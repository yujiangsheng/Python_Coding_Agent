"""
model_loader.py — Tri-backend model loader.

Supports three backends:
  1. **DashScope** (cloud API, recommended): calls the Alibaba Cloud
     DashScope OpenAI-compatible API for Qwen3-Coder models.
  2. **Ollama** (local): calls the local Ollama REST API with
     GGUF-quantised models — fast, memory-efficient.
  3. **Transformers** (fallback): loads the model directly via HuggingFace
     transformers, with GPU > MPS > CPU device priority.

Backend selection via config.model.backend:
  - "dashscope" → DashScope cloud API (default)
  - "ollama"    → local Ollama
  - "transformers" → HuggingFace direct loading
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from agent.exceptions import BackendUnavailableError, ModelNotLoadedError
from agent.utils import retry

logger = logging.getLogger(__name__)

__all__ = ["ModelLoader", "select_device"]

# ======================================================================
# Device helpers (used by transformers backend)
# ======================================================================

def select_device():
    """Select the best available device with priority: CUDA GPU > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"Using CUDA GPU: {gpu_name} ({vram:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        dev = torch.device("cpu")
        logger.info("Using CPU (no GPU/MPS detected)")
    return dev


# ======================================================================
# Backend abstract base class
# ======================================================================

class _Backend(abc.ABC):
    """Every backend must implement load / generate / get_device_info."""

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str: ...

    @abc.abstractmethod
    def get_device_info(self) -> Dict[str, Any]: ...


# ======================================================================
# Unified model loader
# ======================================================================

class ModelLoader:
    """Unified inference interface — delegates to a DashScope, Ollama, or Transformers backend."""

    def __init__(self, config: dict):
        self.config = config
        self._backend: Optional[_Backend] = None

    # --- public API ------------------------------------------------

    def load(self) -> "ModelLoader":
        backend_name = self.config.get("backend", "ollama")

        if backend_name == "dashscope":
            backend = _DashScopeBackend(self.config)
            backend.load()
            self._backend = backend
            return self

        if backend_name == "ollama":
            backend = _OllamaBackend(self.config)
            if backend.is_available():
                backend.load()
                self._backend = backend
                return self
            logger.warning("Ollama not reachable — falling back to transformers")
            backend_name = "transformers"

        if backend_name == "transformers":
            backend = _TransformersBackend(self.config)
            backend.load()
            self._backend = backend
            return self

        raise BackendUnavailableError(f"Unknown backend: {backend_name}")

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def generate(self, messages: list, **kwargs: Any) -> str:
        if self._backend is None:
            raise ModelNotLoadedError("Call .load() before .generate()")
        return self._backend.generate(messages, **kwargs)

    def get_device_info(self) -> dict:
        if self._backend is None:
            raise ModelNotLoadedError("Call .load() before .get_device_info()")
        return self._backend.get_device_info()


# ======================================================================
# Backend: DashScope (OpenAI-compatible API)
# ======================================================================

class _DashScopeBackend(_Backend):
    """Calls the DashScope (Alibaba Cloud) OpenAI-compatible API."""

    def __init__(self, config: dict):
        import os
        self.model_name: str = config.get("dashscope_model", "qwen3-coder-plus")
        self.base_url: str = config.get(
            "dashscope_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.api_key: str = config.get(
            "dashscope_api_key",
            os.environ.get("DASHSCOPE_API_KEY", ""),
        )
        self.max_new_tokens: int = config.get("max_new_tokens", 16384)
        self.temperature: float = config.get("temperature", 0.7)
        self.top_p: float = config.get("top_p", 0.95)
        self.top_k: int = config.get("top_k", 40)

    def load(self):
        if not self.api_key:
            raise BackendUnavailableError(
                "DashScope API key not set.  "
                "Export DASHSCOPE_API_KEY or set dashscope_api_key in config."
            )
        logger.info(
            "Using DashScope backend: %s @ %s",
            self.model_name, self.base_url,
        )

    @retry(max_retries=2, delay=2.0, exceptions=(requests.exceptions.ConnectionError,))
    def generate(self, messages: list, **kwargs) -> str:
        """Call DashScope OpenAI-compatible /chat/completions endpoint."""
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "stream": False,
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            elapsed = time.time() - t0

            usage = data.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            if comp_tokens:
                tps = comp_tokens / elapsed if elapsed > 0 else 0
                logger.info(
                    "DashScope: %d tokens in %.1fs (%.1f tok/s)",
                    comp_tokens, elapsed, tps,
                )
            return content.strip()
        except requests.exceptions.Timeout:
            logger.error("DashScope request timed out (600s)")
            return "[Error: DashScope request timed out]"
        except requests.exceptions.ConnectionError:
            raise  # let @retry handle
        except Exception as e:
            logger.error("DashScope API error: %s", e)
            return "[Error: %s]" % e

    def get_device_info(self) -> dict:
        return {
            "backend": "dashscope",
            "model": self.model_name,
            "url": self.base_url,
            "device": "cloud-api",
            "dtype": "API-managed",
        }


# ======================================================================
# Backend: Ollama (REST API)
# ======================================================================

class _OllamaBackend(_Backend):
    """Calls the local Ollama server via its REST API."""

    def __init__(self, config: dict):
        self.model_name: str = config.get("ollama_model", "qwen3-coder:30b")
        self.base_url: str = config.get("ollama_url", "http://localhost:11434")
        self.max_new_tokens: int = config.get("max_new_tokens", 4096)
        self.temperature: float = config.get("temperature", 0.7)
        self.top_p: float = config.get("top_p", 0.95)
        self.top_k: int = config.get("top_k", 40)

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if any(self.model_name in m for m in models):
                    logger.info(f"Ollama available with model {self.model_name}")
                    return True
                logger.warning(
                    f"Ollama running but model '{self.model_name}' not found. "
                    f"Available: {models}"
                )
                return False
            return False
        except Exception:
            return False

    def load(self):
        logger.info("Using Ollama backend: %s @ %s", self.model_name, self.base_url)

    @retry(max_retries=2, delay=1.0, exceptions=(requests.exceptions.ConnectionError,))
    def generate(self, messages: list, **kwargs) -> str:
        """Call Ollama /api/chat endpoint."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_new_tokens", self.max_new_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "top_k": kwargs.get("top_k", self.top_k),
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            elapsed = time.time() - t0

            # Log token stats if available
            eval_count = data.get("eval_count", 0)
            if eval_count:
                tps = eval_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Ollama: {eval_count} tokens in {elapsed:.1f}s "
                    f"({tps:.1f} tok/s)"
                )
            return content.strip()
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out (300s)")
            return "[Error: Ollama request timed out]"
        except requests.exceptions.ConnectionError:
            raise  # let @retry handle it
        except Exception as e:
            logger.error("Ollama API error: %s", e)
            return f"[Error: {e}]"

    def get_device_info(self) -> dict:
        return {
            "backend": "ollama",
            "model": self.model_name,
            "url": self.base_url,
            "device": "ollama-managed",
            "dtype": "GGUF-quantised",
        }


# ======================================================================
# Backend: Transformers (direct HuggingFace loading)
# ======================================================================

class _TransformersBackend(_Backend):
    """Loads the model directly with HuggingFace transformers."""

    _DTYPE_MAP = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}

    def __init__(self, config: dict):
        self.config = config
        self.device = select_device()
        self._dtype_str: str = config.get("torch_dtype", "auto")
        self.model = None
        self.tokenizer = None
        self.generation_config = None

    @property
    def dtype(self):
        import torch
        if self._dtype_str == "auto":
            if self.device.type == "cuda":
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if self.device.type == "mps":
                return torch.float16
            return torch.float32
        attr = self._DTYPE_MAP.get(self._dtype_str, "float32")
        return getattr(torch, attr)

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        model_name = self.config["name"]
        logger.info(f"Loading model via transformers: {model_name} "
                     f"(dtype={self.dtype}, device={self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.get("trust_remote_code", True),
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "trust_remote_code": self.config.get("trust_remote_code", True),
            "torch_dtype": self.dtype,
        }

        if self.config.get("load_in_4bit"):
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                load_kwargs["device_map"] = "auto"
        elif self.config.get("load_in_8bit"):
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                load_kwargs["device_map"] = "auto"
        else:
            if self.device.type == "cuda":
                load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.get("max_new_tokens", 4096),
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 40),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        n = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"Model loaded: {n:.2f}B params on {self.device}")

    def generate(self, messages: list, **kwargs) -> str:
        import torch
        from transformers import GenerationConfig

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        gen_config = GenerationConfig(**{**self.generation_config.to_dict(), **kwargs})

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def get_device_info(self) -> dict:
        info = {
            "backend": "transformers",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "model": self.config["name"],
        }
        import torch
        if self.device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 2
            )
        return info
