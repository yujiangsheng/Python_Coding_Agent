"""
utils.py — Shared utilities for PyCoder agent.

Consolidates common helpers (JSON parsing, code-fence stripping, retry
logic, token estimation, JSON persistence, and shared quality markers)
so they are not duplicated across modules.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "strip_think_blocks",
    "strip_code_fences",
    "parse_json_response",
    "retry",
    "estimate_tokens",
    "JsonStore",
    "ERROR_MARKERS",
    "UNCERTAINTY_MARKERS",
    "QUALITY_MARKERS_GOOD",
    "QUALITY_MARKERS_BAD",
]


# ======================================================================
# Shared quality / error markers (single source of truth)
# ======================================================================

ERROR_MARKERS: List[str] = [
    "error", "错误", "失败", "traceback", "exception",
]

UNCERTAINTY_MARKERS: List[str] = [
    "不确定", "不太清楚", "不太了解", "可能是", "也许",
    "I'm not sure", "I don't know", "I'm unsure",
    "not certain", "might be", "may not be accurate",
    "无法确认", "需要进一步查证", "无法确定",
]

QUALITY_MARKERS_BAD: List[str] = ERROR_MARKERS + [
    "not sure", "不确定", "可能是", "I don't know",
]

QUALITY_MARKERS_GOOD: List[str] = [
    "```python", "successfully", "成功", "✓",
    "def ", "class ", "return ",
]


# ======================================================================
# Reusable JSON persistence
# ======================================================================

class JsonStore:
    """Lightweight JSON file persistence with load / save / auto-mkdir.

    Eliminates the copy-pasted ``os.makedirs + json.load/dump`` pattern
    used across ErrorRegistry, EvolutionTracker, SkillRegistry, etc.

    Usage::

        store = JsonStore("data/skills.json", default=[])
        store.load()          # populates store.data
        store.data.append(…)
        store.save()
    """

    def __init__(self, path: str, *, default: Any = None):
        self.path = path
        self.data: Any = default if default is not None else {}

    def load(self) -> bool:
        """Load data from disk.  Returns True on success."""
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            logger.error("JsonStore load failed (%s): %s", self.path, e)
            return False

    def save(self):
        """Write current data to disk (creates directories as needed)."""
        dirpath = os.path.dirname(self.path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)


# ======================================================================
# Text / JSON helpers
# ======================================================================

def strip_think_blocks(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks emitted by Qwen-3 models.

    Handles multiple think blocks, nested tags, and partial/unclosed blocks.
    """
    # Remove complete <think>…</think> blocks (non-greedy)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # For unclosed <think> blocks, only strip the tag itself, keep the content
    cleaned = re.sub(r"<think>", "", cleaned)
    return cleaned.strip()


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```lang ... ```) from *text*.

    Returns the inner content with leading/trailing whitespace trimmed.
    Handles fences that appear anywhere in the text (not only at the start).
    """
    text = text.strip()
    # Try to extract content between ``` fences anywhere in text
    m = re.search(r"```(?:\w+)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: strip leading/trailing fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*\n?", "", text)
        text = re.sub(r"\n?\s*```$", "", text)
    return text.strip()


def _find_balanced_json(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Find the first balanced ``open_ch…close_ch`` block in *text*.

    Uses a simple depth counter so nested braces/brackets are handled
    correctly (unlike a greedy regex which grabs too much).
    """
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_response(text: str) -> Any:
    """Robustly extract and parse JSON from LLM output.

    Handles common wrapper patterns emitted by LLMs:
      - ``<think>…</think>`` reasoning blocks (Qwen-3)
      - Markdown code fences (````json … ````)
      - Explanatory prose before / after the JSON

    Strategy (first success wins):
      1. Strip think blocks → strip code fences → ``json.loads``.
      2. Balanced-bracket search for ``[…]`` (array).
      3. Balanced-bracket search for ``{…}`` (object).
      4. Greedy regex fallback.

    Raises ``json.JSONDecodeError`` if nothing works.
    """
    # --- Pre-processing: remove think blocks & code fences ---
    cleaned = strip_think_blocks(text)
    cleaned = strip_code_fences(cleaned)

    # 1. Direct parse of cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Balanced-bracket search for JSON array (most common for suggestions)
    arr = _find_balanced_json(cleaned, "[", "]")
    if arr:
        try:
            return json.loads(arr)
        except json.JSONDecodeError:
            pass

    # 3. Balanced-bracket search for JSON object
    obj = _find_balanced_json(cleaned, "{", "}")
    if obj:
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            pass

    # 4. Greedy regex fallback on *original* text (after think-strip only)
    raw = strip_think_blocks(text)
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # 5. Last resort: try to find ANY json-like content line by line
    for line in cleaned.split("\n"):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("No valid JSON found in response", text[:200], 0)


# ======================================================================
# Retry decorator
# ======================================================================

def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator: retry *func* with exponential back-off on *exceptions*."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            wait = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        logger.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1,
                            exc, wait,
                        )
                        time.sleep(wait)
                        wait *= backoff
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ======================================================================
# Token estimation
# ======================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count with CJK-awareness.

    English / code  ≈ 4 characters per token.
    CJK (Chinese)   ≈ 1.5 characters per token.
    """
    cjk = sum(
        1 for c in text
        if "\u4e00" <= c <= "\u9fff"        # CJK Unified Ideographs
        or "\u3400" <= c <= "\u4dbf"         # Extension A
        or "\uf900" <= c <= "\ufaff"         # CJK Compatibility
    )
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4)
