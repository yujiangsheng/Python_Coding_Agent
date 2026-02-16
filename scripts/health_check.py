#!/usr/bin/env python3
"""项目健康检查脚本（无副作用）。"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple
from urllib.request import Request, urlopen

import yaml
from common import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()


def _ok(msg: str):
    print(f"[OK]   {msg}")


def _warn(msg: str):
    print(f"[WARN] {msg}")


def _fail(msg: str):
    print(f"[FAIL] {msg}")


def check_python() -> Tuple[bool, str]:
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python 版本过低：{version.major}.{version.minor}.{version.micro}（建议 >= 3.9）"


def check_files() -> Tuple[bool, str]:
    required = [
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "evolve.py",
        PROJECT_ROOT / "config.yaml",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "agent" / "core.py",
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.exists()]
    if missing:
        return False, f"缺失关键文件: {', '.join(missing)}"
    return True, "关键文件完整"


def check_configs() -> Tuple[bool, str]:
    cfgs = [
        PROJECT_ROOT / "config.yaml",
        PROJECT_ROOT / "configs" / "config.dev.yaml",
        PROJECT_ROOT / "configs" / "config.prod.yaml",
    ]
    required_keys = [
        "model",
        "memory",
        "execution",
        "self_improvement",
        "memory_agent",
        "reflection",
        "logging",
    ]

    missing_files: List[str] = []
    bad_keys: List[str] = []

    for cfg in cfgs:
        if not cfg.exists():
            missing_files.append(str(cfg.relative_to(PROJECT_ROOT)))
            continue
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            bad_keys.append(f"{cfg.name}: 顶层不是 mapping")
            continue
        for k in required_keys:
            if k not in data:
                bad_keys.append(f"{cfg.name}: 缺少 {k}")

    if missing_files:
        return False, f"配置模板缺失: {', '.join(missing_files)}"
    if bad_keys:
        return False, "配置不完整: " + "; ".join(bad_keys)
    return True, "配置文件可解析且关键字段完整"


def check_imports() -> Tuple[bool, str]:
    modules = [
        "agent.core",
        "agent.code_generator",
        "agent.memory.manager",
        "agent.reflection_agent",
    ]
    failed = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as exc:
            failed.append(f"{m} ({exc})")
    if failed:
        return False, "模块导入失败: " + "; ".join(failed)
    return True, "核心模块可导入"


def check_dirs_writable() -> Tuple[bool, str]:
    dirs = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "evolution",
        PROJECT_ROOT / "data" / "improvements",
    ]
    not_writable = []
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".healthcheck.tmp"
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
        except Exception:
            not_writable.append(str(d.relative_to(PROJECT_ROOT)))
    if not_writable:
        return False, f"目录不可写: {', '.join(not_writable)}"
    return True, "关键数据目录可写"


def check_ollama_connectivity() -> Tuple[bool, str]:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    backend = model_cfg.get("backend", "")
    if backend != "ollama":
        return True, f"当前后端为 {backend}，跳过 Ollama 连通性检查"

    base_url = str(model_cfg.get("ollama_url", "http://localhost:11434")).rstrip("/")
    url = f"{base_url}/api/tags"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=3) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            payload = json.loads(raw) if raw else {}
            count = len(payload.get("models", [])) if isinstance(payload, dict) else 0
            return True, f"Ollama 可访问（models={count}）"
    except Exception as exc:
        return False, f"Ollama 连通失败: {exc}"


def main() -> int:
    os.chdir(PROJECT_ROOT)
    print("== PyCoder Health Check ==")
    checks = [
        ("Python", check_python),
        ("Files", check_files),
        ("Configs", check_configs),
        ("Imports", check_imports),
        ("WritableDirs", check_dirs_writable),
        ("Ollama", check_ollama_connectivity),
    ]

    failed = 0
    for name, fn in checks:
        try:
            ok, msg = fn()
            if ok:
                _ok(f"{name}: {msg}")
            else:
                failed += 1
                _fail(f"{name}: {msg}")
        except Exception as exc:
            failed += 1
            _fail(f"{name}: 未捕获异常: {exc}")

    if failed == 0:
        _ok("系统健康检查通过")
        return 0

    _warn(f"存在 {failed} 项失败，请按 docs/OPERATIONS_TROUBLESHOOTING.md 排查")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
