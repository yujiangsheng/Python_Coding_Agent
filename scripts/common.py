#!/usr/bin/env python3
"""scripts 共享工具。

用于统一：
- 项目根目录定位
- sys.path 注入
- 相对路径解析
- 子命令执行
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root_on_path() -> Path:
    """确保项目根目录在 sys.path 中，便于跨目录导入，并返回项目根路径。"""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def resolve_project_path(path_str: str, default_path: Path, ensure_parent: bool = True) -> Path:
    """解析输出路径：支持绝对路径与项目相对路径。"""
    path = Path(path_str) if path_str else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_project_command(cmd: list[str]) -> Tuple[int, str, str]:
    """在项目根目录执行命令并返回 (exit_code, stdout, stderr)。"""
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()
