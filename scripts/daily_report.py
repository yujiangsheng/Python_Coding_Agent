#!/usr/bin/env python3
"""生成每日运行报告（Markdown）。

聚合来源：
- scripts/health_check.py
- scripts/window_report.py
- data/evolution/evolution_state.json（若存在）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "reports"


def run_command(cmd: list[str]) -> Tuple[int, str, str]:
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def run_health_check(python_exec: str) -> Dict[str, Any]:
    code, out, err = run_command([python_exec, "scripts/health_check.py"])
    lines = [line for line in out.splitlines() if line.strip()]
    ok_count = sum(1 for x in lines if x.startswith("[OK]"))
    fail_count = sum(1 for x in lines if x.startswith("[FAIL]"))
    warn_count = sum(1 for x in lines if x.startswith("[WARN]"))
    return {
        "exit_code": code,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "output": out,
        "stderr": err,
        "passed": code == 0,
    }


def run_window_report(python_exec: str, window: int, compare_prev: bool) -> Dict[str, Any]:
    cmd = [python_exec, "scripts/window_report.py", "--window", str(window)]
    if compare_prev:
        cmd.extend([
            "--compare-prev",
            "--key-tasks",
            "hard_calc_parser",
            "hard_concurrent_pool",
            "med_decorator_retry",
        ])

    code, out, err = run_command(cmd)
    if code != 0:
        return {
            "ok": False,
            "error": f"window_report failed (code={code}): {err or out}",
        }

    try:
        payload = json.loads(out)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"window_report json parse failed: {exc}",
            "raw": out,
        }

    return {
        "ok": True,
        "payload": payload,
    }


def load_state_snapshot() -> Dict[str, Any]:
    state_path = PROJECT_ROOT / "data" / "evolution" / "evolution_state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_markdown(
    generated_at: str,
    health: Dict[str, Any],
    window_result: Dict[str, Any],
    state: Dict[str, Any],
    window: int,
) -> str:
    lines: list[str] = []
    lines.append("# PyCoder 每日报告")
    lines.append("")
    lines.append(f"- 生成时间: {generated_at}")
    lines.append(f"- 项目路径: {PROJECT_ROOT}")
    lines.append(f"- 统计窗口: 最近 {window} 轮")
    lines.append("")

    # Summary
    lines.append("## 概览")
    lines.append("")
    health_status = "PASS" if health.get("passed") else "FAIL"
    lines.append(f"- 健康检查: **{health_status}** (OK={health.get('ok_count', 0)}, FAIL={health.get('fail_count', 0)}, WARN={health.get('warn_count', 0)})")

    if state:
        lines.append(f"- 当前轮次: {state.get('round_number', 'N/A')}")
        lines.append(f"- 总体通过率: {_fmt(state.get('pass_rate', 'N/A'))}")
        lines.append(f"- 累计通过/总数: {state.get('total_passed', 'N/A')}/{state.get('total_benchmarks_run', 'N/A')}")
        lines.append(f"- Best score: {state.get('best_score', 'N/A')} (round {state.get('best_round', 'N/A')})")
    lines.append("")

    # Window metrics
    lines.append("## 窗口指标")
    lines.append("")
    if not window_result.get("ok"):
        lines.append(f"- 生成失败: {window_result.get('error', 'unknown error')}")
        if window_result.get("raw"):
            lines.append("")
            lines.append("```text")
            lines.append(window_result["raw"][:2000])
            lines.append("```")
    else:
        payload = window_result["payload"]
        current = payload.get("current", {}).get("overall", {})
        lines.append(f"- 当前窗口: {current.get('window', 'N/A')} | rounds={current.get('rounds', 'N/A')} | avg={_fmt(current.get('avg_round_pass_ratio', 'N/A'))} | std={_fmt(current.get('std_round_pass_ratio', 'N/A'))} | full_pass={current.get('full_pass_rounds', 'N/A')}")

        if "previous" in payload:
            prev = payload.get("previous", {}).get("overall", {})
            lines.append(f"- 上一窗口: {prev.get('window', 'N/A')} | rounds={prev.get('rounds', 'N/A')} | avg={_fmt(prev.get('avg_round_pass_ratio', 'N/A'))} | std={_fmt(prev.get('std_round_pass_ratio', 'N/A'))} | full_pass={prev.get('full_pass_rounds', 'N/A')}")
            if "both_hard_ge_80" in payload:
                lines.append(f"- 双 hard 任务 >=0.80: {payload.get('both_hard_ge_80')}")

        key_tasks = payload.get("key_tasks", {})
        if key_tasks:
            lines.append("")
            lines.append("### 关键任务")
            for task, item in key_tasks.items():
                if "new" in item:
                    new = item.get("new", {})
                    lines.append(
                        f"- {task}: pass_rate={_fmt(new.get('pass_rate', 'N/A'))}, "
                        f"fails={new.get('fails', 'N/A')}, flips={new.get('flips', 'N/A')}, "
                        f"streak={new.get('longest_pass_streak', 'N/A')}, "
                        f"delta_pass_rate={_fmt(item.get('delta_pass_rate', 'N/A'))}"
                    )
                else:
                    lines.append(
                        f"- {task}: pass_rate={_fmt(item.get('pass_rate', 'N/A'))}, "
                        f"fails={item.get('fails', 'N/A')}, flips={item.get('flips', 'N/A')}, "
                        f"streak={item.get('longest_pass_streak', 'N/A')}"
                    )

    lines.append("")
    lines.append("## 健康检查明细")
    lines.append("")
    lines.append("```text")
    lines.append(health.get("output", "(no output)")[:4000])
    lines.append("```")

    if health.get("stderr"):
        lines.append("")
        lines.append("```text")
        lines.append(health["stderr"][:2000])
        lines.append("```")

    lines.append("")
    lines.append("## 后续建议")
    lines.append("")
    if health.get("passed") and (window_result.get("ok") and window_result.get("payload", {}).get("both_hard_ge_80", True)):
        lines.append("- 当前状态可继续按既定节奏运行，建议每日保留一份报告归档。")
    else:
        lines.append("- 存在异常或回归，请优先查看 docs/OPERATIONS_TROUBLESHOOTING.md 并执行小步修复。")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily markdown report")
    parser.add_argument("--window", type=int, default=10, help="Window size for metrics (default: 10)")
    parser.add_argument("--no-compare-prev", action="store_true", help="Disable previous-window comparison")
    parser.add_argument("--output", type=str, default="", help="Output markdown file path")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    python_exec = sys.executable
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    health = run_health_check(python_exec)
    window_result = run_window_report(
        python_exec=python_exec,
        window=args.window,
        compare_prev=not args.no_compare_prev,
    )
    state = load_state_snapshot()

    md = build_markdown(
        generated_at=now,
        health=health,
        window_result=window_result,
        state=state,
        window=args.window,
    )

    output_path = Path(args.output) if args.output else (REPORT_DIR / f"daily_report_{stamp}.md")
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    print(f"Report generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
