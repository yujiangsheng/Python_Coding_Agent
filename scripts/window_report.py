#!/usr/bin/env python3
"""演化窗口指标报告脚本。

功能：
- 读取 data/evolution/evolution.log
- 统计指定窗口的 overall 与 task 指标
- 可选与前一窗口对比（相同窗口长度）
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "data" / "evolution" / "evolution.log"


def parse_log(path: Path) -> Tuple[Dict[str, Dict[int, int]], Dict[int, float], int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_round = None
    task_round: Dict[str, Dict[int, int]] = {}
    round_pass: Dict[int, float] = {}
    max_round = 0

    for ln in lines:
        if "EVOLUTION ROUND " in ln:
            try:
                current_round = int(ln.rsplit("EVOLUTION ROUND ", 1)[1].strip())
                max_round = max(max_round, current_round)
            except Exception:
                pass
            continue

        marker = None
        status = None
        if "✓ PASS  [" in ln:
            marker = "✓ PASS  ["
            status = 1
        elif "✗ FAIL  [" in ln:
            marker = "✗ FAIL  ["
            status = 0

        if marker and current_round is not None:
            task = ln.split(marker, 1)[1].split("]", 1)[0].strip()
            if task:
                task_round.setdefault(task, {})[current_round] = status
            continue

        if "Round " in ln and "benchmarks:" in ln and "passed" in ln:
            try:
                round_id = int(ln.split("Round ", 1)[1].split(" benchmarks", 1)[0].strip())
                frac = ln.split("benchmarks:", 1)[1].split(" passed", 1)[0].strip().split()[0]
                p, t = frac.split("/")
                round_pass[round_id] = int(p) / int(t)
                max_round = max(max_round, round_id)
            except Exception:
                pass

    return task_round, round_pass, max_round


def flips(seq: List[int]) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def longest_pass_streak(seq: List[int]) -> int:
    best = 0
    cur = 0
    for x in seq:
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize_window(
    task_round: Dict[str, Dict[int, int]],
    round_pass: Dict[int, float],
    start_round: int,
    end_round: int,
) -> Dict:
    ratios = [round_pass[r] for r in range(start_round, end_round + 1) if r in round_pass]
    overall = {
        "window": [start_round, end_round],
        "rounds": len(ratios),
        "avg_round_pass_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "std_round_pass_ratio": round(statistics.pstdev(ratios), 4) if len(ratios) > 1 else 0.0,
        "full_pass_rounds": sum(1 for x in ratios if abs(x - 1.0) < 1e-9),
    }

    tasks = {}
    for task, per_round in sorted(task_round.items()):
        seq = [per_round[r] for r in range(start_round, end_round + 1) if r in per_round]
        if not seq:
            continue
        passes = sum(seq)
        tasks[task] = {
            "n": len(seq),
            "pass_rate": round(passes / len(seq), 4),
            "fails": int(len(seq) - passes),
            "flips": flips(seq),
            "longest_pass_streak": longest_pass_streak(seq),
            "seq": " ".join("P" if x else "F" for x in seq),
        }

    return {"overall": overall, "tasks": tasks}


def compare_selected_tasks(base: Dict, new: Dict, key_tasks: List[str]) -> Dict:
    comp = {}
    for task in key_tasks:
        old_item = base["tasks"].get(task)
        new_item = new["tasks"].get(task)
        if old_item and new_item:
            comp[task] = {
                "base": old_item,
                "new": new_item,
                "delta_pass_rate": round(new_item["pass_rate"] - old_item["pass_rate"], 4),
                "delta_fails": new_item["fails"] - old_item["fails"],
            }
    return comp


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evolution window metrics report")
    parser.add_argument("--window", type=int, default=10, help="Window size in rounds (default: 10)")
    parser.add_argument("--end-round", type=int, default=None, help="Window end round; default uses latest")
    parser.add_argument("--compare-prev", action="store_true", help="Compare with previous same-sized window")
    parser.add_argument(
        "--key-tasks",
        nargs="*",
        default=["hard_calc_parser", "hard_concurrent_pool", "med_decorator_retry"],
        help="Task IDs to highlight",
    )
    args = parser.parse_args()

    if args.window <= 0:
        raise SystemExit("--window must be > 0")

    if not LOG_PATH.exists():
        raise SystemExit(f"Log file not found: {LOG_PATH}")

    task_round, round_pass, latest = parse_log(LOG_PATH)
    end_round = args.end_round if args.end_round is not None else latest
    start_round = max(1, end_round - args.window + 1)

    current = summarize_window(task_round, round_pass, start_round, end_round)

    output = {
        "current": current,
        "latest_round": latest,
        "key_tasks": {},
    }

    if args.compare_prev:
        prev_end = start_round - 1
        if prev_end >= 1:
            prev_start = max(1, prev_end - args.window + 1)
            prev = summarize_window(task_round, round_pass, prev_start, prev_end)
            output["previous"] = prev
            output["key_tasks"] = compare_selected_tasks(prev, current, args.key_tasks)

            parser_rate = output["key_tasks"].get("hard_calc_parser", {}).get("new", {}).get("pass_rate", -1)
            pool_rate = output["key_tasks"].get("hard_concurrent_pool", {}).get("new", {}).get("pass_rate", -1)
            output["both_hard_ge_80"] = parser_rate >= 0.8 and pool_rate >= 0.8

    else:
        output["key_tasks"] = {
            t: current["tasks"][t] for t in args.key_tasks if t in current["tasks"]
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
