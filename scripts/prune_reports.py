#!/usr/bin/env python3
"""清理历史日报文件，按修改时间仅保留最近 N 份。"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "reports"


def list_report_files(report_dir: Path) -> list[Path]:
    if not report_dir.exists():
        return []
    return sorted(
        [p for p in report_dir.glob("*.md") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def prune_reports(keep: int, report_dir: Path) -> tuple[int, list[Path]]:
    files = list_report_files(report_dir)
    if keep < 0:
        raise ValueError("--keep must be >= 0")
    if len(files) <= keep:
        return 0, []

    to_delete = files[keep:]
    for p in to_delete:
        p.unlink(missing_ok=True)
    return len(to_delete), to_delete


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old markdown reports")
    parser.add_argument("--keep", type=int, default=30, help="How many newest reports to keep (default: 30)")
    parser.add_argument("--dir", type=str, default="data/reports", help="Report directory relative to project root")
    args = parser.parse_args()

    report_dir = Path(args.dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    deleted_count, deleted_files = prune_reports(args.keep, report_dir)
    print(f"Report dir: {report_dir}")
    print(f"Kept latest: {args.keep}")
    print(f"Deleted: {deleted_count}")
    for p in deleted_files[:10]:
        print(f" - {p.relative_to(PROJECT_ROOT)}")
    if len(deleted_files) > 10:
        print(f" ... and {len(deleted_files) - 10} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
