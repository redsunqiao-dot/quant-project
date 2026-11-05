#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日更前数据质检：日历与 data_daily / data_ud_new / data_ret 是否齐套。

示例：
  python scripts/check_data_freshness.py
  python scripts/check_data_freshness.py --data-dir ./data --max-lag-days 1
  python scripts/check_data_freshness.py --expect-last 2026-10-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_freshness import check_data_freshness, format_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查本地行情数据新鲜度与齐套")
    p.add_argument("--data-dir", default="./data", help="数据根目录")
    p.add_argument(
        "--expect-last",
        default="",
        help="期望末日 YYYY-MM-DD；默认用 date.pkl 最后一日",
    )
    p.add_argument(
        "--max-lag-days",
        type=int,
        default=0,
        help="允许比期望末日落后的交易日个数",
    )
    p.add_argument(
        "--skip-optional",
        action="store_true",
        help="不检查行业/估值/财务可选目录",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = check_data_freshness(
        args.data_dir,
        expect_calendar_last=(args.expect_last or None),
        max_lag_days=args.max_lag_days,
        check_optional=not args.skip_optional,
    )
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
