#!/usr/bin/env python3
"""
切割/CTR 研究轨（兴趣轨 2）：单独算子 → 验证门 → 报告，不写入主合成/ live。

用法:
  .venv/bin/python scripts/cutting_track_run.py
  .venv/bin/python scripts/cutting_track_run.py --start 2023-01-01 --register
  .venv/bin/python scripts/cutting_track_run.py --update-registry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import ensure_dir, list_dated_stems, zscore_by_date
from src.data.data_loader import DataLoader
from src.research.factor_library import FactorLibrary
from src.research.factor_pipeline import _build_panel
from src.research.factor_validate import validate_factors
from src.strategy.factor_calculator import FactorCalculator

CTR_FACTORS = [
    "ctr_turn_spread_20",
    "ctr_vol_cut_20",
    "ctr_ideal_amp_rev_20",
]

REGISTRY_META = [
    {
        "name": "ctr_turn_spread_20",
        "desc": "CTR 日频：低换日均收益减高换日均收益",
        "formula": "mean(r|turn<=med_20)-mean(r|turn>med_20)",
        "hypothesis": "安静日定价更持续，拥挤日更易均值回复",
        "impl": "ctr_turn_spread_20",
    },
    {
        "name": "ctr_vol_cut_20",
        "desc": "成交量切割：低量日动量减高量日动量",
        "formula": "mean(r|vol<=med_20)-mean(r|vol>med_20)",
        "hypothesis": "聪明动量：信息多在低成交量交易日",
        "impl": "ctr_vol_cut_20",
    },
    {
        "name": "ctr_ideal_amp_rev_20",
        "desc": "理想振幅切割：高振幅日反转",
        "formula": "-mean(r|amp>med_20), amp=high/low-1",
        "hypothesis": "高振幅日过度反应，切割后取反转",
        "impl": "ctr_ideal_amp_rev_20",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="切割/CTR 研究轨：计算 + 验证门")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="")
    p.add_argument(
        "--factor-root",
        default="./factors/cutting_track",
        help="研究轨因子根目录（raw 落盘）",
    )
    p.add_argument(
        "--output-dir",
        default="./outputs/cutting_track",
        help="验证报告目录",
    )
    p.add_argument(
        "--register",
        action="store_true",
        help="若 registry 无条目则登记为 candidate",
    )
    p.add_argument(
        "--update-registry",
        action="store_true",
        help="验证后将指标写回 registry（仍保持 candidate，除非手动改 active）",
    )
    p.add_argument("--ret-col", default="1vwap_pct")
    p.add_argument("--min-abs-ic", type=float, default=0.02)
    return p.parse_args()


def ensure_registered(library: FactorLibrary, do_register: bool) -> None:
    if not do_register:
        return
    for meta in REGISTRY_META:
        if meta["name"] in library._factors:
            continue
        library.register(
            name=meta["name"],
            desc=meta["desc"],
            formula=meta["formula"],
            hypothesis=meta["hypothesis"],
            source="第8–9课 CTR/切割论（日频研究轨）",
            windows={"lookback": 20},
            impl=meta["impl"],
            status="candidate",
            data_deps=["data_daily"],
            card_path="factors/research_cards/cutting_ctr.md",
            checklist_score=8,
        )
        print(f"已登记 candidate: {meta['name']}")
    library.save()


def main() -> int:
    args = parse_args()
    loader = DataLoader(args.data_dir)
    dates = loader.get_all_dates()
    if not dates:
        print("无交易日数据")
        return 1

    start = args.start
    end = args.end or dates[-1]
    target = [d for d in dates if start <= d <= end]
    if len(target) < 60:
        print(f"区间过短: {len(target)} 日")
        return 1

    raw_dir = Path(args.factor_root) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    calc = FactorCalculator(loader)
    calc.factor_dir = raw_dir
    print(f"计算切割轨因子 {CTR_FACTORS}，区间 {target[0]} ~ {target[-1]}")
    calc.calculate_range_and_save(
        start_date=target[0],
        end_date=target[-1],
        factor_names=CTR_FACTORS,
    )

    factor_dates = [
        d for d in list_dated_stems(raw_dir) if target[0] <= d <= target[-1]
    ]
    if not factor_dates:
        print("未写出任何因子文件")
        return 1

    panel = _build_panel(raw_dir, args.data_dir, factor_dates, args.ret_col)
    if panel.empty:
        print("因子与收益无法对齐")
        return 1

    present = [c for c in CTR_FACTORS if c in panel.columns]
    panel = zscore_by_date(panel, present)

    out_dir = ensure_dir(args.output_dir)
    library = FactorLibrary()
    ensure_registered(library, args.register)

    val_library = library if args.update_registry else FactorLibrary()
    summary = validate_factors(
        panel=panel,
        factor_cols=present,
        factor_dir=raw_dir,
        dates=factor_dates,
        library=val_library,
        output_dir=out_dir / "factor_validation",
        ret_col="ret",
        min_abs_ic=args.min_abs_ic,
    )
    if args.update_registry:
        val_library.save()
        print("已写回 registry 验证指标（status 由验证门决定）")

    summary_path = out_dir / "factor_validation" / "validation_summary.csv"
    print("\n=== 切割轨验证摘要 ===")
    print(summary.to_string(index=False))
    print(f"\n报告: {summary_path}")

    passed = summary[summary["status"] == "active"]["factor"].tolist()
    if passed:
        print(f"\n过门 active: {passed}（仍禁止默认进 live/冻结池，需 OOS 与合成对照）")
    else:
        print("\n本轮无 active；可改窗口/切割口径后再跑本脚本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
