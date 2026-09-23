"""
IC 统计指标汇总。

Purpose
- Aggregate IC series into standard metrics (mean/std/IR/win-rate/t-stat).
目的
- 将每日 IC 序列聚合为标准的评估指标（均值、波动率、IR、胜率、t统计量）。

Knowledge points
- IC_IR = mean / std is a signal-to-noise proxy (like information ratio).
  IC_IR (Information Ratio of IC) = IC均值 / IC标准差。
  它衡量因子的稳定性。高 IC 但波动巨大的因子，IR 可能很低，很难在实盘中利用。
  通常优秀的因子 IC_IR > 0.5 甚至 > 1.0。
- Win rate shows consistency but can be misleading with tiny IC magnitudes.
  胜率 (IC > 0 的天数占比) 反映一致性。如果 IC 均值很小但胜率很高，可能只是运气或存活偏差。
- IC t-stat assumes iid; use it as a rough stability check, not a proof.
  IC t统计量假设样本独立同分布 (IID)。虽然金融数据不完全满足此假设，但 t > 2.0 通常被视为统计显著。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.utils import calc_ic_stats, ensure_dir


def summarize_ic_stats(ic_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
    # Expect a file produced by 01_ic_calculation.py.
    # 读取上一步生成的 IC 序列文件。
    ic_df = pd.read_csv(ic_path)
    stats_records = []

    # Summarize all columns that start with ic_ (e.g. ic_spearman, ic_pearson).
    # 自动识别所有以 ic_ 开头的列进行统计。
    ic_cols = [col for col in ic_df.columns if col.startswith("ic_")]
    for col in ic_cols:
        # calc_ic_stats 是核心统计函数，位于 utils.py
        stats = calc_ic_stats(ic_df[col])
        stats_records.append({"metric": col, **stats})

    stats_df = pd.DataFrame(stats_records)
    if output_dir:
        out = ensure_dir(output_dir)
        stats_df.to_csv(out / "ic_summary.csv", index=False)
    return stats_df


if __name__ == "__main__":
    path = "./outputs/day12_ic/ic_series.csv"
    if not Path(path).exists():
        print(f"Missing IC series file: {path}")
    else:
        summary = summarize_ic_stats(path, output_dir="./outputs/day12_ic")
        print(summary)