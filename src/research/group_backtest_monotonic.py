"""
分组回测 (Layering/Quantile Backtest) 与 单调性检查。

Purpose
- Split cross-sectional factor values into quantile groups and compute returns.
- Evaluate monotonicity and long-short spread stability.
目的
- 将每日股票按因子值大小分成 N 组（例如 10 组）。
- 计算每组的次日收益，观察是否“因子越大，收益越大”（单调性）。
- 计算 多空收益 (Long-Short Spread) = 顶层组收益 - 底层组收益。

Knowledge points
- Quantile sorting (qcut) is standard in factor research.
  分位数排序 (qcut) 是因子分析的标准动作。
- Long-short = top group minus bottom group is a core alpha signal proxy.
  多空收益是衡量因子 Alpha 能力的最纯粹指标，因为它抵消了市场波动（Beta）。
- Monotonicity via Spearman correlation checks if higher factor -> higher return.
  单调性检查：计算“组号”与“组收益”的相关性。完美的因子应呈现严格递增或递减。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.common.utils import ensure_dir, group_returns, load_dates, load_factor_series, load_return_series


def run_group_backtest(
    date_path: str,
    factor_dir: str,
    data_dir: str,
    factor_col: Optional[str] = None,
    ret_col: str = "1vwap_pct",
    n_groups: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    # Loop through each date to compute group means and long-short spread.
    # 逐日计算分组收益和多空对冲收益。
    dates = load_dates(date_path, start_date=start_date, end_date=end_date)
    group_records = []
    ls_records = []
    mono_records = []

    for date in dates:
        factor = load_factor_series(factor_dir, date, factor_col=factor_col)
        ret = load_return_series(data_dir, date, ret_col=ret_col)
        if factor.empty or ret.empty:
            continue
            
        # group_returns returns:
        # 1. grouped: Series (index=group_id, value=mean_return)
        # 2. mono: float (Spearman corr between group_id and mean_return)
        grouped, mono = group_returns(factor, ret, n_groups=n_groups)
        if grouped.empty:
            continue

        for group, value in grouped.items():
            group_records.append({"date": date, "group": int(group), "ret": value})

        # Long-short spread uses top minus bottom group mean returns.
        # 多空收益 = 第 N 组收益 - 第 1 组收益。
        if grouped.dropna().empty:
            ls_ret = np.nan
        else:
            ls_ret = grouped.loc[n_groups] - grouped.loc[1]
            
        ls_records.append({"date": date, "long_short": ls_ret})
        mono_records.append({"date": date, "monotonicity": mono})

    group_df = pd.DataFrame(group_records)
    ls_df = pd.DataFrame(ls_records)
    mono_df = pd.DataFrame(mono_records)

    if output_dir:
        out = ensure_dir(output_dir)
        group_df.to_csv(out / "group_return.csv", index=False)
        ls_df.to_csv(out / "long_short.csv", index=False)
        mono_df.to_csv(out / "monotonicity.csv", index=False)

        if not ls_df.empty:
            # Summary stats for Long-Short return
            ls_summary = ls_df["long_short"].agg(["mean", "std"]).reset_index()
            ls_summary.to_csv(out / "long_short_summary.csv", index=False)

        if not mono_df.empty:
            # Summary stats for Monotonicity score
            mono_summary = mono_df["monotonicity"].agg(["mean", "std"]).reset_index()
            mono_summary.to_csv(out / "monotonicity_summary.csv", index=False)

    return {
        "group": group_df,
        "long_short": ls_df,
        "monotonicity": mono_df,
    }


if __name__ == "__main__":
    result = run_group_backtest(
        date_path="./data/date.pkl",
        factor_dir="./factors/neutralized",
        data_dir="./data",
        factor_col=None,
        ret_col="1vwap_pct",
        n_groups=10,
        start_date="2020-01-02",
        end_date="2020-12-31",
        output_dir="./outputs/day12_group",
    )
    print(result["long_short"].head())