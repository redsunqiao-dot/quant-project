"""
IC 计算 (Rank IC / Spearman 和 Normal IC / Pearson)。

Purpose
- Compute daily IC series for a factor against forward returns.
目的
- 计算因子值与未来收益率之间的每日 IC (Information Coefficient) 序列。

Knowledge points
- Spearman IC measures rank monotonicity; Pearson IC measures linearity.
  Spearman IC (秩相关系数) 衡量排名的单调性，对异常值不敏感；Pearson IC 衡量线性关系。
  量化中常用 Spearman IC，因为它只关心因子排序是否正确。
- Use T-day factor with T+1 return (forward return) to avoid leakage.
  关键点：必须使用 T 日的因子值匹配 T+1 日的收益率 (未来收益)。如果匹配 T 日收益，就是用了未来数据。
- Coverage (intersection size) helps diagnose thin or missing data.
  覆盖率（交集大小）指当日既有因子值又有收益率的股票数量。过低的覆盖率会导致 IC 波动剧烈且不可信。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.utils import calc_ic, ensure_dir, load_dates, load_factor_series, load_return_series


def compute_ic_series(
    date_path: str,
    factor_dir: str,
    data_dir: str,
    factor_col: Optional[str] = None,
    ret_col: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    # Load trading dates and loop through each day independently.
    # 加载交易日历，并逐日循环计算。
    dates = load_dates(date_path, start_date=start_date, end_date=end_date)
    records = []

    for date in dates:
        # Load factor and return cross-sections for the same date.
        # 注意：这里的 load_return_series 内部读取的是 T 日对应的“未来收益”。
        # 例如 data_ret/20200102.csv 里存储的是 20200103 的涨跌幅。
        factor = load_factor_series(factor_dir, date, factor_col=factor_col)
        ret = load_return_series(data_dir, date, ret_col=ret_col)
        
        if factor.empty or ret.empty:
            continue
            
        # Compute both rank IC and linear IC for comparison.
        # 计算两种 IC。Rank IC 通常更稳健。
        ic_s = calc_ic(factor, ret, method="spearman")
        ic_p = calc_ic(factor, ret, method="pearson")
        
        records.append(
            {
                "date": date,
                "ic_spearman": ic_s,
                "ic_pearson": ic_p,
                # Coverage is the number of stocks used in the IC computation.
                # 覆盖度：计算两者的索引交集，只有在两个序列中都存在的股票才能计算相关性。
                "coverage": len(factor.dropna().index.intersection(ret.dropna().index)),
            }
        )

    ic_df = pd.DataFrame(records)
    if output_dir:
        out = ensure_dir(output_dir)
        ic_df.to_csv(out / "ic_series.csv", index=False)
    return ic_df


if __name__ == "__main__":
    df = compute_ic_series(
        date_path="./data/date.pkl",
        factor_dir="./factors/neutralized",
        data_dir="./data",
        factor_col=None,
        ret_col="1vwap_pct",
        start_date="2020-01-02",
        end_date="2020-12-31",
        output_dir="./outputs/day12_ic",
    )
    if df.empty:
        print("No IC records generated.")
    else:
        print(df.head())