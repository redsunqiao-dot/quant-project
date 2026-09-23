"""
因子综合打分 (基于 IC, 稳定性, 单调性 和 换手率)。

Purpose
- Combine multiple diagnostics into a single factor score for ranking.
目的
- 将多个维度的评估指标合成一个分数，用于对候选因子进行优胜劣汰的排序。

Knowledge points
- IC/IR capture predictive power and stability.
  IC 和 IC_IR 捕捉预测能力和稳定性。
- Monotonicity reflects the shape of the factor-return relationship.
  单调性反映因子与收益关系的线性程度。
- Turnover penalizes unstable factors that likely incur higher trading costs.
  换手率 (Turnover) 惩罚那些波动剧烈的因子，因为它们会带来高昂的交易成本。
  这里使用 (1 - Rank IC) 作为换手率的代理指标。
- The scoring formula is a heuristic; adjust weights for your use case.
  打分公式通常是经验性的（启发式），实际应用中需要根据策略偏好调整权重。
"""

# '''
# Factor A: ICIR=0.5 mono=0.3
# Factor B: ICIR=0.3 mono=0.8
# Factor C: ICIR=0.6 换手50%

# 1. IC mean

# 2. ICIR

# 3. mono

# 4. cost/turnover
# '''

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.common.utils import (
    calc_ic,
    calc_ic_stats,
    ensure_dir,
    group_returns,
    infer_factor_cols,
    load_dates,
    load_factor_frame,
    load_factor_series,
    load_return_series,
)


def calc_rank_ic(series_a: pd.Series, series_b: pd.Series) -> float:
    # Rank IC between consecutive days approximates factor stability.
    # 计算相邻两天的 Rank IC，用于近似因子稳定性（即换手率的反面）。
    # 如果今天因子排名和昨天差不多，说明因子很稳定，换手率低。
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")


def resolve_factor_cols(factor_dir: str, dates: List[str], factor_cols: Optional[List[str]]) -> List[str]:
    # Infer factor columns from the first available file if not provided.
    # 如果未指定列名，则从第一个有效文件中推断。
    if factor_cols:
        return factor_cols
    for date in dates:
        df = load_factor_frame(factor_dir, date)
        if not df.empty:
            return infer_factor_cols(df.reset_index())
    return []


def score_factors(
    date_path: str,
    factor_dir: str,
    data_dir: str,
    factor_cols: Optional[List[str]] = None,
    ret_col: str = "1vwap_pct",
    n_groups: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = 120,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    # Optionally restrict to a recent window to avoid overly long runs.
    # 限制回测窗口（例如最近 120 天），既能节省计算，又能反映因子的近期表现。
    dates = load_dates(date_path, start_date=start_date, end_date=end_date)
    if max_dates:
        dates = dates[-max_dates:]
    factor_cols = resolve_factor_cols(factor_dir, dates, factor_cols)

    records = []
    for factor_col in factor_cols:
        ic_values = []
        mono_values = []
        turnover_values = []
        prev_factor = None

        for date in dates:
            factor = load_factor_series(factor_dir, date, factor_col=factor_col)
            ret = load_return_series(data_dir, date, ret_col=ret_col)
            if factor.empty or ret.empty:
                continue
            
            # IC and monotonicity are computed per date and aggregated.
            ic_values.append(calc_ic(factor, ret, method="spearman"))
            _, mono = group_returns(factor, ret, n_groups=n_groups)
            mono_values.append(mono)

            if prev_factor is not None:
                # Use 1 - Rank IC as a turnover proxy.
                # Rank IC 越接近 1，说明因子排名没变，换手越低。
                # 所以 Turnover proxy = 1 - Rank IC。
                rank_ic = calc_rank_ic(prev_factor, factor)
                turnover_values.append(1 - rank_ic if not np.isnan(rank_ic) else np.nan)
            prev_factor = factor

        ic_series = pd.Series(ic_values)
        ic_stats = calc_ic_stats(ic_series)
        mono_mean = pd.Series(mono_values).dropna().mean()
        turnover_mean = pd.Series(turnover_values).dropna().mean()

        if pd.isna(mono_mean):
            mono_mean = 0.0
        if pd.isna(turnover_mean):
            turnover_mean = 1.0 # 默认高换手惩罚

        # Weighted score: prioritize IC_IR, then IC_mean, then monotonicity/turnover.
        # 综合打分公式：
        # 40% IC_IR (稳定性调整后的预测力)
        # 20% IC Mean (绝对预测力)
        # 20% Monotonicity (分组单调性)
        # 20% Stability (低换手奖励)
        score = (
            0.4 * (ic_stats.get("ic_ir") or 0)
            + 0.2 * (ic_stats.get("ic_mean") or 0)
            + 0.2 * mono_mean
            + 0.2 * (1 - turnover_mean)
        )

        records.append(
            {
                "factor": factor_col,
                "ic_mean": ic_stats.get("ic_mean"),
                "ic_std": ic_stats.get("ic_std"),
                "ic_ir": ic_stats.get("ic_ir"),
                "ic_win_rate": ic_stats.get("ic_win_rate"),
                "ic_t": ic_stats.get("ic_t"),
                "monotonicity_mean": mono_mean,
                "turnover_mean": turnover_mean,
                "score": score,
                "n": ic_stats.get("n"),
            }
        )

    result = pd.DataFrame(records).sort_values("score", ascending=False)
    if output_dir:
        out = ensure_dir(output_dir)
        result.to_csv(out / "factor_scores.csv", index=False)
    return result


if __name__ == "__main__":
    df = score_factors(
        date_path="./data/date.pkl",
        factor_dir="./factors/neutralized",
        data_dir="./data",
        factor_cols=None,
        ret_col="1vwap_pct",
        n_groups=10,
        start_date="2020-01-02",
        end_date="2020-12-31",
        max_dates=120,
        output_dir="./outputs/day12_scores",
    )
    print(df.head())