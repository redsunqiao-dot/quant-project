"""
优化目标与评估指标 (Optimization Objectives and Evaluation Metrics)。

知识点 (Knowledge Points):
1.  **优化目标 (Optimization Objectives)**: 
    *   在量化研究中，我们不仅关注“哪个因子收益高”，更关注“哪个因子在未来更可靠”。
    *   单一指标往往具有片面性，因此需要构建多维度的评估体系（目标函数），用于因子的筛选和权重分配。

2.  **IC_IR (IC 信息比率)**:
    *   **定义**: IC 的均值除以 IC 的标准差 (Mean/Std)。
    *   **经济含义**: 衡量预测能力的稳定性。如果一个因子 IC 很高但波动巨大（时好时坏），其 IR 会很低。
    *   **重要性**: 在组合构建中，高 IR 的因子往往能提供更平滑的净值曲线。

3.  **多空收益 (Long-Short Return)**:
    *   **定义**: 模拟投资组合，做多因子得分最高的股票组 (Top)，同时做空因子得分最低的组 (Bottom)。
    *   **意义**: 直接衡量因子在扣除市场波动后的纯粹盈利能力 (Alpha)。

4.  **风险调整后收益 (Risk-Adjusted Return / Sharpe-like Ratio)**:
    *   **公式**: Mean(L-S Return) / Std(L-S Return)。
    *   **核心思想**: 类似于夏普比率。它告诉我们：为了获得这些 Alpha 收益，我们承担了多少波动风险。
    *   **实战意义**: 在有交易杠杆或严格风险控制的情况下，性价比（夏普）比纯收益更重要。

5.  **混合打分 (Multi-Objective/Mixed Scoring)**:
    *   **方法**: 将多个维度（如稳定性、盈利能力、换手率等）加权结合。
    *   **示例**: Score = 0.5 * Normalized(IC_IR) + 0.5 * Normalized(Risk_Adj_Return)。
    *   **优势**: 能够选出“德才兼备”的因子，避免被某个极端指标误导。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utils import (  # type: ignore
    calc_ic_summary,
    ensure_dir,
    group_returns,
    infer_factor_columns,
    load_real_panel,
    zscore_by_date,
)


def run(
    output_dir: str = "./outputs/day13_multifactor",
    data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = 240,
    n_groups: int = 10,
) -> None:
    """
    计算并比较不同优化目标下的因子得分。
    """
    # 1. 加载真实数据
    panel = load_real_panel(
        data_dir=data_dir,
        ret_col=ret_horizon,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )
    factor_cols = infer_factor_columns(panel)
    if not factor_cols:
        raise ValueError("No factor columns found in the real panel data.")
    
    # 2. 因子标准化 (Z-Score)
    # 在进行任何跨因子比较前，必须先进行截面标准化，确保量纲一致。
    panel = zscore_by_date(panel, factor_cols)
    panel = panel.dropna(subset=factor_cols + ["ret"])

    records = []
    for col in factor_cols:
        # --- A. 计算基于 IC 的指标 (统计相关性角度) ---
        # 得到 ic_mean, ic_std, ic_ir 等
        ic_stats = calc_ic_summary(panel, col, "ret")
        
        # --- B. 计算基于收益的指标 (模拟实盘角度) ---
        # group_returns 会根据因子值将股票分 10 组，计算每组每期的平均收益
        grouped = group_returns(panel, col, ret_col="ret", n_groups=n_groups)
        
        if grouped.empty or n_groups not in grouped.columns:
            ret_mean = np.nan
            ret_vol = np.nan
        else:
            # 构建多空组合：做多第 10 组 (因子最大)，做空第 1 组 (因子最小)
            ls = grouped[n_groups] - grouped[1]
            ret_mean = ls.mean()       # 多空平均日收益率 (Alpha 强度)
            ret_vol = ls.std(ddof=0)   # 多空收益的波动率 (风险)
        
        # --- C. 构建不同的优化目标得分 (Objectives) ---
        
        # 目标 1: 纯 IC_IR 得分
        # 逻辑：谁预测最稳，谁分就高。适合追求低波动的稳健策略。
        score_ic = ic_stats["ic_ir"]
        
        # 目标 2: 风险调整后收益得分 (Score_Return_Adj)
        # 逻辑：考虑波动后的单位收益。类似于因子的“夏普比率”。
        # 如果波动率为 0 (理论上不可能)，则设为 NaN
        score_ret = ret_mean / ret_vol if ret_vol and ret_vol != 0 else np.nan
        
        # 目标 3: 综合/混合得分 (Multi-Objective)
        # 逻辑：平衡稳定性与收益率。
        # 实际工程中通常需要对各个分项进行 Z-Score 或 Rank 归一化再相加，
        # 这里为了演示简单，直接取等权。
        score_mix = 0.5 * score_ic + 0.5 * score_ret if not np.isnan(score_ret) else score_ic
        
        records.append(
            {
                "factor": col,
                "ic_ir": score_ic,             # 原始稳定性指标
                "ls_mean": ret_mean,           # 原始盈利强度
                "ls_vol": ret_vol,             # 原始波动风险
                "score_return_adj": score_ret, # 性价比得分
                "score_mix": score_mix,        # 综合评分
            }
        )

    # 3. 输出并保存结果
    df = pd.DataFrame(records)
    out = ensure_dir(output_dir)
    df.to_csv(out / "objective_scores.csv", index=False)
    
    print(f"优化目标分析完成。结果已保存至 {out}/objective_scores.csv")
    print("\n因子评分预览 (按综合得分排序):")
    print(df.sort_values("score_mix", ascending=False).round(4))


if __name__ == "__main__":
    run()
