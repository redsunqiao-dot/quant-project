"""
约束条件与可实现性检查 (Constraints and Implementability Checks)。

知识点 (Knowledge Points):
1.  **约束条件 (Constraints)**: 在实际投资组合构建中，理论上的最优权重往往无法直接执行，必须遵守特定的约束规则：
    *   **预算约束 (Budget Constraint)**: 权重之和通常为 1 (完全投资) 或 0 (多空对冲)。
    *   **多头/空头限制 (Long/Short Constraint)**: 仅允许做多 (权重 >= 0) 或限制做空比例。
    *   **最大持仓限制 (Position Limits)**: 单个资产或因子的权重不能超过某个上限 (如 30%)，以防止集中度风险。

2.  **单纯形投影 (Simplex Projection)**:
    *   **概念**: 一种数学优化技术，用于将任意一组数值（如原始因子得分，可能是负数或无界）映射（投影）到一个概率单纯形（Probability Simplex）上。
    *   **效果**: 输出的权重向量满足两个条件：(1) 非负性 (w_i >= 0)，(2) 和为 1 (Sum(w_i) = 1)。
    *   **优势**: 相比简单的归一化 (x / sum(x))，单纯形投影能处理负值输入，并且在几何上寻找距离原始点最近的合规点，保留了原始信号的分布特征。

3.  **权重封顶 (Weight Capping)**:
    *   **目的**: 即使经过归一化，某些极端强势因子的权重仍可能过大。强制封顶 (Capping) 是风险控制的最后一道防线。
    *   **处理**: 将超过阈值 (Cap) 的权重截断为阈值，并将多出的权重按比例分配给其他未超限的因子，是一个迭代过程。

4.  **换手率控制 (Turnover Control / Smoothing)**:
    *   **问题**: 因子模型（尤其是高频因子）计算出的目标权重可能在相邻两期剧烈波动。直接调仓会导致巨大的交易成本 (冲击成本 + 手续费)。
    *   **平滑 (Smoothing)**: 采用指数加权移动平均 (EWMA) 或简单的线性组合来平滑权重。
    *   **公式**: Final_Weight_t = (1 - alpha) * Target_Weight_t + alpha * Final_Weight_{t-1}
        *   alpha (turnover_penalty) 越大，越倾向于保留旧权重，换手率越低，但信号滞后越严重。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    from .utils import (  # type: ignore
        calc_ic_summary,
        cap_weights,
        ensure_dir,
        infer_factor_columns,
        load_real_panel,
        project_simplex,
        zscore_by_date,
    )
except ImportError:  # pragma: no cover
    from src.common.utils import (  # type: ignore
        calc_ic_summary,
        cap_weights,
        ensure_dir,
        infer_factor_columns,
        load_real_panel,
        project_simplex,
        zscore_by_date,
    )


def ic_scores(panel: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
    """
    Compute IC-based scores for each factor using real data.
    计算每个因子的 IC 得分，作为初始权重的依据。

    Args:
        panel: 包含因子值和收益率的面板数据
        factor_cols: 因子列名列表

    Returns:
        pd.Series: 因子名 -> IC均值 的映射
    """

    metrics = {}
    for col in factor_cols:
        # 计算每个因子的 IC 均值
        metrics[col] = calc_ic_summary(panel, col, "ret")['ic_mean']
    
    series = pd.Series(metrics).fillna(0.0)
    
    # 防止全零导致除以零错误
    if series.abs().sum() == 0:
        return pd.Series(1.0 / len(series), index=series.index)
        
    return series


def run(
    output_dir: str = "./outputs/day13_multifactor",
    data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = 240,
    cap: float = 0.3,            # 单个因子的最大权重限制 (30%)
    turnover_penalty: float = 0.2, # 换手惩罚系数 (平滑系数)，值越大越倾向于保留旧权重
) -> None:
    """
    演示如何对因子权重施加约束：归一化、封顶、平滑。
    
    流程:
    1. 加载数据并计算基础 IC 得分。
    2. Simplex Projection: 将原始得分转换为非负且和为 1 的权重。
    3. Weight Capping: 限制单个因子权重上限。
    4. Turnover Smoothing: 考虑上一期权重，平滑本期权重以降低换手。
    """
    # 1. 加载数据
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

    # 预处理：标准化
    panel = zscore_by_date(panel, factor_cols)
    panel = panel.dropna(subset=factor_cols + ["ret"])

    dates = sorted(panel["date"].unique())
    if len(dates) < 2:
        raise ValueError("Not enough dates to demonstrate turnover smoothing.")

    # 划分前后两段时间，用于演示换手率平滑
    # prev_dates: 代表 "上一期" 的数据，用于计算旧权重
    # curr_dates: 代表 "当前期" 的数据，用于计算新目标权重
    split_idx = max(int(len(dates) * 0.7), 1)
    prev_dates = dates[:split_idx]
    curr_dates = dates[split_idx:]
    if not curr_dates:
        curr_dates = prev_dates

    prev_panel = panel[panel["date"].isin(prev_dates)]
    curr_panel = panel[panel["date"].isin(curr_dates)]

    # 计算上一期权重 (作为基准)
    prev_simplex = project_simplex(ic_scores(prev_panel, factor_cols).values)
    # 计算本期原始得分
    raw = ic_scores(curr_panel, factor_cols).values

    # --- 核心约束处理流程 ---

    # 1. 单纯形投影 (Simplex Projection)
    # 将原始得分 (可能包含负值) 映射为合法的投资组合权重 (非负, sum=1)
    # 这是最基础的 "仅做多 (Long-Only)" 约束
    simplex = project_simplex(raw)

    # 2. 权重封顶 (Weight Capping)
    # 确保没有因子的权重超过 cap (例如 0.3)
    # 这是一个迭代过程：截断超限权重 -> 重新分配多余权重 -> 再次检查...
    # utils.cap_weights 内部处理了这个逻辑
    capped = cap_weights(simplex, cap)

    # 3. 换手率控制 (Turnover Smoothing)
    # 模拟实际交易中的平滑处理：不能完全按照新信号调仓
    # 新权重 = (1 - alpha) * 目标权重 + alpha * 旧权重
    # alpha (turnover_penalty) 越大，对旧权重的依赖越大，换手越低
    smoothed = (1 - turnover_penalty) * capped + turnover_penalty * prev_simplex
    
    # 注意：平滑后的线性组合可能再次违反封顶约束 (虽然概率较小)，
    # 在严格场景下，最好再次进行封顶检查
    smoothed = cap_weights(smoothed, cap)

    # 4. 结果对比与输出
    df = pd.DataFrame(
        {
            "factor": factor_cols,
            "raw": raw,                         # 原始得分
            "simplex": simplex,                 # 基础归一化权重 (Long-Only)
            "capped": capped,                   # 封顶后权重 (Max-Weight Constraint)
            "turnover_smoothed": smoothed,      # 考虑换手率后的最终权重 (Turnover Constraint)
        }
    )

    out = ensure_dir(output_dir)
    df.to_csv(out / "weights_constraints.csv", index=False)
    
    print(f"权重约束处理完成。结果已保存至 {out}/weights_constraints.csv")
    print(df.round(4))



if __name__ == "__main__":
    run()
