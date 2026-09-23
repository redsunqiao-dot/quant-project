"""
多因子加权方法与合成因子生成。

知识点 (Knowledge Points):
1.  **多因子模型 (Multifactor Model)**: 将多个因子（Alpha 信号）组合成一个单一的合成得分，比单因子模型能更稳健地预测未来资产收益。
2.  **因子标准化 (Factor Standardization/Z-Score)**: 在横截面（每个日期）上对因子进行归一化，使其均值为 0，标准差为 1。这确保了不同量纲的因子（如市盈率 vs 动量）具有可比性。
3.  **加权方式 (Weighting Schemes)**:
    *   **等权 (Equal Weight)**: 简单平均。假设所有因子效果相同。稳健但忽略了因子质量差异。
    *   **IC 加权 (IC Weighting)**: 权重与信息系数（因子与收益的相关性）成正比。赋予预测能力强的因子更高权重。
    *   **ICIR 加权 (IC_IR Weighting)**: 权重与 IC/IC标准差 成正比。奖励预测稳定且持续的因子。
    *   **分组收益加权 (Return Spread Weighting)**: 根据多空组合（Top组 - Bottom组）的收益差加权。直接衡量因子的盈利能力。
4.  **合成因子 (Composite Signal)**: 标准化后因子的加权和，用于最终的选股排序。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 导入量化分析工具函数
from src.common.utils import (  # type: ignore
    calc_ic_summary,
    ensure_dir,
    group_returns,
    infer_factor_columns,
    load_real_panel,
    normalize_weights,
    zscore_by_date,
)


def compute_factor_metrics(
    panel: pd.DataFrame, factor_cols: List[str], ret_col: str, n_groups: int
) -> pd.DataFrame:
    """
    计算每个因子的绩效指标，用于确定其权重。
    
    计算的指标:
    1. IC Mean: 因子与未来收益率的平均相关系数。衡量预测的准确性。
    2. IC IR (Information Ratio): IC 均值 / IC 标准差。衡量预测的稳定性。IC_IR 越高，说明因子不仅有效，而且发挥稳定，不易失效。
    3. Group Spread (Ret Mean): 头部组（第 N 组）与尾部组（第 1 组）的平均收益差。
       - 这是对因子区分能力的直接度量。
       - 相比 IC (只看线性关系)，分组收益能捕捉到单调但非线性的收益差异。
    """
    records = []
    for col in factor_cols:
        # 1. 计算 IC 统计量 (IC 均值, IC 标准差, IC IR)
        ic_stats = calc_ic_summary(panel, col, ret_col)
        
        # 2. 计算分组收益 (分位数分析)
        # 根据因子值将资产分为 n_groups 组，并计算每组的平均收益
        grouped = group_returns(panel, col, ret_col=ret_col, n_groups=n_groups)
        
        ret_mean = np.nan
        if not grouped.empty and n_groups in grouped.columns:
            # 计算多空收益差: 头部组收益 - 尾部组收益
            # 这代表了因子的纯 Alpha 收益能力 (假设做多 Top，做空 Bottom)
            ret_mean = (grouped[n_groups] - grouped[1]).mean()
            
        records.append(
            {
                "factor": col,
                "ic_mean": ic_stats["ic_mean"],
                "ic_ir": ic_stats["ic_ir"],
                "ret_mean": ret_mean,
            }
        )
    return pd.DataFrame(records).set_index("factor")


def build_composites(
    panel: pd.DataFrame,
    factor_cols: List[str],
    weights_by_method: Dict[str, pd.Series],
) -> Dict[str, pd.DataFrame]:
    """
    基于不同的加权方案构建合成因子信号。
    
    公式: 合成得分 = Sum(权重_i * 因子值_i)
    
    Args:
        panel: 原始数据面板
        factor_cols: 因子列名
        weights_by_method: 不同方法计算出的权重 (Method Name -> Weight Series)
        
    Returns:
        Dict: 方法名 -> 包含合成得分的 DataFrame
    """
    composites = {}
    for name, weights in weights_by_method.items():
        # 对齐权重与因子列，确保顺序一致，缺失权重填充为 0
        aligned = weights.reindex(factor_cols).fillna(0.0)
        
        # 矩阵乘法高效计算加权和 (Vectorized Operation)
        # 矩阵形状: (样本数 N x 因子数 K) @ (因子数 K x 1) -> (样本数 N x 1)
        # 这比循环每一行计算要快得多
        composite = panel[factor_cols].values @ aligned.values
        
        df = panel[["date", "asset"]].copy()
        df["composite"] = composite
        composites[name] = df
    return composites


def run(
    output_dir: str = "./outputs/day13_multifactor",
    data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = 240,
    n_groups: int = 10,
) -> None:
    """
    主执行流程:
    1. 加载真实数据 -> 2. 预处理 (Z-Score) -> 3. 评估因子 -> 4. 计算权重 -> 5. 合成信号
    """
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

    # 2. 预处理: Z-Score 标准化
    # 在组合因子之前至关重要的一步。确保所有因子在同一尺度上（均值为0，标准差为1）。
    panel = zscore_by_date(panel, factor_cols)
    panel = panel.dropna(subset=factor_cols + ["ret"])  # 确保训练样本有效

    # 3. 评估单因子表现
    metrics = compute_factor_metrics(panel, factor_cols, "ret", n_groups)

    # 4. 使用不同方法计算权重
    # normalize_weights 确保 sum(abs(weights)) = 1 或 sum(weights) = 1 (取决于具体实现)
    weights_by_method = {
        # 等权: 每个因子权重为 1/N
        "equal": normalize_weights(pd.Series(1.0, index=factor_cols)),
        
        # IC 加权: 根据预测能力 (相关性) 加权
        "ic": normalize_weights(metrics["ic_mean"]),
        
        # ICIR 加权: 根据风险调整后的预测能力 (稳定性) 加权
        "ic_ir": normalize_weights(metrics["ic_ir"]),
        
        # 收益差加权: 根据历史盈利能力加权
        "ret": normalize_weights(metrics["ret_mean"]),
    }

    # 5. 构建合成因子
    composites = build_composites(panel, factor_cols, weights_by_method)

    # 6. 保存结果
    out = ensure_dir(output_dir)
    # 保存因子绩效指标
    metrics.reset_index().to_csv(out / "factor_metrics.csv", index=False)
    # 保存计算出的权重
    pd.DataFrame(weights_by_method).T.to_csv(out / "weights_by_method.csv", index_label="method")
    # 保存合成信号
    for name, df in composites.items():
        df.to_csv(out / f"composite_factor_{name}.csv", index=False)

    print(f"分析完成。结果已保存至 {out}")


if __name__ == "__main__":
    run()
