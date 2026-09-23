"""
稳定性与过拟合检查 (Stability and Overfitting Checks)。

知识点 (Knowledge Points):
1.  **过拟合检查 (Overfitting Check)**:
    *   **概念**: 模型在训练数据 (In-Sample, IS) 上表现极佳，但在未见的测试数据 (Out-of-Sample, OOS) 上表现糟糕。
    *   **原因**: 模型捕捉了数据中的随机噪声 (Noise) 而非真实的 Alpha 信号。
    *   **检测方法**: 比较 Train IC (IS) 和 Test IC (OOS)。如果两者差异过大 (例如 Gap > 0.03)，或者 Test IC 趋于 0 甚至为负，说明模型大概率失效。

2.  **滚动窗口回测 (Rolling/Sliding Window / Walk-Forward Validation)**:
    *   **目的**: 模拟真实的动态调仓过程。因为因子的表现会随市场环境（如牛熊转换、风格切换）而变化，固定权重的静态回测不具代表性。
    *   **流程**: 
        (1) 使用 T0 ~ T80 的历史数据训练权重。
        (2) 在 T81 ~ T100 的“未来”数据上进行测试。
        (3) 窗口整体向后滑动 (例如步长 20 天)，重复上述过程。
    *   **优势**: 能更真实地反映策略在不同市场时期的适应性和稳健性。

3.  **权重稳定性 (Weight Stability)**:
    *   **重要性**: 即使一个策略 OOS 收益很高，如果权重每天剧烈跳变（今天买入，明天全部卖出），巨大的换手成本 (Transaction Costs) 会吞噬所有利润。
    *   **衡量**: 使用权重换手率 (Turnover) 衡量。平均换手率 > 20% 通常被认为是不可接受的（对于低频策略）。

4.  **权重平滑 (EWMA Smoothing)**:
    *   **公式**: Final_Weight_t = alpha * Target_Weight_t + (1 - alpha) * Final_Weight_{t-1}
    *   **作用**: 过滤掉因子权重的短期波动噪声，降低换手率，增加策略的可落地性。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from .utils import (  # type: ignore
        calc_ic_by_date,
        calc_ic_summary,
        ensure_dir,
        ewma_smooth,
        infer_factor_columns,
        load_real_panel,
        normalize_weights,
        time_split_dates,
        weight_turnover,
        zscore_by_date,
    )
except ImportError:  # pragma: no cover
    # 为了支持直接运行脚本时的相对导入降级
    from src.common.utils import (  # type: ignore
        calc_ic_by_date,
        calc_ic_summary,
        ensure_dir,
        ewma_smooth,
        infer_factor_columns,
        load_real_panel,
        normalize_weights,
        time_split_dates,
        weight_turnover,
        zscore_by_date,
    )


def estimate_ic_weights(panel: pd.DataFrame, factor_cols: List[str]) -> pd.Series:
    """
    根据给定的训练数据，计算 IC 加权权重。
    
    原理:
    IC (Information Coefficient) 衡量了因子预测收益的能力。
    使用 IC 作为权重 (IC Weighting) 是一种简单有效的线性组合方法：
    表现越好的因子，分配的权重越大。

    Args:
        panel: 包含因子和收益率的面板数据
        factor_cols: 因子列名列表

    Returns:
        归一化后的权重 Series (sum = 1)
    """
    metrics = {}
    for col in factor_cols:
        # 计算该因子在训练集上的平均 IC
        metrics[col] = calc_ic_summary(panel, col, "ret")["ic_mean"]
    
    # 归一化权重，确保权重之和为 1 (或绝对值之和为 1，取决于 normalize_weights 实现)
    return normalize_weights(pd.Series(metrics))


def composite_ic(panel: pd.DataFrame, factor_cols: List[str], weights: pd.Series) -> float:
    """
    计算合成因子在给定数据集上的平均 IC。
    用于评估训练集 (IS) 和测试集 (OOS) 的预测能力。

    Args:
        panel: 数据面板
        factor_cols: 因子列表
        weights: 因子权重

    Returns:
        float: 平均 Rank IC
    """
    # 确保权重与因子列对齐，缺失填充 0
    aligned = weights.reindex(factor_cols).fillna(0.0)
    tmp = panel.copy()

    # 计算合成因子得分 (Composite Score)
    # 矩阵乘法: (N_samples x N_factors) @ (N_factors x 1) -> (N_samples x 1)
    tmp["composite"] = tmp[factor_cols].values @ aligned.values

    # 计算合成因子与下期收益率的 Rank IC (Spearman Correlation)
    ic = calc_ic_by_date(tmp, "composite", "ret")
    
    # 返回 IC 均值
    return ic.mean() if not ic.empty else float("nan")


def calculate_backtest_return(panel: pd.DataFrame, factor_cols: List[str], weights: pd.Series) -> dict:
    """
    计算基于合成因子的简单分层回测指标 (Simple Layered Backtest)。
    注意：这只是一个简化的无成本回测，用于快速评估模型区分度。
    
    逻辑:
    1. 计算每个股票的合成得分。
    2. 每日按得分排序。
    3. 做多 Top 20% (Top Quintile) 的股票。
    4. 统计该组合的日均收益。

    Returns:
        包含累计收益、年化收益、夏普比率、最大回撤的字典。
    """
    import numpy as np

    aligned = weights.reindex(factor_cols).fillna(0.0)
    tmp = panel.copy()

    # 1. 生成合成信号
    tmp["composite"] = tmp[factor_cols].values @ aligned.values

    # 2. 模拟每日调仓
    daily_returns = []
    # 遍历每一天
    for date in tmp["date"].unique():
        day_data = tmp[tmp["date"] == date].copy()
        if len(day_data) < 10:  # 样本量太少，统计意义不足，跳过
            continue

        # 3. 核心逻辑：做多头部
        # 按合成因子降序排序，取前 20% 的股票
        day_data = day_data.sort_values("composite", ascending=False)
        top_n = max(1, len(day_data) // 5) # Top 20%
        top_stocks = day_data.head(top_n)

        # 4. 计算这一天持仓组合的平均收益 (假设等权买入)
        mean_ret = top_stocks["ret"].mean()
        daily_returns.append(mean_ret)

    # 如果没有有效交易日，返回空指标
    if not daily_returns:
        return {
            "cumulative_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }

    # 5. 计算各项绩效指标
    # 累积收益: (1+r1)*(1+r2)... - 1
    cumulative_return = np.prod([1 + r for r in daily_returns]) - 1

    num_days = len(daily_returns)
    # 年化收益: 考虑复利效应，转换为每年收益率 (假设一年 252 个交易日)
    annualized_return = (1 + cumulative_return) ** (252 / num_days) - 1 if num_days > 0 else float("nan")

    # 夏普比率 (Sharpe Ratio): 衡量每单位风险带来的超额收益。
    # 公式: (均值 / 标准差) * sqrt(252)
    # 通常 > 1.0 为合格，> 2.0 为优秀
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else float("nan")

    # 最大回撤 (Max Drawdown): 历史上资产净值从最高点下跌的最大幅度。
    # 衡量策略的最坏情况风险。
    cumulative = np.cumprod([1 + r for r in daily_returns])
    running_max = np.maximum.accumulate(cumulative) # 历史最高点序列
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def detect_overfitting_signals(overfit_records: pd.DataFrame, turnover: pd.Series) -> dict:
    """
    【核心函数】检测过拟合的危险信号 (Red Flags)。
    
    诊断逻辑:
    1. IC Gap (IC 衰减): 训练 IC 远高于测试 IC。
       如果 Train IC = 0.10, Test IC = 0.02，说明模型只是记住了历史答案 (Overfitting)，
       而没有学到通用规律。
    2. Test IC Value: 测试集 IC 为负或极低。
       如果 Test IC < 0，说明模型预测完全反了，或者失效。
    3. Turnover (换手率): 权重变换过快。
       如果权重每天剧烈波动 (Turnover > 20%)，说明模型对数据中的噪声过于敏感，
       且实盘中交易成本会极高。
    """
    signals = {
        "has_overfitting": False,
        "warnings": [],
        "details": {}
    }

    # 1. 计算 IC 指标
    train_ic_mean = overfit_records["train_ic"].mean()
    test_ic_mean = overfit_records["test_ic"].mean()
    # 计算衰减幅度
    ic_gap = train_ic_mean - test_ic_mean

    signals["details"]["train_ic_mean"] = train_ic_mean
    signals["details"]["test_ic_mean"] = test_ic_mean
    signals["details"]["ic_gap"] = ic_gap

    # 警告规则 1：训练/测试差异过大 (Gap > 0.03)
    if ic_gap > 0.03 and train_ic_mean > 0.05:
        signals["has_overfitting"] = True
        signals["warnings"].append(
            f"⚠️  训练 IC ({train_ic_mean:.4f}) 显著高于测试 IC ({test_ic_mean:.4f})，差异 {ic_gap:.4f}。存在过拟合风险 (High Generalization Error)。"
        )

    # 警告规则 2：测试集表现极差 (IC < 0.01)
    if test_ic_mean < 0.01:
        signals["has_overfitting"] = True
        signals["warnings"].append(
            f"⚠️  测试集 IC 过低 ({test_ic_mean:.4f})，模型在样本外几乎没有预测能力 (No Predictive Power)。"
        )

    # 2. 计算权重换手率
    if not turnover.empty:
        avg_turnover = turnover.mean()
        max_turnover = turnover.max()

        signals["details"]["avg_turnover"] = avg_turnover
        signals["details"]["max_turnover"] = max_turnover

        # 警告规则 3：策略太不稳定 (Avg Turnover > 20%)
        # 这意味着每次调仓平均要更换 20% 的权重，成本很高。
        if avg_turnover > 0.20:
            signals["has_overfitting"] = True
            signals["warnings"].append(
                f"⚠️  权重平均换手率过高 ({avg_turnover:.2%})，交易成本将吞噬收益 (High Turnover)。"
            )

    return signals


def run(
    output_dir: str = "./outputs/day13_multifactor",
    data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = 320,
) -> None:
    """
    执行滚动窗口分析流程 (Workflow)：
    1. 加载并预处理数据
    2. 滑动窗口切分 (Walk-Forward Split)
    3. IS 训练权重 (Estimate Weights)
    4. OOS 评估性能 (Evaluate OOS Performance)
    5. 汇总并诊断稳定性 (Diagnostics)
    """

    '''
    [80 train][20test]
    [20] [80 train][20test]
    [20][20][80 train][20test] 
    '''
    # -------------------------------------------------------------------------
    # 1. 准备数据
    # -------------------------------------------------------------------------
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

    # 横截面标准化 (Z-Score)
    panel = zscore_by_date(panel, factor_cols)
    panel = panel.dropna(subset=factor_cols + ["ret"])

    dates = sorted(panel["date"].unique())
    # 滚动窗口至少需要一定长度的历史数据
    if len(dates) < 120:
        raise ValueError("数据量太少，不足以进行滚动窗口分析 (至少需要 120 天)。")

    # -------------------------------------------------------------------------
    # 2. 生成滚动时间窗口 (Walk-Forward Setup)
    # -------------------------------------------------------------------------
    # train_size: 训练窗口长度 (如 80 天) —— 用于学习因子权重
    # test_size: 测试/预测窗口长度 (如 20 天) —— 用于验证策略效果
    # step: 每次滑动的步长 (如 20 天) —— 模拟时间推移
    splits = time_split_dates(dates, train_size=80, test_size=20, step=20)

    weight_records = []
    overfit_records = []

    # -------------------------------------------------------------------------
    # 3. 滚动模拟循环
    # -------------------------------------------------------------------------
    for idx, (train_dates, test_dates) in enumerate(splits, start=1):
        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"].isin(test_dates)]

        if train.empty or test.empty:
            continue

        # [Step A] 在 In-Sample (IS) 数据上训练模型权重
        # 这里使用简单的 IC 加权，也可以换成回归、优化器等
        weights = estimate_ic_weights(train, factor_cols)
        weight_records.append(pd.Series(weights, name=f"split_{idx}"))

        # [Step B] 评估性能：比较 IS 和 OOS 的表现
        # 计算 IS IC
        train_ic = composite_ic(train, factor_cols, weights)
        # 计算 OOS IC (关键指标)
        test_ic = composite_ic(test, factor_cols, weights)

        # 计算简单回测收益
        train_backtest = calculate_backtest_return(train, factor_cols, weights)
        test_backtest = calculate_backtest_return(test, factor_cols, weights)

        # 记录关键指标用于后续诊断
        overfit_records.append(
            {
                "split": idx,
                "train_ic": train_ic,  
                "test_ic": test_ic,   
                "train_return": train_backtest["cumulative_return"],
                "test_return": test_backtest["cumulative_return"],
                "test_sharpe": test_backtest["sharpe_ratio"],
                "test_max_dd": test_backtest["max_drawdown"],
            }
        )

    # -------------------------------------------------------------------------
    # 4. 结果汇总与过拟合诊断
    # -------------------------------------------------------------------------
    if not weight_records:
        raise ValueError("没有生成有效的回测窗口。")

    weights_df = pd.DataFrame(weight_records)
    overfit_df = pd.DataFrame(overfit_records)

    # 计算稳定性指标
    turnover = weight_turnover(weights_df)           # 原始权重换手率
    smoothed = ewma_smooth(weights_df, alpha=0.3)    # (可选) 计算平滑后的权重供参考

    # 打印诊断报告
    print("\n" + "=" * 80)
    print("【稳定性与过拟合检查】诊断报告 (Stability & Overfitting Report)")
    print("=" * 80)

    signals = detect_overfitting_signals(overfit_df, turnover)
    
    print(f"\n[1] 核心指标汇总 (Key Metrics):")
    print(f"    - 平均 IS IC (Train):  {signals['details']['train_ic_mean']:.4f}")
    print(f"    - 平均 OOS IC (Test):  {signals['details']['test_ic_mean']:.4f}")
    print(f"    - IC 差异 (Gap):       {signals['details']['ic_gap']:.4f}")
    print(f"    - 平均权重换手率 (Turnover): {signals['details'].get('avg_turnover', 0):.2%}")

    print(f"\n[2] 过拟合风险判定 (Risk Assessment):")
    if signals["has_overfitting"]:
        print("    状态: ❌ 检测到高风险 (High Risk)。建议简化模型或重新筛选因子。")
        for w in signals["warnings"]:
            print(f"    {w}")
    else:
        print("    状态: ✅ 表现稳健，未发现明显过拟合 (Stable)。")

    # -------------------------------------------------------------------------
    # 5. 保存结果
    # -------------------------------------------------------------------------
    out = ensure_dir(output_dir)
    weights_df.to_csv(out / "weights_raw.csv", index_label="split")
    overfit_df.to_csv(out / "overfit_checks.csv", index=False)
    
    # 保存文本诊断报告
    with open(out / "diagnostic_report.txt", "w", encoding="utf-8") as f:
        f.write("Diagnostic Report\n" + "="*20 + "\n")
        f.write(f"Has Overfitting: {signals['has_overfitting']}\n")
        for w in signals["warnings"]:
            f.write(f"- {w}\n")

    print(f"\n✓ 诊断完成，结果已保存至 {out}")


if __name__ == "__main__":
    run()