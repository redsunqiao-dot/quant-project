"""
多因子回测验证与归因分析 (Multi-Factor Backtest Validation & Attribution)。

知识点 (Knowledge Points):
1.  **回测验证 (Backtest Validation)**:
    *   **目的**: 验证“合成因子”在考虑了交易成本、滑点和调仓规则后，是否仍能产生超额收益。
    *   **方法**: 使用与 Day8 相同的标准回测引擎 (BacktestEngine) 运行策略，确保评价标准一致，而非简化的纯收益叠加。

2.  **滚动窗口权重 (Rolling Window Weights)**:
    *   **动态调整**: 市场环境在变，因子权重也应随之调整。我们使用滚动窗口（如过去 60 天）计算因子的 IC_IR，以此动态确定权重。
    *   **优势**: 相比固定权重，能更快适应市场风格切换（例如从价值风格切换到动量风格）。

3.  **归因分析 (Attribution Analysis)**:
    *   **问题**: 策略赚钱了，到底是哪个因子贡献的？
    *   **方法**: 将合成因子的多空收益分解为各个单因子的贡献。
    *   **公式**: 因子贡献 = 因子权重 * (因子多头均值 - 因子空头均值)。这实际上是 Brinson 归因模型在因子投资中的一种简化应用。

4.  **Raw vs Neutralized**:
    *   **对比**: 比较原始数据 (Raw) 和中性化后数据 (Neutralized) 的回测结果。
    *   **意义**: 验证中性化（如行业中性、市值中性）是否真的提升了夏普比率和纯 Alpha 能力。通常中性化会降低总收益（因为过滤了Beta），但能显著降低波动和回撤，提高夏普比率。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from src.common.utils import (  # type: ignore
        calc_ic_summary,
        ensure_dir,
        infer_factor_columns,
        industry_neutralize_by_date,
        load_real_panel,
        zscore_by_date,
    )
except ImportError:  # pragma: no cover
    from .utils import (  # type: ignore
        calc_ic_summary,
        ensure_dir,
        infer_factor_columns,
        industry_neutralize_by_date,
        load_real_panel,
        zscore_by_date,
    )

from src.backtest.backtest_engine_strategy import BacktestEngine
from src.strategy.strategy_base import Strategy

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data")
DEFAULT_OUTPUT_MULTIFACTOR = str(PROJECT_ROOT / "outputs" / "day13_multifactor")
DEFAULT_OUTPUT_REPORT = str(PROJECT_ROOT / "outputs" / "day13_report")

# ---------------------------------------------------------------------------
# Utilities (工具函数)
# ---------------------------------------------------------------------------


def normalize_signed_weights(raw: pd.Series) -> pd.Series:
    """
    归一化权重但保留符号。
    
    目的:
    预测时，因子的方向（正向/负向）很重要。normalize_signed_weights 确保权重绝对值之和为1，
    但保留原始权重的正负号，这样合成因子就能正确反映各单因子的多空方向。

    Args:
        raw: 原始权重 Series

    Returns:
        pd.Series: 归一化后的权重
    """
    values = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    total = values.abs().sum()
    if total == 0:
        # 如果总和为0，分配等权
        return pd.Series(np.full(len(values), 1.0 / len(values)), index=values.index)
    return values / total


def compute_rolling_weights(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    lookback: int,
    min_history: int,
    metric: str = "ic_ir",
) -> pd.DataFrame:
    """
    估计滚动因子权重 (Rolling Weights Estimation)。

    逻辑:
    1. 在每个时间点，回溯过去 `lookback` 天的数据。
    2. 计算每个因子在这段时间内的绩效指标 (默认使用 IC_IR)。
    3. 根据指标归一化生成权重。
    
    Args:
        panel: 包含因子和收益的数据面板
        factor_cols: 因子列表
        lookback: 回溯窗口长度 (天)
        min_history: 最小所需历史数据 (天)
        metric: 权重依据指标 ("ic" 或 "ic_ir")

    Returns:
        pd.DataFrame: 包含 'date' 和各因子权重的 DataFrame
    """

    dates = sorted(panel["date"].unique())
    records: List[Dict[str, float]] = []
    
    # 遍历每个交易日 (模拟伴随式回测)
    for idx, date in enumerate(dates):
        # 确定滚动窗口范围: [date - lookback, date)
        # 注意: 这里严格使用过去数据，不包含当日 (Idx)，防止 Look-ahead Bias
        start = max(0, idx - lookback)
        history_dates = dates[start:idx]
        
        # 数据不足则跳过
        if len(history_dates) < max(1, min_history):
            continue

        subset = panel[panel["date"].isin(history_dates)]
        if subset.empty:
            continue

        metrics = {}
        for col in factor_cols:
            # 计算该因子在历史窗口内的 IC 表现
            stats = calc_ic_summary(subset, col, "ret")
            if metric == "ic":
                metrics[col] = stats["ic_mean"]
            else:
                # IC_IR = IC Mean / IC Std
                # 侧重于奖励发挥稳定的因子 (Stable Alpha)
                metrics[col] = stats["ic_ir"]

        # 归一化权重 (保持符号)
        weights = normalize_signed_weights(pd.Series(metrics, index=factor_cols))
        
        record: Dict[str, float] = {"date": date}
        for col in factor_cols:
            record[col] = float(weights.get(col, 0.0))
        records.append(record)

    weights_df = pd.DataFrame(records).sort_values("date")
    return weights_df.reset_index(drop=True)


def build_signals_and_stats(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    weights_df: pd.DataFrame,
    top_n: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    生成每日合成信号并进行归因统计。
    
    核心步骤:
    1. 合成: Composite_Score = Sum(Weight_i * Factor_Value_i)
    2. 选股: 根据合成得分排序，生成 Long/Short 组合。
    3. 归因: 分解组合收益来源。
    
    Args:
        panel: 数据面板
        factor_cols: 因子列表
        weights_df: 每日权重表
        top_n: 多空组合选股数量

    Returns:
        signals: 每日的选股信号 (BacktestEngine 需要的输入)。
        contrib_df: 每个因子对当日多空收益的贡献分解。
        ls_df: 合成因子的多空组合表现。
    """

    panel_by_date = {date: df.copy() for date, df in panel.groupby("date")}
    signals: Dict[str, pd.DataFrame] = {}
    contrib_records: List[Dict[str, float]] = []
    ls_records: List[Dict[str, float]] = []

    for _, row in weights_df.iterrows():
        date = row["date"]
        # 获取当日（实际上是基于历史计算出的）目标权重
        weights = row[list(factor_cols)].astype(float)
        
        day_panel = panel_by_date.get(date)
        if day_panel is None or day_panel.empty:
            continue

        # 1. 计算合成因子得分: Matrix(N_stocks x N_factors) @ Vector(N_factors x 1)
        day_panel = day_panel.copy()
        day_panel["factor_value"] = day_panel[list(factor_cols)].values @ weights.values
        day_panel = day_panel.dropna(subset=["factor_value", "ret"])
        if day_panel.empty:
            continue

        # 保存信号用于回测引擎 (BacktestEngine)
        signals[date] = (
            day_panel[["asset", "factor_value"]]
            .rename(columns={"asset": "code"})
            .assign(date=date)
        )

        # 2. 构建多空组合进行分析 (Long-Short Portfolio)
        ranked = day_panel.sort_values("factor_value", ascending=False)
        long = ranked.head(top_n)
        short = ranked.tail(top_n)
        if long.empty or short.empty:
            continue

        long_ret = long["ret"].mean()
        short_ret = short["ret"].mean()
        
        # 记录多空收益
        ls_records.append(
            {
                "date": date,
                "long_ret": float(long_ret),
                "short_ret": float(short_ret),
                "ls_ret": float(long_ret - short_ret),
            }
        )

        # 3. 归因分析 (Attribution)
        # 某个因子的贡献 = 权重 * (该因子在多头组的均值 - 该因子在空头组的均值)
        # 解释：如果我们在多头组选了高动量股票，空头组选了低动量股票，那么动量因子就对多空收益有正贡献。
        contrib = {"date": date, "ls_ret": float(long_ret - short_ret)}
        for col in factor_cols:
            # Spread: 多头组该因子均值 - 空头组该因子均值
            gap = float(long[col].mean() - short[col].mean())
            contrib[f"{col}_gap"] = gap
            # Contribution: Spread * Weight
            contrib[f"{col}_contribution"] = gap * float(weights[col])
            contrib[f"{col}_weight"] = float(weights[col])
        contrib_records.append(contrib)

    contrib_df = pd.DataFrame(contrib_records)
    ls_df = pd.DataFrame(ls_records)
    return signals, contrib_df, ls_df


def summarize_contributions(
    weights_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    factor_cols: Sequence[str],
    variant: str,
) -> pd.DataFrame:
    """
    汇总各因子的贡献度统计 (Factor Contribution Summary)。

    知识点:
    1. **因子归因 (Factor Attribution)**:
       - 在多因子模型中，我们需要知道每个因子对最终收益的贡献程度。
       - 这类似于投资组合管理中的 Brinson 归因模型，将总收益分解为各个决策维度的贡献。

    2. **贡献度计算**:
       - Score Gap: 多头组合与空头组合在该因子上的平均值差异（衡量因子的选股能力）
       - Contribution = Weight * Score Gap（衡量该因子对最终收益的实际贡献）
       - Contribution %: 该因子贡献占总贡献的比例

    3. **权重稳定性 (Weight Stability)**:
       - avg_weight: 该因子的平均权重，反映其长期重要性
       - weight_std: 权重的标准差，反映权重的波动程度
       - 理想情况下，重要因子的权重应该稳定，避免频繁大幅调整

    Args:
        weights_df: 包含各时间点因子权重的 DataFrame
        contrib_df: 包含各时间点因子贡献的 DataFrame
        factor_cols: 因子列表
        variant: 变体名称（如 "raw" 或 "neutralized"）

    Returns:
        pd.DataFrame: 因子贡献度汇总表，包含平均权重、权重波动、平均得分差、贡献度等指标
    """

    if weights_df.empty:
        return pd.DataFrame()

    gap_means = pd.Series(dtype=float)
    contrib_means = pd.Series(dtype=float)
    if not contrib_df.empty:
        gap_cols = [f"{col}_gap" for col in factor_cols]
        contrib_cols = [f"{col}_contribution" for col in factor_cols]
        gap_means = contrib_df[gap_cols].mean()
        contrib_means = contrib_df[contrib_cols].mean()

    summary_rows = []
    for col in factor_cols:
        summary_rows.append(
            {
                "variant": variant,
                "factor": col,
                "avg_weight": weights_df[col].mean(),
                "weight_std": weights_df[col].std(ddof=0),
                "avg_score_gap": gap_means.get(f"{col}_gap", np.nan),
                "score_contribution": contrib_means.get(f"{col}_contribution", np.nan),
            }
        )

    summary = pd.DataFrame(summary_rows)
    total = summary["score_contribution"].sum()
    if not np.isnan(total) and total != 0:
        summary["contribution_pct"] = summary["score_contribution"] / total
    else:
        summary["contribution_pct"] = np.nan
    return summary


class PrecomputedSignalStrategy(Strategy):
    """
    预计算信号策略 (Precomputed Signal Strategy)。

    设计模式: 适配器模式 (Adapter Pattern)
    - 作用: 将外部计算好的因子得分适配到 BacktestEngine 的标准接口

    核心优势:
    1. **关注点分离 (Separation of Concerns)**:
       - 因子计算逻辑与回测引擎逻辑解耦
       - 可以独立测试因子合成逻辑和回测引擎逻辑

    2. **复用回测引擎 (Reuse Backtest Infrastructure)**:
       - 不需要重复实现交易成本、滑点、调仓逻辑
       - 确保评价标准与 Day8 的单因子回测一致
       - 避免"回测引擎不一致"导致的评价偏差

    3. **灵活性 (Flexibility)**:
       - 可以对同一组信号使用不同的回测参数（如 top_n、调仓频率）
       - 可以对信号进行后处理（如平滑、过滤）而不影响因子计算

    使用场景:
    - 多因子合成后的回测验证
    - 机器学习模型预测结果的回测
    - 外部信号源（如研报推荐）的回测
    """

    def __init__(self, name: str, signals: Dict[str, pd.DataFrame]):
        super().__init__(name)
        self.signals = signals

    def calculate_factor(self, date: str, data_loader, **kwargs) -> pd.DataFrame:  # noqa: D401
        """
        计算因子值（实际上是查表返回预计算的信号）。

        知识点:
        - BacktestEngine 会在每个交易日调用此方法获取因子值
        - 我们直接从预先计算好的字典中查询，避免重复计算
        - 这是一种"缓存策略"，提高了回测效率

        Args:
            date: 交易日期
            data_loader: 数据加载器（此处未使用，因为信号已预计算）
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含 code, date, factor_value 列的信号表
        """
        # 直接查表获取当日信号
        df = self.signals.get(date)
        if df is None:
            return pd.DataFrame(columns=["code", "date", "factor_value"])
        return df.copy()

    def generate_signal(self, factor_df: pd.DataFrame, top_n: int = 10) -> List[str]:  # noqa: D401
        """
        生成交易信号（选股）。

        知识点 - 选股逻辑 (Stock Selection):
        1. **排序 (Ranking)**: 按因子值降序排列
        2. **截断 (Truncation)**: 取前 top_n 只股票
        3. **等权 (Equal Weight)**: BacktestEngine 默认对选中的股票等权配置

        为什么用等权而非因子值加权？
        - 简单稳健，避免因子值异常导致的极端仓位
        - 减少换手率（因子值微小变化不会导致仓位大幅调整）
        - 学术研究表明，等权组合的长期表现往往不输于优化加权

        Args:
            factor_df: 因子数据表
            top_n: 选股数量

        Returns:
            List[str]: 选中的股票代码列表
        """
        if factor_df.empty:
            return []
        # 标准选股逻辑: 取 Top N
        ranked = factor_df.sort_values("factor_value", ascending=False)
        return ranked.head(top_n)["code"].tolist()


def _series_max_drawdown(returns: pd.Series) -> float:
    """
    计算最大回撤 (Maximum Drawdown, MDD)。

    知识点:
    1. **最大回撤定义**:
       - MDD 衡量从历史最高点到最低点的最大损失幅度
       - 公式: MDD = min((当前净值 / 历史最高净值) - 1)

    2. **为什么重要**:
       - 风险度量: MDD 反映策略在最糟糕情况下的损失
       - 心理承受: 投资者更关注"最大回撤"而非波动率
       - Calmar Ratio: 年化收益 / MDD，衡量"收益回撤比"

    3. **计算步骤**:
       - 累计净值 = (1 + 每日收益率).cumprod()
       - 历史最高净值 = 累计净值.cummax()
       - 回撤序列 = 累计净值 / 历史最高净值 - 1
       - 最大回撤 = 回撤序列的最小值（负数，绝对值越大回撤越大）

    Args:
        returns: 收益率序列

    Returns:
        float: 最大回撤（负数，如 -0.25 表示 25% 的回撤）
    """
    if returns.empty:
        return np.nan
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return float(drawdown.min())


def run_backtest_for_variant(
    variant: str,
    signals: Dict[str, pd.DataFrame],
    data_dir: str,
    top_n: int,
    rebalance_freq: int,
) -> dict:
    """
    对指定变体（如 Raw 或 Neutralized）运行标准回测 (Standard Backtest)。

    知识点:
    1. **回测引擎统一性 (Backtest Consistency)**:
       - 使用与 Day8 相同的 BacktestEngine，确保评价标准一致
       - 避免"因子评价用简化模型，回测用完整模型"导致的结果偏差
       - 这是量化研究中的重要原则：Research-Production Parity

    2. **交易成本建模 (Transaction Cost Modeling)**:
       - enable_cost=True: 启用交易成本模拟
       - 包括：佣金、印花税、冲击成本（滑点）
       - 真实策略的收益 ≈ 理论收益 - 交易成本
       - 忽略成本的回测结果会严重高估策略表现

    3. **调仓频率 (Rebalance Frequency)**:
       - 高频调仓: 更快适应市场，但交易成本更高
       - 低频调仓: 降低成本，但可能错过市场机会
       - 需要在"信息捕获"和"成本控制"之间权衡
       - 常见选择: 5 日（周度）或 20 日（月度）

    4. **IC 计算 (Information Coefficient)**:
       - calculate_ic=True: 计算因子预测能力指标
       - IC = 因子值与未来收益的相关系数
       - 用于验证因子的预测性是否在回测中得到体现

    Args:
        variant: 变体名称（如 "raw" 或 "neutralized"）
        signals: 预计算的交易信号字典 {日期: 信号表}
        data_dir: 数据目录
        top_n: 每次调仓选股数量
        rebalance_freq: 调仓频率（天数）

    Returns:
        dict: 回测报告，包含收益、风险、IC 等指标
    """
    if not signals:
        raise ValueError(f"Variant {variant} does not have any trading signals")

    available_dates = sorted(signals.keys())
    engine = BacktestEngine(data_dir=data_dir)
    # 使用预计算信号策略
    strategy = PrecomputedSignalStrategy(name=f"Composite_{variant}", signals=signals)
    
    # 运行标准回测引擎
    # 启用 enable_cost=True 以模拟真实交易成本
    report = engine.run(
        start_date=available_dates[0],
        end_date=available_dates[-1],
        strategy=strategy,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
        enable_cost=True,  # 关键: 开启成本模拟
        calculate_ic=True,
    )
    return report


def build_backtest_record(
    variant: str,
    report: dict,
    ls_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    将回测报告转换为扁平的记录格式，方便保存为 CSV (Flatten Backtest Report)。

    知识点 - 关键回测指标解读:

    1. **收益指标 (Return Metrics)**:
       - total_return: 总收益率（期间累计收益）
       - annual_return: 年化收益率 = (1 + total_return)^(252/交易日数) - 1
       - 年化便于不同期间策略的横向对比

    2. **风险指标 (Risk Metrics)**:
       - annual_volatility: 年化波动率 = 日收益标准差 * sqrt(252)
       - max_drawdown: 最大回撤（见 _series_max_drawdown 函数）
       - 波动率衡量收益的不确定性，回撤衡量极端损失

    3. **风险调整收益 (Risk-Adjusted Return)**:
       - sharpe_ratio: 夏普比率 = (年化收益 - 无风险利率) / 年化波动率
         * 衡量"每单位风险获得多少超额收益"
         * 通常 > 1 为良好，> 2 为优秀
       - calmar_ratio: 卡玛比率 = 年化收益 / abs(最大回撤)
         * 衡量"每单位回撤获得多少收益"
         * 对回撤敏感的投资者更关注此指标

    4. **胜率指标 (Win Rate Metrics)**:
       - win_rate: 盈利交易日占比
       - ic_win_rate: IC > 0 的交易日占比
       - 高胜率不一定高收益（可能是"赚小亏大"）

    5. **换手率 (Turnover)**:
       - avg_turnover: 平均每日持仓变动比例
       - 高换手率意味着高交易成本
       - 换手率 = 0.5 表示每天有 50% 的持仓发生变化

    6. **IC 指标 (Information Coefficient)**:
       - ic_mean: 平均 IC，衡量因子预测能力的均值
       - ic_std: IC 标准差，衡量因子预测能力的稳定性
       - ir (Information Ratio): IC_IR = ic_mean / ic_std
         * 类似于夏普比率，衡量"信息质量"
         * IR > 0.5 通常认为是好因子

    7. **多空组合指标 (Long-Short Portfolio Metrics)**:
       - ls_mean: 多空收益均值（多头收益 - 空头收益）
       - ls_vol: 多空收益波动率
       - ls_cum_return: 多空累计收益
       - ls_max_drawdown: 多空最大回撤
       - ls_win_rate: 多空正收益日占比
       - 多空组合剔除了市场整体波动（Beta），更能体现因子的 Alpha

    Args:
        variant: 变体名称
        report: BacktestEngine 返回的回测报告
        ls_df: 多空组合收益数据

    Returns:
        Dict[str, float]: 扁平化的回测指标记录
    """
    record: Dict[str, float] = {"variant": variant}

    returns = report.get("daily_returns")
    if isinstance(returns, pd.Series) and not returns.empty:
        record["start_date"] = returns.index[0]
        record["end_date"] = returns.index[-1]
    else:
        record["start_date"] = None
        record["end_date"] = None

    keys = [
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "avg_turnover",
        "ic_mean",
        "ic_std",
        "ir",
        "ic_win_rate",
    ]
    for key in keys:
        record[key] = float(report.get(key, np.nan)) if key in report else np.nan

    if ls_df is not None and not ls_df.empty:
        ls_series = ls_df.set_index("date")["ls_ret"].astype(float)
        record["ls_mean"] = ls_series.mean()
        record["ls_vol"] = ls_series.std(ddof=0)
        record["ls_cum_return"] = (1 + ls_series).prod() - 1
        record["ls_max_drawdown"] = _series_max_drawdown(ls_series)
        record["ls_win_rate"] = (ls_series > 0).mean()
        record["long_mean"] = ls_df["long_ret"].mean()
        record["short_mean"] = ls_df["short_ret"].mean()
    else:
        record.update(
            {
                "ls_mean": np.nan,
                "ls_vol": np.nan,
                "ls_cum_return": np.nan,
                "ls_max_drawdown": np.nan,
                "ls_win_rate": np.nan,
                "long_mean": np.nan,
                "short_mean": np.nan,
            }
        )

    return record


def export_multifactor_outputs(
    output_dir: Path,
    weights_ts: pd.DataFrame,
    contrib_summary: pd.DataFrame,
    backtest_compare: pd.DataFrame,
) -> None:
    """
    导出多因子分析的中间结果 (Export Multifactor Outputs)。

    输出文件说明:
    1. **weights_timeseries.csv**: 权重时间序列
       - 记录每个交易日各因子的权重
       - 用于分析权重的动态变化和稳定性
       - 可视化权重演变，观察市场风格切换

    2. **contribution_summary.csv**: 因子贡献度汇总
       - 各因子对收益的平均贡献度
       - 权重的均值和标准差
       - 用于识别"关键因子"和"冗余因子"

    3. **backtest_compare.csv**: 回测结果对比
       - Raw vs Neutralized 的性能对比
       - 完整的回测指标（收益、风险、IC、换手等）
       - 用于验证数据预处理（标准化、中性化）的有效性

    Args:
        output_dir: 输出目录
        weights_ts: 权重时间序列表
        contrib_summary: 因子贡献度汇总表
        backtest_compare: 回测对比结果表
    """
    ensure_dir(output_dir)

    if not weights_ts.empty:
        cols = ["variant", "date"] + [c for c in weights_ts.columns if c not in {"variant", "date"}]
        weights_ts[cols].to_csv(output_dir / "weights_timeseries.csv", index=False)

    if not contrib_summary.empty:
        contrib_summary.to_csv(output_dir / "contribution_summary.csv", index=False)

    if not backtest_compare.empty:
        backtest_compare.to_csv(output_dir / "backtest_compare.csv", index=False)


def export_report(
    report_dir: Path,
    summary_df: pd.DataFrame,
    contrib_summary: pd.DataFrame,
    weights_ts: pd.DataFrame,
) -> None:
    """
    导出 HTML 格式的回测分析报告 (Export HTML Report)。

    报告结构:
    1. **回测对比表**: Raw vs Neutralized 的完整指标对比
    2. **因子贡献度**: 各因子对收益的贡献分析
    3. **权重稳定性**: 权重换手率统计（均值、标准差、最大值）

    设计理念:
    - 自包含: HTML 文件包含所有样式，无需外部 CSS
    - 简洁清晰: 使用表格展示，便于快速浏览
    - 可打印: 适合导出为 PDF 或打印存档

    Args:
        report_dir: 报告输出目录
        summary_df: 回测汇总表
        contrib_summary: 因子贡献度表
        weights_ts: 权重时间序列表
    """
    ensure_dir(report_dir)
    summary_path = report_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # 将指标分组，避免表格过宽
    perf_cols = ["variant", "start_date", "end_date", "total_return", "annual_return",
                 "annual_volatility", "sharpe_ratio", "max_drawdown", "calmar_ratio",
                 "win_rate", "avg_turnover"]
    ic_cols = ["variant", "ic_mean", "ic_std", "ir", "ic_win_rate"]
    ls_cols = ["variant", "ls_mean", "ls_vol", "ls_cum_return", "ls_max_drawdown",
               "ls_win_rate", "long_mean", "short_mean"]

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Day13 Multifactor Report</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;margin-bottom:24px;}"
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:right;}"
        "th{text-align:center;background:#f6f6f6;}"
        "h1,h2,h3{font-weight:600;}"
        "h3{font-size:16px;margin-top:20px;margin-bottom:10px;color:#333;}</style></head><body>",
        "<h1>Day13 多因子评价报告</h1>",
        "<p>与 Day8 引擎保持一致：成本、滑点、调仓频率统一，使得因子评估结果能够在回测验证阶段对齐。</p>",
        "<h2>回测对比 (Raw vs Industry Neutralized)</h2>",
    ]

    # 1. 绩效指标表
    if all(col in summary_df.columns for col in perf_cols):
        html_parts.append("<h3>1. 回测绩效指标 (Performance Metrics)</h3>")
        html_parts.append(summary_df[perf_cols].to_html(index=False, float_format="%.4f"))

    # 2. IC指标表
    if all(col in summary_df.columns for col in ic_cols):
        html_parts.append("<h3>2. 因子IC指标 (IC Metrics)</h3>")
        html_parts.append(summary_df[ic_cols].to_html(index=False, float_format="%.4f"))

    # 3. 多空分析表
    if all(col in summary_df.columns for col in ls_cols):
        html_parts.append("<h3>3. 多空组合分析 (Long-Short Analysis)</h3>")
        html_parts.append(summary_df[ls_cols].to_html(index=False, float_format="%.4f"))

    if not contrib_summary.empty:
        html_parts.append("<h2>因子贡献度</h2>")
        html_parts.append(contrib_summary.to_html(index=False, float_format="%.4f"))

    if not weights_ts.empty:
        turnover = (
            weights_ts.groupby("variant")["weight_turnover"]
            .agg(["mean", "std", "max"])
            .reset_index()
        )
        html_parts.append("<h2>权重时序稳定性</h2>")
        html_parts.append(turnover.to_html(index=False, float_format="%.4f"))

    html_parts.append("</body></html>")
    (report_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")


def run(
    data_dir: str = DEFAULT_DATA_DIR,
    output_multifactor: str = DEFAULT_OUTPUT_MULTIFACTOR,
    output_report: str = DEFAULT_OUTPUT_REPORT,
    ret_col: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = 260,
    lookback: int = 60,
    min_history: int = 40,
    top_n: int = 30,
    rebalance_freq: int = 5,
) -> None:
    """
    执行完整的 Day13 多因子回测验证流程 (Full Multi-Factor Validation Workflow)。

    流程概览:
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. 数据加载与预处理                                          │
    │    - 加载因子和收益数据                                      │
    │    - 生成 Raw 和 Neutralized 两个变体                       │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 2. 滚动窗口权重估计 (Rolling Weight Estimation)              │
    │    - 对每个交易日，回溯 lookback 天计算因子 IC_IR           │
    │    - 根据 IC_IR 归一化生成因子权重                           │
    │    - 记录权重时间序列，用于分析权重稳定性                    │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 3. 合成信号与归因分析 (Signal Composition & Attribution)     │
    │    - 合成因子得分 = Σ(权重_i × 因子值_i)                    │
    │    - 生成多空组合选股信号                                    │
    │    - 计算各因子对多空收益的贡献                              │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 4. 标准回测引擎验证 (Backtest Engine Validation)             │
    │    - 使用 Day8 的 BacktestEngine 运行策略                   │
    │    - 启用交易成本、滑点等真实因素                            │
    │    - 计算完整的回测指标（收益、风险、IC、换手等）            │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 5. 结果汇总与报告导出                                        │
    │    - 对比 Raw vs Neutralized 的表现                         │
    │    - 生成因子贡献度报告                                      │
    │    - 输出 CSV 数据文件和 HTML 可视化报告                    │
    └─────────────────────────────────────────────────────────────┘

    核心知识点:

    1. **变体对比 (Variant Comparison)**:
       - Raw: 原始因子数据，包含市值、行业等结构性偏差
       - Neutralized: 截面标准化后的数据，剔除了系统性偏差
       - 对比目的: 验证数据预处理是否提升了因子表现

    2. **滚动权重 vs 固定权重 (Rolling vs Fixed Weights)**:
       - 固定权重: 一次性计算全样本权重，整个回测期使用相同权重
         * 优点: 简单稳定
         * 缺点: 无法适应市场风格变化（如价值-成长轮动）
       - 滚动权重: 每个时间点基于过去数据动态调整权重
         * 优点: 适应市场变化，类似"在线学习"
         * 缺点: 可能过度拟合短期波动

    3. **Look-Ahead Bias 防范 (Avoiding Look-Ahead Bias)**:
       - 严格使用"过去数据"计算权重: [t-lookback, t) 计算权重用于 t 日
       - 不包含当日数据，防止"未来信息泄露"
       - 这是量化回测的金科玉律：Never use future data

    4. **归因分析的价值 (Value of Attribution)**:
       - 识别"关键因子": 哪些因子贡献最大？
       - 发现"冗余因子": 哪些因子权重低且贡献小？
       - 优化方向: 剔除冗余因子，降低模型复杂度

    5. **评价标准一致性 (Evaluation Consistency)**:
       - 因子评价阶段（Day10-12）使用 IC、IC_IR 等指标
       - 回测验证阶段（Day13）使用标准回测引擎
       - 两阶段使用相同的数据处理和评价框架，确保结果可信

    参数说明:
        data_dir: 数据目录，包含因子和收益数据
        output_multifactor: 多因子分析结果输出目录
        output_report: HTML 报告输出目录
        ret_col: 收益率列名（如 "1vwap_pct" 表示次日 VWAP 收益率）
        start_date: 回测起始日期（None 表示使用数据中的最早日期）
        end_date: 回测结束日期（None 表示使用数据中的最晚日期）
        max_dates: 最大回测天数（用于快速测试，None 表示使用全部数据）
        lookback: 滚动窗口长度（天），用于计算因子权重
        min_history: 最小历史数据长度（天），不足则跳过该日
        top_n: 多空组合选股数量（如 30 表示多头 30 只、空头 30 只）
        rebalance_freq: 调仓频率（天），如 5 表示每 5 个交易日调仓一次

    输出文件:
        {output_multifactor}/weights_timeseries.csv: 权重时间序列
        {output_multifactor}/contribution_summary.csv: 因子贡献度汇总
        {output_multifactor}/backtest_compare.csv: 回测结果对比
        {output_report}/summary.csv: 回测指标汇总表
        {output_report}/report.html: HTML 可视化报告
    """

    data_dir = str(data_dir)
    output_multifactor = str(output_multifactor)
    output_report = str(output_report)

    # -------------------------------------------------------------------------
    # 1. 加载和预处理数据
    # -------------------------------------------------------------------------
    panel = load_real_panel(
        data_dir=data_dir,
        ret_col=ret_col,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )
    factor_cols = infer_factor_columns(panel)
    if not factor_cols:
        raise ValueError("No factor columns found in the panel data")

    # 定义两个对比变体: 原始数据 (Raw) vs 行业中性化数据 (Industry Neutralized)
    # Raw: 原始Barra因子，包含行业和市值等系统性偏差
    # Neutralized: 行业内标准化，消除行业间差异，提取纯Alpha
    variants = {
        "raw": panel.copy(),
        "neutralized": industry_neutralize_by_date(panel, factor_cols, data_dir),
    }

    weights_frames = []
    contrib_frames = []
    ls_by_variant: Dict[str, pd.DataFrame] = {}
    signals_by_variant: Dict[str, Dict[str, pd.DataFrame]] = {}

    # -------------------------------------------------------------------------
    # 2. 遍历变体，计算权重、信号和归因
    # -------------------------------------------------------------------------
    for variant, variant_panel in variants.items():
        print(f"\n[{variant}] Rolling weight estimation (滚动权重计算)...")
        weights_df = compute_rolling_weights(
            variant_panel,
            factor_cols,
            lookback=lookback,
            min_history=min_history,
            metric="ic_ir",
        )
        if weights_df.empty:
            print(f"[Skip] {variant} does not have enough history for weights")
            continue

        weights_df["variant"] = variant
        weight_cols = list(factor_cols)
        # 计算权重换手率: Sum(abs(Weight_t - Weight_{t-1}))
        weights_df["weight_turnover"] = weights_df[weight_cols].diff().abs().sum(axis=1)

        print(f"[{variant}] Building composite signals and attribution (合成信号与归因)...")
        signals, contrib_df, ls_df = build_signals_and_stats(
            variant_panel,
            factor_cols,
            weights_df,
            top_n=top_n,
        )
        if not signals:
            print(f"[Skip] {variant} does not produce valid signals")
            continue

        signals_by_variant[variant] = signals
        ls_df["variant"] = variant
        ls_by_variant[variant] = ls_df

        summary = summarize_contributions(weights_df, contrib_df, factor_cols, variant)
        if not summary.empty:
            contrib_frames.append(summary)

        weights_frames.append(weights_df)

    if not signals_by_variant:
        raise RuntimeError("No variants produced usable signals. Check the data inputs.")

    # -------------------------------------------------------------------------
    # 3. 运行回测引擎 (BacktestEngine)
    # -------------------------------------------------------------------------
    backtest_records: List[Dict[str, float]] = []
    for variant, signals in signals_by_variant.items():
        print(f"\n[{variant}] Running backtest through the Day8 engine (标准化回测)...")
        report = run_backtest_for_variant(
            variant=variant,
            signals=signals,
            data_dir=data_dir,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
        )
        ls_df = ls_by_variant.get(variant, pd.DataFrame())
        backtest_records.append(build_backtest_record(variant, report, ls_df))

    # -------------------------------------------------------------------------
    # 4. 汇总与导出
    # -------------------------------------------------------------------------
    weights_ts = pd.concat(weights_frames, ignore_index=True) if weights_frames else pd.DataFrame()
    contrib_summary = pd.concat(contrib_frames, ignore_index=True) if contrib_frames else pd.DataFrame()
    backtest_compare = pd.DataFrame(backtest_records)

    export_multifactor_outputs(
        output_dir=Path(output_multifactor),
        weights_ts=weights_ts,
        contrib_summary=contrib_summary,
        backtest_compare=backtest_compare,
    )

    export_report(
        report_dir=Path(output_report),
        summary_df=backtest_compare,
        contrib_summary=contrib_summary,
        weights_ts=weights_ts,
    )

    print("\n✅ Day13 multi-factor backtest + report pipeline completed.")
    print(f"Outputs saved to: {output_multifactor} and {output_report}")


if __name__ == "__main__":
    run()