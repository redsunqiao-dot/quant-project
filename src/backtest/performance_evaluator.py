import pandas as pd
import numpy as np
from typing import List


class PerformanceEvaluator:
    """绩效评估器 - Version 1.0"""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化绩效评估器

        Args:
            risk_free_rate: 无风险利率(年化),默认3%
        """
        # 无风险利率通常参考国债收益率或银行定期存款利率，用于计算夏普比率等风险调整收益指标
        self.risk_free_rate = risk_free_rate
        print("✅ 绩效评估器初始化成功")

    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> dict:
        """
        计算最大回撤

        Args:
            cumulative_returns: 累计净值序列

        Returns:
            包含最大回撤及相关信息的字典
        """
        # 计算历史最高净值 (expanding max)
        # cummax() 会返回截止到当前位置的最大值序列，用于确定每一个时间点的"高水位线"
        running_max = cumulative_returns.cummax()

        # 计算回撤率
        # 公式: (当前净值 - 历史最高净值) / 历史最高净值
        # 结果通常为负数或0，0表示当前即为历史新高
        drawdown = (cumulative_returns - running_max) / running_max

        # 最大回撤 (取最小的负值，即回撤幅度最大的点)
        max_dd = drawdown.min()

        # 找到最大回撤发生的位置 (时间点)
        max_dd_idx = drawdown.idxmin()

        # 找到回撤开始位置(之前的最高点)
        # 在最大回撤点之前的时间段内，找到净值最高点的位置
        max_dd_start = cumulative_returns[:max_dd_idx].idxmax()

        return {
            'max_drawdown': abs(max_dd), # 转换为正数表示幅度，如 0.25 表示 25% 的回撤
            'max_dd_start': max_dd_start,
            'max_dd_end': max_dd_idx,
            'drawdown_series': drawdown
        }

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        计算夏普比率 (Sharpe Ratio)
        衡量每承担一单位总风险所能获得的超额回报。

        Args:
            returns: 日收益率序列
            periods_per_year: 每年的交易周期数(日频=252, 周频=52, 月频=12)

        Returns:
            夏普比率
        """
        # 年化收益率计算
        # 简单年化: 日均收益 * 交易天数 (假设单利)
        # 严谨做法可以是 (1 + 日均收益)^252 - 1，但在高频波动下简单年化是常用近似
        annual_return = returns.mean() * periods_per_year

        # 年化波动率计算
        # 波动率随时间的平方根扩展，因此乘以 sqrt(252)
        annual_volatility = returns.std() * np.sqrt(periods_per_year)

        # 夏普比率计算
        # 公式: (组合年化收益 - 无风险利率) / 组合年化波动率
        if annual_volatility > 0:
            sharpe = (annual_return - self.risk_free_rate) / annual_volatility
        else:
            sharpe = 0.0

        return sharpe

    def calculate_calmar_ratio(
        self,
        cumulative_returns: pd.Series,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        计算卡玛比率(Calmar Ratio)

        定义: 年化收益率 / 最大回撤
        意义: 每承担1%回撤风险,能获得多少收益。相比夏普比率，卡玛比率更关注尾部风险（最大回撤）。

        Args:
            cumulative_returns: 累计净值序列
            returns: 日收益率序列
            periods_per_year: 每年的交易周期数

        Returns:
            卡玛比率
        """
        # 年化收益率
        annual_return = returns.mean() * periods_per_year

        # 最大回撤
        max_dd_info = self.calculate_max_drawdown(cumulative_returns)
        max_dd = max_dd_info['max_drawdown']

        # 卡玛比率
        # 注意：如果最大回撤为0（一直上涨），则比率为无穷大，这里处理为0或极大值需视情况而定
        if max_dd > 0:
            calmar = annual_return / max_dd
        else:
            calmar = 0.0

        return calmar

    def calculate_ic(
        self,
        factor_values: pd.Series,
        next_returns: pd.Series,
        method: str = 'spearman'
    ) -> float:
        """
        计算单日 IC(Information Coefficient)
        IC 衡量因子值与下期收益率的相关性，反映因子的预测能力。

        Args:
            factor_values: 因子值序列 (截面数据，即同一时间点不同股票的因子值)
            next_returns: 下期收益率序列 (对应股票的下期收益)
            method: 
                - 'pearson': 皮尔逊相关系数 (线性相关，对离群值敏感)
                - 'spearman': 斯皮尔曼秩相关系数 (排序相关，更常用，鲁棒性更强)

        Returns:
            IC 值 (-1 到 1 之间)
        """
        # 数据对齐(取交集)
        # 确保只计算同时拥有因子值和收益率的股票
        common_idx = factor_values.index.intersection(next_returns.index)
        if len(common_idx) < 10:  # 样本太少统计意义不大
            return np.nan

        factor = factor_values.loc[common_idx]
        returns = next_returns.loc[common_idx]

        # 去除 NaN (缺失值处理)
        valid_idx = factor.notna() & returns.notna()
        factor = factor[valid_idx]
        returns = returns[valid_idx]

        if len(factor) < 10:
            return np.nan

        # 计算相关系数
        if method == 'spearman':
            ic = factor.corr(returns, method='spearman')
        else:
            ic = factor.corr(returns, method='pearson')

        return ic

    def calculate_ic_ir(self, ic_series: pd.Series) -> dict:
        """
        计算 IC/IR 统计指标 (Information Ratio of IC)
        衡量因子预测能力的稳定性。

        Args:
            ic_series: IC 时间序列 (每天计算出的 IC 值构成的序列)

        Returns:
            包含 IC 统计指标的字典
        """
        ic_clean = ic_series.dropna()

        if len(ic_clean) == 0:
            return {
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'ir': np.nan,
                'ic_win_rate': np.nan
            }

        # IC 均值: 因子的平均预测能力
        ic_mean = ic_clean.mean()
        
        # IC 标准差: 因子预测能力的波动性
        ic_std = ic_clean.std()
        
        # IR (信息比率): IC均值 / IC标准差
        # IR 越高，说明因子不仅预测能力强，而且表现稳定
        ir = ic_mean / ic_std if ic_std > 0 else 0

        # IC 胜率(IC > 0 的比例): 因子预测方向正确的概率
        ic_win_rate = (ic_clean > 0).sum() / len(ic_clean)

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'ic_win_rate': ic_win_rate
        }

    def calculate_group_returns(
        self,
        factor_values: pd.Series,
        next_returns: pd.Series,
        n_groups: int = 10
    ) -> pd.DataFrame:
        """
        计算分组收益 (分层回测)
        检验因子的单调性：理想情况下，多头组(因子值最高)收益最高，空头组(因子值最低)收益最低。

        Args:
            factor_values: 因子值序列
            next_returns: 下期收益率序列
            n_groups: 分组数量 (默认10组)

        Returns:
            分组收益统计表
        """
        # 数据对齐
        common_idx = factor_values.index.intersection(next_returns.index)
        df = pd.DataFrame({
            'factor': factor_values.loc[common_idx],
            'return': next_returns.loc[common_idx]
        })
        df = df.dropna()

        if len(df) < n_groups:
            return pd.DataFrame()

        # 按因子值分组
        # pd.qcut: 按照分位数进行等频分组 (每组股票数量大致相同)
        # labels=False: 返回分组的整数索引 (0, 1, ..., n-1)
        df['group'] = pd.qcut(df['factor'], q=n_groups, labels=False, duplicates='drop')

        # 计算每组收益的统计量
        group_stats = df.groupby('group')['return'].agg([
            ('mean_return', 'mean'), # 组内平均收益
            ('std_return', 'std'),   # 组内收益标准差
            ('count', 'count')       # 组内股票数量
        ])

        # 添加可读标签
        group_stats['group_label'] = [f'G{i+1}' for i in range(len(group_stats))]

        return group_stats

    def generate_report(
        self,
        cumulative_returns: pd.Series,
        returns: pd.Series
    ) -> dict:
        """
        生成完整的绩效报告

        Args:
            cumulative_returns: 累计净值序列
            returns: 日收益率序列

        Returns:
            绩效指标字典
        """
        # 基础统计
        # 总收益率: (期末净值 / 期初净值) - 1
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)

        # 最大回撤
        max_dd_info = self.calculate_max_drawdown(cumulative_returns)

        # 夏普比率
        sharpe = self.calculate_sharpe_ratio(returns)

        # 卡玛比率
        calmar = self.calculate_calmar_ratio(cumulative_returns, returns)

        # 胜率(正收益日占比): 交易日中赚钱天数的比例
        win_rate = (returns > 0).sum() / len(returns)

        report = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'max_drawdown': max_dd_info['max_drawdown'],
            'max_dd_start': max_dd_info['max_dd_start'],
            'max_dd_end': max_dd_info['max_dd_end'],
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'best_day': returns.max(),
            'worst_day': returns.min()
        }

        return report

    def print_report(self, report: dict):
        """打印绩效报告"""
        print("\n" + "=" * 60)
        print("📊 绩效评估报告")
        print("=" * 60)
        print(f"总收益率:       {report['total_return']*100:>10.2f}%")
        print(f"年化收益率:     {report['annual_return']*100:>10.2f}%")
        print(f"年化波动率:     {report['annual_volatility']*100:>10.2f}%")
        print(f"夏普比率:       {report['sharpe_ratio']:>10.2f}")
        print(f"最大回撤:       {report['max_drawdown']*100:>10.2f}%")
        print(f"卡玛比率:       {report['calmar_ratio']:>10.2f}")
        print(f"胜率:           {report['win_rate']*100:>10.2f}%")
        print(f"最佳单日:       {report['best_day']*100:>10.2f}%")
        print(f"最差单日:       {report['worst_day']*100:>10.2f}%")
        print("=" * 60)


# ========== 测试代码 ==========
if __name__ == '__main__':
    # 生成模拟数据
    np.random.seed(42)
    n_days = 252

    # 模拟日收益率(均值0.1%,标准差1.5%)
    # np.random.normal: 生成正态分布的随机数
    returns = pd.Series(
        np.random.normal(0.001, 0.015, n_days),
        index=pd.date_range('2020-01-01', periods=n_days, freq='D')
    )

    # 计算累计净值
    # cumprod(): 累乘，计算净值曲线 (1+r1)*(1+r2)*...
    cumulative_returns = (1 + returns).cumprod()

    # 初始化评估器
    evaluator = PerformanceEvaluator(risk_free_rate=0.03)

    # 生成报告
    report = evaluator.generate_report(cumulative_returns, returns)
    evaluator.print_report(report)

    # 测试 IC 计算 (因子分析部分)
    print("\n" + "=" * 60)
    print("📊 IC 分析测试")
    print("=" * 60)

    # 模拟截面数据：100只股票的因子值和下期收益率
    # 知识点：IC计算是截面(Cross-sectional)概念，即同一时刻不同股票之间的相关性
    factor_values = pd.Series(np.random.randn(100), index=[f'stock_{i}' for i in range(100)])
    next_returns = pd.Series(np.random.randn(100) * 0.02, index=[f'stock_{i}' for i in range(100)])

    ic = evaluator.calculate_ic(factor_values, next_returns)
    print(f"单日 IC: {ic:.4f}")

    # 模拟时间序列数据：多日的 IC 值序列
    # 知识点：IR计算是时间序列(Time-series)概念，衡量IC在时间维度上的稳定性
    ic_series = pd.Series(np.random.normal(0.02, 0.05, 100))
    ic_stats = evaluator.calculate_ic_ir(ic_series)
    print(f"\nIC 均值: {ic_stats['ic_mean']:.4f}")
    print(f"IC 标准差: {ic_stats['ic_std']:.4f}")
    print(f"IR (IC均值/IC标准差): {ic_stats['ir']:.4f}")
    print(f"IC 胜率 (IC>0占比): {ic_stats['ic_win_rate']:.2%}")

    # 测试分组收益
    print("\n" + "=" * 60)
    print("📊 分组收益测试")
    print("=" * 60)
    
    # 使用之前生成的模拟数据 factor_values 和 next_returns
    # 分为5组查看单调性
    group_stats = evaluator.calculate_group_returns(factor_values, next_returns, n_groups=5)
    print("分组统计结果:")
    print(group_stats)