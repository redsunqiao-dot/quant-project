import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

# 导入自定义模块
# Data Loader: 负责加载本地 CSV 数据
from src.data.data_loader import DataLoader
# Portfolio Manager: 负责管理持仓、资金和交易成本
from src.backtest.portfolio_manager import PortfolioManager
# Performance Evaluator: 负责计算回测绩效指标(夏普、回撤等)
from src.backtest.performance_evaluator import PerformanceEvaluator
# Strategy: 负责因子计算 + 选股信号
from src.strategy.strategy_base import Strategy


class BacktestEngine:
    """
    回测引擎 - Version 1.1 (Strategy 版)
    
    功能: 将数据、策略、交易、评估等模块串联起来,执行完整的历史回测。
    设计模式: 外观模式 (Facade Pattern) 的一种体现,为复杂的子系统提供统一的接口。
    
    知识点速览:
    1) 回测主流程: 盘前(信号) -> 盘中(调仓) -> 盘后(收益/IC) -> 汇总(绩效/统计)。
    2) T 日信号依赖 T-1 收盘数据; 本示例用 T+1 日收益近似持仓收益(存放在当日 data_ret 文件)。
    3) 交易成本只在调仓日扣除, 将绝对成本折算为资金占比后从收益率中减去。
    4) RankIC: 因子排序与未来收益排序的相关性,用于衡量因子有效性。
    5) 策略统一接口: calculate_factor + generate_signal。
    6) 组合收益: 这里采用等权平均作为最简持仓收益计算方式。
    """

    def __init__(
        self,
        data_dir: str = './data',
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.00012, # 万1.2 佣金
        slippage_rate: float = 0.001,    # 千一滑点
        stamp_duty: float = 0.0005,      # 万五印花税（仅卖出）
        transfer_fee_rate: float = 0.00001,  # 过户费约万0.1
        risk_free_rate: float = 0.03     # 3% 无风险利率
    ):
        """
        初始化回测引擎

        Args:
            data_dir: 数据目录,存放原始行情数据
            initial_capital: 初始资金,用于模拟账户
            commission_rate: 佣金费率,买卖双向收取
            slippage_rate: 滑点费率,模拟成交价的劣势偏移
            stamp_duty: 印花税,仅卖出收取（默认 0.05%）
            transfer_fee_rate: 过户费,买卖双向
            risk_free_rate: 无风险利率,用于计算夏普比率
        """
        print("=" * 60)
        print("🚀 初始化回测引擎")
        print("=" * 60)

        # 1. 初始化数据加载器
        self.loader = DataLoader(data_dir)
        
        # 2. 初始化投资组合管理器 (管理资金和持仓)
        self.portfolio = PortfolioManager(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            stamp_duty=stamp_duty,
            transfer_fee_rate=transfer_fee_rate,
        )
        
        # 3. 初始化绩效评估器
        self.evaluator = PerformanceEvaluator(risk_free_rate=risk_free_rate)

        print("=" * 60)
        print("✅ 回测引擎初始化完成")
        print("=" * 60)

    def run(
        self,
        start_date: str,
        end_date: str,
        strategy: Strategy,
        top_n: int = 10,           # 每日持仓数量
        rebalance_freq = 1,   # 调仓频率(天)或'month_start'
        enable_cost: bool = True,  # 是否计算交易成本
        calculate_ic: bool = True, # 是否计算 IC 指标
        n_groups: int = 0          # 分组数量(>1 时计算分组回测)
    ) -> dict:
        """
        运行回测
        
        流程:
        1. 准备数据: 确定交易日
        2. 逐日循环: 生成信号 -> 调仓 -> 结算收益 -> 统计IC
        3. 汇总结果: 计算各类绩效指标

        Args:
            start_date: 回测起始日期
            end_date: 回测结束日期
            strategy: 策略对象 (负责因子计算 + 选股信号)
            top_n: 选股数量 (每日持有排名前 N 的股票)
            rebalance_freq: 调仓频率 (1=每天, 5=每周) 或 'month_start'/'month_end'
            enable_cost: 是否启用交易成本 (佣金、印花税、滑点)
            calculate_ic: 是否计算 IC (RankIC)
            n_groups: 分组选股数量(例如 5/10)，<=1 时跳过分组回测

        Returns:
            回测结果字典, 包含收益序列、绩效指标、统计数据等
        """
        print("\n" + "=" * 60)
        print("🚀 开始回测")
        print("=" * 60)
        print(f"📅 回测区间: {start_date} 至 {end_date}")
        print(f"📊 策略: {strategy.name}")
        print(f"📊 选股数量: {top_n}")
        print(f"📊 调仓频率: 每 {rebalance_freq} 天")
        print(f"📊 交易成本: {'启用' if enable_cost else '禁用'}")
        print("=" * 60)

        # 1. 准备交易日列表
        # 获取所有可用日期, 并截取回测区间
        # 注意: 回测日期必须是可交易日,否则会导致取数失败或收益空缺
        trade_dates = self.loader.get_all_dates()
        backtest_dates = [d for d in trade_dates if start_date <= d <= end_date]
        print(f"\n✅ 回测交易日数量: {len(backtest_dates)}")

        # 2. 逐日回测主循环
        # 将"日级逻辑"拆分为可复用的小步骤,便于讲解与扩展
        daily_metrics = self._run_daily_backtest(
            backtest_dates=backtest_dates,
            strategy=strategy,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
            enable_cost=enable_cost,
            calculate_ic=calculate_ic,
            n_groups=n_groups
        )
        daily_returns_list = daily_metrics['daily_returns_list']
        daily_returns_dates = daily_metrics['daily_returns_dates']
        ic_list = daily_metrics['ic_list']
        ic_dates = daily_metrics['ic_dates']
        turnover_list = daily_metrics['turnover_list']
        group_returns = daily_metrics['group_returns']
        group_ls = daily_metrics['group_ls']
        group_ls_dates = daily_metrics['group_ls_dates']

        print("...")
        print(f"✅ 回测完成,共 {len(daily_returns_list)} 个交易日")

        # 4. 回测结束: 汇总统计
        # 4.1 构造收益率序列
        # 使用日期作为索引,方便后续画图与对齐
        returns_series = pd.Series(daily_returns_list, index=daily_returns_dates)
        
        # 4.2 计算累计净值 (1 + r).cumprod()
        # 累计净值是回测结果展示的核心曲线
        cumulative_returns = (1 + returns_series).cumprod()

        # 4.3 生成基础评估报告 (调用 Evaluator)
        # 包括总收益、年化、波动、夏普、回撤等常见指标
        report = self.evaluator.generate_report(cumulative_returns, returns_series)

        # 5. 补充 IC 统计
        if calculate_ic and len(ic_list) > 0:
            ic_series = pd.Series(ic_list, index=ic_dates)
            ic_stats = self.evaluator.calculate_ic_ir(ic_series)
            report['ic_mean'] = ic_stats['ic_mean']       # IC 均值
            report['ic_std'] = ic_stats['ic_std']         # IC 标准差
            report['ir'] = ic_stats['ir']                 # 信息比率 (IR)
            report['ic_win_rate'] = ic_stats['ic_win_rate'] # IC 胜率

        # 6. 补充交易统计 (来自 Portfolio)
        trade_stats = self.portfolio.get_statistics()
        report['total_cost'] = trade_stats['total_cost']  # 总交易成本
        report['trade_count'] = trade_stats['trade_count'] # 总交易次数
        report['avg_turnover'] = np.mean(turnover_list) if turnover_list else 0 # 平均换手率

        # 7. 分组回测数据
        if group_returns:
            report['group_returns'] = pd.DataFrame(group_returns)
        else:
            report['group_returns'] = pd.DataFrame()

        if group_ls:
            report['group_ls_returns'] = pd.Series(group_ls, index=group_ls_dates)
        else:
            report['group_ls_returns'] = pd.Series(dtype=float)

        # 8. 附带原始序列数据 (便于绘图)
        report['cumulative_returns'] = cumulative_returns
        report['daily_returns'] = returns_series
        report['ic_series'] = pd.Series(ic_list, index=ic_dates) if ic_list else None

        return report

    def _run_daily_backtest(
        self,
        backtest_dates: List[str],
        strategy: Strategy,
        top_n: int,
        rebalance_freq,
        enable_cost: bool,
        calculate_ic: bool,
        n_groups: int
    ) -> dict:
        """
        逐日回测主循环 (已拆分为小步骤,便于单测或替换逻辑)
        说明:
        - 因子在 T 日收盘后计算,信号在 T+1 日开盘执行(避免同日回测前视)
        - 收益使用 T 日 -> T+1 日的 forward return
        """
        metrics = self._init_daily_metrics()
        pending_signal = None

        for i, date in enumerate(backtest_dates):
            # 0) 开盘前: 执行上一交易日生成的调仓信号
            is_rebalance_day = False
            rebalance_info = None
            if pending_signal is not None:
                is_rebalance_day = True
                rebalance_info = self._rebalance_portfolio(
                    index=i,
                    date=date,
                    rebalance_freq=rebalance_freq,
                    selected_stocks=pending_signal,
                    turnover_list=metrics['turnover_list']
                )
                pending_signal = None

            # 1) 盘后: 读取 T 日收益(代表 T->T+1 forward return)
            ret_df = self._get_today_returns(date)

            # 2) 结算持仓收益(非日频再平衡)
            portfolio_ret = self._calculate_portfolio_return(
                ret_df=ret_df,
                enable_cost=enable_cost,
                is_rebalance_day=is_rebalance_day,
                rebalance_info=rebalance_info
            )
            # 记录当日收益并更新账户资金
            self._record_daily_return(portfolio_ret, date, metrics)
            self.portfolio.update_capital(portfolio_ret)

            # 3) 盘后: 计算因子(用于 IC / 分组 / 下一交易日信号)
            factor_df = self._calculate_factor(date, strategy)

            if calculate_ic and factor_df is not None and not factor_df.empty:
                # RankIC: 因子 vs forward return
                self._record_daily_ic(
                    date=date,
                    ret_df=ret_df,
                    factor_df=factor_df,
                    metrics=metrics
                )
            if n_groups and n_groups > 1 and factor_df is not None and not factor_df.empty:
                self._record_group_returns(
                    date=date,
                    ret_df=ret_df,
                    factor_df=factor_df,
                    n_groups=n_groups,
                    metrics=metrics
                )

            # 4) 判断是否生成下一交易日信号
            if factor_df is not None and not factor_df.empty:
                if self._should_rebalance(i, date, backtest_dates, rebalance_freq):
                    selected_stocks = self._generate_signal(
                        factor_df=factor_df,
                        strategy=strategy,
                        top_n=top_n
                    )
                    if selected_stocks:
                        pending_signal = selected_stocks

        return metrics

    def _init_daily_metrics(self) -> dict:
        # 统一管理回测过程中生成的日频序列
        return {
            'daily_returns_list': [],
            'daily_returns_dates': [],
            'ic_list': [],
            'ic_dates': [],
            'turnover_list': [],
            'group_returns': [],
            'group_ls': [],
            'group_ls_dates': []
        }

    def _calculate_factor(self, date: str, strategy: Strategy) -> pd.DataFrame:
        # 因子计算由策略负责
        return strategy.calculate_factor(date, self.loader)

    def _generate_signal(
        self,
        factor_df: pd.DataFrame,
        strategy: Strategy,
        top_n: int
    ) -> list:
        # 选股信号由策略负责
        return strategy.generate_signal(factor_df=factor_df, top_n=top_n)

    def _should_rebalance(
        self,
        index: int,
        date: str,
        backtest_dates: List[str],
        rebalance_freq
    ) -> bool:
        """
        调仓条件:
        - int: 每隔 N 个交易日
        - str:
          - 'month_start'/'m' 表示每月首个交易日
          - 'month_end' 表示每月最后一个交易日
        """
        if isinstance(rebalance_freq, str):
            freq = rebalance_freq.lower()
            if freq in {'m', 'month_start'}:
                if index == 0:
                    return True
                prev_date = backtest_dates[index - 1]
                return prev_date[:7] != date[:7]
            if freq in {'month_end', 'month_last'}:
                if index >= len(backtest_dates) - 1:
                    return False
                next_date = backtest_dates[index + 1]
                return next_date[:7] != date[:7]
            return False

        # 默认: 每隔 rebalance_freq 天
        return index % int(rebalance_freq) == 0

    def _rebalance_portfolio(
        self,
        index: int,
        date: str,
        rebalance_freq: int,
        selected_stocks: list,
        turnover_list: list
    ) -> dict:
        # 触发调仓,并把换手率用于后续统计
        rebalance_info = self.portfolio.rebalance(selected_stocks)
        turnover_list.append(rebalance_info['turnover'])

        print_limit = 5 * rebalance_freq if isinstance(rebalance_freq, int) else 5
        if index < print_limit:
            # 仅打印前几次调仓,避免刷屏
            print(f"{date} | 选股: {selected_stocks[:3]}... | "
                  f"换手: {rebalance_info['turnover']:.1%} | "
                  f"成本: {rebalance_info['total_cost']:.0f}元")

        return rebalance_info

    def _get_today_returns(self, date: str) -> Optional[pd.DataFrame]:
        # 使用 T+1 日收益作为 T 日持仓收益的近似(存放在当日文件)
        ret_df = self.loader.get_daily_returns(date)
        if not ret_df.empty:
            ret_df = ret_df.set_index('code')

        return ret_df

    def _calculate_portfolio_return(
        self,
        ret_df: Optional[pd.DataFrame],
        enable_cost: bool,
        is_rebalance_day: bool,
        rebalance_info: Optional[dict]
    ) -> Optional[float]:
        # 无收益数据时直接跳过
        if ret_df is None or ret_df.empty:
            return 0.0

        # 组合收益(考虑权重漂移)
        ret_series = ret_df['1vwap_pct']
        portfolio_ret = self.portfolio.compute_portfolio_return(ret_series)
        if enable_cost and is_rebalance_day and rebalance_info is not None:
            # 交易成本折算为资金占比,从当日收益中扣除
            cost_rate = rebalance_info['total_cost'] / self.portfolio.current_capital
            portfolio_ret -= cost_rate

        return portfolio_ret

    def _record_daily_return(self, portfolio_ret: float, date: str, metrics: dict) -> None:
        # 记录日收益序列,用于后续净值和指标计算
        metrics['daily_returns_list'].append(portfolio_ret)
        metrics['daily_returns_dates'].append(date)

    def _record_daily_ic(
        self,
        date: str,
        ret_df: Optional[pd.DataFrame],
        factor_df: Optional[pd.DataFrame],
        metrics: dict
    ) -> None:
        # IC: 因子值与未来收益的相关性,衡量因子方向与强度
        if ret_df is None or ret_df.empty:
            return

        if factor_df is None or factor_df.empty:
            return

        factor_series = factor_df.set_index('code')['factor_value']
        ret_series = ret_df['1vwap_pct']

        # RankIC 基于秩相关 (如 Spearman)
        ic = self.evaluator.calculate_ic(factor_series, ret_series)
        if not np.isnan(ic):
            metrics['ic_list'].append(ic)
            metrics['ic_dates'].append(date)

    def _record_group_returns(
        self,
        date: str,
        ret_df: Optional[pd.DataFrame],
        factor_df: Optional[pd.DataFrame],
        n_groups: int,
        metrics: dict
    ) -> None:
        if ret_df is None or ret_df.empty:
            return
        if factor_df is None or factor_df.empty:
            return

        factor = factor_df[['code', 'factor_value']].dropna()
        if factor.empty:
            return

        merged = factor.set_index('code').join(
            ret_df[['1vwap_pct']],
            how='inner'
        ).dropna()
        if len(merged) < n_groups:
            return

        try:
            merged['group'] = pd.qcut(
                merged['factor_value'],
                q=n_groups,
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            return

        if merged['group'].nunique() < n_groups:
            return

        group_ret = merged.groupby('group')['1vwap_pct'].mean()
        for g, r in group_ret.items():
            metrics['group_returns'].append({
                'date': date,
                'group': int(g),
                'ret': float(r)
            })

        # 多空 = Top 组 - Bottom 组
        top_group = group_ret.loc[group_ret.index.max()]
        bottom_group = group_ret.loc[group_ret.index.min()]
        metrics['group_ls'].append(float(top_group - bottom_group))
        metrics['group_ls_dates'].append(date)

    def print_report(self, report: dict):
        """
        格式化打印回测报告
        """
        print("\n" + "=" * 60)
        print("📊 完整绩效报告")
        print("=" * 60)

        # 基础绩效
        print(f"总收益率:       {report['total_return']*100:>10.2f}%")
        print(f"年化收益率:     {report['annual_return']*100:>10.2f}%")
        print(f"年化波动率:     {report['annual_volatility']*100:>10.2f}%")
        print(f"夏普比率:       {report['sharpe_ratio']:>10.2f}")
        print(f"最大回撤:       {report['max_drawdown']*100:>10.2f}%")
        print(f"卡玛比率:       {report['calmar_ratio']:>10.2f}")
        print(f"胜率:           {report['win_rate']*100:>10.2f}%")

        # IC 统计
        if 'ic_mean' in report:
            print("\n" + "-" * 60)
            print("📊 IC 分析")
            print("-" * 60)
            print(f"IC 均值:        {report['ic_mean']:>10.4f}")
            print(f"IC 标准差:      {report['ic_std']:>10.4f}")
            print(f"IR:             {report['ir']:>10.4f}")
            print(f"IC 胜率:        {report['ic_win_rate']*100:>10.2f}%")

        # 交易统计
        print("\n" + "-" * 60)
        print("📊 交易统计")
        print("-" * 60)
        print(f"总交易成本:     {report['total_cost']:>10.0f} 元")
        print(f"交易次数:       {report['trade_count']:>10.0f}")
        print(f"平均换手率:     {report['avg_turnover']*100:>10.2f}%")

        group_ls = report.get('group_ls_returns')
        if group_ls is not None and not group_ls.empty:
            print("\n" + "-" * 60)
            print("📊 分组多空 (Group Long-Short)")
            print("-" * 60)
            print(f"平均多空收益:  {group_ls.mean()*100:>10.2f}%")
            print(f"多空标准差:    {group_ls.std()*100:>10.2f}%")
            print(f"样本期数:      {len(group_ls):>10d}")

        # 净值曲线
        cumret = report['cumulative_returns']
        print("\n" + "-" * 60)
        print("📊 累计净值曲线(前5天)")
        print("-" * 60)
        for date in cumret.index[:5]:
            print(f"  {date}    {cumret.loc[date]:.4f}")
        print("  ...")
        print("📊 累计净值曲线(后5天)")
        print("-" * 60)
        for date in cumret.index[-5:]:
            print(f"  {date}    {cumret.loc[date]:.4f}")

        print("\n" + "=" * 60)
        print("✅ 报告生成完成!")
        print("=" * 60)


# ========== 主程序入口 ==========
if __name__ == '__main__':
    from src.strategy.strategy_base import MomentumStrategy, ReversalStrategy

    for strategy in [MomentumStrategy(period=20), ReversalStrategy(period=5)]:
        # 1. 实例化回测引擎 (配置资金与费率)
        engine = BacktestEngine(
            data_dir='./data',
            initial_capital=1000000,
            commission_rate=0.00012,
            slippage_rate=0.001,
            stamp_duty=0.0005,
            risk_free_rate=0.03
        )

        # 2. 运行回测 (配置策略参数)
        report = engine.run(
            start_date='2020-03-01',
            end_date='2020-12-31',
            strategy=strategy,
            top_n=10,
            rebalance_freq=5,
            enable_cost=True,
            calculate_ic=True,
            n_groups=5
        )

        # 3. 打印报告
        engine.print_report(report)
