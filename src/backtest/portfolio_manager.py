import pandas as pd
from typing import List, Tuple


class PortfolioManager:
    """组合管理器 - Version 1.0"""

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.00012,  # 万1.2 佣金
        slippage_rate: float = 0.001,     # 0.1% 滑点
        stamp_duty: float = 0.0005,       # 0.05% 印花税(仅卖出，2023 年下调后)
        transfer_fee_rate: float = 0.00001,  # 过户费约万 0.1，买卖双向
    ):
        """
        初始化组合管理器

        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率(买卖双向)
            slippage_rate: 滑点费率(买卖双向)
            stamp_duty: 印花税(仅卖出时收取)
            transfer_fee_rate: 过户费率(买卖双向)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.stamp_duty = stamp_duty
        self.transfer_fee_rate = transfer_fee_rate

        # 当前持仓(股票代码列表)
        self.current_holdings = []
        # 当前权重(用于非日频再平衡)
        self.current_weights = {}

        # 交易统计
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_stamp_duty = 0.0
        self.total_transfer_fee = 0.0
        self.trade_count = 0

        print("✅ 组合管理器初始化成功")
        print(f"💰 初始资金: {initial_capital:,.0f} 元")
        print(f"📊 佣金费率: {commission_rate*10000:.1f} 万分之")
        print(f"📊 滑点费率: {slippage_rate*100:.2f}%")
        print(f"📊 印花税: {stamp_duty*100:.2f}% (仅卖出)")
        print(f"📊 过户费: {transfer_fee_rate*10000:.2f} 万分之 (买卖双向)")

    def calculate_turnover(
        self,
        old_holdings: List[str],
        new_holdings: List[str]
    ) -> Tuple[List[str], List[str], float]:
        """
        计算调仓换手率

        Args:
            old_holdings: 旧持仓列表
            new_holdings: 新持仓列表

        Returns:
            (买入列表, 卖出列表, 换手率)
        """
        old_set = set(old_holdings)
        new_set = set(new_holdings)

        # 需要卖出的股票(旧持仓中不在新信号里的)
        to_sell = list(old_set - new_set)

        # 需要买入的股票(新信号中不在旧持仓里的)
        to_buy = list(new_set - old_set)

        # 换手率 = (买入数量 + 卖出数量) / (2 * 持仓数量)
        if len(old_holdings) > 0:
            turnover = (len(to_buy) + len(to_sell)) / (2 * len(old_holdings))
        else:
            turnover = 1.0  # 首次建仓,换手率100%

        return to_buy, to_sell, turnover

    def calculate_trade_cost(
        self,
        trade_value: float,
        is_buy: bool
    ) -> float:
        """
        计算单笔交易的总成本

        Args:
            trade_value: 交易金额
            is_buy: True=买入, False=卖出

        Returns:
            总成本
        """
        # 1. 佣金(买卖双向)
        commission = trade_value * self.commission_rate

        # 2. 滑点(买卖双向)
        slippage = trade_value * self.slippage_rate

        # 3. 印花税(仅卖出)
        stamp = 0.0
        if not is_buy:
            stamp = trade_value * self.stamp_duty

        # 4. 过户费(买卖双向)
        transfer = trade_value * self.transfer_fee_rate

        total_cost = commission + slippage + stamp + transfer

        # 记录统计
        self.total_commission += commission
        self.total_slippage += slippage
        self.total_stamp_duty += stamp
        self.total_transfer_fee += transfer

        return total_cost

    def rebalance(
        self,
        new_holdings: List[str],
        equal_weight: bool = True
    ) -> dict:
        """
        执行调仓操作

        Args:
            new_holdings: 新的持仓列表
            equal_weight: 是否等权重配置

        Returns:
            调仓信息字典
        """
        # 1. 计算换手
        to_buy, to_sell, turnover = self.calculate_turnover(
            self.current_holdings,
            new_holdings
        )

        # 2. 计算每只股票的仓位
        if equal_weight and len(new_holdings) > 0:
            weight_per_stock = 1.0 / len(new_holdings)
        else:
            weight_per_stock = 0.0

        # 3. 计算交易成本
        total_cost = 0.0

        # 卖出成本
        if len(to_sell) > 0:
            if self.current_weights:
                for stock in to_sell:
                    weight = self.current_weights.get(stock, 0.0)
                    sell_value = self.current_capital * weight
                    cost = self.calculate_trade_cost(sell_value, is_buy=False)
                    total_cost += cost
            else:
                sell_value_per_stock = (
                    self.current_capital * (1.0 / len(self.current_holdings))
                    if len(self.current_holdings) > 0
                    else 0.0
                )
                for stock in to_sell:
                    cost = self.calculate_trade_cost(sell_value_per_stock, is_buy=False)
                    total_cost += cost

        # 买入成本
        if len(to_buy) > 0:
            buy_value_per_stock = self.current_capital * weight_per_stock
            for stock in to_buy:
                cost = self.calculate_trade_cost(buy_value_per_stock, is_buy=True)
                total_cost += cost

        # 4. 更新持仓
        self.current_holdings = new_holdings.copy()
        # 4.1 更新权重（默认等权）
        if equal_weight and len(new_holdings) > 0:
            self.current_weights = {code: weight_per_stock for code in new_holdings}
        else:
            self.current_weights = {}
        self.trade_count += len(to_buy) + len(to_sell)

        # 5. 返回调仓信息
        rebalance_info = {
            'to_buy': to_buy,
            'to_sell': to_sell,
            'turnover': turnover,
            'total_cost': total_cost,
            'weight_per_stock': weight_per_stock
        }

        return rebalance_info

    def compute_portfolio_return(self, returns: pd.Series) -> float:
        """
        计算组合收益并更新权重(考虑权重漂移)。

        Args:
            returns: 以股票代码为索引的收益率序列

        Returns:
            组合收益率
        """
        if not self.current_holdings:
            return 0.0

        if self.current_weights:
            weights = self.current_weights.copy()
        else:
            # 无权重时默认等权
            weight = 1.0 / len(self.current_holdings)
            weights = {code: weight for code in self.current_holdings}

        # 计算组合收益(缺失收益按 0 处理)
        portfolio_ret = 0.0
        for code, w in weights.items():
            r = returns.get(code, 0.0)
            portfolio_ret += w * r

        # 更新权重（按收益漂移）
        updated = {code: w * (1 + returns.get(code, 0.0)) for code, w in weights.items()}
        total = sum(updated.values())
        if total > 0:
            self.current_weights = {code: w / total for code, w in updated.items()}

        return portfolio_ret

    def update_capital(self, daily_return: float):
        """
        更新资金(根据当日收益率)

        Args:
            daily_return: 当日收益率(例如 0.02 表示涨2%)
        """
        self.current_capital *= (1 + daily_return)

    def get_statistics(self) -> dict:
        """获取交易统计"""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_stamp_duty': self.total_stamp_duty,
            'total_transfer_fee': self.total_transfer_fee,
            'total_cost': (
                self.total_commission
                + self.total_slippage
                + self.total_stamp_duty
                + self.total_transfer_fee
            ),
            'trade_count': self.trade_count
        }


# ========== 测试代码 ==========
if __name__ == '__main__':
    pm = PortfolioManager(initial_capital=1000000)

    # 模拟第一天建仓
    print("\n=== Day 1: 首次建仓 ===")
    day1_holdings = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH', '000858.SZ']
    info = pm.rebalance(day1_holdings)
    print(f"买入: {info['to_buy']}")
    print(f"卖出: {info['to_sell']}")
    print(f"换手率: {info['turnover']:.2%}")
    print(f"交易成本: {info['total_cost']:.2f} 元")

    # 模拟第二天调仓
    print("\n=== Day 2: 调仓 ===")
    day2_holdings = ['000002.SZ', '600000.SH', '600519.SH', '601318.SH', '000651.SZ']
    info = pm.rebalance(day2_holdings)
    print(f"买入: {info['to_buy']}")
    print(f"卖出: {info['to_sell']}")
    print(f"换手率: {info['turnover']:.2%}")
    print(f"交易成本: {info['total_cost']:.2f} 元")

    # 打印统计
    print("\n=== 交易统计 ===")
    stats = pm.get_statistics()
    print(f"总佣金: {stats['total_commission']:.2f} 元")
    print(f"总滑点: {stats['total_slippage']:.2f} 元")
    print(f"总印花税: {stats['total_stamp_duty']:.2f} 元")
    print(f"总过户费: {stats.get('total_transfer_fee', 0):.2f} 元")
    print(f"总成本: {stats['total_cost']:.2f} 元")
    print(f"交易次数: {stats['trade_count']}")
