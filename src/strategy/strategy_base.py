"""
策略基类模块 (Strategy Base)。

知识点 (Knowledge Points):
1.  **策略模式 (Strategy Pattern)**:
    *   定义了一系列算法（这里是不同的交易策略），并将每个算法封装起来，使它们可以互换。
    *   `Strategy` 基类定义了统一的接口，回测引擎只依赖这个接口，而不需要知道具体策略的实现细节。这符合“开闭原则”（对扩展开放，对修改关闭）。

2.  **抽象基类 (Abstract Base Class, ABC)**:
    *   使用 `abc` 模块强制子类必须实现特定的方法 (`calculate_factor`, `generate_signal`)。
    *   如果在子类中没有实现这些抽象方法，实例化时会报错，从而保证了接口的一致性。

3.  **因子与信号分离 (Separation of Factor and Signal)**:
    *   **因子 (Factor)**: 描述资产某种特征的数值（如过去20日涨幅）。因子本身不直接对应买卖。
    *   **信号 (Signal)**: 基于因子值生成的具体交易指令（如买入 Top 10）。
    *   分离的好处是解耦：同一个因子可以对应多种选股逻辑（选Top、选Bottom、多空配对等）。
"""

from abc import ABC, abstractmethod
import pandas as pd

from src.data.tradeable_filter import keep_tradeable


class Strategy(ABC):
    """
    策略基类 - 抽象接口
    所有具体的策略都必须继承此类并实现其抽象方法。
    """

    def __init__(self, name: str):
        """
        初始化策略

        Args:
            name: 策略名称，用于在回测报告中标识
        """
        self.name = name

    @abstractmethod
    def calculate_factor(
        self,
        date: str,
        data_loader,
        **kwargs
    ) -> pd.DataFrame:
        """
        【核心步骤1】计算因子值 (子类必须实现)

        Args:
            date: 当前回测日期
            data_loader: 数据加载器实例，用于获取历史行情
            **kwargs: 预留的其他参数

        Returns:
            DataFrame: 必须包含 'code' (股票代码), 'date' (日期), 'factor_value' (因子值) 列。
            这个 DataFrame 将作为下一步生成信号的输入。
        """
        pass

    @abstractmethod
    def generate_signal(
        self,
        factor_df: pd.DataFrame,
        top_n: int = 10
    ) -> list:
        """
        【核心步骤2】生成选股信号 (子类必须实现)

        Args:
            factor_df: 上一步计算出的因子数据
            top_n: 计划选出的股票数量

        Returns:
            list: 选中的股票代码列表 (例如 ['000001.SZ', '600519.SH'])
        """
        pass


# ========== 示例策略 1: 动量策略 ==========
class MomentumStrategy(Strategy):
    """
    动量策略 (Momentum Strategy): 
    基于“强者恒强”的假设，买入过去 N 日涨幅最大的股票。
    """

    def __init__(self, period: int = 20):
        super().__init__(name=f'Momentum_{period}')
        self.period = period

    def calculate_factor(self, date: str, data_loader, **kwargs) -> pd.DataFrame:
        # 1. 获取所有交易日列表，定位当前日期索引
        trade_dates = data_loader.get_all_dates()
        if date not in trade_dates:
            return pd.DataFrame()

        current_idx = trade_dates.index(date)
        # 如果历史数据不足 N 天，无法计算动量
        if current_idx < self.period:
            return pd.DataFrame()

        # 2. 获取 N 天前的日期
        past_date = trade_dates[current_idx - self.period]

        # 3. 获取当日和 N 天前的行情数据
        current_data = data_loader.get_daily_data(date)
        past_data = data_loader.get_daily_data(past_date)

        if current_data.empty or past_data.empty:
            return pd.DataFrame()

        # 4. 合并数据，计算收益率 (动量因子)
        merged = pd.merge(
            current_data[['code', 'close']],
            past_data[['code', 'close']],
            on='code',
            suffixes=('_now', '_past')
        )

        # 动量因子 = (当前收盘价 / N天前收盘价) - 1
        merged['factor_value'] = (merged['close_now'] / merged['close_past']) - 1
        merged['date'] = date

        factor_df = merged[['code', 'date', 'factor_value']]
        # 剔除停牌、涨跌停、ST、低价和低流动性股票
        return keep_tradeable(factor_df, date, data_loader)

    def generate_signal(self, factor_df: pd.DataFrame, top_n: int = 10) -> list:
        if factor_df.empty:
            return []

        # 降序排序: 因子值(涨幅)越大越好
        sorted_df = factor_df.sort_values('factor_value', ascending=False)
        
        # 取前 Top N 只股票
        selected = sorted_df.head(top_n)['code'].tolist()

        return selected


# ========== 示例策略 2: 反转策略 ==========
class ReversalStrategy(Strategy):
    """
    反转策略 (Reversal Strategy):
    基于“均值回归”的假设，买入过去 N 日跌幅最大的股票，预期其会反弹。
    """

    def __init__(self, period: int = 5):
        super().__init__(name=f'Reversal_{period}')
        self.period = period

    def calculate_factor(self, date: str, data_loader, **kwargs) -> pd.DataFrame:
        # 计算逻辑与动量完全相同，也是计算过去 N 日的收益率
        trade_dates = data_loader.get_all_dates()
        if date not in trade_dates:
            return pd.DataFrame()

        current_idx = trade_dates.index(date)
        if current_idx < self.period:
            return pd.DataFrame()

        past_date = trade_dates[current_idx - self.period]

        current_data = data_loader.get_daily_data(date)
        past_data = data_loader.get_daily_data(past_date)

        if current_data.empty or past_data.empty:
            return pd.DataFrame()

        merged = pd.merge(
            current_data[['code', 'close']],
            past_data[['code', 'close']],
            on='code',
            suffixes=('_now', '_past')
        )

        merged['factor_value'] = (merged['close_now'] / merged['close_past']) - 1
        merged['date'] = date

        factor_df = merged[['code', 'date', 'factor_value']]
        # 剔除停牌、涨跌停、ST、低价和低流动性股票
        return keep_tradeable(factor_df, date, data_loader)

    def generate_signal(self, factor_df: pd.DataFrame, top_n: int = 10) -> list:
        if factor_df.empty:
            return []

        # 升序排序: 因子值(涨幅)越小越好 (跌幅越大越好)
        sorted_df = factor_df.sort_values('factor_value', ascending=True)
        
        # 取前 Top N 只股票 (即跌幅最大的)
        selected = sorted_df.head(top_n)['code'].tolist()

        return selected


# ========== 测试代码 ==========
if __name__ == '__main__':
    from src.data.data_loader import DataLoader

    # 1. 初始化数据加载器
    loader = DataLoader('./data')
    
    # 选取一个测试日期 (确保有足够历史数据)
    if len(loader.trade_dates) > 50:
        test_date = loader.trade_dates[50]
        print(f"测试日期: {test_date}")

        # 2. 测试动量策略
        print("\n=== 测试动量策略 ===")
        momentum = MomentumStrategy(period=20)
        factor_df = momentum.calculate_factor(test_date, loader)
        if not factor_df.empty:
            print(f"因子计算成功: {len(factor_df)} 只股票")
            # 打印部分因子值
            print(factor_df.head())
            selected = momentum.generate_signal(factor_df, top_n=5)
            print(f"选股结果 (Top 5 动量): {selected}")

        # 3. 测试反转策略
        print("\n=== 测试反转策略 ===")
        reversal = ReversalStrategy(period=5)
        factor_df = reversal.calculate_factor(test_date, loader)
        if not factor_df.empty:
            print(f"因子计算成功: {len(factor_df)} 只股票")
            selected = reversal.generate_signal(factor_df, top_n=5)
            print(f"选股结果 (Top 5 跌幅): {selected}")
    else:
        print("数据不足，无法测试")
