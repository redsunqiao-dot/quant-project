"""
信号生成器 - 完整可复制版本
对应教案 Day 7 Version 4.0
"""

import pandas as pd
from src.data.data_loader import DataLoader
from src.strategy.factor_calculator import FactorCalculator


class SignalGenerator:
    """信号生成器 - Version 1.0"""

    def __init__(self, data_loader: DataLoader, factor_calculator: FactorCalculator):
        """
        初始化信号生成器

        Args:
            data_loader: 数据加载器实例
            factor_calculator: 因子计算器实例
        """
        self.loader = data_loader
        self.calculator = factor_calculator
        print("✅ 信号生成器初始化成功")

    def generate_daily_signal(
        self,
        date: str,
        top_n: int = 10,
        filter_limit: bool = True,
        factor_col: str = 'momentum_20'
    ) -> list:
        """
        生成单日选股信号

        Args:
            date: 当前日期（字符串）
            top_n: 选股数量
            filter_limit: 是否过滤涨跌停
            factor_col: 因子列名（例如 momentum_20）

        Returns:
            list: 选中的股票代码列表
        """
        # 1. 读取当日因子值
        factor_df = self.calculator.load_factor(date)
        if factor_df.empty:
            return []

        # 2. 设置 code 为索引，方便排序
        factor_df = factor_df.set_index('code')

        # 3. 过滤涨跌停和停牌股票
        if filter_limit:
            status_df = self.loader.get_daily_status(date)
            if not status_df.empty:
                # 只保留正常交易的股票（paused=0, zt=0, dt=0）
                status_df = status_df.set_index('code')
                tradeable = status_df[(status_df['paused'] == 0) &
                                     (status_df['zt'] == 0) &
                                     (status_df['dt'] == 0)]
                # 取交集
                factor_df = factor_df[factor_df.index.isin(tradeable.index)]

        # 4. 检查因子列是否存在
        if factor_col not in factor_df.columns:
            return []

        # 5. 过滤 NaN 值
        factor_df = factor_df.dropna(subset=[factor_col])

        if len(factor_df) == 0:
            return []

        # 6. 按因子值降序排序，选出 Top N
        sorted_df = factor_df.sort_values(factor_col, ascending=False)
        selected = sorted_df.head(top_n).index.tolist()

        return selected


# ========== 测试代码 ==========
if __name__ == '__main__':
    from src.data.data_loader import DataLoader
    from src.strategy.factor_calculator import FactorCalculator

    # 初始化
    loader = DataLoader('./data')
    calculator = FactorCalculator(loader)
    signal_gen = SignalGenerator(loader, calculator)

    # 先计算并保存一个因子（如果还没有）
    if len(loader.trade_dates) > 50:
        test_date = loader.trade_dates[50]  # 选择第 51 个交易日
        factor_df = calculator.calculate_momentum_daily(test_date, period=20)
        if not factor_df.empty:
            from src.common.utils import write_factor_frame
            save_path = write_factor_frame(factor_df, calculator.factor_dir, test_date)
            print(f"临时保存因子到: {save_path}")

        # 测试单日选股
        factor_col = calculator.momentum_col(20)
        selected_stocks = signal_gen.generate_daily_signal(
            date=test_date,
            top_n=10,
            filter_limit=True,
            factor_col=factor_col
        )

        print(f"\n📅 {test_date} 选股结果:")
        print(f"选中股票数量: {len(selected_stocks)}")
        print(f"前 5 只: {selected_stocks[:5]}")

        # 查看选中股票的因子值
        factor_df_show = calculator.load_factor(test_date)
        if not factor_df_show.empty:
            factor_df_show = factor_df_show.set_index('code')
            print(f"\n因子值排名（Top 5）:")
            for stock in selected_stocks[:5]:
                if stock in factor_df_show.index and factor_col in factor_df_show.columns:
                    print(f"{stock:12s}  {factor_df_show.loc[stock, factor_col]:.4f}")
    else:
        print("交易日数据不足，无法进行测试")
