"""
数据加载器 - Version 2.0：添加数据读取接口
对应教案 Day 7 第二步
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import List

class DataLoader:
    """数据加载器 - Version 2.0：添加数据读取接口"""

    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)

        # 加载交易日列表
        with open(self.data_dir / 'date.pkl', 'rb') as f:
            all_trade_dates = pickle.load(f)

        # 过滤出实际有数据文件的交易日
        self.trade_dates = []
        for date in all_trade_dates:
            if (self.data_dir / 'data_daily' / f'{date}.csv').exists():
                self.trade_dates.append(date)

        print(f"✅ 数据加载器初始化成功")
        print(f"📅 交易日数量: {len(self.trade_dates)}")
        if not self.trade_dates:
            print("⚠️ 未找到任何交易日数据，请检查 data_dir 是否正确")

    def get_all_dates(self) -> List[str]:
        """获取所有交易日"""
        return self.trade_dates

    # ===== 新增：数据读取接口 =====
    def get_daily_data(self, date: str) -> pd.DataFrame:
        """
        读取单日原始行情数据

        Args:
            date: 日期字符串（例如 '2020-01-02'）

        Returns:
            DataFrame: 列包含 date, code, open, close, high, low, volume 等
        """
        file_path = self.data_dir / 'data_daily' / f'{date}.csv'
        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        return df

    def get_daily_returns(self, date: str) -> pd.DataFrame:
        """
        读取单日收益率数据

        Args:
            date: 日期字符串

        Returns:
            DataFrame: 列包含 code, date, 1vwap_pct, 5vwap_pct, 10vwap_pct
        """
        file_path = self.data_dir / 'data_ret' / f'{date}.csv'
        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        return df

    def get_daily_status(self, date: str) -> pd.DataFrame:
        """
        读取单日交易状态数据

        Args:
            date: 日期字符串

        Returns:
            DataFrame: 列包含 date, code, paused, zt, dt 等
        """
        file_path = self.data_dir / 'data_ud_new' / f'{date}.csv'
        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        return df


# ========== 测试代码 ==========
if __name__ == '__main__':
    loader = DataLoader('./data')

    if len(loader.trade_dates) > 10:
        # 测试读取单日行情数据
        test_date = loader.trade_dates[10]
        daily_data = loader.get_daily_data(test_date)
        print(f"\n📊 {test_date} 行情数据形状: {daily_data.shape}")
        print(f"前 3 行:\n{daily_data.head(3)}")

        # 测试读取单日收益率
        daily_ret = loader.get_daily_returns(test_date)
        print(f"\n📈 {test_date} 收益率数据形状: {daily_ret.shape}")
        print(f"前 3 行:\n{daily_ret.head(3)}")

        # 测试读取交易状态
        daily_status = loader.get_daily_status(test_date)
        print(f"\n📋 {test_date} 交易状态数据形状: {daily_status.shape}")
        print(f"前 3 行（包含涨跌停信息）:\n{daily_status[['code', 'paused', 'zt', 'dt']].head(3)}")
    else:
        print("交易日数据不足，无法进行测试")
