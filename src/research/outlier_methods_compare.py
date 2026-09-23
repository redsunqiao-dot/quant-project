import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class OutlierMethodComparator:
    """
    用于比较不同去极值（Outlier Treatment）方法对因子数据分布影响的类。
    
    主要功能：
    1. 加载因子数据。
    2. 应用不同的去极值方法（Sigma法, 分位数法, MAD法）。
    3. 计算并记录处理后的统计量（如均值、标准差、偏度、峰度等）。
    4. 输出比较结果，帮助选择最适合当前因子的去极值方案。
    """
    def __init__(self, factor_col: Optional[str] = None):
        """
        初始化比较器。
        
        Args:
            factor_col: 因子在CSV文件中的列名。如果为None，后续会自动尝试推断。
        """
        self.factor_col = factor_col

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        加载并筛选交易日期列表。
        
        Args:
            date_path: 存储日期序列的pickle文件路径。
            start_date: 起始日期（字符串，如 '2020-01-01'）。
            end_date: 结束日期。
            
        Returns:
            list: 筛选后的日期列表。
        """
        with open(date_path, "rb") as f:
            dates = pickle.load(f)
        # 根据起止日期过滤
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def load_factor_series(self, factor_dir: str, date: str, factor_col: Optional[str] = None) -> pd.Series:
        """
        读取指定日期的因子数据（兼容 Parquet / CSV）。
        """
        from src.common.utils import resolve_dated_file, read_frame

        file = resolve_dated_file(factor_dir, date)
        if file is None:
            raise FileNotFoundError(f"No factor file for {date} in {factor_dir}")
        df = read_frame(file)

        # 将股票代码设为索引，方便后续对齐处理
        if "code" in df.columns:
            df = df.set_index("code")

        col = factor_col or self.factor_col
        # 如果未指定列名，尝试排除常见的元数据列后自动选择第一列
        if col is None:
            meta_cols = {"date", "trade_date", "datetime", "timestamp"}
            candidates = [c for c in df.columns if c not in meta_cols]
            if not candidates:
                raise ValueError(f"No factor column found in file {file}")
            col = candidates[0]

        return df[col].astype(float)

    @staticmethod
    def winsor_sigma(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
        """
        3-Sigma去极值法（标准差法）。
        
        原理：
        假设数据服从正态分布，将超过 Mean ± n_sigma * Std 的值截断到边界值。
        通常 n_sigma 取 3.0。
        
        注意：此方法对均值和标准差本身对异常值敏感，若原始数据异常值极其严重，效果可能不佳。
        """
        mean = series.mean()
        std = series.std()
        
        # 标准差为0或NaN时无法处理，直接返回原序列
        if pd.isna(std) or std == 0:
            return series
            
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        return series.clip(lower, upper)

    @staticmethod
    def winsor_percentile(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
        """
        百分位去极值法（Percentile Clipping）。
        
        原理：
        将数据中小于下分位（如1%）和大于上分位（如99%）的值，分别赋值为下分位数和上分位数。
        
        优点：简单直观，不依赖分布假设。
        缺点：如果数据尾部很长但并非异常，可能会丢失信息。
        """
        lower = series.quantile(lower_q)
        upper = series.quantile(upper_q)
        return series.clip(lower, upper)

    @staticmethod
    def winsor_mad(series: pd.Series, n_mad: float = 3.5) -> pd.Series:
        """
        中位数绝对偏差去极值法（MAD, Median Absolute Deviation）。
        
        原理：
        MAD 是比标准差更鲁棒的波动率估计量。
        1. 计算中位数 Median。
        2. 计算绝对偏差的中位数 MAD = median(|Xi - Median|)。
        3. 边界 = Median ± n_mad * (1.4826 * MAD)。
        
        系数 1.4826 是为了让 MAD 在正态分布下也是标准差的一致估计量。
        通常 n_mad 取 3.0 或 3.5（对应约 3 Sigma 的范围）。
        
        优点：对异常值极不敏感，是量化中常用的稳健去极值方法。
        """
        median = series.median()
        mad = (series - median).abs().median()
        
        if mad == 0 or pd.isna(mad):
            return series
            
        # 1.4826 是正态分布下的比例因子，Scale Factor for Normal Distribution
        scaled_mad = 1.4826 * mad
        lower = median - n_mad * scaled_mad
        upper = median + n_mad * scaled_mad
        return series.clip(lower, upper)

    @staticmethod
    def describe_series(series: pd.Series) -> dict:
        """
        计算序列的统计描述，用于评估去极值效果。
        包含：均值、标准差、偏度（Skewness）、峰度（Kurtosis）、极值和分位数。
        """
        return {
            "mean": series.mean(),
            "std": series.std(),
            "skew": series.skew(),   # 偏度：衡量分布的不对称性
            "kurt": series.kurt(),   # 峰度：衡量分布的尾部厚度（肥尾程度）
            "min": series.min(),
            "p01": series.quantile(0.01),
            "p05": series.quantile(0.05),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
            "max": series.max(),
        }

    def compare_outlier_methods(
        self,
        date_path: str,
        factor_dir: str,
        factor_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        执行比较流程的主函数。
        
        遍历每一天的数据，分别应用多种去极值方法，并记录处理后的统计特征。
        """
        dates = self.load_dates(date_path, start_date=start_date, end_date=end_date)
        records = []

        # 定义要比较的方法字典
        methods = {
            "raw": lambda x: x,  # 原始数据，不做处理
            "sigma": self.winsor_sigma,
            "percentile": self.winsor_percentile,
            "mad": self.winsor_mad,
        }

        print(f"Comparing outlier methods for {len(dates)} dates...")
        
        for date in dates:
            try:
                series = self.load_factor_series(factor_dir, date, factor_col=factor_col)
            except FileNotFoundError:
                print(f"Warning: File for {date} not found, skipping.")
                continue
                
            # 清洗基础数据：处理无穷大和NaN
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                continue

            # 对每种方法进行处理并记录统计量
            for name, fn in methods.items():
                # 注意：传入 copy() 防止修改原始数据
                cleaned = fn(series.copy())
                stats = self.describe_series(cleaned)
                stats.update({"date": date, "method": name})
                records.append(stats)

        # 汇总所有结果
        df = pd.DataFrame(records)
        
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            
            # 保存详细的逐日、逐方法记录
            detail_path = out / "outlier_methods_compare.csv"
            df.to_csv(detail_path, index=False)
            print(f"Detailed comparison saved to {detail_path}")

            # 计算各方法的平均表现（汇总统计）
            summary = df.groupby("method").mean(numeric_only=True).reset_index()
            summary_path = out / "outlier_methods_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"Summary statistics saved to {summary_path}")

        return df


if __name__ == "__main__":
    # 示例用法
    # 1. 初始化比较器
    comparator = OutlierMethodComparator(factor_col=None)
    
    # 2. 运行比较
    # 注意：这里的路径是相对于项目根目录的，请根据实际情况调整
    comparator.compare_outlier_methods(
        date_path="./data/date.pkl",         # 包含日期列表的pkl文件
        factor_dir="./factors/raw",          # 原始因子数据目录
        factor_col=None,                     # 自动推断因子列
        start_date="2020-01-02",             # 开始日期
        end_date="2020-12-31",               # 结束日期
        output_dir="./outputs/outlier_methods_compare", # 结果输出目录
    )