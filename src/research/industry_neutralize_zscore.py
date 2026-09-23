"""
行业内标准化（行业中性化） - Industry Neutralization

功能说明：
    在每个时间截面上，将因子数据按“行业”分组，在每个行业内部进行标准化（Z-Score）或排名（Rank）。
    处理后，每个行业内部的因子均值为0，标准差为1（Z-Score模式）。

核心知识点讲解：
    1. 为什么要进行行业中性化？
       - 消除行业间差异：某些因子天然在特定行业数值偏大（例如银行股的PB通常很低，科技股PB很高）。如果不处理，全市场选股时会倾向于选中特定行业的股票。
       - 避免行业贝塔（Industry Beta）：我们希望选取的是“同行业中更优秀”的股票（Alpha），而不是单纯押注某个行业的整体上涨。
    
    2. 两种常见方法：
       - Z-Score (标准化)： (x - mean) / std。保留了分布的形态和极端值信息。
       - Rank (分位数)： 将数值转换为排名百分比。对异常值极度鲁棒，但丢失了具体的差异幅度信息。
       
    3. 实现技巧：
       - Pandas GroupBy + Transform：這是实现分组计算最高效的方法，可以直接将计算结果映射回原始索引，保持数据形状不变。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.common.utils import resolve_dated_file, read_frame, write_factor_frame


class IndustryNeutralizer:
    """
    行业中性化处理器
    
    负责按日期批量读取因子文件和行业分类文件，执行行业内标准化操作。
    """
    
    def __init__(self, method: str = "zscore", industry_col: str = "industry"):
        """
        初始化
        
        Parameters:
        -----------
        method : str, default 'zscore'
            中性化方法。
            - 'zscore': 行业内减均值除以标准差。
            - 'rank': 行业内求百分比排名。
        industry_col : str, default 'industry'
            行业分类文件中，记录行业名称/代码的列名。
        """
        self.method = method
        self.industry_col = industry_col

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        读取交易日列表并筛选区间。
        通常用于控制回测或数据处理的时间范围。
        """
        with open(date_path, "rb") as f:
            dates = pickle.load(f)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def load_csv_with_index(self, path: Path, index_col: str = "code") -> pd.DataFrame:
        """
        通用表格读取并设置索引（兼容 CSV / Parquet）。
        
        会自动移除 'date' 或 'datetime' 列，防止这些非数值列干扰后续的矩阵运算。
        """
        df = read_frame(path)
        if index_col in df.columns:
            df = df.set_index(index_col)
        # 移除日期等非因子列，避免后续计算混入字符串报错
        drop_cols = [col for col in df.columns if col.lower() in {"date", "datetime"}]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    def within_industry_standardize(self, series: pd.Series, industry: pd.Series, method: Optional[str] = None) -> pd.Series:
        """
        核心逻辑：行业内标准化
        
        Parameters:
        -----------
        series : pd.Series
            原始因子值（单列）
        industry : pd.Series
            对应的行业分类标签
        method : str, optional
            'zscore' 或 'rank'
            
        Returns:
        --------
        pd.Series
            处理后的因子序列，索引与输入保持一致。
        """
        method = method or self.method
        
        # 构造临时 DataFrame 用于 GroupBy
        # 这样确保 value 和 industry 是通过索引自动对齐的
        df = pd.DataFrame({"value": series, "industry": industry})
        
        if method == "zscore":
            # transform 会返回与原 DataFrame 长度相同的序列
            # 逻辑：(x - mean) / std
            # 细节：check std != 0 避免除零错误
            standardized = df.groupby("industry")["value"].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean()
            )
        elif method == "rank":
            # pct=True 返回 [0, 1] 之间的百分比排名
            standardized = df.groupby("industry")["value"].transform(lambda x: x.rank(pct=True))
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return standardized

    def neutralize_by_industry(
        self,
        factor_df: pd.DataFrame,
        industry_series: pd.Series,
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        对所有因子列做行业中性化。
        
        Process:
        1. 筛选数值类型的列。
        2. 遍历每一列，调用 within_industry_standardize 进行处理。
        3. 组装结果。
        """
        # 仅处理数值列，忽略字符串列（如股票名称等）
        numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
        out = pd.DataFrame(index=factor_df.index)
        
        for col in numeric_cols:
            # 替换无穷大值，防止标准差计算异常
            series = factor_df[col].replace([np.inf, -np.inf], np.nan)
            
            # 调用核心分组标准化逻辑
            out[col] = self.within_industry_standardize(series, industry_series, method=method)
            
        # 提示用户如果有非数值列被跳过
        non_numeric = set(factor_df.columns) - set(numeric_cols)
        if non_numeric:
            print(f"Skip non-numeric columns: {', '.join(sorted(non_numeric))}")
            
        return out

    def process_folder(
        self,
        date_path: str,
        factor_dir: str,
        industry_dir: str,
        output_dir: str,
        method: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        按日期批量进行行业中性化处理。
        
        Parameters:
        -----------
        date_path : str
            日期列表文件路径 (pkl)
        factor_dir : str
            原始因子文件夹
        industry_dir : str
            行业分类文件夹 (每天一个csv，记录当天股票的行业)
        output_dir : str
            输出文件夹
        """
        dates = self.load_dates(date_path, start_date=start_date, end_date=end_date)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Starting industry neutralization ({method or self.method})...")
        
        for date in dates:
            factor_file = resolve_dated_file(factor_dir, date)
            industry_file = Path(industry_dir) / f"{date}.csv"
            
            # 必须同时存在因子文件和行业文件
            if factor_file is None or not industry_file.exists():
                continue

            factor_df = self.load_csv_with_index(factor_file)
            industry_df = self.load_csv_with_index(industry_file)
            
            # 获取行业列
            industry_col_name = self.industry_col
            if industry_col_name not in industry_df.columns:
                # 兜底：如果找不到指定列名，选择第一列作为行业标签
                industry_col_name = industry_df.columns[0]
            industry_series = industry_df[industry_col_name]

            # 关键步骤：对齐索引 (Intersection)
            # 确保因子表和行业表的股票代码完全一致，剔除不匹配的行
            common_index = factor_df.index.intersection(industry_series.index)
            factor_df = factor_df.loc[common_index]
            industry_series = industry_series.loc[common_index]

            if factor_df.empty:
                continue

            # 执行中性化
            neutralized = self.neutralize_by_industry(factor_df, industry_series, method=method)
            
            # 保存结果（Parquet）
            write_factor_frame(neutralized, out_path, date, index_as_code=True)

        print(f"Industry neutralization done. Output -> {output_dir}")


if __name__ == "__main__":
    # 使用示例
    neutralizer = IndustryNeutralizer(method="zscore", industry_col="industry")
    
    neutralizer.process_folder(
        date_path="./data/date.pkl",
        factor_dir="./factors/preprocessed",     # 输入：预处理过的因子
        industry_dir="./data/data_industry",     # 输入：行业数据
        output_dir="./factors/industry_neutralized", # 输出
        method="zscore",
        start_date="2020-01-02",
        end_date="2020-12-31",
    )