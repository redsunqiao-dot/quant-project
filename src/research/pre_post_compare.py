"""
处理前后效果对比分析 (Pre/Post Processing Comparison)

功能说明：
    用于对比因子在经过一系列预处理（如去极值、标准化、行业中性化）前后，其预测能力（IC）和选股收益的变化。
    
核心指标：
    1. IC (Information Coefficient): 
       - 因子值与下期收益率的 Spearman 秩相关系数。
       - 衡量因子的排序能力。
       
    2. Top-N 收益 (Group Return):
       - 选取因子值最大的前 N 只股票，计算其平均收益。
       - 衡量因子在头部的选股能力。

知识点讲解：
    - 中性化的代价：通常中性化会降低 IC 的绝对值（因为剔除了风格带来的“伪Alpha”），但会提高 IC_IR（稳定性）。
    - 纯净 Alpha：我们希望处理后的因子，虽然 IC 可能变小，但每一分 IC 都是真实的 Alpha，而不是靠买大盘股/特定行业躺赢的 Beta。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class PrePostComparator:
    """
    处理前后对比分析器
    
    负责加载两组因子数据（Raw vs Processed），并分别计算 IC 和 Top-N 收益序列，最后输出对比结果。
    """
    
    def __init__(self, ret_col: str = "1vwap_pct", top_n: int = 50):
        """
        Parameters:
        -----------
        ret_col : str
            用于计算 IC 的收益率列名（通常是 T+1 日的收益）。
        top_n : int
            计算头部组合收益时选取的股票数量。
        """
        self.ret_col = ret_col
        self.top_n = top_n

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """读取交易日列表。"""
        with open(date_path, "rb") as f:
            dates = pickle.load(f)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def load_factor_series(self, factor_dir: str, date: str, factor_col: Optional[str] = None) -> pd.Series:
        """
        读取单日因子数据（兼容 Parquet / CSV）。
        如果 factor_col 未指定，自动选择第一列数值列作为因子列。
        """
        from src.common.utils import resolve_dated_file, read_frame

        file = resolve_dated_file(factor_dir, date)
        if file is None:
            raise FileNotFoundError(f"No factor file for {date} in {factor_dir}")
        df = read_frame(file)
        if "code" in df.columns:
            df = df.set_index("code")
        # 移除日期等元数据列
        drop_cols = [col for col in df.columns if col.lower() in {"date", "datetime"}]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if factor_col is None:
            # 自动寻找第一个数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric factor columns found in {file}")
            factor_col = numeric_cols[0]
        return df[factor_col]

    def load_return_series(self, ret_dir: str, date: str, ret_col: Optional[str] = None) -> pd.Series:
        """读取单日收益率数据（兼容 Parquet / CSV）。"""
        from src.common.utils import resolve_dated_file, read_frame

        file = resolve_dated_file(ret_dir, date)
        if file is None:
            raise FileNotFoundError(f"No return file for {date} in {ret_dir}")
        df = read_frame(file)
        if "code" in df.columns:
            df = df.set_index("code")
        col = ret_col or self.ret_col
        return df[col]

    def calc_ic(self, factor_series: pd.Series, ret_series: pd.Series) -> float:
        """
        计算 Rank IC (Spearman Correlation)。
        更推荐使用 Rank IC 而非 Pearson IC，因为它对异常值不敏感，且不假设线性关系。
        """
        # 对齐数据并去除空值
        aligned = pd.concat([factor_series, ret_series], axis=1).dropna()
        if aligned.empty:
            return np.nan
        # 使用 spearman 方法计算秩相关系数
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")

    def calc_topn_return(self, factor_series: pd.Series, ret_series: pd.Series, top_n: Optional[int] = None) -> float:
        """
        计算多头组合（Top N）的平均收益。
        """
        aligned = pd.concat([factor_series, ret_series], axis=1).dropna()
        if aligned.empty:
            return np.nan
        count = top_n if top_n is not None else self.top_n
        
        # 按因子值降序排列，取前 N 只
        top = aligned.sort_values(by=aligned.columns[0], ascending=False).head(count)
        
        # 计算这 N 只股票收益率的均值
        return top.iloc[:, 1].mean()

    def calc_series(
        self,
        dates,
        factor_dir: str,
        ret_dir: str,
        factor_col: Optional[str] = None,
        ret_col: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        """
        批量计算指定时间段内的 IC 序列和收益序列。
        """
        ic_values = []
        ret_values = []
        for date in dates:
            try:
                factor = self.load_factor_series(factor_dir, date, factor_col=factor_col)
                ret = self.load_return_series(ret_dir, date, ret_col=ret_col)
            except FileNotFoundError:
                # 缺失文件时填 NaN
                ic_values.append(np.nan)
                ret_values.append(np.nan)
                continue

            ic_values.append(self.calc_ic(factor, ret))
            ret_values.append(self.calc_topn_return(factor, ret, top_n=top_n))

        ic_series = pd.Series(ic_values, index=dates, name="ic")
        ret_series = pd.Series(ret_values, index=dates, name="ret")
        return ic_series, ret_series

    @staticmethod
    def summarize_ic(ic_series: pd.Series) -> dict:
        """计算 IC 统计量：均值、标准差、IR (Information Ratio)。"""
        mean = ic_series.mean()
        std = ic_series.std()
        # IC_IR = Mean / Std，衡量因子预测能力的稳定性
        ir = mean / std if std != 0 else np.nan
        return {"mean": mean, "std": std, "ir": ir}

    def compare_pre_post(
        self,
        dates,
        raw_factor_dir: str,
        neutral_factor_dir: str,
        ret_dir: str,
        factor_col: Optional[str] = None,
        ret_col: Optional[str] = None,
        top_n: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """
        主流程：对比处理前后的效果。
        
        Parameters:
        -----------
        dates : list
            日期列表
        raw_factor_dir : str
            原始因子目录
        neutral_factor_dir : str
            处理后（如中性化后）因子目录
        ret_dir : str
            收益率数据目录
        """
        # 1. 计算原始因子的表现
        ic_raw, ret_raw = self.calc_series(
            dates,
            factor_dir=raw_factor_dir,
            ret_dir=ret_dir,
            factor_col=factor_col,
            ret_col=ret_col,
            top_n=top_n,
        )
        
        # 2. 计算处理后因子的表现
        ic_neu, ret_neu = self.calc_series(
            dates,
            factor_dir=neutral_factor_dir,
            ret_dir=ret_dir,
            factor_col=factor_col,
            ret_col=ret_col,
            top_n=top_n,
        )

        # 3. 汇总数据
        ic_df = pd.DataFrame({"ic_raw": ic_raw, "ic_neutral": ic_neu})
        ret_df = pd.DataFrame({"ret_raw": ret_raw, "ret_neutral": ret_neu})
        
        # 计算累计净值 (Cumulative Returns)，方便画图
        cum_df = (1 + ret_df.fillna(0)).cumprod()

        # 4. 保存结果
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ic_df.to_csv(out / "ic_compare.csv")
            ret_df.to_csv(out / "ret_compare.csv")
            cum_df.to_csv(out / "cumret_compare.csv")

        return ic_df, ret_df, cum_df


if __name__ == "__main__":
    comparator = PrePostComparator(ret_col="1vwap_pct", top_n=50)
    dates = comparator.load_dates("./data/date.pkl", start_date="2020-01-02", end_date="2020-12-31")
    
    # 示例：对比原始因子 (raw) 与 行业中性化后 (industry_neutralized) 的效果
    ic_df, ret_df, cum_df = comparator.compare_pre_post(
        dates,
        raw_factor_dir="./factors/raw",
        neutral_factor_dir="./factors/industry_neutralized",
        ret_dir="./data/data_ret",
        factor_col=None,
        top_n=50,
        output_dir="./outputs/pre_post_cmp",
    )

    print("IC Summary (raw):", comparator.summarize_ic(ic_df["ic_raw"]))
    print("IC Summary (neutral):", comparator.summarize_ic(ic_df["ic_neutral"]))