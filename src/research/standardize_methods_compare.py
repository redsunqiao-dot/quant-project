"""
标准化方法对比

功能说明：
    对比不同标准化方法在 IC 上的表现，帮助选择更稳健的预处理方式。

知识点讲解：
    1) Z-Score：适合近似正态；易受异常值影响。
    2) Rank：只保留排序信息，鲁棒但会损失幅度信息。
    3) Robust：基于中位数与 MAD，抗异常能力强。
    4) MinMax：缩放到 [0,1]，对极值敏感。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_SKIP_COLS = {"code", "date", "industry"}


class StandardizeMethodComparator:
    """
    标准化方法对比分析器

    核心知识点：
    1. Z-Score (Standard Score):
       - 公式: (x - μ) / σ
       - 特点: 均值为0，方差为1。保留了数据的分布形态（如果原数据是偏态的，处理后依然是偏态）。
       - 适用: 大多数线性模型（回归、PCA）。
    
    2. Rank (Percentile):
       - 公式: rank / count
       - 特点: 均匀分布 [0, 1]。完全消除异常值影响，也消除了分布形态。
       - 适用: 对排序敏感的策略（如 TopN 选股），或者数据噪音极大的情况。
       - 缺点: 丢失了"幅度"信息。因子值 10 和 100 可能只差一个排名，但在 Z-Score 中差很远。
       
    3. MinMax (Normalization):
       - 公式: (x - min) / (max - min)
       - 特点: 严格限制在 [0, 1]。
       - 缺点: 对极值非常敏感。如果有一个超级大的异常值，其他所有数都会被压缩到 0 附近。
    """
    def __init__(self, factor_col: Optional[str] = None, ret_col: str = "1vwap_pct"):
        # factor_col=None 时会自动选择第一个非 code 列
        self.factor_col = factor_col
        self.ret_col = ret_col

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """读取交易日列表并按区间筛选。"""
        with open(date_path, "rb") as f:
            dates = pickle.load(f)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def load_factor_series(self, factor_dir: str, date: str, factor_col: Optional[str] = None) -> pd.Series:
        """读取单日因子序列（兼容 Parquet / CSV）。"""
        from src.common.utils import resolve_dated_file, read_frame

        file = resolve_dated_file(factor_dir, date)
        if file is None:
            raise FileNotFoundError(f"No factor file for {date} in {factor_dir}")
        df = read_frame(file)
        if "code" in df.columns:
            df = df.set_index("code")
        col = factor_col or self.factor_col
        if col is None:
            candidates = [c for c in df.columns if c not in DEFAULT_SKIP_COLS]
            if not candidates:
                raise ValueError(f"{file} 中找不到因子列")
            col = candidates[0]
        return pd.to_numeric(df[col], errors="coerce")

    def load_return_series(self, ret_dir: str, date: str, ret_col: Optional[str] = None) -> pd.Series:
        """读取单日收益序列（兼容 Parquet / CSV）。"""
        from src.common.utils import resolve_dated_file, read_frame

        file = resolve_dated_file(ret_dir, date)
        if file is None:
            raise FileNotFoundError(f"No return file for {date} in {ret_dir}")
        df = read_frame(file)
        if "code" in df.columns:
            df = df.set_index("code")
        col = ret_col or self.ret_col
        return pd.to_numeric(df[col], errors="coerce")

    @staticmethod
    def zscore(series: pd.Series) -> pd.Series:
        """Z-Score 标准化。"""
        std = series.std()
        return (series - series.mean()) / std if std != 0 else series - series.mean()

    @staticmethod
    def robust_zscore(series: pd.Series) -> pd.Series:
        """鲁棒 Z-Score（使用中位数与 MAD）。"""
        median = series.median()
        mad = (series - median).abs().median()
        return (series - median) / mad if mad != 0 else series - median

    @staticmethod
    def rank_pct(series: pd.Series) -> pd.Series:
        """排序分位数（0~1）。"""
        return series.rank(pct=True)

    @staticmethod
    def minmax(series: pd.Series) -> pd.Series:
        """MinMax 归一化。"""
        denom = series.max() - series.min()
        return (series - series.min()) / denom if denom != 0 else series * 0

    @staticmethod
    def calc_ic(factor_series: pd.Series, ret_series: pd.Series, method: str = "spearman") -> float:
        """计算指定类型的 IC（支持 spearman 与 pearson）。"""
        aligned = pd.concat([factor_series, ret_series], axis=1).dropna()
        if aligned.empty:
            return np.nan
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method)

    def compare_standardize_methods(
        self,
        date_path: str,
        factor_dir: str,
        ret_dir: str,
        factor_col: Optional[str] = None,
        ret_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """批量对比标准化方法的 IC。"""
        dates = self.load_dates(date_path, start_date=start_date, end_date=end_date)
        methods = {
            "zscore": self.zscore,
            "rank": self.rank_pct,
            "robust": self.robust_zscore,
            "minmax": self.minmax,
        }

        records = []
        for date in dates:
            try:
                factor = self.load_factor_series(factor_dir, date, factor_col=factor_col)
                ret = self.load_return_series(ret_dir, date, ret_col=ret_col)
            except FileNotFoundError:
                continue

            factor = factor.replace([np.inf, -np.inf], np.nan)
            for name, fn in methods.items():
                # 每种标准化方法各算一次 IC
                standardized = fn(factor.dropna())
                spearman_ic = self.calc_ic(standardized, ret, method="spearman")
                pearson_ic = self.calc_ic(standardized, ret, method="pearson")
                records.append(
                    {
                        "date": date,
                        "method": name,
                        "spearman_ic": spearman_ic,
                        "pearson_ic": pearson_ic,
                    }
                )

        df = pd.DataFrame(records)
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / "standardize_method_ic.csv", index=False)

            # 汇总平均 IC 与 IR
            summary = df.groupby("method").agg(
                {
                    "spearman_ic": ["mean", "std"],
                    "pearson_ic": ["mean", "std"],
                }
            )
            summary.columns = [
                "spearman_mean",
                "spearman_std",
                "pearson_mean",
                "pearson_std",
            ]
            summary = summary.reset_index()
            summary["spearman_ir"] = summary["spearman_mean"] / summary["spearman_std"].replace(0, np.nan)
            summary["pearson_ir"] = summary["pearson_mean"] / summary["pearson_std"].replace(0, np.nan)
            summary.to_csv(out / "standardize_method_summary.csv", index=False)

        return df


if __name__ == "__main__":
    comparator = StandardizeMethodComparator(factor_col=None, ret_col="1vwap_pct")
    comparator.compare_standardize_methods(
        date_path="./data/date.pkl",
        factor_dir="./factors/raw",
        ret_dir="./data/data_ret",
        factor_col=None,
        ret_col="1vwap_pct",
        start_date="2020-01-02",
        end_date="2020-12-31",
        output_dir="./outputs/standardize_compare",
    )
