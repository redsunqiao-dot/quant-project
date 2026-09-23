"""
IC 衰减分析

功能说明：
    比较不同持有期（1/5/10 日等）的 IC 水平，衡量因子预测力衰减速度，
    为调仓频率提供依据。

知识点讲解：
    1) IC 衰减：预测力随持有期延长往往下降。
    2) IC_IR：均值/波动，用于比较稳定性。
    3) 视角：短期因子适合高频调仓，长周期因子适合低频持有。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ICDecayAnalyzer:
    """
    IC 衰减分析器 (Information Coefficient Decay)

    核心知识点：
    1. 什么是 IC 衰减？
       - 因子包含的信息会随着时间流逝而失效。
       - 例如："突发新闻"可能只在1天内有效（IC Decay 快），"价值低估"可能在3个月内都有效（IC Decay 慢）。
    
    2. 为什么要看不同 Horizon (持有期) 的 IC？
       - 1日 IC 高，10日 IC 低 -> 这是一个短线因子，必须高频换手（交易成本高）。
       - 1日 IC 一般，20日 IC 高 -> 这是一个中长线因子，可以低频换手（交易成本低，容量大）。
       
    3. 决策指导：
       - 如果 IC(5天) ≈ IC(1天)，说明因子预测力能维持一周，我们就可以改为"周频调仓"，大幅节省手续费。
    """
    def __init__(self, factor_col: Optional[str] = None, horizons: Optional[list] = None):
        # horizons 对应收益文件中的列名，如 1vwap_pct (持有1天), 5vwap_pct (持有5天)
        self.factor_col = factor_col
        self.horizons = horizons or ["1vwap_pct", "5vwap_pct", "10vwap_pct"]

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
        from src.common.utils import load_factor_series as _load

        return _load(factor_dir, date, factor_col=factor_col or self.factor_col)

    @staticmethod
    def load_return_df(ret_dir: str, date: str) -> pd.DataFrame:
        """读取单日收益矩阵（多持有期；兼容 Parquet / CSV）。"""
        from src.common.utils import resolve_dated_file, read_frame

        file = resolve_dated_file(ret_dir, date)
        if file is None:
            raise FileNotFoundError(f"No return file for {date} in {ret_dir}")
        df = read_frame(file)
        if "code" in df.columns:
            df = df.set_index("code")
        return df

    @staticmethod
    def calc_ic(factor_series: pd.Series, ret_series: pd.Series) -> float:
        """计算 Spearman IC。"""
        aligned = pd.concat([factor_series, ret_series], axis=1).dropna()
        if aligned.empty:
            return np.nan
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")

    def ic_decay(
        self,
        date_path: str,
        factor_dir: str,
        ret_dir: str,
        factor_col: Optional[str] = None,
        horizons: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """批量计算各持有期的 IC 衰减曲线。"""
        horizons = horizons or self.horizons
        dates = self.load_dates(date_path, start_date=start_date, end_date=end_date)
        records = []

        for date in dates:
            try:
                factor = self.load_factor_series(factor_dir, date, factor_col=factor_col)
                ret_df = self.load_return_df(ret_dir, date)
            except FileNotFoundError:
                continue

            for col in horizons:
                if col not in ret_df.columns:
                    continue
                # 同一天因子对不同持有期收益做 IC
                ic = self.calc_ic(factor, ret_df[col])
                records.append({"date": date, "horizon": col, "ic": ic})

        df = pd.DataFrame(records)
        
        if df.empty:
            print("WARNING: No IC decay records generated. Check data paths.")
            return df
            
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / "ic_decay.csv", index=False)

            # 汇总平均 IC 与 IR
            summary = df.groupby("horizon")["ic"].agg(["mean", "std"]).reset_index()
            summary["ir"] = summary["mean"] / summary["std"].replace(0, np.nan)
            summary.to_csv(out / "ic_decay_summary.csv", index=False)

        return df


if __name__ == "__main__":
    analyzer = ICDecayAnalyzer(factor_col=None)
    analyzer.ic_decay(
        date_path="./data/date.pkl",
        factor_dir="./factors/neutralized",
        ret_dir="./data/data_ret",
        factor_col=None,
        start_date="2020-01-02",
        end_date="2020-12-31",
        output_dir="./outputs/ic_decay",
    )
