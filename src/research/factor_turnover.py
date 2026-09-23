"""
因子换手率分析

功能说明：
    通过相邻交易日的因子秩相关（Rank IC）估计因子稳定度，
    使用 1 - Rank IC 近似换手率。

知识点讲解：
    1) Rank IC：衡量两天因子排序的相似度。
    2) 换手率：排序差异越大，换手越高，交易成本越高。
    3) 稳定因子更适合低频持有，换手控制更容易。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class FactorTurnoverAnalyzer:
    def __init__(self, factor_col: Optional[str] = None):
        # 指定因子列名，None 则自动推断
        self.factor_col = factor_col

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """读取交易日列表并按区间过滤。"""
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
    def calc_rank_ic(series_a: pd.Series, series_b: pd.Series) -> float:
        """
        计算秩自相关 (Rank Autocorrelation)
        
        核心知识点：
        1. 什么是因子自相关？
           - 今天因子值排名前 10 的股票，明天还在前 10 吗？
           - 相关性越高，说明因子变化越慢，持仓越稳定。
           
        2. 与换手率 (Turnover) 的关系：
           - 估算换手率 ≈ 2 * (1 - Rank_IC) （对于双边换手）
           - Rank IC = 1.0 -> 因子排名完全没变 -> 无需调仓 -> 换手率 0。
           - Rank IC = 0.0 -> 因子排名完全随机 -> 每次大洗牌 -> 换手率 100% (理论上)。
           
        3. 评价标准：
           - 基本面因子 (价值/成长)：Rank IC 通常 > 0.9 (非常稳定)。
           - 量价因子 (反转/技术)：Rank IC 通常 < 0.7 (变化快)。
           - 如果一个基本面因子的 Rank IC 很低，说明数据可能有问题（在剧烈跳动）。
        """
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if aligned.empty:
            return np.nan
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")

    def factor_turnover(
        self,
        date_path: str,
        factor_dir: str,
        factor_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """批量计算日度换手（1 - Rank IC）。"""
        dates = self.load_dates(date_path, start_date=start_date, end_date=end_date)
        records = []

        for prev_date, curr_date in zip(dates[:-1], dates[1:]):
            try:
                prev_series = self.load_factor_series(factor_dir, prev_date, factor_col=factor_col)
                curr_series = self.load_factor_series(factor_dir, curr_date, factor_col=factor_col)
            except FileNotFoundError:
                continue

            # Rank IC 越高，说明两天因子排序更接近
            rank_ic = self.calc_rank_ic(prev_series, curr_series)
            # 用 1 - Rank IC 近似换手率（仅粗略估计）
            turnover = 1 - rank_ic if not np.isnan(rank_ic) else np.nan
            records.append({
                "date": curr_date,
                "rank_ic": rank_ic,
                "turnover": turnover,
                "coverage": prev_series.dropna().shape[0],
            })

        df = pd.DataFrame(records)
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / "factor_turnover.csv", index=False)

            # 汇总平均水平，辅助判断稳定性
            summary = df[["rank_ic", "turnover"]].agg(["mean", "std"]).reset_index()
            summary.to_csv(out / "factor_turnover_summary.csv", index=False)

        return df


if __name__ == "__main__":
    analyzer = FactorTurnoverAnalyzer(factor_col=None)
    analyzer.factor_turnover(
        date_path="./data/date.pkl",
        factor_dir="./factors/neutralized",
        factor_col=None,
        start_date="2020-01-02",
        end_date="2020-12-31",
        output_dir="./outputs/turnover",
    )
