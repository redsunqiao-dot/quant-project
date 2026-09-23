"""
合成分策略：读取 factors/composite 日截面，按综合分 Top-N 选股。

用于把研究管线产出的 composite 接到 BacktestEngine / live 清单。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.utils import resolve_dated_file, read_frame
from src.data.tradeable_filter import keep_tradeable, is_board_allowed
from src.strategy.strategy_base import Strategy


class CompositeStrategy(Strategy):
    """读预计算合成分，按 factor_value 降序取 Top N。"""

    def __init__(
        self,
        factor_dir: str = "./factors/composite",
        score_col: str = "composite",
        name: str = "Composite",
    ):
        super().__init__(name=name)
        self.factor_dir = Path(factor_dir)
        self.score_col = score_col

    def calculate_factor(self, date: str, data_loader, **kwargs) -> pd.DataFrame:
        path = resolve_dated_file(self.factor_dir, date)
        if path is None:
            return pd.DataFrame(columns=["code", "date", "factor_value"])

        df = read_frame(path)
        if df.empty or "code" not in df.columns:
            return pd.DataFrame(columns=["code", "date", "factor_value"])
        if self.score_col not in df.columns:
            raise KeyError(
                f"{path} 缺少列 {self.score_col}，可用列: {list(df.columns)}"
            )

        out = pd.DataFrame(
            {
                "code": df["code"].astype(str),
                "date": date,
                "factor_value": pd.to_numeric(df[self.score_col], errors="coerce"),
            }
        ).dropna(subset=["factor_value"])
        return keep_tradeable(out, date, data_loader)

    def generate_signal(self, factor_df: pd.DataFrame, top_n: int = 10) -> list:
        if factor_df.empty:
            return []
        ranked = factor_df.sort_values("factor_value", ascending=False)
        selected = []
        for code in ranked["code"].tolist():
            if not is_board_allowed(str(code)):
                continue
            selected.append(str(code))
            if len(selected) >= top_n:
                break
        return selected
