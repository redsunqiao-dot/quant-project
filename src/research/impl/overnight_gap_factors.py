"""
隔夜跳空类日频因子（方正 / 国盛量价淘金一）。

方向统一为越大越好：研报 IC 为负的表达式取负。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _overnight_ret(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """隔夜收益：今开 / 昨收 - 1。"""
    prev_close = close.shift(1)
    return open_.astype(float) / prev_close.replace(0, np.nan) - 1.0


def compute_overnight_gap_panels(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    turn: pd.DataFrame,
    jump_n: int = 10,
    corr_n: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    - fz_jump_10: -mean(|overnight|, 10)  方正隔夜跳空
    - gs_ovn_turn_corr_20: -corr(|overnight|, turn.shift(1), 20)
      国盛：隔夜绝对涨跌与昨日换手相关（知情交易优势代理）
    """
    ovn = _overnight_ret(open_, close)
    abs_ovn = ovn.abs()

    fz_jump = -abs_ovn.rolling(jump_n, min_periods=jump_n).mean()

    # 隔夜对应「昨日量」：用昨换手与今 |隔夜| 做滚动相关
    prev_turn = turn.astype(float).shift(1)
    corr = abs_ovn.rolling(corr_n, min_periods=corr_n).corr(prev_turn)
    gs_corr = -corr

    return {
        "fz_jump_10": fz_jump.replace([np.inf, -np.inf], np.nan),
        "gs_ovn_turn_corr_20": gs_corr.replace([np.inf, -np.inf], np.nan),
    }


def overnight_gap_lookback(jump_n: int = 10, corr_n: int = 20) -> int:
    return max(jump_n, corr_n) + 5
