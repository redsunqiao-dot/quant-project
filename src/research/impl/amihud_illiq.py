"""
中投 / Amihud ILLIQ 非流动性因子（日频）。

ILLIQ = mean(|ret| / money, N)；越大流动性越差。
管线已有市值中性；保留原方向，由验证门判定。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_illiq_panels(
    close: pd.DataFrame,
    money: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """zt_illiq_20: 过去 window 日 |日收益|/成交额 均值。"""
    ret = close.astype(float).pct_change()
    amt = money.astype(float).replace(0, np.nan)
    daily = ret.abs() / amt
    illiq = daily.rolling(window, min_periods=window).mean()
    return {
        "zt_illiq_20": illiq.replace([np.inf, -np.inf], np.nan),
    }


def illiq_lookback(window: int = 20) -> int:
    return window + 5
