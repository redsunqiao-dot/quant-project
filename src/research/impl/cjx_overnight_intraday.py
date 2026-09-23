"""
中信建投《隔夜-日内异象》日频子集（N=20）。

无集合竞价量时，跳过隔夜/日内成交量类；仅用开收盘价。
方向：表中已取负的保持；波动类取负使越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_cjx_oi_panels(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    - cjx_intraday_rev_20: -mean(close/open-1, 20)
    - cjx_overnight_mom_20: mean(open/close_lag-1, 20)  原文无负号
    - cjx_overnight_vol_20: -std(overnight, 20)
    - cjx_oi_spread_20: mean(overnight - intraday, 20)
    """
    open_f = open_.astype(float)
    close_f = close.astype(float)
    overnight = open_f / close_f.shift(1).replace(0, np.nan) - 1.0
    intraday = close_f / open_f.replace(0, np.nan) - 1.0

    cjx_intraday_rev = -intraday.rolling(window, min_periods=window).mean()
    cjx_overnight_mom = overnight.rolling(window, min_periods=window).mean()
    cjx_overnight_vol = -overnight.rolling(window, min_periods=window).std()
    cjx_oi_spread = (overnight - intraday).rolling(window, min_periods=window).mean()

    out = {
        "cjx_intraday_rev_20": cjx_intraday_rev,
        "cjx_overnight_mom_20": cjx_overnight_mom,
        "cjx_overnight_vol_20": cjx_overnight_vol,
        "cjx_oi_spread_20": cjx_oi_spread,
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def cjx_oi_lookback(window: int = 20) -> int:
    return window + 5
