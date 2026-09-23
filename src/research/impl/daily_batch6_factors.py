"""
日频批次：MAX 彩票、异质波动代理、Parkinson 极差波动。
方向统一为越大越好（低波动/低 MAX）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_batch6_panels(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    - df_maxret_20: -max(日收益, 20)  彩票偏好 MAX
    - xn_idio_vol_20: -std(收益-截面均值, 20)  异质波动代理
    - pk_range_vol_20: -Parkinson 极差波动
    """
    ret = close.astype(float).pct_change()

    maxret = ret.rolling(window, min_periods=window).max()
    df_maxret = -maxret

    resid = ret.sub(ret.mean(axis=1), axis=0)
    xn_idio = -resid.rolling(window, min_periods=window).std()

    # Parkinson: sqrt( sum(ln(H/L)^2) / (4 n ln2) )
    hl = np.log(high.astype(float) / low.astype(float).replace(0, np.nan))
    pk_var = (hl ** 2).rolling(window, min_periods=window).sum() / (
        4.0 * window * np.log(2.0)
    )
    pk_range_vol = -np.sqrt(pk_var)

    return {
        "df_maxret_20": df_maxret.replace([np.inf, -np.inf], np.nan),
        "xn_idio_vol_20": xn_idio.replace([np.inf, -np.inf], np.nan),
        "pk_range_vol_20": pk_range_vol.replace([np.inf, -np.inf], np.nan),
    }


def batch6_lookback(window: int = 20) -> int:
    return window + 5
