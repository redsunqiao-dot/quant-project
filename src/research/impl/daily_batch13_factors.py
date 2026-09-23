"""
日频批次13：Yang-Zhang 波动、收益峰度、量价相关。
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_batch13_panels(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 20,
    kurt_window: int = 60,
) -> Dict[str, pd.DataFrame]:
    open_ = open_.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    volume = volume.astype(float)
    ret = close.pct_change()

    # Yang-Zhang 简化：过夜方差 + 开收方差 + Rogers-Satchell
    log_oc = np.log(open_ / close.shift(1).replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    with np.errstate(all="ignore"):
        rs = (
            np.log(high / close.replace(0, np.nan)) * np.log(high / open_.replace(0, np.nan))
            + np.log(low / close.replace(0, np.nan)) * np.log(low / open_.replace(0, np.nan))
        )
    # k ≈ 0.34 常用近似
    k = 0.34
    vo = log_oc.rolling(window, min_periods=window).var()
    vc = log_co.rolling(window, min_periods=window).var()
    vrs = rs.rolling(window, min_periods=window).mean()
    yz = vo + k * vc + (1.0 - k) * vrs
    yz_vol = -np.sqrt(yz.clip(lower=0.0))

    # 收益峰度取负（厚尾投机）
    kurt = ret.rolling(kurt_window, min_periods=kurt_window).kurt()
    ret_kurt = -kurt

    # 量价相关取负（量随价齐升往往过热）
    pv_corr = -ret.rolling(window, min_periods=window).corr(volume)

    return {
        "yz_vol_20": yz_vol.replace([np.inf, -np.inf], np.nan),
        "ret_kurt_60": ret_kurt.replace([np.inf, -np.inf], np.nan),
        "pv_corr_vol_20": pv_corr.replace([np.inf, -np.inf], np.nan),
    }


def batch13_lookback(window: int = 20, kurt_window: int = 60) -> int:
    return max(window, kurt_window) + 5
