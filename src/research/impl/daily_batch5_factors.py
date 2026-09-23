"""
日频批次：西南方向波动率 + 方正换手翻转反转。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_sw_dir_vol_panels(
    close: pd.DataFrame,
    window: int = 40,
) -> Dict[str, pd.DataFrame]:
    """
    - sw_dir_vol_diff_40: -(VOL+ - VOL-)
    - sw_mod_skew_40: -修正偏度（分母用 VOL++VOL-）
    """
    ret = np.log(close.astype(float) / close.astype(float).shift(1))
    arr = ret.values
    n_dates, n_codes = arr.shape
    dv = np.full((n_dates, n_codes), np.nan)
    skew = np.full((n_dates, n_codes), np.nan)
    sqrt_n = np.sqrt(float(window))

    for i in range(window - 1, n_dates):
        w = arr[i - window + 1 : i + 1]
        pos = w > 0
        neg = w < 0
        pos_cnt = pos.sum(axis=0).astype(float)
        neg_cnt = neg.sum(axis=0).astype(float)
        pos_sum = np.where(pos, w, 0.0).sum(axis=0)
        neg_sum = np.where(neg, w, 0.0).sum(axis=0)
        pos_mean = np.divide(pos_sum, pos_cnt, out=np.zeros(n_codes), where=pos_cnt > 0)
        neg_mean = np.divide(neg_sum, neg_cnt, out=np.zeros(n_codes), where=neg_cnt > 0)
        vp = np.where(pos, (w - pos_mean) ** 2, 0.0).sum(axis=0)
        vn = np.where(neg, (w - neg_mean) ** 2, 0.0).sum(axis=0)
        ok = (pos_cnt >= 2) & (neg_cnt >= 2)
        dv[i] = np.where(ok, vp - vn, np.nan)
        vol_tot = vp + vn
        mean_all = np.nanmean(w, axis=0)
        m3 = np.nansum((w - mean_all) ** 3, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sk = sqrt_n * m3 / (vol_tot ** 1.5)
        skew[i] = np.where(ok & (vol_tot > 0), sk, np.nan)

    idx, cols = ret.index, ret.columns
    return {
        "sw_dir_vol_diff_40": pd.DataFrame(-dv, index=idx, columns=cols).replace(
            [np.inf, -np.inf], np.nan
        ),
        "sw_mod_skew_40": pd.DataFrame(-skew, index=idx, columns=cols).replace(
            [np.inf, -np.inf], np.nan
        ),
    }


def compute_fz_rev_turn_flip_panels(
    close: pd.DataFrame,
    turn: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """方正「日间反转-换手翻转」取负。"""
    ret = close.astype(float).pct_change()
    d_turn = turn.astype(float) - turn.astype(float).shift(1)
    cs_mean = d_turn.mean(axis=1)
    coin = d_turn.lt(cs_mean, axis=0)
    flipped = ret.where(~coin, -ret)
    mean_flip = flipped.rolling(window, min_periods=window).mean()
    return {
        "fz_rev_turn_flip_20": (-mean_flip).replace([np.inf, -np.inf], np.nan),
    }


def batch5_lookback() -> int:
    return 45
