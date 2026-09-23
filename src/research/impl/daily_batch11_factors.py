"""
日频批次11–12：Rogers-Satchell/半方差/同工作日反转/残差动量/温和动量/隔夜日内反转/低换手动量。
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.research.impl.daily_batch7_factors import _same_weekday_mom


def compute_batch11_panels(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    turn: pd.DataFrame,
    vol_window: int = 20,
    mom_window: int = 60,
    mild_window: int = 20,
    mad_window: int = 60,
    mad_k: float = 1.96,
) -> Dict[str, pd.DataFrame]:
    open_ = open_.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    turn = turn.astype(float)
    ret = close.pct_change()

    # Rogers-Satchell
    with np.errstate(all="ignore"):
        rs = (
            np.log(high / close.replace(0, np.nan)) * np.log(high / open_.replace(0, np.nan))
            + np.log(low / close.replace(0, np.nan)) * np.log(low / open_.replace(0, np.nan))
        )
    rs_vol = -rs.rolling(vol_window, min_periods=vol_window).mean()

    # 下行半方差
    neg = ret.clip(upper=0.0)
    semi = -np.sqrt((neg ** 2).rolling(vol_window, min_periods=vol_window).mean())

    # 同工作日动量取负 → 日频反转
    wd_rev = -_same_weekday_mom(ret, n_weeks=12)

    # 特质/残差动量：截面去均值收益累积
    resid = ret.sub(ret.mean(axis=1), axis=0)
    resid_mom = resid.rolling(mom_window, min_periods=mom_window).sum()

    # 东方温和收益：非 MAD 极端日收益之和
    med = ret.rolling(mad_window, min_periods=mad_window).median()
    mad = (ret - med).abs().rolling(mad_window, min_periods=mad_window).median()
    is_ext = (ret - med).abs() > (mad_k * mad.replace(0, np.nan))
    mild = ret.where(~is_ext, 0.0).where(mad.notna())
    mild_mom = mild.rolling(mild_window, min_periods=mild_window).sum()

    # 隔夜/日内反转
    ovn = open_ / close.shift(1) - 1.0
    idt = close / open_ - 1.0
    ovn_rev = -ovn.rolling(vol_window, min_periods=vol_window).mean()
    idt_rev = -idt.rolling(vol_window, min_periods=vol_window).mean()

    # 低换手动量：换手低于截面中位数才保留动量，否则 0
    mom = close / close.shift(mom_window) - 1.0
    turn_ma = turn.rolling(vol_window, min_periods=vol_window).mean()
    turn_med = turn_ma.median(axis=1)
    low_turn = turn_ma.le(turn_med, axis=0)
    lt_mom = mom.where(low_turn, 0.0).where(turn_ma.notna())

    return {
        "rs_vol_20": rs_vol.replace([np.inf, -np.inf], np.nan),
        "semi_down_20": semi.replace([np.inf, -np.inf], np.nan),
        "xn_same_wd_rev_12": wd_rev.replace([np.inf, -np.inf], np.nan),
        "ct_resid_mom_60": resid_mom.replace([np.inf, -np.inf], np.nan),
        "df_mild_mom_20": mild_mom.replace([np.inf, -np.inf], np.nan),
        "ovn_rev_20": ovn_rev.replace([np.inf, -np.inf], np.nan),
        "idt_rev_20": idt_rev.replace([np.inf, -np.inf], np.nan),
        "fz_lt_mom_60": lt_mom.replace([np.inf, -np.inf], np.nan),
    }


def batch11_lookback(mom_window: int = 60, mad_window: int = 60) -> int:
    return max(mom_window, mad_window, 12 * 5) + 5
