"""
日频批次9：东北协偏度 + 东方极端收益反转（日频 MAD 代理）。
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_batch9_panels(
    close: pd.DataFrame,
    csk_window: int = 120,
    mad_window: int = 60,
    ext_window: int = 20,
    mad_k: float = 1.96,
) -> Dict[str, pd.DataFrame]:
    """
    - db_csk_xyy_120: -CSK_XYY（个股对市场极端波动敏感度）
    - db_csk_down_120: -下行协偏度
    - df_ext_rev_20: -sum(极端日收益, 20)；极端=|r-中位|>k*MAD
    """
    ret = close.astype(float).pct_change()
    mkt = ret.mean(axis=1)

    csk = _rolling_csk_xyy(ret, mkt, csk_window, mode="all")
    csk_down = _rolling_csk_xyy(ret, mkt, csk_window, mode="down")

    med = ret.rolling(mad_window, min_periods=mad_window).median()
    mad = (ret - med).abs().rolling(mad_window, min_periods=mad_window).median()
    is_ext = (ret - med).abs() > (mad_k * mad.replace(0, np.nan))
    # 非极端记 0，便于滚动求和；MAD 未就绪处置空
    ext_ret = ret.where(is_ext, 0.0).where(mad.notna())
    ext_sum = ext_ret.rolling(ext_window, min_periods=ext_window).sum()
    df_ext_rev = -ext_sum

    return {
        "db_csk_xyy_120": (-csk).replace([np.inf, -np.inf], np.nan),
        "db_csk_down_120": (-csk_down).replace([np.inf, -np.inf], np.nan),
        "df_ext_rev_20": df_ext_rev.replace([np.inf, -np.inf], np.nan),
    }


def _rolling_csk_xyy(
    ret: pd.DataFrame,
    mkt: pd.Series,
    window: int,
    mode: str = "all",
) -> pd.DataFrame:
    """
    CSK_XYY = E[(X-EX)(Y-EY)^2] / (σx σy^2)
    mode=down/up 时仅在 Y</> EY 的子样本上估计。
    """
    ret_v = ret.to_numpy(dtype=float)
    mkt_v = mkt.to_numpy(dtype=float)
    n, m = ret_v.shape
    out = np.full((n, m), np.nan, dtype=float)
    min_obs = max(window // 3, 20)

    for i in range(window - 1, n):
        x = ret_v[i - window + 1 : i + 1]
        y = mkt_v[i - window + 1 : i + 1]
        y_mean = np.nanmean(y)
        if mode == "down":
            sel = y < y_mean
        elif mode == "up":
            sel = y > y_mean
        else:
            sel = np.ones(window, dtype=bool)
        if int(np.nansum(sel)) < min_obs:
            continue
        xs = x[sel]
        ys = y[sel]
        y_m = np.nanmean(ys)
        yd = ys - y_m
        x_m = np.nanmean(xs, axis=0)
        xd = xs - x_m
        with np.errstate(all="ignore"):
            num = np.nanmean(xd * (yd ** 2)[:, None], axis=0)
            sx = np.nanstd(xs, axis=0, ddof=0)
            sy = np.nanstd(ys, ddof=0)
            denom = sx * (sy ** 2)
            val = np.where(denom > 1e-12, num / denom, np.nan)
            # 窗口内有效点过少的列置空
            valid = np.sum(np.isfinite(xs), axis=0) >= min_obs
            out[i] = np.where(valid, val, np.nan)
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def batch9_lookback(csk_window: int = 120, mad_window: int = 60) -> int:
    return max(csk_window, mad_window) + 5
