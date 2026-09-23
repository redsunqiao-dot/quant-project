"""
长江证券《高波环境下的有效因子》日频可复现子集。

研报以子类因子等权合成大类；此处只落地不依赖分钟/FF3/大单的价量式。
方向统一为越大越好（研报方向为 -1 的取负）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_cj_highvol_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    turnover: pd.DataFrame,
    skew_window: int = 20,
    corr_window: int = 20,
    cv_window: int = 20,
    mom_long: int = 240,
    mom_skip: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    返回因子面板（index=date, columns=code）。

    - cj_ret_skew_20: -rolling_skew(ret, 20)（研报收益偏度方向 -1）
    - cj_mom_240_skip20: close_{t-20}/close_{t-240}-1（长期动量剔近月）
    - cj_rank_mom_220: 日收益截面分位秩在 [t-240,t-20] 求和（约 220 日）
    - cj_corr_vol_close_20: -rolling_corr(volume, close, 20)（量价相关方向 -1）
    - cj_turn_cv_20: std(turn)/mean(turn)（换手变异系数，研报方向 +1）
    """
    close_f = close.astype(float)
    vol_f = volume.replace(0, np.nan).astype(float)
    turn_f = turnover.replace(0, np.nan).astype(float)
    ret = close_f.pct_change()

    skew = ret.rolling(skew_window, min_periods=skew_window).skew()
    cj_ret_skew_20 = -skew

    cj_mom_240_skip20 = close_f.shift(mom_skip) / close_f.shift(mom_long) - 1.0

    # 截面分位秩（越大涨幅秩越高），再对 [t-240, t-20] 求和
    cs_rank = ret.rank(axis=1, pct=True)
    win = mom_long - mom_skip
    cj_rank_mom_220 = cs_rank.shift(mom_skip).rolling(win, min_periods=win).sum()

    # 滚动相关：逐列对齐后用 cov/std 近似，避免 DataFrame.rolling.corr 版本差异
    mean_c = close_f.rolling(corr_window, min_periods=corr_window).mean()
    mean_v = vol_f.rolling(corr_window, min_periods=corr_window).mean()
    std_c = close_f.rolling(corr_window, min_periods=corr_window).std()
    std_v = vol_f.rolling(corr_window, min_periods=corr_window).std()
    cov = (close_f * vol_f).rolling(corr_window, min_periods=corr_window).mean() - mean_c * mean_v
    corr = cov / (std_c * std_v).replace(0, np.nan)
    cj_corr_vol_close_20 = -corr

    mu = turn_f.rolling(cv_window, min_periods=cv_window).mean()
    sd = turn_f.rolling(cv_window, min_periods=cv_window).std()
    cj_turn_cv_20 = sd / mu.replace(0, np.nan)

    return {
        "cj_ret_skew_20": cj_ret_skew_20,
        "cj_mom_240_skip20": cj_mom_240_skip20,
        "cj_rank_mom_220": cj_rank_mom_220,
        "cj_corr_vol_close_20": cj_corr_vol_close_20,
        "cj_turn_cv_20": cj_turn_cv_20,
    }


def cj_highvol_lookback(mom_long: int = 240) -> int:
    return mom_long + 5
