"""
日频批次：乖离率 / 换手变化率 / 理想振幅 / 日间反转-波动翻转。
方向统一为越大越好（研报 IC 为负的表达式取负）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_fz_bias_panels(close: pd.DataFrame, window: int = 60) -> Dict[str, pd.DataFrame]:
    """方正乖离率：-(close/MA - 1)。"""
    ma = close.astype(float).rolling(window, min_periods=window).mean()
    bias = close.astype(float) / ma.replace(0, np.nan) - 1.0
    return {"fz_bias_60": (-bias).replace([np.inf, -np.inf], np.nan)}


def compute_dw_pct_turn_panels(
    turn: pd.DataFrame,
    avg_n: int = 20,
    base_n: int = 40,
) -> Dict[str, pd.DataFrame]:
    """
    东吴 PctTurn20：
    每日 pct = turn / mean(turn 过去 base_n 日, 不含当日) - 1，
    再对最近 avg_n 日取均值，取负。
    """
    t = turn.astype(float)
    base = t.shift(1).rolling(base_n, min_periods=base_n).mean()
    pct = t / base.replace(0, np.nan) - 1.0
    mean_pct = pct.rolling(avg_n, min_periods=avg_n).mean()
    return {"dw_pct_turn_20": (-mean_pct).replace([np.inf, -np.inf], np.nan)}


def compute_ky_ideal_amp_panels(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
    lam: float = 0.25,
) -> Dict[str, pd.DataFrame]:
    """
    开源理想振幅：V = mean(amp|高价λ日) - mean(amp|低价λ日)，取负。
    amp = high/low - 1。
    """
    amp = high.astype(float) / low.astype(float).replace(0, np.nan) - 1.0
    c = close.astype(float)
    k = max(1, int(round(window * lam)))

    # 逐日滚动：用过去 window 日截面内按收盘价切高/低价态
    dates = list(c.index)
    codes = list(c.columns)
    out = pd.DataFrame(index=c.index, columns=codes, dtype=float)

    amp_v = amp.values
    close_v = c.values
    n_dates, n_codes = amp_v.shape

    for i in range(window - 1, n_dates):
        a_win = amp_v[i - window + 1 : i + 1]  # (W, C)
        c_win = close_v[i - window + 1 : i + 1]
        # 按列：高/低价日振幅均值
        # argsort close ascending → 前 k 低价，后 k 高价
        order = np.argsort(c_win, axis=0)  # (W, C)
        # gather amp by order
        # for each col j: low idx = order[:k,j], high idx = order[-k:,j]
        low_idx = order[:k, :]
        high_idx = order[-k:, :]
        # advanced indexing per column
        cols_idx = np.arange(n_codes)[None, :]
        amp_low = a_win[low_idx, cols_idx].mean(axis=0)
        amp_high = a_win[high_idx, cols_idx].mean(axis=0)
        out.iloc[i] = -(amp_high - amp_low)

    return {"ky_ideal_amp_20": out.replace([np.inf, -np.inf], np.nan)}


def compute_fz_rev_vol_flip_panels(
    close: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    方正「日间反转-波动翻转」：
    mean_ret、std_ret 用过去 window 日；
    截面上 vol < 均值则翻转 mean_ret 符号；
    研报 Rank IC 为负 → 入库再取负。
    """
    ret = close.astype(float).pct_change()
    mean_ret = ret.rolling(window, min_periods=window).mean()
    vol = ret.rolling(window, min_periods=window).std()
    vol_mean = vol.mean(axis=1)
    coin = vol.lt(vol_mean, axis=0)  # 低波=硬币，翻转
    flipped = mean_ret.where(~coin, -mean_ret)
    return {"fz_rev_vol_flip_20": (-flipped).replace([np.inf, -np.inf], np.nan)}


def batch4_lookback() -> int:
    return 60 + 45  # bias60 + pct_turn base
