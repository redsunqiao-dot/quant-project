"""
日频批次8：开源长端动量 + 方正原始惊恐。
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_batch8_panels(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    mom_window: int = 160,
    keep_q: float = 0.7,
    panic_window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    - ky_long_mom_160: 160 日低振幅 70% 日收益之和（长端动量 1.0）
    - ky_long_mom2_160: 超额收益版 + 对 20 日涨跌幅截面中性（长端动量 2.0 简化）
    - fz_panic_ret_20: -sum(惊恐度*收益, 20)（原始惊恐收益，越大越好）
    """
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    ret = close.pct_change()

    amp_v1 = high / low.replace(0, np.nan) - 1.0
    amp_v2 = (high - low) / close.shift(1).replace(0, np.nan)
    mkt = ret.mean(axis=1)
    excess = ret.sub(mkt, axis=0)

    ky1 = _low_amp_ret_sum(ret, amp_v1, mom_window, keep_q)
    alpha_low = _low_amp_ret_sum(excess, amp_v2, mom_window, keep_q)
    rev20 = close / close.shift(20) - 1.0
    ky2 = _cs_residual(alpha_low, rev20)

    # 惊恐度 = |r-rm| / (|r|+|rm|+0.1)；原始惊恐收益 = sum(惊恐度*r)
    # 研报 RankIC 为负，取负使越大越好
    dev = ret.sub(mkt, axis=0).abs()
    base = ret.abs().add(mkt.abs(), axis=0) + 0.1
    saliency = dev / base
    panic_ret = (saliency * ret).rolling(panic_window, min_periods=panic_window).sum()
    fz_panic = -panic_ret

    return {
        "ky_long_mom_160": ky1.replace([np.inf, -np.inf], np.nan),
        "ky_long_mom2_160": ky2.replace([np.inf, -np.inf], np.nan),
        "fz_panic_ret_20": fz_panic.replace([np.inf, -np.inf], np.nan),
    }


def _low_amp_ret_sum(
    ret: pd.DataFrame,
    amp: pd.DataFrame,
    window: int,
    keep_q: float,
) -> pd.DataFrame:
    """窗口内振幅最低 keep_q 分位交易日的收益求和。"""
    ret_v = ret.to_numpy(dtype=float)
    amp_v = amp.to_numpy(dtype=float)
    n, m = ret_v.shape
    out = np.full((n, m), np.nan, dtype=float)
    min_valid = max(window // 2, 20)
    for i in range(window - 1, n):
        a = amp_v[i - window + 1 : i + 1]
        r = ret_v[i - window + 1 : i + 1]
        with np.errstate(all="ignore"):
            q = np.nanquantile(a, keep_q, axis=0)
            mask = a <= q
            # 振幅全空的列跳过
            valid_cnt = np.sum(~np.isnan(a), axis=0)
            s = np.nansum(np.where(mask & ~np.isnan(r), r, 0.0), axis=0)
            out[i] = np.where(valid_cnt >= min_valid, s, np.nan)
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def _cs_residual(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """逐日截面：y 对 x 一元回归残差。"""
    y_v = y.to_numpy(dtype=float)
    x_v = x.to_numpy(dtype=float)
    n, m = y_v.shape
    out = np.full((n, m), np.nan, dtype=float)
    for i in range(n):
        yy = y_v[i]
        xx = x_v[i]
        mask = np.isfinite(yy) & np.isfinite(xx)
        if mask.sum() < 30:
            continue
        xm = xx[mask] - xx[mask].mean()
        ym = yy[mask] - yy[mask].mean()
        denom = float(np.dot(xm, xm))
        if denom <= 1e-12:
            out[i, mask] = ym
            continue
        beta = float(np.dot(xm, ym) / denom)
        out[i, mask] = ym - beta * xm
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def batch8_lookback(mom_window: int = 160, panic_window: int = 20) -> int:
    return max(mom_window, panic_window, 20) + 5
