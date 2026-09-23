"""
国盛《高低位放量事件簇》日频近似。

研报原文日频定义（第二节）：
- 低位放量：收盘价处于过去 120 日 10% 分位及以下，且成交量 > 均值+1.5σ
- 高位放量：收盘价处于过去 120 日 90% 分位及以上，且成交量 > 均值+1.5σ

通道策略回看过去 5 日是否触发；低位正向、高位负向。
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_gs_hl_vol_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 120,
    low_q: float = 0.10,
    high_q: float = 0.90,
    vol_z: float = 1.5,
    event_window: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    gs_low_vol_evt_5  : 近 event_window 日低位放量事件次数（正向）
    gs_high_vol_evt_5 : -近 event_window 日高位放量事件次数（负向取反）
    gs_hl_vol_net_5   : 近 event_window 日 (低位次数 - 高位次数)
    """
    close_f = close.astype(float)
    vol_f = volume.astype(float).replace(0, np.nan)

    # 当日收盘在过去 lookback 日窗内的分位
    pct = close_f.rolling(lookback, min_periods=lookback).rank(pct=True)
    vol_mean = vol_f.rolling(lookback, min_periods=lookback).mean()
    vol_std = vol_f.rolling(lookback, min_periods=lookback).std()
    surge = vol_f > (vol_mean + vol_z * vol_std)

    low_evt = ((pct <= low_q) & surge).astype(float)
    high_evt = ((pct >= high_q) & surge).astype(float)

    # 窗口内有效观测不足时置空，避免全 0 假信号
    min_valid = max(3, event_window // 2)
    valid = (
        close_f.notna().rolling(event_window, min_periods=min_valid).sum() >= min_valid
    )
    low_sum = low_evt.rolling(event_window, min_periods=1).sum().where(valid)
    high_sum = high_evt.rolling(event_window, min_periods=1).sum().where(valid)
    net = (low_sum - high_sum).where(valid)

    # 连续软分数：分位偏离 × 放量 z（仅同向贡献），便于截面排序
    z = (vol_f - vol_mean) / vol_std.replace(0, np.nan)
    z_pos = z.clip(lower=0.0)
    low_soft = (z_pos * (low_q - pct).clip(lower=0.0)).where(pct.notna() & z.notna())
    high_soft = (-(z_pos * (pct - high_q).clip(lower=0.0))).where(pct.notna() & z.notna())
    soft_net = (low_soft - (-high_soft)).where(pct.notna() & z.notna())

    return {
        "gs_low_vol_evt_5": low_sum.replace([np.inf, -np.inf], np.nan),
        "gs_high_vol_evt_5": (-high_sum).replace([np.inf, -np.inf], np.nan),
        "gs_hl_vol_net_5": net.replace([np.inf, -np.inf], np.nan),
        "gs_low_vol_soft_120": low_soft.replace([np.inf, -np.inf], np.nan),
        "gs_high_vol_soft_120": high_soft.replace([np.inf, -np.inf], np.nan),
        "gs_hl_vol_soft_120": soft_net.replace([np.inf, -np.inf], np.nan),
    }


def gs_hl_vol_lookback(lookback: int = 120, event_window: int = 5) -> int:
    return lookback + event_window + 5
