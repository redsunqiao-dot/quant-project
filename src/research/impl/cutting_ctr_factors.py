"""
切割论 / 换手率切割刀（CTR）日频可落地子集。

无分钟数据时用「滚动窗内按中位数切割」近似课上 CTR / 聪明动量切割：
- 低换手的日收益 vs 高换手的日收益
- 低成交量的日收益 vs 高成交量的日收益
- 高振幅日的反转（理想反转切割的日频代理）
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _masked_mean(
    ret: pd.DataFrame,
    mask: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """滚动窗内仅在 mask 为 True 的交易日上求收益均值。"""
    w = float(window)
    min_count = max(window // 2, 5)
    m = mask.astype(float)
    num = (ret * m).rolling(window, min_periods=min_count).sum()
    den = m.rolling(window, min_periods=min_count).sum()
    return num / den.replace(0, np.nan)


def compute_cutting_ctr_panels(
    close: pd.DataFrame,
    turn: pd.DataFrame,
    volume: pd.DataFrame,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    - ctr_turn_spread_20: 窗内低换日均收益 - 高换日均收益（CTR 核心）
    - ctr_vol_cut_20: 窗内低成交量日均收益 - 高成交量日均收益（聪明动量日频代理）
    - ctr_ideal_amp_rev_20: 高振幅日收益均值取负（理想反转切割代理）
    """
    ret = close.astype(float).pct_change()
    turn_f = turn.astype(float)
    vol_f = volume.astype(float)

    turn_med = turn_f.rolling(window, min_periods=window).median()
    is_low_turn = turn_f <= turn_med
    is_high_turn = turn_f > turn_med
    mom_low_turn = _masked_mean(ret, is_low_turn, window)
    mom_high_turn = _masked_mean(ret, is_high_turn, window)
    ctr_turn_spread = (mom_low_turn - mom_high_turn).replace([np.inf, -np.inf], np.nan)

    vol_med = vol_f.rolling(window, min_periods=window).median()
    is_low_vol = vol_f <= vol_med
    is_high_vol = vol_f > vol_med
    mom_low_vol = _masked_mean(ret, is_low_vol, window)
    mom_high_vol = _masked_mean(ret, is_high_vol, window)
    ctr_vol_cut = (mom_low_vol - mom_high_vol).replace([np.inf, -np.inf], np.nan)

    if high is not None and low is not None:
        amp = high.astype(float) / low.astype(float).replace(0, np.nan) - 1.0
    else:
        amp = ret.abs()
    amp_med = amp.rolling(window, min_periods=window).median()
    is_high_amp = amp > amp_med
    ctr_ideal_amp_rev = (-_masked_mean(ret, is_high_amp, window)).replace(
        [np.inf, -np.inf], np.nan
    )

    return {
        "ctr_turn_spread_20": ctr_turn_spread,
        "ctr_vol_cut_20": ctr_vol_cut,
        "ctr_ideal_amp_rev_20": ctr_ideal_amp_rev,
    }


def cutting_ctr_lookback(window: int = 20) -> int:
    return window + 5
