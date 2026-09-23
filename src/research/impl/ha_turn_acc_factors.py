"""
华安《加速换手因子》日频子集：放量上涨日加速换手。

研报 Rank IC 为负，入库取负使越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_ha_turn_acc_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    turn: pd.DataFrame,
    signal_n: int = 3,
    sum_n: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    日加速换手 = turn_t - turn_{t-1}
    放量上涨日：close > MA(close,3) 且 volume > MA(volume,3)
    ha_volup_turn_acc_20 = -sum(加速换手 | 放量上涨日, 过去20日)
    """
    turn_f = turn.astype(float)
    close_f = close.astype(float)
    vol_f = volume.astype(float).replace(0, np.nan)

    d_turn = turn_f - turn_f.shift(1)
    close_ma = close_f.rolling(signal_n, min_periods=signal_n).mean()
    vol_ma = vol_f.rolling(signal_n, min_periods=signal_n).mean()
    mask = (close_f > close_ma) & (vol_f > vol_ma)

    selected = d_turn.where(mask, 0.0)
    # 无有效信号的日子仍计 0；窗口不足保持 NaN
    raw = selected.rolling(sum_n, min_periods=max(5, sum_n // 2)).sum()
    # 整窗无数据时置空
    valid = turn_f.notna().rolling(sum_n, min_periods=max(5, sum_n // 2)).sum() >= max(5, sum_n // 2)
    factor = -raw.where(valid)

    return {
        "ha_volup_turn_acc_20": factor.replace([np.inf, -np.inf], np.nan),
    }


def ha_turn_acc_lookback(signal_n: int = 3, sum_n: int = 20) -> int:
    return signal_n + sum_n + 5
