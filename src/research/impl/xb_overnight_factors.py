"""
西部证券《用隔夜交易策略增强指数增强》日频因子子集。

研报用因子预测隔夜收益，Rank IC 多为负（越大隔夜越差）。
入库方向：取负，使越大越好，对接管线 1 日前瞻收益。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _ewm_halflife(df: pd.DataFrame, halflife: int, min_periods: int) -> pd.DataFrame:
    """按列做 EWM（半衰期），与研报 EMA 口径接近。"""
    return df.astype(float).ewm(halflife=halflife, min_periods=min_periods, adjust=False).mean()


def compute_xb_overnight_panels(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    ema_halflife: int = 60,
    ema_min_periods: int = 120,
) -> Dict[str, pd.DataFrame]:
    """
    - xb_vol_shock: -(volume / EMA_hl60(volume) - 1)
    - xb_amplitude: -((high - low) / close)
    - xb_intraday: -(close / open - 1)
    """
    vol = volume.replace(0, np.nan).astype(float)
    ema = _ewm_halflife(vol, ema_halflife, ema_min_periods)
    shock = vol / ema.replace(0, np.nan) - 1.0
    xb_vol_shock = -shock

    amp = (high.astype(float) - low.astype(float)) / close.astype(float).replace(0, np.nan)
    xb_amplitude = -amp

    intraday = close.astype(float) / open_.astype(float).replace(0, np.nan) - 1.0
    xb_intraday = -intraday

    return {
        "xb_vol_shock": xb_vol_shock.replace([np.inf, -np.inf], np.nan),
        "xb_amplitude": xb_amplitude.replace([np.inf, -np.inf], np.nan),
        "xb_intraday": xb_intraday.replace([np.inf, -np.inf], np.nan),
    }


def xb_overnight_lookback(ema_min_periods: int = 120) -> int:
    return ema_min_periods + 5
