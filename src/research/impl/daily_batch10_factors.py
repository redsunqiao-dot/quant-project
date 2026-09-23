"""
日频批次10：上行/复合协偏度 + Garman-Klass 低波。
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.research.impl.daily_batch9_factors import _rolling_csk_xyy


def compute_batch10_panels(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    csk_window: int = 120,
    gk_window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    - db_csk_up_120: 上行协偏度（研报 RankIC 为正）
    - db_csk_comp_120: 截面 z(上行) - z(下行原始)
    - gk_vol_20: -mean(Garman-Klass, 20)
    """
    open_ = open_.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    ret = close.pct_change()
    mkt = ret.mean(axis=1)

    csk_up = _rolling_csk_xyy(ret, mkt, csk_window, mode="up")
    csk_down = _rolling_csk_xyy(ret, mkt, csk_window, mode="down")
    # 复合：上行减下行（下行原始值高=更糟；研报为正交化后相减，这里用截面 z 近似）
    comp = _cs_z(csk_up) - _cs_z(csk_down)

    # Garman-Klass 日波动，再取窗口均值后取负
    hl = np.log(high / low.replace(0, np.nan))
    co = np.log(close / open_.replace(0, np.nan))
    gk = 0.5 * (hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (co ** 2)
    gk_vol = -gk.rolling(gk_window, min_periods=gk_window).mean()

    return {
        "db_csk_up_120": csk_up.replace([np.inf, -np.inf], np.nan),
        "db_csk_comp_120": comp.replace([np.inf, -np.inf], np.nan),
        "gk_vol_20": gk_vol.replace([np.inf, -np.inf], np.nan),
    }


def _cs_z(panel: pd.DataFrame) -> pd.DataFrame:
    """逐日截面标准化。"""
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def batch10_lookback(csk_window: int = 120, gk_window: int = 20) -> int:
    return max(csk_window, gk_window) + 5
