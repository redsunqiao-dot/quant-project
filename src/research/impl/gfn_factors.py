"""
国金 GFlowNet 研报图表 18 举例因子（日频 OHLCV 可复现子集）。

方向统一为越大越好：研报 IC 为负的表达式取负。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """截面去均值。"""
    return df.sub(df.mean(axis=1), axis=0)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面标准化。"""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _cs_winsorize(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """截面分位去极值。"""
    lo = df.quantile(lower_q, axis=1)
    hi = df.quantile(upper_q, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def compute_gfn_example_panels(
    open_: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    返回若干因子面板（index=date, columns=code）。

    - gfn_logvol_var10: -ts_var(cs_demean(log(volume)), 10)
    - gfn_volpct_max20: -cs_zscore(ts_max(ts_pct_change(volume, 10), 20))
    - gfn_sqrt_open_low_vol: cs_winsorize(sqrt(open) / (low * volume))
    """
    vol = volume.replace(0, np.nan).astype(float)
    log_vol = np.log(vol)
    demeaned = _cs_demean(log_vol)
    # 滚动方差（样本方差）
    var10 = demeaned.rolling(10, min_periods=10).var()
    gfn_logvol_var10 = -var10

    vol_pct10 = vol.pct_change(10)
    max20 = vol_pct10.rolling(20, min_periods=20).max()
    gfn_volpct_max20 = -_cs_zscore(max20)

    raw = np.sqrt(open_.astype(float).clip(lower=0)) / (
        low.astype(float).replace(0, np.nan) * vol
    )
    gfn_sqrt_open_low_vol = _cs_winsorize(raw.replace([np.inf, -np.inf], np.nan))

    return {
        "gfn_logvol_var10": gfn_logvol_var10,
        "gfn_volpct_max20": gfn_volpct_max20,
        "gfn_sqrt_open_low_vol": gfn_sqrt_open_low_vol,
    }


def gfn_lookback() -> int:
    return 30
