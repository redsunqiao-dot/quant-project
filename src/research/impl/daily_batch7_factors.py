"""
日频批次7：民生低波改进 + 西南相同工作日动量。
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_batch7_panels(
    close: pd.DataFrame,
    vol_window: int = 60,
    wd_weeks: int = 12,
) -> Dict[str, pd.DataFrame]:
    """
    - ms_dastd_60: -std(ret, 60)  民生经典低波 DASTD（约 3 个月）
    - ms_rank_vol_60: -std(截面收益分位, 60)  民生分位数波动
    - xn_same_wd_mom_12: 过去 12 个同工作日收益之和  西南相同工作日动量
    """
    close = close.astype(float)
    ret = close.pct_change()

    ms_dastd = -ret.rolling(vol_window, min_periods=vol_window).std()

    # 每日截面分位（升序百分位，低收益→低分位），再对其时间序列求波动
    rank = ret.rank(axis=1, pct=True, method="average")
    ms_rank_vol = -rank.rolling(vol_window, min_periods=vol_window).std()

    xn_same_wd = _same_weekday_mom(ret, n_weeks=wd_weeks)

    return {
        "ms_dastd_60": ms_dastd.replace([np.inf, -np.inf], np.nan),
        "ms_rank_vol_60": ms_rank_vol.replace([np.inf, -np.inf], np.nan),
        "xn_same_wd_mom_12": xn_same_wd.replace([np.inf, -np.inf], np.nan),
    }


def _same_weekday_mom(ret: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    """按工作日分组滚动求和，再拼回原日历。"""
    orig_index = ret.index
    dt_index = pd.to_datetime(orig_index)
    work = ret.copy()
    work.index = dt_index

    parts = []
    for wd in range(5):
        sub = work.loc[work.index.weekday == wd]
        if sub.empty:
            continue
        rolled = sub.rolling(n_weeks, min_periods=n_weeks).sum()
        parts.append(rolled)
    if not parts:
        return pd.DataFrame(np.nan, index=orig_index, columns=ret.columns)

    out = pd.concat(parts).sort_index()
    out = out.reindex(dt_index)
    out.index = orig_index
    return out


def batch7_lookback(vol_window: int = 60, wd_weeks: int = 12) -> int:
    return max(vol_window, wd_weeks * 5) + 5
