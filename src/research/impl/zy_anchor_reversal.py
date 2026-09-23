"""
中银国际锚定反转因子（日频近似）。

参数按研报：i=2 周≈10 日判趋势，j=13 周≈65 日定锚。
因子值越小越看多 → 入库取负。
可选：乘以锚点窗收益波动（研报增强二）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_zy_anchor_panels(
    close: pd.DataFrame,
    trend_i: int = 10,
    anchor_j: int = 65,
) -> Dict[str, pd.DataFrame]:
    """
    上涨：锚=窗口最低；下跌：锚=窗口最高；raw = close/anchor - 1。
    - zy_anchor_rev: -raw
    - zy_anchor_rev_vol: -raw * std(ret, j)
    """
    c = close.astype(float)
    roll_min = c.rolling(anchor_j, min_periods=anchor_j).min()
    roll_max = c.rolling(anchor_j, min_periods=anchor_j).max()
    up = c >= c.shift(trend_i)
    anchor = roll_min.where(up, roll_max)
    raw = c / anchor.replace(0, np.nan) - 1.0

    ret = c.pct_change()
    vol = ret.rolling(anchor_j, min_periods=anchor_j).std()

    return {
        "zy_anchor_rev": (-raw).replace([np.inf, -np.inf], np.nan),
        "zy_anchor_rev_vol": (-raw * vol).replace([np.inf, -np.inf], np.nan),
    }


def zy_anchor_lookback(trend_i: int = 10, anchor_j: int = 65) -> int:
    return max(trend_i, anchor_j) + 5
