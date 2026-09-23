"""
东吴 UBL 因子（研报口径）。

UBL_raw = zscore(desize(蜡烛上_std)) + zscore(desize(威廉下_mean))
入库方向：ubl = -UBL_raw（越大越好，与管线一致）。

时间序列特征：
1) 蜡烛上影 Upper = High - max(Open, Close)
2) 威廉下影 WMS_Lower = min(Open, Close) - rolling_min(Low, N)
3) 5 日滚动标准化 → 再取 20 日 std / mean
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def rolling_zscore_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """按列做滚动 Z-Score。"""
    mean = df.rolling(window, min_periods=window).mean()
    std = df.rolling(window, min_periods=window).std()
    return (df - mean) / std.replace(0, np.nan)


def shadow_mean_std(
    raw: pd.DataFrame,
    z_window: int = 10,
    feat_window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """原始影线 → 滚动 z → feat 窗 mean/std。"""
    z = rolling_zscore_df(raw, z_window)
    feat_mean = z.rolling(feat_window, min_periods=feat_window).mean()
    feat_std = z.rolling(feat_window, min_periods=feat_window).std()
    return feat_mean, feat_std


def desize_cross_section(factor: pd.Series, size: pd.Series) -> pd.Series:
    """单日截面：因子对 ln(市值) OLS 取残差。"""
    aligned = pd.concat([factor.rename("f"), size.rename("s")], axis=1).dropna()
    if len(aligned) < 30:
        return pd.Series(np.nan, index=factor.index)
    y = aligned["f"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(aligned)), aligned["s"].to_numpy(dtype=float)])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    out = pd.Series(np.nan, index=factor.index, dtype=float)
    out.loc[aligned.index] = resid
    return out


def cs_zscore(series: pd.Series) -> pd.Series:
    """截面 Z-Score。"""
    s = series.astype(float)
    std = s.std()
    if std == 0 or pd.isna(std):
        return s * 0.0
    return (s - s.mean()) / std


def build_component_panels(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    z_window: int = 10,
    feat_window: int = 20,
    wms_n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (蜡烛上_std, 威廉下_mean)，索引为日期、列为股票。
    """
    upper = high - np.maximum(open_, close)
    low_n = low.rolling(wms_n, min_periods=wms_n).min()
    wms_lower = np.minimum(open_, close) - low_n

    _, candle_up_std = shadow_mean_std(upper, z_window=z_window, feat_window=feat_window)
    wms_low_mean, _ = shadow_mean_std(wms_lower, z_window=z_window, feat_window=feat_window)
    return candle_up_std, wms_low_mean


def combine_ubl_cross_section(
    candle_up_std: pd.Series,
    wms_low_mean: pd.Series,
    size: pd.Series,
    larger_better: bool = True,
) -> pd.Series:
    """
    单日：desize → zscore → 相加；默认取负使越大越好。
    """
    a = desize_cross_section(candle_up_std, size)
    b = desize_cross_section(wms_low_mean, size)
    ubl_raw = cs_zscore(a) + cs_zscore(b)
    return -ubl_raw if larger_better else ubl_raw


def compute_ubl_by_date(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    size_panel: pd.DataFrame,
    dates: list,
    z_window: int = 10,
    feat_window: int = 20,
    wms_n: int = 20,
    larger_better: bool = True,
) -> pd.DataFrame:
    """
    批量计算若干交易日的 UBL。

    Returns
    -------
    DataFrame
        index=date, columns=code，值为 ubl
    """
    candle_up_std, wms_low_mean = build_component_panels(
        open_, high, low, close,
        z_window=z_window,
        feat_window=feat_window,
        wms_n=wms_n,
    )
    rows = {}
    for date in dates:
        if date not in candle_up_std.index or date not in size_panel.index:
            continue
        size = size_panel.loc[date]
        if isinstance(size, pd.DataFrame):
            size = size.iloc[:, 0]
        ubl = combine_ubl_cross_section(
            candle_up_std.loc[date],
            wms_low_mean.loc[date],
            size,
            larger_better=larger_better,
        )
        rows[date] = ubl
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.sort_index()


def ubl_lookback(z_window: int = 10, feat_window: int = 20, wms_n: int = 20) -> int:
    """计算 UBL 所需最少历史交易日。"""
    return z_window + feat_window + wms_n
