"""
因子活跃态（Episodic）：用滚动 IC 判定因子当日是否参与合成。

规则（仅用过去信息，无前视）：
- 对每个因子算日度 Spearman IC；
- 滚动 window 日 IC 均值 > 0 且滚动 IR > 阈值 → 活跃；
- 合成日 t 使用截至 t-1 的滚动统计（再 shift 1）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.common.utils import calc_ic_by_date, normalize_weights


def build_ic_wide(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    ret_col: str = "ret",
) -> pd.DataFrame:
    """宽表：index=date，columns=factor，值=当日 IC。"""
    series = {}
    for col in factor_cols:
        if col not in panel.columns:
            continue
        ic = calc_ic_by_date(panel, col, ret_col=ret_col)
        if not ic.empty:
            series[col] = ic
    if not series:
        return pd.DataFrame()
    wide = pd.DataFrame(series).sort_index()
    return wide


def rolling_active_mask(
    ic_wide: pd.DataFrame,
    window: int = 20,
    min_ic_mean: float = 0.0,
    min_ir: float = 0.10,
) -> pd.DataFrame:
    """
    返回与 ic_wide 同形的 0/1 活跃掩码（已对统计量做 lag-1）。
    样本不足 window 时默认保持活跃，避免冷启动把因子全关掉。
    """
    if ic_wide.empty:
        return ic_wide.copy()

    roll_mean = ic_wide.rolling(window, min_periods=max(5, window // 2)).mean()
    roll_std = ic_wide.rolling(window, min_periods=max(5, window // 2)).std()
    roll_ir = roll_mean / roll_std.replace(0.0, np.nan)

    # 用昨日状态决定今日是否纳入（避免用到当日 IC）
    mean_lag = roll_mean.shift(1)
    ir_lag = roll_ir.shift(1)
    active = (mean_lag > min_ic_mean) & (ir_lag > min_ir)

    # 冷启动：历史不足时视为活跃
    warm = ic_wide.notna().cumsum() >= max(5, window // 2)
    active = active.where(warm.shift(1).fillna(False), True)
    return active.fillna(True).astype(int)


def apply_active_gate(
    base_weights: pd.Series,
    active_mask: pd.DataFrame,
    dates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    将静态权重按日门控并逐日归一。
    返回 index=date, columns=factor 的权重表。
    """
    cols = [c for c in base_weights.index if c in active_mask.columns]
    if not cols:
        return pd.DataFrame()

    mask = active_mask[cols]
    if dates is not None:
        mask = mask.reindex([d for d in dates if d in mask.index])

    base = base_weights.reindex(cols).fillna(0.0).clip(lower=0.0)
    rows = []
    index = []
    for dt, row in mask.iterrows():
        w = base * row.astype(float)
        w = normalize_weights(w)
        rows.append(w.values)
        index.append(dt)
    return pd.DataFrame(rows, index=index, columns=cols)


def summarize_active_share(active_mask: pd.DataFrame) -> pd.Series:
    """各因子活跃日占比，便于诊断。"""
    if active_mask.empty:
        return pd.Series(dtype=float)
    return active_mask.mean().sort_values()
