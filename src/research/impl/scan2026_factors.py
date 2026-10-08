"""
2026 课包增量：日频可落地的研报因子。

- ha_path_convex_20: 华安学海拾珠「股价路径凸性」日频代理
- sw_chip_cost_60: 申万「筹码平均成本相对现价」衰减近似
- yd_gp_delta: 源达「毛利率变动」用 PIT gp_margin 相对约一季前之差
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_path_convex_panels(close: pd.DataFrame, window: int = 20) -> Dict[str, pd.DataFrame]:
    """
    凸性 = (首尾中点 - 窗口均价) / 中点。
    文献：凸性负向预测收益，故取负。
    """
    px = close.astype(float)
    first = px.shift(window - 1)
    mid = (first + px) / 2.0
    mean = px.rolling(window, min_periods=window).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        conv = (mid - mean) / mid.replace(0, np.nan)
    factor = (-conv).replace([np.inf, -np.inf], np.nan)
    return {"ha_path_convex_20": factor}


def compute_chip_cost_panels(
    close: pd.DataFrame,
    turn: pd.DataFrame,
    money: pd.DataFrame,
    volume: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    每日新筹码按 VWAP 入场，存量按换手均匀衰减。
    因子 = (持仓均价 - 收盘) / 收盘，浮亏为正，偏反转。
    换手为百分数（0.21 表示 0.21%）。
    """
    px = close.astype(float)
    to = (turn.reindex_like(px).astype(float) / 100.0).clip(lower=0.0, upper=0.99)
    vol = volume.reindex_like(px).astype(float).replace(0, np.nan)
    amt = money.reindex_like(px).astype(float)
    vwap = (amt / vol).fillna(px)

    n_dates, n_codes = px.shape
    chip = np.zeros(n_codes, dtype=float)
    cost = np.zeros(n_codes, dtype=float)
    out = np.full((n_dates, n_codes), np.nan, dtype=float)

    px_v = px.to_numpy(dtype=float)
    to_v = to.to_numpy(dtype=float)
    amt_v = amt.to_numpy(dtype=float)
    vwap_v = vwap.to_numpy(dtype=float)

    for i in range(n_dates):
        remain = 1.0 - np.where(np.isfinite(to_v[i]), to_v[i], 0.0)
        remain = np.clip(remain, 0.0, 1.0)
        a = np.where(np.isfinite(amt_v[i]) & (amt_v[i] > 0), amt_v[i], 0.0)
        v = vwap_v[i]
        p = px_v[i]
        old = chip * remain
        new = old + a
        add = np.where(np.isfinite(v), v * a, 0.0)
        old_mass = cost * old
        cost = np.where(new > 0, (old_mass + add) / new, cost)
        chip = new
        valid = np.isfinite(p) & (p > 0) & (chip > 0)
        out[i] = np.where(valid, (cost - p) / p, np.nan)

    frame = pd.DataFrame(out, index=px.index, columns=px.columns)
    return {"sw_chip_cost_60": frame.replace([np.inf, -np.inf], np.nan)}


def compute_gp_delta_panels(fundamental: pd.DataFrame, lag: int = 60) -> Dict[str, pd.DataFrame]:
    """毛利率 TTM 相对 lag 日前 PIT 之差（报告更新时跳动）。"""
    if fundamental is None or fundamental.empty or "gp_margin" not in fundamental.columns:
        return {}
    gp = (
        fundamental.pivot_table(index="date", columns="code", values="gp_margin", aggfunc="last")
        .sort_index()
        .astype(float)
    )
    delta = gp - gp.shift(lag)
    return {"yd_gp_delta": delta.replace([np.inf, -np.inf], np.nan)}


def scan2026_lookback() -> int:
    return 120
