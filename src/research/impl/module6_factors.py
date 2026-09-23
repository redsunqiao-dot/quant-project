"""
模块 6 样例因子：估值 / 质量 / 成长 / PEG / 股息率。

数据：
- data_valuation：pe_ttm, pb_mrq
- data_fundamental：roe, yoy_ni, yoy_eps（PIT）
- data/_cache/dividend + 收盘价：股息率

方向统一为越大越好。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    if df is None or df.empty or value not in df.columns:
        return pd.DataFrame()
    return (
        df.pivot_table(index="date", columns="code", values=value, aggfunc="last")
        .sort_index()
        .astype(float)
    )


def compute_div_yield_panel(
    events: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """
    股息率：近 365 日已除权现金分红合计 / 当日收盘价。

    PIT：announce_date <= 交易日，且 event_date 落在 (t-365, t]。
    等价：交易日 t ∈ [max(event, announce), event+365)。
    """
    if events is None or events.empty or close is None or close.empty:
        return pd.DataFrame()

    need = {"code", "announce_date", "event_date", "cash_ps"}
    if not need.issubset(events.columns):
        return pd.DataFrame()

    ev = events.copy()
    ev["announce_date"] = pd.to_datetime(ev["announce_date"], errors="coerce")
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev["cash_ps"] = pd.to_numeric(ev["cash_ps"], errors="coerce")
    ev = ev.dropna(subset=["code", "announce_date", "event_date", "cash_ps"])
    ev = ev[ev["cash_ps"] > 0]
    if ev.empty:
        return pd.DataFrame()

    dates = pd.to_datetime(pd.Index(close.index).astype(str))
    date_vals = dates.to_numpy()
    code_to_j = {c: j for j, c in enumerate(close.columns)}
    cash = np.zeros((len(dates), len(close.columns)), dtype=float)

    for row in ev.itertuples(index=False):
        j = code_to_j.get(row.code)
        if j is None:
            continue
        start = max(row.announce_date, row.event_date)
        end = row.event_date + pd.Timedelta(days=365)
        i0 = int(np.searchsorted(date_vals, np.datetime64(start), side="left"))
        i1 = int(np.searchsorted(date_vals, np.datetime64(end), side="left"))
        if i0 < i1:
            cash[i0:i1, j] += float(row.cash_ps)

    px = close.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        dy = np.where(px > 0, cash / px, np.nan)
    dy = np.where(cash > 0, dy, np.nan)
    return pd.DataFrame(dy, index=close.index, columns=close.columns).replace(
        [np.inf, -np.inf], np.nan
    )


def compute_module6_panels(
    valuation: pd.DataFrame,
    fundamental: pd.DataFrame,
    close: Optional[pd.DataFrame] = None,
    dividend_events: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    valuation / fundamental 为长表，需含 date, code 及对应字段。

    m6_ep_ttm   : 盈利收益率 1/PE（PE>0）
    m6_bp_mrq   : 账面市值比 1/PB（PB>0）
    m6_roe      : ROE
    m6_yoy_ni   : 净利润同比（截断极端）
    m6_peg_inv  : -PE/增速，增速取 yoy_eps 与 yoy_ni 的较大者且 >5%
    m6_dy       : 近一年现金分红 / 收盘价（需分红缓存 + close）
    """
    pe = _pivot(valuation, "pe_ttm")
    pb = _pivot(valuation, "pb_mrq")
    roe = _pivot(fundamental, "roe")
    yoy_ni = _pivot(fundamental, "yoy_ni")
    yoy_eps = _pivot(fundamental, "yoy_eps")

    out: Dict[str, pd.DataFrame] = {}
    if not pe.empty:
        out["m6_ep_ttm"] = (1.0 / pe.where(pe > 0)).replace([np.inf, -np.inf], np.nan)
    if not pb.empty:
        out["m6_bp_mrq"] = (1.0 / pb.where(pb > 0)).replace([np.inf, -np.inf], np.nan)
    if not roe.empty:
        out["m6_roe"] = roe.replace([np.inf, -np.inf], np.nan)
    if not yoy_ni.empty:
        out["m6_yoy_ni"] = yoy_ni.clip(lower=-2.0, upper=5.0).replace(
            [np.inf, -np.inf], np.nan
        )

    if not pe.empty and (not yoy_eps.empty or not yoy_ni.empty):
        if not yoy_eps.empty and not yoy_ni.empty:
            growth = yoy_eps.combine(yoy_ni, np.fmax)
        elif not yoy_eps.empty:
            growth = yoy_eps.copy()
        else:
            growth = yoy_ni.copy()
        growth = growth.clip(lower=-2.0, upper=5.0)
        peg = pe.where(pe > 0) / growth.where(growth > 0.05)
        out["m6_peg_inv"] = (-peg).replace([np.inf, -np.inf], np.nan)

    if close is not None and dividend_events is not None:
        dy = compute_div_yield_panel(dividend_events, close)
        if not dy.empty:
            out["m6_dy"] = dy

    return out


def module6_lookback() -> int:
    """截面因子，几乎无时序窗口。"""
    return 5


def load_valuation_long(data_dir: str | Path, dates: Sequence[str]) -> pd.DataFrame:
    """按日读取估值截面并纵向拼接。"""
    root = Path(data_dir) / "data_valuation"
    frames: List[pd.DataFrame] = []
    for date in dates:
        path = root / f"{date}.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path)
        if "date" not in part.columns:
            part["date"] = date
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_fundamental_long(data_dir: str | Path, dates: Sequence[str]) -> pd.DataFrame:
    """按日读取财务 PIT 截面并纵向拼接。"""
    root = Path(data_dir) / "data_fundamental"
    frames: List[pd.DataFrame] = []
    for date in dates:
        path = root / f"{date}.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path)
        if "date" not in part.columns:
            part["date"] = date
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
