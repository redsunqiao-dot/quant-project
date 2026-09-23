"""
因子分族：动量 / 反转 / 量价波动 分层估权。

动机（海外文献 Don't Mix Value & Momentum）：
方向冲突的信号硬合成会互相抵消；先族内估权，再族间合成更稳。
本仓库暂无完整估值因子，族划分按现有 active 价量语义。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from src.common.utils import normalize_weights


FAMILY_MOM = "mom"
FAMILY_REV = "rev"
FAMILY_VOL_LIQ = "vol_liq"
FAMILY_OTHER = "other"


def classify_family(name: str) -> str:
    """按因子名启发式分族。"""
    n = name.lower()

    # 反转优先（名称里常同时含 mom/rev 字样时以 rev 为准）
    rev_keys = (
        "reversal",
        "_rev",
        "rev_",
        "ext_rev",
        "intraday_rev",
        "ovn_rev",
        "idt_rev",
        "anchor_rev",
        "rev_vol",
        "rev_turn",
    )
    if any(k in n for k in rev_keys):
        return FAMILY_REV

    mom_keys = (
        "momentum",
        "long_mom",
        "rank_mom",
        "mild_mom",
        "lt_mom",
        "same_wd_mom",
        "mom_",
    )
    if any(k in n for k in mom_keys):
        return FAMILY_MOM

    vol_liq_keys = (
        "vol",
        "illiq",
        "turn",
        "amplitude",
        "intraday",
        "dastd",
        "range",
        "skew",
        "kurt",
        "corr_vol",
        "pv_corr",
        "semi_down",
        "idio",
        "gfn_",
        "xb_",
        "ha_",
        "yz_",
        "gk_",
        "rs_",
        "liquidity",
    )
    if any(k in n for k in vol_liq_keys):
        return FAMILY_VOL_LIQ

    return FAMILY_OTHER


def group_by_family(factor_cols: Sequence[str]) -> Dict[str, List[str]]:
    """族名 -> 因子列表（仅含非空族）。"""
    groups: Dict[str, List[str]] = {}
    for name in factor_cols:
        fam = classify_family(name)
        groups.setdefault(fam, []).append(name)
    return groups


def hierarchical_weights(
    metrics: pd.DataFrame,
    factor_cols: Sequence[str],
    metric_col: str = "ic_ir",
    family_scheme: str = "sqrt_size",
) -> pd.Series:
    """
    分层权重：
    1) 族内按 metric_col 归一；
    2) 族间：sqrt_size（默认，按 √成员数）/ equal / metric；
    3) 最终权重 = 族权 × 族内权。
    """
    cols = list(factor_cols)
    if not cols:
        return pd.Series(dtype=float)

    raw = metrics.reindex(cols)[metric_col].astype(float)
    groups = group_by_family(cols)

    fam_names = list(groups.keys())
    if family_scheme == "metric":
        fam_score = {
            fam: float(raw.reindex(members).clip(lower=0).mean())
            for fam, members in groups.items()
        }
    elif family_scheme == "equal":
        fam_score = {fam: 1.0 for fam in fam_names}
    else:
        # sqrt_size：抑制「单因子一族」独占 1/N 族权
        fam_score = {fam: float(len(members) ** 0.5) for fam, members in groups.items()}
    fam_w = normalize_weights(pd.Series(fam_score))

    out = pd.Series(0.0, index=cols, dtype=float)
    for fam, members in groups.items():
        within = normalize_weights(raw.reindex(members))
        out.loc[members] = within.values * float(fam_w.loc[fam])

    return normalize_weights(out)


def family_assignment_table(factor_cols: Sequence[str]) -> pd.DataFrame:
    """便于落盘检查的分族表。"""
    rows = [{"factor": n, "family": classify_family(n)} for n in factor_cols]
    return pd.DataFrame(rows)
