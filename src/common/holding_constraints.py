"""
持仓硬约束：单行业只数上限、候选池截断到 Top-N。

等权组合下单票权重自然为 1/N；行业约束在选股时生效。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence


def effective_top_n(top_n: int, max_weight_per_stock: float = 0.0) -> int:
    """
    等权下单票权重 = 1/N；若设了单票上限，则 N 至少为 ceil(1/max_weight)。
    max_weight_per_stock <= 0 时不调整。
    """
    if top_n <= 0:
        return 0
    if max_weight_per_stock <= 0:
        return top_n
    need = int(math.ceil(1.0 / float(max_weight_per_stock) - 1e-12))
    return max(top_n, need)


def apply_industry_cap(
    ranked_codes: Sequence[str],
    industry_by_code: Mapping[str, str],
    top_n: int,
    max_per_industry: int = 2,
) -> List[str]:
    """
    按因子排名依次选股，同一行业最多 max_per_industry 只，直到满 top_n。

    industry 缺失的股票单独记为 "__NA__"，彼此不互占行业名额。
    max_per_industry <= 0 时退化为截断 Top-N。
    """
    if top_n <= 0:
        return []
    if max_per_industry <= 0:
        return [str(c) for c in ranked_codes[:top_n]]

    selected: List[str] = []
    counts: Dict[str, int] = defaultdict(int)
    for code in ranked_codes:
        c = str(code)
        ind = industry_by_code.get(c)
        if ind is None or (isinstance(ind, float) and ind != ind) or str(ind).strip() == "":
            key = f"__NA__::{c}"
        else:
            key = str(ind)
            if counts[key] >= max_per_industry:
                continue
            counts[key] += 1
        selected.append(c)
        if len(selected) >= top_n:
            break
    return selected


def load_industry_map(
    date: str,
    industry_dir: str = "./data/data_industry",
    code_col: str = "code",
    industry_col: str = "industry",
) -> Dict[str, str]:
    """读单日行业截面 → code -> industry。"""
    from pathlib import Path

    from src.common.utils import read_frame, resolve_dated_file

    path = resolve_dated_file(Path(industry_dir), date)
    if path is None:
        return {}
    df = read_frame(path)
    if df.empty or code_col not in df.columns:
        return {}
    col = industry_col if industry_col in df.columns else None
    if col is None:
        for cand in ("industry", "industry_name", "sw_l1", "申万一级"):
            if cand in df.columns:
                col = cand
                break
    if col is None:
        return {}
    out: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        code = str(getattr(row, code_col))
        ind = getattr(row, col)
        if ind is None or (isinstance(ind, float) and ind != ind):
            continue
        out[code] = str(ind)
    return out
