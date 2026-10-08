"""
平滑换因子：冻结名单变更时，在 horizon 交易日内把合成权重从旧名单过渡到新名单。

用法：在 composite_universe.json 增加：
  "smooth_replace": {
    "prev": "factors/composite_universe_prev.json",
    "start_date": "2026-10-01",
    "horizon_days": 20
  }
未配置或 prev 不存在时不启用（日更行为不变）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def load_factor_list(path: str | Path) -> List[str]:
    """读取冻结名单中的因子序列（去重保序）。"""
    import json

    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    factors = [str(x) for x in meta.get("factors", []) if str(x).strip()]
    seen = set()
    ordered: List[str] = []
    for name in factors:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def blend_progress(days_since: int, horizon_days: int) -> float:
    """
    过渡进度：0=全旧名单，1=全新名单。
    horizon_days<=0 时一步到位。
    """
    if horizon_days <= 0:
        return 1.0
    if days_since <= 0:
        return 0.0
    return float(min(1.0, days_since / float(horizon_days)))


def blend_weight_vectors(
    w_new: pd.Series,
    w_prev: pd.Series,
    progress: float,
) -> pd.Series:
    """按进度混合新旧权重，再归一到和为 1（全零则等权）。"""
    idx = list(dict.fromkeys(list(w_prev.index) + list(w_new.index)))
    a = w_prev.reindex(idx).fillna(0.0).astype(float)
    b = w_new.reindex(idx).fillna(0.0).astype(float)
    p = float(np.clip(progress, 0.0, 1.0))
    mixed = (1.0 - p) * a + p * b
    s = float(mixed.sum())
    if s <= 0:
        return pd.Series(1.0 / max(len(idx), 1), index=idx)
    return mixed / s


def resolve_smooth_replace(
    universe_meta: dict,
    universe_path: str | Path,
) -> Optional[dict]:
    """
    解析 smooth_replace 配置；无效则返回 None。
    prev 路径相对仓库根或相对当前 universe 文件目录。
    """
    cfg = universe_meta.get("smooth_replace")
    if not cfg or not cfg.get("enabled", True):
        return None
    prev = cfg.get("prev") or cfg.get("prev_path")
    start = cfg.get("start_date")
    horizon = int(cfg.get("horizon_days", 20))
    if not prev or not start:
        return None
    uni = Path(universe_path)
    prev_path = Path(prev)
    if not prev_path.is_absolute():
        cand = uni.parent / prev_path
        if cand.exists():
            prev_path = cand
        elif not prev_path.exists():
            # 相对项目根
            root_cand = Path(".") / prev
            if root_cand.exists():
                prev_path = root_cand
    if not prev_path.exists():
        print(f"警告: smooth_replace.prev 不存在，跳过平滑换因子: {prev_path}")
        return None
    return {
        "prev_path": prev_path,
        "start_date": str(start),
        "horizon_days": horizon,
        "prev_factors": load_factor_list(prev_path),
    }


def apply_smooth_replace_to_daily_weights(
    daily_weights: pd.DataFrame,
    dates: Sequence[str],
    w_new_static: pd.Series,
    w_prev_static: pd.Series,
    start_date: str,
    horizon_days: int,
) -> pd.DataFrame:
    """
    对每日权重做旧→新过渡。
    daily_weights 可为 None（用静态新权重扩成日表）。
    """
    date_list = [str(d) for d in dates]
    cols = list(dict.fromkeys(list(w_prev_static.index) + list(w_new_static.index)))
    if daily_weights is None:
        base = pd.DataFrame(
            [w_new_static.reindex(cols).fillna(0.0).values] * len(date_list),
            index=date_list,
            columns=cols,
        )
    else:
        base = daily_weights.reindex(columns=cols).fillna(0.0)
        # 缺行用静态新权重填
        for d in date_list:
            if d not in base.index:
                base.loc[d] = w_new_static.reindex(cols).fillna(0.0)
        base = base.reindex(date_list).fillna(0.0)

    if start_date not in date_list:
        # 起始日不在本批日期内：整段按相对 start 的进度
        start_idx = -1
        for i, d in enumerate(date_list):
            if d >= start_date:
                start_idx = i
                break
        if start_idx < 0:
            # 全部早于 start：保持旧权重
            out = pd.DataFrame(
                [w_prev_static.reindex(cols).fillna(0.0).values] * len(date_list),
                index=date_list,
                columns=cols,
            )
            return out.div(out.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        start_idx = date_list.index(start_date)

    rows = []
    for i, d in enumerate(date_list):
        if i < start_idx:
            prog = 0.0
            w = blend_weight_vectors(w_new_static, w_prev_static, prog)
        else:
            prog = blend_progress(i - start_idx, horizon_days)
            # 新侧用当日滚动/静态权重（若有），旧侧用静态旧名单
            w_new_day = base.loc[d]
            s = float(w_new_day.sum())
            if s > 0:
                w_new_day = w_new_day / s
            else:
                w_new_day = w_new_static.reindex(cols).fillna(0.0)
            w = blend_weight_vectors(w_new_day, w_prev_static, prog)
        rows.append(w.reindex(cols).fillna(0.0))
    return pd.DataFrame(rows, index=date_list, columns=cols)
