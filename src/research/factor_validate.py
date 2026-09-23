"""
因子验证与失效监控（第九课模块 5）。

流程：IC / 分组单调性 / 换手粗估 → 准入判定 → 更新 FactorLibrary。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.common.utils import (
    calc_ic_by_date,
    calc_ic_summary,
    ensure_dir,
    group_returns,
    load_factor_series,
    load_return_series,
)
from src.research.factor_library import FactorLibrary


def _mono_mean_from_panel(
    panel: pd.DataFrame,
    factor_col: str,
    ret_col: str = "ret",
    n_groups: int = 10,
) -> float:
    """逐日算分组单调性，再取均值。"""
    scores = []
    for _, day in panel.groupby("date"):
        factor = day.set_index("asset")[factor_col]
        ret = day.set_index("asset")[ret_col]
        _, mono = group_returns(factor, ret, n_groups=n_groups)
        if mono is not None and not (isinstance(mono, float) and np.isnan(mono)):
            scores.append(float(mono))
    if not scores:
        return float("nan")
    return float(np.nanmean(scores))


def _turnover_mean(
    factor_dir: str | Path,
    dates: Sequence[str],
    factor_col: str,
) -> float:
    """相邻日因子秩相关换手：turnover ≈ 1 - rank_ic。"""
    records = []
    for prev_date, curr_date in zip(dates[:-1], dates[1:]):
        prev = load_factor_series(str(factor_dir), prev_date, factor_col=factor_col)
        curr = load_factor_series(str(factor_dir), curr_date, factor_col=factor_col)
        if prev.empty or curr.empty:
            continue
        aligned = pd.concat([prev, curr], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            continue
        rank_ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
        if pd.isna(rank_ic):
            continue
        records.append(1.0 - float(rank_ic))
    if not records:
        return float("nan")
    return float(np.nanmean(records))


class FactorMonitor:
    """根据 IC 时间序列判断因子健康度。"""

    def __init__(
        self,
        warn_ic: float = 0.01,
        critical_drop: float = -0.005,
        lookback_short: int = 30,
        lookback_long: int = 250,
    ):
        self.warn_ic = warn_ic
        self.critical_drop = critical_drop
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def check(self, ic_series: pd.Series) -> str:
        """
        返回 HEALTHY / WARNING / CRITICAL。
        ic_series 按时间排序的日度 IC。
        """
        series = ic_series.dropna().astype(float)
        if series.empty:
            return "WARNING"
        current = float(series.iloc[-1])
        short = series.iloc[-self.lookback_short :]
        long = series.iloc[-self.lookback_long :]
        short_mean = float(short.mean()) if not short.empty else np.nan
        long_mean = float(long.mean()) if not long.empty else np.nan

        if current < 0:
            return "CRITICAL"
        if not np.isnan(short_mean) and not np.isnan(long_mean):
            if short_mean - long_mean < self.critical_drop:
                return "CRITICAL"
        if current < self.warn_ic:
            return "WARNING"
        return "HEALTHY"


def decide_status(
    metrics: Dict[str, float],
    min_abs_ic: float = 0.02,
    require_mono_sign: bool = True,
) -> str:
    """
    准入规则：
    - 因子方向已统一为「越大越好」，故要求 IC_mean >= min_abs_ic
    - 可选：分组单调性同为正（与 IC 同号）
    """
    ic_mean = metrics.get("ic_mean", np.nan)
    mono_mean = metrics.get("mono_mean", np.nan)
    if pd.isna(ic_mean) or float(ic_mean) < min_abs_ic:
        return "inactive"
    if require_mono_sign and not pd.isna(mono_mean):
        if float(mono_mean) < 0:
            return "inactive"
    return "active"


def validate_factor(
    panel: pd.DataFrame,
    factor_col: str,
    factor_dir: str | Path,
    dates: Sequence[str],
    ret_col: str = "ret",
    n_groups: int = 10,
    min_abs_ic: float = 0.02,
    monitor: Optional[FactorMonitor] = None,
) -> Dict[str, float | str]:
    """验证单个因子，返回指标字典（含 status / health）。"""
    if factor_col not in panel.columns:
        return {
            "factor": factor_col,
            "ic_mean": np.nan,
            "ic_ir": np.nan,
            "ic_std": np.nan,
            "mono_mean": np.nan,
            "turnover_mean": np.nan,
            "status": "inactive",
            "health": "WARNING",
        }

    ic_stats = calc_ic_summary(panel, factor_col, ret_col)
    mono_mean = _mono_mean_from_panel(panel, factor_col, ret_col=ret_col, n_groups=n_groups)
    turnover_mean = _turnover_mean(factor_dir, dates, factor_col)

    metrics = {
        "factor": factor_col,
        "ic_mean": ic_stats["ic_mean"],
        "ic_std": ic_stats["ic_std"],
        "ic_ir": ic_stats["ic_ir"],
        "mono_mean": mono_mean,
        "turnover_mean": turnover_mean,
    }
    status = decide_status(metrics, min_abs_ic=min_abs_ic)
    monitor = monitor or FactorMonitor()
    ic_series = calc_ic_by_date(panel, factor_col, ret_col)
    health = monitor.check(ic_series)
    # 未准入时健康度至少 WARNING
    if status != "active" and health == "HEALTHY":
        health = "WARNING"
    metrics["status"] = status
    metrics["health"] = health
    return metrics


def validate_factors(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    factor_dir: str | Path,
    dates: Sequence[str],
    library: FactorLibrary,
    output_dir: str | Path,
    ret_col: str = "ret",
    n_groups: int = 10,
    min_abs_ic: float = 0.02,
) -> pd.DataFrame:
    """
    批量验证并写回注册表。
    结果写入 output_dir/validation_summary.csv。
    """
    monitor = FactorMonitor()
    rows = []
    now = datetime.now().strftime("%Y-%m-%d")
    for col in factor_cols:
        result = validate_factor(
            panel=panel,
            factor_col=col,
            factor_dir=factor_dir,
            dates=dates,
            ret_col=ret_col,
            n_groups=n_groups,
            min_abs_ic=min_abs_ic,
            monitor=monitor,
        )
        rows.append(result)
        library.update_metrics(
            name=col,
            metrics={
                "ic_mean": result["ic_mean"],
                "ic_std": result["ic_std"],
                "ic_ir": result["ic_ir"],
                "mono_mean": result["mono_mean"],
                "turnover_mean": result["turnover_mean"],
            },
            status=str(result["status"]),
            health=str(result["health"]),
            updated_at=now,
        )

    summary = pd.DataFrame(rows)
    out = ensure_dir(output_dir)
    summary.to_csv(out / "validation_summary.csv", index=False)
    return summary
