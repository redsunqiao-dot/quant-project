"""
日更前数据新鲜度 / 齐套检查（无拉数，只读本地 data/）。

用于 CI 外的运维断言：日历末日、日线/涨跌停/收益是否对齐，
以及行业、估值等可选目录的覆盖提示。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.common.utils import list_dated_stems


REQUIRED_DIRS = ("data_daily", "data_ud_new", "data_ret")
OPTIONAL_DIRS = ("data_industry", "data_valuation", "data_fundamental")


@dataclass
class FreshnessReport:
    """一次质检结果。"""

    data_dir: Path
    calendar_last: Optional[str] = None
    dir_last: Dict[str, Optional[str]] = field(default_factory=dict)
    missing_vs_calendar: Dict[str, List[str]] = field(default_factory=dict)
    gaps_in_dir: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_calendar_dates(data_dir: Path) -> List[str]:
    """读取 date.pkl 交易日列表，统一为 YYYY-MM-DD 字符串。"""
    path = data_dir / "date.pkl"
    if not path.exists():
        raise FileNotFoundError(f"缺少交易日历: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out: List[str] = []
    for d in raw:
        if hasattr(d, "strftime"):
            out.append(d.strftime("%Y-%m-%d"))
        else:
            s = str(d).replace("/", "-")[:10]
            out.append(s)
    return sorted(set(out))


def _last_or_none(dates: Sequence[str]) -> Optional[str]:
    return dates[-1] if dates else None


def check_data_freshness(
    data_dir: str | Path = "./data",
    *,
    expect_calendar_last: Optional[str] = None,
    max_lag_days: int = 0,
    check_optional: bool = True,
) -> FreshnessReport:
    """
    检查本地数据齐套与新鲜度。

    - 必选目录 data_daily / data_ud_new / data_ret 的末日须与日历末日一致
      （可用 expect_calendar_last 覆盖「期望末日」）。
    - max_lag_days: 允许比日历末日落后的交易日个数（默认 0）。
    - 可选目录仅告警，不记入 errors。
    """
    root = Path(data_dir)
    report = FreshnessReport(data_dir=root)

    if not root.exists():
        report.errors.append(f"数据目录不存在: {root}")
        return report

    try:
        cal = load_calendar_dates(root)
    except FileNotFoundError as exc:
        report.errors.append(str(exc))
        return report

    if not cal:
        report.errors.append("交易日历为空")
        return report

    report.calendar_last = cal[-1]
    target = expect_calendar_last or report.calendar_last
    if target not in cal:
        report.errors.append(f"期望末日 {target} 不在 date.pkl 中")
        return report

    target_idx = cal.index(target)
    min_ok_idx = max(0, target_idx - max_lag_days)
    min_ok_date = cal[min_ok_idx]

    for name in REQUIRED_DIRS:
        folder = root / name
        if not folder.exists():
            report.errors.append(f"缺少必选目录: {folder}")
            report.dir_last[name] = None
            continue
        stems = list_dated_stems(folder)
        last = _last_or_none(stems)
        report.dir_last[name] = last
        if last is None:
            report.errors.append(f"{name} 无任何按日文件")
            continue
        if last < min_ok_date:
            report.errors.append(
                f"{name} 末日 {last} 落后于允许窗口（须 ≥ {min_ok_date}，日历/期望={target}）"
            )
        # 近窗缺日：从 min_ok_date 到 target，目录里应有文件
        need = [d for d in cal if min_ok_date <= d <= target]
        have = set(stems)
        missing = [d for d in need if d not in have]
        if missing:
            report.missing_vs_calendar[name] = missing
            report.errors.append(
                f"{name} 在 {min_ok_date}~{target} 缺 {len(missing)} 日，"
                f"例: {missing[:5]}"
            )

    if check_optional:
        for name in OPTIONAL_DIRS:
            folder = root / name
            if not folder.exists():
                report.warnings.append(f"可选目录缺失: {name}")
                report.dir_last[name] = None
                continue
            stems = list_dated_stems(folder)
            last = _last_or_none(stems)
            report.dir_last[name] = last
            if last is None:
                report.warnings.append(f"{name} 目录为空")
            elif last < min_ok_date:
                report.warnings.append(
                    f"{name} 末日 {last} 落后于 {min_ok_date}（可选，不阻断）"
                )

    return report


def format_report(report: FreshnessReport) -> str:
    """人类可读摘要。"""
    lines = [
        f"data_dir={report.data_dir}",
        f"calendar_last={report.calendar_last}",
        f"ok={report.ok}",
    ]
    for k, v in sorted(report.dir_last.items()):
        lines.append(f"  last[{k}]={v}")
    for e in report.errors:
        lines.append(f"ERROR: {e}")
    for w in report.warnings:
        lines.append(f"WARN: {w}")
    return "\n".join(lines)
