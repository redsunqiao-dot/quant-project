"""
股票首次出现日（近似上市日）缓存。

用 data_daily 中该代码第一次出现的交易日作为 list_date。
baostock 可用时可用 ipoDate 覆盖；当前用于「上市不满 1 年」过滤。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def build_list_date_cache(
    daily_dir: str | Path = "./data/data_daily",
    cache_path: str | Path = "./data/_cache/list_dates.parquet",
    force: bool = False,
) -> pd.Series:
    """
    扫描日线，得到 code -> 首次出现日期。

    Returns
    -------
    Series
        index=code，值=YYYY-MM-DD 字符串
    """
    daily_dir = Path(daily_dir)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    first: Dict[str, str] = {}
    if cache_path.exists() and not force:
        old = pd.read_parquet(cache_path)
        if "code" in old.columns and "list_date" in old.columns:
            first = dict(zip(old["code"].astype(str), old["list_date"].astype(str)))

    dates = sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))
    if not dates:
        raise FileNotFoundError(f"无日线文件: {daily_dir}")

    known = set(first)
    meta_path = cache_path.with_suffix(".meta.txt")
    last_cached_scan: Optional[str] = None
    if meta_path.exists() and not force:
        last_cached_scan = meta_path.read_text(encoding="utf-8").strip() or None

    if force or not first:
        scan_dates = dates
    elif last_cached_scan:
        scan_dates = [d for d in dates if d >= last_cached_scan]
        if not scan_dates:
            scan_dates = [dates[-1]]
    else:
        scan_dates = dates

    for i, date in enumerate(scan_dates):
        path = daily_dir / f"{date}.csv"
        codes = pd.read_csv(path, usecols=["code"])["code"].astype(str)
        new_codes = [c for c in codes.unique() if c not in known]
        for c in new_codes:
            first[c] = date
            known.add(c)
        if (i + 1) % 200 == 0:
            print(f"list_date 扫描进度 {i+1}/{len(scan_dates)}")

    out = pd.DataFrame(
        {"code": list(first.keys()), "list_date": list(first.values())}
    ).sort_values("code")
    out.to_parquet(cache_path, index=False)
    meta_path.write_text(dates[-1], encoding="utf-8")
    return out.set_index("code")["list_date"]


def load_list_dates(
    cache_path: str | Path = "./data/_cache/list_dates.parquet",
    daily_dir: str | Path = "./data/data_daily",
) -> pd.Series:
    """读取缓存；不存在则构建。"""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return build_list_date_cache(daily_dir=daily_dir, cache_path=cache_path)
    df = pd.read_parquet(cache_path)
    return df.set_index("code")["list_date"].astype(str)
