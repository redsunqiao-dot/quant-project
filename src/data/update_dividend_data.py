"""
用 baostock 拉取分红事件并缓存。

原始缓存：data/_cache/dividend/{bs_code}.parquet
用途：模块6股息率 m6_dy = 近一年已宣告现金分红合计 / 收盘价
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

import baostock as bs
import pandas as pd

from src.data.update_market_data import (
    cache_file,
    ensure_login,
    is_ashare_code,
    load_stock_table,
    read_bs_result,
    to_project_code,
)

DIV_COLS = [
    "code",
    "dividPreNoticeDate",
    "dividAgmPumDate",
    "dividPlanAnnounceDate",
    "dividPlanDate",
    "dividRegistDate",
    "dividOperateDate",
    "dividPayDate",
    "dividCashPsBeforeTax",
    "dividStocksPs",
]


def fetch_dividend_year(bs_code: str, year: int) -> pd.DataFrame:
    """拉取单只股票单年分红。"""
    last_error = ""
    for attempt in range(5):
        result = bs.query_dividend_data(code=bs_code, year=str(year), yearType="report")
        if result.error_code != "0":
            last_error = result.error_msg
            if "未登录" in last_error or "异常" in last_error:
                ensure_login()
            time.sleep(0.4 * (attempt + 1))
            continue
        frame = read_bs_result(result)
        if frame.empty:
            return pd.DataFrame(columns=DIV_COLS)
        keep = [c for c in DIV_COLS if c in frame.columns]
        return frame[keep].copy()
    raise RuntimeError(f"{bs_code} {year} 分红拉取失败: {last_error}")


def load_cached_dividend(
    cache_dir: Path,
    bs_code: str,
    years: Sequence[int],
    refresh: bool = False,
) -> pd.DataFrame:
    """按股票缓存多年度分红事件。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_file(cache_dir, bs_code)
    cached = pd.DataFrame()
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)

    have_years = set()
    if not cached.empty and "year" in cached.columns:
        have_years = set(int(y) for y in cached["year"].dropna().unique())

    need_years = [y for y in years if y not in have_years]
    pieces = [cached] if not cached.empty else []
    for year in need_years:
        part = fetch_dividend_year(bs_code, year)
        if part.empty:
            # 占位，避免反复空拉
            part = pd.DataFrame([{c: None for c in DIV_COLS}])
            part["code"] = bs_code
        part["year"] = year
        pieces.append(part)
        time.sleep(0.05)

    frames = [f for f in pieces if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged["code"] = merged["code"].fillna(bs_code).astype(str)
    merged = merged.drop_duplicates(
        subset=[
            "code",
            "year",
            "dividPlanAnnounceDate",
            "dividOperateDate",
            "dividCashPsBeforeTax",
        ],
        keep="last",
    )
    merged.to_parquet(path, index=False)
    return merged


def _worker(args) -> str:
    cache_dir, bs_code, years, refresh = args
    load_cached_dividend(Path(cache_dir), bs_code, years, refresh=refresh)
    return bs_code


def cache_all(
    codes: Sequence[str],
    years: Sequence[int],
    cache_dir: Path,
    workers: int,
    refresh: bool,
) -> None:
    """批量补分红缓存。"""
    tasks = [(str(cache_dir), c, list(years), refresh) for c in codes]
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            try:
                _worker(task)
            except Exception as exc:
                print(f"警告: {task[1]} 失败: {exc}")
                ensure_login()
            if i % 50 == 0:
                print(f"分红缓存进度 {i}/{len(tasks)}")
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # 子进程内需各自登录
        def _worker_login(a):
            ensure_login()
            return _worker(a)

        futs = [pool.submit(_worker_login, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0:
                print(f"分红缓存进度 {done}/{len(tasks)}")


def load_all_dividend_events(cache_dir: Path) -> pd.DataFrame:
    """读入全部缓存并规范化为项目代码。"""
    files = sorted(cache_dir.glob("*.parquet"))
    frames: List[pd.DataFrame] = []
    for path in files:
        part = pd.read_parquet(path)
        if part.empty:
            continue
        frames.append(part)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["dividCashPsBeforeTax"], how="all")
    if raw.empty:
        return pd.DataFrame()

    raw["cash_ps"] = pd.to_numeric(raw["dividCashPsBeforeTax"], errors="coerce")
    raw = raw[raw["cash_ps"].notna() & (raw["cash_ps"] > 0)].copy()
    if raw.empty:
        return pd.DataFrame()

    # 事件生效日：除权日优先，其次派息/股权登记
    for col in [
        "dividOperateDate",
        "dividPayDate",
        "dividRegistDate",
        "dividPlanDate",
    ]:
        if col not in raw.columns:
            raw[col] = ""
        raw[col] = raw[col].fillna("").astype(str).str.strip()

    raw["event_date"] = raw["dividOperateDate"]
    raw.loc[raw["event_date"] == "", "event_date"] = raw["dividPayDate"]
    raw.loc[raw["event_date"] == "", "event_date"] = raw["dividRegistDate"]
    raw.loc[raw["event_date"] == "", "event_date"] = raw["dividPlanDate"]

    for col in [
        "dividPlanAnnounceDate",
        "dividAgmPumDate",
        "dividPreNoticeDate",
    ]:
        if col not in raw.columns:
            raw[col] = ""
        raw[col] = raw[col].fillna("").astype(str).str.strip()

    raw["announce_date"] = raw["dividPlanAnnounceDate"]
    raw.loc[raw["announce_date"] == "", "announce_date"] = raw["dividAgmPumDate"]
    raw.loc[raw["announce_date"] == "", "announce_date"] = raw["dividPreNoticeDate"]
    # 无公告日则退回事件日（保守：按事件日可见）
    raw.loc[raw["announce_date"] == "", "announce_date"] = raw["event_date"]

    raw = raw[(raw["event_date"] != "") & (raw["announce_date"] != "")].copy()

    def _norm_code(c: str) -> str:
        c = str(c)
        if c.startswith(("sh.", "sz.")):
            return to_project_code(c)
        return c

    raw["code"] = raw["code"].map(_norm_code)
    return raw[["code", "announce_date", "event_date", "cash_ps", "year"]].drop_duplicates()


def parse_args():
    parser = argparse.ArgumentParser(description="拉取分红事件缓存（baostock）")
    parser.add_argument("--data-dir", dest="data_dir", default="./data")
    parser.add_argument("--start", default="", help="用于确定股票池的起始交易日")
    parser.add_argument("--end", default="", help="用于确定股票池的结束交易日")
    parser.add_argument("--year-start", type=int, default=2022)
    parser.add_argument("--year-end", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="调试：最多拉 N 只")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "_cache" / "dividend"
    year_end = args.year_end or int(time.strftime("%Y"))
    years = list(range(args.year_start, year_end + 1))

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")

    try:
        end = args.end or time.strftime("%Y-%m-%d")
        start = args.start or end
        stocks = load_stock_table(start, end)
        codes = [c for c in stocks["code"].tolist() if is_ashare_code(c)]
        if args.limit > 0:
            codes = codes[: args.limit]

        pending = []
        for code in codes:
            path = cache_file(cache_dir, code)
            if args.refresh or not path.exists():
                pending.append(code)
            else:
                # 年份不全也补
                try:
                    cached = pd.read_parquet(path)
                    have = set(int(y) for y in cached.get("year", pd.Series(dtype=int)).dropna().unique())
                    if any(y not in have for y in years):
                        pending.append(code)
                except Exception:
                    pending.append(code)

        print(f"股票池 {len(codes)}，待补分红缓存 {len(pending)}，年份 {years[0]}~{years[-1]}")
        if pending:
            cache_all(pending, years, cache_dir, args.workers, refresh=args.refresh)

        events = load_all_dividend_events(cache_dir)
        print(f"有效分红事件 {len(events)} 条，股票 {events['code'].nunique() if not events.empty else 0} 只")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
