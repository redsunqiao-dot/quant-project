"""
用 baostock 增量更新日频估值截面。

写入 data/data_valuation/{date}.csv
字段：code, date, pe_ttm, pb_mrq, ps_ttm, pcf_ttm
口径：不复权日线附带的 TTM/MRQ 估值（adjustflag=3）
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
    last_csv_date,
    load_stock_table,
    load_trade_dates,
    read_bs_result,
    to_project_code,
    write_by_date,
)


VALUATION_FIELDS = "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM"
VALUATION_COLS = ["date", "code", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]


def fetch_valuation_kline(bs_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取单只股票日频估值字段。"""
    last_error = ""
    for attempt in range(5):
        result = bs.query_history_k_data_plus(
            bs_code,
            VALUATION_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        if result.error_code != "0":
            last_error = result.error_msg
            if "未登录" in last_error or "异常" in last_error:
                ensure_login()
            time.sleep(0.5 * (attempt + 1))
            continue
        frame = read_bs_result(result)
        if frame.empty:
            return frame
        for col in ["close", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame
    raise RuntimeError(f"{bs_code} 估值拉取失败: {last_error}")


def load_cached_valuation(
    cache_dir: Path,
    bs_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """按股票缓存估值日线；支持向前续增与向后回补。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_file(cache_dir, bs_code)
    cached = pd.DataFrame()
    if path.exists():
        cached = pd.read_parquet(path)

    pieces = []
    if cached.empty:
        pieces.append(fetch_valuation_kline(bs_code, start_date, end_date))
    else:
        cmin = str(cached["date"].min())
        cmax = str(cached["date"].max())
        # 向更早日期回补
        if cmin > start_date:
            pieces.append(fetch_valuation_kline(bs_code, start_date, cmin))
        pieces.append(cached)
        # 向更新日期续增
        if cmax < end_date:
            pieces.append(fetch_valuation_kline(bs_code, cmax, end_date))

    frames = [f for f in pieces if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    merged.to_parquet(path, index=False)
    return merged[merged["date"].between(start_date, end_date)].copy()


def collect_pending(stocks: pd.DataFrame, cache_dir: Path, start_date: str, end_date: str) -> List[str]:
    """找出缓存缺失，或未覆盖 [start_date, end_date] 的股票。"""
    pending = []
    for rec in stocks.itertuples(index=False):
        path = cache_file(cache_dir, rec.code)
        if not path.exists():
            pending.append(rec.code)
            continue
        if rec.outDate and rec.outDate < start_date:
            continue
        dates = pd.read_parquet(path, columns=["date"])
        if dates.empty:
            pending.append(rec.code)
            continue
        dmin = str(dates["date"].min())
        dmax = str(dates["date"].max())
        if dmax < end_date or dmin > start_date:
            pending.append(rec.code)
    return pending


def _cache_chunk(payload: tuple) -> int:
    """子进程批量缓存估值。"""
    codes, start_date, end_date, cache_dir = payload
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")
    done = 0
    try:
        for idx, bs_code in enumerate(codes, start=1):
            if idx % 30 == 0:
                ensure_login()
                time.sleep(1.0)
            try:
                load_cached_valuation(Path(cache_dir), bs_code, start_date, end_date)
                done += 1
                # baostock 对高频请求敏感，单进程也略作间隔
                time.sleep(0.05)
            except RuntimeError as exc:
                print(f"跳过 {bs_code}: {exc}")
                time.sleep(1.0)
    finally:
        bs.logout()
    return done


def cache_all(
    stocks: Sequence[str],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    workers: int,
) -> None:
    """并行缓存估值日线。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 50
    chunks = [list(stocks[i : i + chunk_size]) for i in range(0, len(stocks), chunk_size)]
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_cache_chunk, (chunk, start_date, end_date, str(cache_dir)))
            for chunk in chunks
        ]
        for future in as_completed(futures):
            try:
                completed += future.result()
                print(f"估值缓存进度 {completed}/{len(stocks)}")
            except Exception as exc:
                print(f"批次失败: {exc}")
                time.sleep(2.0)


def assemble_valuation(cache_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """把缓存拼成项目口径截面长表。"""
    frames = []
    for path in sorted(cache_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame[frame["date"].between(start_date, end_date)]
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=VALUATION_COLS)

    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "date": raw["date"],
            "code": raw["code"].map(to_project_code),
            "pe_ttm": raw["peTTM"],
            "pb_mrq": raw["pbMRQ"],
            "ps_ttm": raw["psTTM"],
            "pcf_ttm": raw["pcfNcfTTM"],
        }
    )
    return out[VALUATION_COLS]


def resolve_range(
    data_dir: Path,
    start: str,
    end: str,
) -> List[str]:
    """决定要补的交易日：默认对齐 data_daily，只补估值目录缺失段。"""
    daily_dir = data_dir / "data_daily"
    val_dir = data_dir / "data_valuation"
    calendar = load_trade_dates("2014-01-01", time.strftime("%Y-%m-%d"))
    if not calendar:
        raise RuntimeError("无法获取交易日历")
    today = time.strftime("%Y-%m-%d")
    default_end = max(d for d in calendar if d <= today)

    last_daily = last_csv_date(daily_dir)
    if last_daily:
        default_end = min(default_end, last_daily)

    last_val = last_csv_date(val_dir)
    if start:
        start_date = start
    elif last_val:
        later = [d for d in calendar if d > last_val]
        start_date = later[0] if later else default_end
    else:
        # 无估值文件时，默认只补最近 240 个有日线的交易日，避免首次全历史过慢
        daily_dates = sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))
        if not daily_dates:
            raise FileNotFoundError("没有 data_daily，请先跑 update_market_data.py")
        start_date = daily_dates[max(0, len(daily_dates) - 240)]

    end_date = end or default_end
    return [d for d in calendar if start_date <= d <= end_date]


def parse_args():
    parser = argparse.ArgumentParser(description="增量更新日频估值（baostock）")
    parser.add_argument("--data-dir", dest="data_dir", default="./data")
    parser.add_argument("--start", default="", help="起始日期")
    parser.add_argument("--end", default="", help="结束日期")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有估值文件")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数（baostock 建议 1，避免封禁）")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    val_dir = data_dir / "data_valuation"
    cache_dir = data_dir / "_cache" / "valuation"

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")

    try:
        need_dates = resolve_range(data_dir, args.start, args.end)
        if not need_dates:
            print("没有需要更新的交易日")
            return
        print(f"估值更新区间: {need_dates[0]} ~ {need_dates[-1]}，共 {len(need_dates)} 天")

        stocks = load_stock_table(need_dates[0], need_dates[-1])
        pending = collect_pending(stocks, cache_dir, need_dates[0], need_dates[-1])
        print(f"股票池 {len(stocks)}，待补估值缓存 {len(pending)}，并行 {args.workers}")

        if pending:
            cache_all(pending, need_dates[0], need_dates[-1], cache_dir, args.workers)

        frame = assemble_valuation(cache_dir, need_dates[0], need_dates[-1])
        written = write_by_date(frame, val_dir, skip_existing=not args.overwrite)
        print(f"新增估值文件 {written} 个，目录最后一日: {last_csv_date(val_dir)}")
        if not frame.empty:
            sample = frame[frame["date"] == need_dates[-1]]
            print(
                f"样例 {need_dates[-1]}: {len(sample)} 只，"
                f"pe_ttm 非空 {sample['pe_ttm'].notna().sum()}，"
                f"pb_mrq 非空 {sample['pb_mrq'].notna().sum()}"
            )
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
