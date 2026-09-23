"""
用 baostock 更新季频财务，并按 pubDate 生成时点正确的日频截面。

原始缓存：data/_cache/fundamental/{bs_code}.parquet
日频 PIT：data/data_fundamental/{date}.csv

字段：
code, date, roe, np_margin, gp_margin, net_profit, eps_ttm,
yoy_ni, yoy_eps, pub_date, stat_date

注意：只用 pub_date <= 交易日 的最新财报，避免未来函数。
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import baostock as bs
import pandas as pd

from src.data.update_market_data import (
    cache_file,
    ensure_login,
    last_csv_date,
    load_stock_table,
    load_trade_dates,
    read_bs_result,
    to_project_code,
    write_by_date,
)


FUND_COLS = [
    "date",
    "code",
    "roe",
    "np_margin",
    "gp_margin",
    "net_profit",
    "eps_ttm",
    "yoy_ni",
    "yoy_eps",
    "pub_date",
    "stat_date",
]


def year_quarter_range(start_year: int, end_year: int) -> List[Tuple[int, int]]:
    """生成 (年, 季) 列表。"""
    out = []
    for year in range(start_year, end_year + 1):
        for quarter in (1, 2, 3, 4):
            out.append((year, quarter))
    return out


def fetch_profit(bs_code: str, year: int, quarter: int) -> pd.DataFrame:
    """单季利润指标。"""
    result = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
    if result.error_code != "0":
        return pd.DataFrame()
    return read_bs_result(result)


def fetch_growth(bs_code: str, year: int, quarter: int) -> pd.DataFrame:
    """单季成长指标。"""
    result = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
    if result.error_code != "0":
        return pd.DataFrame()
    return read_bs_result(result)


def fetch_stock_fundamentals(
    bs_code: str,
    periods: Sequence[Tuple[int, int]],
) -> pd.DataFrame:
    """拉取一只股票多个季度的利润+成长，按 pubDate/statDate 合并。"""
    rows = []
    for year, quarter in periods:
        profit = fetch_profit(bs_code, year, quarter)
        growth = fetch_growth(bs_code, year, quarter)
        if profit.empty and growth.empty:
            continue
        if profit.empty:
            merged = growth.copy()
        elif growth.empty:
            merged = profit.copy()
        else:
            keys = [c for c in ["code", "pubDate", "statDate"] if c in profit.columns and c in growth.columns]
            merged = profit.merge(growth, on=keys, how="outer")
        merged["year"] = year
        merged["quarter"] = quarter
        rows.append(merged)
        time.sleep(0.02)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    rename = {
        "pubDate": "pub_date",
        "statDate": "stat_date",
        "roeAvg": "roe",
        "npMargin": "np_margin",
        "gpMargin": "gp_margin",
        "netProfit": "net_profit",
        "epsTTM": "eps_ttm",
        "YOYNI": "yoy_ni",
        "YOYEPSBasic": "yoy_eps",
    }
    frame = frame.rename(columns=rename)
    keep = ["code", "pub_date", "stat_date", "roe", "np_margin", "gp_margin", "net_profit", "eps_ttm", "yoy_ni", "yoy_eps", "year", "quarter"]
    for col in keep:
        if col not in frame.columns:
            frame[col] = pd.NA
    out = frame[keep].copy()
    for col in ["roe", "np_margin", "gp_margin", "net_profit", "eps_ttm", "yoy_ni", "yoy_eps"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["pub_date"] = out["pub_date"].astype(str).str.strip()
    out["stat_date"] = out["stat_date"].astype(str).str.strip()
    out = out[out["pub_date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    out = out.drop_duplicates(subset=["pub_date", "stat_date"], keep="last")
    return out.sort_values(["pub_date", "stat_date"]).reset_index(drop=True)


def load_cached_fundamental(
    cache_dir: Path,
    bs_code: str,
    periods: Sequence[Tuple[int, int]],
    refresh: bool = False,
) -> pd.DataFrame:
    """按股票缓存季报；refresh=True 时强制重拉。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_file(cache_dir, bs_code)
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if not cached.empty:
            return cached
    frame = fetch_stock_fundamentals(bs_code, periods)
    if not frame.empty:
        frame.to_parquet(path, index=False)
    return frame


def _cache_chunk(payload: tuple) -> int:
    """子进程缓存一批股票的季报。"""
    codes, periods, cache_dir, refresh = payload
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")
    done = 0
    try:
        for idx, bs_code in enumerate(codes, start=1):
            if idx % 5 == 0:
                ensure_login()
                time.sleep(1.0)
            try:
                load_cached_fundamental(Path(cache_dir), bs_code, periods, refresh=refresh)
                done += 1
                time.sleep(0.15)
            except Exception as exc:
                print(f"跳过 {bs_code}: {exc}")
                time.sleep(1.0)
    finally:
        bs.logout()
    return done


def cache_all(
    stocks: Sequence[str],
    periods: Sequence[Tuple[int, int]],
    cache_dir: Path,
    workers: int,
    refresh: bool,
) -> None:
    """并行缓存季报。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 30
    chunks = [list(stocks[i : i + chunk_size]) for i in range(0, len(stocks), chunk_size)]
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_cache_chunk, (chunk, list(periods), str(cache_dir), refresh))
            for chunk in chunks
        ]
        for future in as_completed(futures):
            try:
                completed += future.result()
                print(f"财务缓存进度 {completed}/{len(stocks)}")
            except Exception as exc:
                print(f"批次失败: {exc}")
                time.sleep(2.0)


def load_all_fund_cache(cache_dir: Path) -> pd.DataFrame:
    """读取全部季报缓存。"""
    frames = []
    for path in sorted(cache_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["code"] = raw["code"].map(to_project_code)
    return raw


def build_pit_panel(fund: pd.DataFrame, dates: Sequence[str]) -> pd.DataFrame:
    """
    把季报展开成日频 PIT 截面。
    每个交易日取 pub_date <= 当日 的最新一条（pub_date 最大，同日再比 stat_date）。
    """
    if fund.empty or not dates:
        return pd.DataFrame(columns=FUND_COLS)

    fund = fund.copy()
    fund = fund[fund["pub_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    fund = fund.sort_values(["code", "pub_date", "stat_date"])
    rows = []
    date_index = pd.Index(sorted(dates))
    for code, group in fund.groupby("code", sort=False):
        g = group.drop_duplicates(subset=["pub_date"], keep="last")
        # 每个 pub_date 生效到下一份财报公布前
        pub_dates = g["pub_date"].tolist()
        for i, pub in enumerate(pub_dates):
            start = pub
            end = pub_dates[i + 1] if i + 1 < len(pub_dates) else "9999-12-31"
            # 半开区间 [start, end)
            use_dates = date_index[(date_index >= start) & (date_index < end)]
            if len(use_dates) == 0:
                continue
            rec = g.iloc[i]
            part = pd.DataFrame({"date": use_dates})
            part["code"] = code
            for col in ["roe", "np_margin", "gp_margin", "net_profit", "eps_ttm", "yoy_ni", "yoy_eps", "pub_date", "stat_date"]:
                part[col] = rec[col]
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=FUND_COLS)
    out = pd.concat(rows, ignore_index=True)
    return out[FUND_COLS]


def resolve_dates(data_dir: Path, start: str, end: str) -> List[str]:
    """PIT 落盘日期默认对齐估值或日线最近 240 天。"""
    calendar = load_trade_dates("2014-01-01", time.strftime("%Y-%m-%d"))
    today = time.strftime("%Y-%m-%d")
    default_end = max(d for d in calendar if d <= today)
    last_daily = last_csv_date(data_dir / "data_daily")
    if last_daily:
        default_end = min(default_end, last_daily)

    fund_dir = data_dir / "data_fundamental"
    last_fund = last_csv_date(fund_dir)
    if start:
        start_date = start
    elif last_fund:
        later = [d for d in calendar if d > last_fund]
        start_date = later[0] if later else default_end
    else:
        daily_dates = sorted(p.stem for p in (data_dir / "data_daily").glob("????-??-??.csv"))
        if not daily_dates:
            raise FileNotFoundError("没有 data_daily，请先跑 update_market_data.py")
        start_date = daily_dates[max(0, len(daily_dates) - 240)]
    end_date = end or default_end
    return [d for d in calendar if start_date <= d <= end_date]


def parse_args():
    parser = argparse.ArgumentParser(description="更新季频财务并生成 PIT 日截面（baostock）")
    parser.add_argument("--data-dir", dest="data_dir", default="./data")
    parser.add_argument("--start", default="", help="PIT 起始日期")
    parser.add_argument("--end", default="", help="PIT 结束日期")
    parser.add_argument("--year-start", type=int, default=2019, help="季报起始年")
    parser.add_argument("--year-end", type=int, default=0, help="季报结束年，默认今年")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 PIT 文件")
    parser.add_argument("--refresh-cache", action="store_true", help="强制重拉季报缓存")
    parser.add_argument("--skip-fetch", action="store_true", help="只用地图缓存生成 PIT")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    fund_dir = data_dir / "data_fundamental"
    cache_dir = data_dir / "_cache" / "fundamental"
    year_end = args.year_end or int(time.strftime("%Y"))
    periods = year_quarter_range(args.year_start, year_end)

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")

    try:
        need_dates = resolve_dates(data_dir, args.start, args.end)
        if not need_dates:
            print("没有需要更新的交易日")
            return
        print(f"财务 PIT 区间: {need_dates[0]} ~ {need_dates[-1]}，共 {len(need_dates)} 天")
        print(f"季报年份: {args.year_start} ~ {year_end}")

        stocks = load_stock_table(need_dates[0], need_dates[-1])
        codes = stocks["code"].tolist()
        if not args.skip_fetch:
            pending = []
            for code in codes:
                path = cache_file(cache_dir, code)
                if args.refresh_cache or not path.exists():
                    pending.append(code)
            print(f"股票池 {len(codes)}，待补财务缓存 {len(pending)}，并行 {args.workers}")
            if pending:
                cache_all(pending, periods, cache_dir, args.workers, refresh=args.refresh_cache)

        fund = load_all_fund_cache(cache_dir)
        if fund.empty:
            raise RuntimeError("财务缓存为空，无法生成 PIT")
        print(f"财务缓存记录 {len(fund)} 条，股票 {fund['code'].nunique()} 只")

        panel = build_pit_panel(fund, need_dates)
        written = write_by_date(panel, fund_dir, skip_existing=not args.overwrite)
        print(f"新增财务 PIT 文件 {written} 个，最后一日: {last_csv_date(fund_dir)}")
        if not panel.empty:
            sample = panel[panel["date"] == need_dates[-1]]
            print(
                f"样例 {need_dates[-1]}: {len(sample)} 只，"
                f"roe 非空 {sample['roe'].notna().sum()}，"
                f"yoy_ni 非空 {sample['yoy_ni'].notna().sum()}"
            )
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
