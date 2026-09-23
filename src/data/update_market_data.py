"""
增量更新沪深 A 股日线（后复权）和交易状态（不复权）。

数据源：baostock
- data_daily：后复权 OHLC，成交额保留原始金额，成交量按 成交额/收盘价 对齐课上口径
- data_ud_new：不复权开盘价、昨收、涨跌停价、停牌、涨停、跌停
已有课上文件不会被覆盖。
"""

import argparse
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List, Optional, Sequence

import baostock as bs
import numpy as np
import pandas as pd


DAILY_COLS = ["date", "code", "open", "close", "low", "high", "volume", "money", "turnover_ratio"]
UD_COLS = ["date", "code", "open", "pre_close", "high_limit", "low_limit", "paused", "zt", "dt", "st"]
KLINE_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST"


def round2(value: float) -> float:
    """按四舍五入保留两位小数，对齐 A 股涨跌停价算法。"""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def to_project_code(bs_code: str) -> str:
    """sz.000001 / sh.600519 -> 000001.XSHE / 600519.XSHG"""
    market, number = bs_code.split(".")
    if market == "sz":
        return f"{number}.XSHE"
    if market == "sh":
        return f"{number}.XSHG"
    raise ValueError(f"不支持的市场代码: {bs_code}")


def is_ashare_code(bs_code: str) -> bool:
    """只保留沪深 A 股，排除指数、B 股、北交所。"""
    return bs_code.startswith(("sh.6", "sz.00", "sz.001", "sz.002", "sz.003", "sz.30"))


def limit_ratio(project_code: str, is_st: bool) -> float:
    """根据板块和 ST 状态返回涨跌停幅度。"""
    if is_st:
        return 0.05
    number = project_code.split(".")[0]
    if number.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def read_bs_result(result) -> pd.DataFrame:
    """把 baostock 查询结果转成 DataFrame。"""
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if not rows:
        return pd.DataFrame(columns=result.fields)
    return pd.DataFrame(rows, columns=result.fields)


def last_csv_date(folder: Path) -> Optional[str]:
    """返回目录中最后一份 YYYY-MM-DD.csv 的日期。"""
    files = sorted(folder.glob("????-??-??.csv"))
    if not files:
        return None
    return files[-1].stem


def migrate_data_dir(data_dir: Path) -> None:
    """
    若 data 仍指向课上目录的软链，则改成本地目录：
    日线和状态用硬链接（不占额外空间），Barra/收益/行业继续软链到课上。
    """
    if not data_dir.is_symlink():
        return

    src = data_dir.resolve()
    print(f"将数据目录从软链迁移到本地: {src} -> {data_dir}")
    data_dir.unlink()
    data_dir.mkdir()

    for name in ["data_daily", "data_ud_new"]:
        dst = data_dir / name
        dst.mkdir()
        for src_file in sorted((src / name).glob("*.csv")):
            os.link(src_file, dst / src_file.name)

    for name in ["data_barra", "data_ret", "data_industry"]:
        os.symlink(src / name, data_dir / name)

    shutil.copy2(src / "date.pkl", data_dir / "date.pkl")
    print("数据目录迁移完成")


def load_trade_dates(start_date: str, end_date: str) -> List[str]:
    """读取区间内的交易日。"""
    result = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    calendar = read_bs_result(result)
    if calendar.empty:
        return []
    trading = calendar[calendar["is_trading_day"] == "1"]
    return trading["calendar_date"].tolist()


def load_stock_table(start_date: str, end_date: str) -> pd.DataFrame:
    """读取在更新区间内上市过的沪深 A 股（含区间内退市）。"""
    result = bs.query_stock_basic()
    basic = read_bs_result(result)
    if basic.empty:
        raise RuntimeError("无法获取股票列表")

    stocks = basic[basic["type"] == "1"].copy()
    stocks = stocks[stocks["code"].map(is_ashare_code)]
    stocks = stocks[stocks["ipoDate"] <= end_date]
    stocks["outDate"] = stocks["outDate"].fillna("").astype(str).str.strip()
    stocks = stocks[(stocks["outDate"] == "") | (stocks["outDate"] >= start_date)]
    return stocks[["code", "outDate"]].reset_index(drop=True)


def cache_file(cache_dir: Path, bs_code: str) -> Path:
    return cache_dir / f"{bs_code.replace('.', '_')}.parquet"


def collect_pending_stocks(
    stocks: pd.DataFrame,
    cache_hfq: Path,
    cache_raw: Path,
    end_date: str,
) -> List[str]:
    """只拉缺失缓存，或仍在上市但缓存未到结束日的股票。"""
    pending = []
    for rec in stocks.itertuples(index=False):
        need = False
        for folder in (cache_hfq, cache_raw):
            path = cache_file(folder, rec.code)
            if not path.exists():
                need = True
                break
            if rec.outDate and rec.outDate < end_date:
                continue
            dates = pd.read_parquet(path, columns=["date"])
            if dates.empty:
                continue
            if str(dates["date"].max()) < end_date:
                need = True
                break
        if need:
            pending.append(rec.code)
    return pending


def ensure_login() -> None:
    """会话失效时重新登录 baostock。"""
    try:
        bs.logout()
    except Exception:
        pass
    time.sleep(0.3)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")


def fetch_kline(bs_code: str, start_date: str, end_date: str, adjustflag: str) -> pd.DataFrame:
    """拉取单只股票一段日线，失败或掉线时重试。"""
    last_error = ""
    for attempt in range(5):
        result = bs.query_history_k_data_plus(
            bs_code,
            KLINE_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
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
        numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame
    raise RuntimeError(f"{bs_code} 拉取失败: {last_error}")


def load_cached_kline(
    cache_dir: Path,
    bs_code: str,
    start_date: str,
    end_date: str,
    adjustflag: str,
) -> pd.DataFrame:
    """按股票缓存日线，已缓存到结束日则直接读本地。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{bs_code.replace('.', '_')}.parquet"
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        if not cached.empty and str(cached["date"].max()) >= end_date:
            return cached[cached["date"].between(start_date, end_date)].copy()
        fetch_start = start_date
        if not cached.empty:
            fetch_start = str(cached["date"].max())
        extra = fetch_kline(bs_code, fetch_start, end_date, adjustflag)
        if extra.empty:
            cached.to_parquet(cache_file, index=False)
            return cached[cached["date"].between(start_date, end_date)].copy()
        merged = pd.concat([cached, extra], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date")
        merged.to_parquet(cache_file, index=False)
        return merged[merged["date"].between(start_date, end_date)].copy()

    frame = fetch_kline(bs_code, start_date, end_date, adjustflag)
    if not frame.empty:
        frame.to_parquet(cache_file, index=False)
    return frame


def build_daily_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """把后复权 K 线转成课上 data_daily 列。"""
    if frame.empty:
        return pd.DataFrame(columns=DAILY_COLS)
    out = pd.DataFrame()
    out["date"] = frame["date"]
    out["code"] = frame["code"].map(to_project_code)
    out["open"] = frame["open"].map(round2)
    out["close"] = frame["close"].map(round2)
    out["low"] = frame["low"].map(round2)
    out["high"] = frame["high"].map(round2)
    close = frame["close"].replace(0, pd.NA)
    out["volume"] = (frame["amount"] / close).fillna(0.0).round(1)
    out["money"] = frame["amount"].round(1)
    out["turnover_ratio"] = frame["turn"].fillna(0.0).round(4)
    return out[DAILY_COLS]


def build_ud_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """把不复权 K 线转成课上 data_ud_new 列。"""
    if frame.empty:
        return pd.DataFrame(columns=UD_COLS)
    valid = frame["preclose"].notna() & (frame["preclose"] > 0)
    src = frame.loc[valid].copy()
    codes = src["code"].map(to_project_code)
    numbers = codes.str.split(".").str[0]
    is_st = src["isST"].astype(str) == "1"
    chi_next = numbers.str.match(r"^(300|301|688|689)")
    ratio = np.where(is_st, 0.05, np.where(chi_next, 0.20, 0.10))
    pre = src["preclose"].astype(float)
    close = src["close"].astype(float)
    high_limit = (pre * (1.0 + ratio)).map(round2)
    low_limit = (pre * (1.0 - ratio)).map(round2)
    paused = (src["tradestatus"].astype(str) == "0").astype(float)
    close_r = close.map(round2)
    return pd.DataFrame(
        {
            "date": src["date"],
            "code": codes,
            "open": src["open"].map(round2),
            "pre_close": pre.map(round2),
            "high_limit": high_limit,
            "low_limit": low_limit,
            "paused": paused,
            "zt": ((paused == 0) & (close_r >= high_limit)).astype(int),
            "dt": ((paused == 0) & (close_r <= low_limit)).astype(int),
            "st": is_st.astype(int),
        },
        columns=UD_COLS,
    )


def write_by_date(frame: pd.DataFrame, output_dir: Path, skip_existing: bool) -> int:
    """按日期落盘 CSV，返回新写文件数。"""
    if frame.empty:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for date, group in frame.groupby("date", sort=True):
        path = output_dir / f"{date}.csv"
        if skip_existing and path.exists():
            continue
        group = group.sort_values("code")
        group.to_csv(path, index=False)
        written += 1
    return written


def _cache_chunk(payload: tuple) -> int:
    """子进程：登录 baostock 后把一批股票的日线写入本地缓存。"""
    codes, start_date, end_date, cache_hfq, cache_raw = payload
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")
    done = 0
    try:
        for idx, bs_code in enumerate(codes, start=1):
            if idx % 15 == 0:
                ensure_login()
            try:
                load_cached_kline(Path(cache_hfq), bs_code, start_date, end_date, "1")
                load_cached_kline(Path(cache_raw), bs_code, start_date, end_date, "3")
                done += 1
            except RuntimeError as exc:
                print(f"跳过 {bs_code}: {exc}")
    finally:
        bs.logout()
    return done


def cache_all_stocks(
    stocks: Sequence[str],
    start_date: str,
    end_date: str,
    cache_hfq: Path,
    cache_raw: Path,
    workers: int,
) -> None:
    """并行把全部股票缓存到本地。"""
    cache_hfq.mkdir(parents=True, exist_ok=True)
    cache_raw.mkdir(parents=True, exist_ok=True)
    chunk_size = 40
    chunks = [list(stocks[i : i + chunk_size]) for i in range(0, len(stocks), chunk_size)]
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _cache_chunk,
                (chunk, start_date, end_date, str(cache_hfq), str(cache_raw)),
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            completed += future.result()
            print(f"已拉取 {completed}/{len(stocks)}")


def assemble_from_cache(cache_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """读取缓存中指定区间的全部日线。"""
    frames = []
    for path in sorted(cache_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame[frame["date"].between(start_date, end_date)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def materialize_ret_dir(data_dir: Path) -> Path:
    """若 data_ret 仍是课上软链，改成本地目录并用硬链接继承旧文件。"""
    ret_dir = data_dir / "data_ret"
    if ret_dir.is_symlink():
        src = ret_dir.resolve()
        print(f"将 data_ret 从软链迁移到本地: {src}")
        ret_dir.unlink()
        ret_dir.mkdir()
        for src_file in src.glob("*.csv"):
            os.link(src_file, ret_dir / src_file.name)
        print(f"已继承课上收益文件 {len(list(ret_dir.glob('*.csv')))} 个")
    ret_dir.mkdir(parents=True, exist_ok=True)
    return ret_dir


def _daily_vwap(daily_dir: Path, date: str) -> pd.DataFrame:
    """读取单日 VWAP = 成交额 / 成交量。"""
    frame = pd.read_csv(daily_dir / f"{date}.csv")
    volume = frame["volume"].replace(0, pd.NA)
    frame["vwap"] = frame["money"] / volume
    return frame[["code", "vwap"]]


def update_forward_returns(data_dir: Path, skip_existing: bool = True) -> int:
    """
    用日线补全前瞻收益。T 日文件存 T 到 T+n 的 VWAP 收益，与课上 data_ret 口径一致。
    """
    ret_dir = materialize_ret_dir(data_dir)
    daily_dir = data_dir / "data_daily"
    dates = sorted(path.stem for path in daily_dir.glob("????-??-??.csv"))
    if len(dates) < 2:
        print("日线不足，无法计算前瞻收益")
        return 0

    pending = []
    for i, date in enumerate(dates[:-1]):
        path = ret_dir / f"{date}.csv"
        if skip_existing and path.exists():
            continue
        pending.append(i)
    if not pending:
        print("前瞻收益已齐，无需补写")
        return 0

    print(f"待补收益文件 {len(pending)} 个，起始日 {dates[pending[0]]}")
    cache: dict = {}

    def get_vwap(date: str) -> pd.DataFrame:
        if date not in cache:
            if len(cache) >= 20:
                cache.pop(next(iter(cache)))
            cache[date] = _daily_vwap(daily_dir, date)
        return cache[date]

    written = 0
    horizons = (1, 5, 10, 15)
    for n, i in enumerate(pending, start=1):
        date = dates[i]
        base = get_vwap(date).rename(columns={"vwap": "vwap0"})
        for horizon in horizons:
            col = f"{horizon}vwap_pct"
            j = i + horizon
            if j >= len(dates):
                base[col] = pd.NA
                continue
            future = get_vwap(dates[j]).rename(columns={"vwap": "vwap_f"})
            base = base.merge(future, on="code", how="left")
            base[col] = base["vwap_f"] / base["vwap0"] - 1.0
            base = base.drop(columns=["vwap_f"])
        base["date"] = date
        out = base[["code", "date", "1vwap_pct", "5vwap_pct", "10vwap_pct", "15vwap_pct"]]
        out = out.dropna(subset=["1vwap_pct"])
        out.to_csv(ret_dir / f"{date}.csv", index=False)
        written += 1
        if n % 100 == 0 or n == len(pending):
            print(f"收益进度: {n}/{len(pending)}")
    print(f"新增收益文件 {written} 个，最后一日 {last_csv_date(ret_dir)}")
    return written


def update_date_pkl(data_dir: Path, extra_dates: List[str]) -> None:
    """把新交易日并入 date.pkl，供 DataLoader 识别。"""
    pkl_path = data_dir / "date.pkl"
    old = []
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            old = pickle.load(f)
    merged = sorted(set(old) | set(extra_dates))
    with open(pkl_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"date.pkl 已更新，共 {len(merged)} 个日期，最后一日 {merged[-1]}")


def parse_args():
    parser = argparse.ArgumentParser(description="增量更新日线和交易状态")
    parser.add_argument("--data-dir", dest="data_dir", default="./data")
    parser.add_argument("--start", default="", help="起始日期，默认从本地最后一日的下一交易日开始")
    parser.add_argument("--end", default="", help="结束日期，默认到最近一个交易日")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有日期文件")
    parser.add_argument("--workers", type=int, default=4, help="并行拉取进程数")
    parser.add_argument("--returns-only", action="store_true", help="只补前瞻收益，不拉行情")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    migrate_data_dir(data_dir)
    if args.returns_only:
        update_forward_returns(data_dir, skip_existing=not args.overwrite)
        return

    daily_dir = data_dir / "data_daily"
    ud_dir = data_dir / "data_ud_new"
    cache_root = data_dir / "_cache"

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")

    try:
        calendar = load_trade_dates("2014-01-01", time.strftime("%Y-%m-%d"))
        if not calendar:
            raise RuntimeError("无法获取交易日历")
        today = time.strftime("%Y-%m-%d")
        default_end = max(d for d in calendar if d <= today)

        last_daily = last_csv_date(daily_dir)
        last_ud = last_csv_date(ud_dir)
        if args.start:
            start_date = args.start
        else:
            anchors = [d for d in [last_daily, last_ud] if d]
            start_from = min(anchors) if anchors else calendar[0]
            later = [d for d in calendar if d > start_from]
            start_date = later[0] if later else default_end

        end_date = args.end or default_end
        need_dates = [d for d in calendar if start_date <= d <= end_date]
        if not need_dates:
            print("没有需要更新的交易日")
            return

        print(f"更新区间: {need_dates[0]} ~ {need_dates[-1]}，共 {len(need_dates)} 个交易日")
        print(f"本地日线最后一日: {last_daily}，状态最后一日: {last_ud}")

        stocks = load_stock_table(need_dates[0], need_dates[-1])
        pending = collect_pending_stocks(
            stocks,
            cache_root / "hfq",
            cache_root / "raw",
            need_dates[-1],
        )
        print(f"股票池 {len(stocks)}，待补缓存 {len(pending)}，并行进程: {args.workers}")

        if pending:
            cache_all_stocks(
                pending,
                need_dates[0],
                need_dates[-1],
                cache_root / "hfq",
                cache_root / "raw",
                workers=args.workers,
            )

        skip_existing = not args.overwrite
        hfq_all = assemble_from_cache(cache_root / "hfq", need_dates[0], need_dates[-1])
        raw_all = assemble_from_cache(cache_root / "raw", need_dates[0], need_dates[-1])
        daily_all = build_daily_rows(hfq_all)
        ud_all = build_ud_rows(raw_all)

        n_daily = write_by_date(daily_all, daily_dir, skip_existing)
        n_ud = write_by_date(ud_all, ud_dir, skip_existing)
        update_date_pkl(data_dir, calendar)
        print(f"新增日线文件 {n_daily} 个，新增状态文件 {n_ud} 个")
        print(f"日线目录最后一日: {last_csv_date(daily_dir)}")
        print(f"状态目录最后一日: {last_csv_date(ud_dir)}")
        update_forward_returns(data_dir, skip_existing=skip_existing)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
