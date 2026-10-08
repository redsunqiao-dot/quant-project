"""
用 Tushare Pro 按日截面增量更新估值（baostock 备源）。

环境变量：TUSHARE_TOKEN 或 TS_TOKEN

示例：
  export TUSHARE_TOKEN=你的token
  .venv/bin/python update_valuation_data_tushare.py
  .venv/bin/python update_valuation_data_tushare.py --start 2026-09-01 --end 2026-09-30 --overwrite

写入 data/data_valuation/{date}.csv
字段：date, code, pe_ttm, pb_mrq, ps_ttm, pcf_ttm
口径：daily_basic 的 pe_ttm / pb / ps_ttm / pcf_ncf_ttm
"""

from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.update_market_data import last_csv_date, write_by_date
from src.data.update_market_data_tushare import from_ts_code, resolve_token, yyyymmdd
from src.data.update_valuation_data import VALUATION_COLS

BASIC_FIELDS = "ts_code,trade_date,pe_ttm,pb,ps_ttm,pcf_ncf_ttm"
SLEEP_SEC = 0.35
MAX_RETRY = 3


def load_calendar(data_dir: Path) -> List[str]:
    """优先用本地 date.pkl；否则用 data_daily 文件名。"""
    pkl = data_dir / "date.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return [str(d) for d in pickle.load(f)]
    daily_dir = data_dir / "data_daily"
    return sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))


def resolve_val_dates(
    data_dir: Path,
    start: str,
    end: str,
    overwrite: bool = False,
) -> List[str]:
    """
    决定要补的交易日：默认对齐 data_daily 末日，只补估值目录缺失段。
    无估值文件时默认最近 240 个有日线的交易日。
    """
    calendar = load_calendar(data_dir)
    if not calendar:
        raise RuntimeError("无交易日历，请先具备 data/date.pkl 或 data_daily")

    daily_dir = data_dir / "data_daily"
    val_dir = data_dir / "data_valuation"
    today = datetime.now().strftime("%Y-%m-%d")
    last_daily = last_csv_date(daily_dir)
    default_end = min(last_daily, today) if last_daily else today
    end_date = end or default_end

    last_val = last_csv_date(val_dir)
    if start:
        start_date = start
    elif last_val and not overwrite:
        later = [d for d in calendar if last_val < d <= end_date]
        return later
    else:
        daily_dates = sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))
        if not daily_dates:
            raise FileNotFoundError("没有 data_daily，请先跑行情增量")
        start_date = daily_dates[max(0, len(daily_dates) - 240)]

    dates = [d for d in calendar if start_date <= d <= end_date]
    if not overwrite:
        dates = [d for d in dates if not (val_dir / f"{d}.csv").exists()]
    return dates


def fetch_daily_basic(pro, trade_date: str) -> pd.DataFrame:
    """拉取单日估值截面；失败重试。"""
    d = yyyymmdd(trade_date)
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            basic = pro.daily_basic(trade_date=d, fields=BASIC_FIELDS)
            time.sleep(SLEEP_SEC)
            if basic is None or basic.empty:
                raise RuntimeError(f"daily_basic 空：{trade_date}（检查积分/权限或日期）")
            return basic
        except Exception as exc:
            last_exc = exc
            print(f"daily_basic {trade_date} 第 {attempt} 次失败: {exc}")
            time.sleep(SLEEP_SEC * attempt)
    raise RuntimeError(f"daily_basic 失败 {trade_date}: {last_exc}")


def basic_to_valuation(basic: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """Tushare daily_basic → 项目估值列。"""
    frame = basic.copy()
    frame["code"] = frame["ts_code"].map(from_ts_code)
    frame = frame.dropna(subset=["code"])
    out = pd.DataFrame(
        {
            "date": trade_date,
            "code": frame["code"].astype(str),
            "pe_ttm": pd.to_numeric(frame["pe_ttm"], errors="coerce"),
            "pb_mrq": pd.to_numeric(frame["pb"], errors="coerce"),
            "ps_ttm": pd.to_numeric(frame["ps_ttm"], errors="coerce"),
            "pcf_ttm": pd.to_numeric(frame["pcf_ncf_ttm"], errors="coerce"),
        }
    )
    return out[VALUATION_COLS].sort_values("code").reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Tushare 增量补日频估值（baostock 备源）")
    p.add_argument("--data-dir", dest="data_dir", default="./data")
    p.add_argument("--start", default="", help="起始日期")
    p.add_argument("--end", default="", help="结束日期，默认对齐 data_daily 末日")
    p.add_argument("--token", default="", help="也可设环境变量 TUSHARE_TOKEN")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有估值文件")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = resolve_token(args.token)
    import tushare as ts

    pro = ts.pro_api(token)
    data_dir = Path(args.data_dir)
    val_dir = data_dir / "data_valuation"

    dates = resolve_val_dates(
        data_dir, args.start, args.end, overwrite=args.overwrite
    )
    if not dates:
        print(f"没有需要补的估值交易日（目录最后一日: {last_csv_date(val_dir)}）")
        return 0

    print(f"Tushare 估值待补 {dates[0]} ~ {dates[-1]}，共 {len(dates)} 天")
    frames: List[pd.DataFrame] = []
    for date in dates:
        try:
            basic = fetch_daily_basic(pro, date)
            day = basic_to_valuation(basic, date)
            frames.append(day)
            pe_n = int(day["pe_ttm"].notna().sum())
            pb_n = int(day["pb_mrq"].notna().sum())
            print(f"估值 {date}: {len(day)} 只，pe_ttm 非空 {pe_n}，pb_mrq 非空 {pb_n}")
        except Exception as exc:
            # 权限不足等硬错误直接失败，避免写半截区间
            raise RuntimeError(f"拉取估值失败 {date}: {exc}") from exc

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=VALUATION_COLS)
    written = write_by_date(frame, val_dir, skip_existing=not args.overwrite)
    print(f"新增估值文件 {written} 个，目录最后一日: {last_csv_date(val_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
