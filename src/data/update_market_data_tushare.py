"""
用 Tushare Pro 增量补日线 / 交易状态（baostock 备源）。

环境变量：
  TUSHARE_TOKEN 或 TS_TOKEN

单日覆盖示例（纠正 09-28 应急数据）:
  export TUSHARE_TOKEN=你的token
  .venv/bin/python update_market_data_tushare.py --start 2026-09-28 --end 2026-09-28 --overwrite

口径：
- data_daily：后复权价按「锚定日本地收盘 × (当日不复权价/昨收)」链式接上
- data_ud_new：不复权开盘/昨收 + 涨跌停（优先 stk_limit，否则按规则推算）
- turnover：来自 daily_basic.turnover_rate（百分数，与 baostock turn 一致）
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.data.update_market_data import (
    DAILY_COLS,
    UD_COLS,
    last_csv_date,
    round2,
    update_date_pkl,
    update_forward_returns,
    write_by_date,
)
from src.data.update_market_data_akshare import (
    build_rows_from_raw_day,
    need_dates,
    to_symbol,
)


def resolve_token(cli_token: str = "") -> str:
    token = (
        cli_token
        or os.environ.get("TUSHARE_TOKEN")
        or os.environ.get("TS_TOKEN")
        or ""
    ).strip()
    if not token:
        raise RuntimeError(
            "未找到 Tushare token。请设置环境变量 TUSHARE_TOKEN，"
            "或传 --token。注册: https://tushare.pro/register"
        )
    return token


def to_ts_code(project_code: str) -> str:
    num = to_symbol(project_code)
    if project_code.endswith(".XSHG"):
        return f"{num}.SH"
    return f"{num}.SZ"


def from_ts_code(ts_code: str) -> Optional[str]:
    raw = str(ts_code).strip()
    if "." not in raw:
        return to_project_code(raw)
    num, mkt = raw.split(".", 1)
    mkt = mkt.upper()
    if mkt == "SH":
        return f"{num.zfill(6)}.XSHG"
    if mkt == "SZ":
        return f"{num.zfill(6)}.XSHE"
    return None


def yyyymmdd(date: str) -> str:
    return date.replace("-", "")


def fetch_day_panels(pro, trade_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    拉取单日：不复权行情 + 每日指标(换手) + 涨跌停价。
    trade_date: YYYY-MM-DD
    """
    d = yyyymmdd(trade_date)
    daily = pro.daily(trade_date=d)
    time.sleep(0.35)
    basic = pro.daily_basic(
        trade_date=d,
        fields="ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio",
    )
    time.sleep(0.35)
    try:
        limit = pro.stk_limit(trade_date=d)
        time.sleep(0.35)
    except Exception as exc:
        print(f"stk_limit 不可用，将按规则推算涨跌停: {exc}")
        limit = pd.DataFrame()

    if daily is None or daily.empty:
        raise RuntimeError(f"Tushare daily 空：{trade_date}（检查积分/权限或日期）")
    if basic is None:
        basic = pd.DataFrame()
    return daily, basic, limit if limit is not None else pd.DataFrame()


def fill_one_day_tushare(
    pro,
    data_dir: Path,
    date: str,
    anchor_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tushare 单日 → data_daily / data_ud_new。"""
    daily_anchor = pd.read_csv(data_dir / "data_daily" / f"{anchor_date}.csv")
    hfq_map = daily_anchor.set_index("code")["close"].astype(float).to_dict()

    raw, basic, limit = fetch_day_panels(pro, date)
    raw = raw.copy()
    raw["code"] = raw["ts_code"].map(from_ts_code)
    raw = raw.dropna(subset=["code"])

    turn_map = {}
    if not basic.empty and "turnover_rate" in basic.columns:
        basic = basic.copy()
        basic["code"] = basic["ts_code"].map(from_ts_code)
        turn_map = (
            basic.dropna(subset=["code"])
            .set_index("code")["turnover_rate"]
            .astype(float)
            .to_dict()
        )

    limit_map = {}
    if not limit.empty:
        limit = limit.copy()
        limit["code"] = limit["ts_code"].map(from_ts_code)
        # 列名兼容
        up_col = "up_limit" if "up_limit" in limit.columns else None
        down_col = "down_limit" if "down_limit" in limit.columns else None
        if up_col and down_col:
            for _, r in limit.dropna(subset=["code"]).iterrows():
                limit_map[str(r["code"])] = (float(r[up_col]), float(r[down_col]))

    # ST：用 stock_basic 名称（缓存一次）
    st_codes = set()
    try:
        sb = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")
        time.sleep(0.35)
        if sb is not None and not sb.empty:
            for _, r in sb.iterrows():
                name = str(r.get("name", ""))
                if "ST" in name.upper():
                    code = from_ts_code(r["ts_code"])
                    if code:
                        st_codes.add(code)
    except Exception as exc:
        print(f"stock_basic 拉 ST 失败，跳过 ST 标记: {exc}")

    daily_rows = []
    ud_rows = []
    miss = 0
    for _, r in raw.iterrows():
        code = str(r["code"])
        if code not in hfq_map:
            continue
        try:
            pre = float(r["pre_close"])
            open_ = float(r["open"])
            high = float(r["high"])
            low = float(r["low"])
            close = float(r["close"])
            amount = float(r.get("amount", 0.0) or 0.0) * 1000.0
            # Tushare daily.amount 单位是千元 → 乘 1000 成元，与本地 money 口径一致
            turn = float(turn_map.get(code, 0.0) or 0.0)
            is_st = code in st_codes
            drow, urow = build_rows_from_raw_day(
                code=code,
                date=date,
                open_=open_,
                high=high,
                low=low,
                close=close,
                pre_close=pre,
                amount=amount,
                turn=turn,
                is_st=is_st,
                anchor_hfq_close=float(hfq_map[code]),
                anchor_raw_close=pre,
            )
            if code in limit_map:
                urow["high_limit"] = round2(limit_map[code][0])
                urow["low_limit"] = round2(limit_map[code][1])
                close_r = round2(close)
                paused = float(urow["paused"])
                urow["zt"] = int((paused == 0) and (close_r >= urow["high_limit"]))
                urow["dt"] = int((paused == 0) and (close_r <= urow["low_limit"]))
            daily_rows.append(drow)
            ud_rows.append(urow)
        except Exception:
            miss += 1
            continue

    print(
        f"Tushare {date}: 对齐 {len(daily_rows)} 只，跳过 {miss}，"
        f"换手>0 {sum(1 for x in daily_rows if x['turnover_ratio'] > 0)}"
    )
    return pd.DataFrame(daily_rows, columns=DAILY_COLS), pd.DataFrame(ud_rows, columns=UD_COLS)


def parse_args():
    p = argparse.ArgumentParser(description="Tushare 增量补行情（baostock 备源）")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--start", default="", help="起始补日")
    p.add_argument("--end", default="", help="结束补日，默认今天")
    p.add_argument("--token", default="", help="也可设环境变量 TUSHARE_TOKEN")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = resolve_token(args.token)
    import tushare as ts

    pro = ts.pro_api(token)
    data_dir = Path(args.data_dir)
    daily_dir = data_dir / "data_daily"
    ud_dir = data_dir / "data_ud_new"
    today = datetime.now().strftime("%Y-%m-%d")
    last = last_csv_date(daily_dir)
    if not last:
        raise RuntimeError("本地无 data_daily，无法锚定复权")

    end = args.end or today
    dates = need_dates(data_dir, args.start, end, overwrite=args.overwrite)
    if not dates and (args.start or args.end):
        # 显式指定且 overwrite：强制该区间
        lo = args.start or last
        hi = end
        cal = pd.bdate_range(lo, hi).strftime("%Y-%m-%d").tolist()
        dates = [d for d in cal if (args.overwrite or not (daily_dir / f"{d}.csv").exists())]
        if args.overwrite and args.start and args.end and args.start == args.end:
            dates = [args.start]
    if not dates:
        print(f"没有需要补的交易日（本地最后一日 {last}）")
        return 0

    print(f"Tushare 待补 {dates[0]} ~ {dates[-1]}，共 {len(dates)} 天；锚定用各日前一本地日")
    for date in dates:
        anchors = [
            d
            for d in sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))
            if d < date
        ]
        if not anchors:
            print(f"跳过 {date}：无锚定日")
            continue
        anchor = anchors[-1]
        print(f"补 {date}，锚定 {anchor}")
        daily, ud = fill_one_day_tushare(pro, data_dir, date, anchor)
        if daily.empty:
            print(f"{date} 无数据")
            continue
        write_by_date(daily, daily_dir, skip_existing=False)
        write_by_date(ud, ud_dir, skip_existing=False)
        update_date_pkl(data_dir, [date])

    update_forward_returns(data_dir, skip_existing=not args.overwrite)
    print(f"日线最后一日: {last_csv_date(daily_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
