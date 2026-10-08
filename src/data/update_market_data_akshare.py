"""
用 akshare / 腾讯等增量补日线 / 交易状态，作 baostock 不可用时的备源。

口径对齐本仓库：
- data_daily：后复权价位按「本地末日收盘 × (当日不复权价 / 昨收)」链式接上，避免混用两套复权基数
- data_ud_new：不复权开盘/昨收 + 与 baostock 相同的涨跌停算法
- volume = 成交额 / 收盘价（与 update_market_data 一致）

模式：
- spot：新浪快照（无换手，抄锚定日）
- hist / em：东财（含换手；em 直连 push2his）
- tx：腾讯历史日K（价量稳；换手优先用本地 hfq cache，否则抄锚定日）
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import akshare as ak
import numpy as np
import pandas as pd
import requests

from src.data.update_market_data import (
    DAILY_COLS,
    UD_COLS,
    last_csv_date,
    limit_ratio,
    round2,
    update_date_pkl,
    update_forward_returns,
    write_by_date,
)


def disable_env_proxy() -> None:
    """避免 macOS/系统代理让 eastmoney 请求走坏掉的 proxy。"""
    import os

    for key in list(os.environ):
        if "proxy" in key.lower():
            os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    _orig_init = requests.Session.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.trust_env = False

    requests.Session.__init__ = _init  # type: ignore[method-assign]


def to_symbol(project_code: str) -> str:
    return str(project_code).split(".")[0].zfill(6)


def to_project_code(symbol: str) -> Optional[str]:
    num = str(symbol).zfill(6)
    if not num.isdigit():
        return None
    if num.startswith(("5", "9")):
        return None
    if num.startswith("6"):
        return f"{num}.XSHG"
    return f"{num}.XSHE"


def load_universe_codes(data_dir: Path, anchor_date: str) -> List[str]:
    daily = pd.read_csv(data_dir / "data_daily" / f"{anchor_date}.csv")
    return sorted(daily["code"].astype(str).unique().tolist())


def need_dates(
    data_dir: Path,
    start: str,
    end: str,
    overwrite: bool = False,
) -> List[str]:
    """待补（或 overwrite 时待重写）的交易日。"""
    pkl = data_dir / "date.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            calendar = [str(d) for d in pickle.load(f)]
    else:
        calendar = []
    last = last_csv_date(data_dir / "data_daily")
    today = datetime.now().strftime("%Y-%m-%d")
    hi = end or today
    if start:
        lo = start
    elif last:
        # 默认从本地最后一日的下一日历候选开始
        later = [d for d in calendar if d > last]
        lo = later[0] if later else hi
    else:
        lo = hi
    if not calendar:
        calendar = pd.bdate_range(lo, hi).strftime("%Y-%m-%d").tolist()
    out = []
    for d in calendar:
        if d < lo or d > hi:
            continue
        exists = (data_dir / "data_daily" / f"{d}.csv").exists()
        if exists and not overwrite:
            continue
        if last and d <= last and not overwrite:
            continue
        out.append(d)
    return out


def to_sina_symbol(project_code: str) -> str:
    num = to_symbol(project_code)
    if project_code.endswith(".XSHG"):
        return f"sh{num}"
    return f"sz{num}"


def fetch_spot_sina(codes: Sequence[str], batch_size: int = 400) -> pd.DataFrame:
    """
    新浪实时行情（不复权），按股票池批量拉取。
    字段：name,open,pre_close,close,high,low,...,amount,...,date
    """
    session = requests.Session()
    session.trust_env = False
    headers = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
    rows = []
    symbols = [to_sina_symbol(c) for c in codes]
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        url = "https://hq.sinajs.cn/list=" + ",".join(chunk)
        last_err = None
        text = ""
        for attempt in range(3):
            try:
                resp = session.get(url, headers=headers, timeout=30)
                resp.encoding = "gbk"
                text = resp.text
                break
            except Exception as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        if not text:
            print(f"新浪批次失败 @{i}: {last_err}")
            continue
        for line in text.strip().splitlines():
            # var hq_str_sz000001="...";
            if '="' not in line:
                continue
            left, right = line.split('="', 1)
            sym = left.split("_")[-1]
            payload = right.rstrip('";')
            if not payload:
                continue
            parts = payload.split(",")
            if len(parts) < 10:
                continue
            num = sym[2:]
            code = to_project_code(num)
            if not code:
                continue
            try:
                open_ = float(parts[1])
                pre = float(parts[2])
                close = float(parts[3])
                high = float(parts[4])
                low = float(parts[5])
                amount = float(parts[9])
            except ValueError:
                continue
            if pre <= 0 or close <= 0:
                continue
            name = parts[0]
            quote_date = parts[30] if len(parts) > 30 else ""
            rows.append(
                {
                    "symbol": num,
                    "name": name,
                    "open": open_,
                    "pre_close": pre,
                    "close": close,
                    "high": high,
                    "low": low,
                    "amount": amount,
                    "turn": 0.0,  # 新浪基础行情无换手，记 0
                    "code": code,
                    "quote_date": quote_date,
                    "is_st": ("ST" in name.upper()),
                }
            )
        time.sleep(0.05)
        if (i // batch_size + 1) % 5 == 0:
            print(f"新浪进度 {min(i + batch_size, len(symbols))}/{len(symbols)}")
    return pd.DataFrame(rows)


def fetch_spot(codes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """优先新浪批量快照；失败再试 akshare 东财 spot。"""
    if codes:
        spot = fetch_spot_sina(codes)
        if not spot.empty:
            return spot
        print("新浪快照为空，尝试 akshare eastmoney spot")
    spot = ak.stock_zh_a_spot_em()
    rename = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "昨收": "pre_close",
        "成交量": "volume_lot",
        "成交额": "amount",
        "换手率": "turn",
    }
    spot = spot.rename(columns=rename)
    keep = [c for c in rename.values() if c in spot.columns]
    spot = spot[keep].copy()
    spot["code"] = spot["symbol"].map(to_project_code)
    spot = spot.dropna(subset=["code", "close", "pre_close"])
    spot = spot[spot["pre_close"].astype(float) > 0]
    for col in ["open", "high", "low", "close", "pre_close", "amount", "turn"]:
        if col in spot.columns:
            spot[col] = pd.to_numeric(spot[col], errors="coerce")
    spot["is_st"] = spot["name"].astype(str).str.contains("ST", case=False, na=False)
    return spot


def _em_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/",
        }
    )
    return session


def fetch_em_kline_raw(
    symbol: str,
    start: str,
    end: str,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    东财 push2his 不复权日线（需 Referer）。
    kline: date,open,close,high,low,volume,amount,amp,pct,chg,turn
    """
    session = session or _em_session()
    num = str(symbol).zfill(6)
    market = 1 if num.startswith("6") else 0
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "0",
        "secid": f"{market}.{num}",
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    last_err = None
    for attempt in range(4):
        try:
            resp = session.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params=params,
                timeout=20,
            )
            if resp.status_code != 200 or not resp.content:
                raise RuntimeError(f"HTTP {resp.status_code} empty={not resp.content}")
            payload = resp.json()
            data = payload.get("data") or {}
            klines = data.get("klines") or []
            rows = []
            for line in klines:
                parts = str(line).split(",")
                if len(parts) < 11:
                    continue
                rows.append(
                    {
                        "date": parts[0],
                        "open": float(parts[1]),
                        "close": float(parts[2]),
                        "high": float(parts[3]),
                        "low": float(parts[4]),
                        "amount": float(parts[6]),
                        "turn": float(parts[10]),
                    }
                )
            return pd.DataFrame(rows)
        except Exception as exc:
            last_err = exc
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"{symbol} 东财拉取失败: {last_err}")


def fill_dates_em(
    data_dir: Path,
    dates: Sequence[str],
    anchor_date: str,
    workers: int = 16,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """用东财 kline（含换手）补日；按锚定日链式接后复权。"""
    codes = load_universe_codes(data_dir, anchor_date)
    hfq_map = (
        pd.read_csv(data_dir / "data_daily" / f"{anchor_date}.csv")
        .set_index("code")["close"]
        .astype(float)
        .to_dict()
    )
    end = max(dates)
    date_set = set(dates)
    daily_rows: List[dict] = []
    ud_rows: List[dict] = []
    errors = 0

    def _one(code: str) -> Tuple[List[dict], List[dict]]:
        session = _em_session()
        time.sleep(0.05)  # 限流：每票略停
        full = fetch_em_kline_raw(to_symbol(code), anchor_date, end, session=session)
        if full.empty or code not in hfq_map:
            return [], []
        full = full.sort_values("date").reset_index(drop=True)
        full["pre_close"] = full["close"].shift(1)
        a = full[full["date"] == anchor_date]
        if a.empty:
            return [], []
        anchor_raw = float(a.iloc[0]["close"])
        if anchor_raw <= 0:
            return [], []
        out_d, out_u = [], []
        for _, r in full.iterrows():
            if r["date"] not in date_set:
                continue
            pre = float(r["pre_close"]) if pd.notna(r["pre_close"]) else float("nan")
            if not (pre > 0):
                continue
            drow, urow = build_rows_from_raw_day(
                code=code,
                date=str(r["date"]),
                open_=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                pre_close=pre,
                amount=float(r["amount"]) if pd.notna(r["amount"]) else 0.0,
                turn=float(r["turn"]) if pd.notna(r["turn"]) else 0.0,
                is_st=False,
                anchor_hfq_close=float(hfq_map[code]),
                anchor_raw_close=anchor_raw,
            )
            out_d.append(drow)
            out_u.append(urow)
        return out_d, out_u

    print(f"东财 kline 并行拉取 {len(codes)} 只，workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                dpart, upart = fut.result()
                daily_rows.extend(dpart)
                ud_rows.extend(upart)
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"跳过 {futs[fut]}: {exc}")
            if done % 200 == 0 or done == len(codes):
                print(
                    f"东财进度 {done}/{len(codes)}，成功行 {len(daily_rows)}，失败 {errors}",
                    flush=True,
                )
            time.sleep(0.02)
    return pd.DataFrame(daily_rows, columns=DAILY_COLS), pd.DataFrame(ud_rows, columns=UD_COLS)


def to_tx_symbol(project_code: str) -> str:
    """本仓库 code → 腾讯 sh/sz 代码。"""
    return to_sina_symbol(project_code)


def bs_cache_to_project_code(bs_code: str) -> Optional[str]:
    """baostock cache 代码 sz.000001 → 000001.XSHE。"""
    raw = str(bs_code)
    if raw.startswith("sz."):
        return f"{raw[3:].zfill(6)}.XSHE"
    if raw.startswith("sh."):
        return f"{raw[3:].zfill(6)}.XSHG"
    return None


def load_hfq_turn_map(data_dir: Path, dates: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """
    从本地 baostock hfq cache 读取指定日换手。
    腾讯历史 K 线无换手字段时，用已拉到的官方 turn 补上。
    """
    date_set = set(str(d) for d in dates)
    cache_dir = data_dir / "_cache" / "hfq"
    out: Dict[Tuple[str, str], float] = {}
    if not cache_dir.exists():
        return out
    files = list(cache_dir.glob("*.parquet"))
    for i, path in enumerate(files, 1):
        try:
            frame = pd.read_parquet(path, columns=["date", "code", "turn"])
            frame = frame[frame["date"].astype(str).isin(date_set)]
            if frame.empty:
                continue
            for _, row in frame.iterrows():
                code = bs_cache_to_project_code(row["code"])
                if not code:
                    continue
                turn = float(row["turn"]) if pd.notna(row["turn"]) else 0.0
                if turn > 0:
                    out[(code, str(row["date"]))] = turn
        except Exception:
            continue
        if i % 1000 == 0:
            print(f"hfq 换手扫描 {i}/{len(files)}，命中 {len(out)}", flush=True)
    print(f"hfq 换手可用 {len(out)} 条（覆盖待补日）", flush=True)
    return out


def fetch_tx_spot_turns(
    codes: Sequence[str],
    batch_size: int = 300,
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    腾讯实时行情换手（qt.gtimg.cn）。
    返回 (报价日 YYYY-MM-DD 或 None, {project_code: turn})。
    """
    session = requests.Session()
    session.trust_env = False
    turns: Dict[str, float] = {}
    date_votes: Dict[str, int] = {}
    symbols = [to_tx_symbol(c) for c in codes]
    code_by_sym = {to_tx_symbol(c): c for c in codes}
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        url = "http://qt.gtimg.cn/q=" + ",".join(chunk)
        try:
            resp = session.get(url, timeout=20)
            resp.encoding = "gbk"
        except Exception as exc:
            print(f"腾讯实时批次失败 @{i}: {exc}")
            continue
        for part in resp.text.strip().split(";"):
            part = part.strip()
            if '="' not in part:
                continue
            body = part.split('="', 1)[1].rstrip('";')
            fields = body.split("~")
            if len(fields) < 39:
                continue
            num = str(fields[2]).zfill(6)
            # 从请求符号反查市场
            # fields 不够区分市场时，用 num 规则
            code = code_by_sym.get(f"sh{num}") or code_by_sym.get(f"sz{num}")
            if not code:
                continue
            try:
                turn = float(fields[38])
            except (TypeError, ValueError):
                continue
            if turn > 0:
                turns[code] = turn
            ts = str(fields[30]) if len(fields) > 30 else ""
            if len(ts) >= 8 and ts[:8].isdigit():
                d = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                date_votes[d] = date_votes.get(d, 0) + 1
        time.sleep(0.05)
    quote_date = None
    if date_votes:
        quote_date = max(date_votes.items(), key=lambda x: x[1])[0]
    print(f"腾讯实时换手 {len(turns)} 只，主日期 {quote_date}", flush=True)
    return quote_date, turns


def fetch_tx_kline_raw(
    project_code: str,
    start: str,
    end: str,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    腾讯 ifzq 不复权日K。
    返回 date/open/close/high/low/amount；成交额≈手数*100*收盘（与官方 amount 接近）。
    不含换手。
    """
    session = session or requests.Session()
    session.trust_env = False
    symbol = to_tx_symbol(project_code)
    # count 取区间跨度上界，接口单次最多约 640
    params = {
        "_var": "kline_day",
        "param": f"{symbol},day,{start},{end},640,",
    }
    # 主站易 501，优先代理域名
    urls = (
        "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get",
        "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get",
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
    )
    last_err = None
    for attempt in range(4):
        for url in urls:
            try:
                resp = session.get(url, params=params, timeout=20)
                text = resp.text
                if resp.status_code != 200 or not text or text.lstrip().startswith("<!"):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                if "=" in text:
                    text = text.split("=", 1)[1]
                payload = json.loads(text)
                node = (payload.get("data") or {}).get(symbol) or {}
                rows_raw = node.get("day") or node.get("qfqday") or node.get("hfqday") or []
                rows = []
                for item in rows_raw:
                    if not item or len(item) < 6:
                        continue
                    date = str(item[0])
                    if date < start or date > end:
                        continue
                    open_ = float(item[1])
                    close = float(item[2])
                    high = float(item[3])
                    low = float(item[4])
                    vol_lot = float(item[5])
                    # 腾讯日K无成交额列时，用 手*100*收盘 近似
                    if len(item) >= 9 and item[8] not in (None, ""):
                        try:
                            amount = float(item[8]) * 10000.0
                        except (TypeError, ValueError):
                            amount = vol_lot * 100.0 * close
                    else:
                        amount = vol_lot * 100.0 * close
                    rows.append(
                        {
                            "date": date,
                            "open": open_,
                            "close": close,
                            "high": high,
                            "low": low,
                            "amount": amount,
                            "turn": 0.0,
                        }
                    )
                return pd.DataFrame(rows)
            except Exception as exc:
                last_err = exc
                continue
        time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"{project_code} 腾讯拉取失败: {last_err}")


def fill_dates_tx(
    data_dir: Path,
    dates: Sequence[str],
    anchor_date: str,
    workers: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """腾讯历史日K补数；换手优先 hfq cache，否则锚定日。"""
    codes = load_universe_codes(data_dir, anchor_date)
    daily_anchor = pd.read_csv(data_dir / "data_daily" / f"{anchor_date}.csv")
    hfq_map = daily_anchor.set_index("code")["close"].astype(float).to_dict()
    turn_anchor = (
        daily_anchor.set_index("code")["turnover_ratio"].astype(float).to_dict()
    )
    turn_hfq = load_hfq_turn_map(data_dir, dates)
    spot_date, turn_spot = fetch_tx_spot_turns(codes)
    end = max(dates)
    date_set = set(dates)
    daily_rows: List[dict] = []
    ud_rows: List[dict] = []
    errors = 0
    turn_from_cache = 0
    turn_from_spot = 0
    turn_from_anchor = 0

    def _one(code: str) -> Tuple[List[dict], List[dict], int, int, int]:
        session = requests.Session()
        session.trust_env = False
        time.sleep(0.03)
        full = fetch_tx_kline_raw(code, anchor_date, end, session=session)
        if full.empty or code not in hfq_map:
            return [], [], 0, 0, 0
        full = full.sort_values("date").reset_index(drop=True)
        full["pre_close"] = full["close"].shift(1)
        a = full[full["date"] == anchor_date]
        if a.empty:
            return [], [], 0, 0, 0
        anchor_raw = float(a.iloc[0]["close"])
        if anchor_raw <= 0:
            return [], [], 0, 0, 0
        out_d, out_u = [], []
        n_cache = n_spot = n_anchor = 0
        for _, r in full.iterrows():
            if r["date"] not in date_set:
                continue
            pre = float(r["pre_close"]) if pd.notna(r["pre_close"]) else float("nan")
            if not (pre > 0):
                continue
            dstr = str(r["date"])
            turn = float(turn_hfq.get((code, dstr), 0.0) or 0.0)
            src = "cache"
            if turn <= 0 and spot_date == dstr:
                turn = float(turn_spot.get(code, 0.0) or 0.0)
                src = "spot"
            if turn <= 0:
                turn = float(turn_anchor.get(code, 0.0) or 0.0)
                src = "anchor"
            if src == "cache":
                n_cache += 1
            elif src == "spot":
                n_spot += 1
            else:
                n_anchor += 1
            drow, urow = build_rows_from_raw_day(
                code=code,
                date=dstr,
                open_=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                pre_close=pre,
                amount=float(r["amount"]) if pd.notna(r["amount"]) else 0.0,
                turn=turn,
                is_st=False,
                anchor_hfq_close=float(hfq_map[code]),
                anchor_raw_close=anchor_raw,
            )
            out_d.append(drow)
            out_u.append(urow)
        return out_d, out_u, n_cache, n_spot, n_anchor

    print(f"腾讯日K 并行拉取 {len(codes)} 只，workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                dpart, upart, nc, ns, na = fut.result()
                daily_rows.extend(dpart)
                ud_rows.extend(upart)
                turn_from_cache += nc
                turn_from_spot += ns
                turn_from_anchor += na
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"跳过 {futs[fut]}: {exc}")
            if done % 200 == 0 or done == len(codes):
                print(
                    f"腾讯进度 {done}/{len(codes)}，成功行 {len(daily_rows)}，"
                    f"换手cache={turn_from_cache} spot={turn_from_spot} "
                    f"anchor={turn_from_anchor}，失败 {errors}",
                    flush=True,
                )
            time.sleep(0.01)
    return pd.DataFrame(daily_rows, columns=DAILY_COLS), pd.DataFrame(ud_rows, columns=UD_COLS)


def fetch_hist_raw(symbol: str, start: str, end: str) -> pd.DataFrame:
    """单票不复权日线。"""
    start_s = start.replace("-", "")
    end_s = end.replace("-", "")
    last_err = None
    for attempt in range(4):
        try:
            frame = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_s,
                end_date=end_s,
                adjust="",
            )
            if frame is None or frame.empty:
                return pd.DataFrame()
            frame = frame.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交额": "amount",
                    "换手率": "turn",
                }
            )
            frame["date"] = frame["date"].astype(str)
            for col in ["open", "high", "low", "close", "amount", "turn"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            return frame
        except Exception as exc:
            last_err = exc
            time.sleep(0.4 * (attempt + 1))
    print(f"跳过 {symbol}: {last_err}")
    return pd.DataFrame()


def build_rows_from_raw_day(
    code: str,
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    pre_close: float,
    amount: float,
    turn: float,
    is_st: bool,
    anchor_hfq_close: float,
    anchor_raw_close: float,
) -> Tuple[dict, dict]:
    """
    用不复权当日价 + 锚定日本地后复权收盘，生成一行 daily / ud。
    链式：hfq = anchor_hfq * (raw / anchor_raw)
    """
    if pre_close <= 0 or anchor_raw_close <= 0 or anchor_hfq_close <= 0:
        raise ValueError("锚定或昨收无效")
    # 用锚定 raw 与当日 raw 的比值接到本地 hfq；若有昨收则优先用昨收对齐更稳
    scale = anchor_hfq_close / anchor_raw_close
    h_open = round2(open_ * scale)
    h_high = round2(high * scale)
    h_low = round2(low * scale)
    h_close = round2(close * scale)
    money = float(amount) if pd.notna(amount) else 0.0
    vol = round(money / h_close, 1) if h_close else 0.0
    daily = {
        "date": date,
        "code": code,
        "open": h_open,
        "close": h_close,
        "low": h_low,
        "high": h_high,
        "volume": vol,
        "money": round(money, 1),
        "turnover_ratio": round(float(turn or 0.0), 4),
    }
    ratio = limit_ratio(code, is_st)
    high_limit = round2(pre_close * (1.0 + ratio))
    low_limit = round2(pre_close * (1.0 - ratio))
    close_r = round2(close)
    paused = 1.0 if money <= 0 else 0.0
    ud = {
        "date": date,
        "code": code,
        "open": round2(open_),
        "pre_close": round2(pre_close),
        "high_limit": high_limit,
        "low_limit": low_limit,
        "paused": paused,
        "zt": int((paused == 0) and (close_r >= high_limit)),
        "dt": int((paused == 0) and (close_r <= low_limit)),
        "st": int(is_st),
    }
    return daily, ud


def fill_one_day_spot(
    data_dir: Path,
    date: str,
    anchor_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """用全市场快照填单日（适合补最近一个交易日）。"""
    daily_anchor = pd.read_csv(data_dir / "data_daily" / f"{anchor_date}.csv")
    hfq_map = daily_anchor.set_index("code")["close"].astype(float).to_dict()
    # 新浪无换手时，沿用锚定日换手，避免可交易过滤（MIN_TURNOVER）全灭
    turn_map = daily_anchor.set_index("code")["turnover_ratio"].astype(float).to_dict()
    universe = list(hfq_map.keys())
    spot = fetch_spot(universe)
    print(f"spot 拉取 {len(spot)} 只")
    if "quote_date" in spot.columns and not spot.empty:
        qd = spot["quote_date"].mode()
        if len(qd) and str(qd.iloc[0]) and str(qd.iloc[0]) != date:
            print(f"警告: 快照主日期为 {qd.iloc[0]}，目标补日 {date}")
    daily_rows = []
    ud_rows = []
    miss = 0
    for _, row in spot.iterrows():
        code = row["code"]
        if code not in hfq_map:
            continue
        try:
            pre = float(row["pre_close"])
            turn = float(row["turn"]) if pd.notna(row.get("turn")) else 0.0
            if turn <= 0:
                turn = float(turn_map.get(code, 0.0) or 0.0)
            drow, urow = build_rows_from_raw_day(
                code=code,
                date=date,
                open_=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                pre_close=pre,
                amount=float(row["amount"]) if pd.notna(row["amount"]) else 0.0,
                turn=turn,
                is_st=bool(row["is_st"]),
                anchor_hfq_close=float(hfq_map[code]),
                anchor_raw_close=pre,
            )
            daily_rows.append(drow)
            ud_rows.append(urow)
        except Exception:
            miss += 1
            continue
    print(f"spot 对齐写入 {len(daily_rows)} 只，跳过 {miss}")
    return pd.DataFrame(daily_rows, columns=DAILY_COLS), pd.DataFrame(ud_rows, columns=UD_COLS)


def fill_dates_hist(
    data_dir: Path,
    dates: Sequence[str],
    anchor_date: str,
    sleep_s: float = 0.08,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """逐票 hist 补多日；慢，仅作 spot 不可用或跨多日时使用。"""
    codes = load_universe_codes(data_dir, anchor_date)
    hfq_map = (
        pd.read_csv(data_dir / "data_daily" / f"{anchor_date}.csv")
        .set_index("code")["close"]
        .astype(float)
        .to_dict()
    )
    end = max(dates)
    fetch_start = anchor_date
    daily_rows: List[dict] = []
    ud_rows: List[dict] = []
    date_set = set(dates)
    for i, code in enumerate(codes, start=1):
        if code not in hfq_map:
            continue
        full = fetch_hist_raw(to_symbol(code), fetch_start, end)
        time.sleep(sleep_s)
        if full.empty:
            continue
        full = full.sort_values("date").reset_index(drop=True)
        full["pre_close"] = full["close"].shift(1)
        a = full[full["date"] == anchor_date]
        if a.empty:
            continue
        anchor_raw = float(a.iloc[0]["close"])
        if anchor_raw <= 0:
            continue
        for _, r in full.iterrows():
            if r["date"] not in date_set:
                continue
            pre = float(r["pre_close"]) if pd.notna(r["pre_close"]) else np.nan
            if not (pre > 0):
                continue
            try:
                drow, urow = build_rows_from_raw_day(
                    code=code,
                    date=str(r["date"]),
                    open_=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    pre_close=pre,
                    amount=float(r["amount"]) if pd.notna(r["amount"]) else 0.0,
                    turn=float(r["turn"]) if pd.notna(r["turn"]) else 0.0,
                    is_st=False,
                    anchor_hfq_close=float(hfq_map[code]),
                    anchor_raw_close=anchor_raw,
                )
                daily_rows.append(drow)
                ud_rows.append(urow)
            except Exception:
                continue
        if i % 100 == 0:
            print(f"hist 进度 {i}/{len(codes)}")
    return pd.DataFrame(daily_rows, columns=DAILY_COLS), pd.DataFrame(ud_rows, columns=UD_COLS)


def parse_args():
    p = argparse.ArgumentParser(description="akshare 增量补行情（baostock 备源）")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--start", default="", help="起始补日，默认本地日线最后一日的下一交易日")
    p.add_argument("--end", default="", help="结束补日，默认今天")
    p.add_argument(
        "--mode",
        choices=["auto", "spot", "hist", "em", "tx"],
        default="auto",
        help="auto/spot=新浪快照；hist/em=东财；tx=腾讯日K(换手优先 hfq cache)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sleep", type=float, default=0.08, help="hist 模式每票间隔秒")
    p.add_argument("--workers", type=int, default=16, help="em 模式并行线程数")
    return p.parse_args()


def main() -> int:
    disable_env_proxy()
    args = parse_args()
    data_dir = Path(args.data_dir)
    daily_dir = data_dir / "data_daily"
    ud_dir = data_dir / "data_ud_new"
    today = datetime.now().strftime("%Y-%m-%d")
    last = last_csv_date(daily_dir)
    if not last:
        raise RuntimeError("本地无 data_daily，无法锚定复权")
    end = args.end or today
    start = args.start
    dates = need_dates(data_dir, start, end, overwrite=args.overwrite)
    if not dates:
        # 若 pkl 无新日但用户指定了区间，仍按工作日缺文件补
        if args.start or args.end:
            rng = pd.bdate_range(args.start or last, end).strftime("%Y-%m-%d").tolist()
            dates = [
                d
                for d in rng
                if d > last and not (daily_dir / f"{d}.csv").exists() and d <= end
            ]
        if not dates:
            print(f"没有需要补的交易日（本地最后一日 {last}）")
            return 0

    print(f"本地锚定日 {last}，待补 {dates[0]} ~ {dates[-1]}，共 {len(dates)} 天")
    mode = args.mode
    if mode == "auto":
        mode = "spot" if len(dates) == 1 else "hist"

    if mode == "em":
        # 锚定日取待补首日前一个本地日线日
        first = dates[0]
        daily_dates = sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))
        anchors = [d for d in daily_dates if d < first]
        if not anchors:
            raise RuntimeError("无法找到东财补数锚定日")
        anchor = anchors[-1]
        print(f"东财模式锚定日 {anchor}（覆盖待补日）")
        daily, ud = fill_dates_em(
            data_dir, dates, anchor, workers=max(1, args.workers)
        )
        if daily.empty:
            print("东财结果为空")
            return 1
        n1 = write_by_date(daily, daily_dir, skip_existing=False)
        n2 = write_by_date(ud, ud_dir, skip_existing=False)
        update_date_pkl(data_dir, dates)
        print(f"东财写入日线 {n1}、状态 {n2}；换手>0 {(daily['turnover_ratio']>0).sum()}/{len(daily)}")
        update_forward_returns(data_dir, skip_existing=not args.overwrite)
        print(f"日线最后一日: {last_csv_date(daily_dir)}")
        return 0

    if mode == "tx":
        first = dates[0]
        daily_dates = sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))
        anchors = [d for d in daily_dates if d < first]
        if not anchors:
            raise RuntimeError("无法找到腾讯补数锚定日")
        anchor = anchors[-1]
        print(f"腾讯模式锚定日 {anchor}（覆盖待补日）")
        daily, ud = fill_dates_tx(
            data_dir, dates, anchor, workers=max(1, args.workers)
        )
        if daily.empty:
            print("腾讯结果为空")
            return 1
        n1 = write_by_date(daily, daily_dir, skip_existing=False)
        n2 = write_by_date(ud, ud_dir, skip_existing=False)
        update_date_pkl(data_dir, dates)
        print(
            f"腾讯写入日线 {n1}、状态 {n2}；"
            f"换手>0 {(daily['turnover_ratio']>0).sum()}/{len(daily)}"
        )
        # 覆盖单日时不要全量重算历史收益；只补缺文件
        update_forward_returns(data_dir, skip_existing=True)
        print(f"日线最后一日: {last_csv_date(daily_dir)}")
        return 0

    if mode == "spot":
        if len(dates) != 1:
            print("spot 模式仅支持单日，改用 hist")
            mode = "hist"
        else:
            # spot 锚定：待补日前一本地日
            anchors = [d for d in sorted(p.stem for p in daily_dir.glob("????-??-??.csv")) if d < dates[0]]
            anchor = anchors[-1] if anchors else last
            daily, ud = fill_one_day_spot(data_dir, dates[0], anchor)
            if daily.empty:
                print("spot 结果为空，回退 hist")
                mode = "hist"
            else:
                n1 = write_by_date(daily, daily_dir, skip_existing=not args.overwrite)
                n2 = write_by_date(ud, ud_dir, skip_existing=not args.overwrite)
                update_date_pkl(data_dir, dates)
                print(f"spot 写入日线 {n1}、状态 {n2}")
                update_forward_returns(data_dir, skip_existing=not args.overwrite)
                print(f"日线最后一日: {last_csv_date(daily_dir)}")
                return 0

    # hist 锚定同理
    anchors = [d for d in sorted(p.stem for p in daily_dir.glob("????-??-??.csv")) if d < dates[0]]
    hist_anchor = anchors[-1] if anchors else last
    daily, ud = fill_dates_hist(data_dir, dates, hist_anchor, sleep_s=args.sleep)
    n1 = write_by_date(daily, daily_dir, skip_existing=not args.overwrite)
    n2 = write_by_date(ud, ud_dir, skip_existing=not args.overwrite)
    update_date_pkl(data_dir, dates)
    print(f"hist 写入日线 {n1}、状态 {n2}")
    update_forward_returns(data_dir, skip_existing=not args.overwrite)
    print(f"日线最后一日: {last_csv_date(daily_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
