"""
补全申万一级行业分类到 data/data_industry。

公开源拿不到逐日历史调入调出，这里用最新申万一级成分铺到缺失交易日。
课上已有文件（到 2022-01-21）默认不覆盖。
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional

import akshare as ak
import pandas as pd


def migrate_industry_dir(data_dir: Path) -> Path:
    """若 industry 仍是课上软链，则改成本地目录并硬链旧文件。"""
    industry_dir = data_dir / "data_industry"
    if industry_dir.is_symlink():
        src = industry_dir.resolve()
        print(f"迁移行业目录: {src} -> {industry_dir}")
        industry_dir.unlink()
        industry_dir.mkdir()
        for src_file in sorted(src.glob("*.csv")):
            os.link(src_file, industry_dir / src_file.name)
    elif not industry_dir.exists():
        industry_dir.mkdir(parents=True)
    return industry_dir


def to_project_code(number: str) -> Optional[str]:
    """6 位代码转项目内 code。"""
    num = str(number).zfill(6)
    if not num.isdigit():
        return None
    if num.startswith(("5", "9")):
        return None
    if num.startswith("6"):
        return f"{num}.XSHG"
    return f"{num}.XSHE"


def fetch_sw_l1_map() -> pd.DataFrame:
    """拉取最新申万一级成分，返回 code,industry。"""
    info = ak.sw_index_first_info()
    rows = []
    for _, row in info.iterrows():
        industry = str(row["行业名称"])
        symbol = str(row["行业代码"]).replace(".SI", "")
        cons = ak.index_component_sw(symbol=symbol)
        if cons.empty:
            print(f"跳过空成分: {industry}")
            continue
        code_col = "证券代码" if "证券代码" in cons.columns else cons.columns[1]
        for raw in cons[code_col].tolist():
            code = to_project_code(raw)
            if code:
                rows.append({"code": code, "industry": industry})
        print(f"已拉取 {industry}: {len(cons)} 只")
    if not rows:
        raise RuntimeError("未能获取任何申万一级成分")
    out = pd.DataFrame(rows).drop_duplicates(subset=["code"], keep="last")
    print(f"行业映射合计 {len(out)} 只，行业数 {out['industry'].nunique()}")
    return out.sort_values("code")


def load_need_dates(data_dir: Path, start: str, end: str) -> List[str]:
    """取有日线、缺行业文件的交易日。"""
    daily_dir = data_dir / "data_daily"
    industry_dir = data_dir / "data_industry"
    pkl = data_dir / "date.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            dates = pickle.load(f)
    else:
        dates = sorted(p.stem for p in daily_dir.glob("????-??-??.csv"))
    need = []
    for date in dates:
        if start and date < start:
            continue
        if end and date > end:
            continue
        if not (daily_dir / f"{date}.csv").exists():
            continue
        if (industry_dir / f"{date}.csv").exists():
            continue
        need.append(date)
    return need


def write_industry_files(
    industry_dir: Path,
    mapping: pd.DataFrame,
    dates: List[str],
    daily_dir: Path,
) -> int:
    """按日写出行业文件；只保留当日日线里出现的股票。"""
    written = 0
    map_idx = mapping.set_index("code")["industry"]
    for date in dates:
        daily = pd.read_csv(daily_dir / f"{date}.csv", usecols=["code"])
        codes = daily["code"].astype(str)
        industry = map_idx.reindex(codes).dropna()
        if industry.empty:
            continue
        out = industry.reset_index()
        out.columns = ["code", "industry"]
        out = out.sort_values("code")
        out.to_csv(industry_dir / f"{date}.csv", index=False)
        written += 1
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="补全申万一级行业分类")
    parser.add_argument("--data-dir", dest="data_dir", default="./data")
    parser.add_argument("--start", default="2022-01-24", help="默认从课上行业结束后开始")
    parser.add_argument("--end", default="", help="结束日期，默认到最新有日线的交易日")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有行业文件")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    industry_dir = migrate_industry_dir(data_dir)
    daily_dir = data_dir / "data_daily"

    last_daily = None
    daily_files = sorted(daily_dir.glob("????-??-??.csv"))
    if daily_files:
        last_daily = daily_files[-1].stem
    end = args.end or last_daily or time.strftime("%Y-%m-%d")

    if args.overwrite:
        dates = []
        pkl = data_dir / "date.pkl"
        all_dates = pickle.load(open(pkl, "rb")) if pkl.exists() else [p.stem for p in daily_files]
        for date in all_dates:
            if args.start and date < args.start:
                continue
            if date > end:
                continue
            if (daily_dir / f"{date}.csv").exists():
                dates.append(date)
    else:
        dates = load_need_dates(data_dir, args.start, end)

    if not dates:
        print("没有需要补的行业日期")
        print(f"行业目录最后一日: {sorted(industry_dir.glob('*.csv'))[-1].stem if list(industry_dir.glob('*.csv')) else None}")
        return

    print(f"待补行业: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 天")
    mapping = fetch_sw_l1_map()
    cache = data_dir / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(cache / "sw_l1_industry_map.csv", index=False)

    n = write_industry_files(industry_dir, mapping, dates, daily_dir)
    files = sorted(industry_dir.glob("????-??-??.csv"))
    print(f"新增行业文件 {n} 个")
    print(f"行业目录: {files[0].stem} ~ {files[-1].stem}，共 {len(files)} 天")


if __name__ == "__main__":
    main()
