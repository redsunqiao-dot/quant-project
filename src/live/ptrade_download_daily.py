# -*- coding: utf-8 -*-
"""
在 PTrade 研究笔记本里整段运行，把全市场日线落到研究目录。

运行前把笔记本内核选成 Python 3.11。多股票 get_price 在该版本返回
带 code 列的 DataFrame。数据不含当天，最早可到 2005 年。

默认从 2020-01-01 拉到上一交易日，后复权，每年一个 CSV：
    get_research_path() 下的 market_daily_post_2020.csv 起
股票列表是运行当天的 A 股，不含已退市代码。
"""

import pandas as pd


# 研究里改这两行即可
START = "2020-01-01"
BATCH = 100
FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "preclose",
    "high_limit",
    "low_limit",
    "is_open",
]


def _year_slices(start, end):
    """按自然年切开，避免一次取出全部历史。"""
    start_year = int(start[:4])
    slices = []
    for year in range(start_year, end.year + 1):
        lo = start if year == start_year else "%s-01-01" % year
        hi = "%s-12-31" % year
        end_str = end.strftime("%Y-%m-%d")
        if hi > end_str:
            hi = end_str
        if lo <= hi:
            slices.append((year, lo, hi))
    return slices


def _with_date_column(frame):
    """行索引是日期，落盘前变成 date 列。"""
    out = frame.reset_index()
    first = out.columns[0]
    if first != "date":
        out = out.rename(columns={first: "date"})
    return out


def download_daily():
    end = get_trading_day(-1)
    codes = get_Ashares()
    root = get_research_path()
    print("股票数 %s，区间 %s ~ %s，复权 post" % (len(codes), START, end))
    print("输出目录 %s" % root)

    for year, lo, hi in _year_slices(START, end):
        frames = []
        for offset in range(0, len(codes), BATCH):
            chunk = codes[offset:offset + BATCH]
            raw = get_price(
                chunk,
                start_date=lo,
                end_date=hi,
                frequency="1d",
                fields=FIELDS,
                fq="post",
            )
            if raw is None or len(raw) == 0:
                print("空批次 %s offset=%s" % (year, offset))
                continue
            frames.append(_with_date_column(raw))
            print("完成 %s offset=%s 行数=%s" % (year, offset, len(raw)))
        if not frames:
            print("该年没有数据 %s" % year)
            continue
        out = pd.concat(frames, ignore_index=True)
        path = root + "market_daily_post_%s.csv" % year
        out.to_csv(path, index=False)
        print("写出 %s 行数=%s" % (path, len(out)))


download_daily()
