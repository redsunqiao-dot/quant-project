"""
可交易股票过滤。

日线 close 已是后复权，价格门槛必须用不复权的 data_ud_new。
"""

from typing import Set

import pandas as pd


MIN_PRICE = 2.0
MIN_VOLUME = 1e5
MIN_TURNOVER = 0.0005


def is_board_allowed(code: str) -> bool:
    """
    小资金可交易板块：排除科创板(688/689)与北交所。
    主板 / 能买的创业板保留。
    """
    raw = str(code).strip()
    num = raw.split(".")[0]
    suf = raw.split(".")[-1].upper() if "." in raw else ""
    if suf in {"BJ", "BSE"}:
        return False
    if num.startswith(("688", "689")):
        return False
    # 北交所常见号段：8xxxxx / 43xxxx / 920xxx
    if num.startswith(("8", "43", "920")):
        return False
    return True


def _is_chinext(code: str) -> bool:
    number = str(code).split(".")[0]
    return number.startswith(("300", "301", "688", "689"))


def infer_st(status: pd.DataFrame) -> pd.Series:
    """识别 ST。优先用状态文件的 st 列；没有则按约 5% 涨跌停推断（含创业板/科创板）。"""
    if "st" in status.columns:
        return status["st"].fillna(0).astype(int) == 1
    pre = status["pre_close"].astype(float)
    high = status["high_limit"].astype(float)
    ratio = high / pre - 1.0
    return (ratio < 0.08) & (pre > 0)


def tradeable_codes(daily: pd.DataFrame, status: pd.DataFrame) -> Set[str]:
    """返回当日可买入的股票代码。"""
    if daily.empty or status.empty:
        return set()

    merged = pd.merge(
        daily[["code", "volume", "turnover_ratio"]],
        status,
        on="code",
        how="inner",
    )
    if merged.empty:
        return set()

    price = merged["open"].astype(float)
    if "pre_close" in merged.columns:
        price = merged["pre_close"].astype(float)

    mask = (
        (merged["paused"] == 0)
        & (merged["zt"] == 0)
        & (merged["dt"] == 0)
        & (price >= MIN_PRICE)
        & (merged["volume"] >= MIN_VOLUME)
        & (merged["turnover_ratio"] >= MIN_TURNOVER)
        & (~infer_st(merged))
    )
    return set(merged.loc[mask, "code"].astype(str))


def keep_tradeable(factor_df: pd.DataFrame, date: str, data_loader) -> pd.DataFrame:
    """在已算好的因子表上剔除不可交易股票，保留原有列。"""
    if factor_df.empty:
        return factor_df

    daily = data_loader.get_daily_data(date)
    status = data_loader.get_daily_status(date)
    keep = tradeable_codes(daily, status)
    if not keep:
        return factor_df.iloc[0:0].copy()
    return factor_df[factor_df["code"].astype(str).isin(keep)].copy()
