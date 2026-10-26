# -*- coding: utf-8 -*-
"""
PTrade 试用策略：复制到券商 PTrade「新建策略」，周期选日线，先跑模拟交易。

试用范围（对应本仓库日更链路里能在云端算完的一段）：
- 股票池：中证500（000905.SS），并去掉科创板、北交所
- 三个日频价量因子，截面 z-score 后等权，方向与仓库一致（越大越好）
  cj_corr_vol_close_20：-corr(成交量, 收盘价, 20)
  idt_rev_20：-mean(收盘/开盘-1, 20)
  xb_amplitude：-(最高-最低)/收盘，用最近一根日线
- 每 5 个交易日调仓，持有 4 只，等权
- 盘前用截至上一交易日的后复权日线打分，盘中下单

不做市值行业中性化，不用 240 日 IC_IR，也不读本机 factors/composite。
行情字段按 PTrade 单字段多标的返回：行是日期，列是代码。
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.index_code = "000905.SS"
    g.top_n = 4
    g.rebalance_every = 5
    g.corr_window = 20
    g.idt_window = 20
    g.min_names = 30
    g.min_price = 2.0
    g.min_volume = 100000
    g.history_count = 25
    g.trade_days = 0
    g.targets = {}
    g.do_trade = False

    set_benchmark(g.index_code)
    set_commission(commission_ratio=0.00012, min_commission=5.0, type="STOCK")
    set_slippage(slippage=0.001)
    set_universe(g.index_code)


def before_trading_start(context, data):
    """上一交易日收盘打分，决定今天是否调仓。"""
    g.trade_days += 1
    g.do_trade = False
    g.targets = {}
    if (g.trade_days - 1) % g.rebalance_every != 0:
        return

    stocks = get_index_stocks(g.index_code)
    stocks = [code for code in stocks if _board_allowed(code)]
    if len(stocks) < g.min_names:
        log.info("可交易股票不足，跳过调仓：%s" % len(stocks))
        return

    set_universe(stocks)
    blocked = set(filter_stock_by_status(stocks, ["ST", "HALT", "DELISTING"]))
    limit_map = check_limit(stocks)
    close_raw = _last_row(_load_field(stocks, "close", 1, fq=None))
    volume_raw = _last_row(_load_field(stocks, "volume", 1, fq=None))

    tradable = []
    for code in stocks:
        if code in blocked:
            continue
        if limit_map.get(code) != 0:
            continue
        px = close_raw.get(code, np.nan)
        vol = volume_raw.get(code, np.nan)
        if px != px or vol != vol:
            continue
        if float(px) < g.min_price or float(vol) < g.min_volume:
            continue
        tradable.append(code)
    if len(tradable) < g.min_names:
        log.info("过滤后股票不足，跳过调仓：%s" % len(tradable))
        return

    open_ = _load_field(tradable, "open", g.history_count, fq="post")
    high = _load_field(tradable, "high", g.history_count, fq="post")
    low = _load_field(tradable, "low", g.history_count, fq="post")
    close = _load_field(tradable, "close", g.history_count, fq="post")
    volume = _load_field(tradable, "volume", g.history_count, fq="post")
    if len(close) < g.idt_window:
        log.info("历史长度不足，跳过调仓：%s" % len(close))
        return

    score = _composite_score(open_, high, low, close, volume)
    if len(score) < g.top_n:
        log.info("有效打分不足，跳过调仓：%s" % len(score))
        return

    chosen = list(score.sort_values(ascending=False).head(g.top_n).index)
    weight = round(1.0 / g.top_n, 6)
    g.targets = {code: weight for code in chosen}
    g.do_trade = True

    hold = list(context.portfolio.positions.keys())
    set_universe(list(set(chosen) | set(hold)))
    log.info("调仓日 %s，目标 %s" % (g.trade_days, chosen))


def handle_data(context, data):
    """先清非目标，再把目标调到等权。"""
    if not g.do_trade:
        return

    targets = g.targets
    positions = context.portfolio.positions
    for stock in list(positions.keys()):
        if positions[stock].amount > 0 and stock not in targets:
            order_target(stock, 0)
            log.info("清仓非目标: %s" % stock)

    for stock, weight in targets.items():
        order_target_percent(stock, weight)
        log.info("目标权重 %s -> %.4f" % (stock, weight))

    g.do_trade = False


def _board_allowed(code):
    """排除科创板和北交所，与本仓库清单规则一致。"""
    num = str(code).split(".")[0]
    suf = str(code).split(".")[-1].upper() if "." in str(code) else ""
    if suf in ("BJ", "BSE"):
        return False
    if num.startswith(("688", "689", "8", "43", "920")):
        return False
    return True


def _load_field(stocks, field, count, fq):
    """单字段多标的：index 为日期，columns 为代码。"""
    frame = get_history(
        count,
        frequency="1d",
        field=field,
        security_list=stocks,
        fq=fq,
        include=False,
    )
    return frame.sort_index()


def _last_row(frame):
    if frame is None or len(frame) == 0:
        return pd.Series(dtype=float)
    return frame.iloc[-1]


def _cs_z(series):
    """截面 z-score。样本过少或标准差为 0 时整列为空。"""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < g.min_names:
        return pd.Series(np.nan, index=series.index)
    std = float(clean.std(ddof=0))
    if std == 0.0 or std != std:
        return pd.Series(np.nan, index=series.index)
    return (series - float(clean.mean())) / std


def _composite_score(open_, high, low, close, volume):
    """三个因子各自取最后一天的值，z-score 后等权。三个都有值才保留。"""
    corr = close.tail(g.corr_window).corrwith(volume.tail(g.corr_window))
    corr_factor = -corr

    intraday = close / open_.replace(0, np.nan) - 1.0
    idt_factor = -intraday.tail(g.idt_window).mean()

    last_close = close.iloc[-1].replace(0, np.nan)
    amp_factor = -((high.iloc[-1] - low.iloc[-1]) / last_close)

    scored = pd.concat(
        [
            _cs_z(corr_factor),
            _cs_z(idt_factor),
            _cs_z(amp_factor),
        ],
        axis=1,
    )
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna()
    if scored.empty:
        return pd.Series(dtype=float)
    return scored.mean(axis=1)
