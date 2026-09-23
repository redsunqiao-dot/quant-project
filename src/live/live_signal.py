"""
收盘后生成下一交易日的手工买卖清单。

全市场选股，等权分配资金。建议股数按 100 股一手、用不复权开盘价估算。
买不起一手的票用因子下一名补上。
"""

import json
from pathlib import Path

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.tradeable_filter import keep_tradeable, is_board_allowed


LIVE_DIR = Path("./outputs/live")
HOLDINGS_FILE = LIVE_DIR / "current_holdings.json"
LOT_SIZE = 100


def latest_signal_date(loader: DataLoader) -> str:
    """取同时具备日线和交易状态的最后一个交易日。"""
    dates = loader.get_all_dates()
    for date in reversed(dates):
        daily = loader.data_dir / "data_daily" / f"{date}.csv"
        status = loader.data_dir / "data_ud_new" / f"{date}.csv"
        if daily.exists() and status.exists():
            return date
    raise FileNotFoundError("没有可用的信号日期")


def next_trade_date(loader: DataLoader, signal_date: str) -> str:
    """信号日的下一个交易日；若尚未发生则标为待定。"""
    dates = loader.get_all_dates()
    idx = dates.index(signal_date)
    if idx + 1 < len(dates):
        return dates[idx + 1]
    return "下一交易日"


def load_previous_holdings() -> list:
    """读取上一次保存的持仓代码。"""
    if not HOLDINGS_FILE.exists():
        return []
    payload = json.loads(HOLDINGS_FILE.read_text(encoding="utf-8"))
    return list(payload.get("codes", []))


def save_holdings(signal_date: str, codes: list, capital: float) -> None:
    """保存本期目标持仓，供下次对照。"""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": signal_date,
        "codes": codes,
        "capital": capital,
    }
    HOLDINGS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def unadjusted_price(status: pd.DataFrame) -> pd.Series:
    """不复权参考价，用于估算可买手数。"""
    if "close" in status.columns:
        return status.set_index("code")["close"].astype(float)
    return status.set_index("code")["open"].astype(float)


def pick_affordable(
    ranked_codes: list,
    price: pd.Series,
    top_n: int,
    capital: float,
) -> list:
    """按因子顺序挑选买得起一手的股票，直到满 top_n。"""
    budget = capital / top_n
    selected = []
    for code in ranked_codes:
        if code not in price.index:
            continue
        px = float(price.loc[code])
        if px <= 0:
            continue
        shares = int(budget // (px * LOT_SIZE)) * LOT_SIZE
        if shares < LOT_SIZE:
            continue
        selected.append(code)
        if len(selected) >= top_n:
            break
    return selected


def build_order_table(
    selected: list,
    previous: list,
    price: pd.Series,
    factor_df: pd.DataFrame,
    capital: float,
    signal_date: str,
    exec_date: str,
) -> pd.DataFrame:
    """生成买入/卖出/继续持有清单。"""
    prev_set = set(previous)
    new_set = set(selected)
    factor_map = factor_df.set_index("code")["factor_value"]
    budget = capital / max(len(selected), 1)

    rows = []
    for code in selected:
        px = float(price.loc[code])
        shares = int(budget // (px * LOT_SIZE)) * LOT_SIZE
        amount = shares * px
        action = "继续持有" if code in prev_set else "买入"
        rows.append(
            {
                "signal_date": signal_date,
                "exec_date": exec_date,
                "action": action,
                "code": code,
                "factor_value": float(factor_map.get(code, float("nan"))),
                "ref_price": round(px, 2),
                "shares": shares,
                "amount": round(amount, 2),
                "weight": round(1.0 / len(selected), 4),
            }
        )

    for code in previous:
        if code in new_set:
            continue
        px = float(price.loc[code]) if code in price.index else 0.0
        rows.append(
            {
                "signal_date": signal_date,
                "exec_date": exec_date,
                "action": "卖出",
                "code": code,
                "factor_value": float(factor_map.get(code, float("nan"))),
                "ref_price": round(px, 2) if px else None,
                "shares": None,
                "amount": None,
                "weight": 0.0,
            }
        )

    order = {"卖出": 0, "买入": 1, "继续持有": 2}
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["action_ord"] = table["action"].map(order)
    table = table.sort_values(["action_ord", "factor_value"], ascending=[True, False])
    return table.drop(columns=["action_ord"])


def run_live(args) -> pd.DataFrame:
    """计算最新交易日因子并输出手工执行清单。"""
    from main import build_strategy

    loader = DataLoader(args.data_dir)
    signal_date = latest_signal_date(loader)
    exec_date = next_trade_date(loader, signal_date)
    factor_root = getattr(args, "factor_root", "./factors")
    strategy = build_strategy(
        args.strategy,
        args.period,
        data_dir=args.data_dir,
        factor_root=factor_root,
    )

    factor_df = strategy.calculate_factor(signal_date, loader)
    factor_df = keep_tradeable(factor_df, signal_date, loader)
    if factor_df.empty:
        raise RuntimeError(f"{signal_date} 过滤后没有可交易股票")

    ranked = strategy.generate_signal(factor_df, top_n=len(factor_df))
    ranked = [c for c in ranked if is_board_allowed(c)]
    status = loader.get_daily_status(signal_date)
    price = unadjusted_price(status)
    selected = pick_affordable(ranked, price, args.top_n, args.capital)
    if not selected:
        raise RuntimeError("按当前资金/板块限制，没有买得起一手的股票")

    previous = load_previous_holdings()
    table = build_order_table(
        selected=selected,
        previous=previous,
        price=price,
        factor_df=factor_df,
        capital=args.capital,
        signal_date=signal_date,
        exec_date=exec_date,
    )

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = LIVE_DIR / f"{signal_date}_orders.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_holdings(signal_date, selected, args.capital)

    # 同步导出 PTrade 可粘贴策略（云端一般读不到本机 CSV）
    from src.live.ptrade_export import export_ptrade_files
    from src.live.qmt_export import export_qmt_files

    ptrade_paths = export_ptrade_files(
        codes=selected,
        signal_date=signal_date,
        code_style=getattr(args, "ptrade_code_style", "ss_sz"),
    )
    qmt_paths = export_qmt_files(codes=selected, signal_date=signal_date)

    print(f"信号日: {signal_date}")
    print(f"执行日: {exec_date}")
    print(f"策略: {strategy.name}")
    print(f"资金: {args.capital:.0f} 元，目标持股 {len(selected)} 只")
    print(f"清单已写入: {out_csv}")
    print(f"PTrade目标: {ptrade_paths['targets_json']}")
    print(f"PTrade策略: {ptrade_paths['strategy_py']}")
    print(f"QMT目标: {qmt_paths['targets_json']}")
    print(f"QMT策略: {qmt_paths['strategy_py']}")
    print()
    print(table.to_string(index=False))
    return table
