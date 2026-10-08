#!/usr/bin/env python3
"""
生成作战台回测缓存：扣费前/后净值、换手、当前持仓行业集中度。

  .venv/bin/python scripts/ops_desk/refresh_backtest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops_desk.data_apis import (  # noqa: E402
    BACKTEST_CACHE,
    BACKTEST_CACHE_MAIN,
    BACKTEST_CACHE_ST,
    OUT_DIR,
    build_ops_snapshot,
)


def _downsample_equity(dates: List[str], values: List[float], max_points: int = 240):
    n = len(dates)
    if n <= max_points:
        return [{"date": d, "nav": round(float(v), 6)} for d, v in zip(dates, values)]
    step = max(1, n // max_points)
    out = [
        {"date": dates[i], "nav": round(float(values[i]), 6)}
        for i in range(0, n, step)
    ]
    if out[-1]["date"] != dates[-1]:
        out.append({"date": dates[-1], "nav": round(float(values[-1]), 6)})
    return out


def _metrics_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "avg_turnover",
        "total_cost",
        "trade_count",
        "ic_mean",
        "ir",
    ]
    out = {}
    for k in keys:
        if k in report and report[k] is not None:
            try:
                out[k] = float(report[k])
            except (TypeError, ValueError):
                out[k] = report[k]
    return out


def run_one(*, enable_cost: bool, args) -> Dict[str, Any]:
    from main import (
        COMPOSITE_DIRS,
        DEFAULT_END,
        DEFAULT_START,
        build_strategy,
        composite_subdir,
        resolve_composite_dates,
    )
    from src.backtest.backtest_engine_strategy import BacktestEngine
    from src.common.config import load_config, resolve_config_path

    cfg = load_config(resolve_config_path(args.config or None), reload=True)
    bt = cfg.get("backtest") or {}
    paths = cfg.get("paths") or {}
    data_dir = str(paths.get("data_dir") or "./data")
    factor_root = str(paths.get("factor_root") or "./factors")
    strategy_name = args.strategy or str(bt.get("strategy") or "composite")
    top_n = int(args.top_n or bt.get("top_n") or 4)
    rebalance = int(args.rebalance or bt.get("rebalance") or 5)
    capital = float(args.capital or bt.get("capital") or 50000.0)

    key = strategy_name.lower()
    start = args.start or DEFAULT_START
    end = args.end or DEFAULT_END
    comp_name = composite_subdir(key, args.composite_name or "")
    if key in COMPOSITE_DIRS:
        start, end = resolve_composite_dates(
            factor_root, start, end, composite_name=comp_name
        )

    strategy = build_strategy(
        strategy_name,
        int(bt.get("period") or 20),
        data_dir=data_dir,
        factor_root=factor_root,
        composite_name=args.composite_name or "",
    )
    engine = BacktestEngine(
        data_dir=data_dir,
        initial_capital=capital,
        commission_rate=float(bt.get("commission_rate") or 0.00012),
        slippage_rate=float(bt.get("slippage_rate") or 0.001),
        stamp_duty=float(bt.get("stamp_duty") or 0.0005),
        transfer_fee_rate=float(bt.get("transfer_fee_rate") or 0.00001),
        risk_free_rate=float(bt.get("risk_free_rate") or 0.03),
    )
    report = engine.run(
        start_date=start,
        end_date=end,
        strategy=strategy,
        top_n=top_n,
        rebalance_freq=rebalance,
        enable_cost=enable_cost,
        calculate_ic=True,
        n_groups=0,
    )
    cum = report["cumulative_returns"]
    dates = [str(x)[:10] for x in cum.index.astype(str)]
    values = [float(x) for x in cum.values]
    return {
        "enable_cost": enable_cost,
        "strategy": strategy_name,
        "top_n": top_n,
        "rebalance": rebalance,
        "capital": capital,
        "metrics": _metrics_from_report(report),
        "equity": _downsample_equity(dates, values),
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "n_days": len(dates),
    }


def _write_track(track: str, args) -> Dict[str, Any]:
    """跑一条轨的扣费前/后回测并落盘。"""
    print(f"回测 {track} 扣费后…")
    with_cost = run_one(enable_cost=True, args=args)
    print(f"回测 {track} 扣费前…")
    no_cost = run_one(enable_cost=False, args=args)
    ops = build_ops_snapshot()
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ok": True,
        "track": track,
        "with_cost": with_cost,
        "no_cost": no_cost,
        "industry_live": {
            "main": ops["main"]["industry"],
            "st": ops["st"]["industry"],
        },
        "note": "行业集中度为当前 live 持仓（行业表按最近可用日对齐）",
    }
    path = BACKTEST_CACHE_MAIN if track == "main" else BACKTEST_CACHE_ST
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    if track == "main":
        # 出图页仍读旧单文件，同步主轨
        BACKTEST_CACHE.write_text(text, encoding="utf-8")
    print(f"已写入 {path}")
    m1 = with_cost["metrics"]
    m0 = no_cost["metrics"]
    print(
        f"{track} 扣费后 收益={m1.get('total_return'):.2%} 夏普={m1.get('sharpe_ratio'):.2f} "
        f"换手={m1.get('avg_turnover'):.2%}"
    )
    print(
        f"{track} 扣费前 收益={m0.get('total_return'):.2%} 夏普={m0.get('sharpe_ratio'):.2f}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新作战台回测缓存")
    parser.add_argument("--config", default="")
    parser.add_argument("--strategy", default="")
    parser.add_argument("--composite-name", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--rebalance", type=int, default=0)
    parser.add_argument("--capital", type=float, default=0.0)
    parser.add_argument(
        "--track",
        default="main",
        choices=("main", "st", "both"),
        help="main=冻结 composite；st=paper_st；both=两条都跑",
    )
    args = parser.parse_args()

    jobs = []
    if args.track in ("main", "both"):
        jobs.append(("main", ""))
    if args.track in ("st", "both"):
        jobs.append(("st", str(ROOT / "configs" / "paper_st.json")))

    for track, cfg in jobs:
        # 每条轨用独立 argparse 命名空间，避免短线配置污染主轨
        ns = argparse.Namespace(**vars(args))
        ns.config = cfg if track == "st" else (args.config or "")
        if track == "st" and not ns.strategy:
            ns.strategy = "composite_st"
        if track == "main" and not ns.strategy:
            ns.strategy = "composite"
        _write_track(track, ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
