#!/usr/bin/env python3
"""
模拟盘流水线（不自动下单）。

默认读 configs/default.json：主轨 live，默认不再重写合成分。
短窗轨请用:
  python scripts/run_paper_pipeline.py --config configs/paper_st.json
  或 python scripts/run_paper_st_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# 保证仓库根在 path 中
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config, resolve_config_path
from src.common.logging_setup import get_logger, setup_logging
from src.live.paper_ledger import append_run


def parse_args():
    parser = argparse.ArgumentParser(description="模拟盘流水线：管线 + live + 账本")
    parser.add_argument(
        "--config",
        default="",
        help="配置文件路径，默认 configs/default.json 或 QUANT_CONFIG",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="跳过因子管线，仅跑 live + 账本",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="跳过 live（仅适合调试管线）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = resolve_config_path(args.config or None)
    cfg = load_config(cfg_path, reload=True)

    setup_logging(task="paper")
    log = get_logger("paper")
    log.info("加载配置: %s (version=%s)", cfg_path, cfg.get("version"))

    paths = cfg.get("paths") or {}
    pipeline_cfg = cfg.get("pipeline") or {}
    live_cfg = cfg.get("live") or {}
    paper_cfg = cfg.get("paper") or {}
    backtest_cfg = cfg.get("backtest") or {}

    data_dir = paths.get("data_dir") or "./data"
    factor_root = paths.get("factor_root") or "./factors"
    universe = paths.get("composite_universe") or "factors/composite_universe.json"
    composite_name = str(
        pipeline_cfg.get("composite_name")
        or paths.get("composite_name")
        or "composite"
    )
    live_dir = str(paths.get("live_dir") or "./outputs/live")
    paper_dir_path = str(paths.get("paper_dir") or "./outputs/paper")
    research_out = str(
        paper_cfg.get("research_output_dir") or "./outputs/factor_research"
    )
    top_n = int(live_cfg.get("top_n") or 4)
    capital = float(live_cfg.get("capital") or 50000.0)
    strategy = str(live_cfg.get("strategy") or "composite")

    run_pipeline = bool(paper_cfg.get("run_pipeline", True)) and not args.skip_pipeline
    run_live = bool(paper_cfg.get("run_live", True)) and not args.skip_live

    if run_pipeline:
        from src.research.factor_pipeline import run_factor_pipeline

        lookback = int(
            paper_cfg.get("pipeline_lookback_days")
            or pipeline_cfg.get("lookback_days")
            or 60
        )
        log.info(
            "开始因子管线 lookback=%s weight=%s composite=%s",
            lookback,
            pipeline_cfg.get("weight_method"),
            composite_name,
        )
        run_factor_pipeline(
            data_dir=data_dir,
            factor_root=factor_root,
            output_dir=research_out,
            start_date=None,
            end_date=None,
            lookback_days=lookback,
            weight_method=str(pipeline_cfg.get("weight_method") or "ic_ir"),
            use_family_weights=bool(pipeline_cfg.get("use_family_weights", False)),
            use_active_state=bool(pipeline_cfg.get("use_active_state", False)),
            active_window=int(pipeline_cfg.get("active_window") or 20),
            active_min_ir=float(pipeline_cfg.get("active_min_ir") or 0.10),
            composite_universe_path=universe,
            use_rolling_weights=bool(pipeline_cfg.get("use_rolling_weights", False)),
            rolling_lookback=int(pipeline_cfg.get("rolling_lookback") or 60),
            rolling_min_history=int(pipeline_cfg.get("rolling_min_history") or 20),
            composite_name=composite_name,
        )
        log.info("因子管线完成")
    else:
        log.info("跳过因子管线")

    if not run_live:
        log.info("跳过 live，结束")
        return 0

    from src.live import live_signal
    from src.live.live_signal import run_live as run_live_fn

    ns = SimpleNamespace(
        strategy=strategy,
        top_n=top_n,
        capital=capital,
        period=int(backtest_cfg.get("period") or 20),
        data_dir=data_dir,
        factor_root=factor_root,
        composite_name=composite_name,
        live_dir=live_dir,
    )

    previous = live_signal.load_previous_holdings(live_dir)
    log.info(
        "开始 live 清单 strategy=%s top_n=%s capital=%s live_dir=%s",
        strategy,
        top_n,
        capital,
        live_dir,
    )
    run_live_fn(ns)

    _, holdings_path = live_signal.resolve_live_paths(live_dir)
    holdings = json.loads(holdings_path.read_text(encoding="utf-8"))
    codes = list(holdings.get("codes") or [])
    signal_date = str(holdings.get("date") or "")
    buy = [c for c in codes if c not in previous]
    sell = [c for c in previous if c not in codes]

    from src.data.data_loader import DataLoader

    loader = DataLoader(data_dir)
    trade_date = live_signal.next_trade_date(loader, signal_date) if signal_date else ""

    ledger_path = append_run(
        signal_date=signal_date,
        trade_date=trade_date,
        codes=codes,
        buy=buy,
        sell=sell,
        capital=capital,
        strategy=strategy,
        extra={"previous": previous, "config": str(cfg_path)},
        paper_dir_path=paper_dir_path,
    )
    log.info(
        "模拟盘账本已写 %s | signal=%s trade=%s hold=%s buy=%s sell=%s",
        ledger_path,
        signal_date,
        trade_date,
        codes,
        buy,
        sell,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
