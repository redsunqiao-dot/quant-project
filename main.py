"""
本地多因子研究入口。

默认跑课上主链路：算因子 -> 过滤 -> 预处理 -> IC 评价 -> 合成。
回测和收盘清单是可选后续步骤，不是默认目标。

默认参数来自 configs/default.json（可用 --config / QUANT_CONFIG 覆盖）；CLI 优先。
"""

import argparse
from pathlib import Path

from src.backtest.backtest_engine_strategy import BacktestEngine
from src.common.config import load_config, resolve_config_path
from src.common.logging_setup import get_logger, setup_logging
from src.common.utils import list_dated_stems
from src.research.factor_pipeline import run_factor_pipeline
from src.strategy.strategy_base import MomentumStrategy, ReversalStrategy


DEFAULT_START = "2020-03-02"
DEFAULT_END = "2021-06-30"

# 以下常量保留作回退；正常路径由 configs/default.json 提供
DEFAULT_DATA_DIR = "./data"
DEFAULT_FACTOR_ROOT = "./factors"
LIVE_CAPITAL = 50000.0
LIVE_TOP_N = 4
PIPELINE_LOOKBACK = 240


def _cfg_defaults(config_path: str = ""):
    """从版本化配置读取入口默认值。"""
    cfg = load_config(resolve_config_path(config_path or None), reload=True)
    paths = cfg.get("paths") or {}
    pipeline = cfg.get("pipeline") or {}
    backtest = cfg.get("backtest") or {}
    live = cfg.get("live") or {}
    return {
        "data_dir": paths.get("data_dir") or DEFAULT_DATA_DIR,
        "factor_root": paths.get("factor_root") or DEFAULT_FACTOR_ROOT,
        "composite_universe": paths.get("composite_universe")
        or "factors/composite_universe.json",
        "lookback_days": int(pipeline.get("lookback_days") or PIPELINE_LOOKBACK),
        "weight_method": str(pipeline.get("weight_method") or "ic_ir"),
        "use_family_weights": bool(pipeline.get("use_family_weights", False)),
        "use_active_state": bool(pipeline.get("use_active_state", False)),
        "active_window": int(pipeline.get("active_window") or 20),
        "active_min_ir": float(pipeline.get("active_min_ir") or 0.10),
        "use_rolling_weights": bool(pipeline.get("use_rolling_weights", False)),
        "rolling_lookback": int(pipeline.get("rolling_lookback") or 60),
        "rolling_min_history": int(pipeline.get("rolling_min_history") or 20),
        "composite_name": str(
            pipeline.get("composite_name")
            or paths.get("composite_name")
            or "composite"
        ),
        "live_dir": str(paths.get("live_dir") or "./outputs/live"),
        "strategy": str(backtest.get("strategy") or live.get("strategy") or "composite"),
        "top_n_live": int(live.get("top_n") or LIVE_TOP_N),
        "capital_live": float(live.get("capital") or LIVE_CAPITAL),
        "rebalance": int(backtest.get("rebalance") or 5),
        "period": int(backtest.get("period") or 20),
        "commission_rate": float(backtest.get("commission_rate") or 0.00012),
        "slippage_rate": float(backtest.get("slippage_rate") or 0.001),
        "stamp_duty": float(backtest.get("stamp_duty") or 0.0005),
        "transfer_fee_rate": float(backtest.get("transfer_fee_rate") or 0.00001),
        "risk_free_rate": float(backtest.get("risk_free_rate") or 0.03),
    }


# 合成分策略名 → factors 下子目录
COMPOSITE_DIRS = {
    "composite": "composite",
    "composite_st": "composite_st",
}


def composite_subdir(strategy_name: str, composite_name: str = "") -> str:
    """解析合成分落盘子目录。"""
    if composite_name:
        return str(composite_name).strip()
    key = strategy_name.lower()
    if key in COMPOSITE_DIRS:
        return COMPOSITE_DIRS[key]
    return "composite"


def build_strategy(
    name: str,
    period: int,
    data_dir: str,
    factor_root: str,
    composite_name: str = "",
):
    """按名称构造策略对象，供回测和清单使用。"""
    key = name.lower()
    if key == "momentum":
        return MomentumStrategy(period=period)
    if key == "reversal":
        return ReversalStrategy(period=period)
    if key == "cpv":
        from src.strategy.cpv_strategy import CPVStrategy
        return CPVStrategy(data_dir=data_dir)
    if key in COMPOSITE_DIRS or (composite_name and key.startswith("composite")):
        from src.strategy.composite_strategy import CompositeStrategy
        sub = composite_subdir(key, composite_name)
        label = "CompositeST" if sub == "composite_st" else "Composite"
        return CompositeStrategy(
            factor_dir=str(Path(factor_root) / sub),
            name=label,
        )
    raise ValueError(
        "未知策略: "
        f"{name}，可选 momentum / reversal / cpv / composite / composite_st / multifactor"
    )


def resolve_composite_dates(
    factor_root: str,
    start: str,
    end: str,
    composite_name: str = "composite",
) -> tuple[str, str]:
    """合成分回测区间对齐到 factors/<composite_name> 已有日期。"""
    sub = composite_name or "composite"
    dates = list_dated_stems(Path(factor_root) / sub)
    if not dates:
        raise FileNotFoundError(
            f"未找到合成分文件: {Path(factor_root) / sub}。"
            f"请先跑管线生成 factors/{sub}。"
        )
    lo, hi = dates[0], dates[-1]
    # 未指定或仍是单因子默认历史窗时，直接用合成分覆盖区间
    if not start or start == DEFAULT_START:
        start = lo
    if not end or end == DEFAULT_END:
        end = hi
    start = max(start, lo)
    end = min(end, hi)
    if start > end:
        raise ValueError(
            f"回测区间与合成分无交集: 请求落在 {lo}~{hi} 之外"
        )
    print(f"合成分覆盖 {lo} ~ {hi}，回测使用 {start} ~ {end}")
    return start, end


def run_backtest(args) -> dict:
    """运行策略回测并打印报告。"""
    data_dir = Path(args.data_dir)
    if not (data_dir / "date.pkl").exists():
        raise FileNotFoundError(f"未找到行情数据: {data_dir / 'date.pkl'}")

    key = args.strategy.lower()
    if key == "multifactor":
        # 第13课完整验证：滚动权重 + raw/neu + 含成本引擎
        from src.research.backtest_validation import run as run_day13_validation

        run_day13_validation(
            data_dir=str(data_dir),
            start_date=args.start or None,
            end_date=args.end or None,
            top_n=args.top_n,
            rebalance_freq=args.rebalance,
        )
        return {}

    start = args.start or DEFAULT_START
    end = args.end or DEFAULT_END
    comp_name = composite_subdir(
        key, getattr(args, "composite_name", "") or ""
    )
    if key in COMPOSITE_DIRS:
        start, end = resolve_composite_dates(
            args.factor_root, start, end, composite_name=comp_name
        )

    strategy = build_strategy(
        args.strategy,
        args.period,
        data_dir=str(data_dir),
        factor_root=args.factor_root,
        composite_name=getattr(args, "composite_name", "") or "",
    )
    engine = BacktestEngine(
        data_dir=str(data_dir),
        initial_capital=args.capital,
        commission_rate=getattr(args, "commission_rate", 0.00012),
        slippage_rate=getattr(args, "slippage_rate", 0.001),
        stamp_duty=getattr(args, "stamp_duty", 0.0005),
        transfer_fee_rate=getattr(args, "transfer_fee_rate", 0.00001),
        risk_free_rate=getattr(args, "risk_free_rate", 0.03),
    )
    report = engine.run(
        start_date=start,
        end_date=end,
        strategy=strategy,
        top_n=args.top_n,
        rebalance_freq=args.rebalance,
        enable_cost=True,
        calculate_ic=True,
        n_groups=5,
        max_per_industry=getattr(args, "max_per_industry", 2),
    )
    engine.print_report(report)
    return report


def parse_args():
    # 先读 --config，再填其余默认，避免循环依赖
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="", help=argparse.SUPPRESS)
    pre_args, _ = pre.parse_known_args()
    d = _cfg_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="本地多因子研究入口")
    parser.add_argument(
        "--config",
        default="",
        help="配置文件（默认 configs/default.json 或环境变量 QUANT_CONFIG）",
    )
    parser.add_argument("--pipeline", action="store_true", help="跑因子研究主链路（默认）")
    parser.add_argument("--backtest", action="store_true", help="跑策略回测")
    parser.add_argument("--live", action="store_true", help="用最新交易日生成下一交易日买卖清单")
    parser.add_argument(
        "--strategy",
        default=d["strategy"],
        help="回测/清单: composite(默认) / composite_st / multifactor / momentum / reversal / cpv",
    )
    parser.add_argument("--start", default="", help="研究或回测开始日期，研究默认最近 240 个交易日")
    parser.add_argument("--end", default="", help="研究或回测结束日期，默认最新交易日")
    parser.add_argument("--top-n", dest="top_n", type=int, default=None, help="回测/清单持股数量")
    parser.add_argument("--period", type=int, default=d["period"], help="动量/反转回看天数")
    parser.add_argument("--rebalance", type=int, default=d["rebalance"], help="调仓间隔（交易日）")
    parser.add_argument("--capital", type=float, default=None, help="回测/清单资金")
    parser.add_argument("--data-dir", dest="data_dir", default=d["data_dir"], help="数据目录")
    parser.add_argument(
        "--factor-root",
        dest="factor_root",
        default=d["factor_root"],
        help="因子根目录（composite 读 factors/composite）",
    )
    parser.add_argument(
        "--weight",
        default=d["weight_method"],
        help="合成加权: equal / ic / ic_ir / ret / family_ic_ir",
    )
    parser.add_argument(
        "--family-weights",
        action="store_true",
        default=d["use_family_weights"],
        help="开启分族估权（默认跟随配置，一般为关闭）",
    )
    parser.add_argument(
        "--active-state",
        action="store_true",
        default=d["use_active_state"],
        help="开启滚动 IC 活跃态门控（默认跟随配置，一般为关闭）",
    )
    parser.add_argument(
        "--active-window",
        type=int,
        default=d["active_window"],
        help="活跃态滚动 IC 窗口（交易日）",
    )
    parser.add_argument(
        "--active-min-ir",
        type=float,
        default=d["active_min_ir"],
        help="活跃态滚动 IC_IR 下限",
    )
    parser.add_argument(
        "--composite-universe",
        default=d["composite_universe"],
        help="冻结合成因子名单 JSON；空字符串则回退 registry active",
    )
    parser.add_argument(
        "--composite-name",
        default=d["composite_name"],
        help="合成分落盘子目录名（默认 composite；短窗轨 composite_st）",
    )
    parser.add_argument(
        "--live-dir",
        dest="live_dir",
        default=d["live_dir"],
        help="live 清单与持仓目录（短窗轨可用 outputs/live_st）",
    )
    parser.add_argument(
        "--rolling-weights",
        action=argparse.BooleanOptionalAction,
        default=d["use_rolling_weights"],
        help="合成用滚动 IC_IR 权重（默认跟随配置）",
    )
    parser.add_argument(
        "--rolling-lookback",
        type=int,
        default=d["rolling_lookback"],
        help="滚动 IC_IR 回看交易日数",
    )
    parser.add_argument(
        "--rolling-min-history",
        type=int,
        default=d["rolling_min_history"],
        help="滚动估权最少历史交易日",
    )
    args = parser.parse_args()
    # 成本参数挂到 args，供回测引擎使用
    args.commission_rate = d["commission_rate"]
    args.slippage_rate = d["slippage_rate"]
    args.stamp_duty = d["stamp_duty"]
    args.transfer_fee_rate = d["transfer_fee_rate"]
    args.risk_free_rate = d["risk_free_rate"]
    args.lookback_days = d["lookback_days"]

    if args.live:
        args.top_n = d["top_n_live"] if args.top_n is None else args.top_n
        args.capital = d["capital_live"] if args.capital is None else args.capital
    elif args.backtest:
        args.top_n = d["top_n_live"] if args.top_n is None else args.top_n
        args.capital = d["capital_live"] if args.capital is None else args.capital
    else:
        args.top_n = 10 if args.top_n is None else args.top_n
        args.capital = 1000000.0 if args.capital is None else args.capital
    return args


if __name__ == "__main__":
    parsed = parse_args()
    setup_logging(task="main")
    log = get_logger("main")
    if parsed.live:
        log.info("模式=live strategy=%s top_n=%s", parsed.strategy, parsed.top_n)
        from src.live.live_signal import run_live
        run_live(parsed)
    elif parsed.backtest:
        log.info("模式=backtest strategy=%s", parsed.strategy)
        run_backtest(parsed)
    else:
        uni = (parsed.composite_universe or "").strip() or None
        log.info(
            "模式=pipeline weight=%s lookback=%s composite=%s",
            parsed.weight,
            parsed.lookback_days,
            parsed.composite_name,
        )
        run_factor_pipeline(
            data_dir=parsed.data_dir,
            start_date=parsed.start or None,
            end_date=parsed.end or None,
            lookback_days=parsed.lookback_days,
            weight_method=parsed.weight,
            use_family_weights=parsed.family_weights,
            use_active_state=parsed.active_state,
            active_window=parsed.active_window,
            active_min_ir=parsed.active_min_ir,
            composite_universe_path=uni,
            use_rolling_weights=parsed.rolling_weights,
            rolling_lookback=parsed.rolling_lookback,
            rolling_min_history=parsed.rolling_min_history,
            composite_name=parsed.composite_name,
        )
