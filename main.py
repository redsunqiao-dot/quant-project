"""
本地多因子研究入口。

默认跑课上主链路：算因子 -> 过滤 -> 预处理 -> IC 评价 -> 合成。
回测和收盘清单是可选后续步骤，不是默认目标。
"""

import argparse
from pathlib import Path

from src.backtest.backtest_engine_strategy import BacktestEngine
from src.common.utils import list_dated_stems
from src.research.factor_pipeline import run_factor_pipeline
from src.strategy.strategy_base import MomentumStrategy, ReversalStrategy


DEFAULT_START = "2020-03-02"
DEFAULT_END = "2021-06-30"
DEFAULT_DATA_DIR = "./data"
DEFAULT_FACTOR_ROOT = "./factors"
LIVE_CAPITAL = 50000.0
LIVE_TOP_N = 4
PIPELINE_LOOKBACK = 240


def build_strategy(name: str, period: int, data_dir: str, factor_root: str):
    """按名称构造策略对象，供回测和清单使用。"""
    key = name.lower()
    if key == "momentum":
        return MomentumStrategy(period=period)
    if key == "reversal":
        return ReversalStrategy(period=period)
    if key == "cpv":
        from src.strategy.cpv_strategy import CPVStrategy
        return CPVStrategy(data_dir=data_dir)
    if key == "composite":
        from src.strategy.composite_strategy import CompositeStrategy
        return CompositeStrategy(factor_dir=str(Path(factor_root) / "composite"))
    raise ValueError(
        f"未知策略: {name}，可选 momentum / reversal / cpv / composite / multifactor"
    )


def resolve_composite_dates(factor_root: str, start: str, end: str) -> tuple[str, str]:
    """合成分回测区间对齐到 factors/composite 已有日期。"""
    dates = list_dated_stems(Path(factor_root) / "composite")
    if not dates:
        raise FileNotFoundError(
            f"未找到合成分文件: {Path(factor_root) / 'composite'}。"
            "请先跑 python main.py --pipeline 生成 factors/composite。"
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
    if key == "composite":
        start, end = resolve_composite_dates(args.factor_root, start, end)

    strategy = build_strategy(
        args.strategy,
        args.period,
        data_dir=str(data_dir),
        factor_root=args.factor_root,
    )
    engine = BacktestEngine(
        data_dir=str(data_dir),
        initial_capital=args.capital,
        commission_rate=0.00012,
        slippage_rate=0.001,
        stamp_duty=0.0005,
        transfer_fee_rate=0.00001,
        risk_free_rate=0.03,
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
    )
    engine.print_report(report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="本地多因子研究入口")
    parser.add_argument("--pipeline", action="store_true", help="跑因子研究主链路（默认）")
    parser.add_argument("--backtest", action="store_true", help="跑策略回测")
    parser.add_argument("--live", action="store_true", help="用最新交易日生成下一交易日买卖清单")
    parser.add_argument(
        "--strategy",
        default="composite",
        help="回测/清单: composite(默认) / multifactor / momentum / reversal / cpv",
    )
    parser.add_argument("--start", default="", help="研究或回测开始日期，研究默认最近 240 个交易日")
    parser.add_argument("--end", default="", help="研究或回测结束日期，默认最新交易日")
    parser.add_argument("--top-n", dest="top_n", type=int, default=None, help="回测/清单持股数量")
    parser.add_argument("--period", type=int, default=20, help="动量/反转回看天数")
    parser.add_argument("--rebalance", type=int, default=5, help="调仓间隔（交易日）")
    parser.add_argument("--capital", type=float, default=None, help="回测/清单资金")
    parser.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR, help="数据目录")
    parser.add_argument(
        "--factor-root",
        dest="factor_root",
        default=DEFAULT_FACTOR_ROOT,
        help="因子根目录（composite 读 factors/composite）",
    )
    parser.add_argument("--weight", default="ic_ir", help="合成加权: equal / ic / ic_ir / ret / family_ic_ir")
    parser.add_argument(
        "--family-weights",
        action="store_true",
        help="开启分族估权（默认关闭）",
    )
    parser.add_argument(
        "--active-state",
        action="store_true",
        help="开启滚动 IC 活跃态门控（默认关闭）",
    )
    parser.add_argument(
        "--active-window",
        type=int,
        default=20,
        help="活跃态滚动 IC 窗口（交易日）",
    )
    parser.add_argument(
        "--active-min-ir",
        type=float,
        default=0.10,
        help="活跃态滚动 IC_IR 下限",
    )
    parser.add_argument(
        "--composite-universe",
        default="factors/composite_universe.json",
        help="冻结合成因子名单 JSON；空字符串则回退 registry active",
    )
    parser.add_argument(
        "--rolling-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="合成用滚动 IC_IR 权重（默认关；加 --rolling-weights 开启）",
    )
    parser.add_argument(
        "--rolling-lookback",
        type=int,
        default=60,
        help="滚动 IC_IR 回看交易日数",
    )
    parser.add_argument(
        "--rolling-min-history",
        type=int,
        default=20,
        help="滚动估权最少历史交易日",
    )
    args = parser.parse_args()
    if args.live:
        args.top_n = LIVE_TOP_N if args.top_n is None else args.top_n
        args.capital = LIVE_CAPITAL if args.capital is None else args.capital
    elif args.backtest:
        args.top_n = LIVE_TOP_N if args.top_n is None else args.top_n
        args.capital = LIVE_CAPITAL if args.capital is None else args.capital
    else:
        args.top_n = 10 if args.top_n is None else args.top_n
        args.capital = 1000000.0 if args.capital is None else args.capital
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.live:
        from src.live.live_signal import run_live
        run_live(parsed)
    elif parsed.backtest:
        run_backtest(parsed)
    else:
        uni = (parsed.composite_universe or "").strip() or None
        run_factor_pipeline(
            data_dir=parsed.data_dir,
            start_date=parsed.start or None,
            end_date=parsed.end or None,
            lookback_days=PIPELINE_LOOKBACK,
            weight_method=parsed.weight,
            use_family_weights=parsed.family_weights,
            use_active_state=parsed.active_state,
            active_window=parsed.active_window,
            active_min_ir=parsed.active_min_ir,
            composite_universe_path=uni,
            use_rolling_weights=parsed.rolling_weights,
            rolling_lookback=parsed.rolling_lookback,
            rolling_min_history=parsed.rolling_min_history,
        )
