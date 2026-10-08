#!/usr/bin/env python3
"""
短窗轨实验：lookback=60 估权下的冻结池 LOO + 估权对照。

约定：
- 只写 outputs/st_track_* 与可选 factors/composite_st_* 实验目录
- 不覆盖 factors/composite（主研究盘）
- 默认交易参数对齐 paper_st：Top5 / 每 3 日调仓

用法:
  .venv/bin/python scripts/st_track_loo_weights.py
  .venv/bin/python scripts/st_track_loo_weights.py --apply-best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.backtest_engine_strategy import BacktestEngine
from src.common.utils import (
    list_dated_stems,
    normalize_weights,
    read_frame,
    resolve_dated_file,
    write_factor_frame,
)
from src.data.data_loader import DataLoader
from src.data.tradeable_filter import is_board_allowed, keep_tradeable
from src.research.factor_pipeline import (
    _build_panel,
    build_rolling_ic_ir_weights,
    load_composite_universe,
)
from src.research.multifactor_weights import compute_factor_metrics
from src.strategy.strategy_base import Strategy


class StCompositeStrategy(Strategy):
    """预计算合成分 + 与主轨一致的可交易/板块过滤。"""

    def __init__(self, name: str, signals: Dict[str, pd.DataFrame]):
        super().__init__(name=name)
        self.signals = signals

    def calculate_factor(self, date: str, data_loader, **kwargs) -> pd.DataFrame:
        df = self.signals.get(date)
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "date", "factor_value"])
        out = df.copy()
        if "factor_value" not in out.columns and "composite" in out.columns:
            out = out.rename(columns={"composite": "factor_value"})
        out["date"] = date
        return keep_tradeable(out[["code", "date", "factor_value"]], date, data_loader)

    def generate_signal(self, factor_df: pd.DataFrame, top_n: int = 10) -> list:
        if factor_df.empty:
            return []
        ranked = factor_df.sort_values("factor_value", ascending=False)
        selected = []
        for code in ranked["code"].tolist():
            if not is_board_allowed(str(code)):
                continue
            selected.append(str(code))
            if len(selected) >= top_n:
                break
        return selected


def metrics_to_weights(metrics: pd.DataFrame, method: str) -> pd.Series:
    """把因子指标转成非负归一权重。"""
    if method == "ic":
        raw = metrics["ic_mean"].astype(float)
    elif method == "ic_ir":
        raw = metrics["ic_ir"].astype(float)
    else:
        raise ValueError(f"未知静态加权: {method}")
    # 与管线一致：负向贡献置 0 后再归一
    return normalize_weights(raw.clip(lower=0.0))


def load_factor_matrix(
    neu_dir: Path,
    dates: Sequence[str],
    factor_cols: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """按日读取中性化因子截面。"""
    out: Dict[str, pd.DataFrame] = {}
    for date in dates:
        path = resolve_dated_file(neu_dir, date)
        if path is None:
            continue
        df = read_frame(path)
        if df.empty or "code" not in df.columns:
            continue
        use = [c for c in factor_cols if c in df.columns]
        if not use:
            continue
        piece = df[["code"] + use].copy()
        piece["code"] = piece["code"].astype(str)
        out[str(date)] = piece
    return out


def scores_from_static_weights(
    day_frames: Dict[str, pd.DataFrame],
    weights: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """静态权重 → 每日 composite 信号表。"""
    cols = [c for c in weights.index.tolist() if float(weights.get(c, 0.0)) != 0.0]
    if not cols:
        cols = list(weights.index)
    w = weights.reindex(cols).fillna(0.0).astype(float)
    signals: Dict[str, pd.DataFrame] = {}
    for date, df in day_frames.items():
        use = [c for c in cols if c in df.columns]
        if not use:
            continue
        ww = w.reindex(use).fillna(0.0)
        total = float(ww.sum())
        if total <= 0:
            continue
        ww = ww / total
        score = df[use].fillna(0.0).to_numpy(dtype=float) @ ww.to_numpy(dtype=float)
        signals[date] = pd.DataFrame(
            {"code": df["code"].astype(str).values, "factor_value": score, "date": date}
        )
    return signals


def scores_from_daily_weights(
    day_frames: Dict[str, pd.DataFrame],
    daily_weights: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """逐日权重 → 信号表。"""
    signals: Dict[str, pd.DataFrame] = {}
    for date, df in day_frames.items():
        if date not in daily_weights.index:
            continue
        w = daily_weights.loc[date].astype(float)
        use = [c for c in w.index.tolist() if c in df.columns and float(w.get(c, 0.0)) != 0.0]
        if not use:
            continue
        ww = w.reindex(use).fillna(0.0)
        total = float(ww.sum())
        if total <= 0:
            continue
        ww = ww / total
        score = df[use].fillna(0.0).to_numpy(dtype=float) @ ww.to_numpy(dtype=float)
        signals[date] = pd.DataFrame(
            {"code": df["code"].astype(str).values, "factor_value": score, "date": date}
        )
    return signals


def run_bt(
    signals: Dict[str, pd.DataFrame],
    *,
    name: str,
    data_dir: str,
    top_n: int,
    rebalance: int,
    capital: float,
) -> dict:
    """含成本回测，返回关键指标。"""
    dates = sorted(signals.keys())
    if len(dates) < 5:
        return {"tag": name, "error": "signals_too_short"}
    engine = BacktestEngine(
        data_dir=data_dir,
        initial_capital=capital,
        commission_rate=0.00012,
        slippage_rate=0.001,
        stamp_duty=0.0005,
        transfer_fee_rate=0.00001,
        risk_free_rate=0.03,
    )
    strategy = StCompositeStrategy(name=name, signals=signals)
    report = engine.run(
        start_date=dates[0],
        end_date=dates[-1],
        strategy=strategy,
        top_n=top_n,
        rebalance_freq=rebalance,
        enable_cost=True,
        calculate_ic=True,
        n_groups=5,
    )
    # report 结构随引擎略异，尽量稳健抽取
    total_ret = report.get("total_return")
    if total_ret is None and "cumulative_returns" in report:
        cr = report["cumulative_returns"]
        total_ret = float(cr.iloc[-1] - 1.0) if hasattr(cr, "iloc") else None
    metrics = report.get("metrics") or report
    def pick(*keys):
        for k in keys:
            if isinstance(metrics, dict) and k in metrics:
                return metrics[k]
            if k in report:
                return report[k]
        return None

    return {
        "tag": name,
        "start": dates[0],
        "end": dates[-1],
        "n_days": len(dates),
        "total_return": pick("total_return", "总收益率"),
        "annual_return": pick("annual_return", "annualized_return"),
        "sharpe": pick("sharpe_ratio", "sharpe"),
        "max_dd": pick("max_drawdown", "max_dd"),
        "ic_mean": pick("ic_mean", "IC_mean"),
    }


def extract_report_numbers(report: dict) -> dict:
    """从 BacktestEngine.print 同源结构抽数。"""
    # engine.run 返回的是 evaluator 结果字典
    out = {}
    for k in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown", "ic_mean"):
        if k in report:
            out[k] = report[k]
    # 有的版本嵌套在 performance / summary
    for nest in ("performance", "summary", "stats"):
        block = report.get(nest)
        if isinstance(block, dict):
            for k in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown", "ic_mean"):
                if k in block and k not in out:
                    out[k] = block[k]
    return out


def backtest_quiet(
    signals: Dict[str, pd.DataFrame],
    *,
    name: str,
    data_dir: str,
    top_n: int,
    rebalance: int,
    capital: float,
) -> dict:
    import contextlib
    import io

    dates = sorted(signals.keys())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        engine = BacktestEngine(
            data_dir=data_dir,
            initial_capital=capital,
            commission_rate=0.00012,
            slippage_rate=0.001,
            stamp_duty=0.0005,
            transfer_fee_rate=0.00001,
            risk_free_rate=0.03,
        )
        strategy = StCompositeStrategy(name=name, signals=signals)
        report = engine.run(
            start_date=dates[0],
            end_date=dates[-1],
            strategy=strategy,
            top_n=top_n,
            rebalance_freq=rebalance,
            enable_cost=True,
            calculate_ic=True,
            n_groups=5,
        )
    return {
        "tag": name,
        "start": dates[0],
        "end": dates[-1],
        "n_days": len(dates),
        "total_return": report.get("total_return"),
        "annual_return": report.get("annual_return"),
        "sharpe": report.get("sharpe_ratio"),
        "max_dd": report.get("max_drawdown"),
        "ic_mean": report.get("ic_mean"),
    }


def write_signals_to_dir(signals: Dict[str, pd.DataFrame], out_dir: Path) -> int:
    """把信号写成合成分目录，供 --strategy composite_st 使用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for date, df in signals.items():
        frame = df.rename(columns={"factor_value": "composite"})[["code", "date", "composite"]]
        write_factor_frame(frame, out_dir, date)
        n += 1
    return n


def parse_args():
    p = argparse.ArgumentParser(description="短窗轨 LOO + 估权对照")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--factor-root", default="./factors")
    p.add_argument("--universe", default="factors/composite_universe.json")
    p.add_argument("--lookback", type=int, default=60, help="静态估权样本交易日数")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--rebalance", type=int, default=3)
    p.add_argument("--capital", type=float, default=50000.0)
    p.add_argument(
        "--apply-best",
        action="store_true",
        help="把夏普最优的一版写入 factors/composite_st",
    )
    p.add_argument(
        "--skip-loo",
        action="store_true",
        help="只跑估权对照，不做 LOO",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    root = Path(args.factor_root)
    neu_dir = root / "neutralized"
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    factors, meta = load_composite_universe(args.universe)
    print(f"冻结池 {len(factors)} 个: {meta.get('name')}")

    loader = DataLoader(data_dir)
    all_dates = [d for d in list_dated_stems(neu_dir) if resolve_dated_file(neu_dir, d)]
    if len(all_dates) < args.lookback + 5:
        raise RuntimeError(f"neutralized 交易日不足: {len(all_dates)}")

    # 回测覆盖：有中性化因子的全日；估权窗：末日往前 lookback
    weight_dates = all_dates[-args.lookback :]
    print(
        f"信号区间 {all_dates[0]} ~ {all_dates[-1]} (n={len(all_dates)})；"
        f"估权窗 {weight_dates[0]} ~ {weight_dates[-1]} (n={len(weight_dates)})"
    )

    weight_panel = _build_panel(neu_dir, data_dir, weight_dates, "1vwap_pct")
    if weight_panel.empty:
        raise RuntimeError("估权面板为空，检查 data_ret 与 neutralized")

    use_cols = [c for c in factors if c in weight_panel.columns]
    missing = [c for c in factors if c not in use_cols]
    if missing:
        print(f"警告: 面板缺因子 {missing}")
    if len(use_cols) < 5:
        raise RuntimeError("可用因子过少")

    day_frames = load_factor_matrix(neu_dir, all_dates, use_cols)
    print(f"可读截面 {len(day_frames)} 天")

    rows = []

    def eval_static(tag: str, cols: List[str], method: str) -> dict:
        sub = weight_panel[["date", "asset", "ret"] + cols].dropna(subset=cols, how="all")
        metrics = compute_factor_metrics(sub, cols, "ret", n_groups=10)
        w = metrics_to_weights(metrics, method).reindex(cols).fillna(0.0)
        w = normalize_weights(w)
        signals = scores_from_static_weights(day_frames, w)
        result = backtest_quiet(
            signals,
            name=tag,
            data_dir=data_dir,
            top_n=args.top_n,
            rebalance=args.rebalance,
            capital=args.capital,
        )
        result["method"] = method
        result["n_factors"] = len(cols)
        result["dropped"] = ""
        result["_signals"] = signals
        result["_weights"] = w
        return result

    # 1) 估权对照：ic_ir / ic / rolling_ic_ir
    print("\n=== 估权对照 ===")
    baseline = eval_static("w_ic_ir_lb60", use_cols, "ic_ir")
    rows.append(baseline)
    print(
        f"{baseline['tag']}: ret={baseline['total_return']} sharpe={baseline['sharpe']}"
    )

    ic_row = eval_static("w_ic_lb60", use_cols, "ic")
    rows.append(ic_row)
    print(f"{ic_row['tag']}: ret={ic_row['total_return']} sharpe={ic_row['sharpe']}")

    # 滚动需要带收益的更长面板
    full_panel = _build_panel(neu_dir, data_dir, all_dates, "1vwap_pct")
    roll_dates = sorted(full_panel["date"].astype(str).unique())
    daily_w = build_rolling_ic_ir_weights(
        full_panel,
        use_cols,
        roll_dates,
        lookback=args.lookback,
        min_history=20,
    )
    roll_signals = scores_from_daily_weights(day_frames, daily_w)
    roll_row = backtest_quiet(
        roll_signals,
        name="w_rolling_ic_ir_60",
        data_dir=data_dir,
        top_n=args.top_n,
        rebalance=args.rebalance,
        capital=args.capital,
    )
    roll_row["method"] = "rolling_ic_ir"
    roll_row["n_factors"] = len(use_cols)
    roll_row["dropped"] = ""
    roll_row["_signals"] = roll_signals
    rows.append(roll_row)
    print(f"{roll_row['tag']}: ret={roll_row['total_return']} sharpe={roll_row['sharpe']}")

    # 2) LOO（在 ic_ir + lookback60 上）
    if not args.skip_loo:
        print("\n=== LOO (ic_ir, lookback=60) ===")
        for drop in use_cols:
            cols = [c for c in use_cols if c != drop]
            tag = f"loo_drop_{drop}"
            row = eval_static(tag, cols, "ic_ir")
            row["dropped"] = drop
            rows.append(row)
            print(
                f"{tag}: ret={row['total_return']} sharpe={row['sharpe']} mdd={row['max_dd']}"
            )

    # 整理输出（去掉超大对象）
    table = []
    best = None
    best_sharpe = -1e9
    signal_bank = {}
    for r in rows:
        signal_bank[r["tag"]] = r.get("_signals")
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        table.append(rec)
        sh = rec.get("sharpe")
        if sh is not None and not (isinstance(sh, float) and np.isnan(sh)):
            if float(sh) > best_sharpe:
                best_sharpe = float(sh)
                best = rec

    df = pd.DataFrame(table)
    csv_path = out_dir / "st_track_loo_weights.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n已写 {csv_path}")

    # 排序摘要
    if "sharpe" in df.columns:
        show = df.sort_values("sharpe", ascending=False).head(12)
        print("\n=== Top by sharpe ===")
        print(show[["tag", "method", "dropped", "total_return", "sharpe", "max_dd", "n_factors"]].to_string(index=False))

    summary = {
        "lookback": args.lookback,
        "top_n": args.top_n,
        "rebalance": args.rebalance,
        "universe": meta.get("name"),
        "n_factors": len(use_cols),
        "signal_start": all_dates[0],
        "signal_end": all_dates[-1],
        "weight_start": weight_dates[0],
        "weight_end": weight_dates[-1],
        "best": best,
        "csv": str(csv_path),
    }
    summary_path = out_dir / "st_track_loo_weights_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"摘要: {summary_path}")

    if args.apply_best and best and best["tag"] in signal_bank:
        n = write_signals_to_dir(signal_bank[best["tag"]], root / "composite_st")
        # 同步一份实验快照
        snap = root / f"composite_st__{best['tag']}"
        write_signals_to_dir(signal_bank[best["tag"]], snap)
        print(f"已应用最优 {best['tag']} → factors/composite_st ({n} 天)；快照 {snap}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
