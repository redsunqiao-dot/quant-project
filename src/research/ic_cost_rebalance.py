"""
IC 衰减 × 调仓成本对照（第九课模块 2/3）。

对每个因子、每种调仓间隔：
1. 看对应持有期 IC（1/5/10 日）
2. 模拟 Top-N 调仓换手
3. 用当前默认费率估年化成本拖累
4. 用分组多空毛收益减成本，判断是否过成本门槛

输出：
outputs/factor_research/ic_cost_rebalance/
  - ic_by_horizon.csv
  - rebalance_cost_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.common.utils import (
    ensure_dir,
    infer_factor_cols,
    list_dated_stems,
    load_factor_series,
    load_return_series,
    read_frame,
    resolve_dated_file,
)
from src.data.data_loader import DataLoader


# 默认费率与回测引擎一致
DEFAULT_COMMISSION = 0.00012
DEFAULT_SLIPPAGE = 0.001
DEFAULT_STAMP = 0.0005

# 调仓间隔 -> 收益列
FREQ_RET_COL = {
    1: "1vwap_pct",
    5: "5vwap_pct",
    10: "10vwap_pct",
}


def buy_cost_rate(
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
) -> float:
    return commission + slippage


def sell_cost_rate(
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    stamp: float = DEFAULT_STAMP,
) -> float:
    return commission + slippage + stamp


def round_trip_rate(
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    stamp: float = DEFAULT_STAMP,
) -> float:
    """一次完整双边换手（卖旧买新）约占净值的比例上限。"""
    return buy_cost_rate(commission, slippage) + sell_cost_rate(commission, slippage, stamp)


def _resolve_factor_dir(factor_root: Path) -> Path:
    """优先用中性化结果，其次预处理。"""
    for name in ("neutralized", "preprocessed", "filtered", "raw"):
        folder = factor_root / name
        if len(list_dated_stems(folder)) >= 20:
            return folder
    raise FileNotFoundError(f"在 {factor_root} 下找不到足够的因子截面")


def calc_ic_by_horizon(
    factor_dir: Path,
    data_dir: Path,
    dates: Sequence[str],
    factor_cols: Sequence[str],
    horizons: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """逐日算各持有期 Spearman IC，再汇总。"""
    horizons = horizons or FREQ_RET_COL
    records = []
    for date in dates:
        factors = {}
        for col in factor_cols:
            s = load_factor_series(str(factor_dir), date, factor_col=col)
            if not s.empty:
                factors[col] = s
        if not factors:
            continue
        rets = {}
        for freq, ret_col in horizons.items():
            r = load_return_series(str(data_dir), date, ret_col=ret_col)
            if not r.empty:
                rets[freq] = r
        if not rets:
            continue
        for col, factor in factors.items():
            for freq, ret in rets.items():
                aligned = pd.concat([factor, ret], axis=1, join="inner").dropna()
                if len(aligned) < 30:
                    continue
                if aligned.iloc[:, 0].std(ddof=0) == 0 or aligned.iloc[:, 1].std(ddof=0) == 0:
                    continue
                ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
                records.append(
                    {
                        "date": date,
                        "factor": col,
                        "horizon_days": freq,
                        "ret_col": FREQ_RET_COL[freq],
                        "ic": ic,
                    }
                )
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail
    summary = (
        detail.groupby(["factor", "horizon_days", "ret_col"], as_index=False)["ic"]
        .agg(ic_mean="mean", ic_std="std", n_days="count")
    )
    summary["ic_ir"] = summary["ic_mean"] / summary["ic_std"].replace(0, np.nan)
    return summary


def _top_holdings(factor: pd.Series, top_n: int) -> List[str]:
    s = factor.dropna().sort_values(ascending=False)
    return s.head(top_n).index.astype(str).tolist()


def _turnover(old: Sequence[str], new: Sequence[str]) -> float:
    old_set, new_set = set(old), set(new)
    if not old_set:
        return 1.0
    buy = len(new_set - old_set)
    sell = len(old_set - new_set)
    return (buy + sell) / (2.0 * len(old_set))


def _group_long_short(factor: pd.Series, ret: pd.Series, n_groups: int = 10) -> float:
    """单日多空：顶组减底组收益。"""
    aligned = pd.concat([factor.rename("f"), ret.rename("r")], axis=1, join="inner").dropna()
    if len(aligned) < n_groups * 5:
        return np.nan
    try:
        aligned["g"] = pd.qcut(aligned["f"].rank(method="first"), n_groups, labels=False) + 1
    except ValueError:
        return np.nan
    means = aligned.groupby("g")["r"].mean()
    if n_groups not in means.index or 1 not in means.index:
        return np.nan
    return float(means.loc[n_groups] - means.loc[1])


def simulate_rebalance_costs(
    factor_dir: Path,
    data_dir: Path,
    dates: Sequence[str],
    factor_cols: Sequence[str],
    freqs: Sequence[int] = (1, 5, 10),
    top_n: int = 50,
    n_groups: int = 10,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    stamp: float = DEFAULT_STAMP,
) -> pd.DataFrame:
    """
    按调仓间隔模拟 Top-N 组合换手，并估算年化成本与多空净收益。
    成本近似：单次调仓成本率 ≈ turnover * (买费率 + 卖费率)
    """
    rt = round_trip_rate(commission, slippage, stamp)
    rows = []
    for col in factor_cols:
        for freq in freqs:
            ret_col = FREQ_RET_COL.get(freq)
            if ret_col is None:
                continue
            reb_dates = list(dates[::freq])
            if len(reb_dates) < 3:
                continue

            turnovers = []
            ls_list = []
            holdings: List[str] = []
            for date in reb_dates:
                factor = load_factor_series(str(factor_dir), date, factor_col=col)
                ret = load_return_series(str(data_dir), date, ret_col=ret_col)
                if factor.empty or ret.empty:
                    continue
                new_hold = _top_holdings(factor, top_n)
                if not new_hold:
                    continue
                turnovers.append(_turnover(holdings, new_hold))
                ls_list.append(_group_long_short(factor, ret, n_groups=n_groups))
                holdings = new_hold

            if not turnovers:
                continue
            avg_turn = float(np.nanmean(turnovers))
            cost_per_reb = avg_turn * rt
            ann_cost = cost_per_reb * (252.0 / freq)
            ls_mean = float(np.nanmean(ls_list)) if ls_list else np.nan
            # 持有期收益年化：一年约 252/freq 个互不重叠持有期
            ann_gross = ls_mean * (252.0 / freq) if not np.isnan(ls_mean) else np.nan
            ann_net = ann_gross - ann_cost if not np.isnan(ann_gross) else np.nan
            rows.append(
                {
                    "factor": col,
                    "rebalance_days": freq,
                    "ret_col": ret_col,
                    "n_rebalances": len(turnovers),
                    "avg_turnover": avg_turn,
                    "round_trip_rate": rt,
                    "cost_per_rebalance": cost_per_reb,
                    "ann_cost": ann_cost,
                    "ls_mean_per_hold": ls_mean,
                    "ann_gross_ls": ann_gross,
                    "ann_net_ls": ann_net,
                    "pass_cost_gate": bool(ann_net > 0) if not np.isnan(ann_net) else False,
                }
            )
    return pd.DataFrame(rows)


def run_ic_cost_study(
    data_dir: str = "./data",
    factor_root: str = "./factors",
    output_dir: str = "./outputs/factor_research/ic_cost_rebalance",
    lookback_days: int = 240,
    top_n: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """主入口：算 IC 衰减表 + 调仓成本表。"""
    data_path = Path(data_dir)
    factor_dir = _resolve_factor_dir(Path(factor_root))
    loader = DataLoader(data_dir)
    all_dates = loader.get_all_dates()
    factor_dates = list_dated_stems(factor_dir)
    dates = [d for d in all_dates if d in set(factor_dates)]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    if start_date:
        dates = [d for d in dates if d >= start_date]
    else:
        dates = dates[-lookback_days:]
    if len(dates) < 20:
        raise RuntimeError(f"可用交易日不足: {len(dates)}，目录 {factor_dir}")

    sample_path = resolve_dated_file(factor_dir, dates[-1])
    sample = read_frame(sample_path)
    if "code" in sample.columns:
        sample = sample.set_index("code")
    factor_cols = infer_factor_cols(sample)
    factor_cols = [c for c in factor_cols if c not in {"date", "index", "asset"}]
    if not factor_cols:
        raise RuntimeError("未识别到因子列")

    print(f"因子目录: {factor_dir}")
    print(f"研究区间: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 天")
    print(f"因子: {factor_cols}")
    print(
        f"费率: 买 {buy_cost_rate()*100:.3f}% / 卖 {sell_cost_rate()*100:.3f}% / "
        f"双边往返 {round_trip_rate()*100:.3f}%"
    )

    ic_summary = calc_ic_by_horizon(factor_dir, data_path, dates, factor_cols)
    cost_summary = simulate_rebalance_costs(
        factor_dir, data_path, dates, factor_cols, top_n=top_n
    )

    # 把同频 IC 并到成本表，方便对照
    if not cost_summary.empty and not ic_summary.empty:
        ic_map = ic_summary.rename(columns={"horizon_days": "rebalance_days"})
        cost_summary = cost_summary.merge(
            ic_map[["factor", "rebalance_days", "ic_mean", "ic_ir"]],
            on=["factor", "rebalance_days"],
            how="left",
        )

    out = ensure_dir(output_dir)
    ic_summary.to_csv(out / "ic_by_horizon.csv", index=False)
    cost_summary.to_csv(out / "rebalance_cost_summary.csv", index=False)

    print("\nIC 按持有期:")
    print(ic_summary.to_string(index=False))
    print("\n调仓成本 vs 多空净收益:")
    print(cost_summary.to_string(index=False))
    print(f"\n结果已写入 {out}")
    return {"ic_by_horizon": ic_summary, "rebalance_cost": cost_summary}


def parse_args():
    parser = argparse.ArgumentParser(description="IC × 调仓成本对照")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--factor-root", default="./factors")
    parser.add_argument("--output-dir", default="./outputs/factor_research/ic_cost_rebalance")
    parser.add_argument("--lookback-days", type=int, default=240)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ic_cost_study(
        data_dir=args.data_dir,
        factor_root=args.factor_root,
        output_dir=args.output_dir,
        lookback_days=args.lookback_days,
        top_n=args.top_n,
        start_date=args.start or None,
        end_date=args.end or None,
    )
