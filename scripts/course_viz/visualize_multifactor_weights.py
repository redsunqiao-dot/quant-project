"""Visualizations for multifactor_weights.py outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.course_viz.visualize_common import ensure_plot_dir, load_panel, read_csv

try:  # pragma: no cover
    from .utils import group_returns  # type: ignore
except ImportError:  # pragma: no cover
    from src.common.utils import group_returns  # type: ignore


def plot_weight_comparison(weights: pd.DataFrame, out_dir: Path) -> None:
    if "method" not in weights.columns:
        return
    ax = weights.set_index("method").T.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Factor Weight Comparison by Method")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Weight")
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "weight_comparison.png")
    plt.close()


def plot_factor_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    if "factor" not in metrics.columns or not {"ic_mean", "ic_ir"}.issubset(metrics.columns):
        return
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    metrics.plot(
        x="factor",
        y="ic_mean",
        kind="bar",
        color="skyblue",
        ax=ax1,
        width=0.4,
        position=1,
        label="IC Mean",
    )
    metrics.plot(
        x="factor",
        y="ic_ir",
        kind="bar",
        color="orange",
        ax=ax2,
        width=0.4,
        position=0,
        label="IC IR",
    )
    ax1.set_ylabel("IC Mean", color="skyblue")
    ax2.set_ylabel("IC IR", color="orange")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Factor Performance Metrics")
    plt.tight_layout()
    plt.savefig(out_dir / "factor_metrics.png")
    plt.close()


def plot_factor_correlation(panel: pd.DataFrame, factor_cols: list[str], out_dir: Path) -> None:
    if not factor_cols:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(panel[factor_cols].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Factor Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "factor_correlation.png")
    plt.close()


def plot_group_monotonicity(panel: pd.DataFrame, outputs_path: Path, out_dir: Path) -> None:
    records = []
    for method in ["equal", "ic", "ic_ir", "ret"]:
        comp = read_csv(outputs_path / f"composite_factor_{method}.csv")
        if comp is None:
            continue
        merged = pd.merge(comp, panel[["date", "asset", "ret"]], on=["date", "asset"], how="inner")
        grouped = group_returns(merged, "composite", ret_col="ret", n_groups=10)
        if grouped.empty:
            continue
        avg = grouped.mean()
        avg.name = method.upper()
        records.append(avg)

    if not records:
        return

    df = pd.concat(records, axis=1)
    ax = df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Mean Return by Quantile Group")
    ax.set_xlabel("Quantile (1=Bottom, 10=Top)")
    ax.set_ylabel("Average Daily Return")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "group_monotonicity.png")
    plt.close()


def plot_backtest_curves(panel: pd.DataFrame, outputs_path: Path, out_dir: Path) -> None:
    plt.figure(figsize=(12, 7))
    plotted = False
    for method in ["equal", "ic", "ic_ir", "ret"]:
        comp = read_csv(outputs_path / f"composite_factor_{method}.csv")
        if comp is None:
            continue
        merged = pd.merge(comp, panel[["date", "asset", "ret"]], on=["date", "asset"], how="inner")
        if merged.empty:
            continue

        def top_ret(df: pd.DataFrame) -> float:
            valid = df.dropna(subset=["composite", "ret"])
            if valid.empty:
                return np.nan
            top_n = max(1, int(len(valid) * 0.1))
            return valid.nlargest(top_n, "composite")["ret"].mean()

        daily = merged.groupby("date", group_keys=False).apply(top_ret).dropna()
        if daily.empty:
            continue
        cum = (1 + daily).cumprod()
        plt.plot(pd.to_datetime(cum.index), cum.values, label=method.upper())
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title("Cumulative Returns of Top 10% Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "backtest_comparison.png")
    plt.close()


def run_visualization(
    output_dir: str = "./outputs/day13_multifactor",
    market_data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = None,
) -> None:
    outputs_path = Path(output_dir)
    plot_dir = ensure_plot_dir(outputs_path, "multifactor_weights")
    panel, factor_cols = load_panel(
        market_data_dir=market_data_dir,
        ret_horizon=ret_horizon,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )

    weights = read_csv(outputs_path / "weights_by_method.csv")
    if weights is not None:
        plot_weight_comparison(weights, plot_dir)

    metrics = read_csv(outputs_path / "factor_metrics.csv")
    if metrics is not None:
        plot_factor_metrics(metrics, plot_dir)

    plot_factor_correlation(panel, factor_cols, plot_dir)
    plot_group_monotonicity(panel, outputs_path, plot_dir)
    plot_backtest_curves(panel, outputs_path, plot_dir)

    print(f"Multifactor weight plots saved to {plot_dir}")


if __name__ == "__main__":
    run_visualization()
