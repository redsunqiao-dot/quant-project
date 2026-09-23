"""Visual dashboards for Day13 multi-factor backtest outputs (English labels)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .visualize_common import ensure_plot_dir, read_csv
except ImportError:  # pragma: no cover
    from scripts.course_viz.visualize_common import ensure_plot_dir, read_csv  # type: ignore


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "day13_multifactor"

# Stick to English-friendly fonts to avoid glyph warnings on different OS.
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = True


def _annotate_bars(ax, values: Iterable[float], fmt: str) -> None:
    for patch, value in zip(ax.patches, values):
        ax.annotate(
            fmt.format(value),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _plot_backtest_compare(df: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_ordered = df.sort_index()

    total_pct = df_ordered["total_return"].mul(100)
    total_pct.plot(kind="bar", color="steelblue", ax=axes[0])
    axes[0].set_title("Total Return vs Variant")
    axes[0].set_ylabel("Total Return (%)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    _annotate_bars(axes[0], total_pct.values, "{:.1f}%")

    ls_panel = df_ordered[["long_mean", "short_mean", "ls_mean"]].mul(10000)
    ls_panel.columns = ["Long", "Short", "Long-Short"]
    ls_panel.plot(kind="bar", ax=axes[1], color=["#2ca02c", "#d62728", "#1f77b4"])
    axes[1].set_title("Daily Return Contribution (bp)")
    axes[1].set_ylabel("Daily Return (bp)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_dir / "backtest_compare.png", dpi=150)
    plt.close(fig)


def _plot_weight_timeseries(weights: pd.DataFrame, plot_dir: Path, top_k: int = 5) -> None:
    weights = weights.copy()
    weights["date"] = pd.to_datetime(weights["date"], errors="coerce")
    factor_cols = [c for c in weights.columns if c not in {"variant", "date", "weight_turnover"}]

    # Ensure all required variants exist and handle potential NaNs
    weights_clean = weights.dropna(subset=["date"]).sort_values(["variant", "date"])
    if weights_clean.empty:
        print("[_plot_weight_timeseries] No valid data after cleaning, skipping plots.")
        return

    # Define a color map for variants - dynamically assign colors
    unique_variants = sorted(weights_clean["variant"].unique())
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    variant_colors = {v: color_palette[i % len(color_palette)] for i, v in enumerate(unique_variants)}

    # Calculate top factors across all variants for consistency
    all_factor_weights = weights_clean[factor_cols].abs().mean()
    top_factors = (
        all_factor_weights.sort_values(ascending=False)
        .head(min(top_k, len(factor_cols)))
        .index.tolist()
    )

    if not top_factors:
        print("[_plot_weight_timeseries] No factors to plot for stability.")
        return

    # --- Plot Weight Stability Comparison ---
    # Determine grid size for subplots
    num_factors = len(top_factors)
    if num_factors == 0:
        print("[_plot_weight_timeseries] No top factors found to plot.")
        return
    
    # Calculate grid dimensions: try to make it as square as possible
    nrows = int(num_factors**0.5)
    ncols = (num_factors + nrows - 1) // nrows # equivalent to ceil(num_factors / nrows)
    
    if nrows * ncols < num_factors: # Adjust if nrows * ncols is not enough
        ncols += 1

    fig_stability, axes_stability = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
    )
    axes_stability_flat = axes_stability.flatten()

    for i, factor in enumerate(top_factors):
        ax = axes_stability_flat[i]
        for variant in weights_clean["variant"].unique():
            subset = weights_clean[weights_clean["variant"] == variant]
            # Use the defined color map
            color = variant_colors.get(variant, 'black') # Default to black if variant not in map
            ax.plot(subset["date"], subset[factor], label=variant, color=color)
        ax.set_title(f"Weight Stability: {factor}")
        ax.set_ylabel("Weight")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability

    # Hide unused subplots
    for j in range(num_factors, len(axes_stability_flat)):
        fig_stability.delaxes(axes_stability_flat[j])

    fig_stability.suptitle("Factor Weight Stability Comparison (Raw vs Neutralized)", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle
    plt.savefig(plot_dir / "weights_stability_comparison.png", dpi=150)
    plt.close(fig_stability)

    # --- Plot Weight Turnover Comparison ---
    if "weight_turnover" in weights_clean.columns:
        fig_turnover, ax_turnover = plt.subplots(figsize=(10, 4))
        for variant in weights_clean["variant"].unique():
            subset = weights_clean[weights_clean["variant"] == variant]
            # Use the defined color map
            color = variant_colors.get(variant, 'black') # Default to black if variant not in map
            ax_turnover.plot(subset["date"], subset["weight_turnover"], label=variant, color=color)
        ax_turnover.set_title("Weight Turnover Comparison (Raw vs Neutralized)")
        ax_turnover.set_ylabel("Turnover")
        ax_turnover.grid(alpha=0.3)
        ax_turnover.legend(loc="best")
        ax_turnover.tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig(plot_dir / "weights_turnover_comparison.png", dpi=150)
        plt.close(fig_turnover)
    else:
        print("[_plot_weight_timeseries] 'weight_turnover' column not found, skipping turnover plot.")


def _plot_contribution_summary(contrib: pd.DataFrame, plot_dir: Path) -> None:
    pivot = contrib.pivot(index="variant", columns="factor", values="contribution_pct")
    pivot = pivot.fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Factor Contribution Share")
    ax.set_ylabel("Contribution %")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
    plt.tight_layout()
    plt.savefig(plot_dir / "factor_contribution.png", dpi=150)
    plt.close(fig)


def run_visualization(output_dir: str = str(DEFAULT_OUTPUT_DIR)) -> None:
    outputs_path = Path(output_dir)
    plot_dir = ensure_plot_dir(outputs_path, "backtest_validation")

    base_df = read_csv(outputs_path / "backtest_compare.csv")
    required = {"variant", "total_return", "ls_mean", "long_mean", "short_mean"}
    if base_df is not None and required.issubset(base_df.columns):
        summary_df = base_df.dropna(subset=["variant"]).set_index("variant")
        summary_df = summary_df[list(required - {"variant"})].astype(float)
        _plot_backtest_compare(summary_df, plot_dir)
    else:
        print("[Skip] Missing or malformed backtest_compare.csv")

    weights_df = read_csv(outputs_path / "weights_timeseries.csv")
    if weights_df is not None and {"variant", "date"}.issubset(weights_df.columns):
        _plot_weight_timeseries(weights_df, plot_dir)
    else:
        print("[Skip] weights_timeseries.csv missing required columns")

    contrib_df = read_csv(outputs_path / "contribution_summary.csv")
    if contrib_df is not None and {"variant", "factor", "contribution_pct"}.issubset(contrib_df.columns):
        contrib_df = contrib_df.copy()
        contrib_df["contribution_pct"] = contrib_df["contribution_pct"].astype(float)
        _plot_contribution_summary(contrib_df, plot_dir)
    else:
        print("[Skip] contribution_summary.csv missing required columns")

    print(f"Visualization suite saved to {plot_dir}")


if __name__ == "__main__":
    run_visualization()
