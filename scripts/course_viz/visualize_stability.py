"""Visualizations for stability_overfit_checks.py outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scripts.course_viz.visualize_common import ensure_plot_dir, read_csv


def plot_heatmap(df, title: str, path: Path) -> None:
    """Plot a heatmap for factor weights."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, cmap="coolwarm", center=0, annot=False, fmt=".2f", cbar_kws={"label": "Weight"})
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Factor", fontsize=12)
    plt.ylabel("Split", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_train_test_ic(overfit_df, plot_dir: Path) -> None:
    """Plot train vs test IC comparison to detect overfitting."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Subplot 1: IC series comparison
    ax1 = axes[0]
    ax1.plot(overfit_df["split"], overfit_df["train_ic"], marker="o", label="Train IC", linewidth=2, color="#2E86AB")
    ax1.plot(overfit_df["split"], overfit_df["test_ic"], marker="s", label="Test IC (OOS)", linewidth=2, color="#A23B72")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.fill_between(overfit_df["split"], overfit_df["train_ic"], overfit_df["test_ic"], alpha=0.2, color="red")
    ax1.set_title("Train vs Test IC - overfitting check", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Split", fontsize=11)
    ax1.set_ylabel("IC", fontsize=11)
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # Subplot 2: IC gap chart
    ax2 = axes[1]
    ic_gap = overfit_df["train_ic"] - overfit_df["test_ic"]
    colors = ["red" if gap > 0.03 else "orange" if gap > 0.02 else "green" for gap in ic_gap]
    ax2.bar(overfit_df["split"], ic_gap, color=colors, alpha=0.7)
    ax2.axhline(0.03, color="red", linestyle="--", alpha=0.7, label="Danger threshold (>0.03)")
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_title("IC gap (Train - Test) - large gap implies overfitting", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Split", fontsize=11)
    ax2.set_ylabel("IC Gap", fontsize=11)
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / "train_test_ic_comparison.png", dpi=150)
    plt.close()


def plot_backtest_metrics(overfit_df, plot_dir: Path) -> None:
    """Plot backtest metrics in multiple panels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: cumulative return comparison
    ax1 = axes[0, 0]
    ax1.plot(overfit_df["split"], overfit_df["train_return"], marker="o", label="Train Return", linewidth=2, color="#2E86AB")
    ax1.plot(overfit_df["split"], overfit_df["test_return"], marker="s", label="Test Return (OOS)", linewidth=2, color="#A23B72")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Cumulative return comparison", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Split", fontsize=10)
    ax1.set_ylabel("Cumulative Return", fontsize=10)
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # Subplot 2: out-of-sample Sharpe ratio
    ax2 = axes[0, 1]
    colors_sharpe = ["green" if s > 1.0 else "orange" if s > 0.5 else "red" for s in overfit_df["test_sharpe"]]
    ax2.bar(overfit_df["split"], overfit_df["test_sharpe"], color=colors_sharpe, alpha=0.7)
    ax2.axhline(1.0, color="green", linestyle="--", alpha=0.7, label="Good threshold (>1.0)")
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_title("Test Sharpe ratio", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Split", fontsize=10)
    ax2.set_ylabel("Sharpe Ratio", fontsize=10)
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    # Subplot 3: out-of-sample max drawdown
    ax3 = axes[1, 0]
    ax3.bar(overfit_df["split"], overfit_df["test_max_dd"] * 100, color="darkred", alpha=0.7)
    ax3.axhline(-10, color="red", linestyle="--", alpha=0.7, label="Warning level (-10%)")
    ax3.set_title("Test max drawdown", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Split", fontsize=10)
    ax3.set_ylabel("Max Drawdown (%)", fontsize=10)
    ax3.legend(loc="best")
    ax3.grid(alpha=0.3)

    # Subplot 4: IC vs return scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(overfit_df["test_ic"], overfit_df["test_return"],
                         c=overfit_df["test_sharpe"], cmap="RdYlGn", s=100, alpha=0.7)
    ax4.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax4.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_title("Test IC vs return (color = Sharpe)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Test IC", fontsize=10)
    ax4.set_ylabel("Test Return", fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Sharpe Ratio", fontsize=10)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / "backtest_metrics.png", dpi=150)
    plt.close()


def plot_weight_turnover(turnover_df, plot_dir: Path, threshold: float = 0.20) -> None:
    """Plot weight turnover and highlight risk zones."""
    fig, ax = plt.subplots(figsize=(12, 5))

    col = "turnover"
    if col not in turnover_df.columns:
        value_cols = [c for c in turnover_df.columns if c != "split"]
        if value_cols:
            col = value_cols[0]

    turnover_values = turnover_df[col].values
    splits = turnover_df["split"].values if "split" in turnover_df.columns else range(len(turnover_values))

    colors = []
    for val in turnover_values:
        if val > 0.40:
            colors.append("darkred")  # extremely risky
        elif val > threshold:
            colors.append("orange")   # warning
        else:
            colors.append("green")    # safe

    ax.bar(splits, turnover_values, color=colors, alpha=0.7)
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=2, alpha=0.8, label=f"Warning level ({threshold:.0%})")
    ax.axhline(0.40, color="red", linestyle="--", linewidth=2, alpha=0.8, label="Danger level (40%)")

    ax.set_title("Weight turnover - large swings imply instability", fontsize=13, fontweight="bold")
    ax.set_xlabel("Split", fontsize=11)
    ax.set_ylabel("Turnover Rate", fontsize=11)
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")

    avg_turnover = np.mean(turnover_values)
    ax.text(0.02, 0.95, f"Avg. turnover: {avg_turnover:.2%}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(plot_dir / "weight_turnover_annotated.png", dpi=150)
    plt.close()


def plot_weight_stability_heatmap(raw_df, smoothed_df, plot_dir: Path) -> None:
    """Plot side-by-side heatmaps for raw and smoothed weights."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw weights
    sns.heatmap(raw_df, cmap="coolwarm", center=0, annot=False, ax=axes[0],
                cbar_kws={"label": "Weight"})
    axes[0].set_title("Raw weights", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Factor", fontsize=11)
    axes[0].set_ylabel("Split", fontsize=11)

    # Smoothed weights
    sns.heatmap(smoothed_df, cmap="coolwarm", center=0, annot=False, ax=axes[1],
                cbar_kws={"label": "Weight"})
    axes[1].set_title("EWMA smoothed weights", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Factor", fontsize=11)
    axes[1].set_ylabel("Split", fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_dir / "weight_stability_comparison.png", dpi=150)
    plt.close()


def run_visualization(output_dir: str = "./outputs/day13_multifactor") -> None:
    """Run all stability visualizations."""
    outputs_path = Path(output_dir)
    plot_dir = ensure_plot_dir(outputs_path, "stability")

    print("=" * 80)
    print("Stability & overfitting visualizations")
    print("=" * 80)

    print("\n[1/5] Plotting weight heatmaps...")
    raw = read_csv(outputs_path / "weights_raw.csv", index_col="split")
    smoothed = read_csv(outputs_path / "weights_smoothed.csv", index_col="split")

    if raw is not None and not raw.empty and smoothed is not None and not smoothed.empty:
        plot_weight_stability_heatmap(raw, smoothed, plot_dir)
        print("  ✓ Weight comparison heatmap created")

    print("\n[2/5] Plotting weight turnover...")
    turnover = read_csv(outputs_path / "weight_turnover.csv")
    if turnover is not None and not turnover.empty:
        plot_weight_turnover(turnover, plot_dir, threshold=0.20)
        print("  ✓ Weight turnover chart created")

    print("\n[3/5] Plotting Train vs Test IC...")
    overfit = read_csv(outputs_path / "overfit_checks.csv")
    if overfit is not None and not overfit.empty:
        if {"train_ic", "test_ic"}.issubset(overfit.columns):
            plot_train_test_ic(overfit, plot_dir)
            print("  ✓ IC comparison chart created")

        print("\n[4/5] Plotting backtest metrics...")
        if {"train_return", "test_return", "test_sharpe", "test_max_dd"}.issubset(overfit.columns):
            plot_backtest_metrics(overfit, plot_dir)
            print("  ✓ Backtest metrics figure created")

    print("\n[5/5] Summary...")
    print("=" * 80)
    print(f"✓ All stability plots saved to: {plot_dir}")
    print("\nGenerated figures:")
    print("  • weight_stability_comparison.png - Raw vs smoothed weight comparison")
    print("  • weight_turnover_annotated.png - Turnover with risk annotations")
    print("  • train_test_ic_comparison.png - Train/Test IC comparison")
    print("  • backtest_metrics.png - Backtest metric overview")
    print("=" * 80)


if __name__ == "__main__":
    run_visualization()
