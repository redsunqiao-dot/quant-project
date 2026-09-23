"""Visualizations for nonlinear_models_split.py outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scripts.course_viz.visualize_common import ensure_plot_dir, read_csv


def plot_time_split_strategy(split_df, plot_dir):
    """Plot visualization of the chronological split strategy."""
    if split_df is None or split_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    colors = {"train": "#3498db", "validate": "#f39c12", "test": "#e74c3c"}
    split_names = {"train": "Training", "validate": "Validation", "test": "Test"}

    y_pos = 0.5
    total_width = 0
    rects = []

    for idx, row in split_df.iterrows():
        split_type = row["split"]
        num_dates = row["num_dates"]
        color = colors.get(split_type, "#95a5a6")

        rect = mpatches.Rectangle(
            (total_width, y_pos - 0.3), num_dates, 0.6,
            facecolor=color, edgecolor="black", linewidth=2
        )
        ax.add_patch(rect)
        rects.append(rect)

        ax.text(
            total_width + num_dates / 2, y_pos,
            f"{split_names.get(split_type, split_type)}\n{num_dates} days",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white"
        )

        total_width += num_dates

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Timeline (days)", fontsize=12)
    ax.set_title("Time-series three-way split\nStrict chronological order to avoid look-ahead bias", fontsize=14, fontweight="bold")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    ax.annotate(
        "", xy=(total_width * 0.95, 0.15), xytext=(total_width * 0.05, 0.15),
        arrowprops=dict(arrowstyle="->", lw=2, color="black")
    )
    ax.text(total_width / 2, 0.08, "Time direction ->", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_dir / "time_split_strategy.png", dpi=150)
    plt.close()
    print("  ✓ Time split strategy figure saved")


def plot_rolling_window_results(rolling_df, plot_dir):
    """Plot rolling window evaluation results."""
    if rolling_df is None or rolling_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(rolling_df["window_id"], rolling_df["train_ic"], "o-", label="Train IC", linewidth=2)
    axes[0, 0].plot(rolling_df["window_id"], rolling_df["val_ic"], "s-", label="Validate IC", linewidth=2)
    axes[0, 0].plot(rolling_df["window_id"], rolling_df["test_ic"], "^-", label="Test IC", linewidth=2)
    axes[0, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Window ID")
    axes[0, 0].set_ylabel("IC (Information Coefficient)")
    axes[0, 0].set_title("Rolling window IC trend")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rolling_df["window_id"], rolling_df["train_mse"], "o-", label="Train MSE", linewidth=2)
    axes[0, 1].plot(rolling_df["window_id"], rolling_df["val_mse"], "s-", label="Validate MSE", linewidth=2)
    axes[0, 1].plot(rolling_df["window_id"], rolling_df["test_mse"], "^-", label="Test MSE", linewidth=2)
    axes[0, 1].set_xlabel("Window ID")
    axes[0, 1].set_ylabel("MSE (Mean Squared Error)")
    axes[0, 1].set_title("Rolling window MSE trend")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IC distribution comparison
    ic_data = [
        rolling_df["train_ic"].values,
        rolling_df["val_ic"].values,
        rolling_df["test_ic"].values
    ]
    bp = axes[1, 0].boxplot(ic_data, labels=["Train", "Validate", "Test"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#3498db", "#f39c12", "#e74c3c"]):
        patch.set_facecolor(color)
    axes[1, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_ylabel("IC (Information Coefficient)")
    axes[1, 0].set_title("IC distribution comparison")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].scatter(rolling_df["train_ic"], rolling_df["test_ic"], alpha=0.6, s=100)
    axes[1, 1].plot([rolling_df["train_ic"].min(), rolling_df["train_ic"].max()],
                    [rolling_df["train_ic"].min(), rolling_df["train_ic"].max()],
                    "r--", linewidth=2, label="Ideal line (no overfitting)")
    axes[1, 1].set_xlabel("Train IC")
    axes[1, 1].set_ylabel("Test IC")
    axes[1, 1].set_title("Overfitting check: Train IC vs Test IC")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / "rolling_window_results.png", dpi=150)
    plt.close()
    print("  ✓ Rolling window result figure saved")


def plot_shap_summary(shap_df, plot_dir):
    """Plot a SHAP value summary chart."""
    if shap_df is None or shap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    top_features = shap_df.head(15).sort_values("mean_abs_shap", ascending=True)

    bars = ax.barh(top_features["factor"], top_features["mean_abs_shap"], color="#e74c3c")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP feature importance\nAverage contribution of each factor", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_dir / "shap_summary.png", dpi=150)
    plt.close()
    print("  ✓ SHAP summary figure saved")


def plot_feature_importance_comparison(feature_df, shap_df, plot_dir):
    """Compare feature importance with SHAP contributions."""
    if feature_df is None or shap_df is None or feature_df.empty or shap_df.empty:
        return

    merged = feature_df.merge(shap_df[["factor", "mean_abs_shap"]], on="factor", how="inner")

    if merged.empty:
        return

    merged["importance_norm"] = merged["importance"] / merged["importance"].max()
    merged["shap_norm"] = merged["mean_abs_shap"] / merged["mean_abs_shap"].max()

    top_merged = merged.nlargest(10, "importance_norm")

    fig, ax = plt.subplots(figsize=(12, 8))

    x = range(len(top_merged))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], top_merged["importance_norm"],
                   width, label="Feature Importance", color="#3498db")
    bars2 = ax.bar([i + width/2 for i in x], top_merged["shap_norm"],
                   width, label="SHAP Value", color="#e74c3c")

    ax.set_xlabel("Factor", fontsize=12)
    ax.set_ylabel("Normalized importance", fontsize=12)
    ax.set_title("Feature importance vs SHAP value\nConsistency across methods", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(top_merged["factor"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / "importance_shap_comparison.png", dpi=150)
    plt.close()
    print("  ✓ Feature importance comparison figure saved")


def run_visualization(output_dir: str = "./outputs/day13_multifactor") -> None:
    """Run every nonlinear model visualization."""
    print("=" * 80)
    print("Nonlinear model visualizations")
    print("=" * 80)

    outputs_path = Path(output_dir)
    plot_dir = ensure_plot_dir(outputs_path, "nonlinear_models")

    print("\n[1/5] Plotting feature importance...")
    feature_df = read_csv(outputs_path / "feature_importance_nonlinear.csv")
    if feature_df is not None and {"factor", "importance"}.issubset(feature_df.columns):
        ordered = feature_df.sort_values("importance", ascending=False)
        model_name = ordered.get("model", "").iloc[0] if "model" in ordered.columns and not ordered.empty else ""

        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = ordered.head(15)
        bars = ax.barh(range(len(top_features)), top_features["importance"], color="#3498db")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["factor"])
        ax.invert_yaxis()

        title = "Nonlinear model feature importance"
        if isinstance(model_name, str) and model_name:
            title += f" ({model_name})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance score", fontsize=12)
        ax.set_ylabel("Factor", fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        # Add numeric labels for readability
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f"{width:.4f}", ha="left", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(plot_dir / "feature_importance.png", dpi=150)
        plt.close()
        print("  ✓ Feature importance figure saved")
    else:
        print("  [Skip] feature_importance_nonlinear.csv missing or invalid")

    print("\n[2/5] Plotting time split strategy...")
    split_df = read_csv(outputs_path / "time_split_info.csv")
    plot_time_split_strategy(split_df, plot_dir)

    print("\n[3/5] Plotting SHAP summary...")
    shap_df = read_csv(outputs_path / "shap_summary.csv")
    plot_shap_summary(shap_df, plot_dir)

    print("\n[4/5] Plotting feature importance comparison...")
    if feature_df is not None and shap_df is not None:
        plot_feature_importance_comparison(feature_df, shap_df, plot_dir)
    else:
        print("  [Skip] Need both feature importance and SHAP data")

    print("\n[5/5] Plotting rolling window results...")
    rolling_df = read_csv(outputs_path / "rolling_window_results.csv")
    plot_rolling_window_results(rolling_df, plot_dir)

    print(f"\n✓ All nonlinear visualizations saved to: {plot_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_visualization()
