"""Visualizations for linear_model_weights.py outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.course_viz.visualize_common import ensure_plot_dir, read_csv


def run_visualization(output_dir: str = "./outputs/day13_multifactor") -> None:
    outputs_path = Path(output_dir)
    plot_dir = ensure_plot_dir(outputs_path, "linear_models")

    weights = read_csv(outputs_path / "weights_linear.csv")
    if weights is None or not {"factor", "method", "weight"}.issubset(weights.columns):
        print("[Skip] weights_linear.csv is missing or malformed")
        return

    pivot = weights.pivot(index="factor", columns="method", values="weight")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Linear Model Weights (Ridge vs Lasso)")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Normalized Weight")
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_dir / "linear_model_weights.png")
    plt.close()
    print(f"Linear model plots saved to {plot_dir}")


if __name__ == "__main__":
    run_visualization()
