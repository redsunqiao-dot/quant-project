"""Visualize outputs from optimization_objectives.py."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

from scripts.course_viz.visualize_common import ensure_plot_dir, read_csv


def run_visualization(
    output_dir: str = "./outputs/day13_multifactor",
    start_date: Optional[str] = None,  # unused, keep signature consistency
    end_date: Optional[str] = None,
) -> None:
    outputs_path = Path(output_dir)
    plot_dir = ensure_plot_dir(outputs_path, "optimization_objectives")

    scores = read_csv(outputs_path / "objective_scores.csv")
    if scores is None or "factor" not in scores.columns:
        print("[Skip] objective_scores.csv is missing or malformed")
        return

    metric_cols = [c for c in ["ic_ir", "score_return_adj", "score_mix"] if c in scores.columns]
    if not metric_cols:
        print("[Skip] No score columns to plot")
        return

    melted = scores.melt(id_vars="factor", value_vars=metric_cols, var_name="metric", value_name="score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="factor", y="score", hue="metric")
    plt.title("Optimization Objective Scores")
    plt.xlabel("Factor")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(plot_dir / "objective_scores.png")
    plt.close()
    print(f"Objective score plots saved to {plot_dir}")


if __name__ == "__main__":
    run_visualization()
