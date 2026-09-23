"""
约束条件与权重变化的可视化 (Visualization of Constraints and Weight Changes)。

知识点 (Knowledge Points):
1.  **权重演变全程 (Weight Evolution)**: 展示了因子得分如何一步步变成最终的投资组合权重。
2.  **关键步骤对比 (Key Steps Comparison)**:
    *   **Raw**: 原始因子 IC 预测得分（可能包含负值，和不为 1）。
    *   **Simplex**: 经过单纯形投影，满足非负且和为 1 的基本权重约束。
    *   **Capped**: 经过封顶处理（例如单因子 < 30%），防止权重过度集中于单一因子。
    *   **Turnover Smoothed**: 经过换手率控制，与旧权重平滑，降低交易成本。
3.  **可视化意义**: 帮助检查约束条件是否被正确执行，以及每一步约束对权重的具体影响幅度（如封顶是否削减了过多权重，平滑是否导致信号滞后）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.course_viz.visualize_common import ensure_plot_dir, read_csv


def run_visualization(output_dir: str = "./outputs/day13_multifactor") -> None:
    outputs_path = Path(output_dir)
    plot_dir = ensure_plot_dir(outputs_path, "constraints")

    df = read_csv(outputs_path / "weights_constraints.csv")
    if df is None or "factor" not in df.columns:
        print("[Skip] weights_constraints.csv is missing or malformed")
        return

    value_cols = [c for c in df.columns if c != "factor"]
    if not value_cols:
        print("[Skip] No constraint columns to plot")
        return

    ax = df.set_index("factor")[value_cols].plot(kind="bar", figsize=(12, 6))
    ax.set_title("Constraint Processing of Factor Weights")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Weight")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_dir / "constraints_layers.png")
    plt.close()
    print(f"Constraint plots saved to {plot_dir}")


if __name__ == "__main__":
    run_visualization()
