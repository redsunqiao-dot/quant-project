"""
Visualization of Factor Scoring.
因子打分结果可视化脚本。

This script reads the factor scoring results (factor_scores.csv) and generates plots.
本脚本读取因子打分结果 (factor_scores.csv) 并生成图表。
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_factor_scores(input_file: str, output_file: str):
    path = Path(input_file)
    
    if not path.exists():
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(path)
    
    # Sort by score just in case
    df = df.sort_values("score", ascending=False)
    
    # Calculate weighted components for the score decomposition
    # Score = 0.4 * IC_IR + 0.2 * IC_Mean + 0.2 * Mono + 0.2 * (1 - Turnover)
    # Note: If IC_IR or others are negative, the contribution is negative.
    
    # Handle NaNs
    df = df.fillna(0)
    
    # We'll visualize the top N factors (e.g., top 10)
    top_n = 10
    plot_df = df.head(top_n).copy()
    
    # Calculate components
    # Using the formula from factor_scoring.py
    plot_df["comp_ir"] = 0.4 * plot_df["ic_ir"]
    plot_df["comp_ic"] = 0.2 * plot_df["ic_mean"]
    plot_df["comp_mono"] = 0.2 * plot_df["monotonicity_mean"]
    plot_df["comp_stability"] = 0.2 * (1 - plot_df["turnover_mean"])
    
    factors = plot_df["factor"].astype(str).tolist()
    
    # Setup Plot
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    # Subplot 1: Stacked Bar Chart of Score Components
    ax1 = fig.add_subplot(gs[0])
    
    y_pos = np.arange(len(factors))
    
    # Plotting horizontal bars
    # We need to handle negative values carefully in stacked bars, but for simplicity:
    # We just stack them. Matplotlib stacks positive on positive, negative on negative.
    # If we have mixed, it gets messy. 
    # For now, let's assume they are mostly positive or just plot them as grouped bars?
    # Stacked is better for "total score".
    
    p1 = ax1.barh(y_pos, plot_df["comp_ir"], label='IC IR (40%)', color='skyblue')
    p2 = ax1.barh(y_pos, plot_df["comp_ic"], left=plot_df["comp_ir"], label='IC Mean (20%)', color='orange')
    p3 = ax1.barh(y_pos, plot_df["comp_mono"], left=plot_df["comp_ir"] + plot_df["comp_ic"], label='Monotonicity (20%)', color='lightgreen')
    
    # The stability component (1 - turnover) is usually positive (0 to 1).
    left_stab = plot_df["comp_ir"] + plot_df["comp_ic"] + plot_df["comp_mono"]
    p4 = ax1.barh(y_pos, plot_df["comp_stability"], left=left_stab, label='Stability (1-Turnover) (20%)', color='purple')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(factors)
    ax1.invert_yaxis() # Top factor at top
    ax1.set_xlabel("Weighted Score Contribution")
    ax1.set_title(f"Factor Score Decomposition (Top {len(factors)})", fontsize=14)
    ax1.legend()
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Add total score text at the end of bars
    for i, score in enumerate(plot_df["score"]):
        ax1.text(score + 0.02, i, f"{score:.3f}", va='center', fontweight='bold')

    # Subplot 2: Radar Chart (Spider Web) for the Top 1 Factor
    # Comparing metrics: IC Mean, IC IR, Monotonicity, Stability (1-Turnover)
    # We need to normalize them or just show raw values on different axes (complex).
    # Simple radar: Normalize each metric across the top factors? 
    # Or just plot raw values if they are somewhat comparable (0-1 range).
    # IC is usually small (<0.1), IR can be > 1, Mono ~ 0-1, Stability ~ 0-1.
    # Scales are different.
    
    # Let's just create a text table for the top factor details instead of a misleading radar.
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    if not plot_df.empty:
        top_factor = plot_df.iloc[0]
        text_str = f"Top Factor: {top_factor['factor']}\n\n"
        text_str += f"Total Score: {top_factor['score']:.4f}\n"
        text_str += "-" * 30 + "\n"
        text_str += f"IC Mean:      {top_factor['ic_mean']:.4f}\n"
        text_str += f"IC Std:       {top_factor['ic_std']:.4f}\n"
        text_str += f"IC IR:        {top_factor['ic_ir']:.4f}\n"
        text_str += f"Win Rate:     {top_factor['ic_win_rate']:.2%}\n"
        text_str += "-" * 30 + "\n"
        text_str += f"Monotonicity: {top_factor['monotonicity_mean']:.4f}\n"
        text_str += f"Turnover:     {top_factor['turnover_mean']:.4f}\n"
        text_str += f"Stability:    {(1 - top_factor['turnover_mean']):.4f}\n"
        
        ax2.text(0.1, 0.9, text_str, fontsize=12, family='monospace', va='top')
        ax2.set_title("Top Factor Details", fontsize=14)

    plt.tight_layout()
    
    # Save
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    plot_factor_scores(
        input_file="./outputs/day12_scores/factor_scores.csv",
        output_file="./outputs/day12_scores/factor_score_plot.png"
    )
