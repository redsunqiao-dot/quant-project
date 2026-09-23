"""
Visualization of IC Statistics Summary.
IC 统计指标汇总可视化脚本。

This script reads the IC summary output (ic_summary.csv) and generates comparison charts
for ALL numerical metrics found in the file.
本脚本读取 IC 统计汇总输出 (ic_summary.csv) 并为文件中所有数值型指标生成对比柱状图。
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from pathlib import Path

def generate_dummy_summary():
    """Generate dummy IC summary data if the real file is missing."""
    print("Generating dummy summary data for visualization...")
    return pd.DataFrame([
        {
            "metric": "ic_spearman",
            "ic_mean": 0.052,
            "ic_std": 0.105,
            "ic_ir": 0.495,
            "ic_win_rate": 0.68,
            "ic_t": 2.5,
            "n": 252
        },
        {
            "metric": "ic_pearson",
            "ic_mean": 0.045,
            "ic_std": 0.110,
            "ic_ir": 0.409,
            "ic_win_rate": 0.62,
            "ic_t": 2.1,
            "n": 252
        }
    ])

def plot_ic_stats(input_file: str, output_file: str):
    path = Path(input_file)
    
    if not path.exists():
        print(f"Warning: {input_file} not found.")
        df = generate_dummy_summary()
    else:
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(path)

    if df.empty:
        print("Error: Empty data.")
        return

    # Identify numerical columns to plot (excluding 'metric' or non-numeric)
    # We want to plot every statistic calculated.
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude columns that might not be useful to plot as bars if they are IDs (none here usually)
    # But keep 'n' just in case the user wants to see sample size consistency.
    
    num_plots = len(numeric_cols)
    if num_plots == 0:
        print("No numerical columns to plot.")
        return

    # Calculate grid size (approx square, prefer 2 or 3 columns)
    cols = 2
    rows = math.ceil(num_plots / cols)
    
    # Setup Plot
    fig = plt.figure(figsize=(15, 5 * rows))
    
    # Define a color palette
    colors = ['skyblue', 'lightgreen', 'salmon', 'wheat', 'plum', 'lightblue']
    
    for idx, col_name in enumerate(numeric_cols):
        ax = fig.add_subplot(rows, cols, idx + 1)
        
        # Plot bars
        # We use the 'metric' column as the x-axis labels (e.g. ic_spearman, ic_pearson)
        x_labels = df["metric"].astype(str)
        y_values = df[col_name]
        
        # Choose color cyclically
        bar_color = colors[idx % len(colors)]
        
        bars = ax.bar(x_labels, y_values, color=bar_color, edgecolor='black', alpha=0.7)
        
        ax.set_title(f"Comparison: {col_name}", fontsize=14)
        ax.set_ylabel(col_name)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add a zero line if values can be negative
        if y_values.min() < 0 or col_name in ['ic_mean', 'ic_ir', 'ic_t']:
             ax.axhline(0, color='black', linewidth=0.8)
             
        # Add special reference lines for specific known metrics
        if col_name == "ic_ir":
             ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Threshold 0.5')
             ax.legend()
        elif col_name == "ic_win_rate":
             ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='50%')
             ax.legend()
             ax.set_ylim(0, 1.05) # Win rate is usually 0-1

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            # formatting: integer for 'n', float for others
            if col_name == 'n':
                label_text = f'{int(height)}'
            elif abs(height) < 0.001 and height != 0:
                label_text = f'{height:.1e}'
            else:
                label_text = f'{height:.4f}'
                
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label_text,
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=10, fontweight='bold')

    # Layout adjustment
    plt.tight_layout()
    
    # Save
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    plot_ic_stats(
        input_file="./outputs/day12_ic/ic_summary.csv",
        output_file="./outputs/day12_ic/ic_stats_visualization.png"
    )
