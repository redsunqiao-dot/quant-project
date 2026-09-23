"""
Visualization of Group Backtest & Monotonicity.
分组回测与单调性分析可视化。

This script reads the outputs from group_backtest_monotonic.py and generates plots.
本脚本读取 group_backtest_monotonic.py 的输出文件并生成图表。
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_group_analysis(output_dir: str):
    base_path = Path(output_dir)
    
    group_file = base_path / "group_return.csv"
    ls_file = base_path / "long_short.csv"
    mono_file = base_path / "monotonicity.csv"
    
    if not (group_file.exists() and ls_file.exists()):
        print(f"Error: Required files not found in {output_dir}")
        return

    print(f"Loading data from {output_dir}...")
    group_df = pd.read_csv(group_file)
    ls_df = pd.read_csv(ls_file)
    
    # Optional: Monotonicity
    if mono_file.exists():
        mono_df = pd.read_csv(mono_file)
    else:
        mono_df = None

    # Preprocessing
    group_df["date"] = pd.to_datetime(group_df["date"])
    ls_df["date"] = pd.to_datetime(ls_df["date"])
    if mono_df is not None:
        mono_df["date"] = pd.to_datetime(mono_df["date"])

    # Setup Plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])
    
    # --- Subplot 1: Cumulative Return by Group (Top, Bottom, and others in gray) ---
    ax1 = fig.add_subplot(gs[0, :])
    
    # Pivot group data to get columns as groups
    group_pivot = group_df.pivot(index="date", columns="group", values="ret")
    
    # Calculate cumulative returns: (1 + r).cumprod() - 1
    # Check if returns are in decimals (e.g., 0.01 for 1%) or percent.
    # Looking at head output provided previously: 0.015... so it's decimals.
    cum_ret = (1 + group_pivot).cumprod() - 1
    
    n_groups = group_pivot.shape[1]
    colors = plt.cm.RdYlGn(np.linspace(0, 1, n_groups))
    
    for i, col in enumerate(cum_ret.columns):
        # Determine style based on group
        if col == 1:
            label = f'Group 1 (Bottom)'
            color = 'green'
            alpha = 1.0
            lw = 2
        elif col == n_groups:
            label = f'Group {n_groups} (Top)'
            color = 'red'
            alpha = 1.0
            lw = 2
        else:
            label = None # f'Group {col}'
            color = 'gray'
            alpha = 0.3
            lw = 1
            
        ax1.plot(cum_ret.index, cum_ret[col], label=label, color=color, alpha=alpha, linewidth=lw)
    
    # Also plot Long-Short Cumulative
    ls_df = ls_df.set_index("date")
    ls_cum = (1 + ls_df["long_short"]).cumprod() - 1
    ax1.plot(ls_cum.index, ls_cum, label='Long-Short', color='blue', linestyle='--', linewidth=2)
    
    ax1.set_title(f"Cumulative Returns by Group ({n_groups} Groups)", fontsize=14)
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Average Daily Return by Group (Bar Chart) ---
    ax2 = fig.add_subplot(gs[1, 0])
    avg_ret = group_df.groupby("group")["ret"].mean()
    # Convert to bps or percentage for readability? Let's stick to raw but label well.
    # Actually, visualizing Annualized Return might be better, but mean daily is simple proxy.
    
    bars = ax2.bar(avg_ret.index, avg_ret.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add a trendline?
    z = np.polyfit(avg_ret.index, avg_ret.values, 1)
    p = np.poly1d(z)
    ax2.plot(avg_ret.index, p(avg_ret.index), "r--", alpha=0.6, label='Trend')
    
    ax2.set_title("Average Daily Return by Group (Monotonicity Check)", fontsize=14)
    ax2.set_xlabel("Group ID")
    ax2.set_ylabel("Avg Daily Return")
    ax2.set_xticks(avg_ret.index)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # --- Subplot 3: Long-Short Drawdown ---
    ax3 = fig.add_subplot(gs[1, 1])
    # Calculate Drawdown for Long-Short
    running_max = ls_cum.cummax()
    drawdown = (ls_cum - running_max) # Simple arithmetic drawdown for cum returns closer to 0? 
    # Or proper (1+cum)/(1+max) - 1. 
    # Let's use (NAV - Peak) / Peak. NAV = 1 + cum_ret
    nav = 1 + ls_cum
    dd = (nav - nav.cummax()) / nav.cummax()
    
    ax3.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
    ax3.plot(dd.index, dd, color='red', linewidth=1)
    ax3.set_title("Long-Short Strategy Drawdown", fontsize=14)
    ax3.set_ylabel("Drawdown")
    ax3.grid(True, alpha=0.3)
    
    # --- Subplot 4: Monotonicity Score (Spearman Rho) ---
    if mono_df is not None:
        ax4 = fig.add_subplot(gs[2, :])
        ax4.bar(mono_df["date"], mono_df["monotonicity"], color='orange', alpha=0.5, label='Daily Spearman Corr', width=1.0)
        # Rolling average
        roll_mono = mono_df.set_index("date")["monotonicity"].rolling(20).mean()
        ax4.plot(roll_mono.index, roll_mono, color='brown', linewidth=2, label='20D Rolling Mean')
        
        ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax4.set_title("Daily Monotonicity Score (Spearman Correlation of Group vs Return)", fontsize=14)
        ax4.set_ylabel("Spearman Correlation")
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save
    out_file = base_path / "group_backtest_plot.png"
    plt.savefig(out_file)
    print(f"Visualization saved to {out_file}")

if __name__ == "__main__":
    plot_group_analysis(output_dir="./outputs/day12_group")
