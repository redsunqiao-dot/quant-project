"""
Visualization of Factor Timing.
因子择时可视化脚本。

This script reads the Factor Timing output (ic_timing.csv) and generates plots.
本脚本读取因子择时输出 (ic_timing.csv) 并生成图表。
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_factor_timing(input_file: str, output_file: str):
    path = Path(input_file)
    
    if not path.exists():
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(path)

    # Ensure date is string or datetime
    df["date"] = df["date"].astype(str)
    df["dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("dt")

    # Drop NaNs for plotting rolling metrics (or fill them, but dropping is cleaner for lines)
    # Actually, matplotlib handles NaNs by breaking the line, which is fine.
    
    # Setup Plot
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # Subplot 1: IC Series with Rolling Mean & Factor On/Off Background
    ax1 = fig.add_subplot(gs[0])
    
    # Plot Factor On regions
    # We want to shade the background green where factor_on == 1
    # We can use fill_between
    # Create a boolean mask for where factor is on
    is_on = df["factor_on"] == 1
    
    # To make the fill continuous, we can use transforms or loop through segments.
    # A simple way is fill_between with the x-axis limits or using the data points.
    # Since it's daily data, fill_between works well.
    ylim = (df["ic"].min(), df["ic"].max())
    # Expand ylim slightly
    ylim = (ylim[0] - 0.05, ylim[1] + 0.05)
    
    # Using fill_between with 'where' clause
    ax1.fill_between(df["dt"], ylim[0], ylim[1], where=is_on, 
                     color='green', alpha=0.1, label='Factor Active (On)', transform=ax1.get_xaxis_transform())

    ax1.bar(df["dt"], df["ic"], color='gray', alpha=0.4, label='Daily IC', width=1.0)
    ax1.plot(df["dt"], df["ic_roll_mean"], color='blue', linewidth=2, label='Rolling Mean IC (20D)')
    
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_title("Factor Timing: IC Series & Active Periods", fontsize=14)
    ax1.set_ylabel("IC Value")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    # Set limits explicitly if fill_between messed them up, but usually it's fine if we used data points
    
    # Subplot 2: Rolling IR
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df["dt"], df["ic_roll_ir"], color='purple', linewidth=2, label='Rolling IR (20D)')
    ax2.axhline(0.2, color='red', linestyle='--', label='Threshold (0.2)')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Shade the area where IR > 0.2
    # ax2.fill_between(df["dt"], 0.2, df["ic_roll_ir"], where=(df["ic_roll_ir"] > 0.2), color='purple', alpha=0.2)

    ax2.set_title("Rolling Information Ratio (IR)", fontsize=14)
    ax2.set_ylabel("IR")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Cumulative IC (Filtered vs Unfiltered)
    # We can calculate what the cumulative IC would be if we only traded when factor_on == 1
    # Assuming we hold the portfolio for 1 period, the return is proportional to IC.
    # So we sum IC where factor_on is 1.
    
    df["ic_filtered"] = df["ic"] * df["factor_on"]
    df["cum_ic_raw"] = df["ic"].fillna(0).cumsum()
    df["cum_ic_filtered"] = df["ic_filtered"].fillna(0).cumsum()
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df["dt"], df["cum_ic_raw"], color='gray', linestyle='--', label='Original Cumulative IC')
    ax3.plot(df["dt"], df["cum_ic_filtered"], color='green', linewidth=2, label='Timing Filtered Cumulative IC')
    
    ax3.set_title("Cumulative IC Performance (Timing Effect)", fontsize=14)
    ax3.set_ylabel("Cumulative IC")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    plot_factor_timing(
        input_file="./outputs/day12_ic/ic_timing.csv",
        output_file="./outputs/day12_ic/factor_timing_plot.png"
    )
