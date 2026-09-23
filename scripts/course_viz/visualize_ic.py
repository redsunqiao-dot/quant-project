"""
Visualization of IC series.
IC 序列可视化脚本。

This script reads the IC series output (ic_series.csv) and generates plots.
本脚本读取 IC 序列输出 (ic_series.csv) 并生成图表。
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def generate_dummy_data(dates):
    """Generate dummy IC data if the real file is missing."""
    print("Generating dummy data for visualization...")
    np.random.seed(42)
    n = len(dates)
    # Simulate a decent factor: positive mean IC, some noise
    ic_s = np.random.normal(0.05, 0.1, n)
    ic_p = ic_s * 0.9 + np.random.normal(0, 0.02, n)
    coverage = np.random.randint(2800, 3000, n)
    
    return pd.DataFrame({
        "date": dates,
        "ic_spearman": ic_s,
        "ic_pearson": ic_p,
        "coverage": coverage
    })

def plot_ic(input_file: str, output_file: str):
    path = Path(input_file)
    
    if not path.exists():
        print(f"Warning: {input_file} not found.")
        # Try to find dates from data/date.pkl if possible, else use dummy dates
        try:
            import pickle
            with open("data/date.pkl", "rb") as f:
                dates = pickle.load(f)
                # Just take a subset for demo
                dates = [d for d in dates if d >= "2020-01-01" and d <= "2020-12-31"]
        except:
            print("Could not load real dates, creating dummy dates.")
            dates = pd.date_range(start="2020-01-01", periods=252, freq="B").strftime("%Y%m%d").tolist()
            
        df = generate_dummy_data(dates)
    else:
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(path)

    # Ensure date is string or datetime
    df["date"] = df["date"].astype(str)
    # Parse dates for better plotting if needed, but strings work for labels
    df["dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("dt")

    # 1. Calculate Cumulative IC
    df["cum_ic_spearman"] = df["ic_spearman"].cumsum()
    df["cum_ic_pearson"] = df["ic_pearson"].cumsum()
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Subplot 1: Daily IC Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(df["dt"], df["ic_spearman"], color='skyblue', label='Rank IC (Spearman)', alpha=0.6, width=1.0)
    ax1.plot(df["dt"], df["ic_pearson"], color='green', label='Linear IC (Pearson)', linewidth=1.0, linestyle='-', alpha=0.8)
    ax1.plot(df["dt"], df["ic_spearman"].rolling(20).mean(), color='orange', label='Rank IC 20D MA', linewidth=2)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_title("Daily IC Series Comparison (Spearman vs Pearson)", fontsize=14)
    ax1.set_ylabel("IC Value")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Cumulative IC
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df["dt"], df["cum_ic_spearman"], color='red', linewidth=2, label='Cum Rank IC')
    ax2.plot(df["dt"], df["cum_ic_pearson"], color='green', linewidth=2, linestyle='--', label='Cum Pearson IC')
    ax2.set_title("Cumulative IC Comparison", fontsize=14)
    ax2.set_ylabel("Cumulative IC")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    # Add text for final values
    final_cum_s = df["cum_ic_spearman"].iloc[-1]
    final_cum_p = df["cum_ic_pearson"].iloc[-1]
    ax2.text(df["dt"].iloc[-1], final_cum_s, f"S:{final_cum_s:.2f}", ha='right', va='bottom', fontsize=10, fontweight='bold', color='red')
    ax2.text(df["dt"].iloc[-1], final_cum_p, f"P:{final_cum_p:.2f}", ha='right', va='top', fontsize=10, fontweight='bold', color='green')

    # Subplot 3: IC Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    mean_ic_s = df["ic_spearman"].mean()
    std_ic_s = df["ic_spearman"].std()
    ir_s = mean_ic_s / std_ic_s if std_ic_s != 0 else 0
    
    mean_ic_p = df["ic_pearson"].mean()
    std_ic_p = df["ic_pearson"].std()
    ir_p = mean_ic_p / std_ic_p if std_ic_p != 0 else 0

    ax3.hist(df["ic_spearman"], bins=30, color='skyblue', alpha=0.5, label=f'Spearman (IR={ir_s:.2f})', edgecolor='black')
    ax3.hist(df["ic_pearson"], bins=30, color='green', alpha=0.4, label=f'Pearson (IR={ir_p:.2f})', edgecolor='black')
    
    ax3.axvline(mean_ic_s, color='blue', linestyle='--', linewidth=1)
    ax3.axvline(mean_ic_p, color='darkgreen', linestyle='--', linewidth=1)
    
    ax3.set_title("IC Distribution Comparison", fontsize=14)
    ax3.set_xlabel("IC Value")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Layout adjustment
    plt.tight_layout()
    
    # Save
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    plot_ic(
        input_file="./outputs/day12_ic/ic_series.csv",
        output_file="./outputs/day12_ic/ic_visualization.png"
    )
