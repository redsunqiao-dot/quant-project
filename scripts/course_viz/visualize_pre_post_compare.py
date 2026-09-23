"""
预处理前后对比可视化

功能：
读取 pre_post_compare.py 生成的对比数据 (IC, 累计收益)，
绘制图表直观展示中性化前后的效果差异。

输入文件 (位于 outputs/day10_compare):
- ic_compare.csv
- cumret_compare.csv

输出图片:
- ic_timeseries_compare.png: IC 时序波动对比
- cumret_compare.png: TopN 策略累计净值对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_comparison(output_dir: str):
    out_path = Path(output_dir)
    ic_file = out_path / "ic_compare.csv"
    cum_file = out_path / "cumret_compare.csv"
    
    if not (ic_file.exists() and cum_file.exists()):
        print(f"Error: Data files not found in {output_dir}")
        return

    # 1. 绘制 IC 时序对比
    ic_df = pd.read_csv(ic_file, index_col=0, parse_dates=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(ic_df.index, ic_df["ic_raw"], label="Raw Factor IC", alpha=0.6, color='gray', linestyle='--')
    plt.plot(ic_df.index, ic_df["ic_neutral"], label="Neutralized Factor IC", alpha=0.9, color='dodgerblue', linewidth=1.5)
    
    # 添加 20 日均线使其更清晰
    plt.plot(ic_df.index, ic_df["ic_raw"].rolling(20).mean(), label="Raw IC (MA20)", color='black', linewidth=1)
    plt.plot(ic_df.index, ic_df["ic_neutral"].rolling(20).mean(), label="Neutral IC (MA20)", color='navy', linewidth=2)
    
    plt.title("IC Time Series Comparison: Raw vs Neutralized")
    plt.ylabel("IC (Rank Correlation)")
    plt.xlabel("Date")
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    plt.savefig(out_path / "ic_timeseries_compare.png", dpi=150)
    plt.close()
    print(f"Saved IC plot to {out_path / 'ic_timeseries_compare.png'}")

    # 2. 绘制累计收益对比
    cum_df = pd.read_csv(cum_file, index_col=0, parse_dates=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_df.index, cum_df["ret_raw"], label="Raw Factor (Top 50)", color='gray', linestyle='--')
    plt.plot(cum_df.index, cum_df["ret_neutral"], label="Neutralized Factor (Top 50)", color='red', linewidth=2)
    
    plt.title("Cumulative Return Comparison (Top 50 Stocks)")
    plt.ylabel("Cumulative Net Value (Start=1)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    plt.savefig(out_path / "cumret_compare.png", dpi=150)
    plt.close()
    print(f"Saved Return plot to {out_path / 'cumret_compare.png'}")

    # 3. 绘制日收益分布对比 (基于 ret_compare.csv)
    ret_file = out_path / "ret_compare.csv"
    if ret_file.exists():
        ret_df = pd.read_csv(ret_file, index_col=0, parse_dates=True)
        
        plt.figure(figsize=(10, 6))
        # 绘制直方图
        plt.hist(ret_df["ret_raw"], bins=50, alpha=0.5, label="Raw Returns", color='gray', density=True, edgecolor='black')
        plt.hist(ret_df["ret_neutral"], bins=50, alpha=0.6, label="Neutral Returns", color='dodgerblue', density=True, edgecolor='black')
        
        # 添加核密度估计 (KDE) 曲线，使分布形状更清晰
        try:
            ret_df["ret_raw"].plot(kind='kde', color='black', linestyle='--', label='Raw KDE')
            ret_df["ret_neutral"].plot(kind='kde', color='navy', linestyle='-', label='Neutral KDE')
        except:
            pass # 如果 scipy 缺失可能报错，忽略
            
        plt.title("Daily Return Distribution: Risk & Volatility Check")
        plt.xlabel("Daily Return")
        plt.ylabel("Density (Frequency)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(ret_df.min().min() * 1.1, ret_df.max().max() * 1.1)
        
        plt.savefig(out_path / "ret_distribution_compare.png", dpi=150)
        plt.close()
        print(f"Saved Distribution plot to {out_path / 'ret_distribution_compare.png'}")

if __name__ == "__main__":
    # 假设脚本在 code/ 目录下运行，输出目录在 code/outputs/day10_compare
    output_dir = "outputs/pre_post_cmp"
    plot_comparison(output_dir)
