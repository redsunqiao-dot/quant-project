"""
IC 衰减可视化脚本

功能：
读取 ic_decay.py 生成的 ic_decay_summary.csv 和 ic_decay.csv，
绘制以下图表：
1. IC 衰减柱状图 (Bar Chart): 直观展示不同持有期的 IC 均值。
2. IC 长期走势对比 (Line Chart): 观察不同持有期 IC 随时间的变化趋势。

输入文件 (位于 outputs/ic_decay):
- ic_decay_summary.csv
- ic_decay.csv

输出图片:
- ic_decay_bar.png
- ic_decay_timeseries.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_ic_decay(output_dir: str):
    out_path = Path(output_dir)
    summary_file = out_path / "ic_decay_summary.csv"
    detail_file = out_path / "ic_decay.csv"
    
    # 1. 绘制 IC 衰减柱状图 (基于 Summary)
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        
        # 确保按持有期排序 (这里假设 horizon 命名规则为 '1vwap_pct', '5vwap_pct'...)
        # 提取数字进行排序
        summary_df['days'] = summary_df['horizon'].str.extract('(\d+)').astype(int)
        summary_df = summary_df.sort_values('days')
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(summary_df['horizon'], summary_df['mean'], color='teal', alpha=0.7, edgecolor='black')
        
        # 在柱子上标注数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
            
        plt.title("IC Decay: Mean IC by Holding Period")
        plt.ylabel("Mean IC")
        plt.xlabel("Holding Period (Horizon)")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(out_path / "ic_decay_bar.png", dpi=150)
        plt.close()
        print(f"Saved Bar plot to {out_path / 'ic_decay_bar.png'}")
    else:
        print(f"Summary file not found: {summary_file}")

    # 2. 绘制 IC 长期走势对比 (基于 Detail)
    if detail_file.exists():
        detail_df = pd.read_csv(detail_file, parse_dates=['date'])
        
        plt.figure(figsize=(12, 6))
        
        # 获取所有 horizon
        horizons = detail_df['horizon'].unique()
        
        # 为了排序，同样提取天数
        horizons = sorted(horizons, key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        # 定义不同持有期的颜色和线型
        colors = ['dodgerblue', 'orange', 'green', 'red']
        linestyles = ['-', '--', '-.', ':']
        
        for i, horizon in enumerate(horizons):
            sub_df = detail_df[detail_df['horizon'] == horizon].sort_values('date')
            # 绘制 20 日均线使其更清晰，原始数据太乱
            rolling_ic = sub_df.set_index('date')['ic'].rolling(20).mean()
            
            color = colors[i % len(colors)]
            style = linestyles[i % len(linestyles)]
            
            plt.plot(rolling_ic.index, rolling_ic, label=f"{horizon} (MA20)", color=color, linestyle=style, linewidth=1.5)
            
        plt.title("IC Decay Time Series (20-day Moving Average)")
        plt.ylabel("IC (Rolling Mean)")
        plt.xlabel("Date")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        
        plt.savefig(out_path / "ic_decay_timeseries.png", dpi=150)
        plt.close()
        print(f"Saved Time Series plot to {out_path / 'ic_decay_timeseries.png'}")
    else:
        print(f"Detail file not found: {detail_file}")

if __name__ == "__main__":
    output_dir = "outputs/ic_decay"
    # 假设在 code 目录下运行，需要调整路径
    if Path("code").exists(): # 如果在根目录
        output_dir = "code/outputs/ic_decay"
    
    if Path(output_dir).exists():
        plot_ic_decay(output_dir)
    else:
        # 尝试在当前目录下找
        if Path("outputs/ic_decay").exists():
             plot_ic_decay("outputs/ic_decay")
        else:
             print("Output directory not found.")
