
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_filter_summary(summary_path, output_image):
    if not Path(summary_path).exists():
        print(f"Error: {summary_path} not found. Please run universe_filter.py first.")
        return

    df = pd.read_csv(summary_path)
    # 过滤掉缺失数据的日期
    df = df[df['status'] == 'ok'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制股票数量
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Count', color='tab:blue')
    ax1.plot(df['date'], df['before'], label='Before Filter', color='tab:blue', alpha=0.3, linestyle='--')
    ax1.plot(df['date'], df['after'], label='After Filter', color='tab:blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 绘制剔除率 (Secondary Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Drop Rate (%)', color='tab:red')
    ax2.fill_between(df['date'], df['drop_rate'] * 100, color='tab:red', alpha=0.2, label='Drop Rate')
    ax2.plot(df['date'], df['drop_rate'] * 100, color='tab:red', linewidth=1)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, max(df['drop_rate'] * 100) * 1.2 if not df.empty else 100)

    plt.title('Universe Filter Impact Analysis')
    fig.tight_layout()
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.savefig(output_image)
    print(f"📈 Visualization saved to: {output_image}")
    
    # 打印简要统计
    print("\n=== Filter Statistics Summary ===")
    print(f"Average Stocks Before: {df['before'].mean():.1f}")
    print(f"Average Stocks After:  {df['after'].mean():.1f}")
    print(f"Average Drop Rate:     {df['drop_rate'].mean()*100:.2f}%")
    print(f"Max Drop Rate:         {df['drop_rate'].max()*100:.2f}% on {df.loc[df['drop_rate'].idxmax(), 'date'].date()}")

if __name__ == "__main__":
    SUMMARY_CSV = "factors/filtered/universe_filter_summary.csv"
    OUTPUT_PLOT = "universe_filter_plot.png"
    plot_filter_summary(SUMMARY_CSV, OUTPUT_PLOT)
