import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualize_standardize_comparison(output_dir="./outputs/standardize_compare", ic_metric: str = "pearson"):
    output_path = Path(output_dir)
    ic_df = pd.read_csv(output_path / "standardize_method_ic.csv")
    summary_df = pd.read_csv(output_path / "standardize_method_summary.csv")

    metric_col = f"{ic_metric}_ic"
    mean_col = f"{ic_metric}_mean"
    ir_col = f"{ic_metric}_ir"
    metric_label = ic_metric.title()

    if metric_col not in ic_df.columns:
        raise ValueError(f"IC metric column '{metric_col}' not found in {output_path / 'standardize_method_ic.csv'}")
    for col in (mean_col, ir_col):
        if col not in summary_df.columns:
            raise ValueError(f"Summary column '{col}' not found in {output_path / 'standardize_method_summary.csv'}")

    # 转换日期格式以便绘图
    ic_df['date'] = pd.to_datetime(ic_df['date'])
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 累积 IC 曲线 (Cumulative IC)
    ax1 = fig.add_subplot(2, 2, 1)
    for method in ic_df['method'].unique():
        method_data = ic_df[ic_df['method'] == method].sort_values('date')
        ax1.plot(method_data['date'], method_data[metric_col].cumsum(), label=method)

    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)  # 添加零轴参考线
    ax1.set_title(f"Cumulative {metric_label} IC by Method", fontsize=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"Cumulative {metric_label} IC")
    ax1.legend()
    # 优化 x 轴日期显示
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # 2. IC 分布箱线图 (IC Distribution Boxplot)
    ax2 = fig.add_subplot(2, 2, 2)
    # 修复 FutureWarning: 显式指定 hue
    sns.boxplot(x='method', y=metric_col, hue='method', data=ic_df, ax=ax2, palette="Set2", legend=False)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.3)  # 添加零轴参考线
    ax2.set_title(f"{metric_label} IC Distribution Comparison", fontsize=14)
    ax2.set_xlabel("Method")
    ax2.set_ylabel(f"{metric_label} IC")

    # 3. Mean IC 对比
    ax3 = fig.add_subplot(2, 2, 3)
    # 修复 FutureWarning: 显式指定 hue
    sns.barplot(x='method', y=mean_col, hue='method', data=summary_df, ax=ax3, palette="viridis", legend=False)
    ax3.set_title(f"Mean {metric_label} IC Comparison", fontsize=14)
    ax3.set_xlabel("Method")
    ax3.set_ylabel(f"Mean {metric_label} IC")

    # 4. IR 对比
    ax4 = fig.add_subplot(2, 2, 4)
    # 修复 FutureWarning: 显式指定 hue
    sns.barplot(x='method', y=ir_col, hue='method', data=summary_df, ax=ax4, palette="magma", legend=False)
    ax4.set_title(f"{metric_label} ICIR Comparison", fontsize=14)
    ax4.set_xlabel("Method")
    ax4.set_ylabel("IR")

    plt.tight_layout()
    plot_path = output_path / "standardize_methods_compare.png"
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

if __name__ == "__main__":
    visualize_standardize_comparison()
