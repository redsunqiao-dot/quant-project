import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# 设置中文字体支持（根据系统环境可能需要调整，这里使用默认无衬线字体，若有中文显示问题可提示用户）
plt.rcParams['axes.unicode_minus'] = False 

def load_data(date, raw_dir, neutral_dir, industry_dir):
    # 1. 读取原始因子
    raw_path = Path(raw_dir) / f"{date}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    df_raw = pd.read_csv(raw_path).set_index("code")
    # 假设因子列是除 date 之外的第一列
    factor_col = [c for c in df_raw.columns if c not in ['date', 'datetime', 'industry']][0]
    df_raw = df_raw[[factor_col]].rename(columns={factor_col: 'raw_value'})

    # 2. 读取中性化后因子
    neu_path = Path(neutral_dir) / f"{date}.csv"
    if not neu_path.exists():
        raise FileNotFoundError(f"Neutralized file not found: {neu_path}")
    df_neu = pd.read_csv(neu_path).set_index("code")
    # 假设列名相同或位置相同
    df_neu = df_neu[[factor_col]].rename(columns={factor_col: 'neutral_value'})

    # 3. 读取行业数据
    ind_path = Path(industry_dir) / f"{date}.csv"
    if not ind_path.exists():
        raise FileNotFoundError(f"Industry file not found: {ind_path}")
    df_ind = pd.read_csv(ind_path).set_index("code")
    industry_col = df_ind.columns[0] # 假设第一列是行业
    df_ind = df_ind.rename(columns={industry_col: 'industry'})

    # 4. 合并数据
    merged = df_raw.join(df_neu, how='inner').join(df_ind, how='inner')
    return merged, factor_col

def visualize_industry_neutralize(
    date="2020-02-07",
    raw_dir="./factors/preprocessed",
    neutral_dir="./factors/industry_neutralized",
    industry_dir="./data/data_industry",
    output_dir="./outputs/industry_neutralize"
):
    try:
        df, factor_name = load_data(date, raw_dir, neutral_dir, industry_dir)
    except FileNotFoundError as e:
        print(e)
        return

    # 过滤掉行业样本过少的行业，避免绘图混乱
    ind_counts = df['industry'].value_counts()
    valid_inds = ind_counts[ind_counts > 5].index
    df = df[df['industry'].isin(valid_inds)]
    
    # 按原始因子的中位数对行业进行排序，使图表更美观
    ind_order = df.groupby('industry')['raw_value'].median().sort_values().index

    sns.set_theme(style="whitegrid")
    # 创建画布
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
    
    # 1. 处理前 (Raw)
    sns.boxplot(
        x='industry', 
        y='raw_value', 
        data=df, 
        order=ind_order, 
        ax=axes[0], 
        palette="coolwarm",
        hue='industry',
        legend=False
    )
    axes[0].set_title(f"Before Neutralization: Raw Factor Distribution by Industry ({date})", fontsize=14)
    axes[0].set_ylabel("Factor Value")
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # 2. 处理后 (Neutralized)
    sns.boxplot(
        x='industry', 
        y='neutral_value', 
        data=df, 
        order=ind_order, 
        ax=axes[1], 
        palette="coolwarm",
        hue='industry',
        legend=False
    )
    axes[1].set_title(f"After Neutralization: Industry Neutralized ({date})", fontsize=14)
    axes[1].set_ylabel("Standardized Value (Z-Score)")
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # 调整 X 轴标签
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Industry")

    # 保存图片
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / "industry_neutralize_compare.png"
    plt.tight_layout()
    plt.savefig(save_file)
    print(f"Visualization saved to {save_file}")

    # 打印简单的统计信息
    print("\nSummary Statistics (Mean per Industry):")
    print("-" * 50)
    print(f"{ 'Industry':<15} | { 'Raw Mean':>10} | { 'Neutral Mean':>12}")
    print("-" * 50)
    
    summary = df.groupby('industry')[['raw_value', 'neutral_value']].mean()
    # 展示偏离最大的前5个行业（原始）
    for ind in summary['raw_value'].abs().sort_values(ascending=False).head(5).index:
         print(f"{ind:<15} | {summary.loc[ind, 'raw_value']:10.4f} | {summary.loc[ind, 'neutral_value']:12.4f}")
    print("...")

if __name__ == "__main__":
    visualize_industry_neutralize()
