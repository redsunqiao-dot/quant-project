import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def load_data(date, factor_dir, neutral_dir, barra_dir):
    # 1. 读取原始因子
    factor_path = Path(factor_dir) / f"{date}.csv"
    if not factor_path.exists():
        raise FileNotFoundError(f"Raw factor file not found: {factor_path}")
    df_raw = pd.read_csv(factor_path).set_index("code")
    # 假设因子列是除 date 之外的第一列
    factor_col = [c for c in df_raw.columns if c not in ['date', 'datetime']][0]
    df_raw = df_raw[[factor_col]].rename(columns={factor_col: 'raw_value'})

    # 2. 读取中性化后因子
    neu_path = Path(neutral_dir) / f"{date}.csv"
    if not neu_path.exists():
        raise FileNotFoundError(f"Neutralized file not found: {neu_path}")
    df_neu = pd.read_csv(neu_path).set_index("code")
    df_neu = df_neu[[factor_col]].rename(columns={factor_col: 'neutral_value'})

    # 3. 读取 Barra 风险因子 (Size, Beta)
    barra_path = Path(barra_dir) / f"{date}.csv"
    if not barra_path.exists():
        raise FileNotFoundError(f"Barra file not found: {barra_path}")
    df_barra = pd.read_csv(barra_path).set_index("code")
    
    # 确保有 size 和 beta 列 (根据 factor_neutralization.py 的逻辑，size 可能是 mktcap 需要取对数)
    if 'size' in df_barra.columns:
        df_barra['size_log'] = df_barra['size'] # 假设已经是 size
    elif 'mktcap' in df_barra.columns:
        df_barra['size_log'] = np.log(df_barra['mktcap'].replace(0, np.nan))
    else:
        # 尝试寻找包含 size 的列
        size_cols = [c for c in df_barra.columns if 'size' in c]
        if size_cols:
             df_barra['size_log'] = df_barra[size_cols[0]]
        else:
             raise ValueError("Size column not found in Barra data")

    # 4. 合并数据
    merged = df_raw.join(df_neu, how='inner').join(df_barra[['size_log', 'beta']], how='inner').dropna()
    return merged, factor_col

def visualize_factor_neutralization(
    date="2020-02-07",
    factor_dir="./factors/preprocessed",
    neutral_dir="./factors/neutralized",
    barra_dir="./data/data_barra",
    output_dir="./outputs/factor_neutralization"
):
    try:
        df, factor_name = load_data(date, factor_dir, neutral_dir, barra_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 散点图：原始因子 vs 市值 (Size)
    ax1 = fig.add_subplot(2, 2, 1)
    sns.scatterplot(x='size_log', y='raw_value', data=df, ax=ax1, alpha=0.5, color='steelblue')
    corr_raw_size = spearmanr(df['raw_value'], df['size_log']).correlation
    ax1.set_title(f"Before: Raw Factor vs Size\nCorr: {corr_raw_size:.3f}", fontsize=12)
    ax1.set_xlabel("Log Market Cap (Size)")
    ax1.set_ylabel("Raw Factor Value")
    
    # 2. 散点图：中性化因子 vs 市值 (Size)
    ax2 = fig.add_subplot(2, 2, 2)
    sns.scatterplot(x='size_log', y='neutral_value', data=df, ax=ax2, alpha=0.5, color='darkorange')
    corr_neu_size = spearmanr(df['neutral_value'], df['size_log']).correlation
    ax2.set_title(f"After: Neutral Factor vs Size\nCorr: {corr_neu_size:.3f}", fontsize=12)
    ax2.set_xlabel("Log Market Cap (Size)")
    ax2.set_ylabel("Neutral Factor Value")

    # 3. 相关性对比条形图 (Size & Beta)
    ax3 = fig.add_subplot(2, 2, 3)
    corrs = {
        'Raw vs Size': corr_raw_size,
        'Neu vs Size': corr_neu_size,
        'Raw vs Beta': spearmanr(df['raw_value'], df['beta']).correlation,
        'Neu vs Beta': spearmanr(df['neutral_value'], df['beta']).correlation
    }
    bars = ax3.bar(corrs.keys(), corrs.values(), color=['steelblue', 'darkorange', 'steelblue', 'darkorange'])
    ax3.set_title("Correlation with Risk Factors (Spearman)", fontsize=12)
    ax3.set_ylabel("Correlation Coefficient")
    ax3.axhline(0, color='black', linewidth=0.8)
    
    # 在柱状图上标注数值
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", va='bottom' if yval > 0 else 'top', ha='center')

    # 4. 密度图对比
    ax4 = fig.add_subplot(2, 2, 4)
    sns.kdeplot(df['raw_value'], ax=ax4, label='Raw Factor', fill=True, color='steelblue', alpha=0.3)
    sns.kdeplot(df['neutral_value'], ax=ax4, label='Neutral Factor', fill=True, color='darkorange', alpha=0.3)
    ax4.set_title("Factor Distribution Comparison", fontsize=12)
    ax4.legend()

    plt.suptitle(f"Factor Neutralization Analysis ({factor_name} on {date})", fontsize=16)
    plt.tight_layout()
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / "neutralization_effect.png"
    plt.savefig(save_file)
    print(f"Visualization saved to {save_file}")
    print("\nCorrelation Summary:")
    print("-" * 30)
    for k, v in corrs.items():
        print(f"{k:<15}: {v:.4f}")

if __name__ == "__main__":
    visualize_factor_neutralization()
