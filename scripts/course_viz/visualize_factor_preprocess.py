
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.research.factor_preprocess import FactorPreprocessor

def visualize_preprocessing(date, factor_dir, industry_dir, output_image):
    # Paths
    factor_file = Path(factor_dir) / f"{date}.csv"
    industry_file = Path(industry_dir) / f"{date}.csv"
    
    if not factor_file.exists():
        print(f"Error: Factor file {factor_file} not found.")
        return

    # Setup
    fp = FactorPreprocessor(
        fill_method="industry_median",
        standardize="zscore",
        winsorize=True,
        n_sigma=3.0
    )
    
    df = fp.load_factor_file(factor_file)
    factor_cols = fp.infer_factor_cols(df)
    if not factor_cols:
        print("No factor columns found.")
        return
    
    target = factor_cols[0]
    series = df[target].replace([np.inf, -np.inf], np.nan)
    
    industry_series = None
    if industry_file.exists():
        # Try to find industry column dynamically if needed, or assume sw_l1 or similar
        ind_df = pd.read_csv(industry_file)
        # Simple heuristic to find the industry column (often not 'code' or 'date')
        possible_cols = [c for c in ind_df.columns if c not in ['code', 'date', 'Unnamed: 0']]
        if possible_cols:
            ind_col = possible_cols[0]
            if "code" in ind_df.columns:
                ind_df = ind_df.set_index("code")
            industry_series = ind_df[ind_col]
    
    # Process Steps
    # 1. Winsorize
    s_win = fp.winsorize_series(series, n_sigma=3.0)
    
    # 2. Fill
    s_fill = fp.fill_missing_series(s_win, method="industry_median", industry_series=industry_series)
    
    # 3. Standardize
    s_std = fp.standardize_series(s_fill, method="zscore")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Factor Preprocessing Evolution: {target} ({date})", fontsize=16)
    
    def plot_dist(ax, data, title, color):
        data = data.dropna()
        ax.hist(data, bins=50, color=color, alpha=0.7, density=True)
        ax.set_title(f"{title}\nMean={data.mean():.2f}, Std={data.std():.2f}")
        ax.grid(True, alpha=0.3)
    
    plot_dist(axes[0, 0], series, "1. Original Raw Data", "gray")
    plot_dist(axes[0, 1], s_win, "2. After Winsorization (3σ)", "orange")
    plot_dist(axes[1, 0], s_fill, "3. After Imputation (Ind. Median)", "teal")
    plot_dist(axes[1, 1], s_std, "4. After Standardization (Z-Score)", "royalblue")
    
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"📈 Visualization saved to: {output_image}")

if __name__ == "__main__":
    DATE = "2020-02-07"
    FACTOR_DIR = "./factors/raw"
    INDUSTRY_DIR = "./data/data_industry"
    OUTPUT_IMAGE = "factor_preprocess_plot.png"
    
    visualize_preprocessing(DATE, FACTOR_DIR, INDUSTRY_DIR, OUTPUT_IMAGE)
