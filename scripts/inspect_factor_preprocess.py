
import pandas as pd
import numpy as np
from pathlib import Path
from src.research.factor_preprocess import FactorPreprocessor

def inspect_preprocessing_steps(date, factor_dir, industry_dir):
    print(f"=== Inspecting Factor Preprocessing for {date} ===")
    
    # Paths
    factor_file = Path(factor_dir) / f"{date}.csv"
    industry_file = Path(industry_dir) / f"{date}.csv"
    
    if not factor_file.exists():
        print(f"Error: Factor file not found for {date}")
        return

    # Initialize Preprocessor (using default settings for inspection)
    fp = FactorPreprocessor(
        fill_method="industry_median",
        standardize="zscore",
        winsorize=True,
        n_sigma=3.0,
        industry_col="sw_l1" # Assuming sw_l1 based on common data formats, or will default to first col
    )
    
    # Load Data
    df = fp.load_factor_file(factor_file)
    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
    
    # Identify Factor Columns
    factor_cols = fp.infer_factor_cols(df)
    if not factor_cols:
        print("No factor columns found!")
        return
    
    target_factor = factor_cols[0]
    print(f"\nTarget Factor for Inspection: '{target_factor}'")
    
    series = df[target_factor].copy()
    
    # Load Industry for context
    industry_series = None
    if industry_file.exists():
        industry_series = fp.load_industry_series(industry_file, industry_col="sw_l1")
        print(f"Loaded Industry data for {len(industry_series)} stocks.")
    else:
        print("Industry file not found, will fallback to global median.")

    # Helper to print stats
    def print_stats(name, s):
        print(f"\n--- {name} ---")
        print(f"  Count: {s.count()} (Missing: {s.isna().sum()})")
        print(f"  Mean:  {s.mean():.4f}")
        print(f"  Std:   {s.std():.4f}")
        print(f"  Min:   {s.min():.4f}")
        print(f"  Max:   {s.max():.4f}")
        print(f"  Skew:  {s.skew():.4f}")

    # 0. Original
    print_stats("Original Data", series)
    
    # 0.5 Handle Inf
    series = series.replace([np.inf, -np.inf], np.nan)

    # 1. Winsorization
    series_win = fp.winsorize_series(series, n_sigma=3.0)
    print_stats("Step 1: After Winsorization (3σ)", series_win)
    
    # Check what changed
    changed_mask = (series != series_win) & (~series.isna())
    if changed_mask.any():
        print(f"  -> {changed_mask.sum()} outliers clipped.")
        print(f"  -> Example: Original {series[changed_mask].iloc[0]:.4f} -> Clipped {series_win[changed_mask].iloc[0]:.4f}")
    else:
        print("  -> No outliers clipped.")

    # 2. Imputation
    series_filled = fp.fill_missing_series(series_win, method="industry_median", industry_series=industry_series)
    print_stats("Step 2: After Imputation (Industry Median)", series_filled)
    
    # 3. Standardization
    series_std = fp.standardize_series(series_filled, method="zscore")
    print_stats("Step 3: After Standardization (Z-Score)", series_std)

if __name__ == "__main__":
    DATE = "2020-02-07"
    FACTOR_DIR = "./factors/raw"
    # Assuming industry data might be in data_industry if available, checking path...
    INDUSTRY_DIR = "./data/data_industry" 
    
    inspect_preprocessing_steps(DATE, FACTOR_DIR, INDUSTRY_DIR)
