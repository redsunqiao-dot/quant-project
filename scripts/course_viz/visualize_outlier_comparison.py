import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add current directory to path to ensure import works if needed
sys.path.append(str(Path(__file__).parent))

from src.research.outlier_methods_compare import OutlierMethodComparator

# Configuration
# Assuming we run this script from 'quant-course01/bt-local/code/'
DATA_DIR = Path("../data")
FACTOR_DIR = Path("../factors/raw")
OUTPUT_CSV = Path("outputs/outlier_methods_compare/outlier_methods_compare.csv")
SAMPLE_DATE = "2020-02-07"  # A sample date we saw exists

def set_chinese_font():
    """Attempt to set a Chinese-compatible font."""
    plt.rcParams['axes.unicode_minus'] = False
    try:
        # Common Chinese fonts on Linux/Windows/Mac
        fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
        plt.rcParams['font.family'] = fonts
    except Exception:
        pass

def plot_stats_comparison():
    """
    Plot boxplots comparing the distribution of statistics (Skew, Kurtosis, Std)
    across all days for each method.
    """
    if not OUTPUT_CSV.exists():
        print(f"Comparison CSV not found at {OUTPUT_CSV}. Run outlier_methods_compare.py first.")
        return
    
    print(f"Loading stats from {OUTPUT_CSV}...")
    df = pd.read_csv(OUTPUT_CSV)
    
    metrics = ['std', 'skew', 'kurt']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    unique_methods = df['method'].unique()
    
    for i, metric in enumerate(metrics):
        # Prepare data for boxplot
        data_to_plot = [df[df['method'] == m][metric].dropna() for m in unique_methods]
        
        axes[i].boxplot(data_to_plot, labels=unique_methods, patch_artist=True)
        axes[i].set_title(f'Distribution of Daily {metric.upper()}')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.suptitle("Statistical Properties Stability Across Methods (All Dates)", fontsize=16)
    plt.tight_layout()
    output_file = 'outlier_stats_comparison.png'
    plt.savefig(output_file)
    print(f"Saved stats comparison to {output_file}")

def plot_distribution_demo():
    """
    Load one day of factor data and plot histograms for each method
    to visually show how they clip/transform the data.
    """
    global SAMPLE_DATE
    comp = OutlierMethodComparator()
    
    # Check if factor file exists
    factor_file = FACTOR_DIR / f"{SAMPLE_DATE}.csv"
    if not factor_file.exists():
        print(f"Sample factor file not found: {factor_file}")
        # Try to find any file
        files = list(FACTOR_DIR.glob("*.csv"))
        if files:
            sample_file = files[0]
            print(f"Using alternative sample file: {sample_file}")
            SAMPLE_DATE = sample_file.stem
        else:
            print("No factor files found.")
            return

    try:
        print(f"Loading sample data for {SAMPLE_DATE}...")
        series = comp.load_factor_series(str(FACTOR_DIR), SAMPLE_DATE)
    except Exception as e:
        print(f"Could not load sample data: {e}")
        return

    # Clean data
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    
    methods = {
        "Raw (No Treatment)": lambda x: x,
        "Sigma (3.0)": comp.winsor_sigma,
        "Percentile (1%-99%)": comp.winsor_percentile,
        "MAD (3.5)": comp.winsor_mad,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (name, func) in enumerate(methods.items()):
        # Apply method
        cleaned_data = func(series.copy())
        
        # Calculate stats
        stats = comp.describe_series(cleaned_data)
        
        # Plot histogram
        # Use a fixed range if possible to make comparisons easier, 
        # but outlier handling changes range significantly. 
        # Let's let it auto-scale but share X-axis if feasible? 
        # No, Raw might be huge. Let's keep independent but maybe limit Raw for visibility. 
        
        counts, bins, patches = axes[i].hist(cleaned_data, bins=60, density=True, 
                                             alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add KDE curve (simple approximation)
        try:
            from scipy.stats import gaussian_kde
            density = gaussian_kde(cleaned_data)
            xs = np.linspace(min(cleaned_data), max(cleaned_data), 200)
            axes[i].plot(xs, density(xs), 'r-', lw=2, label='KDE')
        except ImportError:
            pass # Skip if scipy not available

        # Annotation
        text_str = '\n'.join([
            f"Mean: {stats['mean']:.4f}",
            f"Std:  {stats['std']:.4f}",
            f"Skew: {stats['skew']:.2f}",
            f"Kurt: {stats['kurt']:.2f}",
            f"Min/Max: {stats['min']:.2f}/{stats['max']:.2f}"
        ])
        
        axes[i].text(0.95, 0.95, text_str, transform=axes[i].transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"Effect of Outlier Treatment on Factor Distribution ({SAMPLE_DATE})", fontsize=16)
    plt.tight_layout()
    output_file = 'outlier_distribution_demo.png'
    plt.savefig(output_file)
    print(f"Saved distribution demo to {output_file}")

if __name__ == "__main__":
    set_chinese_font()
    plot_stats_comparison()
    plot_distribution_demo()
