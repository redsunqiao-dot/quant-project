
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_factor_turnover(output_dir: str):
    """
    Read factor_turnover.csv and visualize the turnover time series.
    """
    out_path = Path(output_dir)
    detail_file = out_path / "factor_turnover.csv"

    if not detail_file.exists():
        print(f"Detail file not found: {detail_file}")
        return

    detail_df = pd.read_csv(detail_file, parse_dates=['date'])

    plt.figure(figsize=(12, 6))

    # Plot original turnover data
    plt.plot(detail_df['date'], detail_df['turnover'], label="Daily Turnover", color='lightgrey', linewidth=0.8)

    # Plot rolling average of turnover
    rolling_turnover = detail_df.set_index('date')['turnover'].rolling(20).mean()
    plt.plot(rolling_turnover.index, rolling_turnover, label="20-Day Rolling Avg Turnover", color='dodgerblue', linewidth=1.5)

    plt.title("Factor Turnover Time Series")
    plt.ylabel("Turnover (1 - Rank IC)")
    plt.xlabel("Date")
    plt.axhline(rolling_turnover.mean(), color='red', linestyle='--', linewidth=1, label=f'Mean Turnover: {rolling_turnover.mean():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    save_path = out_path / "factor_turnover_timeseries.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Time Series plot to {save_path}")


if __name__ == "__main__":
    output_dir = "outputs/turnover"
    
    # Adjust path if running from root
    if Path("code").exists():
        output_dir = "code/outputs/turnover"
    
    if Path(output_dir).exists():
        plot_factor_turnover(output_dir)
    else:
        # Try to find in current directory
        if Path("outputs/turnover").exists():
             plot_factor_turnover("outputs/turnover")
        else:
             print("Output directory not found.")
