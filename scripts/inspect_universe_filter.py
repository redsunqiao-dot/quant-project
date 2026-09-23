
import pandas as pd
from pathlib import Path
from src.data.universe_filter import UniverseFilter

def inspect_filtering_process(date, factor_dir, data_daily_dir, data_ud_dir):
    print(f"=== Inspecting Universe Filter for {date} ===")
    
    # Initialize Filter
    uf = UniverseFilter()
    
    # Define paths
    factor_file = Path(factor_dir) / f"{date}.csv"
    daily_file = Path(data_daily_dir) / f"{date}.csv"
    status_file = Path(data_ud_dir) / f"{date}.csv"
    
    # Check existence
    if not (factor_file.exists() and daily_file.exists() and status_file.exists()):
        print(f"Error: Missing files for {date}")
        return

    # Load Data
    factor_df = uf.load_csv_with_index(factor_file)
    daily_df = uf.load_csv_with_index(daily_file)
    status_df = uf.load_csv_with_index(status_file)
    
    print(f"Original Factor Universe Size: {len(factor_df)}")
    
    # Merge for inspection
    overlapping_cols = status_df.columns.intersection(daily_df.columns)
    if len(overlapping_cols) > 0:
        status_df = status_df.drop(columns=overlapping_cols)
    merged = daily_df.join(status_df, how="left")
    
    # Align with factor universe
    # We only care about stocks that are in the factor file initially
    merged = merged.loc[merged.index.intersection(factor_df.index)]
    
    # Step-by-step filtering
    current_universe = merged.copy()
    initial_count = len(current_universe)
    
    print(f"\nInitial Candidates (intersection of Daily & Factor): {initial_count}")
    
    # 1. Paused
    paused_mask = pd.Series(True, index=current_universe.index)
    if "paused" in current_universe.columns:
        paused_mask = current_universe["paused"] == 0
    
    removed_paused = current_universe[~paused_mask]
    current_universe = current_universe[paused_mask]
    print(f"Step 1: Removed Paused: {len(removed_paused)} -> Remaining: {len(current_universe)}")
    if len(removed_paused) > 0:
        print(f"   Examples: {removed_paused.index[:5].tolist()}")

    # 2. Limit Up/Down (ZT/DT)
    zt_mask = pd.Series(True, index=current_universe.index)
    dt_mask = pd.Series(True, index=current_universe.index)
    
    if "zt" in current_universe.columns:
        zt_mask = current_universe["zt"] == 0
    if "dt" in current_universe.columns:
        dt_mask = current_universe["dt"] == 0
        
    limit_mask = zt_mask & dt_mask
    removed_limit = current_universe[~limit_mask]
    current_universe = current_universe[limit_mask]
    print(f"Step 2: Removed Limit Up/Down: {len(removed_limit)} -> Remaining: {len(current_universe)}")
    if len(removed_limit) > 0:
        print(f"   Examples: {removed_limit.index[:5].tolist()}")

    # 3. Price
    price_mask = pd.Series(True, index=current_universe.index)
    if "close" in current_universe.columns:
        price_mask = current_universe["close"] >= uf.min_price
    
    removed_price = current_universe[~price_mask]
    current_universe = current_universe[price_mask]
    print(f"Step 3: Removed Price < {uf.min_price}: {len(removed_price)} -> Remaining: {len(current_universe)}")
    if len(removed_price) > 0:
        print(f"   Examples: {removed_price.index[:5].tolist()}")

    # 4. Volume
    volume_mask = pd.Series(True, index=current_universe.index)
    if "volume" in current_universe.columns:
        volume_mask = current_universe["volume"] >= uf.min_volume
        
    removed_volume = current_universe[~volume_mask]
    current_universe = current_universe[volume_mask]
    print(f"Step 4: Removed Volume < {uf.min_volume}: {len(removed_volume)} -> Remaining: {len(current_universe)}")
    if len(removed_volume) > 0:
        print(f"   Examples: {removed_volume.index[:5].tolist()}")

    # 5. Turnover
    turnover_mask = pd.Series(True, index=current_universe.index)
    if "turnover_ratio" in current_universe.columns:
        turnover_mask = current_universe["turnover_ratio"] >= uf.min_turnover
        
    removed_turnover = current_universe[~turnover_mask]
    current_universe = current_universe[turnover_mask]
    print(f"Step 5: Removed Turnover < {uf.min_turnover}: {len(removed_turnover)} -> Remaining: {len(current_universe)}")
    if len(removed_turnover) > 0:
        print(f"   Examples: {removed_turnover.index[:5].tolist()}")

    # 6. ST
    st_mask = pd.Series(True, index=current_universe.index)
    if uf.remove_st:
        if "is_st" in current_universe.columns:
            st_mask = current_universe["is_st"] == 0
        elif "st" in current_universe.columns:
            st_mask = current_universe["st"] == 0
            
    removed_st = current_universe[~st_mask]
    current_universe = current_universe[st_mask]
    print(f"Step 6: Removed ST: {len(removed_st)} -> Remaining: {len(current_universe)}")
    if len(removed_st) > 0:
        print(f"   Examples: {removed_st.index[:5].tolist()}")

    final_count = len(current_universe)
    drop_count = initial_count - final_count
    drop_rate = drop_count / initial_count if initial_count > 0 else 0
    
    print(f"\nFinal Summary:")
    print(f"Before: {initial_count}")
    print(f"After:  {final_count}")
    print(f"Dropped: {drop_count} ({drop_rate:.2%})")

if __name__ == "__main__":
    # Settings
    DATE = "2020-02-07"
    FACTOR_DIR = "./factors/raw"
    DATA_DAILY_DIR = "./data/data_daily"
    DATA_UD_DIR = "./data/data_ud_new"
    
    inspect_filtering_process(DATE, FACTOR_DIR, DATA_DAILY_DIR, DATA_UD_DIR)
