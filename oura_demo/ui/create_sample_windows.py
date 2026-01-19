"""
Create sample windowed CSV files from the Oura parquet data for demo purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Config from preprocessing config
WINDOW_SIZE = 96
TIMESERIES_COLS = [
    "average_hrv",
    "lowest_heart_rate", 
    "age_cva_diff",
    "highest_temperature",
    "stressed_duration",
    "latency",
]
DISCRETE_COLS = ["gender_male"]
CONTINUOUS_COLS = [
    "bmi",
    "hrv_std",
    "awake_mins",
    "age",
    "readiness_score",
    "sleep_score",
    "low_activity_time",
    "deep_mins",
    "sleep_duration",
    "restored_duration",
    "non_wear_time",
    "rem_mins",
    "steps",
    "active_calories",
    "heart_rate_std",
    "sedentary_time",
    "high_activity_met_minutes",
    "hours_after_oura_sleep_start",
    "light_mins",
    "medium_activity_met_minutes",
]
GROUP_COL = "user_id"

ALL_COLS = [GROUP_COL] + TIMESERIES_COLS + DISCRETE_COLS + CONTINUOUS_COLS

# Paths - use the smaller file in the workspace
PARQUET_PATH = Path("/Users/raimi.shah/synthefy-package/daily_ts_10k.parquet")
OUTPUT_DIR = Path("/Users/raimi.shah/synthefy-package/oura_demo/ui/sample_data")


def main():
    print(f"Loading parquet file: {PARQUET_PATH}")
    
    # Load only the columns we need
    df = pd.read_parquet(PARQUET_PATH, columns=ALL_COLS)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Group by user and find users with enough data
    user_counts = df.groupby(GROUP_COL).size()
    valid_users = user_counts[user_counts >= WINDOW_SIZE].index.tolist()
    print(f"Found {len(valid_users):,} users with >= {WINDOW_SIZE} rows")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract windows from a few different users
    num_windows = 5
    windows_created = 0
    
    for user_id in valid_users[:50]:  # Try first 50 valid users
        if windows_created >= num_windows:
            break
            
        user_df = df[df[GROUP_COL] == user_id].reset_index(drop=True)
        
        # Get the first window for this user
        if len(user_df) >= WINDOW_SIZE:
            window_df = user_df.head(WINDOW_SIZE).copy()
            
            # Drop user_id column (not needed for synthesis)
            window_df = window_df.drop(columns=[GROUP_COL])
            
            # Check for NaN values in timeseries columns
            ts_nan_count = window_df[TIMESERIES_COLS].isna().sum().sum()
            if ts_nan_count > 0:
                print(f"  Skipping user {user_id}: {ts_nan_count} NaN values in timeseries")
                continue
            
            # Save as CSV
            output_path = OUTPUT_DIR / f"oura_window_{windows_created + 1}.csv"
            window_df.to_csv(output_path, index=False)
            print(f"✓ Created: {output_path.name} ({len(window_df)} rows)")
            
            windows_created += 1
    
    print(f"\n{'='*50}")
    print(f"Created {windows_created} windowed CSV files in:")
    print(f"  {OUTPUT_DIR}")
    print(f"\nYou can upload these files to the demo at http://localhost:5173")
    print(f"{'='*50}")
    
    # Show sample stats
    if windows_created > 0:
        sample_df = pd.read_csv(OUTPUT_DIR / "oura_window_1.csv")
        print(f"\nSample window stats (timeseries columns):")
        print(sample_df[TIMESERIES_COLS].describe().round(2).to_string())


if __name__ == "__main__":
    main()
