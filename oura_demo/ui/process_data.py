
import pandas as pd
import os

# Define file paths
input_path = os.path.expanduser('~/data/oura/daily_ts_100k_with_custom_split.parquet')
output_dir = os.path.expanduser('~/data/oura/windowed_data')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
try:
    df = pd.read_parquet(input_path)
except FileNotFoundError:
    print(f"Error: Input file not found at {input_path}")
    exit()


# Sort by user and timestamp
df = df.sort_values(by=['user_id', 'timestamp'])

# Get the first user_id
first_user_id = df['user_id'].iloc[0]

# Filter for the first user
user_df = df[df['user_id'] == first_user_id]

# Define window sizes
window_sizes = [192, 96]
num_windows_per_size = 2

# Create and save windowed files
for size in window_sizes:
    for i in range(num_windows_per_size):
        start_index = i * size
        end_index = start_index + size
        if end_index <= len(user_df):
            window = user_df.iloc[start_index:end_index]
            output_filename = f'user_{first_user_id}_window_{i+1}_size_{size}.parquet'
            output_path = os.path.join(output_dir, output_filename)
            window.to_parquet(output_path)
            print(f'Successfully created {output_path}')

