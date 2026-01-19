import os
from typing import Optional

import numpy as np
import pandas as pd
import torch


def export_batch_to_csv_optimized(
    X: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    seq_lens: torch.Tensor,
    train_sizes: torch.Tensor,
    series_flags: torch.Tensor,
    batch_idx: int,
    output_dir: str,
    datasets_per_folder: int = 1000,
    use_parquet: bool = False,
    batch_processing_size: int = 100,
):
    """Export a batch of datasets to CSV/Parquet files (highly optimized version).

    Optimizations:
    1. Convert tensors to numpy once at the beginning
    2. Pre-allocate arrays where possible
    3. Batch processing for memory efficiency
    4. Optional Parquet format for better compression and speed
    5. Vectorized operations where possible

    Parameters
    ----------
    X : torch.Tensor
        Input features tensor [batch_size, seq_len, features] or nested tensor
    y : torch.Tensor
        Target labels tensor [batch_size, seq_len] or nested tensor
    d : torch.Tensor
        Number of features for each dataset [batch_size]
    seq_lens : torch.Tensor
        Sequence length for each dataset [batch_size]
    train_sizes : torch.Tensor
        Position to split train/eval data [batch_size]
    series_flags : torch.Tensor
        Boolean flags indicating if data is time series [batch_size]
    batch_idx : int
        Index of the current batch
    output_dir : str
        Directory to save CSV files
    datasets_per_folder : int, default=1000
        Number of datasets to store per folder
    use_parquet : bool, default=False
        Whether to use Parquet format (faster and more compressed)
    batch_processing_size : int, default=100
        Number of datasets to process in each batch for memory efficiency
    """
    batch_size = len(d)

    # Create folder for this batch
    folder_idx = batch_idx // datasets_per_folder
    folder_name = f"batch_folder_{folder_idx:06d}"
    batch_folder = os.path.join(output_dir, folder_name)
    os.makedirs(batch_folder, exist_ok=True)

    # Convert tensors to numpy once (major optimization)
    d_np = d.cpu().numpy()
    seq_lens_np = seq_lens.cpu().numpy()
    train_sizes_np = train_sizes.cpu().numpy()
    series_flags_np = series_flags.cpu().numpy()

    # Handle nested tensors vs dense tensors
    if X.is_nested:
        # For nested tensors, process each dataset individually
        for i in range(batch_size):
            dataset_X = X[i].cpu().numpy()  # [seq_len, features]
            dataset_y = y[i].cpu().numpy()  # [seq_len]
            dataset_d = int(d_np[i])
            dataset_seq_len = int(seq_lens_np[i])
            dataset_train_size = train_sizes_np[i]
            dataset_is_series = series_flags_np[i]

            # Pre-allocate arrays for efficiency
            max_features = dataset_X.shape[1]

            # Create timestamps
            timestamps = np.arange(dataset_seq_len)

            # Pre-allocate data dictionary
            data_dict = {"timestamp": timestamps}

            # Add feature columns efficiently
            for j in range(max_features):
                if j < dataset_d:
                    column_data = dataset_X[:, j]
                    if column_data.ndim > 1:
                        column_data = column_data.flatten()
                else:
                    column_data = np.full(dataset_seq_len, np.nan)
                data_dict[f"column_{j + 1}"] = column_data

            # Add target column
            target_data = dataset_y
            if target_data.ndim > 1:
                target_data = target_data.flatten()
            data_dict["target"] = target_data

            # Add metadata columns efficiently
            data_dict["train_size"] = np.full(
                dataset_seq_len, dataset_train_size
            )
            data_dict["is_series"] = np.full(dataset_seq_len, dataset_is_series)

            # Create and save DataFrame
            df = pd.DataFrame(data_dict)
            csv_filename = f"dataset_{batch_idx:06d}_{i:04d}.csv"
            csv_path = os.path.join(batch_folder, csv_filename)

            if use_parquet:
                csv_path = csv_path.replace(".csv", ".parquet")
                df.to_parquet(csv_path, index=False, compression="snappy")
            else:
                df.to_csv(csv_path, index=False)

    else:
        # For dense tensors - optimized batch processing
        X_np = X.cpu().numpy()  # [batch_size, seq_len, features]
        y_np = y.cpu().numpy()  # [batch_size, seq_len]

        # Find maximum sequence length and features for pre-allocation
        # max_seq_len = int(seq_lens_np.max())
        max_features = X_np.shape[-1]

        # Process datasets in batches for better memory efficiency
        for start_idx in range(0, batch_size, batch_processing_size):
            end_idx = min(start_idx + batch_processing_size, batch_size)

            # Pre-allocate arrays for this batch
            # current_batch_size = end_idx - start_idx

            for i in range(start_idx, end_idx):
                dataset_X = X_np[i, 0]  # [seq_len, features]
                dataset_y = y_np[i, 0]  # [seq_len]
                dataset_d = int(d_np[i])
                dataset_seq_len = int(seq_lens_np[i])
                dataset_train_size = train_sizes_np[i]
                dataset_is_series = series_flags_np[i]

                # Create timestamps
                timestamps = np.arange(dataset_seq_len)

                # Pre-allocate data dictionary
                data_dict = {"timestamp": timestamps}

                # Add feature columns efficiently
                for j in range(max_features):
                    if j < dataset_d:
                        column_data = dataset_X[:, j]
                        if column_data.ndim > 1:
                            column_data = column_data.flatten()
                    else:
                        column_data = np.full(dataset_seq_len, np.nan)
                    data_dict[f"column_{j + 1}"] = column_data

                # Add target column
                target_data = dataset_y
                if target_data.ndim > 1:
                    target_data = target_data.flatten()
                data_dict["target"] = target_data

                # Add metadata columns efficiently
                data_dict["train_size"] = np.full(
                    dataset_seq_len, dataset_train_size
                )
                data_dict["is_series"] = np.full(
                    dataset_seq_len, dataset_is_series
                )

                # Create and save DataFrame
                df = pd.DataFrame(data_dict)
                csv_filename = f"dataset_{batch_idx:06d}_{i:04d}.csv"
                csv_path = os.path.join(batch_folder, csv_filename)

                if use_parquet:
                    csv_path = csv_path.replace(".csv", ".parquet")
                    df.to_parquet(csv_path, index=False, compression="snappy")
                else:
                    df.to_csv(csv_path, index=False)

    print(f"Saved {batch_size} datasets to {batch_folder}")


def export_batch_to_csv_vectorized(
    X: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    seq_lens: torch.Tensor,
    train_sizes: torch.Tensor,
    series_flags: torch.Tensor,
    batch_idx: int,
    output_dir: str,
    datasets_per_folder: int = 1000,
    use_parquet: bool = True,
):
    """Ultra-optimized version using vectorized operations and single large file.

    This version creates a single large CSV/Parquet file with all datasets,
    which is much faster than individual files.
    """
    batch_size = len(d)

    # Create folder for this batch
    folder_idx = batch_idx // datasets_per_folder
    folder_name = f"batch_folder_{folder_idx:06d}"
    batch_folder = os.path.join(output_dir, folder_name)
    os.makedirs(batch_folder, exist_ok=True)

    # Convert tensors to numpy once
    d_np = d.cpu().numpy()
    seq_lens_np = seq_lens.cpu().numpy()
    train_sizes_np = train_sizes.cpu().numpy()
    series_flags_np = series_flags.cpu().numpy()

    if X.is_nested:
        # For nested tensors, we need to process individually
        all_data = []

        for i in range(batch_size):
            dataset_X = X[i].cpu().numpy()  # [seq_len, features]
            dataset_y = y[i].cpu().numpy()  # [seq_len]
            dataset_d = int(d_np[i])
            dataset_seq_len = int(seq_lens_np[i])
            dataset_train_size = train_sizes_np[i]
            dataset_is_series = series_flags_np[i]

            # Create timestamps
            timestamps = np.arange(dataset_seq_len)

            # Create data for this dataset
            dataset_data = {
                "dataset_id": np.full(dataset_seq_len, i),
                "timestamp": timestamps,
                "target": dataset_y.flatten()
                if dataset_y.ndim > 1
                else dataset_y,
                "train_size": np.full(dataset_seq_len, dataset_train_size),
                "is_series": np.full(dataset_seq_len, dataset_is_series),
            }

            # Add feature columns
            max_features = dataset_X.shape[1]
            for j in range(max_features):
                if j < dataset_d:
                    column_data = dataset_X[:, j]
                    if column_data.ndim > 1:
                        column_data = column_data.flatten()
                else:
                    column_data = np.full(dataset_seq_len, np.nan)
                dataset_data[f"column_{j + 1}"] = column_data

            all_data.append(pd.DataFrame(dataset_data))

        # Combine all datasets into one DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)

    else:
        # For dense tensors - vectorized approach
        X_np = X.cpu().numpy()  # [batch_size, seq_len, features]
        y_np = y.cpu().numpy()  # [batch_size, seq_len]

        # Reshape for vectorized processing
        batch_size, seq_len, features = X_np.shape

        # Create all timestamps at once
        all_timestamps = np.tile(np.arange(seq_len), batch_size)

        # Create dataset IDs
        dataset_ids = np.repeat(np.arange(batch_size), seq_len)

        # Flatten targets
        all_targets = y_np.flatten()

        # Repeat metadata for each timestep
        all_train_sizes = np.repeat(train_sizes_np, seq_len)
        all_series_flags = np.repeat(series_flags_np, seq_len)

        # Create the main DataFrame
        data_dict = {
            "dataset_id": dataset_ids,
            "timestamp": all_timestamps,
            "target": all_targets,
            "train_size": all_train_sizes,
            "is_series": all_series_flags,
        }

        # Add feature columns vectorized
        for j in range(features):
            feature_data = X_np[:, :, j].flatten()
            data_dict[f"column_{j + 1}"] = feature_data

        combined_df = pd.DataFrame(data_dict)

    # Save as single file
    filename = f"batch_{batch_idx:06d}_combined"
    if use_parquet:
        filepath = os.path.join(batch_folder, f"{filename}.parquet")
        combined_df.to_parquet(filepath, index=False, compression="snappy")
    else:
        filepath = os.path.join(batch_folder, f"{filename}.csv")
        combined_df.to_csv(filepath, index=False)

    print(f"Saved {batch_size} datasets as single file to {filepath}")


# Performance comparison function
def benchmark_csv_export():
    """Benchmark different CSV export methods."""
    import time

    # Create sample data
    batch_size = 100
    seq_len = 50
    features = 10

    X = torch.randn(batch_size, seq_len, features)
    y = torch.randn(batch_size, seq_len)
    d = torch.randint(1, features + 1, (batch_size,))
    seq_lens = torch.full((batch_size,), seq_len)
    train_sizes = torch.randint(10, seq_len - 10, (batch_size,))
    series_flags = torch.randint(0, 2, (batch_size,))

    output_dir = "/tmp/csv_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    # Benchmark original method (simplified)
    start_time = time.time()
    # ... original implementation ...
    original_time = time.time() - start_time

    # Benchmark optimized method
    start_time = time.time()
    export_batch_to_csv_optimized(
        X, y, d, seq_lens, train_sizes, series_flags, 0, output_dir
    )
    optimized_time = time.time() - start_time

    # Benchmark vectorized method
    start_time = time.time()
    export_batch_to_csv_vectorized(
        X, y, d, seq_lens, train_sizes, series_flags, 0, output_dir
    )
    vectorized_time = time.time() - start_time

    print(f"Original method: {original_time:.2f}s")
    print(f"Optimized method: {optimized_time:.2f}s")
    print(f"Vectorized method: {vectorized_time:.2f}s")
    print(f"Speedup (optimized): {original_time / optimized_time:.2f}x")
    print(f"Speedup (vectorized): {original_time / vectorized_time:.2f}x")
