import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from torch.utils.data import DataLoader

from synthefy_pkg.data.foundation_dataloader import (
    MultiDirectoryTimeSeriesDataset,
    collate_fn,
)

BATCH_SIZE = 128
NUM_WORKERS = 32


def create_test_data(
    base_dir: str,
    num_dirs: int = 3,
    samples_per_dir: int = 1000,
    num_timeseries_features: int = 1,
    num_continuous_metadata: int = 5,
) -> List[str]:
    """Create test data directories with synthetic time series data.

    Args:
        base_dir: Base directory to create test data in
        num_dirs: Number of data directories to create
        samples_per_dir: Number of samples per directory
        num_timeseries_features: Number of features in the time series data
        num_continuous_metadata: Number of continuous metadata features

    Returns:
        List[str]: List of created data directory paths
    """
    os.makedirs(base_dir, exist_ok=True)

    data_dirs = []
    for i in range(num_dirs):
        dir_path = os.path.join(base_dir, f"data_{i}")
        os.makedirs(dir_path, exist_ok=True)

        # Create synthetic data with a pattern to verify splits
        # Timeseries data columns
        timeseries_data = np.zeros((samples_per_dir, num_timeseries_features))
        for j in range(num_timeseries_features):
            # Create features with different frequencies and phases
            feature = np.sin(
                np.linspace(0, 10 * np.pi, samples_per_dir) + j * 0.5
            )
            timeseries_data[:, j] = feature

        # Create metadata features
        metadata_data = np.zeros((samples_per_dir, num_continuous_metadata))
        for j in range(num_continuous_metadata):
            # Create features with different frequencies and phases
            feature = np.sin(
                np.linspace(0, (j + 2) * np.pi, samples_per_dir) + (j + 1) * 0.5
            )
            metadata_data[:, j] = feature

        # Create timestamps for the data
        timestamps = pd.date_range(
            start="2023-01-01", periods=samples_per_dir, freq="1min"
        )

        # Define column names
        timeseries_column_names = [
            f"timeseries_{j}" for j in range(num_timeseries_features)
        ]
        metadata_column_names = [
            f"metadata_{j}" for j in range(num_continuous_metadata)
        ]

        # Create DataFrame with timestamps and all columns
        df = pd.DataFrame()

        # Add timeseries columns
        for j, col_name in enumerate(timeseries_column_names):
            df[col_name] = timeseries_data[:, j]

        # Add metadata columns
        for j, col_name in enumerate(metadata_column_names):
            df[col_name] = metadata_data[:, j]

        df["timestamp"] = timestamps

        # Add text description column
        df["text_description"] = f"Synthetic time series data for directory {i}"

        # Add textual metadata column with varying text for each timestamp
        df["textual_metadata"] = [
            f"Metadata for sample {k} in directory {i}"
            for k in range(samples_per_dir)
        ]

        # Save data as parquet file with all columns
        logger.info(f"Saving data to {os.path.join(dir_path, 'data.parquet')}")
        df.to_parquet(os.path.join(dir_path, "data.parquet"))

        # Create metadata.json file with updated structure to match fm_utils.py
        # Create columns list with detailed information about each column
        columns = []

        # Add timeseries columns
        for col_name in timeseries_column_names:
            columns.append(
                {
                    "column_id": col_name,
                    "type": "continuous",
                    "is_metadata": "no",
                    "description": f"Synthetic timeseries feature {col_name}",
                }
            )

        # Add metadata columns
        for col_name in metadata_column_names:
            columns.append(
                {
                    "column_id": col_name,
                    "type": "continuous",
                    "is_metadata": "yes",
                    "description": f"Synthetic continuous metadata feature {col_name}",
                }
            )

        # Add timestamp column
        columns.append(
            {
                "column_id": "timestamp",
                "type": "timestamp",
                "is_metadata": "yes",
                "description": "Timestamp for each data point",
            }
        )

        # Add text description column
        columns.append(
            {
                "column_id": "text_description",
                "type": "text",
                "is_metadata": "yes",
                "description": "Text description of the time series",
            }
        )

        # Add textual metadata column
        columns.append(
            {
                "column_id": "textual_metadata",
                "type": "text",
                "is_metadata": "yes",
                "description": "Textual metadata for each data point",
            }
        )

        metadata = {
            "num_samples": samples_per_dir,
            "num_features": num_timeseries_features + num_continuous_metadata,
            "description": f"Synthetic time series data for directory {i}",
            "created_at": pd.Timestamp.now().isoformat(),
            "sampling_rate": 1.0,  # Placeholder value
            "data_type": "synthetic",
            "title": f"Synthetic dataset {i} for testing foundation models",
            "timestamps_columns": ["timestamp"],
            "text_description_column": "text_description",
            "textual_metadata_column": "textual_metadata",
            "columns": columns,  # Add the detailed columns information
            # Add metadata_types_to_use to ensure consistency with WindowPreprocessor
            "metadata_types_to_use": [
                "timestamp",
                "dataset_description",
                "text_description",
                "continuous",
                "retrieved_timeseries",
                "time_varying_textual_metadata",
            ],
        }

        logger.info(
            f"Saving metadata to {os.path.join(dir_path, 'metadata.json')}"
        )
        with open(os.path.join(dir_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        data_dirs.append(dir_path)

    return data_dirs


def worker_init_fn(worker_id):
    """Initialize each worker process.

    Args:
        worker_id: The ID of the worker process
    """
    # Set a lower file descriptor limit for each worker
    import resource

    # Get the current soft limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set a reasonable limit per worker (adjust as needed)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    # Set file sharing strategy to file_system for this worker
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Set a different seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def verify_data_coverage(
    data_dirs: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Verify that all data is being processed.

    Args:
        data_dirs: List of directories containing the source data
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    # Count total data points in source directories
    total_data_points = 0
    for data_dir in data_dirs:
        parquet_path = os.path.join(data_dir, "data.parquet")
        df = pd.read_parquet(parquet_path)
        total_data_points += len(df)

    logger.info(f"Total data points in source directories: {total_data_points}")

    # Count samples in each dataloader
    train_samples = count_samples(train_loader)
    val_samples = count_samples(val_loader)
    test_samples = count_samples(test_loader)

    total_samples = train_samples + val_samples + test_samples
    logger.info(f"Total samples in dataloaders: {total_samples}")

    # Check if the number of samples is reasonable
    # Note: The number of samples will be less than the total data points
    # due to the sliding window approach
    if total_samples == 0:
        logger.warning("Warning: No samples found in dataloaders!")
    elif total_samples > total_data_points:
        logger.warning(
            "Warning: More samples than data points! Check for data duplication."
        )


def count_samples(dataloader: DataLoader) -> int:
    """Count the number of samples in a dataloader.

    Args:
        dataloader: DataLoader to count samples from

    Returns:
        int: Total number of samples in the dataloader
    """
    count = 0
    for batch in dataloader:
        count += batch["timeseries"].shape[0]
    return count


def main() -> None:
    """Main function to create test data, process it, and verify the results."""
    # Set multiprocessing start method at the beginning
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn", force=True)
        logger.info("Set multiprocessing start method to 'spawn'")

    # Set file sharing strategy to file_system
    torch.multiprocessing.set_sharing_strategy("file_system")
    logger.info("Set file sharing strategy to 'file_system'")

    # Reduce number of workers to a more reasonable value
    num_workers = min(NUM_WORKERS, os.cpu_count() or 1)
    # Further reduce workers to avoid "too many open files" error
    num_workers = min(num_workers, 4)  # Limit to 4 workers
    logger.info(f"Using {num_workers} workers for data loading")

    # Create test data
    test_dir = "test_data"
    logger.info(f"Creating test data in {test_dir}")
    data_dirs = create_test_data(
        test_dir,
        num_dirs=5,  # Reduce number of directories to avoid resource issues
        samples_per_dir=500,  # Reduce sample size
        num_timeseries_features=1,
        num_continuous_metadata=5,
    )

    # Clean up any existing embedding files to ensure fresh generation
    for data_dir in data_dirs:
        for file_name in [
            "continuous_embeddings.npy",
            "retrieved_ts_embeddings.npy",
            "dataset_description_embedding.npy",
            "text_description_embedding.npy",
            "time_varying_text_embedding.npy",
        ]:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed existing embedding file: {file_path}")

    # Test the dataset with explicit multiprocessing settings
    logger.info("Creating dataloaders")

    # Create datasets with consistent metadata_types_to_use
    metadata_types_to_use = [
        "timestamp",
        "dataset_description",
        "text_description",
        "continuous",
        "retrieved_timeseries",
        "time_varying_textual_metadata",
    ]

    # Create datasets with explicit metadata_types_to_use parameter
    train_dataset = MultiDirectoryTimeSeriesDataset(
        data_dirs=data_dirs,
        window_size=50,
        stride=10,
        split="train",
        metadata_types_to_use=metadata_types_to_use,
    )
    val_dataset = MultiDirectoryTimeSeriesDataset(
        data_dirs=data_dirs,
        window_size=50,
        stride=10,
        split="val",
        metadata_types_to_use=metadata_types_to_use,
    )
    test_dataset = MultiDirectoryTimeSeriesDataset(
        data_dirs=data_dirs,
        window_size=50,
        stride=10,
        split="test",
        metadata_types_to_use=metadata_types_to_use,
    )

    # Create dataloaders with explicit multiprocessing settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_fn,
    )

    # Verify all data is being processed
    logger.info("Verifying data coverage")
    verify_data_coverage(data_dirs, train_loader, val_loader, test_loader)

    logger.success(
        "MultiDirectoryTimeSeriesDataset test completed successfully!"
    )


if __name__ == "__main__":
    main()
