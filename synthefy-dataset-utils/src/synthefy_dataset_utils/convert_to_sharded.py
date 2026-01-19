#!/usr/bin/env python
"""Convert parquet files to sharded dataset format for fine-tuning.

This module converts any parquet file into the sharded dataset format,
using the existing preprocessing configs from examples/configs/preprocessing_configs/.

Usage as CLI:
    python -m synthefy_dataset_utils.convert_to_sharded \\
        --config path/to/config_ppg_preprocessing.json

Usage as library:
    from synthefy_dataset_utils.convert_to_sharded import convert_parquet_to_sharded

    train_dir, val_dir = convert_parquet_to_sharded(
        config_path="config_ppg_preprocessing.json",
    )

Train/Val/Test Split Configuration
----------------------------------
The `train_val_split` config supports three splitting methods:

1. **Custom Split Column (BYOS) - Highest Priority**
   If your DataFrame contains a column named `custom_split`, this method is used automatically.
   - Column name must be exactly `custom_split`
   - Values must be one of: "train", "val", or "test"
   - The value from the first row of each sliding window determines the split
   - No configuration needed - just include the column in your data.

2. **Timestamp-based**
   Splits by temporal cutoffs. Configure in `train_val_split`:
   ```json
   "train_val_split": {
       "method": "timestamp",
       "timestamp_column": "timestamp",
       "train_end": "2023-01-01T00:00:00",
       "val_end": "2023-06-01T00:00:00"
   }
   ```
   - `timestamp_column`: Column containing datetime values
   - `train_end`: Train includes all samples with window_start < this timestamp
   - `val_end`: Val includes samples with train_end <= window_start < val_end
   - Test includes samples with window_start >= val_end

3. **Ratio-based (Default)**
   Splits samples within each group by proportion:
   ```json
   "train_val_split": {
       "train": 0.8,
       "val": 0.1,
       "shuffle": true
   }
   ```
   - `train`: Proportion for training (0.0-1.0)
   - `val`: Proportion for validation (0.0-1.0)
   - Test gets remaining: 1.0 - train - val
   - `shuffle`: Whether to shuffle before splitting (default: true)

History/Forecast Length Configuration
-------------------------------------
The `history_forecast_split` config supports two methods:

1. **Direct History Length (Preferred)**
   Specify the history length directly as a number:
   ```json
   "history_forecast_split": {
       "history_length": 409
   }
   ```
   - `history_length`: Number of time steps for history (must be < window_size)
   - Forecast length is automatically calculated as: window_size - history_length

2. **Ratio-based (Fallback)**
   Specify history as a ratio of window_size:
   ```json
   "history_forecast_split": {
       "history_ratio": 0.8
   }
   ```
   - `history_ratio`: Fraction of window_size for history (0.0-1.0, default: 0.8)
   - History length = int(window_size * history_ratio)
   - Forecast length = window_size - history_length
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from synthefy_dataset_utils.sharded_dataset_writer import ShardedDatasetWriter

# Default base path from environment variable
DEFAULT_DATASETS_BASE = os.environ.get("SYNTHEFY_DATASETS_BASE")

# Magic column name for BYOS (Bring Your Own Split) - matches preprocess.py convention
CUSTOM_SPLIT_COLUMN = "custom_split"

# Valid split values for BYOS
VALID_SPLIT_VALUES = {"train", "val", "test"}

# Default number of workers for parallel processing
DEFAULT_NUM_WORKERS = 8

# Default samples per shard (larger = fewer files, better I/O)
DEFAULT_SAMPLES_PER_SHARD = 10000


def _process_single_group(
    group_key: Any,
    group_df: pd.DataFrame,
    timeseries_cols: List[str],
    discrete_cols: List[str],
    continuous_cols: List[str],
    covariate_timeseries_cols: List[str],
    dataset_name: str,
    history_length: int,
    forecast_length: int,
    stride: int,
    timestamp_col: Optional[str],
    has_custom_split: bool,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]]:
    """Process a single group and return its samples.

    This function is designed to be called in parallel for each group.
    """
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ] = []

    total_length = history_length + forecast_length

    # Compute group_id from key
    group_id = (
        "_".join(str(k) for k in group_key)
        if isinstance(group_key, tuple)
        else str(group_key)
    )

    # Get metadata values (use first row of group - assumes constant within group)
    # Skip columns that are used as covariate timeseries
    group_meta: Dict[str, Any] = {}
    first_row = group_df.iloc[0]
    for col in discrete_cols:
        if col not in covariate_timeseries_cols:
            group_meta[col] = str(first_row[col])
    for col in continuous_cols:
        if col not in covariate_timeseries_cols:
            group_meta[col] = float(first_row[col])

    # Preload covariate signals for this group as numpy arrays
    covariate_signals: Dict[str, np.ndarray] = {}
    for cov_col in covariate_timeseries_cols:
        if cov_col in group_df.columns:
            covariate_signals[cov_col] = group_df[cov_col].to_numpy()

    # Preload split-relevant columns as numpy arrays (avoid repeated pandas access)
    timestamp_values = None
    if timestamp_col and timestamp_col in group_df.columns:
        timestamp_values = group_df[timestamp_col].to_numpy()

    custom_split_values = None
    if has_custom_split and CUSTOM_SPLIT_COLUMN in group_df.columns:
        custom_split_values = group_df[CUSTOM_SPLIT_COLUMN].to_numpy()

    for ts_col in timeseries_cols:
        if ts_col not in group_df.columns:
            continue

        signal = group_df[ts_col].to_numpy()
        if len(signal) < total_length:
            continue

        num_samples = (len(signal) - total_length) // stride + 1

        # Vectorized NaN check for the entire signal
        signal_isnan = np.isnan(signal)

        for i in range(num_samples):
            start = i * stride
            end_history = start + history_length
            end_target = start + total_length

            # Quick NaN check using precomputed mask
            if signal_isnan[start:end_target].any():
                continue

            history = signal[start:end_history].astype(np.float32)
            target = signal[end_history:end_target].astype(np.float32)

            # Extract covariate windows
            covariates: Dict[str, np.ndarray] = {}
            skip_sample = False
            for cov_col, cov_signal in covariate_signals.items():
                if len(cov_signal) < total_length:
                    skip_sample = True
                    break
                cov_history = cov_signal[start:end_history]
                cov_future = cov_signal[end_history:end_target]
                if np.isnan(cov_history).any() or np.isnan(cov_future).any():
                    skip_sample = True
                    break
                covariates[cov_col] = cov_history.astype(np.float32)
                covariates[f"{cov_col}_future"] = cov_future.astype(np.float32)

            if skip_sample:
                continue

            meta: Dict[str, Any] = {
                "dataset": f"{dataset_name}_{ts_col}",
                "group_id": group_id,
                "history_length": history_length,
                "prediction_length": forecast_length,
                **group_meta,
            }

            # Add split-relevant metadata
            if timestamp_values is not None:
                meta["window_start_timestamp"] = str(timestamp_values[start])

            if custom_split_values is not None:
                meta[CUSTOM_SPLIT_COLUMN] = str(custom_split_values[start])

            samples.append((history, target, covariates, meta))

    return samples


def _process_group_wrapper(args: Tuple) -> List[Tuple]:
    """Wrapper for multiprocessing that unpacks arguments."""
    return _process_single_group(*args)


def create_samples_from_dataframe(
    df: pd.DataFrame,
    timeseries_cols: List[str],
    group_cols: List[str],
    discrete_cols: List[str],
    continuous_cols: List[str],
    dataset_name: str,
    history_length: int,
    forecast_length: int,
    stride: int,
    covariate_timeseries_cols: Optional[List[str]] = None,
    timestamp_col: Optional[str] = None,
    has_custom_split: bool = False,
    num_workers: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]]:
    """Create time series samples from a DataFrame.

    Args:
        df: Input DataFrame
        timeseries_cols: Columns containing time series data (target)
        group_cols: Columns to group by (e.g., subject_id)
        discrete_cols: Discrete metadata columns
        continuous_cols: Continuous metadata columns
        dataset_name: Name for the dataset in metadata
        history_length: Number of history time steps
        forecast_length: Number of forecast time steps
        stride: Sliding window stride
        covariate_timeseries_cols: Columns to use as time series covariates
            (will be saved as {col} for history and {col}_future for forecast)
        timestamp_col: Optional timestamp column for timestamp-based splitting.
            If provided, stores window_start_timestamp in metadata.
        has_custom_split: If True, reads custom_split column and stores in metadata.
        num_workers: Number of parallel workers. If None or 0, uses sequential processing.
            For large datasets (>10k groups), parallel processing is recommended.

    Returns:
        List of (history, target, covariates_dict, metadata) tuples
        where covariates_dict contains {col}_history and {col}_future arrays
    """
    covariate_timeseries_cols = covariate_timeseries_cols or []
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ] = []

    # Label encode discrete columns that will be used as covariates
    # Chronos-2 supports categorical covariates as integer-encoded values
    label_encoders: Dict[str, Dict[Any, int]] = {}
    for col in discrete_cols:
        if col in covariate_timeseries_cols and col in df.columns:
            unique_vals = df[col].unique()
            label_encoders[col] = {
                val: idx for idx, val in enumerate(unique_vals)
            }
            df = df.copy()  # Avoid modifying original
            df[col] = df[col].map(label_encoders[col]).astype(np.float32)

    # Filter to existing columns
    discrete_cols = [c for c in discrete_cols if c in df.columns]
    continuous_cols = [c for c in continuous_cols if c in df.columns]

    # Prepare timestamp column if needed
    if timestamp_col and timestamp_col in df.columns:
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Determine groupby columns
    groupby_cols: Optional[Union[str, List[str]]] = None
    if group_cols:
        existing_cols = [c for c in group_cols if c in df.columns]
        if existing_cols:
            # Use single column directly (not list) to avoid tuple/scalar key mismatch
            groupby_cols = (
                existing_cols[0] if len(existing_cols) == 1 else existing_cols
            )

    # If no grouping, process entire dataframe as one group
    if groupby_cols is None:
        return _process_single_group(
            group_key="all",
            group_df=df.reset_index(drop=True),
            timeseries_cols=timeseries_cols,
            discrete_cols=discrete_cols,
            continuous_cols=continuous_cols,
            covariate_timeseries_cols=covariate_timeseries_cols,
            dataset_name=dataset_name,
            history_length=history_length,
            forecast_length=forecast_length,
            stride=stride,
            timestamp_col=timestamp_col,
            has_custom_split=has_custom_split,
        )

    # OPTIMIZED: Iterate directly over groupby object (single pass, O(n) total)
    # instead of get_group() which is O(n) per call
    groupby = df.groupby(groupby_cols, sort=False)
    num_groups = len(groupby)

    # Decide whether to use parallel processing
    use_parallel = (
        num_workers is not None and num_workers > 0 and num_groups > 100
    )

    if use_parallel:
        # Parallel processing using ProcessPoolExecutor
        print(
            f"Using {num_workers} workers for parallel processing of {num_groups} groups"
        )

        # Collect all group data first (needed for multiprocessing)
        # Note: This requires memory but enables true parallelism
        group_args = []
        for group_key, group_df in groupby:
            group_args.append(
                (
                    group_key,
                    group_df.reset_index(drop=True),
                    timeseries_cols,
                    discrete_cols,
                    continuous_cols,
                    covariate_timeseries_cols,
                    dataset_name,
                    history_length,
                    forecast_length,
                    stride,
                    timestamp_col,
                    has_custom_split,
                )
            )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_group_wrapper, args): i
                for i, args in enumerate(group_args)
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing groups (parallel)",
            ):
                group_samples = future.result()
                samples.extend(group_samples)
    else:
        # Sequential processing - still use efficient iteration
        for group_key, group_df in tqdm(
            groupby, total=num_groups, desc="Processing groups"
        ):
            group_samples = _process_single_group(
                group_key=group_key,
                group_df=group_df.reset_index(drop=True),
                timeseries_cols=timeseries_cols,
                discrete_cols=discrete_cols,
                continuous_cols=continuous_cols,
                covariate_timeseries_cols=covariate_timeseries_cols,
                dataset_name=dataset_name,
                history_length=history_length,
                forecast_length=forecast_length,
                stride=stride,
                timestamp_col=timestamp_col,
                has_custom_split=has_custom_split,
            )
            samples.extend(group_samples)

    return samples


def _split_by_ratio(
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ],
    train_ratio: float,
    val_ratio: float,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split samples into train/val/test sets WITHIN each group by ratio.

    Each group will have samples in train, val, and test sets based on the ratios.
    This ensures all groups are represented in all splits.

    Args:
        samples: List of (history, target, covariates, metadata) tuples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (rest goes to test)
        shuffle: Whether to shuffle samples within each group
        seed: Random seed

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    rng = np.random.default_rng(seed)

    # Group samples by group_id (metadata is at index 3)
    group_to_samples: Dict[str, List] = {}
    for sample in samples:
        group_id = sample[3]["group_id"]
        if group_id not in group_to_samples:
            group_to_samples[group_id] = []
        group_to_samples[group_id].append(sample)

    train_samples: List = []
    val_samples: List = []
    test_samples: List = []

    # Split samples within each group
    for group_id, group_samples in group_to_samples.items():
        if shuffle:
            rng.shuffle(group_samples)

        n_samples = len(group_samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_samples.extend(group_samples[:n_train])
        val_samples.extend(group_samples[n_train : n_train + n_val])
        test_samples.extend(group_samples[n_train + n_val :])

    return train_samples, val_samples, test_samples


def _split_by_timestamp(
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ],
    train_end: str,
    val_end: str,
) -> Tuple[List, List, List]:
    """Split samples into train/val/test sets by timestamp cutoffs.

    Args:
        samples: List of (history, target, covariates, metadata) tuples.
            Each sample's metadata must contain 'window_start_timestamp'.
        train_end: ISO format datetime string. Train includes samples < this.
        val_end: ISO format datetime string. Val includes samples >= train_end and < val_end.
            Test includes samples >= val_end.

    Returns:
        Tuple of (train_samples, val_samples, test_samples)

    Raises:
        ValueError: If samples don't have window_start_timestamp metadata.
    """
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    train_samples: List = []
    val_samples: List = []
    test_samples: List = []

    for sample in samples:
        meta = sample[3]
        if "window_start_timestamp" not in meta:
            raise ValueError(
                "Timestamp-based splitting requires 'window_start_timestamp' in metadata. "
                "Ensure 'timestamp_column' is specified in train_val_split config."
            )

        window_ts = pd.to_datetime(meta["window_start_timestamp"])

        if window_ts < train_end_dt:
            train_samples.append(sample)
        elif window_ts < val_end_dt:
            val_samples.append(sample)
        else:
            test_samples.append(sample)

    return train_samples, val_samples, test_samples


def _split_by_custom_column(
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ],
) -> Tuple[List, List, List]:
    """Split samples into train/val/test sets using pre-existing split labels (BYOS).

    Args:
        samples: List of (history, target, covariates, metadata) tuples.
            Each sample's metadata must contain 'custom_split' with value
            in {"train", "val", "test"}.

    Returns:
        Tuple of (train_samples, val_samples, test_samples)

    Raises:
        ValueError: If samples don't have custom_split metadata or have invalid values.
    """
    train_samples: List = []
    val_samples: List = []
    test_samples: List = []

    for sample in samples:
        meta = sample[3]
        if CUSTOM_SPLIT_COLUMN not in meta:
            raise ValueError(
                f"BYOS splitting requires '{CUSTOM_SPLIT_COLUMN}' in metadata. "
                f"Ensure your DataFrame has a column named '{CUSTOM_SPLIT_COLUMN}'."
            )

        split_value = meta[CUSTOM_SPLIT_COLUMN].lower().strip()

        if split_value not in VALID_SPLIT_VALUES:
            raise ValueError(
                f"Invalid split value '{split_value}' in {CUSTOM_SPLIT_COLUMN} column. "
                f"Valid values are: {VALID_SPLIT_VALUES}"
            )

        if split_value == "train":
            train_samples.append(sample)
        elif split_value == "val":
            val_samples.append(sample)
        else:  # test
            test_samples.append(sample)

    return train_samples, val_samples, test_samples


def split_samples_unified(
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ],
    split_config: Dict[str, Any],
    has_custom_split: bool = False,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Unified sample splitting supporting three methods.

    Method selection (priority order):
    1. If has_custom_split=True → BYOS using metadata["custom_split"]
    2. If split_config["method"] == "timestamp" → timestamp-based
    3. Otherwise → ratio-based using split_config["train"]/["val"]

    Args:
        samples: List of (history, target, covariates, metadata) tuples.
        split_config: Configuration dict with split parameters.
            For ratio: {"train": 0.8, "val": 0.1, "shuffle": true}
            For timestamp: {"method": "timestamp", "timestamp_column": "ts",
                           "train_end": "2023-01-01", "val_end": "2023-06-01"}
        has_custom_split: If True, uses BYOS method (auto-detected from DataFrame).
        seed: Random seed for ratio-based splitting.

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # Priority 1: BYOS (custom_split column detected)
    if has_custom_split:
        print(
            "Using BYOS (Bring Your Own Split) method - detected 'custom_split' column"
        )
        return _split_by_custom_column(samples)

    # Priority 2: Timestamp-based (explicit method in config)
    method = split_config.get("method", "ratio")
    if method == "timestamp":
        timestamp_col = split_config.get("timestamp_column")
        train_end = split_config.get("train_end")
        val_end = split_config.get("val_end")

        if not all([timestamp_col, train_end, val_end]):
            raise ValueError(
                "Timestamp-based splitting requires 'timestamp_column', 'train_end', and 'val_end' "
                "in train_val_split config."
            )

        # Type assertions for pyright (validated above)
        assert isinstance(train_end, str)
        assert isinstance(val_end, str)

        print(
            f"Using timestamp-based splitting: train < {train_end}, val < {val_end}, test >= {val_end}"
        )
        return _split_by_timestamp(samples, train_end, val_end)

    # Priority 3: Ratio-based (default)
    train_ratio = split_config.get("train", 0.8)
    val_ratio = split_config.get("val", 0.1)
    shuffle = split_config.get("shuffle", True)

    print(
        f"Using ratio-based splitting: train={train_ratio}, val={val_ratio}, shuffle={shuffle}"
    )
    return _split_by_ratio(samples, train_ratio, val_ratio, shuffle, seed)


# Keep the old function for backward compatibility
def split_samples_by_group(
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ],
    train_ratio: float,
    val_ratio: float,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split samples into train/val/test sets WITHIN each group.

    DEPRECATED: Use split_samples_unified() instead.

    Each group will have samples in train, val, and test sets based on the ratios.
    This ensures all groups are represented in all splits.

    Args:
        samples: List of (history, target, covariates, metadata) tuples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (rest goes to test)
        shuffle: Whether to shuffle samples within each group
        seed: Random seed

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    return _split_by_ratio(samples, train_ratio, val_ratio, shuffle, seed)


def write_sharded_dataset(
    samples: List[
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
    ],
    output_dir: str,
    samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD,
    covariate_fields: Optional[List[str]] = None,
) -> None:
    """Write samples to sharded dataset format.

    Args:
        samples: List of (history, target, covariates, metadata) tuples
        output_dir: Output directory path
        samples_per_shard: Samples per tar shard (default: 10000)
        covariate_fields: List of covariate field names (will add {name} and {name}_future)
    """
    if not samples:
        return

    covariate_fields = covariate_fields or []

    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)

    writer = ShardedDatasetWriter(
        output_dir, samples_per_shard=samples_per_shard
    )

    # Register all fields: history, target, plus covariates
    fields = ["history", "target"]
    for cov_field in covariate_fields:
        fields.append(cov_field)  # history portion
        fields.append(f"{cov_field}_future")  # future portion
    writer.add_fields(fields)

    for history, target, covariates, meta in tqdm(
        samples, desc=f"Writing {Path(output_dir).name}"
    ):
        arrays = {"history": history, "target": target}
        # Add covariate arrays
        arrays.update(covariates)
        writer.add_sample(
            arrays=arrays,
            metadata=meta,
        )

    writer.close()


def convert_parquet_to_sharded(
    config_path: str,
    data_base_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    samples_per_shard: Optional[int] = None,
) -> Tuple[str, str, str]:
    """Convert a parquet file to sharded dataset format.

    Args:
        config_path: Path to the preprocessing config JSON file
        data_base_path: Base path for data files (default: $SYNTHEFY_DATASETS_BASE)
        output_dir: Output directory (default: sharded/ under $SYNTHEFY_DATASETS_BASE)
        num_workers: Number of parallel workers for processing groups.
            If None, uses config value or defaults to 8 for large datasets (>10k groups).
        samples_per_shard: Number of samples per tar shard.
            If None, uses config value or defaults to 10000.

    Returns:
        Tuple of (train_dir, val_dir, test_dir) paths
    """
    # Resolve base path
    if data_base_path is not None:
        base_path = Path(data_base_path).expanduser()
    elif DEFAULT_DATASETS_BASE is not None:
        base_path = Path(DEFAULT_DATASETS_BASE).expanduser()
    else:
        raise ValueError(
            "Data base path must be provided via argument or SYNTHEFY_DATASETS_BASE environment variable."
        )

    # Load config
    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    # Resolve paths
    parquet_path = base_path / config["filename"]
    if output_dir:
        output_base = Path(output_dir).expanduser()
    else:
        output_base = parquet_path.parent / "sharded"

    # Get params from config
    window_size = config.get("window_size", 256)
    stride = config.get("stride", 32)

    # Get history/forecast split from config
    history_forecast_split = config.get(
        "history_forecast_split",
        {"history_ratio": 0.8},  # Default 80% history, 20% forecast
    )

    # Direct history_length specification
    if "history_length" in history_forecast_split:
        history_length = int(history_forecast_split["history_length"])
        if history_length >= window_size:
            raise ValueError(
                f"history_length ({history_length}) must be less than window_size ({window_size})"
            )
        forecast_length = window_size - history_length
    # Ratio-based (backward compatibility)
    else:
        history_ratio = history_forecast_split.get("history_ratio", 0.8)
        history_length = int(window_size * history_ratio)
        forecast_length = window_size - history_length

    # Get train/val/test split config
    train_val_split = config.get("train_val_split", {"train": 0.8, "val": 0.1})
    shuffle = config.get("shuffle", True)
    seed = config.get("seed", 42)

    # Add shuffle to split config if not already there
    if "shuffle" not in train_val_split:
        train_val_split["shuffle"] = shuffle

    timeseries_cols = config.get("timeseries", {}).get("cols", [])
    group_cols = config.get("group_labels", {}).get("cols", [])
    discrete_cols = config.get("discrete", {}).get("cols", [])
    continuous_cols = config.get("continuous", {}).get("cols", [])
    # Covariate timeseries: use continuous columns as time series covariates (numeric only)
    # Discrete columns remain as scalar metadata (categorical/string)
    covariate_timeseries_cols = continuous_cols
    dataset_name = Path(config["filename"]).stem

    # Get num_workers from argument or config (default to 8 for large datasets)
    if num_workers is None:
        num_workers = config.get("num_workers", DEFAULT_NUM_WORKERS)

    # Get samples_per_shard from argument or config
    if samples_per_shard is None:
        samples_per_shard = config.get("samples_per_shard")
    if samples_per_shard is None:
        samples_per_shard = DEFAULT_SAMPLES_PER_SHARD

    if not timeseries_cols:
        raise ValueError("No timeseries columns in config.timeseries.cols")

    # Load data
    print(f"Loading parquet from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows")
    print(
        f"Window params - History: {history_length}, "
        f"Forecast: {forecast_length}, Stride: {stride}"
    )
    if discrete_cols:
        print(f"Discrete metadata: {discrete_cols}")
    if continuous_cols:
        print(f"Continuous metadata: {continuous_cols}")
    if covariate_timeseries_cols:
        print(f"Covariate timeseries: {covariate_timeseries_cols}")

    # Detect split method and get relevant column info
    has_custom_split = CUSTOM_SPLIT_COLUMN in df.columns
    if has_custom_split:
        print(
            f"Detected '{CUSTOM_SPLIT_COLUMN}' column - will use BYOS splitting"
        )

    # Get timestamp column for timestamp-based splitting
    timestamp_col = None
    if train_val_split.get("method") == "timestamp":
        timestamp_col = train_val_split.get("timestamp_column")
        if not timestamp_col:
            raise ValueError(
                "Timestamp-based splitting requires 'timestamp_column' in train_val_split config."
            )
        if timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        print(f"Using timestamp column '{timestamp_col}' for splitting")

    # Create samples
    samples = create_samples_from_dataframe(
        df=df,
        timeseries_cols=timeseries_cols,
        group_cols=group_cols,
        discrete_cols=discrete_cols,
        continuous_cols=continuous_cols,
        dataset_name=dataset_name,
        history_length=history_length,
        forecast_length=forecast_length,
        stride=stride,
        covariate_timeseries_cols=covariate_timeseries_cols,
        timestamp_col=timestamp_col,
        has_custom_split=has_custom_split,
        num_workers=num_workers,
    )

    if not samples:
        raise ValueError("No samples created! Check config and data.")

    print(f"Created {len(samples)} samples")

    # Split using unified function
    train_samples, val_samples, test_samples = split_samples_unified(
        samples=samples,
        split_config=train_val_split,
        has_custom_split=has_custom_split,
        seed=seed,
    )
    print(
        f"Split: {len(train_samples)} train, {len(val_samples)} val, "
        f"{len(test_samples)} test"
    )

    # Write datasets
    train_dir = str(output_base / f"{dataset_name}_train")
    val_dir = str(output_base / f"{dataset_name}_val")
    test_dir = str(output_base / f"{dataset_name}_test")

    write_sharded_dataset(
        train_samples,
        train_dir,
        samples_per_shard=samples_per_shard,
        covariate_fields=covariate_timeseries_cols,
    )
    print(f"Created {train_dir}")

    write_sharded_dataset(
        val_samples,
        val_dir,
        samples_per_shard=samples_per_shard,
        covariate_fields=covariate_timeseries_cols,
    )
    print(f"Created {val_dir}")

    write_sharded_dataset(
        test_samples,
        test_dir,
        samples_per_shard=samples_per_shard,
        covariate_fields=covariate_timeseries_cols,
    )
    print(f"Created {test_dir}")

    # Build covariate fields info for output
    cov_fields_str = ""
    if covariate_timeseries_cols:
        cov_fields_str = f"""
  # Covariate fields available: {covariate_timeseries_cols}
  past_covariate_fields: {covariate_timeseries_cols}
  future_covariate_fields: {covariate_timeseries_cols}  # if using future covariates"""

    print("\nDone! Use in fine-tuning config:")
    print(f"""
data:
  train_sharded_datasets:
    - {train_dir}
  val_sharded_datasets:
    - {val_dir}
  test_sharded_datasets:
    - {test_dir}
  max_history: {history_length}
  max_forecast: {forecast_length}{cov_fields_str}
""")

    return train_dir, val_dir, test_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert parquet to sharded dataset format"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to preprocessing config JSON file",
    )
    parser.add_argument(
        "--data-base-path",
        type=str,
        default=None,
        help=f"Base path for data files (default: $SYNTHEFY_DATASETS_BASE or {DEFAULT_DATASETS_BASE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: sharded/ under data base path)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"Number of parallel workers for processing groups (default: {DEFAULT_NUM_WORKERS}). "
        "Set to 0 for sequential processing.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=None,
        help=f"Number of samples per tar shard (default: {DEFAULT_SAMPLES_PER_SHARD}).",
    )

    args = parser.parse_args()
    convert_parquet_to_sharded(
        config_path=args.config,
        data_base_path=args.data_base_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        samples_per_shard=args.samples_per_shard,
    )


if __name__ == "__main__":
    main()
