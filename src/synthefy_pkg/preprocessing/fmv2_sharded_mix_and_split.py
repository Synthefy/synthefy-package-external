import argparse
import glob
import io
import json
import os
import pickle
import tarfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from synthefy_pkg.preprocessing.fmv2_preprocess import (
    SCALING_FEATURES,
    TIMESTAMPS_FEATURES,
)

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1


def denormalize_timestamp_features(
    normalized_features,
    year_base=1970,
    year_range=2030 - 1970,
    as_seconds: bool = False,
):
    """Convert normalized timestamp features back to datetime values in a vectorized way.

    Args:
        normalized_features: Numpy array of normalized timestamp features with shape
                            (window_num, len(TIMESTAMPS_FEATURES)) or just (len(TIMESTAMPS_FEATURES),)
        year_base: Base year used in normalization (default: 1970)
        year_range: Range of years used in normalization (default: 60 years)

    Returns:
        pd.DatetimeIndex or pd.Timestamp: Reconstructed timestamps
    """
    # Handle single vector case
    single_vector = normalized_features.ndim == 1
    if single_vector:
        normalized_features = normalized_features.reshape(1, -1)

    # Ensure proper feature count
    feature_count = normalized_features.shape[1]
    assert feature_count == len(TIMESTAMPS_FEATURES), (
        f"Expected {len(TIMESTAMPS_FEATURES)} features, got {feature_count}"
    )

    # Create a dictionary mapping feature names to their column indices
    feature_indices = {name: i for i, name in enumerate(TIMESTAMPS_FEATURES)}

    # Vectorized denormalization for all rows
    years = np.round(
        normalized_features[:, feature_indices["year"]] * year_range + year_base
    ).astype(int)
    months = np.round(
        normalized_features[:, feature_indices["month"]] * 11 + 1
    ).astype(int)
    days = np.round(
        normalized_features[:, feature_indices["day"]] * 30 + 1
    ).astype(int)
    hours = np.round(
        normalized_features[:, feature_indices["hour"]] * 23
    ).astype(int)
    minutes = np.round(
        normalized_features[:, feature_indices["minute"]] * 59
    ).astype(int)
    seconds = np.round(
        normalized_features[:, feature_indices["second"]] * 59
    ).astype(int)

    if as_seconds:
        multipliers = np.array(
            [1, 60, 3600, 86400, 2592000, 31536000]
        )  # [1, 60, 60*60, 24*60*60, 30*24*60*60, 365*24*60*60]
        time_components = np.stack(
            [seconds, minutes, hours, days, months, years], axis=-1
        )
        return np.sum(time_components * multipliers, axis=-1)
    # Create a DatetimeIndex using vectorized operations
    # This will automatically handle invalid dates by creating NaT values
    timestamps = pd.to_datetime(
        {
            "year": years,
            "month": months,
            "day": days,
            "hour": hours,
            "minute": minutes,
            "second": seconds,
        }
    )

    # Find invalid dates (NaT values)
    invalid_mask = pd.isna(timestamps)

    if np.any(invalid_mask):
        raise ValueError("Invalid timestamps found")

    # If input was a single vector, return single timestamp instead of DatetimeIndex
    if single_vector:
        return timestamps[0]

    return timestamps


def check_timestamp_leq(
    timeseries_array: np.ndarray, window_size: int, max_timeval: pd.Timestamp
) -> np.ndarray:
    """
    Check if the last timestamp in the timeseries array are less than or equal to max_timeval.
    TODO: handle timezones

    timeseries_array is of shape (num_windows, num_features)
    returns a boolean array of shape (num_windows,)
    """
    return np.array(
        denormalize_timestamp_features(
            timeseries_array[
                :,
                len(SCALING_FEATURES)
                + len(TIMESTAMPS_FEATURES) * (window_size - 1) : len(
                    SCALING_FEATURES
                )
                + len(TIMESTAMPS_FEATURES) * (window_size),
            ]
        )
        <= max_timeval
    )


def retrieve_timestamp_from_window(
    window: np.ndarray,
    window_size: int,
    idx: int = -1,
    as_seconds: bool = False,
) -> np.ndarray:
    """
    Retrieve the timestamp from the last window
    """
    if idx == -1:
        idx = window_size - 1
    if len(window.shape) == 1:
        return np.array(
            denormalize_timestamp_features(
                window[
                    len(SCALING_FEATURES)
                    + len(TIMESTAMPS_FEATURES) * (idx) : len(SCALING_FEATURES)
                    + len(TIMESTAMPS_FEATURES) * (idx + 1),
                ],
                as_seconds=as_seconds,
            )
        )
    else:
        return np.array(
            denormalize_timestamp_features(
                window[
                    :,
                    len(SCALING_FEATURES)
                    + len(TIMESTAMPS_FEATURES) * (idx) : len(SCALING_FEATURES)
                    + len(TIMESTAMPS_FEATURES) * (idx + 1),
                ],
                as_seconds=as_seconds,
            )
        )


def load_preprocessed_data(
    data_dir: str,
    num_scalars: int,
) -> Tuple[np.ndarray, str, str, np.ndarray, np.ndarray, int, int, int]:
    """
    Load a shard worth of preprocessed data from a directory.

    Args:
        data_dir: Path to directory containing preprocessed data
        num_scalars: Number of scalars to extract from the start of the timeseries
    Returns:
        Tuple containing timeseries data, metadata, and scalers

    Raises:
        FileNotFoundError: If any required files are not found or cannot be loaded
    """
    # get the name of the last folder in the path
    dataset_dir_name = os.path.basename(data_dir)
    dataset_id = int(dataset_dir_name.split("_")[-1].strip("/"))
    timeseries_path = os.path.join(data_dir, "preprocessed_array.npy")
    # load metdata so that we can keep track of column names
    metadata_path = os.path.join(data_dir, f"{dataset_dir_name}_metadata.json")
    embedding_path = os.path.join(data_dir, "description_embedding.npy")

    # Check if all files exist
    missing_files = []
    for path, name in [
        (timeseries_path, "preprocessed_array.npy"),
        (metadata_path, f"{dataset_dir_name}_metadata.json"),
        (embedding_path, "description_embedding.npy"),
    ]:
        if not os.path.exists(path):
            missing_files.append(name)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {data_dir}: {', '.join(missing_files)}"
        )

    # Try to load each file, raising an error if any fail
    try:
        timeseries = np.load(timeseries_path)
        scalars = timeseries[0, :num_scalars]

        metadata = json.load(open(metadata_path, "r"))
        title = metadata["columns"][0][
            "title"
        ]  # TODO: only extracts title for now
        frequency = metadata["frequency"]
        embedding = np.load(embedding_path)
        num_rows = int(metadata["length"])
        num_windows = len(timeseries)
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {data_dir}: {str(e)}")

    logger.info(f"Loaded {len(timeseries)} samples from {data_dir}")

    # Make sure all are non empty:
    if len(timeseries) == 0:
        raise ValueError(f"{data_dir} has no timeseries data, skipping")
    if len(title) == 0:
        raise ValueError(f"{data_dir} has no metadata data, skipping")
    if len(embedding) == 0:
        raise ValueError(f"{data_dir} has no scalers data, skipping")

    return (
        timeseries,
        title,
        frequency,
        embedding,
        scalars,
        num_rows,
        num_windows,
        dataset_id,
    )


def shuffle_and_split_data(
    timeseries: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_seed: Optional[int] = None,
    shuffle: bool = True,
    timestamp_split: pd.Timestamp | None = None,
    window_size: int = 256,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[Dict[str, List[int]]],
]:
    """
    Shuffle and split data into train, validation, and test sets.

    Args:
        timeseries: Timeseries data to split
        metadata: Metadata corresponding to timeseries
        scalers: List of scalers corresponding to timeseries
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        random_seed: Optional random seed for reproducibility
        shuffle: Whether to shuffle the data before splitting

    Returns:
        Tuple of (train_timeseries, train_metadata, train_scalers),
              (val_timeseries, val_metadata, val_scalers),
              (test_timeseries, test_metadata, test_scalers),
              shuffle_mapping (dictionary with indices for each split, or None if no shuffling was done)
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize shuffle mapping as None
    shuffle_mapping = None

    # Calculate split boundaries
    n_samples = len(timeseries)
    train_end = int(np.ceil(n_samples * train_ratio))
    val_end = train_end + int(np.ceil(n_samples * val_ratio))

    # Shuffle data if requested
    if shuffle:
        indices = np.random.permutation(len(timeseries))

        # Create mapping of original indices to split
        shuffle_mapping = {
            "train": indices[:train_end].tolist(),
            "val": indices[train_end:val_end].tolist(),
            "test": indices[val_end:].tolist(),
        }

        timeseries = timeseries[indices]

    # Create splits
    if timestamp_split is not None:
        # the trainable bool will indicate which values can be trained on
        trainable_bool = check_timestamp_leq(
            timeseries, window_size, timestamp_split
        )
        train_timeseries = timeseries[trainable_bool]

        # get the last True in trainable_bool
        train_end = np.where(trainable_bool)[0]
        if len(train_end) > 0:
            train_end = train_end[-1] + 1
        else:
            train_end = 0
        train_timeseries = timeseries[:train_end]

        # assert that the trainable_bool is all True then all False
        assert np.all(trainable_bool[:train_end]) and (
            np.all(~trainable_bool[train_end:])
            or len(trainable_bool[train_end:]) == 0
        ), "trainable_bool must be all True then all False"

        # split up the remaining values into val and test
        val_or_test_timeseries = timeseries[train_end:]
        n_val_or_test = len(val_or_test_timeseries)
        adjusted_val_ratio = val_ratio / (1 - train_ratio)
        val_timeseries = val_or_test_timeseries[
            : int(n_val_or_test * adjusted_val_ratio)
        ]
        test_timeseries = val_or_test_timeseries[
            int(n_val_or_test * adjusted_val_ratio) :
        ]
    else:
        train_timeseries = timeseries[:train_end]

        val_timeseries = timeseries[train_end:val_end]

        test_timeseries = timeseries[val_end:]

    return (
        (train_timeseries),
        (val_timeseries),
        (test_timeseries),
        shuffle_mapping,
    )


def _save_split_data(
    output_subdir: str,
    split_name: str,
    split_dict: Dict[str, Any],
    dataset_indices: Dict[str, Dict[str, int]],
    dataset_shuffle_indices: Dict[str, List[int]],
    shard_num: int = 0,
) -> int:
    """
    Save a single split of data to disk.

    Args:
        output_subdir: Directory to save the data
        split_name: Name of the split
        split_dict: Dictionary containing the split data
            of form:
            split_name: {
                "timeseries": [],
            }

        dataset_indices: Dictionary containing the dataset indices
            of form:
            split_name: {
                "dataset_name": {
                    "start": ...,
                    "end": ...,
                },
                "dataset_name": {
                    "start": ...,
                    "end": ...,
                },
            }
        dataset_shuffle_indices: Dictionary containing the shuffle indices
            of form:
            split_name: {
                "dataset_name": [...],
                "dataset_name": [...],
            }
        shard_num: Shard number to use in filename
    Returns:
        Number of samples in the split
    """
    # Create shard suffix if needed
    shard_suffix = f"_shard_{shard_num}"  # if shard_num > 0 else ""

    # Save Timeseries and Metadata
    timeseries_data = np.concatenate(split_dict["timeseries"], axis=0)

    # Save to disk

    # save as tar.gz
    with tarfile.open(
        os.path.join(
            output_subdir, f"{split_name}_timeseries{shard_suffix}.tar"
        ),
        "w",
    ) as tar:
        # save as separate npy files
        for i in range(timeseries_data.shape[0]):
            # Create BytesIO buffer
            buf = io.BytesIO()
            # Save array to buffer
            np.save(buf, timeseries_data[i : (i + 1)])
            buf.seek(0)

            # Create tarinfo
            tarinfo = tarfile.TarInfo(
                f"{split_name}_timeseries{shard_suffix}_{i}.npy"
            )
            tarinfo.size = buf.getbuffer().nbytes

            # Add buffer to tar
            tar.addfile(tarinfo, buf)

    # Save dataset indices
    with open(
        os.path.join(
            output_subdir, f"{split_name}_dataset_indices{shard_suffix}.json"
        ),
        "w",
    ) as f:
        json.dump(dataset_indices, f, indent=2)

    # Save shuffle indices if they exist
    if dataset_shuffle_indices:
        with open(
            os.path.join(
                output_subdir,
                f"{split_name}_shuffle_indices{shard_suffix}.json",
            ),
            "w",
        ) as f:
            json.dump(dataset_shuffle_indices, f)
        logger.info(
            f"Saved {split_name} shuffle indices to {output_subdir}/{split_name}_shuffle_indices{shard_suffix}.json"
        )

    # Log and return count
    sample_count = len(timeseries_data)
    logger.info(
        f"Saved {sample_count} {split_name} samples to {output_subdir} (shard {shard_num})"
    )
    return sample_count


class DataMixerAndSplitter:
    def __init__(
        self,
        config_path: str,
        allow_pretrain_blind_overlap: bool = False,
    ):
        """
        Initialize the DataMixerAndSplitter.

        Args:
            config_path: Path to the configuration YAML file
            max_batch_size: Maximum number of windows to keep in memory before saving
        """

        self.config_path = config_path
        self.allow_pretrain_blind_overlap = allow_pretrain_blind_overlap
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._validate_config()

        # seed based on config seed
        self.random_seed = self.config.get("random_seed", 42)
        np.random.seed(self.random_seed)

        self.max_shard_size = self.config.get("max_shard_size", 40000)

        self.output_dir = self.config["output_dir"]
        self.input_base_dir = self.config["input_base_dir"]
        self.num_scalars = self.config.get("num_scalars", 3)

        # Get shuffle parameter
        self.shuffle = self.config.get("shuffle", True)

        # Set split ratios
        self.train_ratio = self.config.get("train_ratio", DEFAULT_TRAIN_RATIO)
        self.val_ratio = self.config.get("val_ratio", DEFAULT_VAL_RATIO)
        self.test_ratio = self.config.get(
            "test_ratio", 1 - self.train_ratio - self.val_ratio
        )

        # Validate split ratios
        assert 0 < self.train_ratio < 1, "train_ratio must be between 0 and 1"
        assert 0 < self.val_ratio < 1, "val_ratio must be between 0 and 1"
        assert self.train_ratio + self.val_ratio < 1, (
            "train + val ratio must be < 1"
        )

        self.timestamp_split = self.config.get("timestamp_split", None)
        if self.timestamp_split is not None:
            try:
                self.timestamp_split = pd.Timestamp(self.timestamp_split)
            except Exception as e:
                self.timestamp_split = None
                logger.warning(f"Invalid timestamp_split: {e}, using None")
        self.window_size = self.config.get("window_size", 256)

    def _validate_config(self) -> None:
        """
        Validate the configuration.
        """
        assert self.config.get("selected_datasets"), (
            "selected_datasets must be provided"
        )
        assert self.config.get("output_dir"), "output_dir must be provided"
        assert self.config.get("input_base_dir"), (
            "input_base_dir must be provided"
        )
        assert self.config.get("train_ratio"), "train_ratio must be provided"

        pretrain_datasets = set()
        if not self.allow_pretrain_blind_overlap:
            for dataset_config in self.config["selected_datasets"]:
                if dataset_config["usage"] == "pretrain":
                    pretrain_datasets.add(dataset_config["dataset_name"])
                elif (
                    dataset_config["usage"] != "pretrain"
                    and dataset_config["dataset_name"] in pretrain_datasets
                ):
                    string_to_print = (
                        f"Dataset {dataset_config['dataset_name']} is specified as pretrain and {dataset_config['dataset_name']}"
                        f" is specified as {dataset_config['usage']}. If you want to override this check,"
                        f" add --allow-pretrain-blind-overlap"
                    )
                    raise ValueError(string_to_print)

    def mix_and_split(
        self,
        dataset_set: str,
        dict_only: bool = False,
    ) -> Tuple[int, Dict[str, int]]:
        """
        Mix data from multiple datasets and split into train/val/test sets.
        Saves data in shards to avoid memory issues.

        Args:
            dataset_set: Type of dataset set to process (e.g., "pretrain", "blind")
        """
        # Create output directory if it doesn't exist
        output_subdir = os.path.join(self.output_dir, dataset_set)
        os.makedirs(output_subdir, exist_ok=True)

        # Initialize storage for current shard data
        split_data = {
            "train": {"timeseries": []},
            "val": {"timeseries": []},
            "test": {"timeseries": []},
        }

        # Track dataset indices in the final arrays
        dataset_indices = {"train": {}, "val": {}, "test": {}}
        running_indices = {"train": 0, "val": 0, "test": 0}

        # Track the current shard number for each split
        current_shard = {"train": 0, "val": 0, "test": 0}

        # Track current counts in each split
        current_counts = {"train": 0, "val": 0, "test": 0}
        total_counts = {"train": 0, "val": 0, "test": 0}
        total_rows = 0

        # Store shuffle indices for each dataset and split
        dataset_shuffle_indices = {"train": {}, "val": {}, "test": {}}

        # Process each dataset according to the algorithm
        selected_datasets = sorted(
            self.config.get("selected_datasets", []),
            key=lambda x: x["dataset_name"],
        )

        all_dataset_ids = [
            dataset_config["dataset_name"]
            for dataset_config in selected_datasets
        ]
        all_dataset_ids = [
            int(dataset_id.split("_")[-1]) for dataset_id in all_dataset_ids
        ]
        # create a lookup table for dataset_id to title and embedding and scalars
        # returns a list of tuples of the form (title, frequency, embedding, scalars, num_rows, num_windows)
        dataset_lookup: list[
            tuple[str, str, np.ndarray, np.ndarray, int, int]
        ] = [
            ("", "", np.zeros(0), np.zeros(0), 0, 0)
            for _ in range(np.max(all_dataset_ids) + 1)
        ]

        shard_size_lookup = list()

        for dataset_config in selected_datasets:
            dataset_name = dataset_config["dataset_name"]
            # Skip datasets that don't match the requested set
            if dataset_config["usage"] != dataset_set:
                continue

            dataset_dir = os.path.join(self.input_base_dir, dataset_name)
            if not os.path.exists(dataset_dir):
                logger.warning(
                    f"Dataset directory not found: {dataset_dir}, skipping"
                )
                continue
            logger.info(
                f"Processing dataset: {dataset_name} (usage: {dataset_config['usage']})"
            )

            # Load preprocessed data
            (
                timeseries,
                title,
                frequency,
                embedding,
                scalars,
                num_rows,
                num_windows,
                dataset_id,
            ) = load_preprocessed_data(dataset_dir, self.num_scalars)

            dataset_lookup[dataset_id] = (
                title,
                frequency,
                embedding,
                scalars,
                num_rows,
                num_windows,
            )
            total_rows += num_rows
            # TODO - sampling? how do we use windows_per_million_mixed_windows?

            # Use the function to shuffle and split data
            train_data, val_data, test_data, shuffle_mapping = (
                shuffle_and_split_data(
                    timeseries,
                    self.train_ratio,
                    self.val_ratio,
                    self.random_seed,
                    self.shuffle,
                    timestamp_split=self.timestamp_split,
                    window_size=self.window_size,
                )
            )

            # Store shuffle mapping if shuffling was done
            if shuffle_mapping is not None:
                for split_name, indices in shuffle_mapping.items():
                    dataset_shuffle_indices[split_name][dataset_name] = indices

            # Check if adding this dataset would exceed shard size and save if needed
            for split_name, data in zip(
                ["train", "val", "test"], [train_data, val_data, test_data]
            ):
                split_ts = data

                # Skip if no data for this split
                if len(split_ts) == 0:
                    logger.warning(
                        f"Dataset {dataset_name} has no data for {split_name} split, skipping"
                    )
                    continue

                # Check if adding this dataset would exceed max shard size
                if (
                    current_counts[split_name] > 0
                    and current_counts[split_name] + len(split_ts)
                    > self.max_shard_size
                ):
                    logger.info(
                        f"Saving shard {current_shard[split_name]} for {split_name} set"
                    )
                    # Save current shard
                    samples_saved = _save_split_data(
                        output_subdir=output_subdir,
                        split_name=split_name,
                        split_dict=split_data[split_name],
                        dataset_indices=dataset_indices[split_name],
                        dataset_shuffle_indices=dataset_shuffle_indices[
                            split_name
                        ],
                        shard_num=current_shard[split_name],
                    )
                    total_counts[split_name] += samples_saved

                    # Reset for next shard
                    split_data[split_name] = {
                        "timeseries": [],
                    }
                    dataset_indices[split_name] = {}
                    running_indices[split_name] = 0
                    current_counts[split_name] = 0
                    current_shard[split_name] += 1
                    shard_size_lookup.append(samples_saved)

                # Add to current shard
                split_data[split_name]["timeseries"].append(split_ts)

                # Record dataset indices
                start_idx = running_indices[split_name]
                end_idx = start_idx + len(split_ts)
                dataset_indices[split_name][dataset_name] = {
                    "start": start_idx,
                    "end": end_idx,
                }
                running_indices[split_name] = end_idx
                current_counts[split_name] += len(split_ts)

        # Save any remaining data in each split
        for split_name in ["train", "val", "test"]:
            if current_counts[split_name] > 0:
                samples_saved = _save_split_data(
                    output_subdir=output_subdir,
                    split_name=split_name,
                    split_dict=split_data[split_name],
                    dataset_indices=dataset_indices[split_name],
                    dataset_shuffle_indices=dataset_shuffle_indices[split_name],
                    shard_num=current_shard[split_name],
                )
                total_counts[split_name] += samples_saved
                shard_size_lookup.append(samples_saved)
        # save the dataset_lookup to disk
        with open(os.path.join(output_subdir, "dataset_lookup.pkl"), "wb") as f:
            pickle.dump(dataset_lookup, f)

        # save the shard_size_lookup to disk
        with open(
            os.path.join(output_subdir, "shard_size_lookup.pkl"), "wb"
        ) as f:
            pickle.dump(np.array(shard_size_lookup), f)

        # Save manifest
        saved_config = {
            "config": self.config,
        }
        with open(os.path.join(output_subdir, "saved_config.yaml"), "w") as f:
            yaml.dump(saved_config, f)

        logger.info(
            f"Data mixing and splitting complete for {dataset_set} set:"
        )
        for split_name in ["train", "val", "test"]:
            logger.info(
                f"  {split_name.capitalize()} samples: {total_counts[split_name]}"
            )
        logger.info(f" Cumulative Total rows: {total_rows}")
        logger.info(f"  Saved to: {output_subdir}")
        return total_rows, total_counts


def main():
    """
    Main function to run the data mixing and splitting from command line.

    Usage:
        uv run -m src.synthefy_pkg.preprocessing.fm_mix_and_split
            --config_path /path/to/config.yaml

    example config format:
    ```YAML
        input_base_dir: /home/data/enriched_datasets/
        output_dir: /home/raimi2/data/foundation_model_data_tmp/

        train_ratio: 0.8
        val_ratio: 0.1
        test_ratio: 0.1

        selected_datasets:
        - dataset_name: CPI_partial/CPI_partial_0
            windows_per_million_mixed_windows: 1_000
            usage: pretrain

        - dataset_name: fred_cpi_200/fred_cpi_200_0/
            windows_per_million_mixed_windows: 1_000
            usage: pretrain

        - dataset_name: CPI_partial/CPI_partial_0
            windows_per_million_mixed_windows: 1_000
            usage: blind
    ```

    Also saves a dataset lookup table as dataset_lookup.pkl with the following elements for each dataset:
        title (in text),
        frequency (in text),
        embedding (using sentence embedder),
        scalars (numpy array),
        num_rows from this dataset (int),
        num_windows from this dataset (int),

    """
    parser = argparse.ArgumentParser(
        description="Mix and split preprocessed data for foundation models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--allow-pretrain-blind-overlap",
        action="store_true",
        help="Allow pretrain and blind overlap",
    )

    args = parser.parse_args()

    mixer = DataMixerAndSplitter(
        config_path=args.config,
        allow_pretrain_blind_overlap=(
            args.allow_pretrain_blind_overlap
            if args.allow_pretrain_blind_overlap
            else False
        ),
    )

    mix_and_split_results = {}
    for dataset_set in ["pretrain", "blind"]:
        logger.info(
            f"Mixing and splitting data according to config: {args.config} On the {dataset_set} set"
        )
        mix_and_split_results[dataset_set] = mixer.mix_and_split(
            dataset_set=dataset_set
        )
    for dataset_set, result in mix_and_split_results.items():
        logger.info(f"{dataset_set} total rows: {result[0]}")
        logger.info(
            f"{dataset_set} total counts: train {result[1]['train']}, val {result[1]['val']}, test {result[1]['test']}"
        )
    logger.success("Mixing and splitting complete!")


if __name__ == "__main__":
    main()
