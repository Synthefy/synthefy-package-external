import argparse
import glob
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1


def load_preprocessed_data(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load preprocessed data from a directory.

    Args:
        data_dir: Path to directory containing preprocessed data

    Returns:
        Tuple containing timeseries data, metadata, and scalers

    Raises:
        FileNotFoundError: If any required files are not found or cannot be loaded
    """
    timeseries_path = os.path.join(data_dir, "timeseries.npy")
    metadata_path = os.path.join(data_dir, "metadata.npy")
    scalers_path = os.path.join(data_dir, "scalers.pkl")

    # Check if all files exist
    missing_files = []
    for path, name in [
        (timeseries_path, "timeseries.npy"),
        (metadata_path, "metadata.npy"),
        (scalers_path, "scalers.pkl"),
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
        metadata = np.load(metadata_path, allow_pickle=True)
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {data_dir}: {str(e)}")

    logger.info(f"Loaded {len(timeseries)} samples from {data_dir}")
    return timeseries, metadata, scalers


def save_batch(
    output_dir: str,
    split: str,
    timeseries: np.ndarray,
    metadata: np.ndarray,
    batch_idx: int,
) -> None:
    """
    Save a batch of data to disk.

    Args:
        output_dir: Directory to save the data
        split: Split name ('train', 'val', or 'test')
        timeseries: Timeseries data to save
        metadata: Metadata to save
        batch_idx: Batch index for filename
    """
    os.makedirs(output_dir, exist_ok=True)

    timeseries_path = os.path.join(
        output_dir, f"{split}_timeseries_batch_{batch_idx}.npy"
    )
    metadata_path = os.path.join(
        output_dir, f"{split}_metadata_batch_{batch_idx}.npy"
    )

    np.save(timeseries_path, timeseries)
    np.save(metadata_path, metadata, allow_pickle=True)

    logger.info(
        f"Saved batch {batch_idx} with {len(timeseries)} samples to {output_dir}"
    )


def shuffle_and_split_data(
    timeseries: np.ndarray,
    metadata: np.ndarray,
    scalers: List[Dict],
    train_ratio: float,
    val_ratio: float,
    random_seed: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, List[Dict]],
    Tuple[np.ndarray, np.ndarray, List[Dict]],
    Tuple[np.ndarray, np.ndarray, List[Dict]],
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
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

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
        metadata = metadata[indices]
        # Create a numpy array of scalers to shuffle them the same way
        scalers_array = np.array(scalers, dtype=object)[indices]
        scalers = scalers_array.tolist()

    # Create splits
    train_timeseries = timeseries[:train_end]
    train_metadata = metadata[:train_end]
    train_scalers = scalers[:train_end]

    val_timeseries = timeseries[train_end:val_end]
    val_metadata = metadata[train_end:val_end]
    val_scalers = scalers[train_end:val_end]

    test_timeseries = timeseries[val_end:]
    test_metadata = metadata[val_end:]
    test_scalers = scalers[val_end:]

    return (
        (train_timeseries, train_metadata, train_scalers),
        (val_timeseries, val_metadata, val_scalers),
        (test_timeseries, test_metadata, test_scalers),
        shuffle_mapping,
    )


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

        self.max_shard_size = self.config.get("max_shard_size", 100000)

        self.output_dir = self.config["output_dir"]
        self.input_base_dir = self.config["input_base_dir"]

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

    def mix_and_split(self, dataset_set: str) -> None:
        """
        Mix data from multiple datasets and split into train/val/test sets.

        Args:
            dataset_set: Type of dataset set to process (e.g., "pretrain", "blind")
        """
        # Create output directory if it doesn't exist
        output_subdir = os.path.join(self.output_dir, dataset_set)
        os.makedirs(output_subdir, exist_ok=True)

        # Initialize storage for all data
        split_data = {
            "train": {"timeseries": [], "metadata": [], "scalers": []},
            "val": {"timeseries": [], "metadata": [], "scalers": []},
            "test": {"timeseries": [], "metadata": [], "scalers": []},
        }

        # Track dataset indices in the final arrays
        dataset_indices = {"train": {}, "val": {}, "test": {}}
        running_indices = {"train": 0, "val": 0, "test": 0}

        # Store shuffle indices for each dataset and split
        dataset_shuffle_indices = {"train": {}, "val": {}, "test": {}}

        # Process each dataset according to the algorithm
        selected_datasets = sorted(
            self.config.get("selected_datasets", []),
            key=lambda x: x["dataset_name"],
        )

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

            try:
                # Load preprocessed data
                timeseries, metadata, scalers = load_preprocessed_data(
                    dataset_dir
                )

                # TODO - sampling? how do we use windows_per_million_mixed_windows?

                # Use the function to shuffle and split data
                train_data, val_data, test_data, shuffle_mapping = (
                    shuffle_and_split_data(
                        timeseries,
                        metadata,
                        scalers,
                        self.train_ratio,
                        self.val_ratio,
                        self.random_seed,
                        self.shuffle,
                    )
                )

                # Store shuffle mapping if shuffling was done
                if shuffle_mapping is not None:
                    for split_name, indices in shuffle_mapping.items():
                        dataset_shuffle_indices[split_name][dataset_name] = (
                            indices
                        )

                # Process each split
                for split_name, data in zip(
                    ["train", "val", "test"], [train_data, val_data, test_data]
                ):
                    split_ts, split_md, split_scalers = data
                    if len(split_ts) > 0:
                        # Add to split data
                        split_data[split_name]["timeseries"].append(split_ts)
                        split_data[split_name]["metadata"].append(split_md)
                        split_data[split_name]["scalers"].extend(split_scalers)

                        # Record dataset indices
                        start_idx = running_indices[split_name]
                        end_idx = start_idx + len(split_ts)
                        dataset_indices[split_name][dataset_name] = {
                            "start": start_idx,
                            "end": end_idx,
                        }
                        running_indices[split_name] = end_idx

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue

        # Save shuffle indices for each split
        for split_name, indices_by_dataset in dataset_shuffle_indices.items():
            if indices_by_dataset:
                with open(
                    os.path.join(
                        output_subdir, f"{split_name}_shuffle_indices.json"
                    ),
                    "w",
                ) as f:
                    json.dump(indices_by_dataset, f)
                logger.info(
                    f"Saved {split_name} shuffle indices to {output_subdir}/{split_name}_shuffle_indices.json"
                )

        # Save each split
        split_counts = {}
        for split_name, data in split_data.items():
            if data["timeseries"]:
                # Concatenate data
                timeseries_data = np.concatenate(data["timeseries"], axis=0)
                metadata_data = np.concatenate(data["metadata"], axis=0)

                # Save to disk
                np.save(
                    os.path.join(output_subdir, f"{split_name}_timeseries.npy"),
                    timeseries_data,
                )
                np.save(
                    os.path.join(output_subdir, f"{split_name}_metadata.npy"),
                    metadata_data,
                    allow_pickle=True,
                )

                # Save scalers for this split
                with open(
                    os.path.join(output_subdir, f"{split_name}_scalers.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(data["scalers"], f)

                # Save dataset indices for this split separately
                with open(
                    os.path.join(
                        output_subdir, f"{split_name}_dataset_indices.json"
                    ),
                    "w",
                ) as f:
                    json.dump(dataset_indices[split_name], f, indent=2)

                # Log and store count
                logger.info(
                    f"Saved {len(timeseries_data)} {split_name} samples to {output_subdir}"
                )
                split_counts[f"{split_name}_samples"] = len(timeseries_data)
            else:
                split_counts[f"{split_name}_samples"] = 0

        # Save manifest
        manifest = {
            **split_counts,
            "config": self.config,
        }
        with open(os.path.join(output_subdir, "manifest.yaml"), "w") as f:
            yaml.dump(manifest, f)

        logger.info(
            f"Data mixing and splitting complete for {dataset_set} set:"
        )
        for split_name in ["train", "val", "test"]:
            logger.info(
                f"  {split_name.capitalize()} samples: {split_counts[f'{split_name}_samples']}"
            )
        logger.info(f"  Saved to: {output_subdir}")


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

    for dataset_set in ["pretrain", "blind"]:
        logger.info(
            f"Mixing and splitting data according to config: {args.config} On the {dataset_set} set"
        )
        mixer.mix_and_split(dataset_set=dataset_set)
    logger.success("Mixing and splitting complete!")


if __name__ == "__main__":
    main()
