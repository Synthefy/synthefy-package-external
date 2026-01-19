import argparse
import json
import os
import pickle
import resource
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthefy_pkg.data.foundation_dataloader import (
    MultiDirectoryTimeSeriesDataset,
    collate_fn,
)


def worker_init_fn(worker_id: int) -> None:
    """Initializes worker process with file descriptor limits and seeds."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    torch.multiprocessing.set_sharing_strategy("file_system")
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)


def count_samples(dataloader: DataLoader) -> int:
    """Returns total number of samples in a dataloader."""
    count = 0
    for batch in dataloader:
        count += batch["timeseries"].shape[0]
    return count


def verify_data_coverage(
    data_dirs: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    usage: str,
) -> None:
    """Verifies data processing by comparing source data count with loader samples."""
    total_data_points = 0
    for data_dir in data_dirs:
        # Find all parquet files in the directory
        parquet_files = [
            f for f in os.listdir(data_dir) if f.endswith(".parquet")
        ]
        for parquet_file in parquet_files:
            parquet_path = os.path.join(data_dir, parquet_file)
            df = pd.read_parquet(parquet_path)
            total_data_points += len(df)

    logger.info(
        f"Total {usage} data points in source directories: {total_data_points}"
    )

    train_samples = count_samples(train_loader)
    val_samples = count_samples(val_loader)
    test_samples = count_samples(test_loader)

    total_samples = train_samples + val_samples + test_samples
    logger.info(f"Total {usage} samples in dataloaders: {total_samples}")
    logger.info(f"{usage.capitalize()} train samples: {train_samples}")
    logger.info(f"{usage.capitalize()} validation samples: {val_samples}")
    logger.info(f"{usage.capitalize()} test samples: {test_samples}")

    if total_samples == 0:
        logger.warning(f"Warning: No {usage} samples found in dataloaders!")
    elif total_samples > total_data_points:
        logger.warning(
            f"Warning: More {usage} samples ({total_samples}) than data points ({total_data_points})! "
            "Check for data duplication."
        )


def save_shard(
    output_dir: str,
    split: str,
    shard_idx: int,
    timeseries_data: np.ndarray,
    metadata_data: np.ndarray,
    scalers_list: list,
) -> int:
    """Helper function to save a single shard of data.

    Args:
        output_dir: Directory to save the shard
        split: Split name ('train', 'val', or 'test')
        shard_idx: Index of the shard
        timeseries_data: Timeseries data to save
        metadata_data: Metadata to save
        scalers_list: List of scalers to save

    Returns:
        int: Number of samples saved
    """
    # Save timeseries and metadata
    np.save(
        os.path.join(output_dir, f"{split}_timeseries_shard_{shard_idx}.npy"),
        timeseries_data,
    )
    np.save(
        os.path.join(output_dir, f"{split}_metadata_shard_{shard_idx}.npy"),
        metadata_data,
    )

    # Save scalers
    with open(
        os.path.join(output_dir, f"{split}_scalers_shard_{shard_idx}.pkl"),
        "wb",
    ) as f:
        pickle.dump(scalers_list, f)

    samples_saved = len(timeseries_data)
    return samples_saved


def create_shard_manifest(
    output_dir: str,
    split: str,
    total_shards: int,
    total_samples: int,
    last_shard_samples: int,
    shard_size: Optional[int] = None,
) -> None:
    """Creates and saves a manifest file for sharded data.

    Args:
        output_dir: Directory to save the manifest
        split: Split name ('train', 'val', or 'test')
        total_shards: Total number of shards
        total_samples: Total number of samples across all shards
        last_shard_samples: Number of samples in the last shard
        shard_size: Target number of samples per shard (if specified)
    """
    shard_manifest = {
        "split": split,
        "total_shards": total_shards,
        "samples_per_shard": (
            shard_size if shard_size is not None else total_samples
        ),
        "total_samples": total_samples,
        "last_shard_samples": last_shard_samples,
    }

    with open(
        os.path.join(output_dir, f"{split}_shard_manifest.json"), "w"
    ) as f:
        json.dump(shard_manifest, f, indent=2)


def save_split_data(
    output_dir: str,
    split: str,
    dataloader: DataLoader,
    shard_size: Optional[int] = None,
) -> None:
    """Saves split data to disk in .npy format using shards.

    Args:
        output_dir: Directory to save the data
        split: Split name ('train', 'val', or 'test')
        dataloader: DataLoader containing the data to save
        shard_size: Number of samples per shard. If None, all data is saved in a single shard (shard_0)
    """
    os.makedirs(output_dir, exist_ok=True)

    timeseries_list = []
    metadata_list = []
    scalers_list = []

    sample_count = 0
    shard_idx = 0
    total_samples = 0
    batch_count = 0

    # Create progress bar without a specific total since we can't determine it in advance
    # for IterableDatasets like MultiDirectoryTimeSeriesDataset
    pbar = tqdm(
        desc=f"Saving {split} data",
        unit="batch",
        position=0,
        leave=True,
    )

    for batch in dataloader:
        print(f"iteration {batch_count}")
        batch_count += 1
        timeseries_batch = batch["timeseries"].numpy()
        metadata_batch = batch["metadata"].numpy()
        batch_size = timeseries_batch.shape[0]

        # If we're using limited shard size and this batch would exceed it,
        # save the current shard and start a new one
        if (
            shard_size is not None
            and sample_count > 0
            and sample_count + batch_size > shard_size
        ):
            # Save current shard
            timeseries_data = np.concatenate(timeseries_list, axis=0)
            metadata_data = np.concatenate(metadata_list, axis=0)

            # Save the shard
            samples_saved = save_shard(
                output_dir,
                split,
                shard_idx,
                timeseries_data,
                metadata_data,
                scalers_list,
            )

            # Update progress bar description
            pbar.set_postfix(
                {
                    "batches": batch_count,
                    "shard": shard_idx,
                    "samples_in_shard": samples_saved,
                    "total_samples": total_samples + samples_saved,
                }
            )

            logger.info(
                f"Saved {samples_saved} {split} samples to shard {shard_idx}"
            )

            # Update total sample count
            total_samples += samples_saved

            # Reset lists and increment shard index
            timeseries_list = []
            metadata_list = []
            scalers_list = []
            sample_count = 0
            shard_idx += 1

        # Add current batch to the lists
        timeseries_list.append(timeseries_batch)
        metadata_list.append(metadata_batch)
        scalers_list.extend(batch["scalers"])
        sample_count += batch_size

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix(
            {
                "batches": batch_count,
                "current_shard": shard_idx,
                "samples_collected": sample_count,
                "total_saved": total_samples,
            }
        )

    # Save any remaining data (last or only shard)
    if timeseries_list:
        timeseries_data = np.concatenate(timeseries_list, axis=0)
        metadata_data = np.concatenate(metadata_list, axis=0)

        # Save the final shard
        samples_saved = save_shard(
            output_dir,
            split,
            shard_idx,
            timeseries_data,
            metadata_data,
            scalers_list,
        )

        # Update total sample count
        total_samples += samples_saved

        # Update final progress bar info
        pbar.set_postfix(
            {
                "batches": batch_count,
                "shard": shard_idx,
                "samples_in_shard": samples_saved,
                "total_samples": total_samples,
            }
        )

        if shard_idx == 0:
            # Only one shard was created
            logger.info(
                f"Saved {samples_saved} {split} samples to single shard 0"
            )
        else:
            # Multiple shards were created
            logger.info(
                f"Saved {samples_saved} {split} samples to final shard {shard_idx}"
            )
            logger.info(
                f"Total of {total_samples} {split} samples saved across {shard_idx + 1} shards"
            )

        # Create manifest file
        create_shard_manifest(
            output_dir,
            split,
            shard_idx + 1,
            total_samples,
            samples_saved,
            shard_size,
        )

    # Close progress bar
    pbar.close()


class DataPreprocessor:
    def __init__(self, config_path: str):
        """Initialize DataPreprocessor with config.

        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._validate_config()

        # Set configuration values
        self.input_base_dir = self.config["input_base_dir"]
        self.output_dir = self.config["output_dir"]
        self.window_size = self.config.get("window_size", 50)
        self.stride = self.config.get("stride", 10)
        self.batch_size = self.config.get("batch_size", 128)
        self.num_workers = self.config.get("num_workers", 4)
        self.train_ratio = self.config.get("train_ratio", 0.8)
        self.val_ratio = self.config.get("val_ratio", 0.1)
        self.shard_size: Optional[int] = self.config.get("shard_size", 100000)

        # Set random seed if provided
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
            torch.manual_seed(self.config["random_seed"])

    def _validate_config(self) -> None:
        """Validates the configuration file."""
        required_fields = ["input_base_dir", "output_dir"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: {field}")

    def _get_data_dirs(self, usage: str) -> List[str]:
        """Gets data directories for specific usage (pretrain or blind).

        Args:
            usage: Type of usage ('pretrain' or 'blind')
        """
        data_dirs = []

        if (
            "selected_datasets" in self.config
            and self.config["selected_datasets"]
        ):
            # Filter datasets by usage
            for dataset_config in self.config["selected_datasets"]:
                # Default to 'pretrain' if usage is not specified
                dataset_usage = dataset_config.get("usage", "pretrain")
                if dataset_usage == usage:
                    data_dir = os.path.join(
                        self.input_base_dir, dataset_config["dataset_name"]
                    )
                    if not os.path.exists(data_dir):
                        logger.warning(
                            f"Dataset directory not found: {data_dir}, skipping"
                        )
                        continue
                    data_dirs.append(data_dir)
                    logger.info(
                        f"Found {usage} dataset: {dataset_config['dataset_name']}"
                    )
        else:
            # If no datasets are specified, treat all directories as pretrain
            if usage == "pretrain":
                logger.warning(
                    f"No datasets specified, scanning {self.input_base_dir} for data directories"
                )
                for root, dirs, files in os.walk(self.input_base_dir):
                    if any(file.endswith(".parquet") for file in files):
                        data_dirs.append(root)
                        logger.info(
                            f"Found dataset directory: {os.path.relpath(root, self.input_base_dir)}"
                        )

        if not data_dirs:
            logger.warning(f"No {usage} datasets found")

        return data_dirs

    def process_usage_type(self, usage: str) -> None:
        """Process datasets for a specific usage type."""
        logger.info(f"Processing {usage} datasets...")

        # Get data directories for this usage
        data_dirs = self._get_data_dirs(usage)
        if not data_dirs:
            logger.warning(
                f"Skipping {usage} processing - no valid directories found"
            )
            return

        # Create output subdirectory for this usage
        output_subdir = os.path.join(self.output_dir, usage)
        os.makedirs(output_subdir, exist_ok=True)

        # Clear existing embedding files
        self._clear_existing_embeddings(data_dirs)

        # Get metadata types from config or use defaults
        metadata_types_to_use = self.config.get(
            "metadata_types_to_use",
            [
                "timestamp",
                "dataset_description",
                "text_description",
                "continuous",
                "retrieved_timeseries",
                "time_varying_textual_metadata",
            ],
        )

        datasets = self._create_datasets(data_dirs, metadata_types_to_use)
        dataloaders = self._create_dataloaders(datasets)

        # Verify and save data with usage type
        # verify_data_coverage(data_dirs, *dataloaders, usage=usage)
        self._save_processed_data(
            dataloaders, metadata_types_to_use, output_subdir
        )

    def preprocess(self) -> None:
        """Runs the preprocessing pipeline for both pretrain and blind datasets."""
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
            logger.info("Set multiprocessing start method to 'spawn'")

        torch.multiprocessing.set_sharing_strategy("file_system")

        # Process pretrain and blind datasets separately
        for usage in ["pretrain", "blind"]:
            self.process_usage_type(usage)

    def _clear_existing_embeddings(self, data_dirs: List[str]) -> None:
        """Clears existing embedding files if force_regenerate is True, otherwise preserves them."""
        # Get force_regenerate setting from config, default to False
        force_regenerate = self.config.get("force_regenerate_embeddings", False)

        if not force_regenerate:
            logger.info(
                "Preserving existing embeddings (force_regenerate=False)"
            )
            return

        embedding_files = [
            "continuous_embeddings.npy",
            "retrieved_ts_embeddings.npy",
            "dataset_description_embedding.npy",
            "text_description_embedding.npy",
            "time_varying_text_embedding.npy",
        ]

        for data_dir in data_dirs:
            for file_name in embedding_files:
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed existing embedding file: {file_path}")

    def _create_datasets(
        self, data_dirs: List[str], metadata_types: List[str]
    ) -> tuple:
        """Creates train, validation, and test datasets."""
        logger.info("Creating datasets...")
        dataset_args = {
            "data_dirs": data_dirs,
            "window_size": self.window_size,
            "stride": self.stride,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "metadata_types_to_use": metadata_types,
        }

        return (
            MultiDirectoryTimeSeriesDataset(**dataset_args, split="train"),
            MultiDirectoryTimeSeriesDataset(**dataset_args, split="val"),
            MultiDirectoryTimeSeriesDataset(**dataset_args, split="test"),
        )

    def _create_dataloaders(self, datasets: tuple) -> tuple:
        """Creates train, validation, and test dataloaders."""
        logger.info("Creating dataloaders...")
        dataloader_args = {
            "batch_size": self.batch_size,
            "collate_fn": collate_fn,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "multiprocessing_context": "spawn",
            "worker_init_fn": worker_init_fn,
        }

        return tuple(
            DataLoader(dataset, **dataloader_args) for dataset in datasets
        )

    def _save_processed_data(
        self, dataloaders: tuple, metadata_types: List[str], output_dir: str
    ) -> None:
        """Saves processed data and configuration."""
        logger.info(f"Saving processed data to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        for split, dataloader in zip(["train", "val", "test"], dataloaders):
            save_split_data(output_dir, split, dataloader, self.shard_size)

        # Save manifest with configuration
        manifest = {
            "config": self.config,
            "metadata_types": metadata_types,
        }

        with open(os.path.join(output_dir, "manifest.yaml"), "w") as f:
            yaml.dump(manifest, f)

        logger.success(
            f"Data preprocessing completed successfully for {output_dir}"
        )


def main() -> None:
    """Main function to run preprocessing from config."""
    parser = argparse.ArgumentParser(
        description="Preprocess foundation model data from config"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    preprocessor = DataPreprocessor(args.config)
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
