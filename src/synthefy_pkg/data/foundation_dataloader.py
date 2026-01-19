import gc
import math
import os
import pickle
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.data.data_gen_base import SynthesisBaseDataModule
from synthefy_pkg.preprocessing.fm_preprocess_window import WindowPreprocessor
from synthefy_pkg.preprocessing.fm_text_embedder import (
    FoundationModelTextEmbedder,
)
from synthefy_pkg.utils.basic_utils import seed_everything
from synthefy_pkg.utils.fm_utils import (
    load_data_from_directory,
)


def collate_fn(
    batch: List[Dict[str, Any]], verbose: bool = False
) -> Dict[str, Any]:
    """Custom collate function that handles dictionaries with tensors and scalers.
    Args:
        batch: List of dictionaries, each containing 'timeseries', 'metadata', and 'scalers'
        verbose: Whether to log worker + collate information
    Returns:
        Dict[str, Any]: Dictionary with batched tensors and lists of scalers
    """
    # Log collate function calls when using multiple workers
    worker_info = get_worker_info()
    if worker_info is not None:
        if verbose:
            logger.debug(
                f"Worker {worker_info.id}: Collating batch of size {len(batch)}"
            )

    # Create result dictionary with batched data using list comprehensions
    result = {
        "timeseries": torch.stack([item["timeseries"] for item in batch]),
        "metadata": torch.stack([item["metadata"] for item in batch]),
        "scalers": [item["scalers"] for item in batch],
    }

    return result


class StandardTimeSeriesDataset(Dataset):
    """A simple dataset for loading time series data from standard files.

    This dataset loads data from standard files (timeseries.npy, metadata.npy, scalers.pkl)
    without any splitting.
    """

    def __init__(
        self,
        data_dir: str,
        filename_prefix: Optional[str] = "",
        verbose: bool = False,
    ):
        """Initialize a dataset that loads time series data from standard files.

        Args:
            data_dir: Directory containing the data files
            filename_prefix: Prefix for the data files (train, val, test) are options besides default
            verbose: Whether to log worker + initialization information
        Raises:
            FileNotFoundError: If required files are missing
        """
        super().__init__()
        self.data_dir = data_dir
        self.verbose = verbose

        # Log initialization
        worker_info = get_worker_info()
        if worker_info is None:
            if self.verbose:
                logger.info(
                    f"StandardTimeSeriesDataset initialized in main process for {data_dir}"
                )
        else:
            if self.verbose:
                logger.info(
                    f"StandardTimeSeriesDataset initialized in worker {worker_info.id} "
                    f"(out of {worker_info.num_workers}) for {data_dir}"
                )

        # Find all shard files
        shard_files = [
            f
            for f in os.listdir(data_dir)
            if (
                (
                    f.startswith(f"{filename_prefix}_timeseries_shard_")
                    and f.endswith(".npy")
                )
                or (
                    f.startswith(f"{filename_prefix}_timeseries")
                    and f.endswith(".npy")
                )
                or (filename_prefix == "" and f == "timeseries.npy")
            )
        ]

        # Extract shard indices and sort numerically
        def extract_shard_index(filename: str) -> int:
            """Extract the numeric index from a shard filename.
            Args:
                filename: Name of the shard file
            Returns:
                int: The numeric index of the shard
            """
            # Extract the number between "_timeseries_shard_" and ".npy"
            parts = filename.split("_timeseries_shard_")
            if len(parts) != 2:
                return 0
            try:
                return int(parts[1].split(".npy")[0])
            except ValueError:
                return 0

        # Sort files by numeric shard index
        self.shard_files = sorted(shard_files, key=extract_shard_index)
        logger.debug(
            f"shard_files: {shard_files}, [f for f in os.listdir(data_dir)]: {[f for f in os.listdir(data_dir)]}, self.shard_files: {self.shard_files}"
        )

        if not self.shard_files:
            raise FileNotFoundError(
                f"Timeseries file not found '{filename_prefix}' in {data_dir}"
            )

        # Find corresponding metadata shard files
        self.metadata_shard_files = []
        # Find corresponding scaler shard files
        self.scaler_shard_files = []

        for shard_file in self.shard_files:
            # Replace timeseries_shard with metadata_shard in the filename
            metadata_file = (
                shard_file.replace("_timeseries_shard_", "_metadata_shard_")
                if "_timeseries_shard_" in shard_file
                else shard_file.replace("timeseries.npy", "metadata.npy")
            )
            metadata_path = os.path.join(data_dir, metadata_file)
            if os.path.exists(metadata_path):
                self.metadata_shard_files.append(metadata_path)
            else:
                # If metadata file doesn't exist, use None as a placeholder
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_path}"
                )

            # Get corresponding scaler file
            scaler_file = (
                shard_file.replace(
                    "_timeseries_shard_", "_scalers_shard_"
                ).replace(".npy", ".pkl")
                if "_timeseries_shard_" in shard_file
                else shard_file.replace("timeseries.npy", "scalers.pkl")
            )
            scaler_path = os.path.join(data_dir, scaler_file)

            if os.path.exists(scaler_path):
                self.scaler_shard_files.append(scaler_path)
            else:
                raise FileNotFoundError(
                    f"Scalers file not found: {scaler_path}"
                )

        # Calculate total samples and sample-to-shard mapping
        self.shard_info = []
        self.total_samples = 0
        self.timeseries_data = None
        self.metadata_data = None
        self.scalers = None

        for i, shard_file in enumerate(self.shard_files):
            logger.debug(f"Processing shard {i}: {shard_file}")
            shard_path = os.path.join(data_dir, shard_file)
            # Just load the shape without loading all data
            shard_shape = np.load(shard_path, mmap_mode="r").shape
            shard_samples = shard_shape[0]

            self.shard_info.append(
                {
                    "path": shard_path,
                    "metadata_path": self.metadata_shard_files[i],
                    "scaler_path": self.scaler_shard_files[i],
                    "samples": shard_samples,
                    "start_idx": self.total_samples,
                    "end_idx": self.total_samples + shard_samples - 1,
                }
            )

            if self.timeseries_data is None:
                self.timeseries_data = np.load(shard_path, mmap_mode="r")
            else:
                self.timeseries_data = np.concatenate(
                    [self.timeseries_data, np.load(shard_path, mmap_mode="r")],
                    axis=0,
                )

            if self.metadata_data is None:
                self.metadata_data = np.load(
                    self.metadata_shard_files[i], mmap_mode="r"
                )
            else:
                self.metadata_data = np.concatenate(
                    [
                        self.metadata_data,
                        np.load(self.metadata_shard_files[i], mmap_mode="r"),
                    ],
                    axis=0,
                )

            if self.scalers is None:
                self.scalers = pickle.load(
                    open(self.scaler_shard_files[i], "rb")
                )
            else:
                if isinstance(self.scalers, list):
                    self.scalers.append(
                        pickle.load(open(self.scaler_shard_files[i], "rb"))
                    )
                else:
                    self.scalers.update(
                        pickle.load(open(self.scaler_shard_files[i], "rb"))
                    )

            self.total_samples += shard_samples

        assert self.timeseries_data is not None
        assert self.metadata_data is not None
        assert self.scalers is not None

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return self.total_samples

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, StandardScaler]]]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            dict: Dictionary containing:
                - 'timeseries': tensor of the time series data
                - 'metadata': tensor of metadata (or empty tensor if no metadata)
                - 'scalers': dictionary containing StandardScaler objects (or empty dict if no scalers)

        Raises:
            IndexError: If the index is out of bounds
        """
        # Log access pattern when using multiple workers
        worker_info = get_worker_info()
        if (
            worker_info is not None and idx % 100 == 0
        ) and self.verbose:  # Log only occasionally to avoid spam
            logger.debug(f"Worker {worker_info.id}: Fetching sample {idx}")

        if idx < 0 or idx >= self.total_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {self.total_samples} samples"
            )

        # Load timeseries data
        assert self.timeseries_data is not None
        timeseries = self.timeseries_data[idx].copy()
        timeseries_tensor = torch.tensor(timeseries, dtype=torch.float32)

        # Load metadata if available
        if self.metadata_data is not None:
            metadata = self.metadata_data[idx].copy()
            metadata_tensor = torch.tensor(metadata, dtype=torch.float32)
        else:
            # Create empty metadata tensor if no metadata is available
            metadata_tensor = torch.zeros((1, 1), dtype=torch.float32)

        # Get corresponding scaler if available
        scaler_dict = {}
        if self.scalers is not None:
            # Handle different scaler formats
            if isinstance(self.scalers, dict):
                # If scalers is a dict of scalers for all samples
                scaler_dict = self.scalers
            elif isinstance(self.scalers, list) and idx < len(self.scalers):
                # If scalers is a list of dicts, one per sample
                scaler_dict = self.scalers[idx]

        return {
            "timeseries": timeseries_tensor,
            "metadata": metadata_tensor,
            "scalers": scaler_dict,
        }

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Any]], verbose: bool = False
    ) -> Dict[str, Any]:
        """Custom collate function that handles dictionaries with tensors and scalers.

        Args:
            batch: List of dictionaries, each containing 'timeseries', 'metadata', and 'scalers'
            verbose: Whether to log worker + collate information
        Returns:
            Dict[str, Any]: Dictionary with batched tensors and lists of scalers
        """
        # Log collate function calls when using multiple workers
        worker_info = get_worker_info()
        if worker_info is not None:
            if verbose:
                logger.debug(
                    f"Worker {worker_info.id}: Collating batch of size {len(batch)}"
                )

        # Create result dictionary with batched data using list comprehensions
        result = {
            "timeseries": torch.stack([item["timeseries"] for item in batch]),
            "metadata": torch.stack([item["metadata"] for item in batch]),
            "scalers": [item["scalers"] for item in batch],
        }

        return result


class ShardedTimeSeriesDataset(Dataset):
    """Dataset for loading time series data from sharded numpy files.
    This dataset loads one shard at a time into memory and returns samples from it.
    When all samples from a shard have been consumed, it loads the next shard.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
    ):
        """Initialize a dataset that loads time series data from sharded files.
        Args:
            data_dir: Directory containing the sharded data files
            split: One of 'train', 'val', or 'test'
        Raises:
            FileNotFoundError: If no shard files are found or if metadata/scalers files are missing
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split

        # Find all shard files
        shard_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith(f"{self.split}_timeseries_shard_")
            and f.endswith(".npy")
        ]

        # Extract shard indices and sort numerically
        def extract_shard_index(filename: str) -> int:
            """Extract the numeric index from a shard filename.
            Args:
                filename: Name of the shard file
            Returns:
                int: The numeric index of the shard
            """
            # Extract the number between "_timeseries_shard_" and ".npy"
            parts = filename.split("_timeseries_shard_")
            if len(parts) != 2:
                return 0
            try:
                return int(parts[1].split(".npy")[0])
            except ValueError:
                return 0

        # Sort files by numeric shard index
        self.shard_files = sorted(shard_files, key=extract_shard_index)

        if not self.shard_files:
            raise FileNotFoundError(
                f"No shard files found with prefix '{self.split}' in {data_dir}"
            )

        # Find corresponding metadata shard files
        self.metadata_shard_files = []
        # Find corresponding scaler shard files
        self.scaler_shard_files = []

        for shard_file in self.shard_files:
            # Replace timeseries_shard with metadata_shard in the filename
            metadata_file = shard_file.replace(
                "_timeseries_shard_", "_metadata_shard_"
            )
            metadata_path = os.path.join(data_dir, metadata_file)

            if os.path.exists(metadata_path):
                self.metadata_shard_files.append(metadata_path)
            else:
                # If metadata file doesn't exist, use None as a placeholder
                self.metadata_shard_files.append(None)
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_path}"
                )

            # Get corresponding scaler file
            scaler_file = shard_file.replace(
                "_timeseries_shard_", "_scalers_shard_"
            ).replace(".npy", ".pkl")
            scaler_path = os.path.join(data_dir, scaler_file)

            if os.path.exists(scaler_path):
                self.scaler_shard_files.append(scaler_path)
            else:
                raise FileNotFoundError(
                    f"Scalers file not found: {scaler_path}"
                )

        # Calculate total samples and sample-to-shard mapping
        self.shard_info = []
        self.total_samples = 0

        for i, shard_file in enumerate(self.shard_files):
            shard_path = os.path.join(data_dir, shard_file)
            # Just load the shape without loading all data
            shard_shape = np.load(shard_path, mmap_mode="r").shape
            shard_samples = shard_shape[0]

            self.shard_info.append(
                {
                    "path": shard_path,
                    "metadata_path": self.metadata_shard_files[i],
                    "scaler_path": self.scaler_shard_files[i],
                    "samples": shard_samples,
                    "start_idx": self.total_samples,
                    "end_idx": self.total_samples + shard_samples - 1,
                }
            )

            self.total_samples += shard_samples

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.
        Returns:
            int: Total number of samples
        """
        return self.total_samples

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, StandardScaler]]]:
        """Get a sample from the dataset.
        Args:
            idx: Index of the sample to retrieve
        Returns:
            dict: Dictionary containing:
                - 'timeseries': tensor of the time series data
                - 'metadata': tensor of metadata
                - 'scalers': dictionary containing StandardScaler objects
        Raises:
            IndexError: If the index is out of bounds
            RuntimeError: If the shard containing the index cannot be found
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {self.total_samples} samples"
            )

        # Find which shard contains this index
        target_shard_idx = None
        for i, info in enumerate(self.shard_info):
            if info["start_idx"] <= idx <= info["end_idx"]:
                target_shard_idx = i
                break

        if target_shard_idx is None:
            raise RuntimeError(f"Could not find shard for index {idx}")

        # Get shard info
        shard_info = self.shard_info[target_shard_idx]
        local_idx = idx - shard_info["start_idx"]

        # Load timeseries data
        shard_data = np.load(shard_info["path"], mmap_mode="r")
        sample = shard_data[local_idx].copy()

        # Convert to tensor
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)

        # Load metadata
        metadata_data = np.load(shard_info["metadata_path"], mmap_mode="r")
        metadata = metadata_data[local_idx].copy()
        metadata_tensor = torch.tensor(metadata, dtype=torch.float32)

        # Always load scaler without caching
        with open(shard_info["scaler_path"], "rb") as f:
            scalers = pickle.load(f)

        # Get the specific scaler for this sample from the shard's scalers
        scaler_dict = scalers[local_idx]

        # Return dictionary with the requested structure
        return {
            "timeseries": sample,
            "metadata": metadata_tensor,
            "scalers": scaler_dict,
        }


class MultiDirectoryTimeSeriesDataset(IterableDataset):
    def __init__(
        self,
        data_dirs: List[str],
        window_size: int,
        stride: int = 1,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        buffer_size: int = 10000,
        seed: int = 42,
        metadata_types_to_use: List[str] = [
            "timestamp",
            "dataset_description",
            "text_description",
            "continuous",
            "retrieved_timeseries",
            "time_varying_textual_metadata",
        ],
    ):
        """Initialize a dataset that loads time series data from multiple directories.

        Args:
            data_dirs: List of directories containing raw time series data
            window_size: Size of each time series window
            stride: Step size between consecutive windows
            split: One of 'train', 'val', or 'test'
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            shuffle: Whether to shuffle the data
            buffer_size: Size of buffer used for shuffling
            seed: Random seed for reproducibility
            metadata_types_to_use: List of metadata types to include
        """
        super().__init__()
        self.data_dirs = data_dirs
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed

        seed_everything(self.seed)

        # Validate split ratios
        assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
        assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1"
        assert train_ratio + val_ratio < 1, (
            "train_ratio + val_ratio must be less than 1"
        )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

        # Estimate total samples (placeholder)
        self.total_samples = self._estimate_total_samples()
        self.metadata_types_to_use = metadata_types_to_use

    def _estimate_total_samples(self) -> int:
        """Estimate the total number of samples across all directories.

        Returns:
            int: Estimated total number of samples
        """
        # Implement logic to count total possible windows across all directories
        # This is a placeholder
        return 1000000

    def _is_in_split(self, idx: int, total_windows: int) -> bool:
        """Determine if a window index belongs to the current split.

        Args:
            idx: Index of the window
            total_windows: Total number of windows in the dataset

        Returns:
            bool: True if the window belongs to the current split, False otherwise

        Raises:
            ValueError: If the split is not one of 'train', 'val', or 'test'
        """
        # Special case: if there's only one window, put it in the training set
        if total_windows == 1:
            return self.split == "train"

        train_end = int(total_windows * self.train_ratio)
        val_end = train_end + int(total_windows * self.val_ratio)

        if self.split == "train":
            return idx < train_end
        elif self.split == "val":
            return train_end <= idx < val_end
        elif self.split == "test":
            return idx >= val_end
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __iter__(
        self,
    ) -> Iterator[Dict[str, Union[torch.Tensor, Dict[str, StandardScaler]]]]:
        """Create an iterator over the dataset.

        Returns:
            Iterator: Iterator that yields processed windows
        """
        # Create a base iterator that produces windows
        base_iterator = self._create_base_iterator()

        if self.shuffle and (self.split == "train" or self.split == "val"):
            # For shuffling, we need to collect all windows and scalers
            # This loads everything into memory, which could be an issue for very large datasets
            windows_and_scalers = list(base_iterator)

            # Wrap in a DataPipe for shuffling
            datapipe = IterableWrapper(windows_and_scalers, deepcopy=True)

            # Apply shuffling with the specified buffer size
            # This ensures windows and their corresponding scalers stay together
            shuffled_pipe = ShufflerIterDataPipe(
                datapipe, buffer_size=self.buffer_size
            )
            shuffled_pipe.set_seed(self.seed)
            return iter(shuffled_pipe)
        else:
            # Return the base iterator without shuffling
            return base_iterator

    def _create_base_iterator(
        self,
    ) -> Iterator[Dict[str, Union[torch.Tensor, Dict[str, StandardScaler]]]]:
        """Create the base iterator that yields windows from all directories.

        Returns:
            Iterator: Iterator that yields processed windows
        """
        worker_info = get_worker_info()
        # logger.debug(f"Worker info: {worker_info}")

        # Get directories to process based on worker ID
        if worker_info is None:  # single-process loading
            logger.debug(
                "Running in single-process mode, processing all directories"
            )
            dirs_to_process = self.data_dirs
        else:  # in a worker process
            # Split workload among workers by directories
            per_worker = int(
                math.ceil(len(self.data_dirs) / worker_info.num_workers)
            )
            worker_id = worker_info.id

            start_dir = worker_id * per_worker
            end_dir = min(start_dir + per_worker, len(self.data_dirs))
            dirs_to_process = self.data_dirs[start_dir:end_dir]

        # Process each directory
        for data_dir in dirs_to_process:
            max_retries = 3
            retry_count = 0

            while retry_count <= max_retries:
                try:
                    # Load raw data from the directory
                    raw_data, metadata = load_data_from_directory(data_dir)
                    self._initialize_window_preprocessor(metadata)
                    # Initialize text embedder and generate embeddings
                    text_embedder = FoundationModelTextEmbedder(
                        metadata, data_dir
                    )
                    embeddings_dict = text_embedder.read_or_embed_fm_text_data(
                        data_dir
                    )

                    total_windows = (
                        len(raw_data) - self.window_size
                    ) // self.stride + 1
                    total_windows = max(
                        total_windows, 1
                    )  # always have at least 1 window.
                    logger.debug(
                        f"Directory {data_dir}: {len(raw_data)} rows, {total_windows} windows"
                    )

                    # Create sliding windows and filter by split
                    for window_idx in range(total_windows):
                        if self._is_in_split(window_idx, total_windows):
                            start_idx = window_idx * self.stride
                            window = raw_data.iloc[
                                start_idx : start_idx + self.window_size
                            ]
                            # Get the corresponding slices of text embeddings for this window
                            end_idx = start_idx + self.window_size

                            # window_text_description = np.array([])

                            # TODO: when ready to use non uniform values for a window
                            # text description, uncomment the following code
                            # if len(text_description_embedding) > 0:
                            #     window_text_description = (
                            #         text_description_embedding[
                            #             start_idx:end_idx
                            #         ]
                            #     )

                            window_time_varying_text = np.array([])
                            if (
                                len(
                                    embeddings_dict[
                                        "time_varying_text_embedding"
                                    ]
                                )
                                > 0
                            ):
                                window_time_varying_text = embeddings_dict[
                                    "time_varying_text_embedding"
                                ][start_idx:end_idx]
                            yield self._process_window(
                                window,
                                window_time_varying_text,
                                embeddings_dict,
                            )

                    # If we get here without exception, break the retry loop
                    if retry_count > 0:
                        logger.debug(
                            f"Processed {data_dir} successfully after {retry_count} retries"
                        )
                    break

                except RuntimeError as e:
                    # Check if it's a CUDA OOM error
                    if "CUDA out of memory" in str(e):
                        retry_count += 1
                        if retry_count <= max_retries:
                            logger.warning(
                                f"CUDA OOM error processing directory {data_dir}. Retry {retry_count}/{max_retries}"
                            )
                            # Force garbage collection to free memory
                            gc.collect()

                            # No need to import torch here since it's imported at the module level
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Optional: wait a moment before retrying
                            import time

                            time.sleep(1)
                            continue
                        else:
                            logger.error(
                                f"Failed to process directory {data_dir} after {max_retries} retries due to CUDA OOM"
                            )
                            break
                    else:
                        # For non-OOM errors, log and continue to next directory
                        logger.error(
                            f"Error processing directory {data_dir}: {e}"
                        )
                        break

                except Exception as e:
                    logger.error(f"Error processing directory {data_dir}: {e}")
                    break

                finally:
                    # Clean up resources to avoid file handle leaks

                    if "text_embedder" in locals():
                        text_embedder = locals().get("text_embedder")
                        if text_embedder is not None:
                            del text_embedder
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    # Break out of the retry loop if we've completed successfully
                    if (
                        retry_count == 0
                    ):  # Only break if we didn't have to retry
                        break

    def _initialize_window_preprocessor(self, metadata: Dict[str, Any]) -> None:
        """Initialize the window preprocessor with metadata.

        Args:
            metadata: Dictionary of metadata for this data source
        """
        self.window_preprocessor = WindowPreprocessor(
            window_size=self.window_size,
            metadata=metadata,
            metadata_types_to_use=self.metadata_types_to_use,
            verbose=False,
        )

    def _process_window(
        self,
        window: pd.DataFrame,
        window_time_varying_text: np.ndarray,
        embeddings_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Union[torch.Tensor, Dict[str, StandardScaler]]]:
        """Process a window of data and return a dictionary with the window, metadata, and scaler.

        Args:
            window: DataFrame containing the window data
            window_time_varying_text: Time varying text metadata embeddings for this window
            embeddings_dict: Dictionary containing embeddings for various text metadata

        Returns:
            dict: Dictionary containing:
                - 'timeseries': tensor of the scaled window data
                - 'metadata': tensor of metadata (empty if no metadata)
                - 'scalers': dictionary containing the fitted StandardScaler for timeseries and metadata
        """
        if self.window_preprocessor is None:
            raise ValueError(
                "Window preprocessor is not initialized. Call _initialize_window_preprocessor first."
            )

        processed_window_dict = self.window_preprocessor.preprocess_window(
            window,
            continuous_embeddings=embeddings_dict[
                "continuous_descriptions_embeddings"
            ],
            retrieved_ts_embeddings=embeddings_dict[
                "retrieved_ts_descriptions_embeddings"
            ],
            dataset_description_embedding=embeddings_dict[
                "dataset_description_embedding"
            ],
            text_description_embedding=embeddings_dict[
                "text_description_embedding"
            ],
            time_varying_text_embedding=window_time_varying_text,
        )

        # Convert numpy arrays to tensors
        processed_window_dict["timeseries"] = torch.tensor(
            processed_window_dict["timeseries"].astype(np.float32),
            dtype=torch.float32,
        )
        processed_window_dict["metadata"] = torch.tensor(
            processed_window_dict["metadata"].astype(np.float32),
            dtype=torch.float32,
        )

        return processed_window_dict


class FoundationModelDataLoader(SynthesisBaseDataModule):
    """
    DataLoader for foundation models using StandardTimeSeriesDataset.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the FoundationModelDataLoader.

        Args:
            config: Configuration object containing dataset settings
        """
        logger.info("Initializing FoundationModelDataLoader")

        data_dir = os.path.join(
            str(os.getenv("SYNTHEFY_DATASETS_BASE")),
            config.dataset_config.dataset_name,
        )

        if config.dataset_config.use_sharded_dataset:
            logger.info("Loading train dataset")
            train_dataset = ShardedTimeSeriesDataset(
                data_dir=data_dir, split="train"
            )

            logger.info("Loading val dataset")
            val_dataset = ShardedTimeSeriesDataset(
                data_dir=data_dir, split="val"
            )

            logger.info("Loading test dataset")
            test_dataset = ShardedTimeSeriesDataset(
                data_dir=data_dir, split="test"
            )
        else:
            logger.info("Loading train dataset")
            train_dataset = StandardTimeSeriesDataset(
                data_dir=data_dir, filename_prefix="train"
            )

            logger.info("Loading val dataset")
            val_dataset = StandardTimeSeriesDataset(
                data_dir=data_dir, filename_prefix="val"
            )

            logger.info("Loading test dataset")
            test_dataset = StandardTimeSeriesDataset(
                data_dir=data_dir, filename_prefix="test"
            )

        super().__init__(config, train_dataset, val_dataset, test_dataset)

        # Set batch size and other dataloader parameters from config
        self.batch_size = config.dataset_config.batch_size
        self.num_workers = config.dataset_config.num_workers
        self.shuffle = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=StandardTimeSeriesDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=StandardTimeSeriesDataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=StandardTimeSeriesDataset.collate_fn,
        )
