from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from synthefy_pkg.configs.dataset_configs import DatasetConfig
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.configs.tabicl_config import TabICLPriorConfig
from synthefy_pkg.prior.activations import get_activation_name
from synthefy_pkg.prior.dataset import DisablePrinting, PriorDataset
from synthefy_pkg.prior.probing import (
    construct_mlp_dag,
    get_connectivity_statistics,
    get_dag_statistics,
)
from synthefy_pkg.prior.single_mlp_generate import (
    collect_inputs,
    generate_scm_object,
    get_single_dataset,
)
from synthefy_pkg.utils.basic_utils import seed_everything

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

"""
Utilities for generating and loading prior datasets from disk for training.

This module implements a workflow for generating synthetic tabular data from prior
distributions, saving it efficiently to disk in a sparse format, and loading it during
distributed training.

This module provides two main classes:
- SavePriorDataset: Generates and saves batches of prior data to disk
- LoadPriorDataset: Loads pre-generated prior data from disk for distributed training

The data is saved in a sparse format to reduce storage requirements and loaded
on demand during training. The module supports distributed training by allowing
different processes to load different batches in a coordinated way.

The saved data includes:
- X: Input features in sparse format or nested tensor (for variable-length sequences)
- y: Target labels as regular tensor or nested tensor (for variable-length sequences)
- d: Number of features per dataset
- seq_lens: Sequence length for each dataset
- train_sizes: Position at which to split training and evaluation data
- batch_size: Number of datasets in the batch

See examples/configs/foundation_model_configs/config_icl_synthetic_saved_train.yaml
for an example of a config that can be used to load the saved synthetic data.

Sample run command:
uv run src/synthefy_pkg/prior/genload.py --num_batches 240 --config src/synthefy_pkg/prior/config/synthetic_configs/config_lag_series_overfit.yaml --train_batches 200 --val_batches 20 --test_batches 20

If you want to save CSVs instead, see export_to_csv.py
"""


warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


def dense2sparse(
    dense_tensor: torch.Tensor,
    row_lengths: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a dense tensor with trailing zeros into a compact 1D representation.

    Parameters
    ----------
    dense_tensor : torch.Tensor
        Input tensor of shape (num_rows, num_cols) where each row may contain
        trailing zeros beyond the valid entries

    row_lengths : torch.Tensor
        Tensor of shape (num_rows,) specifying the number of valid entries
        in each row of the dense tensor

    dtype : torch.dtype, default=torch.float32
        Output data type for the sparse representation

    Returns
    -------
    torch.Tensor
        1D tensor of shape (sum(row_lengths),) containing only the valid entries
    """

    assert dense_tensor.dim() == 2, "dense_tensor must be 2D"
    num_rows, num_cols = dense_tensor.shape
    assert row_lengths.shape[0] == num_rows, (
        "row_lengths must match number of rows"
    )
    assert (row_lengths <= num_cols).all(), (
        "row_lengths cannot exceed number of columns"
    )

    indices = torch.arange(num_cols, device=dense_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    sparse = dense_tensor[mask].to(dtype)

    return sparse


def sparse2dense(
    sparse_tensor: torch.Tensor,
    row_lengths: torch.Tensor,
    max_len: int = -1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a dense tensor from its sparse representation.

    This function is the inverse of dense2sparse, reconstructing a padded dense
    tensor from a compact 1D representation and the corresponding row lengths.
    Unused entries in the output are filled with zeros.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        1D tensor containing the valid entries from the original dense tensor

    row_lengths : torch.Tensor
        Number of valid entries for each row in the output tensor

    max_len : Optional[int], default=None
        Maximum length for each row in the output. If None, uses max(row_lengths)

    dtype : torch.dtype, default=torch.float32
        Output data type for the dense representation

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (num_rows, max_len) with zeros padding
    """

    assert sparse_tensor.dim() == 1, "data must be 1D"
    assert row_lengths.sum() == len(sparse_tensor), (
        "data length must match sum of row_lengths"
    )

    num_rows = len(row_lengths)
    max_len = max_len or int(row_lengths.max().item())
    dense = torch.zeros(
        num_rows, max_len, dtype=dtype, device=sparse_tensor.device
    )
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)

    return dense


class SliceNestedTensor:
    """A wrapper for nested tensors that supports slicing along the first dimension.

    This class wraps PyTorch's nested tensor and provides slicing operations
    along the first dimension, which are not natively supported by nested tensors.
    It maintains compatibility with other nested tensor operations by forwarding
    attribute access to the wrapped tensor.

    Parameters
    ----------
    nested_tensor : torch.Tensor
        A nested tensor to wrap
    """

    def __init__(self, nested_tensor):
        self.nested_tensor = nested_tensor
        self.is_nested = nested_tensor.is_nested

    def __getitem__(self, idx):
        """Support slicing operations along the first dimension."""
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = self.nested_tensor.size(0) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step

            indices = list(range(start, stop, step))
            return SliceNestedTensor(
                torch.nested.nested_tensor(
                    [self.nested_tensor[i] for i in indices]
                )
            )
        elif isinstance(idx, int):
            return self.nested_tensor[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped nested tensor."""
        return getattr(self.nested_tensor, name)

    def __len__(self):
        """Return the length of the first dimension."""
        return self.nested_tensor.size(0)

    def to(self, *args, **kwargs):
        """Support the to() method for device/dtype conversion."""
        return SliceNestedTensor(self.nested_tensor.to(*args, **kwargs))


def cat_slice_nested_tensors(tensors: List, dim=0) -> SliceNestedTensor:
    """Concatenate a list of SliceNestedTensor objects along dimension dim.

    Parameters
    ----------
    tensors : List
        List of tensors to concatenate

    dim : int, default=0
        Dimension along which to concatenate

    Returns
    -------
    SliceNestedTensor
        Concatenated tensor wrapped in SliceNestedTensor
    """
    # Extract the wrapped nested tensors
    nested_tensors = [
        t.nested_tensor if isinstance(t, SliceNestedTensor) else t
        for t in tensors
    ]
    return SliceNestedTensor(torch.cat(nested_tensors, dim=dim))


class LoadPriorDataset(IterableDataset):
    """Loads pre-generated prior data sequentially for distributed training.

    Parameters
    ----------
    config : TabICLPriorConfig
        Configuration object containing dataset parameters
    ddp_world_size : int, default=1
        Total number of distributed processes
    ddp_rank : int, default=0
        Rank of current process
    max_batches : int, optional
        Maximum number of batches to load. If None, load indefinitely.
    timeout : int, default=60
        Maximum time in seconds to wait for a batch file
    split : str, default='train'
        Which split to load from ('train', 'val', 'test'). If the split folder
        doesn't exist, falls back to loading from the root directory.
    """

    def __init__(
        self,
        config: TabICLPriorConfig,
        ddp_world_size: int = 1,
        ddp_rank: int = 0,
        max_batches: Optional[int] = None,
        timeout: int = 60,
        split: str = "train",
    ):
        super().__init__()
        if config.prior_dir is None:
            raise ValueError(
                "prior_dir must be set in config to load generated data"
            )

        self.config = config
        self.split = split

        # Determine data directory - check if split folder exists
        base_dir = config.prior_dir
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir) and os.path.isdir(split_dir):
            self.data_dir = split_dir
            logger.info(f"Loading data from {split} split: {self.data_dir}")
        else:
            self.data_dir = base_dir
            logger.warning(
                f"Split folder '{split}' not found, loading from root directory: {self.data_dir}"
            )

        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.current_idx = ddp_rank + config.load_prior_start
        self.timeout = timeout
        self.delete_after_load = config.delete_after_load
        self.device = config.prior_device

        self.max_features = config.max_features
        # Load metadata if available - try both split directory and root directory
        self.metadata = None
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            # Try root directory if not found in split directory
            metadata_file = os.path.join(base_dir, "metadata.json")

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load or parse metadata.json: {e}")

        # Auto-determine max_batches if not provided
        if max_batches is None:
            # Temporarily set max_batches to None so __len__ can do its calculation
            self.max_batches = None
            calculated_length = len(self)
            if calculated_length > 0:
                max_batches = calculated_length
                logger.info(
                    f"Auto-determined max_batches for {split} split: {max_batches}"
                )
            else:
                logger.warning(
                    f"Could not determine max_batches for {split} split, will attempt to load indefinitely"
                )

        self.max_batches = max_batches

        # Buffer for storing datasets that haven't been returned yet
        self.buffer_X = None
        self.buffer_y = None
        self.buffer_d = None
        self.buffer_seq_lens = None
        self.buffer_train_sizes = None
        self.buffer_size = 0
        self.buffer_series_flags = None

        # Track the number of iterations (calls to __next__) separately from file index
        self.iteration_count = 0

    def __repr__(self) -> str:
        """
        Returns a string representation of the LoadPriorDataset.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        repr_str = (
            f"LoadPriorDataset(\n"
            f"  data_dir: {self.data_dir}\n"
            f"  split: {self.split}\n"
            f"  batch_size: {self.config.batch_size}\n"
            f"  ddp_world_size: {self.ddp_world_size}\n"
            f"  ddp_rank: {self.ddp_rank}\n"
            f"  start_from: {self.current_idx - self.ddp_rank}\n"
            f"  max_batches: {self.max_batches or 'Infinite'}\n"
            f"  timeout: {self.timeout}\n"
            f"  delete_after_load: {self.delete_after_load}\n"
            f"  device: {self.device}\n"
        )
        if self.metadata:
            repr_str += "  Loaded Metadata:\n"
            repr_str += (
                f"    prior_type: {self.metadata.get('prior_type', 'N/A')}\n"
            )
            repr_str += f"    batch_size (generated): {self.metadata.get('batch_size', 'N/A')}\n"
            repr_str += f"    batch_size_per_gp: {self.metadata.get('batch_size_per_gp', 'N/A')}\n"
            repr_str += f"    min features: {self.metadata.get('min_features', 'N/A')}\n"
            repr_str += f"    max features: {self.metadata.get('max_features', 'N/A')}\n"
            repr_str += (
                f"    max classes: {self.metadata.get('max_classes', 'N/A')}\n"
            )
            repr_str += f"    seq_len: {self.metadata.get('min_seq_len', 'N/A') or 'None'} - {self.metadata.get('max_seq_len', 'N/A')}\n"
            repr_str += (
                f"    log_seq_len: {self.metadata.get('log_seq_len', 'N/A')}\n"
            )
            repr_str += f"    sequence length varies across groups: {self.metadata.get('seq_len_per_gp', 'N/A')}\n"
            repr_str += f"    train_size: {self.metadata.get('min_train_size', 'N/A')} - {self.metadata.get('max_train_size', 'N/A')}\n"
            repr_str += f"    replay_small: {self.metadata.get('replay_small', 'N/A')}\n"
            if any(
                key in self.metadata
                for key in ["train_batches", "val_batches", "test_batches"]
            ):
                repr_str += f"    Data splits - Train: {self.metadata.get('train_batches', 'N/A')}, "
                repr_str += f"Val: {self.metadata.get('val_batches', 'N/A')}, Test: {self.metadata.get('test_batches', 'N/A')}\n"
        repr_str += ")"

        return repr_str

    def __iter__(self):
        # Reset for new epoch
        self.iteration_count = 0
        self.current_idx = self.ddp_rank + self.config.load_prior_start
        # Clear buffer
        self.buffer_X = None
        self.buffer_y = None
        self.buffer_d = None
        self.buffer_seq_lens = None
        self.buffer_train_sizes = None
        self.buffer_size = 0
        self.buffer_series_flags = None
        return self

    def __len__(self):
        """
        Calculate the number of batches this dataset can provide.

        This method calculates the total number of samples available for this dataset
        instance (accounting for DDP) and divides by the desired batch size to get
        the number of batches that will be yielded.

        Returns
        -------
        int
            Number of batches available for this dataset instance
        """
        # If max_batches is explicitly set, use that
        if self.max_batches is not None:
            return max(
                0,
                (self.max_batches - self.current_idx + self.ddp_world_size - 1)
                // self.ddp_world_size,
            )

        # Try to determine from metadata first
        if self.metadata:
            split_key = f"{self.split}_batches"
            if split_key in self.metadata and self.metadata.get("batch_size"):
                total_file_batches = self.metadata[split_key]
                samples_per_file = self.metadata[
                    "batch_size"
                ]  # This is the batch_size stored in each file
                total_samples = total_file_batches * samples_per_file

                # Account for DDP: each process gets a subset of samples
                samples_for_this_process = (
                    total_samples + self.ddp_world_size - 1
                ) // self.ddp_world_size

                # Calculate how many batches of desired size we can provide
                return max(
                    0,
                    (samples_for_this_process + self.config.batch_size - 1)
                    // self.config.batch_size,
                )

        # Fallback: count available batch files and sum their actual batch sizes
        try:
            batch_files = [
                f
                for f in os.listdir(self.data_dir)
                if f.startswith("batch_") and f.endswith(".pt")
            ]
            if batch_files:
                total_samples = 0
                # Sum up the actual batch_size from each file
                for f in sorted(batch_files):  # Process files in order
                    try:
                        batch_file_path = os.path.join(self.data_dir, f)
                        batch = torch.load(
                            batch_file_path,
                            map_location="cpu",
                            weights_only=True,
                        )
                        file_batch_size = batch.get("batch_size", 0)
                        total_samples += file_batch_size
                    except Exception:
                        # Skip files that can't be loaded
                        continue

                if total_samples > 0:
                    # Account for DDP: each process gets a subset of samples
                    samples_for_this_process = (
                        total_samples + self.ddp_world_size - 1
                    ) // self.ddp_world_size

                    # Calculate how many batches of desired size we can provide
                    return max(
                        0,
                        (samples_for_this_process + self.config.batch_size - 1)
                        // self.config.batch_size,
                    )
        except (OSError, FileNotFoundError):
            pass

        # If we can't determine the length, return 0 to indicate unknown/infinite length
        return 0

    def _load_batch_file(self):
        """Load a single batch file from disk.

        Returns
        -------
        tuple
            A tuple containing X, y, d, seq_lens, train_sizes and the size of the batch
        """
        batch_file = os.path.join(
            self.data_dir, f"batch_{self.current_idx:06d}.pt"
        )

        # Try loading the file for up to timeout seconds
        wait_time = 0
        while not os.path.exists(batch_file):
            if wait_time >= self.timeout:
                # Check if we have max_batches set and current_idx exceeds it
                if (
                    self.max_batches is not None
                    and self.current_idx >= self.max_batches
                ):
                    raise StopIteration(
                        f"Reached end of available batch files. "
                        f"Attempted to load batch {self.current_idx} but max_batches is {self.max_batches}"
                    )
                else:
                    raise RuntimeError(
                        f"Timeout waiting for batch file {batch_file}. "
                        f"Expected batch files: 0 to {self.max_batches - 1 if self.max_batches else 'unknown'}, "
                        f"but batch {self.current_idx} not found."
                    )
            time.sleep(5)
            wait_time += 5

        # CRITICAL: Always load on CPU to prevent CUDA IPC issues in worker processes
        # The collate_fn in the dataloader will move tensors to the target device
        batch = torch.load(batch_file, map_location="cpu", weights_only=True)
        X = batch["X"]
        y = batch["y"]
        d = batch["d"]
        seq_lens = batch["seq_lens"]
        train_sizes = batch["train_sizes"]
        batch_size = batch["batch_size"]
        series_flags = (
            batch["series_flags"]
            if "series_flags" in batch
            else torch.tensor([True] * batch_size, device=self.device)
        )

        if X.is_nested:
            # Wrap nested tensors with SliceNestedTensor
            X = SliceNestedTensor(X)
            y = SliceNestedTensor(y)
        else:
            # Convert sparse tensor to dense
            if self.max_features is not None:
                max_len = self.max_features
            else:
                max_len = -1
            # TODO: if varying seq_lens need to be used, fix this code
            X = sparse2dense(
                X,
                d.repeat_interleave(seq_lens[0]),
                max_len=max_len,
                dtype=torch.float32,
            ).view(batch_size, seq_lens[0], -1)

        # Delete file if requested
        if self.delete_after_load and os.path.exists(batch_file):
            os.remove(batch_file)

        # Prepare next index for this process
        self.current_idx += self.ddp_world_size

        return X, y, d, seq_lens, train_sizes, batch_size, series_flags

    def __next__(self):
        """Load datasets until we have at least batch_size, then return exactly batch_size.

        This method accumulates datasets from multiple files if necessary to return
        the exact number of datasets specified in batch_size. Any extra datasets are
        kept in a buffer for the next iteration.

        Returns
        -------
        tuple
            A tuple containing:
            - X: Input features [batch_size, seq_len, features] or nested tensor
            - y: Target labels [batch_size, seq_len] or nested tensor
            - d: Number of features per dataset
            - seq_lens: Sequence length for each dataset
            - train_sizes: Position at which to split training and evaluation data
        """
        # Check if we've reached the maximum number of iterations
        if (
            self.max_batches is not None
            and self.iteration_count >= self.max_batches
        ):
            raise StopIteration

        # Initialize or use existing buffer
        if self.buffer_size == 0:
            try:
                # Load the first batch
                (
                    X,
                    y,
                    d,
                    seq_lens,
                    train_sizes,
                    file_batch_size,
                    series_flags,
                ) = self._load_batch_file()
                self.buffer_X = X
                self.buffer_y = y
                self.buffer_d = d
                self.buffer_seq_lens = seq_lens
                self.buffer_train_sizes = train_sizes
                self.buffer_size = file_batch_size
                self.buffer_series_flags = series_flags
            except StopIteration:
                # No more files available
                raise StopIteration

        # Keep loading files until we have enough data or no more files
        while self.buffer_size < self.config.batch_size:
            try:
                # Load another batch and append to buffer
                (
                    X,
                    y,
                    d,
                    seq_lens,
                    train_sizes,
                    file_batch_size,
                    series_flags,
                ) = self._load_batch_file()

                # Concatenate with existing buffer
                if self.buffer_X is None:
                    # If buffer is empty, directly assign
                    self.buffer_X = X
                    self.buffer_y = y
                    self.buffer_d = d
                    self.buffer_seq_lens = seq_lens
                    self.buffer_train_sizes = train_sizes
                    self.buffer_size = file_batch_size
                    self.buffer_series_flags = series_flags
                else:
                    # Extract underlying tensors if they are SliceNestedTensor
                    assert (
                        self.buffer_X is not None and self.buffer_y is not None
                    )
                    buffer_X_tensor = (
                        self.buffer_X.nested_tensor
                        if isinstance(self.buffer_X, SliceNestedTensor)
                        else self.buffer_X
                    )
                    X_tensor = (
                        X.nested_tensor
                        if isinstance(X, SliceNestedTensor)
                        else X
                    )
                    buffer_y_tensor = (
                        self.buffer_y.nested_tensor
                        if isinstance(self.buffer_y, SliceNestedTensor)
                        else self.buffer_y
                    )
                    y_tensor = (
                        y.nested_tensor
                        if isinstance(y, SliceNestedTensor)
                        else y
                    )

                    assert isinstance(
                        buffer_X_tensor, torch.Tensor
                    ) and isinstance(X_tensor, torch.Tensor)
                    assert isinstance(
                        buffer_y_tensor, torch.Tensor
                    ) and isinstance(y_tensor, torch.Tensor)
                    assert isinstance(
                        self.buffer_d, torch.Tensor
                    ) and isinstance(d, torch.Tensor)
                    assert isinstance(
                        self.buffer_seq_lens, torch.Tensor
                    ) and isinstance(seq_lens, torch.Tensor)
                    assert isinstance(
                        self.buffer_train_sizes, torch.Tensor
                    ) and isinstance(train_sizes, torch.Tensor)
                    assert isinstance(
                        self.buffer_series_flags, torch.Tensor
                    ) and isinstance(series_flags, torch.Tensor)

                    self.buffer_X = torch.cat(
                        [buffer_X_tensor, X_tensor], dim=0
                    )
                    self.buffer_y = torch.cat(
                        [buffer_y_tensor, y_tensor], dim=0
                    )
                    self.buffer_d = torch.cat([self.buffer_d, d], dim=0)
                    self.buffer_seq_lens = torch.cat(
                        [self.buffer_seq_lens, seq_lens], dim=0
                    )
                    self.buffer_train_sizes = torch.cat(
                        [self.buffer_train_sizes, train_sizes], dim=0
                    )
                    self.buffer_series_flags = torch.cat(
                        [self.buffer_series_flags, series_flags], dim=0
                    )
                    self.buffer_size += file_batch_size
            except StopIteration:
                # No more files available, break and use what we have in buffer
                break
            except Exception as e:
                # If we can't load more files for other reasons, use what we have
                logger.warning(f"Could not load more files: {str(e)}")
                break

        # Extract batch_size datasets (or all if we have fewer)
        output_size = min(self.config.batch_size, self.buffer_size)

        # Prepare output
        assert self.buffer_X is not None and self.buffer_y is not None
        assert (
            self.buffer_d is not None
            and self.buffer_seq_lens is not None
            and self.buffer_train_sizes is not None
        )

        X_out = self.buffer_X[:output_size]
        y_out = self.buffer_y[:output_size]
        d_out = self.buffer_d[:output_size]
        seq_lens_out = self.buffer_seq_lens[:output_size]
        train_sizes_out = self.buffer_train_sizes[:output_size]

        # Update buffer with remaining data
        if output_size < self.buffer_size:
            self.buffer_X = self.buffer_X[output_size:]
            self.buffer_y = self.buffer_y[output_size:]
            self.buffer_d = self.buffer_d[output_size:]
            self.buffer_seq_lens = self.buffer_seq_lens[output_size:]
            self.buffer_train_sizes = self.buffer_train_sizes[output_size:]
            self.buffer_size -= output_size
        else:
            # Buffer is now empty
            self.buffer_X = None
            self.buffer_y = None
            self.buffer_d = None
            self.buffer_seq_lens = None
            self.buffer_train_sizes = None
            self.buffer_size = 0

        # Increment iteration count
        self.iteration_count += 1

        if isinstance(X_out, SliceNestedTensor):
            X_out = X_out.nested_tensor
        if isinstance(y_out, SliceNestedTensor):
            y_out = y_out.nested_tensor

        return X_out, y_out, d_out, seq_lens_out, train_sizes_out


class SavePriorDataset:
    """Generates and saves batches of prior datasets to disk.

    The datasets are saved as individual batch files in the specified directory
    using an atomic file writing pattern to ensure data integrity.

    Parameters
    ----------
    config : TabICLPriorConfig
        Configuration object containing all parameters for dataset generation
    num_batches : int
        Number of batches to generate
    resume_from : int
        Resume generation from this batch index
    num_threads_per_generate : int
        Number of threads to use per generation
    train_batches : int, optional
        Number of batches for training set. If None, all batches go to train.
    val_batches : int, optional
        Number of batches for validation set
    test_batches : int, optional
        Number of batches for test set
    export_csv : bool, default=False
        Whether to export data as CSV files in addition to PyTorch files
    csv_output_dir : str, optional
        Directory to save CSV files. If None and export_csv=True, uses prior_dir/csv
    """

    def __init__(
        self,
        config: TabICLPriorConfig,
        resume_from: int = 0,
        train_batches: int = 0,
        val_batches: int = 0,
        test_batches: int = 0,
        export_csv: bool = False,
        csv_output_dir: Optional[str] = None,
        full_config: Optional[Configuration] = None,
    ):
        self.config = config
        self.resume_from = resume_from
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.num_batches = train_batches + val_batches + test_batches
        self.export_csv = export_csv
        if config.prior_dir is None:
            raise ValueError(
                "prior_dir must be set in config to save generated data"
            )

        self.csv_output_dir = (
            csv_output_dir
            if csv_output_dir is not None
            else os.path.join(config.prior_dir, "csv")
        )  # type: ignore

        self.save_dir = config.prior_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Create subdirectories for train/val/test
        self.train_dir = os.path.join(self.save_dir, "train")
        self.val_dir = os.path.join(self.save_dir, "val")
        self.test_dir = os.path.join(self.save_dir, "test")

        if self.train_batches > 0:
            os.makedirs(self.train_dir, exist_ok=True)
        if self.val_batches > 0:
            os.makedirs(self.val_dir, exist_ok=True)
        if self.test_batches > 0:
            os.makedirs(self.test_dir, exist_ok=True)

        self.save_metadata()

        self.full_config = full_config
        self.dataset_config = (
            full_config.dataset_config if full_config is not None else None
        )
        self.fm_config = (
            full_config.foundation_model_config
            if full_config is not None
            else None
        )

        self.prior = PriorDataset(
            config=config,
            real_data_config=self.dataset_config,
            fm_config=self.fm_config,
        )
        logger.info(self.prior)

    def save_metadata(self):
        """Save metadata about the dataset generation configuration to a JSON file."""
        metadata = {
            "prior_type": self.config.prior_type,
            "batch_size": self.config.batch_size,
            "batch_size_per_gp": self.config.batch_size_per_gp,
            "min_seq_len": self.config.min_seq_len,
            "max_seq_len": self.config.max_seq_len,
            "log_seq_len": self.config.log_seq_len,
            "seq_len_per_gp": self.config.seq_len_per_gp,
            "min_features": self.config.min_features,
            "max_features": self.config.max_features,
            "max_classes": self.config.max_classes,
            "min_train_size": self.config.min_train_size,
            "max_train_size": self.config.max_train_size,
            "replay_small": self.config.replay_small,
            "row_missing_prob": self.config.row_missing_prob,
            "column_has_missing_prob": self.config.column_has_missing_prob,
            "train_batches": self.train_batches,
            "val_batches": self.val_batches,
            "test_batches": self.test_batches,
        }
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_split_and_dir(self, batch_idx):
        """Determine which split a batch belongs to and return the appropriate directory.

        Parameters
        ----------
        batch_idx : int
            Global batch index

        Returns
        -------
        tuple
            (split_name, directory_path, local_batch_idx)
        """
        relative_idx = batch_idx - self.resume_from

        if relative_idx < self.train_batches:
            return "train", self.train_dir, relative_idx
        elif relative_idx < self.train_batches + self.val_batches:
            return "val", self.val_dir, relative_idx - self.train_batches
        else:
            return (
                "test",
                self.test_dir,
                relative_idx - self.train_batches - self.val_batches,
            )

    def save_batch_sparse(
        self, batch_idx, X, y, d, seq_lens, train_sizes, series_flags
    ):
        """Save batch data in sparse format for efficient storage.

        This method handles the conversion between dense and sparse tensor formats
        when appropriate and saves the batch data to a PyTorch file. It uses an atomic
        write pattern (writing to a temporary file and then renaming) to ensure data
        integrity even if the process is interrupted during saving.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch used for file naming

        X : torch.Tensor
            Input features tensor, either in dense format [batch_size, seq_len, features]
            or in nested tensor format for variable sequence lengths

        y : torch.Tensor
            Target labels tensor

        d : torch.Tensor
            Number of features for each dataset

        seq_lens : torch.Tensor
            Sequence length for each dataset

        train_sizes : torch.Tensor
            Position at which to split training and evaluation data
        """

        if self.config.seq_len_per_gp:
            # X and y are nested tensors and they are already sparse
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(
                X.view(-1, H), d.repeat_interleave(T), dtype=torch.float32
            )

        # Determine which split this batch belongs to
        split_name, save_dir, local_idx = self._get_split_and_dir(batch_idx)

        # Create temporary file first
        batch_file = os.path.join(save_dir, f"batch_{local_idx:06d}.pt")
        temp_file = os.path.join(save_dir, f"batch_{local_idx:06d}.pt.tmp")
        torch.save(
            {
                "X": X,
                "y": y,
                "d": d,
                "seq_lens": seq_lens,
                "train_sizes": train_sizes,
                "batch_size": B,
                "series_flags": series_flags,
            },
            temp_file,
        )
        # Atomic rename to ensure file integrity
        os.rename(temp_file, batch_file)

    def run(self):
        """Generate and save batches of prior datasets using PyTorch DataLoader."""
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(
            f"Data split - Train: {self.train_batches}, Val: {self.val_batches}, Test: {self.test_batches}"
        )
        logger.info(
            f"Generating {self.num_batches} batches starting from index {self.resume_from}"
        )

        # Create DataLoader with automatic parallelization
        num_workers = getattr(self.config, "num_workers", 0) or getattr(
            self.config, "n_jobs", 0
        )
        if num_workers == 0:
            num_workers = 0  # Use main process only
        else:
            num_workers = min(
                num_workers, self.num_batches
            )  # Don't exceed batch count

        logger.info(f"Using {num_workers} workers for parallel generation")
        self.prior.config.dataset_length = self.num_batches * args.batch_size

        dataloader = DataLoader(
            self.prior,
            batch_size=args.batch_size,  # Each item is already a batch
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,  # We're saving to disk, not GPU
        )

        # Process batches and save them
        batch_idx = self.resume_from
        for batch_data in tqdm(dataloader, desc="Generating batches"):
            if batch_idx >= self.resume_from + self.num_batches:
                break

            X, y, d, seq_lens, train_sizes, series_flags = batch_data

            # Move tensors to CPU before saving
            X = X.cpu()
            y = y.cpu()
            d = d.cpu()
            seq_lens = seq_lens.cpu()
            train_sizes = train_sizes.cpu()
            series_flags = series_flags.cpu()

            self.save_batch_sparse(
                batch_idx,
                X[:, 0],
                y[:, 0],
                d[:, 0],
                seq_lens,
                train_sizes,
                series_flags,
            )

            # Export to CSV if enabled
            export_csv = getattr(self, "export_csv", False)
            if export_csv:
                csv_dir = getattr(
                    self, "csv_output_dir", os.path.join(self.save_dir, "csv")
                )
                export_batch_to_csv(
                    X,
                    y,
                    d,
                    seq_lens,
                    train_sizes,
                    series_flags,
                    batch_idx,
                    csv_dir,
                )

            batch_idx += 1


class SaveSinglePriorDataset:
    """Generates and saves batches of sampled inputs from a single prior.

    The datasets are saved as individual batch files in the specified directory
    using an atomic file writing pattern to ensure data integrity.

    Parameters
    ----------
    config : TabICLPriorConfig
        Configuration object containing all parameters for dataset generation
    num_batches : int
        Number of batches to generate
    resume_from : int
        Resume generation from this batch index
    train_batches : int, optional
        Number of batches for training set. If None, all batches go to train.
    val_batches : int, optional
        Number of batches for validation set
    test_batches : int, optional
        Number of batches for test set
    """

    def __init__(
        self,
        config: TabICLPriorConfig,
        resume_from: int = 0,
        train_batches: int = 0,
        val_batches: int = 0,
        test_batches: int = 0,
    ):
        self.config = config
        self.resume_from = resume_from
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.num_batches = train_batches + val_batches + test_batches

        if config.prior_dir is None:
            raise ValueError(
                "prior_dir must be set in config to save generated data"
            )

        self.save_dir = config.prior_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Create subdirectories for train/val/test
        self.train_dir = os.path.join(self.save_dir, "train")
        self.val_dir = os.path.join(self.save_dir, "val")
        self.test_dir = os.path.join(self.save_dir, "test")

        if self.train_batches > 0:
            os.makedirs(self.train_dir, exist_ok=True)
        if self.val_batches > 0:
            os.makedirs(self.val_dir, exist_ok=True)
        if self.test_batches > 0:
            os.makedirs(self.test_dir, exist_ok=True)

        # Generate single SCM object for metadata and DAG
        self.dataset_object, self.scm_object, self.params, self.indices_X_y = (
            generate_scm_object(config, config.n_features)
        )
        self.save_metadata()
        self.dag = construct_mlp_dag(
            self.scm_object, self.indices_X_y[0], self.indices_X_y[1]
        )

        # Write DAG statistics
        dag_edges = get_dag_statistics(self.dag)
        dag_edges_json = json.dumps(dag_edges, indent=2)
        with open(os.path.join(self.save_dir, "dag_edges.json"), "w") as f:
            f.write(dag_edges_json)
        logger.info(
            f"DAG edges: {os.path.join(self.save_dir, 'dag_edges.json')}"
        )

        json_string = json.dumps(
            get_connectivity_statistics(
                self.dag,
                self.scm_object,
                self.indices_X_y[0],
                self.indices_X_y[1],
            ),
            indent=2,
        )
        with open(
            os.path.join(self.save_dir, "connectivity_statistics.json"), "w"
        ) as f:
            f.write(json_string)
        logger.info(
            f"Connectivity statistics: {os.path.join(self.save_dir, 'connectivity_statistics.json')}"
        )

    def save_metadata(self):
        """Save metadata about the dataset generation configuration to a JSON file."""
        metadata = {
            "prior_config": {
                "prior_type": self.config.prior_type,
                "batch_size": self.config.batch_size,
                "batch_size_per_gp": self.config.batch_size_per_gp,
                "min_seq_len": self.config.min_seq_len,
                "max_seq_len": self.config.max_seq_len,
                "log_seq_len": self.config.log_seq_len,
                "seq_len_per_gp": self.config.seq_len_per_gp,
                "min_features": self.config.min_features,
                "max_features": self.config.max_features,
                "max_classes": self.config.max_classes,
                "min_train_size": self.config.min_train_size,
                "max_train_size": self.config.max_train_size,
                "replay_small": self.config.replay_small,
                "row_missing_prob": self.config.row_missing_prob,
                "column_has_missing_prob": self.config.column_has_missing_prob,
                "train_batches": self.train_batches,
                "val_batches": self.val_batches,
                "test_batches": self.test_batches,
            },
            "sampled_params": copy.deepcopy(self.params),
        }
        activations = [
            get_activation_name(self.scm_object.layers[i][0])
            for i in range(len(self.scm_object.layers))
            if isinstance(self.scm_object.layers[i], nn.Sequential)
        ]
        metadata["sampled_params"]["mlp_activations"] = activations
        metadata["sampled_params"]["lags"] = (
            metadata["sampled_params"]["lags"].tolist()
            if metadata["sampled_params"]["lags"] is not None
            else None
        )
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        copied_config = copy.deepcopy(self.config.to_dict())
        del copied_config["scm_mlp_activations"]
        copied_config["num_flat_range"] = [
            int(copied_config["num_flat_range"][0]),
            int(copied_config["num_flat_range"][1]),
        ]
        with open(os.path.join(self.save_dir, "prior_config.yaml"), "w") as f:
            yaml.dump(copied_config, f)

    def _get_split_and_dir(self, batch_idx):
        """Determine which split a batch belongs to and return the appropriate directory.

        Parameters
        ----------
        batch_idx : int
            Global batch index

        Returns
        -------
        tuple
            (split_name, directory_path, local_batch_idx)
        """
        relative_idx = batch_idx - self.resume_from

        if relative_idx < self.train_batches:
            return "train", self.train_dir, relative_idx
        elif relative_idx < self.train_batches + self.val_batches:
            return "val", self.val_dir, relative_idx - self.train_batches
        else:
            return (
                "test",
                self.test_dir,
                relative_idx - self.train_batches - self.val_batches,
            )

    def save_batch_sparse(
        self, batch_idx, X, y, d, seq_lens, train_sizes, series_flags
    ):
        """Save batch data in sparse format for efficient storage.

        This method handles the conversion between dense and sparse tensor formats
        when appropriate and saves the batch data to a PyTorch file. It uses an atomic
        write pattern (writing to a temporary file and then renaming) to ensure data
        integrity even if the process is interrupted during saving.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch used for file naming

        X : torch.Tensor
            Input features tensor, either in dense format [batch_size, seq_len, features]
            or in nested tensor format for variable sequence lengths

        y : torch.Tensor
            Target labels tensor

        d : torch.Tensor
            Number of features for each dataset

        seq_lens : torch.Tensor
            Sequence length for each dataset

        train_sizes : torch.Tensor
            Position at which to split training and evaluation data
        """

        if self.config.seq_len_per_gp:
            # X and y are nested tensors and they are already sparse
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(
                X.view(-1, H), d.repeat_interleave(T), dtype=torch.float32
            )

        # Determine which split this batch belongs to
        split_name, save_dir, local_idx = self._get_split_and_dir(batch_idx)

        # Create temporary file first
        batch_file = os.path.join(save_dir, f"batch_{local_idx:06d}.pt")
        temp_file = os.path.join(save_dir, f"batch_{local_idx:06d}.pt.tmp")
        torch.save(
            {
                "X": X,
                "y": y,
                "d": d,
                "seq_lens": seq_lens,
                "train_sizes": train_sizes,
                "batch_size": B,
                "series_flags": series_flags,
            },
            temp_file,
        )
        # Atomic rename to ensure file integrity
        os.rename(temp_file, batch_file)

    def run(self):
        """Generate and save batches of prior datasets."""
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(
            f"Data split - Train: {self.train_batches}, Val: {self.val_batches}, Test: {self.test_batches}"
        )
        logger.info(
            f"Generating {self.num_batches} batches starting from index {self.resume_from}"
        )

        is_series = self.params["dataset_is_tabular"]
        # has_lag = self.params["dataset_has_lag"]
        # has_timestamp = self.params["dataset_has_timestamp"]
        num_inputs_to_collect = 1000000
        fixed_inputs, sample_types = collect_inputs(
            self.scm_object, num_inputs_to_collect
        )

        # plot the fixed inputs
        for i in range(fixed_inputs.shape[1]):
            plt.plot(fixed_inputs[:1000, i])
            plt.savefig(f"fixed_inputs_{i}.png")
            plt.close()

        for batch_idx in tqdm(
            range(self.resume_from, self.resume_from + self.num_batches),
            desc="Generating batches",
        ):
            X, y, d, seq_lens, train_sizes = get_single_dataset(
                self.config,
                self.config.batch_size,
                self.dataset_object,
                self.scm_object,
                self.params,
                self.indices_X_y,
                torch.tensor(fixed_inputs),
            )

            series_flags = torch.ones(
                self.config.batch_size,
                dtype=torch.bool,
                device=self.config.prior_device,
            ) * int(is_series)

            # Move tensors to CPU before saving
            X = X.cpu()
            y = y.cpu()
            d = d.cpu()

            seq_lens = seq_lens.cpu()
            train_sizes = train_sizes.cpu()
            series_flags = series_flags.cpu()
            self.save_batch_sparse(
                batch_idx, X, y, d, seq_lens, train_sizes, series_flags
            )


def export_batch_to_csv(
    X: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    seq_lens: torch.Tensor,
    train_sizes: torch.Tensor,
    series_flags: torch.Tensor,
    batch_idx: int,
    output_dir: str,
    datasets_per_folder: int = 1000,
):
    """Export a batch of datasets to CSV files.

    Each dataset is saved as a separate CSV file with:
    - timestamps as the first column
    - feature columns as column_1, column_2, etc.
    - 1000 datasets per folder by default

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
        # For nested tensors, each element is a different length
        for i in range(batch_size):
            dataset_X = X[i].cpu().numpy()  # [seq_len, features]
            dataset_y = y[i].cpu().numpy()  # [seq_len]
            dataset_d = int(d_np[i])
            dataset_seq_len = int(seq_lens_np[i])
            dataset_train_size = train_sizes_np[i]
            dataset_is_series = series_flags_np[i]

            # Create timestamps (simple sequential for now)
            timestamps = np.arange(dataset_seq_len)

            # Create DataFrame
            data_dict = {"timestamp": timestamps}

            # Add feature columns
            for j in range(int(dataset_d)):
                column_data = dataset_X[:, j]
                # Ensure the column data is 1D
                if column_data.ndim > 1:
                    column_data = column_data.flatten()
                data_dict[f"column_{j + 1}"] = column_data

            # Add target column
            target_data = dataset_y
            if target_data.ndim > 1:
                target_data = target_data.flatten()
            data_dict["target"] = target_data

            # Add metadata columns
            data_dict["train_size"] = np.full(
                int(dataset_seq_len), dataset_train_size
            )
            data_dict["is_series"] = np.full(
                int(dataset_seq_len), dataset_is_series
            )

            df = pd.DataFrame(data_dict)

            # Save CSV
            csv_filename = f"dataset_{batch_idx:06d}_{i:04d}.csv"
            csv_path = os.path.join(batch_folder, csv_filename)
            df.to_csv(csv_path, index=False)

    else:
        # For dense tensors
        X_np = X.cpu().numpy()  # [batch_size, seq_len, features]
        y_np = y.cpu().numpy()  # [batch_size, seq_len]

        for i in range(batch_size):
            dataset_X = X_np[i, 0]  # [seq_len, features]
            dataset_y = y_np[i, 0]  # [seq_len]
            dataset_d = int(d_np[i])
            dataset_seq_len = int(seq_lens_np[i])
            dataset_train_size = train_sizes_np[i]
            dataset_is_series = series_flags_np[i]

            # Create timestamps
            timestamps = np.arange(dataset_seq_len)

            # Create DataFrame
            data_dict = {"timestamp": timestamps}

            # Add feature columns
            for j in range(dataset_X.shape[1]):
                column_data = dataset_X[:, j]
                # Ensure the column data is 1D
                if column_data.ndim > 1:
                    column_data = column_data.flatten()
                if j > dataset_d:
                    column_data = np.full(int(dataset_seq_len), np.nan)
                data_dict[f"column_{j + 1}"] = column_data

            # Add target column
            target_data = dataset_y
            if target_data.ndim > 1:
                target_data = target_data.flatten()
            data_dict["target"] = target_data

            # Add metadata columns
            data_dict["train_size"] = np.full(
                int(dataset_seq_len), dataset_train_size
            )
            data_dict["is_series"] = np.full(
                int(dataset_seq_len), dataset_is_series
            )

            df = pd.DataFrame(data_dict)

            # Save CSV
            csv_filename = f"dataset_{batch_idx:06d}_{i:04d}.csv"
            csv_path = os.path.join(batch_folder, csv_filename)
            df.to_csv(csv_path, index=False)
    logger.info(f"Saved {batch_size} datasets to {batch_folder}")


if __name__ == "__main__":
    # Required for multiprocessing to work with synthetic prior code
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Generate training prior datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--execution-config",
        type=str,
        default=None,
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--resume_from",
        type=int,
        default=0,
        help="Resume generation from this batch index",
    )
    parser.add_argument(
        "--num_threads_per_generate",
        type=int,
        default=1,
        help="Threads per generation",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to use for dataset generation",
    )
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="Path to save prior datasets",
    )
    parser.add_argument(
        "--train_batches",
        type=int,
        default=1,
        help="Number of batches for training set",
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        default=0,
        help="Number of batches for validation set",
    )
    parser.add_argument(
        "--test_batches",
        type=int,
        default=0,
        help="Number of batches for test set",
    )
    parser.add_argument(
        "--single_prior_num",
        type=int,
        default=0,
        help="Number of single prior datasets to generate (default 0, unused)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export data as CSV files in addition to PyTorch files",
    )
    parser.add_argument(
        "--csv_output_dir",
        type=str,
        default=None,
        help="Directory to save CSV files. If None and export_csv=True, uses prior_dir/csv",
    )

    args = parser.parse_args()

    if args.execution_config is not None:
        execution_config = Configuration(config_filepath=args.execution_config)
    else:
        execution_config = None

    # Validate train/val/test splits if provided
    if any(
        [
            args.train_batches < 0,
            args.val_batches < 0,
            args.test_batches < 0,
        ]
    ):
        logger.error(
            "Error: train_batches, val_batches, and test_batches must be non-negative"
        )
        raise ValueError

    elif all(
        [args.train_batches == 0, args.val_batches == 0, args.test_batches == 0]
    ):
        logger.error(
            "Error: at least one of train_batches, val_batches, or test_batches must be non-zero"
        )
        raise ValueError

    # Create config from YAML file
    if args.config is None:
        logger.warning(
            "Warning: Configuration file is not provided, using default parameters"
        )
        config = TabICLPriorConfig()  # Initialize with default parameters
    elif not args.config.endswith((".yaml", ".yml")):
        logger.warning(
            "Warning: Configuration file is not a YAML file, using default parameters"
        )
        config = TabICLPriorConfig()  # Initialize with default parameters
    else:
        config = TabICLPriorConfig.from_yaml(yaml_path=args.config)

    if args.prior_dir is not None:
        config.prior_dir = args.prior_dir

    config.n_jobs = args.n_jobs
    config.num_threads_per_generate = args.num_threads_per_generate
    config.batch_size = args.batch_size

    # Set random seeds from config
    if config.seed is not None:
        seed_everything(config.seed)

    # Create saver and run generation
    if args.single_prior_num > 0:
        for i in range(args.single_prior_num):
            config.prior_dir = os.path.join(args.prior_dir, f"single_prior_{i}")
            saver = SaveSinglePriorDataset(
                config=config,
                resume_from=args.resume_from,
                train_batches=args.train_batches,
                val_batches=args.val_batches,
                test_batches=args.test_batches,
            )
            saver.run()
    else:
        saver = SavePriorDataset(
            config=config,
            resume_from=args.resume_from,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            test_batches=args.test_batches,
            export_csv=args.export_csv,
            csv_output_dir=args.csv_output_dir,
            full_config=execution_config,
        )

        saver.run()
"""
RUN COMMANDS:
uv run src/synthefy_pkg/prior/genload.py   --config src/synthefy_pkg/prior/config/synthetic_configs/eval_datasets/large_convlag_synin_s.yaml   --prior_dir /workspace/data/synthetic_data/large_convlag_synin_s/   --train_batches 10000   --val_batches 100   --test_batches 100   --batch_size 64 --export_csv --n_jobs 16 --num_threads_per_generate 16 --dataset-config examples/configs/dataset_configs/dataset_configs/large_convlag_synin_s.yaml
uv run src/synthefy_pkg/prior/genload.py   --config src/synthefy_pkg/prior/config/synthetic_configs/eval_datasets/medium_convlag_synin_s.yaml   --prior_dir /workspace/data/synthetic_data/medium_convlag_synin_s/   --train_batches 10000   --val_batches 100   --test_batches 100   --batch_size 64 --export_csv --n_jobs 16 --num_threads_per_generate 16
uv run src/synthefy_pkg/prior/genload.py   --config src/synthefy_pkg/prior/config/synthetic_configs/eval_datasets/medium_obslag_synin_s.yaml   --prior_dir /workspace/data/synthetic_data/medium_obslag_synin_s/   --train_batches 10000   --val_batches 100   --test_batches 100   --batch_size 64 --export_csv --n_jobs 16 --num_threads_per_generate 16
uv run src/synthefy_pkg/prior/genload.py   --config src/synthefy_pkg/prior/config/synthetic_configs/eval_datasets/tiny_convlag_synin_ns.yaml   --prior_dir /workspace/data/synthetic_data/tiny_convlag_synin_ns/   --train_batches 10000   --val_batches 100   --test_batches 100   --batch_size 64 --export_csv --n_jobs 16 --num_threads_per_generate 16
uv run src/synthefy_pkg/prior/genload.py   --config src/synthefy_pkg/prior/config/synthetic_configs/eval_datasets/tiny_obslag_synin_ns.yaml   --prior_dir /workspace/data/synthetic_data/tiny_obslag_synin_ns/   --train_batches 10000   --val_batches 100   --test_batches 100   --batch_size 64 --export_csv --n_jobs 16 --num_threads_per_generate 16

RUNNING WITH FULL DATA
uv run src/synthefy_pkg/prior/genload.py   --config src/synthefy_pkg/prior/config/synthetic_configs/eval_datasets/tiny_convlag_mixedin_ns.yaml   --prior_dir /workspace/data/synthetic_data/tiny_obslag_synin_ns/   --train_batches 10000   --val_batches 100   --test_batches 100   --batch_size 64 --export_csv --n_jobs 16 --num_threads_per_generate 16 --execution-config examples/configs/foundation_model_configs/grid_icl_training_configs/embedder_train_with_expert_forecast_real_gift.yaml
uv run src/synthefy_pkg/prior/genload.py   --config src/synthefy_pkg/prior/config/synthetic_configs/eval_datasets/large_convlag_mixedin_s.yaml   --prior_dir /workspace/data/synthetic_data/tiny_obslag_synin_ns/   --train_batches 10000   --val_batches 100   --test_batches 100   --batch_size 64 --export_csv --n_jobs 16 --num_threads_per_generate 16 --execution-config examples/configs/foundation_model_configs/grid_icl_training_configs/embedder_train_with_expert_forecast_real_gift.yaml
"""
