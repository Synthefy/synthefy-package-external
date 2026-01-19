"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from scipy.stats import loguniform
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset, get_worker_info

from synthefy_pkg.configs.dataset_configs import DatasetConfig
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.configs.foundation_model_config import FoundationModelConfig
from synthefy_pkg.configs.tabicl_config import TabICLPriorConfig
from synthefy_pkg.data.v3_sharded_dataloader import V3ShardedDataloader
from synthefy_pkg.prior.hp_sampling import HpSamplerList
from synthefy_pkg.prior.mlp_scm import MLPSCM
from synthefy_pkg.prior.observation import Obs
from synthefy_pkg.prior.tree_scm import TreeSCM
from synthefy_pkg.utils.time_and_freq_utils import (
    FREQUENCIES_TO_SECONDS,
    add_synthetic_time_features,
    convert_timestamp_to_features,
    create_synthetic_date_range,
)

NUM_TIMESTAMP_COLUMNS = 5

warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Your `IterableDataset` has `__len__` defined.*",
    category=UserWarning,
)


def add_synthetic_timestamps(
    X: Tensor,
    d: list[Tensor],
    add_synthetic_timestamps: list[str],
    max_seq_len: int,
    max_features: int,
    frequencies: Optional[Union[list[str], np.ndarray]] = None,
    start_times: Optional[Union[list[int], np.ndarray]] = None,
) -> Tuple[
    Tensor,
    list[Tensor],
    Union[list[str], np.ndarray],
    Union[list[int], np.ndarray],
    int,
]:
    """
    Add synthetic timestamps to the input tensor.

    config.add_synthetic_timestamps is a list of timestamp frequencies:
    - minutely
    - hourly
    - daily
    - monthly

    TODO: currently only handles a single timestamp column in seconds
    """
    # TODO: this is a hack to ensure that the timestamps are not all the sam
    max_time = FREQUENCIES_TO_SECONDS["yearly"] * 100  # 100 years in seconds

    # get a random start time
    if frequencies is None:
        frequencies = np.random.choice(
            add_synthetic_timestamps, replace=True, size=X.shape[0]
        )
    if start_times is None:
        start_times = [
            int(st.item())
            for st in torch.randint(0, max_time, size=(X.shape[0],))
        ]
        total_times = [
            FREQUENCIES_TO_SECONDS[freq] * max_seq_len for freq in frequencies
        ]
        start_times = [
            st - total_time for st, total_time in zip(start_times, total_times)
        ]

    timestamp_columns = list()

    for freq, start_time in zip(frequencies, start_times):
        total_time = FREQUENCIES_TO_SECONDS[freq] * max_seq_len
        # logger.info(f"freq: {freq}, max_seq_len: {self.config.max_seq_len}, max_time: {max_time}, total_time: {total_time}")
        # Convert tensor to int for arange
        start = (
            start_time.item() if isinstance(start_time, Tensor) else start_time
        )
        timestamps = torch.arange(
            start,
            start + total_time,
            FREQUENCIES_TO_SECONDS[freq],
            device=X.device,
        )
        # Compute all components in parallel using tensor operations
        minutes = (
            timestamps % FREQUENCIES_TO_SECONDS["hourly"]
        ) // FREQUENCIES_TO_SECONDS["minutely"]
        hours = (
            timestamps % FREQUENCIES_TO_SECONDS["daily"]
        ) // FREQUENCIES_TO_SECONDS["hourly"]
        days = (
            timestamps % FREQUENCIES_TO_SECONDS["monthly"]
        ) // FREQUENCIES_TO_SECONDS["daily"]
        months = (
            timestamps % FREQUENCIES_TO_SECONDS["yearly"]
        ) // FREQUENCIES_TO_SECONDS["monthly"]
        years = timestamps // FREQUENCIES_TO_SECONDS["yearly"]
        timestamp_columns.append(
            torch.stack(
                (
                    minutes / 60,
                    hours / 24,
                    days / 30,
                    months / 12,
                    years / 100,
                ),
                dim=1,
            )
        )
    # logger.info(f"X.shape: {X.shape}, torch.stack(timestamp_columns).shape: {torch.stack(timestamp_columns).shape}")
    timestamp_columns = torch.stack(timestamp_columns)
    num_added_features = timestamp_columns.shape[-1]

    # replace the first k features with the timestamp columns, but shift d by k
    # roll X by the number of timestamp columns
    # logger.info(f"X before: {X[0,0]}, timestamp_columns: {timestamp_columns[0,0]}, d: {d[0]}")
    X = torch.roll(X, timestamp_columns.shape[-1], dims=-1)
    X[..., : timestamp_columns.shape[-1]] = timestamp_columns
    d = [
        torch.clamp(dv + timestamp_columns.shape[-1], 0, max_features)
        for dv in d
    ]
    # logger.info(f"X after: {X[0,0]}, d: {d[0]}")
    # X = torch.cat([X, torch.stack(timestamp_columns)], dim=-1)
    # replace
    return X, d, frequencies, start_times, num_added_features


class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    config : TabICLPriorConfig
        Configuration object containing all parameters for dataset generation
    """

    def __init__(
        self,
        config: TabICLPriorConfig,
        real_data_config: Optional[DatasetConfig] = None,
        fm_config: Optional[FoundationModelConfig] = None,
    ):
        self.config = config
        self.real_data_config = real_data_config
        self.fm_config = fm_config
        self.validate_train_size_range(
            config.min_train_size, config.max_train_size
        )

    @staticmethod
    def validate_train_size_range(
        min_train_size: Union[int, float], max_train_size: Union[int, float]
    ) -> None:
        """
        Checks if the training size range is valid.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size (position or ratio)

        max_train_size : int|float
            Maximum training size (position or ratio)

        Raises
        ------
        AssertionError
            If training size range is invalid
        ValueError
            If training size types are mismatched or invalid
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(
            max_train_size, (int, float)
        ):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size, (
                "0 < min_train_size < max_train_size"
            )
        elif isinstance(min_train_size, float) and isinstance(
            max_train_size, float
        ):
            assert 0 < min_train_size < max_train_size < 1, (
                "0 < min_train_size < max_train_size < 1"
            )
        else:
            raise ValueError(
                "Both training sizes must be of the same type (int or float)"
            )

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int],
        max_seq_len: int,
        log: bool = False,
        replay_small: bool = False,
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(
        min_train_size: Union[int, float],
        max_train_size: Union[int, float],
        seq_len: int,
    ) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(
            min_train_size, float
        ):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return min(100, max_features)
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
        """

        def filter_unique_features(
            xi: Tensor, di: int | Tensor
        ) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(
                xi[:, unique_mask],
                pad=(0, num_features - di_new),
                mode="constant",
                value=0,
            )
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [
            filter_unique_features(xi, di) for xi, di in zip(X, d)
        ]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        return X_new, d_new

    @staticmethod
    def sanity_check(
        X: Tensor,
        y: Tensor,
        train_size: int,
        n_attempts: int = 10,
        min_classes: int = 2,
    ) -> bool:
        """
        Verifies that both train and test sets contain all classes.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target labels tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """

        def is_valid_split(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return (
                set(unique_tr.tolist()) == set(unique_te.tolist())
                and len(unique_tr) >= min_classes
            )

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True

    def update_config(self, config: TabICLPriorConfig):
        """Update the configuration for this prior."""
        self.config = config


class SCMPrior(Prior):
    """
    Generates synthetic datasets using Structural Causal Models (SCM).

    The data generation process follows a hierarchical structure:
    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary transformations and checks.

    Parameters
    ----------
    config : TabICLPriorConfig
        Configuration object containing all parameters for dataset generation
    """

    def __init__(
        self,
        config: TabICLPriorConfig,
        real_data_config: Optional[DatasetConfig] = None,
        fm_config: Optional[FoundationModelConfig] = None,
    ):
        super().__init__(config, real_data_config, fm_config)
        self.fixed_hp = config.get_scm_fixed_hp()
        self.sampled_hp = config.get_scm_sampled_hp()
        config_dict = {
            "dataset_config": DictConfig(asdict(self.real_data_config))
            if self.real_data_config is not None
            else DictConfig({"dataset_name": "dummy"}),
            "foundation_model_config": DictConfig(asdict(self.fm_config))
            if self.fm_config is not None
            else DictConfig({}),
            "metadata_encoder_config": DictConfig({}),
            "training_config": DictConfig({}),
            "execution_config": DictConfig(
                {
                    "save_path": "dummy",
                    "run_name": "dummy",
                    "experiment_name": "dummy",
                    "dataset_name": "dummy",
                }
            ),
        }

        self.configuration = Configuration(config=DictConfig(config_dict))

        # Create V3ShardedDataloader with num_workers=0 to avoid multiprocessing issues
        self.configuration.dataset_config.num_workers = 0
        # set the batch size to 1
        self.configuration.dataset_config.batch_size = 1
        # set the time_series_length to max_seq_len
        self.configuration.dataset_config.time_series_length = (
            self.config.max_seq_len
        )
        # set the num_correlates to max_features
        self.configuration.dataset_config.num_correlates = (
            self.config.max_features
        )
        self.real_data_loader = None
        if self.real_data_config is not None and self._requires_real_data():
            data_loader = V3ShardedDataloader(self.configuration)
            self.real_data_loader = data_loader.train_dataloader()

    def _requires_real_data(self) -> bool:
        """
        Determine if the current configuration requires real data loading.

        Returns
        -------
        bool
            True if the configuration uses samplers that require real data ('real', 'mixed', 'stl')
        """
        # Check if scm_used_sampler configuration includes samplers that need real data
        sampler_config = self.config.scm_used_sampler
        if "choice_values" in sampler_config:
            sampler_choices = sampler_config["choice_values"]
            # Check if any of the sampler choices require real data
            return any(
                choice in ["real", "mixed", "stl"] for choice in sampler_choices
            )
        return False

    def update_config(self, config: TabICLPriorConfig):
        self.config = config
        self.fixed_hp = config.get_scm_fixed_hp()
        self.sampled_hp = config.get_scm_sampled_hp()

    def hp_sampling(self) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones
        """
        hp_sampler = HpSamplerList(
            self.sampled_hp, device=self.config.prior_device
        )
        sample = hp_sampler.sample()
        return sample

    def _prepare_scm_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and validate SCM parameters with common configuration.

        Parameters
        ----------
        params : dict
            Base parameters for SCM generation

        Returns
        -------
        dict
            Prepared parameters with all necessary configuration
        """
        # Add missing config parameters to SCM params
        params["real_data_loader"] = (
            self.real_data_loader if self.real_data_loader is not None else None
        )
        params["row_missing_prob"] = self.config.row_missing_prob
        params["column_has_missing_prob"] = self.config.column_has_missing_prob
        params["is_regression"] = self.config.is_regression
        params["device"] = self.config.prior_device
        params["max_features"] = self.config.max_features
        params["sampling_mixed_names"] = self.config.scm_mixed_names
        params["sigmoid_mixed_sampling_rate"] = (
            self.config.scm_sigmoid_mixed_sampling_rate
        )
        params["respect_ancestry_for_lag"] = (
            self.config.respect_ancestry_for_lag
        )
        has_one_layer = (
            self.config.scm_num_layers["lower_bound"]
            + self.config.scm_num_layers["max_mean"]
            < 0.5
        )
        params["return_single_output"] = (
            self.config.max_features == 0
            or (
                self.config.max_features == 5
                and self.config.add_synthetic_timestamps
            )
        ) and has_one_layer
        params["use_input_as_target"] = self.config.use_input_as_target

        # reset the sampling to mixed for tabular
        if params["used_sampler"] == "tabular":
            params["sampling"] = "mixed"

        return params

    def _extract_node_info(
        self,
        prior_obj: Union[MLPSCM, TreeSCM],
        indices_X: Tensor,
        indices_y: Optional[Tensor] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract node layer/index information from SCM prior object."""
        if isinstance(prior_obj, MLPSCM):
            # Use the built-in method for accurate node mapping
            node_info_array = np.zeros((len(indices_X), 2), dtype=np.int32)
            for j, feature_idx in enumerate(indices_X.tolist()):
                layer, local_idx = prior_obj.get_local_index_for_node(
                    feature_idx
                )
                node_info_array[j] = [layer, local_idx]

            # Extract target node information if provided
            target_node_info = None
            if indices_y is not None:
                target_node_info = np.zeros((len(indices_y), 2), dtype=np.int32)
                for j, target_idx in enumerate(indices_y.tolist()):
                    layer, local_idx = prior_obj.get_local_index_for_node(
                        target_idx
                    )
                    target_node_info[j] = [layer, local_idx]

            return node_info_array, target_node_info
        elif isinstance(prior_obj, TreeSCM):
            # TreeSCM doesn't have get_local_index_for_node, so use manual mapping
            num_causes = prior_obj.num_causes
            hidden_size = prior_obj.hidden_dim
            num_layers = prior_obj.num_layers

            # Create mapping from global index to (layer, local_index)
            global_to_layer = {}

            # Input layer (layer 0)
            for j in range(num_causes):
                global_to_layer[j] = (0, j)

            # Hidden layers
            counter = num_causes
            for layer_idx in range(num_layers):
                for j in range(hidden_size):
                    global_to_layer[counter] = (layer_idx + 1, j)
                    counter += 1

            # Map feature indices to node information
            node_info_array = np.zeros((len(indices_X), 2), dtype=np.int32)
            for j, feature_idx in enumerate(indices_X.tolist()):
                if feature_idx in global_to_layer:
                    layer, local_idx = global_to_layer[feature_idx]
                    node_info_array[j] = [layer, local_idx]
                else:
                    # Handle negative indices (output nodes)
                    layer = num_layers
                    local_idx = (
                        feature_idx + hidden_size
                        if feature_idx < 0
                        else feature_idx
                    )
                    node_info_array[j] = [layer, local_idx]

            # Extract target node information if provided
            target_node_info = None
            if indices_y is not None:
                target_node_info = np.zeros((len(indices_y), 2), dtype=np.int32)
                for j, target_idx in enumerate(indices_y.tolist()):
                    if target_idx in global_to_layer:
                        layer, local_idx = global_to_layer[target_idx]
                        target_node_info[j] = [layer, local_idx]
                    else:
                        # Handle negative indices (output nodes)
                        layer = num_layers
                        local_idx = (
                            target_idx + hidden_size
                            if target_idx < 0
                            else target_idx
                        )
                        target_node_info[j] = [layer, local_idx]

            return node_info_array, target_node_info
        else:
            # Fallback for other prior types
            return np.zeros((len(indices_X), 2), dtype=np.int32), None

    def _get_prior_class(self, prior_type: str):
        """
        Get the appropriate prior class based on prior type.

        Parameters
        ----------
        prior_type : str
            Type of prior to use ('mlp_scm' or 'tree_scm')

        Returns
        -------
        class
            The prior class to instantiate

        Raises
        ------
        ValueError
            If prior_type is not recognized
        """
        if prior_type == "mlp_scm":
            return MLPSCM
        elif prior_type == "tree_scm":
            return TreeSCM
        else:
            raise ValueError(f"Unknown prior type {prior_type}")

    def _generate_dataset_core(
        self,
        params: Dict[str, Any],
        from_csv: bool = False,
        dict_of_data: dict = {},
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Union[MLPSCM, TreeSCM],
        Dict[str, Any],
        Tensor,
        Tensor,
    ]:
        """
        Core dataset generation logic shared between generate_dataset and generate_dataset_object.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset

        Returns
        -------
        tuple
            Always returns (X, y, d, prior_object, indices_X, indices_y) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
            - prior_object: The SCM prior object
            - indices_X: Indices for X features
            - indices_y: Indices for y labels
        """
        prior_cls = self._get_prior_class(params["prior_type"])
        params = self._prepare_scm_params(params)

        while True:
            prior_object = prior_cls(**params)

            # Always get indices since we might need them
            if from_csv:
                X = torch.tensor(dict_of_data["X"])
                y = torch.tensor(dict_of_data["y"])
                indices_X = torch.tensor(np.arange(X.shape[1]))
                indices_y = indices_X[-1] + 1
                params["original_seq_len"] = X.shape[0]
                params["num_features"] = X.shape[1]
                params["train_size"] = dict_of_data["train_size"]
            else:
                X, y, indices_X, indices_y = prior_object(
                    return_indices=True,
                    exclude_inputs=self.config.exclude_inputs,
                    use_input_as_target=self.config.use_input_as_target,
                )

            X, y = Obs(params)(X, y, indices_X)

            assert "original_seq_len" in params, (
                "params must contain original_seq_len"
            )
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == params["max_features"], (
                f"X.shape[1]: {X.shape[1]}, params['max_features']: {params['max_features']}"
            )
            assert X.shape[0] == params["original_seq_len"]

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor(
                [params["num_features"]],
                device=self.config.prior_device,
                dtype=torch.long,
            )

            # Only keep valid datasets with sufficient features and balanced classes
            if not self.config.is_regression:
                X, d = self.delete_unique_features(X, d)
            if (d > 0).all() and (
                self.sanity_check(X, y, params["train_size"])
                or self.config.is_regression
            ):
                X, y, d = X.squeeze(0), y.squeeze(0), d.squeeze(0)

                if params["counterfactual_num_samples"] > 0:
                    X_cf, y_cf = self._generate_dataset_with_counterfactuals(
                        prior_object, params, indices_X, indices_y
                    )
                    num_drop = (
                        X.shape[0] - X_cf.shape[0]
                    )  # add counterfactuals to the end
                    X = torch.cat([X[:num_drop], X_cf], dim=0)
                    y = torch.cat([y[:num_drop], y_cf], dim=0)
                return X, y, d, prior_object, params, indices_X, indices_y

    def _generate_dataset_with_counterfactuals(
        self,
        prior_object: Union[MLPSCM, TreeSCM],
        params: Dict[str, Any],
        indices_X: Tensor,
        indices_y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Uses a prior object and visible indices to generate a counterfactual dataset.
        """
        # select a random subset to assign as counterfactuals
        counterfactual_indices = torch.randperm(indices_X.shape[0])[
            : params["scm_counterfactual_num_changes"]
        ]

        # generate a counterfactual dataset
        X, y, indices_X, indices_y = prior_object(
            return_indices=True,
            exclude_inputs="allow",
            indices_X_y=(indices_X, indices_y),
            counterfactual_indices=counterfactual_indices,
        )
        num_counterfactual_samples = params["scm_counterfactual_num_samples"]
        return X[-num_counterfactual_samples:], y[-num_counterfactual_samples:]

    @torch.no_grad()
    def generate_dataset(
        self,
        params: Dict[str, Any],
        from_csv: bool = False,
        dict_of_data: dict = {},
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """
        X, y, d, _, _, _, _ = self._generate_dataset_core(
            params, from_csv, dict_of_data
        )
        return X, y, d

    @torch.no_grad()
    def generate_dataset_object(
        self,
        params: Dict[str, Any],
        from_csv: bool = False,
        dict_of_data: dict = {},
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Union[MLPSCM, TreeSCM],
        Dict[str, Any],
        Tensor,
        Tensor,
    ]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            (X, y, d, prior_object, indices_X, indices_y) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
            - prior_object: The SCM prior object used for generation
            - indices_X: Indices for X features
            - indices_y: Indices for y labels
        """
        return self._generate_dataset_core(params, from_csv, dict_of_data)

    def sample_lag_features(
        self,
        min_lag: int,
        max_lag: float,
        num_features: int,
        seq_len: int,
    ) -> Tensor:
        """
        Sample lag of each feature for each dataset.

        Note: Lag sorting based on indices_X is handled in _create_lags in observation.py
        """
        return (
            torch.max(
                torch.rand(num_features, device=self.config.prior_device)
                * seq_len
                * max_lag,
                torch.tensor(min_lag, device=self.config.prior_device),
            )
            .floor()
            .int()
        )

    @torch.no_grad()
    def get_batch(
        self,
        batch_size: Optional[int] = None,
        probing: bool = False,
        from_csv: bool = False,
        dict_of_data: dict = {},
        node_info: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            np.ndarray,
            np.ndarray,
        ],
        list[Any],
    ]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses config.batch_size
        probing : bool, default=False
            If True, returns detailed objects for debugging
        from_csv : bool, default=False
            Whether to load data from CSV files
        dict_of_data : dict, default={}
            Dictionary containing data for CSV loading
        node_info : bool, default=False
            If True, returns additional node information array with layer/index data

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Labels tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)

        series_flags : list
            List of boolean flags indicating if each dataset is a time series

        node_info_array : np.ndarray, optional
            Node information array of shape (batch_size, max_features, 2) where the last dimension
            contains [layer_number, index_within_layer] for each feature. Only returned when node_info=True.
        """
        batch_size = batch_size or self.config.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.config.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(
            self.config.batch_size_per_gp, size_per_gp
        )  # Use same as batch_size_per_gp

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.config.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.config.min_seq_len,
                self.config.max_seq_len,
                log=self.config.log_seq_len,
                replay_small=self.config.replay_small,
            )
            global_train_size = self.sample_train_size(
                self.config.min_train_size,
                self.config.max_train_size,
                global_seq_len,
            )

        series_flags = list()
        timestamp_flags = list()

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.config.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.config.min_seq_len,
                    self.config.max_seq_len,
                    log=self.config.log_seq_len,
                    replay_small=self.config.replay_small,
                )
                gp_train_size = self.sample_train_size(
                    self.config.min_train_size,
                    self.config.max_train_size,
                    gp_seq_len,
                )
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(
                    gp_seq_len, self.config.max_features
                )
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.config.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            dataset_is_tabular = (
                np.random.random() < self.config.tabular_dataset_rate
            )
            dataset_has_lag = (
                np.random.random() < self.config.dataset_has_lag
            ) and not dataset_is_tabular
            if from_csv:
                dataset_has_lag = False
            series_flags += [not dataset_is_tabular] * (
                num_subgps_in_gp * size_per_subgp
            )

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(
                    size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp
                )
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type = self.get_prior()
                subgp_num_features = round(
                    np.random.uniform(
                        low=self.config.min_features, high=gp_max_features
                    )
                )
                subgp_sampled_hp = {
                    k: v() if callable(v) else v
                    for k, v in group_sampled_hp.items()
                }

                assert isinstance(gp_seq_len, int)

                # Same lag for all subgroups in this group
                if dataset_has_lag:
                    lags = self.sample_lag_features(
                        self.config.min_lag,
                        self.config.max_lag,
                        subgp_num_features,
                        gp_seq_len,
                    ).to(
                        self.config.prior_device
                    )  # TODO: these lags are not representative for functional lag
                    original_gp_seq_len = gp_seq_len
                    gp_seq_len = gp_seq_len + int(lags.max().item())
                else:
                    lags = None
                    original_gp_seq_len = gp_seq_len
                    if subgp_sampled_hp.get("use_functional_lag", False):
                        subgp_sampled_hp["use_functional_lag"] = (
                            False  # dataset lag supercedes functional lag
                        )

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    # Each dataset has its own number of classes
                    if (
                        np.random.random() > 0.5
                        and not self.config.is_regression
                    ):
                        ds_num_classes = np.random.randint(
                            2, self.config.max_classes + 1
                        )
                    else:
                        ds_num_classes = 2

                    # Create parameters dictionary for this dataset
                    # all of these parameters are dynamically sampled and cannot be passed in using config
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features
                        if self.config.seq_len_per_gp
                        else self.config.max_features,
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "lags": lags,
                        "original_seq_len": original_gp_seq_len,
                        "dataset_is_tabular": dataset_is_tabular,
                        "dataset_has_lag": dataset_has_lag,
                    }
                    dataset_has_timestamp = (
                        np.random.random()
                        < self.config.dataset_has_timestamp_rate
                    ) and not dataset_is_tabular
                    timestamp_flags.append(dataset_has_timestamp)
                    params["dataset_has_timestamp"] = dataset_has_timestamp

                    if dataset_is_tabular:
                        params["scm_sampling"] = {
                            "distribution": "meta_choice",
                            "choice_values": [
                                "mixed",
                                "normal",
                                "uniform",
                            ],  # "ts", "tabular", "real"
                        }
                    param_list.append(params)
                # Reset seq_len to original value
                if dataset_has_lag:
                    gp_seq_len = original_gp_seq_len

        # Use joblib to generate datasets in parallel
        # Note: n_jobs is set to 1 by default in config to avoid nested parallelism during DDP
        if (
            getattr(self.config, "n_jobs", 1) > 1
            and self.config.prior_device == "cpu"
            and not probing
            and not node_info
        ):
            with joblib.parallel_config(
                n_jobs=getattr(self.config, "n_jobs", 1),
                backend="loky",
                inner_max_num_threads=getattr(
                    self.config, "num_threads_per_generate", 1
                ),
            ):
                results = joblib.Parallel()(
                    joblib.delayed(self.generate_dataset)(
                        params, from_csv, dict_of_data
                    )
                    for params in param_list
                )
        else:
            if probing:
                results = [
                    self.generate_dataset_object(params, from_csv, dict_of_data)
                    for params in param_list
                ]
                return results
            elif node_info:
                results = [
                    self.generate_dataset_object(params, from_csv, dict_of_data)
                    for params in param_list
                ]
            else:
                results = [
                    self.generate_dataset(params, from_csv, dict_of_data)
                    for params in param_list
                ]

        # Handle different return types based on whether node_info is requested
        if node_info:
            # Extract node information from results
            (
                X_list,
                y_list,
                d_list,
                prior_objects,
                params_list,
                indices_X_list,
                indices_y_list,
            ) = zip(*results)

            # Create node information array using the helper function
            node_info_list = []
            target_node_info_list = []
            for prior_obj, indices_X, indices_y in zip(
                prior_objects, indices_X_list, indices_y_list
            ):
                node_info_array_single, target_node_info_single = (
                    self._extract_node_info(prior_obj, indices_X, indices_y)
                )
                node_info_list.append(node_info_array_single)
                target_node_info_list.append(target_node_info_single)
        else:
            # Standard case - only extract X, y, d
            X_list, y_list, d_list = zip(*results)
        # print(f"X_list: {len(X_list)}, y_list: {len(y_list)}, d_list: {len(d_list)}")

        # Combine Results
        if self.config.seq_len_per_gp:
            # Use nested tensrs for variable sequence lengths
            X = nested_tensor(
                [x.to(self.config.prior_device) for x in X_list],
                device=self.config.prior_device,
            )
            y = nested_tensor(
                [y.to(self.config.prior_device) for y in y_list],
                device=self.config.prior_device,
            )
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.config.prior_device)  # (B, T, H)
            y = torch.stack(y_list).to(self.config.prior_device)  # (B, T)
        # print(f"num_gps: {num_gps}, series_flags: {series_flags},")
        series_flags = torch.tensor(
            series_flags, device=self.config.prior_device
        )
        needs_timestamp_flags = torch.tensor(
            timestamp_flags, device=self.config.prior_device
        )

        if len(self.config.add_synthetic_timestamps) > 0:
            if self.config.add_time_stamps_as_features:
                X, d_list, num_added_features, _, frequencies, start_times = (
                    add_synthetic_time_features(
                        X,
                        d_list,
                        self.config.add_synthetic_timestamps,
                        self.config.max_features,
                        needs_timestamp_flags,
                    )
                )
            else:
                X, d_list, frequencies, start_times, num_added_features = (
                    add_synthetic_timestamps(
                        X,
                        d_list,
                        self.config.add_synthetic_timestamps,
                        self.config.max_seq_len,
                        self.config.max_features,
                    )
                )
            # if node info, roll the features to make room for time features
            if node_info:
                for i in range(len(node_info_list)):
                    # Roll the existing node info to make room for time features at the beginning
                    node_info_list[i] = np.roll(
                        node_info_list[i], num_added_features, axis=0
                    )
                    # Set the first len(frequencies) positions to indicate time features
                    for j in range(num_added_features):
                        node_info_list[i][j] = [
                            -1,
                            j,
                        ]  # Use -1 to indicate time feature, j for frequency index
            # y = self.add_synthetic_timestamps(y)

        # Metadata (always regular tensors)
        # print (len(d_list), len(X_list), len(y_list))
        d = torch.stack(d_list).to(
            self.config.prior_device
        )  # Actual number of features after filtering out constant ones
        # seq_lens = torch.tensor(
        #     [params["seq_len"] for params in param_list],
        #     device=self.config.prior_device,
        #     dtype=torch.long,
        # )
        seq_lens = torch.tensor(
            [params["original_seq_len"] for params in param_list],
            device=self.config.prior_device,
            dtype=torch.long,
        )
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list],
            device=self.config.prior_device,
            dtype=torch.long,
        )

        # Convert node_info_list to numpy array and return
        if node_info:
            # Use the actual number of features from node_info_list, not d
            max_features = max(len(ni) for ni in node_info_list)
            node_info_array = np.zeros(
                (len(node_info_list), max_features, 2), dtype=np.int32
            )

            for i, ni in enumerate(node_info_list):
                # Copy all available node info
                node_info_array[i, : len(ni)] = ni

            # Create target node info array (assuming single target per dataset)
            target_node_info_array = np.zeros(
                (len(target_node_info_list), 2), dtype=np.int32
            )
            for i, tni in enumerate(target_node_info_list):
                if tni is not None and len(tni) > 0:
                    target_node_info_array[i] = tni[
                        0
                    ]  # Take first target if multiple

            return (
                X,
                y,
                d,
                seq_lens,
                train_sizes,
                series_flags,
                node_info_array,
                target_node_info_array,
            )
        else:
            return X, y, d, seq_lens, train_sizes, series_flags

    def get_prior(self) -> str:
        """
        Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities.

        Returns
        -------
        str
            The selected prior type name
        """
        if self.config.prior_type == "mix_scm":
            return np.random.choice(
                ["mlp_scm", "tree_scm"],
                p=self.fixed_hp.get("mix_probs", [0.7, 0.3]),
            )
        else:
            return self.config.prior_type


class DummyPrior(Prior):
    """This class creates purely random data. This is useful for testing and debugging
    without the computational overhead of SCM-based generation.

    Parameters
    ----------
    config : TabICLPriorConfig
        Configuration object containing all parameters for dataset generation
    """

    # TODO fix init got swallowed

    @torch.no_grad()
    def get_batch(
        self,
        batch_size: Optional[int] = None,
        from_csv: bool = False,
        dict_of_data: dict = {},
        probing: bool = False,
        node_info: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            np.ndarray,
            np.ndarray,
        ],
        list[Any],
    ]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses config.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape (batch_size, seq_len).
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
            All datasets share the same split position.
        """
        batch_size = batch_size or self.config.batch_size
        seq_len = self.sample_seq_len(
            self.config.min_seq_len,
            self.config.max_seq_len,
            log=self.config.log_seq_len,
        )
        train_size = self.sample_train_size(
            self.config.min_train_size, self.config.max_train_size, seq_len
        )

        X = torch.randn(
            batch_size,
            seq_len,
            self.config.max_features,
            device=self.config.prior_device,
        )

        num_classes = np.random.randint(2, self.config.max_classes + 1)
        y = torch.randint(
            0,
            num_classes,
            (batch_size, seq_len),
            device=self.config.prior_device,
        )

        d = torch.full(
            (batch_size,),
            self.config.max_features,
            device=self.config.prior_device,
        )
        seq_lens = torch.full(
            (batch_size,), seq_len, device=self.config.prior_device
        )
        train_sizes = torch.full(
            (batch_size,), train_size, device=self.config.prior_device
        )
        series_flags = torch.ones(
            batch_size, dtype=torch.bool, device=self.config.prior_device
        )  # dummy returns all series

        # Create dummy node info array if needed
        if node_info:
            node_info_array = np.zeros(
                (batch_size, self.config.max_features, 2), dtype=np.int32
            )

        if probing:
            if node_info:
                return [
                    X,
                    y,
                    d,
                    None,
                    dict(),
                    seq_lens,
                    train_sizes,
                    node_info_array,
                ]
            else:
                return [X, y, d, None, dict(), seq_lens, train_sizes]

        if node_info:
            # Create dummy target node info array
            target_node_info_array = np.zeros((batch_size, 2), dtype=np.int32)
            return (
                X,
                y,
                d,
                seq_lens,
                train_sizes,
                series_flags,
                node_info_array,
                target_node_info_array,
            )
        else:
            return X, y, d, seq_lens, train_sizes, series_flags

    def update_config(self, config: TabICLPriorConfig):
        """Update the configuration for this prior."""
        self.config = config


class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    config : TabICLPriorConfig
        Configuration object containing all parameters for dataset generation
    """

    def __init__(
        self,
        config: TabICLPriorConfig,
        real_data_config: Optional[DatasetConfig] = None,
        fm_config: Optional[FoundationModelConfig] = None,
    ):
        super().__init__()

        if config.prior_type == "dummy":
            self.prior = DummyPrior(
                config, real_data_config=real_data_config, fm_config=fm_config
            )
        elif config.prior_type in ["mlp_scm", "tree_scm", "mix_scm"]:
            # Only pass real_data_config if the configuration actually requires it
            effective_real_data_config = (
                real_data_config
                if self._config_requires_real_data(config)
                else None
            )
            self.prior = SCMPrior(
                config,
                real_data_config=effective_real_data_config,
                fm_config=fm_config,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{config.prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', or 'dummy'."
            )
        self.config = config
        self.counter = 0
        self.check_for_updates_freq = config.check_for_updates_freq

        # Path to the shared curriculum configuration file
        temp_dir = Path(tempfile.gettempdir()) / "synthefy_curriculum"
        self.check_for_curriculum_config = config.check_for_curriculum_config
        self.curriculum_config_path = (
            temp_dir / f"curriculum_config_{config.run_id}.pkl"
        )

    def _config_requires_real_data(self, config: TabICLPriorConfig) -> bool:
        """
        Determine if the configuration requires real data loading.

        Parameters
        ----------
        config : TabICLPriorConfig
            The configuration to check

        Returns
        -------
        bool
            True if the configuration uses samplers that require real data ('real', 'mixed', 'stl')
        """
        # Check if scm_used_sampler configuration includes samplers that need real data
        sampler_config = config.scm_used_sampler
        if "choice_values" in sampler_config:
            sampler_choices = sampler_config["choice_values"]
            # Check if any of the sampler choices require real data
            return any(
                choice in ["real", "mixed", "stl"] for choice in sampler_choices
            )
        return False

    def _check_for_config_updates(self):
        """Check if there's an updated configuration file and apply it."""
        try:
            if (
                self.curriculum_config_path.exists()
                and self.check_for_curriculum_config
            ):
                logger.info(
                    f"Checking for config updates in {self.curriculum_config_path}"
                )
                with open(self.curriculum_config_path, "rb") as f:
                    new_config = pickle.load(f)

                # If config was updated, also update the prior
                if new_config != self.config:
                    logger.info(
                        f"Config updated, updating prior min_train_size {new_config.min_train_size} max_train_size {new_config.max_train_size}"
                    )
                    self.prior.update_config(new_config)
                    return True
            logger.debug(
                f"No config updates found in {self.curriculum_config_path}"
            )
            return False
        except Exception as e:
            # Silently fail to avoid disrupting data loading
            logger.error(f"Error checking for config updates: {e}")
            return False

    def get_batch(
        self,
        batch_size: Optional[int] = None,
        from_csv: bool = False,
        dict_of_data: dict = {},
        node_info: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            np.ndarray,
            np.ndarray,
        ],
    ]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
        """
        result = self.prior.get_batch(
            batch_size,
            from_csv=from_csv,
            dict_of_data=dict_of_data,
            node_info=node_info,
        )
        if isinstance(result, list):
            raise ValueError(
                "Probing mode is not supported for PriorDataset, access directly"
            )
        else:
            return result

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        world_size = 1
        num_workers = 1

        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers

        # In distributed training, we need to account for both processes (GPUs) and workers (for dataloader)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()

        self.length = self.config.dataset_length // (world_size * num_workers)

        return self

    def __len__(self) -> int:
        # This should be the total length of the dataset, not the length of the current worker
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        return self.config.dataset_length // world_size

    def __next__(
        self,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            np.ndarray,
            np.ndarray,
        ],
    ]:
        """
        Returns the next batch from the iterator.
        """
        if (
            self.check_for_updates_freq > 0
            and self.counter % self.check_for_updates_freq == 0
        ):
            self._check_for_config_updates()

        if self.counter >= self.length:
            self.counter = 0
            raise StopIteration

        self.counter += 1
        return self.get_batch(1)  # Next expects batch_size=1

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        sampling_info = ""
        if isinstance(self.prior, SCMPrior):  # Only SCMPrior has sampled_hp
            sampling_info = f"\n  sampling: {self.prior.sampled_hp['sampling']['choice_values']}"

        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.config.prior_type}\n"
            f"  batch_size: {self.config.batch_size}\n"
            f"  batch_size_per_gp: {self.config.batch_size_per_gp}\n"
            f"  features: {self.config.min_features} - {self.config.max_features}\n"
            f"  max classes: {self.config.max_classes}\n"
            f"  seq_len: {self.config.min_seq_len or 'None'} - {self.config.max_seq_len}\n"
            f"  sequence length varies across groups: {self.config.seq_len_per_gp}\n"
            f"  train_size: {self.config.min_train_size} - {self.config.max_train_size}\n"
            f"  device: {self.config.prior_device}\n"
            f"  is_regression: {self.config.is_regression}{sampling_info}\n"
            f"  n_jobs: {self.config.n_jobs}\n"
            f"  num_threads_per_generate: {self.config.num_threads_per_generate}\n"
            f")"
        )

    def update_config(self, config: TabICLPriorConfig):
        self.config = config
        self.prior.update_config(config)


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout
