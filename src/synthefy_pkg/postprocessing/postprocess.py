import gc
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import CRPS as pscore
import einops
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bokeh.layouts import layout
from bokeh.plotting import output_file, save
from dotenv import load_dotenv
from loguru import logger
from matplotlib.axes import Axes
from omegaconf import DictConfig
from tqdm import tqdm

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.postprocessing.utils import (
    _load_metadata_col_names,
    dynamic_downsample_true_and_predicted_values,
    plot_learning_curve,
)
from synthefy_pkg.utils.basic_utils import ENDC, OKYELLOW
from synthefy_pkg.utils.post_preprocess_utils import (
    compute_covariance_matrix,
    create_cross_corr_elements,
    create_metrics_table,
    plot_covariance_matrix,
    plot_covariance_matrix_static,
    process_continuous_columns,
    safe_load_npy_file,
)
from synthefy_pkg.utils.scaling_utils import (
    load_timeseries_col_names,
    unscale_windows_dict,
)

SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")
assert SYNTHEFY_PACKAGE_BASE is not None
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE", "")

# To have a clear plot, limit the number of windows to plot
PLOT_WINDOWS_LIMIT = 9

# To have a clear plot, limit the number of channels to plot
PLOT_CHANNELS_LIMIT = 6

# Validate environment variables
if not SYNTHEFY_DATASETS_BASE:
    raise EnvironmentError(
        "Required environment variable `SYNTHEFY_DATASETS_BASE` is not set."
    )


np.random.seed(42)

COMPILE = False


@dataclass
class PostprocessorConfig:
    dataset_types: List[str] = field(
        default_factory=lambda: ["train", "val", "test"]
    )
    best_worst_windows_number: int = 10
    metric_types: List[str] = field(
        default_factory=lambda: [
            "MAE",
            "MSE",
            "RMSE",
            "MAPE",
            "MDAPE",
            "SMAPE",
        ]
    )
    best_worst_metrics: List[str] = field(
        default_factory=lambda: ["MAE", "RMSE", "MAPE", "MDAPE", "SMAPE"]
    )


class Postprocessor:
    def __init__(
        self,
        config: Optional[DictConfig] = None,
        config_filepath: Optional[str] = None,
        fmv2_filepath: Optional[str] = None,
        splits: List[str] = ["test"],
        downsample_factor: Optional[int] = None,
    ):
        self.config = PostprocessorConfig()
        self.metrics_dict = {}
        self.best_worst_indices = {}
        self.all_data_metrics_dict = {}
        self.windows_data_dict = {}

        # fmv2 = foundation model v2
        if fmv2_filepath is None:
            self._load_configuration(config, config_filepath)
        else:
            self._load_fmv2_data(fmv2_filepath)
        self.config.dataset_types = splits
        self.downsample_factor = downsample_factor
        self.is_foundation_model = False

        # Set task and probabilistic forecast settings first
        self.use_probabilistic_forecast = False
        if self.configuration.task == "synthesis":
            self.label = "Synthesized"
        elif "forecast" in self.configuration.task:
            self.label = "Forecasted"
            self.use_probabilistic_forecast = (
                hasattr(self.configuration, "denoiser_config")
                and self.configuration.denoiser_config is not None
                and self.configuration.denoiser_config.use_probabilistic_forecast
            )
        elif "fmv2" in self.configuration.task:
            self.label = "Forecasted"
        else:
            self.label = "Predicted"

        if fmv2_filepath is None:
            self._initialize_data_structures()  # Now this will use the correct dataset_types
            self._load_data()
            self._unscale_and_save_unscaled_data()
            # Create channel-specific subdirectories for plots
            self._create_channel_subdirectories()
        self._get_eval_metrics_by_dataset_type()

    def _load_configuration(self, config, config_filepath) -> None:
        """Load and process the configuration."""
        self.configuration = Configuration(
            config=config, config_filepath=config_filepath
        )
        self.timeseries_col_names = load_timeseries_col_names(
            self.configuration.dataset_config.dataset_name,
            self.configuration.dataset_config.num_channels,
        )
        self.discrete_col_names, self.continuous_col_names = (
            _load_metadata_col_names(
                self.configuration.dataset_config.dataset_name
            )
        )

    def _load_fmv2_data(self, fmv2_filepath: str) -> None:
        """
        Load FMv2 data from h5 file and configure settings.

        Parameters:
            fmv2_filepath: Path to the h5 file containing FMv2 data
        """
        logger.info(f"Loading FMv2 data from {fmv2_filepath}")

        # Extract dataset name from filepath (assuming last directory in path is dataset name)
        dataset_name = os.path.basename(os.path.dirname(fmv2_filepath))

        self.discrete_col_names, self.continuous_col_names = [], []

        self.windows_data_dict = {"real": {}, "synthesized": {}}
        self.saved_data_path = os.path.dirname(fmv2_filepath)
        self.figsave_path = os.path.join(str(self.saved_data_path), "plots/")
        os.makedirs(self.figsave_path, exist_ok=True)
        # Load data from h5 file
        try:
            with h5py.File(fmv2_filepath, "r") as f:
                # Map the keys from the h5 file to our expected structure
                true_forecast = np.array(f["true_forecast"])
                predicted_forecast = np.array(f["predicted_forecast"])
                history = np.array(f["history"])
                assert isinstance(true_forecast, np.ndarray)
                assert isinstance(predicted_forecast, np.ndarray)
                self.windows_data_dict["real"]["test"] = true_forecast
                self.windows_data_dict["synthesized"]["test"] = (
                    predicted_forecast
                )
                # TODO: update this to actually unscale once the model output
                # includes scaler parameters, now not possible
                self.unscaled_windows_data_dict = self.windows_data_dict

                # Validate shapes
                real_shape = self.windows_data_dict["real"]["test"].shape
                synth_shape = self.windows_data_dict["synthesized"][
                    "test"
                ].shape

                assert len(real_shape) == 3, (
                    f"Expected true_forecast to have shape [num_windows, num_channels, window_size], got {real_shape}"
                )
                assert len(synth_shape) == 3, (
                    f"Expected predicted_forecast to have shape [num_windows, num_channels, window_size], got {synth_shape}"
                )
                assert real_shape == synth_shape, (
                    f"Shape mismatch: true_forecast {real_shape} vs predicted_forecast {synth_shape}"
                )

                # Create configuration using DictConfig
                dataset_config = {
                    "dataset_name": dataset_name,
                    "num_channels": real_shape[1],
                    "forecast_length": real_shape[2],
                    "time_series_length": history.shape[-1],
                }

                config_dict = {
                    "dataset_config": dataset_config,
                    "task": "fmv2",
                    "experiment_name": "",
                    "run_name": "",
                    "generation_save_path": os.path.dirname(
                        os.path.dirname(fmv2_filepath)
                    ),
                }

                self.configuration = DictConfig(config_dict)

                # Load timeseries column names and metadata column names
                self.timeseries_col_names = [
                    f"channel_{i}"
                    for i in range(
                        self.configuration.dataset_config.num_channels
                    )
                ]

                # Create channel-specific subdirectories for plots
                self._create_channel_subdirectories()

                logger.info(
                    f"Successfully loaded FMv2 data with shape {real_shape}"
                )

        except Exception as e:
            raise IOError(
                f"Failed to load FMv2 data from {fmv2_filepath}: {str(e)}"
            )

        for dataset_type in self.config.dataset_types:
            self.metrics_dict[dataset_type] = {}
            self.best_worst_indices[dataset_type] = {
                col_name: {} for col_name in self.timeseries_col_names
            }
            self.all_data_metrics_dict[dataset_type] = {}

    def _get_eval_metrics_by_dataset_type(self):
        """
        Get evaluation metrics for all windows altogether and write into a class attr dict
        """
        for dataset_type in self.config.dataset_types:
            logger.info(f"Processing {dataset_type} data ...")
            self._get_metrics(dataset_type)
        self._save_results()
        self._log_summary()

    def _initialize_data_structures(self) -> None:
        """Initialize data structures used in postprocessing."""
        for dataset_type in self.config.dataset_types:
            self.metrics_dict[dataset_type] = {}
            self.best_worst_indices[dataset_type] = {
                col_name: {} for col_name in self.timeseries_col_names
            }
            self.all_data_metrics_dict[dataset_type] = {}

        self.windows_data_dict = {
            k: {i: np.array([]) for i in self.config.dataset_types}
            for k in [
                "real",
                "synthesized",
            ]
        }
        self.metadata_dict = {
            k: {i: np.array([]) for i in self.config.dataset_types}
            for k in [
                "discrete_conditions",
                "continuous_conditions",
            ]
        }

        self.saved_data_path = os.path.join(
            str(SYNTHEFY_DATASETS_BASE),
            self.configuration.generation_save_path,
            self.configuration.dataset_config.dataset_name,
            self.configuration.experiment_name,
            self.configuration.run_name,
        )

        self.figsave_path = os.path.join(str(self.saved_data_path), "plots/")
        os.makedirs(self.figsave_path, exist_ok=True)
        self.lightning_logs_path = self.configuration.get_lightning_logs_path(
            str(SYNTHEFY_DATASETS_BASE)
        )

    def _load_data(self) -> None:
        """
        Decide whether to run _load_data_h5 or _load_data_pkl based on which files exist.
        """
        type_i = self.config.dataset_types[
            0
        ]  # Using the first dataset type to check if the data exists
        # Check if probabilistic folder exists when using probabilistic forecasting
        data_save_path = os.path.join(self.saved_data_path, f"{type_i}_dataset")
        logger.info(f"HELP: {self.use_probabilistic_forecast=}")
        if self.use_probabilistic_forecast:
            probabilistic_path = os.path.join(
                self.saved_data_path, f"probabilistic_{type_i}_dataset"
            )
            if os.path.exists(probabilistic_path):
                data_save_path = probabilistic_path
            else:
                data_save_path = os.path.join(
                    self.saved_data_path, f"{type_i}_dataset"
                )
                logger.info(
                    f"HELP: Probabilistic data {probabilistic_path} does not exist. Using non-probabilistic save directory instead: {data_save_path}"
                )
        else:
            logger.info(
                f"HELP2: Using non-probabilistic data save directory: {data_save_path}"
            )

        if any(f.endswith(".h5") for f in os.listdir(data_save_path)):
            self._load_data_h5()
        elif any(f.endswith(".pkl") for f in os.listdir(data_save_path)):
            self._load_data_pkl()
        else:
            raise FileNotFoundError(
                f"No .h5 or .pkl files found in {data_save_path}"
            )

    def _load_data_h5(self) -> None:
        """
        Load the true and synthesized timeseries data into the 'all_window_info' dict
        from an h5 file.
        """
        for type_i in self.config.dataset_types:
            if self.use_probabilistic_forecast:
                probabilistic_path = os.path.join(
                    self.saved_data_path, f"probabilistic_{type_i}_dataset"
                )
                if os.path.exists(probabilistic_path):
                    data_save_path = probabilistic_path
                    h5_file = os.path.join(
                        data_save_path,
                        f"probabilistic_{type_i}_combined_data.h5",
                    )
                else:
                    # TODO: Should we raise an error here instead?? might be okay...
                    raise FileNotFoundError(
                        f"Probabilistic data {probabilistic_path} does not exist, but use_probabilistic_forecast is set to True. Please check the config file."
                    )
                    data_save_path = os.path.join(
                        self.saved_data_path, f"{type_i}_dataset"
                    )
                    h5_file = os.path.join(
                        data_save_path, f"{type_i}_combined_data.h5"
                    )
            else:
                data_save_path = os.path.join(
                    self.saved_data_path, f"{type_i}_dataset"
                )
                h5_file = os.path.join(
                    data_save_path, f"{type_i}_combined_data.h5"
                )

            print(OKYELLOW + f"Loading data from {h5_file}" + ENDC)

            data_dict = {}
            with h5py.File(h5_file, "r") as f:
                for key in f.keys():
                    data_dict[key] = np.array(f[key])

            assert len(data_dict.keys()) > 0, (
                f"No data found in {h5_file}. Investigate if experiment.generate_synthetic_data ran correctly."
            )

            if (
                "predicted_forecast" in data_dict
                and "true_forecast" in data_dict
                and "history" in data_dict
            ):
                # NOTE: history data has predicted_forecast as the last forecast_length data points (not group truth)
                # But we before that we have common history data for true and predicted forecasts
                history_data = data_dict["history"][
                    :,
                    :,
                    0,
                    : -self.configuration.dataset_config.forecast_length,
                ]
                # Prepend history data to both real and synthesized data
                self.windows_data_dict["real"][type_i] = np.concatenate(
                    [history_data, data_dict["true_forecast"]], axis=-1
                )
                self.windows_data_dict["synthesized"][type_i] = np.concatenate(
                    [history_data, data_dict["predicted_forecast"]], axis=-1
                )

                # Foundation model keys in h5 files are different than forecasting model keys
                # Having keys predicted_forecast, true_forecast, and history indicates
                # that this is a foundation model
                self.is_foundation_model = True
            else:
                self.windows_data_dict["real"][type_i] = data_dict[
                    "original_timeseries"
                ]
                self.windows_data_dict["synthesized"][type_i] = data_dict[
                    "synthetic_timeseries"
                ]
                self.metadata_dict["discrete_conditions"][type_i] = data_dict[
                    "discrete_conditions"
                ]
                self.metadata_dict["continuous_conditions"][type_i] = data_dict[
                    "continuous_conditions"
                ]

        # Check if we need to adjust real data to match time_series_length
        # This is necessary only for the baseline model case where the
        # real data length is not equal to the time_series_length
        for type_i in self.config.dataset_types:
            if type_i in self.windows_data_dict["real"]:
                real_data = self.windows_data_dict["real"][type_i]
                if hasattr(self.configuration, "dataset_config") and hasattr(
                    self.configuration.dataset_config, "time_series_length"
                ):
                    time_series_length = (
                        self.configuration.dataset_config.time_series_length
                    )
                    if real_data.shape[2] != time_series_length:
                        # Take the last time_series_length data points
                        self.windows_data_dict["real"][type_i] = real_data[
                            :, :, -time_series_length:
                        ]

                # Check for NaNs in the real data and remove windows with NaNs
                real_data = self.windows_data_dict["real"][type_i]
                nan_windows = np.isnan(real_data).any(axis=(1, 2))
                if np.any(nan_windows):
                    print(
                        f"Found {np.sum(nan_windows)} windows with NaNs in {type_i} data. Removing them."
                    )
                    # Keep only windows without NaNs
                    self.windows_data_dict["real"][type_i] = real_data[
                        ~nan_windows
                    ]
                    # Also remove corresponding windows from synthesized data
                    if type_i in self.windows_data_dict["synthesized"]:
                        self.windows_data_dict["synthesized"][type_i] = (
                            self.windows_data_dict["synthesized"][type_i][
                                ~nan_windows
                            ]
                        )

    def _load_data_pkl(self) -> None:
        """
        Load the true and synthesized timeseries data into the
        `all_window_info` dict
        """
        for type_i in self.config.dataset_types:
            if self.use_probabilistic_forecast:
                assert type_i == "test", (
                    "Probabilistic forecast is only supported for test data"
                )
                data_save_path = os.path.join(
                    self.saved_data_path, f"probabilistic_{type_i}_dataset"
                )
            else:
                data_save_path = os.path.join(
                    self.saved_data_path, f"{type_i}_dataset"
                )

            logger.info(f"Loading data from {data_save_path}")

            # Note: We load the .pkl files in ascending order to preserve ordering.
            # This is critical to make sure the data is aligned correctly.
            pkl_files = [
                f for f in os.listdir(data_save_path) if f.endswith(".pkl")
            ]

            data_dict = {
                "original_timeseries": [],
                "synthetic_timeseries": [],
                "discrete_conditions": [],
                "continuous_conditions": [],
            }
            for pkl_file in pkl_files:
                if not os.path.exists(os.path.join(data_save_path, pkl_file)):
                    raise FileNotFoundError(
                        f"File {pkl_file} does not exist in {data_save_path}; possible data corruption."
                    )
                unpickle_data = torch.load(
                    os.path.join(data_save_path, pkl_file)
                )

                for key in data_dict.keys():
                    if key not in unpickle_data:
                        raise ValueError(f"Key {key} not found in {pkl_file}")
                    data_dict[key].append(unpickle_data[key])

            self.windows_data_dict["real"][type_i] = np.concatenate(
                data_dict["original_timeseries"], axis=0
            )
            self.windows_data_dict["synthesized"][type_i] = np.concatenate(
                data_dict["synthetic_timeseries"], axis=0
            )
            self.metadata_dict["discrete_conditions"][type_i] = np.concatenate(
                data_dict["discrete_conditions"], axis=0
            )
            self.metadata_dict["continuous_conditions"][type_i] = (
                np.concatenate(data_dict["continuous_conditions"], axis=0)
            )

    def _unscale_and_save_unscaled_data(self):
        """
        Unscale and save unscaled continuous and discrete data to files.
        """
        # TODO: This is a hack to avoid unscaling the data for foundation models
        # TODO: once scaler parameters are saved use them to unscale the data
        if self.is_foundation_model:
            self.unscaled_windows_data_dict = self.windows_data_dict
            return

        def save_windows(filepath: str, data: np.ndarray) -> None:
            """
            Save data to an npy file
            """
            np.save(filepath, data)

        unscaled_data_path = os.path.join(self.saved_data_path, "unscaled_data")
        if self.use_probabilistic_forecast:
            unscaled_data_path = os.path.join(
                self.saved_data_path, "unscaled_data_probabilistic"
            )
        os.makedirs(unscaled_data_path, exist_ok=True)

        # Save unscaled continuous and discrete conditions (train, val, test) windows
        original_discrete_windows = unscale_windows_dict(
            windows_data_dict=self.metadata_dict.pop("discrete_conditions"),
            window_type="discrete",
            dataset_name=self.configuration.dataset_config.dataset_name,
        )

        continuous_windows = unscale_windows_dict(
            windows_data_dict=self.metadata_dict.pop("continuous_conditions"),
            window_type="continuous",
            dataset_name=self.configuration.dataset_config.dataset_name,
            original_discrete_windows=original_discrete_windows,
        )

        for dataset_type in self.config.dataset_types:
            save_windows(
                os.path.join(
                    unscaled_data_path,
                    f"{dataset_type}_discrete_conditions_unscaled.pkl",
                ),
                original_discrete_windows[dataset_type],
            )
            save_windows(
                os.path.join(
                    unscaled_data_path,
                    f"{dataset_type}_continuous_conditions_unscaled.pkl",
                ),
                continuous_windows[dataset_type],
            )

        # Save unscaled timeseries (train, val, test) windows
        timeseries_dict_keys = ["real", "synthesized"]
        self.unscaled_windows_data_dict = {}

        # we will use the unscaled data to compute metrics
        for timeseries_dict_key in timeseries_dict_keys:
            dict_ = {}
            if (
                self.use_probabilistic_forecast
                and timeseries_dict_key == "synthesized"
            ):
                # reshape the data to (num_windows * num_samples, num_channels, window_size)
                original_shape = list(
                    self.windows_data_dict[timeseries_dict_key].values()
                )[0].shape
                assert len(original_shape) == 4, (
                    f"Probabilistic forecast only generates 4D data; has shape {original_shape}"
                )
                for dataset_type in self.config.dataset_types:
                    dict_[dataset_type] = self.windows_data_dict[
                        timeseries_dict_key
                    ][dataset_type].copy()
                    dict_[dataset_type] = dict_[dataset_type].reshape(
                        -1, original_shape[-2], original_shape[-1]
                    )
            else:
                # Make a deep copy to ensure we don't modify the original data
                dict_ = {
                    k: v.copy()
                    for k, v in self.windows_data_dict[
                        timeseries_dict_key
                    ].items()
                }

            # unscale the data
            unscaled_windows_dict = unscale_windows_dict(
                windows_data_dict=dict_.copy(),  # Pass a copy to prevent modification of original
                window_type="timeseries",
                dataset_name=self.configuration.dataset_config.dataset_name,
                original_discrete_windows=original_discrete_windows,
            )

            if (
                self.use_probabilistic_forecast
                and timeseries_dict_key == "synthesized"
            ):
                # reshape the data to (num_windows, num_samples, num_channels, window_size)
                for dataset_type in self.config.dataset_types:
                    unscaled_windows_dict[dataset_type] = unscaled_windows_dict[
                        dataset_type
                    ].reshape(original_shape)

            for dataset_type in self.config.dataset_types:
                save_windows(
                    os.path.join(
                        unscaled_data_path,
                        f"{dataset_type}_{timeseries_dict_key}_timeseries_unscaled.pkl",
                    ),
                    unscaled_windows_dict[dataset_type],
                )
            self.unscaled_windows_data_dict[timeseries_dict_key] = (
                unscaled_windows_dict
            )

        logger.info(f"Unscaled data saved to: {unscaled_data_path}")
        del self.metadata_dict
        gc.collect()

    def _get_all_data_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        type_i: str,
        scaled: bool = True,
    ) -> None:
        """
        inputs:
            y_true: 3d numpy array of shape (num_windows, num_channels, window_size)
            y_pred: 3d numpy array of shape (num_windows, num_channels, window_size)
            type_i: either 'train', 'val', 'test'
            scaled: whether the data is scaled or not

        Get evaluation metrics for all windows altogether and write into a dict
        named self.all_data_metrics_dict

        Returns: None
        Side Effect: Implicitly updates self.all_data_metrics_dict with metrics from self.config.metric_types, on the split in type_i, for each channel;
        The channels are specified by the order of the columns
        """
        # y_true.shape = (num_windows, num_channels, window_size)
        # y_pred.shape = (num_windows, num_channels, window_size)
        # y_true.transpose(1, 0, 2).shape = (num_channels, num_windows, window_size)
        # y_true.shape[1] = num_channels
        # y_true.transpose(1, 0, 2).reshape(NUM_CHANNELS, -1).shape = (num_channels, num_windows * window_size)

        assert y_true.shape == y_pred.shape, (
            f"y_true and y_pred must have the same shape; {y_true.shape=} {y_pred.shape=}"
        )
        (NUM_WINDOWS, NUM_CHANNELS, WINDOW_SIZE) = y_true.shape
        y_true = y_true.transpose(1, 0, 2).reshape(NUM_CHANNELS, -1)
        y_pred = y_pred.transpose(1, 0, 2).reshape(NUM_CHANNELS, -1)

        # Desired inputs to _calculate_metrics: (window_size, num_channels * num_windows)
        # mean_axis=1, because we want to mean across the (num_channels * num_windows) dim.

        # (metric_types, channels)
        metrics_values_list = self._calculate_metrics(
            y_true, y_pred, mean_axis=1
        )

        if scaled:
            str_to_add = "scaled"
        else:
            str_to_add = "unscaled"

        for i in range(self.configuration.dataset_config.num_channels):
            self.all_data_metrics_dict[type_i][
                self.timeseries_col_names[i] + "_" + str_to_add
            ] = {
                metric_i: metrics_values_list[ind][i]
                for ind, metric_i in enumerate(self.config.metric_types)
            }

    def _get_probabilistic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        type_i: str,
        scaled: bool = True,
    ) -> None:
        """
        Get probabilistic evaluation metrics for all windows altogether and write into a dict
        named self.all_data_metrics_dict

        Inputs:
            y_true: 3d numpy array of shape (num_windows, num_channels, forecast_length)
            y_pred: 3d numpy array of shape (num_windows, num_samples, num_channels, forecast_length)
            type_i: either 'train', 'val', 'test'
            scaled: whether the data is scaled or not

            Note that y_true and y_pred have already clipped the history.
        """

        # flatten y_true and y_pred such that there are multiple predictions per time stamp
        for i in range(self.configuration.dataset_config.num_channels):
            per_channel_y_true = y_true[
                :, i, :
            ]  # (num_windows, forecast_length)
            assert (
                per_channel_y_true.shape[-1]
                == self.configuration.dataset_config.forecast_length
            ), (
                f"forecast_length does not match; should already clip out the history. Real shape: {per_channel_y_true.shape=}; Should have dimension of {self.configuration.dataset_config.forecast_length=}"
            )
            per_channel_y_true = (
                per_channel_y_true.flatten()
            )  # (num_windows * forecast_length)
            per_channel_y_pred = y_pred[
                :, :, i, :
            ]  # (num_windows, num_samples, forecast_length)
            assert (
                per_channel_y_pred.shape[-1]
                == self.configuration.dataset_config.forecast_length
            ), (
                f"forecast_length does not match; should already clip out the history. Real shape: {per_channel_y_pred.shape=}; Should have dimension of {self.configuration.dataset_config.forecast_length=}"
            )
            per_channel_y_pred = np.einsum(
                "bsh->sbh", per_channel_y_pred
            )  # (num_samples, num_windows, forecast_length)
            per_channel_y_pred = per_channel_y_pred.reshape(
                per_channel_y_pred.shape[0], -1
            )  # (num_samples, num_windows * forecast_length)
            assert per_channel_y_true.shape[0] == per_channel_y_pred.shape[1], (
                f"num_windows * forecast_length in y_true and y_pred do not match: {per_channel_y_true.shape[0]} vs {per_channel_y_pred.shape[1]}"
            )

            crps_scores = []
            for j in tqdm(
                range(per_channel_y_true.shape[0]),
                desc=f"Computing CRPS for {self.timeseries_col_names[i]}",
                total=per_channel_y_true.shape[0],
            ):
                crps_scores.append(
                    pscore.CRPS(
                        per_channel_y_pred[:, j], per_channel_y_true[j]
                    ).compute()[0]
                )

            crps_scores = np.array(crps_scores)
            str_to_add = "scaled" if scaled else "unscaled"
            self.all_data_metrics_dict[type_i][
                self.timeseries_col_names[i] + "_" + str_to_add
            ]["CRPS-MEAN"] = crps_scores.mean()
            self.all_data_metrics_dict[type_i][
                self.timeseries_col_names[i] + "_" + str_to_add
            ]["CRPS-STD"] = crps_scores.std()

    def _get_best_worst_windows_inds(
        self,
        real_data: np.ndarray,
        synthesized_data: np.ndarray,
        type_i: str,
    ) -> None:
        """
        Get index values of the best and worst windows by an evaluation metric.
        This function doesn't deal with probabilistic forecast.
        """
        for i in range(self.configuration.dataset_config.num_channels):
            # Get df with evaluation metrics values
            per_channel_real_data = real_data[:, i, :]
            per_channel_synthesized_data = synthesized_data[:, i, :]

            # normalize the data to be between 0 and 1
            max_val = np.max(per_channel_real_data, axis=1, keepdims=True)
            min_val = np.min(per_channel_real_data, axis=1, keepdims=True)
            normalized_real_data = (per_channel_real_data - min_val) / (
                max_val - min_val
            )
            normalized_synthesized_data = (
                per_channel_synthesized_data - min_val
            ) / (max_val - min_val)

            metrics_dict = {}

            # axis=1 means for metric over the batch dimension
            # TODO - use a library function for the metrics to avoid errors.
            for metric in self.config.best_worst_metrics:
                if metric == "MAE":
                    error_list = np.mean(
                        np.abs(
                            normalized_real_data - normalized_synthesized_data
                        ),
                        axis=1,
                    )
                elif metric == "RMSE":
                    error_list = np.sqrt(
                        np.mean(
                            np.square(
                                normalized_real_data
                                - normalized_synthesized_data
                            ),
                            axis=1,
                        )
                    )
                elif metric == "MAPE":
                    error_list = (
                        np.mean(
                            np.abs(
                                normalized_real_data
                                - normalized_synthesized_data
                            )
                            / (np.abs(normalized_real_data) + 1e-10),
                            axis=1,
                        )
                        * 100
                    )
                elif metric == "MDAPE":
                    error_list = (
                        np.median(
                            np.abs(
                                normalized_real_data
                                - normalized_synthesized_data
                            )
                            / (np.abs(normalized_real_data) + 1e-10),
                            axis=1,
                        )
                        * 100
                    )
                elif metric == "SMAPE":
                    # https://permetrics.readthedocs.io/en/latest/pages/regression/SMAPE.html
                    smape_factor = 200
                    error_list = smape_factor * np.mean(
                        np.abs(
                            normalized_real_data - normalized_synthesized_data
                        ),
                        axis=1,
                    )
                metrics_dict[metric] = error_list

                # Store indices for best/worst windows
                best_worst_indices = np.argsort(error_list)
                self.best_worst_indices[type_i][self.timeseries_col_names[i]][
                    metric
                ] = {
                    "Best": best_worst_indices[
                        : self.config.best_worst_windows_number
                    ].tolist(),
                    "Worst": best_worst_indices[
                        -self.config.best_worst_windows_number :
                    ].tolist(),
                }

            # Store metrics DataFrame
            self.metrics_dict[type_i][self.timeseries_col_names[i]] = (
                pd.DataFrame(metrics_dict)
            )

    def _find_nearest_neighbors(
        self, query_timeseries: np.ndarray, reference_timeseries: np.ndarray
    ) -> np.intp:
        """
        Find the nearest neighbors in the reference timeseries for each timeseries in the query timeseries
        """
        # TODO parameterize the metric
        # find the nearest neighbor in the reference timeseries for each timeseries in the query timeseries
        query_timeseries = query_timeseries.reshape(
            query_timeseries.shape[0], -1
        )
        reference_timeseries = reference_timeseries.reshape(
            reference_timeseries.shape[0], -1
        )
        nearest_neighbor = np.argmin(
            np.linalg.norm(reference_timeseries - query_timeseries, axis=1)
        )
        return nearest_neighbor

    def _align_timeseries(self, type_i: str, scaled: bool) -> np.ndarray:
        """
        Align timeseries in the scaled or unscaled space based on the scaled param.
        """
        # Select appropriate data source based on scaling
        data_dict = (
            self.windows_data_dict
            if scaled
            else self.unscaled_windows_data_dict
        )

        # Get query and reference timeseries
        query_timeseries = data_dict["synthesized"][type_i]
        reference_timeseries = np.concatenate(
            list(data_dict["real"].values()), axis=0
        )

        aligned_timeseries_indices = np.array(
            [
                self._find_nearest_neighbors(
                    # make it 3d so pairwise_distances can be used
                    np.expand_dims(query_timeseries[i], axis=0),
                    reference_timeseries,
                )
                for i in range(query_timeseries.shape[0])
            ]
        )
        return reference_timeseries[aligned_timeseries_indices]

    def _get_y_true_and_y_pred(
        self,
        type_i: str,
        windows_data_dict: dict[str, dict[str, np.ndarray]],
        clip_history: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get y_true and y_pred for the given type_i, accounting for probabilistic forecast
        """
        y_true = windows_data_dict["real"][type_i]
        if self.use_probabilistic_forecast:
            y_pred = windows_data_dict["synthesized"][type_i][:, 0, :, :]
        else:
            y_pred = windows_data_dict["synthesized"][type_i]

        if clip_history:
            logger.info(f"Clipping history for {type_i} data")
            # y_true.shape = (num_windows, num_channels, window_size)
            y_true = y_true[
                :, :, -self.configuration.dataset_config.forecast_length :
            ]
            y_pred = y_pred[
                :, :, -self.configuration.dataset_config.forecast_length :
            ]
        return y_true, y_pred

    def _get_metrics(self, type_i: str) -> None:
        """
        Orchestrate evaluation metrics calculation per window and for all
        windows togeter
        1. We obtain the metrics for the scaled data
        2. We obtain the metrics for the unscaled data
        3. We obtain the best and worst windows indices for unscaled data for plotting
        """
        # Ensure no division by zero
        y_true_scaled, y_pred_scaled = self._get_y_true_and_y_pred(
            type_i,
            self.windows_data_dict,
            clip_history=(
                True if "forecast" in self.configuration.task else False
            ),
        )

        self._get_all_data_metrics(
            y_true_scaled, y_pred_scaled, type_i, scaled=True
        )
        if self.use_probabilistic_forecast:
            prob_y_pred_scaled = self.windows_data_dict["synthesized"][type_i]
            prob_y_pred_scaled = prob_y_pred_scaled[
                :, :, :, -self.configuration.dataset_config.forecast_length :
            ]
            self._get_probabilistic_metrics(
                y_true_scaled, prob_y_pred_scaled, type_i, scaled=True
            )

        y_true_unscaled, y_pred_unscaled = self._get_y_true_and_y_pred(
            type_i,
            self.unscaled_windows_data_dict,
            clip_history=(
                True if "forecast" in self.configuration.task else False
            ),
        )
        self._get_all_data_metrics(
            y_true_unscaled, y_pred_unscaled, type_i, scaled=False
        )
        if self.use_probabilistic_forecast:
            prob_y_pred_unscaled = self.unscaled_windows_data_dict[
                "synthesized"
            ][type_i]
            prob_y_pred_unscaled = prob_y_pred_unscaled[
                :, :, :, -self.configuration.dataset_config.forecast_length :
            ]
            self._get_probabilistic_metrics(
                y_true_unscaled, prob_y_pred_unscaled, type_i, scaled=False
            )

        self._get_best_worst_windows_inds(
            y_true_unscaled, y_pred_unscaled, type_i
        )

    def _plot_all_data_fourier(self, type_i: str, windows_data_dict) -> None:
        """
        Plot fourier for all windows together
        """
        y_true, y_pred = self._get_y_true_and_y_pred(type_i, windows_data_dict)
        for i in range(self.configuration.dataset_config.num_channels):
            fft_true = np.abs(np.fft.fft(y_true[:, i, :])).flatten()
            fft_synthetic = np.abs(np.fft.fft(y_pred[:, i, :])).flatten()
            downsampled_true_values, downsampled_pred_values = (
                dynamic_downsample_true_and_predicted_values(
                    fft_true,
                    fft_synthetic,
                    self.configuration.dataset_config.num_channels,
                    downsample_factor=self.downsample_factor,
                )
            )
            self.generate_seaborn_histplot(
                self.timeseries_col_names[i],
                f"Ground Truth vs {self.label} for {self.timeseries_col_names[i]} : {type_i} data",
                f"fourier_all_data_{self.timeseries_col_names[i]}_{type_i}",
                downsampled_true_values,
                downsampled_pred_values,
                channel_name=self.timeseries_col_names[i],
                fourier=True,
            )

    def _plot_all_data_hist(self, type_i: str, windows_data_dict) -> None:
        """
        Plot timeseries histograms of synthesized and real data with all windows
        together
        """
        y_true, y_pred = self._get_y_true_and_y_pred(type_i, windows_data_dict)
        y_true = y_true.transpose(1, 0, 2).reshape(y_true.shape[1], -1)
        y_pred = y_pred.transpose(1, 0, 2).reshape(y_pred.shape[1], -1)
        for i in range(self.configuration.dataset_config.num_channels):
            downsampled_true_values, downsampled_pred_values = (
                dynamic_downsample_true_and_predicted_values(
                    y_true[i, :],
                    y_pred[i, :],
                    self.configuration.dataset_config.num_channels,
                    downsample_factor=self.downsample_factor,
                )
            )
            self.generate_seaborn_histplot(
                self.timeseries_col_names[i],
                f"Ground Truth vs {self.label} Density Distribution for {self.timeseries_col_names[i]} : {type_i} data",
                f"all_data_{self.timeseries_col_names[i]}_{type_i}",
                downsampled_true_values,
                downsampled_pred_values,
                channel_name=self.timeseries_col_names[i],
            )

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, mean_axis: int
    ) -> list:
        """
        Calculate evaluation metrics
        Parameters:
            - y_true: true values array (num_channels, num_windows * window_size)
            - y_pred: synthesized values array (num_channels, num_windows * window_size)
            - mean_axis: axis along which we take the mean
        Return:
            - []: list with evaluation metrics
        """
        epsilon = 1e-10
        y_true_safe = y_true + epsilon

        abs_errors = np.abs(y_pred - y_true)
        squared_errors = abs_errors**2

        metrics = [
            np.mean(abs_errors, axis=mean_axis),  # MAE
            np.mean(squared_errors, axis=mean_axis),  # MSE
            np.sqrt(np.mean(squared_errors, axis=mean_axis)),  # RMSE
            np.mean(np.abs(abs_errors / y_true_safe), axis=mean_axis)
            * 100,  # MAPE
            np.median(np.abs(abs_errors / y_true_safe), axis=mean_axis)
            * 100,  # MDAPE
            200
            * np.mean(
                abs_errors / (np.abs(y_true_safe) + np.abs(y_pred) + epsilon),
                axis=mean_axis,
            ),  # SMAPE
        ]

        return metrics

    def _plot_metrics_hist(self, type_i) -> None:
        """
        Plot hist for all windows
        Parameters:
            - type_i: either 'train', 'val', 'test'
        """
        for var_i in self.metrics_dict[type_i].keys():
            df = self.metrics_dict[type_i][str(var_i)]
            for metric_i in df:
                self.generate_seaborn_histplot(
                    metric_i,
                    f"{type_i} {metric_i}: `{str(var_i)}`",
                    f"metric_plot_{str(var_i)}_{metric_i}_{type_i}",
                    df[metric_i],
                    channel_name=str(var_i),
                )

    def _plot_fourier_coeffs(
        self,
        title: str,
        filename_suffix: str,
        true_values: np.ndarray,
        predicted_values: np.ndarray,
    ) -> None:
        """
        Plot fourier coeffs
        Parameters:
            - title: title of the plot
            - filename_suffix: suffix to add to the filename
            - true_values: true values array
            - predicted_values: synthesized values array
        """
        fft_true = np.fft.fft(true_values)
        fft_synthetic = np.fft.fft(predicted_values)

        self.generate_seaborn_histplot(
            "Ground Truth Part FFT",
            title,
            f"fourier_real_{filename_suffix}",
            fft_true.real,
            fft_synthetic.real,
        )

        self.generate_seaborn_histplot(
            "Imag Part FFT",
            title,
            f"fourier_imag_{filename_suffix}",
            fft_true.imag,
            fft_synthetic.imag,
        )

        self.generate_seaborn_histplot(
            "Magnitude FFT",
            title,
            f"fourier_magnitude_{filename_suffix}",
            np.abs(fft_true),
            np.abs(fft_synthetic),
        )

        # Compute the frequency axis
        freq = np.fft.fftfreq(n=true_values.size, d=1 / 10000)
        # Get only positive values
        n = true_values.size
        n_positive = n // 2
        freq = freq[:n_positive]
        fft_true = np.abs(fft_true)[:n_positive]
        fft_synthetic = np.abs(fft_synthetic)[:n_positive]

        plt.figure(figsize=(16, 10))
        plt.plot(freq, fft_true, label="Ground Truth", alpha=0.7)
        plt.plot(freq, fft_synthetic, label=self.label, alpha=0.7)
        plt.title(title, fontsize=14)
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel("Magnitude")
        plt.legend(fontsize=14)
        plt.grid(True)
        # plt.xlim([0, 50])
        plt.savefig(
            os.path.join(
                self.figsave_path,
                f"fourier_magnitude_freq_{filename_suffix}.png",
            ),
            format="png",
            dpi=300,
        )
        plt.close()

    def plot_timeseries(
        self,
        var_axis_label: str,
        title: str,
        filename_suffix: str,
        true_values: np.ndarray,
        predicted_values: np.ndarray,
    ) -> None:
        """
        Line plots of timeseries true vs synthesized
        Parameters:
            - var_axis_label: label of y axis for the variable
            - title: title of the plot
            - filename_suffix: suffix to add to the filename
            - true_values: true values array
            - predicted_values: synthesized values array
        """
        # Create figure with specified size
        fig = plt.figure(figsize=(16, 10))

        # Create axis
        ax = fig.add_subplot(111)

        # Plot data
        ax.plot(
            true_values,
            label="Ground Truth",
            color="blue",
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            predicted_values,
            label=self.label,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

        # Set title and labels
        ax.set_title(title, fontsize=20, pad=20)
        ax.set_xlabel("", fontsize=14)
        ax.set_ylabel(var_axis_label, fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(True)

        # Adjust layout to ensure proper sizing
        plt.tight_layout()

        # Get channel name from var_axis_label to determine save directory
        channel_name = var_axis_label
        save_dir = (
            os.path.join(self.figsave_path, channel_name)
            if channel_name in self.timeseries_col_names
            else self.figsave_path
        )

        # Save figure with explicit size preservation
        fig.savefig(  # type: ignore
            os.path.join(
                save_dir,
                f"line_{filename_suffix}.png",
            ),
            format="png",
            dpi=300,
        )
        plt.close(fig)

    def _plot_comparison_line(
        self, type_i: str, windows_data_dict: dict[str, dict[str, np.ndarray]]
    ) -> None:
        """Plot comparison line plots between ground truth and predicted values.

        Creates subplots showing ground truth vs predicted values for randomly selected
        windows of data. Each channel gets its own figure with multiple subplots.

        Args:
            type_i: Data split type ('train', 'val', or 'test')
            windows_data_dict: Dictionary containing ground truth and predicted data windows

        Returns:
            None. Saves plots to self.figsave_path/{channel_name}.
        """
        # Get and reshape data
        y_true, y_pred = self._get_y_true_and_y_pred(type_i, windows_data_dict)
        y_true = y_true.transpose(1, 0, 2).reshape(y_true.shape[1], -1)
        y_pred = y_pred.transpose(1, 0, 2).reshape(y_pred.shape[1], -1)

        # Calculate number of windows to plot
        ts_length = self.configuration.dataset_config.time_series_length
        total_windows = y_true.shape[1] // ts_length
        plot_windows = min(total_windows, PLOT_WINDOWS_LIMIT)

        # Randomly select and sort window indices
        window_indices = np.sort(
            np.random.choice(total_windows, plot_windows, replace=False)
        )

        # Plot each channel separately
        for channel_idx in range(
            self.configuration.dataset_config.num_channels
        ):
            n_cols = int(math.ceil(math.sqrt(plot_windows)))
            n_rows = int(math.ceil(plot_windows / n_cols))

            fig: Any
            axs: Any
            fig, axs = plt.subplots(
                n_rows,
                n_cols,
                figsize=(16, 10),
                sharex=False,
                sharey=False,
            )

            # Handle single subplot case
            axs = [axs] if isinstance(axs, plt.Axes) else axs.flatten()
            channel_name = self.timeseries_col_names[channel_idx]
            legend_handles, legend_labels = None, None

            # Create each subplot
            for idx, window_idx in enumerate(window_indices):
                window_start = window_idx * ts_length
                window_end = (window_idx + 1) * ts_length

                real_values = y_true[channel_idx, window_start:window_end]
                forecasted_values = y_pred[channel_idx, window_start:window_end]

                # Plot ground truth
                line_real = axs[idx].plot(
                    real_values,
                    label="Ground Truth",
                    linewidth=2,
                    color="blue",
                    alpha=0.8,
                )[0]

                # Plot predicted values
                line_forecast = axs[idx].plot(
                    forecasted_values,
                    label="Predicted",
                    linestyle="--",
                    linewidth=2,
                    color="red",
                    alpha=0.8,
                )[0]

                # Add error region
                error_region = axs[idx].fill_between(
                    range(ts_length),
                    real_values,
                    forecasted_values,
                    color="lavender",
                    alpha=0.7,
                    label="Error region",
                )

                # Add forecast line if applicable
                if "forecast" in self.configuration.task:
                    forecast_start = (
                        self.configuration.dataset_config.time_series_length
                        - self.configuration.dataset_config.forecast_length
                        - 1
                    )
                    axs[idx].axvline(
                        x=forecast_start,
                        color="gray",
                        linestyle="--",
                        alpha=0.8,
                    )

                axs[idx].tick_params(axis="y", labelsize=14)
                axs[idx].tick_params(axis="x", labelsize=14)

                # Store legend handles from first subplot only
                if idx == 0:
                    legend_handles = [line_real, line_forecast, error_region]
                    legend_labels = [
                        "Ground Truth",
                        "Predicted",
                        "Error region",
                    ]

            # Hide unused subplots
            for ax in axs[plot_windows:]:
                ax.set_visible(False)

            # Add title and labels
            plt.suptitle(
                f"Comparison of Ground Truth vs. {self.label} `{channel_name}` Values "
                f"Across {plot_windows} Random {type_i.capitalize()} Windows",
                fontsize=20,
                y=0.97,
            )
            fig.supxlabel("Step", fontsize=14, y=0.02)
            fig.supylabel(channel_name, fontsize=14, x=0.0)

            # Add figure legend if we have handles
            if legend_handles and legend_labels:
                fig.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc="upper right",
                    fontsize=14,
                )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=5, w_pad=5)

            # Save figure in channel-specific subdirectory
            save_dir = os.path.join(self.figsave_path, channel_name)
            fig.savefig(
                os.path.join(
                    save_dir,
                    f"comparison_line_{type_i}_{channel_name}.png",
                )
            )
            plt.close(fig)

    def _plot_dist_violin(self, type_i: str, windows_data_dict) -> None:
        """
        Plot paired violin plots of synthesized and real data for different channels,
        with normalization to ensure comparable scales.

        Parameters:
            - type_i: either 'train', 'val', 'test'
            - windows_data_dict: dictionary containing real and synthesized data
        """
        y_true, y_pred = self._get_y_true_and_y_pred(type_i, windows_data_dict)

        # Reshape the data
        y_true = y_true.transpose(1, 0, 2).reshape(y_true.shape[1], -1)
        y_pred = y_pred.transpose(1, 0, 2).reshape(y_pred.shape[1], -1)

        # Prepare the data for the violin plot with normalization
        data = []
        for i in range(
            min(
                self.configuration.dataset_config.num_channels,
                PLOT_CHANNELS_LIMIT,
            )
        ):
            channel_name = self.timeseries_col_names[i]

            downsampled_true_values, downsampled_pred_values = (
                dynamic_downsample_true_and_predicted_values(
                    y_true[i, :],
                    y_pred[i, :],
                    self.configuration.dataset_config.num_channels,
                    downsample_factor=self.downsample_factor,
                )
            )

            # Combine real and forecasted values for normalization
            combined_values = np.concatenate(
                [downsampled_true_values, downsampled_pred_values]
            )
            min_val = combined_values.min()
            max_val = combined_values.max()

            # Normalize real and forecasted values based on combined min and max
            if min_val == max_val:
                normalized_real = np.zeros_like(downsampled_true_values)
                normalized_forecasted = np.zeros_like(downsampled_pred_values)
            else:
                normalized_real = (downsampled_true_values - min_val) / (
                    max_val - min_val
                )
                normalized_forecasted = (downsampled_pred_values - min_val) / (
                    max_val - min_val
                )

            # Add normalized data to the plot data
            data.extend(
                [
                    (channel_name, "Ground Truth", value)
                    for value in normalized_real
                ]
            )
            data.extend(
                [
                    (channel_name, self.label, value)
                    for value in normalized_forecasted
                ]
            )

        # Convert the data to a DataFrame
        df = pd.DataFrame(data, columns=["Channel", "Type", "Value"])

        # Plot the paired violin plot
        plt.figure(figsize=(16, 10))

        # Define color palette with blue for Ground Truth and red for predicted
        palette = {"Ground Truth": "blue", self.label: "red"}
        sns.violinplot(
            x="Channel",
            y="Value",
            hue="Type",
            data=df,
            split=False,
            inner="box",
            palette=palette,
            alpha=0.8,
        )

        # Customize the plot
        plt.title(f"Ground Truth vs {self.label} : {type_i} Data", fontsize=20)
        plt.xlabel("Channel", fontsize=20)
        plt.ylabel("Normalized Value", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, title_fontsize=20)

        # Save the plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.figsave_path, f"violin_dist_{type_i}.png")
        )
        plt.close()

    def generate_seaborn_histplot(
        self,
        var_axis_label: str,
        title: str,
        filename_suffix: str,
        plot_data1: np.ndarray,
        plot_data2: Union[np.ndarray, None] = None,
        channel_name: Optional[str] = None,
        fourier: bool = False,
    ):
        """
        Plot histograms
        Parameters:
            - var_axis_label: label of y axis for the variable
            - title: title of the plot
            - filename_suffix: suffix to add to the filename
            - true_values: true values array
            - predicted_values: synthesized values array
        """

        ax: Axes | Any
        _, ax = plt.subplots(
            nrows=1,
            ncols=1,
            sharey=True,
            sharex=True,
            figsize=(16, 10),
        )

        df = pd.DataFrame()
        df["Real"] = plot_data1
        df["Synthetic"] = plot_data2

        if len(np.unique(plot_data1)) == 1:
            kde = False
            stat = "count"
        else:
            kde = True
            stat = "density"

        if fourier:
            bins = 10000
        else:
            bins = 100

        sns.histplot(
            data=df,
            x="Real",
            ax=ax,
            bins=bins,
            kde=kde,
            stat=stat,
            common_norm=False,
            color="Blue",
            alpha=0.5,
            line_kws={"linewidth": 5},
            label="Ground Truth",
        )

        if plot_data2 is not None:
            sns.histplot(
                data=df,
                x="Synthetic",
                ax=ax,
                bins=bins,
                kde=kde,
                stat=stat,
                common_norm=False,
                color="Red",
                alpha=0.5,
                line_kws={"linewidth": 5},
                label=self.label,
            )
            # Move the legend outside the plot.
            ax.legend(fontsize=14)

        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_xlabel(var_axis_label, fontsize=14)

        if fourier:
            # choose xlim based on the quartiles
            q_low = float(np.quantile(plot_data1, 0.01))
            q_high = float(np.quantile(plot_data1, 0.9))
            ax.set_xlim(q_low, q_high)

        plt.title(title, fontsize=20)

        # Draw grid
        ax.grid(True)

        plt.tight_layout()

        save_dir = (
            os.path.join(self.figsave_path, channel_name)
            if channel_name is not None
            else self.figsave_path
        )

        plt.savefig(
            os.path.join(
                save_dir,
                f"hist_{filename_suffix}.png",
            ),
        )
        plt.close()

    def _plot_best_worst(self, type_i, windows_data_dict) -> None:
        """
        Plots best and worst windows line, hist and fourier plot for each
        evaluation metric.
        Parameters:
            - type_i: either 'train', 'val', 'test'
        """
        for best_worst in ["Best", "Worst"]:
            for i in range(self.configuration.dataset_config.num_channels):
                for metric_i in self.config.best_worst_metrics:
                    window_ind = self.best_worst_indices[type_i][
                        self.timeseries_col_names[i]
                    ][metric_i][best_worst][0]
                    filename_suffix = f"{self.timeseries_col_names[i]}_{best_worst}_{metric_i}_{type_i}"

                    title = (
                        f"Evaluation Metric {metric_i} – Selected {best_worst} window (index {window_ind}) "
                        f"from {type_i} data for channel `{self.timeseries_col_names[i]}`"
                    )

                    var_axis_label = self.timeseries_col_names[i]

                    # Plot data
                    true_data = windows_data_dict["real"][type_i][
                        window_ind, i, :
                    ]
                    if (
                        hasattr(self.configuration, "denoiser_config")
                        and self.configuration.denoiser_config is not None
                        and self.configuration.denoiser_config.use_probabilistic_forecast
                    ):
                        synthesized_data = windows_data_dict["synthesized"][
                            type_i
                        ][window_ind, 0, i, :]
                    else:
                        synthesized_data = windows_data_dict["synthesized"][
                            type_i
                        ][window_ind, i, :]

                    # Line plot of timeseries
                    self.plot_timeseries(
                        var_axis_label,
                        title,
                        filename_suffix,
                        true_data,
                        synthesized_data,
                    )

    def plot_probabilistic_predictions(
        self,
        var_axis_label,
        title,
        filename_suffix,
        true_data,
        synthesized_data,
    ) -> None:
        """
        Plot probabilistic predictions
        """
        # create a plot with ground truth and probabilistic predictions such that we show a shaded area representing 1 standard deviation
        plt.figure(figsize=(16, 10))
        plt.plot(
            true_data,
            label="Ground Truth",
            color="blue",
            linewidth=2,
            alpha=0.8,
        )
        mean = np.mean(synthesized_data, axis=0)
        std = np.std(synthesized_data, axis=0)
        plt.fill_between(
            range(len(mean)), mean - std, mean + std, color="red", alpha=0.8
        )
        plt.title(title, fontsize=20)
        plt.xlabel("")
        plt.ylabel(var_axis_label)
        plt.legend(fontsize=14)
        plt.grid(True)

        # Determine save directory based on var_axis_label (channel name)
        channel_name = var_axis_label
        save_dir = (
            os.path.join(self.figsave_path, channel_name)
            if channel_name in self.timeseries_col_names
            else self.figsave_path
        )

        plt.savefig(
            os.path.join(
                save_dir,
                f"line_{filename_suffix}.png",
            ),
            format="png",
            dpi=300,
        )
        plt.close()

    def _plot_best_worst_probabilistic(self, type_i, windows_data_dict) -> None:
        """
        Plots best and worst windows line, hist and fourier plot for each
        evaluation metric.
        Parameters:
            - type_i: either 'train', 'val', 'test'
        """
        plt.figure(figsize=(16, 10))
        for best_worst in ["Best", "Worst"]:
            for i in range(self.configuration.dataset_config.num_channels):
                for metric_i in self.config.best_worst_metrics:
                    window_ind = self.best_worst_indices[type_i][
                        self.timeseries_col_names[i]
                    ][metric_i][best_worst][0]
                    filename_suffix = f"{self.timeseries_col_names[i]}_{best_worst}_{metric_i}_{type_i}_probabilistic"

                    title = (
                        f"Evaluation Metric {metric_i} – Selected {best_worst} window (index {window_ind}) "
                        f"from {type_i} data for channel `{self.timeseries_col_names[i]}` (Probabilistic)"
                    )

                    var_axis_label = self.timeseries_col_names[i]

                    # Plot data
                    true_data = windows_data_dict["real"][type_i][
                        window_ind, i, :
                    ]
                    synthesized_data = windows_data_dict["synthesized"][type_i][
                        window_ind, :, i, :
                    ]
                    # Line plot of timeseries
                    self.plot_probabilistic_predictions(
                        var_axis_label,
                        title,
                        filename_suffix,
                        true_data,
                        synthesized_data,
                    )

    def _metrics_table(self, dataset_type):
        """Create a metrics table for the given dataset type with formatted values."""

        rows = []
        metrics = ["MAE", "MSE", "RMSE", "MAPE", "MDAPE", "SMAPE"]

        for channel_name in self.timeseries_col_names:
            for scaled_text, scale_key in [("✓", "scaled"), ("✗", "unscaled")]:
                row = {"Channel": channel_name, "Scaled": scaled_text}
                for metric in metrics:
                    value = self.all_data_metrics_dict[dataset_type][
                        f"{channel_name}_{scale_key}"
                    ][metric]
                    # Format numbers in scientific notation if they're too big or small
                    if abs(value) < 0.01 or abs(value) > 999:
                        row[metric] = f"{value:.1e}"
                    else:
                        row[metric] = f"{value:.1f}"
                rows.append(row)

        df = pd.DataFrame(rows)

        table_styles: Any = [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("margin", "0"),
                    ("padding", "0"),
                    (
                        "font-family",
                        "'Roboto', 'Helvetica Neue', 'Arial', sans-serif",
                    ),
                    ("font-size", "14px"),
                    ("color", "#333"),
                    ("border", "1px solid #ccc"),
                    ("border-radius", "6px"),
                    ("box-shadow", "0 2px 8px rgba(0, 0, 0, 0.1)"),
                ],
            },
            {
                "selector": "thead",
                "props": [
                    ("background-color", "#394B59"),
                    ("display", "table-header-group"),
                ],
            },
            {
                "selector": "tfoot",
                "props": [("display", "table-footer-group")],
            },
            {
                "selector": "tr",
                "props": [("page-break-inside", "avoid")],
            },
            {
                "selector": "thead th",
                "props": [
                    ("color", "#ffffff"),
                    ("font-weight", "bold"),
                    ("border", "1px solid #dddddd"),
                    ("text-align", "center"),
                    ("padding", "12px"),
                    ("font-size", "15px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("border", "1px solid #dddddd"),
                    ("padding", "10px"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#f9f9f9")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#f1f1f1")],
            },
        ]

        html_table = (
            df.style.set_table_styles(table_styles)
            .set_table_attributes('style="width: 100%;"')
            .hide(axis="index")
            .to_html()
        )

        # Save to HTML file
        with open(
            os.path.join(
                self.figsave_path, f"metrics_table_{dataset_type}.html"
            ),
            "w",
        ) as file:
            file.write(html_table)

    def _plot_covariance_matrix(
        self, type_i: str, windows_data_dict: dict[str, dict[str, np.ndarray]]
    ) -> None:
        """
        Plot the covariance matrix for the given dataset type for both real and synthetic data.

        This function computes and visualizes the covariance matrices between time series data
        and continuous metadata for both real and synthetic datasets. It also calculates and
        plots the error percentage between the real and synthetic covariance matrices.

        Parameters:
            type_i (str): The dataset type ('train', 'val', or 'test').
            windows_data_dict (dict): Dictionary containing real and synthetic data windows.

        Returns:
            None: The function saves the generated plots to the specified output directory.
        """
        logger.info(f"Plotting covariance matrix for {type_i} data...")

        clip_history = True if "forecast" in self.configuration.task else False

        # Get real and synthetic data
        y_true, y_pred = self._get_y_true_and_y_pred(
            type_i,
            windows_data_dict,
            clip_history=clip_history,
        )

        # Rearrange the data
        y_true = einops.rearrange(y_true, "b c t -> b t c")
        y_pred = einops.rearrange(y_pred, "b c t -> b t c")

        cont_metadata = safe_load_npy_file(
            f"{type_i}_continuous_conditions.npy",
            os.path.join(
                str(SYNTHEFY_DATASETS_BASE),
                self.configuration.dataset_config.dataset_name,
            ),
        )

        if cont_metadata is None or len(cont_metadata) == 0:
            logger.warning(
                f"No continuous metadata found for {type_i} dataset. Skipping covariance matrix plot."
            )
            return

        if clip_history:
            cont_metadata = cont_metadata[
                :, -self.configuration.dataset_config.forecast_length :, :
            ]

        # Get the real covariance matrix
        cov_matrix_real = compute_covariance_matrix(y_true, cont_metadata)
        fig = plot_covariance_matrix_static(
            cov_matrix_real,
            self.timeseries_col_names,
            self.continuous_col_names,
            title="Covariance Matrix: Real Time Series vs. Continuous Metadata",
            x_axis_label="Continuous Columns",
            y_axis_label="Real Time Series Columns",
            vmin=-1,
            vmax=1,
            show_percentage=False,
        )
        fig.savefig(
            os.path.join(
                self.figsave_path, f"real_covariance_matrix_{type_i}.png"
            ),
        )

        # Get the synthetic covariance matrix plot
        cov_matrix_synth = compute_covariance_matrix(y_pred, cont_metadata)
        fig = plot_covariance_matrix_static(
            cov_matrix_synth,
            self.timeseries_col_names,
            self.continuous_col_names,
            title=f"Covariance Matrix: {self.label} Time Series vs. Continuous Metadata",
            x_axis_label="Continuous Columns",
            y_axis_label="Synthetic Time Series Columns",
            vmin=-1,
            vmax=1,
            show_percentage=False,
        )
        fig.savefig(
            os.path.join(
                self.figsave_path, f"synthetic_covariance_matrix_{type_i}.png"
            ),
        )

        # Calculate error percentage matrix
        error_matrix = np.abs(cov_matrix_real - cov_matrix_synth) / 2 * 100
        fig = plot_covariance_matrix_static(
            error_matrix,
            self.timeseries_col_names,
            self.continuous_col_names,
            title=f"Covariance Error Percentage: Real vs {self.label}",
            x_axis_label="Continuous Columns",
            y_axis_label="Time Series Columns",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=100,
            show_percentage=True,
        )
        fig.savefig(
            os.path.join(
                self.figsave_path, f"difference_covariance_matrix_{type_i}.png"
            ),
        )

        logger.info(f"Covariance matrices for {type_i} saved as PNG")

    def _analyze_real_vs_synthetic_distributions(
        self, type_i: str, windows_data_dict: dict[str, dict[str, np.ndarray]]
    ) -> None:
        # """Analyze distributions between real and synthetic data."""
        logger.info(f"Analyzing distributions for {type_i} data...")

        # Get real and synthetic data
        y_true, y_pred = self._get_y_true_and_y_pred(
            type_i,
            windows_data_dict,
            clip_history=True
            if "forecast" in self.configuration.task
            else False,
        )

        results_records = []
        layout_elements = []

        continuous_elements = process_continuous_columns(
            data_list=[y_true, y_pred],
            col_names=self.timeseries_col_names,
            col_type="timeseries",
            data_labels=["Real", self.label],
            results_records=results_records,
            jsd_threshold=0.3,
            emd_threshold=0.3,
        )
        layout_elements.extend(continuous_elements)
        metric_table = create_metrics_table(
            results_records=results_records,
            jsd_threshold=0.3,
            emd_threshold=0.3,
            out_csv_path=os.path.join(
                self.figsave_path, f"distribution_analysis_{type_i}.csv"
            ),
        )
        layout_elements.extend(metric_table)
        output_file(
            os.path.join(
                self.figsave_path, f"distribution_analysis_{type_i}.html"
            )
        )
        save(layout_elements)

    def analyze_correlations(
        self, type_i: str, windows_data_dict: dict[str, dict[str, np.ndarray]]
    ) -> None:
        """Analyze correlations between time series and continuous columns."""
        logger.info(f"Analyzing correlations for {type_i} data...")

        # Get real and synthetic data
        y_true, y_pred = self._get_y_true_and_y_pred(
            type_i,
            windows_data_dict,
            clip_history=True
            if "forecast" in self.configuration.task
            else False,
        )

        layout_elements = []
        cov_matrix = compute_covariance_matrix(y_true, y_pred)
        cov_elements = plot_covariance_matrix(
            cov_matrix,
            self.timeseries_col_names,
            self.timeseries_col_names,
            title=f"Covariance Matrix: Real Time Series vs. {self.label} Time Series",
            x_axis_label=self.label,
            y_axis_label="Real",
            same_columns_only=True,  # Only show corresponding columns for real vs predicted
        )
        layout_elements.extend(cov_elements)

        cross_corr_elements = create_cross_corr_elements(
            y_true,
            y_pred,
            self.timeseries_col_names,
            self.timeseries_col_names,
            csv_path=os.path.join(
                self.figsave_path, f"cross_correlation_summary_{type_i}.csv"
            ),
            pairwise_corr_figures=True,
            downsample_factor=100,
            same_columns_only=True,  # Only compare corresponding columns for real vs predicted
            col1_name="Real",
            col2_name=self.label,
        )
        layout_elements.extend(cross_corr_elements)

        output_file(
            os.path.join(
                self.figsave_path, f"correlation_analysis_{type_i}.html"
            )
        )
        save(
            layout(
                children=layout_elements,
                sizing_mode="stretch_width",
                margin=20,
            )
        )

    def plot_consolidated_comparison_line(
        self, type_i: str, windows_data_dict: dict[str, dict[str, np.ndarray]]
    ) -> None:
        """Create a consolidated comparison plot with samples from multiple channels.

        Creates a single figure with subplots showing ground truth vs predicted values
        for randomly selected channels. Unlike _plot_comparison_line which creates
        one figure per channel, this creates a single figure combining multiple channels
        and saves it to the main output folder.

        Args:
            type_i: Data split type ('train', 'val', or 'test')
            windows_data_dict: Dictionary containing ground truth and predicted data windows

        Returns:
            None. Saves a single consolidated plot to the main figsave_path.
        """
        # Get and reshape data
        y_true, y_pred = self._get_y_true_and_y_pred(type_i, windows_data_dict)
        y_true = y_true.transpose(1, 0, 2).reshape(y_true.shape[1], -1)
        y_pred = y_pred.transpose(1, 0, 2).reshape(y_pred.shape[1], -1)

        # Calculate number of windows to plot
        ts_length = self.configuration.dataset_config.time_series_length
        total_windows = y_true.shape[1] // ts_length

        # Define number of channels to sample (use at most 4 channels)
        num_channels_to_sample = min(
            4, self.configuration.dataset_config.num_channels
        )

        # Randomly select channels if there are more than 4
        if (
            self.configuration.dataset_config.num_channels
            <= num_channels_to_sample
        ):
            sampled_channel_indices = list(
                range(self.configuration.dataset_config.num_channels)
            )
        else:
            sampled_channel_indices = sorted(
                np.random.choice(
                    self.configuration.dataset_config.num_channels,
                    num_channels_to_sample,
                    replace=False,
                )
            )

        # Select a random window index to visualize (same for all channels)
        window_idx = np.random.randint(0, total_windows)
        window_start = window_idx * ts_length
        window_end = (window_idx + 1) * ts_length

        # Create a 2x2 grid of subplots (or appropriate size)
        n_cols = min(2, num_channels_to_sample)
        n_rows = (
            num_channels_to_sample + n_cols - 1
        ) // n_cols  # Ceiling division

        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(16, 10), sharex=True, sharey=False
        )

        # Handle the case of a single subplot
        if num_channels_to_sample == 1:
            axs = np.array([axs])

        # Flatten axs if it's a 2D array
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        # Store legend handles from first subplot only
        legend_handles, legend_labels = None, None

        # Create each subplot for selected channels
        for i, channel_idx in enumerate(sampled_channel_indices):
            channel_name = self.timeseries_col_names[channel_idx]

            real_values = y_true[channel_idx, window_start:window_end]
            forecasted_values = y_pred[channel_idx, window_start:window_end]

            # Plot ground truth
            line_real = axs[i].plot(
                real_values,
                label="Ground Truth",
                linewidth=2,
                color="blue",
                alpha=0.8,
            )[0]

            # Plot predicted values
            line_forecast = axs[i].plot(
                forecasted_values,
                label=self.label,
                linestyle="--",
                linewidth=2,
                color="red",
                alpha=0.8,
            )[0]

            # Add error region
            error_region = axs[i].fill_between(
                range(ts_length),
                real_values,
                forecasted_values,
                color="lavender",
                alpha=0.7,
                label="Error region",
            )

            # Add forecast line if applicable
            if "forecast" in self.configuration.task:
                forecast_start = (
                    self.configuration.dataset_config.time_series_length
                    - self.configuration.dataset_config.forecast_length
                    - 1
                )
                axs[i].axvline(
                    x=forecast_start,
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                )

            # Add title for this subplot
            axs[i].set_title(f"{channel_name}", fontsize=14)
            axs[i].tick_params(axis="y", labelsize=12)
            axs[i].tick_params(axis="x", labelsize=12)

            # Store legend handles from first subplot only
            if i == 0:
                legend_handles = [line_real, line_forecast, error_region]
                legend_labels = ["Ground Truth", self.label, "Error region"]

        # Hide unused subplots
        for i in range(num_channels_to_sample, len(axs)):
            axs[i].set_visible(False)

        # Add title and labels
        plt.suptitle(
            f"Comparison of Ground Truth vs. {self.label} Values "
            f"Across Multiple Channels (Window #{window_idx}, {type_i.capitalize()} Data)",
            fontsize=20,
            y=0.98,
        )
        fig.supxlabel("Step", fontsize=16)

        # Add figure legend if we have handles
        if legend_handles and legend_labels:
            fig.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="lower center",
                ncol=3,
                fontsize=14,
                bbox_to_anchor=(0.5, 0.01),
            )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])

        # Save the consolidated figure to the main directory
        fig.savefig(  # type: ignore
            os.path.join(
                self.figsave_path,
                f"comparison_line_{type_i}_consolidated.png",
            )
        )
        plt.close(fig)

    def postprocess(self, plot_fourier: bool = False) -> None:
        """
        Main method to create plots.
        """
        logger.info("Plotting started")
        self._plot_learning_curve()

        for dataset_type in self.config.dataset_types:
            logger.info(f"Processing {dataset_type} data ...")
            logger.info("Plotting metrics table")
            self._metrics_table(dataset_type)
            logger.info("Plotting comparison line")
            self._plot_comparison_line(
                dataset_type, self.unscaled_windows_data_dict
            )
            logger.info("Plotting consolidated comparison line")
            self.plot_consolidated_comparison_line(
                dataset_type, self.unscaled_windows_data_dict
            )
            logger.info("Plotting dist violin")
            self._plot_dist_violin(
                dataset_type, self.unscaled_windows_data_dict
            )
            logger.info("Plotting all data hist")
            self._plot_all_data_hist(
                dataset_type, self.unscaled_windows_data_dict
            )
            logger.info("Plotting metrics hist")
            self._plot_metrics_hist(dataset_type)
            logger.info("Plotting best worst")
            self._plot_best_worst(dataset_type, self.unscaled_windows_data_dict)

            if not self.is_foundation_model:
                self._plot_covariance_matrix(
                    dataset_type, self.unscaled_windows_data_dict
                )

            if plot_fourier:
                logger.info("Plotting fourier plots")
                self._plot_all_data_fourier(
                    dataset_type, self.unscaled_windows_data_dict
                )
            else:
                logger.info("Skipping fourier plotting")
            if (
                hasattr(self.configuration, "denoiser_config")
                and self.configuration.denoiser_config is not None
                and self.configuration.denoiser_config.use_probabilistic_forecast
            ):
                logger.info("Plotting best worst probabilistic")
                self._plot_best_worst_probabilistic(
                    dataset_type, self.unscaled_windows_data_dict
                )
            logger.info("Done!")

    def _ensure_metric_dict_exists(self, target_dict, dataset_type, key):
        """Helper method to ensure a metric dictionary exists at the specified key."""
        if key not in target_dict[dataset_type]:
            target_dict[dataset_type][key] = {}
        return target_dict[dataset_type][key]

    def _save_results(self) -> None:
        """Save processing results to files."""
        metrics_serializable = {
            dataset_type: {
                col: {
                    metric: float(value)
                    for metric, value in metric_values.items()
                }
                for col, metric_values in col_metrics.items()
            }
            for dataset_type, col_metrics in self.all_data_metrics_dict.items()
        }

        # Add aggregated statistics across all channels
        for dataset_type in metrics_serializable:
            # Get all scaled channel metrics for this dataset type
            scaled_channels = [
                col_data
                for col, col_data in self.all_data_metrics_dict[
                    dataset_type
                ].items()
                if col.endswith("_scaled")
            ]

            if not scaled_channels:
                continue

            # Calculate aggregate stats for each metric
            for metric in self.config.metric_types:
                # Extract values for this metric across all channels
                metric_values = [
                    float(channel_data[metric])
                    for channel_data in scaled_channels
                    if metric in channel_data
                ]

                if not metric_values:
                    continue

                # Calculate statistics
                mean_value = float(np.mean(metric_values))
                max_value = float(np.max(metric_values))
                min_value = float(np.min(metric_values))
                std_value = float(np.std(metric_values))

                # Prepare dictionaries and store values for both metrics_serializable and self.all_data_metrics_dict
                agg_stats = {
                    "all_channels_scaled_mean": mean_value,
                    "all_channels_scaled_max": max_value,
                    "all_channels_scaled_min": min_value,
                    "all_channels_scaled_std": std_value,
                }

                for stat_key, stat_value in agg_stats.items():
                    # Ensure dictionaries exist and store values
                    metrics_dict = self._ensure_metric_dict_exists(
                        metrics_serializable, dataset_type, stat_key
                    )
                    metrics_dict[metric] = stat_value

                    # Also update the class instance dictionary for logging
                    metrics_dict = self._ensure_metric_dict_exists(
                        self.all_data_metrics_dict, dataset_type, stat_key
                    )
                    metrics_dict[metric] = stat_value

        with open(
            os.path.join(self.figsave_path, "best_worst_indices.json"), "w"
        ) as file:
            json.dump(self.best_worst_indices, file, indent=4)

        with open(
            os.path.join(self.figsave_path, "all_data_metrics.json"), "w"
        ) as file:
            json.dump(metrics_serializable, file, indent=4)

        logger.info(f"Results saved to: {self.figsave_path}")

    def _log_summary(self) -> None:
        """Log a comprehensive summary of all metrics for each channel and dataset type."""

        logger.info("\n" + "=" * 50)
        logger.info("METRICS SUMMARY")
        logger.info("=" * 50)

        channels = list(
            self.all_data_metrics_dict[self.config.dataset_types[0]].keys()
        )
        metrics = self.config.metric_types

        # First log individual channel metrics
        for channel in channels:
            logger.info("\n" + "-" * 30)
            logger.info(f"Channel: {channel}")
            logger.info("-" * 30)

            # Create a formatted table-like output
            # Header
            header = f"{'Metric':<10} | "
            for dataset_type in self.config.dataset_types:
                header += f"{dataset_type.upper():<15} | "
            logger.info(header)
            logger.info("-" * len(header))

            # Metrics rows
            for metric in metrics:
                row = f"{metric:<10} | "
                for dataset_type in self.config.dataset_types:
                    try:
                        value = self.all_data_metrics_dict[dataset_type][
                            channel
                        ][metric]
                        if isinstance(value, (int, float)):
                            row += f"{value:15.4f} | "
                        else:
                            row += f"{str(value):<15} | "
                    except KeyError:
                        row += f"{'N/A':<15} | "
                logger.info(row)

            # Add probabilistic metrics if they exist
            if self.use_probabilistic_forecast:
                logger.info("-" * len(header))
                for dataset_type in self.config.dataset_types:
                    try:
                        crps_mean = self.all_data_metrics_dict[dataset_type][
                            channel
                        ]["CRPS-MEAN"]
                        crps_std = self.all_data_metrics_dict[dataset_type][
                            channel
                        ]["CRPS-STD"]
                        logger.info(f"CRPS-MEAN  | {crps_mean:15.4f} | ")
                        logger.info(f"CRPS-STD   | {crps_std:15.4f} | ")
                    except KeyError:
                        pass

        # Log aggregate channel metrics
        logger.info("\n" + "=" * 50)
        logger.info("AGGREGATE CHANNEL METRICS")
        logger.info("=" * 50)

        # Define the aggregate metrics to display
        aggregate_types = ["Mean", "Max", "Min", "Std"]

        for aggregate_type in aggregate_types:
            logger.info("\n" + "-" * 30)
            logger.info(f"All Channels {aggregate_type}")
            logger.info("-" * 30)

            # Header
            header = f"{'Metric':<10} | "
            for dataset_type in self.config.dataset_types:
                header += f"{dataset_type.upper():<15} | "
            logger.info(header)
            logger.info("-" * len(header))

            # Metrics rows
            for metric in metrics:
                row = f"{metric:<10} | "
                for dataset_type in self.config.dataset_types:
                    try:
                        key = f"all_channels_scaled_{aggregate_type.lower()}"
                        if key in self.all_data_metrics_dict[dataset_type]:
                            value = self.all_data_metrics_dict[dataset_type][
                                key
                            ][metric]
                            if isinstance(value, (int, float)):
                                row += f"{value:15.4f} | "
                            else:
                                row += f"{str(value):<15} | "
                        else:
                            row += f"{'N/A':<15} | "
                    except (KeyError, TypeError):
                        row += f"{'N/A':<15} | "
                logger.info(row)

        logger.info("\n" + "=" * 50)

    def _plot_learning_curve(self) -> None:
        """
        Plots the learning curve for training, validation, and test losses.
        """
        try:
            plot_learning_curve(
                input_logs_dir=self.lightning_logs_path,
                output_fig_path=os.path.join(
                    self.figsave_path, "learning_curve.png"
                ),
                run_name=self.configuration.run_name,
                dataset_name=self.configuration.dataset_name,
            )
        except Exception as e:
            logger.warning(
                f"An error occurred during learning curve plotting: {str(e)}"
            )

    # Add a new method to create channel-specific subdirectories
    def _create_channel_subdirectories(self) -> None:
        """Create subdirectories for each channel in the plots directory."""
        for channel_name in self.timeseries_col_names:
            channel_dir = os.path.join(self.figsave_path, channel_name)
            os.makedirs(channel_dir, exist_ok=True)
        logger.info(
            f"Created channel-specific subdirectories in {self.figsave_path}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run postprocessing on synthetic data."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--plot_fourier",
        type=bool,
        help="Plot fourier plots",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--fmv2_filepath",
        type=str,
        help="Path to the h5 file containing FMv2 data",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    try:
        postprocessor = Postprocessor(
            config_filepath=args.config, fmv2_filepath=args.fmv2_filepath
        )
        postprocessor.postprocess(plot_fourier=args.plot_fourier)
    except Exception as e:
        logger.error(f"An error occurred during postprocessing: {str(e)}")
        raise
