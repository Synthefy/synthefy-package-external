import os
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger

from synthefy_pkg.configs.execution_configurations import (
    Configuration,
    DatasetConfig,
)
from synthefy_pkg.data.data_gen_base import SynthesisBaseDataModule
from synthefy_pkg.data.dataset_utils import get_dataset_paths
from synthefy_pkg.preprocessing.preprocess import DataPreprocessor
from synthefy_pkg.utils.basic_utils import ENDC, OKBLUE
from synthefy_pkg.utils.scaling_utils import (
    load_continuous_scalers,
    load_discrete_encoders,
    load_timeseries_scalers,
)
from synthefy_pkg.utils.synthesis_utils import add_missing_timeseries_columns

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

COMPILE = False


def fail_if_dataset_shape_mismatch(
    timeseries_dataset: Union[np.ndarray, None],
    discrete_conditions: np.ndarray,
    continuous_conditions: np.ndarray,
    time_series_length: int,
    num_channels: int,
    num_discrete_conditions: int,
    num_continuous_conditions: int,
    num_timestamp_conditions: Union[int, None] = None,
    timestamp_conditions: Union[np.ndarray, None] = None,
) -> None:
    """
    inputs:
        timeseries_dataset: Union[np.ndarray, None] - (num_windows, num_channels, num_timesteps). Can be None in the case of synthesis since we only require metadata
        discrete_conditions: np.ndarray (num_windows, num_timesteps, num_discrete_conditions) or 2d for time invariant metadata: (num_windows, num_discrete_conditions)
        continuous_conditions: np.ndarray (num_windows, num_timesteps, num_continuous_conditions)
        time_series_length: int
        num_channels: int
        num_discrete_conditions: int
        num_continuous_conditions: int
        num_timestamp_conditions: Union[int, None]. Can be None in the case of synthesis.
        timestamp_conditions: Union[np.ndarray, None] - (num_windows, num_timesteps, num_timestamp_conditions). Can be None in the case of synthesis.
    returns:
        None - Raises an error if the shape of the dataset is not correct
    """

    expected_timeseries_shape = (-1, num_channels, time_series_length)
    expected_discrete_conditions_shape = (
        -1,
        time_series_length,
        num_discrete_conditions,
    )
    expected_time_invariant_discrete_conditions_shape = (
        -1,
        num_discrete_conditions,
    )
    expected_continuous_conditions_shape = (
        -1,
        time_series_length,
        num_continuous_conditions,
    )

    # Check timeseries shape
    if (
        timeseries_dataset is not None
        and timeseries_dataset.shape[1:] != expected_timeseries_shape[1:]
    ):
        raise ValueError(
            f"Timeseries dataset shape mismatch. Expected {expected_timeseries_shape} according to the config, got {timeseries_dataset.shape}"
        )

    # Check discrete conditions shape if they exist
    if discrete_conditions.shape[0] > 0 and len(discrete_conditions.shape) == 3:
        if (
            discrete_conditions.shape[1:]
            != expected_discrete_conditions_shape[1:]
        ):
            raise ValueError(
                f"Discrete conditions shape mismatch. Expected {expected_discrete_conditions_shape} according to the config, got {discrete_conditions.shape}"
            )

    if discrete_conditions.shape[0] > 0 and len(discrete_conditions.shape) == 2:
        if (
            discrete_conditions.shape[1:]
            != expected_time_invariant_discrete_conditions_shape[1:]
        ):
            raise ValueError(
                f"Discrete conditions shape mismatch. Expected {expected_time_invariant_discrete_conditions_shape} according to the config, got {discrete_conditions.shape}"
            )

    # Check continuous conditions shape if they exist
    if continuous_conditions.shape[0] > 0:
        if (
            continuous_conditions.shape[1:]
            != expected_continuous_conditions_shape[1:]
        ):
            raise ValueError(
                f"Continuous conditions shape mismatch. Expected {expected_continuous_conditions_shape} according to the config, got {continuous_conditions.shape}"
            )

    # Check timestamp conditions shape(s) if they exist
    if timestamp_conditions is not None:
        # Check if the timestep dimension is correct
        if timestamp_conditions.shape[1] != expected_timeseries_shape[2]:
            raise ValueError(
                f"Timestamp conditions shape mismatch. Expected {expected_timeseries_shape[2]} timesteps, got {timestamp_conditions.shape[1]}"
            )

        # Check if the number of timestamp conditions matches the config
        if timestamp_conditions.shape[2] != num_timestamp_conditions:
            raise ValueError(
                f"Timestamp conditions shape mismatch. Expected {num_timestamp_conditions} timestamp conditions according to the config, got {timestamp_conditions.shape[2]}"
            )


class SynthesisModelv1Dataset(torch.utils.data.Dataset):  # pyright: ignore
    def __init__(
        self,
        config: DatasetConfig,
        train: bool = False,
        val: bool = False,
        test: bool = False,
        timeseries_dataset: np.ndarray | None = None,
        discrete_conditions: np.ndarray | None = None,
        continuous_conditions: np.ndarray | None = None,
    ) -> None:
        """
        inputs:
            config: DatasetConfig
            train: bool
            val: bool
            test: bool
        """

        logger.info("Using SynthesisModelv1Dataset")
        # Typically, the config passed in here is the `dataset_config`. This is different than `ForecastingDataset`.
        torch.manual_seed(config.seed)

        if (
            timeseries_dataset is None
            or discrete_conditions is None
            or continuous_conditions is None
        ):
            dataset_log_dir = os.path.join(  # pyright: ignore
                os.getenv("SYNTHEFY_DATASETS_BASE"),  # pyright: ignore
                config.dataset_name,
            )
            experiment = config.experiment

            (
                self.timeseries_dataset_loc,
                self.discrete_conditions_loc,
                self.continuous_conditions_loc,
            ) = get_dataset_paths(
                dataset_log_dir,
                experiment,
                train=train,
                val=val,
                test=test,
            )
            logger.info(
                OKBLUE
                + "The dataset location is : "
                + self.timeseries_dataset_loc
                + ENDC
            )
            logger.info(
                OKBLUE
                + "The discrete labels location is : "
                + self.discrete_conditions_loc
                + ENDC
            )
            logger.info(
                OKBLUE
                + "The continuous labels location is : "
                + self.continuous_conditions_loc
                + ENDC
            )

            self.timeseries_dataset = np.load(
                self.timeseries_dataset_loc, allow_pickle=True
            )
            self.discrete_conditions = np.load(
                self.discrete_conditions_loc, allow_pickle=True
            )
            self.continuous_conditions = np.load(
                self.continuous_conditions_loc, allow_pickle=True
            )

        else:
            logger.info("Using provided timeseries dataset, and metadata")
            self.timeseries_dataset = timeseries_dataset
            self.discrete_conditions = discrete_conditions
            self.continuous_conditions = continuous_conditions

        self.continuous_conditions_exist = (
            self.continuous_conditions.shape[0] > 0
        )
        self.discrete_conditions_exist = self.discrete_conditions.shape[0] > 0

        fail_if_dataset_shape_mismatch(
            timeseries_dataset=self.timeseries_dataset,
            discrete_conditions=self.discrete_conditions,
            continuous_conditions=self.continuous_conditions,
            time_series_length=config.time_series_length,
            num_channels=config.num_channels,
            num_discrete_conditions=config.num_discrete_conditions,
            num_continuous_conditions=config.num_continuous_labels,
        )

    def __len__(self):
        return self.timeseries_dataset.shape[0]

    def __getitem__(self, index):
        timeseries_full = self.timeseries_dataset[index]

        discrete_label_embedding = (
            self.discrete_conditions[index]
            if self.discrete_conditions_exist
            else np.array([])
        )
        continuous_label_embedding = (
            self.continuous_conditions[index]
            if self.continuous_conditions_exist
            else np.array([])
        )

        return {
            "timeseries_full": timeseries_full,
            "discrete_label_embedding": discrete_label_embedding,
            "continuous_label_embedding": continuous_label_embedding,
        }


class SynthesisModelv1DatasetFromMetadata(torch.utils.data.Dataset):  # pyright: ignore
    def _process_metadata(
        self, metadata: pd.DataFrame, preprocess_config_path: str
    ):
        df = add_missing_timeseries_columns(
            df=metadata, preprocess_config_path=preprocess_config_path
        )

        preprocess_config = yaml.safe_load(open(preprocess_config_path))
        preprocessor = DataPreprocessor(preprocess_config_path)
        if not preprocessor.use_label_col_as_discrete_metadata:
            preprocessor.group_labels_cols = []

        logger.info(
            "Setting stride to window size since the metadata is provided."
        )
        preprocessor.stride = preprocessor.window_size
        if preprocess_config.get("add_lag_data", False):
            preprocessor.stride = preprocessor.window_size

        self.saved_scalers = {
            "timeseries": load_timeseries_scalers(self.dataset_name),
            "continuous": load_continuous_scalers(self.dataset_name),
        }
        self.encoders = load_discrete_encoders(self.dataset_name)
        preprocessor.process_data(
            df,
            saved_scalers=self.saved_scalers,
            saved_encoders=self.encoders,
            save_files_on=False,
        )

        return (
            preprocessor.windows_data_dict["timeseries"]["windows"],
            preprocessor.windows_data_dict["continuous"]["windows"],
            preprocessor.windows_data_dict["discrete"]["windows"],
        )

    def __init__(
        self,
        config: DatasetConfig,
        metadata_for_synthesis: pd.DataFrame,
        preprocess_config_path: str,
    ) -> None:
        logger.info("Using SynthesisModelv1DatasetFromMetadata")

        torch.manual_seed(config.seed)
        self.time_series_length = config.time_series_length
        self.num_channels = config.num_channels
        self.experiment = config.experiment
        self.dataset_name = config.dataset_name

        if "windows" not in metadata_for_synthesis:
            timeseries, continuous_conditions, discrete_conditions = (
                self._process_metadata(
                    metadata_for_synthesis, preprocess_config_path
                )
            )
            self.timeseries = timeseries
            self.continuous_conditions = continuous_conditions
            self.discrete_conditions = discrete_conditions

        elif "windows" in metadata_for_synthesis:
            list_of_metadata = metadata_for_synthesis["windows"]
            num_windows = len(list_of_metadata)
            window_size = len(list_of_metadata[0])
            # Process each dictionary in the list and concatenate results
            timeseries_list = []
            continuous_list = []
            discrete_list = []
            for metadata_dict in list_of_metadata:
                timeseries, cont_cond, disc_cond = self._process_metadata(
                    metadata_dict, preprocess_config_path
                )
                timeseries_list.append(timeseries)
                continuous_list.append(cont_cond)
                discrete_list.append(disc_cond)

            self.timeseries = np.concatenate(timeseries_list, axis=0)
            self.continuous_conditions = np.concatenate(continuous_list, axis=0)
            self.discrete_conditions = np.concatenate(discrete_list, axis=0)
        else:
            raise ValueError(
                "metadata_for_synthesis must be a dictionary or a list of dictionaries"
            )

        self.timeseries = self.timeseries.transpose(0, 2, 1)
        self.continuous_conditions_exist = (
            self.continuous_conditions.shape[0] > 0
        )
        self.discrete_conditions_exist = self.discrete_conditions.shape[0] > 0

        if (
            not self.discrete_conditions_exist
            and not self.continuous_conditions_exist
        ):
            raise ValueError(
                "No conditions exist - please use input JSON that has metadata"
            )
        # if the metadata is not present, set its feature dim to shape 0
        if not self.discrete_conditions_exist:
            self.discrete_conditions = np.zeros((num_windows, window_size, 0))
        if not self.continuous_conditions_exist:
            self.continuous_conditions = np.zeros((num_windows, window_size, 0))

        fail_if_dataset_shape_mismatch(
            timeseries_dataset=(
                self.timeseries if self.timeseries is not None else None
            ),
            discrete_conditions=self.discrete_conditions,
            continuous_conditions=self.continuous_conditions,
            time_series_length=config.time_series_length,
            num_channels=config.num_channels,
            num_discrete_conditions=config.num_discrete_conditions,
            num_continuous_conditions=config.num_continuous_labels,
        )

    def __len__(self):
        # one of continuous/discrete must exist
        if self.discrete_conditions_exist:
            return self.discrete_conditions.shape[0]
        else:
            return self.continuous_conditions.shape[0]

    def __getitem__(self, index):
        # unused since only inference for synthesis
        timeseries_full = (
            np.zeros((self.num_channels, self.time_series_length))
            if self.timeseries is None
            else self.timeseries[index]
        )

        discrete_label_embedding = (
            self.discrete_conditions[index]
            if self.discrete_conditions_exist
            else np.array([])
        )
        continuous_label_embedding = (
            self.continuous_conditions[index]
            if self.continuous_conditions_exist
            else np.array([])
        )

        return {
            "timeseries_full": timeseries_full,
            "discrete_label_embedding": discrete_label_embedding,
            "continuous_label_embedding": continuous_label_embedding,
        }


class ForecastingDataset(torch.utils.data.Dataset):  # pyright: ignore
    def __init__(
        self,
        config: Configuration,
        train: bool = False,
        val: bool = False,
        test: bool = False,
    ) -> None:
        torch.manual_seed(config.seed)
        dataset_config = config.dataset_config
        dataset_log_dir = os.path.join(
            os.getenv("SYNTHEFY_DATASETS_BASE"),  # pyright: ignore
            config.dataset_config.dataset_name,
        )

        if train:
            self.timeseries_dataset_loc = os.path.join(
                dataset_log_dir, "train_timeseries.npy"
            )
            self.discrete_conditions_loc = os.path.join(
                dataset_log_dir, "train_discrete_conditions.npy"
            )
            self.continuous_conditions_loc = os.path.join(
                dataset_log_dir, "train_continuous_conditions.npy"
            )
            self.timestamp_conditions_loc = os.path.join(
                dataset_log_dir, "train_timestamp_conditions.npy"
            )
        elif val:
            self.timeseries_dataset_loc = os.path.join(
                dataset_log_dir, "val_timeseries.npy"
            )
            self.discrete_conditions_loc = os.path.join(
                dataset_log_dir, "val_discrete_conditions.npy"
            )
            self.continuous_conditions_loc = os.path.join(
                dataset_log_dir, "val_continuous_conditions.npy"
            )
            self.timestamp_conditions_loc = os.path.join(
                dataset_log_dir, "val_timestamp_conditions.npy"
            )
        elif test:
            self.timeseries_dataset_loc = os.path.join(
                dataset_log_dir, "test_timeseries.npy"
            )
            self.discrete_conditions_loc = os.path.join(
                dataset_log_dir, "test_discrete_conditions.npy"
            )
            self.continuous_conditions_loc = os.path.join(
                dataset_log_dir, "test_continuous_conditions.npy"
            )
            self.timestamp_conditions_loc = os.path.join(
                dataset_log_dir, "test_timestamp_conditions.npy"
            )

        logger.info(
            OKBLUE
            + "The dataset location is : "
            + self.timeseries_dataset_loc
            + ENDC
        )
        logger.info(
            OKBLUE
            + "The discrete labels location is : "
            + self.discrete_conditions_loc
            + ENDC
        )
        logger.info(
            OKBLUE
            + "The continuous labels location is : "
            + self.continuous_conditions_loc
            + ENDC
        )
        logger.info(
            OKBLUE
            + "The timestamp labels location is : "
            + self.timestamp_conditions_loc
            + ENDC
        )

        self.timeseries_dataset = np.load(
            self.timeseries_dataset_loc, allow_pickle=True
        )
        self.discrete_conditions = np.load(
            self.discrete_conditions_loc, allow_pickle=True
        )
        self.continuous_conditions = np.load(
            self.continuous_conditions_loc, allow_pickle=True
        )
        self.timestamp_conditions = np.load(
            self.timestamp_conditions_loc, allow_pickle=True
        )

        self.discrete_conditions_exist = self.discrete_conditions.shape[0] > 0
        self.continuous_conditions_exist = (
            self.continuous_conditions.shape[0] > 0
        )
        self.timestamp_conditions_exist = (
            True if dataset_config.use_timestamp else False
        )

        fail_if_dataset_shape_mismatch(
            timeseries_dataset=self.timeseries_dataset,
            discrete_conditions=self.discrete_conditions,
            continuous_conditions=self.continuous_conditions,
            time_series_length=dataset_config.time_series_length,
            num_channels=dataset_config.num_channels,
            num_discrete_conditions=dataset_config.num_discrete_conditions,
            num_continuous_conditions=dataset_config.num_continuous_labels,
            num_timestamp_conditions=dataset_config.num_timestamp_labels,
            timestamp_conditions=self.timestamp_conditions,
        )

    def __len__(self):
        return self.timeseries_dataset.shape[0]

    def __getitem__(self, index):
        timeseries_full = self.timeseries_dataset[index]

        continuous_label_embedding = (
            self.continuous_conditions[index]
            if self.continuous_conditions_exist
            else np.array([])
        )
        discrete_label_embedding = (
            self.discrete_conditions[index]
            if self.discrete_conditions_exist
            else np.array([])
        )
        timestamp_label_embedding = (
            self.timestamp_conditions[index]
            if self.timestamp_conditions_exist
            else np.array([])
        )

        return {
            "timeseries_full": timeseries_full,
            "discrete_label_embedding": discrete_label_embedding,
            "continuous_label_embedding": continuous_label_embedding,
            "timestamp_label_embedding": timestamp_label_embedding,
        }


class SynthesisModelv1DataLoader(SynthesisBaseDataModule):
    def __init__(
        self,
        config: DatasetConfig,
        metadata_for_synthesis: Optional[pd.DataFrame] = None,
        preprocess_config_path: Optional[str] = None,
        timeseries_dataset: Optional[np.ndarray] = None,
        discrete_conditions: Optional[np.ndarray] = None,
        continuous_conditions: Optional[np.ndarray] = None,
    ):
        if metadata_for_synthesis is None:
            logger.info("Loading train dataset")
            train_dataset = SynthesisModelv1Dataset(
                config,
                train=True,
                timeseries_dataset=timeseries_dataset,
                discrete_conditions=discrete_conditions,
                continuous_conditions=continuous_conditions,
            )
            logger.info("Loading val dataset")
            val_dataset = SynthesisModelv1Dataset(
                config,
                val=True,
                timeseries_dataset=timeseries_dataset,
                discrete_conditions=discrete_conditions,
                continuous_conditions=continuous_conditions,
            )
            logger.info("Loading test dataset")
            test_dataset = SynthesisModelv1Dataset(
                config,
                test=True,
                timeseries_dataset=timeseries_dataset,
                discrete_conditions=discrete_conditions,
                continuous_conditions=continuous_conditions,
            )
        else:
            assert preprocess_config_path is not None, (
                "preprocess_config_path must be provided if metadata_for_synthesis is provided"
            )
            logger.info(
                "Creating synthetic dataset for the metadata conditions"
            )
            train_dataset = None
            val_dataset = None
            test_dataset = SynthesisModelv1DatasetFromMetadata(
                config,
                metadata_for_synthesis=metadata_for_synthesis,
                preprocess_config_path=preprocess_config_path,  # pyright: ignore
            )
        super().__init__(config, train_dataset, val_dataset, test_dataset)  # pyright: ignore


class ForecastingDataLoader(SynthesisBaseDataModule):
    def __init__(self, config: Configuration):
        logger.info("Loading train dataset")
        train_dataset = ForecastingDataset(config, train=True)
        logger.info("Loading val dataset")
        val_dataset = ForecastingDataset(config, val=True)
        logger.info("Loading test dataset")
        test_dataset = ForecastingDataset(config, test=True)
        super().__init__(
            config, train_dataset, val_dataset, test_dataset, forecasting=True
        )
