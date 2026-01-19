import json
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dotenv import load_dotenv
from loguru import logger

SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")
assert SYNTHEFY_PACKAGE_BASE is not None
load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
assert SYNTHEFY_DATASETS_BASE is not None, (
    "SYNTHEFY_DATASETS_BASE environment variable must be set."
)

COMPILE = True


@dataclass
class BaseConfig(ABC):
    config_source: Union[str, Dict[str, Any]]
    config: Dict[str, Any] = field(init=False)
    filename: Optional[str] = field(init=False)
    timestamps_col: List[str] = field(init=False)
    group_labels_config: Dict[str, Any] = field(init=False)
    timeseries_config: Dict[str, Any] = field(init=False)
    continuous_config: Dict[str, Any] = field(init=False)
    discrete_config: Dict[str, Any] = field(init=False)
    text_config: Dict[str, Any] = field(init=False)
    text_cols: List[str] = field(init=False)
    window_size: int = field(init=False)
    group_labels_cols: List[str] = field(init=False)
    timeseries_cols: List[str] = field(init=False)
    use_label_col_as_discrete_metadata: bool = field(init=False)
    continuous_cols: List[str] = field(init=False)
    discrete_cols: List[str] = field(init=False)
    freq_map: Dict[str, str] = field(
        default_factory=lambda: {"H": "hour", "T": "minute", "D": "day"}
    )
    window_types: List[str] = field(init=False)
    windows_filenames_dict: Dict[str, Dict[str, str]] = field(init=False)
    # windows_data_dict - dict that stores np.arrays of different dataset types
    # dataset_types are train, val, test.
    windows_data_dict: Dict[str, Dict[str, np.ndarray]] = field(init=False)
    output_path: str = field(init=False)

    def __post_init__(self):
        (
            self.config,
            self.filename,
            self.use_label_col_as_discrete_metadata,
            self.timestamps_col,
            self.timestamps_config,
            self.group_labels_config,
            self.timeseries_config,
            self.continuous_config,
            self.discrete_config,
            self.text_config,
            self.window_size,
        ) = self.get_configs(self.config_source)

        self.group_labels_cols = self.group_labels_config.get("cols", [])
        self.timeseries_cols = self.timeseries_config.get("cols", [])
        self.continuous_cols = self.continuous_config.get("cols", [])
        self.discrete_cols = self.discrete_config.get("cols", [])
        text_cols = self.text_config.get("text_col_name", None)
        self.text_cols = [text_cols] if text_cols is not None else []

        self.window_types = [
            "timeseries",
            "continuous",
            "discrete",
            "timestamp",
            "timestamp_conditions",
            "original_discrete",
            "original_text",
        ]
        self.dataset_types = ["train", "val", "test"]
        self.windows_filenames_dict = {
            "timeseries": {
                "window_filename": "timeseries",
            },
            "continuous": {
                "window_filename": "continuous_conditions",
            },
            "discrete": {
                "window_filename": "discrete_conditions",
            },
            "timestamp": {"window_filename": "timestamps_original"},
            "timestamp_conditions": {
                "window_filename": "timestamp_conditions",
            },
            "original_discrete": {
                "window_filename": "original_discrete_windows",
            },
            "original_text": {
                "window_filename": "original_text_conditions",
            },
        }
        self.windows_data_dict = {
            "timeseries": {
                "windows": np.array([]),
            },
            "continuous": {
                "windows": np.array([]),
            },
            "discrete": {
                "windows": np.array([]),
            },
            "timestamp": {"windows": np.array([])},
            "timestamp_conditions": {"windows": np.array([])},
            "original_discrete": {"windows": np.array([])},
            "original_text": {"windows": np.array([])},
        }
        self.freq_map = {
            "H": "hour",
            "T": "minute",
            "D": "day",
        }
        # TODO - the below is buggy. We should have a better way to get the output path ("_" can be in the dataset name)
        # Use custom_output_path if provided in the config otherwise default
        custom_output_path = self.config.get("custom_output_path", None)
        if custom_output_path not in [None, False, ""]:
            self.output_path = custom_output_path
        elif self.filename:
            self.output_path = os.path.join(
                str(SYNTHEFY_DATASETS_BASE), self.filename.split("/")[0]
            )
        else:
            raise ValueError(
                "Custom output path or filename not provided in the config, cannot determine output path."
            )

        path_type = "CUSTOM" if custom_output_path else "DEFAULT"
        logger.warning(f"Using {path_type} output path: {self.output_path}")

        try:
            os.makedirs(self.output_path, exist_ok=True)
        except Exception as e:
            self.output_path = os.path.join(
                str(SYNTHEFY_DATASETS_BASE), str(self.filename).split("/")[0]
            )
            logger.error(
                f"Error creating output path: {e} - not setting custom output path - using default path: {self.output_path}"
            )

    def get_configs(
        self, config_source: Union[str, Dict[str, Any]]
    ) -> Tuple[
        Dict[str, Any],  # config
        Optional[str],  # filename
        bool,  # use_label_col_as_discrete_metadata
        List[str],  # timestamps_col
        Dict[str, Any],  # timestamps_config
        Dict[str, Any],  # group_labels_config
        Dict[str, Any],  # timeseries_config
        Dict[str, Any],  # continuous_config
        Dict[str, Any],  # discrete_config
        Dict[str, Any],  # text_metadata
        int,  # window_size
    ]:
        """
        Loads configuration from either a JSON file path or a dictionary.

        Parameters:
            - config_source: either a path to the config file or a config dictionary
        Returns:
            - config: Complete configuration dictionary
            - filename: Name of the data file (csv|parquet)
            - use_label_col_as_discrete_metadata: Boolean flag for label column usage in discrete conditions
            - timestamps_col: List of timestamp column names
            - timestamps_config: Timestamp configuration dictionary
            - group_labels_config: Group labels configuration dictionary
            - timeseries_config: Time series configuration dictionary
            - continuous_config: Continuous variables configuration dictionary
            - discrete_config: Discrete variables configuration dictionary
            - window_size: Size of the window
        """
        if isinstance(config_source, str):
            logger.warning(f"Using configuration from file: {config_source}")
            with open(os.path.join(config_source), "r") as file:
                config = json.load(file)
        else:
            logger.warning("Using configuration from dictionary.")
            config = config_source

        if "window_size" not in config:
            raise ValueError("'window_size' is missing from the configuration.")
        if "filename" not in config:
            logger.warning("'filename' is missing from the configuration.")

        filename: Optional[str] = config.get("filename", None)
        window_size: int = config["window_size"]

        return (
            config,
            filename,
            config.get("use_label_col_as_discrete_metadata", True),
            config.get("timestamps_col", []),
            config.get("timestamps", {}),
            config.get("group_labels", {}),
            config.get("timeseries", {}),
            config.get("continuous", {}),
            config.get("discrete", {}),
            config.get("text_metadata", {}),
            window_size,
        )
