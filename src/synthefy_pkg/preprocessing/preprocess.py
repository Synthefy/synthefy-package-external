import gc
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from synthefy_pkg.preprocessing.base_config import (
    SYNTHEFY_DATASETS_BASE,
    BaseConfig,
)
from synthefy_pkg.preprocessing.base_text_embedder import (
    EMBEDDED_COL_NAMES_PREFIX,
    BaseTextEmbedder,
)
from synthefy_pkg.preprocessing.embedding_encoder import EmbeddingEncoder
from synthefy_pkg.utils.basic_utils import check_missing_cols
from synthefy_pkg.utils.memory_monitor import (
    BYTES_PER_FLOAT32,
    BYTES_PER_MB,
    CHECK_STEP,
    COARSE_CHECK_STEP,
    DEFAULT_ALLOCATION_SAFETY_FACTOR,
    DEFAULT_GROUP_CHECK_FREQUENCY,
    LARGE_NUMBER_OF_WINDOW_THRESHOLD,
    MemoryMonitor,
    MemoryThresholds,
)
from synthefy_pkg.utils.preprocessing_utils import (
    create_group_label_to_split_idx_dict,
    find_group_train_val_test_split_inds_stratified,
    validate_split_indices,
)

# from synthefy_pkg.utils.expiry import check_codebase_expired  # turning off as it takes too long to run
from synthefy_pkg.utils.scaling_utils import (
    SCALER_FILENAMES,
    SCALER_TYPES,
    fit_scaler,
    transform_using_scaler,
)
from synthefy_pkg.utils.train_config_utils import generate_default_train_config

COMPILE = True

# check_codebase_expired() # turning off as it takes too long to run

logger.add("preprocess.log", level="INFO")

TIMESTAMPS_FEATURES_PREFIX: str = "timestamps_feature_"
DISCRETE_METADATA_ENCODER_TYPES: List[str] = ["onehot", "embedding"]
# TODO: add more encoder types here
ENCODER_TYPES: Dict[str, Any] = {}
TIMESTAMPS_FEATURE_TYPES: List[str] = ["Y", "M", "D", "H", "T", "S"]

DEFAULT_DISCRETE_ENCODER = "onehot"
DEFAULT_SCALER = "standard"
DEFAULT_TEXT_EMBEDDING_SCALER = "none"
LAG_COL_FORMAT = "{col}_lag_{window_size}"
LAG_DATA_FILENAME_SUFFIX = "_with_lag_data.parquet"

MONTHS_IN_A_YEAR: int = 12
DAYS_IN_A_MONTH: int = 31
HOURS_IN_A_DAY: int = 24
MINUTES_IN_AN_HOUR: int = 60
SECONDS_IN_A_MINUTE: int = 60

SD_CLIP_FACTOR = 5
np.random.seed(42)


COMPILE = True


def validate_memory_monitoring_config(config: Dict[str, Any]) -> bool:
    """
    Validate the memory monitoring configuration.

    Parameters:
    - config (Dict[str, Any]): The configuration dictionary containing memory_monitoring section

    Returns:
    - bool: True if all required fields are present and valid, False otherwise
    """
    try:
        memory_config = config.get("memory_monitoring")
        if memory_config is None:
            return False

        required_fields = [
            "process_memory_mb",
            "system_memory_percent",
            "check_interval",
        ]

        # Check if all required fields are present
        for field in required_fields:
            if field not in memory_config:
                logger.warning(
                    f"Missing required field '{field}' in memory_monitoring config"
                )
                return False

        # Validate field types and values
        if (
            not isinstance(memory_config["process_memory_mb"], (int, float))
            or memory_config["process_memory_mb"] <= 0
        ):
            logger.warning("process_memory_mb must be a positive number")
            return False

        if (
            not isinstance(memory_config["system_memory_percent"], (int, float))
            or memory_config["system_memory_percent"] <= 0
            or memory_config["system_memory_percent"] > 100
        ):
            logger.warning(
                "system_memory_percent must be a positive number between 0 and 100"
            )
            return False

        if (
            not isinstance(memory_config["check_interval"], (int, float))
            or memory_config["check_interval"] <= 0
        ):
            logger.warning("check_interval must be a positive number")
            return False

        return True

    except Exception as e:
        logger.warning(f"Error validating memory monitoring config: {e}")
        return False


def find_window_start_idxs(
    window_size: int, stride: int, group_len_dict: Dict[str, int]
) -> Dict[str, List[int]]:
    """
    Find the starting indices for each window in a df of grouped data.

    Parameters:
    - window_size (int): The size of each window.
    - stride (int): The number of steps to move the window forward.
    - group_len_dict (Dict[str, int]): Dictionary mapping group keys to number of windows.

    Returns:
    - List[int]: Starting indices for each window.
    """
    window_start_row_indices = {}
    group_start_idx = 0
    for group_key, group_len in group_len_dict.items():
        # Calculate number of complete windows that fit in this group
        num_windows = max(0, (group_len - window_size) // stride + 1)

        # Get window start indices for this group
        window_start_row_indices[group_key] = [
            group_start_idx + i * stride for i in range(num_windows)
        ]

        # Update group_start_idx to start of next group, ensuring no overlap
        group_start_idx += group_len

    return window_start_row_indices


def find_group_train_val_test_split_inds_byos(
    splits_values: np.ndarray,
) -> Dict[str, List[int]]:
    """
    Creates a dictionary mapping data split categories to their corresponding indices based on provided split values.
    BYOS stands for "Bring Your Own Split", indicating that the split values are predefined.

    Parameters:
    -----------
    splits_values : np.ndarray
        An array containing split labels for each window (each value is the first value of each window).
        Each value should be one of: "train", "test", or "val".

    Returns:
    --------
    Dict[str, List[int]]
        A dictionary with three keys: "train", "test", and "val". Each key maps to a list
        of indices where that split type occurs in the input array which corresponds to
        the window indices for each dataset type.

    Example:
    --------
    >>> splits = np.array(["train", "test", "train", "val"])
    >>> find_group_train_val_test_split_inds_byos(splits)
    {
        "train": [0, 2],
        "test": [1],
        "val": [3]
    }
    """
    split_inds_dict = {"train": [], "test": [], "val": []}

    for index, value in enumerate(splits_values):
        if value in split_inds_dict:
            split_inds_dict[value].append(index)

    return split_inds_dict


def find_dataset_type_df_indices(
    split_inds_dict: Dict[str, List[int]],
    window_size: int,
    window_start_row_indices: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    """
    Create a dictionary mapping data split categories to their corresponding row indices in the DataFrame.
    Parameters:
    -----------
    split_inds_dict : Dict[str, List[int]]
        A dictionary mapping data split categories to their window indices
    window_size : int
        Size of each window
    window_start_row_indices : Dict[str, List[int]]
        Dictionary mapping group names to lists of starting row indices for each window in that group
    Returns:
    --------
    Dict[str, List[int]]
        A dictionary with three keys: "train", "test", and "val". Each key maps to a list
        of row indices in the DataFrame for that split type.
    Example:
    --------
    >>> split_inds_dict = {
        "train": [0, 2],
        "test": [1],
        "val": [3]
        }
    >>> window_start_row_indices = {
        "group1": [0, 50],
        "group2": [100, 150]
        }
    >>> window_size = 10
    >>> # Result structure:
    >>> # {
    >>> #     "train": [0, 1, 2, 3, 4, 5],
    >>> #     "val": [4, 5, 6, 7]
    >>> #     "test": [6, 7],
    >>> # }
    """
    # Create a mapping from window index to its corresponding row indices
    window_to_rows = {}
    window_idx = 0

    # For each group
    for group_starts in window_start_row_indices.values():
        # For each window start index in the group
        for start_idx in group_starts:
            # Map this window index to its row range
            window_to_rows[window_idx] = list(
                range(start_idx, start_idx + window_size)
            )
            window_idx += 1

    # Create the output dictionary with row indices for each split
    df_indices_dict = {
        split_type: [] for split_type in ["train", "val", "test"]
    }

    # For each split type (train/val/test)
    for split_type, window_indices in split_inds_dict.items():
        # For each window index in this split
        for window_idx in window_indices:
            # Add the corresponding row indices to the list
            df_indices_dict[split_type].extend(window_to_rows[window_idx])

        # Sort the indices and remove duplicates
        df_indices_dict[split_type] = sorted(
            list(set(df_indices_dict[split_type]))
        )

    return df_indices_dict


def generate_windows(
    df: pd.DataFrame,
    window_start_row_indices: Dict[str, List[int]],
    window_size: int,
    memory_monitor: Optional[MemoryMonitor] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generate windows from a DataFrame using known start indices and window size.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - window_start_row_indices (Dict[str, List[int]]): Dictionary of starting indices for each window.
    - window_size (int): The size of each window.
    - memory_monitor (Optional[MemoryMonitor]): Memory monitor for checking limits during generation.

    Yields:
    - np.ndarray: A window of data as a NumPy array.
    """
    # Calculate total number of windows for tqdm
    total_windows = sum(
        len(indices) for indices in window_start_row_indices.values()
    )

    logger.warning(
        f"About to generate {total_windows:,} windows - this may consume significant memory!"
    )

    with tqdm(total=total_windows, desc="Generating windows") as pbar:
        window_count = 0
        group_count = 0

        for group_name, indices in window_start_row_indices.items():
            group_count += 1

            # Check memory at configured group frequency
            if (
                memory_monitor
                and group_count % DEFAULT_GROUP_CHECK_FREQUENCY == 0
            ):
                memory_monitor.check_memory_limits(
                    f"window_generation_group_{group_count}_{group_name}"
                )

            for start_idx in indices:
                yield df.iloc[start_idx : start_idx + window_size].values
                pbar.update(1)
                window_count += 1

                if (
                    memory_monitor
                    and total_windows > LARGE_NUMBER_OF_WINDOW_THRESHOLD
                    and window_count % COARSE_CHECK_STEP == 0
                ):
                    memory_monitor.check_memory_limits(
                        f"window_generation_frequent_{window_count}"
                    )

                elif memory_monitor and window_count % CHECK_STEP == 0:
                    memory_monitor.check_memory_limits(
                        f"window_generation_progress_{window_count}"
                    )


def add_time_features(
    df: pd.DataFrame, timestamps_col: str, timestamps_features_list: List[str]
) -> pd.DataFrame:
    """
    Adds normalized time features to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to enhance.
    - timestamps_col (str): The timestamp column to derive features from.
    - timestamps_features_list (List[str]): List of time features to add.

    Returns:
    - pd.DataFrame: Enhanced DataFrame with added time features.
    """
    # Ensure df is a copy to avoid SettingWithCopyWarning
    if not pd.api.types.is_datetime64_any_dtype(df[timestamps_col]):
        df[timestamps_col] = pd.to_datetime(df[timestamps_col])

    for feature in timestamps_features_list:
        timestamps_feat_colname = f"{TIMESTAMPS_FEATURES_PREFIX}{feature}"
        if feature == "Y":
            if df[timestamps_col].dt.year.nunique() == 1:
                df.loc[:, timestamps_feat_colname] = 0.0
            else:
                df.loc[:, timestamps_feat_colname] = (
                    df[timestamps_col].dt.year
                    - df[timestamps_col].dt.year.min()
                ) / (
                    df[timestamps_col].dt.year.max()
                    - df[timestamps_col].dt.year.min()
                ) - 0.5
        elif feature == "M":
            df.loc[:, timestamps_feat_colname] = (
                df[timestamps_col].dt.month - (MONTHS_IN_A_YEAR / 2)
            ) / MONTHS_IN_A_YEAR
        elif feature == "D":
            df.loc[:, timestamps_feat_colname] = (
                df[timestamps_col].dt.day - (DAYS_IN_A_MONTH / 2)
            ) / DAYS_IN_A_MONTH
        elif feature == "H":
            df.loc[:, timestamps_feat_colname] = (
                df[timestamps_col].dt.hour - (HOURS_IN_A_DAY / 2)
            ) / HOURS_IN_A_DAY
        elif feature == "T":
            df.loc[:, timestamps_feat_colname] = (
                df[timestamps_col].dt.minute - (MINUTES_IN_AN_HOUR / 2)
            ) / MINUTES_IN_AN_HOUR
        elif feature == "S":
            df.loc[:, timestamps_feat_colname] = (
                df[timestamps_col].dt.second - (SECONDS_IN_A_MINUTE / 2)
            ) / SECONDS_IN_A_MINUTE
        else:
            raise ValueError(f"Invalid feature: {feature}")

    df.drop(columns=[timestamps_col], inplace=True)
    return df


def check_start_indices(
    df: pd.DataFrame,
    window_start_row_indices: Dict[str, List[int]],
    window_size: int,
) -> None:
    """
    Check that the start indices for each group are correct.
    """
    logger.info("Checking start indices")
    for v in window_start_row_indices.values():
        for start_idx in v:
            tmp = df.iloc[start_idx : start_idx + window_size]
            assert (tmp.iloc[0].values == tmp.iloc[-1].values).all()  # pyright: ignore

    logger.info("All start indices are correct")


def _add_lag_data(
    df: pd.DataFrame,
    timeseries_cols: List[str],
    window_size: int,
) -> pd.DataFrame:
    """
    input: df with timeseries_cols
    output: df with timeseries_cols and lag_timeseries_cols
    description:
        Add lag data to the timeseries_cols.
        Shift by the window_size.
        Fill the first window_size rows with 0 since it has no lag data.
        Downstream, we will rename config file and update the continuous_cols to include these new lag columns.
    """
    df = df.copy()
    for col in timeseries_cols:
        lag_col = LAG_COL_FORMAT.format(col=col, window_size=window_size)
        df[lag_col] = df[col].shift(window_size)
        # Use iloc to index by position
        df.iloc[:window_size, df.columns.get_loc(lag_col)] = 0  # pyright: ignore

    return df


def group_generator(
    df: pd.DataFrame,
    group_columns: List[str],
    window_size: int,
    timestamps_col: List[str],
    intervals: Dict[str, Any],
    group_len_dict: Dict[str, int],
    timestamps_features_list: List[str],
    save_labels_description: bool,
    timeseries_cols: List[str],
    continuous_cols: List[str],
    discrete_cols: List[str],
    group_labels_cols: List[str],
    add_lag_data: bool = False,
    memory_monitor: Optional[MemoryMonitor] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Generates groups from the DataFrame based on the specified parameters.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_columns (List[str]): The columns to group by.
    - window_size (int): The size of the window.
    - stride (int): The stride for the window.
    - timestamps_col (List[str]): timestamps colname as a list with 1 element.
    - window_from_beginning (bool): Whether to start the window from the beginning.
    - intervals (Dict[str, Any]): Dictionary to store intervals for the timestamp column.
    - group_len_dict (Dict[str, int]): Dictionary to store the size of each group.
    - timestamps_features_list (List[str]): List of timestamp features to add.
    - save_labels_description (bool): Whether to save the labels description.
    - memory_monitor (Optional[MemoryMonitor]): Memory monitor for tracking usage.

    Yields:
    - pd.DataFrame: The group DataFrame.
    """
    group_count = 0
    for group_key, group in (
        df.groupby(group_columns) if group_columns else [(None, df)]
    ):
        group_count += 1

        if memory_monitor and group_count % DEFAULT_GROUP_CHECK_FREQUENCY == 0:
            memory_monitor.check_memory_limits(
                f"group_generator_processing_group_{group_count}"
            )
        if len(group) < window_size:
            continue

        if timestamps_col:
            group[timestamps_col[0]] = pd.to_datetime(group[timestamps_col[0]])
            group = group.sort_values(by=timestamps_col[0])

        group_len = len(group)
        group_len_dict[str(group_key)] = group_len

        # add the lag data
        if timeseries_cols is not None and add_lag_data:
            group = _add_lag_data(group, timeseries_cols, window_size)

        if timestamps_col:
            if save_labels_description:
                if intervals.get(timestamps_col[0], None) is None:
                    intervals[timestamps_col[0]] = (
                        group[timestamps_col[0]].iloc[1]
                        - group[timestamps_col[0]].iloc[0]
                        if len(group[timestamps_col[0]]) > 1
                        else pd.Timedelta(0)
                    )
                else:
                    intervals[timestamps_col[0]] = min(
                        intervals[timestamps_col[0]],
                        group[timestamps_col[0]].iloc[1]
                        - group[timestamps_col[0]].iloc[0],
                    )
            group = pd.concat(
                [
                    add_time_features(
                        group[timestamps_col].copy(),
                        timestamps_col[0],
                        timestamps_features_list,
                    ),
                    group,
                ],
                axis=1,
            )
        # Replace NaNs with last non-NaN value. If the series starts with NaNs (e.g. no earlier values), then backfill.
        # Finally, if there is an all-NaN series, fill with 0.
        # This is the only place NaNs are replaced.
        # Use predefined column types to handle NaN filling appropriately
        numeric_cols = timeseries_cols + continuous_cols
        categorical_cols = discrete_cols + group_labels_cols
        if numeric_cols:
            group[numeric_cols] = group[numeric_cols].ffill().bfill().fillna(0)
        if categorical_cols:
            group[categorical_cols] = group[categorical_cols].ffill().bfill()
        yield group


def generate_labels_description(
    df: pd.DataFrame,
    intervals: Dict[str, Any],
    discrete_cols: List[str],
    continuous_cols: List[str],
    timestamps_col: List[str],
    group_labels_cols: List[str],
) -> Dict[str, Any]:
    """
    Generate and save a description of labels for different data types.
    discrete_labels: dict of value counts for each discrete column
    group_labels_combinations: list of unique combinations of group labels
    continuous_labels: dict of min, max, mean for each continuous column
    time_labels: dict of min, max, interval for each timestamp column

    Args:
        - df (pd.DataFrame): The input DataFrame.
        - group_labels_combinations (List[str]): List of unique group label combinations.
        - intervals (Dict[str, Any]): Dictionary timestamp_col to its minimum interval between first 2 rows of its groups.
        - discrete_cols (List[str]): List of discrete columns.
        - continuous_cols (List[str]): List of continuous columns.
        - timestamps_col (List[str]): List of timestamp columns.
        - group_labels_cols (List[str]): List of group label columns.
    Returns:
        - Dict[str, Any]: A dictionary containing the description of labels.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    logger.info("Generating labels description")
    if len(group_labels_cols) > 1:
        group_labels_combinations = (
            df[group_labels_cols]
            .groupby(group_labels_cols)
            .size()
            .index.map(lambda x: "-".join(map(str, x)))
            .tolist()
        )
    elif len(group_labels_cols) == 1:
        group_labels_combinations = df[group_labels_cols[0]].unique().tolist()
    else:
        group_labels_combinations = []

    labels_description = {
        "discrete_labels": (
            {
                col: df[col].value_counts().to_dict()
                for col in list(set(discrete_cols + group_labels_cols))
            }
            if list(set(discrete_cols + group_labels_cols))
            else {}
        ),
        "group_labels_combinations": {
            "-".join(group_labels_cols): group_labels_combinations
        },
        "continuous_labels": (
            df[continuous_cols].agg(["min", "max", "mean"]).to_dict()
            if continuous_cols
            else {}
        ),
        "time_labels": (
            {
                col: {
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "interval": intervals.get(col, pd.Timedelta(0)),
                }
                for col in timestamps_col
            }
            if timestamps_col
            else {}
        ),
        "group_label_cols": group_labels_cols,
    }
    return labels_description


def save_colnames_to_json(
    output_dir: str,
    timestamps_colnames: List[str],
    timeseries_colnames: List[str],
    continuous_colnames: List[str],
    discrete_colnames: List[str],
    original_discrete_colnames: List[str],
    original_text_colnames: List[str],
) -> None:
    """
    Save column names to a json.
    """
    # if it already exists, that means discrete metadata was time invariant/already saved, so add to it instead
    if os.path.exists(os.path.join(output_dir, "colnames.json")):
        with open(os.path.join(output_dir, "colnames.json"), "r") as f:
            colnames_dict = json.load(f)
        original_discrete_colnames = colnames_dict["original_discrete_colnames"]
        discrete_colnames = colnames_dict["discrete_colnames"]

    with open(os.path.join(output_dir, "colnames.json"), "w") as f:
        json.dump(
            {
                "timestamps_colnames": timestamps_colnames,
                "timeseries_colnames": timeseries_colnames,
                "continuous_colnames": continuous_colnames,
                "discrete_colnames": discrete_colnames,
                "original_discrete_colnames": original_discrete_colnames,
                "original_text_colnames": original_text_colnames,
            },
            f,
        )


def convert_scaler_dict_to_scaler_to_feature_dict(
    original_dict: Dict[str, Dict[str, str]],
) -> Dict[str, List[str]]:
    """
    Inverts a dictionary mapping columns to their scaler types into a dictionary mapping
    scaler types to lists of columns that use them.
    Parameters
    ----------
    original_dict : Dict[str, Dict[str, str]]
        Dictionary mapping column names to their scaler configurations
        Example: {"col1": {"scaler_type": "standard"}, "col2": {"scaler_type": "robust"}}
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping scaler types to lists of columns
        Example: {"standard": ["col1"], "robust": ["col2"]}
    """
    inverted_dict = defaultdict(list)
    for key, value in original_dict.items():
        inverted_dict[value["scaler_type"]].append(key)
    return dict(inverted_dict)


@dataclass
class DataPreprocessor(BaseConfig):
    group_labels_scalers_info: Dict[str, str] = field(init=False)
    timeseries_scalers_info: Dict[str, str] = field(init=False)
    continuous_scalers_info: Dict[str, str] = field(init=False)
    discrete_scalers_info: Dict[str, str] = field(init=False)
    window_from_beginning: bool = field(init=False)
    shuffle: bool = field(init=False)
    stride: int = field(init=False)
    train_val_split: Dict[str, float] = field(init=False)
    metadata_dict: Dict[str, Any] = field(default_factory=dict)
    final_discrete_cols: List[str] = field(init=False)
    original_discrete_cols: List[str] = field(init=False)
    already_saved_discrete: bool = False
    add_lag_data: bool = False
    scale_by_metadata: bool = False
    scale_by_metadata_cols: List[str] = field(default_factory=list)
    memory_monitor: Optional[MemoryMonitor] = field(init=False)
    allow_overlap: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.group_labels_scalers_info = self.group_labels_config.get(
            "scalers", {}
        )
        self.timeseries_scalers_info = self.timeseries_config.get("scalers", {})
        self.continuous_scalers_info = self.continuous_config.get("scalers", {})
        self.discrete_scalers_info = self.discrete_config.get("scalers", {})
        self.final_discrete_cols = []
        self.original_discrete_cols = []
        self.encoder_type = self.config.get("search_encoder", None)
        self.metadata_dict["discrete_encoders_info"] = {}
        self.timestamps_features_colnames = []
        self.use_custom_split = False
        self.add_lag_data = self.config.get("add_lag_data", False)
        self.allow_overlap = self.config.get("allow_overlap", False)

        # Initialize memory monitoring
        if validate_memory_monitoring_config(self.config):
            memory_config = self.config["memory_monitoring"]
            memory_thresholds = MemoryThresholds(
                process_memory_mb=memory_config["process_memory_mb"],
                system_memory_percent=memory_config["system_memory_percent"],
                check_interval=memory_config["check_interval"],
            )
            self.memory_monitor = MemoryMonitor(memory_thresholds)
            logger.info(
                f"Memory monitoring enabled with limits: "
                f"Process: {memory_thresholds.process_memory_mb}MB, "
                f"System: {memory_thresholds.system_memory_percent}%"
            )
        else:
            logger.warning(
                "Memory monitoring not enabled: invalid configuration"
            )
            self.memory_monitor = None

        np.random.seed(42)
        (
            self.window_from_beginning,
            self.shuffle,
            self.stride,
            self.val_stride,
            self.test_stride,
            self.train_val_split,
            self.shuffle_partitions_len,
        ) = self.validate_config(self.config)
        if self.timestamps_col:
            self.validate_timestamps_config()
        else:
            self.timestamps_features_list = []
            self.timestamps_features_colnames = []
            self.timestamps_scalers_info = {}

        if not self.shuffle:
            logger.warning(
                "Shuffle is set to False in the preprocessing config file. Verify this is intended."
            )

        if self.text_cols:
            self.text_embedder = BaseTextEmbedder.from_config(self.text_config)

        logger.info(f"Window size: {self.window_size}")
        logger.info(f"Stride: {self.stride}")
        if self.group_labels_cols:
            logger.info(f"Group labels: {self.group_labels_cols}")
        logger.info(f"Timeseries columns: {len(self.timeseries_cols)}")
        logger.info(f"Continuous columns: {len(self.continuous_cols)}")
        logger.info(f"Discrete columns: {len(self.discrete_cols)}")

    def load_and_prep_rawdata(
        self,
        fillna_outliers: bool = True,
        save_labels_description: bool = True,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Loads data from a CSV or parquet file, selects specified columns and
        fills missing values, infinities, outliers with 0.
        Adds time related features if timestamps_col is provided.
        Saves labels description.

        Parameters:
        - fillna_outliers (bool): If filling NaN and outliers with zeros is needed.
        - save_labels_description (bool): If saving labels description is needed.
        - df (pd.DataFrame): Optional DataFrame to use instead of loading from file (only for API now)

        Returns:
        - pd.DataFrame: Processed DataFrame.
        """
        logger.info("Importing data")
        required_cols = (
            list(self.timestamps_col)
            + list(self.group_labels_cols)
            + list(self.timeseries_cols)
            + list(self.continuous_cols)
            + list(self.discrete_cols)
            + list(self.text_cols)
        )
        start_time = time.time()

        if df is None and self.filename:
            file_path = os.path.join(str(SYNTHEFY_DATASETS_BASE), self.filename)
            logger.info(f"Loading data from {file_path}")
            if self.filename.lower().endswith(".csv"):
                df = pd.read_csv(file_path).reset_index(drop=True)
            elif self.filename.lower().endswith(".parquet"):
                df = pd.read_parquet(file_path).reset_index(drop=True)
            else:
                logger.error(f"Unsupported file format: {self.filename}")
                sys.exit(1)
        else:
            logger.info("Using provided DataFrame")

        assert df is not None, "Dataframe is None"

        check_missing_cols(required_cols, df)

        # Check if custom train/val/test split col was provided
        if "split" in df.columns:
            self.use_custom_split = True
            required_cols.extend(["split"])

        df = df[required_cols]
        assert df is not None
        df, self.group_len_dict, intervals = self.process_groups_and_timestamps(
            df, save_labels_description=save_labels_description
        )

        if self.timestamps_col:
            self.metadata_dict[self.timestamps_col[0]] = (
                df[self.timestamps_col[0]]
                .dt.strftime("%Y-%m-%d %H:%M:%S")
                .tolist()
            )
        self.metadata_dict["columns"] = list(df.columns)

        if save_labels_description:
            labels_description = generate_labels_description(
                df,
                intervals,  # pyright: ignore
                self.discrete_cols,
                self.continuous_cols,
                self.timestamps_col,
                self.group_labels_cols,
            )
            logger.info("Saving labels description")
            with open(
                os.path.join(self.output_path, "labels_description.pkl"), "wb"
            ) as file:
                pickle.dump(labels_description, file)

        # if self.group_labels_cols:
        #     if False:
        # for col in self.group_labels_cols:
        #     if any("-" in str(value) for value in df[col].unique()):
        #         raise ValueError(
        #             f"Group-label col {col} has '-' in the value. Please remove '-' from the group-label col."
        #         )

        if fillna_outliers:
            logger.info("Clipping outliers with mean +/- 5SD")
            df[list(self.timeseries_cols) + list(self.continuous_cols)] = (
                self.replace_outliers(
                    df[list(self.timeseries_cols) + list(self.continuous_cols)]
                )
            )

        logger.info(
            f"Data loaded in {round(time.time() - start_time, 2)} seconds; {df.shape = }"
        )
        return df

    def process_groups_and_timestamps(
        self,
        df: pd.DataFrame,
        save_labels_description: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, Any]]:
        """
        Calculate the required length for windowing for each group and drop the extra rows from beginning.
        Sort the groups by timestamps_col.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - Tuple[pd.DataFrame, List[int],  Dict[str, Any]]: preprocessed DataFrame, group lengths dict, intervals for each timestamp column.
        """
        logger.info(f"Prepreprocessing groups of {df.shape=}")
        group_len_dict = {}
        intervals = {}
        if self.timestamps_col:
            # Append new col names to continuous cols list
            self.continuous_cols = list(self.continuous_cols) + list(
                self.timestamps_features_colnames
            )
            self.continuous_scalers_info.update(self.timestamps_scalers_info)
            logger.info(f"Adding time features: {self.timestamps_scalers_info}")

        # Memory check before group processing
        if self.memory_monitor:
            self.memory_monitor.check_memory_limits("before_group_processing")

        # Concatenate all groups using the generator
        processed_df = pd.concat(
            group_generator(
                df,
                self.group_labels_cols,
                self.window_size,
                self.timestamps_col,
                intervals,
                group_len_dict,
                self.timestamps_features_list,
                save_labels_description,
                self.timeseries_cols,
                self.continuous_cols,
                self.discrete_cols,
                self.group_labels_cols,
                self.add_lag_data,
                self.memory_monitor,
            ),
            ignore_index=True,
        )

        # Memory check after group processing and pd.concat
        if self.memory_monitor:
            self.memory_monitor.check_memory_limits(
                "after_group_processing_and_concat"
            )
        if self.add_lag_data:
            for col in self.timeseries_cols:
                self.continuous_cols.append(
                    LAG_COL_FORMAT.format(col=col, window_size=self.window_size)
                )
            processed_df.reset_index(drop=True).to_parquet(
                os.path.join(
                    self.output_path,
                    self.config["filename"]
                    .split("/")[-1]
                    .replace(".csv", LAG_DATA_FILENAME_SUFFIX)
                    .replace(".parquet", LAG_DATA_FILENAME_SUFFIX),
                )
            )

        logger.info(f"Updated groups of {processed_df.shape=}")
        return (
            processed_df,
            group_len_dict,
            intervals,
        )

    def replace_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clips values outside ± SD_CLIP_FACTOR * std from mean; does not fill NaNs."""
        for col in df.columns:
            values = df[col].values.astype(np.float32)
            mean = np.nanmean(values)  # ignore the NaNs
            std = np.nanstd(values)
            df.loc[:, col] = np.clip(
                values, mean - SD_CLIP_FACTOR * std, mean + SD_CLIP_FACTOR * std
            )
        return df

    def embed_timeseries_for_search(
        self, timeseries_data: np.ndarray, encoder_type: Optional[str]
    ) -> None:
        """
        Encode timeseries data for further use in search.

        Parameters:
        - timeseries_data (np.ndarray): The timeseries data to encode.
        - encoder_type (Optional[str]): The type of encoder to use. If None, skip encoding.
        """
        if encoder_type is None or encoder_type not in ENCODER_TYPES:
            logger.warning(f"No valid encoder type specified ({encoder_type}), skipping encoding")
            return
        logger.info(f"{encoder_type} encoding timeseries data")
        encoder = ENCODER_TYPES[encoder_type](self.config)
        encoded_data = encoder.encode(
            timeseries_data, col_names=self.timeseries_cols
        )
        encoded_data_path = os.path.join(
            self.output_path, "ts_search_encoded_timeseries_data.pkl"
        )

        with open(encoded_data_path, "wb") as file:
            pickle.dump(
                {"encoder": encoder, "encoded_data": encoded_data}, file
            )
        logger.info(
            f"Saved {encoder_type} encoder and encoded timeseries data to {encoded_data_path}"
        )
        logger.info(f"{encoded_data.shape=}")

    def validate_timestamps_config(self) -> None:
        logger.info("Validating timestamps config")
        self.timestamps_features_list = self.timestamps_config.get(
            "timestamps_features", ["Y", "M", "D"]
        )
        if not set(self.timestamps_features_list).issubset(
            TIMESTAMPS_FEATURE_TYPES
        ):
            invalid_features = set(self.timestamps_features_list) - set(
                TIMESTAMPS_FEATURE_TYPES
            )
            raise ValueError(
                f"Invalid timestamps features: {invalid_features}. Allowed features are: {TIMESTAMPS_FEATURE_TYPES}"
            )
        self.timestamps_features_colnames = [
            f"{TIMESTAMPS_FEATURES_PREFIX}{feature}"
            for feature in self.timestamps_features_list
        ]
        timestamps_scalers_info = self.timestamps_config.get("scalers", {})
        if self.timestamps_config.get("scalers"):
            timestamps_scalers_info = {
                f"{TIMESTAMPS_FEATURES_PREFIX}{feature}": v
                for feature, v in timestamps_scalers_info.items()
            }
        for col in set(self.timestamps_features_colnames) - set(
            timestamps_scalers_info.keys()
        ):
            logger.warning(
                f"Scaler not specified for `{col}` of `timestamps_features`, using default minmax scaler"
            )
            timestamps_scalers_info.update({col: {"scaler_type": "minmax"}})
        self.timestamps_scalers_info = {
            k: timestamps_scalers_info[k]
            for k in self.timestamps_features_colnames
        }

    def validate_config(
        self, config: Dict[str, Any]
    ) -> Tuple[
        bool,
        bool,
        int,
        int,
        int,
        Dict[str, float],
        Optional[int],
    ]:
        """
        Validate config getting additional parameters necessary
        for preprocessing.

        Parameters:
        - config (Dict[str, Any]): The configuration dictionary.

        Returns:
        - Tuple[List[str], bool, bool, int, List[float]]: Configuration for timestamps features, window from beginning, shuffle, stride, train test split.
        """

        if self.discrete_cols:
            if set(self.group_labels_cols).intersection(
                set(self.discrete_cols)
            ):
                raise ValueError(
                    "There cannot be overlap in group_labels_cols and discrete_cols."
                )

        # Make sure group-label cols do not include "-" in the name since we will use it to combine the group labels
        for col in self.group_labels_cols:
            if "-" in col or "Label-Tuple-" in col:
                raise ValueError(
                    "Group-label cols cannot include '-' or 'Label-Tuple-' in the name."
                )

        stride = config.get("stride", self.window_size // 2)
        val_stride = config.get("val_stride", stride)
        test_stride = config.get("test_stride", stride)
        logger.info(
            f"Train stride: {stride}, Val stride: {val_stride}, Test stride: {test_stride}"
        )

        shuffle = config.get("shuffle", False)
        if shuffle:
            val_stride = test_stride = stride
            logger.warning(
                "Shuffle is enabled, using train_stride for all splits"
            )
        shuffle_partitions_len = config.get("shuffle_partitions_len", None)

        return (
            config.get("window_from_beginning", False),
            shuffle,
            config.get("stride", self.window_size // 2),
            val_stride,
            test_stride,
            config.get("train_val_split", {"train": 0.8, "val": 0.1}),
            shuffle_partitions_len,
        )

    def check_scaler_types(
        self, scalers_info: Dict[str, Dict[str, Any]], key: str
    ) -> None:
        """
        Check if the scalers_info listed in the config are supported.
        Parameters:
        - scalers_info (Dict[str, str]): Dictionary of scalers_info.
        - key (str): The key to check in the scalers_info dictionary "timeseries" or "continuous".
        """
        combined_scaler_types = set(SCALER_TYPES).union(
            DISCRETE_METADATA_ENCODER_TYPES
        )
        scalers_from_config = {
            value.get("scaler_type", None)
            for value in scalers_info.values()
            if value.get("scaler_type", None) is not None
        }

        all_exist = scalers_from_config.issubset(combined_scaler_types)
        if not all_exist:
            missed_types = scalers_from_config - combined_scaler_types
            error_message = (
                f"{missed_types} is not currently supported scaler. Please select one of: {list(SCALER_TYPES.keys())}"
                if key != "discrete"
                else f"{missed_types} is not currently supported discrete encoder. Please select one of: {DISCRETE_METADATA_ENCODER_TYPES}"
            )
            raise ValueError(error_message)

    def validate_scalers(self) -> None:
        """
        Validate scalers and encoders info dictionaries for different window types.
        Impute with default scaler or encoder info if missing for a column.
        """
        scaler_defaults = {
            "timeseries": DEFAULT_SCALER,
            "continuous": DEFAULT_SCALER,
            "discrete": DEFAULT_DISCRETE_ENCODER,
            "group_labels": DEFAULT_DISCRETE_ENCODER,
        }

        all_discrete_cols = (
            self.discrete_cols + self.group_labels_cols
            if self.use_label_col_as_discrete_metadata
            else self.discrete_cols
        )

        for key, default_type in scaler_defaults.items():
            logger.info(f"Validating `{key}` scalers info")
            cols = getattr(self, f"{key}_cols", [])
            scalers_info = getattr(self, f"{key}_scalers_info", {})

            if not cols or len(cols) == 0:
                logger.debug(f"No columns to validate for `{key}`.")
                continue

            self.check_scaler_types(scalers_info, key)

            for col in cols:
                scaler_type = scalers_info.get(col, {}).get("scaler_type")
                if scaler_type is None:
                    logger.warning(
                        f"Scaler info not specified for `{col}` of `{key}`, using default '{default_type}' scaler info."
                    )
                    scalers_info.update({col: {"scaler_type": default_type}})

                if EMBEDDED_COL_NAMES_PREFIX in col:
                    assert (
                        scalers_info.get(col, {}).get("scaler_type", None)
                        == DEFAULT_TEXT_EMBEDDING_SCALER
                    )

                # Handle scaling conditions for relevant keys
                if key in {"timeseries", "continuous"}:
                    scale_conditions = scalers_info[col].get(
                        "scale_by_discrete_metadata", None
                    )
                    if scale_conditions is not None:
                        group_labels = scale_conditions.get("group_labels", [])
                        if group_labels:
                            self.scale_by_metadata = True
                            self.scale_by_metadata_cols.extend(group_labels)
                            if not set(group_labels).issubset(
                                set(all_discrete_cols)
                            ):
                                raise ValueError(
                                    f"Scaling group labels for '{col}': {group_labels} are not in {all_discrete_cols}."
                                )
                            logger.info(
                                f"Will performing metadata specific scaling for `{col}` by discrete/group label cols: {group_labels}."
                            )

            # Order the scalers_info based on the columns' order
            ordered_scalers_info = {col: scalers_info[col] for col in cols}
            setattr(self, f"{key}_scalers_info", ordered_scalers_info)
            logger.debug(
                f"Ordered scalers_info for `{key}`: {ordered_scalers_info}."
            )

        logger.info("Scalers and encoders validated successfully.")

    def validate_discrete_cols_for_scale_by_metadata(
        self, df: pd.DataFrame, df_row_inds_dataset_types: Dict[str, List[int]]
    ) -> None:
        """
        Validate that all discrete values from the df for the columns
        in self.scale_by_metadata_cols are present in the training set.
        This ensures that the training set has at least one occurrence of each
        unique discrete label, which is necessary for proper scaling.
        """
        if not self.scale_by_metadata_cols:
            return

        for col in self.scale_by_metadata_cols:
            total_unique_values = set(df[col].unique())
            train_unique_values = set(
                df.loc[df_row_inds_dataset_types["train"], col].unique()
            )
            missing_values = total_unique_values - train_unique_values

            if missing_values:
                raise ValueError(
                    f"The discrete column '{col}' contains the following values "
                    f"that are not present in the training set: {missing_values}. "
                    "All discrete values must appear in the training set for proper scaling."
                )
        logger.info("Finished validating discrete cols for scale by metadata")

    def save_discrete_encoders(
        self, saved_encoders: Dict[str, Union[OneHotEncoder, EmbeddingEncoder]]
    ):
        encoder_path = os.path.join(
            self.output_path,
            SCALER_FILENAMES["discrete"],
        )
        with open(encoder_path, "wb") as f:
            pickle.dump(saved_encoders, f)

    # TODO: ensure discrete cols of training set has all labels
    def encode_by_type(
        self,
        encoder_type: Union[OneHotEncoder, EmbeddingEncoder],
        df: pd.DataFrame,
        extended_encode_cols: List[str],
        saved_encoder: Optional[Union[OneHotEncoder, EmbeddingEncoder]] = None,
    ) -> Tuple[pd.DataFrame, Union[OneHotEncoder, EmbeddingEncoder]]:
        """
        Encode specified columns in the DataFrame by type.

        Parameters:
        - encoder_type (Union[OneHotEncoder, EmbeddingEncoder]): The type of encoder to use.
        - df (pd.DataFrame): DataFrame to encode.
        - extended_encode_cols (List[str]): Columns to encode.
        - saved_encoder (Optional[Union[OneHotEncoder, EmbeddingEncoder]]): Saved encoder to use.

        Returns:
        - Tuple[pd.DataFrame, Union[OneHotEncoder, EmbeddingEncoder]]: Encoded DataFrame and the encoder used.
        """
        # The encode_cols are cols that must be dropped after encoding
        # we need the group label cols for grouping later
        self.original_discrete_cols.extend(extended_encode_cols)

        # "embedding" type discrete encoder needs to be converted to string
        df[extended_encode_cols] = df[extended_encode_cols].astype(str)
        if not saved_encoder:
            logger.warning(f"Generating encoder for: {encoder_type}")
            if encoder_type == "onehot":
                saved_encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                encoded = saved_encoder.fit_transform(df[extended_encode_cols])
            else:
                saved_encoder = EmbeddingEncoder()
                encoded = saved_encoder.fit_transform(df[extended_encode_cols])
        else:
            logger.warning(f"Using saved encoder for: {encoder_type}")
            encoded = saved_encoder.transform(df[extended_encode_cols])

        # new columns are discrete_cols_{} where {} is the unique value in the column
        new_columns = list(saved_encoder.get_feature_names_out())
        self.metadata_dict["discrete_encoders_info"][f"{encoder_type}_cols"] = (
            new_columns
        )
        encoded_df = pd.DataFrame(encoded, columns=new_columns, index=df.index)  # pyright: ignore
        df = pd.concat([df, encoded_df], axis=1)
        logger.info(f"{encoder_type} encoded cols: {len(new_columns)}")

        self.final_discrete_cols.extend(list(new_columns))
        return df, saved_encoder

    def encode_discrete(
        self,
        df: pd.DataFrame,
        saved_encoders: Dict[str, Union[OneHotEncoder, EmbeddingEncoder]] = {},
        save_files_on: bool = True,
    ) -> pd.DataFrame:
        """
        Encodes specified discrete columns in the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame to encode.
        - saved_encoders (Dict[str, Union[OneHotEncoder, EmbeddingEncoder]]): Saved encoders from training preprocessing.

        Returns:
        - pd.DataFrame: DataFrame with encoded columns appended.
        """

        # Find scaler types and corresponding discrete cols to apply it
        scaler_to_cols = {}

        common_dict = self.discrete_scalers_info.copy()
        if self.use_label_col_as_discrete_metadata:
            common_dict.update(self.group_labels_scalers_info)

        scaler_to_cols = convert_scaler_dict_to_scaler_to_feature_dict(
            common_dict  # pyright: ignore
        )

        for key, encode_cols in scaler_to_cols.items():
            logger.info(f"{key} encoding cols: {encode_cols}")
            df, saved_encoders[key] = self.encode_by_type(
                key,  # pyright: ignore
                df,
                encode_cols,
                saved_encoders.get(key, None),  # pyright: ignore
            )

        # Ensure the order of remaining columns is preserved
        remaining_cols = [
            col for col in df.columns if col not in self.original_discrete_cols
        ]

        # Move the original discrete cols to the end
        df[self.final_discrete_cols] = df[self.final_discrete_cols].astype(
            np.float32
        )
        df = df[remaining_cols + self.original_discrete_cols]

        if save_files_on:
            self.save_discrete_encoders(saved_encoders)

        return df

    def reshape_for_timeseries(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms a DataFrame into a 3D numpy array of sliding windows.

        Parameters:
        - df (pd.DataFrame): The DataFrame all data.

        Returns:
        - np.ndarray: A 3D numpy array where the first dimension indices the windows, the second dimension indices the timestamps within each window, and the third dimension indices the features.
        """
        # extract_windows_idxs = []
        # start = 0
        # for num_windows in self.num_windows_dict:
        #     extract_windows_idxs.extend([start + i * self.stride for i in range(num_windows)])
        #     start = extract_windows_idxs[-1] + self.window_size//self.stride

        # windows = np.lib.stride_tricks.sliding_window_view(
        #     df, window_shape=(self.window_size,), axis=0
        # )[extract_windows_idxs]
        # windows = np.transpose(windows, (0, 2, 1))
        # return windows

        logger.info("Started windowing")

        # Memory check before generating windows
        if self.memory_monitor:
            self.memory_monitor.check_memory_limits("before_generate_windows")
            memory_stats = self.memory_monitor.get_current_memory_usage()
            logger.info(
                f"Starting windowing with {memory_stats['process_memory_mb']:.1f}MB memory usage"
            )

        # Early feasibility check before allocating large window arrays
        if self.memory_monitor:
            total_windows_count = sum(
                len(indices)
                for indices in self.window_start_row_indices.values()
            )
            num_features_total = df.shape[1]
            # Rough estimate: windows_count * window_size * num_features * 4 bytes (float32)
            estimated_required_mb = (
                total_windows_count
                * self.window_size
                * num_features_total
                * BYTES_PER_FLOAT32
                / float(BYTES_PER_MB)
            )
            # Allow some headroom for Python/list/overheads
            self.memory_monitor.ensure_allocation_possible(
                required_mb=estimated_required_mb,
                operation_name="pre-windowing_estimation",
                safety_factor=DEFAULT_ALLOCATION_SAFETY_FACTOR,
            )

        windows_generator = generate_windows(
            df,
            self.window_start_row_indices,
            self.window_size,
            self.memory_monitor,
        )
        windows_list = list(windows_generator)
        windows_3d_array = np.stack(windows_list)
        logger.info(f"Finished windowing, resulting: {windows_3d_array.shape=}")
        return windows_3d_array

    def add_fourier_features(
        self,
        timeseries_windows: np.ndarray,
    ) -> np.ndarray:
        """
        Adds Fourier features to the timeseries windows.

        Parameters:
        - timeseries_windows (np.ndarray): The timeseries windows to add features to.

        Returns:
        - np.ndarray: The timeseries windows with added Fourier features.
        """
        stats = [
            "mean_magnitude",
            "median_magnitude",
            "std_magnitude",
            "mean_phase",
            "median_phase",
            "std_phase",
            "ts_sum_values",
            "ts_median",
            "ts_mean",
            "ts_length",
            "ts_std",
            "ts_var",
            "ts_rms",
            "ts_max",
            "ts_abs_max",
            "ts_min",
        ]
        num_coefficients = self.config.get("fourier_params", {}).get(
            "num_coefficients", 25
        )
        # coefficients cannot be more than the number of time steps
        num_coefficients = min(num_coefficients, timeseries_windows.shape[1])

        label_num = self.config.get("fourier_params", {}).get("label_num", 1)
        size_for_one_ts_feature = num_coefficients + len(stats) + 1
        discrete_windows = np.zeros(
            (
                timeseries_windows.shape[0],
                timeseries_windows.shape[1],
                size_for_one_ts_feature * timeseries_windows.shape[2],
            )
        )
        discrete_windows[:, :, -1] = self.config.get("channel_label", 0)

        discrete_cols = []
        for i in range(timeseries_windows.shape[0]):
            for j in range(timeseries_windows.shape[2]):
                y = timeseries_windows[i, :, j]

                N = len(y)
                coefficients = np.fft.fft(y)[:num_coefficients] / N
                real_coefficients = np.real(coefficients) + 1

                magnitude = np.abs(coefficients)
                phase = np.angle(coefficients)
                statistics = {
                    "mean_magnitude": np.mean(magnitude),
                    "median_magnitude": np.median(magnitude),
                    "std_magnitude": np.std(magnitude),
                    "mean_phase": np.mean(phase),
                    "median_phase": np.median(phase),
                    "std_phase": np.std(phase),
                    # basic time series statistics
                    "ts_sum_values": np.sum(y),
                    "ts_median": np.median(y),
                    "ts_mean": np.mean(y),
                    "ts_length": len(y),
                    "ts_std": np.std(y),
                    "ts_var": np.var(y),
                    "ts_rms": np.sqrt(np.mean(y**2)),
                    "ts_max": np.max(y),
                    "ts_abs_max": np.max(np.abs(y)),
                    "ts_min": np.min(y),
                }

                # add coefficients
                for k in range(timeseries_windows.shape[1]):
                    # because we fill in per timeseries feature
                    discrete_windows[
                        i,
                        k,
                        j * size_for_one_ts_feature : (j + 1)
                        * size_for_one_ts_feature,
                    ] = np.concatenate(
                        [
                            np.array(real_coefficients),
                            np.array(list(statistics.values())),
                            np.array([label_num]),
                        ]
                    )
                discrete_cols = [
                    f"fourier_{i}" for i in range(len(real_coefficients))
                ] + list(statistics.keys())
                discrete_cols.append("channel_label")

        self.discrete_cols = list(self.discrete_cols) + list(discrete_cols)
        logger.warning(f"Added fourier features: {discrete_cols}")

        return discrete_windows

    def process_groups(
        self,
        df: pd.DataFrame,
    ) -> int:
        """
        Processes grouped df by applying a sliding window transformation, one-hot encoding discrete columns,
        and then concatenating windowed df.

        Parameters:
        - df (pd.DataFrame): DataFrame to process.

        Returns:
        - n_samples: number of windows
        """
        logger.info("Started windowing")

        total_windows = self.reshape_for_timeseries(df)

        # Break into separate windows: timeseries, continuous, discrete
        # Number of columns in each category
        logger.info("Dividing into window types")

        # Memory check before window slicing
        if self.memory_monitor:
            self.memory_monitor.check_memory_limits("before_window_slicing")

        num_total_cols = total_windows.shape[2]
        num_timestamps = len(self.timestamps_col)
        num_timeseries = len(self.timeseries_cols)
        num_continuous = len(self.continuous_cols)
        num_original_text = len(self.text_cols)
        num_discrete = (
            len(self.final_discrete_cols)
            if not self.already_saved_discrete
            else 0
        )
        # TODO @bekzat/raimi - why is this getting changed to not include self.group_labels_cols????
        num_original_discrete = (
            len(self.original_discrete_cols)
            if not self.already_saved_discrete
            else 0
        )
        # num_original_discrete = (
        #     len(self.discrete_cols) + len(self.group_labels_cols)
        #     if self.use_label_col_as_discrete_metadata
        #     else len(self.discrete_cols)
        # )
        logger.info(
            f"num_timeseries: {num_timeseries}, num_continuous: {num_continuous}, num_discrete: {num_discrete}, num_original_discrete: {num_original_discrete}"
        )
        logger.info(f"total_windows.shape: {total_windows.shape}")

        assert (
            num_timestamps
            + num_timeseries
            + num_continuous
            + num_original_text
            + num_discrete
            + num_original_discrete
            == num_total_cols
        ), (
            "Number of columns in total windows do not match the sum of the columns in each category"
        )

        # Calculate cumulative indices for slicing
        start_idx_timestamps = 0
        start_idx_timeseries = num_timestamps
        start_idx_continuous = start_idx_timeseries + num_timeseries
        start_idx_original_text = start_idx_continuous + num_continuous
        start_idx_discrete = start_idx_original_text + num_original_text
        start_idx_original_discrete = start_idx_discrete + num_discrete
        end_idx = num_total_cols

        # Perform the slicing
        self.windows_data_dict["timestamp"]["windows"] = total_windows[
            :, :, start_idx_timestamps:start_idx_timeseries
        ]
        self.windows_data_dict["timeseries"]["windows"] = total_windows[
            :, :, start_idx_timeseries:start_idx_continuous
        ].astype(np.float32)
        self.windows_data_dict["continuous"]["windows"] = total_windows[
            :, :, start_idx_continuous:start_idx_original_text
        ].astype(np.float32)
        self.windows_data_dict["original_text"]["windows"] = total_windows[
            :, :, start_idx_original_text:start_idx_discrete
        ]
        self.windows_data_dict["discrete"]["windows"] = total_windows[
            :, :, start_idx_discrete:start_idx_original_discrete
        ].astype(np.float32)
        self.windows_data_dict["original_discrete"]["windows"] = total_windows[
            :, :, start_idx_original_discrete:end_idx
        ]
        del total_windows
        gc.collect()

        if self.config.get("add_fourier", False):
            discrete_windows_new = self.add_fourier_features(
                timeseries_windows=self.windows_data_dict["timeseries"][
                    "windows"
                ]
            )
            self.windows_data_dict["discrete"]["windows"] = np.concatenate(
                (
                    self.windows_data_dict["discrete"]["windows"],
                    discrete_windows_new,
                ),
                axis=2,
            )

            # Memory check after Fourier features
            if self.memory_monitor:
                self.memory_monitor.check_memory_limits(
                    "after_fourier_features"
                )

        for k in self.windows_data_dict.keys():
            # Don't log about the already saved discrete/original_discrete windows
            if k in ["discrete", "original_discrete"] and (
                self.already_saved_discrete
            ):
                continue
            logger.info(
                f"{k}.shape = {self.windows_data_dict[k]['windows'].shape}"
            )

        # Memory check after all window processing
        if self.memory_monitor:
            self.memory_monitor.check_memory_limits(
                "after_all_window_processing"
            )
            memory_stats = self.memory_monitor.get_current_memory_usage()
            logger.info(
                f"All windows processed. Peak memory: {memory_stats['peak_memory_mb']:.1f}MB, "
                f"Current: {memory_stats['process_memory_mb']:.1f}MB"
            )

        assert (
            self.windows_data_dict["timeseries"]["windows"].shape[0]
            == self.windows_data_dict["continuous"]["windows"].shape[0]
            == self.windows_data_dict["discrete"]["windows"].shape[0]
        ), "Number of windows are not equal"
        assert (
            self.windows_data_dict["timeseries"]["windows"].shape[1]
            == self.windows_data_dict["continuous"]["windows"].shape[1]
            == self.windows_data_dict["discrete"]["windows"].shape[1]
        ), "Window sizes are not equal"
        assert self.windows_data_dict["timeseries"]["windows"].shape[2] == len(
            self.timeseries_cols
        ), "Time series features not equal"
        assert self.windows_data_dict["continuous"]["windows"].shape[2] == len(
            self.continuous_cols
        ), "Continuous features not equal"

        logger.info("Finished processing groups")
        return self.windows_data_dict["timeseries"]["windows"].shape[0]

    def save_discrete_windows(self, windows: np.ndarray):
        """Save discrete windows for train, val, and test splits.

        Parameters:
            windows (np.ndarray): Windows to split and save
        """
        split_windows_generator = self.split_windows(windows)
        for data_type, windows_part in zip(
            ["train", "val", "test"], split_windows_generator
        ):
            data_filename_discrete = f"{data_type}_{self.windows_filenames_dict['discrete']['window_filename']}.npy"
            logger.info(
                f"saved: {data_filename_discrete}: {windows_part[:, : len(self.final_discrete_cols)].shape}"
            )
            np.save(
                os.path.join(self.output_path, data_filename_discrete),
                windows_part[:, : len(self.final_discrete_cols)].astype(
                    np.float32
                ),
            )
            data_filename_original = f"{data_type}_{self.windows_filenames_dict['original_discrete']['window_filename']}.npy"
            logger.info(
                f"saved: {data_filename_original}: {windows_part[:, len(self.final_discrete_cols) :].shape}"
            )
            np.save(
                os.path.join(self.output_path, data_filename_original),
                windows_part[:, len(self.final_discrete_cols) :],
            )

        # save the discrete cols to discrete_windows_columns.json
        with open(
            os.path.join(self.output_path, "discrete_windows_columns.json"), "w"
        ) as file:
            json.dump(self.final_discrete_cols, file)

    def encode_only_group_discrete(
        self,
        df: pd.DataFrame,
        saved_encoders: Dict[str, Union[OneHotEncoder, EmbeddingEncoder]] = {},
        save_files_on: bool = True,
    ) -> None:
        """
        Encode discrete and save windows if only group labels are provided
        as discrete metadata.
        Parameters:
            - df (pd.DataFrame): DataFrame containing one row per group, with group cols
        """
        encoded_df = self.encode_discrete(
            df, saved_encoders=saved_encoders, save_files_on=save_files_on
        )
        encoded_array = encoded_df.to_numpy()

        del encoded_df
        gc.collect()

        assert (
            len(self.final_discrete_cols) + len(self.original_discrete_cols)
            == encoded_array.shape[1]
        )
        if save_files_on:
            self.save_discrete_windows(encoded_array)
        else:
            # TODO: bekzat: modify process_groups to skip changing this if df was passed as arg to process_data
            self.windows_data_dict["discrete"]["windows"] = encoded_array[
                :, : len(self.final_discrete_cols)
            ]
            self.windows_data_dict["original_discrete"]["windows"] = (
                encoded_array[:, len(self.final_discrete_cols) :]
            )

        self.num_already_saved_discrete_conditions = len(
            self.final_discrete_cols
        )
        # Set use_label_col_as_discrete_metadata to False to make number of discrete conditions zero
        # after windowing
        self.use_label_col_as_discrete_metadata = False

        self.already_saved_discrete = True

    def check_if_time_invariant_discrete_windows(
        self, df: pd.DataFrame
    ) -> bool:
        """
        Check if for each window the discrete cols have only 1 unique value (are time-invariant).

        Parameters:
            - df (pd.DataFrame): DataFrame before encoding discrete columns

        Returns:
            - bool: True if all windows have constant discrete values
        """
        if self.already_saved_discrete:
            return False

        discrete_cols_to_consider = self.discrete_cols + (
            self.group_labels_cols
            if self.use_label_col_as_discrete_metadata
            else []
        )

        if not discrete_cols_to_consider:
            return False

        discrete_array_2d = df[discrete_cols_to_consider].to_numpy()

        # Check each window for uniform values
        for start_indices in self.window_start_row_indices.values():
            for start_idx in start_indices:
                end_idx = start_idx + self.window_size
                window_slice = discrete_array_2d[start_idx:end_idx, :]

                # If any window is not uniform, return False
                if not np.all(window_slice == window_slice[0, :]):
                    return False

        return True

    # TODO: we should flatten the column that is time invariant, but keep the others in the data to be encoded
    def flatten_discrete_before_process_groups(
        self, df: pd.DataFrame, save_files_on: bool = True
    ) -> pd.DataFrame:
        """
        Flatten and encode discrete columns when all windows have constant values.

        Parameters:
            - df (pd.DataFrame): DataFrame before encoding discrete columns
            - save_files_on (bool): If True, saves the discrete conditions as numpy files

        Returns:
            - pd.DataFrame: DataFrame with discrete columns removed if processed
        """
        discrete_cols_to_consider = self.discrete_cols + (
            self.group_labels_cols
            if self.use_label_col_as_discrete_metadata
            else []
        )

        discrete_array_2d = df[discrete_cols_to_consider].to_numpy()

        # Get just the first row indices for each window
        first_row_indices = [
            start_idx
            for start_indices in self.window_start_row_indices.values()
            for start_idx in start_indices
        ]

        # Take only the first row of each window directly
        flattened_discrete = discrete_array_2d[first_row_indices]

        # Create DataFrame for flattened discrete data
        flattened_df = pd.DataFrame(
            flattened_discrete, columns=discrete_cols_to_consider
        )

        # Encode the flattened discrete data
        encoded_df = self.encode_discrete(
            flattened_df, saved_encoders={}, save_files_on=save_files_on
        )

        encoded_array = encoded_df.to_numpy()

        if save_files_on:
            self.save_discrete_windows(encoded_array)
            # time invariant case the original_discrete_colnames becomes empty (see end of this function)
            with open(
                os.path.join(self.output_path, "colnames.json"), "w"
            ) as f:
                json.dump(
                    {
                        "original_discrete_colnames": self.original_discrete_cols,
                        "discrete_colnames": self.final_discrete_cols,
                    },
                    f,
                )
        else:
            self.windows_data_dict["discrete"]["windows"] = encoded_array[
                :, : len(self.final_discrete_cols)
            ]
            self.windows_data_dict["original_discrete"]["windows"] = (
                encoded_array[:, len(self.final_discrete_cols) :]
            )

        # Update state
        self.num_already_saved_discrete_conditions = len(
            self.final_discrete_cols
        )
        self.num_already_saved_original_discrete_cols = len(
            self.original_discrete_cols
        )
        self.use_label_col_as_discrete_metadata = False
        self.already_saved_discrete = True

        # Remove discrete cols from df
        df.drop(columns=discrete_cols_to_consider, inplace=True)
        self.final_discrete_cols = []
        self.original_discrete_cols = []
        self.discrete_cols = []

        return df

    def _process_discrete_columns(
        self,
        df: pd.DataFrame,
        saved_encoders: Dict[str, Union[OneHotEncoder, EmbeddingEncoder]],
        save_files_on: bool,
    ) -> pd.DataFrame:
        """
        Process discrete columns in the DataFrame based on configuration settings.

        Args:
            df: Input DataFrame containing discrete and/or group label columns
            saved_encoders: Dictionary of pre-trained encoders for inference
            save_files_on: Whether to save encoded data to files

        Returns:
            DataFrame with processed discrete columns
        """
        if self.discrete_cols or (
            self.group_labels_cols and self.use_label_col_as_discrete_metadata
        ):
            if (
                len(self.discrete_cols) == 0
                and len(self.group_labels_cols) > 0
                and save_files_on
                and not self.scale_by_metadata
            ):
                logger.info(
                    "No discrete cols specified, using group labels as discrete metadata"
                )
                self.encode_only_group_discrete(
                    df.loc[
                        list(
                            chain.from_iterable(
                                self.window_start_row_indices.values()
                            )
                        ),
                        self.group_labels_cols,
                    ].reset_index(drop=True),
                    saved_encoders=saved_encoders,
                    save_files_on=save_files_on,
                )
                df = df.drop(columns=self.group_labels_cols)

            elif (
                self.check_if_time_invariant_discrete_windows(df)
                and not saved_encoders
                and not self.scale_by_metadata
            ):  # only do this if we are not doing inference (using saved encoders)
                df = self.flatten_discrete_before_process_groups(
                    df, save_files_on=save_files_on
                )
            else:
                df = self.encode_discrete(
                    df,
                    saved_encoders=saved_encoders,
                    save_files_on=save_files_on,
                )
        else:
            self.save_discrete_encoders(
                saved_encoders={}
            )  # no discrete conditions

        return df

    def _process_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process text column by transforming it into embeddings and updating the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing the text column

        Returns:
            pd.DataFrame: DataFrame with text column replaced by embeddings
        """
        # Transform text column to embeddings and replace it in the DataFrame
        text_df = df[self.text_cols]  # View of original DataFrame
        transformed_text = self.text_embedder.transform(text_df)

        # Check for intersection between existing continuous cols and text embedding cols
        intersection = set(self.continuous_cols).intersection(
            transformed_text.columns
        )
        if intersection:
            raise ValueError(
                f"Text embedding column names {intersection} conflict with existing continuous columns"
            )

        self.continuous_cols = list(self.continuous_cols) + list(
            transformed_text.columns
        )
        self.continuous_scalers_info.update(
            {col: {"scaler_type": "none"} for col in transformed_text.columns}  # pyright: ignore
        )
        df = pd.concat([df, transformed_text], axis=1)

        # Clean up
        del text_df
        del transformed_text
        gc.collect()
        logger.info("Finished processing text column")

        return df

    def process_data(
        self,
        input_df: Optional[pd.DataFrame] = None,
        saved_scalers: Dict[str, Any] = {},
        saved_encoders: Dict[str, Union[OneHotEncoder, EmbeddingEncoder]] = {},
        save_files_on: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Main method to orchestrate the loading, processing, and saving of data based on configured parameters.

        Parameters:
        - df (pd.DataFrame): Loaded data.
        - saved_scalers (Dict[str, Any]): Saved scalers from training preprocessing.
        - saved_encoders (Dict[str, Union[OneHotEncoder, EmbeddingEncoder]]): Saved encoders from training preprocessing.
        - save_files_on (bool): Whether to save the processed data and metadata.
        """

        if self.memory_monitor:
            self.memory_monitor.reset_metrics()
            overall_start_ts = time.perf_counter()
            self.memory_monitor.check_memory_limits("process_data_start")

        # Only for API: otherwise for 2d discrete conditions we won't be able to use
        # original discrete windows, since they will get dropped after saving 2d discrete
        # conditions
        if saved_scalers and not save_files_on:
            self.scale_by_metadata = True

        self.validate_scalers()
        df = self.load_and_prep_rawdata(
            save_labels_description=save_files_on, df=input_df
        )
        assert df is not None, (
            "DataFrame is None after attempting to load data."
        )

        if self.use_custom_split:
            logger.warning("Using BYOS data split")
            self.window_start_row_indices = find_window_start_idxs(
                self.window_size, self.stride, self.group_len_dict
            )
            self.split_inds_dict = find_group_train_val_test_split_inds_byos(
                df.loc[  # pyright: ignore
                    list(
                        chain.from_iterable(
                            self.window_start_row_indices.values()
                        )
                    ),
                    "split",
                ]
                .reset_index(drop=True)
                .values
            )
            df = df.drop(columns=["split"])

        else:
            logger.warning("Using stratified data split")
            self.split_inds_dict, self.window_start_row_indices = (
                find_group_train_val_test_split_inds_stratified(
                    self.group_len_dict,
                    self.train_val_split,
                    self.shuffle,
                    self.window_size,
                    self.stride,
                    self.val_stride,
                    self.test_stride,
                    self.window_from_beginning,
                    self.shuffle_partitions_len,
                    allow_overlap=self.allow_overlap,
                )
            )
        if save_files_on:
            validate_split_indices(self.split_inds_dict)

        if self.group_labels_cols:
            check_start_indices(
                df[self.group_labels_cols],
                self.window_start_row_indices,
                self.window_size,
            )
        df_row_inds_dataset_types = find_dataset_type_df_indices(
            self.split_inds_dict,
            self.window_size,
            self.window_start_row_indices,
        )

        if self.memory_monitor:
            self.memory_monitor.check_memory_limits(
                "after_find_dataset_type_df_indices"
            )
        self.validate_discrete_cols_for_scale_by_metadata(
            df, df_row_inds_dataset_types
        )

        if len(self.text_cols) > 0:
            df = self._process_text_column(df)
            if self.memory_monitor:
                self.memory_monitor.check_memory_limits("after_text_processing")

        # Validate scalers only after adding all necessary cols to the data
        self.validate_scalers()

        group_label_to_split_idx_dict = create_group_label_to_split_idx_dict(
            self.window_start_row_indices,
            df_row_inds_dataset_types,
            allow_overlap=self.allow_overlap,
        )

        # Keep this order since the windows are separated into types in this order
        logger.info("Reordering columns")
        if self.memory_monitor:
            self.memory_monitor.check_memory_limits(
                "before_dataframe_reordering"
            )

        df = df[
            list(self.timestamps_col)
            + (
                list(self.group_labels_cols)
                if self.use_label_col_as_discrete_metadata
                else []
            )
            + list(self.timeseries_cols)
            + list(self.continuous_cols)
            + list(self.text_cols)
            + list(self.discrete_cols)
        ]
        assert isinstance(df, pd.DataFrame)

        df = self._process_discrete_columns(df, saved_encoders, save_files_on)

        # TODO - make this helper function
        if len(saved_scalers) == 0:
            original_discrete_for_scalers = (
                self.original_discrete_cols if self.scale_by_metadata else []
            )
            saved_scalers["timeseries"] = fit_scaler(
                df.loc[
                    df_row_inds_dataset_types["train"],
                    list(self.timeseries_cols) + original_discrete_for_scalers,
                ],
                self.timeseries_scalers_info,
            )
            saved_scalers["continuous"] = fit_scaler(
                df.loc[
                    df_row_inds_dataset_types["train"],
                    list(self.continuous_cols) + original_discrete_for_scalers,
                ],
                self.continuous_scalers_info,
            )

        self.n_samples = self.process_groups(df)

        self.apply_saved_scalers(saved_scalers)
        self.windows_data_dict["timestamp_conditions"]["windows"] = (
            self.windows_data_dict["continuous"]["windows"][
                :, :, -len(self.timestamps_features_colnames) :
            ]
        )

        if save_files_on:
            self.save_scalers(saved_scalers)
            self.save_constraint_channel_minmax_values()
            self.save_windows()
            save_colnames_to_json(
                self.output_path,
                self.timestamps_col,
                self.timeseries_cols,
                self.continuous_cols,
                self.final_discrete_cols,
                self.original_discrete_cols,
                self.text_cols,
            )

            # save the group label to idx mapping to json
            with open(
                os.path.join(
                    self.output_path, "group_label_to_split_idx_dict.pkl"
                ),
                "wb",
            ) as file:
                pickle.dump(group_label_to_split_idx_dict, file)

            # save the column names to json
            with open(
                os.path.join(
                    self.output_path, "timeseries_windows_columns.json"
                ),
                "w",
            ) as file:
                json.dump(self.timeseries_cols, file)
            with open(
                os.path.join(
                    self.output_path, "continuous_windows_columns.json"
                ),
                "w",
            ) as file:
                json.dump(self.continuous_cols, file)
            if not self.already_saved_discrete:
                with open(
                    os.path.join(
                        self.output_path, "discrete_windows_columns.json"
                    ),
                    "w",
                ) as file:
                    json.dump(self.final_discrete_cols, file)
            with open(
                os.path.join(self.output_path, "discrete_encoders_info.json"),
                "w",
            ) as file:
                json.dump(self.metadata_dict["discrete_encoders_info"], file)

            # dump metadata to json
            with open(
                os.path.join(self.output_path, "metadata_dict.json"), "w"
            ) as file:
                json.dump(self.metadata_dict, file)

            logger.info("Metadata info dictionary saved to data_dict.json")

            logger.info(f"all data saved to {self.output_path}")

            # pre-fill/update the training config
            if self.config.get("prefill_training_configs", False):
                self.prefill_training_config()

            if self.memory_monitor:
                total_runtime_s = time.perf_counter() - overall_start_ts
                self.memory_monitor.log_memory_summary("preprocessing_complete")
                self.memory_monitor.log_overhead_summary(total_runtime_s)

            return self.log_action_update_for_training_config()

    def get_dataset_config_to_update(self) -> Dict[str, Any]:
        """
        Gets the dataset config to update for synthesis/forecasting training configs.
        """
        dataset_config_to_update = {}
        dataset_config_to_update["time_series_length"] = self.window_size
        dataset_config_to_update["num_channels"] = len(self.timeseries_cols)

        if not self.already_saved_discrete:
            dataset_config_to_update["num_discrete_conditions"] = len(
                self.final_discrete_cols
            )
        else:
            dataset_config_to_update["num_discrete_conditions"] = (
                self.num_already_saved_discrete_conditions
            )
        dataset_config_to_update["num_continuous_labels"] = len(
            self.continuous_cols
        )
        dataset_config_to_update["num_timestamp_labels"] = len(
            self.timestamps_features_list
        )
        return dataset_config_to_update

    def prefill_training_config(self) -> None:
        """
        Prefills the training config with the data from the preprocessing.
        """
        dataset_config_to_update = self.get_dataset_config_to_update()
        dataset_name = self.config["filename"].split("/")[0]
        _ = generate_default_train_config(
            task="synthesis",
            dataset_name=dataset_name,
            time_series_length=dataset_config_to_update["time_series_length"],
            num_channels=dataset_config_to_update["num_channels"],
            num_discrete_conditions=dataset_config_to_update[
                "num_discrete_conditions"
            ],
            num_continuous_labels=dataset_config_to_update[
                "num_continuous_labels"
            ],
            num_timestamp_labels=dataset_config_to_update[
                "num_timestamp_labels"
            ],
            save_to_examples_dir=True,
        )  # config['prefill_training_configs'] must be true if we are prefilling training configs
        _ = generate_default_train_config(
            task="forecast",
            dataset_name=dataset_name,
            time_series_length=dataset_config_to_update["time_series_length"],
            num_channels=dataset_config_to_update["num_channels"],
            num_discrete_conditions=dataset_config_to_update[
                "num_discrete_conditions"
            ],
            num_continuous_labels=dataset_config_to_update[
                "num_continuous_labels"
            ],
            num_timestamp_labels=dataset_config_to_update[
                "num_timestamp_labels"
            ],
            save_to_examples_dir=True,
        )  # config['prefill_training_configs'] must be true if we are prefilling training configs

    def get_concatted_timeseries(self) -> np.ndarray:
        """
        Concatenates timeseries data from train, val, and test sets.

        Returns:
        - np.ndarray: Concatenated timeseries data.
        """
        timeseries_train = self.windows_data_dict["timeseries"]["train_windows"]
        timeseries_val = self.windows_data_dict["timeseries"]["val_windows"]
        timeseries_test = self.windows_data_dict["timeseries"]["test_windows"]

        concatenated_timeseries = np.concatenate(
            (timeseries_train, timeseries_val, timeseries_test), axis=0
        )

        return concatenated_timeseries

    def log_action_update_for_training_config(self) -> Dict[str, Any]:
        """
        Logs and saves into json file the action to update the trining configs for synthesis and forecast.
        """
        dataset_config_to_update = self.get_dataset_config_to_update()
        # Save dataset_config_to_update to json
        with open(
            os.path.join(self.output_path, "dataset_config_to_update.json"), "w"
        ) as file:
            json.dump(dataset_config_to_update, file)

        logger.warning(
            f"Please update the dataset_config parameter in the config.yaml for this dataset with:\n{dataset_config_to_update}"
        )
        if self.add_lag_data:
            logger.warning(
                "If you wish to run preprocessing again on the new generated data with the lag columns or want to use APIs, please do the following:."
            )
            logger.warning(
                f'1. Update the "continuous" "cols" to include the new lag columns: {json.dumps(self.continuous_cols[-len(self.timeseries_cols) :])}'
            )
            logger.warning(
                '2. Update the "add_lag_data" to false in the config.yaml for this dataset if you want to run preprocessing again on the new data.'
            )
            logger.warning(
                f"""3. Update the filename to include the new lag data: {
                    self.config["filename"]
                    .replace(".csv", LAG_DATA_FILENAME_SUFFIX)
                    .replace(".parquet", LAG_DATA_FILENAME_SUFFIX)
                }"""
            )
        return dataset_config_to_update

    def split_windows(
        self, windows: np.ndarray
    ) -> Generator[np.ndarray, None, None]:
        """
        Splits the data into training, validation, and test sets based on predefined ratios.

        Parameters:
        - windows (np.ndarray): The windows to split.

        Yields:
        - np.ndarray: Train, validation, and test data.
        """
        yield windows[self.split_inds_dict["train"]]
        yield windows[self.split_inds_dict["val"]]
        yield windows[self.split_inds_dict["test"]]

    def apply_saved_scalers(
        self, scalers: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> None:
        """
        Apply scalers that were trained on train dataset on the train/val/test
        widnows of timeseries and continuous.
        Parameters:
        - scalers (Dict[str, Dict[str, List[Dict[str, Any]]]]): dict containing timeseries and continuous scalers to apply.
            >>> scalers = {
            ...     "temperature": [
            ...         {"tuple": {"location": "NYC", "season": "summer"}, "scaler": Scaler()},
            ...         {"tuple": {"location": "NYC", "season": "winter"}, "scaler": Scaler()},
            ...         ...
            ...     ],
            ...     "pressure": [{"scaler": Scaler()}]
            >>> }
        """
        # Apply existing scalers to the original windows
        for window_type in ["timeseries", "continuous"]:
            windows = self.windows_data_dict[window_type]["windows"]
            if windows.size == 0:
                continue
            # Pass full data for timeseries/continuous for apply the scaler
            self.windows_data_dict[window_type]["windows"] = (
                transform_using_scaler(
                    windows=windows,
                    timeseries_or_continuous=window_type,
                    original_discrete_windows=self.windows_data_dict[
                        "original_discrete"
                    ]["windows"],
                    scalers=scalers[window_type],
                    col_names=(
                        self.timeseries_cols
                        if window_type == "timeseries"
                        else self.continuous_cols
                    ),
                    original_discrete_colnames=self.original_discrete_cols,
                    transpose_timeseries=False,
                )
            )

    def save_windows(self) -> None:
        """
        Scales the train, val, and test sets for each window type, fitting on the train set and applying the same transformation to others.
        """
        window_types = [
            "timeseries",
            "continuous",
            "discrete",
            "timestamp",
            "original_discrete",
            "original_text",
        ]
        for window_type in window_types:
            if window_type in ["discrete", "original_discrete"] and (
                self.already_saved_discrete
            ):
                continue
            logger.info(f"Saving windows for: {window_type}")
            window_generator = self.split_windows(
                self.windows_data_dict[window_type].pop("windows")
            )
            for data_type, windows in zip(
                ["train", "val", "test"], window_generator
            ):
                if window_type == "timeseries":
                    windows = np.transpose(windows, (0, 2, 1))

                data_filename = f"{data_type}_{self.windows_filenames_dict[window_type]['window_filename']}.npy"
                np.save(
                    os.path.join(self.output_path, data_filename),
                    windows,
                )
                logger.info(f"saved: {data_filename}: {windows.shape}")
                if window_type == "continuous":
                    # We need to save scaled timestamps features in a separate file as well as in continuous conditions file
                    # since it is used in forecast. They are the last columns in the continuous windows data.
                    if len(self.timestamps_col) > 0:
                        windows = windows[
                            :, :, -len(self.timestamps_features_colnames) :
                        ]

                    else:
                        windows = np.empty(
                            (windows.shape[0], windows.shape[1], 0)
                        )

                    data_filename = f"{data_type}_{self.windows_filenames_dict['timestamp_conditions']['window_filename']}.npy"

                    np.save(
                        os.path.join(self.output_path, data_filename),
                        windows,
                    )
                    logger.info(f"saved: {data_filename}: {windows.shape}")

    def save_constraint_channel_minmax_values(self) -> None:
        """
        Save the min and max values from train and val sets into a JSON file.
        Used in generate_synthetic_data with constraints.
        For windows of shape (B, W, C), computes one min and max value per channel
        by operating across all batches and timesteps.
        The format is {channel_name: {"min": min_value, "max": max_value}, ...}
        """
        window_generator = self.split_windows(
            self.windows_data_dict["timeseries"]["windows"]
        )

        # Get train and val windows
        train_windows, val_windows = (
            next(window_generator),
            next(window_generator),
        )

        # Concatenate train and val along batch dimension
        windows = np.concatenate(
            [train_windows, val_windows], axis=0
        )  # Shape: (B, W, C)

        # Compute min/max across all batches and timesteps for each channel
        min_values = np.min(windows, axis=(0, 1))  # Shape: (C,)
        max_values = np.max(windows, axis=(0, 1))  # Shape: (C,)

        # Create dictionary mapping channel names to their min/max values
        channel_minmax = {}
        for i, channel_name in enumerate(self.timeseries_cols):
            channel_minmax[channel_name] = {
                "min": float(min_values[i]),
                "max": float(max_values[i]),
            }

        # Save as JSON
        minmax_filename = "constraints_channel_minmax_values.json"
        with open(os.path.join(self.output_path, minmax_filename), "w") as f:
            json.dump(channel_minmax, f)

        logger.info(f"saved: {minmax_filename}")

    def save_scalers(
        self,
        scalers_to_save: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> None:
        """
        Save the timeseries and continuous scalers.
        Parameters:
        -----------
        scalers_to_save: Dict[str, Dict[str, List[Dict[str, Any]]]]
        Examples:
        --------
        >>> scalers_to_save = {
        ...     "timeseries": {
        ...        "temperature": [
        ...            {"tuple": {"location": "NYC", "season": "summer"}, "scaler": Scaler()},
        ...            {"tuple": {"location": "NYC", "season": "winter"}, "scaler": Scaler()},
        ...            ...
        ...        ],
        ...        "pressure": [{"tuple": None, "scaler": Scaler()}]
        ...      },
        ...     "continuous": {
        ...        "wind_speed": [
        ...            {"tuple": None, "scaler": Scaler()}
        ...        ],
        ...      }
        >>>     }
        """
        for window_type in ["timeseries", "continuous"]:
            scaler_filename = SCALER_FILENAMES[window_type]
            # Save the fitted scalers
            with open(
                os.path.join(self.output_path, scaler_filename), "wb"
            ) as file:
                pickle.dump(scalers_to_save[window_type], file)

            logger.info(f"Saved {window_type} scalers to {scaler_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="config file path", required=True
    )

    config_path = parser.parse_args().config

    preprocessor = DataPreprocessor(config_source=config_path)
    preprocessor.process_data()
