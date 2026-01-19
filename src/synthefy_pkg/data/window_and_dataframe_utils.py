import asyncio
import json
import os
import pickle
import re
from typing import Any, Dict, List, Literal, Tuple

import einops
import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger

from synthefy_pkg.app.data_models import WindowFilters
from synthefy_pkg.data.synthefy_dataset import SynthefyDataset
from synthefy_pkg.preprocessing.base_text_embedder import (
    EMBEDDED_COL_NAMES_PREFIX,
)
from synthefy_pkg.preprocessing.preprocess import (
    LAG_COL_FORMAT,
    TIMESTAMPS_FEATURES_PREFIX,
)
from synthefy_pkg.utils.scaling_utils import unscale_windows_dict


async def load_dataset_files(
    dataset_name: str,
    split: Literal["train", "val", "test"],
    timeseries: np.ndarray | None = None,
    discrete_original: np.ndarray | None = None,
    continuous: np.ndarray | None = None,
    timestamps_original: np.ndarray | None = None,
    text: np.ndarray | None = None,
    group_label_to_split_idx_dict: Dict[str, Dict[int, Tuple]] | None = None,
    group_label_cols: List[str] | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Dict[int, Any]] | None,
    List[str] | None,
]:
    """Load and validate all dataset files.

    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split ("train"/"val"/"test")
        timeseries, discrete_original, continuous, timestamps_original, text: Optional pre-loaded arrays

    Returns:
        Tuple of (timeseries, discrete_original, continuous, timestamps_original, text) arrays

    Raises:
        FileNotFoundError: If dataset directory or required files don't exist
    """
    SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
    SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")
    assert (
        SYNTHEFY_DATASETS_BASE is not None and SYNTHEFY_PACKAGE_BASE is not None
    ), "ENV SYNTHEFY_DATASETS_BASE or SYNTHEFY_PACKAGE_BASE is not set"

    # Validate dataset directory exists
    dataset_dir = os.path.join(SYNTHEFY_DATASETS_BASE, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Load window data from files
    data_files = {
        "timeseries": f"{split}_timeseries.npy",  # shape is (B, C, W)
        "discrete": f"{split}_original_discrete_windows.npy",  # shape is (B, W, C)
        "continuous": f"{split}_continuous_conditions.npy",  # shape is (B, W, C)
        "timestamps": f"{split}_timestamps_original.npy",  # shape is (B, W)
        "text": f"{split}_original_text_conditions.npy",  # shape is (B, W, C)
    }

    logger.info(f"Loading dataset files for {dataset_name} split {split}")
    # Load each data file
    if timeseries is None:
        timeseries = np.load(
            os.path.join(
                SYNTHEFY_DATASETS_BASE, dataset_name, data_files["timeseries"]
            )
        )
        assert isinstance(timeseries, np.ndarray)
    if discrete_original is None:
        discrete_original = np.load(
            os.path.join(
                SYNTHEFY_DATASETS_BASE, dataset_name, data_files["discrete"]
            ),
            allow_pickle=True,
        )
        assert isinstance(discrete_original, np.ndarray)
    if continuous is None:
        continuous = np.load(
            os.path.join(
                SYNTHEFY_DATASETS_BASE, dataset_name, data_files["continuous"]
            )
        )
        assert isinstance(continuous, np.ndarray)
    if timestamps_original is None:
        timestamps_original = np.load(
            os.path.join(
                SYNTHEFY_DATASETS_BASE, dataset_name, data_files["timestamps"]
            ),
            allow_pickle=True,
        )
        assert isinstance(timestamps_original, np.ndarray)
    # Load text data if it exists
    if text is None:
        text_file = os.path.join(
            SYNTHEFY_DATASETS_BASE,
            dataset_name,
            data_files["text"],
        )
        if os.path.exists(text_file):
            text = np.load(text_file, allow_pickle=True)
            assert isinstance(text, np.ndarray)
        else:
            text = np.zeros((0, 0, 0))  # for backwards compatibility

    if group_label_to_split_idx_dict is None:
        # read the preprocess_config.json file and see if use_label_col_as_discrete_metadata is True
        # if so, then we don't need to load the group_label_to_split_idx_dict
        # if not, then we need to load the group_label_to_split_idx_dict
        preprocess_config = json.load(
            open(
                os.path.join(
                    SYNTHEFY_PACKAGE_BASE,
                    f"examples/configs/preprocessing_configs/config_{dataset_name}_preprocessing.json",
                )
            )
        )
        use_label_col_as_discrete_metadata = preprocess_config.get(
            "use_label_col_as_discrete_metadata", False
        )
        group_label_to_split_idx_dict_file = os.path.join(
            SYNTHEFY_DATASETS_BASE,
            dataset_name,
            "group_label_to_split_idx_dict.pkl",
        )

        if use_label_col_as_discrete_metadata or not os.path.exists(
            group_label_to_split_idx_dict_file
        ):
            logger.warning(
                f"Group label to split idx dict file {group_label_to_split_idx_dict_file} does not exist. "
                "Cannot add group labels to the dataframe."
            )
            group_label_to_split_idx_dict = None
        else:
            with open(group_label_to_split_idx_dict_file, "rb") as f:
                group_label_to_split_idx_dict = pickle.load(f)

        group_label_cols = preprocess_config.get("group_labels", {}).get(
            "cols", []
        )
    logger.info(
        f"Successfully loaded dataset files for {dataset_name} split {split}"
    )

    return (
        timeseries,
        discrete_original,
        continuous,
        timestamps_original,
        text,
        group_label_to_split_idx_dict,
        group_label_cols,
    )


def filter_continuous_features_by_pattern(
    continuous: np.ndarray,
    continuous_colnames: list[str],
    pattern: str,
    match_type: Literal["prefix", "lag_format"] = "prefix",
    timeseries_colnames: list[str] | None = None,
) -> Tuple[np.ndarray, list[str]]:
    """Filter continuous data and column names based on prefix or lag format matching.

    Args:
        continuous: Continuous data array
        continuous_colnames: List of column names for continuous data
        pattern: Pattern to match against column names
        match_type: Type of matching to perform:
            - "prefix": Match columns that start with pattern
            - "lag_format": Match columns that follow LAG_COL_FORMAT format
                where {col} is a string and {window_size} is an integer
        timeseries_colnames: List of column names for timeseries data -> required when match_type is "lag_format"

    Returns:
        tuple of:
            - Filtered continuous data array with matched features removed
            - Filtered list of column names with matched features removed
    """
    if match_type == "lag_format" and timeseries_colnames is None:
        raise ValueError(
            "timeseries_colnames is required when match_type is 'lag_format'"
        )

    matched_indices = []
    filtered_continuous_colnames = []

    for i, col in enumerate(continuous_colnames):
        is_match = False

        if match_type == "prefix":
            is_match = col.startswith(pattern)
        elif match_type == "lag_format":
            # Extract the base column name and window size using regex based on LAG_COL_FORMAT
            # LAG_COL_FORMAT is "{col}_lag_{window_size}", we need to escape the curly braces
            pattern_str = LAG_COL_FORMAT.replace("{", "\\{").replace("}", "\\}")
            # Replace the placeholders with regex capture groups
            pattern_str = pattern_str.replace("\\{col\\}", "(.+)").replace(
                "\\{window_size\\}", "(\\d+)"
            )
            match = re.match(pattern_str, col)
            if match:
                base_col = match.group(1)
                is_match = (
                    isinstance(timeseries_colnames, list)
                    and base_col in timeseries_colnames
                )

        if is_match:
            matched_indices.append(i)
        else:
            filtered_continuous_colnames.append(col)

    # Remove the matched feature columns from the continuous data
    if len(continuous.shape) > 0 and len(matched_indices) > 0:
        continuous = np.delete(continuous, matched_indices, axis=1)

    return continuous, filtered_continuous_colnames


async def convert_windows_to_dataframe(
    dataset_name: str,
    split: Literal["train", "val", "test"] = "train",
    timeseries: np.ndarray | None = None,
    discrete_original: np.ndarray | None = None,
    continuous: np.ndarray | None = None,
    timestamps_original: np.ndarray | None = None,
    text: np.ndarray | None = None,
    group_label_to_split_idx_dict: Dict[str, Dict[int, Tuple]] | None = None,
    group_label_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Input:
        dataset_name: str, name of the dataset to convert
        split: str - "train"/"val"/"test"
        The following arguments are optional and can be used to avoid loading the data from files
        timeseries: np.ndarray, shape is (B, C, W)
        discrete: np.ndarray, shape is (B, W, C) or (B, W, 0) if no discrete features
        continuous: np.ndarray, shape is (B, W, C) or (B, W, 0) if no continuous features
        timestamps: np.ndarray, shape is (B, W) or (B, 0) if no timestamps ??? TODO - check this
        text: np.ndarray, shape is (B, W, C) or (B, W, 0) if no text features ??? TODO - check this
    Output:
        df: pd.DataFrame, dataframe with the timeseries, discrete, and continuous windows
        the df has a column "window_idx" indicating the data per window.
        filtered_continuous_colnames: List[str], list of continuous column names that were not filtered out

    Note: This assumes the files are already in the correct location on disk.
    """
    SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
    assert SYNTHEFY_DATASETS_BASE is not None, (
        "ENV SYNTHEFY_DATASETS_BASE is not set"
    )

    # Load all dataset files
    (
        timeseries,
        discrete_original,
        continuous,
        timestamps_original,
        text,
        group_label_to_split_idx_dict,
        group_label_cols,
    ) = await load_dataset_files(
        dataset_name=dataset_name,
        split=split,
        timeseries=timeseries,
        discrete_original=discrete_original,
        continuous=continuous,
        timestamps_original=timestamps_original,
        text=text,
        group_label_to_split_idx_dict=group_label_to_split_idx_dict,
        group_label_cols=group_label_cols,
    )

    # Unscale the data
    logger.info("Unscaling continuous and timeseries features.")
    continuous = unscale_windows_dict(
        windows_data_dict={split: continuous},
        window_type="continuous",
        dataset_name=dataset_name,
        original_discrete_windows={split: discrete_original.copy()},
    )[split]

    # unscale the data
    timeseries = unscale_windows_dict(
        windows_data_dict={split: timeseries},
        window_type="timeseries",
        dataset_name=dataset_name,
        original_discrete_windows={split: discrete_original.copy()},
    )[split]

    window_size = timeseries.shape[2]
    # reshape them all to be from (B, C, W) to (B*C, W)
    # flip timeseries since its different shape than metadata originally
    timeseries = einops.rearrange(timeseries, "b c w -> b w c")
    timeseries = einops.rearrange(timeseries, "b w c -> (b w) c")

    # TODO - check if this works for last dim being 0
    if len(discrete_original.shape) == 3:
        discrete_original = einops.rearrange(
            discrete_original, "b w c-> (b w) c"
        )
    else:
        # Reshape discrete to match the first dimension of timeseries
        # Repeat each row window_size times to match shape
        discrete_original = np.repeat(discrete_original, window_size, axis=0)

    continuous = einops.rearrange(continuous, "b w c -> (b w) c")
    if timestamps_original.shape[-1] > 0:
        timestamps_original = einops.rearrange(
            timestamps_original, "b w c -> (b w) c"
        )
    if text.shape[-1] > 0:
        text = einops.rearrange(text, "b w c -> (b w) c")

    colnames = json.load(
        open(
            os.path.join(SYNTHEFY_DATASETS_BASE, dataset_name, "colnames.json")
        )
    )

    # Filter out timestamp features, text embedding features, lag col features
    # from continuous columns (if they exist)

    continuous_colnames = colnames.get("continuous_colnames", [])
    timeseries_colnames = colnames["timeseries_colnames"]
    continuous, filtered_continuous_colnames = (
        filter_continuous_features_by_pattern(
            continuous,
            continuous_colnames,
            pattern=TIMESTAMPS_FEATURES_PREFIX,
            match_type="prefix",
        )
    )
    continuous, filtered_continuous_colnames = (
        filter_continuous_features_by_pattern(
            continuous,
            filtered_continuous_colnames,
            pattern=EMBEDDED_COL_NAMES_PREFIX,
            match_type="prefix",
        )
    )
    continuous, filtered_continuous_colnames = (
        filter_continuous_features_by_pattern(
            continuous,
            filtered_continuous_colnames,
            pattern=LAG_COL_FORMAT,
            match_type="lag_format",
            timeseries_colnames=timeseries_colnames,
        )
    )

    columns = (
        colnames["timeseries_colnames"]
        + colnames.get("original_discrete_colnames", [])
        + filtered_continuous_colnames
        + (
            colnames.get("timestamps_colnames", [])
            if timestamps_original.shape[-1] > 0
            else []
        )
        + colnames.get("original_text_colnames", [])
    )

    data_arrays = np.concatenate(
        [
            arr
            for arr in [
                timeseries,
                discrete_original,
                continuous,
                timestamps_original,
                text,
            ]
            if arr is not None and arr.shape[-1] > 0
        ],
        axis=1,
    )

    df = pd.DataFrame(
        data_arrays,
        columns=columns,
    )

    df["window_idx"] = np.repeat(
        np.arange(int(len(df) / window_size)), window_size
    )
    if (
        group_label_cols is not None
        and group_label_to_split_idx_dict is not None
    ):
        # add each group label col from the tuple values
        for i, col in enumerate(group_label_cols):
            df[col] = df["window_idx"].map(
                lambda x: group_label_to_split_idx_dict[split][x][i]
            )

    return df, filtered_continuous_colnames


async def convert_h5_file_to_dataframe(
    dataset_name: str,
    h5_file_path: str,
    split: Literal["train", "val", "test"] = "test",
    synthetic_or_original: str = "synthetic",
    timestamps_original: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Convert an h5 file to a dataframe.
    """
    if synthetic_or_original not in ["synthetic", "original"]:
        raise ValueError(
            f"Invalid value for synthetic_or_original: {synthetic_or_original}"
        )
    logger.info(f"Loading h5 file: {h5_file_path} for {synthetic_or_original=}")

    with h5py.File(h5_file_path, "r") as f:
        timeseries = np.array(f[f"{synthetic_or_original}_timeseries"])
        discrete_conditions = np.array(f["discrete_conditions"])
        continuous_conditions = np.array(f["continuous_conditions"])

    original_discrete_windows = unscale_windows_dict(
        windows_data_dict={split: discrete_conditions},
        window_type="discrete",
        dataset_name=dataset_name,
    )

    df, _ = await convert_windows_to_dataframe(
        dataset_name=dataset_name,
        split=split,
        timeseries=timeseries,
        discrete_original=original_discrete_windows[split],
        continuous=continuous_conditions,
        timestamps_original=timestamps_original,
    )
    return df


if __name__ == "__main__":
    # examples
    # df, _ = asyncio.run(convert_windows_to_dataframe("rrest"))
    # print(df.head())

    # Convert oura train windows to dataframe
    df, filtered_continuous_colnames = asyncio.run(
        convert_windows_to_dataframe(
            dataset_name="oura",
            split="train",
        )
    )
    from pdb import set_trace; set_trace()
    print(f"Dataframe shape: {df.shape}")
    print(f"Filtered continuous columns: {len(filtered_continuous_colnames)}")
    print(df.head())
