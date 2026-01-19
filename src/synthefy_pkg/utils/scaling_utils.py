import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import einops
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from synthefy_pkg.utils.basic_utils import load_pickle_from_path
from synthefy_pkg.utils.dataset_utils import load_column_names

COMPILE = False

SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")
assert SYNTHEFY_PACKAGE_BASE is not None
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))
SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")

SCALER_TYPES: Dict[str, Any] = {
    "none": FunctionTransformer,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

SCALER_FILENAMES: Dict[str, str] = {
    "timeseries": "timeseries_scalers.pkl",
    "continuous": "continuous_scalers.pkl",
    "discrete": "encoders_dict.pkl",
}


def load_timeseries_scalers(
    dataset_name: str, raise_error_if_not_exists: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    scalers: Dict[str, List[Dict[str, Any]]] = load_pickle_from_path(
        os.path.join(
            str(SYNTHEFY_DATASETS_BASE),
            dataset_name,
            SCALER_FILENAMES["timeseries"],
        ),
        raise_error_if_not_exists=raise_error_if_not_exists,
        default_return_value={},
    )
    return scalers


def load_continuous_scalers(
    dataset_name: str, raise_error_if_not_exists: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    scalers: Dict[str, List[Dict[str, Any]]] = load_pickle_from_path(
        os.path.join(
            str(SYNTHEFY_DATASETS_BASE),
            dataset_name,
            SCALER_FILENAMES["continuous"],
        ),
        raise_error_if_not_exists=raise_error_if_not_exists,
        default_return_value={},
    )
    return scalers


def load_discrete_encoders(
    dataset_name: str, raise_error_if_not_exists: bool = False
) -> Dict[str, Any]:
    encoders: Dict[str, Dict[str, Any]] = load_pickle_from_path(
        os.path.join(
            str(SYNTHEFY_DATASETS_BASE),
            dataset_name,
            SCALER_FILENAMES["discrete"],
        ),
        raise_error_if_not_exists=raise_error_if_not_exists,
        default_return_value={},
    )
    return encoders


def load_timeseries_col_names(
    dataset_name: str, num_channels: int
) -> List[str]:
    """Load timeseries column names from file or use default indices."""
    filepath = os.path.join(
        str(SYNTHEFY_DATASETS_BASE),
        dataset_name,
        "timeseries_windows_columns.json",
    )

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            timeseries_col_names = json.load(f)
        if len(timeseries_col_names) != num_channels:
            raise ValueError(
                "Mismatch between loaded column names and expected number of timeseries."
            )
    else:
        timeseries_col_names = [str(i) for i in range(num_channels)]
    return timeseries_col_names


def inverse_transform_discrete(
    windows_data: np.ndarray,
    dataset_name: Optional[str] = None,
    encoders: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Inverse transform encoded discrete columns.

    Parameters:
    - windows_data (Dict[str, np.ndarray]): Encoded discrete windows, 3D np.array.
    - encoders (Dict[str, Any]): Encoders of struct {encoder_type: encoder}.

    Returns:
    - Tuple[List[str], Dict[str, np.ndarray]]:
        - List of decoded column names.
        - Dictionary of decoded discrete conditions (3D array).
        - Returns empty list and empty np.array if encoders is empty.
    """

    if encoders is None and dataset_name is not None:
        encoders = load_discrete_encoders(dataset_name)

    if not encoders:
        logger.debug("No encoders provided for inverse transformation")
        return [], np.array([])

    final_col_names = []
    start_col_ind = 0
    output_arrays_list = []
    decoded_array = np.array([])
    sum_encoded_cols = 0

    for _, encoder in encoders.items():
        final_col_names.extend(list(encoder.feature_names_in_))
        encoded_cols = encoder.get_feature_names_out()
        end_col_ind = start_col_ind + len(encoded_cols)
        sum_encoded_cols += len(encoded_cols)
        if len(windows_data.shape) == 3:
            sliced_discrete_windows = windows_data[
                :, :, start_col_ind:end_col_ind
            ]
            windows_2d = sliced_discrete_windows.reshape(-1, len(encoded_cols))
            decoded_2d = encoder.inverse_transform(windows_2d)
            # Reshape the 2D array back to the original 3D shape
            decoded_array = decoded_2d.reshape(
                sliced_discrete_windows.shape[0],  # Number of windows
                sliced_discrete_windows.shape[1],  # Window size
                -1,  # Number of original features (before one-hot encoding)
            )
        else:
            windows_2d = windows_data[:, start_col_ind:end_col_ind]
            decoded_array = encoder.inverse_transform(windows_2d)

        output_arrays_list.append(decoded_array)

        start_col_ind = end_col_ind

    assert windows_data.shape[-1] == sum_encoded_cols
    output_array = np.concatenate(output_arrays_list, axis=-1)

    return final_col_names, output_array


def unscale_windows_dict(
    windows_data_dict: Dict[str, np.ndarray],
    window_type: str,
    dataset_name: str,
    original_discrete_windows: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Unscale the given windows data dictionary's each val by window type (timeseries, continuous, discrete).

    Parameters:
        - windows_data_dict (Dict[str, np.ndarray]): Dictionary of windows data, possible keys: 'train', 'test', 'val'.
            expected shapes:
                - timeseries: (num_windows, num_channels, window_size)
                - continuous: (num_windows, window_size, num_continuous_features)
                - discrete: (num_windows, window_size, num_discrete_features)
                - probabilistic timeseries: (num_windows, num_samples, num_channels, window_size)
                    Note: num_samples is the number of samples in the probabilistic forecast; e.g. 1000 samples means 1000 alternate forecasts
        - window_type (str): Type of windows data ('timeseries', 'continuous', 'discrete').
        - dataset_name (str): Name of the dataset (tektronix, opanga etc).
        - original_discrete_windows (Dict[str, np.ndarray]): original discrete windows used for metadata dependent scaling.
            shape (num_windows, window_size, num_discrete_features).
    Returns:
        - windows_data_dict (Dict[str, np.ndarray]): Dictionary of transformed windows data.
    """

    # To support probabilistic forecast, we need to reshape the timeseries data to a 3D array
    # If the shape is 4D (num_windows, num_samples, num_channels, window_size), then it is a probabilistic forecast
    # This transformation only applies to timeseries data because the history (including metadata) is not probabilistic.

    unscaled_windows_data_dict = {
        key: windows_data.copy()
        for key, windows_data in windows_data_dict.items()
    }

    for key, windows_data in unscaled_windows_data_dict.items():
        if window_type == "timeseries" or window_type == "continuous":
            windows_data_dict[key] = transform_using_scaler(
                windows=windows_data,
                timeseries_or_continuous=window_type,
                original_discrete_windows=(
                    original_discrete_windows[key]
                    if original_discrete_windows
                    else None
                ),
                dataset_name=dataset_name,
                inverse_transform=True,
                transpose_timeseries=True,
            )
        elif window_type == "discrete":
            _, unscaled_windows_data_dict[key] = inverse_transform_discrete(
                windows_data, dataset_name
            )
        else:
            raise ValueError(f"Invalid window type: {window_type}")

    return unscaled_windows_data_dict


def fit_scaler(
    df: pd.DataFrame,
    scaler_info: Dict[str, Any] = {},
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fits scalers for specified columns in the DataFrame, with support for both global scaling
    and group-specific scaling based on discrete conditions.
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the columns to be scaled
    scaler_info : Dict[str, Any], default={}
        Configuration dictionary specifying scaling parameters for each column.
        If empty, returns an empty dictionary.
        For non-empty cases, expects structure:
        - scaler_type: Type of scaler to use (e.g., "standard", "minmax")
        - scale_by_discrete_metadata: (Optional) If present, specifies grouping columns
          for fitting separate scalers per group
            - group_labels: List of column names to group by
    Returns:
    --------
    Dict[str, Any]
        Dictionary mapping column names to their scaling information:
        - Empty dict if scaler_info is empty
        - For globally scaled columns:
            {col_name: [{"scaler": fitted_scaler}]}
        - For group-specific scaling:
            {col_name: [{"tuple": {group_col: value, ...}, "scaler": fitted_scaler}]}
    Examples:
    --------
    >>> # Empty case
    >>> fit_scaler(df, {})
    {}
    >>> # Non-empty case
    >>> scaler_info = {
    ...     "temperature": {
    ...         "scaler_type": "standard",
    ...         "scale_by_discrete_metadata": {
    ...             "group_labels": ["location", "season"]
    ...         }
    ...     },
    ...     "pressure": {
    ...         "scaler_type": "standard"
    ...     }
    ... }
    >>> scalers = fit_scaler(df, scaler_info)
    >>> # Result structure:
    >>> # {
    >>> #     "temperature": [
    >>> #         {"tuple": {"location": "NYC", "season": "summer"}, "scaler": Scaler()},
    >>> #         {"tuple": {"location": "NYC", "season": "winter"}, "scaler": Scaler()},
    >>> #         ...
    >>> #     ],
    >>> #     "pressure": [{"scaler": Scaler()}]
    >>> # }
    """
    logger.info(f"Fitting scalers for {scaler_info}")
    scalers_dict = {}
    for col_name, col_config in scaler_info.items():
        # Get the appropriate scaler class
        if (
            scaler_class := SCALER_TYPES.get(col_config["scaler_type"].lower())
        ) is None:
            raise ValueError(
                f"Unknown scaler type: {col_config['scaler_type']}"
            )
        assert scaler_class is not None
        # Check if we need to scale by groups
        if (
            "scale_by_discrete_metadata" in col_config
            and len(col_config["scale_by_discrete_metadata"]["group_labels"])
            > 0
        ):
            group_labels = col_config["scale_by_discrete_metadata"][
                "group_labels"
            ]
            scalers_dict[col_name] = []

            # Group the data and fit a scaler for each group
            for group_values, group_data in df.groupby(group_labels):
                # Convert group_values to tuple if it's not already
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)

                # Create group tuple dictionary
                group_dict = dict(zip(group_labels, group_values))

                # Fit scaler for this group
                scaler = scaler_class()
                scaler.fit(np.array(group_data[col_name]).reshape(-1, 1))

                scalers_dict[col_name].append(
                    {"tuple": group_dict, "scaler": scaler}
                )
        else:
            # Fit a single scaler for the entire column
            scaler = scaler_class()
            scaler.fit(np.array(df[col_name]).reshape(-1, 1))
            scalers_dict[col_name] = [{"scaler": scaler}]

        if len(scalers_dict[col_name]) > 0:
            logger.info(
                f"Found {len(scalers_dict[col_name])} scaling discrete label combinations for feature '{col_name}'"
            )

    logger.info("Fitted scalers")
    return scalers_dict


def get_transform_func(inverse: bool) -> Callable:
    """Returns the appropriate transform function based on inverse flag."""
    return lambda scaler, data: (
        scaler.inverse_transform(data) if inverse else scaler.transform(data)
    )


def transform_with_global_scaler(
    windows: np.ndarray, feature_idx: int, scaler: Any, transform_func: Callable
) -> np.ndarray:
    """
    Transform a feature using a global scaler (no discrete conditions).
    Parameters:
        - windows (np.ndarray): (num_windows, window_size, num_features).
        - feature_idx (int): Index of the feature to transform in the windows array.
        - scaler (Any): Fitted global scaler.
        - transform_func (Callable): Function to apply transformation, either transform or
            inverse_transform from the scaler.
    Returns:
        - np.ndarray: Transformed windows array with the same shape as input windows (num_windows, window_size, num_features).
    Raises:
        ValueError: If feature_idx is out of bounds or windows has incorrect shape.
    """
    if len(windows.shape) != 3:
        raise ValueError(
            f"Expected 3D windows array, got shape {windows.shape}"
        )
    if feature_idx >= windows.shape[2]:
        raise ValueError(
            f"feature_idx {feature_idx} out of bounds for windows shape {windows.shape}"
        )

    feature_data = einops.rearrange(
        windows[:, :, feature_idx], "b t -> (b t) 1"
    )
    transformed = transform_func(scaler, feature_data)
    windows[:, :, feature_idx] = einops.rearrange(
        transformed, "(b t) 1 -> b t", b=windows.shape[0]
    )
    return windows


def transform_with_conditional_scaler(
    windows: np.ndarray,
    feature_idx: int,
    timeseries_or_continuous_col_name: str,
    col_info: List[Dict[str, Any]],
    original_discrete_windows: np.ndarray,
    discrete_col_to_idx: Dict[str, int],
    transform_func: Callable,
) -> np.ndarray:
    """
    Transform a feature using condition-specific scalers based on discrete conditions.
    Parameters:
        - windows (np.ndarray): (num_windows, window_size, num_features)
        - feature_idx (int): Index of the feature to transform in the windows array.
        - timeseries_or_continuous_col_name (str): Name of the column being transformed (used for error messages).
        - col_info (List[Dict[str, Any]]): List of condition-specific scalers and their conditions.
            Example structure:
            [
                {
                    "tuple": {"location": "NYC", "season": "summer"},
                    "scaler": scaler1
                },
                {
                    "tuple": {"location": "LA", "season": "winter"},
                    "scaler": scaler2
                }
            ]
        - original_discrete_windows (np.ndarray): Original discrete condition windows,
            shape (num_windows, window_size, num_discrete_features).
        - discrete_col_to_idx (Dict[str, int]): Mapping from discrete column names to their
            indices in original_discrete_windows, e.g., {"location": 0, "season": 1}.
        - transform_func (Callable): Function to apply transformation, either transform or
            inverse_transform from the scaler.
    Returns:
        - np.ndarray: Transformed windows array with the same shape as input windows (num_windows, window_size, num_features).
    Raises:
        ValueError: If any data points don't match any of the provided discrete conditions.
    Notes:
        - Transforms the data in-place by modifying the input windows array.
        - Each data point must match exactly one set of discrete conditions.
    """

    # add basic validation
    if len(windows.shape) != 3:
        raise ValueError(
            f"Expected 3D windows array, got shape {windows.shape}"
        )
    if feature_idx >= windows.shape[2]:
        raise ValueError(
            f"feature_idx {feature_idx} out of bounds for windows shape {windows.shape}"
        )

    feature_data = einops.rearrange(
        windows[:, :, feature_idx], "b t -> (b t) 1"
    )
    try:
        discrete_condition_cols = list(col_info[0]["tuple"].keys())
    except IndexError:
        raise ValueError(f"col_info is not valid: {col_info=}")

    # Create discrete matrix (shape: (num_windows * window_size, num_discrete_conditions_in_col_info))
    # and transformed data arrays

    # get only the discrete features from the col info
    discrete_matrix = original_discrete_windows[
        :, :, [discrete_col_to_idx[col] for col in discrete_condition_cols]
    ]
    discrete_matrix = einops.rearrange(discrete_matrix, "b t f -> (b t) f")

    transformed_data = np.zeros_like(feature_data)
    any_match_for_transform = np.zeros(len(feature_data), dtype=bool)

    # Apply transformations for each condition
    for scaler_info in col_info:
        conditions_values = np.array(
            [scaler_info["tuple"][col] for col in discrete_condition_cols]
        )
        mask = np.all(
            discrete_matrix == conditions_values, axis=1
        )  # compare horizonally which are the conditions
        if mask.any():
            transformed_data[mask] = transform_func(
                scaler_info["scaler"], feature_data[mask]
            )
            any_match_for_transform |= mask

    # Check for unmatched conditions that were not transformed above- this condition is only for error handling
    # We should never hit this if preprocessing worked correctly
    if np.any(~any_match_for_transform):
        unmatched_indices = np.where(~any_match_for_transform)[0]
        unmatched_discrete_values = {
            col: discrete_matrix[unmatched_indices, i]
            for i, col in enumerate(discrete_condition_cols)
        }
        error_msg = (
            f"No matching scaler found for feature '{timeseries_or_continuous_col_name}' with some discrete "
            f"conditions. Unmatched combinations: {unmatched_discrete_values}"
        )
        raise ValueError(error_msg)

    windows[:, :, feature_idx] = einops.rearrange(
        transformed_data, "(b t) 1 -> b t", b=windows.shape[0]
    )
    return windows


def transform_using_scaler(
    windows: np.ndarray,
    timeseries_or_continuous: str,
    original_discrete_windows: Optional[np.ndarray] = None,
    scalers: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    col_names: Optional[List[str]] = None,
    original_discrete_colnames: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    inverse_transform: bool = False,
    transpose_timeseries: bool = True,
) -> np.ndarray:
    """
    Transform the given windows data using the fitted scalers.
    Parameters:
        - windows (np.ndarray): timeseries/continuous data/metadata windows of
            shape (num_windows, window_size, num_channels/num_features).
            note - for timeseries it is (num_windows, num_channels, window_size)
        - timeseries_or_continuous (str): Type of window data ('timeseries', 'continuous').
        - original_discrete_windows (np.ndarray): original discrete windows of
            shape (num_windows, window_size, num_discrete_features). Required when using
            condition-specific scaling.
        - scalers (Dict[str, List[Dict[str, Any]]]): fitted scalers for each column.
            Example structure:
            {
                "feature1": [{"scaler": global_scaler}],  # Global scaling
                "feature2": [  # Condition-specific scaling
                    {"tuple": {"location": "NYC", "season": "summer"}, "scaler": scaler1},
                    {"tuple": {"location": "LA", "season": "winter"}, "scaler": scaler2}
                ]
            }
        - col_names (List[str]): column names for the input `timeseries_or_continuous` data.
        - original_discrete_colnames (List[str]): column names for the original discrete data.
            Used to create discrete_col_to_idx mapping {col_name: index}.
        - dataset_name (str): dir name where the raw data is stored.
        - inverse_transform (bool): whether to inverse transform the data.
        - transpose_timeseries (bool): whether to transpose the timeseries data's last 2 dims.
            Only False in preprocessing.
    Returns:
        - np.ndarray: transformed windows data with same shape as input windows.
    Notes:
        - dataset_name is required if scalers/col_names or both are not provided.
        - First priority is given to the provided scalers and col_names then to the dataset_name.
        - If dataset_name is provided, it will load the scalers (depending on `timeseries_or_continuous`
          from SCALER_FILENAMES) and col_names (json file with all window type col names) from the
          dataset directory.
        - If no scalers are available (empty dictionary), returns the input windows unchanged.
    """
    # extra safety check
    if transpose_timeseries and timeseries_or_continuous == "timeseries":
        windows = einops.rearrange(windows, "b t c -> b c t")
    if transpose_timeseries and not timeseries_or_continuous == "timeseries":
        logger.warning(
            "Not transposing timeseries data as it is not timeseries"
        )

    # Load scalers and column names if not provided
    if scalers is None or col_names is None:
        if dataset_name is None:
            raise ValueError(
                "dataset_name is required when scalers or col_names are not provided"
            )

        # Load scalers based on window type
        if scalers is None:
            logger.info(f"Loading scalers for {timeseries_or_continuous}")
            if timeseries_or_continuous == "timeseries":
                scalers = load_timeseries_scalers(dataset_name)
            elif timeseries_or_continuous == "continuous":
                scalers = load_continuous_scalers(dataset_name)
            else:
                raise ValueError(
                    f"Invalid timeseries_or_continuous: {timeseries_or_continuous}"
                )

            # Return original windows if scalers dictionary is empty
            if not scalers:
                logger.warning(
                    f"No scalers found for {timeseries_or_continuous}. Returning original windows."
                )
                # Restore original shape if needed
                if (
                    transpose_timeseries
                    and timeseries_or_continuous == "timeseries"
                ):
                    windows = einops.rearrange(windows, "b c t -> b t c")
                return windows

        if col_names is None or original_discrete_colnames is None:
            logger.info(f"Loading column names for {timeseries_or_continuous}")
            # Load column names
            try:
                colnames_dict = load_column_names(dataset_name)
                if col_names is None:
                    col_names = colnames_dict[
                        f"{timeseries_or_continuous}_colnames"
                    ]
                if original_discrete_colnames is None:
                    original_discrete_colnames = colnames_dict[
                        "original_discrete_colnames"
                    ]
            except Exception:
                logger.warning(
                    f"Error loading column names for dataset: {dataset_name}. Using generic column names."
                )
                col_names_len = (
                    windows.shape[1]
                    if (
                        timeseries_or_continuous == "timeseries"
                        and transpose_timeseries
                    )
                    else windows.shape[2]
                )
                col_names = [
                    f"{timeseries_or_continuous}_col_{i}"
                    for i in range(col_names_len)
                ]
                original_discrete_colnames = [
                    f"original_discrete_col_{i}" for i in range(col_names_len)
                ]

    # Check if scalers is empty (could happen if provided explicitly)
    if not scalers:
        logger.warning(
            f"Empty scalers dictionary provided for {timeseries_or_continuous}. Returning original windows."
        )
        # Restore original shape if needed
        if transpose_timeseries and timeseries_or_continuous == "timeseries":
            windows = einops.rearrange(windows, "b c t -> b t c")
        return windows

    logger.info(
        f"Transforming windows of type `{timeseries_or_continuous}` with scalers: {scalers}"
    )
    # Create a mapping of discrete column names to their indices
    discrete_col_to_idx = (
        {col: idx for idx, col in enumerate(original_discrete_colnames)}
        if original_discrete_colnames
        else {}
    )

    # Get the transform function once
    transform_func = get_transform_func(inverse_transform)

    # Iterate through each feature
    for feature_idx, col_name in enumerate(col_names):
        # Example: {"tuple": {"location": "NYC", "season": "summer"}, "scaler": Scaler()}
        col_info = scalers[col_name]

        # Global case: single scaler with tuple=None
        if len(col_info) == 1 and col_info[0].get("tuple", None) is None:
            windows = transform_with_global_scaler(
                windows, feature_idx, col_info[0]["scaler"], transform_func
            )
            continue

        assert original_discrete_windows is not None, (
            "original_discrete_windows is required for conditional scaling - something went wrong in preprocessing"
        )
        transform_with_conditional_scaler(
            windows,
            feature_idx,
            col_name,
            col_info,
            original_discrete_windows,
            discrete_col_to_idx,
            transform_func,
        )

    if transpose_timeseries and timeseries_or_continuous == "timeseries":
        windows = einops.rearrange(windows, "b c t -> b t c")
    return windows


def transform_timeseries_constraints(
    values_dict: Dict[str, float],
    dataset_name: Optional[str] = None,
    discrete_conditions: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Transform (scale) a dictionary of single timeseries values using fitted scalers.

    This function applies the appropriate scaling transformation to each value in the input
    dictionary based on the column name and any discrete conditions. It supports both global
    scaling and condition-specific scaling.

    Parameters:
        - values_dict (Dict[str, float]): Dictionary mapping timeseries channel names to their values.
        - dataset_name (Optional[str]): Name of the dataset, required to load the appropriate scalers.
        - discrete_conditions (Optional[Dict[str, Any]]): Discrete conditions for conditional scaling,
          required when any channel uses condition-specific scaling.

    Returns:
        - Dict[str, float]: Dictionary with transformed (scaled) values.

    Raises:
        - ValueError: If dataset_name is not provided, if no scaler is found for a column,
          or if discrete_conditions are required but not provided.

    Notes:
        - This function only performs forward transformation (scaling), not inverse transformation.
        - For channels with condition-specific scaling, the discrete_conditions must match
          exactly one of the condition combinations used during scaler fitting.
    """
    # Load scalers if not provided
    if dataset_name is None:
        raise ValueError("dataset_name must be provided")

    scalers = load_timeseries_scalers(dataset_name)

    # Create output dictionary
    transformed_dict = {}

    for col_name, value in values_dict.items():
        if col_name not in scalers:
            raise ValueError(
                f"No scaler found for column '{col_name}', keeping original value"
            )

        col_info = scalers[col_name]

        # Global case: single scaler with no conditions
        if len(col_info) == 1 and col_info[0].get("tuple", None) is None:
            scaler = col_info[0]["scaler"]
            transformed_value = scaler.transform(np.array([[value]]))
            transformed_dict[col_name] = transformed_value[0][0]
            continue

        # Conditional scaling case
        if discrete_conditions is None:
            raise ValueError(
                f"Discrete conditions required for conditional scaling of column '{col_name}'"
            )

        # Find the matching scaler based on discrete conditions
        matching_scaler = None
        for scaler_info in col_info:
            conditions = scaler_info["tuple"]
            if all(
                discrete_conditions.get(cond_name) == cond_value
                for cond_name, cond_value in conditions.items()
            ):
                matching_scaler = scaler_info["scaler"]
                break

        if matching_scaler is None:
            condition_str = ", ".join(
                f"{k}={v}" for k, v in discrete_conditions.items()
            )
            raise ValueError(
                f"No matching scaler found for column '{col_name}' with conditions: {condition_str}"
            )

        transformed_value = matching_scaler.transform(np.array([[value]]))
        transformed_dict[col_name] = transformed_value[0][0]

    return transformed_dict
