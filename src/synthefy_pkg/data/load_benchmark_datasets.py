"""
Load CSV datasets for sequential quick predictions.

"""

import json
import os
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from synthefy_pkg.utils.time_and_freq_utils import (
    FREQ_MAP,
    FREQUENCIES_TO_SECONDS,
    add_synthetic_time_features,
    convert_timestamp_to_features,
)


def add_synthetic_timestamps(
    X: Tensor,
) -> Tensor:
    """
    Add synthetic timestamps to the input tensor.

    config.add_synthetic_timestamps is a list of timestamp frequencies:
        - minutely
        - hourly
        - daily
        - monthly

    TODO: currently only handles a single timestamp column in seconds
    """

    add_synthetic_timestamps = [
        "minutely",
        "hourly",
        "daily",
        "weekly",
        "monthly",
    ]
    # get a random start time
    frequencies = np.random.choice(
        add_synthetic_timestamps, replace=True, size=X.shape[0]
    )

    # TODO: this is a hack to ensure that the timestamps are not all the sam
    max_time = FREQUENCIES_TO_SECONDS["yearly"] * 100  # 100 years in seconds

    timestamp_columns = list()

    max_seq_len = X.shape[1]
    for freq in frequencies:
        total_time = FREQUENCIES_TO_SECONDS[freq] * max_seq_len
        # logger.info(f"freq: {freq}, max_seq_len: {self.config.max_seq_len}, max_time: {max_time}, total_time: {total_time}")
        start_time = torch.randint(
            0, max_time - total_time, (1,), device=X.device
        )
        # Convert tensor to int for arange
        start = start_time.item()
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

    # replace the first k features with the timestamp columns, but shift d by k
    # roll X by the number of timestamp columns
    # logger.info(f"X before: {X[0,0]}, timestamp_columns: {timestamp_columns[0,0]}, d: {d[0]}")
    new_X = torch.zeros(
        X.shape[0], X.shape[1], X.shape[-1] + timestamp_columns.shape[-1]
    )
    new_X[:, :, : timestamp_columns.shape[-1]] = timestamp_columns
    new_X[:, :, timestamp_columns.shape[-1] :] = X
    return new_X


def check_json(json_path, is_regression: bool = False):
    """Checks if the dataset is valid for the given task type"""
    with open(json_path, "r") as f:
        data = json.load(f)

    return is_regression == (data["task_type"] == "regression")


def load_talent_data(
    suite_name, base_path="/workspace/data/talent_benchmark/data", split="test"
):
    folder_path = os.path.join(base_path, suite_name)

    arrays = {"N": None, "C": None, "y": None}
    for feature in ["N", "C", "y"]:
        file_name = f"{feature}_{split}.npy"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            try:
                # First try loading without allow_pickle
                arrays[feature] = np.load(file_path)
            except ValueError:
                # If that fails, try with allow_pickle=True
                arrays[feature] = np.load(file_path, allow_pickle=True)

    # Combine N and C arrays if they exist
    if arrays["N"] is not None and arrays["C"] is not None:
        # Concatenate along feature dimension (axis=1)
        X = np.concatenate([arrays["N"], arrays["C"]], axis=1)
    elif arrays["N"] is not None:
        X = arrays["N"]
    elif arrays["C"] is not None:
        X = arrays["C"]
    else:
        raise ValueError("Neither N nor C arrays exist")

    y = arrays["y"]
    assert X is not None
    assert y is not None
    assert X.shape[0] == y.shape[0]

    return X, y


def load_talent_dataset(dataset_name: str, base_path: str) -> dict[str, Any]:
    """Load talent dataset and return standardized format as a dictionary"""
    X_train_raw, y_train = load_talent_data(
        dataset_name, base_path, split="train"
    )
    X_test_raw, y_test = load_talent_data(dataset_name, base_path, split="test")

    assert X_train_raw.shape[0] == y_train.shape[0], (
        "X_train_raw and y_train must have the same number of rows"
    )
    assert X_test_raw.shape[0] == y_test.shape[0], (
        "X_test_raw and y_test must have the same number of rows"
    )
    assert X_train_raw.shape[1] == X_test_raw.shape[1], (
        "X_train_raw and X_test_raw must have the same number of columns"
    )

    # Convert to standardized format
    X_train = pd.DataFrame(X_train_raw)
    X_test = pd.DataFrame(X_test_raw)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return {
        "dataset_name": dataset_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def get_timestamp_if_exists(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Checks for and extracts timestamp from a DataFrame.
    It first looks for a column with datetime type.
    If not found, it tries to construct a timestamp from columns named
    'year', 'month', 'day', 'hour', 'minute', 'second' (case-insensitive).
    The columns used to construct the timestamp are dropped from the DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        A tuple containing:
        - The DataFrame with timestamp columns removed.
        - A numpy array of timestamps, or None if no timestamp is found.
    """
    # First, check for an existing datetime column
    date_cols = [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    if len(date_cols) > 0:
        timestamp = df[date_cols[0]].values
        df = df.drop(columns=[date_cols[0]])
        return df, np.array(timestamp)

    # If no datetime column, check for date components
    column_map = {c.lower(): c for c in df.columns}
    date_parts = ["year", "month", "day", "hour", "minute", "second"]

    found_parts_data = {}
    found_cols = []

    for part in date_parts:
        if part in column_map:
            col_name = column_map[part]
            found_parts_data[part] = df[col_name]
            found_cols.append(col_name)

    # Check for required components
    if (
        "year" in found_parts_data
        and "month" in found_parts_data
        and "day" in found_parts_data
    ):
        # fill missing optional parts with 0
        for part in ["hour", "minute", "second"]:
            if part not in found_parts_data:
                found_parts_data[part] = 0

        timestamp = pd.to_datetime(pd.DataFrame(found_parts_data))
        df = df.drop(columns=found_cols)
        return df, np.array(timestamp.values)

    return df, None


def check_existing_time_features(df: pd.DataFrame) -> bool:
    """
    Check if the DataFrame already contains the 14 specific time series features
    that would be generated by convert_timestamp_to_features.

    Args:
        df: The input DataFrame.

    Returns:
        bool: True if all 14 time series features are present, False otherwise.
    """
    expected_time_features = [
        "year",
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_month_sin",
        "day_of_month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "week_of_year_sin",
        "week_of_year_cos",
        "month_of_year_sin",
        "month_of_year_cos",
        "running_index",
    ]

    # Check if all expected features are present
    existing_features = set(df.columns)
    missing_features = [
        feat for feat in expected_time_features if feat not in existing_features
    ]

    return len(missing_features) == 0


def get_existing_time_features_count(df: pd.DataFrame) -> int:
    """
    Count how many of the expected time series features are present in the DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        int: Number of expected time features found.
    """
    expected_time_features = [
        "year",
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_month_sin",
        "day_of_month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "week_of_year_sin",
        "week_of_year_cos",
        "month_of_year_sin",
        "month_of_year_cos",
        "running_index",
    ]

    existing_features = set(df.columns)
    found_features = [
        feat for feat in expected_time_features if feat in existing_features
    ]
    return len(found_features)


def load_csv_dataset(
    suite_name,
    use_tabpfn_features=False,
    base_path="/workspace/data/synthetic_data",
):
    df = pd.read_csv(os.path.join(base_path, f"{suite_name}"))

    # Check if the DataFrame already has time series features
    if use_tabpfn_features:
        has_existing_time_features = check_existing_time_features(df)

        if has_existing_time_features:
            # If time features already exist, use them directly
            # logger.info("Found existing time series features in DataFrame, using them directly")
            column_names = df.columns.tolist()[:-1]
            X = df.iloc[:, :-1]  # Get all columns except last as X
            y = df.iloc[:, -1].values  # Get last column as y
            num_ts_features = get_existing_time_features_count(df)
            timestamp = None
        else:
            # Original logic for handling timestamps and adding synthetic features
            df, timestamp = get_timestamp_if_exists(df)
            column_names = df.columns.tolist()[:-1]

            X = df.iloc[:, :-1]  # Get all columns except last as X
            y = df.iloc[:, -1].values  # Get last column as y

            if timestamp is not None:
                # Convert existing timestamps to features
                res, ts_column_names = convert_timestamp_to_features(
                    pd.to_datetime(timestamp)
                )
                res = res.numpy()
                X = np.concatenate([res, X], axis=1)
                num_ts_features = res.shape[1]
                column_names = ts_column_names + column_names

            else:
                # Add synthetic time features when no timestamps exist
                # Calculate max_features needed (original features + time features + buffer)
                original_features = X.shape[1]
                max_features = (
                    original_features + 50
                )  # Allow some extra features for time features

                # Convert X to torch tensor with proper shape for add_synthetic_time_features
                # Pad the tensor to max_features to avoid size mismatch
                X_padded = np.zeros((X.shape[0], max_features))
                X_padded[:, :original_features] = X.values

                X_tensor = torch.tensor(
                    X_padded, dtype=torch.float32
                ).unsqueeze(0)  # Add batch dimension

                # Create dummy d tensor (number of features per dataset)
                d = [torch.tensor(original_features, dtype=torch.int32)]

                available_frequencies = [
                    "minutely",
                    "hourly",
                    "daily",
                    "monthly",
                ]

                # Add synthetic time features
                (
                    X_with_time,
                    updated_d,
                    num_ts_features,
                    ts_column_names,
                    frequencies,
                    start_times,
                ) = add_synthetic_time_features(
                    X_tensor,
                    d,
                    available_frequencies,
                    max_features=max_features,
                    series_flags=torch.ones(
                        X_tensor.shape[0],
                        dtype=torch.bool,
                        device=X_tensor.device,
                    ),
                )
                assert original_features + num_ts_features <= max_features, (
                    "Number of features should be less than or equal to max_features - increase the extra features"
                )

                # Convert back to numpy and remove batch dimension
                X = X_with_time.squeeze(0).numpy()[
                    :, : original_features + num_ts_features
                ]
                column_names = ts_column_names + column_names
    else:
        has_existing_time_features = check_existing_time_features(df)

        if has_existing_time_features:
            # If time features already exist, use them directly
            # logger.info("Found existing time series features in DataFrame, using them directly")
            # remove all the time features
            columns_to_remove = []
            for col in df.columns:
                if "feat" in col or "target" in col:
                    continue
                columns_to_remove.append(col)
            df = df.drop(columns=columns_to_remove)

        column_names = df.columns.tolist()[:-1]
        X = df.iloc[:, :-1]  # Get all columns except last as X
        y = df.iloc[:, -1].values  # Get last column as y
        num_ts_features = get_existing_time_features_count(df)
        timestamp = None
        X = add_synthetic_timestamps(torch.tensor(X.values).unsqueeze(0))
        X = X.squeeze(0).numpy()
        num_ts_features = X.shape[1] - df.shape[1] + 1
        column_names = [
            f"ts_feat_{i}" for i in range(num_ts_features)
        ] + column_names

    train_size = int(0.7 * X.shape[0])
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    # Split timestamps if they exist
    timestamp_train = None
    timestamp_test = None
    if timestamp is not None:
        timestamp_train = timestamp[:train_size]
        timestamp_test = timestamp[train_size:]

    assert X_train.shape[0] == y_train.shape[0], (
        "X_train and y_train must have the same number of rows"
    )
    assert X_test.shape[0] == y_test.shape[0], (
        "X_test and y_test must have the same number of rows"
    )
    assert X_train.shape[1] == X_test.shape[1], (
        "X_train and X_test must have the same number of columns"
    )

    return {
        "X_train": pd.DataFrame(X_train, columns=column_names),
        "X_test": pd.DataFrame(X_test, columns=column_names),
        "y_train": y_train,
        "y_test": y_test,
        "num_timestamp_features": num_ts_features,
        "timestamp_train": timestamp_train,
        "timestamp_test": timestamp_test,
    }
