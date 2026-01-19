"""
Helper module that contains a copy of the get_timestamp_range function for testing.
This avoids the import chain issues with the database connection.
"""

import pandas as pd


def get_timestamp_range(
    df: pd.DataFrame, timestamp_column: str, convert_to_datetime: bool = True
) -> tuple:
    """
    Determine the minimum and maximum timestamps for a given column in a dataframe.

    Args:
        df: pandas DataFrame containing timestamp data
        timestamp_column: Name of the column containing timestamp data
        convert_to_datetime: Whether to try converting the column to datetime if it's not already

    Returns:
        tuple: (min_timestamp, max_timestamp)

    Raises:
        ValueError: If the timestamp column doesn't exist, can't be ordered as timestamps,
                   or contains only null values
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' not found in dataframe")

    # Make a copy of the column to avoid modifying the original dataframe
    ts_series = df[timestamp_column].copy()

    # Check if we need to convert to datetime
    if convert_to_datetime and not pd.api.types.is_datetime64_any_dtype(
        ts_series
    ):
        try:
            ts_series = pd.to_datetime(ts_series, format="mixed")
        except Exception as e:
            raise ValueError(
                f"Failed to convert column '{timestamp_column}' to datetime: {str(e)}"
            )

    # Check if there are null values and handle accordingly
    if ts_series.isna().any():
        non_null_count = ts_series.notna().sum()
        if non_null_count == 0:
            raise ValueError(
                f"Column '{timestamp_column}' contains only null values"
            )

    # Check if the series is empty
    if len(ts_series) == 0:
        raise ValueError(f"Column '{timestamp_column}' is empty")

    # Verify the data is orderable
    try:
        # Drop nulls for the min/max calculation
        clean_series = ts_series.dropna()
        if len(clean_series) == 0:
            raise ValueError(
                f"Column '{timestamp_column}' contains only null values"
            )

        min_timestamp = clean_series.min()
        max_timestamp = clean_series.max()
    except TypeError:
        raise ValueError(
            f"Column '{timestamp_column}' contains values that cannot be ordered"
        )

    return min_timestamp, max_timestamp
