"""
Data loading and column categorization utilities for DataSummarizer.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import polars as pl
from loguru import logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file is not found
        json.JSONDecodeError: If the JSON file is malformed
    """
    logger.info(f"Loading configuration from: {config_path}")

    try:
        absolute_path = os.path.abspath(config_path)
        with open(absolute_path, "r") as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from the configuration file: {config_path}"
        )
        raise json.JSONDecodeError(
            "Error decoding JSON from the configuration file",
            config_path,
            0,
        )


def load_data(data_inputs: List[Union[pl.DataFrame, str]]) -> pl.DataFrame:
    """
    Load data from various formats into a Polars DataFrame.
    If multiple inputs are provided, they will be concatenated vertically.

    Args:
        data_inputs: List of paths to data files or Polars DataFrames.

    Returns:
        Concatenated Polars DataFrame.
    """
    dataframes = []

    for i, data_input in enumerate(data_inputs):
        if isinstance(data_input, pl.DataFrame):
            df = data_input
        elif isinstance(data_input, str):
            if data_input.endswith(".csv"):
                df = pl.scan_csv(data_input).collect()
            elif data_input.endswith(".parquet"):
                df = pl.scan_parquet(data_input).collect()
            elif data_input.endswith(".json"):
                df = pl.scan_json(data_input).collect()
            else:
                raise ValueError(
                    f"Unsupported data format for input {i + 1}: {data_input}"
                )
        else:
            raise ValueError(
                f"Invalid data input type for input {i + 1}: {type(data_input)}"
            )

        # Add source identifier column
        df = df.with_columns(pl.lit(f"Source_{i + 1}").alias("_source_id"))
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        # Concatenate all dataframes
        return pl.concat(dataframes, how="vertical_relaxed")


def categorize_columns(data_summarizer) -> None:
    """
    Categorize columns as continuous or discrete, excluding group and timeseries columns.

    Args:
        data_summarizer: DataSummarizer instance to categorize columns for
    """
    # Include timeseries_cols and group_cols in the exclusion set
    columns_to_exclude = set(
        (
            [data_summarizer.timestamps_col]
            if data_summarizer.timestamps_col
            else []
        )
        + (data_summarizer.group_cols if data_summarizer.group_cols else [])
        + (
            data_summarizer.timeseries_cols
            if data_summarizer.timeseries_cols
            else []
        )
    )
    columns_to_exclude.add("__index_level_0__")
    columns_to_exclude.add("_source_id")
    data_summarizer.continuous_cols = []
    data_summarizer.discrete_cols = []

    # First check if columns are specified in config
    if data_summarizer.config:
        config_continuous = data_summarizer.config.get("continuous", {}).get(
            "cols", []
        )
        config_discrete = data_summarizer.config.get("discrete", {}).get(
            "cols", []
        )

        # Add columns specified in config, excluding any that are in timeseries_cols or group_cols
        data_summarizer.continuous_cols.extend(
            [col for col in config_continuous if col not in columns_to_exclude]
        )
        data_summarizer.discrete_cols.extend(
            [col for col in config_discrete if col not in columns_to_exclude]
        )

        # Only auto-categorize columns not already specified
        remaining_cols = [
            col
            for col in data_summarizer.df_data.columns
            if col not in columns_to_exclude
            and col not in config_continuous
            and col not in config_discrete
        ]
    else:
        remaining_cols = [
            col
            for col in data_summarizer.df_data.columns
            if col not in columns_to_exclude
        ]

    for col in remaining_cols:
        if data_summarizer.df_data[col].dtype.is_numeric():
            data_summarizer.continuous_cols.append(col)
        else:
            # Check if it's a list column and log a warning
            if data_summarizer.df_data[col].dtype == pl.List:
                logger.warning(
                    f"Column '{col}' has list data type - categorizing as discrete but some operations may be limited"
                )
            data_summarizer.discrete_cols.append(col)

    # Add timeseries columns to continuous columns for analysis purposes
    # (they are numeric and should be analyzed as continuous variables)
    # But only if they're not explicitly specified in the config
    # If timeseries columns are explicitly specified in config, don't add them to continuous
    if not data_summarizer.config or "timeseries" not in data_summarizer.config:
        for col in data_summarizer.timeseries_cols:
            if (
                col not in data_summarizer.continuous_cols
                and data_summarizer.df_data[col].dtype.is_numeric()
            ):
                data_summarizer.continuous_cols.append(col)


def is_column_safe_for_analysis(df_data: pl.DataFrame, col_name: str) -> bool:
    """
    Check if a column is safe for standard analysis operations.

    Args:
        df_data: Polars DataFrame containing the data
        col_name: Name of the column to check

    Returns:
        True if the column is safe for analysis, False otherwise
    """
    if col_name not in df_data.columns:
        return False

    col_dtype = df_data.schema[col_name]

    # List columns are not safe for standard analysis operations
    if col_dtype == pl.List:
        return False

    # Other potentially problematic types can be added here
    return True


def setup_data_sources_and_config(
    data_input: Union[pl.DataFrame, str, List[Union[pl.DataFrame, str]]],
    config: Optional[Dict[str, Any]],
    group_cols: Optional[List[str]],
) -> tuple[List[Union[pl.DataFrame, str]], List[str], Dict[str, Any]]:
    """
    Setup data sources and configuration for DataSummarizer.

    Args:
        data_input: Path to dataset file relative to SYNTHEFY_DATASETS_BASE, absolute path, a Polars DataFrame, or a list of these
        config: Optional configuration dictionary
        group_cols: Optional list of column names to use for grouping

    Returns:
        Tuple of (processed_data_inputs, data_sources, processed_config)
    """
    # Clean up and validate paths
    data_sources = []  # Track individual data sources
    if isinstance(data_input, (str, pl.DataFrame)):
        # Single data source - convert to list for uniform handling
        data_input = [data_input]

    # Process each data input
    processed_inputs = []
    for i, single_input in enumerate(data_input):
        if isinstance(single_input, str):
            single_input = single_input.strip()
            if not single_input.startswith("/"):
                datasets_base = os.getenv("SYNTHEFY_DATASETS_BASE")
                if not datasets_base:
                    raise ValueError(
                        "SYNTHEFY_DATASETS_BASE environment variable must be set"
                    )
                single_input = os.path.join(datasets_base, single_input)
            data_sources.append(f"Source {i + 1}: {single_input}")
        else:
            data_sources.append(f"Source {i + 1}: DataFrame")
        processed_inputs.append(single_input)

    # Load config if it's a string path
    if isinstance(config, str):
        processed_config = load_config(config)
    else:
        processed_config = config

    return (
        processed_inputs,
        data_sources,
        processed_config if processed_config is not None else {},
    )


def setup_timestamp_and_group_columns(
    data_summarizer,
    config: Optional[Dict[str, Any]],
    group_cols: Optional[List[str]],
) -> None:
    """
    Setup timestamp and group columns based on config or auto-detection.

    Args:
        data_summarizer: DataSummarizer instance to setup columns for
        config: Configuration dictionary
        group_cols: Optional list of column names to use for grouping
    """
    if config:
        # Handle timestamp_col that could be either str or list
        timestamps_col = config.get("timestamps_col")
        if timestamps_col:
            # Convert string to list of one element if needed
            if isinstance(timestamps_col, str):
                timestamp_col = timestamps_col
            else:
                # Ensure it's a non-empty list before accessing first element
                timestamp_col = timestamps_col[0] if timestamps_col else None

            if (
                timestamp_col
                and timestamp_col in data_summarizer.df_data.columns
            ):
                data_summarizer.timestamps_col = timestamp_col
            elif "timestamp" in data_summarizer.df_data.columns:
                data_summarizer.timestamps_col = "timestamp"
            else:
                logger.warning(
                    f"Timestamp column {timestamp_col} not found in data columns {data_summarizer.df_data.columns}"
                )

        # Always use config values if available, otherwise fall back to parameters
        data_summarizer.group_cols = config.get("group_labels", {}).get(
            "cols", []
        )
        if not data_summarizer.group_cols and group_cols is not None:
            data_summarizer.group_cols = group_cols

        data_summarizer.timeseries_cols = config.get("timeseries", {}).get(
            "cols", []
        )
    else:
        # No config provided, use auto-detection and group_cols parameter
        logger.info("No config provided, using auto-detection")
        data_summarizer.group_cols = (
            group_cols if group_cols is not None else []
        )

        # Auto-detect timestamp column
        datetime_cols = [
            col
            for col in data_summarizer.df_data.columns
            if data_summarizer.df_data.schema[col]
            in [pl.Datetime, pl.Date, pl.Time]
        ]

        if len(datetime_cols) > 1:
            logger.warning(
                f"Multiple datetime columns found: {datetime_cols}. Using first column: {datetime_cols[0]}"
            )

        data_summarizer.timestamps_col = (
            datetime_cols[0] if datetime_cols else None
        )
        if not data_summarizer.timestamps_col:
            logger.info("No datetime column found in data")

    # Validate group columns if they exist
    if data_summarizer.group_cols:
        missing_cols = [
            col
            for col in data_summarizer.group_cols
            if col not in data_summarizer.df_data.columns
        ]
        if missing_cols:
            raise ValueError(f"Group columns not found in data: {missing_cols}")
