import datetime
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from dateutil.parser import parse
from loguru import logger
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Default threshold for null values in timestamp columns
# This represents the maximum proportion of null values allowed in a column
# for it to be considered a valid timestamp index candidate
DEFAULT_NULL_THRESHOLD = 0.15


class SmartApproximationUtils:
    @staticmethod
    def check_column_for_timestamp_index(args):
        """
        Function to run in a subprocess. Checks if a column is a good timestamp index.
        Arguments:
        - args: tuple of (column name, series, null_threshold)
        Returns:
        - tuple of (column name or None, list of messages) where messages contain debug/info/error logs
        """
        col, series, null_threshold = args
        messages = []
        messages.append(
            ("debug", f"Checking column '{col}' as timestamp index candidate")
        )

        try:
            # Skip purely numeric columns to prevent interpreting them as epoch timestamps
            if is_numeric_dtype(series):
                messages.append(
                    (
                        "debug",
                        f"Skipping numeric column '{col}' to avoid misinterpreting as timestamp",
                    )
                )
                return None, messages

            if not is_datetime64_any_dtype(series):
                messages.append(
                    (
                        "debug",
                        f"Column '{col}' is not datetime type, attempting conversion",
                    )
                )
                # Try to convert to datetime - first standard parsing, then mixed format
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        series = pd.to_datetime(series)
                        # Check if any warnings were raised during parsing
                        if w:
                            for warning in w:
                                warning_msg = str(warning.message)
                                if any(
                                    phrase in warning_msg
                                    for phrase in [
                                        "Could not infer format",
                                        "falling back to",
                                        "Parsing dates in",
                                        "dayfirst=",
                                        "specify a format to silence this warning",
                                    ]
                                ):
                                    messages.append(
                                        (
                                            "error",
                                            f"Column '{col}' has mixed timestamp formats. Please ensure that the timestamps are in the format '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d'.",
                                        )
                                    )
                                    return None, messages
                    except ValueError:
                        try:
                            series = pd.to_datetime(series, format="mixed")
                            messages.append(
                                (
                                    "error",
                                    f"Column '{col}' has mixed timestamp formats. Please ensure that the timestamps are in the format '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d'.",
                                )
                            )
                            return None, messages
                        except ValueError:
                            messages.append(
                                (
                                    "debug",
                                    f"Could not parse '{col}' with timestamp format",
                                )
                            )
                            return None, messages

            # Check missingness
            null_ratio = series.isna().mean()
            if null_ratio > null_threshold:
                messages.append(
                    (
                        "debug",
                        f"Column '{col}' has too many nulls: {null_ratio:.2%} > {null_threshold:.2%}",
                    )
                )
                return None, messages

            non_null_series = series.dropna()
            # Check if column is empty
            if len(non_null_series) == 0:
                messages.append(("debug", f"Column '{col}' is empty"))
                return None, messages

            messages.append(
                ("info", f"Column '{col}' is a valid timestamp index candidate")
            )
            return col, messages

        except Exception as e:
            messages.append(
                (
                    "warning",
                    f"Error checking column '{col}' as timestamp index: {str(e)}",
                )
            )
            return None, messages

    @staticmethod
    def find_timestamp_index_candidates_parallel(
        df, null_threshold=DEFAULT_NULL_THRESHOLD
    ):
        """
        Parallel version of find_timestamp_index_candidates.
        Uses multiprocessing to check each column in parallel.

        Parameters:
        - df (pd.DataFrame): The DataFrame to analyze
        - null_threshold (float): Max proportion of allowed nulls

        Returns:
        - Tuple of (candidates, error_messages) where:
          - candidates: List of column names suitable for time series indexing
          - error_messages: List of error messages to display to the user
        """
        logger.info(
            f"Finding timestamp index candidates in parallel for DataFrame with {len(df.columns)} columns"
        )
        # Bundle arguments for each column
        args = [(col, df[col], null_threshold) for col in df.columns]

        # Use multiprocessing only if there are enough columns to justify the overhead
        if len(df.columns) < 10:  # Threshold can be adjusted based on testing
            logger.debug(
                "Too few columns for parallel processing, using sequential method"
            )
            return SmartApproximationUtils.find_timestamp_index_candidates(
                df, null_threshold
            )

        # Use all CPU cores
        try:
            num_cpu_cores_to_use = max(cpu_count() // 4, 1)
            with Pool(num_cpu_cores_to_use) as pool:
                logger.debug(
                    f"Using {num_cpu_cores_to_use} CPU cores for parallel processing"
                )
                results = pool.map(
                    SmartApproximationUtils.check_column_for_timestamp_index,
                    args,
                )
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            return SmartApproximationUtils.find_timestamp_index_candidates(
                df, null_threshold
            )

        # Filter successful candidates
        candidates = []
        error_messages = []
        for result in results:
            col, messages = result
            # Log all messages from the subprocess
            for level, message in messages:
                if level == "debug":
                    logger.debug(message)
                elif level == "info":
                    logger.info(message)
                elif level == "warning":
                    logger.warning(message)
                elif level == "error":
                    logger.error(message)
                    error_messages.append(message)

            if col is not None:
                candidates.append(col)

        logger.info(
            f"Found {len(candidates)} timestamp index candidates: {candidates}"
        )
        return candidates, error_messages

    @staticmethod
    def find_timestamp_index_candidates(
        df, null_threshold=DEFAULT_NULL_THRESHOLD
    ):
        """
        Identify columns that are suitable for use as a time series index.

        Criteria:
        - Must be datetime dtype (after attempting conversion)
        - Must have low missingness (default: <=15% nulls)
        - Must be unique
        - Must be sorted (monotonic increasing or decreasing)

        Parameters:
        - df (pd.DataFrame): The DataFrame to analyze
        - null_threshold (float): Max proportion of allowed nulls

        Returns:
        - Tuple of (candidates, error_messages) where:
          - candidates: List of column names that could be used as a time series index
          - error_messages: List of error messages to display to the user
        """
        logger.info(
            f"Finding timestamp index candidates sequentially for DataFrame with {len(df.columns)} columns"
        )
        # Bundle arguments for each column
        args = [(col, df[col], null_threshold) for col in df.columns]

        # Process each column sequentially
        candidates = []
        error_messages = []
        for arg in args:
            result = SmartApproximationUtils.check_column_for_timestamp_index(
                arg
            )
            col, messages = result
            # Log all messages from the subprocess
            for level, message in messages:
                if level == "debug":
                    logger.debug(message)
                elif level == "info":
                    logger.info(message)
                elif level == "warning":
                    logger.warning(message)
                elif level == "error":
                    logger.error(message)
                    error_messages.append(message)

            if col is not None:
                candidates.append(col)

        logger.info(
            f"Found {len(candidates)} timestamp index candidates: {candidates}"
        )
        return candidates, error_messages

    @staticmethod
    def _is_datetime_like(value):
        """
        Determine if a value is likely to be a datetime value.
        Uses pandas datetime conversion logic for more accurate detection.
        """
        if value is None:
            return False

        # Check if already a datetime type
        if isinstance(value, (pd.Timestamp, datetime.datetime)):
            return True

        # Convert to string for checking
        str_value = str(value)

        # Try standard datetime formats first (more efficient)
        try:
            # Try ISO format
            pd.to_datetime(str_value, format="%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            try:
                # Try date only format
                pd.to_datetime(str_value, format="%Y-%m-%d")
                return True
            except ValueError:
                # Finally try with flexible parsing
                try:
                    result = pd.to_datetime(str_value, errors="coerce")
                    # Check if the conversion produced a valid datetime (not NaT)
                    return not pd.isna(result)
                except Exception:
                    return False

    @staticmethod
    def get_categorical_features(
        df: pd.DataFrame, max_unique_values: int = 100
    ) -> dict:
        logger.info(
            f"Getting improved categorical values from DataFrame with {len(df.columns)} columns"
        )
        result = {}

        for col in df.columns:
            series = df[col].dropna()
            dtype = series.dtype
            logger.debug(f"Analyzing column '{col}' with dtype {dtype}")

            # Skip if all values are NaN
            if series.empty:
                logger.debug(
                    f"Skipping column '{col}' because it's empty after dropping NaN values"
                )
                continue

            unique_values = series.unique()
            num_unique = len(unique_values)
            logger.debug(f"Column '{col}' has {num_unique} unique values")

            # Case 1: Explicitly categorical or object (excluding datetime-like)
            if isinstance(dtype, pd.CategoricalDtype) or is_object_dtype(dtype):
                # Skip mixed-type that are semantically numeric and high-cardinality
                all_numeric_like = (
                    pd.to_numeric(series, errors="coerce").notnull().all()
                )
                logger.debug(
                    f"Column '{col}' is numeric-like: {all_numeric_like}"
                )
                if all_numeric_like and num_unique > max_unique_values:
                    logger.debug(
                        f"Skipping column '{col}' because it's numeric-like with high cardinality ({num_unique} > {max_unique_values})"
                    )
                    continue
                logger.debug(
                    f"Adding column '{col}' as categorical with {num_unique} unique values"
                )
                result[col] = set(unique_values)

            # Case 2: Numeric low-cardinality
            elif is_numeric_dtype(dtype) and num_unique <= max_unique_values:
                logger.debug(
                    f"Adding column '{col}' as numeric with low cardinality ({num_unique} ≤ {max_unique_values})"
                )
                # Numeric types should always be sortable
                result[col] = set(unique_values)
            else:
                logger.debug(
                    f"Skipping column '{col}' because it's not categorical or numeric with low cardinality"
                )

        logger.info(
            f"Found {len(result)} categorical-like columns out of {len(df.columns)} total columns"
        )
        return result
