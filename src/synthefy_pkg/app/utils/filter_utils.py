from io import BytesIO
from itertools import product
from typing import Dict, List, Optional, Union

import boto3
import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.app.data_models import (
    GroupLabelColumnFilters,
    SupportedAggregationFunctions,
)
from synthefy_pkg.app.utils.s3_utils import upload_file_to_s3


class FilterUtils:
    @staticmethod
    def _get_pandas_aggregation_function(
        aggregation_func: SupportedAggregationFunctions,
    ):
        """
        Convert SupportedAggregationFunctions enum to pandas-compatible aggregation function.

        Args:
            aggregation_func: The aggregation function enum value

        Returns:
            A pandas-compatible aggregation function (string or callable)
        """
        if aggregation_func == SupportedAggregationFunctions.MODE:
            # For mode, return a lambda that computes mode and handles empty results
            def mode_func(x):
                if len(x) == 0:
                    return None
                # Ensure x is a pandas Series for mode calculation
                if not isinstance(x, pd.Series):
                    x = pd.Series(x)
                mode_result = x.mode()
                if len(mode_result) > 0:
                    return mode_result.iloc[0]
                else:
                    # If no mode (all values are unique), return the first value
                    return x.iloc[0] if len(x) > 0 else None

            return mode_func
        else:
            # For other aggregation functions, return the string value directly
            return aggregation_func.value

    @staticmethod
    def _expand_filter_values_with_type_conversion(
        values: List[Union[str, int, float]],
    ) -> List[Union[str, int, float]]:
        """
        Expand filter values to include both original and converted data types.

        This method tries to convert string values to numeric types (int, float) and
        numeric values to string types, allowing the backend to match against both
        data types without knowing the actual column type in advance.

        Args:
            values: List of filter values from the frontend

        Returns:
            List of expanded values including original and converted types
        """
        seen = set()
        unique_values = []

        for value in values:
            # Always include the original value
            key = (value, type(value))
            if key not in seen:
                seen.add(key)
                unique_values.append(value)

            if isinstance(value, str):
                # Try to convert string to numeric types
                try:
                    # Try integer conversion first
                    if "." not in value:
                        int_val = int(value)
                        int_key = (int_val, type(int_val))
                        if int_key not in seen:
                            seen.add(int_key)
                            unique_values.append(int_val)
                    else:
                        # Try float conversion
                        float_val = float(value)
                        float_key = (float_val, type(float_val))
                        if float_key not in seen:
                            seen.add(float_key)
                            unique_values.append(float_val)
                except (ValueError, TypeError):
                    # If conversion fails, just keep the original string
                    pass
            elif isinstance(value, (int, float)):
                # Convert numeric values to string
                str_val = str(value)
                str_key = (str_val, type(str_val))
                if str_key not in seen:
                    seen.add(str_key)
                    unique_values.append(str_val)

        return unique_values

    @staticmethod
    def build_filter_path(
        filters: GroupLabelColumnFilters,
    ) -> str:
        """
        Build filter path for S3 keys, handling both filtered and unfiltered cases.

        Args:
            filters: GroupLabelColumnFilters instance (may have empty filter list for unfiltered data)

        Returns:
            String representing the filter path component
        """
        # Handle unfiltered case - when filter list is empty
        if not filters.filter:
            return f"unfiltered_agg={filters.aggregation_func.value}"

        combined_filters: Dict[str, List[str | int | float]] = {}

        # Flatten and merge filters
        for filter_dict in filters.filter:
            for column, values in filter_dict.items():
                if column not in combined_filters:
                    combined_filters[column] = []
                combined_filters[column].extend(values)

        # Remove duplicates and sort values
        for column in combined_filters:
            combined_filters[column] = sorted(set(combined_filters[column]))

        # Sort columns and construct path parts
        sorted_columns = sorted(combined_filters.keys())
        path_parts = [
            f"{col}={'+'.join([str(val) for val in combined_filters[col]])}"
            for col in sorted_columns
        ]

        # Include aggregation function in the path to ensure different aggregation
        # functions create different cache keys
        path_parts.append(f"agg={filters.aggregation_func.value}")

        return "/".join(path_parts)

    @staticmethod
    def get_s3_path(
        bucket: str,
        user_id: str,
        dataset_file_name: str,
        ext: str,
        filters: GroupLabelColumnFilters,
    ) -> str:
        filter_path = FilterUtils.build_filter_path(filters)
        return f"s3://{bucket}/{user_id}/foundation_models/{dataset_file_name}/{filter_path}/data.{ext}"

    @staticmethod
    def get_s3_key(
        user_id: str,
        dataset_file_name: str,
        ext: str,
        filters: GroupLabelColumnFilters,
    ) -> str:
        filter_path = FilterUtils.build_filter_path(filters)
        return f"{user_id}/foundation_models/{dataset_file_name}/{filter_path}/data.{ext}"

    @staticmethod
    def get_metadata_key(
        user_id: str,
        dataset_file_name: str,
        filters: GroupLabelColumnFilters,
    ) -> str:
        filter_path = FilterUtils.build_filter_path(filters)
        return f"{user_id}/foundation_models/{dataset_file_name}/{filter_path}/metadata.json"

    @staticmethod
    def save_dataframe_to_s3(
        df: pd.DataFrame,
        bucket: str,
        path: str,
        file_format: str = "csv",
    ) -> None:
        """
        Save a pandas DataFrame to an S3 bucket at the specified path.

        Args:
            df: The pandas DataFrame to save
            bucket: The S3 bucket name
            path: The path within the bucket (without s3:// prefix)
            file_format: The format to save the data in ('csv' or 'parquet')
        """
        import os
        import tempfile

        s3_client = boto3.client("s3")

        # Remove s3:// prefix and bucket name if present in the path
        if path.startswith("s3://"):
            path = path[5:]  # Remove 's3://'
            if path.startswith(f"{bucket}/"):
                path = path[len(bucket) + 1 :]  # Remove bucket name and slash

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_format.lower()}"
        )
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            # Save DataFrame to the temporary file in specified format
            if file_format.lower() == "csv":
                df.to_csv(temp_file_path, index=False)
            elif file_format.lower() == "parquet":
                df.to_parquet(temp_file_path, index=False)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_format}. Use 'csv' or 'parquet'."
                )

            # Upload the file to S3
            upload_file_to_s3(s3_client, temp_file_path, bucket, path)

        finally:
            # Delete the temporary file regardless of success or failure
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @staticmethod
    def _aggregate_dataframe(
        df: pd.DataFrame,
        timestamp_column: str,
        aggregation_func: SupportedAggregationFunctions,
        additional_groupby_keys: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Helper function to aggregate dataframe by timestamp and optionally additional keys.

        Args:
            df: The pandas DataFrame to aggregate
            timestamp_column: Name of the timestamp column
            aggregation_func: The aggregation function to apply
            additional_groupby_keys: Optional list of additional columns to group by

        Returns:
            Aggregated DataFrame

        Raises:
            ValueError: If duplicate timestamps exist and non-numeric columns would need aggregation
        """
        if df[timestamp_column].is_unique:
            return df

        # Build groupby keys
        groupby_keys = [timestamp_column]
        if additional_groupby_keys:
            groupby_keys.extend(additional_groupby_keys)

        # Check for non-numeric columns that would need aggregation
        numeric_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        non_numeric_columns = [
            col
            for col in df.columns
            if col not in numeric_columns and col not in groupby_keys
        ]

        # Check if non-numeric columns actually vary within groups
        problematic_columns = []
        if non_numeric_columns:
            grouped_df = df.groupby(groupby_keys)
            problematic_columns = [
                col
                for col in non_numeric_columns
                if grouped_df[col].nunique().gt(1).any()
            ]

        if problematic_columns:
            error_message = (
                f"Dataset contains duplicate timestamps and non-numeric columns with varying values that require aggregation: {problematic_columns}. "
                f"Aggregation strategy for non-numeric columns is not yet configurable. "
                f"Please either: 1) Remove duplicate timestamps from your data, or 2) Convert non-numeric columns to numeric, or 3) Ensure non-numeric columns have consistent values within timestamp groups."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Get pandas aggregation function
        pandas_agg_func = FilterUtils._get_pandas_aggregation_function(
            aggregation_func
        )

        # Create aggregation dictionary for columns not in groupby_keys
        agg_dict = {}
        for column in df.columns:
            if column not in groupby_keys:
                if column in non_numeric_columns:
                    # For non-numeric columns (which we've verified have consistent values within groups),
                    # use 'first' to take any value since they're all the same
                    agg_dict[column] = "first"
                else:
                    # For numeric columns, use the user-specified aggregation function
                    agg_dict[column] = pandas_agg_func

        return df.groupby(groupby_keys).agg(agg_dict).reset_index()

    @staticmethod
    def filter_and_aggregate_dataframe(
        df: pd.DataFrame,
        filters: GroupLabelColumnFilters,
        timestamp_column: str,
    ) -> pd.DataFrame:
        """
        Filter a pandas DataFrame based on GroupLabelColumnFilters.

        Args:
            df: The pandas DataFrame to filter
            filters: GroupLabelColumnFilters instance containing filter criteria (may have empty filter list for unfiltered data)
            timestamp_column: Name of the timestamp column currently selected (used for aggregation)
        Returns:
            A filtered pandas DataFrame containing only rows that match the filter criteria

        Logic:
            - If filters.filter is empty, returns the entire dataframe with aggregation applied if needed
            - Each dictionary in the filter list represents a group of AND conditions
            - Multiple dictionaries in the filter list are combined with OR logic
            - Within each dictionary, values for the same column are combined with OR logic
            - Different columns within the same dictionary are combined with AND logic
            - Filter values are automatically converted to both string and numeric types to handle
              cases where the user's input type doesn't match the column's data type

        Examples:
            filter=[{"category": ["A"], "location": ["CA"]}, {"category": ["B"], "location": ["TX"]}]
            means: (Category=A AND location=CA) OR (Category=B AND location=TX)

            filter=[{"category": ["A", "B"], "location": ["CA", "TX"]}]
            means: (Category=A OR Category=B) AND (location=CA OR location=TX)
        """
        if not filters.filter:
            # No column filters - return entire dataframe with aggregation if needed
            return FilterUtils._aggregate_dataframe(
                df, timestamp_column, filters.aggregation_func
            )

        # Create a copy to avoid modifying the original dataframe
        filtered_df = df.copy()

        # Initialize a mask for the AND combination of all filter groups
        combined_mask = pd.Series(True, index=filtered_df.index)

        for filter_dict in filters.filter:
            # Initialize mask for this filter group (combined with OR)
            group_mask = pd.Series(False, index=filtered_df.index)

            # Process each column in this filter group
            for column, values in filter_dict.items():
                if column in filtered_df.columns and values:
                    # Expand filter values to include both original and converted data types
                    expanded_values = (
                        FilterUtils._expand_filter_values_with_type_conversion(
                            values
                        )
                    )

                    logger.debug(
                        f"Filtering column '{column}' with expanded values: {expanded_values}"
                    )

                    # Try to match with the expanded values (original + converted types)
                    column_mask = filtered_df[column].isin(expanded_values)

                    # Different columns are combined with OR
                    group_mask = group_mask | column_mask

            # Combine this group's result with the overall result using AND
            combined_mask = combined_mask & group_mask

        # Apply the combined mask to the dataframe
        filtered_df = filtered_df[combined_mask]

        # Get filter keys for groupby
        filter_keys = []
        for filter_dict in filters.filter:
            filter_keys.extend(list(filter_dict.keys()))
        filter_keys = list(set(filter_keys))

        # Aggregate the filtered dataframe
        return FilterUtils._aggregate_dataframe(
            filtered_df, timestamp_column, filters.aggregation_func, filter_keys
        )

    @staticmethod
    def get_s3_base_path(
        user_id: str,
        dataset_name: str,
    ) -> str:
        """
        Constructs the base S3 path for foundation models without filters.

        Args:
            user_id: The user ID
            dataset_name: The name of the dataset

        Returns:
            A string representing the S3 path for the base dataset in the format:
            {user_id}/foundation_models/{dataset_name}
        """
        return f"{user_id}/foundation_models/{dataset_name}"

    @staticmethod
    def validate_filter_columns(
        df: pd.DataFrame, filters: GroupLabelColumnFilters
    ) -> tuple[bool, Optional[list[str]]]:
        """
        Validates if all columns specified in the filters exist in the DataFrame.

        Args:
            df: The pandas DataFrame to check against
            filters: GroupLabelColumnFilters instance containing filter criteria

        Returns:
            A tuple containing:
            - bool: True if all filter columns exist in the DataFrame, False otherwise
            - list[str]: List of missing column names if any exist, None if all columns exist
        """

        if not filters.filter:
            logger.info("No filters provided, validation passed")
            return True, None

        df_columns = set(df.columns)
        missing_columns = set()

        # Check each filter group
        for filter_dict in filters.filter:
            for column in filter_dict.keys():
                if column not in df_columns:
                    missing_columns.add(column)

        if missing_columns:
            logger.warning(
                f"These columns do not exist in the dataframe: {sorted(missing_columns)}"
            )
            return False, sorted(missing_columns)

        logger.info("All filter columns exist in DataFrame, validation passed")
        return True, None

    @staticmethod
    def generate_filter_combinations(
        group_filters: "GroupLabelColumnFilters",
    ) -> list[list[dict[str, list]]]:
        """
        Given a GroupLabelColumnFilters instance, generate all possible combinations
        of filter values as a list of lists of single-key filter dicts suitable for use with filter_and_aggregate_dataframe.
        Each inner list represents a combination, and each dict is a single column-value pair.
        Example output:
        [
          [{'A': [1]}, {'B': [3]}],
          [{'A': [1]}, {'B': [4]}],
          [{'A': [2]}, {'B': [3]}],
          [{'A': [2]}, {'B': [4]}],
        ]
        """

        # Merge all filter dicts into one, deduplicating values
        merged = {}
        for d in group_filters.filter:
            for k, v in d.items():
                merged.setdefault(k, set()).update(v)
        # Sort for reproducibility
        keys = sorted(merged)
        values_lists = [sorted(merged[k]) for k in keys]
        # Generate all combinations as list of lists of single-key dicts
        combos = [
            [{k: [v]} for k, v in zip(keys, values)]
            for values in product(*values_lists)
        ]
        return combos
