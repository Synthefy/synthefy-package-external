"""
This file contains the WindowPreprocessor class for preprocessing data windows for Foundation Models.

The class takes a pandas DataFrame with the following columns:
- Timestamps
- Group Label
- Timeseries Columns
- Continuous Columns
- Textual Metadata Column
- Text Description Column

And preprocesses them into numpy arrays with the following transformations:

Timestamp column:
- Converted to normalized values between 0-1 representing relative position in window.
- Shape: (window_size,)

Dataset description:
- Text embedded using sentence transformer
- Shape: (text_embedding_dim,)

Text description column:
- Text embedded using sentence transformer
- Missing values filled with missing_text_value=0
- Shape: (text_embedding_dim,)

Continuous columns:
- Column names embedded using sentence transformer
- Values normalized using StandardScaler
- Missing values filled with null_value
- Fixed output size regardless of input columns: shape (N_EXOGENOUS*(text_embedding_dim + window_size),)
- Missing columns filled with null_value

Timeseries columns:
- Values normalized using StandardScaler
- Missing values filled with null_value
- Shape: (num_channels, window_size)

Retrieved timeseries:
- Similar to continuous columns but with different fixed size
- Shape: (N_RETRIEVED_TIMESERIES*(text_embedding_dim + window_size),)

Textual metadata column:
- Text embedded using sentence transformer
- Missing values filled with missing_text_value=0
- Shape: (window_size * text_embedding_dim,)

The preprocess_window() method returns:
- Dictionary containing:
  - timeseries: Numpy array with timeseries data
  - metadata: Numpy array with combined metadata (timestamp, dataset_description, text_description,
    continuous, retrieved_timeseries, time_varying_textual_metadata)
  - scalers: Dictionary of fitted scalers for each numeric column (timeseries and continuous columns)

The class supports configurable metadata types through the metadata_types_to_use parameter,
allowing users to include or exclude specific metadata components.
"""

import datetime
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from einops import rearrange
from loguru import logger
from sklearn.preprocessing import StandardScaler

from synthefy_pkg.preprocessing.fm_text_embedder import TEXT_EMBEDDING_DIM
from synthefy_pkg.utils.fm_utils import get_columns_info_dict_from_metadata_json

TEMP_FIX = True
# Length of the exogenous features
N_EXOGENOUS = 100

# Length of the retrieved timeseries
N_RETRIEVED_TIMESERIES = 10

# Dimension of the text embeddings (sentence transformer)
NULL_VALUE = -999999.0

TIMESTAMPS_FEATURES = ["year", "month", "day", "hour", "minute"]


class WindowPreprocessor:
    def __init__(
        self,
        window_size: int,
        metadata: Dict[str, Any],
        metadata_types_to_use: List[str] = [
            "timestamp",
            "dataset_description",
            "text_description",
            "continuous",
            "retrieved_timeseries",
            "time_varying_textual_metadata",
        ],
        verbose: bool = False,
    ):
        """Initialize the window preprocessor with column names to process.

        Args:
            window_size: Size of the window to process
            metadata: Metadata dictionary containing column information with keys like:
                - "columns": List of dictionaries with column information, each containing:
                    - "type": Column type (e.g., "continuous", "text")
                    - "column_id": Name of the column
                    - "description" or "title": Description of the column
                    - "is_metadata": flag indicating if column is metadata
                - "timestamps_columns": List of timestamp column names
                - "title": Dataset title/description
                - "text_description_column": Name of text description column
                - "textual_metadata_column": Name of textual metadata column
                Example:
                {
                    "title": "Weather Data 2020-2022",
                    "timestamps_columns": ["date"],
                    "text_description_column": "forecast_summary",
                    "textual_metadata_column": "weather_conditions",
                    "columns": [
                        {
                            "type": "continuous",
                            "column_id": "temperature",
                            "description": "Temperature in Celsius",
                            "is_metadata": "yes"
                        },
                        {
                            "type": "continuous",
                            "column_id": "humidity",
                            "description": "Humidity percentage"
                            "is_metadata": "no"
                        }
                    ]
                }
            metadata_types_to_use: List of metadata types to include in the metadata vector
            verbose: Whether to log timing information
        """
        self.columns_info_dict = get_columns_info_dict_from_metadata_json(
            metadata, v1=True
        )

        self.window_size = window_size
        self.timestamp_column = (
            metadata["timestamps_columns"][0]
            if "timestamps_columns" in metadata
            else None
        )

        self.timeseries_columns = list(
            self.columns_info_dict.get("timeseries", {}).keys()
        )
        self.continuous_columns = list(
            self.columns_info_dict.get("continuous", {}).keys()
        )

        self.textual_metadata_column = metadata.get(
            "textual_metadata_column", None
        )
        self.dataset_description = metadata.get("title", None)
        self.text_description_column = metadata.get(
            "text_description_column", None
        )
        self.null_value = NULL_VALUE
        self.metadata_types_to_use = metadata_types_to_use
        self.verbose = verbose

        # Define mapping of metadata types to their length contributions
        self.metadata_type_to_length = {
            "timestamp": (
                self.window_size * len(TIMESTAMPS_FEATURES)
                if "timestamp" in self.metadata_types_to_use
                else 0
            ),
            "dataset_description": (
                TEXT_EMBEDDING_DIM
                if "dataset_description" in self.metadata_types_to_use
                else 0
            ),
            # TODO: phase 2 - TEXT_EMBEDDING_DIM * self.window_size
            "text_description": (
                TEXT_EMBEDDING_DIM
                if "text_description" in self.metadata_types_to_use
                else 0
            ),
            "continuous": (
                N_EXOGENOUS * (TEXT_EMBEDDING_DIM + window_size)
                if "continuous" in self.metadata_types_to_use
                else 0
            ),
            "retrieved_timeseries": (
                N_RETRIEVED_TIMESERIES * (TEXT_EMBEDDING_DIM + window_size)
                if "retrieved_timeseries" in self.metadata_types_to_use
                else 0
            ),
            "time_varying_textual_metadata": (
                TEXT_EMBEDDING_DIM * self.window_size
                if "time_varying_textual_metadata" in self.metadata_types_to_use
                else 0
            ),
        }
        self.metadata_vector_length, self.metadata_type_to_indices_slice = (
            self.calculate_metadata_vector_length()
        )

    def calculate_metadata_vector_length(
        self,
    ) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Calculate the length of the metadata vector."""
        metadata_vector_length = 0

        metadata_type_to_indices_slice = {}
        # Calculate total length by iterating through metadata types to use
        for metadata_type in self.metadata_types_to_use:
            if self.metadata_type_to_length[metadata_type] > 0:
                length_contribution = self.metadata_type_to_length[
                    metadata_type
                ]
                metadata_vector_length += length_contribution
                if self.verbose:
                    logger.debug(
                        f"Metadata type '{metadata_type}' contributes {length_contribution} to vector length"
                    )
                metadata_type_to_indices_slice[metadata_type] = {
                    "start_idx": metadata_vector_length - length_contribution,
                    "end_idx": metadata_vector_length,
                }

        return metadata_vector_length, metadata_type_to_indices_slice

    def normalize_numeric_columns(
        self, df: pd.DataFrame, columns: list, impute_nan: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize numeric columns and fit new scalers.

        This method handles several scenarios:
        - Fits new StandardScaler for each column
        - All-NaN columns: Sets scaler to None and fills with null_value
        - Partial NaN values: Fits scaler on non-NaN values, fills NaNs with null_value

        Args:
            df: Input dataframe containing numeric columns to normalize
            columns: List of column names to normalize
            impute_nan: Whether to impute NaN values with null_value

        Returns:
            Tuple containing:
            - DataFrame with normalized columns (NaN values replaced with null_value)
            - Dictionary mapping column names to fitted StandardScaler objects or None
        """
        result_df = df.copy()
        scalers = {}

        for col in columns:
            # check if more than 1 col starts with the same name
            # TODO - REMOVE THIS. currently bug in standardized data
            if (
                TEMP_FIX
                and (col + "_x" in df.columns)
                and (col + "_y" in df.columns)
            ):
                col = col + "_x"

            # Get original values and reshape using einops
            original_values = df[col].to_numpy()
            original_values_2d = rearrange(original_values, "W -> W 1")

            # If all values are NaN, fill with null_value and store None in scalers
            if np.isnan(original_values).all():
                result_df[col] = self.null_value
                scalers[col] = None
                continue

            # Create new scaler
            scaler = StandardScaler()
            # Fit on non-missing values only
            non_missing_mask = ~np.isnan(original_values)
            if np.any(non_missing_mask):
                non_missing_values = original_values[non_missing_mask]
                non_missing_values_2d = rearrange(
                    non_missing_values, "W -> W 1"
                )
                scaler.fit(non_missing_values_2d)
            scalers[col] = scaler

            # Transform values if scaler exists
            if scalers[col] is not None:
                transformed = scalers[col].transform(original_values_2d)
                transformed = rearrange(transformed, "W 1 -> W")
                result_df[col] = pd.Series(transformed)
                # Use null_value for missing values
                if impute_nan:
                    result_df[col] = result_df[col].fillna(self.null_value)
            else:
                result_df[col] = (
                    self.null_value if impute_nan else original_values
                )

        return result_df, scalers

    def load_scalers(self, path: str) -> Dict:
        """Load scalers from pickle file.

        Args:
            path: Path to load scalers from

        Returns:
            Dictionary of loaded scalers
        """
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Scalers file not found at {path}")

    def _pad_array_to_size(
        self,
        array: np.ndarray,
        expected_size: int,
        pad_value: Optional[float] = None,
        pad_from_start: bool = False,
    ) -> np.ndarray:
        """Pad array with padding values to reach expected size.

        Args:
            array: Input numpy array
            expected_size: Expected size of the output array
            pad_value: Value to use for padding. If None, uses self.null_value
            pad_from_start: If True, pad from the start of the array. If False, pad from the end.
                pad_from_start is used when windows are < MODEL_WINDOW_SIZE/DEFAULT_WINDOW_SIZE
        Returns:
            Padded numpy array with padding values
        """
        if len(array) == expected_size:
            return array

        if pad_value is None:
            pad_value = self.null_value

        padding = np.full(expected_size - len(array), pad_value)
        if pad_from_start:
            return np.concatenate([padding, array])
        else:
            return np.concatenate([array, padding])

    def _pad_numeric_array_to_size(
        self,
        array: np.ndarray,
        expected_size: int,
        elements_per_column: int,
        column_embeddings_dim: int,
    ) -> np.ndarray:
        """Pad numeric array with a pattern of embeddings and values to reach expected size.

        Args:
            array: Input 1D numpy array
            expected_size: Expected size of the output array
            elements_per_column: Number of elements per column (embedding_dim + window_size)
            column_embeddings_dim: Dimension of column embeddings

        Returns:
            Padded numpy array with pattern of missing_text_value for embeddings and null_value for values
        """
        if len(array) >= expected_size:
            return array[:expected_size]

        # Calculate how many complete columns we need to add
        remaining_elements = expected_size - len(array)
        complete_columns_to_add = remaining_elements // elements_per_column

        padding = []
        for _ in range(complete_columns_to_add):
            # Add embedding padding (pad with zeros)
            padding.append(np.zeros(column_embeddings_dim))
            # Add values padding (pad with null_value)
            padding.append(np.full(self.window_size, self.null_value))

        result = np.concatenate([array] + padding)
        assert len(result) == expected_size, (
            f"Padded array length {len(result)} does not match expected size {expected_size}"
        )
        return result

    def _validate_metadata_exists(
        self,
        metadata_type: str,
        columns: Optional[Union[str, List[str]]],
        df: pd.DataFrame,
    ) -> None:
        """Validate that the metadata type exists in the dataframe.

        Args:
            metadata_type: Type of metadata to validate
            columns: Name of the column or list of columns to validate in the df
            df: DataFrame to check for columns

        Raises:
            ValueError if metadata type is in metadata_types_to_use but column(s) not found in dataframe
        """
        if metadata_type not in self.metadata_types_to_use:
            return

        if columns is None:
            return

        if isinstance(columns, str):
            # Single column case
            if columns not in df.columns:
                raise ValueError(
                    f"Metadata type '{metadata_type}' specified but column '{columns}' not found in dataframe"
                )
        elif isinstance(columns, list):
            if len(columns) == 0:
                raise ValueError(
                    f"Metadata type '{metadata_type}' specified but no columns provided"
                )
            # List of columns case
            for col in columns:
                if (
                    TEMP_FIX
                    and (col + "_x" in df.columns)
                    and (col + "_y" in df.columns)
                ):
                    col = col + "_x"
                if col not in df.columns and metadata_type:
                    raise ValueError(
                        f"Metadata type '{metadata_type}' specified but column '{col}' not found in dataframe"
                    )

    def _process_timeseries_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process timeseries columns by normalizing values.

        Args:
            df: Input dataframe containing the data to process
            columns: List of column names to process

        Returns:
            A tuple containing:
            - Processed numpy array with shape (num_columns, window_size)
            - Dictionary of scalers used for normalization
        """
        # If no columns provided, return empty array and empty scalers dict
        if not columns:
            return np.array([]), {}

        # Check if all columns exist in the dataframe
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Columns {missing_columns} not found in dataframe"
            )

        timeseries_start = datetime.datetime.now()

        # Normalize columns
        normalized_df, scalers = self.normalize_numeric_columns(
            df, columns, impute_nan=False
        )

        processed_data = []
        # Process each column
        for col in columns:
            # check if more than 1 col starts with the same name
            # TODO - REMOVE THIS. currently bug in standardized data
            if (
                TEMP_FIX
                and (col + "_x" in df.columns)
                and (col + "_y" in df.columns)
            ):
                col = col + "_x"

            # Get normalized values
            normalized_values = normalized_df[col].to_numpy()
            if len(normalized_values) < self.window_size:
                normalized_values = self._pad_array_to_size(
                    normalized_values, self.window_size, pad_from_start=True
                )

            processed_data.append(normalized_values)

        # Concatenate the processed data if any exists
        if processed_data:
            result = np.array(processed_data)
        else:
            result = np.array([])

        if self.verbose:
            elapsed = (
                datetime.datetime.now() - timeseries_start
            ).total_seconds()
            logger.debug(f"Timeseries processing took {elapsed:.4f}s")

        return result, scalers

    def _process_exogenous_retrieved_columns(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]],
        metadata_type: Literal["continuous", "retrieved_timeseries"],
        column_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process numeric columns by embedding column names and normalizing values.

        This method processes the specified columns, creating a normalized
        representation that includes both column name embeddings and their corresponding values.

        Args:
            df: Input dataframe containing the data to process
            columns: List of column names to process
            metadata_type: Type of metadata being processed ("continuous" or "retrieved_timeseries")
            column_embeddings: Embeddings for the column names with shape (num_columns, TEXT_EMBEDDING_DIM)
        Returns:
            A tuple containing:
            - Processed numpy array with embeddings and values interleaved in the format:
              [col1_embedding, col1_values, col2_embedding, col2_values, ...]
            - Dictionary of scalers used for normalization

        Notes:
            - Column name embeddings are created using the sentence transformer
            - Values are normalized using newly fitted scalers
            - Missing values and values for columns with None scalers are filled with null_value
            - If metadata_type is not in metadata_types_to_use or columns is None, returns array of null_values
        """
        self._validate_metadata_exists(metadata_type, columns, df)
        if self.metadata_type_to_length[metadata_type] == 0:
            return (
                np.full(
                    self.metadata_type_to_length[metadata_type], self.null_value
                ),
                {},
            )

        expected_size = self.metadata_type_to_length[metadata_type]
        if columns is None:
            return (
                self._pad_numeric_array_to_size(
                    np.array([]),
                    expected_size,
                    elements_per_column=TEXT_EMBEDDING_DIM + self.window_size,
                    column_embeddings_dim=TEXT_EMBEDDING_DIM,
                ),
                {},
            )

        start_time = datetime.datetime.now()
        processed_data = []

        # Calculate how many columns we need to process to reach expected_size
        cols_to_process = min(len(columns), N_EXOGENOUS)
        # Normalize columns
        normalized_df, scalers = self.normalize_numeric_columns(
            df, columns[:cols_to_process], impute_nan=True
        )

        if cols_to_process < len(columns):
            logger.warning(
                f"Only processing {cols_to_process} of {len(columns)} columns to fit within expected size {expected_size}."
            )

        # Process only the necessary columns
        for i, col in enumerate(columns[:cols_to_process]):
            # check if more than 1 col starts with the same name
            # TODO - REMOVE THIS. currently bug in standardized data
            if (
                TEMP_FIX
                and (col + "_x" in df.columns)
                and (col + "_y" in df.columns)
            ):
                col = col + "_x"

            # Get column name embedding
            one_col_embedding = column_embeddings[i]

            # Get normalized values
            normalized_values = normalized_df[col].to_numpy()
            normalized_values = self._pad_array_to_size(
                normalized_values, self.window_size, pad_from_start=True
            )

            processed_data.append(one_col_embedding)
            processed_data.append(normalized_values)

        # First concatenate the processed data we have
        if processed_data:
            result = np.concatenate(processed_data)
        else:
            result = np.array([])

        # Pad with pattern of embeddings and values if needed to reach expected size
        result = self._pad_numeric_array_to_size(
            result,
            expected_size,
            elements_per_column=TEXT_EMBEDDING_DIM + self.window_size,
            column_embeddings_dim=TEXT_EMBEDDING_DIM,
        )

        if self.verbose:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            logger.debug(f"{metadata_type} processing took {elapsed:.4f}s")
        return result, scalers

    def _process_time_varying_text_metadata(
        self,
        df: pd.DataFrame,
        column: Optional[str],
        time_varying_text_embedding: np.ndarray,
    ) -> np.ndarray:
        """Process time varying text metadata column by embedding values.

        Args:
            df: Input dataframe
            column: Name of the metadata text column
            time_varying_text_embedding: Embeddings for the metadata text column
                with shape (window_size, TEXT_EMBEDDING_DIM)
        Returns:
            Processed numpy array with text embeddings as a 1D array with shape (TEXT_EMBEDDING_DIM * window_size,)
        """
        self._validate_metadata_exists(
            "time_varying_textual_metadata", column, df
        )
        if (
            self.metadata_type_to_length["time_varying_textual_metadata"] == 0
            or column is None
        ):
            return np.zeros(
                self.metadata_type_to_length["time_varying_textual_metadata"],
            )

        if len(time_varying_text_embedding) == 0:
            raise ValueError(
                "Time varying text embedding is empty. Please check the input data."
            )
        textual_metadata_start = datetime.datetime.now()
        # Flatten the 2D array to 1D
        metadata = rearrange(
            time_varying_text_embedding,
            "W TEXT_EMBEDDING_DIM -> (W TEXT_EMBEDDING_DIM)",
        )
        if self.verbose:
            elapsed = (
                datetime.datetime.now() - textual_metadata_start
            ).total_seconds()
            logger.debug(f"Textual metadata processing took {elapsed:.4f}s")

        return metadata

    def _process_text_description_column(
        self,
        df: pd.DataFrame,
        column: Optional[str],
        text_description_embedding: np.ndarray,
    ) -> np.ndarray:
        """Process text description column by embedding values.

        Args:
            df: Input window dataframe
            column: Name of the text description column
            text_description_embedding: Embeddings for the text description column
                with shape (window_size, TEXT_EMBEDDING_DIM)
        Returns:
            Processed numpy array with text embeddings as a 1D array with
                shape (TEXT_EMBEDDING_DIM * window_size,)
        """
        self._validate_metadata_exists("text_description", column, df)
        if (
            self.metadata_type_to_length["text_description"] == 0
            or column is None
        ):
            return np.zeros(
                self.metadata_type_to_length["text_description"],
            )

        if len(text_description_embedding) == 0:
            raise ValueError(
                "Text description embedding is empty. Please check the input data."
            )
        text_description_start = datetime.datetime.now()
        text_description = text_description_embedding
        # TODO: use whole column
        # text_description = rearrange(
        #     text_description_embedding,
        #     "W TEXT_EMBEDDING_DIM -> (W TEXT_EMBEDDING_DIM)",
        # )

        if self.verbose:
            elapsed = (
                datetime.datetime.now() - text_description_start
            ).total_seconds()
            logger.debug(f"Text description processing took {elapsed:.4f}s")

        return text_description

    def _get_timestamp_array(
        self, df: pd.DataFrame, timestamp_column: str | None
    ) -> np.ndarray:
        """Extract yearly, monthly, daily, hourly and minutely features from timestamps.

        Args:
            df: Input dataframe
            timestamp_column: Name of the timestamp column

        Returns:
            Normalized timestamp features as a 1D numpy array with shape (window_size * 5,)
            containing yearly, monthly, daily, hourly and minutely features,
            or array of null_values if timestamp processing is disabled or column is None
        """
        self._validate_metadata_exists("timestamp", timestamp_column, df)

        if (
            self.metadata_type_to_length["timestamp"] == 0
            or timestamp_column is None
        ):
            return np.full(
                self.metadata_type_to_length["timestamp"], self.null_value
            )

        timestamp_start = datetime.datetime.now()
        # Convert to datetime and extract datetime components
        timestamp_values = pd.to_datetime(df[timestamp_column], format="mixed")

        # Extract features using TIMESTAMPS_FEATURES constant
        feature_values = []
        for feature in TIMESTAMPS_FEATURES:
            # Use dt accessor to get datetime components
            feature_values.append(getattr(timestamp_values.dt, feature).values)

        # MinMax normalize with fixed ranges
        normalization_ranges = {
            "year": (1980, 2030),
            "month": (1, 12),
            "day": (1, 31),
            "hour": (0, 23),
            "minute": (0, 59),
        }

        normalized_features = []
        for feature, values in zip(TIMESTAMPS_FEATURES, feature_values):
            if feature == "year":
                norm_values = (values - normalization_ranges[feature][0]) / (
                    normalization_ranges[feature][1]
                    - normalization_ranges[feature][0]
                )
            elif feature in ["month", "day"]:
                norm_values = (values - 1) / (
                    normalization_ranges[feature][1] - 1
                )
            else:
                norm_values = values / normalization_ranges[feature][1]

            padded = self._pad_array_to_size(
                norm_values, self.window_size, pad_from_start=True
            )
            normalized_features.append(padded)

        # Concatenate all features
        timestamp_features = np.concatenate(normalized_features)

        if self.verbose:
            elapsed = (
                datetime.datetime.now() - timestamp_start
            ).total_seconds()
            logger.debug(f"Timestamp processing took {elapsed:.4f}s")

        return timestamp_features

    def _process_dataset_description(
        self,
        dataset_description: Optional[str],
        dataset_description_embedding: np.ndarray,
    ) -> np.ndarray:
        """Process dataset description by embedding values.

        Args:
            dataset_description: Dataset description
            dataset_description_embedding: Embedding of dataset description

        Returns:
            Processed numpy array with text embeddings as a 1D array with shape (TEXT_EMBEDDING_DIM,)
        """
        if self.verbose:
            dataset_description_start = datetime.datetime.now()
        if (
            dataset_description is None
            or self.metadata_type_to_length["dataset_description"] == 0
        ):
            return np.zeros(self.metadata_type_to_length["dataset_description"])

        # Check for empty array and correct dimension in one go
        if len(dataset_description_embedding) != TEXT_EMBEDDING_DIM:
            raise ValueError(
                f"Dataset description embedding dimension {len(dataset_description_embedding)} "
                f"does not match TEXT_EMBEDDING_DIM {TEXT_EMBEDDING_DIM}. "
                "Please check the input data."
            )

        if self.verbose:
            elapsed = (
                datetime.datetime.now() - dataset_description_start
            ).total_seconds()
            logger.debug(f"Dataset description processing took {elapsed:.4f}s")

        return dataset_description_embedding

    def _combine_metadata(
        self,
        timestamp_array: np.ndarray,
        dataset_description_array: np.ndarray,
        text_description_array: np.ndarray,
        continuous_array: np.ndarray,
        retrieved_ts_array: np.ndarray,
        time_varying_textual_metadata_array: np.ndarray,
    ) -> np.ndarray:
        """Combine different types of metadata into a single array.

        Args:
            timestamp_array: Array of normalized timestamps
            dataset_description_array: Embedding of dataset description
            text_description_array: Embedding of text descriptions
            continuous_array: Array of continuous features
            retrieved_ts_array: Array of retrieved timeseries features
            time_varying_textual_metadata_array: Array of textual metadata

        Returns:
            Combined metadata as a single 1D numpy array with length matching self.metadata_vector_length
        """
        combine_start = datetime.datetime.now()
        # Filter out None values and only include metadata types specified in metadata_types_to_use
        arrays_to_combine = []
        arrays_to_combine.append(timestamp_array)
        arrays_to_combine.append(dataset_description_array)
        arrays_to_combine.append(text_description_array)
        arrays_to_combine.append(continuous_array)
        arrays_to_combine.append(retrieved_ts_array)
        arrays_to_combine.append(time_varying_textual_metadata_array)

        # Concatenate arrays if any exist
        output_array = np.concatenate(arrays_to_combine)
        if self.verbose:
            elapsed = (datetime.datetime.now() - combine_start).total_seconds()
            logger.debug(f"Combining metadata took {elapsed:.4f}s")
        return output_array

    def preprocess_window(
        self,
        window: pd.DataFrame,
        continuous_embeddings: np.ndarray,
        retrieved_ts_embeddings: np.ndarray,
        dataset_description_embedding: np.ndarray,
        text_description_embedding: np.ndarray,
        time_varying_text_embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """Preprocess a window of data.

        This method processes a window of data by:
        1. Processing timestamps into normalized values
        2. Processing timeseries columns with normalization
        3. Processing continuous columns with normalization and embedding
        4. Processing retrieved timeseries (if applicable)
        5. Processing textual metadata
        6. Processing text descriptions
        7. Embedding dataset description
        8. Combining all metadata into a single array

        Args:
            window: DataFrame containing the window data
            continuous_embeddings: Embeddings for continuous column names (num_continuous_cols, TEXT_EMBEDDING_DIM)
            retrieved_ts_embeddings: Embeddings for retrieved timeseries column names (num_retrieved_cols, TEXT_EMBEDDING_DIM)
            dataset_description_embedding: Embedding for dataset description (TEXT_EMBEDDING_DIM,)
            text_description_embedding: Pre-computed embeddings for text description column
                for this specific window, shape (window_size, TEXT_EMBEDDING_DIM) or empty array
            time_varying_text_embedding: Pre-computed embeddings for textual metadata column
                for this specific window, shape (window_size, TEXT_EMBEDDING_DIM) or empty array

        Returns:
            Dictionary containing:
            - timeseries: Numpy array with timeseries data
            - metadata: Numpy array with combined metadata (timestamp, dataset_description, text_description,
              continuous, retrieved_timeseries, time_varying_textual_metadata)
            - scalers: Dictionary of fitted scalers for each numeric column (timeseries and continuous columns)
        """

        # Reset index to ensure proper alignment
        window = window.reset_index(drop=True)
        start_time = datetime.datetime.now()
        readable_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        if self.verbose:
            logger.info(
                f"Preprocessing window of size {len(window)} starting at {readable_time}"
            )
        # Process timestamp using the normalization function
        # TODO - handle missing timestamps!
        timestamp_array = self._get_timestamp_array(
            df=window,
            timestamp_column=self.timestamp_column,
        )

        # Process timeseries columns
        timeseries_array, timeseries_scalers = self._process_timeseries_columns(
            df=window,
            columns=self.timeseries_columns,
        )
        # Process continuous columns
        continuous_array, continuous_scalers = (
            self._process_exogenous_retrieved_columns(
                df=window,
                columns=self.continuous_columns,
                metadata_type="continuous",
                column_embeddings=continuous_embeddings,
            )
        )

        # Process retrieved timeseries
        retrieved_ts_array, retrieved_ts_scalers = (
            self._process_exogenous_retrieved_columns(
                df=window,
                columns=None,
                metadata_type="retrieved_timeseries",
                column_embeddings=retrieved_ts_embeddings,
            )
        )

        # Organize scalers by type to prevent key collisions
        scalers = {
            "timeseries": timeseries_scalers,
            "continuous": continuous_scalers,
            "retrieved_timeseries": retrieved_ts_scalers,
        }

        # Process time varying textual metadata using the helper function
        time_varying_textual_metadata_array = (
            self._process_time_varying_text_metadata(
                window,
                self.textual_metadata_column,
                time_varying_text_embedding,
            )
        )

        # Process text description column using the helper function
        text_description_array = self._process_text_description_column(
            window, self.text_description_column, text_description_embedding
        )

        # Embed dataset description and duplicate for each row in the window
        dataset_description_array = self._process_dataset_description(
            dataset_description=self.dataset_description,
            dataset_description_embedding=dataset_description_embedding,
        )

        # Create the metadata array by combining all components
        metadata_array = self._combine_metadata(
            timestamp_array=timestamp_array,
            dataset_description_array=dataset_description_array,
            text_description_array=text_description_array,
            continuous_array=continuous_array,
            retrieved_ts_array=retrieved_ts_array,
            time_varying_textual_metadata_array=time_varying_textual_metadata_array,
        )

        assert metadata_array.shape[0] == self.metadata_vector_length, (
            f"Metadata array shape: {metadata_array.shape[0]} != metadata vector length: {self.metadata_vector_length}"
        )

        if self.verbose:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Total preprocessing time: {elapsed:.4f}s")

        # Reshape metadata_array to have shape (1, length)
        metadata_array = metadata_array.reshape(1, -1)
        # Return the dictionary with the required format
        return {
            "timeseries": timeseries_array,
            "metadata": metadata_array,
            "scalers": scalers,
        }
