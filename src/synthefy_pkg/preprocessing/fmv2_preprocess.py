"""
Foundation Model Window Preprocessor

This module provides the FMV2_Preprocessor class for transforming time-series data windows
into numerical representations suitable for foundation models. It handles multiple data types
and creates fixed-size representations for each window.

Input:
- pandas DataFrame with timestamps and timeseries data (only 2 columns always: timestamp and value)

Output Features:
- Timestamps: Normalized temporal features extracted from timestamps (position in window)
- Timeseries data: Normalized using StandardScaler with consistent dimensionality
- Retrieved timeseries: Similar context timeseries with consistent dimensionality
- Time-varying textual metadata: Text embedded via transformer with consistent dimensionality

The preprocessor handles missing values, applies appropriate normalization, and ensures
consistent output shapes regardless of input variations. Output is structured as a dictionary
containing the processed data arrays and fitted scalers for later inverse transformations.
"""

import argparse
import datetime
import glob
import multiprocessing
import os

# Add pickle import
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import holidays
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from filelock import FileLock
from holidays import HolidayBase
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from synthefy_pkg.preprocessing.fm_text_embedder import TEXT_EMBEDDING_DIM
from synthefy_pkg.utils.fm_utils import (
    get_columns_info_dict_from_metadata_json,
    load_dataframe_from_directory,
    load_metadata_from_directory,
)

SCALING_FEATURES = [
    "mean",
    "var",
    "scale",
]

TIMESTAMPS_FEATURES = [
    "year",
    "quarter",
    "month",
    "week",
    "dayofweek",
    "day",
    "hour",
    "minute",
    "second",
    "holiday",
    "timezone",
]

NORM_RANGES = {
    "year": (1970, 2030),
    "quarter": (1, 4),
    "month": (1, 12),
    "week": (1, 53),
    "dayofweek": (0, 6),
    "day": (1, 31),
    "hour": (0, 23),
    "minute": (0, 59),
    "second": (0, 59),
    "timezone": (-12, 14),  # Typical timezone range in hours
    # Binary features don't need ranges
}


def convert_time_to_vector(
    result: np.ndarray,
    features: list[str],
    norm_ranges: dict,
    timestamp_series: pd.Series,
    holidays: HolidayBase,
    timezone_offset: float,
    nan_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert time values to a vector representation
    """
    # Extract and normalize each feature
    for i, feature in enumerate(features):
        if feature == "holiday":
            # Check if date is a US holiday
            holiday_flags = np.array(
                [
                    1.0 if pd.Timestamp(ts).date() in holidays else 0.0
                    for ts in timestamp_series
                ]
            )
            result[:, i] = holiday_flags
        elif feature == "timezone":
            # Add timezone offset feature
            tz_values = np.full(len(timestamp_series), timezone_offset)
            min_val, max_val = norm_ranges[feature]
            # Normalize to 0-1 range
            result[:, i] = (tz_values - min_val) / (max_val - min_val)
        elif feature == "week":
            # Use isocalendar().week instead of weekofyear
            values = timestamp_series.dt.isocalendar().week.values.astype(float)
            max_val = norm_ranges[feature][1]
            result[:, i] = (values - 1) / (max_val - 1)
        elif feature == "second":
            # Extract seconds
            values = timestamp_series.dt.second.values.astype(float)
            max_val = norm_ranges[feature][1]
            result[:, i] = values / max_val
        else:
            # Numeric features
            values = getattr(timestamp_series.dt, feature).values.astype(float)

            if feature in norm_ranges:
                # Normalize based on feature type
                if feature == "year":
                    min_val, max_val = norm_ranges[feature]
                    result[:, i] = (values - min_val) / (max_val - min_val)
                elif feature in ["month", "day", "quarter"]:
                    # These features start at 1, normalize to 0-1 range
                    max_val = norm_ranges[feature][1]
                    result[:, i] = (values - 1) / (max_val - 1)
                else:
                    # These features start at 0, normalize to 0-1 range
                    max_val = norm_ranges[feature][1]
                    result[:, i] = values / max_val

        # Apply the NaN mask to preserve original NaNs
        result[nan_mask, i] = np.nan
    return result


def _process_single_subdir(args: Tuple[str, str, int, int, bool]) -> None:
    """Process a single subdirectory of time series data.

    Args:
        args: Tuple containing (subdir_path, dataset_name, window_size, stride, verbose)
    """
    subdir, dataset_name, window_size, stride, verbose = args
    try:
        # Extract dataset ID from directory name
        dir_basename = os.path.basename(subdir)
        match = re.search(f"{dataset_name}_(\d+)$", dir_basename)
        if not match:
            logger.error(f"Could not extract dataset ID from {dir_basename}")
            return

        # Load metadata specific to this subdirectory
        try:
            metadata = load_metadata_from_directory(subdir)
            if verbose:
                logger.info(f"Loaded metadata from {subdir}")
        except Exception as e:
            logger.error(f"Failed to load metadata from {subdir}: {str(e)}")
            return

        # Load dataframe
        try:
            df = load_dataframe_from_directory(subdir)
            if len(df) == 0:
                logger.error(f"Empty dataframe loaded from {subdir}")
                return
        except Exception as e:
            logger.error(f"Failed to load dataframe from {subdir}: {str(e)}")
            return

        # Create preprocessor instance for this subdirectory
        preprocessor = FMV2_Preprocessor(
            window_size=window_size,
            stride=stride,
            metadata=metadata,
            data_dir=subdir,
            verbose=verbose,
        )

        # Process the data and save to the subdirectory
        processed_array = preprocessor.preprocess_df(df)
        output_path = os.path.join(subdir, "preprocessed_array.npy")
        lock_path = output_path + ".lock"

        # Use file lock for thread-safe saving
        with FileLock(lock_path):
            np.save(output_path, processed_array)

        if verbose:
            logger.info(f"Successfully processed and saved data for {subdir}")

    except Exception as e:
        logger.error(f"Error processing {subdir}: {str(e)}")


def inverse_time_vector(
    vectorized_timestamps: np.ndarray,
    features: list[str] = TIMESTAMPS_FEATURES,
    norm_ranges: dict = NORM_RANGES,
    default_year: int = 2000,
    default_month: int = 1,
    default_day: int = 1,
    default_hour: int = 0,
    default_minute: int = 0,
    default_second: int = 0,
) -> np.ndarray:
    """
    Inverse of convert_time_to_vector.
    Args:
        vectorized_timestamps: np.ndarray of shape (T, len(features)) or (batch_size, T, len(features))
    Returns:
        np.ndarray of shape (T,) with pd.Timestamp objects, or (batch_size, T) for batched input
    """
    # Ensure input is a numpy array (handle torch.Tensor input)
    try:
        import torch

        if isinstance(vectorized_timestamps, torch.Tensor):
            vectorized_timestamps = vectorized_timestamps.detach().cpu().numpy()
    except ImportError:
        pass
    vectorized_timestamps = np.asarray(vectorized_timestamps)

    if vectorized_timestamps.ndim == 2:
        T, num_feat = vectorized_timestamps.shape
        timestamps = np.empty((T,), dtype=object)
        feature_idx = {f: i for i, f in enumerate(features)}

        years = np.round(
            vectorized_timestamps[:, feature_idx["year"]]
            * (norm_ranges["year"][1] - norm_ranges["year"][0])
            + norm_ranges["year"][0]
        ).astype(int)
        months = np.round(
            vectorized_timestamps[:, feature_idx["month"]]
            * (norm_ranges["month"][1] - 1)
            + 1
        ).astype(int)
        days = np.round(
            vectorized_timestamps[:, feature_idx["day"]]
            * (norm_ranges["day"][1] - 1)
            + 1
        ).astype(int)
        hours = np.round(
            vectorized_timestamps[:, feature_idx["hour"]]
            * norm_ranges["hour"][1]
        ).astype(int)
        minutes = np.round(
            vectorized_timestamps[:, feature_idx["minute"]]
            * norm_ranges["minute"][1]
        ).astype(int)
        seconds = np.round(
            vectorized_timestamps[:, feature_idx["second"]]
            * norm_ranges["second"][1]
        ).astype(int)

        for i in range(T):
            try:
                if np.any(
                    np.isnan(
                        [
                            years[i],
                            months[i],
                            days[i],
                            hours[i],
                            minutes[i],
                            seconds[i],
                        ]
                    )
                ):
                    timestamps[i] = pd.NaT
                else:
                    y, m, d = years[i], months[i], days[i]
                    try:
                        max_day = pd.Timestamp(
                            year=y, month=m, day=1
                        ).days_in_month
                        d = min(d, max_day)
                    except Exception:
                        d = default_day
                    timestamps[i] = pd.Timestamp(
                        year=y,
                        month=m,
                        day=d,
                        hour=hours[i],
                        minute=minutes[i],
                        second=seconds[i],
                    )
            except Exception:
                timestamps[i] = pd.NaT
        return timestamps
    elif vectorized_timestamps.ndim == 3:
        batch_size, T, num_feat = vectorized_timestamps.shape
        return np.stack(
            [
                inverse_time_vector(
                    vectorized_timestamps[i],
                    features=features,
                    norm_ranges=norm_ranges,
                    default_year=default_year,
                    default_month=default_month,
                    default_day=default_day,
                    default_hour=default_hour,
                    default_minute=default_minute,
                    default_second=default_second,
                )
                for i in range(batch_size)
            ]
        )
    else:
        raise ValueError(
            f"Input must be 2D or 3D array, got shape {vectorized_timestamps.shape}"
        )


def batch_process_directories(
    data_parent_dir: str,
    dataset_name: str,
    window_size: int = 256,
    stride: int = 1,
    verbose: bool = False,
    batch_size: Optional[int] = None,  # Number of subdirs to process at once
    n_workers: Optional[int] = None,  # Number of parallel workers
) -> None:
    """Process multiple subdirectories containing time series data in parallel.

    Args:
        data_parent_dir: Parent directory containing subdirectories in format {dataset_name}_{id}
        dataset_name: Name of the dataset to filter subdirectories
        window_size: Window size for processing
        stride: Stride between windows
        verbose: Enable verbose logging
        batch_size: Number of subdirectories to process at once (None for all)
        n_workers: Number of parallel workers (None for CPU count - 1)
    """
    # Start the overall timer
    start_time = datetime.datetime.now()
    if verbose:
        logger.info(
            f"Starting batch processing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # Set default number of workers
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        if verbose:
            logger.info(f"Using {n_workers} parallel workers")

    # Find all subdirectories matching the pattern
    pattern = f"{dataset_name}_*"
    subdirs = [
        d
        for d in glob.glob(os.path.join(data_parent_dir, pattern))
        if os.path.isdir(d)
    ]

    if not subdirs:
        raise ValueError(
            f"No subdirectories matching pattern '{pattern}' found in {data_parent_dir}"
        )

    # Sort subdirectories by ID for consistent ordering
    def extract_id(dir_name):
        match = re.search(f"{dataset_name}_(\d+)$", os.path.basename(dir_name))
        return int(match.group(1)) if match else float("inf")

    subdirs = sorted(subdirs, key=extract_id)

    if verbose:
        logger.info(f"Found {len(subdirs)} subdirectories to process")

    # Process in batches if specified
    if batch_size is None:
        batch_size = len(subdirs)

    # Process subdirectories in batches
    for i in range(0, len(subdirs), batch_size):
        batch_start_time = datetime.datetime.now()
        batch_subdirs = subdirs[i : i + batch_size]

        if verbose:
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(subdirs) - 1) // batch_size + 1} ({len(batch_subdirs)} subdirs)"
            )

        # Prepare args for the process_subdir function
        process_args = [
            (subdir, dataset_name, window_size, stride, verbose)
            for subdir in batch_subdirs
        ]

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Track progress if verbose
            if verbose:
                list(
                    tqdm(
                        executor.map(_process_single_subdir, process_args),
                        total=len(batch_subdirs),
                        desc="Processing subdirectories",
                    )
                )
            else:
                list(executor.map(_process_single_subdir, process_args))

        # Log batch completion time
        if verbose:
            batch_end_time = datetime.datetime.now()
            batch_elapsed = (batch_end_time - batch_start_time).total_seconds()
            logger.info(
                f"Batch {i // batch_size + 1} completed in {batch_elapsed:.2f} seconds"
            )

    # Calculate and log total processing time
    end_time = datetime.datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    total_minutes = total_elapsed / 60

    # Format message based on time
    if total_minutes < 1:
        time_msg = f"{total_elapsed:.2f} seconds"
    else:
        time_msg = f"{total_minutes:.2f} minutes ({total_elapsed:.2f} seconds)"

    logger.info(f"Total batch processing time: {time_msg}")


class FMV2_Preprocessor:
    """Window-based preprocessor for time-series data for foundation models.

    This class transforms time-series data into fixed-size numerical representations
    suitable for foundation models. It divides input data into windows and processes
    each window to include:

    1. Scaling statistics (mean, variance, scale)
    2. Temporal features from timestamps
    3. Text embeddings of series descriptions
    4. Normalized time-series values
    5. Dataset identification

    Each window is processed independently to facilitate parallel inference
    and consistent dimensionality across different inputs.

    Attributes:
        window_size (int): Number of timesteps in each window
        stride (int): Step size between consecutive windows
        metadata (Dict[str, Any]): Dataset metadata containing column specifications
        data_dir (str): Directory path for saving/loading preprocessing artifacts
        verbose (bool): Whether to log detailed processing information
        device (torch.device): Device for running text embedding model
        timestamps_column (str): Name of the timestamps column
        timeseries_column (str): Name of the timeseries value column
        timeseries_description (str): Description of the timeseries
        null_value: Value to use for nulls/missing data (np.nan)
        expected_output_length (int): Expected vector length of processed output
        dataset_id (int): Unique identifier for the dataset
    """

    def __init__(
        self,
        window_size: int,
        stride: int,
        metadata: Dict[str, Any],
        data_dir: str,
        verbose: bool = False,
    ):
        """Initialize the window preprocessor with configuration and metadata.

        Args:
            window_size: Number of timesteps in each window
            metadata: Dictionary containing dataset metadata including column specifications
            data_dir: Directory path where preprocessing artifacts will be saved
            verbose: If True, log detailed timing and processing information
        """
        self.metadata = metadata
        columns_info_dict = get_columns_info_dict_from_metadata_json(
            metadata, v1=False
        )
        self.window_size = window_size
        self.stride = stride
        self.data_dir = data_dir.rstrip("/")
        self.verbose = verbose
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.timestamps_column = metadata["timestamp_columns"][0]
        if "timeseries" not in columns_info_dict:
            raise ValueError("No 'timeseries' columns found in metadata")

        timeseries_columns = list(columns_info_dict["timeseries"].keys())
        if len(timeseries_columns) != 1:
            raise ValueError(
                f"Expected exactly 1 timeseries column, but found {len(timeseries_columns)}"
            )

        self.timeseries_column = timeseries_columns[0]
        self.timeseries_description = columns_info_dict["timeseries"][
            self.timeseries_column
        ]["description"]

        self.null_value = np.nan
        self.expected_output_length = (
            len(SCALING_FEATURES)
            + len(TIMESTAMPS_FEATURES) * self.window_size
            + TEXT_EMBEDDING_DIM
            + self.window_size
            + 1
        )
        self.dataset_id = int(os.path.basename(self.data_dir).split("_")[-1])

    def _get_timestamp_array(
        self,
        timestamp_values: Union[pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        """Extract and normalize temporal features from timestamp values, then create windows.

        Converts timestamps into exactly 11 normalized temporal features and creates windows
        using strided views for memory efficiency.

        Args:
            timestamp_values: pandas Series or DataFrame of timestamp values

        Returns:
            Windowed normalized timestamp features as a numpy array
            with shape (num_windows, window_size * len(TIMESTAMPS_FEATURES))
        """
        timestamp_start = datetime.datetime.now()

        # Check if timestamp_values is a DataFrame and extract the Series
        if isinstance(timestamp_values, pd.DataFrame):
            # Extract the first column as a Series
            timestamp_series = timestamp_values.iloc[:, 0]
        else:
            timestamp_series = timestamp_values

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(timestamp_series):
            timestamp_series = pd.to_datetime(timestamp_series, errors="coerce")

        # Use the global constant
        features = TIMESTAMPS_FEATURES

        # Create a result array to store features for each timestamp
        result = np.zeros((len(timestamp_series), len(features)))

        # Create mask for NaN values in original timestamps to preserve them
        nan_mask = timestamp_series.isna().values

        # Normalization ranges for each feature
        norm_ranges = NORM_RANGES

        # Get timezone from metadata if available
        timezone_offset = 0
        if "timezone" in self.metadata:
            tz_string = self.metadata["timezone"]
            # Parse timezone string like "GMT-6"
            if isinstance(tz_string, str) and tz_string.startswith("GMT"):
                try:
                    # Extract the numerical part (e.g., -6 from "GMT-6")
                    timezone_offset = float(tz_string[3:])
                except ValueError:
                    logger.warning(
                        f"Could not parse timezone string: {tz_string}, using default offset 0"
                    )
            elif isinstance(tz_string, (int, float)):
                # If it's already a number, use it directly
                timezone_offset = float(tz_string)
            else:
                logger.warning(
                    f"Could not parse timezone string: {tz_string}, using default offset 0"
                )
        else:
            logger.warning(
                "No timezone found in metadata, using default offset 0"
            )

        # Create US holidays calendar
        us_holidays = holidays.country_holidays("US")

        result = convert_time_to_vector(
            result,
            features,
            norm_ranges,
            timestamp_series,
            us_holidays,
            timezone_offset,
            np.array(nan_mask),
        )

        # Ensure correct shape for the special case where data_length == window_size
        if len(timestamp_series) == self.window_size:
            # Special case: reshape output directly for single window
            timestamps_window_shape = (1, self.window_size * len(features))
        else:
            # Standard case: calculate number of windows
            num_windows = (
                len(timestamp_series) - self.window_size
            ) // self.stride + 1
            timestamps_window_shape = (
                num_windows,
                self.window_size * len(features),
            )

        windowed_timestamps = self._create_strided_windows(
            data=result,
        )

        # Reshape to expected output shape
        windowed_timestamps = windowed_timestamps.reshape(
            timestamps_window_shape
        )

        if self.verbose:
            elapsed = (
                datetime.datetime.now() - timestamp_start
            ).total_seconds()
            logger.debug(
                f"Timestamp feature extraction and windowing took {elapsed:.4f}s"
            )

        return windowed_timestamps

    def inverse_timestamp_array(
        self,
        windowed_timestamps: np.ndarray,
        base_timestamp: Union[pd.Timestamp, str, None] = None,
    ) -> pd.DataFrame:
        """Convert processed timestamp feature arrays back to original timestamps.

        This method performs the inverse operation of _get_timestamp_array by:
        1. Reshaping the windowed features to per-timestamp features
        2. Denormalizing the features to their original ranges
        3. Reconstructing timestamps from the denormalized features

        Args:
            windowed_timestamps: Numpy array of processed timestamp features with shape
                                (num_windows, window_size * len(TIMESTAMPS_FEATURES))
            base_timestamp: Optional reference timestamp to start reconstructing from.
                           If None, will use the current date as reference.

        Returns:
            DataFrame containing reconstructed timestamps for each window with columns:
            - 'window_idx': Index of the window
            - 'position': Position within the window (0 to window_size-1)
            - 'timestamp': Reconstructed timestamp
            - 'is_holiday': Whether the timestamp falls on a holiday
            - 'timezone_offset': Timezone offset in hours
        """
        # Get dimensions
        num_windows = windowed_timestamps.shape[0]
        num_features = len(TIMESTAMPS_FEATURES)

        # Reshape to separate window and timestamp dimensions
        # From: (num_windows, window_size * num_features)
        # To:   (num_windows, window_size, num_features)
        reshaped = windowed_timestamps.reshape(
            num_windows, self.window_size, num_features
        )

        # Create window_idx and position arrays using broadcasting
        window_indices = (
            np.arange(num_windows)[:, np.newaxis]
            .repeat(self.window_size, axis=1)
            .flatten()
        )
        positions = np.tile(np.arange(self.window_size), num_windows)

        # Reshape the features array to (num_windows * window_size, num_features)
        features_flat = reshaped.reshape(-1, num_features)

        # Skip timestamps that are all NaN (indicates padding or missing data)
        valid_mask = ~np.isnan(features_flat).all(axis=1)

        # Filter out invalid entries
        valid_window_indices = window_indices[valid_mask]
        valid_positions = positions[valid_mask]
        valid_features = features_flat[valid_mask]

        # If base_timestamp is provided as string, convert to Timestamp
        if isinstance(base_timestamp, str):
            base_timestamp = pd.Timestamp(base_timestamp)
        elif base_timestamp is None:
            base_timestamp = pd.Timestamp.now()

        # Store original features for out-of-bounds detection
        original_features = valid_features.copy()

        # Prepare arrays for each denormalized feature using exact denormalization logic from original function
        # First, clamp normalized values to [0, 1] range to prevent extreme out-of-bounds issues
        valid_features = np.clip(valid_features, 0, 1)

        years = np.round(
            valid_features[:, 0]
            * (NORM_RANGES["year"][1] - NORM_RANGES["year"][0])
            + NORM_RANGES["year"][0]
        ).astype(int)
        months = np.round(
            valid_features[:, 2] * (NORM_RANGES["month"][1] - 1) + 1
        ).astype(int)
        days = np.round(
            valid_features[:, 5] * (NORM_RANGES["day"][1] - 1) + 1
        ).astype(int)
        hours = np.round(valid_features[:, 6] * NORM_RANGES["hour"][1]).astype(
            int
        )
        minutes = np.round(
            valid_features[:, 7] * NORM_RANGES["minute"][1]
        ).astype(int)
        seconds = np.round(
            valid_features[:, 8] * NORM_RANGES["second"][1]
        ).astype(int)
        holidays = valid_features[:, 9] > 0.5  # Binary threshold for holiday
        timezones = (
            valid_features[:, 10]
            * (NORM_RANGES["timezone"][1] - NORM_RANGES["timezone"][0])
            + NORM_RANGES["timezone"][0]
        )

        # Prepare to store reconstructed timestamps
        timestamps = []
        timestamp_indices = []

        # Process each valid entry
        for i in range(len(valid_window_indices)):
            # Get original normalized values to check for out-of-bounds
            orig_year_norm = original_features[i, 0]
            orig_month_norm = original_features[i, 2]
            orig_day_norm = original_features[i, 5]
            orig_hour_norm = original_features[i, 6]
            orig_minute_norm = original_features[i, 7]
            orig_second_norm = original_features[i, 8]

            # Get denormalized values
            year, month, day = years[i], months[i], days[i]
            hour, minute, second = hours[i], minutes[i], seconds[i]

            try:
                # Check if values were out of bounds (outside 0-1 range) and use base timestamp for those
                # Create a timestamp with valid values for each component
                timestamp = pd.Timestamp(
                    year=base_timestamp.year
                    if orig_year_norm < 0 or orig_year_norm > 1
                    else year,
                    month=base_timestamp.month
                    if orig_month_norm < 0 or orig_month_norm > 1
                    else month,
                    day=1,  # Temporary value, will fix below
                    hour=base_timestamp.hour
                    if orig_hour_norm < 0 or orig_hour_norm > 1
                    else hour,
                    minute=base_timestamp.minute
                    if orig_minute_norm < 0 or orig_minute_norm > 1
                    else minute,
                    second=base_timestamp.second
                    if orig_second_norm < 0 or orig_second_norm > 1
                    else second,
                )

                # Fix the day separately to handle month length issues
                target_day = (
                    base_timestamp.day
                    if orig_day_norm < 0 or orig_day_norm > 1
                    else day
                )
                # Ensure day is valid for the month
                max_days = timestamp.days_in_month
                valid_day = min(target_day, max_days)

                # Create final timestamp with correct day
                timestamp = timestamp.replace(day=valid_day)

            except (ValueError, pd.errors.OutOfBoundsDatetime):
                # If timestamp creation still fails, fall back entirely to the base timestamp
                timestamp = base_timestamp

            timestamps.append(timestamp)
            timestamp_indices.append(i)

        # Create result DataFrame
        result_data = {
            "window_idx": valid_window_indices[timestamp_indices],
            "position": valid_positions[timestamp_indices],
            "timestamp": timestamps,
            "is_holiday": holidays[timestamp_indices],
            "timezone_offset": timezones[timestamp_indices],
        }

        result_df = pd.DataFrame(result_data)

        # Sort by window index and position
        result_df = result_df.sort_values(
            ["window_idx", "position"]
        ).reset_index(drop=True)

        return result_df

    def window_and_scale_column(
        self, df: pd.DataFrame, column_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Window and standardize a single column of data using vectorized operations.

        Creates fixed-size windows from a column and standardizes each window using
        NumPy's vectorized operations, matching sklearn's StandardScaler behavior.
        Each window is normalized independently based on its own statistics.

        Args:
            df: Input dataframe containing the column to process
            column_name: Name of the column to window and standardize

        Returns:
            Tuple containing:
            - Numpy array of standardized windows with shape (num_windows, window_size)
            - Numpy array of scaler parameters with shape (num_windows, len(SCALING_FEATURES))
              where SCALING_FEATURES are [mean, var, scale]

        Raises:
            ValueError: If column_name is not in the dataframe
            ValueError: If the dataframe is empty
            ValueError: If window_size is larger than the dataframe length
        """
        # Check if column exists in dataframe
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")

        # Check if dataframe is empty
        if len(df) == 0:
            raise ValueError("Input dataframe is empty")

        # Check if window size is valid
        if self.window_size > len(df):
            raise ValueError(
                f"Window size ({self.window_size}) is larger than dataframe length ({len(df)})"
            )

        # Extract column data as numpy array
        data = df[column_name].to_numpy()

        # Use the helper function to create windows
        windowed_data = self._create_strided_windows(data=data)

        # Standardize the windowed data
        windowed_data, scaler_params = self._standardize_windows(windowed_data)

        return windowed_data, scaler_params

    def _standardize_windows(
        self, windowed_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize windows of data using vectorized operations.

        Args:
            windowed_data: Numpy array of windowed data with shape (num_windows, window_size)

        Returns:
            Tuple containing:
            - Numpy array of standardized windows with shape (num_windows, window_size)
            - Numpy array of scaler parameters with shape (num_windows, len(SCALING_FEATURES))
        """
        # Get number of windows
        num_windows = windowed_data.shape[0]

        # Initialize array for scaler parameters
        scaler_params = np.zeros((num_windows, len(SCALING_FEATURES)))

        # Create a mask for non-NaN values
        valid_mask = ~np.isnan(windowed_data)

        # Count valid values in each window (for Bessel's correction)
        n_samples_per_window = np.sum(valid_mask, axis=1, keepdims=True)

        # Handle edge case of windows with no valid data points
        zero_samples_mask = (n_samples_per_window == 0).flatten()

        # Calculate mean using valid values only
        # Sum valid values and divide by count (equivalent to nanmean)
        sums = np.sum(
            np.where(valid_mask, windowed_data, 0), axis=1, keepdims=True
        )
        window_means = np.zeros_like(sums)
        valid_windows = n_samples_per_window > 0
        window_means[valid_windows] = (
            sums[valid_windows] / n_samples_per_window[valid_windows]
        )

        # Calculate variance with Bessel's correction (n-1) like sklearn
        # Only for windows with more than 1 sample to avoid division by zero
        diff_squared = np.where(
            valid_mask, (windowed_data - window_means) ** 2, 0
        )
        sum_diff_squared = np.sum(diff_squared, axis=1, keepdims=True)

        # Initialize variance array
        window_vars = np.zeros_like(sum_diff_squared)

        # Only calculate variance for windows with more than 1 valid sample
        # Use n-1 in denominator for Bessel's correction (like sklearn)
        multi_sample_windows = n_samples_per_window > 1
        window_vars[multi_sample_windows] = sum_diff_squared[
            multi_sample_windows
        ] / (n_samples_per_window[multi_sample_windows] - 1)

        # Calculate scale (std dev) as square root of variance
        window_scales = np.sqrt(window_vars)

        # Replace zero scales with 1 to avoid division by zero (sklearn behavior)
        window_scales[window_scales == 0] = 1.0

        # Store scaler parameters
        scaler_params[:, 0] = window_means.flatten()  # mean
        scaler_params[:, 1] = (
            window_vars.flatten()
        )  # variance with Bessel's correction
        scaler_params[:, 2] = window_scales.flatten()  # scale

        # Standardize the windows in-place (z-score normalization)
        # Only transform valid (non-NaN) values
        standardized_data = np.where(
            valid_mask,
            (windowed_data - window_means) / window_scales,
            windowed_data,
        )  # Keep NaNs as NaNs

        # Handle windows with zero valid samples - leave as NaN
        if np.any(zero_samples_mask):
            scaler_params[zero_samples_mask, :] = np.nan

        return standardized_data, scaler_params

    def _create_strided_windows(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Create windows from data using memory-efficient strided views.

        Args:
            data: Input data as numpy array

        Returns:
            Numpy array of windowed data with shape (num_windows, window_size)

        Raises:
            ValueError: If window_size is larger than the data length
            ValueError: If stride is less than 1
        """
        data_length = len(data)

        # Add parameter validation
        if self.window_size > data_length:
            raise ValueError(
                f"Window size ({self.window_size}) is larger than data length ({data_length})"
            )

        if self.stride < 1:
            raise ValueError(f"Stride must be at least 1, got {self.stride}")

        # Handle the case when data_length equals window_size
        if data_length == self.window_size:
            # Special case: exactly one window
            if data.ndim == 1:
                return np.array([data])
            else:
                return np.array([data])

        # Calculate number of windows for standard case
        num_windows = (data_length - self.window_size) // self.stride + 1

        # Handle case when no windows can be created
        if num_windows <= 0:
            raise ValueError(
                f"Cannot create windows with size={self.window_size} and stride={self.stride} "
                f"from data with length={data_length}"
            )

        # Create window shape based on input data shape
        if data.ndim == 1:
            window_shape = (num_windows, self.window_size)
            strides = (self.stride * data.strides[0], data.strides[0])
        else:
            # For 2D data like timestamp features with shape (n_samples, n_features)
            window_shape = (num_windows, self.window_size, data.shape[1])
            strides = (
                self.stride * data.strides[0],
                data.strides[0],
                data.strides[1],
            )

        # Create strided view
        strided_data = np.lib.stride_tricks.as_strided(
            data, shape=window_shape, strides=strides, writeable=False
        )

        # Copy to ensure memory safety
        windowed_data = np.copy(strided_data)

        return windowed_data

    def _pad_dataframe_to_window_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pad a dataframe with additional rows to reach the required window size.

        When a dataframe is shorter than the required window size, this function:
        1. For timestamps: Infers the frequency from existing data and creates
           properly sequenced timestamps for the padding rows
        2. For other columns: Pads with NaN values

        Args:
            df: Input dataframe that may need padding

        Returns:
            Padded dataframe with length >= window_size
        """
        # Return original df if it's already >= window_size
        if len(df) >= self.window_size:
            return df

        # Calculate how many rows we need to add
        padding_rows = self.window_size - len(df)

        # Create a DataFrame for padding
        padding_df = pd.DataFrame(index=range(-padding_rows, 0))

        # Intelligently pad the timestamp column by inferring frequency
        if len(df) >= 2:
            # Try to infer frequency from existing timestamps
            timestamps = pd.to_datetime(df[self.timestamps_column])

            # Calculate time deltas between consecutive timestamps
            time_diffs = timestamps.diff().dropna()
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()

                # Create properly spaced timestamps for padding by stepping backward from first timestamp
                first_timestamp = timestamps.iloc[0]
                padding_timestamps = [
                    first_timestamp - (i + 1) * median_diff
                    for i in range(padding_rows)
                ]
                padding_timestamps.reverse()  # To maintain chronological order

                padding_df[self.timestamps_column] = padding_timestamps

                if self.verbose:
                    logger.info(
                        f"Padded timestamp column with {padding_rows} values using inferred frequency: {median_diff}"
                    )
            else:
                # Fallback to NaN if we can't infer frequency
                padding_df[self.timestamps_column] = np.nan

                if self.verbose:
                    logger.warning(
                        "Unable to infer timestamp frequency, padding with NaN values"
                    )
        else:
            # Not enough data to infer frequency
            padding_df[self.timestamps_column] = np.nan

            if self.verbose:
                logger.warning(
                    "Not enough timestamps to infer frequency, padding with NaN values"
                )

        # Pad other columns with NaN values
        for col in df.columns:
            if col != self.timestamps_column:
                padding_df[col] = [np.nan] * padding_rows

        # Concatenate the padding with the original dataframe
        padded_df = pd.concat([padding_df, df]).reset_index(drop=True)

        if self.verbose:
            logger.info(
                f"Padded dataframe with {padding_rows} rows to reach window_size={self.window_size}"
            )

        return padded_df

    def preprocess_df(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Preprocess a dataframe into a numerical representation for foundation models.

        Main preprocessing method that transforms raw time-series data into
        consistent numerical representations suitable for foundation models. The output
        concatenates multiple processed features into a single vector for each window:

        1. Scaling features (mean, var, scale): length = len(SCALING_FEATURES)
        2. Timestamp features (year, month, etc.): length = len(TIMESTAMPS_FEATURES) * window_size
        3. Timeseries description embedding: length = TEXT_EMBEDDING_DIM
        4. Normalized timeseries values: length = window_size
        5. Dataset ID (int): length = 1

        Args:
            df: DataFrame with at minimum timestamps and timeseries value columns

        Returns:
            Numpy array with shape (num_windows, expected_output_length) containing
            the processed data for each window

        Raises:
            ValueError: If input dataframe is missing required columns
            ValueError: If output array shape doesn't match expected dimensions
            ValueError: If input dataframe is empty
        """
        # Validate input
        if len(df) <= 1:
            raise ValueError("Input dataframe is empty")

        if self.timestamps_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{self.timestamps_column}' not found in dataframe"
            )

        if self.timeseries_column not in df.columns:
            raise ValueError(
                f"Timeseries column '{self.timeseries_column}' not found in dataframe"
            )

        # Pad the dataframe if needed to reach the window size
        df = self._pad_dataframe_to_window_size(df)

        # Reset index to ensure proper alignment
        df = df.reset_index(drop=True)

        start_time = datetime.datetime.now()
        readable_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        if self.verbose:
            logger.info(
                f"Preprocessing window of size {len(df)} starting at {readable_time}"
            )

        timestamps_array = self._get_timestamp_array(df[self.timestamps_column])
        # Process timeseries columns
        timeseries_array, timeseries_scalers_array = (
            self.window_and_scale_column(
                df=df,
                column_name=self.timeseries_column,
            )
        )
        timeseries_description = self.embed_column_description()

        # Get number of windows from timeseries_array
        num_windows = timeseries_array.shape[0]

        # Broadcast timeseries_description to match number of windows
        # Shape: (num_windows, TEXT_EMBEDDING_DIM)
        broadcasted_description = np.tile(
            timeseries_description, (num_windows, 1)
        )

        # Create dataset_id array with one id for each window
        dataset_id_array = np.full((num_windows, 1), self.dataset_id)

        # Concatenate all arrays to get the final output
        output_array = np.concatenate(
            [
                timeseries_scalers_array,  # Shape: (num_windows, len(SCALING_FEATURES))
                timestamps_array,  # Shape: (num_windows, len(TIMESTAMPS_FEATURES) * window_size)
                broadcasted_description,  # Shape: (num_windows, TEXT_EMBEDDING_DIM)
                timeseries_array,  # Shape: (num_windows, window_size)
                dataset_id_array,  # Shape: (num_windows, 1)
            ],
            axis=1,
        )

        # Check second dimension (vector length)
        assert output_array.shape[1] == self.expected_output_length, (
            f"Output array shape: {output_array.shape[1]} != expected output length: {self.expected_output_length}"
        )

        if self.verbose:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Total preprocessing time: {elapsed:.4f}s")

        return output_array

    def embed_column_description(
        self,
    ) -> np.ndarray:
        """Embed a column description string into a fixed-size numerical representation.

        Uses a sentence transformer model to convert a textual column description
        into a vector representation with consistent dimensionality. This embedding
        captures semantic meaning of the description in a format suitable for
        machine learning models.

        First checks if a pre-computed embedding file exists in data_dir,
        and uses that if available to avoid recomputing the embedding.

        If description text is missing, returns a zero vector of size TEXT_EMBEDDING_DIM.

        Returns:
            Numpy array with shape (TEXT_EMBEDDING_DIM,) containing the embedding vector

        Note:
            Uses the "sentence-transformers/all-MiniLM-L6-v2" model which creates
            384-dimensional embeddings (matching TEXT_EMBEDDING_DIM)
        """
        # Check if pre-computed embedding exists
        embedding_path = os.path.join(
            self.data_dir, "description_embedding.npy"
        )
        if os.path.exists(embedding_path):
            if self.verbose:
                logger.info(
                    f"Loading pre-computed embedding from {embedding_path}"
                )
            return np.load(embedding_path)

        # Handle empty description by returning zeros
        if (
            not self.timeseries_description
            or len(self.timeseries_description) == 0
        ):
            if self.verbose:
                logger.warning(
                    "Empty timeseries description, returning zero embedding"
                )
            zero_embedding = np.zeros(TEXT_EMBEDDING_DIM)

            # Save the zero embedding for future use
            if self.verbose:
                logger.info(f"Saving zero embedding to {embedding_path}")
            np.save(embedding_path, zero_embedding)

            return zero_embedding

        # If no pre-computed embedding exists and description is not empty, generate a new one
        if self.verbose:
            logger.info("Computing description embedding...")

        # Initialize the sentence transformer model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model = model.to(self.device)
        model = model.eval()

        # Generate embedding
        with torch.no_grad():
            embedding = model.encode(
                self.timeseries_description,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embedding_array = embedding.cpu().numpy()

        # Save the embedding for future use
        if self.verbose:
            logger.info(f"Saving embedding to {embedding_path}")
        np.save(embedding_path, embedding_array)

        return embedding_array


def main():
    """Run the foundation model window preprocessor from the command line.

    This function:
    1. Parses command line arguments for data directory, window size, stride, and verbosity
    2. Checks if data_dir contains subdirectories in format dataset_name_id
    3. If subdirectories found, processes them in parallel
    4. If no subdirectories, processes single directory
    5. Saves processed arrays to respective directories

    Returns:
        numpy.ndarray: The processed array if processing a single directory, None if batch processing
    """
    # Start total execution timer
    script_start_time = datetime.datetime.now()

    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Process time series data for foundation models"
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="Directory containing data to process",
        )
        parser.add_argument(
            "--window_size",
            type=int,
            default=256,
            help="Window size for processing",
        )
        parser.add_argument(
            "--stride", type=int, default=1, help="Stride between windows"
        )
        parser.add_argument(
            "--n_workers",
            type=int,
            default=40,
            help="Number of parallel workers for batch processing. If not provided, uses (CPU count - 1) workers. Set to 1 for sequential processing.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Number of subdirectories to process in each batch",
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose logging"
        )
        args = parser.parse_args()

        # Validate directory exists
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {args.data_dir}"
            )

        # Check if directory contains subdirectories with pattern *_digits
        subdirs = []
        for d in os.listdir(args.data_dir):
            if os.path.isdir(os.path.join(args.data_dir, d)):
                match = re.search(r"_(\d+)$", d)
                if match is not None:
                    subdirs.append(d)

        if subdirs:
            # Extract dataset name from first subdir
            match = re.match(r"(.+)_\d+$", subdirs[0])
            if not match:
                raise ValueError(
                    f"Could not extract dataset name from subdirectory pattern: {subdirs[0]}"
                )

            dataset_name = match.group(1)
            logger.info(
                f"Found subdirectories with pattern '{dataset_name}_*', using parallel processing"
            )

            # Set number of workers
            if args.n_workers == 1:
                logger.info("Using sequential processing (n_workers=1)")
            else:
                n_workers = (
                    args.n_workers
                    if args.n_workers is not None
                    else max(1, multiprocessing.cpu_count() - 1)
                )
                logger.info(
                    f"Using parallel processing with {n_workers} workers"
                )

            # Batch processing mode
            batch_process_directories(
                data_parent_dir=args.data_dir,
                dataset_name=dataset_name,
                window_size=args.window_size,
                stride=args.stride,
                verbose=args.verbose,
                batch_size=args.batch_size,
                n_workers=args.n_workers,
            )
            logger.info("Batch processing complete")
            return None  # Return None for batch processing

        else:
            # Single directory processing
            logger.info(
                f"No subdirectories found, processing single directory: {args.data_dir}"
            )
            metadata = load_metadata_from_directory(args.data_dir)
            df = load_dataframe_from_directory(args.data_dir)

            # Validate data
            if len(df) == 0:
                raise ValueError("Loaded dataframe is empty")

            # Initialize preprocessor
            logger.info(
                f"Initializing preprocessor with window_size={args.window_size}, stride={args.stride}"
            )
            preprocessor = FMV2_Preprocessor(
                window_size=args.window_size,
                stride=args.stride,
                metadata=metadata,
                data_dir=args.data_dir,
                verbose=args.verbose,
            )

            # Run preprocessing
            logger.info("Starting preprocessing")
            processed_array = preprocessor.preprocess_df(df)

            # Save the processed array with file lock
            output_path = os.path.join(args.data_dir, "preprocessed_array.npy")
            lock_path = output_path + ".lock"

            logger.info(f"Saving preprocessed array to {output_path}")
            with FileLock(lock_path):
                np.save(output_path, processed_array)

            # Log completion and array shape
            logger.info(
                f"Preprocessing complete. Output shape: {processed_array.shape}"
            )

            return processed_array  # Return the processed array for single directory processing

        # Calculate and log total script execution time
        script_end_time = datetime.datetime.now()
        script_elapsed = (script_end_time - script_start_time).total_seconds()
        script_minutes = script_elapsed / 60

        if script_minutes < 1:
            time_msg = f"{script_elapsed:.2f} seconds"
        else:
            time_msg = (
                f"{script_minutes:.2f} minutes ({script_elapsed:.2f} seconds)"
            )

        logger.info(f"Total script execution time: {time_msg}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")

        # Still log execution time even if there was an error
        script_end_time = datetime.datetime.now()
        script_elapsed = (script_end_time - script_start_time).total_seconds()
        logger.info(
            f"Script execution terminated after {script_elapsed:.2f} seconds due to error"
        )

        raise


if __name__ == "__main__":
    main()
