"""Time series features extraction using tsfeatures package."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import tsfeatures
from loguru import logger
from tsfeatures import (
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    hurst,
    hw_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    series_length,
    stability,
    stl_features,
    unitroot_kpss,
    unitroot_pp,
)

from synthefy_pkg.preprocessing.data_summary_utils.decompositions import (
    _group_by_timestamp,
    _should_group_by_idx,
    apply_grouping,
)


def _apply_grouped_feature_extraction(
    df: pl.DataFrame,
    column: str,
    timestamp_column: str,
    feature_func: Callable[..., Any],
    aggregation_method: str = "mean_variance",
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Helper function to apply feature extraction with grouping by timestamp.

    Args:
        df: Polars DataFrame
        column: Column name for time series analysis
        timestamp_column: Column name for timestamp to group by
        feature_func: Feature extraction function to apply to each group
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")
        *args, **kwargs: Arguments to pass to feature_func

    Returns:
        Dictionary with aggregated features
    """

    # Create a wrapper function that sets group_by_timestamp=False to avoid infinite recursion
    def _feature_wrapper(group_df, timestamps, *args, **kwargs):
        return feature_func(group_df, *args, **kwargs, group_by_timestamp=False)

    # Use apply_grouping to handle timestamp grouping
    grouped_results = apply_grouping(
        df, _feature_wrapper, timestamp_column, None, column, *args, **kwargs
    )
    return _aggregate_grouped_features(
        grouped_results, aggregation_method, column
    )


def _aggregate_grouped_features(
    grouped_results: List[Dict[str, Any]],
    aggregation_method: str = "mean_variance",
    column: str = "",
) -> Dict[str, Any]:
    """
    Aggregate grouped feature extraction results.

    Args:
        grouped_results: List of feature dictionaries from each group
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")

    Returns:
        Dictionary with aggregated features
    """
    if not grouped_results:
        return {}

    # Get all unique keys across all groups
    all_keys = set()
    for result in grouped_results:
        all_keys.update(result.keys())

    aggregated = {}

    for key in all_keys:
        # Collect values for this key across all groups
        values = []
        for result in grouped_results:
            if key in result and result[key] is not None:
                val = result[key]
                # Skip if already a dictionary (already aggregated)
                if isinstance(val, dict):
                    continue
                try:
                    # Convert to float if possible
                    val = float(val)
                    if not np.isnan(val) and not np.isinf(val):
                        values.append(val)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue

        if not values:
            aggregated[key] = None
            continue

        values = np.array(values)

        if aggregation_method == "mean_variance":
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "variance": float(np.var(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
        elif aggregation_method == "mean":
            aggregated[key] = float(np.mean(values))
        elif aggregation_method == "variance":
            aggregated[key] = float(np.var(values))
        elif aggregation_method == "all":
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "variance": float(np.var(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "count": len(values),
            }
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregation_method}"
            )
    aggregated["unique_id"] = result["unique_id"]
    assert column == result["unique_id"]

    return aggregated


def _extract_ts_features_single(
    df: pl.DataFrame,
    timestamps: Optional[np.ndarray],
    column: str,
    features: Optional[List[str]] = None,
    freq: Optional[int] = None,
    scale: bool = True,
    parallel: bool = False,
    aggregation_method: str = "mean_variance",
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """
    Extract time series features for a single group.
    """
    # Get the column and find non-null indices
    column_data = df.get_column(column)
    non_null_mask = ~column_data.is_null()
    non_null_indices = np.where(non_null_mask)[0]

    # Extract series and corresponding timestamps
    series = column_data.drop_nulls().to_numpy()

    if len(series) < 10:
        return {"error": "Insufficient data for feature extraction"}

    # Create timestamps if not provided
    if timestamps is not None:
        # Use the non-null indices to select corresponding timestamps
        time_col_values = timestamps[non_null_indices]
    else:
        # Create synthetic daily timestamps if none provided
        time_col_values = pd.date_range(
            start="2020-01-01", periods=len(series), freq="D"
        )

    # Convert to pandas DataFrame for tsfeatures (tsfeatures expects DataFrame with unique_id, ds, and y columns)
    # Note: tsfeatures requires integer frequencies, so we use integer approximations
    freq_mapping = {
        # Weekly variants
        "W-FRI": 7,
        "W-SUN": 7,
        "W-MON": 7,
        "W": 7,
        # Daily frequencies (1-31 days)
        "D": 1,
        "B": 1,  # Business daily
        # Hourly frequencies
        "H": 1,  # Hourly - use 1 as integer approximation
        # Minute frequencies
        "T": 1,  # Minute
        # Second frequencies
        "S": 1,  # Secondly
        # Microsecond frequencies
        "U": 1,  # Microsecond
        "L": 1,  # Millisecond
        "N": 1,  # Nanosecond
        # Monthly, quarterly, yearly frequencies
        "M": 30,
        "BM": 30,  # Month end, Business month end
        "Q": 90,
        "BQ": 90,  # Quarter end, Business quarter end
        "Y": 365,
        "BY": 365,  # Year end, Business year end
    }

    # Programmatically add day-based frequencies (2D-31D)
    for i in range(2, 32):
        freq_mapping[f"{i}D"] = i

    # Programmatically add hour-based frequencies (2H-23H)
    for i in range(2, 24):
        freq_mapping[f"{i}H"] = i

    # Programmatically add minute-based frequencies (2T-59T)
    for i in range(2, 60):
        freq_mapping[f"{i}T"] = i

    # Programmatically add second-based frequencies (2S-59S)
    for i in range(2, 60):
        freq_mapping[f"{i}S"] = i

    # Programmatically add millisecond-based frequencies (2L-999L)
    for i in range(2, 1000):
        freq_mapping[f"{i}L"] = i

    # Programmatically add microsecond-based frequencies (2U-999U)
    for i in range(2, 1000):
        freq_mapping[f"{i}U"] = i

    # Programmatically add nanosecond-based frequencies (2N-999N)
    for i in range(2, 1000):
        freq_mapping[f"{i}N"] = i
    ts_df = pd.DataFrame(
        {
            "unique_id": column,  # Single series identifier
            "ds": time_col_values,  # Time index (tsfeatures expects this column)
            "y": series,  # Target variable (tsfeatures expects column named 'y')
        }
    )

    # some features may be especially slow
    feature_calls_dict = {
        "acf_features": acf_features,
        "arch_stat": arch_stat,
        "crossing_points": crossing_points,
        "entropy": entropy,
        "flat_spots": flat_spots,
        "heterogeneity": heterogeneity,
        "holt_parameters": holt_parameters,
        "lumpiness": lumpiness,
        "nonlinearity": nonlinearity,
        "pacf_features": pacf_features,
        "stl_features": stl_features,
        "stability": stability,
        "hw_parameters": hw_parameters,
        "unitroot_kpss": unitroot_kpss,
        "unitroot_pp": unitroot_pp,
        "series_length": series_length,
        "hurst": hurst,
    }

    used_feature_calls = (
        [feature_calls_dict[feature] for feature in features]
        if features is not None
        else list(feature_calls_dict.values())
    )
    # Extract features using tsfeatures
    if fast_mode:
        # Use only basic features for faster processing
        basic_features = ["mean", "var", "acf1", "trend", "seasonality"]
        if features is None:
            features = basic_features

    # Handle frequency inference issues with integer timestamps proactively
    if ts_df["ds"].dtype in [np.int64, np.int32, np.int16, np.int8]:
        # Convert integer timestamps to datetime to avoid frequency inference issues
        ts_df = ts_df.copy()
        ts_df["ds"] = pd.date_range(
            start="2020-01-01", periods=len(ts_df), freq="D"
        )

    # Extract features
    # Use default features with frequency mapping
    # Optimize threading: use 1 thread for single series to avoid multiprocessing overhead
    threads = 1 if len(ts_df["unique_id"].unique()) == 1 else None

    try:
        extracted_features = tsfeatures.tsfeatures(
            ts_df,
            freq=freq,
            scale=scale,
            dict_freqs=freq_mapping,
            features=used_feature_calls,
            threads=threads,
        )
    except Exception as e:
        # Check if this is the specific frequency inference error
        if "Failed to infer frequency from the `ds` column" in str(e):
            # Retry with explicit frequency=1
            logger.warning(
                f"Failed to infer frequency from the `ds` column, retrying with frequency=1: {e}"
            )
            extracted_features = tsfeatures.tsfeatures(
                ts_df,
                freq=1,
                scale=scale,
                dict_freqs=freq_mapping,
                features=used_feature_calls,
                threads=threads,
            )
        else:
            # Re-raise if it's a different error
            raise e

    # Convert to regular Python types and handle NaN values
    result = {}
    for col in extracted_features.columns:
        value = extracted_features[col].iloc[0]
        key = col
        if isinstance(value, (np.integer, np.floating)):
            if np.isnan(value):
                result[key] = None
            else:
                result[key] = float(value)
        elif isinstance(value, (pd.Series, np.ndarray)):
            # Handle pandas Series or numpy arrays
            if isinstance(value, pd.Series) and value.isna().any():
                result[key] = None
            elif np.isnan(value).any():
                result[key] = None
            else:
                result[key] = value
        else:
            # For other types (strings, etc.), just store as-is
            result[key] = value

    return result


def _combine_ts_features_results(
    grouped_results: List[Dict[str, Any]], aggregation_method: str
) -> Dict[str, Any]:
    """
    Combine time series features from multiple groups.
    """
    # Filter out error results
    valid_results = [r for r in grouped_results if "error" not in r]

    if not valid_results:
        return {"error": "No valid feature extraction results"}

    # Get all feature names
    all_features = set()
    for result in valid_results:
        all_features.update(result.keys())

    combined_result = {}

    for feature in all_features:
        # Collect values, handling NaNs and Nones properly
        values = []
        for r in valid_results:
            value = r.get(feature, 0.0)
            if isinstance(value, (np.integer, np.floating)):
                if pd.isna(value) or value is None:
                    values.append(0.0)
                else:
                    values.append(float(value))
            else:
                values.append(value)

        if not values:
            combined_result[feature] = 0.0
            continue

        values = [v for v in values if v is not None]
        if len(values) == 0:
            combined_result[feature] = None
            continue

        if isinstance(values[0], (np.integer, np.floating, int, float)):
            if aggregation_method == "mean":
                combined_result[feature] = float(np.mean(values))
            elif aggregation_method == "variance":
                combined_result[feature] = float(np.var(values))
            elif aggregation_method == "mean_variance":
                combined_result[feature] = float(np.mean(values))
                combined_result[f"{feature}_variance"] = float(np.var(values))
            else:  # "all"
                combined_result[feature] = values
        else:
            combined_result[feature] = values[
                0
            ]  # assumed to be all the same, probably None

    # Add metadata
    combined_result["_num_groups"] = len(valid_results)
    combined_result["_total_groups"] = len(grouped_results)
    combined_result["_grouped_results"] = grouped_results

    return combined_result


def extract_ts_features(
    df: pl.DataFrame,
    column: str,
    timestamp_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    freq: Optional[int] = None,
    scale: bool = True,
    parallel: bool = False,
    aggregation_method: str = "mean_variance",
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """
    Extract time series features using the tsfeatures package.

    Args:
        df: Polars DataFrame
        column: Column name for time series analysis
        timestamp_column: Column name for timestamp
        features: List of specific features to extract (default: all available)
        freq: Frequency of the time series (e.g., 'D' for daily, 'H' for hourly)
        scale: Whether to scale the features
        parallel: Whether to use parallel processing
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")

    Returns:
        Dictionary with extracted features (aggregated if group_by_timestamp=True)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    series = df.get_column(column).drop_nulls().to_numpy()

    if len(series) < 10:
        return dict()

    # get the timestamp column
    # create daily data if timestamp column is not provided
    if timestamp_column is not None:
        time_col_values = df.get_column(timestamp_column).to_numpy()
    else:
        # Check if timestamps column already exists
        if "timestamps" in df.columns:
            time_col_values = df.get_column("timestamps").to_numpy()
        else:
            # Look for any column with sequential datetime values
            datetime_col = None
            for col in df.columns:
                try:
                    col_values = df.get_column(col).to_numpy()
                    # Check if column contains datetime-like values
                    if hasattr(col_values, "dtype") and (
                        np.issubdtype(col_values.dtype, np.datetime64)
                        or str(col_values.dtype).startswith("datetime")
                    ):
                        # Check if values are sequential (increasing)
                        if len(col_values) > 1 and np.all(
                            col_values[1:] > col_values[:-1]
                        ):
                            datetime_col = col
                            break
                except (TypeError, ValueError, AttributeError):
                    # Skip columns that can't be processed as datetime
                    continue

            if datetime_col is not None:
                # Rename the datetime column to 'timestamps'
                df = df.rename({datetime_col: "timestamps"})
                time_col_values = df.get_column("timestamps").to_numpy()
            else:
                # Create synthetic daily timestamps if none provided
                time_col_values = pd.date_range(
                    start="2020-01-01", periods=len(series), freq="D"
                )
                # Add timestamps column to the DataFrame using Polars syntax
                df = df.with_columns(pl.Series("timestamps", time_col_values))
    # Use automatic grouping logic similar to decompositions
    grouped_results = apply_grouping(
        df,
        _extract_ts_features_single,
        timestamp_column if timestamp_column is not None else "timestamps",
        None,
        column,
        features,
        freq,
        scale,
        parallel,
        aggregation_method,
        fast_mode,
    )

    # Combine results if multiple groups
    if len(grouped_results) > 1:
        return _combine_ts_features_results(grouped_results, aggregation_method)
    else:
        return grouped_results[0]


def extract_statistical_features(
    df: pl.DataFrame,
    column: str,
    timestamp_column: Optional[str] = None,
    group_by_timestamp: bool = False,
    aggregation_method: str = "mean_variance",
) -> Dict[str, Any]:
    """
    Extract basic statistical features from time series.

    Args:
        df: Polars DataFrame
        column: Column name for time series analysis
        timestamp_column: Column name for timestamp to group by
        group_by_timestamp: Whether to group by timestamp and aggregate results
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")

    Returns:
        Dictionary with statistical features (aggregated if group_by_timestamp=True)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Handle grouping by timestamp if requested
    if group_by_timestamp:
        if timestamp_column is None:
            raise ValueError(
                "timestamp_column must be provided when group_by_timestamp=True"
            )

        return _apply_grouped_feature_extraction(
            df,
            column,
            timestamp_column,
            extract_statistical_features,
            aggregation_method,
        )

    series = df.get_column(column).drop_nulls().to_numpy()

    if len(series) < 2:
        raise ValueError("Insufficient data for statistical features")

    # Basic statistical features
    features = {
        "length": len(series),
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "median": float(np.median(series)),
        "skewness": float(_calculate_skewness(series)),
        "kurtosis": float(_calculate_kurtosis(series)),
        "range": float(np.max(series) - np.min(series)),
        "iqr": float(np.percentile(series, 75) - np.percentile(series, 25)),
        "cv": float(np.std(series) / np.mean(series))
        if np.mean(series) != 0
        else 0.0,
    }

    return features


def extract_trend_features(
    df: pl.DataFrame,
    column: str,
    window: Optional[int] = None,
    timestamp_column: Optional[str] = None,
    group_by_timestamp: bool = False,
    aggregation_method: str = "mean_variance",
) -> Dict[str, Any]:
    """
    Extract trend-related features from time series.

    Args:
        df: Polars DataFrame
        column: Column name for time series analysis
        window: Window size for trend calculation (default: len(series) // 10)
        timestamp_column: Column name for timestamp to group by
        group_by_timestamp: Whether to group by timestamp and aggregate results
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")

    Returns:
        Dictionary with trend features (aggregated if group_by_timestamp=True)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Handle grouping by timestamp if requested
    if group_by_timestamp:
        if timestamp_column is None:
            raise ValueError(
                "timestamp_column must be provided when group_by_timestamp=True"
            )

        return _apply_grouped_feature_extraction(
            df,
            column,
            timestamp_column,
            extract_trend_features,
            aggregation_method,
            window,
        )

    series = df.get_column(column).drop_nulls().to_numpy()

    if len(series) < 3:
        raise ValueError("Insufficient data for trend features")

    if window is None:
        window = max(3, len(series) // 10)

    # Linear trend
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    trend_strength = np.corrcoef(x, series)[0, 1]

    # Moving average trend
    if len(series) >= window * 2:
        ma_start = np.mean(series[:window])
        ma_end = np.mean(series[-window:])
        ma_trend = (ma_end - ma_start) / ma_start if ma_start != 0 else 0.0
    else:
        ma_trend = 0.0

    # First and last value trend
    first_last_trend = (
        (series[-1] - series[0]) / series[0] if series[0] != 0 else 0.0
    )

    features = {
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "trend_strength": float(trend_strength)
        if not np.isnan(trend_strength)
        else 0.0,
        "ma_trend": float(ma_trend),
        "first_last_trend": float(first_last_trend),
        "trend_direction": 1.0 if slope > 0 else (-1.0 if slope < 0 else 0.0),
    }

    return features


def extract_seasonality_features(
    df: pl.DataFrame,
    column: str,
    max_period: Optional[int] = None,
    timestamp_column: Optional[str] = None,
    group_by_timestamp: bool = False,
    aggregation_method: str = "mean_variance",
) -> Dict[str, Any]:
    """
    Extract seasonality-related features from time series.

    Args:
        df: Polars DataFrame
        column: Column name for time series analysis
        max_period: Maximum period to check for seasonality
        timestamp_column: Column name for timestamp to group by
        group_by_timestamp: Whether to group by timestamp and aggregate results
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")

    Returns:
        Dictionary with seasonality features (aggregated if group_by_timestamp=True)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Handle grouping by timestamp if requested
    if group_by_timestamp:
        if timestamp_column is None:
            raise ValueError(
                "timestamp_column must be provided when group_by_timestamp=True"
            )

        return _apply_grouped_feature_extraction(
            df,
            column,
            timestamp_column,
            extract_seasonality_features,
            aggregation_method,
            max_period,
        )

    series = df.get_column(column).drop_nulls().to_numpy()

    if len(series) < 10:
        raise ValueError("Insufficient data for seasonality features")

    if max_period is None:
        max_period = min(len(series) // 2, 100)

    # Autocorrelation-based seasonality
    autocorr = _calculate_autocorrelation(series, max_lag=max_period)

    # Find peaks in autocorrelation (potential seasonality)
    peaks = _find_peaks(autocorr)

    if len(peaks) > 0:
        # Get the most prominent peak
        main_peak = peaks[np.argmax([autocorr[p] for p in peaks])]
        seasonality_strength = autocorr[main_peak]
        seasonality_period = main_peak
    else:
        seasonality_strength = 0.0
        seasonality_period = 0

    # Variance ratio (seasonal vs total variance)
    if len(series) >= 4:
        # Simple seasonal decomposition
        seasonal_variance = _estimate_seasonal_variance(series)
        total_variance = np.var(series)
        variance_ratio = (
            seasonal_variance / total_variance if total_variance > 0 else 0.0
        )
    else:
        variance_ratio = 0.0

    features = {
        "seasonality_strength": float(seasonality_strength),
        "seasonality_period": int(seasonality_period),
        "variance_ratio": float(variance_ratio),
        "has_seasonality": 1.0 if seasonality_strength > 0.3 else 0.0,
    }

    return features


def extract_volatility_features(
    df: pl.DataFrame,
    column: str,
    window: Optional[int] = None,
    timestamp_column: Optional[str] = None,
    group_by_timestamp: bool = False,
    aggregation_method: str = "mean_variance",
) -> Dict[str, Any]:
    """
    Extract volatility-related features from time series.

    Args:
        df: Polars DataFrame
        column: Column name for time series analysis
        window: Window size for volatility calculation
        timestamp_column: Column name for timestamp to group by
        group_by_timestamp: Whether to group by timestamp and aggregate results
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")

    Returns:
        Dictionary with volatility features (aggregated if group_by_timestamp=True)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Handle grouping by timestamp if requested
    if group_by_timestamp:
        if timestamp_column is None:
            raise ValueError(
                "timestamp_column must be provided when group_by_timestamp=True"
            )

        return _apply_grouped_feature_extraction(
            df,
            column,
            timestamp_column,
            extract_volatility_features,
            aggregation_method,
            window,
        )

    series = df.get_column(column).drop_nulls().to_numpy()

    if len(series) < 2:
        raise ValueError("Insufficient data for volatility features")

    if window is None:
        window = max(2, len(series) // 20)

    # Returns (if applicable)
    if len(series) > 1:
        returns = np.diff(series) / series[:-1]
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]

        if len(returns) > 0:
            return_volatility = float(np.std(returns))
            return_skewness = float(_calculate_skewness(returns))
            return_kurtosis = float(_calculate_kurtosis(returns))
        else:
            return_volatility = 0.0
            return_skewness = 0.0
            return_kurtosis = 0.0
    else:
        return_volatility = 0.0
        return_skewness = 0.0
        return_kurtosis = 0.0

    # Rolling volatility
    if len(series) >= window * 2:
        rolling_vol = []
        for i in range(window, len(series)):
            window_data = series[i - window : i]
            rolling_vol.append(np.std(window_data))

        if rolling_vol:
            rolling_volatility_mean = float(np.mean(rolling_vol))
            rolling_volatility_std = float(np.std(rolling_vol))
        else:
            rolling_volatility_mean = 0.0
            rolling_volatility_std = 0.0
    else:
        rolling_volatility_mean = 0.0
        rolling_volatility_std = 0.0

    # GARCH-like features (simplified)
    if len(series) >= 10:
        # Simple volatility clustering measure
        volatility_clustering = _calculate_volatility_clustering(series)
    else:
        volatility_clustering = 0.0

    features = {
        "return_volatility": return_volatility,
        "return_skewness": return_skewness,
        "return_kurtosis": return_kurtosis,
        "rolling_volatility_mean": rolling_volatility_mean,
        "rolling_volatility_std": rolling_volatility_std,
        "volatility_clustering": float(volatility_clustering),
    }

    return features


def extract_all_ts_features(
    df: pl.DataFrame,
    column: str,
    include_tsfeatures: bool = True,
    freq: Optional[int] = None,
    timestamp_column: Optional[str] = None,
    group_by_timestamp: bool = False,
    aggregation_method: str = "mean_variance",
) -> Dict[str, Any]:
    """
    Extract all available time series features.

    Args:
        df: Polars DataFrame
        column: Column name for time series analysis
        include_tsfeatures: Whether to include tsfeatures package features
        freq: Frequency for tsfeatures (if used)
        timestamp_column: Column name for timestamp (optional)
        group_by_timestamp: Whether to group by timestamp and aggregate results
        aggregation_method: Method for aggregation ("mean_variance", "mean", "variance", "all")

    Returns:
        Dictionary with all extracted features organized by category (aggregated if group_by_timestamp=True)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    all_features = {}

    # Statistical features
    all_features["statistical"] = extract_statistical_features(
        df,
        column,
        timestamp_column=timestamp_column,
        aggregation_method=aggregation_method,
    )

    # Trend features
    all_features["trend"] = extract_trend_features(
        df,
        column,
        timestamp_column=timestamp_column,
        aggregation_method=aggregation_method,
    )

    # Seasonality features
    all_features["seasonality"] = extract_seasonality_features(
        df,
        column,
        timestamp_column=timestamp_column,
        aggregation_method=aggregation_method,
    )

    # Volatility features
    all_features["volatility"] = extract_volatility_features(
        df,
        column,
        timestamp_column=timestamp_column,
        aggregation_method=aggregation_method,
    )

    # tsfeatures package features (if requested and available)
    if include_tsfeatures:
        tsfeatures_result = extract_ts_features(
            df,
            column,
            timestamp_column=timestamp_column,
            freq=freq,
            aggregation_method=aggregation_method,
        )
        if "error" not in tsfeatures_result and len(tsfeatures_result) > 0:
            all_features["tsfeatures"] = tsfeatures_result
        else:
            all_features["tsfeatures"] = {"note": "tsfeatures failed"}

    return all_features


# Helper functions
def _calculate_skewness(values: np.ndarray) -> float:
    """Calculate skewness of a numpy array."""
    if len(values) < 3:
        return 0.0

    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0

    skewness = np.mean(((values - mean) / std) ** 3)
    return float(skewness)


def _calculate_kurtosis(values: np.ndarray) -> float:
    """Calculate kurtosis of a numpy array."""
    if len(values) < 4:
        return 0.0

    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0

    kurtosis = np.mean(((values - mean) / std) ** 4) - 3
    return float(kurtosis)


def _calculate_autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Calculate autocorrelation function."""
    n = len(series)
    max_lag = min(max_lag, n - 1)

    autocorr = np.zeros(max_lag + 1)
    autocorr[0] = 1.0

    for lag in range(1, max_lag + 1):
        if lag < n:
            x1 = series[:-lag]
            x2 = series[lag:]
            corr = np.corrcoef(x1, x2)[0, 1]
            autocorr[lag] = corr if not np.isnan(corr) else 0.0

    return autocorr


def _find_peaks(array: np.ndarray, min_height: float = 0.1) -> List[int]:
    """Find peaks in an array above a minimum height."""
    peaks = []
    for i in range(1, len(array) - 1):
        if (
            array[i] > array[i - 1]
            and array[i] > array[i + 1]
            and array[i] > min_height
        ):
            peaks.append(i)
    return peaks


def _estimate_seasonal_variance(series: np.ndarray) -> float:
    """Estimate seasonal variance using simple approach."""
    if len(series) < 4:
        return 0.0

    # Simple seasonal decomposition: subtract moving average
    window = min(4, len(series) // 4)
    if window < 2:
        return 0.0

    ma = np.convolve(series, np.ones(window) / window, mode="valid")
    seasonal = series[window - 1 :] - ma

    return float(np.var(seasonal)) if len(seasonal) > 0 else 0.0


def _calculate_volatility_clustering(series: np.ndarray) -> float:
    """Calculate simple volatility clustering measure."""
    if len(series) < 10:
        return 0.0

    # Calculate absolute returns
    returns = np.abs(np.diff(series))

    # Calculate autocorrelation of absolute returns
    autocorr = _calculate_autocorrelation(returns, min(20, len(returns) // 2))

    # Return the average autocorrelation (excluding lag 0)
    if len(autocorr) > 1:
        return float(np.mean(autocorr[1:]))
    else:
        return 0.0
