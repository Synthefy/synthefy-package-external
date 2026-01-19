"""Decomposition methods for time series analysis."""

import time
from multiprocessing import Value
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import polars as pl
from loguru import logger
from scipy import linalg
from scipy.sparse.linalg import svds

from synthefy_pkg.preprocessing.data_summary_utils.stl_decomposition import (
    perform_stl_decomposition,
)
from synthefy_pkg.preprocessing.data_summary_utils.summary_utils import (
    _interpolate_nans,
)


def _convert_time_column(time_series: pl.Series) -> np.ndarray:
    """Convert a time series column to a numerical array."""
    # Convert to numerical values, handling datetime objects
    if time_series.dtype in [pl.Datetime, pl.Date]:
        # Convert datetime to Unix timestamp (seconds since epoch)
        time_series = time_series.dt.timestamp()
    elif time_series.dtype == pl.Utf8:
        # Try to parse as datetime first, then convert to timestamp
        try:
            time_series = time_series.str.strptime(pl.Datetime, format=None)
            time_series = time_series.dt.timestamp()
        except ValueError:
            # If datetime parsing fails, try to convert to numeric directly
            try:
                time_series = time_series.cast(pl.Float64, strict=False)
            except ValueError:
                # Final fallback: use index
                time_series = pl.Series(list(range(len(time_series))))

    time_series_np = time_series.to_numpy()
    return time_series_np


def _diagonal_average_vectorized(Xi: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorized diagonal averaging for SSA reconstruction.

    Args:
        Xi: Component matrix (L x K)
        n: Length of original series

    Returns:
        Vectorized diagonal average
    """
    L, K = Xi.shape
    result = np.zeros(n)

    # Create indices for diagonal averaging
    for j in range(n):
        # Find all pairs (k, l) such that k + l = j
        k_start = max(0, j - K + 1)
        k_end = min(L, j + 1)

        if k_start < k_end:
            k_indices = np.arange(k_start, k_end)
            l_indices = j - k_indices

            # Ensure l_indices are valid
            valid_mask = (l_indices >= 0) & (l_indices < K)
            k_indices = k_indices[valid_mask]
            l_indices = l_indices[valid_mask]

            if len(k_indices) > 0:
                result[j] = np.mean(Xi[k_indices, l_indices])

    return result


def apply_grouping(
    df: pl.DataFrame,
    decomposition_func: Callable[..., Dict[str, Any]],
    timestamp_column: Optional[str] = None,
    limit_cols: Optional[int] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Apply grouping to the data.
    """
    if _should_group_by_idx(df):
        return _group_by_idx(
            df,
            "idx",
            decomposition_func,
            limit_cols,
            timestamp_column,
            *args,
            **kwargs,
        )
    elif timestamp_column:
        return _group_by_timestamp(
            df,
            timestamp_column,
            decomposition_func,
            limit_cols,
            *args,
            **kwargs,
        )
    # else, no grouping, no timestamp column
    return [decomposition_func(df, None, *args, **kwargs)]


def _group_by_timestamp(
    df: pl.DataFrame,
    timestamp_column: str,
    decomposition_func: Callable[..., Dict[str, np.ndarray]],
    limit_cols: Optional[int] = None,
    *args,
    **kwargs,
) -> List[Dict[str, np.ndarray]]:
    """
    Helper function to group data by timestamp and apply decomposition function to each group.

    Args:
        df: Polars DataFrame
        timestamp_column: Column name for timestamps to group data by
        decomposition_func: Function to apply to each group
        *args, **kwargs: Arguments to pass to decomposition_func

    Returns:
        List of decomposition results for each timestamp group
    """
    if timestamp_column not in df.columns:
        raise ValueError(
            f"Timestamp column {timestamp_column} not found in DataFrame"
        )

    grouped_results = []

    # Get timestamp column as numeric numpy array
    # Handle datetime conversion properly
    timestamp_series = df.get_column(timestamp_column)

    timestamps = _convert_time_column(timestamp_series)

    # Vectorized approach: find where timestamps decrease (new segment starts)
    # np.diff(timestamps) < 0 gives True where timestamp decreases
    # np.where finds the indices of these True values
    # Add 0 at the beginning and len(timestamps) at the end
    segment_boundaries = np.concatenate(
        [[0], np.where(np.diff(timestamps) < 0)[0] + 1, [len(timestamps)]]
    )

    # Process each segment
    for i in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i + 1]

        # Extract segment
        group_df = df.slice(start_idx, end_idx - start_idx)
        segment_timestamps = timestamps[start_idx:end_idx]

        # Skip segments that are too short for meaningful analysis
        if len(group_df) < 5:
            logger.warning(
                f"Skipping timestamp segment {i} with only {len(group_df)} data points"
            )
            continue

        # Apply decomposition function to this group
        result = decomposition_func(
            group_df, segment_timestamps, *args, **kwargs
        )
        grouped_results.append(result)

    # Limit the number of timestamp groups to process
    if limit_cols is not None:
        grouped_results = grouped_results[:limit_cols]

    return grouped_results


def _should_group_by_idx(
    df: pl.DataFrame,
    idx_column: str = "idx",
) -> bool:
    """
    Check if DataFrame should be grouped by idx column.

    Args:
        df: Polars DataFrame
        idx_column: Column name for idx values to check

    Returns:
        True if idx column exists and has more than one unique value
    """
    if idx_column not in df.columns:
        return False

    unique_values = df.get_column(idx_column).n_unique()
    return unique_values > 1


def _group_by_idx(
    df: pl.DataFrame,
    idx_column: str,
    decomposition_func: Callable[..., Dict[str, np.ndarray]],
    limit_cols: Optional[int] = None,
    timestamp_column: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, np.ndarray]]:
    """
    Helper function to group data by idx column and apply decomposition function to each group.

    Args:
        df: Polars DataFrame
        idx_column: Column name for idx values to group data by
        decomposition_func: Function to apply to each group
        *args, **kwargs: Arguments to pass to decomposition_func

    Returns:
        List of decomposition results for each idx group
    """
    if idx_column not in df.columns:
        raise ValueError(f"Idx column {idx_column} not found in DataFrame")

    grouped_results = []

    # Get unique idx values and sort them
    unique_idx_values = df.get_column(idx_column).unique().sort()

    # Limit the number of idx values to process
    if limit_cols is not None:
        unique_idx_values = unique_idx_values[:limit_cols]

    # Process each idx group
    for idx_value in unique_idx_values:
        # Filter DataFrame for this idx value
        group_df = df.filter(pl.col(idx_column) == idx_value)

        # Skip groups that are too short for meaningful analysis
        if len(group_df) < 20:
            logger.warning(
                f"Skipping idx group {idx_value} with only {len(group_df)} data points"
            )
            continue

        # Get timestamps for this group (assuming there's a timestamp column)
        # We'll need to determine the timestamp column name
        if timestamp_column is not None:
            timestamps = group_df.get_column(timestamp_column).to_numpy()
        else:
            timestamp_columns = [
                col
                for col in df.columns
                if "time" in col.lower() or "timestamp" in col.lower()
            ]

            if timestamp_columns:
                # Use the first timestamp column found
                timestamp_column = timestamp_columns[0]
                timestamp_series = group_df.get_column(timestamp_column)
                timestamps = _convert_time_column(timestamp_series)
            else:
                # If no timestamp column found, create a simple range
                timestamps = np.arange(len(group_df))

        # Apply decomposition function to this group
        result = decomposition_func(group_df, timestamps, *args, **kwargs)
        grouped_results.append(result)

    return grouped_results


def combined_decomposition(
    result_dict: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combine the results of the fourier decomposition for each timestamp group.
    Handles arrays of different sizes by padding to the maximum size.
    """
    if not result_dict:
        return {}

    combined_result = {}
    num_decompositions = len(result_dict)
    for key in result_dict[0].keys():
        # Get all arrays for this key
        values = [result[key] for result in result_dict]

        if isinstance(values[0], np.ndarray):
            # Find the maximum number of dimensions and size along each dimension
            max_ndim = max(arr.ndim for arr in values)
            max_shape = []
            for dim in range(max_ndim):
                max_dim_size = max(
                    arr.shape[dim] if dim < arr.ndim else 1 for arr in values
                )
                max_shape.append(max_dim_size)
            max_shape = tuple(max_shape)

            # Pad all arrays to the maximum shape
            padded_arrays = []
            for arr in values:
                if arr.shape == max_shape:
                    padded_arrays.append(arr)
                else:
                    # Create padding only for the array's actual dimensions
                    padding = []
                    for i in range(arr.ndim):
                        current_size = arr.shape[i]
                        max_size = max_shape[i]
                        if current_size < max_size:
                            padding.append((0, max_size - current_size))
                        else:
                            padding.append((0, 0))

                    # Pad the array with NaN values
                    padded_arr = np.pad(
                        arr, padding, mode="constant", constant_values=np.nan
                    )
                    padded_arrays.append(padded_arr)

            # Ensure all arrays have the same shape before concatenation
            reshaped_arrays = []
            for arr in padded_arrays:
                if arr.ndim < max_ndim:
                    # Add dimensions to match max_ndim
                    new_shape = arr.shape + (1,) * (max_ndim - arr.ndim)
                    arr = arr.reshape(new_shape)

                # Ensure the array matches the target shape exactly
                if arr.shape != max_shape:
                    # Reshape to target shape, padding with NaN if needed
                    target_arr = np.full(max_shape, np.nan)
                    # Copy data into the target array
                    slices = tuple(
                        slice(0, min(arr.shape[i], max_shape[i]))
                        for i in range(arr.ndim)
                    )
                    target_arr[slices] = arr[slices]
                    arr = target_arr

                reshaped_arrays.append(arr)

            # Concatenate the reshaped arrays
            combined_result[key] = np.concatenate(reshaped_arrays)
        elif isinstance(values[0], dict):  # just take dicts as they are
            combined_result[key] = values
        elif (
            isinstance(values[0], float)
            or isinstance(values[0], int)
            or isinstance(values[0], (np.floating, np.integer))
        ):
            combined_result[key] = np.mean(values)  # take the mean of floats
        elif isinstance(values[0], str):
            combined_result[key] = values[0]
        elif values[0] is None:
            combined_result[key] = None
        else:
            raise ValueError(f"Unsupported value type: {type(values[0])}")
    combined_result["num_decompositions"] = num_decompositions
    return combined_result


def _fourier_single(
    df: pl.DataFrame,
    timestamps: Optional[np.ndarray],
    column: str,
    n_components: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Single-series Fourier decomposition without timestamp grouping."""
    start_time = time.time()
    series = df.get_column(column).to_numpy()
    series = _interpolate_nans(series, method="linear")
    series = (series - series.mean()) / (series.std() + 1e-10)

    # Perform FFT
    fft = np.fft.fft(series)
    freqs = np.fft.fftfreq(len(series))

    # Calculate amplitudes and phases
    amplitudes = np.abs(fft)
    phases = np.angle(fft)

    # Sort by amplitude (descending)
    sorted_indices = np.argsort(amplitudes)[::-1]

    if n_components is not None:
        sorted_indices = sorted_indices[:n_components]

    # Reconstruct signal with top components
    reconstructed = np.zeros_like(series, dtype=complex)
    for idx in sorted_indices:
        # Handle DC component (frequency 0) separately
        if freqs[idx] == 0:
            reconstructed += fft[idx] / len(
                series
            )  # DC component doesn't need complex exponential
        else:
            reconstructed += (
                fft[idx]
                * np.exp(1j * 2 * np.pi * freqs[idx] * np.arange(len(series)))
                / len(series)
            )

    residual = np.abs(series - reconstructed)

    logger.info(
        f"Fourier decomposition time: {time.time() - start_time} on series of length {len(series)}"
    )

    return {
        "frequencies": freqs[sorted_indices],
        "amplitudes": amplitudes[sorted_indices],
        "phases": phases[sorted_indices],
        "reconstructed": np.real(reconstructed),
        "residual": residual,
        "mean_residual": np.mean(residual),
        "normalized_original": series,
        "denormalized_reconstructed": reconstructed * series.std()
        + series.mean(),
    }


def fourier_decomposition(
    df: pl.DataFrame,
    column: str,
    n_components: Optional[int] = None,
    timestamp_column: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform Fourier decomposition on a time series column.

    Args:
        df: Polars DataFrame
        column: Column name for decomposition
        n_components: Number of frequency components to return (default: all)
        timestamp_column: Column name for timestamps to group data by (if provided)

    Returns:
        If timestamp_column is None: Dictionary with 'frequencies', 'amplitudes', 'phases', 'reconstructed'
        If timestamp_column is provided: List of decomposition results for each timestamp group
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Use apply_grouping to handle timestamp grouping
    grouped_results = apply_grouping(
        df, _fourier_single, timestamp_column, None, column, n_components
    )
    # Return both combined results and grouped results for plotting flexibility
    combined_result = combined_decomposition(grouped_results)
    combined_result["_grouped_results"] = (
        grouped_results  # Store grouped results for plotting
    )
    return combined_result


def _ssa_single(
    df: pl.DataFrame,
    timestamps: Optional[np.ndarray],
    column: str,
    window_size: Optional[int] = None,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """Single-series SSA without timestamp grouping."""
    start_time = time.time()
    series = df.get_column(column).to_numpy()
    series = _interpolate_nans(series, method="linear")
    series = (series - series.mean()) / (series.std() + 1e-10)
    n = len(series)

    # Optimize window size for performance
    if window_size is None:
        # Use smaller window size for large series to improve performance
        if n > 1000:
            window_size = min(n // 8, 200)  # Smaller window for large series
        else:
            window_size = n // 4

    # Limit window size to prevent excessive computation
    window_size = min(window_size, min(n // 2, 500))

    # Create trajectory matrix more efficiently
    L = window_size
    K = n - L + 1
    X = np.zeros((L, K))

    # Vectorized trajectory matrix creation
    for i in range(K):
        X[:, i] = series[i : i + L]

    # Use truncated SVD for better performance
    if n_components is None:
        n_components = 20  # Limit components for performance

    # Use SVD - sparse for large matrices, regular for small ones
    max_k = min(L, K) - 1

    if max_k <= 0:
        # Matrix too small for sparse SVD, use regular SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # Filter out near-zero singular values
        mask = S > 1e-10
        if np.any(mask):
            U, S, Vt = U[:, mask], S[mask], Vt[mask, :]
        else:
            U, S, Vt = U[:, :1], S[:1], Vt[:1, :]
    else:
        # Use sparse SVD for better performance
        k_value = min(n_components, max_k)
        U, S, Vt = svds(X, k=k_value)

    # Handle case where svds returns None or empty results
    if S is None or len(S) == 0 or U is None or Vt is None:
        logger.warning(
            f"SSA SVD failed for column {column}, returning empty result"
        )
        return {
            "eigenvalues": np.array([]),
            "eigenvectors": np.array([]),
            "reconstructed": series,
            "residual": np.zeros_like(series),
            "mean_residual": 0.0,
            "normalized_original": series,
            "denormalized_reconstructed": series,
        }

    # Sort by singular values (svds doesn't guarantee order)
    idx = np.argsort(S)[::-1]
    U, S, Vt = U[:, idx], S[idx], Vt[idx, :]

    # Optimized reconstruction with vectorized diagonal averaging
    reconstructed = np.zeros_like(series)

    for i in range(min(n_components, len(S))):
        # Reconstruct component i
        Xi = np.outer(U[:, i], Vt[i, :]) * S[i]

        # Vectorized diagonal averaging
        reconstructed += _diagonal_average_vectorized(Xi, n)

    residual = np.abs(series - reconstructed)

    logger.info(
        f"SSA decomposition time: {time.time() - start_time:.2f}s on series of length {len(series)} (window_size={window_size}, components={n_components})"
    )

    return {
        "eigenvalues": S[:n_components],
        "eigenvectors": U[:, :n_components],
        "reconstructed": reconstructed,
        "residual": residual,
        "mean_residual": np.mean(residual),
        "normalized_original": series,
        "denormalized_reconstructed": reconstructed * series.std()
        + series.mean(),
    }


def singular_spectrum_analysis(
    df: pl.DataFrame,
    column: str,
    window_size: Optional[int] = None,
    n_components: Optional[int] = None,
    timestamp_column: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform Singular Spectrum Analysis (SSA) on a time series.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        window_size: Embedding window size (default: len(df) // 4)
        n_components: Number of components to reconstruct (default: all)
        timestamp_column: Column name for timestamps to group data by (if provided)

    Returns:
        If timestamp_column is None: Dictionary with 'eigenvalues', 'eigenvectors', 'reconstructed'
        If timestamp_column is provided: List of decomposition results for each timestamp group
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Use apply_grouping to handle timestamp grouping
    grouped_results = apply_grouping(
        df,
        _ssa_single,
        timestamp_column,
        None,
        column,
        window_size,
        n_components,
    )
    # Return both combined results and grouped results for plotting flexibility
    combined_result = combined_decomposition(grouped_results)
    combined_result["_grouped_results"] = (
        grouped_results  # Store grouped results for plotting
    )
    return combined_result


def _sindy_single(
    df: pl.DataFrame,
    timestamps: Optional[np.ndarray],
    column: str,
    time_column: Optional[str] = None,
    poly_order: int = 2,
    threshold: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Single-series SINDy without timestamp grouping."""
    start_time = time.time()
    series = df.get_column(column).to_numpy()
    series = _interpolate_nans(series, method="linear")

    # Only normalize the data series, not time
    series_normalized = (series - series.mean()) / (series.std() + 1e-10)

    if time_column:
        time_series_raw = df.get_column(time_column).to_numpy()
        time_series_raw = _interpolate_nans(time_series_raw, method="linear")
        time_series = _convert_time_column(pl.Series(time_series_raw))
        # Don't normalize time_series - keep original scale
        time_series = (time_series - time_series.mean()) / time_series.std()
    else:
        time_series = np.arange(len(series))

    # Use normalized data for gradient calculation
    dx_dt = np.gradient(series_normalized, time_series)

    # Create polynomial library using normalized data
    n = len(series_normalized)
    library_terms = []
    for order in range(poly_order + 1):
        if order == 0:
            library_terms.append(np.ones(n))
        else:
            library_terms.append(series_normalized**order)

    Theta = np.column_stack(library_terms)

    # Solve and reconstruct
    try:
        result = linalg.lstsq(Theta, dx_dt)
        if result is not None and result[0] is not None:
            coefficients = result[0]
        else:
            coefficients = np.zeros(Theta.shape[1])
    except ImportError:
        coefficients = np.linalg.lstsq(Theta, dx_dt, rcond=None)[0]

    # Apply thresholding
    coefficients[np.abs(coefficients) < threshold] = 0

    # Reconstruct using coefficients and library terms
    reconstructed_dx = Theta @ coefficients
    reconstructed = np.cumsum(reconstructed_dx) + series_normalized[0]

    residual = np.abs(series_normalized - reconstructed)

    logger.info(
        f"SINDy decomposition time: {time.time() - start_time} on series of length {len(series)}"
    )

    return {
        "coefficients": coefficients,
        "library_terms": np.array(library_terms),
        "reconstructed": reconstructed,
        "residual": residual,
        "normalized_original": series_normalized,  # Return original scale
    }


def sindy_decomposition(
    df: pl.DataFrame,
    column: str,
    time_column: Optional[str] = None,
    poly_order: int = 2,
    threshold: float = 0.1,
    timestamp_column: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform SINDy (Sparse Identification of Nonlinear Dynamics) decomposition.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        poly_order: Maximum polynomial order for library
        threshold: Sparsity threshold for coefficient selection
        timestamp_column: Column name for timestamps to group data by (if provided)

    Returns:
        Dictionary with 'coefficients', 'library_terms', 'reconstructed'
        If timestamp_column is provided, results are concatenated across all timestamp groups
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Use apply_grouping to handle timestamp grouping
    grouped_results = apply_grouping(
        df,
        _sindy_single,
        timestamp_column,
        None,
        column,
        time_column,
        poly_order,
        threshold,
    )
    # Return both combined results and grouped results for plotting flexibility
    combined_result = combined_decomposition(grouped_results)
    combined_result["_grouped_results"] = (
        grouped_results  # Store grouped results for plotting
    )
    return combined_result


# utilize pysindy for decomposition
def _pysindy_single(
    df: pl.DataFrame,
    timestamps: Optional[np.ndarray],
    column: str,
    poly_order: int = 2,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """Single-series PySINDy without timestamp grouping."""
    import pysindy

    start_time = time.time()

    series = df.get_column(column).to_numpy()
    series = _interpolate_nans(series, method="linear")
    series = (series - series.mean()) / (1e-5 + series.std())

    # Check if series is long enough for PySINDy (minimum 3 points for differentiation)
    if len(series) < 3:
        logger.warning(
            f"Series too short for PySINDy ({len(series)} points), returning empty result"
        )
        return {
            "coefficients": np.array([]),
            "library_terms": np.array([]),
            "reconstructed": series,
            "residual": np.zeros_like(series),
            "mean_residual": 0.0,
            "denormalized_reconstructed": series,
            "normalized_original": series,
        }

    # Assume uniform time sampling, using a small timestep for stability, uniform sampling
    # time_series = timestamps
    timestep = 1 / 1000
    logger.info(
        f"Using timestep of {timestep} for data of range {np.min(series)} to {np.max(series)}"
    )
    time_series = np.arange(len(series)) * timestep

    # Create pysindy model
    model = pysindy.SINDy(
        feature_library=pysindy.PolynomialLibrary(degree=poly_order),
        optimizer=pysindy.STLSQ(threshold=threshold),
    )

    # Ensure data is in the correct format for PySINDy
    # PySINDy expects 2D arrays for features but 1D for time
    series_2d = series.reshape(-1, 1)

    # Use the original 1D time series as PySINDy expects
    model.fit(series_2d, t=time_series)
    coefficients = np.array(model.coefficients()).flatten()

    # Create polynomial library terms manually for reconstruction
    library_terms = []
    for order in range(poly_order + 1):
        if order == 0:
            library_terms.append(np.ones(len(series)))
        else:
            library_terms.append(series**order)

    Theta = np.column_stack(library_terms)

    # Reconstruct using coefficients and library terms
    reconstructed_dx = Theta @ coefficients

    # Scale by timestep before integrating
    reconstructed = np.cumsum(reconstructed_dx * timestep) + series[0]

    residual = np.abs(series - reconstructed)

    logger.info(
        f"PySINDy decomposition time: {time.time() - start_time} on series of length {len(series)}"
    )

    return {
        "coefficients": coefficients,
        "library_terms": np.array(library_terms),
        "reconstructed": reconstructed,
        "residual": residual,
        "mean_residual": np.mean(residual),
        "denormalized_reconstructed": reconstructed * series.std()
        + series.mean(),
        "normalized_original": series,
    }


def pysindy_decomposition(
    df: pl.DataFrame,
    column: str,
    time_column: Optional[str] = None,
    poly_order: int = 2,
    threshold: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Perform SINDy (Sparse Identification of Nonlinear Dynamics) decomposition using pysindy.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        time_column: Time column for derivatives (if None, assumes uniform sampling)
        poly_order: Maximum polynomial order for library
        threshold: Sparsity threshold for coefficient selection
        timestamp_column: Column name for timestamps to group data by (if provided)

    Returns:
        Dictionary with 'coefficients', 'library_terms', 'reconstructed'
        If timestamp_column is provided, results are concatenated across all timestamp groups
    """
    import pysindy  # type: ignore

    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Use apply_grouping to handle timestamp grouping
    grouped_results = apply_grouping(
        df, _pysindy_single, time_column, None, column, poly_order, threshold
    )
    # Return both combined results and grouped results for plotting flexibility
    combined_result = combined_decomposition(grouped_results)
    combined_result["_grouped_results"] = (
        grouped_results  # Store grouped results for plotting
    )
    return combined_result


# def pyts_ssa_decomposition(
#     df: pl.DataFrame,
#     column: str,
#     window_size: Optional[int] = None,
#     n_components: Optional[int] = None,
#     timestamp_column: Optional[str] = None,
#     grouping: str = "auto",
# ) -> Dict[str, np.ndarray]:
#     """
#     Perform Singular Spectrum Analysis (SSA) using the pyts library.

#     Args:
#         df: Polars DataFrame
#         column: Column name for analysis
#         window_size: Embedding window size (default: len(df) // 4)
#         n_components: Number of components to reconstruct (default: all)
#         timestamp_column: Column name for timestamps to group data by (if provided)
#         grouping: Grouping strategy ('auto', 'eigenvector', 'kmeans', 'dbscan')

#     Returns:
#         Dictionary with 'eigenvalues', 'eigenvectors', 'reconstructed', 'groups'
#     """
#     try:
#         from pyts.decomposition import SingularSpectrumAnalysis
#     except ImportError:
#         raise ImportError(
#             "pyts library not found. Install with: pip install pyts"
#         )

#     if column not in df.columns:
#         raise ValueError(f"Column {column} not found in DataFrame")

#     # If timestamp column is provided, use grouping helper
#     if timestamp_column:
#         grouped_results = _group_by_timestamp(
#             df,
#             timestamp_column,
#             _pyts_ssa_single,
#             column,
#             window_size,
#             n_components,
#             grouping,
#         )
#         return combined_decomposition(grouped_results)

#     # Original single-series logic
#     return _pyts_ssa_single(
#         df,
#         df.get_column(timestamp_column).to_numpy()
#         if timestamp_column
#         else None,
#         column,
#         window_size,
#         n_components,
#         grouping,
#     )


# def _pyts_ssa_single(
#     df: pl.DataFrame,
#     timestamps: Optional[np.ndarray],
#     column: str,
#     window_size: Optional[int] = None,
#     n_components: Optional[int] = None,
#     grouping: str = "auto",
# ) -> Dict[str, Any]:
#     """Single-series SSA using pyts without timestamp grouping."""
#     from pyts.decomposition import SingularSpectrumAnalysis

#     series = df.get_column(column).drop_nulls().to_numpy()

#     # Ensure series is 2D for pyts (n_samples, n_features)
#     if series.ndim == 1:
#         series = series.reshape(-1, 1)

#     # Set default window size if not provided
#     if window_size is None:
#         window_size = max(series.shape[0] // 4, 2)

#     # Create SSA instance and fit
#     ssa = SingularSpectrumAnalysis(window_size=window_size, groups=grouping)

#     # Fit and transform
#     reconstructed = ssa.fit_transform(series)

#     # Get component groups
#     groups = ssa.groups

#     # Determine number of components to use
#     if n_components is None:
#         n_components = len(groups) if groups is not None else 0
#     else:
#         n_components = min(
#             n_components, len(groups) if groups is not None else 0
#         )

#     # Convert reconstructed back to 1D if it was originally 1D
#     if reconstructed.ndim == 2 and reconstructed.shape[1] == 1:
#         reconstructed = reconstructed.flatten()

#     return {
#         "reconstructed": reconstructed,
#         "groups": groups,
#         "window_size": window_size,
#         "n_components": n_components,
#         "ssa_object": ssa,
#     }


def _stl_single(
    df: pl.DataFrame,
    timestamps: Optional[np.ndarray],
    column: str,
    period: Optional[int] = None,
    seasonal: Optional[int] = None,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    low_pass_jump: int = 1,
    robust: bool = False,
) -> Dict[str, np.ndarray]:
    """Single-series STL decomposition without timestamp grouping."""
    start_time = time.time()

    series = df.get_column(column).to_numpy()
    series = _interpolate_nans(series, method="linear")

    # Normalize the series
    series_normalized = (series - series.mean()) / (series.std() + 1e-10)

    # Perform STL decomposition
    stl_result = perform_stl_decomposition(
        series_normalized,
        period=period,
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        low_pass_deg=low_pass_deg,
        seasonal_jump=seasonal_jump,
        trend_jump=trend_jump,
        low_pass_jump=low_pass_jump,
        robust=robust,
        return_components=True,
        return_numpy=True,
    )

    # Reconstruct the signal by combining trend, seasonal, and residual
    reconstructed = (
        stl_result["trend"] + stl_result["seasonal"] + stl_result["resid"]
    )

    non_residual_reconstructed = stl_result["trend"] + stl_result["seasonal"]
    residual = np.abs(series_normalized - reconstructed)
    residual_ts_only = np.abs(series_normalized - non_residual_reconstructed)

    logger.info(
        f"STL decomposition time: {time.time() - start_time} on series of length {len(series)}"
    )

    return {
        "trend": stl_result["trend"],
        "seasonal": stl_result["seasonal"],
        "resid": stl_result["resid"],
        "reconstructed": reconstructed,
        "non_residual_reconstructed": non_residual_reconstructed,
        "residual_complete": residual,
        "residual": residual_ts_only,
        "mean_residual": np.mean(residual_ts_only),
        "normalized_original": series_normalized,
        # "period": period,
        # "stl_result": stl_result.get("stl_result"),
    }


def stl_decomposition(
    df: pl.DataFrame,
    column: str,
    period: Optional[int] = None,
    seasonal: Optional[int] = None,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    low_pass_jump: int = 1,
    robust: bool = False,
    timestamp_column: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform STL (Seasonal and Trend decomposition using Loess) decomposition on a time series.

    Args:
        df: Polars DataFrame
        column: Column name for decomposition
        period: Period of the seasonal component. If None, will be inferred from the data
        seasonal: Seasonal smoothing parameter. Must be odd. If None, will be set to period
        trend: Trend smoothing parameter. Must be odd. If None, will be set to period
        low_pass: Low-pass smoothing parameter. Must be odd. If None, will be set to period
        seasonal_deg: Degree of seasonal LOESS (0, 1, or 2)
        trend_deg: Degree of trend LOESS (0, 1, or 2)
        low_pass_deg: Degree of low-pass LOESS (0, 1, or 2)
        seasonal_jump: Jump size for seasonal LOESS
        trend_jump: Jump size for trend LOESS
        low_pass_jump: Jump size for low-pass LOESS
        robust: Whether to use robust estimation
        timestamp_column: Column name for timestamps to group data by (if provided)

    Returns:
        If timestamp_column is None: Dictionary with 'trend', 'seasonal', 'resid', 'reconstructed'
        If timestamp_column is provided: Results are concatenated across all timestamp groups
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Use apply_grouping to handle timestamp grouping
    grouped_results = apply_grouping(
        df,
        _stl_single,
        timestamp_column,
        None,
        column,
        period,
        seasonal,
        trend,
        low_pass,
        seasonal_deg,
        trend_deg,
        low_pass_deg,
        seasonal_jump,
        trend_jump,
        low_pass_jump,
        robust,
    )
    # Return both combined results and grouped results for plotting flexibility
    combined_result = combined_decomposition(grouped_results)
    combined_result["_grouped_results"] = (
        grouped_results  # Store grouped results for plotting
    )
    return combined_result


def forecast_fourier(
    decomposition_params: Dict[str, np.ndarray],
    forecast_steps: int,
    original_length: int,
    original_mean: float = 0.0,
    original_std: float = 1.0,
    series_to_forecast: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, None]]:
    """
    Forecast using Fourier decomposition parameters.

    Args:
        decomposition_params: Dictionary with 'frequencies', 'amplitudes', 'phases'
        forecast_steps: Number of steps to forecast
        original_length: Length of the original time series
        original_mean: Mean of the original time series (for denormalization)
        original_std: Standard deviation of the original time series (for denormalization)

    Returns:
        Dictionary with 'forecast' and 'historical_decomposition'
    """
    frequencies = decomposition_params["frequencies"]
    amplitudes = decomposition_params["amplitudes"]
    phases = decomposition_params["phases"]

    # Create time indices for forecasting
    forecast_indices = np.arange(
        original_length, original_length + forecast_steps
    )

    # Reconstruct forecast using the learned frequency components
    forecast = np.zeros(forecast_steps)

    for i, (freq, amp, phase) in enumerate(
        zip(frequencies, amplitudes, phases)
    ):
        if freq == 0:  # DC component
            forecast += amp / original_length
        else:
            forecast += (amp / original_length) * np.cos(
                2 * np.pi * freq * forecast_indices + phase
            )

    # Reconstruct historical decomposition
    historical_time = np.arange(original_length)
    historical_decomposition = np.zeros(original_length)

    for i, (freq, amp, phase) in enumerate(
        zip(frequencies, amplitudes, phases)
    ):
        if freq == 0:  # DC component
            historical_decomposition += amp / original_length
        else:
            historical_decomposition += (amp / original_length) * np.cos(
                2 * np.pi * freq * historical_time + phase
            )

    # Denormalize both
    forecast = forecast * original_std + original_mean
    historical_decomposition = (
        historical_decomposition * original_std + original_mean
    )
    normalized_forecast = forecast / original_std + original_mean
    normalized_original = None
    if series_to_forecast is not None:
        normalized_original = series_to_forecast / original_std + original_mean

    return {
        "forecast": forecast,
        "normalized_forecast": normalized_forecast,
        "historical_decomposition": historical_decomposition,
        "normalized_original": normalized_original,
    }


def forecast_ssa(
    decomposition_params: Dict[str, np.ndarray],
    forecast_steps: int,
    original_length: int,
    original_mean: float = 0.0,
    original_std: float = 1.0,
    series_to_forecast: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, None]]:
    """
    Forecast using SSA decomposition parameters.

    Args:
        decomposition_params: Dictionary with 'eigenvalues', 'eigenvectors', 'window_size'
        forecast_steps: Number of steps to forecast
        original_length: Length of the original time series
        original_mean: Mean of the original time series (for denormalization)
        original_std: Standard deviation of the original time series (for denormalization)

    Returns:
        Dictionary with 'forecast' and 'historical_decomposition'
    """
    eigenvectors = decomposition_params["eigenvectors"]
    window_size = decomposition_params.get("window_size", eigenvectors.shape[0])

    # Use the last window_size values to forecast
    # This is a simple approach - in practice, you might want more sophisticated forecasting
    forecast = np.zeros(forecast_steps)

    # For SSA, we can use the eigenvectors to project forward
    # This is a simplified approach - more sophisticated methods exist
    for step in range(forecast_steps):
        # Use the last few values and eigenvectors to predict next value
        if step < window_size:
            # Simple linear combination of eigenvectors
            forecast[step] = np.mean(
                eigenvectors[:, : min(3, eigenvectors.shape[1])].flatten()
            )
        else:
            # Use trend continuation
            forecast[step] = (
                forecast[step - 1]
                + (forecast[step - 1] - forecast[step - 2]) * 0.1
            )

    # Get historical decomposition from SSA
    historical_decomposition = decomposition_params.get(
        "trend", np.zeros(original_length)
    )
    if "seasonal" in decomposition_params:
        historical_decomposition += decomposition_params["seasonal"]

    # Denormalize both
    forecast = forecast * original_std + original_mean
    historical_decomposition = (
        historical_decomposition * original_std + original_mean
    )
    normalized_forecast = forecast / original_std + original_mean
    normalized_original = None
    if series_to_forecast is not None:
        normalized_original = series_to_forecast / original_std + original_mean

    return {
        "forecast": forecast,
        "historical_decomposition": historical_decomposition,
        "normalized_forecast": normalized_forecast,
        "normalized_original": normalized_original,
    }


def forecast_sindy(
    decomposition_params: Dict[str, np.ndarray],
    forecast_steps: int,
    original_length: int,
    original_mean: float = 0.0,
    original_std: float = 1.0,
    time_step: float = 0.001,
    series_to_forecast: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, None]]:
    """
    Forecast using SINDy decomposition parameters.

    Args:
        decomposition_params: Dictionary with 'coefficients', 'library_terms'
        forecast_steps: Number of steps to forecast
        original_length: Length of the original time series
        original_mean: Mean of the original time series (for denormalization)
        original_std: Standard deviation of the original time series (for denormalization)
        time_step: Time step for integration

    Returns:
        Dictionary with 'forecast' and 'historical_decomposition'
    """
    coefficients = decomposition_params["coefficients"]

    # Start with the last value from the original series
    last_value = decomposition_params.get("normalized_original", [0])[-1]

    forecast = np.zeros(forecast_steps)
    current_value = last_value

    for step in range(forecast_steps):
        # Create polynomial features for current value
        features = np.array(
            [current_value**i for i in range(len(coefficients))]
        )

        # Calculate derivative using SINDy coefficients
        derivative = np.dot(coefficients, features)

        # Integrate forward using Euler method
        current_value += derivative * time_step
        forecast[step] = current_value

    # Get historical decomposition from SINDy (polynomial fit)
    historical_time = np.arange(original_length)
    historical_decomposition = np.zeros(original_length)

    for i, coef in enumerate(coefficients):
        historical_decomposition += coef * (historical_time**i)

    # Denormalize both
    forecast = forecast * original_std + original_mean
    historical_decomposition = (
        historical_decomposition * original_std + original_mean
    )
    normalized_forecast = forecast / original_std + original_mean
    normalized_original = None
    if series_to_forecast is not None:
        normalized_original = series_to_forecast / original_std + original_mean

    return {
        "forecast": forecast,
        "historical_decomposition": historical_decomposition,
        "normalized_forecast": normalized_forecast,
        "normalized_original": normalized_original,
    }


def forecast_stl(
    decomposition_params: Dict[str, np.ndarray],
    forecast_steps: int,
    original_length: int,
    original_mean: float = 0.0,
    original_std: float = 1.0,
    period: Optional[int] = None,
    series_to_forecast: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, None]]:
    """
    Forecast using STL decomposition parameters.

    Args:
        decomposition_params: Dictionary with 'trend', 'seasonal', 'resid'
        forecast_steps: Number of steps to forecast
        original_length: Length of the original time series
        original_mean: Mean of the original time series (for denormalization)
        original_std: Standard deviation of the original time series (for denormalization)
        period: Period of the seasonal component

    Returns:
        Dictionary with 'forecast' and 'historical_decomposition'
    """
    trend = decomposition_params["trend"]
    seasonal = decomposition_params["seasonal"]
    resid = decomposition_params["resid"]

    # Infer period if not provided
    if period is None:
        # Try to find period from seasonal component
        if len(seasonal) > 1:
            # Simple autocorrelation-based period detection
            logger.warning(
                f"Using autocorrelation-based period detection for STL decomposition with seasonal component of length {len(seasonal)}"
            )
            autocorr = np.correlate(seasonal, seasonal, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            # Find first peak after lag 1
            peaks = np.where(np.diff(autocorr) < 0)[0]
            if len(peaks) > 0:
                period = peaks[0] + 1
            else:
                period = len(seasonal) // 4
        else:
            period = len(seasonal) // 4

    # Ensure period is not None
    if period is None:
        period = 1

    forecast = np.zeros(forecast_steps)

    for step in range(forecast_steps):
        # Extrapolate trend (simple linear extrapolation)
        if len(trend) >= 2:
            trend_slope = trend[-1] - trend[-2]
            trend_forecast = trend[-1] + trend_slope * (step + 1)
        else:
            trend_forecast = trend[-1] if len(trend) > 0 else 0

        # Extrapolate seasonal (repeat the pattern)
        seasonal_idx = (original_length + step) % period
        if seasonal_idx < len(seasonal):
            seasonal_forecast = seasonal[seasonal_idx]
        else:
            seasonal_forecast = seasonal[-1] if len(seasonal) > 0 else 0

        # Residual forecast (assume mean of residuals)
        resid_forecast = np.mean(resid) if len(resid) > 0 else 0

        # Combine components
        forecast[step] = trend_forecast + seasonal_forecast + resid_forecast

    # Get historical decomposition (trend + seasonal + residual)
    historical_decomposition = trend + seasonal + resid

    # Denormalize both
    forecast = forecast * original_std + original_mean
    historical_decomposition = (
        historical_decomposition * original_std + original_mean
    )
    normalized_forecast = forecast / original_std + original_mean
    normalized_original = None
    if series_to_forecast is not None:
        normalized_original = series_to_forecast / original_std + original_mean

    return {
        "forecast": forecast,
        "historical_decomposition": historical_decomposition,
        "normalized_forecast": normalized_forecast,
        "normalized_original": normalized_original,
    }


def forecast_with_decomposition(
    df: pl.DataFrame,
    column: str,
    decomposition_type: str,
    forecast_steps: int,
    decomposition_params: Optional[Dict[str, np.ndarray]] = None,
    timestamp_column: Optional[str] = None,
    series_to_forecast: Optional[np.ndarray] = None,
    **decomposition_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Perform forecasting using decomposition parameters.

    Args:
        df: Polars DataFrame with training data
        column: Column name for forecasting
        decomposition_type: Type of decomposition ('fourier', 'ssa', 'sindy', 'stl')
        forecast_steps: Number of steps to forecast
        decomposition_params: Pre-computed decomposition parameters (if None, will compute)
        timestamp_column: Column name for timestamps to group data by (if None, no grouping)
        series_to_forecast: Optional test series for comparison (if None, no comparison)
        **decomposition_kwargs: Additional arguments for decomposition

    Returns:
        Dictionary with 'forecast', 'decomposition_params', and other relevant information.
        If grouping is applied, returns a list of dictionaries for each group.
    """

    def _forecast_single_group(
        group_df: pl.DataFrame, group_timestamps: Optional[np.ndarray] = None
    ) -> Dict[
        str, Union[np.ndarray, Dict[str, np.ndarray], str, int, float, None]
    ]:
        """Helper function to forecast a single group."""
        if decomposition_params is None:
            # Compute decomposition parameters for this group
            if decomposition_type.lower() == "fourier":
                group_decomposition_params = fourier_decomposition(
                    group_df, column, **decomposition_kwargs
                )
            elif decomposition_type.lower() == "ssa":
                group_decomposition_params = singular_spectrum_analysis(
                    group_df, column, **decomposition_kwargs
                )
            elif decomposition_type.lower() == "sindy":
                group_decomposition_params = sindy_decomposition(
                    group_df, column, **decomposition_kwargs
                )
            elif decomposition_type.lower() == "stl":
                group_decomposition_params = stl_decomposition(
                    group_df, column, **decomposition_kwargs
                )
            else:
                raise ValueError(
                    f"Unsupported decomposition type: {decomposition_type}"
                )
        else:
            group_decomposition_params = decomposition_params

        # Get original series statistics for denormalization
        original_series = group_df.get_column(column).to_numpy()
        original_series = _interpolate_nans(original_series, method="linear")
        original_mean = float(np.mean(original_series))
        original_std = float(np.std(original_series))
        original_length = len(original_series)

        # Perform forecasting based on decomposition type
        if decomposition_type.lower() == "fourier":
            forecast_result = forecast_fourier(
                group_decomposition_params,
                forecast_steps,
                original_length,
                original_mean,
                original_std,
                series_to_forecast=series_to_forecast,
            )
        elif decomposition_type.lower() == "ssa":
            forecast_result = forecast_ssa(
                group_decomposition_params,
                forecast_steps,
                original_length,
                original_mean,
                original_std,
                series_to_forecast=series_to_forecast,
            )
        elif decomposition_type.lower() == "sindy":
            forecast_result = forecast_sindy(
                group_decomposition_params,
                forecast_steps,
                original_length,
                original_mean,
                original_std,
                series_to_forecast=series_to_forecast,
            )
        elif decomposition_type.lower() == "stl":
            forecast_result = forecast_stl(
                group_decomposition_params,
                forecast_steps,
                original_length,
                original_mean,
                original_std,
                series_to_forecast=series_to_forecast,
            )

        result = {
            "forecast": forecast_result["forecast"],
            "historical_decomposition": forecast_result[
                "historical_decomposition"
            ],
            "normalized_forecast": forecast_result.get("normalized_forecast"),
            "normalized_original": forecast_result.get("normalized_original"),
            "decomposition_params": group_decomposition_params,
            "decomposition_type": decomposition_type,
            "forecast_steps": forecast_steps,
            "original_length": original_length,
            "original_mean": original_mean,
            "original_std": original_std,
            "test_values": series_to_forecast
            if series_to_forecast is not None
            else np.array([]),
        }
        return result

    # Apply grouping if requested
    res = apply_grouping(df, _forecast_single_group, timestamp_column, None)
    combined_result = combined_decomposition(res)
    combined_result["_grouped_results"] = (
        res  # Store grouped results for plotting
    )
    return combined_result


def evaluate_forecast(
    actual: np.ndarray,
    forecast: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate forecast accuracy using various metrics.

    Args:
        actual: Actual values
        forecast: Forecasted values
        metrics: List of metrics to compute (default: ['mae', 'mse', 'rmse', 'mape'])

    Returns:
        Dictionary with metric names and values
    """
    if metrics is None:
        metrics = ["mae", "mse", "rmse", "mape"]

    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast arrays must have the same length")

    # Remove any NaN or infinite values
    mask = np.isfinite(actual) & np.isfinite(forecast)
    actual_clean = actual[mask]
    forecast_clean = forecast[mask]

    if len(actual_clean) == 0:
        return {metric: np.nan for metric in metrics}

    results = {}

    if "mae" in metrics:
        results["mae"] = np.mean(np.abs(actual_clean - forecast_clean))

    if "mse" in metrics:
        results["mse"] = np.mean((actual_clean - forecast_clean) ** 2)

    if "rmse" in metrics:
        results["rmse"] = (
            np.sqrt(results["mse"])
            if "mse" in results
            else np.sqrt(np.mean((actual_clean - forecast_clean) ** 2))
        )

    if "mape" in metrics:
        # Avoid division by zero
        non_zero_mask = actual_clean != 0
        if np.any(non_zero_mask):
            results["mape"] = (
                np.mean(
                    np.abs(
                        (
                            actual_clean[non_zero_mask]
                            - forecast_clean[non_zero_mask]
                        )
                        / actual_clean[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            results["mape"] = np.nan

    return results
