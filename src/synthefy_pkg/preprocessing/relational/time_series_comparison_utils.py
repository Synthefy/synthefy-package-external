import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from numba import jit
from scipy import stats
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from statsmodels.tsa.stattools import grangercausalitytests

SERIES_COMPARISON_METHODS = [
    "euclidean",
    "dtw",
    "twed",
    "lcss",
    "granger",
    "cross_corr",
    "pearson",
]


def euclidean_distance(ts1, ts2):
    """
    Calculate Euclidean distance between two time series of equal length.

    Args:
        ts1 (numpy.ndarray): First time series
        ts2 (numpy.ndarray): Second time series

    Returns:
        float: Euclidean distance between the series
    """
    # Ensure equal length
    min_length = min(len(ts1), len(ts2))
    ts1 = ts1[:min_length]
    ts2 = ts2[:min_length]

    return np.sqrt(np.sum((ts1 - ts2) ** 2))


def dynamic_time_warping(ts1, ts2, window=None):
    """
    Calculate Dynamic Time Warping distance between two time series.
    Handles time series of different lengths using a highly optimized implementation.

    Args:
        ts1 (numpy.ndarray): First time series
        ts2 (numpy.ndarray): Second time series
        window (int): Sakoe-Chiba band width (None=no constraint)

    Returns:
        float: DTW distance between the series
    """
    # Convert to correct format
    ts1 = np.array(ts1, dtype=np.float64)
    ts2 = np.array(ts2, dtype=np.float64)

    # Early termination for edge cases
    n, m = len(ts1), len(ts2)
    if n == 0 or m == 0:
        return np.inf
    if n == 1 and m == 1:
        return abs(ts1[0] - ts2[0])

    return _fast_dtw_small(ts1, ts2, window)


@jit(nopython=True)
def _fast_dtw_small(ts1, ts2, window=None):
    """JIT-compiled DTW for small series"""
    n, m = len(ts1), len(ts2)

    # Set window size if not specified
    if window is None:
        window = max(n, m)

    # Initialize distance matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill the matrix using dynamic programming
    for i in range(1, n + 1):
        # Apply window constraint
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = abs(ts1[i - 1] - ts2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return dtw_matrix[n, m]


def time_warp_edit_distance(ts1, ts2, lambda_param=1.0, nu_param=0.001):
    """
    Calculate Time Warp Edit Distance between two time series.
    TWED combines time warping with edit distance concepts.

    Args:
        ts1 (numpy.ndarray): First time series with timestamps
        ts2 (numpy.ndarray): Second time series with timestamps
        lambda_param (float): Stiffness parameter
        nu_param (float): Penalty for deletion/insertion operations

    Returns:
        float: TWED distance between the series
    """
    # If timestamps not provided, use indices
    if isinstance(ts1, np.ndarray) and ts1.ndim == 1:
        t1 = np.arange(len(ts1))
        x1 = ts1
    else:
        t1, x1 = ts1[:, 0], ts1[:, 1]

    if isinstance(ts2, np.ndarray) and ts2.ndim == 1:
        t2 = np.arange(len(ts2))
        x2 = ts2
    else:
        t2, x2 = ts2[:, 0], ts2[:, 1]

    # Convert to numpy arrays for Numba compatibility
    t1 = np.asarray(t1, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    return _fast_twed(t1, x1, t2, x2, lambda_param, nu_param)


@jit(nopython=True)
def _fast_twed(t1, x1, t2, x2, lambda_param, nu_param):
    """
    JIT-compiled implementation of TWED calculation.
    """
    n, m = len(x1), len(x2)

    # Initialize distance matrix
    dp = np.zeros((n + 1, m + 1))

    # Vectorized initialization of first column
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + (
            abs(x1[i - 1]) + nu_param * abs(t1[i - 1] - t1[i - 2])
            if i > 1
            else abs(x1[i - 1])
        )

    # Vectorized initialization of first row
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + (
            abs(x2[j - 1]) + nu_param * abs(t2[j - 1] - t2[j - 2])
            if j > 1
            else abs(x2[j - 1])
        )

    # Pre-compute absolute differences for all pairs to avoid repeated calculations
    # This avoids repeating these calculations in the inner loop
    for i in range(1, n + 1):
        i_idx = i - 1

        for j in range(1, m + 1):
            j_idx = j - 1

            # Calculate distances more efficiently
            dist1 = abs(x1[i_idx] - x2[j_idx])  # Current points

            # Previous points
            if i > 1 and j > 1:
                dist2 = abs(x1[i_idx - 1] - x2[j_idx - 1])
                time_dist1 = nu_param * abs(t1[i_idx] - t2[j_idx])
                time_dist2 = nu_param * abs(t1[i_idx - 1] - t2[j_idx - 1])
            else:
                dist2 = 0
                time_dist1 = time_dist2 = 0

            # Cost for each operation
            c1 = dist1 + dist2 + time_dist1 + time_dist2  # Match

            # Deletion/insertion time penalties
            time_penalty1 = (
                nu_param * abs(t1[i_idx] - t1[i_idx - 1]) if i > 1 else 0
            )
            time_penalty2 = (
                nu_param * abs(t2[j_idx] - t2[j_idx - 1]) if j > 1 else 0
            )

            c2 = abs(x1[i_idx]) + time_penalty1 + lambda_param  # Deletion
            c3 = abs(x2[j_idx]) + time_penalty2 + lambda_param  # Insertion

            # Choose minimum cost operation
            dp[i, j] = min(
                dp[i - 1, j - 1] + c1, dp[i - 1, j] + c2, dp[i, j - 1] + c3
            )

    return dp[n, m]


def longest_common_subsequence(ts1, ts2, epsilon=0.1, window=None):
    """
    Calculate Longest Common Subsequence similarity between two time series.
    Handles noise using epsilon threshold for matching.

    Args:
        ts1 (numpy.ndarray): First time series
        ts2 (numpy.ndarray): Second time series
        epsilon (float): Threshold for considering two points as matching
        window (int): Window constraint limit (None=no constraint)

    Returns:
        float: LCSS similarity (0-1 range, higher is more similar)
    """
    # Convert to numpy arrays
    ts1 = np.asarray(ts1, dtype=np.float64)
    ts2 = np.asarray(ts2, dtype=np.float64)

    n, m = len(ts1), len(ts2)

    # Edge cases
    if n == 0 or m == 0:
        return 0.0
    if n == 1 and m == 1:
        return 1.0 if abs(ts1[0] - ts2[0]) < epsilon else 0.0

    # Choose implementation based on input size
    lcss_val = _fast_lcss_small(ts1, ts2, epsilon, window)

    # Return normalized similarity (0-1 range)
    return lcss_val / min(n, m)


@jit(nopython=True)
def _fast_lcss_small(ts1, ts2, epsilon=0.1, window=None):
    """
    JIT-compiled LCSS for small time series (full matrix approach)
    """
    n, m = len(ts1), len(ts2)

    # Set window size if not specified
    if window is None:
        window = max(n, m)

    # Initialize LCSS matrix
    lcss = np.zeros((n + 1, m + 1))

    # Pre-compute distance matrix for performance
    # This avoids repeatedly calculating distances in the inner loop
    match_matrix = np.zeros((n, m), dtype=np.bool_)
    for i in range(n):
        j_start = max(0, i - window)
        j_end = min(m, i + window + 1)
        for j in range(j_start, j_end):
            match_matrix[i, j] = abs(ts1[i] - ts2[j]) < epsilon

    # Fill the LCSS matrix
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            # If points match (using pre-computed match matrix)
            if match_matrix[i - 1, j - 1]:
                lcss[i, j] = lcss[i - 1, j - 1] + 1
            else:
                lcss[i, j] = max(lcss[i - 1, j], lcss[i, j - 1])

    return lcss[n, m]


@lru_cache(maxsize=1024)
def _cached_granger_test(ts1_tuple, ts2_tuple, max_lag):
    """
    Cached version of Granger causality test.
    Converts arrays to tuples for hashability.
    """
    # Convert back to arrays
    ts1 = np.array(ts1_tuple)
    ts2 = np.array(ts2_tuple)

    # Quick early exits for invalid cases
    if (
        len(ts1) <= max_lag + 2
        or len(ts2) <= max_lag + 2
        or np.std(ts1) < 1e-10
        or np.std(ts2) < 1e-10
    ):
        return 1.0

    # Stack time series for Granger test
    data = np.column_stack((ts2, ts1))

    # Run test with minimal computation
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except InfeasibleTestError:
        return 1.0

    # Extract only the p-values we need
    p_values = [result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
    return np.max(p_values)


def granger_causality(ts1, ts2, max_lag=5):
    """
    Calculate linear Granger causality between two time series.
    Tests if ts1 Granger-causes ts2.

    Args:
        ts1 (numpy.ndarray): First time series (potential cause)
        ts2 (numpy.ndarray): Second time series (potential effect)
        max_lag (int): Maximum number of lags to test

    Returns:
        float: Maximum p-value across lags (lower values suggest causality)
    """
    # Early exit if either series has too many NaN/inf values
    ts1_clean = np.array(ts1, dtype=np.float64)
    ts2_clean = np.array(ts2, dtype=np.float64)

    # Ensure equal length
    min_length = min(len(ts1_clean), len(ts2_clean))
    ts1_clean = ts1_clean[:min_length]
    ts2_clean = ts2_clean[:min_length]

    # Convert to tuples for caching
    ts1_tuple = tuple(ts1_clean)
    ts2_tuple = tuple(ts2_clean)

    # Call cached function
    result = 1 - _cached_granger_test(ts1_tuple, ts2_tuple, max_lag)
    # print(f"Granger causality calculation time: {end_time - start_time} seconds, result: {result}")
    return result


def fast_granger_causality(ts1, ts2, max_lag=5):
    """
    Calculate linear Granger causality between two time series using a simplified,
    parallelism-friendly implementation.
    Tests if ts1 Granger-causes ts2.

    Args:
        ts1 (numpy.ndarray): First time series (potential cause)
        ts2 (numpy.ndarray): Second time series (potential effect)
        max_lag (int): Maximum number of lags to test

    Returns:
        float: Granger causality measure (higher values suggest stronger causality)
    """

    # Quick validation
    ts1 = np.asarray(ts1, dtype=np.float64)
    ts2 = np.asarray(ts2, dtype=np.float64)

    # Ensure equal length
    min_length = min(len(ts1), len(ts2))
    if min_length <= max_lag + 2:
        return 0.0  # Not enough data for testing

    ts1 = ts1[:min_length]
    ts2 = ts2[:min_length]

    # Check for constant series
    if np.std(ts1) < 1e-10 or np.std(ts2) < 1e-10:
        return 0.0

    # Number of samples in the output
    n_samples = min_length - max_lag

    # Create target variable
    y = ts2[max_lag:]

    # Create indices for all lags at once
    # For each position in the output, we need values from (t-1) to (t-max_lag)
    lag_indices = np.arange(max_lag)[:, None] + np.arange(n_samples)

    # Use advanced indexing to get all lags at once for both series
    # This creates matrices of shape (max_lag, n_samples)
    ts2_lags = ts2[lag_indices]
    ts1_lags = ts1[lag_indices]

    # Transpose to get (n_samples, max_lag) shape and ensure C-contiguous layout
    X_restricted = np.ascontiguousarray(ts2_lags.T)

    # Create the unrestricted model features by concatenating both lag matrices
    X_unrestricted = np.hstack([X_restricted, np.ascontiguousarray(ts1_lags.T)])

    # Fit restricted model (only ts2 lags)
    model_restricted = LinearRegression(fit_intercept=True)
    model_restricted.fit(X_restricted, y)
    restricted_pred = model_restricted.predict(X_restricted)
    restricted_rss = np.sum((y - restricted_pred) ** 2)

    # Fit unrestricted model (both ts1 and ts2 lags)
    model_unrestricted = LinearRegression(fit_intercept=True)
    model_unrestricted.fit(X_unrestricted, y)
    unrestricted_pred = model_unrestricted.predict(X_unrestricted)
    unrestricted_rss = np.sum((y - unrestricted_pred) ** 2)

    # Handle cases where models are identical or have errors
    if (
        restricted_rss <= 0
        or unrestricted_rss <= 0
        or restricted_rss <= unrestricted_rss
    ):
        return 0.0

    # Calculate improvement ratio (simpler than F-test but effective)
    improvement = (restricted_rss - unrestricted_rss) / restricted_rss

    return improvement


# def gpu_granger_causality(ts1, ts2, max_lag=2):
#     """
#     Calculate linear Granger causality between two time series using batch operations.
#     Tests if ts1 Granger-causes ts2.

#     Args:
#         ts1: First time series (potential cause)
#         ts2: Second time series (potential effect)
#         max_lag: Maximum lag to test

#     Returns:
#         Best F-statistic across all tested lags
#     """
#     import cudf
#     import cupy as cp
#     import numpy as np
#     from cuml.linear_model import LinearRegression

#     # Ensure input is right shape and length
#     n = len(ts2)
#     if len(ts1) != n:
#         raise ValueError("Both time series must have the same length")

#     if n <= max_lag:
#         raise ValueError("Time series length must be greater than max_lag")

#     # Create all lagged features for both time series at once
#     # This creates features for the maximum lag, which we'll subset later
#     X_full = cudf.DataFrame()
#     y = cudf.Series(ts2[max_lag:])

#     # Create lagged features for ts2
#     for i in range(1, max_lag + 1):
#         X_full[f"ts2_lag_{i}"] = ts2[max_lag - i : -i]

#     # Create lagged features for ts1
#     for i in range(1, max_lag + 1):
#         X_full[f"ts1_lag_{i}"] = ts1[max_lag - i : -i]

#     # Prepare arrays to store RSS values for different lag values
#     restricted_rss_array = cp.zeros(max_lag)
#     unrestricted_rss_array = cp.zeros(max_lag)
#     n_obs = len(y)

#     # Get column names for easy filtering
#     ts2_cols = [f"ts2_lag_{i}" for i in range(1, max_lag + 1)]

#     # Create a single restricted model and single unrestricted model
#     # Then use them to calculate RSS for all lag values
#     restricted_model = LinearRegression(algorithm="eig", fit_intercept=True)
#     unrestricted_model = LinearRegression(algorithm="eig", fit_intercept=True)

#     # Process all lags at once
#     for lag in range(1, max_lag + 1):
#         # Get columns for this lag
#         restricted_cols = [f"ts2_lag_{i}" for i in range(1, lag + 1)]
#         unrestricted_cols = restricted_cols + [
#             f"ts1_lag_{i}" for i in range(1, lag + 1)
#         ]

#         # Fit restricted model
#         X_restricted = X_full[restricted_cols]
#         restricted_model.fit(X_restricted, y)
#         restricted_pred = restricted_model.predict(X_restricted)
#         restricted_residuals = y - restricted_pred
#         restricted_rss = (restricted_residuals**2).sum()

#         # Fit unrestricted model
#         X_unrestricted = X_full[unrestricted_cols]
#         unrestricted_model.fit(X_unrestricted, y)
#         unrestricted_pred = unrestricted_model.predict(X_unrestricted)
#         unrestricted_residuals = y - unrestricted_pred
#         unrestricted_rss = (unrestricted_residuals**2).sum()

#         # Store results
#         restricted_rss_array[lag - 1] = float(restricted_rss)
#         unrestricted_rss_array[lag - 1] = float(unrestricted_rss)

#     # Calculate F-statistics for all lags at once using array operations
#     df_restricted = cp.arange(1, max_lag + 1)
#     df_unrestricted = 2 * df_restricted

#     # Avoid division by zero
#     valid_mask = (restricted_rss_array > 0) & (unrestricted_rss_array > 0)

#     # Initialize with zeros
#     f_statistics = cp.zeros(max_lag)

#     # Only calculate where valid
#     if cp.any(valid_mask):
#         f_statistics[valid_mask] = (
#             (
#                 restricted_rss_array[valid_mask]
#                 - unrestricted_rss_array[valid_mask]
#             )
#             / df_restricted[valid_mask]
#         ) / (
#             unrestricted_rss_array[valid_mask]
#             / (n_obs - df_unrestricted[valid_mask])
#         )

#     # Return the best F-statistic
#     return float(cp.max(f_statistics))


def cross_correlation(ts1, ts2, max_lag=None, normalize=True):
    """
    Calculate cross-correlation between two time series.

    Args:
        ts1 (numpy.ndarray): First time series
        ts2 (numpy.ndarray): Second time series
        max_lag (int): Maximum lag to consider. If None, use min(len(ts1), len(ts2))-1
        normalize (bool): Whether to normalize the result

    Returns:
        tuple: (correlations, lags) - correlation values and corresponding lags
    """
    # Ensure arrays
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)

    # Set default max_lag
    if max_lag is None:
        max_lag = min(20, min(len(ts1), len(ts2)) - 1)

    # Calculate cross-correlation
    correlations = correlate(ts1, ts2, mode="full")

    # Get lag values
    n = len(correlations)
    lags = np.arange(-(n // 2), n // 2 + 1)

    # Center and truncate to requested max_lag
    center = n // 2
    start = center - max_lag
    end = center + max_lag + 1
    correlations = correlations[start:end]
    lags = lags[start:end]

    # Normalize if requested
    if normalize:
        # Normalize to [0, 1] range
        # First normalize to [-1, 1] using proper cross-correlation formula
        norm = np.sqrt(np.sum(ts1**2) * np.sum(ts2**2))
        if norm > 0:
            correlations = correlations / norm

        # Then shift and scale to [0, 1] range
        correlations = (correlations + 1) / 2

    return correlations, lags


def pearson_correlation(ts1, ts2):
    """
    Calculate Pearson correlation coefficient between two time series.

    Args:
        ts1 (numpy.ndarray): First time series
        ts2 (numpy.ndarray): Second time series

    Returns:
        float: Pearson correlation coefficient (-1 to 1)
    """
    # Ensure equal length
    min_length = min(len(ts1), len(ts2))
    ts1 = ts1[:min_length]
    ts2 = ts2[:min_length]

    # Calculate correlation coefficient
    result = stats.pearsonr(ts1, ts2)
    corr = result[0] if isinstance(result, tuple) else result.statistic

    if np.isnan(np.array(corr)):
        return 0.0
    return corr


def compare_time_series(ts1, ts2, method="euclidean"):
    """
    Compare two time series using multiple methods.

    Args:
        ts1 (numpy.ndarray): First time series
        ts2 (numpy.ndarray): Second time series
        methods (list): List of method names to use. If None, use all methods.
            Valid options: 'euclidean', 'dtw', 'twed', 'lcss', 'granger',
                          'cross_corr', 'pearson'

    Returns:
        point value comparison between two time series
    """
    if method == "euclidean":
        euclidian = euclidean_distance(ts1, ts2)
        assert not np.isnan(euclidian)
        return euclidian
    elif method == "dtw":
        dtw = dynamic_time_warping(ts1, ts2)
        assert not np.isnan(dtw)
        return dtw
    elif method == "twed":
        twed = time_warp_edit_distance(ts1, ts2)
        assert not np.isnan(twed)
        return twed
    elif method == "lcss":
        lcss = longest_common_subsequence(ts1, ts2)
        assert not np.isnan(lcss)
        return lcss
    elif method == "granger":
        # granger = granger_causality(ts1, ts2)
        granger = fast_granger_causality(ts1, ts2)

        assert not np.isnan(granger)
        return granger
    elif method == "cross_corr":
        corr, lags = cross_correlation(ts1, ts2)
        assert not np.isnan(corr).any()
        # key_statistics = {
        #     "max_corr": np.max(np.abs(corr)),
        #     "lag_at_max": lags[np.argmax(np.abs(corr))],
        #     "full_corr": corr,
        #     "lags": lags,
        # }
        # logger.info(f"Cross-correlation statistics: {key_statistics}")
        return np.max(np.abs(corr))
    elif method == "pearson":
        pearson = pearson_correlation(ts1, ts2)
        return pearson
    else:
        raise ValueError(f"Invalid method: {method}")
