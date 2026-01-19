"""Relational operations for data analysis."""

import time
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import entropy
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

from synthefy_pkg.preprocessing.data_summary_utils.decompositions import (
    _convert_time_column,
    apply_grouping,
)
from synthefy_pkg.preprocessing.data_summary_utils.summary_utils import (
    _interpolate_nans,
)
from synthefy_pkg.preprocessing.relational.convergent_cross_mapping import (
    ConvergentCrossMapping,
)
from synthefy_pkg.preprocessing.relational.dlinear import (
    DLinear,
    WindowedTimeSeries,
)
from synthefy_pkg.preprocessing.relational.time_series_comparison_utils import (
    fast_granger_causality,
)


def _preprocess_series_for_relational_analysis(
    df: pl.DataFrame,
    col1: str,
    col2: str,
    interpolate_nans: bool = True,
    interpolation_method: str = "linear",
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, Tuple[float, float]]]:
    """
    Helper function to perform common preprocessing operations for relational analysis functions.

    This function consolidates the common preprocessing steps used across transfer entropy,
    mutual information, cross correlation, granger causality, and dlinear causality functions.

    Args:
        df: Polars DataFrame containing the time series data
        col1: First column name
        col2: Second column name
        interpolate_nans: Whether to interpolate NaN values (default: True)
        interpolation_method: Method for interpolation ('linear', 'forward', 'backward')
        normalize: Whether to normalize the series (default: False)

    Returns:
        Tuple of (processed_series1, processed_series2, normalization_stats) where:
        - processed_series1, processed_series2: numpy arrays
        - normalization_stats: dict with keys 'col1' and 'col2', each containing (mean, std) tuples

    Raises:
        ValueError: If columns are not found in DataFrame
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1} or {col2} not found in DataFrame")

    # Convert to numpy arrays
    series1 = df.get_column(col1).to_numpy()
    series2 = df.get_column(col2).to_numpy()

    # Handle NaN values
    if interpolate_nans:
        series1 = _interpolate_nans(series1, interpolation_method)
        series2 = _interpolate_nans(series2, interpolation_method)
    else:
        # Remove rows where either series has NaN
        valid_mask = ~(np.isnan(series1) | np.isnan(series2))
        series1 = series1[valid_mask]
        series2 = series2[valid_mask]

    # Align series to same length
    if len(series1) != len(series2):
        min_len = min(len(series1), len(series2))
        series1 = series1[:min_len]
        series2 = series2[:min_len]

    # Store original statistics for denormalization
    normalization_stats = {
        col1: (float(np.mean(series1)), float(np.std(series1))),
        col2: (float(np.mean(series2)), float(np.std(series2))),
    }

    # Apply normalization if requested
    if normalize:
        series1 = (series1 - normalization_stats[col1][0]) / max(
            normalization_stats[col1][1], 1e-6
        )
        series2 = (series2 - normalization_stats[col2][0]) / max(
            normalization_stats[col2][1], 1e-6
        )

    return series1, series2, normalization_stats


# Training utilities borrowed from dlinear.py
def to_device(batch, device):
    """Move batch to device, handling lists and tuples."""
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    return batch.to(device)


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, grad_clip: Optional[float] = None
):
    """Train model for one epoch using DataLoader."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = to_device(xb, device), to_device(yb, device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate_model(model, loader, device, loss_fn):
    """Evaluate model using DataLoader."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb, yb = to_device(xb, device), to_device(yb, device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total_loss += loss.item() * len(xb)
        n_samples += len(xb)
    return total_loss / max(1, n_samples)


class TransferEntropyEstimator:
    """
    Transfer entropy estimator with automatic binning for continuous data.

    Transfer entropy measures the directional flow of information between two
    time series. It quantifies how much information from one series helps
    predict the future of another series, beyond what can be predicted from
    the target series' own past.

    This implementation uses histogram binning to discretize continuous data
    and estimates transfer entropy using the discrete formula.
    """

    def __init__(
        self,
        bins: int = 5,
        binning_method: str = "equal",
        lag_source: int = 1,
        lag_target: int = 1,
        normalize: bool = True,
        normalized: bool = False,
        min_samples_per_bin: int = 1,
    ):
        """
        Initialize the transfer entropy estimator.

        Args:
            bins: Number of bins for discretization
            binning_method: Method for binning ('equal', 'quantile', 'kmeans')
            lag_source: Number of lags to consider for source series
            lag_target: Number of lags to consider for target series
            normalize: Whether to normalize the result by the target's entropy
            normalized: Whether to return normalized transfer entropy (0-1 range)
            min_samples_per_bin: Minimum samples required per bin for reliability
        """
        self.bins = bins
        self.binning_method = binning_method
        self.lag_source = lag_source
        self.lag_target = lag_target
        self.normalize = normalize
        self.normalized = normalized
        self.min_samples_per_bin = min_samples_per_bin

        if binning_method not in ["equal", "quantile", "kmeans"]:
            raise ValueError(f"Unknown binning method: {binning_method}")

    def _discretize_equal(self, data: np.ndarray) -> np.ndarray:
        """Discretize data using equal-width bins."""
        data_min, data_max = data.min(), data.max()
        if data_max == data_min:
            return np.zeros_like(data, dtype=int)

        bin_edges = np.linspace(data_min, data_max, self.bins + 1)
        bin_edges[0] = -np.inf  # Include minimum value
        bin_edges[-1] = np.inf  # Include maximum value

        return np.digitize(data, bin_edges) - 1

    def _discretize_quantile(self, data: np.ndarray) -> np.ndarray:
        """Discretize data using quantile-based bins."""
        if self.bins == 1:
            return np.zeros_like(data, dtype=int)

        quantiles = np.linspace(0, 1, self.bins + 1)
        bin_edges = np.quantile(data, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        return np.digitize(data, bin_edges) - 1

    def _discretize_kmeans(self, data: np.ndarray) -> np.ndarray:
        """Discretize data using k-means clustering."""
        from sklearn.cluster import KMeans

        if len(data) < self.bins:
            return np.zeros_like(data, dtype=int)

        # Reshape for k-means
        data_reshaped = data.reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.bins, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(data_reshaped)

        return labels

    def discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Discretize continuous data into bins.

        Args:
            data: Input time series data

        Returns:
            Discretized data with integer bin labels
        """
        if self.binning_method == "equal":
            return self._discretize_equal(data)
        elif self.binning_method == "quantile":
            return self._discretize_quantile(data)
        elif self.binning_method == "kmeans":
            return self._discretize_kmeans(data)
        else:
            raise ValueError(f"Unknown binning method: {self.binning_method}")

    def _validate_bins(self, discretized_data: np.ndarray) -> bool:
        """Check if discretization produced valid bins."""
        unique_bins = np.unique(discretized_data)

        # Check if we have enough bins
        if len(unique_bins) < 2:
            return False

        # Check if each bin has enough samples
        for bin_val in unique_bins:
            bin_count = np.sum(discretized_data == bin_val)
            if bin_count < self.min_samples_per_bin:
                return False

        return True

    def _estimate_joint_probability(
        self, source: np.ndarray, target: np.ndarray, target_lagged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate joint probability distributions needed for transfer entropy.

        Args:
            source: Discretized source series
            target: Discretized target series
            target_lagged: Discretized lagged target series

        Returns:
            Tuple of (p_yt_ytlag, p_yt_ytlag_xs, p_ytlag_xs)
        """
        # Create joint states
        joint_yt_ytlag = np.column_stack([target, target_lagged])
        joint_yt_ytlag_xs = np.column_stack([target, target_lagged, source])
        joint_ytlag_xs = np.column_stack([target_lagged, source])

        # Estimate probabilities using frequency counting
        p_yt_ytlag = self._estimate_probability(joint_yt_ytlag)
        p_yt_ytlag_xs = self._estimate_probability(joint_yt_ytlag_xs)
        p_ytlag_xs = self._estimate_probability(joint_ytlag_xs)

        return p_yt_ytlag, p_yt_ytlag_xs, p_ytlag_xs

    def _estimate_probability(self, joint_states: np.ndarray) -> np.ndarray:
        """Estimate probability distribution from joint states."""
        # Count occurrences of each unique state
        unique_states, counts = np.unique(
            joint_states, axis=0, return_counts=True
        )

        # Create probability array
        max_vals = np.max(joint_states, axis=0) + 1
        prob_shape = tuple(max_vals)
        probabilities = np.zeros(prob_shape)

        # Fill in probabilities
        for state, count in zip(unique_states, counts):
            probabilities[tuple(state)] = count

        # Normalize
        probabilities = probabilities / np.sum(probabilities)

        return probabilities

    def _calculate_conditional_entropy(
        self,
        p_yt_ytlag: np.ndarray,
        p_yt_ytlag_xs: np.ndarray,
        p_ytlag_xs: np.ndarray,
    ) -> float:
        """Calculate conditional entropy terms for transfer entropy."""
        # Avoid log(0) by adding small epsilon
        eps = 1e-10

        # Calculate conditional probabilities
        p_yt_given_ytlag = p_yt_ytlag / (
            np.sum(p_yt_ytlag, axis=0, keepdims=True) + eps
        )
        p_yt_given_ytlag_xs = p_yt_ytlag_xs / (
            np.sum(p_yt_ytlag_xs, axis=0, keepdims=True) + eps
        )

        # Calculate entropies
        h_yt_given_ytlag = -np.sum(p_yt_ytlag * np.log(p_yt_given_ytlag + eps))
        h_yt_given_ytlag_xs = -np.sum(
            p_yt_ytlag_xs * np.log(p_yt_given_ytlag_xs + eps)
        )

        return h_yt_given_ytlag - h_yt_given_ytlag_xs

    def estimate(self, source: np.ndarray, target: np.ndarray) -> float:
        """
        Estimate transfer entropy from source to target.

        Args:
            source: Source time series (potential cause)
            target: Target time series (potential effect)

        Returns:
            Transfer entropy value (bits)
        """
        # Ensure equal length
        min_length = min(len(source), len(target))
        if min_length < max(self.lag_source, self.lag_target) + 1:
            logger.debug("Insufficient data for transfer entropy calculation")
            return 0.0

        source = source[:min_length]
        target = target[:min_length]

        # Discretize data
        source_disc = self.discretize(source)
        target_disc = self.discretize(target)

        # Validate discretization
        if not self._validate_bins(source_disc) or not self._validate_bins(
            target_disc
        ):
            logger.debug("Invalid discretization, reducing number of bins")
            if (
                self.bins - 1 < 2
                or np.std(source_disc) < 1e-6
                or np.std(target_disc) < 1e-6
            ):
                logger.debug("Too few bins, returning 0.0")
                return 0.0
            self.bins = max(2, self.bins - 1)
            return self.estimate(source, target)

        # Create lagged series
        max_lag = max(self.lag_source, self.lag_target)
        source_lagged = source_disc[
            max_lag - self.lag_source : min_length - self.lag_source
        ]
        target_lagged = target_disc[
            max_lag - self.lag_target : min_length - self.lag_target
        ]
        target_current = target_disc[max_lag:]

        # Ensure all series have same length
        min_len = min(
            len(source_lagged), len(target_lagged), len(target_current)
        )
        source_lagged = source_lagged[:min_len]
        target_lagged = target_lagged[:min_len]
        target_current = target_current[:min_len]

        # Estimate joint probabilities
        p_yt_ytlag, p_yt_ytlag_xs, p_ytlag_xs = (
            self._estimate_joint_probability(
                source_lagged, target_current, target_lagged
            )
        )

        # Calculate transfer entropy
        te = self._calculate_conditional_entropy(
            p_yt_ytlag, p_yt_ytlag_xs, p_ytlag_xs
        )

        # Normalize if requested
        if self.normalize:
            target_entropy = entropy(
                np.bincount(target_current) / len(target_current)
            )
            if target_entropy > 0:
                te = te / target_entropy

        # Apply normalized scaling if requested (0-1 range)
        if self.normalized:
            # Normalize by the minimum of source and target entropies
            source_entropy = entropy(
                np.bincount(source_lagged) / len(source_lagged)
            )
            target_entropy = entropy(
                np.bincount(target_current) / len(target_current)
            )
            min_entropy = min(float(source_entropy), float(target_entropy))
            if min_entropy > 0:
                te = te / min_entropy
            te = min(1.0, float(te))  # Cap at 1.0

        return max(0.0, float(te))  # Transfer entropy should be non-negative


def cross_correlation(
    df: pl.DataFrame,
    col1: str,
    col2: str,
    max_lag: Optional[int] = None,
    method: str = "pearson",
    normalize: bool = True,
    interpolate_nans: bool = True,
    interpolation_method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cross-correlation between two columns with optional lag and normalization.

    Args:
        df: Polars DataFrame
        col1: First column name
        col2: Second column name
        max_lag: Maximum lag to compute (default: len(df) // 4)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        normalize: Whether to normalize the time series before correlation (default: True)

    Returns:
        Tuple of (lags, cross_correlations)
    """
    # Use helper function for common preprocessing
    series1, series2, normalization_stats = (
        _preprocess_series_for_relational_analysis(
            df,
            col1,
            col2,
            interpolate_nans,
            interpolation_method,
            normalize=False,
        )
    )

    if max_lag is None:
        max_lag = len(series1) // 4

    # # Normalize time series if requested
    # if normalize:
    #     # Z-score normalization: (x - mean) / std
    #     series1 = (series1 - np.mean(series1)) / np.std(series1)
    #     series2 = (series2 - np.mean(series2)) / np.std(series2)

    # Alternative: Min-max normalization to [0,1] range
    # series1 = (series1 - np.min(series1)) / (np.max(series1) - np.min(series1))
    # series2 = (series2 - np.min(series2)) / (np.max(series2) - np.min(series2))

    # Calculate cross-correlation
    if method == "pearson":
        # Use numpy correlate for efficiency
        corr = np.correlate(series1, series2, mode="full")

        # Normalize correlation values to [0, 1] range
        if normalize:
            # Proper normalization to [0, 1] range
            # First normalize to [-1, 1] using proper cross-correlation formula
            norm_factor = np.sqrt(np.sum(series1**2) * np.sum(series2**2))
            if norm_factor > 0:
                corr = corr / norm_factor

            # Then shift and scale to [0, 1] range
            corr = (corr + 1) / 2
        else:
            # For non-normalized series, apply standard normalization then shift to [0,1]
            norm_factor = np.sqrt(np.sum(series1**2) * np.sum(series2**2))
            if norm_factor > 0:
                corr = corr / norm_factor
            corr = (corr + 1) / 2

    else:
        # For other methods, use scipy if available
        corr = signal.correlate(series1, series2, mode="full")

        # Apply appropriate normalization to [0, 1] range
        if normalize:
            # Proper normalization to [0, 1] range
            norm_factor = np.sqrt(np.sum(series1**2) * np.sum(series2**2))
            if norm_factor > 0:
                corr = corr / norm_factor
            corr = (corr + 1) / 2
        else:
            # For non-normalized series, apply standard normalization then shift to [0,1]
            norm_factor = np.sqrt(np.sum(series1**2) * np.sum(series2**2))
            if norm_factor > 0:
                corr = corr / norm_factor
            corr = (corr + 1) / 2

    # Extract relevant lag range
    lags = np.arange(1, max_lag + 1)
    center_idx = len(corr) // 2
    start_idx = center_idx + 1
    end_idx = min(len(corr), center_idx + max_lag + 1)

    return lags, corr[start_idx:end_idx]


def transfer_entropy(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    bins: int = 5,
    binning_method: str = "equal",
    lag_source: int = 1,
    lag_target: int = 1,
    normalize: bool = True,
    normalized: bool = False,
    interpolate_nans: bool = True,
    interpolation_method: str = "linear",
) -> float:
    """
    Calculate transfer entropy between two columns in a Polars DataFrame.

    Transfer entropy measures the directional flow of information between two
    time series. It quantifies how much information from one series helps
    predict the future of another series, beyond what can be predicted from
    the target series' own past.

    Args:
        df: Polars DataFrame containing the time series data
        source_col: Source column name (potential cause)
        target_col: Target column name (potential effect)
        bins: Number of bins for discretization
        binning_method: Method for binning ('equal', 'quantile', 'kmeans')
        lag_source: Number of lags to consider for source series
        lag_target: Number of lags to consider for target series
        normalize: Whether to normalize the result by the target's entropy
        normalized: Whether to return normalized transfer entropy (0-1 range)
        interpolate_nans: Whether to interpolate NaN values (assumes temporal alignment)
        interpolation_method: Method for interpolation ('linear', 'forward', 'backward')

    Returns:
        Transfer entropy value (bits or normalized 0-1 range)
    """
    # Use helper function for common preprocessing (transfer entropy always normalizes)
    source_series, target_series, normalization_stats = (
        _preprocess_series_for_relational_analysis(
            df,
            source_col,
            target_col,
            interpolate_nans,
            interpolation_method,
            normalize=True,
        )
    )

    # Create estimator and calculate transfer entropy
    estimator = TransferEntropyEstimator(
        bins=bins,
        binning_method=binning_method,
        lag_source=lag_source,
        lag_target=lag_target,
        normalize=normalize,
        normalized=normalized,
    )

    return estimator.estimate(source_series, target_series)


def mutual_information(
    df: pl.DataFrame,
    col1: str,
    col2: str,
    bins: int = 5,
    binning_method: str = "equal",
    normalize: bool = True,
    interpolate_nans: bool = True,
    interpolation_method: str = "linear",
) -> float:
    """
    Calculate mutual information between two columns in a Polars DataFrame.

    Mutual information measures the amount of information shared between two
    variables, regardless of the type of relationship (linear or non-linear).

    Args:
        df: Polars DataFrame containing the data
        col1: First column name
        col2: Second column name
        bins: Number of bins for discretization
        binning_method: Method for binning ('equal', 'quantile', 'kmeans')
        normalize: Whether to normalize the series before analysis (default: True)
        interpolate_nans: Whether to interpolate NaN values (assumes temporal alignment)
        interpolation_method: Method for interpolation ('linear', 'forward', 'backward')

    Returns:
        Mutual information value (bits)
    """
    # Use helper function for common preprocessing
    series1, series2, _ = _preprocess_series_for_relational_analysis(
        df, col1, col2, interpolate_nans, interpolation_method, normalize
    )

    # Create estimator for discretization
    estimator = TransferEntropyEstimator(
        bins=bins, binning_method=binning_method
    )

    x_disc = estimator.discretize(series1)
    y_disc = estimator.discretize(series2)

    # Calculate joint and marginal probabilities
    joint_counts = np.histogram2d(x_disc, y_disc, bins=bins)[0]
    joint_probs = joint_counts / np.sum(joint_counts)

    x_probs = np.sum(joint_probs, axis=1)
    y_probs = np.sum(joint_probs, axis=0)

    # Calculate mutual information (vectorized)
    eps = 1e-10

    # Create mask for non-zero joint probabilities
    mask = joint_probs > 0

    # Calculate marginal product matrix
    marginal_product = np.outer(x_probs, y_probs)

    # Vectorized mutual information calculation
    # MI = Σ p(x,y) * log₂(p(x,y) / (p(x) * p(y)))
    mi_terms = joint_probs * np.log2(
        joint_probs / (marginal_product + eps) + eps
    )

    # Sum only over non-zero terms
    mi = float(np.sum(mi_terms[mask]))

    return max(0.0, mi)


def normalized_mutual_information(
    df: pl.DataFrame,
    col1: str,
    col2: str,
    bins: int = 5,
    binning_method: str = "equal",
) -> float:
    """
    Calculate normalized mutual information between two columns (0-1 range).

    Normalized mutual information measures the amount of information shared between
    two variables, scaled to a 0-1 range where 0 indicates independence and 1
    indicates perfect dependence.

    Args:
        df: Polars DataFrame containing the data
        col1: First column name
        col2: Second column name
        bins: Number of bins for discretization
        binning_method: Method for binning ('equal', 'quantile', 'kmeans')

    Returns:
        Normalized mutual information value (0-1 range)
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1} or {col2} not found in DataFrame")

    # Convert to numpy arrays and handle missing values
    series1 = df.get_column(col1).drop_nulls().to_numpy()
    series2 = df.get_column(col2).drop_nulls().to_numpy()

    if len(series1) != len(series2):
        # Align series to same length
        min_len = min(len(series1), len(series2))
        series1 = series1[:min_len]
        series2 = series2[:min_len]

    # Create estimator for discretization
    estimator = TransferEntropyEstimator(
        bins=bins, binning_method=binning_method
    )

    x_disc = estimator.discretize(series1)
    y_disc = estimator.discretize(series2)

    # Calculate joint and marginal probabilities
    joint_counts = np.histogram2d(x_disc, y_disc, bins=bins)[0]
    joint_probs = joint_counts / np.sum(joint_counts)

    x_probs = np.sum(joint_probs, axis=1)
    y_probs = np.sum(joint_probs, axis=0)

    # Calculate mutual information
    eps = 1e-10
    mask = joint_probs > 0
    marginal_product = np.outer(x_probs, y_probs)

    mi_terms = joint_probs * np.log2(
        joint_probs / (marginal_product + eps) + eps
    )
    mi = float(np.sum(mi_terms[mask]))

    # Calculate individual entropies
    h_x = entropy(x_probs)
    h_y = entropy(y_probs)

    # Normalize by the minimum entropy
    min_entropy = min(float(h_x), float(h_y))
    if min_entropy > 0:
        nmi = mi / min_entropy
    else:
        nmi = 0.0

    return min(1.0, max(0.0, nmi))  # Ensure 0-1 range


def granger_causality(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    max_lag: int = 5,
    normalize: bool = False,
    interpolate_nans: bool = True,
    interpolation_method: str = "linear",
) -> float:
    """
    Calculate Granger causality between two columns in a Polars DataFrame.

    Granger causality tests whether one time series helps predict another time series
    beyond what can be predicted from the target series' own past. This is a linear
    causality test that measures the improvement in prediction accuracy.

    Args:
        df: Polars DataFrame containing the time series data
        source_col: Source column name (potential cause)
        target_col: Target column name (potential effect)
        max_lag: Maximum number of lags to test
        normalize: Whether to normalize the series before analysis (default: False)
        interpolate_nans: Whether to interpolate NaN values (assumes temporal alignment)
        interpolation_method: Method for interpolation ('linear', 'forward', 'backward')

    Returns:
        Granger causality measure (0-1 range, higher values suggest stronger causality)
    """
    # Use helper function for common preprocessing
    source_series, target_series, _ = (
        _preprocess_series_for_relational_analysis(
            df,
            source_col,
            target_col,
            interpolate_nans,
            interpolation_method,
            normalize,
        )
    )

    # Calculate Granger causality
    gc_value = fast_granger_causality(
        source_series, target_series, max_lag=max_lag
    )

    # Ensure the result is in [0, 1] range and return as float
    return max(0.0, min(1.0, float(gc_value)))


def dlinear_causality(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    max_lag: int = 5,
    normalize: bool = False,
    interpolate_nans: bool = True,
    interpolation_method: str = "linear",
) -> float:
    """
    Calculate causality between two columns using DLinear model.

    This function uses the DLinear (Decomposition Linear) model to test whether one
    time series helps predict another time series. It trains two models:
    1. A restricted model that only uses the target series' own past
    2. An unrestricted model that uses both the source and target series' past

    The causality measure is based on the improvement in prediction accuracy when
    including the source series.

    Args:
        df: Polars DataFrame containing the time series data
        source_col: Source column name (potential cause)
        target_col: Target column name (potential effect)
        max_lag: Maximum number of lags to use for prediction (used as sequence length)
        normalize: Whether to normalize the series before analysis (default: False)
        interpolate_nans: Whether to interpolate NaN values (assumes temporal alignment)
        interpolation_method: Method for interpolation ('linear', 'forward', 'backward')

    Returns:
        DLinear causality measure (0-1 range, higher values suggest stronger causality)
    """
    # Use helper function for common preprocessing
    source_series, target_series, _ = (
        _preprocess_series_for_relational_analysis(
            df,
            source_col,
            target_col,
            interpolate_nans,
            interpolation_method,
            normalize,
        )
    )

    # Check for sufficient data
    min_length = max(max_lag * 2, 50)  # Need enough data for training
    if len(source_series) < min_length or len(target_series) < min_length:
        logger.debug("Not enough data for DLinear causality")
        return 0.0

    # Check for constant series
    if np.std(source_series) < 1e-10 or np.std(target_series) < 1e-10:
        logger.debug("Constant series not compatible for DLinear causality")
        return 0.0

    # Set up parameters
    seq_len = max_lag
    pred_len = 1  # Predict one step ahead
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data for DLinear
    # Create multivariate time series: [target, source]
    combined_data = np.column_stack([target_series, source_series])

    # Split data for training and testing
    train_size = int(0.7 * len(combined_data))
    train_data = combined_data[:train_size]
    test_data = combined_data[train_size:]

    if len(test_data) < seq_len + pred_len:
        return 0.0

    # Create datasets
    train_dataset = WindowedTimeSeries(train_data, seq_len, pred_len)
    test_dataset = WindowedTimeSeries(test_data, seq_len, pred_len)

    if len(train_dataset) < 10 or len(test_dataset) < 5:
        logger.debug("Not enough data for DLinear causality")
        return 0.0

    # Create custom dataset wrapper for restricted model (target series only)
    class RestrictedDataset(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            x, y = self.base_dataset[idx]
            # Only use target series (channel 0)
            return x[:, 0:1], y[:, 0:1]  # (seq_len, 1), (pred_len, 1)

    # Create datasets
    restricted_train_dataset = RestrictedDataset(train_dataset)
    restricted_test_dataset = RestrictedDataset(test_dataset)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=False
    )
    restricted_train_loader = DataLoader(
        restricted_train_dataset, batch_size=32, shuffle=True, drop_last=False
    )
    restricted_test_loader = DataLoader(
        restricted_test_dataset, batch_size=32, shuffle=False, drop_last=False
    )

    # Create models
    channels = 2  # target and source
    num_layers = 1
    hidden_size = 128
    activation = "relu"

    # Restricted model: only uses target series (channel 0)
    restricted_model = DLinear(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=1,  # Only target channel
        individual=False,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation,
    ).to(device)

    # Unrestricted model: uses both target and source series
    unrestricted_model = DLinear(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,  # Both channels
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation,
        individual=False,
    ).to(device)

    # Training parameters
    epochs = 20
    lr = 5e-3
    optimizer_restricted = Adam(
        restricted_model.parameters(), lr=lr, weight_decay=1e-4
    )
    optimizer_unrestricted = Adam(
        unrestricted_model.parameters(), lr=lr, weight_decay=1e-4
    )
    loss_fn = nn.MSELoss()

    # Train restricted model (only target series)
    for epoch in range(epochs):
        train_one_epoch(
            restricted_model,
            restricted_train_loader,
            optimizer_restricted,
            loss_fn,
            device,
        )

    # Train unrestricted model (both series) - only predict target series
    def unrestricted_loss_fn(pred, target):
        # pred: (batch, pred_len, 2), target: (batch, pred_len, 2)
        # Only use target series (channel 0) for loss calculation
        pred_target = pred[:, :, 0:1]  # (batch, pred_len, 1)
        target_only = target[:, :, 0:1]  # (batch, pred_len, 1)
        return loss_fn(pred_target, target_only)

    for epoch in range(epochs):
        train_one_epoch(
            unrestricted_model,
            train_loader,
            optimizer_unrestricted,
            unrestricted_loss_fn,
            device,
        )

    # Evaluate models on test data
    restricted_mse = evaluate_model(
        restricted_model, restricted_test_loader, device, loss_fn
    )
    unrestricted_mse = evaluate_model(
        unrestricted_model, test_loader, device, unrestricted_loss_fn
    )

    # Calculate improvement ratio
    logger.info(
        f"Restricted MSE: {restricted_mse}, Unrestricted MSE: {unrestricted_mse}"
    )
    if (
        restricted_mse <= 0
        or unrestricted_mse <= 0
        or restricted_mse <= unrestricted_mse
    ):
        return 0.0

    improvement = (restricted_mse - unrestricted_mse) / restricted_mse

    # Ensure the result is in [0, 1] range
    return max(0.0, min(1.0, float(improvement)))


def convergent_cross_mapping(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    max_lag: int = 5,  # Kept for compatibility but not used by CCM
    interpolate_nans: bool = True,
    interpolation_method: str = "linear",
    max_library_size: int = 20,
) -> float:
    """
    Calculate convergent cross-mapping between two columns in a Polars DataFrame.

    Convergent cross-mapping (CCM) is a method for detecting causality in complex
    ecosystems. It tests whether the historical record of one variable can be used
    to predict the current state of another variable, indicating causal influence.
    CCM uses time delay embedding and does not rely on lag-based analysis.

    Args:
        df: Polars DataFrame containing the time series data
        source_col: Source column name (potential cause)
        target_col: Target column name (potential effect)
        max_lag: Kept for compatibility but not used by CCM algorithm
        interpolate_nans: Whether to interpolate NaN values
        interpolation_method: Method for interpolation ('linear', 'nearest', etc.)

    Returns:
        float: Convergent cross-mapping score between 0 and 1
    """
    start_time = time.time()
    # Convert to numpy arrays
    source_series = df.get_column(source_col).to_numpy()
    target_series = df.get_column(target_col).to_numpy()

    # Ensure both series have the same length (temporal alignment)
    if len(source_series) != len(target_series):
        min_len = min(len(source_series), len(target_series))
        source_series = source_series[:min_len]
        target_series = target_series[:min_len]

    # Handle NaN values using the same logic as granger causality
    if interpolate_nans:
        # Interpolate NaN values while preserving temporal alignment
        source_series = _interpolate_nans(source_series, interpolation_method)
        target_series = _interpolate_nans(target_series, interpolation_method)
    else:
        # Remove rows where either series has NaN (original behavior)
        valid_mask = ~(np.isnan(source_series) | np.isnan(target_series))
        source_series = source_series[valid_mask]
        target_series = target_series[valid_mask]

    # Ensure we have enough data points for CCM (needs sufficient data for embedding)
    min_length = 50  # Minimum data points needed for CCM
    if len(source_series) < min_length or len(target_series) < min_length:
        return 0.0

    # Create combined time series matrix for CCM
    combined_series = np.column_stack([source_series, target_series])

    # Initialize CCM with appropriate parameters
    ccm = ConvergentCrossMapping(
        d_embed=3,  # Embedding dimension
        k=None,  # Will be set to d_embed + 1
        verbose=False,
        max_library_size=min(
            15, len(combined_series) // 2
        ),  # Limit for performance
        significance_threshold=0.05,  # Statistical significance threshold
    )

    # Fit the CCM model
    causal_matrix = ccm.fit(combined_series)

    # Extract the causal strength from source to target (0->1 direction)
    ccm_score = causal_matrix[1, 0]  # target influenced by source

    logger.info(f"Convergent cross-mapping time: {time.time() - start_time}")
    # Ensure the result is in [0, 1] range and return as float
    return max(0.0, min(1.0, float(ccm_score)))


def _relational_single(
    df: pl.DataFrame,
    timestamps: Optional[np.ndarray],
    relational_type: str,
    source_col: str,
    target_col: str,
    **kwargs,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Apply a single relational operation to a grouped subset of data.

    Args:
        df: Polars DataFrame (subset for this group)
        timestamps: Optional timestamps for this group
        relational_type: Type of relational operation ('transfer_entropy', 'mutual_information',
                         'granger_causality', 'convergent_cross_mapping', 'dlinear_causality', 'cross_correlation')
        source_col: Source column name
        target_col: Target column name
        **kwargs: Additional parameters for the relational function

    Returns:
        Dictionary containing the relational result
    """
    if source_col not in df.columns or target_col not in df.columns:
        return {
            "value": 0.0,
            "error": f"Columns {source_col} or {target_col} not found",
        }

    if relational_type == "transfer_entropy":
        result = transfer_entropy(df, source_col, target_col, **kwargs)
        return {"value": result}

    elif relational_type == "mutual_information":
        result = mutual_information(df, source_col, target_col, **kwargs)
        return {"value": result}

    elif relational_type == "granger_causality":
        result = granger_causality(df, source_col, target_col, **kwargs)
        return {"value": result}

    elif relational_type == "convergent_cross_mapping":
        result = convergent_cross_mapping(df, source_col, target_col, **kwargs)
        return {"value": result}

    elif relational_type == "dlinear_causality":
        result = dlinear_causality(df, source_col, target_col, **kwargs)
        return {"value": result}

    elif relational_type == "cross_correlation":
        lags, corr_values = cross_correlation(
            df, source_col, target_col, **kwargs
        )
        max_corr_idx = np.argmax(np.abs(corr_values))
        max_corr = corr_values[max_corr_idx]
        max_lag = lags[max_corr_idx]
        return {
            "value": max_corr,
            "max_lag": max_lag,
            "correlation_range": f"{corr_values.min():.4f} to {corr_values.max():.4f}",
        }

    else:
        return {
            "value": 0.0,
            "error": f"Unknown relational type: {relational_type}",
        }


def combined_relational(
    result_dict: List[Dict[str, Union[float, np.ndarray, str]]],
    aggregation_method: str = "mean_skip_zeros",
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Combine the results of relational operations for each group.

    Args:
        result_dict: List of relational results for each group
        aggregation_method: Method to aggregate results ('mean', 'max', 'median', 'weighted_mean')

    Returns:
        Combined relational result
    """
    if not result_dict:
        return {"value": 0.0, "error": "No results to combine"}

    # Extract values, filtering out errors
    valid_results = [r for r in result_dict if "error" not in r]

    if not valid_results:
        return {"value": 0.0, "error": "No valid results to combine"}

    # Get the main value to aggregate
    values = [r["value"] for r in valid_results]

    # Convert values to numpy array, handling mixed types
    try:
        values_array = np.array(values)
    except (ValueError, TypeError):
        # If conversion fails, handle mixed types manually
        numeric_values = []
        for v in values:
            if isinstance(v, (int, float)):
                numeric_values.append(float(v))
            elif isinstance(v, np.ndarray):
                numeric_values.append(float(np.mean(v)))
            else:
                numeric_values.append(0.0)
        values_array = np.array(numeric_values)

    skip_zero_values = values_array[values_array != 0]
    if len(skip_zero_values) == 0:
        combined_value = 0.0
    elif aggregation_method == "mean_skip_zeros":
        combined_value = np.mean(skip_zero_values)
    elif aggregation_method == "max_skip_zeros":
        combined_value = np.max(skip_zero_values)
    elif aggregation_method == "median_skip_zeros":
        combined_value = np.median(skip_zero_values)
    if aggregation_method == "mean":
        combined_value = np.mean(values_array)
    elif aggregation_method == "max":
        combined_value = np.max(values_array)
    elif aggregation_method == "median":
        combined_value = np.median(values_array)
    elif aggregation_method == "weighted_mean":
        # Weight by the number of data points in each group (if available)
        weights = []
        for r in valid_results:
            if "data_points" in r:
                data_points = r["data_points"]
                if isinstance(
                    data_points, (list, tuple, np.ndarray)
                ) and hasattr(data_points, "__len__"):
                    weights.append(len(data_points))
                else:
                    weights.append(1)
            else:
                weights.append(1)
        combined_value = np.average(values_array, weights=weights)
    else:
        combined_value = np.mean(values_array)  # Default to mean

    # Combine additional metrics if they exist
    combined_result: Dict[str, Any] = {"value": float(combined_value)}

    # Check if all results have the same additional keys
    additional_keys = set()
    for r in valid_results:
        additional_keys.update(r.keys())
    additional_keys.discard("value")  # Remove the main value key

    for key in additional_keys:
        if all(key in r for r in valid_results):
            if key == "max_lag":
                # For max_lag, take the most common value
                lags = [r[key] for r in valid_results]
                combined_result[key] = float(
                    int(np.round(np.mean(np.array(lags))))
                )
            elif key == "correlation_range":
                # For correlation range, combine the ranges
                ranges = [str(r[key]) for r in valid_results]
                combined_result[key] = "; ".join(ranges)
            else:
                # For other metrics, take the mean
                metric_values = [r[key] for r in valid_results]
                try:
                    combined_result[key] = float(
                        np.mean(np.array(metric_values))
                    )
                except (ValueError, TypeError):
                    # If conversion fails, handle mixed types
                    numeric_values = []
                    for v in metric_values:
                        if isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                        elif isinstance(v, np.ndarray):
                            numeric_values.append(float(np.mean(v)))
                        else:
                            numeric_values.append(0.0)
                    combined_result[key] = float(
                        np.mean(np.array(numeric_values))
                    )

    # Add metadata about the combination
    combined_result["_num_groups"] = float(len(valid_results))
    combined_result["_num_groups_skip_zeros"] = float(len(skip_zero_values))
    combined_result["_total_groups"] = float(len(result_dict))
    combined_result["_grouped_results"] = (
        result_dict  # Store grouped results for analysis
    )

    return combined_result


def unified_relational_analysis(
    df: pl.DataFrame,
    relational_type: str,
    source_col: str,
    target_col: str,
    timestamp_column: Optional[str] = None,
    aggregation_method: str = "mean",
    limit_cols: Optional[int] = None,
    **kwargs,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Unified function to perform relational analysis with grouping support.

    This function applies the same grouping logic as decomposition functions,
    performing the specified relational operation on each group and then
    combining the results.

    Args:
        df: Polars DataFrame containing the data
        relational_type: Type of relational operation ('transfer_entropy', 'mutual_information',
                         'granger_causality', 'convergent_cross_mapping', 'dlinear_causality', 'cross_correlation')
        source_col: Source column name
        target_col: Target column name
        timestamp_column: Optional timestamp column for grouping
        aggregation_method: Method to aggregate results across groups ('mean', 'max', 'median', 'weighted_mean')
        **kwargs: Additional parameters for the relational function

    Returns:
        Dictionary containing the combined relational result with metadata
    """
    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Columns {source_col} or {target_col} not found in DataFrame"
        )

    # Use apply_grouping to handle timestamp grouping (same as decomposition)
    grouped_results = apply_grouping(
        df,
        _relational_single,
        timestamp_column,
        limit_cols,
        relational_type,
        source_col,
        target_col,
        **kwargs,
    )

    # Combine results using the same pattern as decomposition
    combined_result = combined_relational(grouped_results, aggregation_method)

    return combined_result
