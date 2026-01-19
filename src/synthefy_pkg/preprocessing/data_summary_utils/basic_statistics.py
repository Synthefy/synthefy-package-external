"""Basic statistical operations for data analysis."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import scipy.stats


def calculate_basic_statistics(
    df: pl.DataFrame, columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate basic statistics for numeric columns.

    Args:
        df: Polars DataFrame
        columns: List of columns to analyze (default: all numeric columns)

    Returns:
        Dictionary with statistics for each column
    """
    if columns is None:
        # Get all numeric columns
        columns = [col for col in df.columns if df.schema[col].is_numeric()]

    # Return empty dict if no columns specified
    if not columns:
        return {}

    stats = {}
    for col in columns:
        if col not in df.columns:
            continue

        col_data = df.get_column(col).drop_nulls().drop_nans()
        if col_data.len() == 0:
            continue

        values = col_data.to_numpy()

        if not np.issubdtype(values.dtype, np.number):
            continue

        stats[col] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "skewness": float(_calculate_skewness(values)),
            "kurtosis": float(_calculate_kurtosis(values)),
        }

    return stats


def get_quantiles(
    df: pl.DataFrame, column: str, quantiles: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate quantiles for a column.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        quantiles: List of quantiles to compute (default: [0.25, 0.5, 0.75])

    Returns:
        Dictionary with quantile values
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]

    col_data = df.get_column(column).drop_nulls()
    if col_data.len() == 0:
        return {}

    values = col_data.to_numpy()
    result = {}

    for q in quantiles:
        result[f"q{int(q * 100)}"] = float(np.percentile(values, q * 100))

    return result


def detect_outliers(
    df: pl.DataFrame, column: str, method: str = "iqr", threshold: float = 1.5
) -> Dict[str, Union[List[int], List[float], int]]:
    """
    Detect outliers in a column using various methods.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection

    Returns:
        Dictionary with outlier information
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    col_data = df.get_column(column).drop_nulls()
    if col_data.len() == 0:
        return {"outlier_indices": [], "outlier_values": [], "outlier_count": 0}

    values = col_data.to_numpy()
    indices = np.arange(len(values))

    if method == "iqr":
        outlier_mask = _detect_outliers_iqr(values, threshold)
    elif method == "zscore":
        outlier_mask = _detect_outliers_zscore(values, threshold)
    elif method == "isolation_forest":
        outlier_mask = _detect_outliers_isolation_forest(values)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    outlier_indices = indices[outlier_mask].tolist()
    outlier_values = values[outlier_mask].tolist()

    return {
        "outlier_indices": outlier_indices,
        "outlier_values": outlier_values,
        "outlier_count": len(outlier_indices),
    }


def get_data_length(df: pl.DataFrame) -> Dict[str, int]:
    """
    Get basic information about data length and dimensions.

    Args:
        df: Polars DataFrame

    Returns:
        Dictionary with length information
    """
    return {
        "rows": df.height,
        "columns": df.width,
        "total_cells": df.height * df.width,
        "memory_usage_bytes": int(df.estimated_size()),
    }


def calculate_correlations(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
) -> pl.DataFrame:
    """
    Calculate correlation matrix for numeric columns.

    Args:
        df: Polars DataFrame
        columns: List of columns to analyze (default: all numeric columns)
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Polars DataFrame with correlation matrix
    """
    if columns is None:
        columns = [col for col in df.columns if df.schema[col].is_numeric()]

    if len(columns) < 2:
        return pl.DataFrame()

    # Filter to only numeric columns and drop nulls
    numeric_df = df.select(columns).drop_nulls()

    if numeric_df.height == 0:
        return pl.DataFrame()

    # Convert to numpy for correlation calculation
    values = numeric_df.to_numpy()

    if method == "pearson":
        corr_matrix = np.corrcoef(values.T)
    elif method == "spearman":
        # Ensure we get a matrix, handle edge cases
        corr_matrix, p_values = scipy.stats.spearmanr(values, axis=0)
        if values.shape[1] == 2:
            n_corr_matrix = np.eye(2)
            n_corr_matrix[0, 1] = corr_matrix
            n_corr_matrix[1, 0] = corr_matrix
            corr_matrix = n_corr_matrix
        # Ensure corr_matrix is always 2D with correct shape
        corr_matrix = np.asarray(corr_matrix)
        if corr_matrix.ndim == 0:
            # Single value, create 1x1 matrix
            corr_matrix = np.array([[corr_matrix]])
        elif corr_matrix.ndim == 1:
            # 1D array, reshape to square matrix
            n_cols = values.shape[1]
            corr_matrix = corr_matrix.reshape(n_cols, n_cols)
    elif method == "kendall":
        n_cols = values.shape[1]
        corr_matrix = np.eye(n_cols)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                corr, _ = scipy.stats.kendalltau(values[:, i], values[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Create correlation DataFrame
    corr_df = pl.DataFrame(corr_matrix, schema=columns).with_row_index("column")

    return corr_df.with_columns(
        pl.col("column").map_elements(lambda x: columns[x])
    )


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


def _detect_outliers_iqr(values: np.ndarray, threshold: float) -> np.ndarray:
    """Detect outliers using IQR method."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    return (values < lower_bound) | (values > upper_bound)


def _detect_outliers_zscore(values: np.ndarray, threshold: float) -> np.ndarray:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((values - np.mean(values)) / np.std(values))
    return z_scores > threshold


def _detect_outliers_isolation_forest(values: np.ndarray) -> np.ndarray:
    """Detect outliers using Isolation Forest."""
    try:
        from sklearn.ensemble import IsolationForest

        # Reshape for sklearn
        X = values.reshape(-1, 1)
        iso_forest = IsolationForest(random_state=42, contamination="auto")
        predictions = iso_forest.fit_predict(X)

        # -1 indicates outliers
        return predictions == -1

    except ImportError:
        # Fallback to IQR method if sklearn not available
        return _detect_outliers_iqr(values, 1.5)
