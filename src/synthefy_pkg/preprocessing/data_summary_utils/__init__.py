"""Data summary utilities for preprocessing operations."""

from synthefy_pkg.preprocessing.data_summary_utils.autoregressions import (
    arima_analysis,
    autocorrelation,
    garch_analysis,
)
from synthefy_pkg.preprocessing.data_summary_utils.basic_statistics import (
    calculate_basic_statistics,
    calculate_correlations,
    detect_outliers,
    get_data_length,
    get_quantiles,
)
from synthefy_pkg.preprocessing.data_summary_utils.decompositions import (
    fourier_decomposition,
    sindy_decomposition,
    singular_spectrum_analysis,
)
from synthefy_pkg.preprocessing.data_summary_utils.relational_operations import (
    TransferEntropyEstimator,
    cross_correlation,
    mutual_information,
    transfer_entropy,
)
from synthefy_pkg.preprocessing.data_summary_utils.ts_features import (
    extract_all_ts_features,
    extract_seasonality_features,
    extract_statistical_features,
    extract_trend_features,
    extract_ts_features,
    extract_volatility_features,
)

__all__ = [
    # Basic statistics
    "calculate_basic_statistics",
    "calculate_correlations",
    "detect_outliers",
    "get_data_length",
    "get_quantiles",
    # Time series features
    "extract_ts_features",
    "extract_statistical_features",
    "extract_trend_features",
    "extract_seasonality_features",
    "extract_volatility_features",
    "extract_all_ts_features",
    # Decompositions
    "fourier_decomposition",
    "singular_spectrum_analysis",
    "sindy_decomposition",
    # Relational operations
    "cross_correlation",
    "transfer_entropy",
    "mutual_information",
    "TransferEntropyEstimator",
    # Autoregressions
    "autocorrelation",
    "arima_analysis",
    "garch_analysis",
]
