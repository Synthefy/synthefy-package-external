"""
Analysis functions for DataSummarizer.
"""

import time
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.preprocessing.data_summary_utils.autoregressions import (
    autocorrelation,
)
from synthefy_pkg.preprocessing.data_summary_utils.basic_statistics import (
    calculate_basic_statistics,
    calculate_correlations,
    detect_outliers,
    get_quantiles,
)
from synthefy_pkg.preprocessing.data_summary_utils.decompositions import (
    forecast_with_decomposition,
    fourier_decomposition,
    pysindy_decomposition,
    singular_spectrum_analysis,
    stl_decomposition,
)
from synthefy_pkg.preprocessing.data_summary_utils.relational_operations import (
    convergent_cross_mapping,
    cross_correlation,
    granger_causality,
    mutual_information,
    transfer_entropy,
    unified_relational_analysis,
)
from synthefy_pkg.preprocessing.data_summary_utils.ts_features import (
    extract_ts_features,
)


def perform_basic_statistics_analysis(data_summarizer):
    """Perform basic statistics analysis on numeric columns."""
    # Get basic statistics for all numeric columns
    basic_stats = calculate_basic_statistics(data_summarizer.df_data)

    # Convert to DataFrame for display
    stats_data = []
    for col, stats in basic_stats.items():
        stats_data.append(
            {
                "Column": col,
                "Count": stats["count"],
                "Mean": f"{stats['mean']:.4f}",
                "Std": f"{stats['std']:.4f}",
                "Min": f"{stats['min']:.4f}",
                "Max": f"{stats['max']:.4f}",
                "Median": f"{stats['median']:.4f}",
                "Skewness": f"{stats['skewness']:.4f}",
                "Kurtosis": f"{stats['kurtosis']:.4f}",
            }
        )

    data_summarizer.basic_stats_df = pd.DataFrame(stats_data)
    logger.info(f"Basic statistics calculated for {len(basic_stats)} columns")


def perform_ts_features_analysis(data_summarizer):
    """Perform time series features analysis on time series columns."""
    ts_features_data = []
    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    for i, col in enumerate(all_cols):
        ts_features = extract_ts_features(
            data_summarizer.df_data, col, data_summarizer.timestamps_col
        )
        if len(ts_features) != 0:
            ts_features_data.append(ts_features)
    data_summarizer.ts_features_df = pd.DataFrame(ts_features_data)
    logger.info(
        f"Time series features calculated for {len(ts_features_data)} columns"
    )
    logger.info(
        f"Time series features: {data_summarizer.ts_features_df.columns}"
    )
    # Clean up the DataFrame - replace applymap with map for each column, and handle nans
    for col in data_summarizer.ts_features_df.columns:
        data_summarizer.ts_features_df[col] = data_summarizer.ts_features_df[
            col
        ].map(
            lambda x: (
                x["mean"]
            )  # TODO: right now only uses the mean, not the variance
            if isinstance(x, dict)
            else x
        )


def perform_correlation_analysis(data_summarizer):
    """Perform correlation analysis on numeric columns."""
    # Calculate correlations using all numeric columns
    corr_matrix = calculate_correlations(data_summarizer.df_data)

    if corr_matrix.height > 0:
        # Convert to pandas for easier manipulation
        data_summarizer.correlation_df = corr_matrix.to_pandas()
        logger.info(
            f"Correlation matrix calculated with dimensions {corr_matrix.height}x{corr_matrix.width}"
        )
    else:
        data_summarizer.correlation_df = None
        logger.info("No correlation matrix generated (insufficient data)")


def perform_outlier_analysis(data_summarizer):
    """Perform outlier analysis on numeric columns."""
    outlier_data = []

    # Analyze outliers for each numeric column
    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    for col in all_cols:
        outliers = detect_outliers(data_summarizer.df_data, col, method="iqr")
        outlier_count = outliers["outlier_count"]
        if isinstance(outlier_count, int):
            outlier_percentage = (
                outlier_count / len(data_summarizer.df_data)
            ) * 100
            outlier_data.append(
                {
                    "Column": col,
                    "Outlier Count": outlier_count,
                    "Outlier Percentage": f"{outlier_percentage:.2f}%",
                    "Method": "IQR",
                }
            )
        else:
            outlier_data.append(
                {
                    "Column": col,
                    "Outlier Count": "Error",
                    "Outlier Percentage": "Error",
                    "Method": "IQR",
                }
            )

    data_summarizer.outlier_df = pd.DataFrame(outlier_data)
    logger.info(f"Outlier analysis completed for {len(outlier_data)} columns")


def perform_quantile_analysis(data_summarizer):
    """Perform quantile analysis on numeric columns."""
    quantile_data = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    for col in all_cols:
        quantiles = get_quantiles(data_summarizer.df_data, col)
        if quantiles:
            quantile_data.append(
                {
                    "Column": col,
                    "Q25": f"{quantiles.get('q25', 'N/A')}",
                    "Q50 (Median)": f"{quantiles.get('q50', 'N/A')}",
                    "Q75": f"{quantiles.get('q75', 'N/A')}",
                }
            )

    data_summarizer.quantile_df = pd.DataFrame(quantile_data)
    logger.info(f"Quantile analysis completed for {len(quantile_data)} columns")


def perform_autocorrelation_analysis(data_summarizer):
    """Perform autocorrelation analysis on time series columns."""
    autocorr_data = []
    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    for col in all_cols:
        if data_summarizer.timestamps_col:
            # Perform autocorrelation analysis
            lags, autocorr_values = autocorrelation(
                data_summarizer.df_data, col, max_lag=100
            )

            # Get key autocorrelation values
            autocorr_data.append(
                {
                    "Column": col,
                    "Lag 1": f"{autocorr_values[1] if len(autocorr_values) > 1 else 'N/A':.4f}",
                    "Lag 5": f"{autocorr_values[5] if len(autocorr_values) > 5 else 'N/A':.4f}",
                    "Lag 10": f"{autocorr_values[10] if len(autocorr_values) > 10 else 'N/A':.4f}",
                    "Lag 20": f"{autocorr_values[20] if len(autocorr_values) > 20 else 'N/A':.4f}",
                    "Lag 50": f"{autocorr_values[50] if len(autocorr_values) > 50 else 'N/A':.4f}",
                    "Lag 100": f"{autocorr_values[100] if len(autocorr_values) > 100 else 'N/A':.4f}",
                    "Max Lag": len(lags) - 1,
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                }
            )
        else:
            autocorr_data.append(
                {
                    "Column": col,
                    "Lag 1": "No timestamp",
                    "Lag 5": "No timestamp",
                    "Lag 10": "No timestamp",
                    "Lag 20": "No timestamp",
                    "Lag 50": "No timestamp",
                    "Lag 100": "No timestamp",
                    "Max Lag": "N/A",
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                }
            )

    data_summarizer.autocorr_df = pd.DataFrame(autocorr_data)
    logger.info(
        f"Autocorrelation analysis completed for {len(autocorr_data)} columns"
    )


def perform_decomposition_analysis(
    data_summarizer, compute_all: bool = False, execute_forecast: bool = False
):
    """Perform decomposition analysis on time series columns."""

    decomposition_data = []
    data_summarizer.decomposition_results = OrderedDict[str, Dict[str, Any]]()

    if compute_all:
        all_cols = (
            data_summarizer.timeseries_cols + data_summarizer.continuous_cols
        )
    else:
        all_cols = data_summarizer.timeseries_cols

    for col in all_cols:
        if not data_summarizer.df_data.get_column(col).dtype.is_numeric():
            continue

        col_results = {}

        # Try Fourier decomposition
        start_time = time.time()
        fourier_result = fourier_decomposition(
            data_summarizer.df_data,
            col,
            n_components=10,
            timestamp_column=data_summarizer.timestamps_col,
        )
        full_fourier_result = fourier_decomposition(
            data_summarizer.df_data,
            col,
            n_components=100,
            timestamp_column=data_summarizer.timestamps_col,
        )
        logger.info(
            f"Fourier decomposition time: {time.time() - start_time} on series of length {len(data_summarizer.df_data[col])}"
        )

        if data_summarizer.test_df is not None and execute_forecast:
            fourier_forecast_result = forecast_with_decomposition(
                df=data_summarizer.df_data,
                column=col,
                decomposition_type="fourier",
                forecast_steps=len(data_summarizer.test_df),
                decomposition_params=fourier_result,
                timestamp_column=data_summarizer.timestamps_col,
                series_to_forecast=data_summarizer.test_df[col].to_numpy(),
            )
        else:
            fourier_forecast_result = None

        if fourier_result and "frequencies" in fourier_result:
            col_results["fourier"] = fourier_result
            col_results["full_fourier"] = full_fourier_result
            if fourier_forecast_result is not None:
                assert isinstance(
                    fourier_forecast_result["normalized_forecast"],
                    np.ndarray,
                )
                assert isinstance(
                    fourier_forecast_result["normalized_original"],
                    np.ndarray,
                )
                fourier_result["forecast"] = fourier_forecast_result[
                    "normalized_forecast"
                ]
                fourier_result["test_values"] = fourier_forecast_result[
                    "normalized_original"
                ]
                assert isinstance(
                    fourier_forecast_result["_grouped_results"], list
                )
                for i in range(len(fourier_result["_grouped_results"])):
                    fourier_result["_grouped_results"][i]["forecast"] = (
                        fourier_forecast_result["_grouped_results"][i][
                            "forecast"
                        ]
                    )
                    fourier_result["_grouped_results"][i]["test_values"] = (
                        fourier_forecast_result["_grouped_results"][i][
                            "test_values"
                        ]
                    )
            else:
                fourier_result["forecast"] = np.ndarray([])
                fourier_result["test_values"] = np.ndarray([])
            residual = (
                np.mean(fourier_result["mean_residual"])
                if fourier_result.get("mean_residual", None) is not None
                else None
            )
            decomposition_data.append(
                {
                    "Column": col,
                    "Decomposition Type": "Fourier",
                    "Components": len(fourier_result["frequencies"])
                    / fourier_result["num_decompositions"],
                    "Status": "Success",
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                    "Residual": residual,
                }
            )
        else:
            decomposition_data.append(
                {
                    "Column": col,
                    "Decomposition Type": "Fourier",
                    "Components": "N/A",
                    "Status": "Failed",
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                    "Residual": -1,
                }
            )

        # Try singular spectrum analysis
        start_time = time.time()
        ssa_result = singular_spectrum_analysis(
            data_summarizer.df_data,
            col,
            n_components=7,
            timestamp_column=data_summarizer.timestamps_col,
        )
        logger.info(
            f"SSA decomposition time: {time.time() - start_time} on series of length {len(data_summarizer.df_data[col])}"
        )
        if data_summarizer.test_df is not None and execute_forecast:
            ssa_forecast_result = forecast_with_decomposition(
                df=data_summarizer.df_data,
                column=col,
                decomposition_type="ssa",
                forecast_steps=len(data_summarizer.test_df),
                decomposition_params=ssa_result,
                timestamp_column=data_summarizer.timestamps_col,
                series_to_forecast=data_summarizer.test_df[col].to_numpy(),
            )
        else:
            ssa_forecast_result = None
        if ssa_result and "reconstructed" in ssa_result:
            col_results["ssa"] = ssa_result
            if ssa_forecast_result is not None:
                normalized_forecast = ssa_forecast_result.get(
                    "normalized_forecast"
                )
                normalized_original = ssa_forecast_result.get(
                    "normalized_original"
                )
                if (
                    normalized_forecast is not None
                    and normalized_original is not None
                ):
                    assert isinstance(normalized_forecast, np.ndarray)
                    assert isinstance(normalized_original, np.ndarray)
                    ssa_result["forecast"] = normalized_forecast
                    ssa_result["test_values"] = normalized_original
            else:
                ssa_result["forecast"] = np.ndarray([])
                ssa_result["test_values"] = np.ndarray([])

            residual = (
                np.mean(ssa_result["mean_residual"])
                if ssa_result.get("mean_residual", None) is not None
                else None
            )
            decomposition_data.append(
                {
                    "Column": col,
                    "Decomposition Type": "SSA",
                    "Components": len(ssa_result.get("eigenvalues", []))
                    / ssa_result["num_decompositions"],
                    "Status": "Success",
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                    "Residual": residual,
                }
            )
        else:
            decomposition_data.append(
                {
                    "Column": col,
                    "Decomposition Type": "SSA",
                    "Components": "N/A",
                    "Status": "Failed",
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                    "Residual": -1,
                }
            )

        # Try SINDy decomposition if time column exists
        if data_summarizer.timestamps_col:
            start_time = time.time()
            sindy_result = pysindy_decomposition(
                data_summarizer.df_data,
                col,
                data_summarizer.timestamps_col,
                poly_order=20,
            )
            if data_summarizer.test_df is not None and execute_forecast:
                sindy_forecast_result = forecast_with_decomposition(
                    df=data_summarizer.df_data,
                    column=col,
                    decomposition_type="sindy",
                    forecast_steps=len(data_summarizer.test_df),
                    decomposition_params=sindy_result,
                    timestamp_column=data_summarizer.timestamps_col,
                    series_to_forecast=data_summarizer.test_df[col].to_numpy(),
                )
            else:
                sindy_forecast_result = None

            if sindy_result and "reconstructed" in sindy_result:
                col_results["sindy"] = sindy_result
                if sindy_forecast_result is not None:
                    assert isinstance(
                        sindy_forecast_result["normalized_forecast"],
                        np.ndarray,
                    )
                    assert isinstance(
                        sindy_forecast_result["normalized_original"],
                        np.ndarray,
                    )
                    sindy_result["forecast"] = sindy_forecast_result[
                        "normalized_forecast"
                    ]
                    sindy_result["test_values"] = sindy_forecast_result[
                        "normalized_original"
                    ]
                else:
                    sindy_result["forecast"] = np.ndarray([])
                    sindy_result["test_values"] = np.ndarray([])
                residual = (
                    np.mean(sindy_result["mean_residual"])
                    if sindy_result.get("mean_residual", None) is not None
                    else None
                )
                decomposition_data.append(
                    {
                        "Column": col,
                        "Decomposition Type": "SINDy",
                        "Components": len(sindy_result.get("coefficients", []))
                        / sindy_result["num_decompositions"],
                        "Status": "Success",
                        "Residual": residual,
                        "Type": "Time Series"
                        if col in data_summarizer.timeseries_cols
                        else "Continuous",
                    }
                )
            else:
                decomposition_data.append(
                    {
                        "Column": col,
                        "Decomposition Type": "SINDy",
                        "Components": "N/A",
                        "Status": "Failed",
                        "Residual": -1,
                        "Type": "Time Series"
                        if col in data_summarizer.timeseries_cols
                        else "Continuous",
                    }
                )

        # Try STL decomposition
        start_time = time.time()
        stl_result = stl_decomposition(
            data_summarizer.df_data,
            col,
            timestamp_column=data_summarizer.timestamps_col,
        )
        logger.info(
            f"STL decomposition time: {time.time() - start_time} on series of length {len(data_summarizer.df_data[col])}, {col}"
        )
        if data_summarizer.test_df is not None and execute_forecast:
            # TODO: forecasting with decomposition is not viable for multiple batches
            stl_forecast_result = forecast_with_decomposition(
                df=data_summarizer.df_data,
                column=col,
                decomposition_type="stl",
                forecast_steps=len(data_summarizer.test_df),
                decomposition_params=stl_result,
                timestamp_column=data_summarizer.timestamps_col,
                series_to_forecast=data_summarizer.test_df[col].to_numpy(),
            )
        else:
            stl_forecast_result = None
        if stl_result and "reconstructed" in stl_result:
            col_results["stl"] = stl_result
            if stl_forecast_result is not None:
                assert isinstance(
                    stl_forecast_result["normalized_forecast"], np.ndarray
                )
                assert isinstance(
                    stl_forecast_result["normalized_original"], np.ndarray
                )
                stl_result["forecast"] = stl_forecast_result[
                    "normalized_forecast"
                ]
                stl_result["test_values"] = stl_forecast_result[
                    "normalized_original"
                ]
            else:
                stl_result["forecast"] = np.ndarray([])
                stl_result["test_values"] = np.ndarray([])
            residual = (
                np.mean(stl_result["mean_residual"])
                if stl_result.get("mean_residual", None) is not None
                else None
            )
            decomposition_data.append(
                {
                    "Column": col,
                    "Decomposition Type": "STL",
                    "Components": 3,  # trend, seasonal, residual
                    "Status": "Success",
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                    "Residual": residual,
                }
            )
        else:
            decomposition_data.append(
                {
                    "Column": col,
                    "Decomposition Type": "STL",
                    "Components": "N/A",
                    "Status": "Failed",
                    "Type": "Time Series"
                    if col in data_summarizer.timeseries_cols
                    else "Continuous",
                    "Residual": -1,
                }
            )

        # Store results for this column
        if col_results:
            data_summarizer.decomposition_results[col] = col_results

    data_summarizer.decomposition_df = pd.DataFrame(decomposition_data)
    logger.info(
        f"Decomposition analysis completed for {len(decomposition_data)} plots"
    )


def perform_cross_correlation_analysis(
    data_summarizer, compute_all: bool = False
):
    """Perform cross-correlation analysis between time series columns."""
    cross_corr_data = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols

    target_cols = data_summarizer.timeseries_cols

    if len(all_cols) >= 2 and data_summarizer.timestamps_col:
        # Analyze cross-correlations between pairs of time series
        for i, col1 in enumerate(all_cols):
            if compute_all:
                used_target_col = all_cols[i + 1 :]
            else:
                used_target_col = [col for col in target_cols if col != col1]

            for col2 in used_target_col:
                corr_result = unified_relational_analysis(
                    data_summarizer.df_data,
                    relational_type="cross_correlation",
                    source_col=col1,
                    target_col=col2,
                    timestamp_column=data_summarizer.timestamps_col,
                    aggregation_method="mean",
                    max_lag=10,
                    normalize=True,
                    interpolate_nans=True,
                    interpolation_method="linear",
                )
                max_corr = corr_result["value"]
                max_lag = corr_result.get("max_lag", 0)
                correlation_range = corr_result.get("correlation_range", "N/A")

                # Store additional information about the correlation analysis
                all_values = corr_result.get("all_values", [])
                all_corr_values = corr_result.get("all_corr_values", [])
                all_lags = corr_result.get("all_lags", [])
                num_groups = corr_result.get("_num_groups", 0)

                cross_corr_data.append(
                    {
                        "Series 1": col1,
                        "Series 2": col2,
                        "Max Correlation": f"{max_corr:.4f}",
                        "Max Correlation Lag": max_lag,
                        "Correlation Range": correlation_range,
                        "All Values": all_values,  # Store all individual max correlations
                        "All Correlation Arrays": all_corr_values,  # Store all correlation arrays
                        "All Lag Arrays": all_lags,  # Store all lag arrays
                        "Number of Groups": num_groups,
                        "Aggregation Method": corr_result.get(
                            "aggregation_method", "mean"
                        ),
                        "_num_groups": corr_result.get("_num_groups", 0),
                        "_num_groups_skip_zeros": corr_result.get(
                            "_num_groups_skip_zeros", 0
                        ),
                        "_total_groups": corr_result.get("_total_groups", 0),
                        "_grouped_results": corr_result.get(
                            "_grouped_results", []
                        ),
                    }
                )

    if cross_corr_data:
        data_summarizer.cross_corr_df = pd.DataFrame(cross_corr_data)
        logger.info(
            f"Cross-correlation analysis completed for {len(cross_corr_data)} pairs"
        )
    else:
        data_summarizer.cross_corr_df = None
        logger.info(
            "No cross-correlation analysis performed (insufficient time series data)"
        )


def perform_transfer_entropy_analysis(
    data_summarizer, compute_all: bool = False
):
    """Perform transfer entropy analysis between time series columns."""
    transfer_entropy_data = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols

    target_cols = data_summarizer.timeseries_cols

    if len(all_cols) >= 2 and data_summarizer.timestamps_col:
        # Analyze transfer entropy between pairs of time series
        for i, col1 in enumerate(all_cols):
            if compute_all:
                used_target_col = all_cols[i + 1 :]
            else:
                used_target_col = [col for col in target_cols if col != col1]

            for col2 in used_target_col:
                # Test both directions: col1 -> col2 and col2 -> col1
                for source_col, target_col in [(col1, col2), (col2, col1)]:
                    # Test multiple lags to find optimal lag
                    max_te = 0.0
                    optimal_lag = 1

                    # Store all transfer entropy results for different lags
                    all_te_results = []
                    for lag in [1, 5, 10, 20, 50]:  # Test lags 1, 2, 3
                        te_result = unified_relational_analysis(
                            data_summarizer.df_data,
                            relational_type="transfer_entropy",
                            source_col=source_col,
                            target_col=target_col,
                            timestamp_column=data_summarizer.timestamps_col,
                            aggregation_method="mean",
                            bins=50,
                            binning_method="equal",
                            lag_source=lag,
                            lag_target=1,
                            normalize=True,
                        )
                        te = te_result["value"]
                        all_te_results.append(
                            {"lag": lag, "value": te, "result": te_result}
                        )

                        # Only compare if te is numeric
                        if (
                            isinstance(te, (int, float, np.number))
                            and te > max_te
                        ):
                            max_te = te
                            optimal_lag = lag

                    # Add result if we got a valid transfer entropy
                    if max_te > 0.0 or optimal_lag > 0:
                        transfer_entropy_data.append(
                            {
                                "Source": source_col,
                                "Target": target_col,
                                "Transfer Entropy": f"{max_te:.6f}",
                                "Optimal Lag": optimal_lag,
                                "Max Transfer Entropy": f"{max_te:.6f}",
                                "Normalized": True,
                                "Bins": 50,
                                "Binning Method": "equal",
                                "All TE Results": all_te_results,  # Store all lag results
                                "All Values": [
                                    r["value"] for r in all_te_results
                                ],  # Store all values
                                "_num_groups": te_result.get("_num_groups", 0),
                                "_num_groups_skip_zeros": te_result.get(
                                    "_num_groups_skip_zeros", 0
                                ),
                                "_total_groups": te_result.get(
                                    "_total_groups", 0
                                ),
                                "_grouped_results": te_result.get(
                                    "_grouped_results", []
                                ),
                            }
                        )

    if transfer_entropy_data:
        data_summarizer.transfer_entropy_df = pd.DataFrame(
            transfer_entropy_data
        )
        logger.info(
            f"Transfer entropy analysis completed for {len(transfer_entropy_data)} pairs"
        )
    else:
        data_summarizer.transfer_entropy_df = None
        logger.info(
            "No transfer entropy analysis performed (insufficient time series data)"
        )


def perform_mutual_information_analysis(
    data_summarizer, compute_all: bool = False
):
    """Perform mutual information analysis between time series columns."""
    mutual_info_data = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols

    target_cols = data_summarizer.timeseries_cols

    if len(all_cols) >= 2:
        # Analyze mutual information between pairs of time series
        for i, col1 in enumerate(all_cols):
            if compute_all:
                used_target_col = all_cols[i + 1 :]
            else:
                used_target_col = [col for col in target_cols if col != col1]

            for col2 in used_target_col:
                mi_result = unified_relational_analysis(
                    data_summarizer.df_data,
                    relational_type="mutual_information",
                    source_col=col1,
                    target_col=col2,
                    timestamp_column=data_summarizer.timestamps_col,
                    aggregation_method="mean",
                    bins=20,
                    binning_method="equal",
                    normalize=True,
                    interpolate_nans=True,
                    interpolation_method="linear",
                )
                mi = mi_result["value"]

                # Store additional information about the mutual information analysis
                all_values = mi_result.get("all_values", [])
                num_groups = mi_result.get("_num_groups", 0)

                mutual_info_data.append(
                    {
                        "Column 1": col1,
                        "Column 2": col2,
                        "Mutual Information": f"{mi:.6f}",
                        "Bins": 20,
                        "Binning Method": "equal",
                        "All Values": all_values,  # Store all individual MI values
                        "Number of Groups": num_groups,
                        "Aggregation Method": mi_result.get(
                            "aggregation_method", "mean"
                        ),
                        "_num_groups": mi_result.get("_num_groups", 0),
                        "_num_groups_skip_zeros": mi_result.get(
                            "_num_groups_skip_zeros", 0
                        ),
                        "_total_groups": mi_result.get("_total_groups", 0),
                        "_grouped_results": mi_result.get(
                            "_grouped_results", []
                        ),
                    }
                )

    if mutual_info_data:
        data_summarizer.mutual_info_df = pd.DataFrame(mutual_info_data)
        logger.info(
            f"Mutual information analysis completed for {len(mutual_info_data)} pairs"
        )
    else:
        data_summarizer.mutual_info_df = None
        logger.info(
            "No mutual information analysis performed (insufficient data)"
        )


def perform_granger_causality_analysis(
    data_summarizer, compute_all: bool = False
):
    """Perform Granger causality analysis between time series columns."""
    granger_causality_data = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols

    target_cols = data_summarizer.timeseries_cols

    if len(all_cols) >= 2 and data_summarizer.timestamps_col:
        # Analyze Granger causality between pairs of time series
        for i, col1 in enumerate(all_cols):
            if compute_all:
                used_target_col = all_cols[i + 1 :]
            else:
                used_target_col = [col for col in target_cols if col != col1]

            for col2 in used_target_col:
                # Test both directions: col1 -> col2 and col2 -> col1
                for source_col, target_col in [(col1, col2), (col2, col1)]:
                    # Test multiple lags to find optimal lag
                    max_gc = 0.0
                    optimal_lag = 1

                    # Store all Granger causality results for different lags
                    all_gc_results = []
                    for lag in [50]:  # Test different lags
                        gc_result = unified_relational_analysis(
                            data_summarizer.df_data,
                            relational_type="granger_causality",
                            source_col=source_col,
                            target_col=target_col,
                            timestamp_column=data_summarizer.timestamps_col,
                            aggregation_method="mean",
                            max_lag=lag,
                            normalize=True,
                            interpolate_nans=True,
                            interpolation_method="linear",
                        )
                        gc = gc_result["value"]
                        all_gc_results.append(
                            {"lag": lag, "value": gc, "result": gc_result}
                        )

                        # Only compare if gc is numeric
                        if (
                            isinstance(gc, (int, float, np.number))
                            and gc > max_gc
                        ):
                            max_gc = gc
                            optimal_lag = lag

                    # Add result if we got a valid Granger causality
                    if max_gc > 0.0 or optimal_lag > 0:
                        granger_causality_data.append(
                            {
                                "Source": source_col,
                                "Target": target_col,
                                "Granger Causality": f"{max_gc:.6f}",
                                "Optimal Lag": optimal_lag,
                                "Max Granger Causality": f"{max_gc:.6f}",
                                "Interpolate NaNs": True,
                                "Interpolation Method": "linear",
                                "All GC Results": all_gc_results,  # Store all lag results
                                "All Values": [
                                    r["value"] for r in all_gc_results
                                ],  # Store all values
                                "_num_groups": gc_result.get("_num_groups", 0),
                                "_num_groups_skip_zeros": gc_result.get(
                                    "_num_groups_skip_zeros", 0
                                ),
                                "_total_groups": gc_result.get(
                                    "_total_groups", 0
                                ),
                                "_grouped_results": gc_result.get(
                                    "_grouped_results", []
                                ),
                            }
                        )

    if granger_causality_data:
        data_summarizer.granger_causality_df = pd.DataFrame(
            granger_causality_data
        )
        logger.info(
            f"Granger causality analysis completed for {len(granger_causality_data)} pairs"
        )
    else:
        data_summarizer.granger_causality_df = None
        logger.info(
            "No Granger causality analysis performed (insufficient data)"
        )


def perform_dlinear_analysis(data_summarizer, compute_all: bool = False):
    """Perform DLinear causality analysis between time series columns."""
    dlinear_data = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols

    target_cols = data_summarizer.timeseries_cols

    if len(all_cols) >= 2 and data_summarizer.timestamps_col:
        # Analyze DLinear causality between pairs of time series
        for i, col1 in enumerate(all_cols):
            if compute_all:
                used_target_col = all_cols[i + 1 :]
            else:
                used_target_col = [col for col in target_cols if col != col1]

            for col2 in used_target_col:
                # Test both directions: col1 -> col2 and col2 -> col1
                for source_col, target_col in [(col1, col2), (col2, col1)]:
                    # Test multiple lags to find optimal lag
                    max_dlinear = 0.0
                    optimal_lag = 1

                    # Store all DLinear results for different lags
                    all_dlinear_results = []
                    for lag in [20]:  # Test different lags
                        dlinear_result = unified_relational_analysis(
                            data_summarizer.df_data,
                            relational_type="dlinear_causality",
                            source_col=source_col,
                            target_col=target_col,
                            timestamp_column=data_summarizer.timestamps_col,
                            limit_cols=20,
                            aggregation_method="mean",
                            max_lag=lag,
                            normalize=True,
                            interpolate_nans=True,
                            interpolation_method="linear",
                        )
                        dlinear = dlinear_result["value"]
                        all_dlinear_results.append(
                            {
                                "lag": lag,
                                "value": dlinear,
                                "result": dlinear_result,
                            }
                        )

                        # Only compare if dlinear is numeric
                        if (
                            isinstance(dlinear, (int, float, np.number))
                            and dlinear > max_dlinear
                        ):
                            max_dlinear = dlinear
                            optimal_lag = lag

                    # Add result if we got a valid DLinear causality
                    if max_dlinear > 0.0 or optimal_lag > 0:
                        dlinear_data.append(
                            {
                                "Source": source_col,
                                "Target": target_col,
                                "DLinear Causality": f"{max_dlinear:.6f}",
                                "Optimal Lag": optimal_lag,
                                "Max DLinear Causality": f"{max_dlinear:.6f}",
                                "Interpolate NaNs": True,
                                "Interpolation Method": "linear",
                                "All DLinear Results": all_dlinear_results,  # Store all lag results
                                "All Values": [
                                    r["value"] for r in all_dlinear_results
                                ],  # Store all values
                                "_num_groups": dlinear_result.get(
                                    "_num_groups", 0
                                ),
                                "_num_groups_skip_zeros": dlinear_result.get(
                                    "_num_groups_skip_zeros", 0
                                ),
                                "_total_groups": dlinear_result.get(
                                    "_total_groups", 0
                                ),
                                "_grouped_results": dlinear_result.get(
                                    "_grouped_results", []
                                ),
                            }
                        )

    if dlinear_data:
        data_summarizer.dlinear_df = pd.DataFrame(dlinear_data)
        logger.info(
            f"DLinear causality analysis completed for {len(dlinear_data)} pairs"
        )
    else:
        data_summarizer.dlinear_df = None
        logger.info(
            "No DLinear causality analysis performed (insufficient data)"
        )


def perform_convergent_cross_mapping_analysis(
    data_summarizer, compute_all: bool = False
):
    """Perform convergent cross-mapping analysis between time series columns."""
    convergent_cross_mapping_data = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols

    target_cols = data_summarizer.timeseries_cols

    if len(all_cols) >= 2 and data_summarizer.timestamps_col:
        # Analyze convergent cross-mapping between pairs of time series
        for i, col1 in enumerate(all_cols):
            if compute_all:
                used_target_col = all_cols[i + 1 :]
            else:
                used_target_col = [col for col in target_cols if col != col1]

            for col2 in used_target_col:
                # Test both directions: col1 -> col2 and col2 -> col1
                for source_col, target_col in [(col1, col2), (col2, col1)]:
                    ccm_result = unified_relational_analysis(
                        data_summarizer.df_data,
                        relational_type="convergent_cross_mapping",
                        source_col=source_col,
                        target_col=target_col,
                        limit_cols=20,
                        timestamp_column=data_summarizer.timestamps_col,
                        aggregation_method="mean",
                        max_lag=5,  # Parameter not used by CCM but kept for compatibility
                        interpolate_nans=True,
                        interpolation_method="linear",
                    )
                    ccm = ccm_result["value"]

                    # Store additional information about the CCM analysis
                    all_values = ccm_result.get("all_values", [])
                    num_groups = ccm_result.get("_num_groups", 0)

                    # Only include pairs with significant CCM
                    convergent_cross_mapping_data.append(
                        {
                            "Source": source_col,
                            "Target": target_col,
                            "Convergent Cross-Mapping": ccm,
                            "All Values": all_values,  # Store all individual CCM values
                            "Number of Groups": num_groups,
                            "Aggregation Method": ccm_result.get(
                                "aggregation_method", "mean"
                            ),
                            "_num_groups": ccm_result.get("_num_groups", 0),
                            "_num_groups_skip_zeros": ccm_result.get(
                                "_num_groups_skip_zeros", 0
                            ),
                            "_total_groups": ccm_result.get("_total_groups", 0),
                            "_grouped_results": ccm_result.get(
                                "_grouped_results", []
                            ),
                        }
                    )

    if convergent_cross_mapping_data:
        data_summarizer.convergent_cross_mapping_df = pd.DataFrame(
            convergent_cross_mapping_data
        )
        logger.info(
            f"Convergent cross-mapping analysis completed for {len(convergent_cross_mapping_data)} pairs"
        )
    else:
        data_summarizer.convergent_cross_mapping_df = None
        logger.info(
            "No convergent cross-mapping analysis performed (insufficient data)"
        )
