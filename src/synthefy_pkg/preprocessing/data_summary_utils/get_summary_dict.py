"""
Summary dictionary generation utilities for DataSummarizer.
"""

from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger


def get_summary_dict(data_summarizer) -> dict:
    """
    Returns a JSON-serializable dictionary containing all summary data.

    Args:
        data_summarizer: DataSummarizer instance with analysis results

    Returns:
        Dictionary containing all summary data
    """
    # Ensure comprehensive analysis has been performed
    data_summarizer._ensure_analysis_complete()

    return {
        "time_series_summary": (
            data_summarizer.time_series_df.to_dict(orient="records")
            if data_summarizer.time_series_df is not None
            else None
        ),
        "time_range": (
            data_summarizer.time_range.to_dict(orient="records")
            if data_summarizer.time_range is not None
            else None
        ),
        "metadata_summary": (
            data_summarizer.metadata_df.to_dict(orient="records")
            if data_summarizer.metadata_df is not None
            else None
        ),
        "columns_by_type": data_summarizer.num_columns_by_type,
        "sample_counts": (
            data_summarizer.sample_counts.to_dict(orient="records")
            if isinstance(data_summarizer.sample_counts, pd.DataFrame)
            else data_summarizer.sample_counts
        ),
        "basic_statistics": (
            data_summarizer.basic_stats_df.to_dict(orient="records")
            if hasattr(data_summarizer, "basic_stats_df")
            and data_summarizer.basic_stats_df is not None
            else None
        ),
        "ts_features": (
            data_summarizer.ts_features_df.to_dict(orient="records")
            if hasattr(data_summarizer, "ts_features_df")
            and data_summarizer.ts_features_df is not None
            else None
        ),
        "correlation_matrix": (
            data_summarizer.correlation_df.to_dict(orient="records")
            if hasattr(data_summarizer, "correlation_df")
            and data_summarizer.correlation_df is not None
            else None
        ),
        "outlier_analysis": (
            data_summarizer.outlier_df.to_dict(orient="records")
            if hasattr(data_summarizer, "outlier_df")
            and data_summarizer.outlier_df is not None
            else None
        ),
        "quantile_analysis": (
            data_summarizer.quantile_df.to_dict(orient="records")
            if hasattr(data_summarizer, "quantile_df")
            and data_summarizer.quantile_df is not None
            else None
        ),
        "autocorrelation_analysis": (
            data_summarizer.autocorr_df.to_dict(orient="records")
            if hasattr(data_summarizer, "autocorr_df")
            and data_summarizer.autocorr_df is not None
            else None
        ),
        "decomposition_analysis": (
            data_summarizer.decomposition_df.to_dict(orient="records")
            if hasattr(data_summarizer, "decomposition_df")
            and data_summarizer.decomposition_df is not None
            else None
        ),
        "cross_correlation_analysis": (
            data_summarizer.cross_corr_df.to_dict(orient="records")
            if hasattr(data_summarizer, "cross_corr_df")
            and data_summarizer.cross_corr_df is not None
            else None
        ),
        "granger_causality_analysis": (
            data_summarizer.granger_causality_df.to_dict(orient="records")
            if hasattr(data_summarizer, "granger_causality_df")
            and data_summarizer.granger_causality_df is not None
            else None
        ),
        "dlinear_analysis": (
            data_summarizer.dlinear_df.to_dict(orient="records")
            if hasattr(data_summarizer, "dlinear_df")
            and data_summarizer.dlinear_df is not None
            else None
        ),
        "convergent_cross_mapping_analysis": (
            data_summarizer.convergent_cross_mapping_df.to_dict(
                orient="records"
            )
            if hasattr(data_summarizer, "convergent_cross_mapping_df")
            and data_summarizer.convergent_cross_mapping_df is not None
            else None
        ),
        "mutual_information_analysis": (
            data_summarizer.mutual_info_df.to_dict(orient="records")
            if hasattr(data_summarizer, "mutual_info_df")
            and data_summarizer.mutual_info_df is not None
            else None
        ),
        "transfer_entropy_analysis": (
            data_summarizer.transfer_entropy_df.to_dict(orient="records")
            if hasattr(data_summarizer, "transfer_entropy_df")
            and data_summarizer.transfer_entropy_df is not None
            else None
        ),
        "decomposition_plots": (
            len(data_summarizer.decomposition_plots)
            if hasattr(data_summarizer, "decomposition_plots")
            and data_summarizer.decomposition_plots
            else 0
        ),
        "cross_correlation_matrix_plot": (
            "Generated"
            if hasattr(data_summarizer, "cross_corr_matrix_plot")
            and data_summarizer.cross_corr_matrix_plot
            else "Not Generated"
        ),
        "autocorrelation_plots": (
            len(data_summarizer.autocorr_plots)
            if hasattr(data_summarizer, "autocorr_plots")
            and data_summarizer.autocorr_plots
            else 0
        ),
        "report_path": data_summarizer.settings.json_save_path,  # type: ignore
    }
