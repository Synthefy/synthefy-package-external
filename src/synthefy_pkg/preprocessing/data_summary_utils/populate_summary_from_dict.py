from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from synthefy_pkg.preprocessing.data_summarizer import DataSummarizer

# Import plot creation functions
from synthefy_pkg.preprocessing.data_summary_utils.create_plots import (
    create_autocorrelation_plots,
    create_convergent_cross_mapping_matrix_plot,
    create_cross_correlation_matrix_plot,
    create_decomposition_plots,
    create_dlinear_matrix_plot,
    create_granger_causality_matrix_plot,
    create_mutual_information_matrix_plot,
    create_transfer_entropy_matrix_plot,
)


def populate_data_summarizer_from_dict(
    summarizer: DataSummarizer,
    summary_dict: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> DataSummarizer:
    """
    Create a DataSummarizer instance from a summary dictionary.

    This function reconstructs a DataSummarizer by loading dataframes from the dictionary
    and regenerating plots for HTML generation.

    Args:
        summary_dict: Dictionary containing summary data from get_summary_dict()
        output_dir: Optional output directory for plots and HTML generation

    Returns:
        DataSummarizer instance with loaded data and regenerated plots
    """
    logger.info("Creating DataSummarizer from summary dictionary...")

    # Create a minimal DataSummarizer instance

    # Load metadata if available
    if "metadata" in summary_dict:
        metadata = summary_dict["metadata"]
        summarizer.dataset_name = metadata.get("dataset_name", "unknown")
        summarizer.timeseries_cols = metadata.get("timeseries_cols", [])
        summarizer.timestamps_col = metadata.get("timestamps_col", None)
        summarizer.group_cols = metadata.get("group_cols", [])
        summarizer.continuous_cols = metadata.get("continuous_cols", [])
        summarizer.discrete_cols = metadata.get("discrete_cols", [])
        summarizer.num_columns_by_type = metadata.get("num_columns_by_type", {})

    # Load dataframes from dictionary
    dataframe_mappings = {
        "time_series_summary": "time_series_df",
        "time_range": "time_range",
        "metadata_summary": "metadata_df",
        "sample_counts": "sample_counts",
        "basic_statistics": "basic_stats_df",
        "ts_features": "ts_features_df",
        "correlation_matrix": "correlation_df",
        "outlier_analysis": "outlier_df",
        "quantile_analysis": "quantile_df",
        "autocorrelation_analysis": "autocorr_df",
        "decomposition_analysis": "decomposition_df",
        "cross_correlation_analysis": "cross_corr_df",
        "granger_causality_analysis": "granger_causality_df",
        "dlinear_analysis": "dlinear_df",
        "convergent_cross_mapping_analysis": "convergent_cross_mapping_df",
        "mutual_information_analysis": "mutual_info_df",
        "transfer_entropy_analysis": "transfer_entropy_df",
    }

    # Convert dictionary records back to DataFrames
    for dict_key, attr_name in dataframe_mappings.items():
        if dict_key in summary_dict and summary_dict[dict_key] is not None:
            df = pd.DataFrame(summary_dict[dict_key])
            setattr(summarizer, attr_name, df)
            logger.debug(f"Loaded {attr_name} with {len(df)} rows")
        else:
            setattr(summarizer, attr_name, None)

    # Set plot-related attributes
    summarizer.decomposition_plots = []
    summarizer.autocorr_plots = []
    summarizer.cross_corr_matrix_plot = None

    # Regenerate plots if data is available
    logger.info("Regenerating plots from loaded data...")

    # Regenerate decomposition plots
    if (
        hasattr(summarizer, "decomposition_df")
        and summarizer.decomposition_df is not None
    ):
        create_decomposition_plots(summarizer)
        logger.info(
            f"Regenerated {len(summarizer.decomposition_plots)} decomposition plots"
        )

    # Regenerate autocorrelation plots
    if (
        hasattr(summarizer, "autocorr_df")
        and summarizer.autocorr_df is not None
    ):
        create_autocorrelation_plots(summarizer)
        logger.info(
            f"Regenerated {len(summarizer.autocorr_plots)} autocorrelation plots"
        )

    # Regenerate cross-correlation matrix plot
    if (
        hasattr(summarizer, "cross_corr_df")
        and summarizer.cross_corr_df is not None
    ):
        create_cross_correlation_matrix_plot(summarizer)
        logger.info("Regenerated cross-correlation matrix plot")

    # Regenerate transfer entropy matrix plot
    if hasattr(summarizer, "transfer_entropy_df"):
        create_transfer_entropy_matrix_plot(summarizer)
        logger.info("Regenerated transfer entropy matrix plot")

    # Regenerate mutual information matrix plot
    if hasattr(summarizer, "mutual_info_df"):
        create_mutual_information_matrix_plot(summarizer)
        logger.info("Regenerated mutual information matrix plot")

    # Regenerate granger causality matrix plot
    if hasattr(summarizer, "granger_causality_df"):
        create_granger_causality_matrix_plot(summarizer)
        logger.info("Regenerated granger causality matrix plot")

    # Regenerate dlinear matrix plot
    if hasattr(summarizer, "dlinear_df"):
        create_dlinear_matrix_plot(summarizer)
        logger.info("Regenerated dlinear matrix plot")

    # Regenerate convergent cross mapping matrix plot
    if hasattr(summarizer, "convergent_cross_mapping_df"):
        create_convergent_cross_mapping_matrix_plot(summarizer)
        logger.info("Regenerated convergent cross mapping matrix plot")

    logger.info("Successfully created DataSummarizer from summary dictionary")
    return summarizer
