"""
Metadata summarization utilities for DataSummarizer.
"""

import datetime
from typing import List, Optional

import pandas as pd
import polars as pl
from loguru import logger


def summarize_metadata(data_summarizer) -> None:
    """
    Summarize metadata including number of columns, types, ranges, and units.

    Args:
        data_summarizer: DataSummarizer instance to populate with metadata
    """
    metadata = []

    # Process continuous columns
    for col in data_summarizer.continuous_cols:
        # if the column has all null values, we skip it
        if data_summarizer.df_data.get_column(col).null_count() < len(
            data_summarizer.df_data
        ):
            col_data = data_summarizer.df_data.get_column(col)
            col_range = f"{col_data.min()} to {col_data.max()}"
            outliers_pct = _calculate_outliers(
                col_data, data_summarizer.OUTLIER_STD_THRESHOLD
            )
            metadata.append(
                {
                    "Column Name": col,
                    "Type": "Continuous",
                    "Range": col_range,
                    f"% Outliers (>{data_summarizer.OUTLIER_STD_THRESHOLD} SD)": f"{outliers_pct:.2f}%",
                }
            )
        else:
            logger.info(f"Skipping column {col} due to all null values")
            metadata.append(
                {
                    "Column Name": col,
                    "Type": "Continuous",
                    "Range": "N/A",
                    f"% Outliers (>{data_summarizer.OUTLIER_STD_THRESHOLD} SD)": "N/A",
                }
            )

    # Process discrete columns
    for col in data_summarizer.discrete_cols:
        col_data = data_summarizer.df_data.get_column(col)

        # Skip columns with list data types as they don't support n_unique()
        if col_data.dtype == pl.List:
            logger.warning(
                f"Skipping column '{col}' with list data type (unsupported for metadata analysis)"
            )
            metadata.append(
                {
                    "Column Name": col,
                    "Type": "Discrete (List)",
                    "Range": "List data type - analysis skipped",
                    f"% Outliers (>{data_summarizer.OUTLIER_STD_THRESHOLD} SD)": "N/A",
                }
            )
            continue

        value_counts = (
            col_data.drop_nulls().value_counts().sort("count", descending=True)
        )
        top_values = value_counts.head(data_summarizer.TOP_VALUES_COUNT)
        sample_vals = top_values[col].to_list()
        n_unique = col_data.n_unique()
        col_range = (
            f"{n_unique} unique values: {', '.join(map(str, sample_vals))}"
            + ("..." if n_unique > data_summarizer.TOP_VALUES_COUNT else "")
        )

        metadata.append(
            {
                "Column Name": col,
                "Type": "Discrete",
                "Range": col_range,
                f"% Outliers (>{data_summarizer.OUTLIER_STD_THRESHOLD} SD)": "N/A",
            }
        )

    metadata_df = pl.DataFrame(metadata)
    if "Type" in metadata_df.columns and metadata_df.height > 0:
        metadata_df = metadata_df.sort("Type")

    metadata_df = metadata_df.to_pandas().reset_index(drop=True)

    # Ensure all required fields are present
    metadata_df["Type"] = metadata_df.get("Type", "Unknown")
    metadata_df["Range"] = metadata_df.get("Range", "N/A")

    # Calculate sample counts
    sample_counts = pl.DataFrame(
        {"Metric": ["Total Samples"], "Counts": [len(data_summarizer.df_data)]}
    )

    # Add source-specific sample counts if multiple sources
    if len(data_summarizer.data_sources) > 1:
        source_counts = (
            data_summarizer.df_data.group_by("_source_id")
            .agg(pl.count().alias("count"))
            .sort("_source_id")
        )
        for row in source_counts.iter_rows(named=True):
            sample_counts = pl.concat(
                [
                    sample_counts,
                    pl.DataFrame(
                        {
                            "Metric": [f"Samples from {row['_source_id']}"],
                            "Counts": [row["count"]],
                        }
                    ),
                ],
                how="vertical",
            )

    # Calculate samples per group if we have group columns
    if data_summarizer.group_cols:
        for col in data_summarizer.group_cols:
            total_group_samples = (
                data_summarizer.df_data.group_by(col)
                .agg(pl.count())
                .select(pl.col("count").sum())
                .item()
            )
            sample_counts = pl.concat(
                [
                    sample_counts,
                    pl.DataFrame(
                        {
                            "Metric": [f"Total Samples - {col}"],
                            "Counts": [total_group_samples],
                        }
                    ),
                ],
                how="vertical",
            )
    else:
        logger.info(
            "No group columns provided, skipping group sample calculation."
        )

    sample_counts = sample_counts.to_pandas()

    # Ensure all required fields are present
    sample_counts["Metric"] = sample_counts.get("Metric", "Unknown")
    sample_counts["Counts"] = sample_counts.get("Counts", 0)

    data_summarizer.metadata_df = metadata_df
    data_summarizer.num_columns_by_type = (
        metadata_df["Type"].value_counts().to_dict()
    )
    data_summarizer.sample_counts = sample_counts


def summarize_time_series(data_summarizer) -> None:
    """
    Summarize time series information for each KPI.

    Args:
        data_summarizer: DataSummarizer instance to populate with time series data
    """
    time_series = []
    # Calculate time range if we have a timestamp column
    if data_summarizer.timestamps_col:
        _calculate_time_intervals(data_summarizer)
    # Only process time series if we have timeseries columns defined
    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    for col_name in all_cols:
        if data_summarizer.df_data.get_column(col_name).null_count() < len(
            data_summarizer.df_data
        ):
            col_data = data_summarizer.df_data.get_column(col_name)
            col_range = f"{col_data.min()} to {col_data.max()}"
            missing_pct = (col_data.null_count() / len(col_data)) * 100
            outliers_pct = _calculate_outliers(
                col_data, data_summarizer.OUTLIER_STD_THRESHOLD
            )

            time_series.append(
                {
                    "Column Name": col_name,
                    "Range": col_range,
                    "Missing Percentage": f"{missing_pct:.2f}%",
                    f"% Outliers (>{data_summarizer.OUTLIER_STD_THRESHOLD} SD)": f"{outliers_pct:.2f}%",
                }
            )
        else:
            logger.info(f"Skipping column {col_name} due to all null values")
            time_series.append(
                {
                    "Column Name": col_name,
                    "Range": "N/A",
                    "Missing Percentage": "100%",
                    f"% Outliers (>{data_summarizer.OUTLIER_STD_THRESHOLD} SD)": "N/A",
                }
            )
    time_series_df = pd.DataFrame(time_series)

    # Store results as instance variables
    data_summarizer.time_series_df = time_series_df

    # Initialize visualization attributes
    data_summarizer.decomposition_plots = []
    data_summarizer.cross_corr_matrix_plot = None
    data_summarizer.transfer_entropy_matrix_plot = None
    data_summarizer.mutual_info_matrix_plot = None
    data_summarizer.autocorr_plots = []

    # Initialize analysis DataFrames
    data_summarizer.basic_stats_df = None
    data_summarizer.correlation_df = None
    data_summarizer.outlier_df = None
    data_summarizer.quantile_df = None
    data_summarizer.autocorr_df = None
    data_summarizer.decomposition_df = None
    data_summarizer.cross_corr_df = None
    data_summarizer.transfer_entropy_df = None
    data_summarizer.mutual_info_df = None


def _calculate_outliers(
    col_data: pl.Series, outlier_std_threshold: float
) -> float:
    """
    Calculate the fraction of outliers in a column using the +/- 3 standard deviations method.

    Args:
        col_data: Polars Series containing the column data
        outlier_std_threshold: Number of standard deviations to use as threshold

    Returns:
        Percentage of outliers in the column
    """
    if not col_data.dtype.is_numeric():
        logger.info(
            f"Skipping outlier calculation for non-numeric column: {col_data.name}"
        )
        return 0.0

    mean = col_data.mean()
    std_dev = col_data.std()

    if mean is None or std_dev is None:
        # Should never reach this...
        return 0.0

    lower_bound = mean - outlier_std_threshold * std_dev  # type: ignore
    upper_bound = mean + outlier_std_threshold * std_dev  # type: ignore

    # Count outliers using Polars boolean operations
    outliers_count = col_data.filter(
        (col_data < lower_bound) | (col_data > upper_bound)
    ).len()
    total_count = col_data.len()

    return (outliers_count / total_count * 100) if total_count > 0 else 0.0


def _calculate_time_intervals(data_summarizer) -> None:
    """Calculate the minimum, maximum timestamps and the most common time interval between timestamps."""
    if not data_summarizer.timestamps_col:
        logger.warning("No timestamp column defined")
        return

    try:
        # Ensure timestamp column is in datetime format
        if (
            data_summarizer.df_data[data_summarizer.timestamps_col].dtype
            == pl.Utf8
        ):
            logger.info(
                f"Converting {data_summarizer.timestamps_col} from string to datetime"
            )
            data_summarizer.df_data = data_summarizer.df_data.with_columns(
                [
                    pl.col(data_summarizer.timestamps_col)
                    .str.strptime(pl.Datetime, format=None)
                    .alias(data_summarizer.timestamps_col)
                ]
            )

        # Calculate intervals
        sorted_timestamps = data_summarizer.df_data.select(
            pl.col(data_summarizer.timestamps_col)
        ).sort(data_summarizer.timestamps_col)
        intervals = sorted_timestamps.with_columns(
            pl.col(data_summarizer.timestamps_col).diff().alias("interval")
        )
        min_time = sorted_timestamps[data_summarizer.timestamps_col].min()
        max_time = sorted_timestamps[data_summarizer.timestamps_col].max()
        min_interval = intervals["interval"].drop_nulls().min()
        max_interval = intervals["interval"].drop_nulls().max()

        # Calculate the most common interval
        interval_counts = intervals["interval"].drop_nulls().value_counts()
        most_common_interval = interval_counts.sort(
            "count", descending=True
        ).row(0)[0]

        # Convert datetime values to human-readable strings
        min_time_str = (
            min_time.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(min_time, datetime.datetime)
            else str(min_time)
        )
        max_time_str = (
            max_time.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(max_time, datetime.datetime)
            else str(max_time)
        )

        # Format time intervals to be human-readable
        def format_timedelta(td):
            if td is None:
                return "N/A"

            seconds = td.total_seconds()

            # Handle zero timedelta case
            if seconds == 0:
                return "0 seconds"

            days, remainder = divmod(seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)

            parts = []
            if days > 0:
                parts.append(f"{int(days)} days")
            if hours > 0:
                parts.append(f"{int(hours)} hours")
            if minutes > 0:
                parts.append(f"{int(minutes)} minutes")
            if seconds > 0 or not parts:
                parts.append(f"{seconds:.2f} seconds")

            return ", ".join(parts)

        min_interval_str = format_timedelta(min_interval)
        max_interval_str = format_timedelta(max_interval)
        most_common_interval_str = format_timedelta(most_common_interval)

        data_summarizer.time_range = pl.DataFrame(
            {
                "Min Time": [min_time_str],
                "Max Time": [max_time_str],
                "Min Interval": [min_interval_str],
                "Max Interval": [max_interval_str],
                "Most Common Interval": [most_common_interval_str],
            }
        ).to_pandas()

    except Exception as e:
        logger.warning(f"Could not calculate time intervals: {str(e)}")
        # Fallback to basic time range without intervals
        try:
            min_time = data_summarizer.df_data[
                data_summarizer.timestamps_col
            ].min()
            max_time = data_summarizer.df_data[
                data_summarizer.timestamps_col
            ].max()

            # Convert datetime values to human-readable strings
            min_time_str = (
                min_time.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(min_time, datetime.datetime)
                else str(min_time)
            )
            max_time_str = (
                max_time.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(max_time, datetime.datetime)
                else str(max_time)
            )

            data_summarizer.time_range = pl.DataFrame(
                {
                    "Min Time": [min_time_str],
                    "Max Time": [max_time_str],
                    "Min Interval": ["Could not calculate"],
                    "Max Interval": ["Could not calculate"],
                    "Most Common Interval": ["Could not calculate"],
                }
            ).to_pandas()
        except Exception:
            data_summarizer.time_range = pl.DataFrame(
                {
                    "Min Time": ["Could not calculate"],
                    "Max Time": ["Could not calculate"],
                    "Min Interval": ["Could not calculate"],
                    "Max Interval": ["Could not calculate"],
                    "Most Common Interval": ["Could not calculate"],
                }
            ).to_pandas()
