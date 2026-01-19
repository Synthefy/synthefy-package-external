"""Bar plot analysis for fm-evals results.

This module provides mid-level analysis functions that bridge the gap between
high-level aggregated metrics and qualitative individual sample plots.

dataframes are created from the dataset_result_format.py file. expected to be in the format of:

Each batch of NC correlates will translate to one DataFrame. The DataFrames include:
- Timestamps, history values, target values, and forecast values
- Model information (model_name from forecast samples)
- Individual sample metrics (MAE, MAPE) for each correlate
- Sample metadata (sample_id, column_name, etc.)
- Split information (history vs target)
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from synthefy_pkg.fm_evals.formats.dataset_result_format import (
    DatasetResultFormat,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)


def create_grouped_bar_plots(  # grouping by qualitative data
    dataframes: List[pd.DataFrame],
    group_by: str,
    metric: str = "mae",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    show_counts: bool = False,
    **kwargs,
):
    """
    Create grouped bar plots showing model performance across different groups.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames from DatasetResultFormat.to_dfs()
    group_by : str
        Column name to group by (e.g., 'column_name', 'sample_id', etc.)
    metric : str, default "mae"
        Metric to plot ('mae' or 'mape')
    output_path : Optional[str], default None
        Path to save the plot. If None, plot is not saved.
    figsize : Tuple[int, int], default (12, 8)
        Figure size (width, height)
    title : Optional[str], default None
        Plot title. If None, auto-generated.
    show_counts : bool, default False
        Whether to show count of samples in each group
    **kwargs
        Additional arguments passed to seaborn.barplot

    Returns
    -------
    Figure
        The matplotlib figure object

    Examples
    --------
    # Group by target column (note: for some datasets (like traffic) this is synonymous with sample_id)
    >>> create_grouped_bar_plots(dfs, group_by='column_name', metric='mae')
    # Group by sample ID
    >>> create_grouped_bar_plots(dfs, group_by='sample_id', metric='mape')
    # Group by model
    >>> create_grouped_bar_plots(dfs, group_by='model_name', metric='mape')
    """
    if not dataframes:
        raise ValueError("dataframes list cannot be empty")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract metric columns and metadata
    metric_data = []

    for _, row in combined_df.iterrows():
        # Find all metric columns for this row
        metric_cols = [
            col for col in combined_df.columns if col.endswith(f"_{metric}")
        ]

        for metric_col in metric_cols:
            if pd.notna(row[metric_col]):
                # Extract the correlate name from the metric column
                correlate_name = metric_col.replace(f"_{metric}", "")

                # Get the corresponding group value
                group_col = f"{correlate_name}_{group_by}"
                if group_col in combined_df.columns:
                    group_value = row[group_col]
                else:
                    # If no specific group column, try to get from the main group_by column
                    group_value = row.get(group_by, None)

                # Skip if group value is None, NaN, or 'unknown'
                if (
                    pd.isna(group_value)
                    or group_value == "unknown"
                    or group_value is None
                ):
                    continue

                # Get model name
                model_col = f"{correlate_name}_model_name"
                model_name = row.get(model_col, "unknown")

                metric_data.append(
                    {
                        "group": group_value,
                        "model": model_name,
                        "metric_value": row[metric_col],
                        "correlate": correlate_name,
                    }
                )

    if not metric_data:
        raise ValueError(f"No {metric} data found in the provided dataframes")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(metric_data)

    # Group by the specified column and calculate mean metrics
    grouped_data = (
        plot_df.groupby(["group", "model"])["metric_value"].mean().reset_index()
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, plt.Axes), "Expected single Axes object"

    # Create grouped bar plot
    sns.barplot(
        data=grouped_data,
        x="group",
        y="metric_value",
        hue="model",
        ax=ax,
        **kwargs,
    )

    # Customize the plot
    if title is None:
        title = f"{metric.upper()} by {group_by}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(group_by.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(f"{metric.upper()}", fontsize=12)

    # Rotate x-axis labels if there are many groups
    if len(grouped_data["group"].unique()) > 10:
        plt.xticks(rotation=45, ha="right")

    # Add count annotations if requested
    if show_counts:
        for i, group in enumerate(grouped_data["group"].unique()):
            group_data = grouped_data[grouped_data["group"] == group]
            total_count = len(plot_df[plot_df["group"] == group])

            # Add count text above the bars
            for j, (_, row) in enumerate(group_data.iterrows()):
                ax.text(
                    i,
                    row["metric_value"] + 0.01 * ax.get_ylim()[1],
                    f"n={total_count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def analyze_model_performance_by_group(
    dataframes: List[pd.DataFrame],
    group_by: str,
    metric: str = "mae",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
) -> Dict:
    """
    Analyze which model performs best in each group and count frequency.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames from DatasetResultFormat.to_dfs()
    group_by : str
        Column name to group by (e.g., 'column_name', 'sample_id', etc.)
    metric : str, default "mae"
        Metric to analyze ('mae' or 'mape')
    output_path : Optional[str], default None
        Path to save the win count visualization. If None, plot is not saved.
    figsize : Tuple[int, int], default (12, 8)
        Figure size (width, height)
    title : Optional[str], default None
        Plot title. If None, auto-generated.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'best_model_by_group': Dict mapping group values to best model names
        - 'model_win_counts': Dict mapping model names to number of groups they win
        - 'group_metrics': DataFrame with mean metrics for each group-model combination
        - 'summary_stats': Dict with overall statistics
        - 'win_count_figure': Figure object showing win count visualization
    """
    if not dataframes:
        raise ValueError("dataframes list cannot be empty")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract metric columns and metadata
    metric_data = []

    for _, row in combined_df.iterrows():
        # Find all metric columns for this row
        metric_cols = [
            col for col in combined_df.columns if col.endswith(f"_{metric}")
        ]

        for metric_col in metric_cols:
            if pd.notna(row[metric_col]):
                # Extract the correlate name from the metric column
                correlate_name = metric_col.replace(f"_{metric}", "")

                # Get the corresponding group value
                group_col = f"{correlate_name}_{group_by}"
                if group_col in combined_df.columns:
                    group_value = row[group_col]
                else:
                    # If no specific group column, try to get from the main group_by column
                    group_value = row.get(group_by, None)

                # Skip if group value is None, NaN, or 'unknown'
                if (
                    pd.isna(group_value)
                    or group_value == "unknown"
                    or group_value is None
                ):
                    continue

                # Get model name
                model_col = f"{correlate_name}_model_name"
                model_name = row.get(model_col, "unknown")

                metric_data.append(
                    {
                        "group": group_value,
                        "model": model_name,
                        "metric_value": row[metric_col],
                        "correlate": correlate_name,
                    }
                )

    if not metric_data:
        raise ValueError(f"No {metric} data found in the provided dataframes")

    # Create DataFrame for analysis
    plot_df = pd.DataFrame(metric_data)

    # Calculate mean metrics for each group-model combination
    group_metrics = (
        plot_df.groupby(["group", "model"])["metric_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    group_metrics.columns = ["group", "model", "mean", "std", "count"]

    # Find best model for each group
    best_model_by_group = {}
    for group in group_metrics["group"].unique():
        group_data = group_metrics[group_metrics["group"] == group]
        best_model = group_data.loc[group_data["mean"].idxmin(), "model"]
        best_model_by_group[group] = best_model

    # Count how many groups each model wins
    model_win_counts = {}
    for model in group_metrics["model"].unique():
        win_count = sum(
            1
            for best_model in best_model_by_group.values()
            if best_model == model
        )
        model_win_counts[model] = win_count

    # Calculate summary statistics
    total_groups = len(best_model_by_group)
    summary_stats = {
        "total_groups": total_groups,
        "total_models": len(group_metrics["model"].unique()),
        "metric_analyzed": metric,
        "grouping_column": group_by,
        "win_percentages": {
            model: (count / total_groups) * 100
            for model, count in model_win_counts.items()
        },
    }

    # Create win count visualization
    win_count_figure = None
    if model_win_counts:
        fig, ax = plt.subplots(figsize=figsize)
        assert isinstance(ax, plt.Axes), "Expected single Axes object"

        # Create bar plot of win counts
        models = list(model_win_counts.keys())
        counts = list(model_win_counts.values())

        bars = ax.bar(models, counts, color="skyblue", alpha=0.7)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(counts),
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Customize the plot
        if title is None:
            title = f"Model Win Counts by {group_by} ({metric.upper()})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Number of Groups Won", fontsize=12)
        ax.set_ylim(0, max(counts) * 1.1)  # Add 10% padding

        # Rotate x-axis labels if needed
        if len(models) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        win_count_figure = fig

    return {
        "best_model_by_group": best_model_by_group,
        "model_win_counts": model_win_counts,
        "group_metrics": group_metrics,
        "summary_stats": summary_stats,
        "win_count_figure": win_count_figure,
    }


def print_model_performance_summary(
    dataframes: List[pd.DataFrame], group_by: str, metric: str = "mae"
) -> None:
    """
    Print a summary of model performance analysis.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames from DatasetResultFormat.to_dfs()
    group_by : str
        Column name to group by
    metric : str, default "mae"
        Metric to analyze ('mae' or 'mape')
    """
    analysis = analyze_model_performance_by_group(dataframes, group_by, metric)

    print(
        f"\n=== Model Performance Summary ({metric.upper()} by {group_by}) ==="
    )
    print(f"Total groups analyzed: {analysis['summary_stats']['total_groups']}")
    print(f"Total models: {analysis['summary_stats']['total_models']}")

    print("\nModel Win Counts:")
    for model, count in analysis["model_win_counts"].items():
        percentage = analysis["summary_stats"]["win_percentages"][model]
        print(f"  {model}: {count} groups ({percentage:.1f}%)")

    print("\nBest Model by Group:")
    for group, model in analysis["best_model_by_group"].items():
        print(f"  {group}: {model}")

    # Show overall statistics
    group_metrics = analysis["group_metrics"]
    print("\nOverall Statistics:")
    for model in group_metrics["model"].unique():
        model_data = group_metrics[group_metrics["model"] == model]
        mean_metric = model_data["mean"].mean()
        std_metric = model_data["mean"].std()
        print(f"  {model}: {mean_metric:.4f} ± {std_metric:.4f}")


def create_metric_grouped_bar_plots(  # quantitative data grouping
    dataframes: List[pd.DataFrame],
    metric: str = "mae",
    bins: int = 10,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    show_counts: bool = False,
    **kwargs,
):
    """
    Create bar plots showing frequency of samples in different metric ranges.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames from DatasetResultFormat.to_dfs()
    metric : str, default "mae"
        Metric to analyze ('mae' or 'mape')
    bins : int, default 10
        Number of bins to create for metric ranges
    output_path : Optional[str], default None
        Path to save the plot. If None, plot is not saved.
    figsize : Tuple[int, int], default (12, 8)
        Figure size (width, height)
    title : Optional[str], default None
        Plot title. If None, auto-generated.
    show_counts : bool, default False
        Whether to show count of samples in each bin
    **kwargs
        Additional arguments passed to seaborn.barplot

    Returns
    -------
    Figure
        The matplotlib figure object

    Examples
    --------
    >>> # Analyze MAE distribution across models
    >>> create_metric_grouped_bar_plots(dfs, metric='mae', bins=10)

    >>> # Analyze MAPE distribution with custom bins
    >>> create_metric_grouped_bar_plots(dfs, metric='mape', bins=15)
    """
    if not dataframes:
        raise ValueError("dataframes list cannot be empty")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract metric columns
    metric_data = []

    for _, row in combined_df.iterrows():
        # Find all metric columns for this row
        metric_cols = [
            col for col in combined_df.columns if col.endswith(f"_{metric}")
        ]

        for metric_col in metric_cols:
            if pd.notna(row[metric_col]):
                # Extract the correlate name from the metric column
                correlate_name = metric_col.replace(f"_{metric}", "")

                # Get model name
                model_col = f"{correlate_name}_model_name"
                model_name = row.get(model_col, "unknown")

                # Skip if model name is 'unknown'
                if model_name == "unknown":
                    continue

                metric_data.append(
                    {
                        "model": model_name,
                        "metric_value": row[metric_col],
                        "correlate": correlate_name,
                    }
                )

    if not metric_data:
        raise ValueError(f"No {metric} data found in the provided dataframes")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(metric_data)

    # Get unique samples by correlate to avoid counting individual data points
    sample_metrics = (
        plot_df.groupby(["correlate", "model"])["metric_value"]
        .first()
        .reset_index()
    )

    # Create bins for the metric values
    sample_metrics["metric_bin"] = pd.cut(
        sample_metrics["metric_value"], bins=bins, include_lowest=True
    )

    # Count samples in each bin for each model
    bin_counts = (
        sample_metrics.groupby(["metric_bin", "model"])
        .size()
        .reset_index(name="count")
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, plt.Axes), "Expected single Axes object"

    # Create grouped bar plot
    sns.barplot(
        data=bin_counts, x="metric_bin", y="count", hue="model", ax=ax, **kwargs
    )

    # Customize the plot
    if title is None:
        title = f"{metric.upper()} Distribution by Model (Sample Count)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{metric.upper()} Range", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add count annotations if requested
    if show_counts:
        for i, bin_label in enumerate(bin_counts["metric_bin"].unique()):
            bin_data = bin_counts[bin_counts["metric_bin"] == bin_label]
            total_count = bin_data["count"].sum()

            # Add count text above the bars
            for j, (_, row) in enumerate(bin_data.iterrows()):
                ax.text(
                    i,
                    row["count"] + 0.01 * ax.get_ylim()[1],
                    f"n={total_count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_error_distribution_plot(
    dataframes: List[pd.DataFrame],
    metric: str = "mae",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    **kwargs,
):
    """
    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames from DatasetResultFormat.to_dfs()
    metric : str, default "mae"
        Metric to analyze ('mae' or 'mape')
    output_path : Optional[str], default None
        Path to save the plot. If None, plot is not saved.
    figsize : Tuple[int, int], default (15, 10)
        Figure size (width, height)
    title : Optional[str], default None
        Plot title. If None, auto-generated.
    **kwargs
        Additional arguments passed to seaborn plotting functions

    Returns
    -------
    Figure
        The matplotlib figure object

    Examples
    --------
    >>> # Analyze MAE error distribution
    >>> create_error_distribution_plot(dfs, metric='mae')

    >>> # Analyze MAPE error distribution
    >>> create_error_distribution_plot(dfs, metric='mape')
    """
    if not dataframes:
        raise ValueError("dataframes list cannot be empty")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract metric columns
    metric_data = []

    for _, row in combined_df.iterrows():
        # Find all metric columns for this row
        metric_cols = [
            col for col in combined_df.columns if col.endswith(f"_{metric}")
        ]

        for metric_col in metric_cols:
            if pd.notna(row[metric_col]):
                # Extract the correlate name from the metric column
                correlate_name = metric_col.replace(f"_{metric}", "")

                # Get model name
                model_col = f"{correlate_name}_model_name"
                model_name = row.get(model_col, "unknown")

                # Skip if model name is 'unknown'
                if model_name == "unknown":
                    continue

                # Get column name for additional analysis
                col_name_col = f"{correlate_name}_column_name"
                column_name = row.get(col_name_col, correlate_name)

                metric_data.append(
                    {
                        "model": model_name,
                        "column_name": column_name,
                        "correlate": correlate_name,
                        "metric_value": row[metric_col],
                    }
                )

    if not metric_data:
        raise ValueError(f"No {metric} data found in the provided dataframes")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(metric_data)

    # Create the plot with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize)

    # Plot 1: Histogram of error distribution by model (counting samples)
    # First, get unique samples by correlate to avoid counting individual data points
    sample_metrics = (
        plot_df.groupby(["correlate", "model"])["metric_value"]
        .first()
        .reset_index()
    )

    sns.histplot(
        data=sample_metrics,
        x="metric_value",
        hue="model",
        bins=30,
        alpha=0.7,
        ax=ax1,
        **kwargs,
    )
    ax1.set_title(
        f"{metric.upper()} Distribution by Model (Sample Count)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel(f"{metric.upper()}", fontsize=10)
    ax1.set_ylabel("Number of Samples", fontsize=10)

    # Plot 2: Box plot showing error statistics by model (using sample metrics)
    sns.boxplot(data=sample_metrics, x="model", y="metric_value", ax=ax2)
    ax2.set_title(
        f"{metric.upper()} Statistics by Model (Sample Level)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Model", fontsize=10)
    ax2.set_ylabel(f"{metric.upper()}", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_aggregated_plots(
    dataframes: List[pd.DataFrame],
    group_by: Union[str, List[str]],
    metric: str = "mae",
    aggregation: Optional[str] = None,
    plot_type: str = "bar",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    show_counts: bool = False,
    additional_filters: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames from DatasetResultFormat.to_dfs()
    group_by : Union[str, List[str]]
        Column name(s) to group by. Can be:
        - Single string: 'column_name', 'sample_id', 'model_name', etc.
        - List of strings: ['column_name', 'model_name'] for multi-level grouping
    metric : str, default "mae"
        Metric to plot ('mae' or 'mape')
    aggregation : str, default "mean"
        Aggregation method: 'mean', 'sum', 'median', 'std', 'count', 'min', 'max'
    plot_type : str, default "bar"
        Type of plot: 'bar', 'line', 'box', 'violin', 'histogram'
    output_path : Optional[str], default None
        Path to save the plot. If None, plot is not saved.
    figsize : Tuple[int, int], default (12, 8)
        Figure size (width, height)
    title : Optional[str], default None
        Plot title. If None, auto-generated with metric and aggregation info.
    show_counts : bool, default False
        Whether to show count of samples in each group (for bar plots)
    additional_filters : Optional[Dict[str, Any]], default None
        Additional filters to apply to the data before aggregation.
        Format: {'column_name': value} or {'column_name': [value1, value2]}
    **kwargs
        Additional arguments passed to seaborn plotting functions

    Returns
    -------
    Figure
        The matplotlib figure object

    Examples
    --------
    >>> # Simple bar plot by column name
    >>> create_aggregated_plots(dfs, group_by='column_name', metric='mae')

    >>> # Multi-level grouping with aggregation
    >>> create_aggregated_plots(dfs, group_by=['column_name', 'model_name'],
    ...                        aggregation='median', plot_type='box')

    >>> # Line plot with additional filters
    >>> create_aggregated_plots(dfs, group_by='model_name', metric='mape',
    ...                        plot_type='line',
    ...                        additional_filters={'column_name': ['col1', 'col2']})
    """
    if not dataframes:
        raise ValueError("dataframes list cannot be empty")

    # Validate inputs
    valid_aggregations = ["mean", "sum", "median", "std", "count", "min", "max"]
    if aggregation is not None and aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {valid_aggregations}")

    # Determine if user explicitly provided aggregation
    explicit_aggregation = aggregation is not None

    valid_plot_types = ["bar", "line", "box", "histogram"]
    if plot_type not in valid_plot_types:
        raise ValueError(f"plot_type must be one of {valid_plot_types}")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract metric columns and metadata
    metric_data = []

    for _, row in combined_df.iterrows():
        # Find all metric columns for this row
        metric_cols = [
            col for col in combined_df.columns if col.endswith(f"_{metric}")
        ]

        for metric_col in metric_cols:
            if pd.notna(row[metric_col]):
                # Extract the correlate name from the metric column
                correlate_name = metric_col.replace(f"_{metric}", "")

                # Get model name
                model_col = f"{correlate_name}_model_name"
                model_name = row.get(model_col, "unknown")

                # Skip if model name is 'unknown'
                if model_name == "unknown":
                    continue

                # Get column name
                col_name_col = f"{correlate_name}_column_name"
                column_name = row.get(col_name_col, correlate_name)

                # Get sample ID
                sample_id_col = f"{correlate_name}_sample_id"
                sample_id = row.get(sample_id_col, "unknown")

                # Create data point
                data_point = {
                    "model_name": model_name,
                    "column_name": column_name,
                    "sample_id": sample_id,
                    "correlate": correlate_name,
                    "metric_value": row[metric_col],
                }

                # Add any additional columns that might be useful for grouping
                for col in combined_df.columns:
                    if col.startswith(f"{correlate_name}_") and col not in [
                        f"{correlate_name}_mae",
                        f"{correlate_name}_mape",
                        f"{correlate_name}_model_name",
                        f"{correlate_name}_column_name",
                        f"{correlate_name}_sample_id",
                    ]:
                        # Extract the base column name
                        base_col = col.replace(f"{correlate_name}_", "")
                        data_point[base_col] = row[col]

                metric_data.append(data_point)

    if not metric_data:
        raise ValueError(f"No {metric} data found in the provided dataframes")

    # Create DataFrame for analysis
    plot_df = pd.DataFrame(metric_data)

    # Apply additional filters if provided
    if additional_filters:
        for col, value in additional_filters.items():
            if col in plot_df.columns:
                if isinstance(value, list):
                    plot_df = plot_df[plot_df[col].isin(value)]
                else:
                    plot_df = plot_df[plot_df[col] == value]

    if plot_df.empty:
        raise ValueError("No data remaining after applying filters")

    # Handle grouping
    if isinstance(group_by, str):
        group_columns = [group_by]
    else:
        group_columns = group_by

    # Validate group columns exist
    missing_cols = [col for col in group_columns if col not in plot_df.columns]
    if missing_cols:
        raise ValueError(f"Group columns not found in data: {missing_cols}")

    if explicit_aggregation and aggregation == "count":
        # For count aggregation, we count unique samples per group
        if len(group_columns) > 1:
            aggregated_data = (
                plot_df.groupby(group_columns[0])["correlate"]
                .nunique()
                .reset_index()
            )
            aggregated_data.columns = [group_columns[0], "count"]
            y_column = "count"
        else:
            aggregated_data = (
                plot_df.groupby(group_columns)["correlate"]
                .nunique()
                .reset_index()
            )
            aggregated_data.columns = list(group_columns) + ["count"]
            y_column = "count"
    elif explicit_aggregation:
        # For other explicit aggregations, we aggregate the metric values
        if len(group_columns) > 1:
            aggregated_data = (
                plot_df.groupby(group_columns[0])["metric_value"]
                .agg(aggregation)
                .reset_index()
            )
            aggregated_data.columns = [
                group_columns[0],
                f"{aggregation}_{metric}",
            ]
            y_column = f"{aggregation}_{metric}"
        else:
            aggregated_data = (
                plot_df.groupby(group_columns)["metric_value"]
                .agg(aggregation)
                .reset_index()
            )
            aggregated_data.columns = list(group_columns) + [
                f"{aggregation}_{metric}"
            ]
            y_column = f"{aggregation}_{metric}"
    else:
        # No explicit aggregation - show all combinations
        if len(group_columns) > 1:
            aggregated_data = (
                plot_df.groupby(group_columns)["metric_value"]
                .mean()
                .reset_index()
            )
            aggregated_data.columns = list(group_columns) + [f"mean_{metric}"]
            y_column = f"mean_{metric}"
        else:
            aggregated_data = (
                plot_df.groupby(group_columns)["metric_value"]
                .mean()
                .reset_index()
            )
            aggregated_data.columns = list(group_columns) + [f"mean_{metric}"]
            y_column = f"mean_{metric}"

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, plt.Axes), "Expected single Axes object"

    # Generate title if not provided
    if title is None:
        if explicit_aggregation and aggregation == "count":
            if len(group_columns) > 1:
                title = f"Number of Samples by {group_columns[0]} (aggregated across {', '.join(group_columns[1:])})"
            else:
                title = f"Number of Samples by {group_columns[0]}"
        elif explicit_aggregation:
            if len(group_columns) > 1:
                title = f"{aggregation.title()} {metric.upper()} by {group_columns[0]} (aggregated across {', '.join(group_columns[1:])})"
            else:
                title = f"{aggregation.title()} {metric.upper()} by {group_columns[0]}"
        else:
            if len(group_columns) > 1:
                title = f"Mean {metric.upper()} by {group_columns[0]} and {group_columns[1]}"
            else:
                title = f"Mean {metric.upper()} by {group_columns[0]}"

    # Create different plot types
    if plot_type == "bar":
        if explicit_aggregation or len(group_columns) == 1:
            # For explicit aggregation or single grouping variable, use primary group
            sns.barplot(
                data=aggregated_data,
                x=group_columns[0],
                y=y_column,
                ax=ax,
                **kwargs,
            )
        else:
            # For multiple grouping variables without explicit aggregation, use hue
            sns.barplot(
                data=aggregated_data,
                x=group_columns[0],
                y=y_column,
                hue=group_columns[1],
                ax=ax,
                **kwargs,
            )

        # Add count annotations if requested
        if show_counts and (not explicit_aggregation or aggregation != "count"):
            for i, group in enumerate(
                aggregated_data[group_columns[0]].unique()
            ):
                group_data = aggregated_data[
                    aggregated_data[group_columns[0]] == group
                ]
                total_count = len(plot_df[plot_df[group_columns[0]] == group])

                for j, (_, row) in enumerate(group_data.iterrows()):
                    ax.text(
                        i,
                        row[y_column] + 0.01 * ax.get_ylim()[1],
                        f"n={total_count}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    elif plot_type == "line":
        if explicit_aggregation or len(group_columns) == 1:
            sns.lineplot(
                data=aggregated_data,
                x=group_columns[0],
                y=y_column,
                ax=ax,
                **kwargs,
            )
        else:
            sns.lineplot(
                data=aggregated_data,
                x=group_columns[0],
                y=y_column,
                hue=group_columns[1],
                ax=ax,
                **kwargs,
            )

    elif plot_type == "box":
        # For box plots, we need the original data, not aggregated
        if len(group_columns) == 1:
            sns.boxplot(
                data=plot_df,
                x=group_columns[0],
                y="metric_value",
                ax=ax,
                **kwargs,
            )
        else:
            sns.boxplot(
                data=plot_df,
                x=group_columns[0],
                y="metric_value",
                ax=ax,
                **kwargs,
            )
        y_column = "metric_value"  # Update for axis labels

    elif plot_type == "histogram":
        if len(group_columns) == 1:
            sns.histplot(
                data=plot_df,
                x="metric_value",
                hue=group_columns[0],
                ax=ax,
                **kwargs,
            )
            ax.set_xlabel(f"{metric.upper()}", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
        else:
            sns.histplot(
                data=plot_df,
                x="metric_value",
                hue=group_columns[0],
                ax=ax,
                **kwargs,
            )
            ax.set_xlabel(f"{metric.upper()}", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)

    # Customize the plot
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set axis labels
    if plot_type != "histogram":
        ax.set_xlabel(group_columns[0].replace("_", " ").title(), fontsize=12)

        if explicit_aggregation and aggregation == "count":
            ax.set_ylabel("Number of Samples", fontsize=12)
        elif explicit_aggregation:
            ax.set_ylabel(
                f"{aggregation.title()} {metric.upper()}", fontsize=12
            )
        else:
            ax.set_ylabel(f"Mean {metric.upper()}", fontsize=12)

    # Rotate x-axis labels if there are many groups
    if len(aggregated_data[group_columns[0]].unique()) > 10:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_sample_analysis_plots(
    dataframes: List[pd.DataFrame],
    metric: str = "mae",
    random_samples: int = 0,
    median_samples: int = 0,
    best_samples: int = 0,
    worst_samples: int = 0,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    seed: int = 42,
    **kwargs,
):
    """
    Create plots showing forecasts for samples with different metric characteristics.
    Generates a multi-page PDF with plots for all models in the dataframes.
    """
    if not dataframes:
        raise ValueError("dataframes list cannot be empty")

    # Extract metric data for each sample
    sample_metrics = []
    for df_idx, df in enumerate(dataframes):
        metric_cols = [col for col in df.columns if col.endswith(f"_{metric}")]
        for metric_col in metric_cols:
            if pd.notna(df[metric_col].iloc[0]):
                correlate_name = metric_col.replace(f"_{metric}", "")
                model_col = f"{correlate_name}_model_name"
                model_name = (
                    df[model_col].iloc[0]
                    if model_col in df.columns
                    else "unknown"
                )
                if model_name == "unknown":
                    continue
                sample_id_col = f"{correlate_name}_sample_id"
                sample_id = (
                    df[sample_id_col].iloc[0]
                    if sample_id_col in df.columns
                    else f"batch_{df_idx}"
                )
                col_name_col = f"{correlate_name}_column_name"
                column_name = (
                    df[col_name_col].iloc[0]
                    if col_name_col in df.columns
                    else correlate_name
                )
                actual_values = (
                    df[correlate_name].values
                    if correlate_name in df.columns
                    else None
                )
                forecast_values = (
                    df[f"{correlate_name}_forecast"].values
                    if f"{correlate_name}_forecast" in df.columns
                    else None
                )
                if actual_values is not None and forecast_values is not None:
                    if isinstance(actual_values, (list, pd.Series)):
                        actual_values = np.array(actual_values)
                    if isinstance(forecast_values, (list, pd.Series)):
                        forecast_values = np.array(forecast_values)
                    if len(actual_values) > 0 and len(forecast_values) > 0:
                        sample_metrics.append(
                            {
                                "sample_id": sample_id,
                                "correlate": correlate_name,
                                "model_name": model_name,
                                "column_name": column_name,
                                "metric_value": df[metric_col].iloc[0],
                                "actual_values": actual_values,
                                "forecast_values": forecast_values,
                                "row_index": len(sample_metrics),
                            }
                        )
    if not sample_metrics:
        raise ValueError(f"No {metric} data found in the provided dataframes")
    metrics_df = pd.DataFrame(sample_metrics)
    metrics_df = metrics_df.drop_duplicates(
        subset=["sample_id", "correlate", "model_name"]
    )
    valid_metrics = metrics_df[~pd.isna(metrics_df["metric_value"])]
    if len(valid_metrics) == 0:
        raise ValueError(f"No valid {metric} values found")
    # Group by model
    all_models = valid_metrics["model_name"].unique()
    figs = []
    pdf_pages = None
    if output_path and output_path.lower().endswith(".pdf"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pdf_pages = PdfPages(output_path)
    for model in all_models:
        model_metrics = valid_metrics[valid_metrics["model_name"] == model]
        if len(model_metrics) == 0:
            continue
        # Sort by metric value
        model_metrics_sorted = model_metrics.sort_values("metric_value")
        total_indices = []
        selection_labels = []
        # Random
        if random_samples > 0:
            np.random.seed(seed)
            num = min(random_samples, len(model_metrics_sorted))
            indices = np.random.choice(
                len(model_metrics_sorted), num, replace=False
            )
            total_indices.extend(
                model_metrics_sorted.iloc[indices].index.tolist()
            )
            selection_labels.extend(["Random"] * num)
        # Best
        if best_samples > 0:
            num = min(best_samples, len(model_metrics_sorted))
            indices = model_metrics_sorted.head(num).index.tolist()
            total_indices.extend(indices)
            selection_labels.extend(["Best"] * num)
        # Worst
        if worst_samples > 0:
            num = min(worst_samples, len(model_metrics_sorted))
            indices = model_metrics_sorted.tail(num).index.tolist()
            total_indices.extend(indices)
            selection_labels.extend(["Worst"] * num)
        # Median
        if median_samples > 0:
            num = min(median_samples, len(model_metrics_sorted))
            median_idx = len(model_metrics_sorted) // 2
            start_idx = median_idx - num // 2
            end_idx = start_idx + num
            indices = model_metrics_sorted.iloc[
                start_idx:end_idx
            ].index.tolist()
            total_indices.extend(indices)
            selection_labels.extend(["Median"] * num)
        # Default to 3 random if nothing selected
        if len(total_indices) == 0:
            np.random.seed(seed)
            num = min(3, len(model_metrics_sorted))
            indices = np.random.choice(
                len(model_metrics_sorted), num, replace=False
            )
            total_indices.extend(
                model_metrics_sorted.iloc[indices].index.tolist()
            )
            selection_labels.extend(["Random"] * num)
        num_samples = len(total_indices)
        if num_samples == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(num_samples, 1, figsize=figsize)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            elif isinstance(axes, np.ndarray):
                axes = axes.flatten()
        for i, (idx, label) in enumerate(zip(total_indices, selection_labels)):
            sample_data = metrics_df.loc[idx]
            actual_values = sample_data["actual_values"]
            forecast_values = sample_data["forecast_values"]
            if isinstance(actual_values, (list, pd.Series)):
                actual_values = np.array(actual_values)
            if isinstance(forecast_values, (list, pd.Series)):
                forecast_values = np.array(forecast_values)
            forecast_mask = ~pd.isna(forecast_values)
            history_length = len(forecast_values) - np.sum(forecast_mask)
            time_axis = np.arange(len(actual_values))
            current_ax = (
                axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            )  # type: ignore
            current_ax.plot(  # type: ignore
                time_axis,
                actual_values,
                label="True",
                color="blue",
                linewidth=2,
            )  # type: ignore
            if np.sum(forecast_mask) > 0:
                forecast_time = time_axis[forecast_mask]
                forecast_vals = forecast_values[forecast_mask]
                current_ax.plot(  # type: ignore
                    forecast_time,
                    forecast_vals,
                    label="Predicted",
                    color="red",
                    linewidth=2,
                    linestyle="--",
                )  # type: ignore
            if history_length > 0:
                current_ax.axvline(  # type: ignore
                    x=history_length - 0.5,
                    color="black",
                    linestyle=":",
                    alpha=0.7,
                    label="Forecast Start",
                )  # type: ignore
            metric_val = sample_data["metric_value"]
            metric_label = metric.upper()
            if label == "Random":
                title = f"{label} Sample - {sample_data['column_name']} ({metric_label}: {metric_val:.4f}, Seed: {seed})\nModel: {model}"
            else:
                title = f"{label} Sample - {sample_data['column_name']} ({metric_label}: {metric_val:.4f})\nModel: {model}"
            current_ax.set_title(title, fontsize=12, fontweight="bold")  # type: ignore
            current_ax.set_xlabel("Time Step", fontsize=10)  # type: ignore
            current_ax.set_ylabel("Value", fontsize=10)  # type: ignore
            current_ax.legend()  # type: ignore
            current_ax.grid(True, alpha=0.3)  # type: ignore
            info_text = f"Model: {model}\nSample ID: {sample_data['sample_id']}"
            current_ax.text(  # type: ignore
                0.02,
                0.98,
                info_text,
                transform=current_ax.transAxes,  # type: ignore
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
        plt.tight_layout()
        if pdf_pages:
            pdf_pages.savefig(fig)
            plt.close(fig)
        figs.append(fig)
    if pdf_pages:
        pdf_pages.close()
    # If not saving to PDF, save the last fig as PNG if requested
    if output_path and not output_path.lower().endswith(".pdf") and figs:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        figs[-1].savefig(output_path, dpi=300, bbox_inches="tight")
    return figs[-1] if figs else None
