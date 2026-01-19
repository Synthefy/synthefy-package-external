import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from synthefy_pkg.preprocessing.data_summary_utils.decompositions import (
    apply_grouping,
)


def create_simple_plot(data: pd.DataFrame, x: str, y: str, title: str) -> Any:
    """
    Create a simple plot with the given data.
    """
    fig = plt.figure()
    plt.plot(data[x], data[y])
    plt.title(title)
    plt.show()
    return fig


def create_stacked_time_series_plots(
    data_summarizer,
    save_dir: str = "data_visualizations",
    limit_groups: Optional[int] = None,
) -> None:
    """
    Create stacked time series plots for each node/group, showing all features stacked vertically.
    Uses apply_grouping function from decompositions to handle grouping consistently.

    Args:
        data_summarizer: DataSummarizer instance with data and column information
        save_dir: Directory to save the plots (default: "data_visualizations")
        limit_groups: Optional limit on number of groups to plot (similar to limit_cols in decompositions)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get all feature columns (timeseries + continuous)
    feature_cols = (
        data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    )

    if not feature_cols:
        logger.warning("No feature columns found for plotting")
        return

    # Get timestamp column for x-axis
    timestamp_col = data_summarizer.timestamps_col

    # Use apply_grouping from decompositions to handle grouping consistently
    # Create a counter to track sequential group numbers
    group_counter = {"count": 0}

    def _plot_group_function_with_counter(df, timestamps, *args, **kwargs):
        group_counter["count"] += 1
        # Extract the parameters we need
        feature_cols = kwargs.get("feature_cols", [])
        timestamp_col = kwargs.get("timestamp_col", None)
        save_dir = kwargs.get("save_dir", "data_visualizations")

        return _plot_group_function(
            df,
            timestamps,
            feature_cols,
            timestamp_col,
            save_dir,
            group_id=f"group_{group_counter['count']}",
        )

    apply_grouping(
        data_summarizer.df_data,
        _plot_group_function_with_counter,
        timestamp_column=timestamp_col,
        limit_cols=limit_groups,
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
        save_dir=save_dir,
    )

    logger.info(f"Created stacked time series plots in {save_dir}")


def _plot_group_function(
    df: pl.DataFrame,
    timestamps: Optional[Any],
    feature_cols: List[str],
    timestamp_col: Optional[str],
    save_dir: str,
    group_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot function that can be used with apply_grouping from decompositions.
    Creates a single stacked plot for the given group data.

    Args:
        df: Polars DataFrame containing the group data
        timestamps: Optional timestamps array (not used in plotting)
        feature_cols: List of feature column names to plot
        timestamp_col: Optional timestamp column name for x-axis
        save_dir: Directory to save the plot
        group_id: Optional group identifier (if not provided, will generate one)

    Returns:
        Dictionary with plot information (for consistency with decomposition functions)
    """
    # Convert to pandas for plotting
    df_pandas = df.to_pandas()

    # Use provided group_id or create a fallback identifier
    if group_id is None:
        group_id = f"group_{hash(str(df_pandas.head(3).values.tobytes()))}"

    # Create the stacked plot for this group
    _create_single_stacked_plot(
        df_pandas, feature_cols, timestamp_col, group_id, save_dir
    )

    # Return a dictionary for consistency with decomposition functions
    return {"plot_created": True, "group_id": group_id}


def _create_single_stacked_plot(
    data: pd.DataFrame,
    feature_cols: List[str],
    timestamp_col: Optional[str],
    group_id: str,
    save_dir: str,
) -> None:
    """
    Create a single stacked plot for one group/node.
    If there are more than 20 features, arranges them in multiple columns side by side.

    Args:
        data: DataFrame containing the data for this group
        feature_cols: List of feature column names to plot
        timestamp_col: Optional timestamp column name for x-axis
        group_id: Identifier for this group (used in filename and title)
        save_dir: Directory to save the plot
    """
    if data.empty:
        logger.warning(f"No data found for group {group_id}")
        return

    # Determine x-axis data
    if timestamp_col and timestamp_col in data.columns:
        x_data = pd.to_datetime(data[timestamp_col])
        x_label = "Time"
    else:
        x_data = range(len(data))
        x_label = "Index"

    # Split features into chunks of 20 to avoid matplotlib size limits
    max_features_per_column = 20
    feature_chunks = [
        feature_cols[i : i + max_features_per_column]
        for i in range(0, len(feature_cols), max_features_per_column)
    ]

    n_features = len(feature_cols)
    n_columns = len(feature_chunks)

    logger.info(
        f"Creating single plot for group {group_id} with {n_features} features arranged in {n_columns} columns"
    )

    # Create figure with subplots arranged in columns
    # Each column will have up to 20 features stacked vertically
    fig, axes = plt.subplots(
        max_features_per_column,
        n_columns,
        figsize=(6 * n_columns, 3 * max_features_per_column),
        sharex=True,
    )

    # Handle matplotlib's inconsistent return types for subplots
    # When n_columns=1, matplotlib returns a single column of axes
    # When n_columns>1, matplotlib returns a 2D array of axes
    if n_columns == 1:
        if max_features_per_column == 1:
            axes_array: np.ndarray = np.array([[axes]])
        else:
            axes_array = axes.reshape(-1, 1)
    else:
        axes_array = axes

    # Plot each chunk in its own column
    for col_idx, feature_chunk in enumerate(feature_chunks):
        for row_idx, feature in enumerate(feature_chunk):
            ax = axes_array[row_idx, col_idx]
            if feature in data.columns:
                # Check if column has numeric data
                y_data = pd.to_numeric(data[feature], errors="coerce")

                # Plot the data
                ax.plot(x_data, y_data, linewidth=1, alpha=0.8)
                ax.set_ylabel(feature, fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{feature}", fontsize=8)

                # Add some styling
                ax.tick_params(axis="x", rotation=45, labelsize=6)
                ax.tick_params(axis="y", labelsize=6)

            else:
                logger.warning(f"Feature {feature} not found in data")
                ax.text(
                    0.5,
                    0.5,
                    f"Feature {feature} not found",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax.set_ylabel(feature, fontsize=8)

        # Hide unused subplots in this column
        for row_idx in range(len(feature_chunk), max_features_per_column):
            axes_array[row_idx, col_idx].set_visible(False)

    # Set x-axis labels on the bottom row
    for col_idx in range(n_columns):
        bottom_row_idx = min(
            len(feature_chunks[col_idx]) - 1, max_features_per_column - 1
        )
        axes_array[bottom_row_idx, col_idx].set_xlabel(x_label, fontsize=8)

    # Set overall title
    title = f"Time Series Features - {group_id} ({n_features} features in {n_columns} columns)"
    fig.suptitle(title, fontsize=12, y=0.98)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the single plot
    filename = f"stacked_timeseries_{group_id}.jpg"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight", format="jpg")
    plt.close()

    logger.info(f"Saved stacked plot: {filepath}")
