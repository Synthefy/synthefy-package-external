"""
Plot creation utilities for DataSummarizer.
"""

import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from loguru import logger

from synthefy_pkg.preprocessing.data_summary_utils.autoregressions import (
    autocorrelation,
)
from synthefy_pkg.preprocessing.data_summary_utils.create_simple_plot import (
    create_stacked_time_series_plots,
)


def create_decomposition_plot(
    result: dict, col: str, title: str, timestamp_col: Optional[str] = None
) -> figure:
    """
    Create a decomposition plot for a time series column.

    Args:
        result: Decomposition result dictionary (may contain _grouped_results for plotting)
        col: Column name
        title: Plot title
        timestamp_col: Optional timestamp column name

    Returns:
        Bokeh figure object
    """
    # If result contains grouped results, use only the first group for plotting
    if "_grouped_results" in result:
        grouped_results = result["_grouped_results"]
        if grouped_results:
            # Take only the first group for plotting (first four groups available)
            result = grouped_results[0]
    p = figure(
        width=600,
        height=400,
        title=f"{title} Decomposition: {col}",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    if (
        "forecast" in result and result["forecast"].shape == ()
    ) or "forecast" not in result:
        # no forecast in this branch, just plot the decomposition
        x_data = np.arange(len(result["normalized_original"]))
        y_data_original = result["normalized_original"]
        y_data_forecast = result["reconstructed"]
    else:
        x_data = np.arange(
            len(result["normalized_original"]) + len(result["test_values"])
        )
        y_data_original = np.concatenate(
            [result["normalized_original"], result["test_values"]]
        )
        y_data_forecast = np.concatenate(
            [result["reconstructed"], result["forecast"]]
        )

    source_original = ColumnDataSource(
        data={
            "x": x_data,
            "y": y_data_original,
        }
    )
    p.line(
        "x",
        "y",
        source=source_original,
        line_color="blue",
        legend_label="Original",
        line_width=2,
    )

    source_recon = ColumnDataSource(
        data={
            "x": x_data,
            "y": y_data_forecast,
        }
    )
    p.line(
        "x",
        "y",
        source=source_recon,
        line_color="red",
        legend_label="Reconstructed",
        line_width=2,
    )

    p.xaxis.axis_label = "Index"
    p.yaxis.axis_label = col
    p.legend.location = "top_left"
    p.grid.grid_line_alpha = 0.2

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Value", "@y"),
            (
                "Time" if timestamp_col else "Index",
                "@x",
            ),
        ]
    )
    p.add_tools(hover)
    return p


def create_decomposition_plots(data_summarizer) -> None:
    """
    Create decomposition plots for time series columns.

    Args:
        data_summarizer: DataSummarizer instance with decomposition results
    """
    decomposition_plots = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    for col in all_cols:
        if not isinstance(
            data_summarizer.df_data.get_column(col).dtype,
            (pl.Float64, pl.Int64, pl.Float32, pl.Int32),
        ):
            continue

        col_results = data_summarizer.decomposition_results.get(col)
        if not col_results:
            logger.warning(f"No decomposition results found for column: {col}")
            continue

        # Create plots for each decomposition type
        for decomp_type, result in col_results.items():
            if decomp_type == "full_fourier":
                # Create frequency spectrum plot
                # Handle grouped results by taking the first group
                full_fourier_result = result
                if "_grouped_results" in result:
                    grouped_results = result["_grouped_results"]
                    full_fourier_result = (
                        grouped_results[0] if grouped_results else {}
                    )

                if (
                    "frequencies" in full_fourier_result
                    and "amplitudes" in full_fourier_result
                ):
                    # plot as period instead of frequency
                    full_fourier_result["periods"] = (
                        1 / full_fourier_result["frequencies"]
                    )
                    full_fourier_result["amplitudes"] = full_fourier_result[
                        "amplitudes"
                    ]

                    p_spectrum = figure(
                        width=600,
                        height=300,
                        title=f"Frequency Spectrum: {col}",
                        toolbar_location="above",
                        tools="pan,wheel_zoom,box_zoom,reset,save",
                    )

                    source_spectrum = ColumnDataSource(
                        data={
                            "period": full_fourier_result["periods"],
                            "amp": full_fourier_result["amplitudes"],
                        }
                    )

                    p_spectrum.vbar(
                        x="period",
                        top="amp",
                        width=0.1,
                        source=source_spectrum,
                        fill_color="green",
                        alpha=0.7,
                    )
                    p_spectrum.xaxis.axis_label = "Period"
                    p_spectrum.yaxis.axis_label = "Amplitude"
                    p_spectrum.grid.grid_line_alpha = 0.2

                    hover_spectrum = HoverTool(
                        tooltips=[
                            ("Period", "@period"),
                            ("Amplitude", "@amp"),
                        ]
                    )
                    p_spectrum.add_tools(hover_spectrum)

                    decomposition_plots.append(p_spectrum)
            elif decomp_type == "fourier":
                p = create_decomposition_plot(
                    result, col, "Fourier", data_summarizer.timestamps_col
                )
                decomposition_plots.append(p)

            elif decomp_type == "ssa":
                p = create_decomposition_plot(
                    result, col, "SSA", data_summarizer.timestamps_col
                )
                decomposition_plots.append(p)

            elif decomp_type == "sindy":
                p = create_decomposition_plot(
                    result, col, "SINDy", data_summarizer.timestamps_col
                )
                decomposition_plots.append(p)

            elif decomp_type == "stl":
                p = create_decomposition_plot(
                    result, col, "STL", data_summarizer.timestamps_col
                )
                decomposition_plots.append(p)

                # Create component plots for STL decomposition
                # Handle grouped results by taking the first group
                stl_result = result
                if "_grouped_results" in result:
                    grouped_results = result["_grouped_results"]
                    stl_result = grouped_results[0] if grouped_results else {}

                if (
                    "trend" in stl_result
                    and "seasonal" in stl_result
                    and "resid" in stl_result
                ):
                    # Create a multi-component plot showing trend, seasonal, and residual
                    p_components = figure(
                        width=600,
                        height=400,
                        title=f"STL Components: {col}",
                        toolbar_location="above",
                        tools="pan,wheel_zoom,box_zoom,reset,save",
                    )

                    x_data = np.arange(len(stl_result["trend"]))

                    # Plot trend component
                    source_trend = ColumnDataSource(
                        data={"x": x_data, "y": stl_result["trend"]}
                    )
                    p_components.line(
                        "x",
                        "y",
                        source=source_trend,
                        line_color="red",
                        legend_label="Trend",
                        line_width=2,
                    )

                    # Plot seasonal component
                    source_seasonal = ColumnDataSource(
                        data={"x": x_data, "y": stl_result["seasonal"]}
                    )
                    p_components.line(
                        "x",
                        "y",
                        source=source_seasonal,
                        line_color="green",
                        legend_label="Seasonal",
                        line_width=1,
                    )

                    # Plot residual component
                    source_resid = ColumnDataSource(
                        data={"x": x_data, "y": stl_result["resid"]}
                    )
                    p_components.line(
                        "x",
                        "y",
                        source=source_resid,
                        line_color="blue",
                        legend_label="Residual",
                        line_width=1,
                    )

                    p_components.xaxis.axis_label = "Index"
                    p_components.yaxis.axis_label = col
                    p_components.legend.location = "top_left"
                    p_components.grid.grid_line_alpha = 0.2

                    # Add hover tool
                    hover = HoverTool(
                        tooltips=[
                            ("Value", "@y"),
                            ("Index", "@x"),
                        ]
                    )
                    p_components.add_tools(hover)

                    decomposition_plots.append(p_components)

    data_summarizer.decomposition_plots = decomposition_plots
    logger.info(f"Created {len(decomposition_plots)} decomposition plots")


def create_heatmap_data(
    df, value_col, series1_col, series2_col, symmetric=True
):
    """
    Create heatmap data from a DataFrame with pairwise relationships.

    Args:
        df: DataFrame containing pairwise data
        value_col: Column name containing the values to plot
        series1_col: Column name for first series
        series2_col: Column name for second series
        symmetric: Whether the relationship is symmetric (e.g., correlation)

    Returns:
        tuple: (series_list, corr_matrix, x_coords, y_coords, values, colors)
    """
    if df is None or df.empty:
        return None, None, [], [], [], []

    # Get unique series names
    series_list = list(set(df[series1_col].tolist() + df[series2_col].tolist()))
    n_series = len(series_list)

    # Initialize matrix
    corr_matrix = np.zeros((n_series, n_series))

    # Fill diagonal with appropriate default value
    if symmetric:
        np.fill_diagonal(corr_matrix, 1.0)  # For correlation
    else:
        np.fill_diagonal(corr_matrix, 0.0)  # For transfer entropy

    # Fill matrix with values
    for _, row in df.iterrows():
        value = float(row[value_col]) if row[value_col] != "Error" else 0.0
        series1_idx = series_list.index(row[series1_col])
        series2_idx = series_list.index(row[series2_col])
        corr_matrix[series1_idx, series2_idx] = value

        if symmetric:
            corr_matrix[series2_idx, series1_idx] = value

    # Create heatmap coordinates and values
    x_coords = []
    y_coords = []
    values = []
    colors = []

    for i, series1 in enumerate(series_list):
        for j, series2 in enumerate(series_list):
            x_coords.append(series1)
            y_coords.append(series2)
            value = corr_matrix[i, j]
            values.append(value)
            colors.append(get_heatmap_color(value, symmetric))

    return series_list, corr_matrix, x_coords, y_coords, values, colors


def get_heatmap_color(value, symmetric=True):
    """
    Get color for heatmap based on value and relationship type.

    Args:
        value: The value to color-code
        symmetric: Whether the relationship is symmetric (affects color mapping)

    Returns:
        str: Hex color code
    """
    if symmetric:
        # Color mapping for symmetric relationships (like correlation)
        if value == 1.0:
            return "#000080"  # Perfect positive correlation - dark blue
        elif value > 0.7:
            return "#0000ff"  # Strong positive correlation - blue
        elif value > 0.3:
            return "#4d4dff"  # Moderate positive correlation - light blue
        elif value > 0:
            return "#b3b3ff"  # Weak positive correlation - very light blue
        elif value == 0:
            return "#ffffff"  # No correlation - white
        elif value > -0.3:
            return "#ffb3b3"  # Weak negative correlation - very light red
        elif value > -0.7:
            return "#ff4d4d"  # Moderate negative correlation - light red
        elif value > -1:
            return "#ff0000"  # Strong negative correlation - red
        else:
            return "#800000"  # Perfect negative correlation - dark red
    else:
        # Color mapping for asymmetric relationships (like transfer entropy)
        if value >= 0.8:
            return "#000080"  # Very high - dark blue
        elif value >= 0.6:
            return "#0000ff"  # High - blue
        elif value >= 0.4:
            return "#4d4dff"  # Medium-high - light blue
        elif value >= 0.2:
            return "#8080ff"  # Medium - lighter blue
        elif value >= 0.1:
            return "#b3b3ff"  # Low-medium - very light blue
        elif value >= 0.05:
            return "#e6e6ff"  # Low - very light blue
        elif value > 0:
            return "#f0f0ff"  # Very low - almost white
        else:
            return "#ffffff"  # Zero or negative - white


def create_heatmap_plot(
    series_list,
    x_coords,
    y_coords,
    values,
    colors,
    title,
    x_label,
    y_label,
    value_label="Value",
):
    """
    Create a Bokeh heatmap plot with the given data.

    Args:
        series_list: List of series names
        x_coords: X coordinates for rectangles
        y_coords: Y coordinates for rectangles
        values: Values to display
        colors: Colors for rectangles
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        value_label: Label for values in hover tool

    Returns:
        Bokeh figure object
    """
    p = figure(
        width=600,
        height=500,
        title=title,
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=series_list,
        y_range=series_list,
    )

    # Create data source
    source = ColumnDataSource(
        data={
            "x": x_coords,
            "y": y_coords,
            "value": values,
            "color": colors,
        }
    )

    # Create rectangles for heatmap
    p.rect(
        x="x",
        y="y",
        width=0.9,
        height=0.9,
        source=source,
        fill_color="color",
        fill_alpha=0.7,
        line_color="white",
        line_width=1,
    )

    # Add value labels
    p.text(
        x="x",
        y="y",
        text="value",
        source=source,
        text_font_size="10px",
        text_color="black",
        text_align="center",
        text_baseline="middle",
    )

    # Configure axes
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.grid.grid_line_alpha = 0.0
    p.xaxis.major_label_orientation = 0.8

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Series 1", "@x"),
            ("Series 2", "@y"),
            (value_label, "@value"),
        ]
    )
    p.add_tools(hover)

    return p


def create_cross_correlation_matrix_plot(data_summarizer) -> None:
    """Create a heatmap visualization of the cross-correlation matrix."""
    if (
        not hasattr(data_summarizer, "cross_corr_df")
        or data_summarizer.cross_corr_df is None
        or data_summarizer.cross_corr_df.empty
    ):
        data_summarizer.cross_corr_matrix_plot = None
        return

    # Create heatmap data using helper function
    series_list, _, x_coords, y_coords, values, colors = create_heatmap_data(
        data_summarizer.cross_corr_df,
        "Max Correlation",
        "Series 1",
        "Series 2",
        symmetric=True,  # Cross-correlation is symmetric
    )

    if series_list is None:
        data_summarizer.cross_corr_matrix_plot = None
        return

    # Create the plot using helper function
    p = create_heatmap_plot(
        series_list=series_list,
        x_coords=x_coords,
        y_coords=y_coords,
        values=values,
        colors=colors,
        title="Cross-Correlation Matrix Heatmap",
        x_label="Time Series",
        y_label="Time Series",
        value_label="Correlation",
    )

    data_summarizer.cross_corr_matrix_plot = p
    logger.info("Created cross-correlation matrix heatmap")


def create_transfer_entropy_matrix_plot(data_summarizer) -> None:
    """Create a heatmap visualization of the transfer entropy matrix."""
    if (
        not hasattr(data_summarizer, "transfer_entropy_df")
        or data_summarizer.transfer_entropy_df is None
        or data_summarizer.transfer_entropy_df.empty
    ):
        data_summarizer.transfer_entropy_matrix_plot = None
        return

    # Create heatmap data using helper function
    series_list, _, x_coords, y_coords, values, colors = create_heatmap_data(
        data_summarizer.transfer_entropy_df,
        "Transfer Entropy",
        "Source",
        "Target",
        symmetric=False,  # Transfer entropy is directional
    )

    if series_list is None:
        data_summarizer.transfer_entropy_matrix_plot = None
        return

    # Create the plot using helper function
    p = create_heatmap_plot(
        series_list=series_list,
        x_coords=x_coords,
        y_coords=y_coords,
        values=values,
        colors=colors,
        title="Transfer Entropy Matrix Heatmap",
        x_label="Source Series",
        y_label="Target Series",
        value_label="Transfer Entropy",
    )

    data_summarizer.transfer_entropy_matrix_plot = p
    logger.info("Created transfer entropy matrix heatmap")


def create_mutual_information_matrix_plot(data_summarizer) -> None:
    """Create a heatmap visualization of the mutual information matrix."""
    if (
        not hasattr(data_summarizer, "mutual_info_df")
        or data_summarizer.mutual_info_df is None
        or data_summarizer.mutual_info_df.empty
    ):
        data_summarizer.mutual_info_matrix_plot = None
        return

    # Create heatmap data using helper function
    series_list, _, x_coords, y_coords, values, colors = create_heatmap_data(
        data_summarizer.mutual_info_df,
        "Mutual Information",
        "Column 1",
        "Column 2",
        symmetric=True,  # Mutual information is symmetric
    )

    if series_list is None:
        data_summarizer.mutual_info_matrix_plot = None
        return

    # Create the plot using helper function
    p = create_heatmap_plot(
        series_list=series_list,
        x_coords=x_coords,
        y_coords=y_coords,
        values=values,
        colors=colors,
        title="Mutual Information Matrix Heatmap",
        x_label="Series",
        y_label="Series",
        value_label="Mutual Information",
    )

    data_summarizer.mutual_info_matrix_plot = p
    logger.info("Created mutual information matrix heatmap")


def create_autocorrelation_plots(data_summarizer) -> None:
    """Create autocorrelation plots for time series columns."""
    autocorr_plots = []

    all_cols = data_summarizer.timeseries_cols + data_summarizer.continuous_cols
    for col in all_cols:
        if data_summarizer.timestamps_col:
            # Perform autocorrelation analysis
            lags, autocorr_values = autocorrelation(
                data_summarizer.df_data, col, max_lag=100
            )

            if len(lags) > 1 and len(autocorr_values) > 1:
                # Create autocorrelation plot
                p = figure(
                    width=600,
                    height=400,
                    title=f"Autocorrelation: {col}",
                    toolbar_location="above",
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                )

                source = ColumnDataSource(
                    data={"lags": lags, "autocorr": autocorr_values}
                )

                # Plot autocorrelation values
                p.line(
                    "lags",
                    "autocorr",
                    source=source,
                    line_color="blue",
                    line_width=2,
                )
                p.scatter(
                    "lags",
                    "autocorr",
                    source=source,
                    fill_color="blue",
                    size=6,
                    alpha=0.7,
                )

                # Add horizontal lines at 0, ±0.5, and ±1
                p.line(
                    [0, max(lags)],
                    [0, 0],
                    line_color="black",
                    line_dash="dashed",
                    alpha=0.5,
                )
                p.line(
                    [0, max(lags)],
                    [0.5, 0.5],
                    line_color="red",
                    line_dash="dotted",
                    alpha=0.3,
                )
                p.line(
                    [0, max(lags)],
                    [-0.5, -0.5],
                    line_color="red",
                    line_dash="dotted",
                    alpha=0.3,
                )
                p.line(
                    [0, max(lags)],
                    [1, 1],
                    line_color="green",
                    line_dash="dotted",
                    alpha=0.3,
                )
                p.line(
                    [0, max(lags)],
                    [-1, -1],
                    line_color="green",
                    line_dash="dotted",
                    alpha=0.3,
                )

                p.xaxis.axis_label = "Lag"
                p.yaxis.axis_label = "Autocorrelation"
                p.grid.grid_line_alpha = 0.2

                # Add hover tool
                hover = HoverTool(
                    tooltips=[
                        ("Lag", "@lags"),
                        ("Autocorrelation", "@autocorr"),
                    ]
                )
                p.add_tools(hover)

                autocorr_plots.append(p)

    data_summarizer.autocorr_plots = autocorr_plots
    logger.info(f"Created {len(autocorr_plots)} autocorrelation plots")


def create_granger_causality_matrix_plot(data_summarizer) -> None:
    """Create a heatmap visualization of the Granger causality matrix."""
    if (
        not hasattr(data_summarizer, "granger_causality_df")
        or data_summarizer.granger_causality_df is None
        or data_summarizer.granger_causality_df.empty
    ):
        data_summarizer.granger_causality_matrix_plot = None
        return

    # Create heatmap data using helper function
    series_list, _, x_coords, y_coords, values, colors = create_heatmap_data(
        data_summarizer.granger_causality_df,
        "Granger Causality",
        "Source",
        "Target",
        symmetric=False,  # Granger causality is directional
    )

    if series_list is None:
        data_summarizer.granger_causality_matrix_plot = None
        return

    # Create the plot using helper function
    p = create_heatmap_plot(
        series_list=series_list,
        x_coords=x_coords,
        y_coords=y_coords,
        values=values,
        colors=colors,
        title="Granger Causality Matrix Heatmap",
        x_label="Source Series",
        y_label="Target Series",
        value_label="Granger Causality",
    )

    data_summarizer.granger_causality_matrix_plot = p
    logger.info("Created Granger causality matrix heatmap")


def create_dlinear_matrix_plot(data_summarizer) -> None:
    """Create a heatmap visualization of the DLinear causality matrix."""
    if (
        not hasattr(data_summarizer, "dlinear_df")
        or data_summarizer.dlinear_df is None
        or data_summarizer.dlinear_df.empty
    ):
        data_summarizer.dlinear_matrix_plot = None
        return

    # Create heatmap data using helper function
    series_list, _, x_coords, y_coords, values, colors = create_heatmap_data(
        data_summarizer.dlinear_df,
        "DLinear Causality",
        "Source",
        "Target",
        symmetric=False,  # DLinear causality is directional
    )

    if series_list is None:
        data_summarizer.dlinear_matrix_plot = None
        return

    # Create the plot using helper function
    p = create_heatmap_plot(
        series_list=series_list,
        x_coords=x_coords,
        y_coords=y_coords,
        values=values,
        colors=colors,
        title="DLinear Causality Matrix Heatmap",
        x_label="Source Series",
        y_label="Target Series",
        value_label="DLinear Causality",
    )

    data_summarizer.dlinear_matrix_plot = p
    logger.info("Created DLinear causality matrix heatmap")


def create_convergent_cross_mapping_matrix_plot(data_summarizer) -> None:
    """Create a heatmap visualization of the convergent cross-mapping matrix."""
    if (
        not hasattr(data_summarizer, "convergent_cross_mapping_df")
        or data_summarizer.convergent_cross_mapping_df is None
        or data_summarizer.convergent_cross_mapping_df.empty
    ):
        data_summarizer.convergent_cross_mapping_matrix_plot = None
        return

    # Create heatmap data using helper function
    series_list, _, x_coords, y_coords, values, colors = create_heatmap_data(
        data_summarizer.convergent_cross_mapping_df,
        "Convergent Cross-Mapping",
        "Source",
        "Target",
        symmetric=False,  # CCM is directional
    )

    if series_list is None:
        data_summarizer.convergent_cross_mapping_matrix_plot = None
        return

    # Create the plot using helper function
    p = create_heatmap_plot(
        series_list=series_list,
        x_coords=x_coords,
        y_coords=y_coords,
        values=values,
        colors=colors,
        title="Convergent Cross-Mapping Matrix Heatmap",
        x_label="Source Series",
        y_label="Target Series",
        value_label="CCM Score",
    )

    data_summarizer.convergent_cross_mapping_matrix_plot = p
    logger.info("Created convergent cross-mapping matrix heatmap")


def create_stacked_time_series_visualizations(
    data_summarizer,
    limit_groups: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Create stacked time series plots for each node/group."""
    logger.info("Creating stacked time series visualizations...")
    if output_dir is None:
        output_dir = "data_visualizations"
    save_dir = os.path.join(output_dir, "stacked_time_series")
    os.makedirs(save_dir, exist_ok=True)
    create_stacked_time_series_plots(
        data_summarizer, save_dir=save_dir, limit_groups=limit_groups
    )
    logger.info("Created stacked time series visualizations")
