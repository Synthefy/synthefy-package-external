"""
HTML report generation utilities for DataSummarizer.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    Div,
    HoverTool,
    TableColumn,
)
from bokeh.plotting import figure
from loguru import logger


def generate_html_report(
    data_summarizer,
    output_html: str = "data_summary_report.html",
) -> str:
    """
    Generate an interactive HTML report with metadata, time series summaries, tables,
    and interactive plots using the Bokeh module.

    Args:
        data_summarizer: DataSummarizer instance with analysis results
        output_html: Path for the output HTML file

    Returns:
        Path to the generated HTML file
    """
    # Ensure comprehensive analysis has been completed
    data_summarizer._ensure_analysis_complete()

    output_dir = os.path.dirname(os.path.abspath(output_html))
    os.makedirs(output_dir, exist_ok=True)
    output_file(output_html, title="Data Summary Report", mode="cdn")

    layout_elements = []

    # Global CSS styling
    css = """
    <style>
    body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px; }
    h1, h2, h3, h4 { color: #333; }
    .section-header { margin-top: 30px; }
    .container { max-width: 1200px; margin: 0 auto; padding: 30px; margin-left: 30px; margin-right: 30px; }
    </style>
    """
    css_div = Div(text=css, width=800)
    layout_elements.append(css_div)

    # Container wrapper
    container_start = Div(text="<div class='container'>", width=800)
    layout_elements.append(container_start)

    header = Div(
        text="<h1 style='text-align:center;'>Data Summary Report</h1>",
        width=800,
    )
    layout_elements.append(header)

    # Data Sources Information
    if len(data_summarizer.data_sources) > 1:
        sources_div = Div(
            text="<h2 class='section-header'>Data Sources</h2>",
            width=800,
        )
        layout_elements.append(sources_div)

        sources_text = "<ul>"
        for source in data_summarizer.data_sources:
            sources_text += f"<li>{source}</li>"
        sources_text += "</ul>"

        sources_info = Div(
            text=sources_text,
            width=800,
        )
        layout_elements.append(sources_info)

    # Time Series Summary Table
    if (
        data_summarizer.timeseries_cols
        and data_summarizer.time_series_df is not None
    ):
        ts_div = Div(
            text="<h2 class='section-header'>Time Series Summary</h2>",
            width=800,
        )
        layout_elements.append(ts_div)
        ts_source = ColumnDataSource(data_summarizer.time_series_df)
        ts_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.time_series_df.columns
        ]
        ts_table = DataTable(
            source=ts_source,
            columns=ts_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(ts_table)

    # Time Range Table
    if (
        data_summarizer.timestamps_col
        and data_summarizer.time_range is not None
    ):
        tr_div = Div(
            text="<h2 class='section-header'>Time Range</h2>", width=800
        )
        layout_elements.append(tr_div)
        tr_source = ColumnDataSource(data_summarizer.time_range)
        tr_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.time_range.columns
        ]
        tr_table = DataTable(
            source=tr_source,
            columns=tr_columns,
            width=800,
            height=100,
            sizing_mode="stretch_width",
        )
        layout_elements.append(tr_table)

    # Metadata Summary Table
    if (
        data_summarizer.metadata_df is not None
        and not data_summarizer.metadata_df.empty
    ):
        meta_div = Div(
            text="<h2 class='section-header'>Metadata Summary</h2>",
            width=800,
        )
        layout_elements.append(meta_div)
        meta_source = ColumnDataSource(data_summarizer.metadata_df)
        meta_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.metadata_df.columns
        ]
        meta_table = DataTable(
            source=meta_source,
            columns=meta_columns,
            width=800,
            height=300,
            sizing_mode="stretch_width",
        )
        layout_elements.append(meta_table)

    # Sample Counts Table
    if (
        data_summarizer.sample_counts is not None
        and not data_summarizer.sample_counts.empty
    ):
        sc_div = Div(
            text="<h2 class='section-header'>Sample Counts</h2>", width=800
        )
        layout_elements.append(sc_div)
        sc_source = ColumnDataSource(data_summarizer.sample_counts)
        sc_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.sample_counts.columns
        ]
        sc_table = DataTable(
            source=sc_source,
            columns=sc_columns,
            width=800,
            height=100,
            sizing_mode="stretch_width",
        )
        layout_elements.append(sc_table)

    # Number of Columns by Type Table
    if data_summarizer.num_columns_by_type:
        nct_div = Div(
            text="<h2 class='section-header'>Number of Columns by Type</h2>",
            width=800,
        )
        layout_elements.append(nct_div)
        num_columns_df = pd.DataFrame(
            list(data_summarizer.num_columns_by_type.items()),
            columns=["Type", "Count"],
        )
        nct_source = ColumnDataSource(num_columns_df)
        nct_columns = [
            TableColumn(field=col, title=col) for col in num_columns_df.columns
        ]
        nct_table = DataTable(
            source=nct_source,
            columns=nct_columns,
            width=400,
            height=100,
            sizing_mode="stretch_width",
        )
        layout_elements.append(nct_table)

    # Add analysis tables
    _add_analysis_tables(data_summarizer, layout_elements)

    # Add visualization plots
    _add_visualization_plots(data_summarizer, layout_elements)

    # Add time series plots
    _add_time_series_plots(data_summarizer, layout_elements)

    # Add statistical distribution plots
    _add_distribution_plots(data_summarizer, layout_elements)

    # Add discrete variable plots
    _add_discrete_plots(data_summarizer, layout_elements)

    # Column Lists Section
    _add_column_lists(data_summarizer, layout_elements)

    # Close container div
    container_end = Div(text="</div>", width=800)
    layout_elements.append(container_end)

    report_layout = column(*layout_elements, sizing_mode="scale_width")
    save(report_layout)
    logger.info(f"Bokeh interactive HTML report saved to {output_html}")

    # data_summarizer.cleanup()
    return output_html


def _add_analysis_tables(data_summarizer, layout_elements):
    """Add analysis tables to the layout."""
    # Basic Statistics Table
    if (
        hasattr(data_summarizer, "basic_stats_df")
        and data_summarizer.basic_stats_df is not None
        and not data_summarizer.basic_stats_df.empty
    ):
        bs_div = Div(
            text="<h2 class='section-header'>Basic Statistics</h2>",
            width=800,
        )
        layout_elements.append(bs_div)
        bs_source = ColumnDataSource(data_summarizer.basic_stats_df)
        bs_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.basic_stats_df.columns
        ]
        bs_table = DataTable(
            source=bs_source,
            columns=bs_columns,
            width=800,
            height=300,
            sizing_mode="stretch_width",
        )
        layout_elements.append(bs_table)

    # Correlation Matrix Table
    if (
        hasattr(data_summarizer, "correlation_df")
        and data_summarizer.correlation_df is not None
        and not data_summarizer.correlation_df.empty
    ):
        corr_div = Div(
            text="<h2 class='section-header'>Correlation Matrix</h2>",
            width=800,
        )
        layout_elements.append(corr_div)
        corr_source = ColumnDataSource(data_summarizer.correlation_df)
        corr_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.correlation_df.columns
        ]
        corr_table = DataTable(
            source=corr_source,
            columns=corr_columns,
            width=800,
            height=300,
            sizing_mode="stretch_width",
        )
        layout_elements.append(corr_table)

    # Outlier Analysis Table
    if (
        hasattr(data_summarizer, "outlier_df")
        and data_summarizer.outlier_df is not None
        and not data_summarizer.outlier_df.empty
    ):
        outlier_div = Div(
            text="<h2 class='section-header'>Outlier Analysis</h2>",
            width=800,
        )
        layout_elements.append(outlier_div)
        outlier_source = ColumnDataSource(data_summarizer.outlier_df)
        outlier_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.outlier_df.columns
        ]
        outlier_table = DataTable(
            source=outlier_source,
            columns=outlier_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(outlier_table)

    # Quantile Analysis Table
    if (
        hasattr(data_summarizer, "quantile_df")
        and data_summarizer.quantile_df is not None
        and not data_summarizer.quantile_df.empty
    ):
        quantile_div = Div(
            text="<h2 class='section-header'>Quantile Analysis</h2>",
            width=800,
        )
        layout_elements.append(quantile_div)
        quantile_source = ColumnDataSource(data_summarizer.quantile_df)
        quantile_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.quantile_df.columns
        ]
        quantile_table = DataTable(
            source=quantile_source,
            columns=quantile_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(quantile_table)

    # Autocorrelation Analysis Table
    if (
        hasattr(data_summarizer, "autocorr_df")
        and data_summarizer.autocorr_df is not None
        and not data_summarizer.autocorr_df.empty
    ):
        autocorr_div = Div(
            text="<h2 class='section-header'>Autocorrelation Analysis</h2>",
            width=800,
        )
        layout_elements.append(autocorr_div)
        autocorr_source = ColumnDataSource(data_summarizer.autocorr_df)
        autocorr_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.autocorr_df.columns
        ]
        autocorr_table = DataTable(
            source=autocorr_source,
            columns=autocorr_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(autocorr_table)

    # Decomposition Analysis Table
    if (
        hasattr(data_summarizer, "decomposition_df")
        and data_summarizer.decomposition_df is not None
        and not data_summarizer.decomposition_df.empty
    ):
        decomp_div = Div(
            text="<h2 class='section-header'>Decomposition Analysis</h2>",
            width=800,
        )
        layout_elements.append(decomp_div)
        decomp_source = ColumnDataSource(data_summarizer.decomposition_df)
        decomp_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.decomposition_df.columns
        ]
        decomp_table = DataTable(
            source=decomp_source,
            columns=decomp_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(decomp_table)

    # Cross-Correlation Analysis Table
    if (
        hasattr(data_summarizer, "cross_corr_df")
        and data_summarizer.cross_corr_df is not None
        and not data_summarizer.cross_corr_df.empty
    ):
        cross_corr_div = Div(
            text="<h2 class='section-header'>Cross-Correlation Analysis</h2>",
            width=800,
        )
        layout_elements.append(cross_corr_div)
        cross_corr_source = ColumnDataSource(data_summarizer.cross_corr_df)
        cross_corr_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.cross_corr_df.columns
        ]
        cross_corr_table = DataTable(
            source=cross_corr_source,
            columns=cross_corr_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(cross_corr_table)

    # Transfer Entropy Analysis Table
    if (
        hasattr(data_summarizer, "transfer_entropy_df")
        and data_summarizer.transfer_entropy_df is not None
        and not data_summarizer.transfer_entropy_df.empty
    ):
        te_table_div = Div(
            text="<h2 class='section-header'>Transfer Entropy Analysis</h2>",
            width=800,
        )
        layout_elements.append(te_table_div)
        te_source = ColumnDataSource(data_summarizer.transfer_entropy_df)
        te_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.transfer_entropy_df.columns
        ]
        te_table = DataTable(
            source=te_source,
            columns=te_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(te_table)

    # Mutual Information Analysis Table
    if (
        hasattr(data_summarizer, "mutual_info_df")
        and data_summarizer.mutual_info_df is not None
        and not data_summarizer.mutual_info_df.empty
    ):
        mi_table_div = Div(
            text="<h2 class='section-header'>Mutual Information Analysis</h2>",
            width=800,
        )
        layout_elements.append(mi_table_div)
        mi_source = ColumnDataSource(data_summarizer.mutual_info_df)
        mi_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.mutual_info_df.columns
        ]
        mi_table = DataTable(
            source=mi_source,
            columns=mi_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(mi_table)

    # Granger Causality Analysis Table
    if (
        hasattr(data_summarizer, "granger_causality_df")
        and data_summarizer.granger_causality_df is not None
        and not data_summarizer.granger_causality_df.empty
    ):
        gc_table_div = Div(
            text="<h2 class='section-header'>Granger Causality Analysis</h2>",
            width=800,
        )
        layout_elements.append(gc_table_div)
        gc_source = ColumnDataSource(data_summarizer.granger_causality_df)
        gc_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.granger_causality_df.columns
        ]
        gc_table = DataTable(
            source=gc_source,
            columns=gc_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(gc_table)

    # DLinear Causality Analysis Table
    if (
        hasattr(data_summarizer, "dlinear_df")
        and data_summarizer.dlinear_df is not None
        and not data_summarizer.dlinear_df.empty
    ):
        dlinear_table_div = Div(
            text="<h2 class='section-header'>DLinear Causality Analysis</h2>",
            width=800,
        )
        layout_elements.append(dlinear_table_div)
        dlinear_source = ColumnDataSource(data_summarizer.dlinear_df)
        dlinear_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.dlinear_df.columns
        ]
        dlinear_table = DataTable(
            source=dlinear_source,
            columns=dlinear_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(dlinear_table)

    # CCM Causality Analysis Table
    if (
        hasattr(data_summarizer, "ccm_df")
        and data_summarizer.ccm_df is not None
        and not data_summarizer.ccm_df.empty
    ):
        ccm_table_div = Div(
            text="<h2 class='section-header'>CCM Causality Analysis</h2>",
            width=800,
        )
        layout_elements.append(ccm_table_div)
        ccm_source = ColumnDataSource(data_summarizer.ccm_df)
        ccm_columns = [
            TableColumn(field=col, title=col)
            for col in data_summarizer.ccm_df.columns
        ]
        ccm_table = DataTable(
            source=ccm_source,
            columns=ccm_columns,
            width=800,
            height=200,
            sizing_mode="stretch_width",
        )
        layout_elements.append(ccm_table)


def _add_visualization_plots(data_summarizer, layout_elements):
    """Add visualization plots to the layout."""
    # Cross-Correlation Matrix Heatmap
    if (
        hasattr(data_summarizer, "cross_corr_matrix_plot")
        and data_summarizer.cross_corr_matrix_plot is not None
    ):
        cross_corr_plot_div = Div(
            text="<h2 class='section-header'>Cross-Correlation Matrix Visualization</h2>",
            width=800,
        )
        layout_elements.append(cross_corr_plot_div)
        layout_elements.append(data_summarizer.cross_corr_matrix_plot)

    # Decomposition Plots
    if (
        hasattr(data_summarizer, "decomposition_plots")
        and data_summarizer.decomposition_plots
    ):
        decomp_plots_div = Div(
            text="<h2 class='section-header'>Decomposition Analysis Visualizations</h2>",
            width=800,
        )
        layout_elements.append(decomp_plots_div)

        # Create a grid layout for decomposition plots
        if len(data_summarizer.decomposition_plots) > 1:
            # Use gridplot for multiple plots
            decomp_grid = gridplot(
                data_summarizer.decomposition_plots,
                ncols=2,
                sizing_mode="scale_width",
                merge_tools=False,
            )
            layout_elements.append(decomp_grid)
        else:
            # Single plot
            for plot in data_summarizer.decomposition_plots:
                layout_elements.append(plot)

    # Transfer Entropy Plot
    if (
        hasattr(data_summarizer, "transfer_entropy_matrix_plot")
        and data_summarizer.transfer_entropy_matrix_plot
    ):
        layout_elements.append(data_summarizer.transfer_entropy_matrix_plot)

    # Mutual Information Plot
    if (
        hasattr(data_summarizer, "mutual_info_matrix_plot")
        and data_summarizer.mutual_info_matrix_plot
    ):
        layout_elements.append(data_summarizer.mutual_info_matrix_plot)

    # Granger Causality Plot
    if (
        hasattr(data_summarizer, "granger_causality_matrix_plot")
        and data_summarizer.granger_causality_matrix_plot
    ):
        layout_elements.append(data_summarizer.granger_causality_matrix_plot)

    # DLinear Causality Plot
    if (
        hasattr(data_summarizer, "dlinear_matrix_plot")
        and data_summarizer.dlinear_matrix_plot
    ):
        layout_elements.append(data_summarizer.dlinear_matrix_plot)

    # Convergent Cross-Mapping Matrix Plot
    if (
        hasattr(data_summarizer, "convergent_cross_mapping_matrix_plot")
        and data_summarizer.convergent_cross_mapping_matrix_plot
    ):
        layout_elements.append(
            data_summarizer.convergent_cross_mapping_matrix_plot
        )

    # Autocorrelation Plots
    if (
        hasattr(data_summarizer, "autocorr_plots")
        and data_summarizer.autocorr_plots
    ):
        autocorr_plots_div = Div(
            text="<h2 class='section-header'>Autocorrelation Analysis Visualizations</h2>",
            width=800,
        )
        layout_elements.append(autocorr_plots_div)

        # Create a grid layout for autocorrelation plots
        if len(data_summarizer.autocorr_plots) > 1:
            # Use gridplot for multiple plots
            autocorr_grid = gridplot(
                data_summarizer.autocorr_plots,
                ncols=2,
                sizing_mode="scale_width",
                merge_tools=False,
            )
            layout_elements.append(autocorr_grid)
        else:
            # Single plot
            for plot in data_summarizer.autocorr_plots:
                layout_elements.append(plot)


def _add_time_series_plots(data_summarizer, layout_elements):
    """Add time series plots to the layout."""
    if not data_summarizer.skip_plots and data_summarizer.timeseries_cols:
        ts_plots_div = Div(
            text="<h2 class='section-header'>Time Series Plots</h2>",
            width=800,
        )
        layout_elements.append(ts_plots_div)

        df_data = data_summarizer.df_data
        # Consider total data points as len(data) * len(columns)
        num_rows = len(df_data)
        num_cols = len(df_data.columns)
        total_data_points = num_rows * num_cols

        if total_data_points > data_summarizer.MAX_SAMPLES:
            # Calculate sampling ratio based on both rows and columns
            sample_ratio = total_data_points / data_summarizer.MAX_SAMPLES
            # Calculate step size for row sampling
            step = max(1, int(sample_ratio))
            # Sample rows evenly
            df_data = df_data.gather_every(n=step)
            logger.info(
                f"Downsampled from {num_rows} to {len(df_data)} rows to stay under {data_summarizer.MAX_SAMPLES} data points"
            )

        ts_plots = []
        all_cols = (
            data_summarizer.timeseries_cols + data_summarizer.continuous_cols
        )
        for col in all_cols:
            plot_data = (
                df_data.select([data_summarizer.timestamps_col, col])
                .drop_nulls()
                .sort(data_summarizer.timestamps_col)
            )
            pd_data = plot_data.to_pandas()
            if not pd.api.types.is_datetime64_any_dtype(
                pd_data[data_summarizer.timestamps_col]
            ):
                pd_data[data_summarizer.timestamps_col] = pd.to_datetime(
                    pd_data[data_summarizer.timestamps_col]
                )
            source = ColumnDataSource(
                data={
                    "x": pd_data[data_summarizer.timestamps_col],
                    "y": pd_data[col],
                }
            )
            p = figure(
                width=600,
                height=300,
                x_axis_type="datetime",
                title=f"Time Series: {col}",
                toolbar_location="above",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            p.line("x", "y", source=source, line_width=2)
            hover = HoverTool(
                tooltips=[("Time", "@x{%F %T}"), (col, "@y")],
                formatters={"@x": "datetime"},
                mode="vline",
            )
            p.add_tools(hover)
            p.xaxis.axis_label = data_summarizer.timestamps_col
            p.yaxis.axis_label = col
            p.grid.grid_line_alpha = 0.2
            ts_plots.append(p)
        if ts_plots:
            ts_grid = gridplot(
                ts_plots,
                ncols=2,
                sizing_mode="scale_width",
                merge_tools=False,
            )
            layout_elements.append(ts_grid)


def _add_distribution_plots(data_summarizer, layout_elements):
    """Add statistical distribution plots to the layout."""
    if not data_summarizer.skip_plots and data_summarizer.continuous_cols:
        dist_plots_div = Div(
            text="<h2 class='section-header'>Statistical Distribution Plots</h2>",
            width=800,
        )
        layout_elements.append(dist_plots_div)

        df_data = data_summarizer.df_data
        # Apply sampling if needed
        num_rows = len(df_data)
        num_cols = len(df_data.columns)
        total_data_points = num_rows * num_cols
        if total_data_points > data_summarizer.MAX_SAMPLES:
            sample_ratio = total_data_points / data_summarizer.MAX_SAMPLES
            step = max(1, int(sample_ratio))
            df_data = df_data.gather_every(n=step)

        dist_plots = []
        for col in data_summarizer.continuous_cols:
            col_data = df_data.get_column(col)
            pd_series = col_data.to_pandas()

            if pd.api.types.is_numeric_dtype(pd_series):
                p = figure(
                    width=600,
                    height=300,
                    title=f"Histogram: Distribution of {col} Values",
                    toolbar_location="above",
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                )

                # Filter outliers for better visualization
                mean = pd_series.mean()
                std = pd_series.std()
                filtered_data = pd_series[
                    (
                        pd_series
                        >= mean - data_summarizer.OUTLIER_STD_THRESHOLD * std
                    )
                    & (
                        pd_series
                        <= mean + data_summarizer.OUTLIER_STD_THRESHOLD * std
                    )
                ]

                hist, edges = np.histogram(
                    filtered_data, bins="auto", density=True
                )
                source = ColumnDataSource(
                    data={
                        "left": edges[:-1],
                        "right": edges[1:],
                        "top": hist,
                    }
                )
                p.quad(
                    top="top",
                    bottom=0,
                    left="left",
                    right="right",
                    source=source,
                    fill_alpha=0.75,
                    fill_color="steelblue",
                )
                hover = HoverTool(
                    tooltips=[
                        ("Bin", "@left to @right"),
                        ("Density", "@top"),
                    ]
                )
                p.add_tools(hover)
                p.xaxis.axis_label = col
                p.yaxis.axis_label = "Density"
                p.grid.grid_line_alpha = 0.2
                dist_plots.append(p)

        if dist_plots:
            dist_grid = gridplot(
                dist_plots,
                ncols=2,
                sizing_mode="scale_width",
                merge_tools=False,
            )
            layout_elements.append(dist_grid)


def _add_discrete_plots(data_summarizer, layout_elements):
    """Add discrete variable plots to the layout."""
    if not data_summarizer.skip_plots and data_summarizer.discrete_cols:
        disc_plots_div = Div(
            text="<h2 class='section-header'>Discrete Variable Plots</h2>",
            width=800,
        )
        layout_elements.append(disc_plots_div)

        df_data = data_summarizer.df_data
        # Apply sampling if needed
        num_rows = len(df_data)
        num_cols = len(df_data.columns)
        total_data_points = num_rows * num_cols
        if total_data_points > data_summarizer.MAX_SAMPLES:
            sample_ratio = total_data_points / data_summarizer.MAX_SAMPLES
            step = max(1, int(sample_ratio))
            df_data = df_data.gather_every(n=step)

        disc_plots = []
        for col in data_summarizer.discrete_cols:
            col_data = df_data.get_column(col)
            pd_series = col_data.to_pandas()

            p = figure(
                width=600,
                height=300,
                title=f"Bar Chart: Frequency Count of {col} Categories",
                toolbar_location="above",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )

            if pd.api.types.is_numeric_dtype(pd_series):
                # For numeric discrete variables, create histogram
                hist, edges = np.histogram(pd_series, bins="auto")
                source = ColumnDataSource(
                    data={
                        "left": edges[:-1],
                        "right": edges[1:],
                        "top": hist,
                    }
                )
                p.quad(
                    top="top",
                    bottom=0,
                    left="left",
                    right="right",
                    source=source,
                    fill_alpha=0.75,
                    fill_color="lightcoral",
                )
                hover = HoverTool(
                    tooltips=[
                        ("Bin", "@left to @right"),
                        ("Count", "@top"),
                    ]
                )
                p.add_tools(hover)
                p.xaxis.axis_label = col
                p.yaxis.axis_label = "Count"
                # Update title for numeric discrete variables
                p.title.text = (  # type: ignore
                    f"Histogram: Count Distribution of {col} Values"  # type: ignore
                )
            else:
                # For categorical variables, create bar chart
                pd_series = pd_series.astype(str)
                counts = pd_series.value_counts().head(
                    15
                )  # Show top 15 categories
                cats = list(counts.index)
                values = list(counts.values)
                source = ColumnDataSource(data={"cats": cats, "counts": values})
                p.vbar(
                    x="cats",
                    top="counts",
                    width=0.5,
                    source=source,
                    fill_color="lightgreen",
                    fill_alpha=0.7,
                )
                hover = HoverTool(
                    tooltips=[
                        ("Category", "@cats"),
                        ("Count", "@counts"),
                    ]
                )
                p.add_tools(hover)
                p.xaxis.major_label_orientation = 0.8
                p.xaxis.axis_label = col
                p.yaxis.axis_label = "Count"

            p.grid.grid_line_alpha = 0.2
            disc_plots.append(p)

        if disc_plots:
            disc_grid = gridplot(
                disc_plots,
                ncols=2,
                sizing_mode="scale_width",
                merge_tools=False,
            )
            layout_elements.append(disc_grid)


def _add_column_lists(data_summarizer, layout_elements):
    """Add column lists section to the layout."""
    continuous_cols_str = ", ".join(
        [f'"{col}"' for col in data_summarizer.continuous_cols]
    )
    discrete_cols_str = ", ".join(
        [f'"{col}"' for col in data_summarizer.discrete_cols]
    )

    # Column Lists Section
    column_lists_div = Div(
        text="<h2 class='section-header'>Column Lists for Easy Copying</h2>",
        width=800,
    )
    layout_elements.append(column_lists_div)

    continuous_cols_div = Div(
        text=f"<h3>Continuous Columns:</h3><pre>{continuous_cols_str}</pre>",
        width=800,
    )
    layout_elements.append(continuous_cols_div)

    discrete_cols_div = Div(
        text=f"<h3>Discrete Columns:</h3><pre>{discrete_cols_str}</pre>",
        width=800,
    )
    layout_elements.append(discrete_cols_div)
