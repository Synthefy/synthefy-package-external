"""
Driver script for fm-evals plotting functionality.

This module provides a high-level interface for creating various plots and analyses
from fm-evals results. It includes the standard analysis function and provides
convenient methods for generating different types of visualizations.

Usage:
    from synthefy_pkg.fm_evals.visualizations.plotter import Plotter

    # Create a plotter instance with dataframes
    plotter = Plotter(dataframes)

    # Or create with file paths
    plotter = Plotter.from_files(['/path/to/results1.h5', '/path/to/results2.csv'])

    # Generate standard analysis
    plotter.create_standard_analysis("/path/to/output")

    # Generate quick analysis
    quick_analysis(group_by='column_name', metric='mae')

    # Create specific plots
    plotter.create_grouped_bar_plot(group_by='column_name', metric='mae')
    plotter.create_error_distribution_plot(metric='mape')
    plotter.create_sample_analysis_plots(random_samples=5, best_samples=3)
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
from synthefy_pkg.fm_evals.visualizations.plotting_utils import (
    analyze_model_performance_by_group,
    create_aggregated_plots,
    create_error_distribution_plot,
    create_grouped_bar_plots,
    create_metric_grouped_bar_plots,
    create_sample_analysis_plots,
    print_model_performance_summary,
)


def _load_dataframes_from_files(file_paths: List[str]) -> List[pd.DataFrame]:
    all_dataframes = []

    for file_path in file_paths:
        try:
            print(f"Loading file: {file_path}")
            dataset_result = DatasetResultFormat.load_from_file(file_path)
            dataframes = dataset_result.to_dfs()
            all_dataframes.extend(dataframes)
            print(f"Loaded {len(dataframes)} dataframes from {file_path}")
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {str(e)}")
            continue

    if not all_dataframes:
        raise ValueError(
            "No dataframes could be loaded from the provided files"
        )

    print(f"Total dataframes loaded: {len(all_dataframes)}")
    return all_dataframes


class Plotter:
    """
    High-level interface for creating plots from fm-evals results.

    This class provides convenient methods for generating various types of plots
    and analyses from fm-evals dataframes.
    """

    def __init__(self, dataframes: List[pd.DataFrame]):
        """
        Initialize the Plotter with dataframes.

        Parameters
        ----------
        dataframes : List[pd.DataFrame]
            List of DataFrames from DatasetResultFormat.to_dfs()
        """
        if not dataframes:
            raise ValueError("dataframes list cannot be empty")
        self.dataframes = dataframes

    @classmethod
    def from_files(cls, file_paths: Union[str, List[str]]) -> "Plotter":
        """
        Create a Plotter instance from file paths.

        Parameters
        ----------
        file_paths : Union[str, List[str]]
            Single file path or list of file paths to load.
            Supported formats: .h5, .csv, .pkl

        Returns
        -------
        Plotter
            Plotter instance with loaded dataframes
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if not file_paths:
            raise ValueError("file_paths cannot be empty")

        dataframes = _load_dataframes_from_files(file_paths)
        return cls(dataframes)

    @classmethod
    def from_dataset_results(
        cls,
        dataset_results: Union[DatasetResultFormat, List[DatasetResultFormat]],
    ) -> "Plotter":
        """
        Create a Plotter instance from DatasetResultFormat objects.

        Parameters
        ----------
        dataset_results : Union[DatasetResultFormat, List[DatasetResultFormat]]
            Single DatasetResultFormat or list of DatasetResultFormat objects

        Returns
        -------
        Plotter
            Plotter instance with loaded dataframes
        """
        if isinstance(dataset_results, DatasetResultFormat):
            dataset_results = [dataset_results]

        if not dataset_results:
            raise ValueError("dataset_results cannot be empty")

        all_dataframes = []
        for dataset_result in dataset_results:
            dataframes = dataset_result.to_dfs()
            all_dataframes.extend(dataframes)

        if not all_dataframes:
            raise ValueError(
                "No dataframes could be extracted from the dataset results"
            )

        return cls(all_dataframes)

    def create_standard_analysis(
        self,
        output_dir: str,
        group_by_options: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Dict[str, str]:
        """
        Create a standard set of plots for fm-evals analysis.
        Can keep updating this to add all the common plots a customer might want to see.

        Parameters
        ----------
        output_dir : str
            Directory to save all plots
        group_by_options : Optional[List[str]], default None
            List of columns to group by. If None, uses common defaults.
        metrics : Optional[List[str]], default None
            List of metrics to analyze. If None, uses ['mae', 'mape'].
        figsize : Tuple[int, int], default (12, 8)
            Figure size for all plots

        Returns
        -------
        Dict[str, str]
            Dictionary mapping plot names to their file paths
        """
        # Set defaults
        if group_by_options is None:
            group_by_options = ["column_name", "model_name", "sample_id"]

        if metrics is None:
            metrics = ["mae", "mape"]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        plot_paths = {}

        # Generate plots for each metric and grouping option
        for metric in metrics:
            for group_by in group_by_options:
                # Skip if group_by is a list (handled separately)
                if isinstance(group_by, list):
                    continue

                # 1. Basic aggregated bar plot
                try:
                    fig = create_aggregated_plots(
                        self.dataframes,
                        group_by,
                        metric,
                        "mean",
                        "bar",
                        figsize=figsize,
                    )
                    path = os.path.join(
                        output_dir, f"{metric}_by_{group_by}_bar.png"
                    )
                    fig.savefig(path, dpi=300, bbox_inches="tight")  # type: ignore
                    plt.close(fig)
                    plot_paths[f"{metric}_by_{group_by}_bar"] = path
                except Exception as e:
                    print(
                        f"Warning: Could not create {metric}_by_{group_by}_bar plot: {e}"
                    )

                # 2. Box plot for distribution
                try:
                    fig = create_aggregated_plots(
                        self.dataframes,
                        group_by,
                        metric,
                        "mean",
                        "box",
                        figsize=figsize,
                    )
                    path = os.path.join(
                        output_dir, f"{metric}_by_{group_by}_box.png"
                    )
                    fig.savefig(path, dpi=300, bbox_inches="tight")  # type: ignore
                    plt.close(fig)
                    plot_paths[f"{metric}_by_{group_by}_box"] = path
                except Exception as e:
                    print(
                        f"Warning: Could not create {metric}_by_{group_by}_box plot: {e}"
                    )

        # Generate multi-level grouping plots
        for metric in metrics:
            # Model vs Column name
            try:
                fig = create_aggregated_plots(
                    self.dataframes,
                    ["column_name", "model_name"],
                    metric,
                    plot_type="bar",
                    figsize=figsize,
                )
                path = os.path.join(
                    output_dir, f"{metric}_by_column_model_bar.png"
                )
                fig.savefig(path, dpi=300, bbox_inches="tight")  # type: ignore
                plt.close(fig)
                plot_paths[f"{metric}_by_column_model_bar"] = path
            except Exception as e:
                print(
                    f"Warning: Could not create {metric}_by_column_model_bar plot: {e}"
                )

        # Generate histogram plots
        for metric in metrics:
            try:
                fig = create_aggregated_plots(
                    self.dataframes,
                    "model_name",
                    metric,
                    "mean",
                    "histogram",
                    figsize=figsize,
                )
                path = os.path.join(
                    output_dir, f"{metric}_distribution_histogram.png"
                )
                fig.savefig(path, dpi=300, bbox_inches="tight")  # type: ignore
                plt.close(fig)
                plot_paths[f"{metric}_distribution_histogram"] = path
            except Exception as e:
                print(
                    f"Warning: Could not create {metric}_distribution_histogram plot: {e}"
                )

        print(f"Generated {len(plot_paths)} plots in {output_dir}")
        return plot_paths

    def create_grouped_bar_plot(
        self,
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
        """
        return create_grouped_bar_plots(
            self.dataframes,
            group_by,
            metric,
            output_path,
            figsize,
            title,
            show_counts,
            **kwargs,
        )

    def analyze_model_performance(
        self,
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
            Dictionary containing analysis results and win count figure
        """
        return analyze_model_performance_by_group(
            self.dataframes, group_by, metric, output_path, figsize, title
        )

    def print_performance_summary(
        self, group_by: str, metric: str = "mae"
    ) -> None:
        """
        Print a summary of model performance analysis.

        Parameters
        ----------
        group_by : str
            Column name to group by
        metric : str, default "mae"
            Metric to analyze ('mae' or 'mape')
        """
        print_model_performance_summary(self.dataframes, group_by, metric)

    def create_metric_distribution_plot(
        self,
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
        """
        return create_metric_grouped_bar_plots(
            self.dataframes,
            metric,
            bins,
            output_path,
            figsize,
            title,
            show_counts,
            **kwargs,
        )

    def create_error_distribution_plot(
        self,
        metric: str = "mae",
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10),
        title: Optional[str] = None,
        **kwargs,
    ):
        """
        Create comprehensive error distribution plots showing what kinds of errors models are making.

        Parameters
        ----------
        metric : str, default "mae"
            Metric to analyze ('mae' or 'mape')
        output_path : Optional[str], default None
            Path to save the plot. If None, plot is not saved.
        figsize : Tuple[int, int], default (10, 10)
            Figure size (width, height)
        title : Optional[str], default None
            Plot title. If None, auto-generated.
        **kwargs
            Additional arguments passed to seaborn plotting functions

        Returns
        -------
        Figure
            The matplotlib figure object
        """
        return create_error_distribution_plot(
            self.dataframes, metric, output_path, figsize, title, **kwargs
        )

    def create_aggregated_plot(
        self,
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
        Create aggregated plots with flexible grouping and aggregation options.

        Parameters
        ----------
        group_by : Union[str, List[str]]
            Column name(s) to group by
        metric : str, default "mae"
            Metric to plot ('mae' or 'mape')
        aggregation : str, default "mean"
            Aggregation method: 'mean', 'sum', 'median', 'std', 'count', 'min', 'max'
        plot_type : str, default "bar"
            Type of plot: 'bar', 'line', 'box', 'histogram'
        output_path : Optional[str], default None
            Path to save the plot. If None, plot is not saved.
        figsize : Tuple[int, int], default (12, 8)
            Figure size (width, height)
        title : Optional[str], default None
            Plot title. If None, auto-generated.
        show_counts : bool, default False
            Whether to show count of samples in each group
        additional_filters : Optional[Dict[str, Any]], default None
            Additional filters to apply to the data before aggregation.
        **kwargs
            Additional arguments passed to seaborn plotting functions

        Returns
        -------
        Figure
            The matplotlib figure object
        """
        return create_aggregated_plots(
            self.dataframes,
            group_by,
            metric,
            aggregation,
            plot_type,
            output_path,
            figsize,
            title,
            show_counts,
            additional_filters,
            **kwargs,
        )

    def create_sample_analysis_plots(
        self,
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

        Parameters
        ----------
        metric : str, default "mae"
            Metric to use for sample selection
        random_samples : int, default 0
            Number of random samples to include
        median_samples : int, default 0
            Number of median-performing samples to include
        best_samples : int, default 0
            Number of best-performing samples to include
        worst_samples : int, default 0
            Number of worst-performing samples to include
        output_path : Optional[str], default None
            Path to save the plot. If None, plot is not saved.
        figsize : Tuple[int, int], default (15, 10)
            Figure size (width, height)
        seed : int, default 42
            Random seed for reproducible random sample selection
        **kwargs
            Additional arguments passed to plotting functions

        Returns
        -------
        Figure
            The matplotlib figure object
        """
        return create_sample_analysis_plots(
            self.dataframes,
            metric,
            random_samples,
            median_samples,
            best_samples,
            worst_samples,
            output_path,
            figsize,
            seed,
            **kwargs,
        )

    def quick_analysis(
        self,
        output_dir: str,
        group_by: str = "column_name",
        metric: str = "mae",
    ) -> Dict[str, str]:
        """
        Quick analysis method for generating basic plots.
        This is a convenience method that generates the most commonly used plots.

        Parameters
        ----------
        output_dir : str
            Directory to save all plots
        group_by : str, default "column_name"
            Column name to group by
        metric : str, default "mae"
            Metric to analyze ('mae' or 'mape')

        Returns
        -------
        Dict[str, str]
            Dictionary mapping plot names to their file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        plot_paths = {}

        # Generate basic plots
        try:
            # Grouped bar plot
            fig = self.create_grouped_bar_plot(group_by, metric)
            path = os.path.join(output_dir, f"{metric}_by_{group_by}_bar.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")  # type: ignore
            plt.close(fig)
            plot_paths[f"{metric}_by_{group_by}_bar"] = path
        except Exception as e:
            print(
                f"Warning: Could not create {metric}_by_{group_by}_bar plot: {e}"
            )

        try:
            # Error distribution plot
            fig = self.create_error_distribution_plot(metric)
            path = os.path.join(output_dir, f"{metric}_error_distribution.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")  # type: ignore
            plt.close(fig)
            plot_paths[f"{metric}_error_distribution"] = path
        except Exception as e:
            print(
                f"Warning: Could not create {metric}_error_distribution plot: {e}"
            )

        try:
            # Model performance analysis
            analysis = self.analyze_model_performance(group_by, metric)
            if analysis["win_count_figure"]:
                path = os.path.join(
                    output_dir, f"{metric}_model_win_counts.png"
                )
                analysis["win_count_figure"].savefig(
                    path, dpi=300, bbox_inches="tight"
                )  # type: ignore
                plt.close(analysis["win_count_figure"])
                plot_paths[f"{metric}_model_win_counts"] = path
        except Exception as e:
            print(
                f"Warning: Could not create {metric}_model_win_counts plot: {e}"
            )

        print(f"Generated {len(plot_paths)} plots in {output_dir}")
        return plot_paths


# some convenience functions:


def quick_analysis(
    dataframes: List[pd.DataFrame],
    output_dir: str,
    group_by: str = "column_name",
    metric: str = "mae",
) -> Dict[str, str]:
    plotter = Plotter(dataframes)
    return plotter.quick_analysis(output_dir, group_by, metric)


def quick_analysis_from_files(
    file_paths: Union[str, List[str]],
    output_dir: str,
    group_by: str = "column_name",
    metric: str = "mae",
) -> Dict[str, str]:
    plotter = Plotter.from_files(file_paths)
    return plotter.quick_analysis(output_dir, group_by, metric)


def create_standard_analysis(
    dataframes: List[pd.DataFrame],
    output_dir: str,
    group_by_options: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Dict[str, str]:
    plotter = Plotter(dataframes)
    return plotter.create_standard_analysis(
        output_dir, group_by_options, metrics, figsize
    )


def create_standard_analysis_from_files(
    file_paths: Union[str, List[str]],
    output_dir: str,
    group_by_options: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Dict[str, str]:
    plotter = Plotter.from_files(file_paths)
    return plotter.create_standard_analysis(
        output_dir, group_by_options, metrics, figsize
    )
