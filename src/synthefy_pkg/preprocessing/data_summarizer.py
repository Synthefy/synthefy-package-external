import gc
import platform
import signal
import threading
import time
from collections import OrderedDict
from concurrent.futures import TimeoutError
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

# Import data summary utilities
from synthefy_pkg.preprocessing.data_summary_utils.create_plots import (
    create_autocorrelation_plots,
    create_convergent_cross_mapping_matrix_plot,
    create_cross_correlation_matrix_plot,
    create_decomposition_plots,
    create_dlinear_matrix_plot,
    create_granger_causality_matrix_plot,
    create_mutual_information_matrix_plot,
    create_stacked_time_series_visualizations,
    create_transfer_entropy_matrix_plot,
)
from synthefy_pkg.preprocessing.data_summary_utils.generate_summary_html import (
    generate_html_report,
)
from synthefy_pkg.preprocessing.data_summary_utils.get_summary_dict import (
    get_summary_dict,
)
from synthefy_pkg.preprocessing.data_summary_utils.load_and_categorize import (
    categorize_columns,
    load_data,
    setup_data_sources_and_config,
    setup_timestamp_and_group_columns,
)
from synthefy_pkg.preprocessing.data_summary_utils.perform_analysis import (
    perform_autocorrelation_analysis,
    perform_basic_statistics_analysis,
    perform_convergent_cross_mapping_analysis,
    perform_correlation_analysis,
    perform_cross_correlation_analysis,
    perform_decomposition_analysis,
    perform_dlinear_analysis,
    perform_granger_causality_analysis,
    perform_mutual_information_analysis,
    perform_outlier_analysis,
    perform_quantile_analysis,
    perform_transfer_entropy_analysis,
    perform_ts_features_analysis,
)
from synthefy_pkg.preprocessing.data_summary_utils.summarize_metadata import (
    summarize_metadata,
    summarize_time_series,
)

COMPILE = False


class DataSummarizer:
    def __init__(
        self,
        data_input: Union[pl.DataFrame, str, List[Union[pl.DataFrame, str]]],
        save_path: str,
        config: Optional[Dict[str, Any]] = None,
        group_cols: Optional[List[str]] = None,
        skip_plots: bool = False,
        test_df: Optional[pl.DataFrame] = None,
        compute_all: bool = True,
        analysis_functions: Optional[List[str]] = None,
        execute_forecast: bool = False,
    ):
        """
        Initialize the DataSummarizer with required and optional inputs.

        Args:
            data_input: Path to dataset file relative to SYNTHEFY_DATASETS_BASE, absolute path, a Polars DataFrame, or a list of these
            save_path: Path where plots and reports will be saved
            config: Optional configuration dictionary
            group_cols: Optional list of column names to use for grouping. Takes precedence over config group_labels if both are provided.
            skip_plots: Optional boolean to skip generating plots in the report. Defaults to False.
            test_df: Optional test DataFrame for forecasting analysis
            compute_all: Optional boolean to compute analysis on all columns (True) or only timeseries columns (False). Defaults to True.
            analysis_functions: Optional list of analysis function names to run. If None, runs all analyses.
                               Available functions: 'basic_statistics', 'ts_features', 'correlation', 'outlier',
                               'quantile', 'autocorrelation', 'decomposition', 'cross_correlation',
                               'transfer_entropy', 'granger_causality', 'convergent_cross_mapping', 'mutual_information'
        """
        # Setup constants
        self.MAX_SAMPLES = 1000000
        self.OUTLIER_STD_THRESHOLD = 3.0
        self.TOP_VALUES_COUNT = 3
        self.PLOT_TIMEOUT = 240  # 4 minutes in seconds
        self.skip_plots = skip_plots
        self.test_df = test_df
        self.compute_all = compute_all
        self.analysis_functions = analysis_functions
        self.execute_forecast = execute_forecast
        self.save_path = save_path
        self.dataset_name = ""  # should be set in metadata fn
        # Setup data sources and configuration
        processed_inputs, self.data_sources, self.config = (
            setup_data_sources_and_config(data_input, config, group_cols)
        )

        # Load data
        self.df_data = load_data(processed_inputs)

        # Initialize column attributes
        self.continuous_cols: List[str] = []
        self.discrete_cols: List[str] = []
        self.group_cols: List[str] = []
        self.timeseries_cols: List[str] = []
        self.timestamps_col: Optional[str] = None

        # Setup timestamp and group columns
        setup_timestamp_and_group_columns(self, self.config, group_cols)

        # Categorize remaining columns
        categorize_columns(self)

        # Initialize attributes that will be populated by different functions
        # Attributes set by summarize_metadata():
        self.metadata_df = None
        self.num_columns_by_type = None
        self.sample_counts = None
        self.time_range = None

        # Attributes set by summarize_time_series():
        self.time_series_df = None
        self.skip_plots = skip_plots
        self.test_df = test_df
        self.decomposition_plots = []
        self.cross_corr_matrix_plot = None
        self.transfer_entropy_matrix_plot = None
        self.granger_causality_matrix_plot = None
        self.convergent_cross_mapping_matrix_plot = None
        self.mutual_info_matrix_plot = None
        self.autocorr_plots = []

        # Attributes set by perform_analysis functions:
        # - perform_basic_statistics_analysis():
        self.basic_stats_df = None
        # - perform_ts_features_analysis():
        self.ts_features_df = None
        # - perform_correlation_analysis():
        self.correlation_df = None
        # - perform_outlier_analysis():
        self.outlier_df = None
        # - perform_quantile_analysis():
        self.quantile_df = None
        # - perform_autocorrelation_analysis():
        self.autocorr_df = None
        # - perform_decomposition_analysis():
        self.decomposition_df = None
        self.decomposition_results = None
        # - perform_cross_correlation_analysis():
        self.cross_corr_df = None
        # - perform_transfer_entropy_analysis():
        self.transfer_entropy_df = None
        # - perform_granger_causality_analysis():
        self.granger_causality_df = None
        # - perform_dlinear_analysis():
        self.dlinear_df = None
        # - perform_convergent_cross_mapping_analysis():
        self.convergent_cross_mapping_df = None
        # - perform_mutual_information_analysis():
        self.mutual_info_df = None

        # Perform comprehensive analysis
        self.perform_comprehensive_analysis(
            execute_forecast=execute_forecast,
            output_dir=save_path,
            analysis_functions=analysis_functions,
        )

        self.settings = type(
            "Settings",
            (),
            {"json_save_path": save_path},
        )()

    def summarize_metadata(self):
        """
        Summarize metadata including number of columns, types, ranges, and units.
        """
        summarize_metadata(self)

    def summarize_time_series(self):
        """
        Summarize time series information for each KPI.
        """
        summarize_time_series(self)

    def cleanup(self):
        """Clean up memory by deleting all class attributes and clearing plots"""
        # Clear all plots
        plt.close("all")

        # Save summary attributes we want to keep
        summary_attrs = {
            "time_series_df": getattr(self, "time_series_df", None),
            "time_range": getattr(self, "time_range", None),
            "metadata_df": getattr(self, "metadata_df", None),
            "num_columns_by_type": getattr(self, "num_columns_by_type", None),
            "sample_counts": getattr(self, "sample_counts", None),
            "settings": getattr(self, "settings", None),
            "basic_stats_df": getattr(self, "basic_stats_df", None),
            "correlation_df": getattr(self, "correlation_df", None),
            "outlier_df": getattr(self, "outlier_df", None),
            "quantile_df": getattr(self, "quantile_df", None),
            "autocorr_df": getattr(self, "autocorr_df", None),
            "decomposition_df": getattr(self, "decomposition_df", None),
            "cross_corr_df": getattr(self, "cross_corr_df", None),
            "transfer_entropy_df": getattr(self, "transfer_entropy_df", None),
            "granger_causality_df": getattr(self, "granger_causality_df", None),
            "convergent_cross_mapping_df": getattr(
                self, "convergent_cross_mapping_df", None
            ),
            "ts_features_df": getattr(self, "ts_features_df", None),
            "decomposition_plots": getattr(self, "decomposition_plots", []),
            "cross_corr_matrix_plot": getattr(
                self, "cross_corr_matrix_plot", None
            ),
            "transfer_entropy_matrix_plot": getattr(
                self, "transfer_entropy_matrix_plot", None
            ),
            "granger_causality_matrix_plot": getattr(
                self, "granger_causality_matrix_plot", None
            ),
            "convergent_cross_mapping_matrix_plot": getattr(
                self, "convergent_cross_mapping_matrix_plot", None
            ),
            "autocorr_plots": getattr(self, "autocorr_plots", []),
            "compute_all": getattr(self, "compute_all", True),
            "analysis_functions": getattr(self, "analysis_functions", None),
            "timeseries_cols": getattr(self, "timeseries_cols", []),
            "timestamps_col": getattr(self, "timestamps_col", None),
            "group_cols": getattr(self, "group_cols", []),
            "continuous_cols": getattr(self, "continuous_cols", []),
            "discrete_cols": getattr(self, "discrete_cols", []),
        }

        for attr in list(self.__dict__.keys()):
            delattr(self, attr)

        for attr, value in summary_attrs.items():
            setattr(self, attr, value)

        gc.collect()

    def _timeout_handler(self, signum, frame):
        """Signal handler for timeout"""
        raise TimeoutError("Plotting operation timed out")

    def _plot_with_timeout(self, plot_func, *args, **kwargs):
        """Execute plotting function with timeout"""
        if platform.system() != "Windows":
            # Unix-based systems can use signal
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.PLOT_TIMEOUT)
            result = plot_func(*args, **kwargs)
            signal.alarm(0)  # Disable alarm
            return result
        else:
            # Windows systems use threading
            result = [None]

            def target():
                result[0] = plot_func(*args, **kwargs)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=self.PLOT_TIMEOUT)

            if thread.is_alive():
                plt.close()  # Clean up any partial plots
                logger.warning(
                    f"Plotting timed out after {self.PLOT_TIMEOUT} seconds"
                )
                return None

            return result[0]

    def generate_html_report(
        self,
        output_html: str = "data_summary_report.html",
    ):
        """
        Generate an interactive HTML report with metadata, time series summaries, tables,
        and interactive plots using the Bokeh module.
        """
        return generate_html_report(self, output_html)

    def perform_comprehensive_analysis(
        self,
        compute_all: Optional[bool] = None,
        analysis_functions: Optional[List[str]] = None,
        execute_forecast: bool = False,
        output_dir: Optional[str] = None,
    ):
        """Perform comprehensive analyses based on specified functions."""
        # Use instance variable if not provided
        if compute_all is None:
            compute_all = self.compute_all
        if analysis_functions is None:
            analysis_functions = self.analysis_functions

        logger.info("Starting comprehensive data analysis...")
        start_time = time.time()

        # Create stacked time series visualizations first

        # Define available analysis functions
        available_functions = {
            "stacked_plots": (
                "Starting stacked time series plots",
                lambda: create_stacked_time_series_visualizations(
                    self, output_dir=output_dir
                ),
            ),
            "basic_statistics": (
                "Starting basic statistics analysis...",
                lambda: perform_basic_statistics_analysis(self),
            ),
            "ts_features": (
                "Starting time series features analysis...",
                lambda: perform_ts_features_analysis(self),
            ),
            "correlation": (
                "Starting correlation analysis...",
                lambda: perform_correlation_analysis(self),
            ),
            "outlier": (
                "Starting outlier analysis...",
                lambda: perform_outlier_analysis(self),
            ),
            "quantile": (
                "Starting quantile analysis...",
                lambda: perform_quantile_analysis(self),
            ),
            "autocorrelation": (
                "Starting autocorrelation analysis...",
                lambda: perform_autocorrelation_analysis(self),
            ),
            "decomposition": (
                "Starting decomposition analysis...",
                lambda: perform_decomposition_analysis(
                    self,
                    compute_all=compute_all,
                    execute_forecast=execute_forecast,
                ),
            ),
            "cross_correlation": (
                "Starting cross correlation analysis...",
                lambda: perform_cross_correlation_analysis(
                    self, compute_all=compute_all
                ),
            ),
            "transfer_entropy": (
                "Starting transfer entropy analysis...",
                lambda: perform_transfer_entropy_analysis(
                    self, compute_all=compute_all
                ),
            ),
            "granger_causality": (
                "Starting Granger causality analysis...",
                lambda: perform_granger_causality_analysis(
                    self, compute_all=compute_all
                ),
            ),
            "dlinear": (
                "Starting DLinear causality analysis...",
                lambda: perform_dlinear_analysis(self, compute_all=compute_all),
            ),
            "convergent_cross_mapping": (
                "Starting convergent cross-mapping analysis...",
                lambda: perform_convergent_cross_mapping_analysis(
                    self, compute_all=compute_all
                ),
            ),
            "mutual_information": (
                "Starting mutual information analysis...",
                lambda: perform_mutual_information_analysis(
                    self, compute_all=compute_all
                ),
            ),
        }

        # Determine which functions to run
        if analysis_functions is None:
            functions_to_run = available_functions
            logger.info("Running all analysis functions")
        else:
            functions_to_run = {
                k: v
                for k, v in available_functions.items()
                if k in analysis_functions
            }
            logger.info(
                f"Running selected analysis functions: {list(functions_to_run.keys())}"
            )

        # Perform selected analyses
        for func_name, (log_message, func) in functions_to_run.items():
            analysis_start = time.time()
            logger.info(log_message)
            func()
            logger.info(
                f"{func_name.replace('_', ' ').title()} analysis completed in {time.time() - analysis_start:.2f} seconds"
            )

        # Create visualizations with error handling
        viz_start = time.time()
        logger.info("Starting visualization creation...")

        # Define available plot functions
        available_plots = {
            "decomposition": (
                "Creating decomposition plots...",
                lambda: create_decomposition_plots(self),
            ),
            "cross_correlation": (
                "Creating cross correlation matrix plot...",
                lambda: create_cross_correlation_matrix_plot(self),
            ),
            "transfer_entropy": (
                "Creating transfer entropy matrix plot...",
                lambda: create_transfer_entropy_matrix_plot(self),
            ),
            "granger_causality": (
                "Creating Granger causality matrix plot...",
                lambda: create_granger_causality_matrix_plot(self),
            ),
            "dlinear": (
                "Creating DLinear causality matrix plot...",
                lambda: create_dlinear_matrix_plot(self),
            ),
            "convergent_cross_mapping": (
                "Creating convergent cross-mapping matrix plot...",
                lambda: create_convergent_cross_mapping_matrix_plot(self),
            ),
            "mutual_information": (
                "Creating mutual information matrix plot...",
                lambda: create_mutual_information_matrix_plot(self),
            ),
            "autocorrelation": (
                "Creating autocorrelation plots...",
                lambda: create_autocorrelation_plots(self),
            ),
        }

        # Only create plots for analyses that were run
        plots_to_create = {}
        if analysis_functions is None:
            plots_to_create = available_plots
        else:
            plots_to_create = {
                k: v
                for k, v in available_plots.items()
                if k in analysis_functions
            }

        # Create selected plots
        for plot_name, (log_message, plot_func) in plots_to_create.items():
            plot_start = time.time()
            logger.info(log_message)
            plot_func()
            logger.info(
                f"{plot_name.replace('_', ' ').title()} plots completed in {time.time() - plot_start:.2f} seconds"
            )

        logger.info(
            f"All visualizations completed in {time.time() - viz_start:.2f} seconds"
        )
        logger.info(
            f"Comprehensive analysis completed in {time.time() - start_time:.2f} seconds total"
        )

    def _ensure_analysis_complete(self):
        """Ensure that comprehensive analysis has been completed."""
        if not hasattr(self, "basic_stats_df") or self.basic_stats_df is None:
            logger.info(
                "Comprehensive analysis not yet completed. Running it now..."
            )
            self.perform_comprehensive_analysis(output_dir=self.save_path)
        else:
            logger.info("Comprehensive analysis already completed.")

    def get_summary_dict(self):
        """
        Returns a JSON-serializable dictionary containing all summary data.
        """
        return get_summary_dict(self)
