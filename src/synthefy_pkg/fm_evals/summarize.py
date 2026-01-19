#!/usr/bin/env python3
"""
Data Summarization Script

This script provides data summarization functionality using DataSummarizer,
similar to the eval.py script but focused on generating comprehensive data summaries
instead of running forecasting models.
"""

import argparse
import copy
import json
import os
import random
import sys
import tempfile
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import polars as pl
import yaml
from loguru import logger

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.eval import (
    get_dataloader,
    get_effective_dataset_name,
    get_supported_datasets,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.preprocessing.data_summarizer import DataSummarizer
from synthefy_pkg.preprocessing.data_summary_utils.populate_summary_from_dict import (
    populate_data_summarizer_from_dict,
)


# Local utility functions
def limit_dataset_samples(
    data_df: pd.DataFrame, max_samples: int
) -> pd.DataFrame:
    """Limit the dataset to a maximum number of samples.

    Args:
        data_df: Input DataFrame
        max_samples: Maximum number of samples to keep

    Returns:
        Limited DataFrame with max_samples rows
    """
    if max_samples and len(data_df) > max_samples:
        original_size = len(data_df)
        logger.info(
            f"Limiting dataset from {original_size:,} to {max_samples:,} samples ({max_samples / original_size * 100:.1f}% of original)"
        )
        # Take a representative sample (first max_samples rows)
        limited_df = data_df.head(max_samples)
        logger.info(
            f"Sample limiting applied: {len(limited_df):,} samples retained"
        )
        return limited_df
    elif max_samples and len(data_df) <= max_samples:
        logger.info(
            f"Dataset already within limit: {len(data_df):,} samples (max: {max_samples:,})"
        )
        return data_df
    return data_df


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_keys = ["timeseries", "discrete", "continuous"]
    return all(key in config for key in required_keys)


def ensure_directory_exists(directory: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)


def make_unique_column_names(columns) -> List[str]:
    """Make column names unique by adding numeric suffixes to duplicates.

    Args:
        columns: Column names (Index or List) that may contain duplicates

    Returns:
        List of unique column names with numeric suffixes for duplicates
    """
    seen = {}
    unique_columns = []

    for col in columns:
        if col in seen:
            seen[col] += 1
            unique_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_columns.append(col)

    return unique_columns


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and NaN values properly."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)


class ComprehensiveSummaryGenerator:
    """Generates comprehensive summary statistics across all datasets."""

    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.all_dataset_results = {}
        self.failed_datasets = {}  # Track failed datasets and their errors
        self.summary_statistics = {
            "total_datasets": 0,
            "total_data_points": 0,
            "total_rows": 0,
            "total_columns": 0,
            "dataset_breakdown": {},
            "correlation_summary": {},
            "time_series_summary": {},
            "metadata_summary": {},
        }

    def add_dataset_summary(
        self,
        dataset_name: str,
        summary_dict: Dict[str, Any],
        max_samples_applied: Optional[int] = None,
    ) -> None:
        """Add a single dataset summary to the comprehensive results."""
        # Extract key metrics
        metadata = summary_dict.get("metadata", {})
        time_series_summary = summary_dict.get("time_series_summary", [])
        discrete_summary = summary_dict.get("discrete_summary", [])
        continuous_summary = summary_dict.get("continuous_summary", [])
        correlation_matrix = summary_dict.get("correlation_matrix", [])

        # Calculate dataset-specific statistics
        dataset_stats = {
            "data_points": len(time_series_summary)
            if time_series_summary
            else 0,
            "rows": metadata.get("total_rows", 0),
            "columns": metadata.get("total_columns", 0),
            "time_series_columns": len(time_series_summary),
            "discrete_columns": len(discrete_summary),
            "continuous_columns": len(continuous_summary),
            "correlation_pairs": len(correlation_matrix)
            if correlation_matrix
            else 0,
            "decomposition_plots": summary_dict.get("decomposition_plots", 0),
            "autocorrelation_plots": summary_dict.get(
                "autocorrelation_plots", 0
            ),
            "cross_correlation_matrix": bool(
                summary_dict.get("cross_correlation_matrix_plot")
            ),
            "timestamp": datetime.now().isoformat(),
        }

        # Add sample limiting information if applicable
        if max_samples_applied:
            dataset_stats["sample_limiting_applied"] = True
            dataset_stats["max_samples_limit"] = max_samples_applied
        else:
            dataset_stats["sample_limiting_applied"] = False
            dataset_stats["max_samples_limit"] = None

        # Store dataset results
        self.all_dataset_results[dataset_name] = {
            "summary": summary_dict,
            "statistics": dataset_stats,
        }

        # Update comprehensive statistics
        self.summary_statistics["total_datasets"] += 1
        self.summary_statistics["total_data_points"] += dataset_stats[
            "data_points"
        ]
        self.summary_statistics["total_rows"] += dataset_stats["rows"]
        self.summary_statistics["total_columns"] += dataset_stats["columns"]

        # Store dataset breakdown
        self.summary_statistics["dataset_breakdown"][dataset_name] = (
            dataset_stats
        )

        # Update correlation summary
        if correlation_matrix:
            for corr_data in correlation_matrix:
                if isinstance(corr_data, dict) and "correlation" in corr_data:
                    corr_value = corr_data["correlation"]
                    if isinstance(corr_value, (int, float)):
                        if "correlation_summary" not in self.summary_statistics:
                            self.summary_statistics["correlation_summary"] = {}
                        if (
                            "total_correlations"
                            not in self.summary_statistics[
                                "correlation_summary"
                            ]
                        ):
                            self.summary_statistics["correlation_summary"][
                                "total_correlations"
                            ] = 0
                        self.summary_statistics["correlation_summary"][
                            "total_correlations"
                        ] += 1

        # Update time series summary
        if time_series_summary:
            if (
                "total_time_series"
                not in self.summary_statistics["time_series_summary"]
            ):
                self.summary_statistics["time_series_summary"][
                    "total_time_series"
                ] = 0
            self.summary_statistics["time_series_summary"][
                "total_time_series"
            ] += len(time_series_summary)

        # Update metadata summary
        if metadata:
            if (
                "total_metadata_entries"
                not in self.summary_statistics["metadata_summary"]
            ):
                self.summary_statistics["metadata_summary"][
                    "total_metadata_entries"
                ] = 0
            self.summary_statistics["metadata_summary"][
                "total_metadata_entries"
            ] += len(metadata)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate the final comprehensive report."""
        # Calculate additional aggregate statistics
        self.summary_statistics["average_data_points_per_dataset"] = (
            self.summary_statistics["total_data_points"]
            / max(self.summary_statistics["total_datasets"], 1)
        )
        self.summary_statistics["average_rows_per_dataset"] = (
            self.summary_statistics["total_rows"]
            / max(self.summary_statistics["total_datasets"], 1)
        )
        self.summary_statistics["average_columns_per_dataset"] = (
            self.summary_statistics["total_columns"]
            / max(self.summary_statistics["total_datasets"], 1)
        )

        # Add sample limiting summary
        datasets_with_limiting = sum(
            1
            for stats in self.summary_statistics["dataset_breakdown"].values()
            if stats.get("sample_limiting_applied", False)
        )
        self.summary_statistics["sample_limiting_summary"] = {
            "datasets_with_limiting": datasets_with_limiting,
            "total_datasets": self.summary_statistics["total_datasets"],
            "percentage_limited": (
                datasets_with_limiting
                / max(self.summary_statistics["total_datasets"], 1)
            )
            * 100,
        }

        # Add failed datasets summary
        failed_summary = self.get_failed_datasets_summary()
        self.summary_statistics["failed_datasets_summary"] = failed_summary

        # Generate timestamp
        self.summary_statistics["generated_at"] = datetime.now().isoformat()

        return {
            "comprehensive_summary": self.summary_statistics,
            "all_dataset_results": self.all_dataset_results,
            "failed_datasets": self.failed_datasets,
        }

    def save_comprehensive_report(self, run_id: str) -> str:
        """Save the comprehensive report to JSON file."""
        report = self.generate_comprehensive_report()

        # Create comprehensive results directory
        comprehensive_dir = os.path.join(
            self.output_directory, "comprehensive_results"
        )
        os.makedirs(comprehensive_dir, exist_ok=True)

        # Save comprehensive report
        report_file = os.path.join(
            comprehensive_dir, f"all_datasets_summary_{run_id}.json"
        )
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Saved comprehensive report to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save comprehensive report: {e}")
            raise

        return report_file

    def add_custom_metric(
        self, metric_name: str, value: Any, dataset_name: Optional[str] = None
    ) -> None:
        """Add a custom metric to the summary statistics.

        Args:
            metric_name: Name of the custom metric
            value: Value of the metric
            dataset_name: If provided, adds metric to specific dataset. If None, adds to global summary.
        """
        if dataset_name:
            if dataset_name in self.all_dataset_results:
                if (
                    "custom_metrics"
                    not in self.all_dataset_results[dataset_name]
                ):
                    self.all_dataset_results[dataset_name][
                        "custom_metrics"
                    ] = {}
                self.all_dataset_results[dataset_name]["custom_metrics"][
                    metric_name
                ] = value
            else:
                logger.warning(
                    f"Dataset {dataset_name} not found, cannot add custom metric"
                )
        else:
            # Add to global summary
            if "custom_metrics" not in self.summary_statistics:
                self.summary_statistics["custom_metrics"] = {}
            self.summary_statistics["custom_metrics"][metric_name] = value

    def add_custom_dataset_metric(
        self, dataset_name: str, metric_name: str, value: Any
    ) -> None:
        """Add a custom metric to a specific dataset.

        Args:
            dataset_name: Name of the dataset
            metric_name: Name of the custom metric
            value: Value of the metric
        """
        self.add_custom_metric(metric_name, value, dataset_name)

    def add_custom_global_metric(self, metric_name: str, value: Any) -> None:
        """Add a custom metric to the global summary.

        Args:
            metric_name: Name of the custom metric
            value: Value of the metric
        """
        self.add_custom_metric(metric_name, value)

    def get_dataset_summary(
        self, dataset_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get summary for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset summary if found, None otherwise
        """
        return self.all_dataset_results.get(dataset_name)

    def get_global_summary(self) -> Dict[str, Any]:
        """Get the global summary statistics.

        Returns:
            Global summary statistics
        """
        return self.summary_statistics.copy()

    def export_summary_to_csv(self, run_id: str) -> str:
        """Export dataset breakdown to CSV format for easy analysis.

        Args:
            run_id: Run identifier

        Returns:
            Path to the exported CSV file
        """
        import pandas as pd

        # Create comprehensive results directory
        comprehensive_dir = os.path.join(
            self.output_directory, "comprehensive_results"
        )
        os.makedirs(comprehensive_dir, exist_ok=True)

        # Convert dataset breakdown to DataFrame
        dataset_data = []
        for dataset_name, stats in self.summary_statistics[
            "dataset_breakdown"
        ].items():
            row = {"dataset_name": dataset_name}
            row.update(stats)
            dataset_data.append(row)

        if dataset_data:
            df = pd.DataFrame(dataset_data)

            # Export to CSV
            csv_file = os.path.join(
                comprehensive_dir, f"dataset_breakdown_{run_id}.csv"
            )
            df.to_csv(csv_file, index=False)
            logger.info(f"Dataset breakdown exported to CSV: {csv_file}")
            return csv_file
        else:
            logger.warning("No dataset data available for CSV export")
            return ""

    def record_failed_dataset(
        self, dataset_name: str, error: str, error_type: str = "unknown"
    ) -> None:
        """Record a failed dataset processing attempt.

        Args:
            dataset_name: Name of the dataset that failed
            error: Error message
            error_type: Type of error (e.g., 'data_type_compatibility', 'processing_error')
        """
        self.failed_datasets[dataset_name] = {
            "error": error,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
        }

    def get_failed_datasets_summary(self) -> Dict[str, Any]:
        """Get a summary of failed datasets.

        Returns:
            Dictionary containing failed datasets information
        """
        return {
            "total_failed": len(self.failed_datasets),
            "failed_datasets": self.failed_datasets.copy(),
        }


def generate_unique_id() -> str:
    """Generate a Docker-style unique ID with timestamp suffix."""
    # Generate a short UUID (first 8 characters)
    short_uuid = str(uuid.uuid4())[:8]

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{short_uuid}_{timestamp}"


def generate_expected_json_path(
    args: argparse.Namespace, effective_dataset_name: str, run_id: str
) -> str:
    """Generate the expected JSON file path based on dataset name and output directory structure.

    Args:
        args: Parsed command line arguments
        effective_dataset_name: Effective dataset name
        run_id: Run ID

    Returns:
        Expected path to the JSON file
    """
    # Determine base output directory based on dataset type
    if args.dataset == "gift":
        base_output_dir = os.path.join(
            args.output_directory, args.dataset, args.sub_dataset
        )
    elif args.dataset == "gpt-synthetic":
        dataset_subdir = args.dataset_name if args.dataset_name else ""
        base_output_dir = os.path.join(
            args.output_directory, args.dataset, dataset_subdir
        )
    else:
        base_output_dir = os.path.join(args.output_directory, args.dataset)

    # Create run-specific directory
    run_output_dir = os.path.join(base_output_dir, run_id)

    # Generate dataset basename and JSON filename
    dataset_basename = effective_dataset_name.replace("/", "_")
    json_filename = f"{dataset_basename}_summary_data.json"

    return os.path.join(run_output_dir, json_filename)


def download_json_from_s3(dataset_name: str, local_path: str) -> str:
    """Download existing JSON file from S3 for a dataset.

    Args:
        dataset_name: Name of the dataset
        local_path: Local path to save the downloaded file

    Returns:
        Path to the downloaded file
    """
    # Construct S3 path based on the structure used in download_and_aggregate.py
    if dataset_name.find("gpt-synthetic") != -1:
        s3_path = f"s3://synthefy-fm-eval-datasets/analysis/{'-'.join(dataset_name.split('/'))}"
    else:
        s3_path = f"s3://synthefy-fm-eval-datasets/analysis/{dataset_name}"

    logger.info(f"Searching for JSON files in {s3_path}")
    json_files = list_s3_files(s3_path, file_extension=".json")

    if not json_files:
        raise ValueError(
            f"No JSON files found for dataset {dataset_name} in {s3_path}"
        )

    # Find the summary data file (should end with _summary_data.json)
    summary_file = None
    for file_url in json_files:
        if file_url.endswith("_summary_data.json"):
            summary_file = file_url
            break

    if not summary_file:
        # If no specific summary file found, use the first JSON file
        summary_file = json_files[0]
        logger.warning(f"No _summary_data.json found, using {summary_file}")

    # Parse S3 URL to get bucket and key
    parsed = urlparse(summary_file)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    # Download the file
    s3_client = boto3.client("s3")
    logger.info(f"Downloading {summary_file} to {local_path}")
    s3_client.download_file(bucket, key, local_path)
    logger.info(f"Successfully downloaded {os.path.basename(local_path)}")

    return local_path


def modify_existing_json(
    existing_json_path: str,
    new_analysis_results: Dict[str, Any],
    analysis_functions: List[str],
    overwrite: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """Modify an existing JSON file with new analysis results.

    Args:
        existing_json_path: Path to the existing JSON file
        new_analysis_results: New analysis results from DataSummarizer
        analysis_functions: List of analysis functions that were run
        overwrite: Whether to overwrite the existing file or create a new one
        output_path: Optional output path for modified file (if not overwriting)

    Returns:
        Path to the modified JSON file
    """
    # Safety check: prevent overwriting when output_path is specified
    if output_path and overwrite:
        raise ValueError(
            "Cannot specify both overwrite=True and output_path. Use either overwrite or custom output path."
        )

    # Safety check: ensure either overwrite or output_path is specified
    if not overwrite and not output_path:
        raise ValueError(
            "Must specify either overwrite=True or output_path when modifying existing JSON."
        )

    # Safety check: prevent overwriting the original file when using custom output path
    if output_path and not overwrite:
        # Resolve paths to absolute paths for comparison
        abs_existing_path = os.path.abspath(existing_json_path)
        abs_output_path = os.path.abspath(output_path)

        if abs_existing_path == abs_output_path:
            raise ValueError(
                f"Cannot use output_path '{output_path}' as it would overwrite the original file '{existing_json_path}'. Use --overwrite-json if you want to overwrite the original file."
            )

    logger.info(f"Loading existing JSON from: {existing_json_path}")

    # Load existing JSON
    with open(existing_json_path, "r") as f:
        existing_data = json.load(f)

    logger.info(f"Loaded existing JSON with keys: {list(existing_data.keys())}")

    # Map analysis functions to their corresponding keys in the summary dict
    analysis_function_mapping = {
        "basic_statistics": "basic_statistics",
        "ts_features": "ts_features",
        "correlation": "correlation_matrix",
        "outlier": "outlier_analysis",
        "quantile": "quantile_analysis",
        "autocorrelation": "autocorrelation_analysis",
        "decomposition": "decomposition_analysis",
        "cross_correlation": "cross_correlation_analysis",
        "transfer_entropy": "transfer_entropy_analysis",
        "granger_causality": "granger_causality_analysis",
        "convergent_cross_mapping": "convergent_cross_mapping_analysis",
        "mutual_information": "mutual_information_analysis",
        "dlinear": "dlinear_analysis",
    }

    # Update the existing data with new analysis results
    updated_count = 0
    for analysis_func in analysis_functions:
        if analysis_func in analysis_function_mapping:
            key = analysis_function_mapping[analysis_func]
            if key in new_analysis_results:
                existing_data[key] = new_analysis_results[key]
                updated_count += 1
                logger.info(f"Updated {key} with new {analysis_func} results")
            else:
                logger.warning(
                    f"Key {key} not found in new analysis results for {analysis_func}"
                )
        else:
            logger.warning(f"Unknown analysis function: {analysis_func}")

    # Update metadata to reflect the modification
    if "metadata" in existing_data:
        existing_data["metadata"]["last_modified"] = datetime.now().isoformat()
        existing_data["metadata"]["modified_analysis_functions"] = (
            analysis_functions
        )
        existing_data["metadata"]["modification_count"] = (
            existing_data["metadata"].get("modification_count", 0) + 1
        )

    logger.info(f"Updated {updated_count} analysis sections")

    # Determine output path
    if overwrite:
        final_output_path = existing_json_path
        logger.info(f"Overwriting existing file: {final_output_path}")
    else:
        if output_path:
            final_output_path = output_path
            logger.info(
                f"Creating new file at specified location: {final_output_path}"
            )
        else:
            # Create a new filename with modification timestamp
            base_name = os.path.splitext(existing_json_path)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_path = f"{base_name}_modified_{timestamp}.json"
            logger.info(
                f"Creating new file with timestamp: {final_output_path}"
            )

    # Ensure output directory exists if using custom path
    if output_path and not overwrite:
        output_dir = os.path.dirname(final_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        # Safety check: warn if output file already exists (but don't prevent it)
        if os.path.exists(final_output_path):
            logger.warning(
                f"Output file already exists and will be overwritten: {final_output_path}"
            )
            logger.warning(
                "If this is not intended, please specify a different output path."
            )

    # Save the modified JSON
    with open(final_output_path, "w") as f:
        json.dump(existing_data, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Successfully saved modified JSON to: {final_output_path}")
    return final_output_path


def load_summary_config(config_path: str) -> Dict[str, Any]:
    """Load summary configuration from yaml file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file: {e}")
        sys.exit(1)


def save_summary_run_info(
    args: argparse.Namespace,
    run_id: str,
    output_directory: str,
    dataset_name: str,
    summary_dict: Dict[str, Any],
) -> None:
    """Save summary run information to a YAML file for reproducibility."""
    run_info = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "parameters": {},
    }

    # Convert args to dictionary, filtering out None values
    for key, value in vars(args).items():
        if value is not None:
            run_info["parameters"][key] = value

    # Add summary statistics if available
    if summary_dict:
        run_info["summary_stats"] = {
            "total_rows": summary_dict.get("metadata", {}).get(
                "total_rows", "N/A"
            ),
            "total_columns": summary_dict.get("metadata", {}).get(
                "total_columns", "N/A"
            ),
            "time_series_columns": len(
                summary_dict.get("time_series_summary") or []
            ),
            "discrete_columns": len(summary_dict.get("discrete_summary") or []),
            "continuous_columns": len(
                summary_dict.get("continuous_summary") or []
            ),
        }
    else:
        run_info["summary_stats"] = None

    # Save to the output directory
    run_info_file = os.path.join(
        output_directory, f"summary_run_info_{run_id}.yaml"
    )
    try:
        with open(run_info_file, "w") as file:
            yaml.dump(run_info, file, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved summary run info to: {run_info_file}")
    except Exception as e:
        logger.error(f"Failed to save summary run info: {e}")


def validate_summary_arguments(args: argparse.Namespace) -> None:
    """Validate arguments for data summarization."""
    errors = []

    # Validate JSON modification mode arguments
    if args.modify_existing_json:
        if not args.analysis_functions:
            errors.append(
                "--analysis-functions is required when using --modify-existing-json"
            )
        if not args.dataset and not args.input_file:
            errors.append(
                "Either --dataset or --input-file is required when using --modify-existing-json"
            )
        if not args.output_directory:
            errors.append(
                "--output-directory is required when using --modify-existing-json"
            )

        # Cannot use modify_existing_json with dataset ALL
        if args.dataset and args.dataset.upper() == "ALL":
            errors.append(
                "Cannot use --modify-existing-json with --dataset ALL"
            )

        # Validate output-json-path argument
        if args.output_json_path and args.overwrite_json:
            errors.append(
                "Cannot use --output-json-path with --overwrite-json (overwrite mode ignores custom output path)"
            )

        # Ensure either output-json-path or overwrite-json is specified when modifying existing JSON
        if not args.output_json_path and not args.overwrite_json:
            errors.append(
                "When using --modify-existing-json, you must specify either --output-json-path or --overwrite-json"
            )

    # If --dataset ALL is used, only validate output directory
    if args.dataset and args.dataset.upper() == "ALL":
        if not args.output_directory:
            errors.append(
                "--output-directory is required when using --dataset ALL"
            )
        if args.input_file:
            errors.append(
                "Cannot use --dataset ALL with --input-file. Use one or the other."
            )
        return

    # Basic argument validation
    if not args.summary_config:
        if not args.dataset and not args.input_file:
            errors.append(
                "Either --summary-config, --dataset, or --input-file must be provided"
            )
        if not args.output_directory:
            errors.append(
                "Either --summary-config or --output-directory must be provided"
            )

    # Dataset validation
    if args.dataset and args.dataset.upper() != "ALL":
        supported_datasets = get_supported_datasets()
        if args.dataset not in supported_datasets:
            errors.append(
                f"Unsupported dataset name: {args.dataset}. Supported datasets: {supported_datasets}"
            )

    # File path validation
    if args.config_file and not os.path.exists(args.config_file):
        errors.append(f"Config file not found: {args.config_file}")

    if args.input_file and not os.path.exists(args.input_file):
        errors.append(f"Input file not found: {args.input_file}")

    if args.output_directory:
        # Check if output directory can be created
        try:
            os.makedirs(args.output_directory, exist_ok=True)
        except (OSError, PermissionError) as e:
            errors.append(
                f"Cannot create output directory {args.output_directory}: {e}"
            )

    # Report errors and exit if any
    if errors:
        logger.error("Argument validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)


def parse_summary_arguments() -> argparse.Namespace:
    """Parse command line arguments for the data summarization script."""
    parser = argparse.ArgumentParser(
        description="Data Summarization Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use summary config file
  python summarize.py --summary-config configs/summary_config.yaml
  
  # Use individual arguments
  python summarize.py --dataset fmv3 --config-file configs/fmv3_config.yaml --output-directory results/
  
  # With custom grouping columns
  python summarize.py --dataset gift --data-path /path/to/data --sub-dataset traffic --output-directory results/ --group-cols "App Name,User Type"
  
  # Direct file input (bypasses dataloader system)
  python summarize.py --input-file /path/to/data.csv --output-directory results/ --group-cols "Category,Region"
  
  # Process all supported datasets
  python summarize.py --dataset ALL --output-directory results/
  
  # Process with limited samples (useful for testing)
  python summarize.py --dataset traffic --output-directory results/ --max-samples 1000
  
  # Process all datasets with limited samples
  python summarize.py --dataset ALL --output-directory results/ --max-samples 500
  
  # Modify existing JSON file and overwrite it (automatically finds the file)
  python summarize.py --modify-existing-json --dataset traffic --output-directory results/ --analysis-functions "correlation,outlier" --overwrite-json
  
  # Download JSON from S3 and overwrite it
  python summarize.py --modify-existing-json --download-from-s3 --dataset traffic --output-directory results/ --analysis-functions "autocorrelation,decomposition" --overwrite-json
  
  # Modify JSON with specific analysis functions and save to custom location
  python summarize.py --modify-existing-json --input-file /path/to/data.csv --output-directory results/ --analysis-functions "basic_statistics,ts_features" --output-json-path /path/to/custom_location.json
  
  # Modify JSON and save to specific location (safe - won't overwrite original)
  python summarize.py --modify-existing-json --dataset traffic --output-directory results/ --analysis-functions "correlation" --output-json-path /path/to/custom_location.json
        """,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to process per dataset (optional, useful for testing or partial processing)",
    )
    parser.add_argument(
        "--summary-config",
        type=str,
        help="Path to summary configuration YAML file (alternative to individual arguments)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to summarize (uses dataloader system). Use 'ALL' to process all supported datasets.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name (required for gpt-synthetic dataset) options: energy, manufacturing, retail, supply_chain, traffic",
        default="",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Direct path to input file (CSV, Parquet, etc.) - bypasses dataloader system",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file (required for fmv3 dataset)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data location (required for gift dataset)",
    )
    parser.add_argument(
        "--sub-dataset",
        type=str,
        help="Sub-dataset name (required for gift dataset)",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        help="Directory to save summary results and reports",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        help="Comma-separated list of column names for grouping",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=False,
        help="Skip generating plots in the summary (default: False)",
    )
    parser.add_argument(
        "--analysis-functions",
        type=str,
        default=None,
        help="Comma-separated list of analysis functions to run. Available functions: basic_statistics, ts_features, correlation, outlier, quantile, autocorrelation, decomposition, cross_correlation, transfer_entropy, granger_causality, convergent_cross_mapping, mutual_information. If not specified, runs all analyses.",
    )
    parser.add_argument(
        "--external-dataloader-spec",
        type=str,
        help="External dataloader specification in format 'path::class_name' (required for external dataset)",
    )
    parser.add_argument(
        "--compute-all",
        action="store_true",
        default=False,
        help="Compute analysis on all columns (timeseries + continuous). If not specified, only analyzes timeseries columns (default: False)",
    )
    parser.add_argument(
        "--random-ordering",
        action="store_true",
        default=False,
        help="Randomly order the data (default: False)",
    )
    parser.add_argument(
        "--mixed-datasets",
        type=str,
        nargs="+",
        help="List of dataset names for mixed domain dataloader (required for mixed_domain dataset)",
    )
    parser.add_argument(
        "--replace-metadata",
        action="store_true",
        default=False,
        help="Replace metadata with random time series (default: False)",
    )
    parser.add_argument(
        "--use-other-metadata",
        action="store_true",
        default=False,
        help="Use metadata from different domain datasets (default: False)",
    )
    parser.add_argument(
        "--random-ts-sampling",
        type=str,
        default="mixed_simple",
        help="Type of random time series sampling for metadata (default: mixed_simple)",
    )
    parser.add_argument(
        "--execute-forecast",
        action="store_true",
        default=False,
        help="Execute forecast (default: False)",
    )
    parser.add_argument(
        "--modify-existing-json",
        action="store_true",
        default=False,
        help="Modify existing JSON file with new analysis results (automatically determines path from dataset name). Requires either --output-json-path or --overwrite-json.",
    )
    parser.add_argument(
        "--download-from-s3",
        action="store_true",
        default=False,
        help="Download existing JSON from S3 instead of using local file",
    )
    parser.add_argument(
        "--overwrite-json",
        action="store_true",
        default=False,
        help="Overwrite the existing JSON file instead of creating a new one",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        help="Target path for the modified JSON file (only used when not overwriting)",
    )

    args = parser.parse_args()

    if args.summary_config:
        logger.info(
            f"Loading summary configuration from: {args.summary_config}"
        )
        config = load_summary_config(args.summary_config)
        for key, value in config.items():
            attr_name = key
            if hasattr(args, attr_name):
                setattr(args, attr_name, value)
            else:
                logger.warning(f"Unknown config key: {key}")

    validate_summary_arguments(args)
    return args


def get_summary_dataloader(args: argparse.Namespace):
    """Get the appropriate dataloader for summarization based on the dataset."""
    # Use the existing dataloader from eval.py
    dataloader = get_dataloader(args)

    # For gift dataset, set default values for summarization
    if args.dataset == "gift":
        # Override with default values suitable for summarization
        from synthefy_pkg.fm_evals.dataloading.gift_eval_dataloader import (
            GIFTEvalUnivariateDataloader,
        )

        return GIFTEvalUnivariateDataloader(
            data_dir=args.data_path,
            forecast_length=12,  # Default value for summarization
            history_length=24,  # Default value for summarization
            random_ordering=args.random_ordering,
        )

    return dataloader


def split_long_sequence(sample, batch_count):
    """
    Split a long sequence into non-overlapping windows of 1152 length.
    Each window has history=1024 and forecast=128.

    Args:
        sample: SingleEvalSample object
        batch_count: Current batch count for ID generation

    Returns:
        List of sample data dictionaries
    """
    # Concatenate history and target timestamps and values
    all_timestamps = np.concatenate(
        [sample.history_timestamps, sample.target_timestamps]
    )
    all_values = np.concatenate([sample.history_values, sample.target_values])

    total_length = len(all_timestamps)
    window_size = 1152  # Total window size
    history_size = 1024  # History portion

    split_samples = []

    # Create non-overlapping windows
    for start_idx in range(0, total_length, window_size):
        end_idx = min(start_idx + window_size, total_length)

        # Skip if we don't have enough data for a full window
        if end_idx - start_idx < window_size:
            break

        # Extract window data
        window_timestamps = all_timestamps[start_idx:end_idx]
        window_values = all_values[start_idx:end_idx]

        # Split into history and target
        history_timestamps = window_timestamps[:history_size]
        history_values = window_values[:history_size]
        target_timestamps = window_timestamps[history_size:]
        target_values = window_values[history_size:]

        # Create sample data
        batch_id_sample_id = f"b{batch_count}_" + sample.column_name
        sample_data = {
            "sample_id": batch_id_sample_id,
            "history_timestamps": history_timestamps,
            "history_values": history_values,
            "target_timestamps": target_timestamps,
            "target_values": target_values,
        }
        split_samples.append(sample_data)
        batch_count += 1

    return split_samples


def extract_data_from_dataloader(
    dataloader, max_samples: Optional[int] = None
) -> tuple[pd.DataFrame, List[str], List[str], List[str], List[str], str]:
    """Extract data from dataloader and convert to pandas DataFrame."""
    all_data = []
    dataloader_batch_count = 0  # Counter for dataloader batches (for logging)
    sample_id_counter = 0  # Counter for sample IDs (for data processing)
    total_samples = 0

    # Iterate through all batches in the dataloader
    for batch in dataloader:
        if batch is None:
            logger.warning(
                f"Batch {dataloader_batch_count} returned None, skipping"
            )
            dataloader_batch_count += 1
            continue

        logger.info(f"Processing batch {dataloader_batch_count}")

        # Check if we've reached the max_samples limit
        if max_samples and total_samples >= max_samples:
            logger.info(
                f"Reached max_samples limit ({max_samples}), stopping extraction"
            )
            break

        target_cols = batch.target_cols  # should be the same for all samples
        metadata_cols = (
            batch.metadata_cols
        )  # should be the same for all samples
        leak_cols = batch.leak_cols  # should be the same for all samples
        sample_id_cols = (
            batch.sample_id_cols
        )  # should be the same for all samples
        timestamps_col = (
            batch.timestamp_col
        )  # should be the same for all samples

        # Try to extract data from EvalBatchFormat
        if hasattr(batch, "samples") and batch.samples:
            # Convert EvalBatchFormat to pandas DataFrame
            for sample_list in batch.samples:
                for sample in sample_list:
                    # Check if sequence length exceeds 1152 and split if necessary
                    total_length = len(sample.history_values) + len(
                        sample.target_values
                    )
                    total_batches_separated = 0
                    if total_length > 1152:
                        # Split the sequence into non-overlapping windows
                        split_samples = split_long_sequence(
                            sample, sample_id_counter
                        )
                        for split_sample in split_samples:
                            all_data.append(split_sample)
                            total_batches_separated += 1
                    else:
                        # Create a row with sample data (original logic)
                        batch_id_sample_id = (
                            f"b{sample_id_counter}_" + sample.column_name
                        )
                        sample_data = {
                            "sample_id": batch_id_sample_id,
                            "history_timestamps": sample.history_timestamps,
                            "history_values": sample.history_values,
                            "target_timestamps": sample.target_timestamps,
                            "target_values": sample.target_values,
                        }
                        all_data.append(sample_data)
                        total_batches_separated += 1
                sample_id_counter += total_batches_separated
                total_samples += 1

                # Break outer loop if we've reached max_samples
                if max_samples and total_samples >= max_samples:
                    break
        else:
            logger.warning(
                f"Could not extract data from batch {dataloader_batch_count}, skipping"
            )
            dataloader_batch_count += 1
            continue

        # Increment dataloader batch counter after processing each batch
        dataloader_batch_count += 1

    if not all_data:
        raise ValueError("No data could be extracted from dataloader")

    # If max_samples is specified, sample complete batches instead of individual samples
    if max_samples:
        num_unique_batch_ids = len(
            set(
                [
                    sample_data["sample_id"].split("_")[0]
                    for sample_data in all_data
                ]
            )
        )
        if num_unique_batch_ids > max_samples:
            # Group samples by batch_id (extract batch number from sample_id)
            batch_groups = {}
            for sample_data in all_data:
                sample_id = sample_data["sample_id"]
                # Extract batch_id from sample_id (format: "b{batch_count}_{column_name}")
                batch_id = sample_id.split("_")[0]  # Gets "b{batch_count}"
                if batch_id not in batch_groups:
                    batch_groups[batch_id] = []
                batch_groups[batch_id].append(sample_data)

            # Randomly sample batch_ids
            available_batch_ids = list(batch_groups.keys())
            num_batches_to_sample = min(max_samples, len(available_batch_ids))
            sampled_batch_ids = random.sample(
                available_batch_ids, num_batches_to_sample
            )

            # Collect all samples from the sampled batches
            sampled_data = []
            for batch_id in sampled_batch_ids:
                sampled_data.extend(batch_groups[batch_id])

            all_data = sampled_data
            logger.info(
                f"Sampled {len(sampled_batch_ids)} batches containing {len(all_data)} total samples"
            )

    # Convert to DataFrame
    if isinstance(all_data[0], pd.DataFrame):
        # If we have DataFrames, concatenate them
        data_df = pd.concat(all_data, ignore_index=True)
    else:
        # If we have dictionaries, create DataFrame from them
        # Convert each dictionary to a DataFrame and concatenate to preserve order
        data_dfs = [pd.DataFrame([data]) for data in all_data]
        data_df = pd.concat(data_dfs, ignore_index=True)

    # Ensure all columns are uniquely named and are Series
    def normalize_dataframe(df):
        """Ensure all columns are uniquely named and are Series."""
        # Make column names unique
        if not df.columns.is_unique:
            df.columns = make_unique_column_names(df.columns)

        # Ensure all columns are Series
        for col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        return df

    data_df = normalize_dataframe(data_df)

    # Calculate number of batches from the DataFrame rows
    if "sample_id" in data_df.columns:
        # Extract unique batch IDs from sample_id column
        batch_ids = data_df["sample_id"].str.extract(r"(b\d+)_")[0].unique()
        calculated_batch_count = len(batch_ids)
    else:
        # Fallback to the sample_id_counter if sample_id column doesn't exist
        calculated_batch_count = sample_id_counter

    logger.info(
        f"Extracted {len(data_df)} total samples from {calculated_batch_count} batches, processed {dataloader_batch_count} dataloader batches"
    )
    print(data_df)

    # # Apply sample limit if specified (in case we didn't limit during extraction)
    # if max_samples and len(data_df) > max_samples:
    #     data_df = limit_dataset_samples(data_df, max_samples)
    #     logger.info(f"Limited dataset to {len(data_df)} samples")

    return (
        data_df,
        target_cols,
        metadata_cols,
        leak_cols,
        sample_id_cols,
        timestamps_col,
    )


def convert_nested_list_data_to_dataframe(
    data_df,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert nested list data format to a flattened DataFrame compatible with DataSummarizer.

    This function handles data where each row contains lists of values, such as:
    - sample_id: list[str] - contains identifiers
    - history_timestamps: list[datetime] - contains time series timestamps
    - history_values: list[float] - contains time series values
    - target_timestamps: list[datetime] - contains target timestamps
    - target_values: list[float] - contains target values

    Args:
        data_df: DataFrame with nested list columns

    Returns:
        tuple of (history_df, target_df, combined_df)
        - history_df: DataFrame with history data
        - target_df: DataFrame with target data
        - combined_df: DataFrame with all data combined
    """
    logger.info("Converting nested list data format to flattened DataFrame...")

    # Check if this is the expected nested list format
    expected_columns = [
        "sample_id",
        "history_timestamps",
        "history_values",
        "target_timestamps",
        "target_values",
    ]
    if not all(col in data_df.columns for col in expected_columns):
        logger.warning(
            "Data doesn't match expected nested list format, returning original DataFrame"
        )
        return data_df

    # Verify that the columns contain lists

    logger.info(f"Detected nested list format with columns: {data_df.columns}")

    history_df, target_df, combined_df = (
        EvalBatchFormat.reconstruct_wide_format(data_df)
    )

    # Ensure all columns are Series by flattening DataFrame columns
    def ensure_series_columns(df):
        """Convert DataFrame columns to Series by taking first column."""
        for col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                logger.error(
                    f"Converting DataFrame column {col} to Series with shape {df[col].shape}, which loses data"
                )
                series_col = copy.deepcopy(df[col].iloc[:, -1])
                del df[col]
                df[col] = series_col
        return df

    history_df = ensure_series_columns(history_df)
    target_df = ensure_series_columns(target_df)
    combined_df = ensure_series_columns(combined_df)
    logger.info(
        f"History dataframe columns: {history_df.columns} timestamps: {history_df['timestamp'].head()}"
    )
    if len(history_df.columns) > 100:
        logger.warning(
            f"History dataframe has {len(history_df.columns)} columns, which is too many"
        )
        raise ValueError(
            f"History dataframe has {len(history_df.columns)} columns, which is too many"
        )
    return history_df, target_df, combined_df


def auto_adjust_group_columns(
    data_df, requested_group_cols: Optional[List[str]]
) -> List[str]:
    """Auto-adjust group columns based on what's actually available in the dataset.

    Args:
        data_df: The dataset DataFrame (pandas or polars)
        requested_group_cols: List of requested group columns

    Returns:
        List of group columns that actually exist in the dataset
    """
    if not requested_group_cols:
        return []

    available_columns = set(data_df.columns)
    adjusted_group_cols = []
    missing_cols = []

    for col in requested_group_cols:
        if col in available_columns:
            adjusted_group_cols.append(col)
        else:
            missing_cols.append(col)

    if missing_cols:
        logger.warning(f"Group columns not found in dataset: {missing_cols}")
        logger.info(f"Using available group columns: {adjusted_group_cols}")

    return adjusted_group_cols


def create_group_columns_for_flattened_data(
    flattened_df: pd.DataFrame,
) -> List[str]:
    """Create appropriate group columns for flattened nested list data.

    Args:
        flattened_df: Flattened DataFrame from convert_nested_list_data_to_dataframe

    Returns:
        List of column names that can be used for grouping
    """
    available_columns = list(flattened_df.columns)

    # Priority order for group columns
    group_candidates = ["sample_id", "series_type", "point_index"]

    # Find available group columns
    group_cols = []
    for col in group_candidates:
        if col in available_columns:
            group_cols.append(col)

    # If we have sample_id, that's usually the best grouping column
    if "sample_id" in group_cols:
        logger.info(
            "Using 'sample_id' as primary group column for time series analysis"
        )

    return group_cols


def create_config_for_flattened_data(
    flattened_df: pd.DataFrame,
    timestamps_col: Optional[str] = None,
    target_cols: List[str] = [],
    metadata_cols: List[str] = [],
    leak_cols: List[str] = [],
    sample_id_cols: List[str] = [],
    history_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Create a configuration file specifically for flattened nested list data.

    Args:
        flattened_df: Flattened DataFrame from convert_nested_list_data_to_dataframe

    Returns:
        Configuration dictionary optimized for flattened data
    """
    config = {}

    # Set timestamp column
    if timestamps_col is not None:
        if history_df is not None and timestamps_col not in history_df.columns:
            config["timestamps_col"] = "timestamp"
        else:
            config["timestamps_col"] = timestamps_col
    elif "timestamp" in flattened_df.columns:
        config["timestamps_col"] = "timestamp"

    if target_cols:
        config["timeseries"] = {"cols": target_cols}

    # alternatively define columns from config
    # if metadata_cols:
    #     config["metadata_cols"] = {"cols": metadata_cols}

    # if leak_cols:
    #     config["leak_cols"] = {"cols": leak_cols}

    # if sample_id_cols:
    #     config["sample_id_cols"] = {"cols": sample_id_cols}

    # Set group columns
    group_cols = create_group_columns_for_flattened_data(flattened_df)
    if group_cols:
        config["group_labels"] = {"cols": group_cols}

    # Set timeseries columns
    if "value" in flattened_df.columns:
        config["timeseries"] = {"cols": ["value"]}

    # Set continuous columns (numeric columns)
    numeric_cols = [
        col
        for col in flattened_df.columns
        if (flattened_df[col].dtype in ["int64", "float64", "int32", "float32"])
    ]
    if numeric_cols:
        config["continuous"] = {"cols": numeric_cols}

    # Set discrete columns (non-numeric, non-timestamp columns)
    discrete_cols = [
        col
        for col in flattened_df.columns
        if col not in numeric_cols and col != "timestamp" and col != "value"
    ]
    if discrete_cols:
        config["discrete"] = {"cols": discrete_cols}

    logger.info("Created configuration for flattened data:")
    logger.info(f"  - Timestamp column: {config.get('timestamps_col', 'None')}")
    logger.info(
        f"  - Group columns: {config.get('group_labels', {}).get('cols', [])}"
    )
    logger.info(
        f"  - Timeseries columns: {config.get('timeseries', {}).get('cols', [])}"
    )
    logger.info(
        f"  - Continuous columns: {config.get('continuous', {}).get('cols', [])}"
    )
    logger.info(
        f"  - Discrete columns: {config.get('discrete', {}).get('cols', [])}"
    )

    return config


def summarize_data_from_file(
    input_file: str,
    output_dir: str,
    config: Optional[Dict] = None,
    group_cols: Optional[List[str]] = None,
    skip_plots: bool = False,
    max_samples: Optional[int] = None,
    train_test_split: Optional[float] = None,
    compute_all: bool = False,
    analysis_functions: Optional[List[str]] = None,
    execute_forecast: bool = False,
) -> Dict[str, Any]:
    """Summarize data directly from a file using DataSummarizer."""
    logger.info(f"Loading data directly from file: {input_file}")

    # Load data and apply sample limit if specified
    if max_samples:
        # Load data first to check size
        if input_file.endswith(".csv"):
            data_df = pd.read_csv(input_file)
        elif input_file.endswith(".parquet"):
            data_df = pd.read_parquet(input_file)
        else:
            # Try pandas auto-detection
            data_df = pd.read_csv(input_file)

        # Apply sample limit
        data_df = limit_dataset_samples(data_df, max_samples)

        # Convert pandas DataFrame to polars DataFrame for DataSummarizer
        import polars as pl

        polars_df = pl.from_pandas(data_df)

        # Initialize DataSummarizer with DataFrame instead of file path
        summarizer = DataSummarizer(
            data_input=polars_df,
            save_path=output_dir,
            config=config,
            group_cols=group_cols,
            skip_plots=skip_plots,
            compute_all=compute_all,
            analysis_functions=analysis_functions,
            execute_forecast=execute_forecast,
        )
    else:
        # Initialize DataSummarizer with file path (original behavior)
        summarizer = DataSummarizer(
            data_input=input_file,
            save_path=output_dir,
            config=config,
            group_cols=group_cols,
            skip_plots=skip_plots,
            compute_all=compute_all,
            analysis_functions=analysis_functions,
            execute_forecast=execute_forecast,
        )

    # Generate summaries
    logger.info("Generating metadata summary...")
    summarizer.summarize_metadata()

    logger.info("Generating time series summary...")
    summarizer.summarize_time_series()

    # Ensure comprehensive analysis is completed (this generates all plots)
    logger.info("Performing comprehensive analysis to generate all plots...")
    summarizer.perform_comprehensive_analysis(
        execute_forecast=execute_forecast, output_dir=output_dir
    )

    # Generate HTML report with all plots
    file_basename = os.path.splitext(os.path.basename(input_file))[0]
    html_path = os.path.join(output_dir, f"{file_basename}_data_summary.html")

    logger.info(f"Generating HTML report with all plots at {html_path}...")
    summarizer.generate_html_report(output_html=html_path)

    # Get summary data
    summary_dict = summarizer.get_summary_dict()
    logger.info("Summary generated successfully.")

    # Log plot generation summary
    logger.info("Plot generation summary:")
    logger.info(
        f"  - Decomposition plots: {summary_dict.get('decomposition_plots', 0)}"
    )
    logger.info(
        f"  - Autocorrelation plots: {summary_dict.get('autocorrelation_plots', 0)}"
    )
    logger.info(
        f"  - Cross-correlation matrix: {'Generated' if summary_dict.get('cross_correlation_matrix_plot') else 'Not Generated'}"
    )
    logger.info(
        f"  - Time series plots: {'Generated' if not skip_plots and summary_dict.get('time_series_summary') else 'Skipped'}"
    )
    logger.info(
        f"  - Distribution plots: {'Generated' if not skip_plots and summary_dict.get('metadata_summary') else 'Skipped'}"
    )

    return summary_dict


def summarize_data_from_dataloader(
    args: argparse.Namespace,
    run_output_dir: str,
    effective_dataset_name: str,
    train_test_split: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize data using dataloader system."""
    logger.info("Using dataloader system for data summarization")

    # Get dataloader and extract data for summarization
    dataloader = get_summary_dataloader(args)

    # Load configuration for DataSummarizer if provided
    config = None
    if args.config_file:
        try:
            with open(args.config_file, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded config from: {args.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")

    # Process group_cols into a list if provided
    processed_group_cols = None
    if args.group_cols:
        processed_group_cols = args.group_cols.split(",")
        logger.info(f"Using group columns: {processed_group_cols}")

    # Extract data from dataloader
    (
        data_df,
        target_cols,
        metadata_cols,
        leak_cols,
        sample_id_cols,
        timestamps_col,
    ) = extract_data_from_dataloader(dataloader, args.max_samples)

    # Check if data is in nested list format and convert if needed
    logger.info("Checking data format compatibility...")
    if hasattr(data_df, "columns"):
        # Check if this looks like nested list data
        expected_columns = [
            "sample_id",
            "history_timestamps",
            "history_values",
            "target_timestamps",
            "target_values",
        ]
        if all(col in data_df.columns for col in expected_columns):
            logger.info("=" * 60)
            logger.info("NESTED LIST DATA DETECTED")
            logger.info("=" * 60)
            logger.info(
                "Your data is in nested list format where each row contains:"
            )
            logger.info("  - sample_id: List of identifiers")
            logger.info(
                "  - history_timestamps: List of timestamps for historical data"
            )
            logger.info(
                "  - history_values: List of values for historical data"
            )
            logger.info(
                "  - target_timestamps: List of timestamps for target data"
            )
            logger.info("  - target_values: List of values for target data")
            logger.info("")
            logger.info("Converting to flattened format for analysis...")
            logger.info("=" * 60)

            history_df, target_df, combined_df = (
                convert_nested_list_data_to_dataframe(data_df)
            )
            data_df = history_df
            test_df = target_df

            # If no group columns were specified, create appropriate ones for flattened data
            if not processed_group_cols:
                logger.info(
                    "No group columns specified, creating appropriate ones for flattened data..."
                )
                processed_group_cols = create_group_columns_for_flattened_data(
                    data_df
                )
                logger.info(
                    f"Auto-created group columns: {processed_group_cols}"
                )
            else:
                # Validate that user-specified group columns exist in the flattened data
                available_columns = set(data_df.columns)
                missing_group_cols = [
                    col
                    for col in processed_group_cols
                    if col not in available_columns
                ]
                if missing_group_cols:
                    logger.warning(
                        f"User-specified group columns not found in flattened data: {missing_group_cols}"
                    )
                    logger.warning(
                        f"Available columns: {list(available_columns)}"
                    )
                    logger.info(
                        "Falling back to auto-generated group columns..."
                    )
                    processed_group_cols = (
                        create_group_columns_for_flattened_data(data_df)
                    )
                    logger.info(
                        f"Using auto-generated group columns: {processed_group_cols}"
                    )
                else:
                    logger.info(
                        f"User-specified group columns are valid: {processed_group_cols}"
                    )

            # If no config was provided, create one optimized for flattened data
            if not config:
                logger.info(
                    "No configuration provided, creating one optimized for flattened data..."
                )
                config = create_config_for_flattened_data(
                    data_df,
                    timestamps_col,
                    target_cols,
                    metadata_cols,
                    leak_cols,
                    sample_id_cols,
                    history_df,
                )
            else:
                # Override the config file's group columns to be compatible with flattened data
                logger.info(
                    "Config file provided, but overriding group columns for flattened data compatibility..."
                )
                if "group_labels" in config:
                    old_group_cols = config["group_labels"].get("cols", [])
                    logger.info(
                        f"Original config group columns: {old_group_cols}"
                    )

                # Create new config optimized for flattened data
                flattened_config = create_config_for_flattened_data(
                    data_df,
                    timestamps_col,
                    target_cols,
                    metadata_cols,
                    leak_cols,
                    sample_id_cols,
                    history_df,
                )

                # Merge configs, prioritizing flattened data settings
                config.update(flattened_config)
                logger.info(
                    f"Updated config group columns: {config.get('group_labels', {}).get('cols', [])}"
                )
                logger.info(
                    f"Updated config timeseries columns: {config.get('timeseries', {}).get('cols', [])}"
                )

                # Show what columns are actually available in the flattened data
                logger.info(
                    f"Available columns in flattened data: {list(data_df.columns)}"
                )
                logger.info(f"Flattened data shape: {data_df.shape}")
        else:
            logger.info(
                "Data appears to be in standard format, proceeding without conversion"
            )
            if train_test_split:
                test_df = data_df.iloc[
                    int(len(data_df) * train_test_split) : len(data_df)
                ]
                data_df = data_df.iloc[0 : int(len(data_df) * train_test_split)]
            else:
                test_df = None
    else:
        test_df = None

    if hasattr(data_df, "to_pandas"):  # Already polars
        polars_df = data_df
    else:  # Convert pandas to polars
        # Ensure unique column names for Polars compatibility
        if not data_df.columns.is_unique:
            data_df.columns = make_unique_column_names(data_df.columns)
        # Convert datetime columns to string to avoid PyArrow conversion issues
        for col in data_df.columns:
            if data_df[col].dtype == "datetime64[ns]":
                data_df[col] = data_df[col].astype(str)
        polars_df = pl.from_pandas(data_df)
    if test_df is not None:
        # Ensure unique column names for Polars compatibility
        if not test_df.columns.is_unique:
            test_df.columns = make_unique_column_names(test_df.columns)
        # Convert datetime columns to string to avoid PyArrow conversion issues
        for col in test_df.columns:
            if test_df[col].dtype == "datetime64[ns]":
                test_df[col] = test_df[col].astype(str)
        polars_test_df = pl.from_pandas(test_df)
    else:
        polars_test_df = None

    # Auto-adjust group columns if they are not found in the dataset
    if processed_group_cols:
        processed_group_cols = auto_adjust_group_columns(
            polars_df, processed_group_cols
        )
        logger.info(f"Adjusted group columns: {processed_group_cols}")

    # Also check if config specifies group columns that don't exist
    if config and "group_labels" in config:
        config_group_cols = config.get("group_labels", {}).get("cols", [])
        if config_group_cols:
            available_config_group_cols = auto_adjust_group_columns(
                polars_df, config_group_cols
            )
            if available_config_group_cols != config_group_cols:
                logger.warning(
                    "Config file specifies group columns that don't exist in dataset"
                )
                logger.warning(f"Config group columns: {config_group_cols}")
                logger.warning(
                    f"Available group columns: {available_config_group_cols}"
                )
                # Update the config to use only available columns
                if "group_labels" not in config:
                    config["group_labels"] = {}
                config["group_labels"]["cols"] = available_config_group_cols

    # Final validation and logging of configuration
    logger.info("=" * 60)
    logger.info("FINAL CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Group columns: {processed_group_cols}")
    logger.info(
        f"Config file: {args.config_file if args.config_file else 'None'}"
    )
    if config:
        logger.info(
            f"Config group columns: {config.get('group_labels', {}).get('cols', [])}"
        )
        logger.info(
            f"Config timeseries columns: {config.get('timeseries', {}).get('cols', [])}"
        )
        logger.info(
            f"Config timestamp column: {config.get('timestamps_col', 'None')}"
        )
    logger.info(f"Data shape: {polars_df.shape}")
    logger.info(f"Data columns: {list(polars_df.columns)}")
    logger.info("=" * 60)

    # Process analysis_functions into a list if provided
    processed_analysis_functions = None
    if args.analysis_functions:
        processed_analysis_functions = args.analysis_functions.split(",")
        logger.info(f"Using analysis functions: {processed_analysis_functions}")

    # Initialize DataSummarizer
    summarizer = DataSummarizer(
        data_input=polars_df,  # type: ignore
        save_path=run_output_dir,
        config=config,
        group_cols=processed_group_cols,
        skip_plots=args.skip_plots,
        test_df=polars_test_df,
        compute_all=args.compute_all,
        analysis_functions=processed_analysis_functions,
        execute_forecast=args.execute_forecast,
    )

    # Generate summaries
    logger.info("Generating metadata summary...")
    summarizer.summarize_metadata()

    logger.info("Generating time series summary...")
    summarizer.summarize_time_series()

    # Ensure comprehensive analysis is completed (this generates all plots)
    logger.info("Performing comprehensive analysis to generate all plots...")
    summarizer.perform_comprehensive_analysis()

    # Generate HTML report with all plots
    dataset_basename = effective_dataset_name.replace("/", "_")
    html_path = os.path.join(
        run_output_dir, f"{dataset_basename}_data_summary.html"
    )

    logger.info(f"Generating HTML report with all plots at {html_path}...")
    summarizer.generate_html_report(output_html=html_path)

    # Get summary data
    summary_dict = summarizer.get_summary_dict()
    logger.info("Summary generated successfully.")

    # Log plot generation summary
    logger.info("Plot generation summary:")
    logger.info(
        f"  - Decomposition plots: {summary_dict.get('decomposition_plots', 0)}"
    )
    logger.info(
        f"  - Autocorrelation plots: {summary_dict.get('autocorrelation_plots', 0)}"
    )
    logger.info(
        f"  - Cross-correlation matrix: {'Generated' if summary_dict.get('cross_correlation_matrix_plot') else 'Not Generated'}"
    )
    logger.info(
        f"  - Time series plots: {'Generated' if not args.skip_plots and summary_dict.get('time_series_summary') else 'Skipped'}"
    )
    logger.info(
        f"  - Distribution plots: {'Generated' if not args.skip_plots and summary_dict.get('metadata_summary') else 'Skipped'}"
    )
    summarizer.cleanup()

    return summary_dict


def process_all_datasets(
    args: argparse.Namespace,
    run_id: str,
    output_directory: str,
    comprehensive_generator: ComprehensiveSummaryGenerator,
) -> None:
    """Process all supported datasets and generate comprehensive summary statistics."""
    logger.info("Processing all supported datasets...")

    # Get list of supported datasets
    supported_datasets = get_supported_datasets()
    logger.info(
        f"Found {len(supported_datasets)} supported datasets: {supported_datasets}"
    )

    # Filter out datasets that require special handling or external dependencies
    # These datasets might need specific configs or data paths that aren't available in batch mode

    logger.info(
        f"Processing {len(supported_datasets)} batch-processable datasets"
    )

    for dataset_name in supported_datasets:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'=' * 60}")

            # Create dataset-specific output directory
            dataset_output_dir = os.path.join(
                output_directory, dataset_name, run_id
            )
            os.makedirs(dataset_output_dir, exist_ok=True)

            # Create temporary args for this dataset
            dataset_args = argparse.Namespace()
            for attr, value in vars(args).items():
                setattr(dataset_args, attr, value)
            dataset_args.dataset = dataset_name

            # Set default values for datasets that need them
            if dataset_name == "gift":
                # Skip gift dataset in batch mode as it requires specific data path
                logger.info(
                    f"Skipping {dataset_name} - requires specific data path configuration"
                )
                continue

            # Process the dataset
            try:
                summary_dict = summarize_data_from_dataloader(
                    dataset_args, dataset_output_dir, dataset_name
                )

                # Add to comprehensive generator
                comprehensive_generator.add_dataset_summary(
                    dataset_name, summary_dict, args.max_samples
                )

                logger.info(f"Successfully processed {dataset_name}")

            except Exception as e:
                error_msg = str(e)
                if "n_unique operation not supported for dtype" in error_msg:
                    logger.error(
                        f"Failed to process dataset {dataset_name}: Data type compatibility issue - {error_msg}"
                    )
                    logger.error(
                        "This usually indicates columns with unsupported data types (e.g., list columns)"
                    )
                    comprehensive_generator.record_failed_dataset(
                        dataset_name, error_msg, "data_type_compatibility"
                    )
                else:
                    logger.error(
                        f"Failed to process dataset {dataset_name}: {error_msg}"
                    )
                    comprehensive_generator.record_failed_dataset(
                        dataset_name, error_msg, "processing_error"
                    )
                # Continue with next dataset instead of failing completely
                continue

        except Exception as e:
            logger.error(
                f"Unexpected error processing dataset {dataset_name}: {e}"
            )
            continue

    # Process datasets that require special handling
    special_datasets = ["fmv3", "gpt-synthetic", "external"]
    logger.info(
        f"\nProcessing {len(special_datasets)} special datasets that require configuration..."
    )

    for dataset_name in special_datasets:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing special dataset: {dataset_name}")
            logger.info(f"{'=' * 60}")

            # Skip these in batch mode as they require specific configs
            logger.info(
                f"Skipping {dataset_name} - requires specific configuration in batch mode"
            )
            continue

        except Exception as e:
            logger.error(
                f"Error processing special dataset {dataset_name}: {e}"
            )
            continue

    logger.info(f"\n{'=' * 60}")
    logger.info("COMPREHENSIVE SUMMARY GENERATION")
    logger.info(f"{'=' * 60}")

    # Generate and save comprehensive report
    try:
        report_file = comprehensive_generator.save_comprehensive_report(run_id)
        logger.info(f"Comprehensive summary report saved to: {report_file}")

        # Export to CSV for easy analysis
        csv_file = comprehensive_generator.export_summary_to_csv(run_id)
        if csv_file:
            logger.info(f"Dataset breakdown exported to CSV: {csv_file}")

        # Add some example custom metrics to demonstrate extensibility
        comprehensive_generator.add_custom_global_metric(
            "processing_timestamp", datetime.now().isoformat()
        )
        comprehensive_generator.add_custom_global_metric(
            "total_processing_time_minutes", 0
        )  # Placeholder

        # Demonstrate custom metrics functionality
        demonstrate_custom_metrics_usage(comprehensive_generator)

        # Print summary statistics
        summary_stats = comprehensive_generator.summary_statistics
        logger.info(
            f"Total datasets processed: {summary_stats['total_datasets']}"
        )
        logger.info(f"Total data points: {summary_stats['total_data_points']}")
        logger.info(f"Total rows: {summary_stats['total_rows']}")
        logger.info(f"Total columns: {summary_stats['total_columns']}")
        logger.info(
            f"Average data points per dataset: {summary_stats.get('average_data_points_per_dataset', 0):.2f}"
        )
        logger.info(
            f"Average rows per dataset: {summary_stats.get('average_rows_per_dataset', 0):.2f}"
        )
        logger.info(
            f"Average columns per dataset: {summary_stats.get('average_columns_per_dataset', 0):.2f}"
        )

        # Print sample limiting summary if applicable
        if args.max_samples:
            sample_limiting_summary = summary_stats.get(
                "sample_limiting_summary", {}
            )
            logger.info(
                f"Sample limiting applied to {sample_limiting_summary.get('datasets_with_limiting', 0)} out of {sample_limiting_summary.get('total_datasets', 0)} datasets"
            )
            logger.info(
                f"Percentage of datasets with limiting: {sample_limiting_summary.get('percentage_limited', 0):.1f}%"
            )

        # Print failed datasets summary
        failed_summary = summary_stats.get("failed_datasets_summary", {})
        if failed_summary.get("total_failed", 0) > 0:
            logger.info(
                f"Failed to process {failed_summary['total_failed']} datasets:"
            )

            # Group failures by error type for better reporting
            error_types = {}
            for dataset_name, failure_info in failed_summary.get(
                "failed_datasets", {}
            ).items():
                error_type = failure_info["error_type"]
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(dataset_name)

            for error_type, datasets in error_types.items():
                logger.info(f"  {error_type}: {len(datasets)} datasets")
                for dataset_name in datasets:
                    failure_info = failed_summary["failed_datasets"][
                        dataset_name
                    ]
                    logger.info(
                        f"    - {dataset_name}: {failure_info['error'][:100]}..."
                    )
        else:
            logger.info("All datasets processed successfully!")

    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        raise


def demonstrate_custom_metrics_usage(
    comprehensive_generator: ComprehensiveSummaryGenerator,
) -> None:
    """Demonstrate how to extend the summary system with custom metrics.

    This function shows various ways to add custom metrics to the comprehensive summary.
    Users can modify this function or use it as a template for their own extensions.
    """
    logger.info("Demonstrating custom metrics functionality...")

    # Example 1: Add global custom metrics
    comprehensive_generator.add_custom_global_metric(
        "example_global_metric", "This is a global example"
    )
    comprehensive_generator.add_custom_global_metric(
        "example_numeric_metric", 42
    )
    comprehensive_generator.add_custom_global_metric(
        "example_list_metric", [1, 2, 3, 4, 5]
    )

    # Example 2: Add dataset-specific custom metrics
    for dataset_name in comprehensive_generator.all_dataset_results.keys():
        comprehensive_generator.add_custom_dataset_metric(
            dataset_name,
            "example_dataset_metric",
            f"Custom metric for {dataset_name}",
        )
        comprehensive_generator.add_custom_dataset_metric(
            dataset_name,
            "example_dataset_numeric",
            len(dataset_name),  # Just an example calculation
        )

    # Example 3: Add computed metrics based on existing data
    total_plots = 0
    for (
        dataset_name,
        dataset_data,
    ) in comprehensive_generator.all_dataset_results.items():
        stats = dataset_data.get("statistics", {})
        total_plots += stats.get("decomposition_plots", 0) + stats.get(
            "autocorrelation_plots", 0
        )

    comprehensive_generator.add_custom_global_metric(
        "total_plots_generated", total_plots
    )

    # Example 4: Add metadata about the processing
    processing_metadata = {
        "python_version": sys.version,
        "script_name": "summarize.py",
        "functionality": "comprehensive_dataset_summarization",
        "extensibility": "custom_metrics_supported",
    }
    comprehensive_generator.add_custom_global_metric(
        "processing_metadata", processing_metadata
    )

    logger.info("Custom metrics demonstration completed")


def main():
    """Main function for data summarization using DataSummarizer."""
    args = parse_summary_arguments()

    # Generate unique ID for this run
    run_id = "0"  # generate_unique_id()
    logger.info(f"Summary Run ID: {run_id}")

    # Log max_samples usage if specified
    if args.max_samples:
        logger.info(
            f"Sample limiting enabled: maximum {args.max_samples:,} samples per dataset"
        )

    # Handle JSON modification mode
    if args.modify_existing_json:
        logger.info("=" * 60)
        logger.info("JSON MODIFICATION MODE")
        logger.info("=" * 60)

        # Get effective dataset name
        if args.input_file:
            effective_dataset_name = os.path.splitext(
                os.path.basename(args.input_file)
            )[0]
        else:
            effective_dataset_name = get_effective_dataset_name(args)

        logger.info(f"Dataset: {effective_dataset_name}")

        # Determine the path to the existing JSON file
        if args.download_from_s3:
            # Download from S3 using dataset name
            temp_json_path = os.path.join(
                tempfile.gettempdir(),
                f"{effective_dataset_name}_temp_summary.json",
            )
            # Ensure the directory exists
            os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)
            existing_json_path = download_json_from_s3(
                effective_dataset_name, temp_json_path
            )
            logger.info(f"Downloaded JSON from S3 to: {existing_json_path}")
        else:
            # Generate expected path based on dataset name and output directory
            existing_json_path = generate_expected_json_path(
                args, effective_dataset_name, run_id
            )
            if not os.path.exists(existing_json_path):
                logger.error(
                    f"JSON file not found at expected path: {existing_json_path}"
                )
                logger.error(
                    "Make sure the dataset has been processed before and the output directory is correct"
                )
                sys.exit(1)
            logger.info(f"Using existing JSON file: {existing_json_path}")

        # Process analysis functions
        processed_analysis_functions = None
        generate_html = False
        if args.analysis_functions:
            all_functions = args.analysis_functions.split(",")
            # Check if generate_html is requested
            if "generate_html" in all_functions:
                generate_html = True
                # Remove generate_html from the list of analysis functions
                processed_analysis_functions = [
                    f for f in all_functions if f != "generate_html"
                ]
                logger.info(
                    "HTML generation requested - will run after JSON modification"
                )
            else:
                processed_analysis_functions = all_functions

            logger.info(
                f"Running analysis functions: {processed_analysis_functions}"
            )
        else:
            logger.error(
                "--analysis-functions is required when modifying existing JSON"
            )
            sys.exit(1)

        # We need to run the analysis on the same data to get updated results
        # For now, we'll require the user to provide the dataset information
        if not args.dataset and not args.input_file:
            logger.error(
                "Either --dataset or --input-file is required for JSON modification mode"
            )
            sys.exit(1)

        # Use the provided output directory for analysis
        if not args.output_directory:
            raise ValueError(
                "Output directory is required for JSON modification mode"
            )
        # analysis_output_dir = args.output_directory
        modified_json_output_dir = os.path.dirname(args.output_json_path)
        os.makedirs(modified_json_output_dir, exist_ok=True)

        # Also ensure the output directory exists for analysis
        analysis_output_dir = args.output_directory
        os.makedirs(analysis_output_dir, exist_ok=True)

        # Run the analysis to get new results
        if args.input_file:
            # Direct file input mode
            logger.info(f"Running analysis on file: {args.input_file}")

            # Load configuration if provided
            config = None
            if args.config_file:
                with open(args.config_file, "r") as file:
                    config = yaml.safe_load(file)

            # Process group_cols into a list if provided
            processed_group_cols = None
            if args.group_cols:
                processed_group_cols = args.group_cols.split(",")

            # Generate summary from file with only specified analysis functions
            summary_dict = summarize_data_from_file(
                args.input_file,
                modified_json_output_dir,
                config,
                processed_group_cols,
                args.skip_plots,
                args.max_samples,
                compute_all=args.compute_all,
                analysis_functions=processed_analysis_functions,
            )
        else:
            # Dataloader system mode
            effective_dataset_name = get_effective_dataset_name(args)
            logger.info(
                f"Running analysis on dataset: {effective_dataset_name}"
            )

            # Create temporary args for analysis
            temp_args = argparse.Namespace()
            for attr, value in vars(args).items():
                setattr(temp_args, attr, value)

            summary_dict = summarize_data_from_dataloader(
                temp_args, modified_json_output_dir, effective_dataset_name
            )

        # Modify the existing JSON with new results
        modified_json_path = modify_existing_json(
            existing_json_path,
            summary_dict,
            processed_analysis_functions,
            args.overwrite_json,
            args.output_json_path,  # Use custom output path if provided
        )

        logger.info("JSON modification completed successfully!")
        logger.info(f"Modified JSON saved to: {modified_json_path}")

        # Generate HTML if requested
        if generate_html:
            logger.info("=" * 60)
            logger.info("GENERATING HTML REPORT")
            logger.info("=" * 60)

            # Load the modified JSON data for HTML generation
            with open(modified_json_path, "r") as f:
                json_data = json.load(f)

            # Create output directory for HTML
            html_output_dir = os.path.dirname(modified_json_path)
            # Create DataSummarizer from the JSON data
            data_summarizer = DataSummarizer(
                data_input=args.input_file,
                save_path=html_output_dir,
                config=config,
                group_cols=processed_group_cols,
                skip_plots=args.skip_plots,
            )
            temp_summarizer = populate_data_summarizer_from_dict(
                data_summarizer, json_data, html_output_dir
            )

            # Generate HTML report
            dataset_basename = effective_dataset_name.replace("/", "_")
            html_path = os.path.join(
                html_output_dir, f"{dataset_basename}_data_summary.html"
            )

            logger.info(f"Generating HTML report at: {html_path}")
            temp_summarizer.generate_html_report(output_html=html_path)
            logger.info(f"HTML report generated successfully: {html_path}")

        # Clean up temporary files
        if args.download_from_s3 and os.path.exists(existing_json_path):
            os.remove(existing_json_path)
            logger.info(
                f"Cleaned up temporary downloaded file at {existing_json_path}"
            )

        # Clean up temporary output directory
        import shutil

        return

    # Check if --dataset ALL is specified
    if args.dataset and args.dataset == "ALL":
        # Check for conflicting arguments
        if args.input_file:
            logger.error(
                "Cannot use --dataset ALL with --input-file. Use one or the other."
            )
            sys.exit(1)

        logger.info("Processing all supported datasets...")

        # Create output directory structure
        base_output_dir = os.path.join(args.output_directory, "all_datasets")
        os.makedirs(base_output_dir, exist_ok=True)

        # Create run-specific directory
        run_output_dir = os.path.join(base_output_dir, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        # Initialize comprehensive summary generator
        comprehensive_generator = ComprehensiveSummaryGenerator(run_output_dir)

        # Process all datasets
        process_all_datasets(
            args, run_id, run_output_dir, comprehensive_generator
        )

        logger.info(
            f"All datasets processing completed. Results saved to: {run_output_dir}"
        )
        return

    # Determine if we're using direct file input or dataloader system
    if args.input_file:
        # Direct file input mode
        logger.info(f"Direct file input mode: {args.input_file}")

        # Create output directory structure
        base_output_dir = os.path.join(args.output_directory, "direct_files")
        os.makedirs(base_output_dir, exist_ok=True)

        # Create run-specific directory
        run_output_dir = os.path.join(base_output_dir, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        # Load configuration if provided
        config = None
        if args.config_file:
            try:
                with open(args.config_file, "r") as file:
                    config = yaml.safe_load(file)
                logger.info(f"Loaded config from: {args.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")

        # Process group_cols into a list if provided
        processed_group_cols = None
        if args.group_cols:
            processed_group_cols = args.group_cols.split(",")
            logger.info(f"Using group columns: {processed_group_cols}")

        # Process analysis_functions into a list if provided
        processed_analysis_functions = None
        if args.analysis_functions:
            processed_analysis_functions = args.analysis_functions.split(",")
            logger.info(
                f"Using analysis functions: {processed_analysis_functions}"
            )

        # Generate summary from file
        summary_dict = summarize_data_from_file(
            args.input_file,
            run_output_dir,
            config,
            processed_group_cols,
            args.skip_plots,
            args.max_samples,
            compute_all=args.compute_all,
            analysis_functions=processed_analysis_functions,
        )

        # Use filename as dataset name for logging
        effective_dataset_name = os.path.splitext(
            os.path.basename(args.input_file)
        )[0]

    else:
        # Dataloader system mode
        effective_dataset_name = get_effective_dataset_name(args)
        logger.info(f"Dataset: {effective_dataset_name}")

        # Display conditional arguments based on dataset
        if args.dataset == "fmv3":
            logger.info(f"Config file: {args.config_file}")
        elif args.dataset == "traffic":
            logger.info("Using traffic dataset (loading data directly)")
        elif args.dataset == "solar_alabama":
            logger.info("Using solar_alabama dataset (loading data directly)")
        elif args.dataset == "weather_mpi":
            logger.info("Using weather_mpi dataset (loading data directly)")
        elif args.dataset == "gift":
            logger.info(f"Data path: {args.data_path}")
            logger.info(f"Sub-dataset: {args.sub_dataset}")
        elif args.dataset == "goodrx":
            logger.info("Using GoodRx dataset (loading from S3)")
        elif args.dataset == "spain_energy":
            logger.info("Using Spain energy dataset (loading from S3)")
        elif args.dataset == "gpt-synthetic":
            logger.info("Using GPT synthetic dataset")
            if args.dataset_name:
                logger.info(f"Dataset name: {args.dataset_name}")
            else:
                raise ValueError(
                    "Dataset name is required for gpt-synthetic dataset, options: energy, manufacturing, retail, supply_chain, traffic"
                )
        elif args.dataset == "cursor_tabs":
            logger.info("Using cursor_tabs dataset (loading from S3)")
        elif args.dataset == "walmart_sales":
            logger.info("Using walmart_sales dataset (loading from S3)")
        elif args.dataset == "complex_seasonal_timeseries":
            logger.info(
                "Using complex_seasonal_timeseries dataset (loading from local CSV)"
            )
        elif args.dataset == "external":
            logger.info(
                f"Using external dataloader: {args.external_dataloader_spec}"
            )

        # Create output directory structure
        if args.dataset == "gift":
            logger.info(f"output path: {args.output_directory}")
            logger.info(f"Sub-dataset: {args.sub_dataset}")
            base_output_dir = os.path.join(
                args.output_directory, args.dataset, args.sub_dataset
            )
        elif args.dataset == "gpt-synthetic":
            # Use dataset_name parameter if provided, otherwise default to "retail"
            dataset_subdir = args.dataset_name if args.dataset_name else ""
            base_output_dir = os.path.join(
                args.output_directory,
                args.dataset,
                dataset_subdir,
            )
        else:
            base_output_dir = os.path.join(args.output_directory, args.dataset)
        os.makedirs(base_output_dir, exist_ok=True)
        logger.info(f"Output directory: {base_output_dir}")

        # Create run-specific directory
        run_output_dir = os.path.join(base_output_dir, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        # Generate summary using dataloader system
        summary_dict = summarize_data_from_dataloader(
            args, run_output_dir, effective_dataset_name
        )

    # Save run info
    save_summary_run_info(
        args,
        run_id,
        run_output_dir,
        effective_dataset_name,
        summary_dict,
    )

    # Save summary data in JSON format
    dataset_basename = effective_dataset_name.replace("/", "_")
    summary_data_file = os.path.join(
        run_output_dir, f"{dataset_basename}_summary_data.json"
    )
    try:
        with open(summary_data_file, "w") as f:
            json.dump(summary_dict, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved summary data to: {summary_data_file}")
    except Exception as e:
        logger.error(f"Failed to save summary data: {e}")

    # Print summary statistics
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Dataset: {effective_dataset_name}")
    logger.info(
        f"Total rows: {summary_dict.get('metadata', {}).get('total_rows', 'N/A')}"
    )
    logger.info(
        f"Total columns: {summary_dict.get('metadata', {}).get('total_columns', 'N/A')}"
    )
    logger.info(
        f"Time series columns: {len(summary_dict.get('time_series_summary') or [])}"
    )
    logger.info(
        f"Discrete columns: {len(summary_dict.get('discrete_summary') or [])}"
    )
    logger.info(
        f"Continuous columns: {len(summary_dict.get('continuous_summary') or [])}"
    )

    # Print comprehensive analysis results
    logger.info("\n" + "=" * 50)
    logger.info("COMPREHENSIVE ANALYSIS RESULTS")
    logger.info("=" * 50)
    logger.info(
        f"Basic Statistics: {len(summary_dict.get('basic_statistics') or [])} columns analyzed"
    )
    logger.info(
        f"Correlation Matrix: {len(summary_dict.get('correlation_matrix') or [])} rows"
    )
    logger.info(
        f"Outlier Analysis: {len(summary_dict.get('outlier_analysis') or [])} columns analyzed"
    )
    logger.info(
        f"Quantile Analysis: {len(summary_dict.get('quantile_analysis') or [])} columns analyzed"
    )
    logger.info(
        f"Autocorrelation Analysis: {len(summary_dict.get('autocorrelation_analysis') or [])} columns analyzed"
    )
    logger.info(
        f"Decomposition Analysis: {len(summary_dict.get('decomposition_analysis') or [])} columns analyzed"
    )
    logger.info(
        f"Cross-Correlation Analysis: {len(summary_dict.get('cross_correlation_analysis') or [])} pairs analyzed"
    )

    # Print plot generation summary
    logger.info("\n" + "=" * 50)
    logger.info("PLOT GENERATION SUMMARY")
    logger.info("=" * 50)
    logger.info(
        f"Decomposition Plots: {summary_dict.get('decomposition_plots', 0)} plots generated"
    )
    logger.info(
        f"Autocorrelation Plots: {summary_dict.get('autocorrelation_plots', 0)} plots generated"
    )
    logger.info(
        f"Cross-Correlation Matrix: {'Generated' if summary_dict.get('cross_correlation_matrix_plot') else 'Not Generated'}"
    )
    logger.info(
        f"Time Series Plots: {'Generated' if not args.skip_plots and summary_dict.get('time_series_summary') else 'Skipped'}"
    )
    logger.info(
        f"Distribution Plots: {'Generated' if not args.skip_plots and summary_dict.get('metadata_summary') else 'Skipped'}"
    )

    logger.info(f"Results saved to: {run_output_dir}")

    # Show HTML report location
    if args.input_file:
        file_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        html_file = f"{file_basename}_data_summary.html"
    else:
        dataset_basename = effective_dataset_name.replace("/", "_")
        html_file = f"{dataset_basename}_data_summary.html"

    html_path = os.path.join(run_output_dir, html_file)
    logger.info(f"HTML Report: {html_path}")
    logger.info(f"Data summarization completed successfully for run {run_id}")


if __name__ == "__main__":
    main()

# Example usage of the comprehensive summary system programmatically:
"""
# Import and use the comprehensive summary generator programmatically
from synthefy_pkg.fm_evals.summarize import ComprehensiveSummaryGenerator

# Initialize the generator
generator = ComprehensiveSummaryGenerator("/path/to/output")

# Add custom metrics
generator.add_custom_global_metric("my_custom_metric", "custom_value")
generator.add_custom_dataset_metric("dataset_name", "dataset_metric", "value")

# Add dataset summaries (this would normally be done by the processing functions)
# generator.add_dataset_summary("dataset_name", summary_dict)

# Generate and save the report
report_file = generator.save_comprehensive_report("run_123")

# Export to CSV
csv_file = generator.export_summary_to_csv("run_123")

# Get summaries
global_summary = generator.get_global_summary()
dataset_summary = generator.get_dataset_summary("dataset_name")

# Example usage of JSON modification functionality programmatically:
from synthefy_pkg.fm_evals.summarize import modify_existing_json, download_json_from_s3, generate_expected_json_path

# Method 1: Download JSON from S3 and modify it
temp_path = "/tmp/temp_summary.json"
downloaded_path = download_json_from_s3("traffic", temp_path)

# Run analysis to get new results (this would be done with DataSummarizer)
new_results = {
    "correlation_matrix": [...],  # New correlation results
    "outlier_analysis": [...]     # New outlier results
}

# Modify the existing JSON
modified_path = modify_existing_json(
    downloaded_path,
    new_results,
    ["correlation", "outlier"],
    overwrite=False,  # Create new file instead of overwriting
    output_path="/path/to/custom_location.json"  # Specify custom output path
)

print(f"Modified JSON saved to: {modified_path}")

# Method 2: Use automatic path generation for local files
import argparse
args = argparse.Namespace()
args.dataset = "traffic"
args.output_directory = "results/"
args.sub_dataset = None
args.dataset_name = None

expected_path = generate_expected_json_path(args, "traffic", "0")
print(f"Expected JSON path: {expected_path}")
"""
