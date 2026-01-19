#!/usr/bin/env python3
"""
Script to extract CSV files and create stacked plots from datasets using the same approach as eval.py and summarize.py.

This script:
1. Uses dataset names from eval.py's get_supported_datasets()
2. Uses get_dataloader() function from eval.py for consistent loading
3. Flattens data like summarize.py does
4. Saves one CSV file per dataset to the target folder
5. Creates stacked time series plots for each dataset using DataSummarizer
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

# Import functions from eval.py and summarize.py
from synthefy_pkg.fm_evals.eval import get_dataloader, get_supported_datasets
from synthefy_pkg.fm_evals.summarize import (
    convert_nested_list_data_to_dataframe,
    extract_data_from_dataloader,
)

# Import plotting functionality
from synthefy_pkg.preprocessing.data_summarizer import DataSummarizer

# Add the package root to the path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))


def create_mock_args(dataset_name: str) -> argparse.Namespace:
    """Create mock args object for get_dataloader function."""
    args = argparse.Namespace()
    args.dataset = dataset_name
    args.random_ordering = False
    args.data_path = None
    args.external_dataloader_spec = None
    args.forecast_length = None
    args.history_length = None
    args.config_file = None

    # Special handling for gpt-synthetic dataset
    if dataset_name == "gpt-synthetic":
        # gpt-synthetic requires a specific dataset_name parameter
        # Use 'energy' as the default sub-dataset
        args.dataset_name = "energy"
    else:
        args.dataset_name = None

    return args


def extract_and_flatten_data_from_dataloader(
    dataloader, max_samples: int = 1
) -> pd.DataFrame:
    """
    Extract data from dataloader and flatten it like summarize.py does.

    Args:
        dataloader: The dataloader instance
        max_samples: Maximum number of samples to extract (default: 1)

    Returns:
        Flattened pandas DataFrame
    """
    # Extract data using summarize.py's function
    (
        data_df,
        target_cols,
        metadata_cols,
        leak_cols,
        sample_id_cols,
        timestamps_col,
    ) = extract_data_from_dataloader(dataloader, max_samples)

    # Check if data is in nested list format and convert if needed
    if hasattr(data_df, "columns"):
        expected_columns = [
            "sample_id",
            "history_timestamps",
            "history_values",
            "target_timestamps",
            "target_values",
        ]

        if all(col in data_df.columns for col in expected_columns):
            print("  Converting nested list data to flattened format...")
            history_df, target_df, combined_df = (
                convert_nested_list_data_to_dataframe(data_df)
            )
            return combined_df  # Return the combined flattened data
        else:
            # Data is already in a regular format
            return data_df
    else:
        return data_df


def create_plots_for_dataset(
    dataset_name: str, df: pd.DataFrame, plots_dir: Path
) -> bool:
    """
    Create stacked time series plots for a dataset using DataSummarizer.

    Args:
        dataset_name: Name of the dataset
        df: Flattened DataFrame containing the data
        plots_dir: Directory to save the plots

    Returns:
        True if successful, False otherwise
    """
    print(f"  Creating plots for {dataset_name}...")

    # Create dataset-specific plots directory
    dataset_plots_dir = plots_dir / dataset_name
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)

    # Convert pandas DataFrame to polars DataFrame
    polars_df = pl.from_pandas(df)

    # Create a basic configuration for DataSummarizer
    config = {
        "timestamps_col": "timestamp" if "timestamp" in df.columns else None,
        "timeseries": {"cols": []},
        "continuous": {"cols": []},
        "discrete": {"cols": []},
        "group_labels": {"cols": []},
    }

    # Auto-detect column types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        config["continuous"]["cols"] = numeric_cols

    # Find timestamp column
    timestamp_candidates = ["timestamp", "time", "date", "datetime"]
    timestamp_col = None
    for col in timestamp_candidates:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        config["timestamps_col"] = timestamp_col

    # Create DataSummarizer instance
    DataSummarizer(
        data_input=polars_df,
        save_path=str(dataset_plots_dir),
        config=config,
        skip_plots=False,  # Let DataSummarizer handle plot creation
        compute_all=True,  # Enable analysis and plot generation
        analysis_functions=["stacked_plots"],
    )

    # Generate summaries and plots
    # summarizer.summarize_metadata()
    # summarizer.summarize_time_series()
    # summarizer.perform_comprehensive_analysis(analysis_functions=["stacked_plots"])

    print(f"  Plots saved to {dataset_plots_dir}")
    return True


def extract_csv_from_dataset(
    dataset_name: str, target_dir: Path, plots_dir: Optional[Path] = None
) -> bool:
    """
    Extract CSV data from a single dataset using eval.py's approach.

    Args:
        dataset_name: Name of the dataset (from get_supported_datasets)
        target_dir: Directory to save the CSV file
        plots_dir: Optional directory to save plots

    Returns:
        True if successful, False otherwise
    """
    print(f"Processing dataset: {dataset_name}")

    # Create mock args for get_dataloader
    args = create_mock_args(dataset_name)

    # Get dataloader using eval.py's function
    dataloader = get_dataloader(args)

    # Extract and flatten data (limit to 1 sample for efficiency)
    df = extract_and_flatten_data_from_dataloader(dataloader, max_samples=1)

    if df is None or len(df) == 0:
        print(f"  Empty or invalid data for {dataset_name}")
        return False

    # Save to target directory
    output_file = target_dir / f"{dataset_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} rows to {output_file}")

    # Create plots if plots_dir is provided
    if plots_dir is not None:
        create_plots_for_dataset(dataset_name, df, plots_dir)

    return True


def main():
    """Main function to extract CSV files and create plots from all supported datasets."""

    # Create target directories
    target_dir = Path("results/sample_csvs_eval_style")
    plots_dir = Path("results/sample_plots_eval_style")

    target_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print(f"CSV target directory: {target_dir.absolute()}")
    print(f"Plots target directory: {plots_dir.absolute()}")

    # Get supported datasets from eval.py
    print("Getting supported datasets from eval.py...")
    supported_datasets = get_supported_datasets()
    print(f"Found {len(supported_datasets)} supported datasets")

    # Process each dataset
    successful_extractions = 0
    successful_plots = 0
    failed_extractions = 0
    skipped_datasets = 0

    for dataset_name in supported_datasets:
        # Skip datasets that require complex configuration or have issues
        if dataset_name in [
            "fmv3",
            "gift",
            "synthetic_medium_lag",
            "complex_seasonal_timeseries",
        ]:
            print(
                f"Skipping dataset: {dataset_name} (requires complex configuration)"
            )
            skipped_datasets += 1
            continue

        success = extract_csv_from_dataset(dataset_name, target_dir, plots_dir)
        if success:
            successful_extractions += 1
            # Check if plots were created successfully
            dataset_plots_dir = plots_dir / dataset_name
            if dataset_plots_dir.exists() and any(dataset_plots_dir.iterdir()):
                successful_plots += 1
        else:
            failed_extractions += 1
        print()  # Add blank line for readability

    # Summary
    print("=" * 60)
    print("EXTRACTION AND PLOTTING SUMMARY")
    print("=" * 60)
    print(f"Total datasets available: {len(supported_datasets)}")
    print(f"Skipped datasets: {skipped_datasets}")
    print(f"Datasets processed: {len(supported_datasets) - skipped_datasets}")
    print(f"Successful CSV extractions: {successful_extractions}")
    print(f"Successful plot generations: {successful_plots}")
    print(f"Failed extractions: {failed_extractions}")
    processed_count = len(supported_datasets) - skipped_datasets
    if processed_count > 0:
        print(
            f"CSV extraction success rate: {successful_extractions / processed_count * 100:.1f}%"
        )
        print(
            f"Plot generation success rate: {successful_plots / processed_count * 100:.1f}%"
        )
    print(f"CSV files saved to: {target_dir.absolute()}")
    print(f"Plots saved to: {plots_dir.absolute()}")


if __name__ == "__main__":
    main()
