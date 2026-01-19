#!/usr/bin/env python3
"""
Dataset Processing Script

Usage:
--all-datasets: Process all datasets found in S3 bucket
--datasets: Process specific datasets

python aggregate_tests.py --all-datasets
python aggregate_tests.py --datasets mta_ridership aus_electricity
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from synthefy_pkg.fm_evals.eval import (
    get_dataloader,
    get_effective_dataset_name,
    get_supported_datasets,
)
from synthefy_pkg.fm_evals.scripts.utils import (
    get_datasets_in_s3,
    get_s3_client,
    load_forecast_data,
    read_existing_forecasts_from_s3,
    write_forecasts_to_s3,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process forecast datasets and generate aggregated JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific datasets
  python aggregate_tests.py --datasets mta_ridership aus_electricity

  # Process all datasets in S3 bucket
  python aggregate_tests.py --all-datasets
""",
    )

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        help="List of dataset names to process",
    )
    dataset_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="Process all datasets found in S3 bucket",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.all_datasets:
        return

    if not args.datasets:
        logger.error("No datasets specified")
        sys.exit(1)


def load_raw_dataset_data(
    dataset_name: str, args: Optional[argparse.Namespace] = None
) -> Optional[Dict[str, Any]]:
    """Load raw dataset data by directly accessing the data source."""
    import pandas as pd

    from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files

    if dataset_name not in get_supported_datasets():
        logger.warning(f"Dataset not supported: {dataset_name}")
        return None

    # for datasets that have special paths
    dataset_locations = {
        "traffic": "s3://synthefy-fm-eval-datasets/traffic_PeMS/",
        "rideshare_uber": "s3://synthefy-fm-eval-datasets/rideshare/uber/",
        "rideshare_lyft": "s3://synthefy-fm-eval-datasets/rideshare/lyft/",
        "tx_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "ne_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "ny_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "az_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "cal_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "nm_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "pa_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "tn_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "co_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "car_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "al_daily": f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/",
        "tn_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "pa_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "car_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "cal_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "tx_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "se_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "ne_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "az_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "id_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "or_electricity": f"s3://synthefy-fm-eval-datasets/hourly_electricity/{dataset_name}/",
        "fred_md1": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "fred_md2": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "fred_md3": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "fred_md4": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "fred_md5": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "fred_md6": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "fred_md7": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "fred_md8": f"s3://synthefy-fm-eval-datasets/fred_md/{dataset_name}/",
        "web_visitors": "s3://synthefy-fm-eval-datasets/website_visitors/",
        "ev_sensors": "s3://synthefy-fm-eval-datasets/ev-sensors/",
        "causal_rivers": "s3://synthefy-fm-eval-datasets/causalrivers/",
        "walmart_sales": "s3://synthefy-fm-eval-datasets/walmart-sales/",
        "central_electricity": "s3://synthefy-fm-eval-datasets/europe_electricity/central_electricity/",
        "eastern_electricity": "s3://synthefy-fm-eval-datasets/europe_electricity/eastern_electricity/",
        "western_electricity": "s3://synthefy-fm-eval-datasets/europe_electricity/western_electricity/",
        "southern_electricity": "s3://synthefy-fm-eval-datasets/europe_electricity/southern_electricity/",
        "northern_electricity": "s3://synthefy-fm-eval-datasets/europe_electricity/northern_electricity/",
        "cursor_tabs": "s3://synthefy-fm-eval-datasets/cursor-tabs/",
        "mujoco_halfcheetah_v2": [
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-random-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-expert-v2/",
        ],
        "mujoco_ant_v2": [
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-random-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-expert-v2/",
        ],
        "mujoco_hopper_v2": [
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-v2/",
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-expert-v2/",
        ],
        "mujoco_walker2d_v2": [
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-v2/",
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-expert-v2/",
        ],
    }

    # Use special mapping if needed, otherwise use default pattern
    data_location = dataset_locations.get(
        dataset_name, f"s3://synthefy-fm-eval-datasets/{dataset_name}/"
    )

    # Handle multiple paths for mujoco datasets
    if isinstance(data_location, list):
        csv_files = []
        for path in data_location:
            files = list_s3_files(path, file_extension=".csv")
            csv_files.extend(files)
    else:
        csv_files = list_s3_files(data_location, file_extension=".csv")

    if not csv_files:
        location_str = (
            data_location
            if isinstance(data_location, str)
            else ", ".join(data_location)
        )
        logger.warning(
            f"No CSV files found for {dataset_name} at {location_str}"
        )
        return None

    total_timestamps = 0
    total_data_points = 0
    num_covariates = 0
    timestamps_per_file = []

    first_df = pd.read_csv(csv_files[0])

    # dataset-specific timestamp columnhandling
    if dataset_name in ["traffic", "solar_alabama", "weather_mpi"]:
        # These datasets have year, month, day, hour, minute columns
        if all(
            col in first_df.columns
            for col in ["year", "month", "day", "hour", "minute"]
        ):
            first_df["timestamp"] = pd.to_datetime(
                first_df[["year", "month", "day", "hour", "minute"]]
            )
        first_df = first_df.drop(
            columns=["year", "month", "day", "hour", "minute"], errors="ignore"
        )
        non_timestamp_cols = [
            col for col in first_df.columns if col != "timestamp"
        ]
    elif dataset_name == "aus_electricity":
        # Australian electricity has specific format
        first_df = pd.read_csv(csv_files[0], delimiter=";")
        timestamp_col = first_df.columns[0]
        first_df[timestamp_col] = pd.to_datetime(
            first_df[timestamp_col], format="%d/%m/%Y %H:%M", dayfirst=True
        )
        first_df = first_df.rename(columns={timestamp_col: "timestamp"})
        first_df = first_df.iloc[:, :11]  # Take first 11 columns
        non_timestamp_cols = [
            col for col in first_df.columns if col != "timestamp"
        ]
    else:
        # Default: assume first column is timestamp
        timestamp_col = first_df.columns[0]
        first_df = first_df.rename(columns={timestamp_col: "timestamp"})
        non_timestamp_cols = [
            col for col in first_df.columns if col != "timestamp"
        ]

    if len(non_timestamp_cols) > 0:
        metadata_cols = non_timestamp_cols
        num_covariates = len(metadata_cols)

    # Calculate data frequency from first two timestamps
    data_frequency = "unknown"
    if len(first_df) >= 2 and "timestamp" in first_df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(first_df["timestamp"]):
                first_df["timestamp"] = pd.to_datetime(first_df["timestamp"])

            time_diff = (
                first_df["timestamp"].iloc[1] - first_df["timestamp"].iloc[0]
            )

            if time_diff.total_seconds() < 60:
                data_frequency = f"{int(time_diff.total_seconds())}s"
            elif time_diff.total_seconds() < 3600:
                data_frequency = f"{int(time_diff.total_seconds() / 60)}m"
            elif time_diff.total_seconds() < 86400:
                data_frequency = f"{int(time_diff.total_seconds() / 3600)}h"
            elif time_diff.days < 7:
                data_frequency = f"{time_diff.days}d"
            elif time_diff.days < 30:
                data_frequency = f"{int(time_diff.days / 7)}w"
            elif time_diff.days < 365:
                data_frequency = f"{int(time_diff.days / 30)}M"
            else:
                data_frequency = f"{int(time_diff.days / 365)}Y"
        except Exception as e:  # some datasets are just formatted weirdly, so we can manually set the frequency
            logger.warning(
                f"Could not determine data frequency for {dataset_name}: {e}"
            )
            data_frequency = "unknown"

    for csv_file in csv_files:
        if dataset_name in ["traffic", "solar_alabama", "weather_mpi"]:
            df = pd.read_csv(csv_file)
            df = df.drop(
                columns=["year", "month", "day", "hour", "minute"],
                errors="ignore",
            )
        elif dataset_name == "aus_electricity":
            df = pd.read_csv(csv_file, delimiter=";")
            timestamp_col = df.columns[0]
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col], format="%d/%m/%Y %H:%M", dayfirst=True
            )
            df = df.rename(columns={timestamp_col: "timestamp"})
            df = df.iloc[:, :11]
        else:
            df = pd.read_csv(csv_file)
            timestamp_col = df.columns[0]
            df = df.rename(columns={timestamp_col: "timestamp"})

        file_timestamps = len(df)
        timestamps_per_file.append(file_timestamps)
        total_timestamps += file_timestamps

        data_columns = [col for col in df.columns if col != "timestamp"]
        total_data_points += len(df) * len(data_columns)

    num_files = len(timestamps_per_file)
    avg_timestamps = total_timestamps / num_files if num_files > 0 else 0
    min_timestamps = min(timestamps_per_file) if timestamps_per_file else 0
    max_timestamps = max(timestamps_per_file) if timestamps_per_file else 0

    if min_timestamps != max_timestamps:
        logger.info(
            f"{dataset_name} dataset: {num_files} files, avg timestamps: {avg_timestamps:.1f}, range: {min_timestamps}-{max_timestamps}"
        )
    else:
        logger.info(
            f"{dataset_name} dataset: {num_files} files, {total_timestamps} timestamps, {total_data_points} data points"
        )

    return {
        "num_covariates": num_covariates,
        "num_files": num_files,
        "avg_timestamps": round(avg_timestamps),
        "min_timestamps": min_timestamps,
        "max_timestamps": max_timestamps,
        "total_data_points": total_data_points,
        "data_frequency": data_frequency,
    }


def extract_dataset_metadata(
    dataset_name: str, args: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    metadata = {
        "dataset_name": dataset_name,
        "total_data_points": 0,
        "avg_timestamps": 0,
        "min_timestamps": 0,
        "max_timestamps": 0,
        "num_files": 0,
        "num_covariates": 0,
        "data_frequency": "unknown",
    }

    raw_data = load_raw_dataset_data(dataset_name, args)
    if raw_data is None:
        logger.warning(f"Could not load raw data for {dataset_name}")
        return metadata

    metadata["num_covariates"] = raw_data["num_covariates"]
    metadata["num_files"] = raw_data["num_files"]
    metadata["avg_timestamps"] = raw_data["avg_timestamps"]
    metadata["min_timestamps"] = raw_data["min_timestamps"]
    metadata["max_timestamps"] = raw_data["max_timestamps"]
    metadata["total_data_points"] = raw_data["total_data_points"]
    metadata["data_frequency"] = raw_data["data_frequency"]

    return metadata


def extract_model_mape_values(
    forecast_data: Dict[str, Any],
) -> Dict[str, float]:
    model_mape_values = {}

    models = forecast_data.get("models", {})
    for model_name, model_data in models.items():
        if isinstance(model_data, dict) and "metrics" in model_data:
            metrics = model_data["metrics"]
            if isinstance(metrics, dict) and "mape" in metrics:
                mape_value = metrics["mape"]
                if isinstance(mape_value, (int, float)):
                    model_mape_values[model_name] = float(mape_value)

    return model_mape_values


def generate_aggregated_json(
    dataset_name: str,
    forecast_data: Optional[Dict[str, Any]] = None,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    metadata = extract_dataset_metadata(dataset_name, args)

    # Extract model MAPE values if forecast data is provided
    new_model_mape_values = {}
    if forecast_data:
        new_model_mape_values = extract_model_mape_values(forecast_data)

    s3_key = f"aggregated/{dataset_name}_aggregated.json"
    existing_data = read_existing_forecasts_from_s3(
        "synthefy-fm-dataset-forecasts", s3_key
    )

    # Always create fresh aggregated data with updated metadata
    aggregated_data = {
        "dataset_name": metadata["dataset_name"],
        "total_data_points": metadata["total_data_points"],
        "avg_timestamps": metadata["avg_timestamps"],
        "min_timestamps": metadata["min_timestamps"],
        "max_timestamps": metadata["max_timestamps"],
        "num_files": metadata["num_files"],
        "num_covariates": metadata["num_covariates"],
        "data_frequency": metadata["data_frequency"],
        "model_mape_values": new_model_mape_values,
    }

    # If existing data has model metrics, preserve them and add new ones
    if existing_data and "model_mape_values" in existing_data:
        existing_model_mape_values = existing_data.get("model_mape_values", {})
        # Update with new values, preserving existing ones
        existing_model_mape_values.update(new_model_mape_values)
        aggregated_data["model_mape_values"] = existing_model_mape_values
        logger.info(
            f"Found existing model metrics for {dataset_name}, preserving them and adding new ones"
        )
    else:
        logger.info(
            f"No existing model metrics found for {dataset_name}, using only new metrics"
        )

    # Always overwrite the entire JSON
    write_forecasts_to_s3(
        "synthefy-fm-dataset-forecasts", s3_key, aggregated_data
    )
    logger.info(
        f"Overwrote aggregated JSON in S3: s3://synthefy-fm-dataset-forecasts/{s3_key}"
    )

    print(f"\n--- Aggregated JSON for {dataset_name} ---")
    print(json.dumps(aggregated_data, indent=2))
    print("--- End of JSON ---\n")

    return aggregated_data


def process_dataset(
    dataset_name: str, args: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    logger.info(f"Processing dataset: {dataset_name}")

    # Try to load forecast data for MAPE values, but don't fail if not available
    forecast_data = None
    try:
        forecast_data = load_forecast_data(
            dataset_name, "synthefy-fm-dataset-forecasts"
        )
    except Exception as e:
        logger.warning(f"Could not load forecast data for {dataset_name}: {e}")

    aggregated_data = generate_aggregated_json(
        dataset_name, forecast_data, args
    )
    return {
        "dataset_name": dataset_name,
        "status": "success",
        "aggregated_data": aggregated_data,
    }


def process_datasets(
    datasets: List[str], args: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    logger.info(f"Processing {len(datasets)} datasets")

    all_results = {}
    successful_datasets = 0
    failed_datasets = 0

    for dataset_name in tqdm(datasets, desc="Processing datasets"):
        dataset_results = process_dataset(dataset_name, args)
        all_results[dataset_name] = dataset_results

        if dataset_results["status"] == "success":
            successful_datasets += 1
        else:
            failed_datasets += 1

    summary = {
        "total_datasets": len(datasets),
        "successful_datasets": successful_datasets,
        "failed_datasets": failed_datasets,
    }

    return {
        "summary": summary,
        "results": all_results,
    }


def print_summary(results: Dict[str, Any]) -> None:
    summary = results["summary"]

    print("\n" + "=" * 80)
    print("DATASET PROCESSING SUMMARY")
    print(f"Total datasets: {summary['total_datasets']}")
    print(f"Successful datasets: {summary['successful_datasets']}")
    print(f"Failed datasets: {summary['failed_datasets']}")


def main():
    args = parse_arguments()
    validate_arguments(args)

    if args.all_datasets:
        datasets = get_supported_datasets()
        if not datasets:
            logger.error("No supported datasets found")
            sys.exit(1)
        logger.info(f"Processing all {len(datasets)} supported datasets")
    else:
        datasets = args.datasets
        logger.info(f"Processing {len(datasets)} specified datasets")

    results = process_datasets(datasets, args)
    print_summary(results)


if __name__ == "__main__":
    main()
