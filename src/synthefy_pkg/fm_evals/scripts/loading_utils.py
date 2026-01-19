"""
Utilities for loading forecast metrics and aggregated results from S3 buckets.

Usage:

from synthefy_pkg.fm_evals.scripts.loading_utils import extract_forecast_metrics, extract_aggregated_metrics

# Extract all forecast metrics
extract_forecast_metrics("/path/to/output/forecast_metrics.json")

# Extract forecast metrics for specific models
extract_forecast_metrics("/path/to/output/energy_models", models=["tabpfn_multivariate", "tabpfn_uniivariate"])

# Extract forecast metrics for specific domains
extract_forecast_metrics("/path/to/output/energy_finance", domains=["Energy", "Finance"])

# Extract forecast metrics for specific datasets
extract_forecast_metrics("/path/to/output/specific_datasets", datasets=["solar_alabama", "bitcoin_price"])

# Extract aggregated metrics -- total_data_points, avg_timestamps, num_files, num_covariates, data_frequency
extract_aggregated_metrics("/path/to/output/aggregated_metrics")

# Extract aggregated metrics with filters
extract_aggregated_metrics("/path/to/output/energy_aggregated", models=["model1"], domains=["Energy"])
"""

import json
import os
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def _load_domains_mapping() -> Dict[str, str]:
    """Load the domains mapping from domains.json file."""
    domains_file = "/home/synthefy/synthefy-package/src/synthefy_pkg/fm_evals/scripts/domains.json"
    try:
        with open(domains_file, "r") as f:
            domains_data = json.load(f)

        # Create a mapping from dataset name to domain
        dataset_to_domain = {}
        for domain_group in domains_data.get("datasets", []):
            domain = domain_group.get("domain")
            datasets = domain_group.get("dataset_name", [])
            for dataset in datasets:
                dataset_to_domain[dataset] = domain

        return dataset_to_domain
    except FileNotFoundError:
        print(f"Warning: Domains file not found at {domains_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing domains file: {e}")
        return {}


def _list_s3_objects(bucket_name: str, prefix: str) -> List[str]:
    """List all objects in S3 bucket with given prefix."""
    try:
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        contents = response.get("Contents", [])
        return [obj.get("Key", "") for obj in contents if obj.get("Key")]
    except (ClientError, NoCredentialsError) as e:
        print(f"Error listing S3 objects: {e}")
        return []


def _read_json_from_s3(bucket_name: str, key: str) -> Optional[Dict[str, Any]]:
    """Read JSON data from S3."""
    try:
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except (ClientError, NoCredentialsError, json.JSONDecodeError) as e:
        print(f"Error reading JSON from S3 key {key}: {e}")
        return None


def _filter_by_models(
    data: Dict[str, Any], models: List[str]
) -> Dict[str, Any]:
    """Filter data to include only specified models."""
    if not models:
        return data

    filtered_data = data.copy()
    if "models" in filtered_data:
        filtered_models = {}
        for model in models:
            if model in filtered_data["models"]:
                filtered_models[model] = filtered_data["models"][model]
        filtered_data["models"] = filtered_models

    return filtered_data


def _filter_by_datasets(
    data: Dict[str, Any], datasets: List[str]
) -> Optional[Dict[str, Any]]:
    """Filter data to include only specified datasets."""
    if not datasets:
        return data

    if "dataset_name" in data and data["dataset_name"] not in datasets:
        return None

    return data


def _filter_by_domains(
    data: Dict[str, Any], domains: List[str], domains_mapping: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Filter data to include only specified domains."""
    if not domains:
        return data

    dataset_name = data.get("dataset_name")
    if not dataset_name:
        return data

    dataset_domain = domains_mapping.get(dataset_name)
    if dataset_domain not in domains:
        return None

    return data


def _extract_metrics_only(models_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only MAE and MAPE metrics from models data, removing forecast data."""
    filtered_models = {}
    for model_name, model_data in models_data.items():
        if isinstance(model_data, dict) and "metrics" in model_data:
            # Extract only MAE and MAPE from metrics
            metrics = model_data["metrics"]
            if metrics is not None and isinstance(metrics, dict):
                filtered_models[model_name] = {
                    "mae": metrics.get("mae"),
                    "mape": metrics.get("mape"),
                }
            else:
                # Handle case where metrics is None or not a dict
                filtered_models[model_name] = {"mae": None, "mape": None}
        else:
            if isinstance(model_data, dict):
                filtered_models[model_name] = {
                    "mae": model_data.get("mae"),
                    "mape": model_data.get("mape"),
                }
            else:
                filtered_models[model_name] = {"mae": None, "mape": None}
    return filtered_models


def extract_forecast_metrics(
    output_path: str,
    bucket_name: str = "synthefy-fm-dataset-forecasts",
    models: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> str:
    """
    Parameters
    ----------
    output_path : str
        Full path to save the output JSON file (including filename)
    bucket_name : str
        Name of the S3 bucket containing forecast data
    models : Optional[List[str]]
        List of model names to filter by
    domains : Optional[List[str]]
        List of domain names to filter by
    datasets : Optional[List[str]]
        List of dataset names to filter by

    Returns
    -------
    str
        Path to the output JSON file
    """
    print("Starting forecast metrics extraction...")

    domains_mapping = _load_domains_mapping()

    # List all forecast files
    forecast_files = _list_s3_objects(bucket_name, "")
    forecast_jsons = [
        f for f in forecast_files if f.endswith("_forecasts.json")
    ]
    metrics_jsons = [
        f for f in forecast_files if f.endswith("_forecast_metrics.json")
    ]

    print(
        f"Found {len(forecast_jsons)} forecast files and {len(metrics_jsons)} metrics files"
    )

    combined_metrics = {}

    for file_key in forecast_jsons:
        dataset_name = file_key.replace("_forecasts.json", "")

        if datasets and dataset_name not in datasets:
            continue

        dataset_domain = domains_mapping.get(dataset_name, "Unknown")
        if domains and dataset_domain not in domains:
            continue

        print(f"Processing forecast file: {file_key}")
        data = _read_json_from_s3(bucket_name, file_key)
        if not data:
            continue

        filtered_data = _filter_by_models(data, models or [])

        models_data = filtered_data.get("models", {})
        filtered_models = _extract_metrics_only(models_data)

        combined_metrics[dataset_name] = {
            "domain": dataset_domain,
            "dataset_name": dataset_name,
            "models": filtered_models,
        }

    for file_key in metrics_jsons:
        dataset_name = file_key.replace("_forecast_metrics.json", "")

        if datasets and dataset_name not in datasets:
            continue

        dataset_domain = domains_mapping.get(dataset_name, "Unknown")
        if domains and dataset_domain not in domains:
            continue

        print(f"Processing metrics file: {file_key}")
        data = _read_json_from_s3(bucket_name, file_key)
        if not data:
            continue

        filtered_data = _filter_by_models(data, models or [])

        models_data = filtered_data.get("models", {})
        filtered_models = _extract_metrics_only(models_data)

        combined_metrics[dataset_name] = {
            "domain": dataset_domain,
            "dataset_name": dataset_name,
            "models": filtered_models,
        }

    if not output_path.endswith(".json"):
        output_path += ".json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)

    print(f"Saved combined forecast metrics to {output_path}")
    print(f"Processed {len(combined_metrics)} datasets")

    return output_path


def extract_aggregated_metrics(
    output_path: str,
    bucket_name: str = "synthefy-fm-dataset-forecasts",
    models: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> str:
    """
    Parameters
    ----------
    output_path : str
        Full path to save the output JSON file (including filename)
    bucket_name : str
        Name of the S3 bucket containing forecast data
    models : Optional[List[str]]
        List of model names to filter by
    domains : Optional[List[str]]
        List of domain names to filter by
    datasets : Optional[List[str]]
        List of dataset names to filter by

    Returns
    -------
    str
        Path to the output JSON file
    """
    print("Starting aggregated metrics extraction...")

    # Load domains mapping
    domains_mapping = _load_domains_mapping()

    # List all aggregated files
    aggregated_files = _list_s3_objects(bucket_name, "aggregated/")
    aggregated_jsons = [
        f for f in aggregated_files if f.endswith("_aggregated.json")
    ]

    print(f"Found {len(aggregated_jsons)} aggregated files")

    combined_aggregated = {}

    # Process aggregated files
    for file_key in aggregated_jsons:
        dataset_name = file_key.replace("aggregated/", "").replace(
            "_aggregated.json", ""
        )

        if datasets and dataset_name not in datasets:
            continue

        dataset_domain = domains_mapping.get(dataset_name, "Unknown")
        if domains and dataset_domain not in domains:
            continue

        print(f"Processing aggregated file: {file_key}")
        data = _read_json_from_s3(bucket_name, file_key)
        if not data:
            continue

        combined_aggregated[dataset_name] = {
            "dataset_name": data.get("dataset_name", dataset_name),
            "total_data_points": data.get("total_data_points"),
            "avg_timestamps": data.get("avg_timestamps"),
            "num_files": data.get("num_files"),
            "num_covariates": data.get("num_covariates"),
            "data_frequency": data.get("data_frequency"),
            "domain": dataset_domain,
        }

    if not output_path.endswith(".json"):
        output_path += ".json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(combined_aggregated, f, indent=2)

    print(f"Saved combined aggregated metrics to {output_path}")
    print(f"Processed {len(combined_aggregated)} datasets")

    return output_path
