#!/usr/bin/env python3
"""
Script to download JSON files from S3 and run aggregation statistics.

Downloads files from S3 with structure: fm-eval-datasets/analysis/<name>/<name>/0/<name>_summary_data.json
Runs aggregate_predictability_statistics and aggregate_multivariate_predictability_statistics
Saves results to fm_evals/ directory.
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Add the parent directory to the path to import aggregate_statistics
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.preprocessing.data_summary_utils.aggregate_statistics import (
    aggregate_multivariate_predictability_statistics,
    aggregate_predictability_statistics,
    write_json,
)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file, return empty dict if file doesn't exist."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}


def download_from_s3(
    dataset_names: List[str],
    local_dir: str,
    s3_base_path: str = "s3://synthefy-fm-eval-datasets/analysis/",
    dataset_name: str | None = None,
) -> List[str]:
    """Download JSON files from S3 bucket."""
    os.makedirs(local_dir, exist_ok=True)

    all_s3_files = []
    base_path = s3_base_path.rstrip("/")

    for dataset in dataset_names:
        if dataset == "gpt-synthetic" and dataset_name:
            s3_path = f"{base_path}/{dataset}-{dataset_name}"
        else:
            s3_path = f"{base_path}/{dataset}"

        print(f"Listing JSON files in {s3_path}")
        json_files = list_s3_files(s3_path, file_extension=".json")
        all_s3_files.extend(json_files)

    print(f"Found {len(all_s3_files)} JSON files to download")

    downloaded_files = []
    s3_client = boto3.client("s3")

    for s3_file_url in all_s3_files:
        filename = os.path.basename(s3_file_url)
        local_file_path = os.path.join(local_dir, filename)

        from urllib.parse import urlparse

        parsed = urlparse(s3_file_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        print(f"Downloading {s3_file_url} to {local_file_path}")
        s3_client.download_file(bucket, key, local_file_path)
        downloaded_files.append(local_file_path)
        print(f"Successfully downloaded {filename}")

    return downloaded_files


def run_aggregation_statistics(
    downloaded_files: List[str],
    output_dir: str,
    dataset_names: List[str],
    dataset_name: str | None = None,
    append_results: bool = False,
):
    """Run aggregation statistics on downloaded files."""
    if not downloaded_files:
        print("No files to process. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {len(downloaded_files)} files...")
    for file_path in downloaded_files:
        print(f"  - {file_path}")

    # Create dataset key for predictability results
    if (
        len(dataset_names) == 1
        and dataset_names[0] == "gpt-synthetic"
        and dataset_name
    ):
        result_key = f"{dataset_names[0]}-{dataset_name}"
    elif len(dataset_names) == 1:
        result_key = dataset_names[0]
    else:
        result_key = "_".join(dataset_names)

    # Run predictability statistics
    print("\nRunning aggregate_predictability_statistics...")
    predictability_results = aggregate_predictability_statistics(
        downloaded_files
    )
    predictability_output = os.path.join(
        output_dir, "predictability_statistics.json"
    )

    if append_results:
        existing = load_json(predictability_output)
        existing[result_key] = predictability_results
        predictability_results = existing
        print(f"Appended results for dataset: {result_key}")
    else:
        predictability_results = {result_key: predictability_results}

    write_json(predictability_results, predictability_output)
    print(f"Predictability statistics saved to: {predictability_output}")

    # Run multivariate predictability statistics
    print("\nRunning aggregate_multivariate_predictability_statistics...")
    multivariate_results = aggregate_multivariate_predictability_statistics(
        downloaded_files
    )
    multivariate_output = os.path.join(
        output_dir, "multivariate_predictability_statistics.json"
    )

    # Fix empty keys by using our dataset names
    if multivariate_results and list(multivariate_results.keys())[0] == "":
        # If the key is empty, replace it with our result_key
        multivariate_results = {
            result_key: list(multivariate_results.values())[0]
        }

    if append_results:
        existing = load_json(multivariate_output)
        # multivariate_results is already a dict with dataset names as keys and single float values
        existing.update(multivariate_results)
        multivariate_results = existing
        print(
            f"Appended multivariate results for datasets: {list(multivariate_results.keys())}"
        )
    else:
        # Convert OrderedDict to regular dict for JSON serialization
        multivariate_results = dict(multivariate_results)

    write_json(multivariate_results, multivariate_output)
    print(
        f"Multivariate predictability statistics saved to: {multivariate_output}"
    )

    return predictability_results, multivariate_results


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="Download JSON files from S3 and run aggregation statistics"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="synthefy-fm-eval-datasets",
        help="S3 bucket name",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="List of dataset names to download and process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fm_evals",
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for downloaded files",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep downloaded files after processing",
    )
    parser.add_argument(
        "--s3-base-path",
        type=str,
        default="s3://synthefy-fm-eval-datasets/analysis/",
        help="Base path for S3 bucket",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Specific dataset name for gpt-synthetic dataset",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Append results to existing files instead of overwriting",
    )

    args = parser.parse_args()

    # Set up temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="s3_download_")

    print(f"Using temporary directory: {temp_dir}")
    print(f"Processing datasets: {args.datasets}")
    print(f"Output directory: {args.output_dir}")

    # Download files from S3
    print(f"\nDownloading files from S3 bucket: {args.bucket}")
    downloaded_files = download_from_s3(
        args.datasets, temp_dir, args.s3_base_path, args.dataset_name
    )

    if not downloaded_files:
        print("No files were successfully downloaded. Exiting.")
        return

    # Run aggregation statistics
    run_aggregation_statistics(
        downloaded_files,
        args.output_dir,
        args.datasets,
        args.dataset_name,
        args.append_results,
    )

    print(f"\nProcessing complete! Results saved to: {args.output_dir}")

    # Clean up temporary files unless --keep-files is specified
    if not args.keep_files and temp_dir != args.temp_dir:
        import shutil

        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    elif args.keep_files:
        print(f"Downloaded files kept in: {temp_dir}")


if __name__ == "__main__":
    main()
