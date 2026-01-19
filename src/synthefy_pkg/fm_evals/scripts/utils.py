# general s3/file handling utils, defining things for testing, etc.

import json
import os
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from loguru import logger
from tqdm import tqdm


def get_s3_client():
    return boto3.client("s3")


def read_existing_forecasts_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    """Read existing forecasts from S3, return empty dict if not found."""
    try:
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            logger.info(f"No existing forecasts found at s3://{bucket}/{key}")
            return {}
        else:
            logger.error(f"Error reading from S3: {e}")
            raise
    except Exception as e:
        logger.error(f"Error reading forecasts from S3: {e}")
        raise


def write_forecasts_to_s3(
    bucket: str, key: str, forecasts: Dict[str, Any]
) -> None:
    try:
        s3_client = get_s3_client()
        content = json.dumps(forecasts, indent=2)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            ContentType="application/json",
        )
        logger.info(f"Successfully wrote forecasts to s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Error writing forecasts to S3: {e}")
        raise


def get_datasets_in_s3(s3_bucket: str) -> List[str]:
    try:
        s3_client = get_s3_client()
        response = s3_client.list_objects_v2(Bucket=s3_bucket)

        if "Contents" not in response:
            logger.warning(f"No objects found in S3 bucket {s3_bucket}")
            return []

        datasets = []
        for obj in response["Contents"]:
            key = obj.get("Key", "")
            # Check if it's a forecast file (ends with _forecasts.json)
            if key.endswith("_forecasts.json"):
                dataset_name = key[: -len("_forecasts.json")]
                datasets.append(dataset_name)

        logger.info(
            f"Found {len(datasets)} datasets in S3 bucket {s3_bucket}: {datasets}"
        )
        return sorted(datasets)

    except Exception as e:
        logger.error(
            f"Failed to discover datasets in S3 bucket {s3_bucket}: {e}"
        )
        return []


def load_forecast_data(
    dataset_name: str, s3_bucket: str
) -> Optional[Dict[str, Any]]:
    s3_key = f"{dataset_name}_forecasts.json"

    try:
        forecast_data = read_existing_forecasts_from_s3(s3_bucket, s3_key)
        if forecast_data:
            logger.info(f"Loaded forecast data from S3 for {dataset_name}")
            return forecast_data
    except Exception as e:
        logger.warning(f"Could not load from S3 for {dataset_name}: {e}")

    # for local testing purposes
    try:
        forecast_data = read_existing_forecasts_locally(dataset_name)
        if forecast_data:
            logger.info(
                f"Loaded forecast data from local storage for {dataset_name}"
            )
            return forecast_data
    except Exception as e:
        logger.warning(
            f"Could not load from local storage for {dataset_name}: {e}"
        )

    logger.warning(f"No forecast data found for {dataset_name}")
    return None


# for local testing purposes
def read_existing_forecasts_locally(dataset_name: str) -> Dict[str, Any]:
    """Read existing forecasts from local file, return empty dict if not found."""
    local_dir = (
        "/home/synthefy/synthefy-package/src/synthefy_pkg/fm_evals/test_results"
    )
    local_file = os.path.join(local_dir, f"{dataset_name}_forecasts.json")

    try:
        if os.path.exists(local_file):
            with open(local_file, "r") as f:
                return json.load(f)
        else:
            logger.info(f"No existing local forecasts found at {local_file}")
            return {}
    except Exception as e:
        logger.error(f"Error reading local forecasts: {e}")
        return {}


# for local testing purposes
def save_forecasts_locally(
    dataset_name: str, forecasts: Dict[str, Any]
) -> None:
    """Save forecasts locally in test_results directory."""
    local_dir = (
        "/home/synthefy/synthefy-package/src/synthefy_pkg/fm_evals/test_results"
    )
    os.makedirs(local_dir, exist_ok=True)

    local_file = os.path.join(local_dir, f"{dataset_name}_forecasts.json")
    try:
        with open(local_file, "w") as f:
            json.dump(forecasts, f, indent=2)
        logger.info(f"Successfully saved forecasts locally to {local_file}")
    except Exception as e:
        logger.error(f"Error saving forecasts locally: {e}")
        raise
