import argparse
import os
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import yaml
from botocore.exceptions import ClientError
from loguru import logger

from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat

MAP_NAME_TO_DISPLAY_NAME = {
    "gpt-4o": "GPT-4o",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
}


def get_test_samples(
    batch: EvalBatchFormat,
    test_samples: int,
) -> EvalBatchFormat:
    """
    Extract a number oftest samples from a batch.
    """
    if test_samples <= 0:
        raise ValueError("test_samples must be positive")
    if test_samples >= (batch.batch_size * batch.num_correlates):
        return batch

    samples_per_batch = test_samples // batch.batch_size
    remaining_samples = test_samples % batch.batch_size

    extracted_samples = []

    for batch_idx in range(batch.batch_size):
        if batch_idx < remaining_samples:
            samples_to_take = samples_per_batch + 1
        else:
            samples_to_take = samples_per_batch

        if samples_to_take == 0:
            continue

        row = []
        for idx in range(samples_to_take):
            if idx >= batch.num_correlates:
                break  # if we've taken all correlates, break
            row.append(batch.samples[batch_idx][idx])

        extracted_samples.append(row)

    return EvalBatchFormat(extracted_samples)


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    Parse an S3 URL and return the bucket name and key (object path or prefix).

    Args:
        s3_url (str): The S3 URL in the format 's3://bucket-name/key/path'.

    Returns:
        tuple[str, str]: A tuple containing the bucket name and the S3 key (object path or prefix).

    Example:
        >>> parse_s3_url('s3://my-bucket/data/file.txt')
        ('my-bucket', 'data/file.txt')
    """
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def download_checkpoint_from_s3(s3_url: str) -> str:
    """
    Download a checkpoint file from S3 and return the local path.

    Args:
        s3_url (str): S3 URL of the checkpoint file

    Returns:
        str: Local path to the downloaded checkpoint file

    Raises:
        ValueError: If the S3 URL is invalid or download fails
    """
    try:
        # Parse S3 URL
        bucket, key = parse_s3_url(s3_url)

        # Create S3 client
        s3_client = boto3.client("s3")

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w+b", delete=False, suffix=".ckpt"
        ) as tmp_file:
            local_path = tmp_file.name

        # Download file from S3
        logger.info(f"Downloading checkpoint from {s3_url} to {local_path}...")
        s3_client.download_file(bucket, key, local_path)

        # Verify download
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logger.info(f"Successfully downloaded checkpoint to {local_path}")
            return local_path
        else:
            raise ValueError(
                f"Download appeared to succeed but file is missing or empty: {local_path}"
            )

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            raise ValueError(f"Checkpoint not found in S3: {s3_url}")
        else:
            raise ValueError(f"S3 error downloading checkpoint: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error downloading checkpoint from S3: {str(e)}")


ARGPARSE_EPILOG = """
Examples:
  # Using eval-config file
  python eval.py --eval-config config.yaml

  # Using individual arguments
  python eval.py --dataset traffic --models tabpfn_univariate tabpfn_multivariate --output-directory /path/to/results

Dataset-Specific Required Arguments:

fmv3 dataset:
  Required: --config-file
  Example: python eval.py --dataset fmv3 --models prophet causal_impact --config-file config.yaml --output-directory /path/to/results --output-type pkl csv

traffic dataset:
  No additional parameters required (loads data directly)
  Example: python eval.py --dataset traffic --models tabpfn_univariate tabpfn_multivariate --output-directory /path/to/results --output-type pkl h5

solar_alabama dataset:
  No additional parameters required (loads data directly)
  Example: python eval.py --dataset solar_alabama --models prophet --output-directory /path/to/results --output-type csv

weather_mpi dataset:
  No additional parameters required (loads data directly)
  Example: python eval.py --dataset weather_mpi --models prophet --output-directory /path/to/results --output-type pkl

gift dataset:
  Required: --data-path, --forecast-length, --sub-dataset
  Optional: --history-length
  Example: python eval.py --dataset gift --models tabpfn_univariate --data-path /path/to/data --forecast-length 24 --history-length 168 --sub-dataset traffic --output-directory /path/to/results --output-type pkl csv h5

goodrx dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset goodrx --models prophet tabpfn_univariate --output-directory /path/to/results --output-type pkl csv h5

spain_energy dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset spain_energy --models prophet tabpfn_univariate --output-directory /path/to/results --output-type pkl csv h5

gpt-synthetic dataset:
  Required: --dataset-name
  Example: python eval.py --dataset gpt-synthetic --dataset-name retail --models prophet tabpfn_univariate --output-directory /path/to/results --output-type pkl csv h5

beijing_embassy dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset beijing_embassy --models prophet --output-directory /path/to/results --output-type pkl csv h5

ercot_load dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset ercot_load --models prophet --output-directory /path/to/results --output-type pkl csv h5

open_aq dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset open_aq --models prophet --output-directory /path/to/results --output-type pkl csv h5

beijing_aq dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset beijing_aq --models prophet --output-directory /path/to/results --output-type pkl csv h5

cgm dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset cgm --models prophet --output-directory /path/to/results --output-type pkl csv h5

mn_interstate dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset mn_interstate --models prophet --output-directory /path/to/results --output-type pkl csv h5

cursor_tabs dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset cursor_tabs --models prophet --output-directory /path/to/results --output-type pkl csv h5

walmart_sales dataset:
  No additional parameters required (loads data from S3)
  Example: python eval.py --dataset walmart_sales --models prophet --output-directory /path/to/results --output-type pkl csv h5

synthetic_medium_lag dataset:
  Required: --data-path
  Example: python eval.py --dataset synthetic_medium_lag --models gridicl_univariate --data-path /path/to/data --output-directory /path/to/results --forecast-length 128 --history-length 640

external dataset:
  Required: --external-dataloader-spec
  Example: python eval.py --dataset external --external-dataloader-spec "/path/to/my_dataloader.py::MyCustomDataloader" --models prophet --output-directory /path/to/results --output-type pkl csv

Model-Specific Notes:

sfm_forecaster model:
  Required: --config-file, --history-length, --forecast-length, and either --model-checkpoint-path OR --model-ckpt
  Example with local checkpoint: python eval.py --dataset fmv3 --models sfm_forecaster --config-file config.yaml --model-checkpoint-path /path/to/checkpoint.ckpt --history-length 128 --forecast-length 128 --output-directory /path/to/results --output-type pkl csv
  Example with S3 checkpoint: python eval.py --dataset fmv3 --models sfm_forecaster --config-file config.yaml --model-ckpt s3://synthefy-mlflow-artifacts/.../.ckpt --history-length 128 --forecast-length 128 --output-directory /path/to/results --output-type pkl h5 csv

tabpfn_boosting model:
  Optional: --boosting-models (defaults to ["prophet"])
  Example: python eval.py --dataset fmv3 --models tabpfn_boosting --config-file config.yaml --boosting-models prophet tabpfn_multivariate --output-directory /path/to/results

mitra_boosting model:
  Optional: --boosting-models (defaults to ["prophet"])
  Example: python eval.py --dataset fmv3 --models mitra_boosting --config-file config.yaml --boosting-models prophet tabpfn_multivariate --output-directory /path/to/results

toto model:
  Optional: server URL set via env var TOTO_SERVER_URL (default: http://localhost:8000)
  Example: python eval.py --dataset fmv3 --models toto --config-file config.yaml --output-directory /path/to/results --output-type pkl csv

mitra model:
  Optional: server URL set via env var MITRA_SERVER_URL (default: http://localhost:8001)
  Example: python eval.py --dataset fmv3 --models mitra --config-file config.yaml --output-directory /path/to/results --output-type pkl csv

llm model:
  Optional: --llm-model-names (default: gemini-2.0-flash)
  Example: python eval.py --dataset fmv3 --models llm --config-file config.yaml --output-directory /path/to/results --output-type pkl csv
  Example with custom model: python eval.py --dataset fmv3 --models llm --config-file config.yaml --llm-model-names gemini-2.0-flash --output-directory /path/to/results --output-type pkl csv
  Example with multiple models: python eval.py --dataset fmv3 --models llm --config-file config.yaml --llm-model-names gemini-2.0-flash gpt-4o --output-directory /path/to/results --output-type pkl csv

sarima model:
  Required: --seasonal-period
  Example: python eval.py --dataset fmv3 --models sarima --config-file config.yaml --seasonal-period 12 --output-directory /path/to/results --output-type pkl csv

sarimax_future_leaked model:
  Required: --seasonal-period
  Example: python eval.py --dataset fmv3 --models sarimax_future_leaked --config-file config.yaml --seasonal-period 12 --output-directory /path/to/results --output-type pkl csv
"""
