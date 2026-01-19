import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import aioboto3
import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException, Request, status
from loguru import logger
from mypy_boto3_s3 import S3Client
from pydantic import ValidationError

from synthefy_pkg.app.data_models import S3Source
from synthefy_pkg.app.utils.api_utils import get_train_config_file_name

COMPILE = True
PRESIGNED_URL_EXPIRATION = 604800  # expires in 1 week
S3_OPERATION_TIMEOUT = 120  # seconds
S3_DOWNLOAD_RETRIES = 3  # number of retry attempts for failed downloads
S3_RETRY_DELAY = 2  # seconds to wait between retry attempts
S3_DELETE_BATCH_SIZE = 1000  # S3 limit for batch delete operations


def parse_s3_url(s3_url: str) -> Tuple[str, str]:
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


def upload_directory_to_s3(
    s3_client: S3Client,
    bucket: str,
    local_dir: str,
    s3_dir: str,
    clear_local_dir: bool = True,
) -> bool:
    """
    Upload the entire content of a given local directory to the specified S3 directory and clear local directory.
    Parameters:
        - s3_client: S3Client
        - bucket: bucket name
        - local_dir: local directory path to upload the files from
        - s3_directory: S3 directory path to save the files
        - clear_local_dir: whether to clear the local directory after upload
    Returns:
        - bool: True if successful, False otherwise
    """
    try:
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_file_path, local_dir)
                s3_key = os.path.join(s3_dir, relative_path)

                try:
                    s3_client.upload_file(local_file_path, bucket, s3_key)
                    logger.info(
                        f"Uploaded {local_file_path} to s3://{bucket}/{s3_key}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to upload {local_file_path}: {str(e)}"
                    )
                    return False

        # Clear local directory after successful upload
        if clear_local_dir:
            for root, dirs, files in os.walk(local_dir, topdown=False):
                for filename in files:
                    os.remove(os.path.join(root, filename))
            for dirname in dirs:
                os.rmdir(os.path.join(root, dirname))

        # Remove the local_dir itself if it is empty
        if not os.listdir(local_dir):
            os.rmdir(local_dir)

        return True

    except Exception as e:
        logger.error(f"Error uploading directory {local_dir}: {str(e)}")
        return False


def upload_file_to_s3(
    s3_client: S3Client, local_file: str, bucket: str, s3_key: str
) -> bool:
    """Upload a single file to S3.
    Returns True if successful, False otherwise."""
    try:
        s3_client.upload_file(local_file, bucket, s3_key)
        logger.info(
            f"Successfully uploaded {local_file} to s3://{bucket}/{s3_key}"
        )
        return True
    except Exception as e:
        logger.error(f"Error uploading {local_file}: {str(e)}")
        return False


async def async_upload_file_to_s3(
    async_s3_client: S3Client, local_file: str, bucket: str, s3_key: str
) -> bool:
    """Upload a single file to S3.
    Returns True if successful, False otherwise."""
    try:
        await async_s3_client.upload_file(local_file, bucket, s3_key)  # type: ignore
        logger.info(
            f"Successfully uploaded {local_file} to s3://{bucket}/{s3_key}"
        )
        return True
    except Exception as e:
        logger.error(f"Error uploading {local_file}: {str(e)}")
        return False


def create_presigned_url(
    s3_client: S3Client,
    bucket: str,
    s3_key: str,
    expiration: int = PRESIGNED_URL_EXPIRATION,
) -> Optional[str]:
    """Generate a presigned URL to share an S3 object."""
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=expiration,
        )
        return url
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None


async def acreate_presigned_url(
    aioboto3_session: aioboto3.Session,
    bucket: str,
    s3_key: str,
    expiration: int = PRESIGNED_URL_EXPIRATION,
) -> Optional[str]:
    """Generate a presigned URL to share an S3 object asynchronously, only if the file exists."""
    try:
        async with aioboto3_session.client(
            "s3",
            config=Config(signature_version="s3v4", region_name="us-east-2"),
        ) as s3_client:  # type: ignore
            # First, check if the object exists
            await s3_client.head_object(Bucket=bucket, Key=s3_key)

            # If it exists, generate the presigned URL
            url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return url
    except ClientError as e:
        error_response = getattr(e, "response", {})
        error_code = error_response.get("Error", {}).get("Code")
        if error_code == "404":
            logger.warning(f"File not found in S3: s3://{bucket}/{s3_key}")
        else:
            logger.error(f"Error checking/generating presigned URL: {e}")
        return None


async def handle_s3_source(
    s3_source_str: str, tmp_dir: str, session: aioboto3.Session
) -> Tuple[str, S3Source]:
    """
    Asynchronously handle S3 source configuration parsing and file download.

    Args:
        s3_source_str: JSON string containing S3 source configuration.
        tmp_dir: Temporary directory to save downloaded file.
        session: aioboto3 session

    Returns:
        Tuple containing:
        - Local file path where the file is downloaded.
        - Parsed S3Source object.

    Raises:
        HTTPException: If S3 configuration is invalid or file download fails.
    """
    try:
        # Parse S3 source
        s3_source_dict = json.loads(s3_source_str)
        s3_source_obj = S3Source(**s3_source_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid S3 source configuration: {str(e)}"
        )

    # Validate key is not empty
    if not s3_source_obj.key:
        raise HTTPException(
            status_code=400,
            detail="Invalid S3 source configuration: key cannot be empty",
        )

    # Construct the local file path
    local_path = os.path.join(tmp_dir, os.path.basename(s3_source_obj.key))

    async with session.client(
        "s3",  # type: ignore
        aws_access_key_id=s3_source_obj.access_key_id,
        aws_secret_access_key=s3_source_obj.secret_access_key,
        region_name=s3_source_obj.region,
    ) as s3_client:
        success, error_msg = await download_file_from_s3_async(
            s3_client, s3_source_obj.bucket_name, s3_source_obj.key, local_path
        )
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to download file from S3: {error_msg}",
            )

    return local_path, s3_source_obj


async def download_file_from_s3_async(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    local_path: str,
    timeout: int = S3_OPERATION_TIMEOUT,
    retries: int = S3_DOWNLOAD_RETRIES,
    retry_delay: int = S3_RETRY_DELAY,
) -> Tuple[bool, str]:
    """
    Download asynchronously a single file from S3.
    Returns a tuple of (success: bool, error_message: str).

    Parameters:
        s3_client: Async S3 client instance
        bucket: Name of the S3 bucket
        s3_key: Path to the file in S3
        local_path: Local path where the file should be saved
        timeout: Maximum time in seconds to wait for the download operation
        retries: Number of download attempts before giving up
        retry_delay: Seconds to wait between retry attempts

    Returns:
        Tuple[bool, str]: (success status, error message if any)
    """
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Check if file exists in S3 with timeout
            async with asyncio.timeout(timeout):
                await s3_client.head_object(Bucket=bucket, Key=s3_key)

            # Download file with timeout
            async with asyncio.timeout(timeout):
                await s3_client.download_file(bucket, s3_key, local_path)

            # Verify download
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                logger.info(f"Successfully downloaded {s3_key} to {local_path}")
                return True, ""
            else:
                last_error = f"Download appeared to succeed but file is missing or empty: {local_path}"
                logger.error(last_error)

        except asyncio.TimeoutError:
            last_error = f"Timeout during attempt {attempt} for {s3_key}"
            logger.error(last_error)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":  # type: ignore
                last_error = f"File not found in S3: {s3_key}"
                logger.error(last_error)
                # If the file does not exist, no point retrying
                return False, last_error
            else:
                last_error = f"ClientError during attempt {attempt}: {str(e)}"
                logger.error(last_error)
        except Exception as e:
            last_error = (
                f"Error during attempt {attempt} for {s3_key}: {str(e)}"
            )
            logger.error(last_error)

        if attempt < retries:
            logger.info(
                f"Retrying download for {s3_key} in {retry_delay} seconds... (Attempt {attempt + 1}/{retries})"
            )
            await asyncio.sleep(retry_delay)

    return (
        False,
        f"All {retries} attempts failed for {s3_key}. Last error: {last_error}",
    )


def download_file_from_s3(
    s3_client,
    bucket: str,
    s3_key: str,
    local_path: str,
    timeout: int = S3_OPERATION_TIMEOUT,
    retries: int = S3_DOWNLOAD_RETRIES,
    retry_delay: int = S3_RETRY_DELAY,
) -> Tuple[bool, str]:
    """
    Synchronously download a single file from S3.
    Returns a tuple of (success: bool, error_message: str).
    """
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Check if file exists in S3 (head_object)
            s3_client.head_object(Bucket=bucket, Key=s3_key)

            # Download file (boto3 does not have a built-in timeout,
            # so to enforce timeout you would need a wrapper; most people skip this)
            s3_client.download_file(bucket, s3_key, local_path)

            # Verify download
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                logger.info(f"Successfully downloaded {s3_key} to {local_path}")
                return True, ""
            else:
                last_error = f"Download appeared to succeed but file is missing or empty: {local_path}"
                logger.error(last_error)

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":  # type: ignore
                last_error = f"File not found in S3: {s3_key}"
                logger.error(last_error)
                # If the file does not exist, no point retrying
                return False, last_error
            else:
                last_error = f"ClientError during attempt {attempt}: {str(e)}"
                logger.error(last_error)
        except Exception as e:
            last_error = (
                f"Error during attempt {attempt} for {s3_key}: {str(e)}"
            )
            logger.error(last_error)

        if attempt < retries:
            logger.info(
                f"Retrying download for {s3_key} in {retry_delay} seconds... (Attempt {attempt + 1}/{retries})"
            )
            time.sleep(retry_delay)

    return (
        False,
        f"All {retries} attempts failed for {s3_key}. Last error: {last_error}",
    )


async def download_directory_from_s3_async(
    s3_client: Any, bucket: str, s3_dir: str, local_dir: str
) -> bool:
    """
    Async version: Download the entire content of a given S3 directory to the specified local directory.
    """
    try:
        # Check if the local directory exists and is empty
        if os.path.exists(local_dir) and os.listdir(local_dir):
            logger.error(
                f"Local directory {local_dir} already exists, skipping download"
            )
            return True

        # Check if the S3 directory is entirely empty
        result = await s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_dir)
        if "Contents" not in result:
            logger.error(f"S3 directory {s3_dir} is empty, skipping download")
            return False

        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=s3_dir):
            if "Contents" not in page:
                logger.info(f"No contents found in S3 directory: {s3_dir}")
                return False

            for obj in page["Contents"]:
                s3_key = obj["Key"]
                relative_path = os.path.relpath(s3_key, s3_dir)
                local_file_path = os.path.join(local_dir, relative_path)

                # Skip problematic file types that shouldn't be downloaded
                if _should_skip_file(s3_key):
                    logger.info(f"Skipping problematic file: {s3_key}")
                    continue

                success, error_msg = await download_file_from_s3_async(
                    s3_client, bucket, s3_key, local_file_path
                )
                if not success:
                    logger.error(f"Failed to download {s3_key}: {error_msg}")
                    return False

        return True

    except Exception as e:
        logger.error(f"Error downloading directory {s3_dir}: {str(e)}")
        return False


def _should_skip_file(s3_key: str) -> bool:
    """
    Determine if a file should be skipped during S3 directory downloads.

    Args:
        s3_key: S3 object key to check

    Returns:
        bool: True if the file should be skipped, False otherwise
    """
    # File extensions to skip (temporary/lock files)
    skip_extensions = [
        ".lock",  # HDF5 lock files
        ".tmp",  # Temporary files
        ".temp",  # Temporary files
        "._",  # macOS resource fork files
        ".DS_Store",  # macOS metadata files
    ]

    # File names to skip (exact matches)
    skip_filenames = [
        "Thumbs.db",  # Windows thumbnail cache
        ".gitkeep",  # Git placeholder files
    ]

    filename = os.path.basename(s3_key).lower()

    # Skip files with problematic extensions
    for ext in skip_extensions:
        if filename.endswith(ext.lower()):
            return True

    # Skip specific filenames
    if filename in [name.lower() for name in skip_filenames]:
        return True

    # Skip zero-byte or very small files that might be artifacts
    # (This would require an additional S3 head_object call, so we'll skip this optimization for now)

    return False


async def adelete_s3_object(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    is_directory: bool = False,
) -> bool:
    """Delete a single object or all objects under a prefix (directory) from S3 asynchronously."""

    if is_directory:
        logger.debug(f"Deleting directory from S3: s3://{bucket}/{s3_key}")
        prefix = s3_key if s3_key.endswith("/") else f"{s3_key}/"

        paginator = s3_client.get_paginator("list_objects_v2")
        objects_to_delete = []

        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            contents = page.get("Contents", [])
            objects_to_delete.extend({"Key": obj["Key"]} for obj in contents)

        if not objects_to_delete:
            logger.warning(f"No objects found under: s3://{bucket}/{prefix}")
            return False

        # Delete in chunks of S3_DELETE_BATCH_SIZE (S3 limit)
        for i in range(0, len(objects_to_delete), S3_DELETE_BATCH_SIZE):
            batch = objects_to_delete[i : i + S3_DELETE_BATCH_SIZE]
            await s3_client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": batch, "Quiet": True},
            )

        logger.info(f"Deleted {len(objects_to_delete)} objects under {prefix}")
        return True

    else:
        logger.debug(f"Deleting single object from S3: s3://{bucket}/{s3_key}")
        await s3_client.delete_object(Bucket=bucket, Key=s3_key)
        logger.info(f"Deleted object: s3://{bucket}/{s3_key}")
        return True

    return False


def download_directory_from_s3(
    s3_client, bucket: str, s3_dir: str, local_dir: str
) -> bool:
    """
    Download the entire content of a given S3 directory to the specified local directory (synchronous version).
    """
    try:
        # Check if the local directory exists and is empty
        if os.path.exists(local_dir) and os.listdir(local_dir):
            logger.error(
                f"Local directory {local_dir} already exists, skipping download"
            )
            return True

        # Check if the S3 directory is entirely empty
        result = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_dir)
        if "Contents" not in result:
            logger.error(f"S3 directory {s3_dir} is empty, skipping download")
            return False

        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_dir):
            if "Contents" not in page:
                logger.info(f"No contents found in S3 directory: {s3_dir}")
                return False

            for obj in page["Contents"]:
                s3_key = obj["Key"]
                relative_path = os.path.relpath(s3_key, s3_dir)
                local_file_path = os.path.join(local_dir, relative_path)
                if relative_path != ".":
                    # Skip problematic file types that shouldn't be downloaded
                    if _should_skip_file(s3_key):
                        logger.info(f"Skipping problematic file: {s3_key}")
                        continue

                    success, error_msg = download_file_from_s3(
                        s3_client, bucket, s3_key, local_file_path
                    )
                    if not success:
                        logger.error(
                            f"Failed to download {s3_key}: {error_msg}"
                        )
                        return False

        return True

    except Exception as e:
        logger.error(f"Error downloading directory {s3_dir}: {str(e)}")
        return False


async def download_config_from_s3_async(
    s3_client: Any,
    bucket: str,
    user_id: str,
    dataset_name: str,
    filename: str,
    config_file_path: str,
    overwrite_if_exists: bool = True,
) -> bool:
    """
    Download config file from S3 if it doesn't exist locally.
    """
    if (
        os.path.exists(config_file_path)
        and os.path.getsize(config_file_path) > 0
        and not overwrite_if_exists
    ):
        logger.info(
            f"File already exists locally at {config_file_path}, skipping download"
        )
        return True
    else:
        s3_key = f"{user_id}/{dataset_name}/{filename}"
        success, _ = await download_file_from_s3_async(
            s3_client, bucket, s3_key, config_file_path
        )
        return success


async def download_training_config_from_s3_async(
    s3_client: Any,
    bucket: str,
    user_id: str,
    dataset_name: str,
    task_type: Literal["synthesis", "forecast"],
    config_file_path: str,
    training_job_id: Optional[str] = None,
    overwrite_if_exists: bool = True,
) -> bool:
    """
    Download training config file from S3 with support for both new and old naming conventions.

    This function attempts to download the training config file using both the new format
    (with training_job_id) and the old format (without training_job_id) for backwards compatibility.

    Args:
        s3_client: Async S3 client instance
        bucket: Name of the S3 bucket
        user_id: User identifier for S3 path construction
        dataset_name: Name of the dataset
        task_type: Type of task ("synthesis" or "forecast")
        config_file_path: Local path where the config file should be saved
        training_job_id: Optional training job ID for new format
        overwrite_if_exists: Whether to overwrite if file already exists locally

    Returns:
        bool: True if download was successful, False otherwise
    """
    # Check if file already exists locally and skip if appropriate
    if (
        os.path.exists(config_file_path)
        and os.path.getsize(config_file_path) > 0
        and not overwrite_if_exists
    ):
        logger.info(
            f"Training config already exists locally at {config_file_path}, skipping download"
        )
        return True

    # Try downloading with new format first (if training_job_id provided)
    if training_job_id:
        new_file_name = get_train_config_file_name(
            dataset_name, task_type, training_job_id
        )
        new_file_success = await download_config_from_s3_async(
            s3_client=s3_client,
            bucket=bucket,
            user_id=user_id,
            dataset_name=dataset_name,
            filename=new_file_name,
            config_file_path=config_file_path,
            overwrite_if_exists=overwrite_if_exists,
        )
        if new_file_success:
            return True

    # If new format failed or no training_job_id, try old format
    old_file_name = get_train_config_file_name(dataset_name, task_type)
    old_file_success = await download_config_from_s3_async(
        s3_client=s3_client,
        bucket=bucket,
        user_id=user_id,
        dataset_name=dataset_name,
        filename=old_file_name,
        config_file_path=config_file_path,
        overwrite_if_exists=overwrite_if_exists,
    )

    if not old_file_success:
        logger.warning(
            f"Failed to download {task_type} training config for dataset: {dataset_name}"
        )
        return False

    return True


async def download_preprocessed_data_from_s3_async(
    s3_client: Any,
    bucket: str,
    user_id: str,
    dataset_name: str,
    local_path: str,
    required_files: List[str],
) -> bool:
    """
    Async version: Download preprocessed data files from S3.
    """
    # Check if all files already exist locally
    all_files_exist = True
    for filename in required_files:
        local_file_path = os.path.join(local_path, filename)
        if not (
            os.path.exists(local_file_path)
            and os.path.getsize(local_file_path) > 0
        ):
            all_files_exist = False
            break

    if all_files_exist:
        logger.info(
            f"All preprocessed files already exist in {local_path}, skipping download"
        )
        return True

    # If any file is missing, download all to ensure consistency
    for filename in required_files:
        s3_key = f"{user_id}/{dataset_name}/{filename}"
        local_file_path = os.path.join(local_path, filename)

        file_download, error_msg = await download_file_from_s3_async(
            s3_client, bucket, s3_key, local_file_path
        )
        if not file_download:
            logger.error(f"Failed to download {filename}: {error_msg}")
            return False

    return True


async def get_async_s3_client(session: aioboto3.Session):
    """Get async S3 client using context manager"""
    async with session.client(
        "s3", config=Config(signature_version="s3v4", region_name="us-east-2")
    ) as async_s3_client:  # type: ignore
        return async_s3_client


def get_aioboto3_session(request: Request) -> aioboto3.Session:
    """Get the aioboto3 session from the FastAPI application state."""
    return request.app.state.aioboto3_session


async def download_model_from_s3_async(
    s3_client: Any,
    bucket: str,
    user_id: str,
    dataset_name: str,
    model_save_path: Optional[str] = None,
    training_job_id: Optional[str] = None,
    overwrite_if_exists: bool = True,
) -> bool:
    """Async version: Download model checkpoint from S3 if it doesn't exist locally."""
    if not (training_job_id and model_save_path):
        return False

    # Check if model already exists locally
    if (
        os.path.exists(model_save_path)
        and os.path.getsize(model_save_path) > 0
        and not overwrite_if_exists
    ):
        logger.info(
            f"Model already exists locally at {model_save_path}, skipping download"
        )
        return True

    s3_key = f"{user_id}/training_logs/{dataset_name}/{training_job_id}/output/model/model.ckpt"
    success, _ = await download_file_from_s3_async(
        s3_client, bucket, s3_key, model_save_path
    )
    return success


async def avalidate_file_exists(
    s3_client: Any, bucket_name: str, file_key: str
) -> Tuple[bool, str]:
    """
    Asynchronously validate that the file exists in S3.

    Args:
        s3_client: Async S3 client instance
        bucket_name: Name of the S3 bucket
        file_key: Path to the file in S3

    Returns:
        Tuple[bool, str]:
            - First element is True if file exists, False otherwise
            - Second element is a message describing the result or error
    """
    try:
        logger.debug(f"Checking if file_key exists in S3: {file_key}")
        await s3_client.head_object(Bucket=bucket_name, Key=file_key)
        logger.debug(f"File exists in S3: {file_key}")
        return True, "File exists"
    except ClientError as e:
        logger.error(f"Error checking if file_key exists in S3: {e}")
        logger.error(f"Error response: {e.response}")
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            logger.warning(f"File not found in S3: {file_key}")
            return False, f"The specified file does not exist in S3: {file_key}"
        else:
            # Handle other error codes
            logger.error(f"Unexpected S3 error: {error_code}")
            return False, f"S3 error: {str(e)}"


def read_s3_yaml_file(s3_url: str) -> dict:
    """
    Read and parse the metadata file from S3.

    Args:
        s3_url: S3 URL of the metadata file

    Returns:
        dict: Parsed metadata content
    """
    # Parse S3 URL
    bucket, key = parse_s3_url(s3_url)

    # Create S3 client
    s3_client = boto3.client("s3")

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        mode="w+b", delete=False, suffix=".yaml"
    ) as tmp_file:
        local_path = tmp_file.name

    try:
        # Download file from S3
        print(f"Downloading {s3_url} to {local_path}...")
        success, error_msg = download_file_from_s3(
            s3_client=s3_client,
            bucket=bucket,
            s3_key=key,
            local_path=local_path,
        )

        if not success:
            print(f"Failed to download file: {error_msg}")
            return {}

        # Read and parse YAML file
        print("Reading and parsing YAML file...")
        with open(local_path, "r") as f:
            metadata = yaml.safe_load(f)

        return metadata

    finally:
        # Clean up temporary file
        if os.path.exists(local_path):
            os.unlink(local_path)
