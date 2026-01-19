import asyncio
import json
import os
import tempfile
import traceback
import zipfile
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import aioboto3
import boto3
import numpy as np
import pandas as pd
import torch
from celery import Task
from celery.contrib import rdb
from dateutil.relativedelta import relativedelta
from isodate import parse_duration
from loguru import logger

from synthefy_pkg.app.celery_app import celery_app
from synthefy_pkg.app.config import SyntheticDataAgentSettings
from synthefy_pkg.app.data_models import (
    FoundationModelConfig,
    GroupLabelColumnFilters,
)
from synthefy_pkg.app.routers.foundation_models import (
    _run_forecast_backtest_core,
)
from synthefy_pkg.app.utils.s3_utils import (
    acreate_presigned_url,
    async_upload_file_to_s3,
    create_presigned_url,
    upload_directory_to_s3,
)
from synthefy_pkg.data.window_and_dataframe_utils import (
    convert_h5_file_to_dataframe,
)
from synthefy_pkg.experiments.synthesis_experiment import (
    MAX_STATUS_PERCENTAGE_FOR_CELERY,
    SynthesisExperiment,
)

DEFAULT_SOFT_TIME_LIMIT = 57600  # 16 hours
DEFAULT_TIME_LIMIT = 86400  # 24 hours

SYNTHEFY_DATASETS_BASE = str(os.getenv("SYNTHEFY_DATASETS_BASE"))
assert (
    SYNTHEFY_DATASETS_BASE is not None and SYNTHEFY_DATASETS_BASE != "None"
), "SYNTHEFY_DATASETS_BASE must be set in the environment"


SYNTHETIC_DATA_FILENAME = "synthetic_data.zip"


@celery_app.task(
    bind=True,
    max_retries=3,
    soft_time_limit=DEFAULT_SOFT_TIME_LIMIT,  # 16 hours
    time_limit=DEFAULT_TIME_LIMIT,  # 24 hours
)
def run_synthetic_data_generation(
    self,
    config_path: str,
    model_checkpoint_path: str,
    metadata_for_synthesis: Dict[str, Any],
    preprocess_config_path: str,
    run_name: str,
    user_id: str,
    dataset_name: str,
    split: str,
    settings_dict: Dict[str, Any],
    synthesis_training_job_id: str,
    bucket_name: str | None = None,
):
    """
    Run synthetic data generation as a Celery task

    Inputs:
        config_path: str #synthesis config path
        model_checkpoint_path: str # synthesis ckpt model path
        metadata_for_synthesis: Dict[str, Any] # must be json serializable (dict and not df)
        preprocess_config_path: str
        run_name: str
        user_id: str
        dataset_name: str
        bucket_name: str | None = None # None when running locally

    Outputs:
        status: str
        message: str
        details: Dict[str, Any]
    """

    metadata_for_synthesis = pd.DataFrame(metadata_for_synthesis)  # type: ignore

    logger.info("Entered task for synthetic data generation")

    # Download required files if they don't exist locally
    try:
        import aioboto3

        from synthefy_pkg.app.utils.synthetic_data_utils import (
            ensure_preprocessed_data_downloaded,
            ensure_synthesis_model_downloaded,
            get_files_to_download,
        )

        # Recreate settings object from the serialized dictionary
        settings = SyntheticDataAgentSettings(**settings_dict)

        # Create aioboto3 session (same as FastAPI app)
        aioboto3_session = aioboto3.Session()

        # Get files to download for the split
        files_to_download = get_files_to_download(split)

        # Download preprocessed data files using asyncio.run since this is a sync function
        asyncio.run(
            ensure_preprocessed_data_downloaded(
                settings,
                user_id,
                dataset_name,
                aioboto3_session,
                files_to_download,
            )
        )

        # Use the passed training job ID
        asyncio.run(
            ensure_synthesis_model_downloaded(
                settings,
                user_id,
                dataset_name,
                aioboto3_session,
                synthesis_training_job_id,
            )
        )

    except Exception as e:
        logger.error(f"Failed to download required files: {str(e)}")
        raise e

    # Clear GPU cache before task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Define output directory based on task ID
    output_dir = os.path.join(
        SYNTHEFY_DATASETS_BASE,
        "generation_logs",
        dataset_name,
        run_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Update initial status
        self.update_state(
            state="PROGRESS",
            meta={
                "progress_percentage": 0,
            },
        )

        # Initialize experiment
        experiment = SynthesisExperiment(config_path)

        logger.info("Starting synthetic data generation")
        # Generate synthetic data with progress tracking
        _ = experiment.generate_synthetic_data(
            model_checkpoint_path=model_checkpoint_path,
            metadata_for_synthesis=metadata_for_synthesis,  # type: ignore
            preprocess_config_path=preprocess_config_path,
            output_dir=output_dir,
            task_id=self.request.id,  # Pass the task ID to track progress
            output_filename_prefix=run_name,
        )

        logger.info("Synthetic data generation completed successfully")

        # Load preprocessing config to extract timestamp column
        with open(preprocess_config_path, "r") as f:
            preprocessing_config = json.load(f)

        # Extract timestamps_col from config
        timestamps_col = preprocessing_config.get("timestamps_col", [])
        window_size = preprocessing_config["window_size"]

        if len(timestamps_col) > 0:
            # Extract the timestamp column
            timestamp_values = metadata_for_synthesis[timestamps_col[0]].values

            # Get unique window indices to determine number of windows
            window_indices = metadata_for_synthesis["window_idx"].unique()
            num_windows = len(window_indices)

            # Properly organize timestamps by windows
            timestamps_original = np.zeros((num_windows, window_size, 1))
            for i, window_idx in enumerate(window_indices):
                # Get all rows for this window
                window_mask = metadata_for_synthesis["window_idx"] == window_idx
                window_timestamps = timestamp_values[window_mask]
                timestamps_original[i, :, 0] = window_timestamps

            logger.info(
                f"Extracted timestamp column: {timestamps_col[0]} with shape {timestamps_original.shape}"
            )
        else:
            # Create empty 3D array with shape (num_windows, window_size, 0)
            window_indices = metadata_for_synthesis["window_idx"].unique()
            num_windows = len(window_indices)
            timestamps_original = np.empty((num_windows, window_size, 0))
            logger.info(
                f"No timestamp column found, created empty 3D array with shape {timestamps_original.shape}"
            )

        df_synthetic_data = asyncio.run(
            convert_h5_file_to_dataframe(
                dataset_name=dataset_name,
                h5_file_path=os.path.join(
                    output_dir,
                    f"{run_name}_dataset",
                    f"{run_name}_combined_data.h5",
                ),
                split=cast(Literal["train", "val", "test"], split),
                synthetic_or_original="synthetic",
                timestamps_original=timestamps_original,
            )
        )
        # save the df_synthetic_data to the output_dir
        df_synthetic_data.to_parquet(
            os.path.join(output_dir, "synthetic_data.parquet"), index=False
        )
        # combine all files in the output_dir into a zip file
        zip_file_path = os.path.join(output_dir, SYNTHETIC_DATA_FILENAME)
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    # only save the synthetic_data.parquet file
                    if file == SYNTHETIC_DATA_FILENAME:
                        continue
                    if file != "synthetic_data.parquet":
                        os.remove(os.path.join(root, file))
                        continue
                    file_path = os.path.join(root, file)
                    logger.info(f"Adding file {file} to zip file")
                    zipf.write(file_path, file)
                    # Delete the file after adding it to the zip
                    os.remove(file_path)
                    logger.info(f"Deleted file {file} after adding to zip")

        # Upload results to S3 if bucket provided
        presigned_url = None
        s3_path = None
        if (
            bucket_name is not None
            and os.path.exists(output_dir)
            and os.listdir(output_dir)
        ):
            try:
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "progress_percentage": MAX_STATUS_PERCENTAGE_FOR_CELERY,
                    },
                )

                # Create S3 client
                s3_client = boto3.client("s3")  # pyright: ignore

                # Use the existing utility to upload the directory
                s3_prefix = str(
                    os.path.join(
                        user_id,
                        "generation_logs",
                        dataset_name,
                        run_name,
                    )
                )

                # Upload using the utility from s3_utils.py
                # Note that this will delete the local directory after upload
                success = upload_directory_to_s3(
                    s3_client=s3_client,
                    bucket=bucket_name,
                    local_dir=output_dir,
                    s3_dir=s3_prefix,
                )

                if success:
                    # Generate presigned URL specifically for the zip file
                    zip_s3_key = os.path.join(
                        s3_prefix, SYNTHETIC_DATA_FILENAME
                    )
                    presigned_url = create_presigned_url(
                        s3_client=s3_client,
                        bucket=bucket_name,
                        s3_key=zip_s3_key,
                    )
                    s3_path = f"s3://{bucket_name}/{zip_s3_key}"
                    logger.info(
                        f"Successfully uploaded results to s3://{bucket_name}/{s3_prefix}"
                    )
                else:
                    logger.error("Failed to upload results to S3")
                    raise Exception("Failed to upload results to S3")

            except Exception as e:
                logger.error(f"Error during S3 upload: {str(e)}")
                raise

        return {
            "status": "SUCCESS",
            "progress_percentage": 100,
            "message": "Synthetic data generation completed successfully",
            "task_id": self.request.id,
            "presigned_url": presigned_url,
            "s3_path": s3_path,  # for debugging - TODO - we will remove this in the future
        }

    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={
                "progress_percentage": 0,
                "error": str(e),
                "exc_type": type(e).__name__,
                "exc_message": traceback.format_exc().split("\n"),
            },
        )
        raise e
    finally:
        # Always clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _str_to_relativedelta(delta_str: str) -> relativedelta:
    """Convert a string representation of relativedelta back to a relativedelta object.

    Handles both ISO 8601 duration strings (e.g., "P224D") and relativedelta string representations
    (e.g., "relativedelta(days=+224)").
    """
    try:
        # First try parsing as ISO 8601 duration
        duration = parse_duration(delta_str)
        # Convert timedelta to relativedelta
        days = duration.days
        seconds = duration.seconds
        microseconds = duration.microseconds

        # Calculate years and months from days
        years = days // 365
        remaining_days = days % 365
        months = remaining_days // 30
        days = remaining_days % 30

        # Calculate hours, minutes, seconds
        hours = seconds // 3600
        remaining_seconds = seconds % 3600
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60

        return relativedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )
    except Exception:
        # If ISO 8601 parsing fails, try parsing relativedelta string representation
        try:
            # Extract values from string like "relativedelta(days=+224)"
            import re

            values = {}
            for match in re.finditer(r"(\w+)=([+-]?\d+)", delta_str):
                key, value = match.groups()
                values[key] = int(value)
            return relativedelta(**values)
        except Exception as e:
            raise ValueError(
                f"Could not parse duration string: {delta_str}. Error: {str(e)}"
            )


@celery_app.task(
    bind=True,
    max_retries=3,
    soft_time_limit=DEFAULT_SOFT_TIME_LIMIT,  # 16 hours
    time_limit=DEFAULT_TIME_LIMIT,  # 24 hours
)
def run_forecast_backtest(
    self,
    df_serialized: dict,  # from json.loads(df.to_json())
    forecast_windows: List[Tuple[str, str]],  # List of (start, end) ISO strings
    config_as_dict: Dict[str, Any],
    user_id: str,
    group_filters_serialized: Dict[str, Any] | None,
    bucket_name: str,
    file_path_key: str,
    delta_serialized: str | None = None,
    covariate_grid: List[Dict[str, bool]] | None = None,
) -> Dict[str, Any]:
    """
    Run forecast backtest as a Celery task.

    Args:
        df_serialized: DataFrame as CSV string
        forecast_windows: List of (start, end) ISO strings
        config: FoundationModelConfig as dict
        user_id: ID of the user
        group_filters: Optional group filters for the forecast
        bucket_name: Name of the S3 bucket
        file_path_key: S3 key of the input file
        delta_serialized: Optional serialized relativedelta string
        covariate_grid: Optional list of covariate dictionaries for grid search where keys are covariate names and values are booleans indicating whether to leak them

    Returns:
        Dict containing task status and results
    """
    logger.info("Entered task for forecast backtest")

    def progress_callback(percentage: int, status: str):
        self.update_state(
            state="PROGRESS",
            meta={
                "progress_percentage": percentage,
                "status": status,
                "user_id": user_id,
            },
        )

    try:
        result = asyncio.run(
            _run_forecast_backtest_core(
                df_serialized=df_serialized,
                forecast_windows=forecast_windows,
                config_as_dict=config_as_dict,
                user_id=user_id,
                group_filters_serialized=group_filters_serialized,  # type: ignore
                bucket_name=bucket_name,
                file_path_key=file_path_key,
                delta_serialized=delta_serialized,
                progress_callback=progress_callback,
                covariate_grid=covariate_grid,
            )
        )
        return result
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={
                "progress_percentage": 0,
                "error": str(e),
                "exc_type": type(e).__name__,
                "exc_message": traceback.format_exc().split("\n"),
                "user_id": user_id,
            },
        )
        return {
            "status": "FAILURE",
            "message": str(e),
        }
