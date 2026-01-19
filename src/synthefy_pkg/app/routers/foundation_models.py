import asyncio
import io
import json
import os
import re
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import combinations
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import aioboto3
import httpx
import ijson
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
from botocore.exceptions import ClientError
from dateutil.relativedelta import relativedelta
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from isodate import parse_duration
from loguru import logger
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from smart_open import open as smart_open
from sqlalchemy.orm import Session

from synthefy_pkg.app.celery_app import celery_app
from synthefy_pkg.app.fm_agent.agent import run_agent, run_agent_streaming

matplotlib.use("Agg")
from synthefy.data_models import (
    ForecastV2Request,
    ForecastV2Response,
)
from synthefy_pkg.app.config import (
    FoundationModelApiSettings,
    SynthefyFoundationModelSettings,
)
from synthefy_pkg.app.dao.user_api_keys import validate_api_key
from synthefy_pkg.app.data_models import (
    ApiForecastRequest,
    ApiForecastResponse,
    BacktestAPIForecastRequest,
    BacktestAPIForecastResponse,
    BacktestInfo,
    CategoricalFeaturesResponse,
    CategoricalFeatureValues,
    CovariateGridRequest,
    CovariateGridResponse,
    ForecastDataset,
    FoundationModelChatRequest,
    FoundationModelConfig,
    FoundationModelForecastStreamRequest,
    FoundationModelForecastStreamResponse,
    GroupLabelColumnFilters,
    HaverDatasetMatch,
    HaverMetadataAccessInfo,
    HistoricalDataset,
    ListBacktestsRequest,
    ListBacktestsResponse,
    LocationSearchRequest,
    MetadataDataFrame,
    MetadataVisualizationItem,
    MetadataVisualizationRequest,
    MetadataVisualizationResponse,
    PaginationInfo,
    PredictHQMetadataAccessInfo,
    StatusCode,
    SupportedAggregationFunctions,
    SynthefyDatabaseDirectorySearchRequest,
    SynthefyDatabaseDirectorySearchResponse,
    SynthefyDatabaseMetadataSearchRequest,
    SynthefyDatabaseMetadataSearchResponse,
    TimePeriod,
    UploadResponse,
    WeatherMetadataAccessInfo,
    WeatherParameters,
    WeatherStackLocation,
)

# Import database dependencies for API key validation
from synthefy_pkg.app.db import get_db
from synthefy_pkg.app.middleware.metrics_manager.metrics_manager_utils import (
    track_metadata_usage,
)
from synthefy_pkg.app.utils.api_utils import (
    api_key_required,
    detect_time_frequency,
    format_timestamp_with_optional_fractional_seconds,
    get_settings,
)
from synthefy_pkg.app.utils.auth_utils import (
    AuthenticationUtils,
    get_user_id_from_token_or_api_key,
)
from synthefy_pkg.app.utils.external_metadata_utils import (
    LocationCacheManager,
    load_locations_into_cache,
    prepare_metadata_dataframes,
    process_haver_metadata_access_info,
    process_predicthq_metadata_access_info,
    process_weatherstack_metadata_access_info,
    search_cached_locations,
)
from synthefy_pkg.app.utils.filter_utils import FilterUtils
from synthefy_pkg.app.utils.s3_utils import (
    acreate_presigned_url,
    adelete_s3_object,
    async_upload_file_to_s3,
    avalidate_file_exists,
    get_aioboto3_session,
)
from synthefy_pkg.app.utils.smart_approximation_utils import (
    DEFAULT_NULL_THRESHOLD,
    SmartApproximationUtils,
)
from synthefy_pkg.app.utils.supabase_utils import (
    get_supabase_user,
    get_user_id_from_token_or_body,
    get_user_id_from_token_or_form,
)
from synthefy_pkg.app.utils.timestamp_utils import get_timestamp_range
from synthefy_pkg.fm_evals.forecasting.api.api_forecaster import (
    ForecastV2APIAdapter,
)
from synthefy_pkg.model.foundation_model_service import (
    FoundationModelService,
    apply_full_and_range_modifications,
    apply_point_modifications,
)

# Progress percentage constants for backtest forecasting
PROGRESS_SETUP_PERCENTAGE = 10
PROGRESS_PROCESSING_PERCENTAGE = 90

# Environment variable to control whether to use Celery or direct execution
USE_CELERY = os.getenv("USE_CELERY", "false").lower() == "true"
logger.info(f"USE_CELERY: {USE_CELERY}")
logger.debug(f"USE_CELERY: {USE_CELERY}")

SYNTHEFY_FOUNDATION_MODEL_METADATA_DATASETS_BUCKET = (
    "synthefy-foundation-model-metadata-datasets"
)
MAX_BACKTEST_FORECAST_WINDOWS = 250

# Number of decimal places to round aggregated metrics
METRIC_ROUNDING_PRECISION = 4

COMPILE = False

FOUNDATION_MODEL_SETTINGS = SynthefyFoundationModelSettings()
# Constants for warning messages
SHAP_ANALYSIS_WARNING_MSG = (
    "⚠️ SHAP analysis is enabled. This will significantly increase processing time "
    "and may take several minutes to complete. Please be patient while the analysis runs. "
    "Note: Results may degrade if there are 5 or more covariates. Contact the Synthefy team "
    "for optimal performance with complex datasets."
)
BACKTEST_RESULTS_SUFFIX = "_backtest_results_"


router = APIRouter(tags=["Foundation Models"])

RequestDep = Annotated[Request, Depends()]


@router.put("/api/foundation_models/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    user_id: Optional[str] = Form(None),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
):
    if user_id_from_auth_header:
        user_id = user_id_from_auth_header
    user_id = AuthenticationUtils.validate_user_id_required(user_id)
    """Upload and store an initial dataset with user ID and dataset name."""
    logger.add(sys.stdout, level="DEBUG")

    logger.info(
        f"Starting upload_dataset for user_id: {user_id}, file: {file.filename}"
    )
    try:
        # Get bucket name from config
        logger.debug("Retrieving bucket name from settings")
        settings = get_settings(FoundationModelApiSettings)
        bucket_name = settings.bucket_name
        logger.debug(f"Retrieved bucket name: {bucket_name}")

        if not bucket_name:
            logger.error("S3 bucket name not configured in settings")
            raise ValueError("S3 bucket name not configured")

        if file.filename is None:
            logger.warning("Upload attempted with missing filename")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "file_path_key": None,
                    "status": StatusCode.unprocessable_entity,
                    "message": "File name is required",
                },
            )

        # Validate file extension
        logger.debug(f"Validating file extension for: {file.filename}")
        if not (
            file.filename.endswith(".csv") or file.filename.endswith(".parquet")
        ):
            logger.warning(f"Invalid file extension: {file.filename}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "file_path_key": None,
                    "status": StatusCode.unprocessable_entity,
                    "message": "File must be a CSV or Parquet file (with .csv or .parquet extension)",
                },
            )

        file_name_without_extension = file.filename.split(".")[0]
        # Get S3 base path
        logger.debug(
            f"Generating S3 base path for user_id: {user_id}, file: {file_name_without_extension}"
        )
        s3_base_path = FilterUtils.get_s3_base_path(
            user_id, file_name_without_extension
        )
        s3_file_path = f"{s3_base_path}/{file_name_without_extension}.parquet"

        # Check if user try to upload a new data with the same file name
        async with aioboto3_session.client("s3") as s3_client:  # type: ignore
            # Validate file exists in S3
            exists, message = await avalidate_file_exists(
                s3_client, bucket_name, s3_file_path
            )
            if exists:
                logger.warning(f"File already exists: {s3_file_path}")
                await adelete_s3_object(
                    s3_client, bucket_name, s3_base_path, is_directory=True
                )

        logger.debug(f"Generated S3 file path: {s3_file_path}")

        # Read file content
        logger.debug("Reading file content")
        file_content = await file.read()
        logger.debug(f"Read {len(file_content)} bytes from file")

        # Extract column names from the file
        dataset_columns = []
        timestamp_columns = []
        # TODO: Optimize writing to temp file
        logger.debug("Creating temporary file for processing")
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            logger.debug(f"Wrote content to temporary file: {temp_file.name}")

            if file.filename.endswith(".csv"):
                # For CSV files
                logger.debug("Processing CSV file to extract column names")
                df = pd.read_csv(temp_file.name)
            elif file.filename.endswith(".parquet"):
                # For Parquet files - read only the metadata
                logger.debug("Processing Parquet file to extract column names")
                df = pd.read_parquet(temp_file.name)

            # Common processing for both file types
            dataset_columns = df.columns.tolist()
            logger.debug(f"Extracted {len(dataset_columns)} columns")
            logger.debug("Finding timestamp column candidates")
            timestamp_columns, error_messages = (
                SmartApproximationUtils.find_timestamp_index_candidates(
                    df, null_threshold=DEFAULT_NULL_THRESHOLD
                )
            )
            logger.debug(
                f"Found {len(timestamp_columns) if timestamp_columns else 0} timestamp column candidates"
            )
            if len(timestamp_columns) == 0:
                logger.warning("No timestamp column candidates found")
                if error_messages:
                    # Show specific error messages from timestamp parsing
                    error_message = " ".join(error_messages)
                else:
                    # Fallback to generic message if no specific errors
                    error_message = "No timestamp column identified. Please include a timestamp column of format '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d'."
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "file_path_key": None,
                        "status": StatusCode.bad_request,
                        "message": error_message,
                    },
                )

            # TODO: Remove any Tz info from the timestamp column??

            # df[timestamp_columns[0]] = df[timestamp_columns[0]].dt.tz_localize(None)

            # Convert to parquet and save to a new temporary file
            logger.debug("Converting to parquet format")
            parquet_temp_file = tempfile.NamedTemporaryFile(
                delete=True, suffix=".parquet"
            )
            df.to_parquet(parquet_temp_file.name, index=False)
            logger.debug(f"Saved parquet file to: {parquet_temp_file.name}")

            # Upload parquet file to S3
            logger.debug(f"Uploading parquet file to S3: {s3_file_path}")
            async with aioboto3_session.client("s3") as s3_client:  # type: ignore
                with open(parquet_temp_file.name, "rb") as f:
                    logger.debug("Starting S3 upload_fileobj operation")
                    await s3_client.upload_fileobj(f, bucket_name, s3_file_path)
                    logger.debug("Completed S3 upload_fileobj operation")

        # Create and upload metadata.json
        logger.debug("Creating metadata.json")
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "original_filename": file.filename,
            "user_id": user_id,
        }
        logger.debug(f"Metadata content: {metadata}")

        metadata_key = f"{s3_base_path}/metadata.json"
        logger.debug(f"Uploading metadata to S3: {metadata_key}")
        async with aioboto3_session.client("s3") as s3_client:  # type: ignore
            logger.debug("Starting S3 put_object operation for metadata")
            await s3_client.put_object(
                Bucket=bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata),
                ContentType="application/json",
            )
            logger.debug("Completed S3 put_object operation for metadata")

        logger.info(
            f"Successfully uploaded dataset '{file.filename}' for user '{user_id}'"
        )
        return UploadResponse(
            original_file_key=s3_file_path,
            status=StatusCode.ok,
            dataset_columns=dataset_columns,
            message=f"Dataset '{file.filename}' uploaded successfully for user '{user_id}'",
            timestamp_columns=timestamp_columns,
            time_frequency=detect_time_frequency(df[timestamp_columns[0]])
            if timestamp_columns
            else None,
        )
    except Exception as e:
        logger.exception(f"Error in upload_dataset: {str(e)}")
        # Return error response using JSONResponse for consistency
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "file_path_key": None,
                "status": StatusCode.internal_server_error,
                "message": f"Failed to parse or store dataset: {str(e)}",
            },
        )


class FilterRequest(BaseModel):
    user_id: Optional[str] = None
    file_key: str
    timestamp_column: str
    filters: GroupLabelColumnFilters

    class Config:
        # Ensure all fields are required and not None
        extra = "forbid"  # Prevent extra fields

    @field_validator("file_key")
    @classmethod
    def validate_parquet_extension(cls, v):
        """Validate that file_key ends with .parquet."""
        if not v.endswith(".parquet"):
            raise ValueError("file_key must end with .parquet")
        return v

    @field_validator("user_id", "file_key", mode="before")
    @classmethod
    def validate_not_none(cls, value, info):
        if value is None:
            raise ValueError(f"{info.field_name} cannot be null or None")
        return value


class DatasetInfo(BaseModel):
    """Model for dataset information with validation."""

    original_file_key: str
    original_filename: str
    original_file_name_without_extension: str  # Renamed from dataset_name
    filtered_dataset_key: str
    filtered_dataset_json_blob_key: str | None = Field(
        default=None,
        description="S3 key for the JSON blob representation of the filtered dataset",
    )

    @field_validator("original_file_key")
    @classmethod
    def validate_parquet_extension(cls, v):
        """Validate that original_file_key ends with .parquet."""
        if not v.endswith(".parquet"):
            raise ValueError("original_file_key must end with .parquet")
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_consistency(cls, data):
        """Validate consistency between fields after all individual fields are validated."""
        # Check filename consistency with file_key
        if hasattr(data, "original_file_key") and hasattr(
            data, "original_filename"
        ):
            file_parts = data.original_file_key.split("/")
            expected_filename = file_parts[-1]
            if data.original_filename != expected_filename:
                raise ValueError(
                    f"original_filename '{data.original_filename}' does not match the filename in original_file_key '{expected_filename}'"
                )

        # Check original_file_name_without_extension consistency
        if hasattr(data, "original_filename") and hasattr(
            data, "original_file_name_without_extension"
        ):
            expected_name = data.original_filename.split(".")[0]
            if data.original_file_name_without_extension != expected_name:
                raise ValueError(
                    f"original_file_name_without_extension '{data.original_file_name_without_extension}' does not match the name derived from original_filename '{expected_name}'"
                )

        return data

    @model_validator(mode="after")
    def validate_keys_consistency(self):
        """Validate consistency between filtered_dataset_key and filtered_dataset_json_blob_key."""
        filtered_key = self.filtered_dataset_key
        json_blob_key = self.filtered_dataset_json_blob_key

        if filtered_key and json_blob_key:
            # Get base paths without extension
            filtered_base_path = os.path.splitext(filtered_key)[0]
            json_blob_base_path = os.path.splitext(json_blob_key)[0]

            # Get extensions
            filtered_ext = os.path.splitext(filtered_key)[1]
            json_blob_ext = os.path.splitext(json_blob_key)[1]

            # Validate extensions
            if filtered_ext != ".parquet":
                raise ValueError(
                    f"filtered_dataset_key must end with .parquet, got {filtered_ext}"
                )

            if json_blob_ext != ".json":
                raise ValueError(
                    f"filtered_dataset_json_blob_key must end with .json, got {json_blob_ext}"
                )

            # Check that base paths match
            filtered_name = filtered_base_path.split("/")[-1]
            json_name = json_blob_base_path.split("/")[-1]

            if filtered_name != json_name:
                raise ValueError(
                    f"Base filenames must match. Got {filtered_name} and {json_name}"
                )

            # Check parent directory paths match
            filtered_parent = "/".join(filtered_base_path.split("/")[:-1])
            json_parent = "/".join(json_blob_base_path.split("/")[:-1])

            if filtered_parent != json_parent:
                raise ValueError(
                    f"Parent directory paths must match. Got {filtered_parent} and {json_parent}"
                )

        return self

    @classmethod
    def from_file_key(cls, original_file_key: str, filtered_dataset_key: str):
        """Create a DatasetInfo object from a file key."""
        file_parts = original_file_key.split("/")
        original_filename = file_parts[-1]
        original_file_name_without_extension = original_filename.split(".")[0]

        return cls(
            original_file_key=original_file_key,
            original_filename=original_filename,
            original_file_name_without_extension=original_file_name_without_extension,
            filtered_dataset_key=filtered_dataset_key,
        )


class FilterMetadata(DatasetInfo):
    """Metadata model for filtered datasets."""

    generated_at: str  # ISO format timestamp
    user_id: str  # User who created the filtered dataset
    filters: Dict[str, Any]  # Filter configuration used
    row_count: Optional[int] = None  # Number of rows in the filtered dataset

    class Config:
        extra = "allow"  # Allow additional fields for future extensibility


class FilterResponse(BaseModel):
    """Response model for dataset filtering operations."""

    status: StatusCode
    message: str
    filtered_dataset_key: str | None = Field(
        default=None, description="S3 key of the filtered dataset"
    )
    download_url: str | None = Field(
        default=None,
        description="Pre-signed URL for downloading the filtered dataset",
    )
    filtered_dataset_json_blob_key: str | None = Field(
        default=None,
        description="S3 key for the filtered dataset's JSON metadata blob",
    )
    json_blob_download_url: str | None = Field(
        default=None,
        description="Pre-signed URL for downloading the filtered dataset's JSON metadata",
    )
    metadata_json: FilterMetadata


async def run_filter_core_code(
    filter_request: FilterRequest,
    aioboto3_session: aioboto3.Session,
) -> FilterResponse:
    """Filter a dataset based on GroupLabelColumnFilters."""
    logger.info(
        f"Received filter request for user_id: {filter_request.user_id}"
    )
    try:
        # Access fields directly from the validated model
        settings = get_settings(FoundationModelApiSettings)
        bucket_name = settings.bucket_name

        if not bucket_name:
            logger.error("S3 bucket name not configured in settings")
            raise ValueError("S3 bucket name not configured")

        user_id = filter_request.user_id
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="User ID is required",
            )
        original_file_key = filter_request.file_key
        filters = filter_request.filters
        timestamp_column = filter_request.timestamp_column
        aggregation_func = filters.aggregation_func

        logger.info(
            f"Filtering dataset with file_key: {original_file_key}, user_id: {user_id}, aggregation_func: {aggregation_func.value}"
        )
        logger.debug(f"Applying filters: {filters}")

        # Create a new session for this operation
        session = aioboto3.Session()
        # Use async context manager for S3 client
        async with session.client("s3") as s3_client:  # type: ignore
            # Validate file exists in S3
            exists, message = await avalidate_file_exists(
                s3_client, bucket_name, original_file_key
            )
            if not exists:
                # Raise HTTPException for missing files
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=message
                )

            # Extract file information
            file_info = _extract_file_info(original_file_key)
            original_filename = file_info["file_name"]
            dataset_name = file_info["dataset_file_name"]
            file_format = file_info["extension"]

            # Generate cache key for filtered dataset
            filtered_dataset_key = FilterUtils.get_s3_key(
                user_id=user_id,
                dataset_file_name=dataset_name,
                ext=file_format,
                filters=filters,
            )
            logger.info(
                f"Generated cache key for filtered dataset: {filtered_dataset_key}"
            )

            # Check if filtered dataset already exists
            cached_response = await _get_cached_dataset(
                s3_client,
                session,
                bucket_name,
                filtered_dataset_key,
                user_id,
                dataset_name,
                filters,
            )
            if cached_response:
                return cached_response

            # Create new filtered dataset
            dataset_info = DatasetInfo(
                original_file_key=original_file_key,
                original_filename=original_filename,
                original_file_name_without_extension=dataset_name,
                filtered_dataset_key=filtered_dataset_key,
            )

            return await _create_filtered_dataset(
                s3_client,
                session,
                bucket_name,
                dataset_info,
                user_id,
                filters,
                timestamp_column,
            )

    except HTTPException as _:
        raise
    except Exception as e:
        logger.exception(f"Error in filter_dataset: {str(e)}")
        # Raise HTTPException instead of returning JSONResponse
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Filtering failed: {str(e)}",
        )


@router.put("/api/foundation_models/filter", response_model=FilterResponse)
async def filter_dataset(
    filter_request: FilterRequest = Body(...),
    user_id: Optional[str] = Depends(get_user_id_from_token_or_api_key),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
):
    # Handle authentication and get user_id
    use_access_token = os.getenv("SYNTHEFY_USE_ACCESS_TOKEN", "0") == "1"
    if use_access_token:
        # When SYNTHEFY_USE_ACCESS_TOKEN=1, use the user_id from the dependency
        user_id = AuthenticationUtils.validate_access_token_required(user_id)
        filter_request.user_id = user_id
    else:
        # When SYNTHEFY_USE_ACCESS_TOKEN is not "1", check if user_id is in the request body
        if not filter_request.user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required in request body when SYNTHEFY_USE_ACCESS_TOKEN is not set",
            )
        if (
            filter_request.user_id
            and user_id
            and filter_request.user_id != user_id
        ):
            raise HTTPException(
                status_code=400,
                detail="user_id in request body does not match user_id in authorization header",
            )

    return await run_filter_core_code(
        filter_request=filter_request,
        aioboto3_session=aioboto3_session,
    )


def _validate_file_exists(
    s3_client: Any, bucket_name: str, file_key: str
) -> None:
    """Validate that the file exists in S3.

    Args:
        s3_client: Boto3 S3 client instance
        bucket_name: Name of the S3 bucket
        file_key: Path to the file in S3

    Raises:
        HTTPException: If the file doesn't exist or there's an S3 error
    """
    try:
        logger.debug(f"Checking if file_key exists in S3: {file_key}")
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        logger.debug(f"File exists in S3: {file_key}")
    except ClientError as e:
        logger.error(f"Error checking if file_key exists in S3: {e}")
        logger.error(f"Error response: {e.response}")
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            logger.warning(f"File not found in S3: {file_key}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"The specified file does not exist in S3: {file_key}",
            )
        else:
            logger.error(f"Unexpected S3 error: {error_code}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"S3 error: {str(e)}",
            )


def _extract_file_info(original_file_key: str) -> Dict[str, str]:
    """Extract file information from the file key.

    Args:
        original_file_key: S3 key of the original file

    Returns:
        Dictionary containing file_name, dataset_file_name, and extension
    """
    logger.debug(f"Extracting dataset name from file_key: {original_file_key}")
    file_parts = original_file_key.split("/")
    original_filename = file_parts[-1]
    dataset_name = original_filename.split(".")[0]
    file_format = original_filename.split(".")[-1]

    logger.debug(
        f"Extracted file_name: {original_filename}, dataset_file_name: {dataset_name}, extension: {file_format}"
    )
    return {
        "file_name": original_filename,
        "dataset_file_name": dataset_name,
        "extension": file_format,
    }


async def _get_cached_dataset(
    async_s3_client: Any,
    aioboto3_session: aioboto3.Session,
    bucket_name: str,
    filtered_dataset_key: str,
    user_id: str,
    dataset_name: str,
    filters: GroupLabelColumnFilters,
) -> Optional[FilterResponse]:
    """
    Check if filtered dataset is already cached and return it if found.

    Args:
        s3_client: Boto3 S3 client instance
        bucket_name: Name of the S3 bucket
        filtered_dataset_key: S3 key of the filtered dataset
        user_id: ID of the user who owns the dataset
        dataset_name: Name of the dataset
        filters: Filter configuration applied to the dataset (may have empty filter list for unfiltered data)

    Returns:
        FilterResponse object if cached dataset exists, None otherwise
    """
    try:
        # Try to get the metadata file for the filtered dataset
        logger.debug(
            f"Checking if filtered dataset already exists at: {filtered_dataset_key}"
        )
        await async_s3_client.head_object(
            Bucket=bucket_name, Key=filtered_dataset_key
        )
        logger.info(f"Found cached filtered dataset: {filtered_dataset_key}")

        # If we get here, the filtered dataset exists
        # Get the metadata to find the filtered file path and row count
        metadata_key = FilterUtils.get_metadata_key(
            user_id=user_id,
            dataset_file_name=dataset_name,
            filters=filters,
        )
        logger.debug(f"Retrieving metadata from: {metadata_key}")
        response = await async_s3_client.get_object(
            Bucket=bucket_name, Key=metadata_key
        )
        metadata = await response["Body"].read()
        metadata = json.loads(metadata.decode("utf-8"))
        logger.debug(f"Retrieved metadata: {metadata}")

        # Generate a presigned URL for downloading the filtered dataset
        logger.debug(f"Generating presigned URL for: {filtered_dataset_key}")
        download_url = await acreate_presigned_url(
            aioboto3_session, bucket=bucket_name, s3_key=filtered_dataset_key
        )
        filtered_dataset_json_blob_key = metadata.get(
            "filtered_dataset_json_blob_key"
        )

        json_blob_download_url = None
        if filtered_dataset_json_blob_key:
            json_blob_download_url = await acreate_presigned_url(
                aioboto3_session,
                bucket=bucket_name,
                s3_key=filtered_dataset_json_blob_key,
            )
        logger.debug(
            f"Generated presigned URL: {download_url[:50]}..."
            if download_url
            else "No download URL generated"
        )

        logger.info(
            f"Returning cached filtered dataset with {metadata.get('row_count', 'unknown')} rows"
        )
        return FilterResponse(
            status=StatusCode.ok,
            message="Retrieved cached filtered dataset",
            filtered_dataset_key=filtered_dataset_key,
            download_url=download_url,
            json_blob_download_url=json_blob_download_url,
            filtered_dataset_json_blob_key=metadata.get(
                "filtered_dataset_json_blob_key"
            ),
            metadata_json=FilterMetadata(
                generated_at=metadata["generated_at"],
                original_file_key=metadata["original_file_key"],
                user_id=metadata["user_id"],
                filters=metadata["filters"],
                filtered_dataset_key=metadata["filtered_dataset_key"],
                filtered_dataset_json_blob_key=metadata.get(
                    "filtered_dataset_json_blob_key"
                ),
                row_count=metadata["row_count"],
                original_file_name_without_extension=metadata[
                    "original_file_name_without_extension"
                ],
                original_filename=metadata["original_filename"],
            ),
        )
    except ClientError:
        # Filtered dataset not found in cache
        logger.info(
            "No cached filtered dataset found. Creating new filtered dataset."
        )
        return None


async def _create_filtered_dataset(
    async_s3_client: Any,
    aioboto3_session: aioboto3.Session,
    bucket_name: str,
    dataset_info: DatasetInfo,
    user_id: str,
    filters: GroupLabelColumnFilters,
    timestamp_column: str,
) -> FilterResponse:
    """
    Create a new filtered dataset and upload it to S3.

    Args:
        async_s3_client: Boto3 S3 client instance
        aioboto3_session: aioboto3.Session
        bucket_name: Name of the S3 bucket
        dataset_info: Information about the dataset
        user_id: ID of the user who owns the dataset
        filters: Filter configuration to apply (may have empty filter list for unfiltered data)
        timestamp_column: Name of the timestamp column currently selected

    Returns:
        FilterResponse object with information about the filtered dataset

    Raises:
        Exception: If generating presigned URL fails
        HTTPException: If dataset processing fails
    """
    # Create a temporary directory for processing
    logger.debug("Creating temporary directory for processing")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Process the dataset
            metadata = await _process_dataset_in_temp_dir(
                async_s3_client,
                bucket_name,
                dataset_info,
                user_id,
                filters,
                temp_dir,
                timestamp_column,
            )
            logger.debug("Temporary directory processing complete")
        except HTTPException as http_exc:
            # Handle HTTP exceptions, particularly 400 Bad Request
            if http_exc.status_code == status.HTTP_400_BAD_REQUEST:
                logger.warning(
                    f"Bad request during dataset processing: {http_exc.detail}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(http_exc.detail),
                )
            # Re-raise other HTTP exceptions
            raise http_exc
        except Exception as e:
            # Re-raise any other exceptions
            logger.error(f"Error during dataset processing: {str(e)}")
            raise

    # Generate a presigned URL for downloading the filtered dataset
    logger.debug("Generating presigned URL for filtered dataset")
    download_url = await acreate_presigned_url(
        aioboto3_session,
        bucket=bucket_name,
        s3_key=dataset_info.filtered_dataset_key,
    )
    if download_url is None:
        logger.error("Failed to generate presigned URL")
        raise Exception("Failed to generate presigned URL")

    logger.debug(
        f"Generated presigned URL: {download_url[:50]}..."
        if download_url
        else "No download URL generated"
    )

    # Generate a presigned URL for downloading the JSON blob if it exists
    json_blob_download_url = None
    if dataset_info.filtered_dataset_json_blob_key:
        logger.debug("Generating presigned URL for JSON blob")
        json_blob_download_url = await acreate_presigned_url(
            aioboto3_session,
            bucket=bucket_name,
            s3_key=dataset_info.filtered_dataset_json_blob_key,
        )

        if json_blob_download_url:
            logger.debug(
                f"Generated JSON blob presigned URL: {json_blob_download_url[:50]}..."
            )
        else:
            logger.warning("Failed to generate presigned URL for JSON blob")

    logger.info(
        f"Successfully created filtered dataset with {metadata['row_count']} rows"
    )
    return FilterResponse(
        status=StatusCode.ok,
        message="Dataset filtered successfully",
        filtered_dataset_key=dataset_info.filtered_dataset_key,
        download_url=download_url,
        filtered_dataset_json_blob_key=dataset_info.filtered_dataset_json_blob_key,
        json_blob_download_url=json_blob_download_url,
        metadata_json=FilterMetadata(
            generated_at=metadata["generated_at"],
            original_file_key=metadata["original_file_key"],
            user_id=metadata["user_id"],
            filters=metadata["filters"],
            filtered_dataset_key=metadata["filtered_dataset_key"],
            filtered_dataset_json_blob_key=metadata.get(
                "filtered_dataset_json_blob_key"
            ),
            row_count=metadata["row_count"],
            original_file_name_without_extension=metadata[
                "original_file_name_without_extension"
            ],
            original_filename=metadata["original_filename"],
        ),
    )


async def _process_dataset_in_temp_dir(
    async_s3_client: Any,
    bucket_name: str,
    dataset_info: DatasetInfo,
    user_id: str,
    filters: GroupLabelColumnFilters,
    temp_dir: str,
    timestamp_column: str,
):
    """Process the dataset in a temporary directory.

    This function downloads a dataset from S3, applies filters, and uploads the filtered
    dataset back to S3 along with metadata.

    Args:
        async_s3_client: Boto3 S3 client for AWS operations
        bucket_name (str): Name of the S3 bucket
        dataset_info (DatasetInfo): Information about the dataset
        user_id (str): ID of the user who owns the dataset
        filters (GroupLabelColumnFilters): Filter configuration to apply (may have empty filter list for unfiltered data)
        temp_dir (str): Path to temporary directory for processing
        timestamp_column (str): Name of the timestamp column currently selected

    Returns:
        dict: Metadata about the filtered dataset

    Raises:
        Exception: If upload of filtered dataset or metadata fails
    """
    # Download the original file from S3
    local_input_path = os.path.join(temp_dir, dataset_info.original_filename)
    logger.debug(f"Downloading original file from S3 to: {local_input_path}")
    _ = await async_s3_client.download_file(
        Bucket=bucket_name,
        Key=dataset_info.original_file_key,
        Filename=local_input_path,
    )
    logger.debug("Downloaded original file successfully")

    # Read and filter the dataset
    filtered_df = await _read_and_filter_dataset(
        local_input_path, filters, timestamp_column
    )

    # Save the filtered dataset locally
    local_output_path = await _save_filtered_dataset(
        filtered_df,
        temp_dir,
        dataset_info.original_file_name_without_extension,
    )

    # Also save as JSON blob
    local_json_blob_path = await _save_filtered_dataset_as_json_blob(
        filtered_df, temp_dir, dataset_info.original_file_name_without_extension
    )

    # Upload the filtered dataset to S3
    logger.debug(
        f"Uploading filtered dataset to S3 at: {dataset_info.filtered_dataset_key}"
    )
    upload_success = await async_upload_file_to_s3(
        async_s3_client=async_s3_client,
        local_file=local_output_path,
        bucket=bucket_name,
        s3_key=str(dataset_info.filtered_dataset_key),
    )

    os.remove(local_output_path)
    logger.debug(f"Removed temporary local file: {local_output_path}")

    if not upload_success:
        logger.error(
            f"Failed to upload filtered dataset to S3: {dataset_info.filtered_dataset_key}"
        )
        raise Exception("Failed to upload filtered dataset to S3")

    # Generate a key for the JSON blob and upload it
    json_blob_key = (
        f"{os.path.dirname(dataset_info.filtered_dataset_key)}/data.json"
    )
    logger.debug(f"Uploading JSON blob to S3 at: {json_blob_key}")

    json_upload_success = await async_upload_file_to_s3(
        async_s3_client=async_s3_client,
        local_file=local_json_blob_path,
        bucket=bucket_name,
        s3_key=json_blob_key,
    )

    os.remove(local_json_blob_path)
    logger.debug(f"Removed temporary local JSON file: {local_json_blob_path}")

    # Set the JSON blob key in the dataset_info object if upload was successful
    if json_upload_success:
        dataset_info.filtered_dataset_json_blob_key = json_blob_key
        logger.debug(f"Successfully uploaded JSON blob to: {json_blob_key}")
    else:
        logger.warning(f"Failed to upload JSON blob to S3: {json_blob_key}")
        # Don't raise an exception for JSON upload failure,
        # since the main filtered dataset was uploaded successfully

    # Create and upload metadata
    return await _create_and_upload_metadata(
        async_s3_client,
        bucket_name,
        dataset_info,
        user_id,
        filters,
        filtered_df,
        temp_dir,
    )


async def _read_and_filter_dataset(
    local_input_path: str,
    filters: GroupLabelColumnFilters,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Read and filter a dataset from a local file.

    Args:
        local_input_path: Path to the local file to read
        filters: Filter configuration to apply (may have empty filter list for unfiltered data)
        timestamp_column: Name of the timestamp column currently selected
    Returns:
        Filtered pandas DataFrame

    Raises:
        HTTPException: If filter columns are not found in the dataset
    """
    logger.debug(f"Reading dataset from: {local_input_path}")
    df = pd.read_parquet(local_input_path)
    logger.info(f"Original dataset loaded with {len(df)} rows")

    # If no column filters provided, apply aggregation only if there are duplicate timestamps
    if not filters.filter:
        logger.info(
            "No column filters provided, checking for duplicate timestamps"
        )
        if not df[timestamp_column].is_unique:
            logger.info(
                f"Found duplicate timestamps, applying {filters.aggregation_func.value} aggregation"
            )
            try:
                # Use FilterUtils helper function for aggregation (includes validation)
                df = FilterUtils._aggregate_dataframe(
                    df, timestamp_column, filters.aggregation_func
                )
                logger.info(f"After aggregation: {len(df)} rows")
            except ValueError as e:
                # Convert ValueError from utility function to HTTPException
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )
        else:
            logger.info(
                "No duplicate timestamps found, returning entire dataset"
            )
        return df

    # Validate filter columns before applying filters
    logger.debug("Validating filter columns against dataframe columns")
    is_valid, missing_columns = FilterUtils.validate_filter_columns(df, filters)
    if not is_valid:
        error_message = (
            f"Filter columns not found in dataset: {missing_columns}"
        )
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filter request: {error_message}",
        )

    # Apply filters (including aggregation within FilterUtils.filter_and_aggregate_dataframe)
    logger.debug("Applying filters to dataframe")
    filtered_df = FilterUtils.filter_and_aggregate_dataframe(
        df, filters=filters, timestamp_column=timestamp_column
    )
    logger.info(
        f"Filtered dataset contains {len(filtered_df)} rows (from original {len(df)} rows)"
    )

    return filtered_df


async def _save_filtered_dataset(
    filtered_df: pd.DataFrame,
    temp_dir: str,
    dataset_name: str,
) -> str:
    """
    Save the filtered dataset to a local file.

    Args:
        filtered_df: DataFrame containing the filtered data
        temp_dir: Path to the temporary directory
        dataset_name: Name of the dataset

    Returns:
        Path to the saved local file
    """
    local_output_path = os.path.join(
        temp_dir, f"{dataset_name}_filtered.parquet"
    )
    logger.debug(f"Saving filtered Parquet to: {local_output_path}")
    filtered_df.to_parquet(local_output_path, index=False)

    return local_output_path


async def _save_filtered_dataset_as_json_blob(
    filtered_df: pd.DataFrame, temp_dir: str, dataset_name: str
) -> str:
    """
    Save the filtered dataset as a JSON blob to a local file.

    Args:
        filtered_df: DataFrame containing the filtered data
        temp_dir: Path to the temporary directory
        dataset_name: Name of the dataset

    Returns:
        Path to the saved local JSON file
    """
    local_output_path = os.path.join(temp_dir, f"{dataset_name}_filtered.json")

    logger.debug(
        f"Converting DataFrame to JSON and saving to: {local_output_path}"
    )

    # Convert the DataFrame to a JSON string
    # Orient 'records' gives a list of objects with column names as keys
    json_data = filtered_df.to_json(orient="records", date_format="iso")

    # Write the JSON blob to the file
    with open(local_output_path, "w") as f:
        json.dump(json.loads(json_data), f, indent=2)

    logger.debug(f"JSON blob written to: {local_output_path}")

    return local_output_path


async def _create_and_upload_metadata(
    async_s3_client: Any,
    bucket_name: str,
    dataset_info: DatasetInfo,
    user_id: str,
    filters: GroupLabelColumnFilters,
    filtered_df: pd.DataFrame,
    temp_dir: str,
) -> Dict[str, Any]:
    """
    Create and upload metadata for the filtered dataset.

    Args:
        s3_client: Boto3 S3 client instance
        bucket_name: Name of the S3 bucket
        dataset_info: Information about the dataset
        user_id: ID of the user who owns the dataset
        filters: Filter configuration applied to the dataset
        filtered_df: DataFrame containing the filtered data
        temp_dir: Path to the temporary directory

    Returns:
        Dictionary containing the metadata

    Raises:
        Exception: If upload of metadata fails
    """
    metadata = FilterMetadata(
        # Fields from DatasetInfo
        original_file_key=dataset_info.original_file_key,
        original_filename=dataset_info.original_filename,
        original_file_name_without_extension=dataset_info.original_file_name_without_extension,
        filtered_dataset_key=dataset_info.filtered_dataset_key,
        filtered_dataset_json_blob_key=dataset_info.filtered_dataset_json_blob_key,
        # Additional fields for FilterMetadata
        generated_at=datetime.now().isoformat(),
        user_id=user_id,
        filters=filters.model_dump(),
        row_count=len(filtered_df),
    )

    logger.debug(f"Created metadata: {metadata.model_dump()}")

    # Save metadata locally first
    metadata_local_path = os.path.join(temp_dir, "metadata.json")
    logger.debug(f"Saving metadata to local file: {metadata_local_path}")
    with open(metadata_local_path, "w") as f:
        json.dump(metadata.model_dump(), f)

    # Upload metadata to S3
    metadata_key = FilterUtils.get_metadata_key(
        user_id=user_id,
        dataset_file_name=dataset_info.original_file_name_without_extension,
        filters=filters,
    )
    logger.debug(f"Uploading metadata to S3: {metadata_key}")
    metadata_upload_success = await async_upload_file_to_s3(
        async_s3_client=async_s3_client,
        local_file=metadata_local_path,
        bucket=bucket_name,
        s3_key=metadata_key,
    )

    if not metadata_upload_success:
        logger.error(f"Failed to upload metadata to S3: {metadata_key}")
        raise Exception("Failed to upload metadata to S3")

    return metadata.model_dump()


def _setup_forecast_environment() -> Tuple[FoundationModelApiSettings, str]:
    """Set up the environment for forecast generation.

    Returns:
        Tuple containing settings and bucket_name

    Raises:
        ValueError: If S3 bucket name is not configured
    """
    settings = get_settings(FoundationModelApiSettings)
    bucket_name = settings.bucket_name

    if not bucket_name:
        logger.error("S3 bucket name not configured in settings")
        raise ValueError("S3 bucket name not configured")

    return settings, bucket_name


def _load_dataset_file(
    local_file_path: str, file_name: str
) -> Optional[pd.DataFrame]:
    """Load a dataset file into a pandas DataFrame.

    Args:
        local_file_path: Path to the local file
        file_name: Name of the file

    Returns:
        DataFrame or None if loading fails
    """
    logger.debug(
        f"Loading file {local_file_path}, format: {file_name.split('.')[-1]}"
    )
    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(local_file_path)
            logger.debug("CSV file loaded successfully")
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(local_file_path)
            logger.debug("Parquet file loaded successfully")
        else:
            logger.debug(f"Unsupported file format: {file_name}")
            return None

        return df
    except Exception as e:
        logger.error(f"Error loading dataset file: {str(e)}")
        return None


def _validate_required_columns(
    df: pd.DataFrame, config: FoundationModelConfig
) -> bool:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: DataFrame to validate
        config: Configuration with timestamp_column, timeseries_columns, and covariate_columns

    Returns:
        True if all required columns exist, False otherwise
    """
    required_columns = (
        [config.timestamp_column]
        + config.timeseries_columns
        + (config.covariate_columns)
    )
    logger.debug(f"Required columns: {required_columns}")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_message = (
            f"Required columns missing from dataset: {missing_columns}"
        )
        logger.error(error_message)
        return False

    return True


def _calculate_time_delta(
    df: pd.DataFrame, timestamp_column: str
) -> Union[pd.Timedelta, relativedelta]:
    """Calculate the most common time interval between timestamps.

    Args:
        df: DataFrame containing timestamp data
        timestamp_column: Name of the timestamp column

    Returns:
        Union[pd.Timedelta, relativedelta]: The most common time interval
    """
    # Calculate time differences
    time_diffs = df[timestamp_column].diff().dropna()

    # First check if data appears to be monthly/yearly
    month_diffs = time_diffs.dt.days / 30.44  # Average days per month
    year_diffs = time_diffs.dt.days / 365.25  # Average days per year

    # Round to nearest integer for monthly/yearly detection
    rounded_month_diffs = month_diffs.round()
    rounded_year_diffs = year_diffs.round()

    # Get most common intervals
    mode_month_diff = stats.mode(rounded_month_diffs, keepdims=False)[0]
    mode_year_diff = stats.mode(rounded_year_diffs, keepdims=False)[0]

    # Check if the data appears to be monthly
    if (
        abs(mode_month_diff - 1) < 0.1
    ):  # Allow small deviation from exactly 1 month
        # Check if day of month is consistent
        days_of_month = df[timestamp_column].dt.day
        day_counts = days_of_month.value_counts()
        most_common_day = day_counts.index[0]
        day_frequency = day_counts.iloc[0] / len(days_of_month)

        # Check if dates are consistently on the last day of month
        is_last_day = df[timestamp_column].dt.is_month_end
        last_day_frequency = is_last_day.mean()

        logger.debug(
            f"Monthly data detected. Day of month: {most_common_day}, frequency: {day_frequency:.2%}, "
            f"Last day of month frequency: {last_day_frequency:.2%}"
        )

        if day_frequency > 0.8:  # If more than 80% of dates are on the same day
            logger.debug(
                f"Detected monthly interval (consistent day {most_common_day})"
            )
            return relativedelta(months=1)
        elif (
            last_day_frequency > 0.8
        ):  # If more than 80% of dates are on the last day of month
            logger.debug("Detected monthly interval (last day of month)")
            return relativedelta(months=1)
    elif abs(mode_month_diff - 3) < 0.1:  # Quarterly
        logger.debug("Detected quarterly interval")
        return relativedelta(months=3)
    elif abs(mode_month_diff - 6) < 0.1:  # Semi-annual
        logger.debug("Detected semi-annual interval")
        return relativedelta(months=6)

    # Check if the data appears to be yearly
    if (
        abs(mode_year_diff - 1) < 0.1
    ):  # Allow small deviation from exactly 1 year
        # Check if month and day are consistent
        month_day = df[timestamp_column].dt.strftime("%m-%d")
        month_day_counts = month_day.value_counts()
        most_common_month_day = month_day_counts.index[0]
        month_day_frequency = month_day_counts.iloc[0] / len(month_day)

        logger.debug(
            f"Yearly data detected. Month-day: {most_common_month_day}, frequency: {month_day_frequency:.2%}"
        )

        if (
            month_day_frequency > 0.8
        ):  # If more than 80% of dates are on the same month-day
            logger.debug(
                f"Detected yearly interval (consistent date {most_common_month_day})"
            )
            return relativedelta(years=1)

    # If not monthly/yearly, fall back to seconds-based approach
    time_diffs_seconds = time_diffs.dt.total_seconds()
    logger.debug(
        f"Time differences range (seconds): {time_diffs_seconds.min()} to {time_diffs_seconds.max()}"
    )

    # Handle sub-second intervals by checking if all differences are less than 1 second
    if time_diffs_seconds.max() < 1.0:
        # Check if data appears to be in millisecond frequency (between 0.001 and 1.0 seconds)
        if time_diffs_seconds.min() >= 0.001 and time_diffs_seconds.max() < 1.0:
            # For millisecond data, use millisecond precision
            time_diffs_milliseconds = time_diffs.dt.total_seconds() * 1_000
            rounded_diffs_milliseconds = time_diffs_milliseconds.round()
            logger.debug(
                f"Millisecond data detected. Using millisecond precision: {rounded_diffs_milliseconds.value_counts().head()}"
            )
            mode_diff_milliseconds = stats.mode(
                rounded_diffs_milliseconds, keepdims=False
            )[0]
            time_delta = pd.Timedelta(
                milliseconds=float(mode_diff_milliseconds)
            )
            logger.debug(f"Created millisecond time delta: {time_delta}")
            return time_delta
        else:
            # For sub-millisecond data, use microsecond precision
            time_diffs_microseconds = time_diffs.dt.total_seconds() * 1_000_000
            rounded_diffs_microseconds = time_diffs_microseconds.round()
            logger.debug(
                f"Sub-millisecond data detected. Using microsecond precision: {rounded_diffs_microseconds.value_counts().head()}"
            )
            mode_diff_microseconds = stats.mode(
                rounded_diffs_microseconds, keepdims=False
            )[0]
            time_delta = pd.Timedelta(
                microseconds=float(mode_diff_microseconds)
            )
            logger.debug(f"Created sub-millisecond time delta: {time_delta}")
            return time_delta
    else:
        # For data with intervals >= 1 second, round to nearest second
        rounded_diffs = time_diffs_seconds.round()
        logger.debug(
            f"Rounded time differences: {rounded_diffs.value_counts().head()}"
        )
        mode_diff_seconds = stats.mode(rounded_diffs, keepdims=False)[0]
        time_delta = pd.Timedelta(seconds=float(mode_diff_seconds))
        logger.debug(f"Created time delta: {time_delta}")
        return time_delta


def _generate_future_timestamps(
    df: pd.DataFrame,
    forecasting_timestamp: pd.Timestamp,
    time_delta: pd.Timedelta | relativedelta,
    forecast_length: int,
) -> List[datetime]:
    """Generate future timestamps for forecasting.

    Args:
        df: DataFrame containing historical data
        forecasting_timestamp: Timestamp to start forecasting from
        time_delta: Time interval between forecasts
        forecast_length: Number of future timestamps to generate
    Returns:
        List[datetime]: List of future timestamps
    """
    if len(df) > 1:
        # Generate future timestamps using the time delta
        future_timestamps = [
            (forecasting_timestamp + (i + 1) * time_delta).to_pydatetime()
            for i in range(forecast_length)
        ]
    else:
        logger.debug(
            "Too few data points to determine pattern, defaulting to daily intervals"
        )
        # Default to daily if we can't determine the pattern
        future_timestamps = [
            (forecasting_timestamp + pd.Timedelta(days=i + 1)).to_pydatetime()
            for i in range(forecast_length)
        ]

    logger.debug(
        f"Generated {len(future_timestamps)} future timestamps: {future_timestamps[0]} to {future_timestamps[-1]}"
    )

    return future_timestamps


def _process_timestamps(df: pd.DataFrame, config) -> tuple:
    """Process timestamps in the dataset.

    Args:
        df: DataFrame containing timestamp data
        config: Configuration with timestamp_column, min_timestamp, etc.
        delta: If provided, use this as the stride duration. If not provided, calculate the time delta from the data.

    Returns:
        tuple: (filtered_df, ground_truth_df, forecasting_timestamp, future_timestamps)
            - filtered_df: DataFrame filtered by time range
            - ground_truth_df: DataFrame containing ground truth data
            - forecasting_timestamp: Timestamp to start forecasting from
            - future_timestamps: List of future timestamps
    """

    if len(df) < 2:
        raise ValueError(f"Not enough data: {len(df)=} must be at least 2")

    df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    # sort by the timestamps
    df = df.sort_values(by=config.timestamp_column)
    logger.debug(
        f"Timestamp range: {df[config.timestamp_column].iloc[0]} to {df[config.timestamp_column].iloc[-1]}"
    )

    # Get the timezone from the first timestamp in the dataframe
    tz = df[config.timestamp_column].iloc[0].tz
    logger.debug(f"Timezone: {tz}")

    # Convert min_timestamp and forecasting_timestamp to datetime with the same timezone
    min_timestamp = pd.to_datetime(config.min_timestamp)
    forecasting_timestamp = pd.to_datetime(config.forecasting_timestamp)
    if min_timestamp.tz is None:
        min_timestamp = min_timestamp.tz_localize(tz)
    else:
        min_timestamp = min_timestamp.tz_convert(tz)
    if forecasting_timestamp.tz is None:
        forecasting_timestamp = forecasting_timestamp.tz_localize(tz)
    else:
        forecasting_timestamp = forecasting_timestamp.tz_convert(tz)

    # Adjust timestamps to align with actual data points
    # Get timestamps in the specified range without creating a new DataFrame
    timestamp_mask = (df[config.timestamp_column] >= min_timestamp) & (
        df[config.timestamp_column] <= forecasting_timestamp
    )
    timestamps_in_range = df.loc[timestamp_mask, config.timestamp_column]

    if len(timestamps_in_range) == 0:
        actual_min_timestamp = min_timestamp
        actual_forecasting_timestamp = forecasting_timestamp
        logger.warning(
            f"No data available in the specified range: {min_timestamp} to {forecasting_timestamp}"
        )
    else:
        # Adjust min_timestamp to the first actual timestamp in the filtered data
        actual_min_timestamp = timestamps_in_range.iat[0]
        actual_forecasting_timestamp = timestamps_in_range.iat[-1]
        logger.debug(
            f"Adjusted filtering data from {actual_min_timestamp} to {actual_forecasting_timestamp}"
        )

    # Create a deep copy of the dataframe before filtering
    original_df = df.copy(deep=True)
    logger.debug(
        f"Created deep copy of dataframe with shape: {original_df.shape}"
    )

    # Get the last timestamp in the original data
    last_timestamp = original_df[config.timestamp_column].iloc[-1]
    logger.debug(f"Last timestamp in original data: {last_timestamp}")

    # Calculate time delta first so we can use it for adjusting the forecasting timestamp
    time_delta = _calculate_time_delta(df, config.timestamp_column)

    # Calculate the maximum forecast timestamp using the actual forecasting timestamp
    forecast_end = actual_forecasting_timestamp + (
        time_delta * config.forecast_length
    )
    logger.debug(f"Calculated forecast end: {forecast_end}")

    max_gt_forecast_timestamp = min(last_timestamp, forecast_end)
    logger.debug(f"Maximum forecast timestamp: {max_gt_forecast_timestamp}")

    # Filter original dataframe to get only timeseries columns within the time range
    ground_truth_df = original_df[
        (original_df[config.timestamp_column] >= actual_min_timestamp)
        & (original_df[config.timestamp_column] <= max_gt_forecast_timestamp)
    ][[config.timestamp_column] + config.timeseries_columns]
    logger.debug(f"Filtered dataframe shape: {ground_truth_df.shape}")

    logger.debug("Created deep copy of original dataframe")

    # Filter the dataframe to only include timestamps before the actual forecasting timestamp
    df = df[
        (df[config.timestamp_column] >= actual_min_timestamp)
        # Note - we do not clip the future timestamps because we want to enable leaking metadata in some cases.
        # & (df[config.timestamp_column] <= forecasting_timestamp)
    ]

    logger.debug(f"After filtering, dataset contains {len(df)} rows")

    # Generate future timestamps from the actual forecasting timestamp
    future_timestamps = _generate_future_timestamps(
        df,
        actual_forecasting_timestamp,
        time_delta,
        config.forecast_length,
    )

    # remove tz localization since everthing in same tz now
    # (and downstream models don't like tz)
    if isinstance(df[config.timestamp_column].dtype, pd.DatetimeTZDtype):
        df[config.timestamp_column] = df[
            config.timestamp_column
        ].dt.tz_localize(None)
    if isinstance(
        ground_truth_df[config.timestamp_column].dtype, pd.DatetimeTZDtype
    ):
        ground_truth_df[config.timestamp_column] = ground_truth_df[
            config.timestamp_column
        ].dt.tz_localize(None)
    if getattr(actual_forecasting_timestamp, "tzinfo", None) is not None:
        actual_forecasting_timestamp = actual_forecasting_timestamp.replace(
            tzinfo=None
        )
    future_timestamps = [
        ts.replace(tzinfo=None)
        if getattr(ts, "tzinfo", None) is not None
        else ts
        for ts in future_timestamps
    ]

    return df, ground_truth_df, actual_forecasting_timestamp, future_timestamps


def _generate_model_forecast(
    df: pd.DataFrame,
    config: FoundationModelConfig,
    user_id: str,
    forecasting_timestamp: pd.Timestamp,
    future_timestamps: List[datetime],
    ground_truth_df: pd.DataFrame,
    metadata_dataframes: List[MetadataDataFrame],
) -> ForecastDataset:
    """Generate forecasts using the foundation model.

    Args:
        df: DataFrame containing historical data
        config: Configuration with model_type, timestamp_column, etc.
        user_id: ID of the user
        forecasting_timestamp: Timestamp to start forecasting from
        future_timestamps: List of future timestamps
        ground_truth_df: DataFrame containing ground truth data
        metadata_dataframes: List of metadata dataframes

    Returns:
        ForecastDataset: Forecast results

    Raises:
        Exception: If forecast generation fails
    """
    # Sort by timestamp for the entire dataset
    logger.info("Generating forecast for entire dataset")
    logger.debug(
        f"Calling predict with target_columns={config.timeseries_columns}"
    )

    # Initialize the foundation model
    logger.info(f"Initializing {config.model_type} model")
    logger.debug(
        f"Getting model instance for model_type={config.model_type}, user_id={user_id}"
    )

    model = FoundationModelService.get_model(
        model_type=config.model_type, user_id=user_id
    )
    logger.debug(f"Model initialized: {model.__class__.__name__}")

    df = df.sort_values(by=config.timestamp_column)
    logger.debug("Dataset sorted by timestamp column")

    try:
        # Generate forecasts for the entire dataset
        logger.debug("Starting prediction")
        forecast_df = model.predict(
            target_df=df,
            covariate_columns=config.covariate_columns,
            covariate_columns_to_leak=config.covariate_columns_to_leak,
            target_columns=config.timeseries_columns,
            forecasting_timestamp=forecasting_timestamp,
            metadata_dataframes=metadata_dataframes,
            metadata_dataframes_leak_idxs=config.metadata_dataframes_leak_idxs,
            future_time_stamps=future_timestamps,
            ground_truth_df=ground_truth_df,
            do_llm_explanation=config.do_llm_explanation,
            timestamp_column=config.timestamp_column,
        )

        return forecast_df
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        logger.debug(
            f"Forecast prediction error details: {e.__class__.__name__}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


async def _download_and_load_dataset(
    aioboto3_session: aioboto3.Session,
    bucket_name: str,
    file_path_key: str,
) -> Optional[pd.DataFrame]:
    """Download and load a dataset from S3 into a pandas DataFrame.

    Args:
        aioboto3_session: aioboto3.Session for S3 operations
        bucket_name: Name of the S3 bucket
        file_path_key: S3 key of the file to download

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if loading fails
    """
    if file_path_key is None:
        logger.debug("file_path_key is None")
        return None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = file_path_key.split("/")[-1]
            local_file_path = os.path.join(temp_dir, file_name)
            logger.debug(f"Local file path: {local_file_path}")

            logger.debug(
                f"Downloading file {file_path_key} to {local_file_path}"
            )
            async with aioboto3_session.client("s3") as async_s3_client:  # type: ignore
                await async_s3_client.download_file(
                    Bucket=bucket_name,
                    Key=file_path_key,
                    Filename=local_file_path,
                )
            logger.debug("File downloaded successfully")

            # Load the dataset using the refactored functions
            df = await asyncio.to_thread(
                _load_dataset_file, local_file_path, file_name
            )

            return df
    except Exception as e:
        logger.error(f"Error downloading and loading dataset: {str(e)}")
        return None


async def do_forecast_core_code(
    request: ApiForecastRequest,
    aioboto3_session: aioboto3.Session,
    fastapi_request=None,
) -> ApiForecastResponse:
    """Generate forecasts from uploaded time series data."""
    logger.add(sys.stdout, level="DEBUG")
    try:
        logger.debug("Starting generate_forecast function")
        # Extract the config and user_id from the request
        config: FoundationModelConfig = request.config
        user_id = request.user_id

        logger.info(
            f"Received forecast request for user_id: {user_id}, model: {config.model_type}"
        )
        logger.debug(
            f"Request config: file_path_key={config.file_path_key}, timestamp_column={config.timestamp_column}, forecast_length={config.forecast_length}"
        )

        # Get settings and S3 bucket
        try:
            settings, bucket_name = _setup_forecast_environment()
        except ValueError as e:
            return ApiForecastResponse(
                status=StatusCode.bad_request,
                message=str(e),
                dataset=None,
            )

        # Validate file exists in S3
        logger.debug(f"Validating file exists in S3: {config.file_path_key}")
        async with aioboto3_session.client("s3") as async_s3_client:  # type: ignore
            exists, message = await avalidate_file_exists(
                async_s3_client, bucket_name, config.file_path_key
            )
            if not exists:
                logger.debug(f"File not found: {config.file_path_key}")
                return ApiForecastResponse(
                    status=StatusCode.not_found,
                    message=message,
                    dataset=None,
                )
            else:
                logger.debug(
                    f"File validation successful: {config.file_path_key}"
                )

        # Download and preprocess data
        df = await _download_and_load_dataset(
            aioboto3_session=aioboto3_session,
            bucket_name=bucket_name,
            file_path_key=config.file_path_key,
        )
        if df is None:
            return ApiForecastResponse(
                status=StatusCode.bad_request,
                message="Failed to load dataset",
                dataset=None,
            )

        # Apply modifications to the entire dataset once, right after downloading
        # This ensures all subsequent processing (historical, ground truth, forecasting) uses modified data
        if config.full_and_range_modifications:
            logger.debug(
                "Applying full_and_range_modifications to entire dataset"
            )
            df = await asyncio.to_thread(
                apply_full_and_range_modifications,
                df=df,
                full_and_range_modifications=config.full_and_range_modifications,
                timestamp_column=config.timestamp_column,
            )

        if config.point_modifications:
            logger.debug("Applying point_modifications to entire dataset")
            df = await asyncio.to_thread(
                apply_point_modifications,
                df=df,
                timestamp_column=config.timestamp_column,
                point_modifications=config.point_modifications,
            )

        # Validate required columns
        if not await asyncio.to_thread(_validate_required_columns, df, config):
            return ApiForecastResponse(
                status=StatusCode.bad_request,
                message="Required columns missing from dataset",
                dataset=None,
            )

        # Process timestamps
        (
            df,
            ground_truth_df,
            forecasting_timestamp,
            future_timestamps,
        ) = await asyncio.to_thread(_process_timestamps, df, config)

        # If no data after filtering, return error
        if len(df) == 0:
            logger.debug("No data available in the specified time range")
            return ApiForecastResponse(
                status=StatusCode.bad_request,
                message="No data available in the specified time range",
                dataset=None,
            )

        # Prepare metadata dataframes
        metadata_dataframes, new_leak_idxs = await prepare_metadata_dataframes(
            config.metadata_info_combined,
            settings.metadata_datasets_bucket,
            df[config.timestamp_column].tolist(),
            config.metadata_dataframes_leak_idxs,
        )
        config.metadata_dataframes_leak_idxs = new_leak_idxs

        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="User ID is required",
            )

        # Track metadata usage for billing and analytics
        try:
            # Track metadata usage and store results for API usage middleware
            metadata_usage_stats = track_metadata_usage(
                metadata_info_combined=config.metadata_info_combined,
                metadata_dataframes=metadata_dataframes,
            )

            # Store metadata usage stats in request state for API usage middleware
            if fastapi_request and hasattr(fastapi_request, "state"):
                fastapi_request.state.metadata_usage_stats = (
                    metadata_usage_stats
                )
        except Exception as e:
            logger.error(f"Failed to track metadata usage: {str(e)}")

        try:
            # Generate forecasts
            forecast_df = await asyncio.to_thread(
                _generate_model_forecast,
                df=df,
                config=config,
                user_id=user_id,
                forecasting_timestamp=forecasting_timestamp,
                future_timestamps=future_timestamps,
                ground_truth_df=ground_truth_df,
                metadata_dataframes=metadata_dataframes,
            )

            logger.debug("Returning successful forecast response")
            # Convert the dataframe to a historical dataset (modifications already applied)
            try:
                historical_df = await asyncio.to_thread(
                    _dataframe_to_historical_dataset,
                    df[df[config.timestamp_column] <= forecasting_timestamp],
                    config.timestamp_column,
                    config.timeseries_columns,
                )
            except Exception as e:
                logger.error(
                    f"Error converting dataframe to historical dataset: {str(e)}"
                )
                historical_df = None

            return ApiForecastResponse(
                status=StatusCode.ok,
                message="Forecast generated successfully",
                dataset=forecast_df,
                historical_dataset=historical_df,
            )

        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            logger.debug(
                f"Forecast prediction error details: {e.__class__.__name__}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction error: {str(e)}",
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Error in generate_forecast: {str(e)}")
        logger.debug(
            f"Error type: {type(e)}, error args: {e.args if hasattr(e, 'args') else 'No args'}"
        )

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@router.put(
    "/api/foundation_models/forecast", response_model=ApiForecastResponse
)
async def generate_forecast(
    fastapi_request: Request,
    request: ApiForecastRequest = Body(...),
    user_id: Optional[str] = Depends(get_user_id_from_token_or_api_key),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> ApiForecastResponse:
    """Generate forecasts from uploaded time series data."""

    # Handle authentication and get user_id
    if user_id and request.user_id and request.user_id != user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id in request body does not match user_id in authorization header",
        )
    # Set the user_id from the authentication method
    if user_id:
        request.user_id = user_id

    # Warning for SHAP analysis
    warnings = None
    if request.config.do_shap_analysis:
        logger.warning(SHAP_ANALYSIS_WARNING_MSG)
        warnings = [SHAP_ANALYSIS_WARNING_MSG]

    response = await do_forecast_core_code(
        request,
        aioboto3_session,
        fastapi_request=fastapi_request,
    )
    response.warnings = warnings
    return response


@router.put(
    "/api/foundation_models/forecast/backtest",
    response_model=BacktestAPIForecastResponse,
)
async def generate_forecast_backtest(
    request: BacktestAPIForecastRequest = Body(...),
    user_id: Optional[str] = Depends(get_user_id_from_token_or_api_key),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> BacktestAPIForecastResponse:
    """Generate forecasts from uploaded time series data (now using Celery async task)."""
    logger.add(sys.stdout, level="DEBUG")
    try:
        logger.debug("Starting generate_forecast_backtest function")

        # Handle authentication and get user_id
        if request.user_id and user_id and request.user_id != user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id in request body does not match user_id in authorization header",
            )

        # Set the user_id from the authentication method
        if user_id:
            request.user_id = user_id

        config: FoundationModelConfig = request.config

        # Warning for SHAP analysis
        warnings = None
        if config.do_shap_analysis:
            logger.warning(SHAP_ANALYSIS_WARNING_MSG)
            warnings = [SHAP_ANALYSIS_WARNING_MSG]

        logger.info(
            f"Received forecast request for user_id: {request.user_id}, model: {config.model_type}"
        )
        logger.debug(
            f"Request config: file_path_key={config.file_path_key}, timestamp_column={config.timestamp_column}, forecast_length={config.forecast_length}"
        )

        # Get settings and S3 bucket
        try:
            settings, bucket_name = _setup_forecast_environment()
        except ValueError as e:
            return BacktestAPIForecastResponse(
                status=StatusCode.bad_request,
                message=str(e),
                presigned_url=None,
                task_id=None,
            )

        # Validate file exists in S3
        logger.debug(f"Validating file exists in S3: {config.file_path_key}")
        async with aioboto3_session.client("s3") as async_s3_client:  # type: ignore
            exists, message = await avalidate_file_exists(
                async_s3_client, bucket_name, config.file_path_key
            )
            if not exists:
                logger.debug(f"File not found: {config.file_path_key}")
                return BacktestAPIForecastResponse(
                    status=StatusCode.not_found,
                    message=message,
                    presigned_url=None,
                    task_id=None,
                )
            else:
                logger.debug(
                    f"File validation successful: {config.file_path_key}"
                )

        # Download and preprocess data
        df = await _download_and_load_dataset(
            aioboto3_session=aioboto3_session,
            bucket_name=bucket_name,
            file_path_key=config.file_path_key,
        )
        if df is None:
            return BacktestAPIForecastResponse(
                status=StatusCode.bad_request,
                message="Failed to load dataset",
                presigned_url=None,
                task_id=None,
            )

        # Compute forecast windows if stride is provided
        forecast_windows = []
        delta = None
        error_message = None
        if request.stride:
            forecast_windows, delta, error_message = _compute_forecast_windows(
                min_timestamp=pd.to_datetime(config.min_timestamp),
                start_forecast_timestamp=pd.to_datetime(
                    config.forecasting_timestamp
                ),
                max_forecast_timestamp=pd.to_datetime(
                    df[config.timestamp_column].max()
                ),
                stride=request.stride,
            )
            if error_message:
                return BacktestAPIForecastResponse(
                    status=StatusCode.bad_request,
                    message=error_message,
                    presigned_url=None,
                    task_id=None,
                )
            assert forecast_windows is not None
            assert delta is not None
        else:
            # If no stride, just use the single window from config
            forecast_windows = [
                (
                    pd.to_datetime(config.min_timestamp),
                    pd.to_datetime(config.forecasting_timestamp),
                )
            ]

        # Convert forecast_windows to list of (start, end) ISO strings
        forecast_windows_serializable = [
            (start.isoformat(), end.isoformat())
            for start, end in forecast_windows
        ]
        # Prepare config as dict
        config_dict = (
            config.model_dump()
            if hasattr(config, "model_dump")
            else dict(config)
        )

        if USE_CELERY:
            logger.info("Using Celery to run backtest forecast")

            # Import here to avoid circular import
            from synthefy_pkg.app.tasks import run_forecast_backtest

            # Call the Celery task asynchronously
            task = run_forecast_backtest.delay(
                df_serialized=json.loads(
                    df.to_json()
                ),  # serialize to json string
                forecast_windows=forecast_windows_serializable,
                config_as_dict=config_dict,
                user_id=request.user_id,
                group_filters_serialized=request.group_filters.model_dump(),
                bucket_name=bucket_name,
                file_path_key=config.file_path_key,
                delta_serialized=str(delta) if delta else None,
                covariate_grid=request.covariate_grid,
            )
            # Return the task id or status endpoint as appropriate
            return BacktestAPIForecastResponse(
                status=StatusCode.ok,
                message="Backtest task submitted.",
                presigned_url=None,
                task_id=task.id,
            )
        else:
            logger.info("Using direct execution to run backtest forecast")
            # Use direct execution
            if request.user_id is None:
                raise HTTPException(
                    status_code=401,
                    detail="User ID is required",
                )
            try:
                x = await _run_forecast_backtest_core(
                    df_serialized=json.loads(df.to_json()),
                    forecast_windows=forecast_windows_serializable,
                    config_as_dict=config_dict,
                    user_id=request.user_id,
                    group_filters_serialized=request.group_filters.model_dump(),
                    bucket_name=bucket_name,
                    file_path_key=config.file_path_key,
                    delta_serialized=str(delta) if delta else None,
                    covariate_grid=request.covariate_grid,
                )

            except Exception as e:
                logger.error(f"Error running backtest forecast: {str(e)}")
                return BacktestAPIForecastResponse(
                    status=StatusCode.internal_server_error,
                    message=str(e),
                    presigned_url=None,
                    task_id=None,
                )
            return BacktestAPIForecastResponse(
                status=StatusCode.ok
                if x.get("state") == "SUCCESS" or x.get("status") == "SUCCESS"
                else StatusCode.internal_server_error,
                message=x.get("message", ""),
                presigned_url=x.get("presigned_url"),
                task_id=None,
                warnings=warnings,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in generate_forecast: {str(e)}")
        logger.debug(
            f"Error type: {type(e)}, error args: {e.args if hasattr(e, 'args') else 'No args'}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


class ModelConfigOption(BaseModel):
    """Model for configuration options of a foundation model."""

    mode: List[str]


class FoundationModelInfo(BaseModel):
    """Information about an available foundation model."""

    name: str
    description: str
    config_options: ModelConfigOption


class AvailableModelsResponse(BaseModel):
    """Response model for listing available foundation models."""

    available_models: List[FoundationModelInfo]


@router.get(
    "/api/foundation_models/models", response_model=AvailableModelsResponse
)
async def list_models(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> AvailableModelsResponse:
    """List available foundation models for the users."""

    # Use the new authentication logic
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )

    available_model_list = []
    for (
        model_name,
        model_description,
    ) in FOUNDATION_MODEL_SETTINGS.available_models.items():
        available_model_list.append(
            FoundationModelInfo(
                name=model_name,
                description=model_description,
                config_options=ModelConfigOption(mode=["LOCAL"]),
            )
        )
    return AvailableModelsResponse(
        available_models=available_model_list,
    )


class UiMetadataDataFrame(BaseModel):
    df: Optional[Any] = None
    metadata_json: Optional[dict] = None
    description: str
    file_path_key: str
    download_url: Optional[str] = Field(
        default=None,
        description="Presigned URL for downloading the metadata dataframe",
    )
    timestamp_key: Optional[str] = None
    feature_names: Optional[List[str]] = None


class MetadataDataFramesResponse(BaseModel):
    """Response model for metadata dataframes."""

    dataframes: List[UiMetadataDataFrame]


@router.get(
    "/api/foundation_models/get_metadata_dataframes",
    response_model=MetadataDataFramesResponse,
)
async def get_metadata_dataframes(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
    db: Session = Depends(get_db),
) -> MetadataDataFramesResponse:
    """List available metadata dataframes from the foundation model metadata bucket."""
    logger.info("Starting get_metadata_dataframes request")

    # Use the new authentication logic
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )

    try:
        # Get settings to access the metadata datasets bucket
        logger.debug("Retrieving FoundationModelApiSettings")
        settings = get_settings(FoundationModelApiSettings)
        metadata_bucket = settings.metadata_datasets_bucket
        logger.debug(f"Retrieved metadata_datasets_bucket: {metadata_bucket}")

        if not metadata_bucket:
            logger.warning("Metadata datasets bucket not configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Metadata datasets bucket not configured",
            )

        # List all folders in the univariate directory
        logger.debug(f"Listing folders in s3://{metadata_bucket}/univariate/")
        try:
            logger.debug("Executing S3 list_objects_v2 operation")
            async with aioboto3_session.client("s3") as s3_client:  # type: ignore
                response = await s3_client.list_objects_v2(
                    Bucket=metadata_bucket, Prefix="univariate/", Delimiter="/"
                )
            logger.debug("S3 list_objects_v2 operation completed successfully")
        except ClientError as e:
            logger.error(f"S3 list_objects_v2 operation failed: {str(e)}")
            logger.error(
                f"Error response code: {e.response.get('Error', {}).get('Code')}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal s3 client call to list objects failed",
            )

        # Extract folder names from CommonPrefixes
        logger.debug("Extracting folder names from CommonPrefixes")
        folders = []
        common_prefixes = response.get("CommonPrefixes", [])
        logger.debug(f"Found {len(common_prefixes)} CommonPrefixes")

        for prefix in common_prefixes:
            folder_path = prefix.get("Prefix", "")
            logger.debug(f"Processing prefix: {folder_path}")
            folder_name = folder_path.strip("/").split("/")[-1]
            if folder_name:
                folders.append(folder_name)
                logger.debug(f"Added folder: {folder_name}")
            else:
                logger.warning(
                    f"Skipping empty folder name from prefix: {folder_path}"
                )

        logger.info(f"Found {len(folders)} metadata dataset folders: {folders}")

        # Use aioboto3 to fetch metadata files asynchronously
        logger.debug("Creating async session with aioboto3")
        async with aioboto3_session.client("s3") as s3_client:  # type: ignore
            # Create tasks for fetching metadata for each folder
            logger.debug("Creating tasks for fetching metadata")
            tasks = [
                fetch_folder_metadata(
                    s3_client, metadata_bucket, folder, aioboto3_session
                )
                for folder in folders
            ]
            # Gather all results
            logger.debug(f"Awaiting {len(tasks)} metadata fetch tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"All {len(results)} metadata fetch tasks completed")

        # Process results and filter out errors
        logger.debug("Processing metadata fetch results")
        metadata_dataframes = []
        for i, result in enumerate(results):
            folder = folders[i]
            if isinstance(result, Exception):
                logger.error(
                    f"Error fetching metadata for folder {folder}: {str(result)}"
                )
                continue

            if result is None:
                logger.warning(f"No metadata found for folder {folder}")
                continue

            metadata_dataframes.append(result)
            logger.debug(f"Added metadata dataframe for {folder}")

        logger.info(
            f"Successfully retrieved {len(metadata_dataframes)} metadata dataframes"
        )
        return MetadataDataFramesResponse(dataframes=metadata_dataframes)

    except Exception as e:
        logger.exception(f"Error retrieving metadata dataframes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving metadata dataframes: {str(e)}",
        )


async def fetch_folder_metadata(
    async_s3_client,
    bucket: str,
    folder: str,
    aioboto3_session: aioboto3.Session,
) -> Optional[UiMetadataDataFrame]:
    """Fetch metadata for a single folder asynchronously."""
    try:
        logger.debug(f"Fetching metadata for folder: {folder}")

        # Construct the metadata file path
        metadata_file_key = f"univariate/{folder}/{folder}_metadata.json"
        logger.debug(f"Metadata file key: {metadata_file_key}")

        # Get the metadata JSON file
        try:
            logger.debug(
                f"Executing async S3 get_object for: {metadata_file_key}"
            )
            metadata_response = await async_s3_client.get_object(
                Bucket=bucket, Key=metadata_file_key
            )

            # Read the response body
            metadata_body = await metadata_response["Body"].read()
            metadata_content = json.loads(metadata_body.decode("utf-8"))
            logger.debug(
                f"Metadata JSON parsed successfully: {list(metadata_content.keys())}"
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            logger.error(
                f"S3 get_object operation failed for {metadata_file_key}: {str(e)}"
            )
            logger.error(f"Error code: {error_code}")
            if error_code == "NoSuchKey":
                logger.warning(f"Metadata file not found: {metadata_file_key}")
            return None

        # Extract the description from the metadata
        if "text_descriptor" in metadata_content:
            description = metadata_content["text_descriptor"]
        else:
            try:
                description = metadata_content["columns"][0]["title"]
            except Exception:
                logger.warning(
                    f"No description found for dataset: {folder}, using default description"
                )
                description = f"Dataset: {folder}"

        logger.debug(
            f"Description: {description[:50]}..."
            if len(description) > 50
            else f"Description: {description}"
        )

        # Generate a presigned URL for the JSON file
        json_file_key = f"univariate/{folder}/{folder}.json"
        logger.debug(f"Generating presigned URL for: {json_file_key}")

        try:
            download_url = await acreate_presigned_url(
                aioboto3_session, bucket=bucket, s3_key=json_file_key
            )
            logger.debug(
                f"Presigned URL generated successfully for {json_file_key}"
            )
        except Exception as e:
            logger.error(
                f"Failed to generate presigned URL for {json_file_key}: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate download URL for {folder}: {str(e)}",
            )

        # Construct the parquet file path
        file_path_key = f"univariate/{folder}/{folder}.parquet"
        logger.debug(f"Constructed parquet file path: {file_path_key}")

        # Create and return the metadata dataframe entry
        return UiMetadataDataFrame(
            description=description,
            file_path_key=file_path_key,
            download_url=download_url,
        )

    except Exception as e:
        logger.error(
            f"Error processing metadata for folder {folder}: {str(e)}",
            exc_info=True,
        )
        raise e


@router.post("/api/foundation_models/generate_covariate_grid")
async def generate_covariate_grid_endpoint(
    request: CovariateGridRequest = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> CovariateGridResponse:
    """
    Generate all possible combinations of covariates for grid search.

    Args:
        request: CovariateGridRequest containing configuration for grid generation

    Returns:
        CovariateGridResponse: All possible combinations based on request parameters
    """

    # Use the new authentication logic
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )

    try:
        # Generate covariate combinations with the specified limit
        all_covariate_grid = generate_covariate_combinations(
            request.available_covariates, request.max_combinations
        )

        return CovariateGridResponse(
            status=StatusCode.ok,
            message=f"Generated {len(all_covariate_grid)} covariate combinations (max: {request.max_combinations})",
            covariate_grid=all_covariate_grid,
            total_combinations=len(all_covariate_grid),
        )
    except Exception as e:
        logger.error(f"Error generating covariate grid: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate covariate grid: {str(e)}",
        )


@router.get(
    "/api/foundation_models/categorical_features",
    response_model=CategoricalFeaturesResponse,
)
async def get_categorical_features(
    file_path_key: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
    db: Session = Depends(get_db),
) -> CategoricalFeaturesResponse:
    """
    Retrieve categorical features and their distinct values from a dataset in S3.

    Args:
        file_path_key: The S3 key of the dataset file

    Returns:
        CategoricalFeaturesResponse: List of categorical features and their values
    """
    logger.info(f"Getting categorical features for file: {file_path_key}")

    # Use the new authentication logic
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )

    try:
        # Get bucket name from config
        settings = get_settings(FoundationModelApiSettings)
        bucket_name = settings.bucket_name
        logger.debug(f"Using S3 bucket: {bucket_name}")

        if not bucket_name:
            logger.error("S3 bucket name not configured in settings")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="S3 bucket name not configured in settings",
            )

        # Create a temp file to store the downloaded dataset
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            # Download the file from S3
            logger.debug(f"Downloading file from S3: {file_path_key}")
            async with aioboto3_session.client("s3") as s3_client:  # type: ignore
                try:
                    await s3_client.download_file(
                        bucket_name, file_path_key, temp_file.name
                    )
                except ClientError as e:
                    logger.error(f"Error downloading file: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"File not found: {file_path_key}",
                    )

            # Read the file as a DataFrame
            logger.debug("Reading file into DataFrame")
            if file_path_key.endswith(".csv"):
                df = pd.read_csv(temp_file.name)
            elif file_path_key.endswith(".parquet"):
                df = pd.read_parquet(temp_file.name)
            else:
                logger.error(f"Unsupported file format: {file_path_key}")
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported file format. Only CSV and Parquet are supported.",
                )

            logger.info(f"Successfully loaded DataFrame with shape {df.shape}")

            # Use SmartApproximationUtils to identify categorical features
            logger.debug("Extracting categorical features")
            categorical_features_dict = await asyncio.to_thread(
                SmartApproximationUtils.get_categorical_features, df
            )

            # Convert to response format
            categorical_features_list = []
            for feature_name, values in categorical_features_dict.items():
                categorical_features_list.append(
                    CategoricalFeatureValues(
                        feature_name=feature_name, values=values
                    )
                )

            logger.info(
                f"Found {len(categorical_features_list)} categorical features"
            )

            return CategoricalFeaturesResponse(
                status=StatusCode.ok,
                message=f"Successfully extracted categorical features from {file_path_key}",
                categorical_features=categorical_features_list,
                file_path_key=file_path_key,
            )

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.exception(f"Error extracting categorical features: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract categorical features: {str(e)}",
        )


class TimeGranularity(str, Enum):
    """Enum representing the granularity of a timestamp column."""

    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    MILLISECOND = "millisecond"
    MICROSECOND = "microsecond"
    NANOSECOND = "nanosecond"


class TimestampAnalysis(BaseModel):
    """Model representing analysis of a timestamp column."""

    column_name: str
    min_timestamp: str
    max_timestamp: str
    granularity: TimeGranularity
    count: int
    missing_values: int


class TimestampAnalysisRequest(BaseModel):
    """Request model for timestamp analysis."""

    file_path_key: str
    timestamp_columns: List[str]


class TimestampAnalysisResponse(BaseModel):
    """Response model for timestamp analysis endpoint."""

    status: StatusCode
    message: str
    timestamp_analyses: List[TimestampAnalysis] = []
    file_path_key: str


@router.post(
    "/api/foundation_models/timestamp_analysis",
    response_model=TimestampAnalysisResponse,
)
async def analyze_timestamps(
    request: TimestampAnalysisRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
    db: Session = Depends(get_db),
) -> TimestampAnalysisResponse:
    """
    Analyze timestamp columns in a dataset to determine time ranges and granularity.

    Args:
        request: TimestampAnalysisRequest containing file_path_key and timestamp_columns

    Returns:
        TimestampAnalysisResponse: Analysis results for each timestamp column
    """
    # Use the new authentication logic
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )

    # TODO: Implement the functionality to analyze timestamp columns
    # 1. Determine which bucket to use based on file_path_key
    # 2. Download the file from S3
    # 3. Read the file as a DataFrame
    # 4. For each timestamp column:
    #    - Parse the timestamps
    #    - Determine min and max timestamps
    #    - Infer the granularity
    #    - Count total and missing values
    # 5. Return the analysis results

    return TimestampAnalysisResponse(
        status=StatusCode.ok,
        message="Stub implementation - not yet functional",
        timestamp_analyses=[],
        file_path_key=request.file_path_key,
    )


@router.get(
    "/api/foundation_models/timestamp_range",
    response_model=Dict[str, Any],
)
async def get_timestamp_range_for_column(
    file_path_key: str,
    column_name: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> Dict[str, Any]:
    """
    Get the minimum and maximum timestamps for a specified column in a dataset.

    Args:
        file_path_key: Key of the file in S3
        column_name: Name of the timestamp column to analyze
        aioboto3_session: Aioboto3 session for S3 access

    Returns:
        Dict containing column name and min/max timestamps
    """
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )
    try:
        # Create temporary file to store the downloaded data
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Download file from S3
            settings = get_settings(FoundationModelApiSettings)
            bucket_name = settings.bucket_name
            async with aioboto3_session.client("s3") as s3_client:  # type: ignore
                await s3_client.download_file(
                    bucket_name, file_path_key, temp_file.name
                )

            # Read the parquet file into a dataframe
            if file_path_key.endswith(".csv"):
                df = pd.read_csv(temp_file.name)
            elif file_path_key.endswith(".parquet"):
                df = pd.read_parquet(temp_file.name)
            else:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported file format. Only CSV and Parquet are supported.",
                )

        # Get timestamp range using the utility function
        min_timestamp, max_timestamp = get_timestamp_range(df, column_name)

        # Convert timestamps to ISO format strings for JSON serialization
        if hasattr(min_timestamp, "isoformat"):
            min_timestamp = min_timestamp.isoformat()
        if hasattr(max_timestamp, "isoformat"):
            max_timestamp = max_timestamp.isoformat()

        return {
            "column_name": column_name,
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except ClientError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path_key}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


def _dataframe_to_historical_dataset(
    df: pd.DataFrame, timestamp_column: str, target_columns: List[str]
) -> HistoricalDataset:
    """
    Convert a pandas DataFrame to a HistoricalDataset object for the foundation model API.

    Args:
        df: The pandas DataFrame containing historical data
        timestamp_column: The name of the column containing timestamps
        target_columns: List of column names to include as target variables

    Returns:
        HistoricalDataset: Historical dataset object for API response
    """
    # Convert timestamps to string format with optional fractional seconds
    timestamps = [
        format_timestamp_with_optional_fractional_seconds(ts)
        for ts in df[timestamp_column]
    ]

    # Create values dictionary mapping target columns to their values
    values = {col: df[col].tolist() for col in target_columns}

    # Create and return the HistoricalDataset object
    return HistoricalDataset(
        timestamps=timestamps, values=values, target_columns=target_columns
    )


def _compute_forecast_metrics(
    actuals: List[float | None],
    forecasts: List[float],
    include_all_metrics: bool = False,
) -> Dict[str, float]:
    """Compute forecast metrics between actual and forecast values.

    Args:
        actuals: List of actual observed values (can contain None or NaN values)
        forecasts: List of forecasted values
        include_all_metrics: If True, include all metrics; otherwise, only RMSE and NMAE

    Returns:
        Dictionary containing computed metrics.
    """
    metrics = {}

    # Find the last valid actual value index
    last_valid_idx = len(actuals)
    for i in range(len(actuals) - 1, -1, -1):
        if actuals[i] is not None and not pd.isna(actuals[i]):
            last_valid_idx = i + 1
            break

    # Clip both arrays to the length of valid actuals
    actuals = actuals[:last_valid_idx]
    forecasts = forecasts[:last_valid_idx]

    # Filter out None values and ensure lists are same length
    valid_pairs = [(a, f) for a, f in zip(actuals, forecasts) if a is not None]
    if not valid_pairs:
        if include_all_metrics:
            return {
                "mae": float("nan"),
                "mse": float("nan"),
                "rmse": float("nan"),
                "nmae": float("nan"),
                "mape": float("nan"),
                "smape": float("nan"),
                "residual": float("nan"),
            }
        else:
            return {
                "rmse": float("nan"),
                "nmae": float("nan"),
            }

    actuals_valid, forecasts_valid = zip(*valid_pairs)
    actuals_array = np.array(actuals_valid)
    forecasts_array = np.array(forecasts_valid)

    # Compute RMSE (sqrt of MSE)
    mse = mean_squared_error(actuals_array, forecasts_array)
    rmse = np.sqrt(mse)
    metrics["rmse"] = rmse

    # Compute NMAE (Normalized Mean Absolute Error)
    mae = mean_absolute_error(actuals_array, forecasts_array)
    mean_actual = np.mean(np.abs(actuals_array))
    if mean_actual != 0:
        nmae = mae / mean_actual
        metrics["nmae"] = nmae
    else:
        metrics["nmae"] = float("nan")

    if include_all_metrics:
        metrics["mae"] = mae
        metrics["mse"] = mse
        # Compute Residual (mean of actual - forecast)
        residual = np.mean(actuals_array - forecasts_array)
        metrics["residual"] = residual
        # Compute MAPE (excluding zeros in actuals)
        non_zero_mask = actuals_array != 0
        if np.any(non_zero_mask):
            mape = (
                np.mean(
                    np.abs(
                        (
                            actuals_array[non_zero_mask]
                            - forecasts_array[non_zero_mask]
                        )
                        / actuals_array[non_zero_mask]
                    )
                )
                * 100
            )
            metrics["mape"] = mape
        else:
            metrics["mape"] = float("nan")
        # Compute SMAPE (Symmetric Mean Absolute Percentage Error)
        if np.any(non_zero_mask):
            smape = (
                np.mean(
                    2
                    * np.abs(
                        forecasts_array[non_zero_mask]
                        - actuals_array[non_zero_mask]
                    )
                    / (
                        np.abs(actuals_array[non_zero_mask])
                        + np.abs(forecasts_array[non_zero_mask])
                    )
                )
                * 100
            )
            metrics["smape"] = smape
        else:
            metrics["smape"] = float("nan")

    return metrics


class ForecastResultTracker:
    """Class to track forecast results and history data during backtesting."""

    def __init__(self, include_all_metrics: bool = False):
        self.results = []
        self.history_data = []
        self.actuals_by_group = {}  # Track actuals for NMAE calculation
        self.include_all_metrics = include_all_metrics

    def add_history_data(
        self, timestamp: datetime, value: float, group_keys: tuple
    ):
        """Add historical data point."""
        self.history_data.append(
            {
                "Period": timestamp,
                "History Values": value,
                "Group Keys": group_keys,
            }
        )

    def add_forecast_result(
        self,
        forecast_start_timestamp: datetime,
        forecast_end_timestamp: datetime,
        model_name: str,
        forecasts: List[float],
        ground_truth: List[float | None],
        group_values: tuple[Any, ...],
        group_columns: List[str],
        history_timestamps: List[datetime],
        history_values: List[float],
        future_timestamps: List[datetime],
        covariate_leak_info: Optional[Dict[str, bool]] = None,
    ):
        """Add a forecast result with metrics if actual values are available."""
        history_keep_mask = [
            ts < forecast_start_timestamp for ts in history_timestamps
        ]
        history_timestamps = [
            ts
            for ts, mask in zip(history_timestamps, history_keep_mask)
            if mask
        ]
        history_values = [
            v for v, mask in zip(history_values, history_keep_mask) if mask
        ]

        # Extract covariate columns from the leak info
        covariate_columns = (
            list(covariate_leak_info.keys()) if covariate_leak_info else []
        )

        # Create covariate combination identifier that includes leak information
        covariate_combo_str = self._create_covariate_combination_string(
            covariate_columns, covariate_leak_info
        )

        # Create individual covariate columns with leak flags for CSV output
        covariate_details = self._create_covariate_details(
            covariate_columns, covariate_leak_info
        )

        result_dict = {
            "Group Value": f"({', '.join(f'{col}={val}' for col, val in zip(group_columns, group_values))})",
            "Forecast Start Timestamp": forecast_start_timestamp,
            "Forecast End Timestamp": forecast_end_timestamp,
            # "Model": model_name, #fill in once we have more models
            "Covariates": covariate_columns,
            "Covariates_Str": ", ".join(covariate_columns)
            if covariate_columns
            else "None",
            "Covariate_Combination_ID": covariate_combo_str,
            "Forecasts": forecasts,
            "Ground Truth": ground_truth,
            "History Timestamps": history_timestamps,
            "History Values": history_values,
            "Future Timestamps": future_timestamps,
        }

        # Add individual covariate details to the result dict
        result_dict.update(covariate_details)

        # If we have actual values, compute metrics
        if any(a is not None for a in ground_truth):
            metrics = _compute_forecast_metrics(
                ground_truth, forecasts, self.include_all_metrics
            )
            result_dict.update(metrics)

            # Track actuals for NMAE calculation
            if group_values not in self.actuals_by_group:
                self.actuals_by_group[group_values] = []
            self.actuals_by_group[group_values].extend(
                [a for a in ground_truth if a is not None]
            )

        self.results.append(result_dict)

    def _create_covariate_combination_string(
        self,
        covariate_columns: Optional[List[str]],
        covariate_leak_info: Optional[Dict[str, bool]],
    ) -> str:
        """Create a unique identifier for covariate combination including leak information."""
        if not covariate_columns:
            return "None"

        if covariate_leak_info is None:
            # If no leak info provided, assume all non-leaking
            covariate_leak_info = {col: False for col in covariate_columns}

        # Create string like: "Temperature(no_leak), Humidity(leak), CPI(no_leak)"
        covariate_parts = []
        for col in covariate_columns:
            leak_status = (
                "leak" if covariate_leak_info.get(col, False) else "no_leak"
            )
            covariate_parts.append(f"{col}({leak_status})")

        return ", ".join(covariate_parts)

    def _create_covariate_details(
        self,
        covariate_columns: Optional[List[str]],
        covariate_leak_info: Optional[Dict[str, bool]],
    ) -> Dict[str, Any]:
        """Create detailed covariate information for CSV output."""
        details = {}

        if not covariate_columns:
            details["Covariate_Count"] = 0
            details["Leaked_Covariate_Count"] = 0
            details["Non_Leaked_Covariate_Count"] = 0
            details["Leaked_Covariates"] = "None"
            details["Non_Leaked_Covariates"] = "None"
            return details

        if covariate_leak_info is None:
            # If no leak info provided, assume all non-leaking
            covariate_leak_info = {col: False for col in covariate_columns}

        leaked_covariates = [
            col
            for col in covariate_columns
            if covariate_leak_info.get(col, False)
        ]
        non_leaked_covariates = [
            col
            for col in covariate_columns
            if not covariate_leak_info.get(col, False)
        ]

        details["Covariate_Count"] = len(covariate_columns)
        details["Leaked_Covariate_Count"] = len(leaked_covariates)
        details["Non_Leaked_Covariate_Count"] = len(non_leaked_covariates)
        details["Leaked_Covariates"] = (
            ", ".join(leaked_covariates) if leaked_covariates else "None"
        )
        details["Non_Leaked_Covariates"] = (
            ", ".join(non_leaked_covariates)
            if non_leaked_covariates
            else "None"
        )

        return details

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame(self.results)

    def get_history_dataframe(self) -> pd.DataFrame:
        """Convert history data to DataFrame."""
        return pd.DataFrame(self.history_data)


def _compute_forecast_windows(
    min_timestamp: datetime,
    start_forecast_timestamp: datetime,
    max_forecast_timestamp: datetime,
    stride: str,
) -> (
    Tuple[List[Tuple[datetime, datetime]], relativedelta, None]
    | Tuple[None, None, str]
):
    """Compute forecast windows based on stride.

    Args:
        min_timestamp: Start timestamp for all windows
        start_forecast_timestamp: End timestamp for the last window
        max_forecast_timestamp: End timestamp for the last window
        stride: ISO 8601 duration string (e.g., "P28D" for 28 days, "P1M" for 1 month)

    Returns:
        List of tuples containing (window_start, window_end) timestamps
        delta: The stride duration
        error_message: Error message if there is an error
    Raises:
        ValueError: If stride is invalid or would result in too many windows
    """

    # Parse ISO 8601 duration string
    stride_duration = parse_duration(stride)

    # Convert to relativedelta
    delta = relativedelta(
        years=int(stride_duration.years)  # type: ignore
        if hasattr(stride_duration, "years")
        else 0,
        months=int(stride_duration.months)  # type: ignore
        if hasattr(stride_duration, "months")
        else 0,
        days=int(stride_duration.days)  # type: ignore
        if hasattr(stride_duration, "days")
        else 0,
        hours=int(stride_duration.hours)  # type: ignore
        if hasattr(stride_duration, "hours")
        else 0,
        minutes=int(stride_duration.minutes)  # type: ignore
        if hasattr(stride_duration, "minutes")
        else 0,
        seconds=int(stride_duration.seconds)  # type: ignore
        if hasattr(stride_duration, "seconds")
        else 0,
        microseconds=int(stride_duration.microseconds)  # type: ignore
        if hasattr(stride_duration, "microseconds")
        else 0,
    )

    # Check if delta is zero
    if (
        delta.years == 0
        and delta.months == 0
        and delta.days == 0
        and delta.hours == 0
        and delta.minutes == 0
        and delta.seconds == 0
        and delta.microseconds == 0
    ):
        return (
            None,
            None,
            f"Parsing error: stride duration cannot be zero: {stride=}",
        )

    windows = []
    current_end = start_forecast_timestamp

    # If the first window would exceed max_forecast_timestamp, just return a single window
    if current_end + delta > max_forecast_timestamp:
        return (
            [(min_timestamp, start_forecast_timestamp)],
            delta,
            None,
        )

    while current_end <= max_forecast_timestamp:
        windows.append((min_timestamp, current_end))
        current_end = current_end + delta

        # Check if we would exceed the maximum number of windows
        if len(windows) >= MAX_BACKTEST_FORECAST_WINDOWS:
            return (
                None,
                None,
                f"Stride would result in more than {MAX_BACKTEST_FORECAST_WINDOWS} windows: {stride=}. Use a smaller stride or decrease the forecast period.",
            )
    # Add final window if needed
    if windows and windows[-1][1] < max_forecast_timestamp:
        windows.append((min_timestamp, max_forecast_timestamp))

    return windows, delta, None


def _create_forecast_plot(
    history_timestamps: List[datetime],
    history_values: List[float],
    future_timestamps: List[datetime],
    forecasts: List[float],
    ground_truth: List[float | None],
    group_value: str,
) -> Tuple[bytes, str]:
    try:
        logger.debug(f"Creating plot for group: {group_value}")
        logger.debug(f"History data points: {len(history_timestamps)}")
        logger.debug(f"Forecast data points: {len(future_timestamps)}")

        fig: Figure
        fig, ax = plt.subplots(figsize=(8, 4))  # type: ignore
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()[0]

        # Convert timestamps to datetime if they aren't already
        history_timestamps = [pd.to_datetime(ts) for ts in history_timestamps]
        future_timestamps = [pd.to_datetime(ts) for ts in future_timestamps]

        ax.plot(
            history_timestamps,
            history_values,
            "b-",
            label="Historical Data",
            linewidth=2,
        )
        ax.plot(
            future_timestamps, forecasts, "r--", label="Forecasts", linewidth=2
        )

        valid_gt_indices = [
            i for i, x in enumerate(ground_truth) if x is not None
        ]
        if valid_gt_indices:
            last_valid_idx = valid_gt_indices[-1] + 1
            ax.plot(
                future_timestamps[:last_valid_idx],
                ground_truth[:last_valid_idx],
                "g-",
                label="Ground Truth",
                linewidth=2,
            )

        ax.set_title(f"Forecast Results for {group_value}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)

        safe_group_value = "".join(
            c if c.isalnum() else "_" for c in group_value
        )
        filename = f"forecast_plot_{safe_group_value}.png"
        logger.debug(f"Successfully created plot: {filename}")

        return buf.getvalue(), filename
    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        raise
    finally:
        plt.close(fig)  # Only close the current figure


async def _create_plots_async(
    result_tracker: ForecastResultTracker,
) -> List[Tuple[bytes, str]]:
    """Create plots for all results asynchronously."""
    logger.info(f"Starting to create {len(result_tracker.results)} plots")
    plot_tasks = []
    for i, result in enumerate(result_tracker.results):
        # Create a unique identifier for each plot based on group value and timestamps
        group_value = result["Group Value"]
        start_time = result["Forecast Start Timestamp"].strftime(
            "%Y%m%d_%H%M%S"
        )
        end_time = result["Forecast End Timestamp"].strftime("%Y%m%d_%H%M%S")

        # Add covariate information to group value if covariates are present
        covariate_combination_id = result.get("Covariate_Combination_ID", "")
        covariate_suffix = (
            f"_covs_{covariate_combination_id.replace('(', '_').replace(')', '').replace(', ', '-')}"
            if covariate_combination_id and covariate_combination_id != "None"
            else ""
        )

        logger.debug(
            f"Creating plot {i + 1}/{len(result_tracker.results)} for group: {group_value}"
        )

        task = asyncio.to_thread(
            _create_forecast_plot,
            history_timestamps=result["History Timestamps"],
            history_values=result["History Values"],
            future_timestamps=result["Future Timestamps"],
            forecasts=result["Forecasts"],
            ground_truth=result["Ground Truth"],
            group_value=f"{group_value}_{start_time}_to_{end_time}{covariate_suffix}",
        )
        plot_tasks.append(task)

    # Get all plot results
    plot_results = await asyncio.gather(*plot_tasks)
    logger.info(f"Successfully created all {len(plot_results)} plots")
    return plot_results


@router.get("/api/foundation_models/tasks/{user_id}/{task_id}")
async def get_task_status(
    user_id: str,
    task_id: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Get simplified task status with progress percentage and estimated time remaining.

    Args:
        user_id: ID of the user requesting the task status
        task_id: ID of the task to check
    """
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )
    task = celery_app.AsyncResult(task_id)
    info = task.info or {}

    # Check if task exists and belongs to the user
    if not task.ready() and task.state == "PENDING":
        # For pending tasks, we can't verify ownership yet
        response = {
            "state": task.state,
            "progress_percentage": 0,
        }
    else:
        # For tasks that have started or completed, verify user ownership
        task_user_id = info.get("user_id")
        if task_user_id and task_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not authorized to view this task's status",
            )

        if task.state == "PROGRESS":
            response = {
                "state": task.state,
                "progress_percentage": info.get("progress_percentage", 0),
                "status": info.get("status", None),
            }
        elif task.state == "SUCCESS":
            response = {
                "state": task.state,
                "progress_percentage": 100,
                "presigned_url": info.get("presigned_url", None),
                "message": info.get("message", None),
            }
        elif task.state == "FAILURE":
            response = {
                "state": task.state,
                "progress_percentage": 0,
                "error": str(info),
            }
        else:
            response = {
                "state": task.state,
                "progress_percentage": 0,
            }

    return response


def generate_covariate_combinations(
    available_covariates: List[str],
    max_combinations: int,
) -> List[List[str]]:
    """Generate covariate combinations for grid search, limited by max_combinations.

    Args:
        available_covariates: List of available covariate column names
        max_combinations: Maximum number of combinations to return

    Returns:
        List of covariate combinations, starting with empty list (no covariates),
        limited to max_combinations
    """

    combinations_list = [[]]  # Start with no covariates

    # Generate all combinations of different lengths
    for r in range(1, len(available_covariates) + 1):
        for combo in combinations(available_covariates, r):
            combinations_list.append(list(combo))

            # Stop if we've reached the maximum number of combinations
            if len(combinations_list) >= max_combinations:
                return combinations_list

    return combinations_list


def find_best_covariate_combination(results_df: pd.DataFrame) -> pd.DataFrame:
    """Find the best covariate combination based on RMSE performance.

    This function groups the backtest results by covariate combination and calculates
    comprehensive statistics (mean, median, std, min, max) for RMSE and NMAE metrics.
    The best performing combination is determined by the lowest mean RMSE.

    The returned DataFrame is sorted by mean RMSE in ascending order, so the best
    performing combination (lowest RMSE) appears first. Users can download and
    re-sort the results as needed for their analysis.

    Args:
        results_df: DataFrame containing backtest results with metrics including
                   'Covariate_Combination_ID' column and 'rmse'/'nmae' metric columns

    Returns:
        DataFrame with covariate combinations as index and aggregated metrics as columns.
        Sorted by rmse_mean in ascending order (best performance first).
        Columns include: rmse_mean, rmse_median, rmse_std, rmse_min, rmse_max,
                        and nmae_mean, nmae_median, nmae_std, nmae_min, nmae_max (if available)

    Raises:
        ValueError: If RMSE column is not found in the results DataFrame
    """
    if "rmse" not in results_df.columns:
        logger.warning(
            "RMSE column not found in results, cannot determine best combination"
        )
        raise ValueError(
            "RMSE column not found in results, cannot determine best combination"
        )

    # Use Covariate_Combination_ID if available, fallback to Covariates_Str for backward compatibility
    group_by_column = (
        "Covariate_Combination_ID"
        if "Covariate_Combination_ID" in results_df.columns
        else "Covariates_Str"
    )

    # Build aggregation dictionary with RMSE and NMAE (both always computed)
    agg_dict = {
        "rmse": ["mean", "median", "std", "min", "max"],
        "nmae": ["mean", "median", "std", "min", "max"],
    }

    # Group by covariate combination and calculate comprehensive statistics
    covariate_performance = (
        results_df.groupby(group_by_column)
        .agg(agg_dict)  # type: ignore
        .round(METRIC_ROUNDING_PRECISION)
    )

    # Flatten column names (e.g., ('rmse', 'mean') -> 'rmse_mean')
    covariate_performance.columns = [
        "_".join(col).strip() for col in covariate_performance.columns
    ]

    # Sort by mean RMSE (ascending order - best performance first)
    covariate_performance = covariate_performance.sort_values("rmse_mean")

    return covariate_performance


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

            values = {}
            for match in re.finditer(r"(\w+)=([+-]?\d+)", delta_str):
                key, value = match.groups()
                values[key] = int(value)
            return relativedelta(**values)
        except Exception as e:
            raise ValueError(
                f"Could not parse duration string: {delta_str}. Error: {str(e)}"
            )


async def _run_forecast_backtest_core(
    df_serialized: dict,  # from json.loads(df.to_json())
    forecast_windows: List[Tuple[str, str]],  # List of (start, end) ISO strings
    config_as_dict: Dict[str, Any],
    user_id: str,
    group_filters_serialized: Dict[str, Any],
    bucket_name: str,
    file_path_key: str,
    delta_serialized: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    covariate_grid: List[Dict[str, bool]] | None = None,
) -> Dict[str, Any]:
    """
    Core functionality for running backtest forecasting. Can be used with or without Celery.

    Args:
        df_serialized: DataFrame as JSON dict
        forecast_windows: List of (start, end) ISO strings
        config_as_dict: FoundationModelConfig as dict
        user_id: ID of the user
        group_filters_serialized: Optional group filters for the forecast
        bucket_name: Name of the S3 bucket
        file_path_key: S3 key of the input file
        delta_serialized: Optional serialized relativedelta string
        progress_callback: Optional callback function to report progress (int percentage, str status)

    Returns:
        Dict containing task status and results;
        keys include:
            {
                "state": "SUCCESS" | "FAILURE" | "PENDING" | "PROGRESS"
                "error": str | None
                "results": List[Dict[str, Any]]
            }
    """
    logger.info("Entered backtest forecasting core function")
    # Clear GPU cache before task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config: FoundationModelConfig = FoundationModelConfig(**config_as_dict)

    group_filters = GroupLabelColumnFilters(**group_filters_serialized)
    try:
        if progress_callback:
            progress_callback(0, "Starting backtest forecast")

        # Deserialize DataFrame
        df = pd.DataFrame(df_serialized)

        # Deserialize forecast_windows
        forecast_windows_dt = [
            (pd.to_datetime(start), pd.to_datetime(end))
            for start, end in forecast_windows
        ]
        total_windows = len(forecast_windows_dt)
        total_groups = 1  # Default for no group filters

        # Set up covariate grid
        if covariate_grid is None:
            # Use the original covariate columns as single combination
            covariate_combinations = [
                {col: False for col in config.covariate_columns}
            ]
            # If covariate_columns_to_leak is specified, mark those columns for leaking
            if config.covariate_columns_to_leak:
                for col in config.covariate_columns_to_leak:
                    if col in covariate_combinations[0]:
                        covariate_combinations[0][col] = True
        else:
            covariate_combinations = covariate_grid

        total_covariate_combinations = len(covariate_combinations)
        logger.info(
            f"Testing {total_covariate_combinations} covariate combinations"
        )

        # Initialize result tracker
        result_tracker = ForecastResultTracker()

        metadata_dataframes, new_leak_idxs = await prepare_metadata_dataframes(
            config.metadata_info_combined,
            SYNTHEFY_FOUNDATION_MODEL_METADATA_DATASETS_BUCKET,
            df[config.timestamp_column].tolist(),
            config.metadata_dataframes_leak_idxs,
        )
        config.metadata_dataframes_leak_idxs = new_leak_idxs

        # Process group filters if not empty
        if len(group_filters.filter) > 0:
            combinations = FilterUtils.generate_filter_combinations(
                group_filters
            )

            total_groups = len(combinations)
            total_tasks = (
                total_groups * total_windows * total_covariate_combinations
            )
            logger.info(
                f"Processing {total_groups} groups across {total_windows} forecast windows with {total_covariate_combinations} covariate combinations ({total_tasks} total tasks)"
            )

            task_idx = 0
            for group_idx, filter_dict in enumerate(combinations):
                logger.info(f"Processing filter {filter_dict}")
                df_filtered = FilterUtils.filter_and_aggregate_dataframe(
                    df,
                    GroupLabelColumnFilters(
                        filter=filter_dict,
                        aggregation_func=group_filters.aggregation_func,
                    ),
                    config.timestamp_column,
                )
                if len(df_filtered) == 0:
                    logger.warning(
                        f"No data available for filter {filter_dict} - skipping it."
                    )
                    continue

                # Process each forecast window
                for window_idx, (window_start, window_end) in enumerate(
                    forecast_windows_dt
                ):
                    # Update config with current window timestamps (do this once per window)
                    config.min_timestamp = window_start.isoformat()
                    config.forecasting_timestamp = window_end.isoformat()

                    # Process timestamps once per window (not per covariate combination)
                    try:
                        (
                            df_window,
                            ground_truth_df,
                            forecasting_timestamp,
                            future_timestamps,
                        ) = _process_timestamps(df_filtered, config)

                        if len(df_window) == 0:
                            logger.debug(
                                f"No data available in window {window_start} to {window_end}"
                            )
                            continue

                        # Process each covariate combination
                        for cov_idx, covariate_combo in enumerate(
                            covariate_combinations
                        ):
                            task_idx += 1
                            progress = PROGRESS_SETUP_PERCENTAGE + (
                                task_idx
                                * (
                                    PROGRESS_PROCESSING_PERCENTAGE
                                    - PROGRESS_SETUP_PERCENTAGE
                                )
                                / total_tasks
                            )
                            if progress_callback:
                                progress_callback(
                                    int(progress),
                                    f"Processing group {group_idx + 1}/{total_groups}, window {window_idx + 1}/{total_windows}, covariates {cov_idx + 1}/{total_covariate_combinations}",
                                )

                            # Extract covariate columns and leaking columns from dictionary
                            config.covariate_columns = list(
                                covariate_combo.keys()
                            )
                            config.covariate_columns_to_leak = [
                                col
                                for col, should_leak in covariate_combo.items()
                                if should_leak
                            ]

                            # Generate forecast
                            forecast_df = await asyncio.to_thread(
                                _generate_model_forecast,
                                df=df_window,
                                config=config,
                                user_id=user_id,
                                forecasting_timestamp=forecasting_timestamp,
                                future_timestamps=future_timestamps,
                                ground_truth_df=ground_truth_df,
                                metadata_dataframes=metadata_dataframes,
                            )

                            # Add results to tracker
                            for forecast_group in forecast_df.values:
                                result_tracker.add_forecast_result(
                                    forecast_start_timestamp=forecasting_timestamp,
                                    forecast_end_timestamp=future_timestamps[
                                        -1
                                    ],
                                    model_name=config.model_type,
                                    forecasts=forecast_group.forecasts,
                                    ground_truth=forecast_group.ground_truth,
                                    group_values=tuple(
                                        v[0]
                                        for fd in filter_dict
                                        for v in fd.values()
                                    ),
                                    group_columns=list(
                                        k
                                        for fd in filter_dict
                                        for k in fd.keys()
                                    ),
                                    history_timestamps=df_window[
                                        config.timestamp_column
                                    ].tolist(),
                                    history_values=df_window[
                                        config.timeseries_columns[0]
                                    ].tolist(),
                                    future_timestamps=future_timestamps,
                                    covariate_leak_info=covariate_combo,
                                )

                    except Exception as e:
                        logger.error(f"Error generating forecast: {str(e)}")
                        continue

        # Handle case when group_filters has empty filter list - process entire dataset with aggregation
        else:
            total_tasks = total_windows * total_covariate_combinations
            logger.info(
                f"No column filters provided (empty filter list) - processing entire dataset with aggregation ({group_filters.aggregation_func.value}) across {total_windows} forecast windows and {total_covariate_combinations} covariate combinations ({total_tasks} total tasks)"
            )

            task_idx = 0
            for window_idx, (window_start, window_end) in enumerate(
                forecast_windows_dt
            ):
                # Process each covariate combination
                for cov_idx, covariate_combo in enumerate(
                    covariate_combinations
                ):
                    task_idx += 1
                    progress = PROGRESS_SETUP_PERCENTAGE + (
                        task_idx * PROGRESS_PROCESSING_PERCENTAGE / total_tasks
                    )
                    if progress_callback:
                        progress_callback(
                            int(progress),
                            f"Processing window {window_idx + 1}/{total_windows}, covariates {cov_idx + 1}/{total_covariate_combinations}",
                        )

                    # Update config with current window timestamps and covariates
                    config.min_timestamp = window_start.isoformat()
                    config.forecasting_timestamp = window_end.isoformat()

                    # Extract covariate columns and leaking columns from dictionary
                    config.covariate_columns = list(covariate_combo.keys())
                    config.covariate_columns_to_leak = [
                        col
                        for col, should_leak in covariate_combo.items()
                        if should_leak
                    ]

                    try:
                        # Apply aggregation with empty filters (entire dataset)
                        df_window_filtered = FilterUtils.filter_and_aggregate_dataframe(
                            df,
                            GroupLabelColumnFilters(
                                filter=[],
                                aggregation_func=group_filters.aggregation_func,
                            ),
                            config.timestamp_column,
                        )
                        # Process timestamps on the aggregated data
                        (
                            df_window,
                            ground_truth_df,
                            forecasting_timestamp,
                            future_timestamps,
                        ) = _process_timestamps(df_window_filtered, config)

                        if len(df_window) == 0:
                            logger.debug(
                                f"No data available in window {window_start} to {window_end}"
                            )
                            continue
                        forecast_df = await asyncio.to_thread(
                            _generate_model_forecast,
                            df=df_window,
                            config=config,
                            user_id=user_id,
                            forecasting_timestamp=forecasting_timestamp,
                            future_timestamps=future_timestamps,
                            ground_truth_df=ground_truth_df,
                            metadata_dataframes=metadata_dataframes,
                        )
                        for forecast_group in forecast_df.values:
                            result_tracker.add_forecast_result(
                                forecast_start_timestamp=forecasting_timestamp,
                                forecast_end_timestamp=future_timestamps[-1],
                                model_name=config.model_type,
                                forecasts=forecast_group.forecasts,
                                ground_truth=forecast_group.ground_truth,
                                group_values=(),
                                group_columns=[],
                                history_timestamps=df_window[
                                    config.timestamp_column
                                ].tolist(),
                                history_values=df_window[
                                    config.timeseries_columns[0]
                                ].tolist(),
                                future_timestamps=future_timestamps,
                                covariate_leak_info=covariate_combo,
                            )
                    except Exception as e:
                        logger.error(f"Error generating forecast: {str(e)}")
                        raise

        # Get results and create plots
        if progress_callback:
            progress_callback(
                PROGRESS_SETUP_PERCENTAGE + PROGRESS_PROCESSING_PERCENTAGE,
                "Generating plots and preparing results",
            )

        results_df = result_tracker.get_results_dataframe()
        plot_results = await _create_plots_async(result_tracker)

        # Find best covariate combination if we have multiple combinations
        best_model_analysis_df = None
        if len(covariate_combinations) > 1:
            # TODO: always have rmse, put None if ground truth is not available
            try:
                best_model_analysis_df = find_best_covariate_combination(
                    results_df
                )
                # Get the best combination (first row since sorted by RMSE)
                best_combo = (
                    best_model_analysis_df.index[0]
                    if len(best_model_analysis_df) > 0
                    else "N/A"
                )
                logger.info(f"Best covariate combination: {best_combo}")
            except ValueError as e:
                logger.warning(
                    f"Could not determine best covariate combination: {str(e)}"
                )
                logger.info(
                    "Continuing without best model analysis due to missing RMSE data"
                )
                best_model_analysis_df = None
        else:
            logger.info(
                "Single covariate combination used, no comparison needed"
            )

        # Create a new session for this operation
        session = aioboto3.Session()
        try:
            # Save results to temporary files
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False
            ) as results_file:
                results_df.to_csv(results_file.name, index=False)
                timestamp_formatted = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Create zip file
                with tempfile.NamedTemporaryFile(
                    suffix=".zip", delete=False
                ) as zip_file:
                    with zipfile.ZipFile(zip_file.name, "w") as zipf:
                        # Add CSV
                        zipf.write(
                            results_file.name,
                            os.path.basename(results_file.name),
                        )
                        # Add plots
                        for plot_bytes, filename in plot_results:
                            zipf.writestr(filename, plot_bytes)

                        # Add best model analysis if available
                        if (
                            best_model_analysis_df is not None
                            and not best_model_analysis_df.empty
                        ):
                            # Save DataFrame as CSV
                            analysis_csv = best_model_analysis_df.to_csv(
                                index=True
                            )
                            zipf.writestr(
                                "best_model_analysis.csv", analysis_csv
                            )

                    # Upload to S3 using the session's client
                    async with session.client("s3") as s3_client:  # type: ignore
                        try:
                            zip_key = f"{file_path_key}{BACKTEST_RESULTS_SUFFIX}{timestamp_formatted}.zip"
                            await async_upload_file_to_s3(
                                async_s3_client=s3_client,
                                local_file=zip_file.name,
                                bucket=bucket_name,
                                s3_key=zip_key,
                            )

                            # Generate presigned URL
                            zip_url = await acreate_presigned_url(
                                session, bucket=bucket_name, s3_key=zip_key
                            )

                            # Clean up temporary files
                            os.remove(results_file.name)
                            os.remove(zip_file.name)

                            return {
                                "status": "SUCCESS",
                                "progress_percentage": 100,
                                "message": "Backtest completed successfully",
                                "presigned_url": zip_url,
                                "user_id": user_id,
                            }
                        except Exception as e:
                            logger.error(
                                f"Error uploading results to S3: {str(e)}"
                            )
                            raise
        except Exception as e:
            logger.error(f"Error in backtest results processing: {str(e)}")
            raise
        finally:
            # Always clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        return {
            "state": "FAILED",
            "progress_percentage": 0,
            "message": f"Backtest forecasting failed: {str(e)}",
            "presigned_url": None,
            "user_id": user_id,
        }
    finally:
        # Always clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@router.post("/api/v2/foundation_models/forecast/stream")
async def forecastv2_stream(
    request: ForecastV2Request,
    _: Optional[str] = Depends(get_user_id_from_token_or_api_key),
) -> ForecastV2Response:
    """
    Forecast using fm-evals.
    """
    try:
        forecast_adapter = ForecastV2APIAdapter(model_name=request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        response = forecast_adapter.predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response


@router.post(
    "/api/foundation_models/forecast/stream",
    response_model=FoundationModelForecastStreamResponse,
)
async def forecast_stream(
    request: FoundationModelForecastStreamRequest,
    _: Optional[str] = Depends(get_user_id_from_token_or_api_key),
) -> FoundationModelForecastStreamResponse:
    """
    Stream forecast results for a given forecast window.

    Args:
        request: FoundationModelForecastStreamRequest

    Returns:
        FoundationModelForecastStreamResponse: Streamed forecast results


    TODO still:
        use static_metadata_dataframes
        add leakage support (will be done in predict())
        better naming for metadata dataframes
    """

    # Warning for SHAP analysis
    warnings = None
    if request.do_shap_analysis:
        logger.warning(SHAP_ANALYSIS_WARNING_MSG)
        warnings = [SHAP_ANALYSIS_WARNING_MSG]

    model = FoundationModelService.get_model(model_type=request.model_type)

    historical_df = request.historical_df
    forecasting_timestamp = historical_df["timestamp"].iloc[-1]

    # add the future timestamps, future_timeseries_data for leaking if requested.
    try:
        future_df = request.future_df
        if len(future_df) > 0:
            historical_df = pd.concat(
                [historical_df, future_df], ignore_index=True
            )

    except Exception as e:
        logger.error(f"Error in adding future data to DataFrame: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    future_timestamps = list(pd.to_datetime(request.forecast_timestamps).values)

    metadata_dataframes, new_leak_idxs = await prepare_metadata_dataframes(
        request.synthefy_metadata_info_combined,
        SYNTHEFY_FOUNDATION_MODEL_METADATA_DATASETS_BUCKET,
        historical_df["timestamp"].tolist(),
        request.synthefy_metadata_leak_idxs,
    )
    request.synthefy_metadata_leak_idxs = new_leak_idxs
    try:
        # get leak parameters
        covariate_columns_to_leak = (
            list(request.future_timeseries_data.keys())
            if request.future_timeseries_data
            else None
        )

        if (
            request.synthefy_metadata_info_combined
            and request.synthefy_metadata_leak_idxs
        ):
            metadata_dataframes_leak_idxs = request.synthefy_metadata_leak_idxs
        else:
            metadata_dataframes_leak_idxs = None

        # Generate forecasts for the entire dataset
        logger.debug("Starting prediction")
        ret = await asyncio.to_thread(
            model.predict,
            target_df=historical_df,
            covariate_columns=request.covariates or [],
            target_columns=request.targets,
            forecasting_timestamp=forecasting_timestamp,
            metadata_dataframes=metadata_dataframes,
            future_time_stamps=future_timestamps,
            ground_truth_df=None,
            quantiles=request.quantiles or [0.1, 0.9],
            covariate_columns_to_leak=covariate_columns_to_leak,
            metadata_dataframes_leak_idxs=metadata_dataframes_leak_idxs,
            do_llm_explanation=request.do_llm_explanation,
            do_shap_analysis=request.do_shap_analysis,
            timestamp_column="timestamp",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    forecasts = {}
    forecast_quantiles = {}
    forecast_shap_analysis = {}
    for forecast_group in ret.values:
        forecasts[forecast_group.target_column] = forecast_group.forecasts
        if request.quantiles is None:
            quantiles = [0.1, 0.9]
        else:
            quantiles = request.quantiles

        forecast_quantiles[
            f"{forecast_group.target_column}_p{int(quantiles[0] * 100)}"
        ] = [
            forecast_group.confidence_intervals[i].lower
            for i in range(len(forecast_group.confidence_intervals))
        ]
        forecast_quantiles[
            f"{forecast_group.target_column}_p{int(quantiles[1] * 100)}"
        ] = [
            forecast_group.confidence_intervals[i].upper
            for i in range(len(forecast_group.confidence_intervals))
        ]
        if forecast_group.shap_analysis:
            forecast_shap_analysis[forecast_group.target_column] = (
                forecast_group.shap_analysis
            )

    return FoundationModelForecastStreamResponse(
        forecast_timestamps=ret.timestamps,
        forecast=forecasts,
        forecast_quantiles=forecast_quantiles,
        shap_analysis=forecast_shap_analysis,
        warnings=warnings,
    )


async def stream_first_n_json_entries_from_s3(
    s3_client, bucket, key, n, regex_pattern=None
):
    """
    Stream the first n entries from a large JSON file in S3.
    Yields HaverDatasetMatch objects for each entry that matches the regex pattern if provided.
    Handles both top-level arrays and dicts.
    Reads the entire file into memory before parsing.
    """
    response = await s3_client.get_object(Bucket=bucket, Key=key)
    content = await response["Body"].read()
    count = 0

    pattern = (
        re.compile(regex_pattern, re.IGNORECASE) if regex_pattern else None
    )

    try:
        for obj_json in ijson.items(io.BytesIO(content), "item"):
            if pattern and "description" in obj_json:
                if not pattern.search(obj_json["description"]):
                    continue
            haver_data = {
                "data_source": "haver",
                "description": obj_json.get("description", ""),
                "database_name": obj_json.get("databaseName"),
                "name": obj_json.get("name"),
                "start_date": obj_json.get("startDate", 0),
                "db_path_info": key,
                "file_name": None,
            }
            yield HaverDatasetMatch(
                access_info=HaverMetadataAccessInfo(**haver_data)
            )
            count += 1
            if count >= n:
                return
    except Exception:
        for k, v in ijson.kvitems(io.BytesIO(content), ""):
            if pattern and "description" in v:
                if not pattern.search(v["description"]):
                    continue
            haver_data = {
                "data_source": "haver",
                "description": v.get("description", ""),
                "database_name": v.get("databaseName"),
                "name": v.get("name"),
                "start_date": v.get("startDate", 0),
                "db_path_info": key,
                "file_name": None,  # Explicitly set to None since we're using database_name + name
            }
            yield HaverDatasetMatch(
                access_info=HaverMetadataAccessInfo(**haver_data)
            )
            count += 1
            if count >= n:
                return


@router.post(
    "/api/search-synthefy-database-metadata/",
    response_class=StreamingResponse,
)
async def search_synthefy_database_metadata(
    request: SynthefyDatabaseMetadataSearchRequest = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Stream up to 100 dataset metadata matches from the specified data source in the S3 bucket.
    For each .json file, only the first N entries are streamed using ijson for memory efficiency.
    Now streams as NDJSON (newline-delimited JSON).
    Supports regex pattern matching on the description field.
    """
    settings = get_settings(FoundationModelApiSettings)
    metadata_bucket = settings.metadata_datasets_bucket
    if not metadata_bucket:
        raise HTTPException(
            status_code=500,
            detail="Metadata datasets bucket not configured",
        )

    prefix = f"univariate/{request.data_source}/synthefy-flat"
    session = aioboto3.Session()
    N = 100
    GLOBAL_LIMIT = 500

    async def dataset_match_generator():
        logger.info(
            f"Starting streaming search for prefix: {prefix} (limit {N} entries per file, NDJSON mode, {GLOBAL_LIMIT} total)"
        )
        if request.regex:
            logger.info(f"Using regex pattern: {request.regex}")

        total_count = 0
        async with session.client("s3") as s3_client:  # type: ignore
            paginator = s3_client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=metadata_bucket, Prefix=prefix
            ):
                for obj in page.get("Contents", []):
                    if total_count >= GLOBAL_LIMIT:
                        logger.info(
                            f"Global limit of {GLOBAL_LIMIT} reached, stopping stream."
                        )
                        return
                    key = obj["Key"]
                    if not key.endswith(".json"):
                        continue
                    logger.info(f"Processing file: {key}")
                    entry_count = 0
                    async for match in stream_first_n_json_entries_from_s3(
                        s3_client,
                        metadata_bucket,
                        key,
                        min(N, GLOBAL_LIMIT - total_count),
                        request.regex,
                    ):
                        yield json.dumps(match.model_dump()).encode() + b"\n"
                        entry_count += 1
                        total_count += 1
                        if total_count >= GLOBAL_LIMIT:
                            logger.info(
                                f"Global limit of {GLOBAL_LIMIT} reached, stopping stream."
                            )
                            return
                    logger.info(f"Streamed {entry_count} entries from {key}")

    return StreamingResponse(
        dataset_match_generator(), media_type="application/x-ndjson"
    )


@router.post(
    "/api/search-directory/",
    response_model=SynthefyDatabaseDirectorySearchResponse,
)
async def search_directory(
    request: SynthefyDatabaseDirectorySearchRequest = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> SynthefyDatabaseDirectorySearchResponse:
    """
    List directories and datasets at the specified directory level in the metadata S3 bucket.
    """
    settings = get_settings(FoundationModelApiSettings)
    metadata_bucket = settings.metadata_datasets_bucket
    if not metadata_bucket:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metadata datasets not configured",
        )

    prefix = "univariate/"
    prefix = (
        os.path.join(prefix, request.directory_to_search)
        if request.directory_to_search
        else prefix
    )
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    try:
        session = aioboto3.Session()
        async with session.client("s3") as s3_client:  # type: ignore
            response = await s3_client.list_objects_v2(
                Bucket=metadata_bucket, Prefix=prefix, Delimiter="/"
            )

            directories = []
            for cp in response.get("CommonPrefixes", []):
                folder_path = cp.get("Prefix", "")
                folder_name = folder_path[len(prefix) :].strip("/")
                if folder_name:
                    directories.append(folder_name)

            datasets = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if "/" in key[len(prefix) :]:
                    continue
                if key.endswith(".parquet") or key.endswith(".json"):
                    file_name = key.split("/")[-1]
                    datasets.append(
                        HaverDatasetMatch(
                            access_info=HaverMetadataAccessInfo(
                                file_name=file_name,
                                data_source="haver",
                                description="",
                                start_date=0,
                                database_name=None,
                                name=None,
                            ),
                            db_path_info=key,
                        )
                    )

            return SynthefyDatabaseDirectorySearchResponse(
                directories=directories,
                datasets=datasets,
            )
    except ClientError as e:
        logger.error(f"S3 list_objects_v2 operation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal s3 client call to list objects failed",
        )
    except Exception as e:
        logger.error(f"Error listing S3 directory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing S3 directory: {str(e)}",
        )


@router.post(
    "/api/foundation_models/visualize_metadata",
    response_model=MetadataVisualizationResponse,
)
async def visualize_metadata(
    request: MetadataVisualizationRequest = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> MetadataVisualizationResponse:
    """
    Process a metadata dataset for visualization.

    1. Downloads the haver dataset
    2. Uploads it to S3 in both parquet and JSON formats
    3. Returns presigned URLs for visualization
    """
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )
    logger.info("Starting metadata visualization request")
    try:
        # Get settings
        settings = get_settings(FoundationModelApiSettings)
        bucket_name = settings.bucket_name

        if not bucket_name:
            logger.error("S3 bucket name not configured in settings")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="S3 bucket name not configured",
            )

        # Process the metadata info to get the DataFrame
        if isinstance(request.metadata_access_info, HaverMetadataAccessInfo):
            metadata_df_list = await process_haver_metadata_access_info(
                request.metadata_access_info, settings.metadata_datasets_bucket
            )

        elif isinstance(
            request.metadata_access_info, WeatherMetadataAccessInfo
        ):
            # Use the unified function that handles both cases (with and without target_timestamps)
            logger.info(
                f"Processing WeatherStack data for location {request.metadata_access_info.name}"
                + (
                    f" with {len(request.target_timestamps)} exact timestamps"
                    if request.target_timestamps
                    else " using date range"
                )
            )
            metadata_df_list = await process_weatherstack_metadata_access_info(
                request.metadata_access_info,
                request.target_timestamps,
            )

        elif isinstance(
            request.metadata_access_info, PredictHQMetadataAccessInfo
        ):
            metadata_df_list = await process_predicthq_metadata_access_info(
                request.metadata_access_info
            )

        else:
            raise ValueError("Invalid metadata info type")

        if metadata_df_list is None or len(metadata_df_list) == 0:
            return MetadataVisualizationResponse(
                status=StatusCode.bad_request,
                message="Failed to process metadata dataset",
                datasets=[],
            )

        # Process each metadata dataframe in the list
        processed_datasets = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, metadata_df in enumerate(metadata_df_list):
            if metadata_df is None or not hasattr(metadata_df, "df"):
                logger.warning(
                    f"Skipping metadata dataframe {idx}: invalid or empty"
                )
                continue

            # Create temporary files to store the dataset
            with (
                tempfile.NamedTemporaryFile(
                    suffix=".parquet", delete=False
                ) as parquet_file,
                tempfile.NamedTemporaryFile(
                    suffix=".json", delete=False
                ) as json_file,
            ):
                try:
                    # Generate a safe description for the filenames
                    description = getattr(
                        metadata_df, "description", f"dataset_{idx}"
                    )
                    safe_description = "".join(
                        c if c.isalnum() else "_" for c in description[:50]
                    )

                    # Save DataFrame to parquet
                    metadata_df.df.to_parquet(parquet_file.name, index=False)
                    parquet_s3_key = f"metadata_visualization/{timestamp}_{idx}_{safe_description}.parquet"

                    # Save DataFrame to JSON in the format requested
                    # Convert dataframe to the expected JSON format with date and value fields
                    json_records = metadata_df.df.to_dict(orient="records")
                    with open(json_file.name, "w") as f:
                        json.dump(json_records, f)
                    json_s3_key = f"metadata_visualization/{timestamp}_{idx}_{safe_description}.json"

                    # Upload both files to S3
                    async with aioboto3_session.client("s3") as s3_client:  # type: ignore
                        # Upload parquet file
                        with open(parquet_file.name, "rb") as f:
                            await s3_client.upload_fileobj(
                                f, bucket_name, parquet_s3_key
                            )

                        # Upload JSON file
                        with open(json_file.name, "rb") as f:
                            await s3_client.upload_fileobj(
                                f, bucket_name, json_s3_key
                            )

                    # Generate presigned URLs
                    parquet_download_url = await acreate_presigned_url(
                        aioboto3_session,
                        bucket=bucket_name,
                        s3_key=parquet_s3_key,
                    )

                    json_download_url = await acreate_presigned_url(
                        aioboto3_session,
                        bucket=bucket_name,
                        s3_key=json_s3_key,
                    )

                    if not parquet_download_url and not json_download_url:
                        logger.error(
                            f"Failed to generate presigned URLs for dataset {idx}"
                        )
                        continue

                    display_name = (
                        metadata_df.description
                        if metadata_df.description
                        else f"Dataset {idx + 1}"
                    )

                    # Generate a unique ID for the dataset
                    dataset_id = f"{timestamp}_{idx}_{safe_description}"

                    # Add to processed datasets list
                    processed_datasets.append(
                        MetadataVisualizationItem(
                            id=dataset_id,
                            display_name=display_name,
                            download_url=parquet_download_url,
                            json_download_url=json_download_url,
                        )
                    )

                except Exception as e:
                    logger.error(f"Error processing dataset {idx}: {str(e)}")
                    continue
                finally:
                    # Clean up temporary files
                    if os.path.exists(parquet_file.name):
                        os.remove(parquet_file.name)
                    if os.path.exists(json_file.name):
                        os.remove(json_file.name)

        if len(processed_datasets) == 0:
            return MetadataVisualizationResponse(
                status=StatusCode.internal_server_error,
                message="Failed to process any datasets",
                datasets=[],
            )

        return MetadataVisualizationResponse(
            status=StatusCode.ok,
            message=f"Successfully processed {len(processed_datasets)} dataset(s)",
            datasets=processed_datasets,
        )

    except Exception as e:
        logger.exception(f"Error in visualize_metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing metadata: {str(e)}",
        )


@router.post(
    "/api/foundation_models/chat/stream", response_class=StreamingResponse
)
async def foundation_model_chat_stream(
    request: FoundationModelChatRequest = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Process a chat request from the user and stream the response.
    Returns a streaming response with JSON data.
    """
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )
    try:
        logger.info("Starting foundation model chat request")
        agent_result = run_agent_streaming(request)
        return StreamingResponse(
            agent_result, media_type="application/x-ndjson"
        )
    except Exception as e:
        return StreamingResponse(
            [json.dumps({"error": str(e)}) + "\n"],
            media_type="application/x-ndjson",
        )


_location_cache_manager = LocationCacheManager()


@router.post(
    "/api/locations_search/",
    response_class=StreamingResponse,
)
async def search_locations(
    request: LocationSearchRequest = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
):
    """
    Stream location search results from the cities database.
    The search is case-insensitive and matches partial names.
    Results are streamed as they are found to improve response time.
    """
    # Validate request early
    if not request or not request.search:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search term is required and cannot be empty",
        )

    logger.info(f"Received search request with term: {request.search}")
    bucket_name = SYNTHEFY_FOUNDATION_MODEL_METADATA_DATASETS_BUCKET
    file_key = "city_names/cities_autocomplete.json"

    try:
        async with aioboto3_session.client("s3") as s3_client:  # type: ignore
            # First check if file exists
            try:
                await s3_client.head_object(Bucket=bucket_name, Key=file_key)
            except ClientError as e:
                if (
                    getattr(e, "response", {}).get("Error", {}).get("Code")
                    == "404"
                ):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Cities data file not found: {file_key}",
                    )
                raise

            # Load cache if not already loaded
            if not _location_cache_manager.get("locations"):
                await load_locations_into_cache(
                    s3_client, bucket_name, file_key, _location_cache_manager
                )

            # Use the extracted search function
            return StreamingResponse(
                search_cached_locations(
                    search_term=request.search,
                    cache_manager=_location_cache_manager,
                    limit=100,
                ),
                media_type="application/x-ndjson",
                headers={"X-Content-Type-Options": "nosniff"},
            )

    except Exception as e:
        logger.error(f"Error in location search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching locations: {str(e)}",
        )


class WeatherLocation(BaseModel):
    """Simple location configuration for weather data."""

    name: str
    latitude: float  # Required - get from location search
    longitude: float  # Required - get from location search


class WeatherConfig(BaseModel):
    """Simplified weather configuration for easy customer use."""

    name: str
    location: WeatherLocation
    weather_parameters: List[str] = [
        "temperature",
        "humidity",
        "precip",
        "wind_speed",
    ]  # Simple list of parameters
    frequency: str = "day"  # day, week, month
    units: Literal["m", "s", "f"] = "m"  # m=metric, s=scientific, f=fahrenheit
    start_time: str  # Simplified from min_timestamp
    end_time: str  # Simplified from forecast_timestamp


class HaverConfig(BaseModel):
    """Simplified Haver configuration for easy customer use."""

    database_name: str = Field(
        ..., description="Name of the Haver database (e.g., 'USECON')"
    )
    name: str = Field(..., description="Name of the Haver series (e.g., 'GDP')")
    description: Optional[str] = Field(
        default=None, description="Optional description of the series"
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Optional start time to filter data (ISO format)",
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Optional end time to filter data (ISO format)",
    )


def _convert_weather_config_to_internal_format(
    weather_config: WeatherConfig,
) -> WeatherMetadataAccessInfo:
    """
    Convert simplified WeatherConfig to complex WeatherMetadataAccessInfo format.

    This function handles the conversion between the customer-facing simplified format
    and the internal complex format used by the weather processing functions.
    """
    # Convert simple location to complex WeatherLocation with all optional fields
    location_data = WeatherStackLocation(
        name=weather_config.location.name,
        latitude=weather_config.location.latitude,
        longitude=weather_config.location.longitude,
        country_code=None,  # Set to None since not provided in simplified format
        admin1_code=None,  # Set to None since not provided in simplified format
        population=None,  # Set to None since not provided in simplified format
    )

    # Convert simple list of parameters to complex WeatherParameters with boolean flags
    weather_params = WeatherParameters()
    # Reset all parameters to False first
    for param in weather_params.__fields__:
        setattr(weather_params, param, False)

    # Set requested parameters to True
    for param in weather_config.weather_parameters:
        if hasattr(weather_params, param):
            setattr(weather_params, param, True)

    # Create TimePeriod from simplified start_time and end_time
    time_period = TimePeriod(
        min_timestamp=weather_config.start_time,
        forecast_timestamp=weather_config.end_time,
    )

    # Create the complex WeatherMetadataAccessInfo
    return WeatherMetadataAccessInfo(
        data_source="weather",  # Always set to "weather" in backend
        name=weather_config.name,
        description=f"Weather data for {weather_config.location.name}",
        location_data=location_data,
        weather_parameters=weather_params,
        frequency=weather_config.frequency,
        units=weather_config.units,
        time_period=time_period,
        aggregate_intervals=False,  # Default value, can be made configurable later
    )


def _convert_haver_config_to_internal_format(
    haver_config: HaverConfig,
) -> HaverMetadataAccessInfo:
    """
    Convert simplified HaverConfig to complex HaverMetadataAccessInfo format.

    This function handles the conversion between the customer-facing simplified format
    and the internal complex format used by the Haver processing functions.
    """
    return HaverMetadataAccessInfo(
        data_source="haver",
        database_name=haver_config.database_name,
        name=haver_config.name,
        description=haver_config.description
        or f"{haver_config.name}@{haver_config.database_name}",
        start_date=0,  # Will be handled by filtering logic later
        db_path_info=None,  # Not needed for direct API access
        file_name=None,  # Not using file-based access
    )


class MetadataEnrichmentRequest(BaseModel):
    """Request model for enriching data with metadata."""

    # Metadata configuration (one of these must be provided)
    weather_metadata_info: Optional[WeatherConfig] = None
    haver_metadata_info: Optional[HaverConfig] = None

    # Optional user DataFrame to enrich (if not provided, just returns metadata)
    user_dataframe_json: Optional[Any] = Field(
        default=None,
        description="User's DataFrame in JSON format (from df.to_dict(orient='records'))",
    )
    user_timestamp_column: Optional[str] = Field(
        default=None, description="Name of timestamp column in user DataFrame"
    )

    @model_validator(mode="after")
    @classmethod
    def validate_metadata_info(cls, data):
        # Exactly one metadata info must be provided
        metadata_configs = [
            data.weather_metadata_info,
            data.haver_metadata_info,
        ]
        provided_configs = [
            config for config in metadata_configs if config is not None
        ]

        if len(provided_configs) == 0:
            raise ValueError(
                "At least one metadata configuration must be provided"
            )
        elif len(provided_configs) > 1:
            raise ValueError(
                "Only one metadata configuration can be provided at a time"
            )

        # If user DataFrame is provided, timestamp column is required
        if data.user_dataframe_json and not data.user_timestamp_column:
            raise ValueError(
                "user_timestamp_column is required when user_dataframe_json is provided"
            )

        return data


class MetadataEnrichmentResponse(BaseModel):
    """Response model for metadata enrichment."""

    status: StatusCode
    message: str

    # The result DataFrame (either just metadata or enriched user data)
    result_dataframe_json: Optional[Any] = Field(
        default=None,
        description="Result DataFrame in JSON format (from df.to_dict(orient='records'))",
    )

    # Metadata information
    metadata_columns: List[str] = Field(
        default_factory=list,
        description="List of column names that were added as metadata",
    )
    metadata_description: Optional[str] = Field(
        default=None, description="Description of the metadata source"
    )

    # Enrichment statistics
    original_row_count: Optional[int] = None
    final_row_count: Optional[int] = None
    common_timestamps_count: Optional[int] = None


async def _process_user_dataframe(
    request: MetadataEnrichmentRequest,
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[int]]:
    """Process user DataFrame and extract timestamps."""
    if not request.user_dataframe_json:
        return None, None, None

    user_df = pd.DataFrame(request.user_dataframe_json)
    original_row_count = len(user_df)

    # Validate timestamp column exists
    if request.user_timestamp_column not in user_df.columns:
        raise ValueError(
            f"Timestamp column '{request.user_timestamp_column}' not found in user DataFrame. Available columns: {list(user_df.columns)}"
        )

    # Parse timestamps
    user_df[request.user_timestamp_column] = pd.to_datetime(
        user_df[request.user_timestamp_column]
    )
    user_timestamps = (
        user_df[request.user_timestamp_column]
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        .tolist()
    )

    logger.info(
        f"User DataFrame has {len(user_df)} rows with timestamp range: {user_df[request.user_timestamp_column].min()} to {user_df[request.user_timestamp_column].max()}"
    )

    return user_df, user_timestamps, original_row_count


async def _process_weather_metadata(
    weather_config: WeatherConfig, user_timestamps: Optional[List[str]]
) -> Optional[List[Any]]:
    """Process weather metadata configuration."""
    logger.info("Processing weather metadata")

    # Convert simplified WeatherConfig to internal format
    internal_weather_config = _convert_weather_config_to_internal_format(
        weather_config
    )

    if user_timestamps is not None:
        logger.info("Using user DataFrame timestamp range for weather data")
        return await process_weatherstack_metadata_access_info(
            internal_weather_config, user_timestamps
        )

    # Create proper timestamps for the range based on the frequency
    start_time = pd.to_datetime(weather_config.start_time)
    end_time = pd.to_datetime(weather_config.end_time)

    # Map weather frequency to pandas frequency code
    freq_mapping = {
        "minute": "T",
        "hour": "H",
        "day": "D",
        "week": "W",
        "month": "M",
        "year": "Y",
    }
    pandas_freq = freq_mapping.get(weather_config.frequency.lower())

    if not pandas_freq:
        raise ValueError(
            f"Invalid frequency: {weather_config.frequency}. Valid frequencies are: {list(freq_mapping.keys())}"
        )

    proper_timestamps = (
        pd.date_range(start=start_time, end=end_time, freq=pandas_freq)
        .strftime("%Y-%m-%dT%H:%M:%S")
        .tolist()
    )

    return await process_weatherstack_metadata_access_info(
        internal_weather_config, proper_timestamps
    )


async def _process_haver_metadata(
    haver_config: HaverConfig, user_timestamps: Optional[List[str]]
) -> Optional[List[Any]]:
    """Process Haver metadata configuration."""
    logger.info("Processing Haver metadata")

    # Convert simplified HaverConfig to internal format
    internal_haver_config = _convert_haver_config_to_internal_format(
        haver_config
    )

    # Get settings for bucket name
    settings = get_settings(FoundationModelApiSettings)
    metadata_bucket = settings.metadata_datasets_bucket

    # Fetch Haver data
    metadata_dataframes = await process_haver_metadata_access_info(
        internal_haver_config, metadata_bucket
    )

    if not metadata_dataframes:
        return metadata_dataframes

    # Apply time filtering if specified
    if haver_config.start_time or haver_config.end_time:
        for i, metadata_df_obj in enumerate(metadata_dataframes):
            df = metadata_df_obj.df.copy()

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

                if haver_config.start_time:
                    start_filter = pd.to_datetime(haver_config.start_time)
                    df = df[df["date"] >= start_filter]

                if haver_config.end_time:
                    end_filter = pd.to_datetime(haver_config.end_time)
                    df = df[df["date"] <= end_filter]

                metadata_dataframes[i].df = df

    # Filter to user's timestamp range if provided
    if user_timestamps is not None:
        logger.info("Filtering Haver data to user DataFrame timestamp range")
        user_start = pd.to_datetime(user_timestamps).min()
        user_end = pd.to_datetime(user_timestamps).max()

        for i, metadata_df_obj in enumerate(metadata_dataframes):
            df = metadata_df_obj.df.copy()

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df[(df["date"] >= user_start) & (df["date"] <= user_end)]
                metadata_dataframes[i].df = df

    return metadata_dataframes


def _combine_metadata_dataframes(
    metadata_dataframes: List[Any],
) -> Tuple[Optional[pd.DataFrame], List[str], str]:
    """Combine all metadata DataFrames into a single DataFrame."""
    combined_metadata_df = None
    metadata_columns = []
    metadata_description = ""

    for i, metadata_df_obj in enumerate(metadata_dataframes):
        metadata_df = metadata_df_obj.df.copy()
        description = metadata_df_obj.description or f"Metadata {i}"

        # Build description
        if i == 0:
            metadata_description = description
        else:
            metadata_description += f"; {description}"

        # Rename timestamp column to standard name for merging
        timestamp_col = metadata_df_obj.timestamp_key
        if (
            timestamp_col in metadata_df.columns
            and timestamp_col != "timestamp"
        ):
            metadata_df = metadata_df.rename(
                columns={timestamp_col: "timestamp"}
            )

        # Get value columns (exclude timestamp) and rename to avoid conflicts
        value_columns = [
            col for col in metadata_df.columns if col != "timestamp"
        ]

        # Rename value columns to include the parameter name to avoid conflicts
        column_renames = {}
        for col in value_columns:
            if col == "value":
                # Extract parameter name from metadata description/title
                param_name = getattr(
                    metadata_df_obj, "param_name", f"param_{i}"
                )
                if (
                    hasattr(metadata_df_obj, "metadata_json")
                    and metadata_df_obj.metadata_json
                ):
                    columns_info = metadata_df_obj.metadata_json.get(
                        "columns", []
                    )
                    if columns_info and len(columns_info) > 0:
                        param_name = columns_info[0].get("id", f"param_{i}")
                new_col_name = f"{param_name}"
                column_renames[col] = new_col_name
                metadata_columns.append(new_col_name)
            else:
                # For non-value columns, add suffix to avoid conflicts
                new_col_name = f"{col}_{i}" if col in metadata_columns else col
                column_renames[col] = new_col_name
                metadata_columns.append(new_col_name)

        if column_renames:
            metadata_df = metadata_df.rename(columns=column_renames)

        if combined_metadata_df is None:
            combined_metadata_df = metadata_df
        else:
            # Merge on timestamp - now safe from column conflicts
            combined_metadata_df = combined_metadata_df.merge(
                metadata_df,
                on="timestamp",
                how="outer",
                suffixes=("", "_duplicate"),
            )

            # Remove any columns that got duplicate suffixes
            duplicate_cols = [
                col
                for col in combined_metadata_df.columns
                if col.endswith("_duplicate")
            ]
            if duplicate_cols:
                combined_metadata_df = combined_metadata_df.drop(
                    columns=duplicate_cols
                )

    return combined_metadata_df, metadata_columns, metadata_description


def _merge_user_and_metadata_dataframes(
    user_df: pd.DataFrame,
    combined_metadata_df: pd.DataFrame,
    user_timestamp_column: str,
) -> Tuple[pd.DataFrame, int]:
    """Merge user DataFrame with metadata DataFrame using merge_asof for proper time series alignment."""
    logger.info("Enriching user DataFrame with metadata using merge_asof")

    # Prepare user DataFrame for merging
    user_df_for_merge = user_df.copy()
    user_df_for_merge["timestamp"] = pd.to_datetime(
        user_df_for_merge[user_timestamp_column]
    )

    # Parse metadata timestamps for merging
    combined_metadata_df = combined_metadata_df.copy()
    combined_metadata_df["timestamp"] = pd.to_datetime(
        combined_metadata_df["timestamp"]
    )

    # Handle timezone normalization - make both timezone-naive to ensure compatibility
    # This preserves the logical date/time while removing timezone information
    user_has_tz = user_df_for_merge["timestamp"].dt.tz is not None
    metadata_has_tz = combined_metadata_df["timestamp"].dt.tz is not None

    if user_has_tz:
        user_df_for_merge["timestamp"] = user_df_for_merge[
            "timestamp"
        ].dt.tz_localize(None)
    if metadata_has_tz:
        combined_metadata_df["timestamp"] = combined_metadata_df[
            "timestamp"
        ].dt.tz_localize(None)

    # Sort both DataFrames by timestamp (required for merge_asof)
    user_df_for_merge = user_df_for_merge.sort_values("timestamp")
    combined_metadata_df = combined_metadata_df.sort_values("timestamp")

    # Find common timestamps before merging for reporting
    user_timestamps_set = set(
        user_df_for_merge["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    metadata_timestamps_set = set(
        combined_metadata_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    common_timestamps = user_timestamps_set.intersection(
        metadata_timestamps_set
    )
    common_timestamps_count = len(common_timestamps)

    logger.info(
        f"Found {common_timestamps_count} exact timestamp matches between user data and metadata"
    )
    logger.info(
        f"User data has {len(user_timestamps_set)} timestamps, metadata has {len(metadata_timestamps_set)} timestamps"
    )

    # Use merge_asof for proper time series alignment
    # direction="backward" means: for each user timestamp, find the most recent metadata timestamp that is <= user timestamp
    result_df = pd.merge_asof(
        user_df_for_merge,
        combined_metadata_df,
        on="timestamp",
        direction="backward",  # Use the most recent metadata timestamp that's <= user timestamp
        # tolerance can be added here if needed: tolerance=pd.Timedelta(days=1)
    )

    # Sort by timestamp to maintain chronological order
    result_df = result_df.sort_values("timestamp").reset_index(drop=True)

    # Update the original timestamp column but keep the timestamp column for tests/compatibility
    if user_timestamp_column != "timestamp":
        result_df[user_timestamp_column] = result_df["timestamp"]

    logger.info(f"Enriched DataFrame has {len(result_df)} rows")

    return result_df, common_timestamps_count


def _generate_response_message(
    user_df: Optional[pd.DataFrame],
    original_row_count: Optional[int],
    final_row_count: int,
    common_timestamps_count: Optional[int],
    metadata_count: int,
    metadata_columns: List[str],
) -> str:
    """Generate descriptive message based on the results."""
    if user_df is not None:
        if common_timestamps_count == 0:
            return f"Successfully merged user data ({original_row_count} rows) with metadata ({metadata_count} timestamps) using outer join. No exact timestamp matches - result has {final_row_count} rows with NaN values where data is missing from either source."
        else:
            return f"Successfully merged user data ({original_row_count} rows) with metadata ({metadata_count} timestamps) using outer join. Found {common_timestamps_count} exact matches - result has {final_row_count} rows."
    else:
        return f"Successfully fetched metadata with {len(metadata_columns)} columns"


@router.post(
    "/api/foundation_models/enrich_with_metadata",
    response_model=MetadataEnrichmentResponse,
)
async def enrich_with_metadata(
    request: MetadataEnrichmentRequest = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
    db: Session = Depends(get_db),
) -> MetadataEnrichmentResponse:
    """
    Enrich user data with external metadata (weather) or just fetch metadata.

    This endpoint can:
    1. Just fetch metadata if only metadata config is provided
    2. Enrich user's DataFrame with metadata if both are provided

    For weather: Uses min/max timestamps from user data to determine date range

    The returned metadata columns can be used as covariates in forecasting.
    """
    # Handle authentication and get user_id
    AuthenticationUtils.validate_authentication_headers(
        authorization, x_api_key, db
    )

    logger.info("Starting metadata enrichment request")

    try:
        # Process user DataFrame if provided
        (
            user_df,
            user_timestamps,
            original_row_count,
        ) = await _process_user_dataframe(request)

        # Process metadata based on type
        if request.weather_metadata_info:
            metadata_dataframes = await _process_weather_metadata(
                request.weather_metadata_info, user_timestamps
            )
        elif request.haver_metadata_info:
            metadata_dataframes = await _process_haver_metadata(
                request.haver_metadata_info, user_timestamps
            )
        else:
            metadata_dataframes = None

        if not metadata_dataframes:
            return MetadataEnrichmentResponse(
                status=StatusCode.internal_server_error,
                message="Failed to fetch metadata from external source",
            )

        # Combine all metadata DataFrames
        combined_metadata_df, metadata_columns, metadata_description = (
            _combine_metadata_dataframes(metadata_dataframes)
        )

        if combined_metadata_df is None or combined_metadata_df.empty:
            return MetadataEnrichmentResponse(
                status=StatusCode.internal_server_error,
                message="No metadata could be processed",
            )

        # Handle result based on whether user DataFrame was provided
        if user_df is None:
            result_df = combined_metadata_df
            final_row_count = len(result_df)
            common_timestamps_count = None
            logger.info(
                f"Returning metadata-only DataFrame with {final_row_count} rows"
            )
        else:
            # user_timestamp_column is guaranteed to be not None when user_df is not None
            if request.user_timestamp_column is None:
                raise ValueError(
                    "user_timestamp_column is required when user DataFrame is provided"
                )
            result_df, common_timestamps_count = (
                _merge_user_and_metadata_dataframes(
                    user_df, combined_metadata_df, request.user_timestamp_column
                )
            )
            final_row_count = len(result_df)

        # Convert result to JSON format
        result_json = result_df.to_dict(orient="records")

        # Generate message
        metadata_count = len(set(combined_metadata_df["timestamp"]))
        message = _generate_response_message(
            user_df,
            original_row_count,
            final_row_count,
            common_timestamps_count,
            metadata_count,
            metadata_columns,
        )

        return MetadataEnrichmentResponse(
            status=StatusCode.ok,
            message=message,
            result_dataframe_json=result_json,
            metadata_columns=metadata_columns,
            metadata_description=metadata_description,
            original_row_count=original_row_count,
            final_row_count=final_row_count,
            common_timestamps_count=common_timestamps_count,
        )

    except ValueError as e:
        return MetadataEnrichmentResponse(
            status=StatusCode.bad_request,
            message=str(e),
        )
    except Exception as e:
        logger.exception(f"Error in metadata enrichment: {str(e)}")
        return MetadataEnrichmentResponse(
            status=StatusCode.internal_server_error,
            message=f"Failed to enrich with metadata: {str(e)}",
        )


@router.post(
    "/api/foundation_models/backtests/list",
    response_model=ListBacktestsResponse,
)
async def list_available_backtests(
    request: ListBacktestsRequest = Body(...),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> ListBacktestsResponse:
    """
    List available backtest results for a user with pagination support.

    This endpoint searches for backtest result files in S3 and returns paginated information
    about each backtest including download URLs and execution timestamps.

    Args:
        request: ListBacktestsRequest containing user_id, optional dataset_name, and pagination params

    Returns:
        ListBacktestsResponse with paginated list of available backtests
    """
    logger.info(
        f"Starting list_available_backtests for user_id: {request.user_id}, "
        f"page: {request.page}, page_size: {request.page_size}"
    )

    try:
        # Get settings and S3 bucket
        settings, bucket_name = _setup_forecast_environment()

        # Build more specific prefix to reduce S3 listing scope
        search_prefix = f"{request.user_id}/"

        # Add dataset-specific prefix if filtering by dataset
        if request.dataset_name:
            search_prefix = (
                f"{request.user_id}/foundation_models/{request.dataset_name}/"
            )

        async with aioboto3_session.client("s3") as s3_client:  # type: ignore
            # First pass: collect backtest metadata with early pagination
            paginator = s3_client.get_paginator("list_objects_v2")
            backtest_metadata = []

            async for page in paginator.paginate(
                Bucket=bucket_name, Prefix=search_prefix
            ):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    s3_key = obj["Key"]

                    # Check if this is a backtest result file
                    if BACKTEST_RESULTS_SUFFIX in s3_key and s3_key.endswith(
                        ".zip"
                    ):
                        # Extract dataset name from the file path
                        dataset_name = s3_key.split("/foundation_models/")[
                            1
                        ].split("/")[0]

                        # Filter by dataset name if specified
                        if (
                            request.dataset_name
                            and dataset_name != request.dataset_name
                        ):
                            continue

                        # Extract execution datetime from the filename
                        execution_datetime = (
                            _extract_execution_datetime_from_key(s3_key)
                        )

                        if execution_datetime:
                            backtest_metadata.append(
                                {
                                    "s3_key": s3_key,
                                    "execution_datetime": execution_datetime,
                                    "dataset_name": dataset_name,
                                    "file_size_bytes": obj.get("Size"),
                                }
                            )

            # Sort backtests by execution datetime
            reverse_sort = request.sort_order == "desc"
            backtest_metadata.sort(
                key=lambda x: x["execution_datetime"], reverse=reverse_sort
            )

            # Calculate pagination
            total_items = len(backtest_metadata)
            total_pages = (
                (total_items + request.page_size - 1) // request.page_size
                if total_items > 0
                else 1
            )
            start_index = (request.page - 1) * request.page_size
            end_index = start_index + request.page_size

            # Get the backtest metadata for current page
            current_page_metadata = backtest_metadata[start_index:end_index]

            # Second pass: generate presigned URLs in parallel for the current page
            backtests = []
            if current_page_metadata:
                # Generate presigned URLs in parallel
                presigned_url_tasks = []
                for metadata in current_page_metadata:
                    task = acreate_presigned_url(
                        aioboto3_session,
                        bucket=bucket_name,
                        s3_key=metadata["s3_key"],
                    )
                    presigned_url_tasks.append(task)

                # Wait for all presigned URLs to be generated
                presigned_urls = await asyncio.gather(
                    *presigned_url_tasks, return_exceptions=True
                )

                # Create backtest info objects
                for i, metadata in enumerate(current_page_metadata):
                    download_url = presigned_urls[i]
                    if isinstance(download_url, Exception):
                        logger.warning(
                            f"Failed to generate presigned URL for {metadata['s3_key']}: {download_url}"
                        )
                        continue

                    if download_url and isinstance(download_url, str):
                        backtest_info = BacktestInfo(
                            s3_key=metadata["s3_key"],
                            download_url=download_url,
                            execution_datetime=metadata["execution_datetime"],
                            dataset_name=metadata["dataset_name"],
                            file_size_bytes=metadata["file_size_bytes"],
                        )
                        backtests.append(backtest_info)

            # Create pagination info
            pagination_info = PaginationInfo(
                current_page=request.page,
                page_size=request.page_size,
                total_items=total_items,
                total_pages=total_pages,
                has_next=request.page < total_pages,
                has_previous=request.page > 1,
            )

            logger.info(
                f"Found {total_items} total backtest results for user {request.user_id}, "
                f"returning page {request.page} with {len(backtests)} items"
            )

            return ListBacktestsResponse(
                status=StatusCode.ok,
                message=f"Successfully retrieved page {request.page} of {total_pages} "
                f"({len(backtests)} of {total_items} total backtest results)",
                backtests=backtests,
                pagination=pagination_info,
            )

    except Exception as e:
        logger.error(f"Error listing backtests: {str(e)}")
        return ListBacktestsResponse(
            status=StatusCode.internal_server_error,
            message=f"Failed to list backtests: {str(e)}",
            backtests=[],
            pagination=PaginationInfo(
                current_page=request.page,
                page_size=request.page_size,
                total_items=0,
                total_pages=0,
                has_next=False,
                has_previous=False,
            ),
        )


def _extract_execution_datetime_from_key(s3_key: str) -> Optional[str]:
    """
    Extract execution datetime from backtest S3 key.

    Args:
        s3_key: S3 key of the backtest file

    Returns:
        ISO format datetime string or None if parsing fails
    """
    try:
        # Extract timestamp from the key
        # Pattern: ..._backtest_results_{timestamp}.zip
        if BACKTEST_RESULTS_SUFFIX in s3_key and s3_key.endswith(".zip"):
            # Extract the timestamp part
            timestamp_part = s3_key.split(BACKTEST_RESULTS_SUFFIX)[1].replace(
                ".zip", ""
            )

            # Parse the timestamp using the same format as creation
            dt = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
            return dt.isoformat()

        return None
    except Exception as e:
        logger.warning(
            f"Failed to extract execution datetime from key {s3_key}: {str(e)}"
        )
        return None
