import os
import time
from datetime import datetime
from typing import List

import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, Request
from loguru import logger

from synthefy_pkg.app.config import DataRetrievalSettings
from synthefy_pkg.app.data_models import (
    DatasetInfo,
    DeletePreprocessedDatasetResponse,
    DeleteTrainingJobsResponse,
    ListAllTrainingJobsResponse,
    ListSyntheticDataAgentRunsRequest,
    ListSyntheticDataAgentRunsResponse,
    RetrievePostPreProcessReportResponse,
    RetrievePostprocessReportResponse,
    RetrievePreprocessingConfigRequest,
    RetrievePreprocessingConfigResponse,
    RetrieveSummaryReportResponse,
    RetrieveSyntheticDataAgentRunRequest,
    RetrieveSyntheticDataAgentRunResponse,
    RetrieveTrainingConfigModelRequest,
    RetrieveTrainingConfigModelResponse,
    RetrieveTrainJobIDsResponse,
    SyntheticDataAgentRunInfo,
    TrainingJobInfo,
)
from synthefy_pkg.app.services.summarize_service import DATA_SUMMARY_REPORT_DIR
from synthefy_pkg.app.tasks import SYNTHETIC_DATA_FILENAME
from synthefy_pkg.app.utils.api_utils import (
    delete_s3_objects,
    get_settings,
    get_train_config_file_name,
    s3_prefix_exists,
)
from synthefy_pkg.app.utils.s3_utils import create_presigned_url

COMPILE = True


class DataRetrievalService:
    def __init__(self, settings: DataRetrievalSettings):
        logger.info("Initializing DataRetrievalService...")
        start_time = time.time()

        self.settings = settings
        self.s3_client = boto3.client("s3")

        logger.info(
            f"DataRetrievalService initialized in {time.time() - start_time:.2f}s"
        )

    async def list_preprocessed_datasets(
        self, user_id: str
    ) -> List[DatasetInfo]:
        """
        List all preprocessed datasets for a user, sorted by last modification time (newest first).
        Skip the training_logs folder
        """
        logger.info(f"Listing preprocessed datasets for user {user_id}...")
        start_time = time.time()

        try:
            # Setup pagination for S3 listing
            user_prefix = f"{user_id.strip()}/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.settings.bucket_name, Prefix=user_prefix
            )

            # Track datasets and their latest modification time
            datasets = {}

            # Process all objects in the user's prefix
            for page in pages:
                for obj in page.get("Contents", []):
                    # get dataset name from the path
                    if "Key" not in obj:
                        continue
                    relative_path = obj["Key"][len(user_prefix) :]
                    if not relative_path:
                        continue

                    dataset_name = relative_path.split("/")[0]

                    if (
                        not dataset_name
                        or dataset_name == "training_logs"
                        or dataset_name == "generation_logs"
                        or dataset_name == "foundation_models"
                        or dataset_name == DATA_SUMMARY_REPORT_DIR
                    ):
                        continue

                    # Update last modified time if newer
                    if "LastModified" not in obj:
                        continue
                    last_modified = obj["LastModified"]
                    if (
                        dataset_name not in datasets
                        or last_modified > datasets[dataset_name]
                    ):
                        datasets[dataset_name] = last_modified

            # Convert to list of DatasetInfo objects, sorted by last_modified
            result = [
                DatasetInfo(
                    name=name,
                    last_modified=modified_time.isoformat(),  # Convert datetime to ISO string
                )
                for name, modified_time in sorted(
                    datasets.items(), key=lambda x: x[1], reverse=True
                )
            ]

            logger.info(
                f"Found {len(result)} datasets for user {user_id} "
                f"(took {time.time() - start_time:.2f}s)"
            )
            return result

        except Exception as e:
            logger.error(f"Error listing datasets for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to list datasets: {str(e)}"
            )

    async def list_training_jobs(
        self, user_id: str, dataset_name: str
    ) -> RetrieveTrainJobIDsResponse:
        """
        Retrieve training job IDs for a given user_id and dataset_name.
        Only includes jobs that have a completed model checkpoint.
        Currently this is done by checking for the presence of a model.ckpt file in the output/model/ directory.
        """
        logger.info(
            f"Listing training jobs for dataset {dataset_name} (user: {user_id})..."
        )
        start_time = time.time()

        try:
            # Check if dataset exists
            dataset_prefix = f"{user_id}/{dataset_name}/"
            if not s3_prefix_exists(
                self.s3_client, self.settings.bucket_name, dataset_prefix
            ):
                logger.warning(
                    f"Dataset {dataset_name} not found for user {user_id}"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {dataset_name} not found for user {user_id}",
                )

            # Get all training jobs from the training_logs directory
            training_logs_prefix = f"{user_id}/training_logs/{dataset_name}/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.settings.bucket_name,
                Prefix=training_logs_prefix,
                Delimiter="/",
            )

            synthesis_jobs = []
            forecast_jobs = []

            # Process each potential training job folder
            for page in pages:
                for prefix_obj in page.get("CommonPrefixes", []):
                    logger.info(f"Processing prefix: {prefix_obj}")
                    job_prefix = prefix_obj.get("Prefix", "")
                    job_id = job_prefix.strip("/").split("/")[-1]

                    # Check if model checkpoint exists
                    checkpoint_path = f"{job_prefix}output/model/model.ckpt"
                    try:
                        self.s3_client.head_object(
                            Bucket=self.settings.bucket_name,
                            Key=checkpoint_path,
                        )
                        # Model checkpoint exists, categorize the job
                        if "synthesis" in job_id.lower():
                            synthesis_jobs.append(job_id)
                        elif "forecast" in job_id.lower():
                            forecast_jobs.append(job_id)
                    except Exception:
                        # Skip jobs without model checkpoint
                        logger.debug(
                            f"No model checkpoint found for job {job_id}"
                        )
                        continue

            logger.info(
                f"Found {len(synthesis_jobs)} synthesis and {len(forecast_jobs)} forecast jobs "
                f"for dataset {dataset_name} (user: {user_id})"
            )

            result = RetrieveTrainJobIDsResponse(
                synthesis_train_job_ids=sorted(synthesis_jobs),
                forecast_train_job_ids=sorted(forecast_jobs),
            )

            logger.info(
                f"Retrieved {len(synthesis_jobs)} synthesis and {len(forecast_jobs)} forecast jobs "
                f"for dataset {dataset_name} (user: {user_id}, took {time.time() - start_time:.2f}s)"
            )
            return result

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error listing training jobs: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to list training jobs"
            )

    async def delete_preprocessed_dataset(
        self, user_id: str, dataset_name: str
    ) -> DeletePreprocessedDatasetResponse:
        """
        Delete the preprocessed dataset for a given user_id and dataset_name.
        Training jobs are preserved and must be deleted separately.
        """
        logger.info(
            f"Deleting preprocessed dataset {dataset_name} for user {user_id}..."
        )
        start_time = time.time()

        try:
            dataset_prefix = f"{user_id}/{dataset_name}/"

            # Check if dataset exists
            if not s3_prefix_exists(
                self.s3_client, self.settings.bucket_name, dataset_prefix
            ):
                logger.warning(
                    f"Dataset {dataset_name} not found for user {user_id}"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {dataset_name} not found for user {user_id}",
                )

            # Only delete the preprocessed dataset
            delete_s3_objects(
                self.s3_client, self.settings.bucket_name, dataset_prefix
            )

            logger.info(
                f"Successfully deleted preprocessed dataset {dataset_name} for user {user_id} "
                f"(took {time.time() - start_time:.2f}s)"
            )
            return DeletePreprocessedDatasetResponse(
                status="success",
                message=f"Successfully deleted preprocessed dataset {dataset_name}",
                deleted_dataset_path=dataset_prefix,
            )

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(
                f"Error deleting dataset {dataset_name} for user {user_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail="Failed to delete dataset"
            )

    async def delete_training_jobs(
        self, user_id: str, dataset_name: str, training_job_ids: List[str]
    ) -> DeleteTrainingJobsResponse:
        """
        Delete specific training jobs for a dataset.
        Training jobs can still exist even if the preprocessed dataset no longer exists.
        """
        logger.info(
            f"Deleting {len(training_job_ids)} training jobs for dataset {dataset_name} "
            f"(user: {user_id})"
        )
        start_time = time.time()

        try:
            # Check if training logs directory exists
            training_logs_prefix = f"{user_id}/training_logs/{dataset_name}/"
            if not s3_prefix_exists(
                self.s3_client, self.settings.bucket_name, training_logs_prefix
            ):
                logger.warning(
                    f"No training jobs found for dataset {dataset_name} (user: {user_id})"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"No training jobs found for dataset {dataset_name} (user: {user_id})",
                )

            # Get existing training jobs
            existing_jobs = self._get_training_jobs(user_id, dataset_name)

            # Validate all given job_ids exist
            invalid_jobs = [
                job_id
                for job_id in training_job_ids
                if job_id not in existing_jobs
            ]
            if invalid_jobs:
                raise HTTPException(
                    status_code=404,
                    detail=f"Training jobs not found: {', '.join(invalid_jobs)}",
                )

            # Delete each training job
            for job_id in training_job_ids:
                job_prefix = f"{user_id}/training_logs/{dataset_name}/{job_id}/"
                delete_s3_objects(
                    self.s3_client, self.settings.bucket_name, job_prefix
                )
                logger.info(f"Deleted training job {job_id}")

            logger.info(
                f"Successfully deleted {len(training_job_ids)} training jobs "
                f"for dataset {dataset_name} (user: {user_id}, took {time.time() - start_time:.2f}s)"
            )
            return DeleteTrainingJobsResponse(
                status="success",
                message=f"Successfully deleted {len(training_job_ids)} training jobs",
            )

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error deleting training jobs: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to delete training jobs"
            )

    def _list_job_ids(self, prefix: str) -> List[str]:
        """
        Helper method to list job IDs from a given S3 prefix.
        """
        if not s3_prefix_exists(
            self.s3_client, self.settings.bucket_name, prefix
        ):
            return []

        paginator = self.s3_client.get_paginator("list_objects_v2")
        job_ids = []

        for page in paginator.paginate(
            Bucket=self.settings.bucket_name, Prefix=prefix, Delimiter="/"
        ):
            for prefix_obj in page.get("CommonPrefixes", []):
                # get job ID from path
                job_id = prefix_obj.get("Prefix", "").strip("/").split("/")[-1]
                if job_id:
                    job_ids.append(job_id)

        return sorted(job_ids)

    def _get_training_jobs(self, user_id: str, dataset_name: str) -> List[str]:
        """
        Helper method to get all training job IDs for a dataset.
        """
        training_logs_prefix = f"{user_id}/training_logs/{dataset_name}/"
        return self._list_job_ids(training_logs_prefix)

    async def list_all_training_jobs(
        self, user_id: str
    ) -> ListAllTrainingJobsResponse:
        """
        Retrieve all training jobs for a given user_id across all datasets.
        """
        try:
            # Get the training_logs directory for the user
            training_logs_prefix = f"{user_id}/training_logs/"

            # List all datasets in training_logs
            datasets = self._list_job_ids(training_logs_prefix)
            training_jobs = []

            # For each dataset, get all training jobs
            for dataset_name in datasets:
                dataset_prefix = f"{training_logs_prefix}{dataset_name}/"
                paginator = self.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self.settings.bucket_name,
                    Prefix=dataset_prefix,
                    Delimiter="/",
                )

                # Process each training job folder
                for page in pages:
                    for prefix_obj in page.get("CommonPrefixes", []):
                        job_prefix = prefix_obj.get("Prefix", "")
                        job_id = job_prefix.strip("/").split("/")[-1]

                        # Determine job type from job ID
                        if "synthesis" in job_id.lower():
                            job_type = "synthesis"
                        elif "forecast" in job_id.lower():
                            job_type = "forecast"
                        else:
                            job_type = "embedding"

                        training_jobs.append(
                            TrainingJobInfo(
                                training_job_id=job_id,
                                job_type=job_type,
                                dataset_name=dataset_name,
                            )
                        )

            logger.info(
                f"Found {len(training_jobs)} total training jobs for user {user_id}"
            )
            return ListAllTrainingJobsResponse(training_jobs=training_jobs)

        except Exception as e:
            logger.error(
                f"Error listing all training jobs for user {user_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list all training jobs: {str(e)}",
            )

    async def get_dataset_summary(
        self, user_id: str, dataset_name: str
    ) -> RetrieveSummaryReportResponse:
        """
        Get dataset summary PDF URL for a given user_id and dataset_name.

        Args:
            user_id: User ID requesting the dataset summary
            dataset_name: Name of the dataset to fetch summary for

        Returns:
            RetrieveSummaryReportResponse containing presigned URL to download the PDF

        Raises:
            HTTPException: If summary not found or other errors occur
        """
        logger.info(
            f"Getting dataset summary for {dataset_name} (user: {user_id})..."
        )
        start_time = time.time()

        try:
            # Construct the S3 key for the HTML
            # TODO - make this whole formatted string path from the summarize service not hardcoded
            s3_key = os.path.join(
                user_id,
                DATA_SUMMARY_REPORT_DIR,
                dataset_name,
                f"{dataset_name}.html",
            )

            # Check if file exists in S3
            try:
                self.s3_client.head_object(
                    Bucket=self.settings.bucket_name, Key=s3_key
                )
            except Exception as e:
                logger.error(f"Summary HTML not found: {str(e)}")

                # Check if PDF version exists instead
                pdf_s3_key = s3_key.replace(".html", ".pdf")

                try:
                    self.s3_client.head_object(
                        Bucket=self.settings.bucket_name, Key=pdf_s3_key
                    )
                    # PDF exists, update s3_key to use PDF instead
                    s3_key = pdf_s3_key
                    logger.info(
                        f"Using PDF version instead of HTML for {dataset_name}"
                    )

                except Exception as e:
                    logger.error(
                        f"Neither HTML nor PDF summary found: {str(e)}"
                    )
                    raise HTTPException(
                        status_code=404,
                        detail=f"Summary HTML not found for dataset {dataset_name}",
                    )

            # Generate presigned URL
            presigned_url = create_presigned_url(
                self.s3_client, self.settings.bucket_name, s3_key
            )

            if not presigned_url:
                raise HTTPException(
                    status_code=500, detail="Failed to generate presigned URL"
                )

            logger.info(
                f"Successfully retrieved dataset summary for {dataset_name} "
                f"(user: {user_id}, took {time.time() - start_time:.2f}s)"
            )

            return RetrieveSummaryReportResponse(
                status="success",
                message="Dataset summary retrieved successfully",
                presigned_url=presigned_url,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting dataset summary: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve dataset summary: {str(e)}",
            )

    async def retrieve_postprocess_report(
        self, user_id: str, dataset_name: str, job_id: str
    ) -> RetrievePostprocessReportResponse:
        """
        Retrieve the postprocess report HTML for a specific training job.
        Args:
            user_id: User ID requesting the report
            dataset_name: Name of the dataset
            job_id: Training job ID
        Returns:
            RetrievePostprocessReportResponse containing presigned URL to download the HTML -> presigned_url will be None if report not found
        Raises:
            HTTPException: If report not found or other errors occur
        """
        logger.info(
            f"Retrieving postprocess report for job {job_id} "
            f"(dataset: {dataset_name}, user: {user_id})"
        )
        start_time = time.time()

        try:
            # Construct the S3 key for the HTML
            # TODO: make this formatted string part of postprocess service. this is too messy to copy this in 2 places.
            s3_key = os.path.join(
                f"{user_id}",
                "training_logs",
                dataset_name,
                job_id,
                "output",
                "model",
                "postprocessing",
                "reports",
                "postprocessing_report.html",
            )

            # Check if file exists in S3
            try:
                self.s3_client.head_object(
                    Bucket=self.settings.bucket_name, Key=s3_key
                )
            except Exception as e:
                logger.error(f"Postprocess report not found: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Postprocess report not found for job {job_id}",
                )

            # Generate presigned URL
            presigned_url = create_presigned_url(
                self.s3_client, self.settings.bucket_name, s3_key
            )

            if not presigned_url:
                # return None so UI can check to give user option to generate it.
                return RetrievePostprocessReportResponse(
                    status="error",
                    message="Failed to generate presigned URL",
                    presigned_url=None,
                )

            logger.info(
                f"Successfully retrieved postprocess report "
                f"(took {time.time() - start_time:.2f}s)"
            )

            return RetrievePostprocessReportResponse(
                status="success",
                message="Postprocess report retrieved successfully",
                presigned_url=presigned_url,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving postprocess report: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve postprocess report: {str(e)}",
            )

    async def retrieve_postpreprocess_report(
        self, user_id: str, dataset_name: str
    ) -> RetrievePostPreProcessReportResponse:
        """
        Retrieve the postpreprocess report HTML for a specific training job.
        Args:
            user_id: User ID requesting the report
            dataset_name: Name of the dataset
        Returns:
            RetrievePostPreProcessReportResponse containing presigned URL to download the HTML -> presigned_url will be None if report not found
        Raises:
            HTTPException: If report not found or other errors occur
        """
        logger.info(
            f"Retrieving postpreprocess report for dataset {dataset_name} "
            f"(user: {user_id})"
        )
        start_time = time.time()

        try:
            # Try to find both scaled and unscaled versions of the report
            possible_filenames = [
                f"{dataset_name}_post_preprocessing_scaled.html",
                f"{dataset_name}_post_preprocessing_unscaled.html",
            ]

            presigned_urls = []

            for filename in possible_filenames:
                s3_key = os.path.join(
                    user_id,
                    "reports",
                    dataset_name,
                    filename,
                )

                # Check if file exists in S3
                try:
                    self.s3_client.head_object(
                        Bucket=self.settings.bucket_name, Key=s3_key
                    )

                    # Generate presigned URL
                    presigned_url = create_presigned_url(
                        self.s3_client, self.settings.bucket_name, s3_key
                    )

                    if presigned_url:
                        presigned_urls.append(presigned_url)
                except Exception as e:
                    logger.debug(f"File {filename} not found in S3: {str(e)}")
                    continue

            if not presigned_urls:
                # return None so UI can check to give user option to generate it.
                return RetrievePostPreProcessReportResponse(
                    status="error",
                    message="Failed to generate presigned URLs",
                    presigned_urls=[],
                )

            logger.info(
                f"Successfully retrieved {len(presigned_urls)} postpreprocess reports "
                f"(took {time.time() - start_time:.2f}s)"
            )

            return RetrievePostPreProcessReportResponse(
                status="success",
                message=f"Retrieved {len(presigned_urls)} postpreprocess reports successfully",
                presigned_urls=presigned_urls,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving postpreprocess report: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve postpreprocess report: {str(e)}",
            )

    # TODO - support local path listing
    async def list_synthetic_data(
        self, user_id: str, dataset_name: str
    ) -> ListSyntheticDataAgentRunsResponse:
        """
        List all synthetic data runs for a given user_id and dataset_name.
        """
        logger.info(
            f"Listing synthetic data for dataset {dataset_name} (user: {user_id})..."
        )
        start_time = time.time()

        try:
            # List objects in the generation_logs directory
            prefix = f"{user_id}/generation_logs/{dataset_name}/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.settings.bucket_name, Prefix=prefix, Delimiter="/"
            )

            synthetic_data_runs = []

            # Process each potential synthetic data run folder
            for page in pages:
                for prefix_obj in page.get("CommonPrefixes", []):
                    run_prefix = prefix_obj.get("Prefix", "")
                    run_name = run_prefix.strip("/").split("/")[-1]

                    # Get the last modified time for this run
                    try:
                        response = self.s3_client.list_objects_v2(
                            Bucket=self.settings.bucket_name,
                            Prefix=run_prefix,
                            MaxKeys=1,
                        )
                        if "Contents" in response and response["Contents"]:
                            last_modified = response["Contents"][0].get(
                                "LastModified", ""
                            )

                            synthetic_data_runs.append(
                                SyntheticDataAgentRunInfo(
                                    run_name=run_name,
                                    last_modified=(
                                        last_modified.isoformat()
                                        if isinstance(last_modified, datetime)
                                        else ""
                                    ),
                                )
                            )
                    except Exception as e:
                        logger.error(
                            f"Error getting details for run {run_name}: {str(e)}"
                        )
                        continue

            logger.info(
                f"Found {len(synthetic_data_runs)} synthetic data runs "
                f"(took {time.time() - start_time:.2f}s)"
            )

            return ListSyntheticDataAgentRunsResponse(
                synthetic_data_runs=synthetic_data_runs
            )

        except Exception as e:
            logger.error(f"Error listing synthetic data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list synthetic data: {str(e)}",
            )

    # TODO - support local path retrieval
    async def retrieve_synthetic_data(
        self, user_id: str, dataset_name: str, run_name: str
    ) -> RetrieveSyntheticDataAgentRunResponse:
        """
        Get synthetic data for a specific run.
        """
        logger.info(
            f"Retrieving synthetic data for run {run_name} "
            f"(dataset: {dataset_name}, user: {user_id})"
        )
        start_time = time.time()

        try:
            # Construct the S3 path
            s3_prefix = os.path.join(
                user_id,
                "generation_logs",
                dataset_name,
                run_name,
            )

            # Check if directory exists
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.settings.bucket_name,
                    Prefix=s3_prefix,
                    MaxKeys=1,
                )
                if "Contents" not in response:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Synthetic data not found for run {run_name}",
                    )
            except Exception as e:
                logger.error(
                    f"Error checking synthetic data existence: {str(e)}"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Synthetic data not found for run {run_name}",
                )

            # Generate presigned URL for the zip file
            zip_key = f"{s3_prefix}/{SYNTHETIC_DATA_FILENAME}"
            presigned_url = create_presigned_url(
                self.s3_client, self.settings.bucket_name, zip_key
            )

            if not presigned_url:
                raise HTTPException(
                    status_code=500, detail="Failed to generate presigned URL"
                )

            logger.info(
                f"Successfully retrieved synthetic data "
                f"(took {time.time() - start_time:.2f}s)"
            )

            return RetrieveSyntheticDataAgentRunResponse(
                status="success",
                message="Synthetic data retrieved successfully",
                presigned_url=presigned_url,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving synthetic data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve synthetic data: {str(e)}",
            )

    def check_run_exists(
        self, user_id: str, dataset_name: str, run_name: str
    ) -> bool:
        """
        Check if a synthetic data run already exists.

        Args:
            user_id: User ID
            dataset_name: Name of the dataset
            run_name: Name of the run to check

        Returns:
            bool: True if run exists, False otherwise
        """
        try:
            s3_prefix = os.path.join(
                user_id,
                "generation_logs",
                dataset_name,
                run_name,
            )

            response = self.s3_client.list_objects_v2(
                Bucket=self.settings.bucket_name,
                Prefix=s3_prefix,
                MaxKeys=1,
            )

            return "Contents" in response and len(response["Contents"]) > 0
        except Exception as e:
            logger.error(f"Error checking if run exists: {str(e)}")
            return False

    def dataset_exists(self, user_id: str, dataset_name: str) -> bool:
        """
        Check if a dataset with the given name already exists for this user.

        Args:
            user_id: The ID of the user
            dataset_name: The name to check

        Returns:
            bool: True if dataset exists, False otherwise
        """
        try:
            dataset_prefix = (
                f"{user_id.strip()}/{dataset_name.strip()}_{user_id.strip()}/"
            )
            response = self.s3_client.list_objects_v2(
                Bucket=self.settings.bucket_name,
                Prefix=dataset_prefix,
                MaxKeys=1,
            )
            return "Contents" in response and len(response["Contents"]) > 0
        except Exception as e:
            logger.error(f"Error checking dataset existence: {str(e)}")
            return False

    async def retrieve_training_config_model(
        self, user_id: str, dataset_name: str, training_job_id: str
    ) -> RetrieveTrainingConfigModelResponse:
        """
        Retrieve the training configuration file (config.yaml) for a specific training job.

        Args:
            user_id: User ID requesting the config
            dataset_name: Name of the dataset
            training_job_id: Training job ID

        Returns:
            RetrieveTrainingConfigModelResponse containing presigned URL to download the config YAML.

        Raises:
            HTTPException: If the config file is not found or other errors occur.
        """
        logger.info(
            f"Retrieving training config for job {training_job_id} "
            f"(dataset: {dataset_name}, user: {user_id})"
        )
        start_time = time.time()

        try:
            # Determine the task type from the job ID to use the correct helper
            task_type = (
                "forecast"
                if "forecast" in training_job_id.lower()
                else "synthesis"
            )  # Add more types if needed

            # Define potential S3 keys
            new_config_file_name = get_train_config_file_name(
                dataset_name=dataset_name,
                task=task_type,
                job_name=training_job_id,
            )
            new_s3_key = f"{user_id}/{dataset_name}/{new_config_file_name}"

            old_config_file_name = get_train_config_file_name(
                dataset_name=dataset_name,
                task=task_type,
                # job_name is omitted for old style
            )
            old_s3_key = f"{user_id}/{dataset_name}/{old_config_file_name}"

            keys_to_try = [new_s3_key, old_s3_key]
            found_s3_key = None

            # Attempt to find the config file using the defined keys
            for s3_key in keys_to_try:
                try:
                    self.s3_client.head_object(
                        Bucket=self.settings.bucket_name, Key=s3_key
                    )
                    found_s3_key = s3_key
                    logger.info(f"Found training config using key: {s3_key}")
                    break  # Stop checking once found
                except ClientError as e:
                    error_response = e.response
                    error_details = error_response.get("Error", {})
                    error_code = error_details.get("Code")
                    error_message = error_details.get(
                        "Message", "Unknown S3 error"
                    )

                    if error_code == "404":
                        logger.warning(
                            f"Config file not found with key {s3_key}. Trying next..."
                        )
                        continue  # Try the next key in the list
                    else:
                        # Unexpected Boto3 client error
                        logger.error(f"Error checking key {s3_key}: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error accessing S3 for key {s3_key}: {error_message}",
                        )
                except Exception as e:
                    # Other unexpected errors during S3 check
                    logger.error(
                        f"Unexpected error checking key {s3_key}: {str(e)}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unexpected error checking key {s3_key}: {str(e)}",
                    )

            # If no key was found after trying all possibilities
            if not found_s3_key:
                final_error_detail = (
                    f"Training config file not found for job {training_job_id} "
                    f"(dataset: {dataset_name}) using expected naming conventions. "
                    f"Checked keys: {', '.join(keys_to_try)}"
                )
                logger.error(final_error_detail)
                raise HTTPException(status_code=404, detail=final_error_detail)

            # Generate presigned URL using the found key
            presigned_url = create_presigned_url(
                self.s3_client, self.settings.bucket_name, found_s3_key
            )

            if not presigned_url:
                raise HTTPException(
                    status_code=500, detail="Failed to generate presigned URL"
                )

            logger.info(
                f"Successfully retrieved training config for job {training_job_id} "
                f"(took {time.time() - start_time:.2f}s)"
            )

            return RetrieveTrainingConfigModelResponse(
                status="success",
                message="Training config retrieved successfully",
                presigned_url=presigned_url,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving training config: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve training config: {str(e)}",
            )

    async def retrieve_preprocessing_config(
        self, user_id: str, dataset_name: str
    ) -> RetrievePreprocessingConfigResponse:
        """
        Retrieve the preprocessing configuration file (config_{dataset_name}_preprocessing.json)
        for a specific dataset.

        Args:
            user_id: User ID requesting the config
            dataset_name: Name of the dataset

        Returns:
            RetrievePreprocessingConfigResponse containing presigned URL to download the config JSON.

        Raises:
            HTTPException: If the config file is not found or other errors occur.
        """
        logger.info(
            f"Retrieving preprocessing config for dataset {dataset_name} (user: {user_id})"
        )
        start_time = time.time()

        try:
            # Construct the S3 key for the preprocessing config JSON
            config_file_name = f"config_{dataset_name}_preprocessing.json"
            s3_key = f"{user_id}/{dataset_name}/{config_file_name}"

            # Check if file exists in S3
            try:
                self.s3_client.head_object(
                    Bucket=self.settings.bucket_name, Key=s3_key
                )
            except ClientError as e:
                error_response = e.response
                error_details = error_response.get("Error", {})
                error_code = error_details.get("Code")
                error_message = error_details.get("Message", "Unknown S3 error")

                if error_code == "404":
                    logger.error(
                        f"Preprocessing config file not found at {s3_key}: {str(e)}"
                    )
                    raise HTTPException(
                        status_code=404,
                        detail=f"Preprocessing config file not found for dataset {dataset_name}",
                    )
                else:
                    # Re-raise unexpected Boto3 client error
                    logger.error(f"Error checking key {s3_key}: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error accessing S3: {error_message}",
                    )
            except Exception as e:
                # Other unexpected errors
                logger.error(
                    f"Unexpected error checking key {s3_key}: {str(e)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error: {str(e)}",
                )

            # Generate presigned URL
            presigned_url = create_presigned_url(
                self.s3_client, self.settings.bucket_name, s3_key
            )

            if not presigned_url:
                raise HTTPException(
                    status_code=500, detail="Failed to generate presigned URL"
                )

            logger.info(
                f"Successfully retrieved preprocessing config for dataset {dataset_name} "
                f"(took {time.time() - start_time:.2f}s)"
            )

            return RetrievePreprocessingConfigResponse(
                status="success",
                message="Preprocessing config retrieved successfully",
                presigned_url=presigned_url,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving preprocessing config: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve preprocessing config: {str(e)}",
            )


async def get_data_retrieval_service(
    request: Request,
) -> DataRetrievalService:
    """
    Dependency function to create a DataRetrievalService instance.

    Args:
        request: FastAPI Request object containing path parameters

    Returns:
        DataRetrievalService: Configured service instance
    """
    settings: DataRetrievalSettings = get_settings(DataRetrievalSettings)
    return DataRetrievalService(settings=settings)
