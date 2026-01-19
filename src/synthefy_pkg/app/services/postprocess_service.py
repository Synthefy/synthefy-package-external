import asyncio
import os
from typing import List, Tuple

import aioboto3
import boto3
from fastapi import HTTPException
from loguru import logger

from synthefy_pkg.app.config import PostprocessSettings
from synthefy_pkg.app.data_models import PostprocessRequest, PostprocessResponse
from synthefy_pkg.app.utils.s3_utils import (
    create_presigned_url,
    download_directory_from_s3_async,
    download_preprocessed_data_from_s3_async,
    download_training_config_from_s3_async,
    get_async_s3_client,
    upload_directory_to_s3,
)
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.postprocessing.postprocess import (
    SYNTHEFY_DATASETS_BASE,
    Postprocessor,
)
from synthefy_pkg.postprocessing.report_generator import ReportGenerator
from synthefy_pkg.utils.scaling_utils import SCALER_FILENAMES

COMPILE = True


class PostprocessService:
    def __init__(
        self, settings: PostprocessSettings, aioboto3_session: aioboto3.Session
    ):
        self.settings = settings
        self.aioboto3_session = aioboto3_session
        self.s3_client = boto3.client("s3")  # For synchronous operations
        self._cleanup_paths = []  # Store paths that need cleanup

    def get_cleanup_paths(self) -> List[str]:
        """Return paths that need cleanup after processing."""
        return self._cleanup_paths

    async def postprocess(
        self,
        request: PostprocessRequest,
    ) -> PostprocessResponse:
        self._cleanup_paths = []  # Reset cleanup paths
        user_id = request.user_id
        dataset_name = self.settings.dataset_name
        job_id = request.job_id

        plots_dir = ""
        reports_dir = ""
        config_path = ""
        preprocessed_data_path = ""
        local_dir = ""

        try:
            # Get the config file
            config_path = await self._process_config(
                user_id, dataset_name, job_id
            )
            self._cleanup_paths.append(config_path)
            logger.info(
                f"Config file has been successfully downloaded to {config_path}"
            )

            # Get the preprocessed data
            preprocessed_data_path = await self._process_preprocessed_data(
                user_id, dataset_name
            )
            self._cleanup_paths.append(preprocessed_data_path)
            logger.info(
                f"Preprocessed data has been successfully downloaded to {preprocessed_data_path}"
            )

            # Get the generated data
            local_dir = await self._process_generated_data(
                user_id, dataset_name, job_id, config_path, request.splits
            )
            self._cleanup_paths.append(local_dir)
            logger.info(
                f"Generated data has been successfully downloaded to {local_dir}"
            )

            # Run CPU-intensive postprocessing in a thread pool using asyncio.to_thread
            plots_dir, reports_dir = await asyncio.to_thread(
                self._do_postprocessing,
                config_path=config_path,
                local_dir=local_dir,
            )

            self._cleanup_paths.extend([plots_dir, reports_dir])
            logger.info(
                f"Postprocessing has been successfully completed and the plots are saved output path: {plots_dir}"
            )

            if self.settings.bucket_name != "local":
                # Upload artifacts and get the S3 key
                s3_key = await self._upload_generated_artifacts_to_s3(
                    user_id, dataset_name, job_id, plots_dir, reports_dir
                )

                # Generate presigned URL
                presigned_url = create_presigned_url(
                    self.s3_client,
                    self.settings.bucket_name,
                    f"{s3_key}/reports/postprocessing_report.html",
                )
                if not presigned_url:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to generate presigned URL",
                    )

                return PostprocessResponse(
                    status="success",
                    message="Postprocessing completed successfully",
                    presigned_url=presigned_url,
                )
            else:
                # For local development, return the local file path
                report_path = os.path.join(
                    reports_dir, "postprocessing_report.html"
                )
                return PostprocessResponse(
                    status="success",
                    message="Postprocessing completed successfully",
                    presigned_url=f"file://{report_path}",
                )

        except Exception as e:
            logger.error(f"Error in postprocessing occurred: {str(e)}")
            raise (
                e
                if isinstance(e, HTTPException)
                else HTTPException(
                    status_code=500, detail="An error while postprocessing."
                )
            )

    async def _process_config(
        self, user_id: str, dataset_name: str, job_id: str
    ) -> str:
        bucket_name = self.settings.bucket_name
        if "synthesis" in job_id.lower():
            config_path = self.settings.synthesis_config_path
            task_type = "synthesis"
        elif "forecast" in job_id.lower():
            config_path = self.settings.forecast_config_path
            task_type = "forecast"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job_id: {job_id}. It must contain either 'synthesis' or 'forecast'.",
            )

        config_path = config_path.replace("${dataset_name}", dataset_name)

        if bucket_name != "local":
            async_s3_client = await get_async_s3_client(self.aioboto3_session)
            # Download the training config file from S3 with proper naming conventions
            if not await download_training_config_from_s3_async(
                s3_client=async_s3_client,
                bucket=bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                task_type=task_type,
                config_file_path=config_path,
                training_job_id=job_id,
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download the {task_type} config for dataset: {dataset_name}, job_id: {job_id}",
                )
            logger.info(
                f"config file has been successfully downloaded to {config_path}"
            )

        # Check if config exists locally
        if not os.path.exists(config_path):
            raise HTTPException(
                status_code=404,
                detail=f"Config file does not exist for dataset: {dataset_name}",
            )

        return config_path

    async def _process_preprocessed_data(
        self, user_id: str, dataset_name: str
    ) -> str:
        preprocessed_file_names = [
            SCALER_FILENAMES["discrete"],
            SCALER_FILENAMES["timeseries"],
            SCALER_FILENAMES["continuous"],
            "labels_description.pkl",
            "colnames.json",
            "timeseries_windows_columns.json",
        ]
        preprocessed_data_path = self.settings.preprocessed_data_path.replace(
            "${dataset_name}", dataset_name
        )

        if self.settings.bucket_name != "local":
            async_s3_client = await get_async_s3_client(self.aioboto3_session)
            # Download preprocessed data from S3
            if not await download_preprocessed_data_from_s3_async(
                s3_client=async_s3_client,
                bucket=self.settings.bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                local_path=preprocessed_data_path,
                required_files=preprocessed_file_names,
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download the preprocessed data for dataset: {dataset_name}",
                )

        # Check if preprocessed data exists locally
        for file_name in preprocessed_file_names:
            file_path = os.path.join(preprocessed_data_path, file_name)
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Preprocessed data file {file_name} does not exist for dataset: {dataset_name}",
                )
        return preprocessed_data_path

    async def _process_generated_data(
        self,
        user_id: str,
        dataset_name: str,
        job_id: str,
        config_path: str,
        splits: List[str] = ["test"],
    ) -> str:
        configuration = Configuration(config_filepath=config_path)
        local_dir = configuration.get_save_dir(SYNTHEFY_DATASETS_BASE)

        if self.settings.bucket_name != "local":
            async_s3_client = await get_async_s3_client(self.aioboto3_session)
            # Download generated data from S3
            for split in splits:
                s3_dir = os.path.join(
                    user_id,
                    "training_logs",
                    dataset_name,
                    job_id,
                    "output",
                    "model",
                    f"{split}_dataset",
                )
                if not await download_directory_from_s3_async(
                    s3_client=async_s3_client,
                    bucket=self.settings.bucket_name,
                    s3_dir=s3_dir,
                    local_dir=os.path.join(local_dir, f"{split}_dataset"),
                ):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to download the {split} data.",
                    )
                logger.info(
                    f"{split} data has been successfully downloaded to {local_dir}/{split}_dataset"
                )

        # Check if generated data exists locally
        for split in splits:
            folder_path = os.path.join(local_dir, f"{split}_dataset")
            if not os.path.exists(folder_path) or not os.listdir(folder_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"{split} data does not exist or is empty for dataset: {dataset_name}",
                )
        return local_dir

    def _do_postprocessing(
        self, config_path: str, local_dir: str
    ) -> Tuple[str, str]:
        """Run CPU-intensive postprocessing in a synchronous manner."""
        postprocessor = Postprocessor(config_filepath=config_path)
        postprocessor.postprocess(plot_fourier=False)

        logger.info(
            f"Postprocessing has been completed and the plots are saved output path: {postprocessor.figsave_path}"
        )

        plots_dir = postprocessor.figsave_path
        reports_dir = os.path.join(local_dir, "reports")

        generator = ReportGenerator(config_path=config_path)
        generator.generate_html_report(
            output_html=os.path.join(reports_dir, "postprocessing_report.html"),
        )
        return plots_dir, reports_dir

    async def _upload_generated_artifacts_to_s3(
        self,
        user_id: str,
        dataset_name: str,
        job_id: str,
        plots_dir: str,
        reports_dir: str,
    ) -> str:
        # Upload the postprocessing plots to S3
        s3_key = os.path.join(
            user_id,
            "training_logs",
            dataset_name,
            job_id,
            "output",
            "model",
            "postprocessing",
        )
        if not upload_directory_to_s3(
            self.s3_client,
            self.settings.bucket_name,
            plots_dir,
            os.path.join(s3_key, "plots"),
        ):
            raise HTTPException(
                status_code=500,
                detail="Failed to upload postprocessing plots to S3",
            )
        logger.info(
            f"Uploading {plots_dir} to s3 has been successfully completed."
        )

        # Upload the postprocessing reports to S3
        if not upload_directory_to_s3(
            self.s3_client,
            self.settings.bucket_name,
            reports_dir,
            os.path.join(s3_key, "reports"),
        ):
            raise HTTPException(
                status_code=500,
                detail="Failed to upload postprocessing reports to S3",
            )
        logger.info(
            f"Uploading {reports_dir} to s3 has been successfully completed."
        )

        return s3_key
