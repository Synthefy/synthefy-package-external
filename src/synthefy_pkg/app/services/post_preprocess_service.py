import asyncio
import os
from typing import List

import aioboto3
import boto3
from fastapi import HTTPException
from loguru import logger

from synthefy_pkg.app.config import PostPreProcessSettings
from synthefy_pkg.app.data_models import (
    PostPreProcessRequest,
    PostPreProcessResponse,
)
from synthefy_pkg.app.utils.s3_utils import (
    create_presigned_url,
    download_preprocessed_data_from_s3_async,
    get_async_s3_client,
)
from synthefy_pkg.utils.post_preprocess_utils import post_preprocess_analysis

COMPILE = True


class PostPreProcessService:
    def __init__(
        self,
        settings: PostPreProcessSettings,
        aioboto3_session: aioboto3.Session,
    ):
        self.settings = settings
        self.aioboto3_session = aioboto3_session
        self.s3_client = boto3.client("s3")  # For synchronous operations
        self._cleanup_paths = []  # Store paths that need cleanup

    def get_cleanup_paths(self) -> List[str]:
        """Return paths that need cleanup after processing."""
        return self._cleanup_paths

    async def post_preprocess(
        self,
        request: PostPreProcessRequest,
    ) -> PostPreProcessResponse:
        self._cleanup_paths = []  # Reset cleanup paths
        user_id = request.user_id
        dataset_name = self.settings.dataset_name
        jsd_threshold = request.jsd_threshold
        emd_threshold = request.emd_threshold
        use_scaled_data = request.use_scaled_data
        pairwise_corr_figures = request.pairwise_corr_figures
        downsample_factor = request.downsample_factor

        post_preprocess_dir = "/tmp"
        os.makedirs(post_preprocess_dir, exist_ok=True)
        preprocessed_data_path = ""
        report_path = os.path.join(
            post_preprocess_dir,
            f"{dataset_name}_post_preprocessing_{'scaled' if use_scaled_data else 'unscaled'}.html",
        )

        try:
            # Get the preprocessed data
            preprocessed_data_path = await self._process_preprocessed_data(
                user_id, dataset_name
            )
            self._cleanup_paths.append(preprocessed_data_path)
            logger.info(
                f"Preprocessed data has been successfully downloaded to {preprocessed_data_path}"
            )

            # Run the CPU-intensive analysis in a thread pool using asyncio.to_thread
            await asyncio.to_thread(
                post_preprocess_analysis,
                dataset_dir=preprocessed_data_path,
                jsd_threshold=jsd_threshold,
                emd_threshold=emd_threshold,
                use_scaled_data=use_scaled_data,
                output_dir=post_preprocess_dir,
                pairwise_corr_figures=pairwise_corr_figures,
                downsample_factor=downsample_factor,
            )

            self._cleanup_paths.append(post_preprocess_dir)
            logger.info(
                f"Post preprocessing has been successfully completed and the artifacts are saved at output path: {post_preprocess_dir}"
            )

            if self.settings.bucket_name != "local":
                # Upload artifacts and get the S3 key
                s3_key = await self._upload_report_to_s3(
                    user_id, dataset_name, report_path
                )

                # Generate presigned URL
                presigned_url = create_presigned_url(
                    self.s3_client,
                    self.settings.bucket_name,
                    s3_key,
                )
                if not presigned_url:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to generate presigned URL",
                    )

                return PostPreProcessResponse(
                    status="success",
                    message="Postpreprocessing completed successfully",
                    presigned_url=presigned_url,
                )
            else:
                # For local development, return the local file path
                return PostPreProcessResponse(
                    status="success",
                    message="Postpreprocessing completed successfully",
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

    async def _process_preprocessed_data(
        self, user_id: str, dataset_name: str
    ) -> str:
        preprocessed_file_names = [
            "train_timeseries.npy",
            "val_timeseries.npy",
            "test_timeseries.npy",
            "train_continuous_conditions.npy",
            "val_continuous_conditions.npy",
            "test_continuous_conditions.npy",
            "train_original_discrete_windows.npy",
            "val_original_discrete_windows.npy",
            "test_original_discrete_windows.npy",
            "timeseries_windows_columns.json",
            "continuous_windows_columns.json",
            "colnames.json",
            "timeseries_scalers.pkl",
            "continuous_scalers.pkl",
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

    async def _upload_report_to_s3(
        self,
        user_id: str,
        dataset_name: str,
        report_path: str,
    ) -> str:
        # Upload the post_preprocessing plots to S3
        s3_key = os.path.join(
            user_id,
            "reports",
            dataset_name,
            os.path.basename(report_path),
        )

        # Upload the post_preprocessing report to S3
        self.s3_client.upload_file(
            report_path,
            self.settings.bucket_name,
            s3_key,
        )

        return s3_key
