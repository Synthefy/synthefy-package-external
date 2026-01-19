import os
from typing import List

import aioboto3
from loguru import logger

from synthefy_pkg.app.config import SyntheticDataAgentSettings
from synthefy_pkg.app.utils.api_utils import cleanup_local_directories
from synthefy_pkg.app.utils.s3_utils import (
    download_config_from_s3_async,
    download_model_from_s3_async,
    download_preprocessed_data_from_s3_async,
    download_training_config_from_s3_async,
)


def get_files_to_download(split: str) -> List[str]:
    """Get the list of files that need to be downloaded for a given split."""
    return [
        f"{split}_timeseries.npy",
        f"{split}_timestamps_original.npy",
        f"{split}_original_discrete_windows.npy",
        f"{split}_continuous_conditions.npy",
        # original_text_conditions is optional for backwards compatibility - try download separately
        # f"{request.split_type.value}_original_text_conditions.npy",
        "encoders_dict.pkl",
        "timeseries_scalers.pkl",
        "continuous_scalers.pkl",
        "labels_description.pkl",
        "colnames.json",
    ]


async def ensure_preprocessed_data_downloaded(
    settings: SyntheticDataAgentSettings,
    user_id: str,
    dataset_name: str,
    aioboto3_session: aioboto3.Session,
    files_to_download: List[str],
) -> None:
    """Ensure all required preprocessed data files are downloaded."""
    if settings.bucket_name != "local":
        async with aioboto3_session.client("s3") as async_s3_client:  # pyright: ignore
            if not await download_preprocessed_data_from_s3_async(
                async_s3_client,
                settings.bucket_name,
                user_id,
                dataset_name,
                settings.preprocessed_data_path,
                files_to_download,
            ):
                cleanup_local_directories([settings.preprocessed_data_path])
                raise Exception(
                    f"Failed to download preprocessed data for dataset: {dataset_name}"
                )

            # Download the original_text_conditions file separately (optional for backwards compatibility)
            if not await download_preprocessed_data_from_s3_async(
                async_s3_client,
                settings.bucket_name,
                user_id,
                dataset_name,
                settings.preprocessed_data_path,
                ["original_text_conditions.npy"],
            ):
                logger.warning(
                    f"Failed to download original_text_conditions for dataset: {dataset_name}"
                )

            # Download preprocessing config
            if not await download_config_from_s3_async(
                s3_client=async_s3_client,
                bucket=settings.bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                filename=f"config_{dataset_name}_preprocessing.json",
                config_file_path=settings.preprocess_config_path,
            ):
                cleanup_local_directories([settings.preprocessed_data_path])
                raise Exception(
                    f"Failed to download preprocessing config for dataset: {dataset_name}"
                )
    else:
        logger.info(f"{settings.preprocessed_data_path}")
        logger.info(files_to_download)
        if not all(
            os.path.exists(
                os.path.join(
                    settings.preprocessed_data_path,
                    file,
                )
            )
            for file in files_to_download
        ):
            raise Exception(
                f"Preprocessed data for dataset: {dataset_name} not found in local path"
            )


async def ensure_synthesis_model_downloaded(
    settings: SyntheticDataAgentSettings,
    user_id: str,
    dataset_name: str,
    aioboto3_session: aioboto3.Session,
    training_job_id: str,
) -> None:
    """Ensure synthesis model and config are downloaded for the given training job ID."""
    if settings.bucket_name != "local":
        async with aioboto3_session.client("s3") as async_s3_client:  # pyright: ignore
            # Download synthesis config file
            if not await download_training_config_from_s3_async(
                s3_client=async_s3_client,
                bucket=settings.bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                task_type="synthesis",
                config_file_path=settings.synthesis_config_path,
                training_job_id=training_job_id,
                overwrite_if_exists=True,
            ):
                cleanup_local_directories([settings.synthesis_config_path])
                raise Exception(
                    f"Failed to download synthesis config for dataset: {dataset_name}"
                )

            # Download the model checkpoint
            if not await download_model_from_s3_async(
                s3_client=async_s3_client,
                bucket=settings.bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                model_save_path=settings.synthesis_model_path,
                training_job_id=training_job_id,
            ):
                raise Exception(
                    f"Failed to download the model checkpoint for dataset: {dataset_name}"
                )
    else:
        if not os.path.exists(settings.synthesis_model_path):
            raise Exception(
                f"Model checkpoint for dataset: {dataset_name} not found in local path"
            )
