import builtins
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import boto3
import pandas as pd
from fastapi import HTTPException, UploadFile
from loguru import logger

from synthefy_pkg.app.config import PreprocessSettings
from synthefy_pkg.app.data_models import (
    AWSInfo,
    DatasetConfig,
    PreprocessResponse,
    S3Source,
)
from synthefy_pkg.app.utils.s3_utils import upload_directory_to_s3
from synthefy_pkg.data.synthefy_dataset import SynthefyDataset
from synthefy_pkg.preprocessing.preprocess import DataPreprocessor
from synthefy_pkg.utils.memory_monitor import (
    MemoryLimitError,
    generate_memory_error_guidance,
)

COMPILE = True

EMBED_TS_FOR_SEARCH = False


class PreprocessService:
    def __init__(self, settings: PreprocessSettings):
        """Initialize PreprocessService with the given settings.

        Args:
            settings: Configuration settings for preprocessing service
        """
        logger.info("Initializing PreprocessService...")
        start_time = time.time()

        self.settings = settings

        logger.info(
            f"PreprocessService initialized in {time.time() - start_time:.2f}s"
        )

    def create_output_path(self, dataset_name: str) -> str:
        """Generate and create the output directory path for preprocessed data.

        Args:
            dataset_name: Name of the dataset
        Returns:
            str: Path to the output directory
        """
        output_path = os.path.join(self.settings.dataset_path, dataset_name)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def _save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Persist preprocessing configuration to the output directory.

        Args:
            config: Preprocessing configuration to save
            output_path: Directory path where config will be saved
        """
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved preprocessing config to {config_path}")

    def _read_dataframe(self, file_path: str, suffix: str) -> pd.DataFrame:
        """Read a CSV or Parquet file into a pandas DataFrame.

        Args:
            file_path: Path to the input file
            suffix: File extension (.csv or .parquet)

        Returns:
            pd.DataFrame: Loaded DataFrame

        Raises:
            HTTPException: If file type is unsupported or if there are read errors
        """
        try:
            if suffix == ".csv":
                return pd.read_csv(file_path)
            elif suffix == ".parquet":
                return pd.read_parquet(file_path)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file type: {suffix}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error reading {suffix} file: {str(e)}"
            )

    async def _process_dataframe(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        dataset_name: str,
        aws_upload_info: AWSInfo,
        output_path: str,
    ) -> PreprocessResponse:
        """Process a DataFrame using the provided configuration."""
        try:
            config["custom_output_path"] = output_path
            # reset the filename in the preprocessing config to be based on the dataset name
            original_filename = config["filename"].split("/")[-1]
            config["filename"] = os.path.join(dataset_name, original_filename)
            # Env vars are guaranteed; read directly
            if self.settings.bucket_name != "local":
                config["memory_monitoring"] = {
                    "process_memory_mb": int(
                        os.environ["SYNTHEFY_MEMORY_PROCESS_LIMIT_MB"]
                    ),
                    "system_memory_percent": float(
                        os.environ["SYNTHEFY_MEMORY_SYSTEM_LIMIT_PERCENT"]
                    ),
                    "check_interval": float(
                        os.environ["SYNTHEFY_MEMORY_CHECK_INTERVAL"]
                    ),
                }

            # save for uploading to s3
            self._save_config(
                config,
                f"{output_path}/config_{dataset_name}_preprocessing.json",
            )

            # save for local use - TODO: remove this?
            self._save_config(config, self.settings.preprocess_config_path)
            preprocessor = DataPreprocessor(config_source=config)

            dataset_config_dict = preprocessor.process_data(
                input_df=df,
                saved_scalers={},
                saved_encoders={},
                save_files_on=True,
            )

            if not dataset_config_dict:
                raise ValueError(
                    "Preprocessing failed to return dataset configuration"
                )

            # Load the processed windows and perform feature extraction
            dataset = SynthefyDataset(config_source=config)
            dataset.load_windows(window_types=["timeseries"])

            if EMBED_TS_FOR_SEARCH:
                preprocessor.embed_timeseries_for_search(
                    dataset.windows_data_dict["timeseries"]["windows"],
                    preprocessor.encoder_type,
                )

            logger.info(
                f"Preprocessing and feature extraction completed, output path: {preprocessor.output_path}"
            )
            preprocess_response = PreprocessResponse(
                status="success",
                message="Data preprocessing and feature extraction completed",
                output_path=preprocessor.output_path,
                dataset_config=DatasetConfig(**dataset_config_dict),
            )

            # Upload processed files to S3
            if aws_upload_info.bucket_name != "local":
                self._upload_to_s3(output_path, aws_upload_info)

            return preprocess_response

        except (MemoryLimitError, builtins.MemoryError) as e:
            logger.error(
                f"Memory limit exceeded during preprocessing: {str(e)}"
            )
            # Generate user-friendly guidance based on the preprocessing config
            user_friendly_message = generate_memory_error_guidance(
                config, str(e)
            )
            raise HTTPException(
                status_code=413,  # Payload Too Large
                detail=user_friendly_message,
            )
        except Exception as e:
            logger.error(f"Error in _process_dataframe: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error("Error traceback:", exc_info=True)

            raise HTTPException(
                status_code=500, detail=f"Failed to process data: {str(e)}"
            )

    async def process_file(
        self,
        file_path: str,
        config: Dict[str, Any],
        dataset_name: str,
        aws_upload_info: AWSInfo,
    ) -> PreprocessResponse:
        """Process a file according to the provided configuration."""
        try:
            logger.info(f"Processing file: {file_path}")
            output_path = self.create_output_path(dataset_name)

            # Check file size
            file_size = os.path.getsize(file_path)
            await self._check_file_size(file_size, "file")

            suffix = Path(file_path).suffix
            df = self._read_dataframe(file_path, suffix)
            logger.info(f"DataFrame loaded, shape: {df.shape}")

            # Save file to the output_path
            shutil.copy(file_path, output_path)

            return await self._process_dataframe(
                df, config, dataset_name, aws_upload_info, output_path
            )

        except HTTPException:
            # Re-raise HTTPExceptions (like 413 from memory errors) as-is
            raise
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _check_file_size(self, size: int, source: str) -> None:
        """Check if file size exceeds maximum allowed size.

        Args:
            size: File size in bytes
            source: Source of the file (e.g., 'upload', 's3') for error messaging

        Raises:
            HTTPException: If file size exceeds maximum allowed size
        """
        if size > self.settings.max_file_size:
            raise HTTPException(
                status_code=413,  # Request Entity Too Large
                detail=f"File from {source} too large. Maximum size: {self.settings.max_file_size} bytes",
            )

    def _upload_to_s3(self, output_path: str, aws_upload_info: AWSInfo) -> None:
        """Upload preprocessed files to S3 with organized folder structure and cleanup local files."""
        try:
            s3_client = boto3.client("s3")
            s3_prefix = (
                f"{aws_upload_info.user_id}/{aws_upload_info.dataset_name}/"
            )

            if not upload_directory_to_s3(
                s3_client, aws_upload_info.bucket_name, output_path, s3_prefix
            ):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to upload processed files to S3",
                )

        except Exception as e:
            logger.error(f"Error in upload and cleanup process: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload files to S3: {str(e)}",
            )
