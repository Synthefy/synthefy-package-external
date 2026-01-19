import json
import os
import time
from typing import Any, Dict, Optional, cast

import boto3
import httpx
import yaml
from fastapi import HTTPException
from loguru import logger
from mergedeep import merge

from synthefy_pkg.app.config import TrainSettings
from synthefy_pkg.app.data_models import (
    TrainListJobsRequest,
    TrainListJobsResponse,
    TrainRequest,
    TrainResponse,
    TrainStatusRequest,
    TrainStatusResponse,
    TrainStopRequest,
    TrainStopResponse,
)
from synthefy_pkg.app.utils.api_utils import get_train_config_file_name
from synthefy_pkg.experiments.forecast_experiment import ForecastExperiment
from synthefy_pkg.experiments.synthesis_experiment import SynthesisExperiment
from synthefy_pkg.utils.train_config_utils import generate_default_train_config

COMPILE = True


# Base API endpoint without environment
API_BASE_ENDPOINT = "https://z8qzt18he3.execute-api.us-east-2.amazonaws.com"
API_GATEWAY_KEY = os.getenv("API_GATEWAY_KEY")

DEFAULT_INSTANCE_TYPE = "ml.p3.2xlarge"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_VOLUME_SIZE = 30
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_FORECAST_LENGTH_MULTIPLIER = 0.5

# from sagemaker/lambda_handler.py -> not in container so have to hardcode.
WRONG_ENVIRONMENT_STOP_ERROR = "Cannot stop this job as it belongs to"


def upload_config_to_s3(
    config_dict: Dict[str, Any],
    dataset_name: str,
    task_type: str,
    user_id: str,
    bucket_name: str,
    job_name: str,
) -> None:
    """Upload configuration to S3.

    Args:
        config_dict: Configuration dictionary to upload
        dataset_name: Name of the dataset
        task_type: Type of task ('forecast' or 'synthesis')
        user_id: User identifier
        bucket_name: Name of the S3 bucket
        job_name: Name of the job (based on the lambda handler which creates the job name)
    """
    s3_client = boto3.client("s3")
    config_file_name = get_train_config_file_name(
        dataset_name=dataset_name, task=task_type, job_name=job_name
    )
    s3_key = f"{user_id}/{dataset_name}/{config_file_name}"

    try:
        config_yaml = yaml.dump(config_dict, indent=2)
        logger.info(
            f"Uploading config to S3: Bucket={bucket_name}, Key={s3_key}"
        )
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=config_yaml)
        logger.info("Config uploaded successfully to S3.")
    except Exception as e:
        logger.error(f"Failed to upload config to S3: {str(e)}")
        raise


class TrainService:
    def __init__(
        self, settings: TrainSettings, environment: Optional[str] = None
    ):
        self.settings = settings
        # Determine environment from bucket_name if not explicitly provided
        if environment is None:
            self.environment = (
                "prod" if "prod" in self.settings.bucket_name else "dev"
            )
        else:
            self.environment = environment
        # Construct the API endpoint based on the environment
        self.api_endpoint = f"{API_BASE_ENDPOINT}/{self.environment}/train"
        logger.info(f"Using API endpoint: {self.api_endpoint}")

    def _save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Persist training configuration to the output directory.

        Args:
            config: Training configuration to save
            config_path: Path where config will be saved
        """
        with open(config_path, "w") as f:
            yaml.dump(config, f, indent=2)

    def _get_default_config(
        self,
        dataset_name: str,
        task: str,
        dataset_dims_dict: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Inputs:
        dataset_name - name of the dataset
        task - task type (forecast or synthesis)
        dataset_dims_dict - dictionary containing the dimensions of the dataset (output from preprocessing). can be optional then we will fill with 0s.

        Returns:
        default_config - default configuration for the training task. Includes model_config as a key since sagemaker requires it.
        """
        default_config = generate_default_train_config(
            task=task,
            dataset_name=dataset_name,
            time_series_length=(
                dataset_dims_dict["time_series_length"]
                if dataset_dims_dict
                else 0
            ),
            num_channels=(
                dataset_dims_dict["num_channels"] if dataset_dims_dict else 0
            ),
            num_discrete_conditions=(
                dataset_dims_dict["num_discrete_conditions"]
                if dataset_dims_dict
                else 0
            ),
            num_continuous_labels=(
                dataset_dims_dict["num_continuous_labels"]
                if dataset_dims_dict
                else 0
            ),
            num_timestamp_labels=(
                dataset_dims_dict["num_timestamp_labels"]
                if dataset_dims_dict
                else 0
            ),
            save_to_examples_dir=False,
        )

        final_config = {
            "device": "cuda",
            "task": task,
            "model_config": {
                "dataset_config": default_config["dataset_config"],
                "denoiser_config": default_config["denoiser_config"],
                "metadata_encoder_config": default_config[
                    "metadata_encoder_config"
                ],
                "execution_config": default_config["execution_config"],
                "training_config": default_config["training_config"],
            },
        }

        return final_config

    async def train_model(
        self,
        request: TrainRequest,
    ) -> TrainResponse:
        """
        Train synthesis OR forecast model using the provided configuration
        """
        task_type = request.config["task"]

        # Get default config and merge with user-provided config - needed for both paths
        dataset_config = await self.download_dataset_config(
            request.dataset_name, request.config["user_id"]
        )
        default_config = self._get_default_config(
            request.dataset_name, task_type, dataset_config
        )
        config_dict = merge(default_config, request.config)
        logger.info(f"training config: {config_dict}")

        if dataset_config:
            # check the dimensions of the dataset config
            for key, expected_dim in dataset_config.items():
                if (
                    config_dict["model_config"]["dataset_config"][key]
                    != expected_dim
                ):
                    raise ValueError(
                        f"Expected {key} to be {expected_dim}, got {config_dict['model_config']['dataset_config'][key]}"
                    )

        request.config = cast(Dict[str, Any], config_dict)

        if self.settings.bucket_name == "local":
            # Local training needs checkpoint management
            # TODO: move this to a training_utils so it can be used for forecast + CLIP
            model_checkpoint_path = None

            try:
                # Local training path
                logger.info(
                    f"Starting local training with config: {config_dict}"
                )

                # Initialize appropriate experiment based on task
                if task_type == "forecast":
                    experiment = ForecastExperiment(
                        config_source=cast(
                            Dict[str, Any], config_dict["model_config"]
                        ),
                    )
                elif task_type == "synthesis":
                    experiment = SynthesisExperiment(
                        config_source=cast(
                            Dict[str, Any], config_dict["model_config"]
                        ),
                    )
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")

                experiment.train(model_checkpoint_path=model_checkpoint_path)

                model_checkpoint_path = os.path.join(
                    experiment.training_runner.save_path, "best_model.ckpt"
                )
                return TrainResponse(
                    status="success",
                    message="Model training completed successfully",
                    training_job_name="",  # not applicable for local training (yet)
                )

            except Exception as e:
                logger.error(f"Error during local model training: {str(e)}")
                raise Exception(f"Failed to train model locally: {str(e)}")

        else:
            # SageMaker training - checkpoint management handled by SageMaker
            try:
                logger.info("Starting SageMaker training job")
                sagemaker_response = await self.train_model_sagemaker(
                    request=request
                )
                logger.info(
                    f"SageMaker training response: {sagemaker_response}"
                )

                if (
                    sagemaker_response.status == "success"
                    and sagemaker_response.training_job_name
                ):
                    try:
                        logger.info(
                            f"Uploading config to S3 for job: {sagemaker_response.training_job_name}"
                        )
                        upload_config_to_s3(
                            config_dict=config_dict["model_config"],
                            dataset_name=request.dataset_name,
                            task_type=task_type,
                            user_id=request.config["user_id"],
                            bucket_name=self.settings.bucket_name,
                            job_name=sagemaker_response.training_job_name,
                        )
                    except Exception as upload_error:
                        logger.warning(
                            f"Failed to upload config to S3 for job {sagemaker_response.training_job_name} after successful start: {upload_error}"
                        )

                return sagemaker_response

            except Exception as e:
                logger.error(f"Error during SageMaker model training: {str(e)}")
                raise Exception(f"Failed to start SageMaker training: {str(e)}")

    async def _make_api_request(
        self, payload: Dict[str, Any], error_prefix: str
    ) -> Dict[str, Any]:
        """Make an API request to the training endpoint."""
        if API_GATEWAY_KEY is None:
            raise ValueError(
                "API_GATEWAY_KEY is not set before sagemaker API call."
            )
        try:
            # Increase timeout to match Lambda's timeout (120 seconds)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120.0)
            ) as client:
                response = await client.post(
                    self.api_endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": API_GATEWAY_KEY,
                    },
                )
                if response.status_code == 200:
                    # Parse the response body
                    response_data = response.json()
                    # API Gateway wraps the response, so we need to parse the body
                    if "body" in response_data:
                        return json.loads(response_data["body"])
                    return response_data
                else:
                    raise Exception(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

        except Exception as e:
            logger.error(f"{error_prefix}: {str(e)}")
            raise Exception(f"{error_prefix}: {str(e)}")

    async def train_model_sagemaker(
        self,
        request: TrainRequest,
        instance_type: str = DEFAULT_INSTANCE_TYPE,
    ) -> TrainResponse:
        """Start a SageMaker training job.

        Args:
            request: Training request with model configuration
            instance_type: SageMaker instance type to use

        Returns:
            Training response with job status
        """
        # Train synthesis or forecast model using SageMaker via API Gateway
        try:
            task = request.config.get("task")
            user_id = request.config.get("user_id")

            model_config = request.config["model_config"]
            dataset_config = model_config["dataset_config"]

            # Add dataset_name to dataset_config
            dataset_config["dataset_name"] = request.dataset_name

            # Prepare the request payload
            payload = {
                "task": task,
                "action": "start_training",
                "user_id": user_id,
                "environment": self.environment,
                "dataset_name": request.dataset_name,
                "model_config": {
                    "dataset_config": dataset_config,
                    "denoiser_config": model_config["denoiser_config"],
                    "metadata_encoder_config": model_config[
                        "metadata_encoder_config"
                    ],
                    "execution_config": model_config["execution_config"],
                    "training_config": model_config["training_config"],
                },
                "resource_config": {
                    "InstanceType": instance_type,
                    "InstanceCount": DEFAULT_INSTANCE_COUNT,
                    "VolumeSizeInGB": DEFAULT_VOLUME_SIZE,
                },
            }

            try:
                response_body = await self._make_api_request(
                    payload, "API request failed"
                )
                logger.info(
                    f"Raw API response: {response_body}"
                )  # Add debug logging

                # Extract training job name from response
                training_job_name = response_body.get("training_job_name")
                if not training_job_name:
                    error_message = "Training job name missing from response."
                    logger.error(
                        f"API response missing training_job_name: {response_body}"
                    )
                    return TrainResponse(
                        status="failed",
                        message=error_message,
                        training_job_name="",  # failure
                    )

                return TrainResponse(
                    status="success",
                    message=f"Training job started: {training_job_name}",
                    training_job_name=training_job_name,
                )

            except Exception as api_error:
                # Specifically handle ResourceLimitExceeded
                if "ResourceLimitExceeded" in str(api_error):
                    error_message = (
                        "Training job could not be started due to AWS resource limits. "
                        "Please wait and try again later, or contact support to increase your service quota."
                    )
                    logger.error(f"Resource Limit Exceeded: {api_error}")
                    return TrainResponse(
                        status="resource_limit",  # Consider adding this as a specific status
                        message=error_message,
                        training_job_name="",  # failure
                    )
                # Re-raise other types of errors
                raise

        except Exception as e:
            logger.error(f"Unexpected error in train_model_sagemaker: {str(e)}")
            return TrainResponse(
                status="failed",
                message=f"An unexpected error occurred: {str(e)}",
                training_job_name="",  # failure
            )

    async def get_training_status(
        self, request: TrainStatusRequest
    ) -> TrainStatusResponse:
        """Get the status of a SageMaker training job via API Gateway"""
        payload = {
            "action": "status",
            "training_job_name": request.training_job_name,
            "user_id": request.user_id,
        }

        body = await self._make_api_request(payload, "Status check failed")
        return TrainStatusResponse(**body)

    async def stop_training(
        self, request: TrainStopRequest
    ) -> TrainStopResponse:
        """Stop a SageMaker training job via API Gateway"""
        payload = {
            "action": "stop_training",
            "training_job_name": request.training_job_name,
            "environment": self.environment,
            "user_id": request.user_id,
        }

        response = await self._make_api_request(payload, "Stop training failed")
        if response.get("error", "").startswith(WRONG_ENVIRONMENT_STOP_ERROR):
            raise HTTPException(status_code=403, detail=response["error"])
        return TrainStopResponse(response=response)

    async def download_dataset_config(
        self, dataset_name: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Download the dataset config from S3 or local storage.
        This will be the dataset requirements (time_series_length, num_channels, num_discrete_conditions, num_continuous_labels, num_timestamp_labels)
        """
        if self.settings.bucket_name == "local":
            # local case
            config_key = f"{dataset_name}/dataset_config_to_update.json"
            # For local storage, use the json_save_path from settings
            config_path = os.path.join(self.settings.dataset_path, config_key)
            logger.info(f"getting configs from: {config_path}")
            try:
                with open(config_path, "r") as f:
                    dataset_config = json.load(f)
                logger.info(
                    f"Retrieved local dataset config for {dataset_name}"
                )
            except FileNotFoundError:
                logger.error(f"Config file not found at: {config_path}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset config not found for {dataset_name}. Make sure preprocessing was completed successfully.",
                )
        else:
            config_key = (
                f"{user_id}/{dataset_name}/dataset_config_to_update.json"
            )
            s3_client = boto3.client("s3")
            try:
                logger.info(
                    f"Looking for dataset config at: s3://{self.settings.bucket_name}/{config_key}"
                )
                response = s3_client.get_object(
                    Bucket=self.settings.bucket_name, Key=config_key
                )
                dataset_config = json.loads(
                    response["Body"].read().decode("utf-8")
                )
                logger.info(f"Retrieved dataset config for {dataset_name}")
            except Exception as e:
                logger.error(f"Error fetching dataset config: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset config not found for {dataset_name}. Make sure preprocessing was completed successfully.",
                )
        return dataset_config

    async def get_train_config(
        self,
        dataset_name: str,
        user_id: str,
        task: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get pre-filled training configuration based on preprocessing results.

        Args:
            dataset_name: Name of the dataset
            user_id: User identifier
            task: Type of task ('forecast' or 'synthesis')
            model_name: Optional name of the training model

        Returns:
            Dict containing the complete training configuration
        """
        dataset_config = await self.download_dataset_config(
            dataset_name, user_id
        )
        # Get default config template
        default_config = self._get_default_config(
            dataset_name=dataset_name,
            task=task,
            dataset_dims_dict=dataset_config,
        )

        default_config = default_config["model_config"]  # only need this.
        logger.info(f"Generated training config for {dataset_name} ({task})")
        return default_config

    async def list_training_jobs(
        self, request: TrainListJobsRequest
    ) -> TrainListJobsResponse:
        """List training jobs with filters."""
        # convert request to dict and add the environment as a key
        request_dict = request.model_dump(exclude_unset=True)
        request_dict["environment"] = self.environment
        request_dict["action"] = (
            "list_training_jobs"  # Explicitly set action for timeout handling
        )

        try:
            logger.info(
                f"Listing training jobs in {self.environment} environment"
            )
            response = await self._make_api_request(
                request_dict, "List training jobs failed"
            )

            # Handle both direct Lambda responses and API Gateway responses
            if isinstance(response, dict):
                if "statusCode" in response:
                    # API Gateway response
                    if response["statusCode"] != 200:
                        raise Exception(
                            f"API request failed with status {response['statusCode']}"
                        )
                    try:
                        body = json.loads(response["body"])
                    except json.JSONDecodeError:
                        logger.error("Failed to parse response body")
                        raise
                else:
                    # Direct Lambda response
                    body = response

                return TrainListJobsResponse(**body)

            # Add a default return for when response is not a dict
            return TrainListJobsResponse(
                training_jobs=[],
                total_count=0,
                truncated=False,
                execution_time_seconds=0,
            )

        except Exception as e:
            logger.error(f"Error in list_training_jobs: {str(e)}")
            raise
