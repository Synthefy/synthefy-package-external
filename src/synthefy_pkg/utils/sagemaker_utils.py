import datetime
import json
import os
from typing import Optional

import boto3
from loguru import logger

COMPILE = True
# SageMaker paths and configuration
SAGEMAKER_ROOT = os.environ.get("SM_ROOT", "/opt/ml")
SAGEMAKER_PATHS = {
    "MODEL_DIR": os.environ.get("SM_MODEL_DIR", f"{SAGEMAKER_ROOT}/model"),
    "CONFIG_DIR": f"{SAGEMAKER_ROOT}/input/config",
    "HYPERPARAMETERS_FILE": f"{SAGEMAKER_ROOT}/input/config/hyperparameters.json",
    "TRAIN_DATA_DIR": os.environ.get(
        "SM_CHANNEL_TRAIN", f"{SAGEMAKER_ROOT}/input/data"
    ),
    "OUTPUT_DIR": os.environ.get(
        "SYNTHEFY_DATASETS_BASE", f"{SAGEMAKER_ROOT}/output"
    ),
    "LOG_DIR": os.environ.get(
        "SYNTHEFY_DATASETS_BASE", f"{SAGEMAKER_ROOT}/output/logs"
    ),
    "RUNS_DIR": os.environ.get(
        "SYNTHEFY_DATASETS_BASE", f"{SAGEMAKER_ROOT}/output/runs"
    ),
}

# CloudWatch configuration
CLOUDWATCH_NAMESPACE = "aws/sagemaker/TrainingJobs"
DEFAULT_AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")


class SageMakerMetricsLogger:
    """Handles additional CloudWatch logging when running in SageMaker"""

    def __init__(self):
        self._cloudwatch = None
        self.training_job_name = self._get_training_job_name()

    def _get_training_job_name(self) -> str:
        try:
            with open(SAGEMAKER_PATHS["HYPERPARAMETERS_FILE"], "r") as f:
                hyperparameters = json.load(f)
                return hyperparameters.get("training_job_name", "unknown")
        except Exception as e:
            logger.error(f"Failed to read training job name: {e}")
            return "unknown"

    @property
    def cloudwatch(self):
        if self._cloudwatch is None:
            try:
                region = os.environ.get("AWS_REGION", DEFAULT_AWS_REGION)
                self._cloudwatch = boto3.client(
                    "cloudwatch", region_name=region
                )
                logger.info(f"Initialized CloudWatch client in region {region}")
            except Exception as e:
                logger.error(f"Failed to initialize CloudWatch client: {e}")
                self._cloudwatch = None
        return self._cloudwatch

    def log_metric(self, metric_name: str, value: float) -> None:
        """
        Logs a metric to CloudWatch.
        Note - this only logs epoch-end metrics.
        see timeseries_decoder_forecasting_trainer.py and diffusion_model.py the code to call this.
        """

        if not self.cloudwatch:
            return

        try:
            self.cloudwatch.put_metric_data(
                Namespace=CLOUDWATCH_NAMESPACE,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": float(value),
                        "Unit": "None",
                        "Timestamp": datetime.datetime.now(
                            datetime.timezone.utc
                        ),
                        "Dimensions": [
                            {
                                "Name": "TrainingJobName",
                                "Value": self.training_job_name,
                            }
                        ],
                    }
                ],
            )
            logger.info(f"Emitted metric {metric_name}: {value} to CloudWatch")
        except Exception as e:
            logger.error(f"Failed to log to CloudWatch: {e}")


def is_running_in_sagemaker() -> bool:
    """Check if code is running in SageMaker environment"""
    return os.path.exists(SAGEMAKER_ROOT)


def get_sagemaker_logger() -> Optional[SageMakerMetricsLogger]:
    """Get SageMaker logger if running in SageMaker, else None"""
    if is_running_in_sagemaker():
        return SageMakerMetricsLogger()
    return None


def get_sagemaker_save_dir(base_save_dir: str) -> str:
    """Get the appropriate save directory when running in SageMaker."""
    if not is_running_in_sagemaker():
        return base_save_dir

    logger.info(
        f"Running in SageMaker - saving synthetic data to {SAGEMAKER_PATHS['MODEL_DIR']}"
    )
    return SAGEMAKER_PATHS["MODEL_DIR"]


def get_sagemaker_data_dir(original_data_path: str) -> str:
    """Get the appropriate data directory when running in SageMaker."""
    if not is_running_in_sagemaker():
        return original_data_path

    return SAGEMAKER_PATHS["TRAIN_DATA_DIR"]
