import argparse  # Import argparse
import asyncio
import json
import os
from pathlib import Path

from loguru import logger

from synthefy_pkg.app.config import TrainSettings
from synthefy_pkg.app.data_models import (
    SageMakerTrainingJobStatus,
    TrainListJobsRequest,
    TrainRequest,
    TrainStatusRequest,
    TrainStopRequest,
)
from synthefy_pkg.app.services.train_service import TrainService

COMPILE = True

API_GATEWAY_KEY = os.getenv("API_GATEWAY_KEY")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SageMaker training flow test script."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["training-flow", "list-jobs"],
        help="Type of test to run: 'training-flow' or 'list-jobs'.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Run the full training until completion instead of stopping early",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["dev", "prod"],
        default="dev",
        help="Environment to test against: 'dev' or 'prod'",
    )
    return parser.parse_args()


async def test_sagemaker_training_flow(
    json_path: str,
    test_type: str,
    full_run: bool = False,
    environment: str = "dev",
):
    """
    Manual test to verify the full SageMaker workflow:
    1. Start training OR List training jobs
    2. For training flow:
       - List training jobs
       - Get job status
       - Stop training

    Args:
        json_path: Path to the JSON configuration file
        test_type: Type of test to run ('training-flow' or 'list-jobs')
        full_run: Whether to run the full training until completion
        environment: Environment to test against ('dev' or 'prod')
    """
    if not API_GATEWAY_KEY:
        logger.error("API_GATEWAY_KEY environment variable must be set")
        return

    try:
        # Load the example configuration
        example_path = Path(json_path)
        with open(example_path) as f:
            config_data = json.load(f)

        # Extract user_id from the configuration
        user_id = config_data["config"]["user_id"]

        # Initialize service with non-local bucket to trigger SageMaker path
        settings = TrainSettings(
            bucket_name="synthefy-dev-logs",
            dataset_path="datasets",
            synthesis_config_path="configs/synthesis",
            forecast_config_path="configs/forecast",
            json_save_path="json_files",
        )

        # Create the train service with the specified environment
        train_service = TrainService(settings, environment=environment)
        logger.info(f"Testing against {environment} environment")

        if test_type == "list-jobs":
            # List training jobs
            try:
                logger.info("Listing training jobs...")
                # Force max_results to 1 for minimal load testing
                max_results = 1
                logger.info(f"Using max_results={max_results} to minimize load")

                list_request = TrainListJobsRequest(
                    action="list_training_jobs",
                    client_id=config_data.get("client_id"),
                    status=config_data.get("status"),
                    dataset_name=config_data.get("dataset_name"),
                    max_results=max_results,  # Override with minimal value
                    task=config_data.get("task"),
                )
                list_response = await train_service.list_training_jobs(
                    list_request
                )
                logger.info(f"Found {list_response.total_count} training jobs")
                if list_response.truncated:
                    logger.info("Results were truncated")
                if list_response.timeout:
                    logger.warning(
                        f"Operation timed out after {list_response.execution_time_seconds} seconds"
                    )
            except Exception as e:
                logger.error(f"Failed to list training jobs: {str(e)}")
                return
        else:
            # Start training using the exact payload from the JSON
            try:
                logger.info("Starting training job...")
                train_request = TrainRequest(**config_data)
                train_response = await train_service.train_model(train_request)

                if train_response.status != "success":
                    logger.error(
                        f"Training failed to start: {train_response.message}"
                    )
                    return

                # The training_job_name is in the response message like "Training job started: <job_name>"
                training_job_name = train_response.message.split(": ")[1]
                logger.info(
                    f"Training job started with name: {training_job_name}"
                )

                if full_run:
                    # Add a delay to allow SageMaker to initialize the job
                    logger.info("Waiting 200 seconds for job to initialize...")
                    await asyncio.sleep(200)

                    # Monitor training until completion
                    status_request = TrainStatusRequest(
                        training_job_name=training_job_name, user_id=user_id
                    )
                    while True:
                        try:
                            status_response = (
                                await train_service.get_training_status(
                                    status_request
                                )
                            )
                            status = status_response.status
                            logger.info(f"Current status: {status}")

                            if status in [
                                SageMakerTrainingJobStatus.COMPLETED,
                                SageMakerTrainingJobStatus.FAILED,
                                SageMakerTrainingJobStatus.STOPPED,
                            ]:
                                logger.info(
                                    f"Training finished with status: {status}"
                                )
                                break

                        except Exception as e:
                            logger.warning(f"Error getting status: {str(e)}")
                            logger.info("Will retry in 5 minutes...")

                        # Check status every 5 minutes
                        await asyncio.sleep(300)
                else:
                    # Original behavior: Wait for 180 seconds and then stop
                    logger.info("Waiting 180 seconds...")
                    await asyncio.sleep(180)

                    # Check status
                    logger.info("Checking training status...")
                    try:
                        status_request = TrainStatusRequest(
                            training_job_name=training_job_name, user_id=user_id
                        )
                        status_response = (
                            await train_service.get_training_status(
                                status_request
                            )
                        )
                        logger.info(
                            f"Training status: {status_response.model_dump_json(indent=2)}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to get training status: {e}")

                    # Stop training (runs regardless of status check result)
                    try:
                        logger.info("Stopping training job...")
                        stop_request = TrainStopRequest(
                            training_job_name=training_job_name, user_id=user_id
                        )
                        stop_response = await train_service.stop_training(
                            stop_request
                        )
                        logger.info(
                            f"Stop response: {stop_response.model_dump_json(indent=2)}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to stop training job: {e}")

            except Exception as e:
                if "CapacityError" in str(e):
                    logger.error(
                        f"SageMaker capacity error. Try using a different instance type in your config: {str(e)}"
                    )
                else:
                    logger.error(f"Training flow failed: {str(e)}")
                return

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return


if __name__ == "__main__":
    args = parse_arguments()
    try:
        asyncio.run(
            test_sagemaker_training_flow(
                args.json_path, args.test_type, args.full_run, args.environment
            )
        )
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")

"""
Usage Examples:
--------------
# Required Environment Variables:
export API_GATEWAY_KEY=<your_api_key>
export SYNTHEFY_DATASETS_BASE=/home/ubuntu/data
export SYNTHEFY_PACKAGE_BASE="/home/ubuntu/code/synthefy/synthefy-package"

# Test Training Flow on Dev Environment (default)
# Forecasting:
python src/synthefy_pkg/scripts/sagemaker_test_train.py \
    src/synthefy_pkg/app/tests/test_jsons/sagemaker_twamp_one_month_forecast.json \
    --test-type training-flow

# Test Training Flow on Prod Environment
# Forecasting:
python src/synthefy_pkg/scripts/sagemaker_test_train.py \
    src/synthefy_pkg/app/tests/test_jsons/sagemaker_twamp_one_month_forecast.json \
    --test-type training-flow \
    --environment prod

# Synthesis:
python src/synthefy_pkg/scripts/sagemaker_test_train.py \
    src/synthefy_pkg/app/tests/test_jsons/sagemaker_twamp_one_month_synthesis.json \
    --test-type training-flow \
    --environment dev

# List Training Jobs:
python src/synthefy_pkg/scripts/sagemaker_test_train.py \
    src/synthefy_pkg/app/tests/test_jsons/sagemaker_list_training_jobs.json \
    --test-type list-jobs \
    --environment prod

# Full Training Run:
python src/synthefy_pkg/scripts/sagemaker_test_train.py \
    src/synthefy_pkg/app/tests/test_jsons/sagemaker_twamp_one_month_forecast.json \
    --test-type training-flow \
    --full-run \
    --environment dev
"""
