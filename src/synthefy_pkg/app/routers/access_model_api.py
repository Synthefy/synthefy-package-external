import json
import os
import traceback
from typing import Optional

import aioboto3
from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger

from synthefy_pkg.app.config import SynthefySettings
from synthefy_pkg.app.data_models import (
    ModelAPIAccessRequest,
    ModelAPIAccessResponse,
)
from synthefy_pkg.app.routers.setup_ui import download_inference_files_from_s3
from synthefy_pkg.app.utils.api_utils import get_settings
from synthefy_pkg.app.utils.auth_utils import (
    AuthenticationUtils,
    get_user_id_from_token_or_api_key,
)
from synthefy_pkg.app.utils.s3_utils import (
    acreate_presigned_url,
    download_config_from_s3_async,
    get_aioboto3_session,
)

router = APIRouter(tags=["Model API Access"])
SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")
assert SYNTHEFY_PACKAGE_BASE is not None


def determine_task(
    requested_task: str,
    synthesis_training_job_id: str | None,
    forecast_training_job_id: str | None,
) -> str:
    """
    Determine which task to use based on the requested task and available training job IDs.

    Args:
        requested_task: The task requested by the user ('synthesis' or 'forecast')
        synthesis_training_job_id: Optional ID for synthesis training job
        forecast_training_job_id: Optional ID for forecast training job

    Returns:
        str: The determined task to use

    Raises:
        HTTPException: If invalid task or no training job ID is provided
    """
    if requested_task not in ["synthesis", "forecast"]:
        raise HTTPException(status_code=400, detail="Invalid task provided")

    # Check if any training job ID is provided
    if synthesis_training_job_id is None and forecast_training_job_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"{requested_task.capitalize()} training job ID is required",
        )

    # Use the requested task if its training job is available
    if (
        requested_task == "synthesis"
        and synthesis_training_job_id is not None
        or requested_task == "forecast"
        and forecast_training_job_id is not None
    ):
        return requested_task

    # Otherwise use the available training job
    elif synthesis_training_job_id is not None:
        return "synthesis"
    else:
        return "forecast"


@router.post("/api/access-model-api", response_model=ModelAPIAccessResponse)
async def access_model_api(
    request: ModelAPIAccessRequest = Body(...),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> dict:
    """
    Return formatted Python code for using the synthesis API with the updated endpoint.
    The model downloading is now handled by the synthesis endpoint itself.
    """
    # Handle authentication and get user_id
    if user_id_from_auth_header is not None:
        # Use authenticated user_id from header
        if request.user_id and request.user_id != user_id_from_auth_header:
            raise HTTPException(
                status_code=403,
                detail="User ID in request body does not match authenticated user",
            )
        request.user_id = user_id_from_auth_header
    _ = AuthenticationUtils.validate_user_id_required(request.user_id)

    try:
        logger.info("Starting model access setup")

        settings = get_settings(
            SynthefySettings, dataset_name=request.dataset_name
        )

        # Download preprocess config from S3 if it doesn't exist locally
        config_filename = f"config_{request.dataset_name}_preprocessing.json"

        async with aioboto3_session.client("s3") as async_s3_client:  # type: ignore
            download_success = await download_config_from_s3_async(
                s3_client=async_s3_client,
                bucket=settings.bucket_name,
                user_id=request.user_id,
                dataset_name=request.dataset_name,
                filename=config_filename,
                config_file_path=settings.preprocess_config_path,
                overwrite_if_exists=True,
            )

            if not download_success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download preprocessing config for dataset: {request.dataset_name}",
                )

        with open(settings.preprocess_config_path, "r") as f:
            pre_process_file_config = json.load(f)
            window_size = pre_process_file_config["window_size"]
            group_labels_cols = pre_process_file_config.get(
                "group_labels", {}
            ).get("cols", [])
            timestamps_col = pre_process_file_config.get("timestamps_col", [])

            # Replace the inline code with the helper function call
            task = determine_task(
                requested_task=request.task,
                synthesis_training_job_id=request.synthesis_training_job_id,
                forecast_training_job_id=request.forecast_training_job_id,
            )

            # Determine the training job ID to use in the API URL
            if task == "synthesis" and request.synthesis_training_job_id:
                training_job_id = request.synthesis_training_job_id
            elif task == "forecast" and request.forecast_training_job_id:
                training_job_id = request.forecast_training_job_id
            else:
                # Fallback - this shouldn't happen due to determine_task validation
                training_job_id = (
                    request.synthesis_training_job_id
                    or request.forecast_training_job_id
                )

            with open(
                f"{SYNTHEFY_PACKAGE_BASE}/examples/python_api_example_template.txt",
                "r",
            ) as code:
                python_code_template = code.read()
                python_code_template = python_code_template.format(
                    dataset_name=request.dataset_name,
                    window_size=window_size,
                    group_labels_cols=group_labels_cols,
                    timestamps_col=timestamps_col,
                    task=task,
                    training_job_id=training_job_id,
                    user_id=request.user_id,
                )

            s3_key = str(
                os.path.join(
                    request.user_id,
                    request.dataset_name,
                    pre_process_file_config["filename"].rsplit("/", maxsplit=1)[
                        -1
                    ],
                )
            )
            dataset_link = await acreate_presigned_url(
                aioboto3_session, settings.bucket_name, s3_key
            )

        return {
            "status": "success",
            "formatted_python_code": python_code_template,
            "dataset_link": dataset_link,
        }
    except HTTPException as e:
        logger.error(f"Failed to setup model access: {str(e)}")
        stacktrace = traceback.format_exc()
        logger.error(f"Stacktrace: {stacktrace}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Failed to setup model access: {str(e)}")
        stacktrace = traceback.format_exc()
        logger.error(f"Stacktrace: {stacktrace}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred while setting up model access: {str(e)}",
        )
