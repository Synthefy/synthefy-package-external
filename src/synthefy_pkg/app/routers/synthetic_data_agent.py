import traceback
from typing import Optional

import aioboto3
from fastapi import APIRouter, Body, Depends, HTTPException, Path
from loguru import logger
from pydantic import BaseModel

from synthefy_pkg.app.celery_app import celery_app
from synthefy_pkg.app.config import SyntheticDataAgentSettings
from synthefy_pkg.app.data_models import (
    GenerateCombinationsRequest,
    GridCombinations,
    MetaDataGridSample,
    SyntheticDataGenerationRequest,
    WindowFilters,
)
from synthefy_pkg.app.services.synthetic_data_agent_service import (
    SyntheticDataAgentService,
)
from synthefy_pkg.app.utils.api_utils import get_settings
from synthefy_pkg.app.utils.auth_utils import (
    AuthenticationUtils,
    get_user_id_from_token_or_api_key,
)
from synthefy_pkg.app.utils.s3_utils import get_aioboto3_session

router = APIRouter(tags=["Synthetic Data Agent"])


def get_synthetic_data_agent_service(
    user_id: str = Path(...),
    dataset_name: str = Path(...),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> SyntheticDataAgentService:
    settings: SyntheticDataAgentSettings = get_settings(
        SyntheticDataAgentSettings, dataset_name=dataset_name
    )
    return SyntheticDataAgentService(
        user_id, dataset_name, settings, aioboto3_session
    )


@router.get(
    "/api/synthetic_data_agent/metadata_grid_sample/{user_id}/{dataset_name}"
)
async def get_metadata_grid_sample(
    user_id: str = Path(...),
    dataset_name: str = Path(...),
    service: SyntheticDataAgentService = Depends(
        get_synthetic_data_agent_service
    ),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
) -> MetaDataGridSample:
    """
    Retrieve metadata grid sample for a specific dataset.
    """
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        if user_id and user_id != user_id_from_auth_header:
            raise HTTPException(
                status_code=400,
                detail="user_id in path does not match user_id in authorization header",
            )
        user_id = user_id_from_auth_header

    try:
        return await service.get_metadata_grid_sample()
    except Exception as e:
        logger.error(f"Error retrieving metadata grid sample: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/synthetic_data_agent/generate_combinations/{user_id}/{dataset_name}"
)
async def generate_combinations(
    user_id: str = Path(...),
    dataset_name: str = Path(...),
    request: GenerateCombinationsRequest = Body(...),
    service: SyntheticDataAgentService = Depends(
        get_synthetic_data_agent_service
    ),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
) -> GridCombinations:
    """
    Generate all possible combinations of parameters.
    """
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        if user_id and user_id != user_id_from_auth_header:
            raise HTTPException(
                status_code=400,
                detail="user_id in path does not match user_id in authorization header",
            )
        user_id = user_id_from_auth_header

    try:
        return await service.generate_combinations(request)
    except Exception as e:
        logger.error(f"Error generating combinations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/synthetic_data_agent/generate_synthetic_data/{user_id}/{dataset_name}"
)
async def generate_synthetic_data(
    user_id: str = Path(...),
    dataset_name: str = Path(...),
    request: SyntheticDataGenerationRequest = Body(...),
    service: SyntheticDataAgentService = Depends(
        get_synthetic_data_agent_service
    ),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
):
    """
    Generate synthetic data based on provided combinations.
    Returns a task ID for tracking the background job.
    """
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        if user_id and user_id != user_id_from_auth_header:
            raise HTTPException(
                status_code=400,
                detail="user_id in path does not match user_id in authorization header",
            )
        user_id = user_id_from_auth_header

    try:
        return await service.generate_synthetic_data(request)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(
            f"Error generating synthetic data: {str(e)}\n{error_traceback}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/synthetic_data_agent/tasks/{user_id}/{task_id}")
async def get_task_status(
    user_id: str = Path(...),
    task_id: str = Path(...),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
):
    """
    Get simplified task status with progress percentage and estimated time remaining.
    """
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        if user_id and user_id != user_id_from_auth_header:
            raise HTTPException(
                status_code=400,
                detail="user_id in path does not match user_id in authorization header",
            )
        user_id = user_id_from_auth_header

    task = celery_app.AsyncResult(task_id)
    info = task.info or {}

    if task.state == "PENDING":
        response = {
            "state": task.state,
            "progress_percentage": 0,
        }
    elif task.state == "PROGRESS":
        response = {
            "state": task.state,
            "progress_percentage": info.get("progress_percentage", 0),
        }
    elif task.state == "SUCCESS":
        response = {
            "state": task.state,
            "progress_percentage": 100,
            "presigned_url": info.get("presigned_url", None),
            "s3_path": info.get("s3_path", None),
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
