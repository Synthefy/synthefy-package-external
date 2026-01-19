import json
from typing import Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Path

from synthefy_pkg.app.config import TrainSettings
from synthefy_pkg.app.data_models import (
    GetTrainConfigRequest,
    GetTrainConfigResponse,
    TrainListJobsRequest,
    TrainListJobsResponse,
    TrainRequest,
    TrainResponse,
    TrainStatusRequest,
    TrainStatusResponse,
    TrainStopRequest,
    TrainStopResponse,
)
from synthefy_pkg.app.services.train_service import TrainService
from synthefy_pkg.app.utils.api_utils import get_settings
from synthefy_pkg.app.utils.auth_utils import (
    AuthenticationUtils,
    get_user_id_from_token_or_api_key,
)

COMPILE = False

router = APIRouter(tags=["Train"])


def get_train_service(dataset_name: str = Path(...)) -> TrainService:
    settings = cast(
        TrainSettings, get_settings(TrainSettings, dataset_name=dataset_name)
    )
    return TrainService(settings=settings)


def get_train_service_without_dataset_name() -> TrainService:
    settings = cast(TrainSettings, get_settings(TrainSettings))
    return TrainService(settings=settings)


@router.post("/api/train/status", response_model=TrainStatusResponse)
async def train_status(
    request: TrainStatusRequest,
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    service: TrainService = Depends(get_train_service_without_dataset_name),
):
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        request.user_id = user_id_from_auth_header
    request.user_id = AuthenticationUtils.validate_user_id_required(
        request.user_id
    )

    return await service.get_training_status(request)


@router.post("/api/train/stop", response_model=TrainStopResponse)
async def stop_training(
    request: TrainStopRequest,
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    service: TrainService = Depends(get_train_service_without_dataset_name),
):
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        request.user_id = user_id_from_auth_header
    request.user_id = AuthenticationUtils.validate_user_id_required(
        request.user_id
    )

    return await service.stop_training(request)


@router.post("/api/train/list", response_model=TrainListJobsResponse)
async def list_training_jobs(
    request: TrainListJobsRequest,
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    service: TrainService = Depends(get_train_service_without_dataset_name),
):
    """List training jobs with optional filters"""

    try:
        return await service.list_training_jobs(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/train/{dataset_name}/config", response_model=GetTrainConfigResponse
)
async def get_train_config(
    dataset_name: str,
    request: GetTrainConfigRequest,
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    service: TrainService = Depends(get_train_service),
):
    """Get pre-filled training configuration based on preprocessing results"""
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        request.user_id = user_id_from_auth_header
    request.user_id = AuthenticationUtils.validate_user_id_required(
        request.user_id
    )

    try:
        config = await service.get_train_config(
            dataset_name=dataset_name,
            user_id=request.user_id,
            task=request.task,
            model_name=request.training_model_name,
        )
        return GetTrainConfigResponse(config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/train/{dataset_name}", response_model=TrainResponse)
async def train(
    request: TrainRequest,
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    service: TrainService = Depends(get_train_service),
):
    # save request to json file
    with open("/tmp/train_request.json", "w") as f:
        json.dump(request.model_dump(), f)
    """Endpoint to train synthesis or forecast model"""
    task = request.config.get("task")

    if task not in ["synthesis", "forecast"]:
        raise HTTPException(
            status_code=400, detail=f"Unsupported task type: {task}"
        )

    return await service.train_model(request)
