from fastapi import APIRouter, Depends, HTTPException, Path
from loguru import logger

from synthefy_pkg.app.config import PreTrainedAnomalySettings
from synthefy_pkg.app.data_models import (
    DynamicTimeSeriesData,
    PreTrainedAnomalyV2Request,
    PreTrainedAnomalyV2Response,
)
from synthefy_pkg.app.services.pretrained_anomaly_v2_service import (
    PreTrainedAnomalyV2Service,
)
from synthefy_pkg.app.utils.api_utils import (
    get_settings,
    save_request,
    save_response,
)

COMPILE = False
router = APIRouter(tags=["PreTrained Anomaly Detection"])


def get_pretrained_anomaly_v2_service(
    dataset_name: str = Path(...),
) -> PreTrainedAnomalyV2Service:
    settings: PreTrainedAnomalySettings = get_settings(
        PreTrainedAnomalySettings, dataset_name=dataset_name
    )
    return PreTrainedAnomalyV2Service(settings)


@router.post(
    "/api/v2/anomaly_detection/{dataset_name}",
    response_model=PreTrainedAnomalyV2Response,
)
async def anomaly_detection_v2(
    request: PreTrainedAnomalyV2Request,
    service: PreTrainedAnomalyV2Service = Depends(
        get_pretrained_anomaly_v2_service
    ),
) -> PreTrainedAnomalyV2Response:
    """
    Unified endpoint to preprocess data from uploaded file or S3
    """
    try:
        save_request(
            request, "anomaly_detection_v2", service.settings.json_save_path
        )
        response = service.pretrained_anomaly_detection(request)
        save_response(
            response, "anomaly_detection_v2", service.settings.json_save_path
        )
        return response
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during anomaly detection",
        )


@router.post(
    "/api/v2/anomaly_detection/{dataset_name}/stream",
    response_model=PreTrainedAnomalyV2Response,
)
async def anomaly_detection_v2_stream(
    request: DynamicTimeSeriesData,
    service: PreTrainedAnomalyV2Service = Depends(
        get_pretrained_anomaly_v2_service
    ),
) -> PreTrainedAnomalyV2Response:
    """
    Unified endpoint to preprocess data from uploaded file or S3
    """
    try:
        save_request(
            request,
            "anomaly_detection_v2_stream",
            service.settings.json_save_path,
        )
        response = service.pretrained_anomaly_detection(request)
        save_response(
            response,
            "anomaly_detection_v2_stream",
            service.settings.json_save_path,
        )
        return response
    except Exception as e:
        logger.error(f"Streaming anomaly detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during streaming anomaly detection",
        )
