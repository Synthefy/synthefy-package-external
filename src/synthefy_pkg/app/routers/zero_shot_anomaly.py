from fastapi import APIRouter, Depends

from synthefy_pkg.app.data_models import (
    ZeroShotAnomalyRequest,
    ZeroShotAnomalyResponse,
)
from synthefy_pkg.app.services.zero_shot_anomaly_service import (
    ZeroShotAnomalyService,
)

router = APIRouter(tags=["Zero Shot Anomaly"])

COMPILE = False


def get_zero_shot_anomaly_service():
    return ZeroShotAnomalyService()


@router.post("/api/zero_shot_anomaly", response_model=ZeroShotAnomalyResponse)
async def zero_shot_anomaly(
    request: ZeroShotAnomalyRequest,
    service: ZeroShotAnomalyService = Depends(get_zero_shot_anomaly_service),
):
    return service.zero_shot_anomaly_detection(request)
