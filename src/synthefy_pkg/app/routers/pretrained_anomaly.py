import logging

from fastapi import APIRouter, Depends, HTTPException, Path
from loguru import logger

from synthefy_pkg.app.config import PreTrainedAnomalySettings
from synthefy_pkg.app.data_models import (
    DynamicTimeSeriesData,
    SynthefyRequest,
    SynthefyResponse,
)
from synthefy_pkg.app.services.pretrained_anomaly_service import (
    PreTrainedAnomalyService,
)
from synthefy_pkg.app.utils.api_utils import (
    SelectedAction,
    convert_dynamic_time_series_data_to_synthefy_request,
    convert_label_tuple_to_discrete_metadata,
    convert_synthefy_response_to_dynamic_time_series_data,
    delete_gt_real_timeseries_windows,
    get_settings,
    save_request,
    save_response,
)

COMPILE = False
router = APIRouter(tags=["PreTrained Anomaly Detection"])


def get_pretrained_anomaly_service(
    dataset_name: str = Path(...),
) -> PreTrainedAnomalyService:
    try:
        settings: PreTrainedAnomalySettings = get_settings(
            PreTrainedAnomalySettings, dataset_name=dataset_name
        )
        return PreTrainedAnomalyService(dataset_name, settings)
    except Exception as e:
        logger.error(f"Failed to get pretrained anomaly service: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/api/pretrained_anomaly/{dataset_name}", response_model=SynthefyResponse
)
async def pretrained_anomaly(
    request: SynthefyRequest,
    service: PreTrainedAnomalyService = Depends(get_pretrained_anomaly_service),
):
    try:
        save_request(
            request, "pretrained_anomaly", service.settings.json_save_path
        )
        request = convert_label_tuple_to_discrete_metadata(request)
        request = delete_gt_real_timeseries_windows(request)
        response = service.pretrained_anomaly_detection(request)
        save_response(
            response, "pretrained_anomaly", service.settings.json_save_path
        )
        return response
    except Exception as e:
        logger.error(f"Error processing pretrained anomaly: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/api/pretrained_anomaly/{dataset_name}/stream",
    response_model=DynamicTimeSeriesData,
)
async def pretrained_anomaly_stream(
    request: DynamicTimeSeriesData,
    service: PreTrainedAnomalyService = Depends(get_pretrained_anomaly_service),
) -> DynamicTimeSeriesData:
    try:
        save_request(
            request, "pretrained_anomaly", service.settings.json_save_path
        )

        # Available columns for timeseries. Take intersection of columns in request and preprocess_config timeseries cols
        timeseries_cols = set(request.root.keys()) & set(
            service.preprocess_config.get("timeseries", {}).get("cols", [])
        )
        timeseries_cols = list(timeseries_cols)

        synthefy_request = convert_dynamic_time_series_data_to_synthefy_request(
            request,
            service.preprocess_config.get("group_labels", {}).get("cols", []),
            timeseries_cols,
            service.preprocess_config.get("continuous", {}).get("cols", []),
            service.preprocess_config.get("discrete", {}).get("cols", []),
            service.preprocess_config.get("timestamps_col", []),
            len(request.root[list(request.root.keys())[0]]),
            selected_action=SelectedAction.ANOMALY_DETECTION,
        )

        synthefy_request = convert_label_tuple_to_discrete_metadata(
            synthefy_request
        )
        synthefy_request = delete_gt_real_timeseries_windows(synthefy_request)
        synthefy_response = service.pretrained_anomaly_detection(
            synthefy_request, streaming=True
        )

        response = convert_synthefy_response_to_dynamic_time_series_data(
            synthefy_response, return_only_synthetic=False
        )
        save_response(
            response,
            "pretrained_anomaly_stream",
            service.settings.json_save_path,
        )
        return response
    except Exception as e:
        logger.error(f"Error processing pretrained anomaly stream: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
