import traceback

from fastapi import APIRouter, Depends, HTTPException, Path
from loguru import logger

from synthefy_pkg.app.data_models import (
    DynamicTimeSeriesData,
    SelectedAction,
    SynthefyRequest,
    SynthefyResponse,
)
from synthefy_pkg.app.services.forecast_service import (
    ForecastService,
    ForecastSettings,
)
from synthefy_pkg.app.utils.api_utils import (
    api_key_required,
    convert_dynamic_time_series_data_to_synthefy_request,
    convert_label_tuple_to_discrete_metadata,
    convert_synthefy_response_to_dynamic_time_series_data,
    delete_gt_real_timeseries_windows,
    get_settings,
    save_request,
    save_response,
    trim_response_to_forecast_window,
    update_metadata_from_query,
)

COMPILE = False
router = APIRouter(tags=["Forecast"])


def get_forecast_service(dataset_name: str = Path(...)) -> ForecastService:
    settings: ForecastSettings = get_settings(
        ForecastSettings, dataset_name=dataset_name
    )
    return ForecastService(dataset_name, settings)


@router.post("/api/forecast/{dataset_name}", response_model=SynthefyResponse)
async def forecast(
    request: SynthefyRequest,
    service: ForecastService = Depends(get_forecast_service),
):
    try:
        save_request(request, "forecast", service.settings.json_save_path)
        request = delete_gt_real_timeseries_windows(request)
        request = update_metadata_from_query(request)
        request = convert_label_tuple_to_discrete_metadata(request)
        response = await service.forecast(request, suffix_label="_forecast")
        save_response(response, "forecast", service.settings.json_save_path)
        return response
    except Exception as e:
        logger.error(f"Forecast operation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during forecast operation",
        )


@router.post(
    "/api/forecast/{dataset_name}/stream", response_model=DynamicTimeSeriesData
)
async def forecast_time_series_stream(
    request: DynamicTimeSeriesData,
    _: str = Depends(api_key_required),
    service: ForecastService = Depends(get_forecast_service),
) -> DynamicTimeSeriesData:
    try:
        save_request(
            request, "forecast_stream", service.settings.json_save_path
        )

        synthefy_request = convert_dynamic_time_series_data_to_synthefy_request(
            request,
            service.preprocess_config.get("group_labels", {}).get("cols", []),
            service.preprocess_config.get("timeseries", {}).get("cols", []),
            service.preprocess_config.get("continuous", {}).get("cols", []),
            service.preprocess_config.get("discrete", {}).get("cols", []),
            service.preprocess_config.get("timestamps_col", []),
            service.preprocess_config.get("window_size", None),
            selected_action=SelectedAction.FORECAST,
        )

        synthefy_response = await service.forecast(
            synthefy_request,
            streaming=True,
            true_forecast_with_shifting=True,
            suffix_label="_forecast",
        )
        response = convert_synthefy_response_to_dynamic_time_series_data(
            synthefy_response,
            return_only_synthetic=service.settings.return_only_synthetic_in_streaming_response,
            suffix_label="_forecast",
        )

        # cut it to only include the forecast
        if service.settings.only_include_forecast_in_streaming_response:
            trim_response_to_forecast_window(
                response,
                service.preprocess_config["window_size"],
                service.forecast_length,
            )

        save_response(
            response, "forecast_stream", service.settings.json_save_path
        )

        return response
    except Exception as e:
        logger.error(f"Forecast stream operation failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during forecast stream operation",
        )
