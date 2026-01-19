import os

import aioboto3
from fastapi import APIRouter, Depends, HTTPException, Path
from loguru import logger

from synthefy_pkg.app.config import SynthefySettings, SynthesisSettings
from synthefy_pkg.app.data_models import (
    DynamicTimeSeriesData,
    SelectedAction,
    SynthefyRequest,
    SynthefyResponse,
)
from synthefy_pkg.app.routers.setup_ui import download_inference_files_from_s3
from synthefy_pkg.app.services.synthesis_service import SynthesisService
from synthefy_pkg.app.utils.api_utils import (
    api_key_required,
    convert_dynamic_time_series_data_to_synthefy_request,
    convert_label_tuple_to_discrete_metadata,
    convert_synthefy_response_to_dynamic_time_series_data,
    delete_gt_real_timeseries_windows,
    get_settings,
    save_request,
    save_response,
    update_metadata_from_query,
)
from synthefy_pkg.app.utils.s3_utils import get_aioboto3_session

COMPILE = False
router = APIRouter(tags=["Synthesis"])


def get_synthesis_service(dataset_name: str = Path(...)) -> SynthesisService:
    settings: SynthesisSettings = get_settings(
        SynthesisSettings, dataset_name=dataset_name
    )
    return SynthesisService(dataset_name, settings)


def get_current_training_job_id(
    dataset_name: str, task: str = "synthesis"
) -> str | None:
    """
    Get the currently downloaded training job ID for a dataset and task.

    Args:
        dataset_name: Name of the dataset
        task: Task type ('synthesis' or 'forecast')

    Returns:
        The training job ID of currently downloaded files, or None if not found
    """
    try:
        # Use the same path structure as the model paths
        settings = get_settings(SynthefySettings, dataset_name=dataset_name)

        # Determine the model path based on task
        if task == "synthesis":
            model_path = settings.synthesis_model_path
        else:
            model_path = settings.forecast_model_path

        # Create a tracking file path next to the model
        model_dir = os.path.dirname(model_path)
        tracking_file = os.path.join(
            model_dir, f"current_{task}_training_job_id.txt"
        )

        if os.path.exists(tracking_file):
            with open(tracking_file, "r") as f:
                return f.read().strip()
        return None
    except Exception as e:
        logger.warning(f"Failed to get current training job ID: {str(e)}")
        return None


def set_current_training_job_id(
    dataset_name: str, training_job_id: str, task: str = "synthesis"
) -> None:
    """
    Set the currently downloaded training job ID for a dataset and task.

    Args:
        dataset_name: Name of the dataset
        training_job_id: The training job ID to save
        task: Task type ('synthesis' or 'forecast')
    """
    try:
        # Use the same path structure as the model paths
        settings = get_settings(SynthefySettings, dataset_name=dataset_name)

        # Determine the model path based on task
        if task == "synthesis":
            model_path = settings.synthesis_model_path
        else:
            model_path = settings.forecast_model_path

        # Create a tracking file path next to the model
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        tracking_file = os.path.join(
            model_dir, f"current_{task}_training_job_id.txt"
        )

        with open(tracking_file, "w") as f:
            f.write(training_job_id)
        logger.info(
            f"Set current {task} training job ID for {dataset_name}: {training_job_id}"
        )
    except Exception as e:
        logger.warning(f"Failed to set current training job ID: {str(e)}")


@router.post("/api/synthesis/{dataset_name}", response_model=SynthefyResponse)
async def synthesize_time_series(
    request: SynthefyRequest,
    service: SynthesisService = Depends(get_synthesis_service),
) -> SynthefyResponse:
    try:
        save_request(request, "synthesis", service.settings.json_save_path)
        request = delete_gt_real_timeseries_windows(request)
        request = update_metadata_from_query(request)
        request = convert_label_tuple_to_discrete_metadata(request)
        response = await service.get_time_series_synthesis(request)
        save_response(response, "synthesis", service.settings.json_save_path)
        return response
    except Exception as e:
        logger.error(f"Time series synthesis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during time series synthesis",
        )


@router.post(
    "/api/synthesis/{dataset_name}/stream", response_model=DynamicTimeSeriesData
)
async def synthesize_time_series_stream(
    request: DynamicTimeSeriesData,
    _: str = Depends(api_key_required),
    service: SynthesisService = Depends(get_synthesis_service),
) -> DynamicTimeSeriesData:
    try:
        synthefy_request = convert_dynamic_time_series_data_to_synthefy_request(
            request,
            service.preprocess_config.get("group_labels", {}).get("cols", []),
            service.preprocess_config.get("timeseries", {}).get("cols", []),
            service.preprocess_config.get("continuous", {}).get("cols", []),
            service.preprocess_config.get("discrete", {}).get("cols", []),
            service.preprocess_config.get("timestamps_col", []),
            service.preprocess_config.get("window_size", 0),
            selected_action=SelectedAction.SYNTHESIS,
        )
        save_request(
            request, "synthesis_stream", service.settings.json_save_path
        )
        synthefy_response = await service.get_time_series_synthesis(
            synthefy_request, streaming=True
        )
        response = convert_synthefy_response_to_dynamic_time_series_data(
            synthefy_response,
            return_only_synthetic=service.settings.return_only_synthetic_in_streaming_response,
        )
        save_response(
            response, "synthesis_stream", service.settings.json_save_path
        )
        return response
    except Exception as e:
        logger.error(f"Time series stream synthesis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during stream synthesis",
        )


@router.post(
    "/api/synthesis/{userid}/{dataset_name}/{training_job_id}/stream",
    response_model=DynamicTimeSeriesData,
)
async def synthesize_time_series_stream_with_training_job(
    request: DynamicTimeSeriesData,
    userid: str = Path(...),
    dataset_name: str = Path(...),
    training_job_id: str = Path(...),
    _: str = Depends(api_key_required),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> DynamicTimeSeriesData:
    """
    Synthesis endpoint with training job ID that handles model downloading.
    Downloads model files if they are not already available for the specified training job.
    """
    try:
        logger.info(
            f"Starting synthesis with training job ID: {training_job_id}"
        )
        # Check if we already have the correct model files downloaded
        current_training_job_id = get_current_training_job_id(
            dataset_name, "synthesis"
        )

        if current_training_job_id != training_job_id:
            logger.info(
                f"Downloading model files for training job {training_job_id}"
            )

            # Download model files if we don't have the right training job ID
            synthefy_settings = get_settings(
                SynthefySettings, dataset_name=dataset_name
            )
            if synthefy_settings.bucket_name != "local":
                await download_inference_files_from_s3(
                    synthefy_settings,
                    user_id=userid,
                    dataset_name=dataset_name,
                    preprocessed_data_path=synthefy_settings.preprocessed_data_path,
                    aioboto3_session=aioboto3_session,
                    synthesis_training_job_id=training_job_id,
                    forecast_training_job_id=None,
                    synthesis_model_save_path=synthefy_settings.synthesis_model_path,
                    forecast_model_save_path=None,
                    include_npy_files=False,
                )

                # Mark this training job ID as currently downloaded
                set_current_training_job_id(
                    dataset_name, training_job_id, "synthesis"
                )
            else:
                logger.info("Local bucket mode - skipping S3 download")
        else:
            logger.info(
                f"Model files for training job {training_job_id} already available"
            )

        service: SynthesisService = get_synthesis_service(dataset_name)

        # Process the synthesis request using the standard logic
        synthefy_request = convert_dynamic_time_series_data_to_synthefy_request(
            request,
            service.preprocess_config.get("group_labels", {}).get("cols", []),
            service.preprocess_config.get("timeseries", {}).get("cols", []),
            service.preprocess_config.get("continuous", {}).get("cols", []),
            service.preprocess_config.get("discrete", {}).get("cols", []),
            service.preprocess_config.get("timestamps_col", []),
            service.preprocess_config.get("window_size", 0),
            selected_action=SelectedAction.SYNTHESIS,
        )
        save_request(
            request, "synthesis_stream", service.settings.json_save_path
        )
        synthefy_response = await service.get_time_series_synthesis(
            synthefy_request, streaming=True
        )
        response = convert_synthefy_response_to_dynamic_time_series_data(
            synthefy_response,
            return_only_synthetic=service.settings.return_only_synthetic_in_streaming_response,
        )
        save_response(
            response, "synthesis_stream", service.settings.json_save_path
        )
        return response
    except Exception as e:
        logger.error(f"Time series stream synthesis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during stream synthesis",
        )


@router.get("/api/synthesis/{dataset_name}/health")
async def health_check():
    try:
        logger.debug("Health check endpoint called")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during health check",
        )
