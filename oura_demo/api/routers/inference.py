"""
Inference router - Endpoints for synthesis model inference.

Uses real model inference by default. Set USE_MOCK_SYNTHESIS=true to use mock
mode for UI development (returns original values + 10%).

Requires:
- synthefy_pkg installed
- Model checkpoint (set via SYNTHESIS_MODEL_PATH env var)
- Scalers/encoders saved for the dataset
"""

import os
from typing import Union

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from models import (
    ModelType,
    SynthesisRequest,
    SynthesisResponse,
    TaskType,
)
from services.config_loader import get_config_loader
from services.demo_synthesis_service import (
    DemoSynthesisService,
    MockForecastService,
    MockSynthesisService,
    get_mock_synthesis_service,
    get_synthesis_service,
)

router = APIRouter(prefix="/api", tags=["Inference"])

# Environment variable to control mock mode (defaults to False for real inference)
USE_MOCK_SYNTHESIS = os.getenv("USE_MOCK_SYNTHESIS", "false").lower() == "true"


def _get_service(
    dataset_name: str, model_type: str, task_type: str, use_mock: bool
) -> Union[DemoSynthesisService, MockSynthesisService, MockForecastService]:
    """Get the appropriate synthesis service.

    Args:
        dataset_name: Name of the dataset
        model_type: Model type ('standard' or 'flexible')
        task_type: Task type ('synthesis' or 'forecast')
        use_mock: Whether to use mock service

    Returns:
        DemoSynthesisService, MockSynthesisService, or MockForecastService
    """
    if use_mock:
        if task_type == "forecast":
            logger.info("Using MockForecastService")
        else:
            logger.info("Using MockSynthesisService")
        return get_mock_synthesis_service(dataset_name, task_type=task_type)
    else:
        logger.info(
            f"Using DemoSynthesisService (real model, type={model_type}, task={task_type})"
        )
        return get_synthesis_service(
            dataset_name, model_type=model_type, task_type=task_type
        )


@router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize(
    request: SynthesisRequest,
    mock: bool = Query(
        default=None,
        description="Override mock mode. If not provided, uses USE_MOCK_SYNTHESIS env var.",
    ),
) -> SynthesisResponse:
    """Generate synthetic time series from input data.

    Takes input data (1 window) and generates synthetic time series
    using the synthesis model for the specified dataset.

    Uses real model inference by default. Set mock=true to use mock mode
    (returns original values + 10%) for UI development.

    Args:
        request: SynthesisRequest with dataset_name and data
        mock: Optional override for mock mode (default: use real inference)

    Returns:
        SynthesisResponse with original and synthetic timeseries

    Raises:
        HTTPException: If synthesis fails
    """
    dataset_name = request.dataset_name.value
    model_type = request.model_type.value
    task_type = request.task_type.value
    logger.info(
        f"Synthesis request for dataset: {dataset_name}, model_type: {model_type}, task_type: {task_type}"
    )
    logger.info(
        f"Input data shape: {request.data.num_rows} rows x {len(request.data.column_names)} cols"
    )

    # Determine if we should use mock
    use_mock = mock if mock is not None else USE_MOCK_SYNTHESIS
    logger.info(f"Mock mode: {use_mock}")

    # Get config info
    try:
        loader = get_config_loader(dataset_name)
        window_size = loader.get_window_size()
        num_channels = loader.get_num_channels()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate input data length
    if request.data.num_rows != window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Input data has {request.data.num_rows} rows but expected {window_size} (window_size)",
        )

    # Get synthesis service (mock or real)
    try:
        service = _get_service(dataset_name, model_type, task_type, use_mock)
    except Exception as e:
        logger.exception("Failed to initialize synthesis service")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize synthesis service: {str(e)}",
        )

    # Extract original timeseries
    original_timeseries = service.get_original_timeseries(request.data)

    # Run synthesis
    try:
        # Pass num_samples, ground_truth_prefix_length, and forecast_length if the service supports it
        # For forecast task, always pass forecast_length (use request value)
        effective_forecast_length = None
        if request.task_type == TaskType.FORECAST:
            effective_forecast_length = request.forecast_length
            logger.info(
                f"Starting forecast with num_samples={request.num_samples}, "
                f"forecast_length={effective_forecast_length} (from request.forecast_length={request.forecast_length})"
            )
        else:
            logger.info(
                f"Starting synthesis with num_samples={request.num_samples}, "
                f"ground_truth_prefix_length={request.ground_truth_prefix_length}"
            )
        if isinstance(service, (DemoSynthesisService, MockForecastService)):
            synthetic_timeseries = service.generate(
                request.data,
                num_samples=request.num_samples,
                ground_truth_prefix_length=request.ground_truth_prefix_length,
                forecast_length=effective_forecast_length,
            )
        else:
            # MockSynthesisService doesn't support these params
            synthetic_timeseries = service.generate(request.data)
    except ValueError as e:
        logger.error(f"Synthesis ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Synthesis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}",
        )

    logger.info(f"Generated {len(synthetic_timeseries)} synthetic timeseries")

    # Determine forecast_horizon for response - use the actual value that was used
    if request.task_type == TaskType.FORECAST:
        forecast_horizon = request.forecast_length
        logger.info(f"Response forecast_horizon set to {forecast_horizon} (from request.forecast_length={request.forecast_length})")
    else:
        forecast_horizon = 96  # Not used for synthesis, but included for consistency

    return SynthesisResponse(
        original_timeseries=original_timeseries,
        synthetic_timeseries=synthetic_timeseries,
        window_size=window_size,
        num_channels=num_channels,
        dataset_name=request.dataset_name,
        model_type=request.model_type,
        task_type=request.task_type,
        forecast_horizon=forecast_horizon,
    )
