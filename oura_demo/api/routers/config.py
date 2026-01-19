"""
Config router - Endpoints for config loading and dataset info.
"""

from typing import List

from fastapi import APIRouter, HTTPException
from models import ConfigResponse, DatasetName, RequiredColumns
from services.config_loader import (
    DATASET_CONFIG_MAPPING,
    get_config_loader,
)

router = APIRouter(prefix="/api/config", tags=["Config"])


@router.get("/datasets", response_model=List[str])
async def list_datasets() -> List[str]:
    """List all available dataset names.

    Returns:
        List of dataset names that can be used with other endpoints
    """
    return list(DATASET_CONFIG_MAPPING.keys())


@router.get("/{dataset_name}", response_model=ConfigResponse)
async def get_config(dataset_name: DatasetName) -> ConfigResponse:
    """Get configuration info for a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g., oura, oura_subset, ppg)

    Returns:
        ConfigResponse with required columns, window size, etc.

    Raises:
        HTTPException: If dataset_name is invalid
    """
    try:
        loader = get_config_loader(dataset_name.value)

        return ConfigResponse(
            dataset_name=dataset_name,
            required_columns=loader.get_required_columns(),
            window_size=loader.get_window_size(),
            num_channels=loader.get_num_channels(),
            available_datasets=list(DATASET_CONFIG_MAPPING.keys()),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Config file not found: {e}",
        )
