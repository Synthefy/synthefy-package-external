import os
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Path
from loguru import logger

from synthefy_pkg.app.data_models import SynthefyRequest, SynthefyResponse
from synthefy_pkg.app.services.search_service import (
    SearchService,
    SearchSettings,
)
from synthefy_pkg.app.utils.api_utils import (
    convert_label_tuple_to_discrete_metadata,
    delete_gt_real_timeseries_windows,
    get_settings,
    save_request,
    save_response,
)

COMPILE = False
router = APIRouter(tags=["Search"])


# Cache the SearchService instance
@lru_cache()
def get_search_service_with_cache(dataset_name: str) -> SearchService:
    settings: SearchSettings = get_settings(
        SearchSettings, dataset_name=dataset_name
    )
    return SearchService(dataset_name, settings)


def get_search_service_without_cache(dataset_name: str) -> SearchService:
    settings: SearchSettings = get_settings(
        SearchSettings, dataset_name=dataset_name
    )
    return SearchService(dataset_name, settings)


def get_search_service(dataset_name: str = Path(...)) -> SearchService:
    # Disable caching during tests
    if os.getenv("TESTING") == "true":
        logger.info("Disabling cache for search service")
        return get_search_service_without_cache(dataset_name)
    return get_search_service_with_cache(dataset_name)


@router.post("/api/search/{dataset_name}", response_model=SynthefyResponse)
async def search(
    request: SynthefyRequest,
    service: SearchService = Depends(get_search_service),
):
    try:
        save_request(request, "search", service.settings.json_save_path)
        request = convert_label_tuple_to_discrete_metadata(request)
        request = delete_gt_real_timeseries_windows(request)
        response = await service.search(request)
        save_response(response, "search", service.settings.json_save_path)
        return response
    except Exception as e:
        logger.error(f"Search operation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during search operation",
        )
