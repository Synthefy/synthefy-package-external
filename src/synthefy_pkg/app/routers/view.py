from functools import lru_cache

from fastapi import APIRouter, Depends, Path

from synthefy_pkg.app.data_models import SynthefyRequest, SynthefyResponse
from synthefy_pkg.app.services.view_service import ViewService, ViewSettings
from synthefy_pkg.app.utils.api_utils import (
    convert_label_tuple_to_discrete_metadata,
    get_settings,
    save_request,
    save_response,
)

COMPILE = False


router = APIRouter(tags=["View"])


@lru_cache()
def get_view_service(dataset_name: str = Path(...)) -> ViewService:
    settings: ViewSettings = get_settings(
        ViewSettings, dataset_name=dataset_name
    )
    return ViewService(settings)


# @lru_cache()
# def get_view_service(dataset_name: str):
#     settings: ViewSettings = get_settings(ViewSettings, dataset_name=dataset_name)
#     return ViewService(settings)


# @router.get("/api/view/default", response_model=ViewRequest)
# async def get_default_view_request(
#     service: ViewService = Depends(get_view_service),
# ):
#     return service.get_default_view_request()


@router.post("/api/view/{dataset_name}", response_model=SynthefyResponse)
async def view_time_series(
    request: SynthefyRequest,
    service: ViewService = Depends(get_view_service),
):
    save_request(request, "view", service.settings.json_save_path)
    request = convert_label_tuple_to_discrete_metadata(request)
    response = service.get_time_series_view(request)
    save_response(response, "view", service.settings.json_save_path)
    return response
