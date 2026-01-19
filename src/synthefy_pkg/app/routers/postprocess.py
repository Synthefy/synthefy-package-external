from typing import Optional

import aioboto3
from fastapi import APIRouter, Depends, HTTPException, Path

from synthefy_pkg.app.config import PostprocessSettings
from synthefy_pkg.app.data_models import PostprocessRequest, PostprocessResponse
from synthefy_pkg.app.services.postprocess_service import PostprocessService
from synthefy_pkg.app.utils.api_utils import (
    cleanup_local_directories,
    get_settings,
    save_request,
    save_response,
)
from synthefy_pkg.app.utils.auth_utils import (
    AuthenticationUtils,
    get_user_id_from_token_or_api_key,
)
from synthefy_pkg.app.utils.s3_utils import get_aioboto3_session

COMPILE = False
router = APIRouter(tags=["Postprocess Report"])


def get_postprocess_service(
    dataset_name: str = Path(...),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> PostprocessService:
    settings: PostprocessSettings = get_settings(  # type: ignore
        PostprocessSettings, dataset_name=dataset_name
    )  # type: ignore
    return PostprocessService(settings, aioboto3_session)


@router.post("/api/postprocess/html/{dataset_name}")
async def postprocess(
    request: PostprocessRequest,
    service: PostprocessService = Depends(get_postprocess_service),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
) -> PostprocessResponse:
    # Handle authentication and get user_id
    if user_id_from_auth_header:
        if request.user_id and request.user_id != user_id_from_auth_header:
            raise HTTPException(
                status_code=400,
                detail="user_id in request body does not match user_id in authorization header",
            )
        request.user_id = user_id_from_auth_header
    request.user_id = AuthenticationUtils.validate_user_id_required(
        request.user_id
    )

    try:
        save_request(
            request, "postprocess_html", service.settings.json_save_path
        )
        response = await service.postprocess(request=request)
        save_response(
            response,
            "postprocess_html",
            service.settings.json_save_path,
        )
        return response
    except Exception as e:
        error_response = PostprocessResponse(
            status="failure",
            message=f"Postprocessing HTML report generation failed. {e}",
            presigned_url=None,
        )
        save_response(
            error_response,
            "postprocess_html",
            service.settings.json_save_path,
        )
        raise e
    finally:
        if service.settings.bucket_name != "local":
            cleanup_local_directories(service.get_cleanup_paths())
