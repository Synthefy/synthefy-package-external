import aioboto3
from fastapi import APIRouter, Depends, Path

from synthefy_pkg.app.config import PostPreProcessSettings
from synthefy_pkg.app.data_models import (
    PostPreProcessRequest,
    PostPreProcessResponse,
)
from synthefy_pkg.app.services.post_preprocess_service import (
    PostPreProcessService,
)
from synthefy_pkg.app.utils.api_utils import (
    cleanup_local_directories,
    get_settings,
    save_request,
    save_response,
)
from synthefy_pkg.app.utils.s3_utils import get_aioboto3_session

COMPILE = False
router = APIRouter(tags=["Post Preprocess"])


def get_post_preprocess_service(
    dataset_name: str = Path(...),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> PostPreProcessService:
    settings: PostPreProcessSettings = get_settings(  # type: ignore
        PostPreProcessSettings, dataset_name=dataset_name
    )
    return PostPreProcessService(settings, aioboto3_session)


@router.post("/api/postpreprocess/html/{dataset_name}")
async def post_preprocess(
    request: PostPreProcessRequest,
    service: PostPreProcessService = Depends(get_post_preprocess_service),
) -> PostPreProcessResponse:
    try:
        save_request(
            request,  # type: ignore
            "post_preprocess_html",
            service.settings.json_save_path,
        )
        response = await service.post_preprocess(request=request)
        save_response(
            response,  # type: ignore
            "post_preprocess_html",
            service.settings.json_save_path,
        )
        return response
    except Exception as e:
        error_response = PostPreProcessResponse(
            status="failure",
            message=f"Postpreprocessing HTML report generation failed. {e}",
            presigned_url=None,
        )
        save_response(
            error_response,  # type: ignore
            "post_preprocess_html",
            service.settings.json_save_path,
        )
        raise e
    finally:
        if service.settings.bucket_name != "local":
            cleanup_local_directories(service.get_cleanup_paths())
