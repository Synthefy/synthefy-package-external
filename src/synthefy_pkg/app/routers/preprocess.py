import json
from typing import Optional

import aioboto3
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from loguru import logger
from pydantic import ValidationError
from sqlalchemy.orm import Session

from synthefy_pkg.app.config import PreprocessSettings
from synthefy_pkg.app.data_models import AWSInfo, DataSource, PreprocessResponse
from synthefy_pkg.app.db import get_db
from synthefy_pkg.app.services.data_retrieval_service import (
    DataRetrievalService,
    get_data_retrieval_service,
)
from synthefy_pkg.app.services.preprocess_service import PreprocessService
from synthefy_pkg.app.utils.api_utils import (
    cleanup_tmp_dir,
    get_settings,
    get_user_tmp_dir,
    handle_file_upload,
)
from synthefy_pkg.app.utils.auth_utils import (
    AuthenticationUtils,
    get_user_id_from_token_or_api_key,
)
from synthefy_pkg.app.utils.s3_utils import (
    get_aioboto3_session,
    handle_s3_source,
)

COMPILE = False
router = APIRouter(tags=["Preprocess"])


async def get_preprocess_service(
    request: Request,
) -> PreprocessService:
    user_id = request.path_params["user_id"]
    dataset_name = request.path_params["dataset_name"]
    # Update dataset_name to include user_id since we are storing datasets for
    # for any user_id in the same directory
    # TODO: This should be updated to store datasets in separate directories for
    # each user_id
    settings: PreprocessSettings = get_settings(
        PreprocessSettings, dataset_name=f"{dataset_name}_{user_id}"
    )
    return PreprocessService(settings)


@router.post(
    "/api/preprocess/{user_id}/{dataset_name}",
    response_model=PreprocessResponse,
)
async def preprocess_data(
    user_id: str,
    dataset_name: str,
    data_source: str = Form(...),
    config: str = Form(...),
    file: Optional[UploadFile] = File(None),
    s3_source: Optional[str] = Form(None),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    service: PreprocessService = Depends(get_preprocess_service),
    data_retrieval_service: DataRetrievalService = Depends(
        get_data_retrieval_service
    ),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
):
    """
    Unified endpoint to preprocess data from uploaded file or S3
    """

    # Handle authentication and get user_id
    if user_id_from_auth_header:
        user_id = user_id_from_auth_header
    user_id = AuthenticationUtils.validate_user_id_required(user_id)

    dataset_exists = data_retrieval_service.dataset_exists(
        user_id, dataset_name
    )
    if dataset_exists:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset {dataset_name} already exists for user {user_id}",
        )

    try:
        # Create temporary directory for processing
        tmp_dir = get_user_tmp_dir(user_id, service.settings.dataset_name)

        try:
            config_dict = json.loads(config)
            aws_upload_info = AWSInfo(
                bucket_name=service.settings.bucket_name,
                user_id=user_id,
                dataset_name=service.settings.dataset_name,
            )

            if data_source == DataSource.UPLOAD:
                if file is None:
                    raise HTTPException(
                        status_code=400,
                        detail="file is required when source is 'upload'",
                    )
                file_path = handle_file_upload(file, tmp_dir)
                return await service.process_file(
                    file_path,
                    config_dict,
                    service.settings.dataset_name,
                    aws_upload_info,
                )

            elif data_source == DataSource.S3:
                if not s3_source:
                    raise HTTPException(
                        status_code=400,
                        detail="s3_source configuration is required when source is 's3'",
                    )
                file_path, _ = await handle_s3_source(
                    s3_source, tmp_dir, aioboto3_session
                )
                return await service.process_file(
                    file_path,
                    config_dict,
                    service.settings.dataset_name,
                    aws_upload_info,
                )

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported data source: {data_source}",
                )

        finally:
            cleanup_tmp_dir(tmp_dir)

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format in request"
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions (like 413 from memory errors) as-is
        logger.info(
            f"Returning HTTP error response: {e.status_code} - {e.detail}"
        )
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
