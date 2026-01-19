import json
import os
from typing import Optional

import aioboto3
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from loguru import logger

from synthefy_pkg.app.config import SummarizerSettings
from synthefy_pkg.app.data_models import AWSInfo, DataSource, SummarizeResponse
from synthefy_pkg.app.services.data_retrieval_service import (
    DataRetrievalService,
    get_data_retrieval_service,
)
from synthefy_pkg.app.services.summarize_service import SummarizeService
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
router = APIRouter(tags=["Summarize"])
TMP_DIR = os.path.join(os.getenv("SYNTHEFY_PACKAGE_BASE", ""), "tmp")


async def get_summarize_service(
    user_id: str,
    dataset_name: str,
) -> SummarizeService:
    # Update dataset_name to include user_id
    settings: SummarizerSettings = get_settings(
        SummarizerSettings, dataset_name=f"{dataset_name}_{user_id}"
    )
    return SummarizeService(settings)


@router.post(
    "/api/summarize/{user_id}/{dataset_name}", response_model=SummarizeResponse
)
async def get_data_summary(
    user_id: str,
    dataset_name: str,
    data_source: str = Form(...),
    file: Optional[UploadFile] = File(None),
    s3_source: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
    group_cols: Optional[str] = Form(None),
    skip_plots: bool = Form(default=False),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
    data_retrieval_service: DataRetrievalService = Depends(
        get_data_retrieval_service
    ),
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
):
    """Generate summary statistics and visualizations for the dataset."""

    # Handle authentication and get user_id
    if user_id_from_auth_header:
        if user_id and user_id != user_id_from_auth_header:
            raise HTTPException(
                status_code=400,
                detail="user_id in request body does not match user_id in authorization header",
            )
        user_id = user_id_from_auth_header
    user_id = AuthenticationUtils.validate_user_id_required(user_id)

    # Clean user_id
    user_id = user_id.replace('"', "").replace("'", "")

    # Get summarize service with authenticated user_id
    summarize_service = await get_summarize_service(user_id, dataset_name)

    # Check if dataset exists first
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
        tmp_dir = get_user_tmp_dir(
            user_id, summarize_service.settings.dataset_name
        )

        try:
            # Parse config if provided
            config_dict = json.loads(config) if config else None

            aws_upload_info = AWSInfo(
                bucket_name=summarize_service.settings.bucket_name,
                user_id=user_id,
                dataset_name=summarize_service.settings.dataset_name,
            )

            if data_source == DataSource.UPLOAD and file is not None:
                file_path = handle_file_upload(file, tmp_dir)
            elif data_source == DataSource.S3:
                if not s3_source:
                    raise HTTPException(
                        status_code=400,
                        detail="s3_source configuration is required when source is 's3'",
                    )
                file_path, _ = await handle_s3_source(
                    s3_source, tmp_dir, aioboto3_session
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid data source",
                )
            return await summarize_service.process_file(
                file_path=file_path,
                aws_upload_info=aws_upload_info,
                config_dict=config_dict,
                group_cols=group_cols,
                skip_plots=skip_plots,
                tmp_dir=tmp_dir,
            )
        finally:
            cleanup_tmp_dir(tmp_dir)

    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/summarize/pdf")
async def get_summary_pdf(
    dataset_name: str = Query(
        ..., description="Name of the dataset file (without extension)"
    ),
    user_id_from_auth_header: Optional[str] = Depends(
        get_user_id_from_token_or_api_key
    ),
):
    """Return the generated PDF report as a downloadable file."""

    # Handle authentication and get user_id
    _ = AuthenticationUtils.validate_user_id_required(user_id_from_auth_header)

    json_save_path = TMP_DIR
    pdf_path = os.path.join(json_save_path, f"{dataset_name}_summary.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report not found")

    return FileResponse(
        pdf_path, media_type="application/pdf", filename=f"{dataset_name}.pdf"
    )
