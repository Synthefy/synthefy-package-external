from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from sqlalchemy.orm import Session

from synthefy_pkg.app.dao.user_api_keys import (
    delete_api_key,
    generate_api_key,
    get_api_keys,
    save_api_key,
)
from synthefy_pkg.app.db import get_db
from synthefy_pkg.app.schemas.use_api_keys import (
    UserAPIKeyCreateRequest,
    UserAPIKeyCreateResponse,
    UserAPIKeyDeleteResponse,
    UserAPIKeyResponse,
)
from synthefy_pkg.app.utils.supabase_utils import get_supabase_user_dependency

COMPILE = False

router = APIRouter(tags=["User API Keys"])


@router.post("/api/user_api_key/", response_model=UserAPIKeyCreateResponse)
def create_api_key(
    request: UserAPIKeyCreateRequest,
    fastapi_request: Request,
    user_id: str = Depends(get_supabase_user_dependency),
    db: Session = Depends(get_db),
):
    logger.info("API key creation requested for user_id: {}", user_id)
    try:
        api_key = generate_api_key()
        api_key, name, api_key_id = save_api_key(
            db, user_id, request.name, api_key
        )
        logger.info(
            "API key created successfully for user_id: {} with key_id: {}",
            user_id,
            api_key_id,
        )
        # Store API key details in request state for middleware tracking
        if fastapi_request:
            api_key_details = {
                "generated_api_key_id": api_key_id,
                "api_key_name": name,
                "operation": "api_key_created",
                "api_key": api_key,
            }
            fastapi_request.state.api_key_details = api_key_details
            logger.debug(
                f"Set API key details in request.state: {api_key_details}"
            )
        else:
            logger.debug(
                "Could not set API key details - fastapi_request is None"
            )

        return UserAPIKeyCreateResponse(
            id=api_key_id, name=name, api_key=api_key
        )
    except RuntimeError as e:
        logger.error(
            "Failed to create API key for user_id: {}. Error: {}",
            user_id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key. Please try again later.",
        )


@router.delete(
    "/api/user_api_key/{api_key_id}", response_model=UserAPIKeyDeleteResponse
)
def remove_api_key(
    api_key_id: int,
    fastapi_request: Request,
    user_id: str = Depends(get_supabase_user_dependency),
    db: Session = Depends(get_db),
):
    logger.info(
        "API key deletion requested for user_id: {} and key_id: {}",
        user_id,
        api_key_id,
    )
    try:
        if not delete_api_key(db, user_id, api_key_id):
            logger.warning(
                "API key with ID: {} not found for user_id: {}",
                api_key_id,
                user_id,
            )
            raise HTTPException(status_code=404, detail="API key not found")
        logger.info(
            "API key with ID: {} deleted successfully for user_id: {}",
            api_key_id,
            user_id,
        )

        # Store API key details in request state for middleware tracking
        if fastapi_request:
            api_key_details = {
                "deleted_api_key_id": api_key_id,
                "operation": "api_key_deleted",
            }
            fastapi_request.state.api_key_details = api_key_details
            logger.debug(
                f"Set API key details in request.state: {api_key_details}"
            )
        else:
            logger.debug(
                "Could not set API key details - fastapi_request is None"
            )

        return UserAPIKeyDeleteResponse(message="API key deleted successfully")
    except RuntimeError as e:
        logger.error(
            "Failed to delete API key with ID: {} for user_id: {}. Error: {}",
            api_key_id,
            user_id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key. Please try again later.",
        )


@router.get("/api/user_api_keys/", response_model=list[UserAPIKeyResponse])
def get_user_api_keys(
    fastapi_request: Request,
    user_id: str = Depends(get_supabase_user_dependency),
    db: Session = Depends(get_db),
):
    logger.info("API keys retrieval requested for user_id: {}", user_id)
    try:
        api_keys = get_api_keys(db, user_id)
        logger.info(
            "Retrieved {} API keys for user_id: {}", len(api_keys), user_id
        )

        # Store API key details in request state for middleware tracking
        if fastapi_request:
            api_key_details = {
                "api_keys_count": len(api_keys),
                "operation": "api_keys_listed",
            }
            fastapi_request.state.api_key_details = api_key_details
            logger.debug(
                f"Set API key details in request.state: {api_key_details}"
            )
        else:
            logger.debug(
                "Could not set API key details - fastapi_request is None"
            )

        return api_keys
    except RuntimeError as e:
        logger.error(
            "Failed to retrieve API keys for user_id: {}. Error: {}",
            user_id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API keys. Please try again later.",
        )
