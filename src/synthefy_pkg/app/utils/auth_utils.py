import os
from typing import Optional

from fastapi import Depends, Form, Header, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from synthefy_pkg.app.dao.user_api_keys import validate_api_key
from synthefy_pkg.app.db import get_db
from synthefy_pkg.app.utils.supabase_utils import get_supabase_user
from synthefy_pkg.utils.licensing_utils import check_dev_mode


class AuthenticationUtils:
    """Utility class for handling authentication across the application."""

    @staticmethod
    async def get_user_id_from_token_or_api_key(
        db: Session,
        authorization: Optional[str],
        x_api_key: Optional[str],
    ) -> Optional[str]:
        """
        Authentication dependency that handles both access tokens and API keys.
        When SYNTHEFY_USE_ACCESS_TOKEN=1:
        - Requires either an access token (Authorization header) or an API key (x-api-key header)
        - If both are present, access token takes precedence
        - If neither is present, raises HTTPException
        When SYNTHEFY_USE_ACCESS_TOKEN is not "1":
        - First tries to get user_id from authorization header (if present)
        - Then tries to get user_id from API key (if present)
        - Then tries to get user_id from form data
        - If none found, returns None
        """
        logger.info("Getting user_id from token or API key")
        use_access_token = os.getenv("SYNTHEFY_USE_ACCESS_TOKEN", "0") == "1"
        if use_access_token:
            if authorization:
                return get_supabase_user(authorization)
            elif x_api_key:
                user_id = validate_api_key(db, x_api_key)
                if user_id:
                    return user_id
                raise HTTPException(status_code=401, detail="Invalid API key")
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Authorization or x-api-key required",
                )
        else:
            # Try API key if authorization header is empty
            if x_api_key:
                user_id = validate_api_key(db, x_api_key)
                if user_id:
                    return user_id

            # If none found, return None
            return None

    @staticmethod
    def validate_user_id_required(
        user_id: Optional[str],
        error_message: str = "user_id is required. Please provide it via authorization header, API key, form data, or request body.",
    ) -> str:
        """
        Validate that user_id is not None and raise appropriate error if it is.

        Args:
            user_id: The user_id to validate
            error_message: Custom error message to display

        Returns:
            The validated user_id

        Raises:
            HTTPException: If user_id is None
        """
        if user_id is None:
            raise HTTPException(
                status_code=400,
                detail=error_message,
            )
        return user_id

    @staticmethod
    def validate_access_token_required(
        user_id: Optional[str],
        error_message: str = "Authorization or x-api-key required",
    ) -> str:
        """
        Validate that user_id is not None for access token mode and raise appropriate error if it is.

        Args:
            user_id: The user_id to validate
            error_message: Custom error message to display

        Returns:
            The validated user_id

        Raises:
            HTTPException: If user_id is None
        """
        if user_id is None:
            raise HTTPException(status_code=401, detail=error_message)
        return user_id

    @staticmethod
    def validate_authentication_headers(
        authorization: Optional[str],
        x_api_key: Optional[str],
        db: Session,
    ) -> None:
        """
        Validate authentication headers for endpoints that don't need user_id but require authentication.

        Args:
            authorization: Authorization header
            x_api_key: API key header
            db: Database session

        Raises:
            HTTPException: If authentication is invalid or missing
        """
        use_access_token = os.getenv("SYNTHEFY_USE_ACCESS_TOKEN", "0") == "1"
        if use_access_token:
            if not authorization and not x_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="Either Authorization header or x-api-key header required when SYNTHEFY_USE_ACCESS_TOKEN=1",
                )

            if authorization:
                # Access token takes precedence
                _ = get_supabase_user(authorization)
            elif x_api_key:
                # Validate API key
                _ = validate_api_key(db, x_api_key)
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Either Authorization header or x-api-key header required when SYNTHEFY_USE_ACCESS_TOKEN=1",
                )


async def get_user_id_from_token_or_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Optional[str]:
    """
    FastAPI dependency for getting user_id from token or API key.
    This function can be reused across different routers to avoid code duplication.

    Returns:
        Optional[str]: The user_id if authentication is successful, None otherwise
    """
    return await AuthenticationUtils.get_user_id_from_token_or_api_key(
        db, authorization, x_api_key
    )
