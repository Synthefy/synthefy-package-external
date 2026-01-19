import os
import traceback
from typing import Optional

from fastapi import Depends, Form, Header, HTTPException
from gotrue._sync.gotrue_client import AuthApiError, UserResponse
from loguru import logger
from supabase import create_client

from synthefy_pkg.app.config import SupabaseSettings

# Initialize Supabase client
settings = SupabaseSettings()

client = None


def get_supabase_client(mock_client=None):
    global client

    if mock_client:
        return mock_client

    if client is not None:
        return client
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_API_KEY)
    return client


def get_supabase_user(authorization: str):
    """
    Validates the Supabase session token and retrieves user details.
    :param authorization: The "Authorization" header containing the bearer token.
    :return: The user_id.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    access_token = authorization.split("Bearer ")[1]
    logger.info(f"Access token: {access_token}")

    try:
        # Verify the session token
        response: Optional[UserResponse] = get_supabase_client().auth.get_user(
            access_token
        )
        if not response:
            raise HTTPException(
                status_code=401, detail="Invalid or expired session token"
            )

        if not hasattr(response, "user"):
            raise HTTPException(
                status_code=401, detail="Invalid or expired session token"
            )

        # Return the user's id
        return response.user.id
    except AuthApiError as e:
        # Catch specific Supabase Auth API errors
        logger.warning(
            f"Supabase AuthApiError occurred: Status={e.status}, Message={e.message}, Code={e.code}"
        )
        logger.error(traceback.format_exc())

        # Provide a generic but informative error to the client
        # You can add more specific logic here based on e.status or e.code if needed
        if e.status == 401:
            raise HTTPException(
                status_code=401, detail="Invalid or expired session token."
            )
        elif e.status == 429:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
            )
        else:
            # For other AuthApiErrors, provide a general authentication failure message
            raise HTTPException(
                status_code=401,
                detail="Authentication failed due to a server-side issue.",
            )
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail=e)


def get_supabase_user_dependency(authorization: str = Header(...)):
    """
    FastAPI dependency function for getting user from Supabase token.
    This is used when the function is used as a dependency in endpoint definitions.
    """
    return get_supabase_user(authorization)


def get_user_id_from_token_or_form(
    authorization: Optional[str] = Header(None),
    user_id: Optional[str] = Form(None),
) -> str:
    """
    Conditional dependency that extracts user_id from Supabase access token
    when SYNTHEFY_USE_ACCESS_TOKEN=1 is set, otherwise falls back to form data.

    Args:
        authorization: Optional "Authorization" header containing the bearer token
        user_id: Optional user_id from form data (fallback method)

    Returns:
        The user_id extracted from token or form data

    Raises:
        HTTPException: If token is required but invalid/missing, or if no user_id is provided
    """
    use_access_token = os.getenv("SYNTHEFY_USE_ACCESS_TOKEN", "0") == "1"

    if use_access_token:
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Authorization header required when SYNTHEFY_USE_ACCESS_TOKEN=1",
            )
        return get_supabase_user(authorization)
    else:
        # Fall back to current method (form data)
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required in form data when SYNTHEFY_USE_ACCESS_TOKEN is not set",
            )
        return user_id


def get_user_id_from_token_or_body(
    authorization: Optional[str] = Header(None),
    request_body: Optional[dict] = None,
) -> str:
    """
    Conditional dependency that extracts user_id from Supabase access token
    when SYNTHEFY_USE_ACCESS_TOKEN=1 is set, otherwise extracts from request body.

    This is a helper function that can be used in endpoint functions where
    the request body is already parsed.

    Args:
        authorization: Optional "Authorization" header containing the bearer token
        request_body: Optional request body dict containing user_id

    Returns:
        The user_id extracted from token or request body

    Raises:
        HTTPException: If token is required but invalid/missing, or if no user_id is provided
    """
    use_access_token = os.getenv("SYNTHEFY_USE_ACCESS_TOKEN", "0") == "1"

    if use_access_token:
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Authorization header required when SYNTHEFY_USE_ACCESS_TOKEN=1",
            )
        return get_supabase_user(authorization)
    else:
        # Fall back to current method (request body)
        if not request_body or "user_id" not in request_body:
            raise HTTPException(
                status_code=400,
                detail="user_id is required in request body when SYNTHEFY_USE_ACCESS_TOKEN is not set",
            )
        return request_body["user_id"]
