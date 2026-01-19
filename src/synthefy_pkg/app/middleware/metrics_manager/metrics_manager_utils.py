"""
Utility functions for metrics manager operations.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from loguru import logger

from synthefy_pkg.app.utils.supabase_utils import get_supabase_user


def extract_user_id_from_dataset_name(
    dataset_name: Optional[str],
) -> Optional[str]:
    """Extract user ID (UUID) from the dataset name if present.

    Dataset names often follow a pattern with UUID at the end like:
    'rrest_april_2_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0'

    Args:
        dataset_name: The dataset name to extract user ID from

    Returns:
        The user ID if it can be extracted, None otherwise
    """
    if not dataset_name:
        return None

    try:
        # Look for UUID pattern at the end of the dataset name
        uuid_pattern = r".*_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
        uuid_match = re.match(uuid_pattern, dataset_name)

        if uuid_match:
            user_id = uuid_match.group(1)
            # Validate that this is actually a valid UUID format
            try:
                uuid_obj = uuid.UUID(user_id)
                return str(uuid_obj)
            except ValueError:
                logger.warning(
                    f"Extracted string is not a valid UUID: {user_id}"
                )
                return None
        else:
            logger.debug(
                f"No UUID pattern found in dataset name: {dataset_name}"
            )
            return None
    except Exception as e:
        logger.error(f"Error extracting user ID from dataset name: {str(e)}")
        return None


async def extract_user_id_and_dataset_name(
    request: Request,
    path: str,
    api_key: Optional[str],
    dataset_name: Optional[str],
    validate_api_key_func,
    get_supabase_user_func,
    extract_form_data_func,
    extract_user_id_from_request_body_func,
) -> Tuple[str, Optional[str]]:
    """Extract user_id and dataset_name from multiple sources in order of priority.

    This function consolidates all the user_id extraction logic into a single,
    cleaner function that tries different sources in order of priority.

    Args:
        request: The FastAPI request object
        path: The request path
        api_key: The API key from headers
        dataset_name: The dataset name if already extracted
        validate_api_key_func: Function to validate API key
        get_supabase_user_func: Function to get user from Supabase token
        extract_form_data_func: Function to extract form data
        extract_user_id_from_request_body_func: Function to extract user_id from request body

    Returns:
        Tuple of (user_id, dataset_name) where user_id is never None (falls back to "anonymous")
    """
    user_id = None
    final_dataset_name = dataset_name

    # Priority 1: API Key authentication
    if api_key and os.environ.get("API_KEY_AUTH_ENABLED", "false") == "true":
        x_api_key = request.headers.get("X-API-Key")
        if x_api_key:
            try:
                user_id = validate_api_key_func(x_api_key)
                if user_id:
                    logger.debug(
                        f"API call from API key authenticated user_id: {user_id}"
                    )
            except Exception as e:
                logger.debug(f"Could not validate API key: {str(e)}")

    # Priority 2: Bearer token authentication
    if not user_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                user_id = get_supabase_user_func(auth_header)
                if user_id:
                    logger.debug(
                        f"API call from Bearer token authenticated user_id: {user_id}"
                    )
            except Exception as e:
                logger.debug(f"Could not extract user_id from token: {str(e)}")

    # Priority 3: Dataset name extraction
    if not user_id and dataset_name:
        user_id = extract_user_id_from_dataset_name(dataset_name)
        if user_id:
            logger.debug(f"API call from dataset-derived user_id: {user_id}")

    # Priority 4: Form data extraction (upload endpoint)
    if path == "/api/foundation_models/upload":
        form_data = await extract_form_data_func(request)
        if form_data:
            # Always extract dataset name from form data for upload endpoint
            if not final_dataset_name:
                filename = form_data.get("filename")
                if filename:
                    # Remove file extension from filename to get dataset name
                    final_dataset_name = (
                        filename.rsplit(".", 1)[0]
                        if "." in filename
                        else filename
                    )
            # Extract user_id only if not already determined
            if not user_id:
                user_id = form_data.get("user_id")
                if user_id:
                    logger.debug(f"API call from form data user_id: {user_id}")

    # Priority 5: Request body extraction (backtest and forecast endpoints)
    if not user_id and path in [
        "/api/foundation_models/forecast/backtest",
        "/api/foundation_models/forecast",
    ]:
        user_id = await extract_user_id_from_request_body_func(request)
        if user_id:
            logger.debug(f"API call from request body user_id: {user_id}")

    # Fallback to anonymous
    if not user_id:
        user_id = "anonymous"
        logger.debug("Could not determine user ID, using anonymous")

    return user_id, final_dataset_name


def track_metadata_usage(
    metadata_info_combined: Optional[List],
    metadata_dataframes: List,
) -> Dict[str, Any]:
    """
    Track metadata usage for forecasting endpoints.

    This function calculates metadata usage statistics for weather and haver metadata.
    It compares the requested metadata with the actually loaded metadata to determine success/failure.

    Args:
        metadata_info_combined: List of requested metadata info objects
        metadata_dataframes: List of actually loaded metadata dataframes

    Returns:
        Dictionary containing metadata usage statistics that can be included in forecast_execution events
    """
    if not metadata_info_combined:
        return {}

    # Track each metadata type separately
    weather_metadata_requests = 0
    weather_metadata_successes = 0
    haver_metadata_requests = 0
    haver_metadata_successes = 0

    # Count requested metadata by type
    for metadata_info in metadata_info_combined:
        if hasattr(metadata_info, "data_source"):
            if metadata_info.data_source == "weather":
                weather_metadata_requests += 1
            elif metadata_info.data_source == "haver":
                haver_metadata_requests += 1

    # Count successful metadata loads by checking metadata_dataframes
    # We need to count unique metadata requests, not individual dataframes
    # For weather: multiple dataframes can come from one request (one per weather parameter)
    # For haver: one dataframe per request

    # Track unique successful requests by their metadata info
    successful_weather_requests = set()
    successful_haver_requests = set()

    # Group metadata dataframes by their source request
    for metadata_df in metadata_dataframes:
        if hasattr(metadata_df, "metadata_json") and metadata_df.metadata_json:
            data_source = metadata_df.metadata_json.get("data_source")

            if data_source == "weather":
                # For weather, we need to identify the unique request
                # We can use the description or location info to identify unique requests
                description = metadata_df.metadata_json.get("columns", [{}])[
                    0
                ].get("title", "")
                if description:
                    # Extract location name from title (e.g., "Temperature - New York" -> "New York")
                    location_name = (
                        description.split(" - ")[-1]
                        if " - " in description
                        else description
                    )
                    successful_weather_requests.add(location_name)

            elif data_source == "haver":
                # For haver, each dataframe represents one unique request
                # We can use the description to identify unique requests
                description = metadata_df.metadata_json.get("columns", [{}])[
                    0
                ].get("title", "")
                if description:
                    successful_haver_requests.add(description)

    weather_metadata_successes = len(successful_weather_requests)
    haver_metadata_successes = len(successful_haver_requests)

    # Return metadata usage statistics for inclusion in forecast_execution events
    return {
        "weather_requests": weather_metadata_requests,
        "weather_successes": weather_metadata_successes,
        "weather_failed": weather_metadata_requests
        - weather_metadata_successes,
        "haver_requests": haver_metadata_requests,
        "haver_successes": haver_metadata_successes,
        "haver_failed": haver_metadata_requests - haver_metadata_successes,
    }
