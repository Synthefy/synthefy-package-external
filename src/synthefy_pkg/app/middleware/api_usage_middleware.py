import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from synthefy_pkg.app.dao.user_api_keys import validate_api_key
from synthefy_pkg.app.db import SessionLocal

# Import centralized API endpoints
from synthefy_pkg.app.middleware.api_endpoints import APIEndpoints, APIEventType

# Import database and API key validation functions
from synthefy_pkg.app.middleware.metrics_manager.metrics_manager_utils import (
    extract_user_id_and_dataset_name,
    extract_user_id_from_dataset_name,
)
from synthefy_pkg.app.utils.supabase_utils import get_supabase_user

# Import only the usage logger, not the API utils with database dependencies
from synthefy_pkg.app.utils.usage_logger import log_api_usage_async


class APIUsageMiddleware(BaseHTTPMiddleware):
    """Middleware to track API usage for billing and analytics purposes."""

    def __init__(self, app):
        super().__init__(app)
        logger.info("API Usage Tracking Middleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip tracking for non-billable endpoints
        if not self._should_track_endpoint(path):
            logger.info(f"Skipping tracking for non-billable endpoint: {path}")
            return await call_next(request)

        start_time = time.time()

        # Extract API key from headers (support both X-API-Key and Authorization headers)
        api_key = self._extract_api_key_from_headers(request.headers)

        # Extract dataset name from URL path
        dataset_name = self._extract_dataset_name(path)

        # Extract user_id from URL path for endpoints that have it in the path
        user_id_from_path = self._extract_user_id_from_path(path)

        # For backtest and forecast endpoints, try to extract dataset name from request body
        if not dataset_name and (
            APIEndpoints.is_foundation_models_forecast(path)
            or APIEndpoints.is_foundation_models_backtest(path)
        ):
            dataset_name = await self._extract_dataset_name_from_request_body(
                request
            )

        # Extract user_id and dataset_name from multiple sources
        user_id, dataset_name = await extract_user_id_and_dataset_name(
            request=request,
            path=path,
            api_key=api_key,
            dataset_name=dataset_name,
            validate_api_key_func=self._validate_api_key,
            get_supabase_user_func=get_supabase_user,
            extract_form_data_func=self._extract_form_data,
            extract_user_id_from_request_body_func=self._extract_user_id_from_request_body,
        )

        # If we extracted user_id from path and no user_id was found through other means, use the path user_id
        if user_id_from_path and user_id == "anonymous":
            user_id = user_id_from_path
            logger.debug(f"Using user_id from path: {user_id}")

        # For dataset creation endpoints, construct the compound dataset name
        if dataset_name and user_id and user_id != "anonymous":
            if APIEndpoints.is_dataset_creation_endpoint(path):
                # For dataset creation endpoints, the actual dataset name is {dataset_name}_{user_id}
                compound_dataset_name = f"{dataset_name}_{user_id}"
                logger.debug(
                    f"Constructed compound dataset name: {compound_dataset_name} from {dataset_name} and {user_id}"
                )
                dataset_name = compound_dataset_name

        # Set user_id on request.state for other middleware to use
        request.state.user_id = user_id

        # Process the request
        response = await call_next(request)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Determine if response was successful for billing purposes
        is_successful_response = response.status_code == 200

        # Check if endpoint should be billed (only synthesis, backtest, and generate_synthetic_data)
        should_bill = APIEndpoints.should_bill_endpoint(path)

        # Get BigQuery dataset_id to check if we're in prod environment
        metrics_manager = getattr(request.state, "metrics_manager", None)
        is_prod_environment = False
        if metrics_manager:
            config = metrics_manager.get_config()
            is_prod_environment = config.get("dataset_id") == "metrics_prod"

        # Log API usage to Supabase (billing) - ONLY for successful responses AND billable endpoints AND NOT prod
        if is_successful_response and should_bill and not is_prod_environment:
            try:
                # Using the async version to avoid blocking
                await log_api_usage_async(
                    user_id=user_id,
                    api_key=api_key,  # Will be masked/hashed in logs
                    endpoint=path,
                    dataset_name=dataset_name,
                    processing_time_ms=processing_time_ms,
                    status_code=response.status_code,
                )
            except Exception as e:
                logger.error(f"Failed to log API usage to Supabase: {str(e)}")
        elif is_successful_response and should_bill and is_prod_environment:
            logger.debug(
                f"Skipping Supabase billing log for prod environment (dataset_id=metrics_prod): {path}"
            )
        elif is_successful_response and not should_bill:
            logger.debug(
                f"Skipping Supabase billing log for non-billable endpoint: {path}"
            )
        else:
            logger.debug(
                f"Skipping Supabase billing log for failed request: {response.status_code}"
            )

        # Extract backtest and forecast details early if needed
        backtest_details = {}
        forecast_details = {}

        if APIEndpoints.is_foundation_models_backtest(path) and dataset_name:
            backtest_details = (
                await self._extract_backtest_details_from_request_body(request)
            )
        elif APIEndpoints.is_foundation_models_forecast(path) and dataset_name:
            forecast_details = (
                await self._extract_backtest_details_from_request_body(request)
            )

        # Log to BigQuery (analytics) - ALL responses (successful and failed)
        self._log_api_usage(
            request,
            user_id,
            path,
            processing_time_ms,
            response.status_code,
            dataset_name,
            backtest_details,
            forecast_details,
        )

        return response

    def _validate_api_key(self, x_api_key: str) -> Optional[str]:
        """Validate API key and return user_id if valid."""
        try:
            db = SessionLocal()
            try:
                user_id = validate_api_key(db, x_api_key)
                return user_id
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Could not validate API key: {str(e)}")
            return None

    def _extract_api_key_from_headers(self, headers) -> Optional[str]:
        """Extract API key from request headers, supporting multiple formats."""
        # Try X-API-Key header first (preferred method)
        api_key = headers.get("X-API-Key")
        if api_key:
            return api_key

        # Fall back to Authorization header
        auth_header = headers.get("Authorization", "")

        # Handle Bearer token format
        if auth_header.startswith("Bearer "):
            return auth_header.replace("Bearer ", "")

        return None

    def _should_track_endpoint(self, path: str) -> bool:
        """Check if the endpoint should be tracked for billing."""
        return APIEndpoints.should_track_endpoint(path)

    def _extract_dataset_name(self, path: str) -> Optional[str]:
        """Extract dataset name from URL path if present."""
        return APIEndpoints.extract_dataset_name(path)

    def _extract_user_id_from_path(self, path: str) -> Optional[str]:
        """Extract user_id from URL path if present."""
        return APIEndpoints.extract_user_id_from_path(path)

    async def _extract_form_data(
        self, request: Request
    ) -> Optional[Dict[str, str]]:
        """Extract user_id and filename from form data for upload endpoint.

        Returns:
            Dictionary with user_id and filename if found, None otherwise
        """
        try:
            # Check if content type is multipart/form-data
            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" not in content_type:
                return None

            # Store the original body
            original_body = await request.body()

            # Create a new request with the same body so the endpoint can still read it
            request._body = original_body

            # Parse the form data to extract user_id and filename
            body_str = original_body.decode("utf-8", errors="ignore")

            result = {}

            # Look for user_id in the form data
            import re

            user_id_match = re.search(
                r'name="user_id"\r?\n\r?\n([^\r\n]+)', body_str
            )
            if user_id_match:
                result["user_id"] = str(user_id_match.group(1))

            # Look for filename in the form data
            filename_match = re.search(r'filename="([^"]+)"', body_str)
            if filename_match:
                result["filename"] = filename_match.group(1)

            if result:
                return result
            else:
                logger.debug("No form data found")
                return None

        except Exception as e:
            logger.debug(f"Error extracting form data: {str(e)}")
            return None

    async def _extract_user_id_from_request_body(
        self, request: Request
    ) -> Optional[str]:
        """Extract user_id from request body for backtest and forecast endpoints.

        Returns:
            The user_id if found in request body, None otherwise
        """
        try:
            # Check if content type is application/json
            content_type = request.headers.get("content-type", "")
            if "application/json" not in content_type:
                return None

            # Read the request body
            body = await request.json()

            # Extract user_id from the request body
            if isinstance(body, dict) and "user_id" in body:
                user_id = body.get("user_id")
                if user_id:
                    return str(user_id)

            logger.debug("No user_id found in request body")
            return None

        except Exception as e:
            logger.debug(
                f"Error extracting user_id from request body: {str(e)}"
            )
            return None

    async def _extract_dataset_name_from_request_body(
        self, request: Request
    ) -> Optional[str]:
        """Extract dataset name from request body for backtest and forecast endpoints.

        Returns:
            The dataset name if found in request body, None otherwise
        """
        try:
            # Check if content type is application/json
            content_type = request.headers.get("content-type", "")
            if "application/json" not in content_type:
                return None

            # Read the request body
            body = await request.json()

            # Extract dataset name from config.file_path_key
            if isinstance(body, dict) and "config" in body:
                config = body.get("config", {})
                if isinstance(config, dict) and "file_path_key" in config:
                    file_path_key = config.get("file_path_key")
                    if file_path_key:
                        # Extract dataset name by looking for "foundation_models/" in the path
                        # Examples:
                        # Backtest: "user_id/foundation_models/different_values/different_values.parquet"
                        # Forecast: "user_id/foundation_models/different_values/feature_A=1/agg=sum/data.parquet"
                        # Forecast: "user_id/foundation_models/different_values/unfiltered_agg=sum/data.parquet"
                        if "foundation_models/" in file_path_key:
                            # Split by "foundation_models/" and take the part after it
                            parts = file_path_key.split("foundation_models/")
                            if len(parts) > 1:
                                # Take the first path segment after "foundation_models/"
                                remaining_path = parts[1]
                                if remaining_path:  # Check if there's actually a path after "foundation_models/"
                                    dataset_name = remaining_path.split("/")[0]
                                    if (
                                        dataset_name
                                    ):  # Check if dataset name is not empty
                                        return dataset_name

            logger.debug("No dataset name found in request body")
            return None

        except Exception as e:
            logger.debug(
                f"Error extracting dataset name from request body: {str(e)}"
            )
            return None

    async def _extract_backtest_details_from_request_body(
        self, request: Request
    ) -> Dict[str, Any]:
        """Extract backtest details from request body for backtest endpoint.

        Returns:
            Dictionary containing model_type and forecast_length if found
        """
        try:
            # Check if content type is application/json
            content_type = request.headers.get("content-type", "")
            if "application/json" not in content_type:
                return {}

            # Read the request body
            body = await request.json()

            details = {}

            # Extract details from config
            if isinstance(body, dict) and "config" in body:
                config = body.get("config", {})
                if isinstance(config, dict):
                    details["model_type"] = config.get("model_type")
                    details["forecast_length"] = config.get("forecast_length")

            return details

        except Exception as e:
            logger.debug(
                f"Error extracting backtest details from request body: {str(e)}"
            )
            return {}

    def _log_api_usage(
        self,
        request: Request,
        user_id: str,
        endpoint: str,
        processing_time_ms: float,
        status_code: int,
        dataset_name: Optional[str] = None,
        backtest_details: Optional[Dict[str, Any]] = None,
        forecast_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log request to BigQuery for analytics purposes - tracks ALL responses (successful and failed)."""
        try:
            metrics_manager = getattr(request.state, "metrics_manager", None)
            if not metrics_manager:
                return

            correlation_id = getattr(
                request.state,
                "correlation_id",
                f"req_{int(time.time() * 1000)}",
            )

            # Determine event type and extra payload based on endpoint and success status
            event_type, extra_payload = self._determine_event_type_and_payload(
                endpoint=endpoint,
                dataset_name=dataset_name,
                status_code=status_code,
                request=request,
                backtest_details=backtest_details,
                forecast_details=forecast_details,
                user_id=user_id,
            )

            # Single call to record_api_usage with all necessary arguments
            metrics_manager.record_api_usage(
                user_id=user_id,
                api_key=None,  # Not available in this context
                endpoint=endpoint,
                dataset_name=dataset_name,
                processing_time_ms=processing_time_ms,
                status_code=status_code,
                correlation_id=correlation_id,
                event_type=event_type,
                extra_payload=extra_payload,
            )

        except Exception:
            # Silent fail to avoid interfering with request processing
            pass

    def _determine_event_type_and_payload(
        self,
        endpoint: str,
        dataset_name: Optional[str],
        status_code: int,
        request: Request,
        backtest_details: Optional[Dict[str, Any]] = None,
        forecast_details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Determine event type and extra payload based on endpoint and conditions."""

        # Default values for general API requests
        event_type = APIEventType.API_REQUEST.value
        extra_payload = {
            "level": "INFO" if status_code == 200 else "ERROR",
            "message": f"API request: {request.method} {endpoint}",
            "method": request.method,
            "module": "api_usage_middleware",
        }

        # Get endpoint configuration from centralized API endpoints for all status codes
        endpoint_config = APIEndpoints.get_endpoint_config(endpoint)

        if endpoint_config and endpoint_config.event_type:
            event_type = endpoint_config.event_type

            # Add operation if available
            if endpoint_config.operation:
                extra_payload["operation"] = endpoint_config.operation

            # Special handling for forecast and backtest endpoints (only for successful responses)
            if status_code == 200:
                if (
                    APIEndpoints.is_foundation_models_forecast(endpoint)
                    and forecast_details
                ):
                    model_type = forecast_details.get("model_type")
                    forecast_length = forecast_details.get("forecast_length")
                    if model_type:
                        extra_payload["model_type"] = str(model_type)
                    if forecast_length:
                        extra_payload["forecast_length"] = str(forecast_length)

                    # Add metadata usage information if available
                    metadata_usage_stats = getattr(
                        request.state, "metadata_usage_stats", None
                    )
                    if metadata_usage_stats:
                        extra_payload["weather_success"] = (
                            metadata_usage_stats.get("weather_successes", 0)
                        )
                        extra_payload["weather_failed"] = (
                            metadata_usage_stats.get("weather_failed", 0)
                        )
                        extra_payload["haver_success"] = (
                            metadata_usage_stats.get("haver_successes", 0)
                        )
                        extra_payload["haver_failed"] = (
                            metadata_usage_stats.get("haver_failed", 0)
                        )

                elif (
                    APIEndpoints.is_foundation_models_backtest(endpoint)
                    and backtest_details
                ):
                    model_type = backtest_details.get("model_type")
                    forecast_length = backtest_details.get("forecast_length")
                    if model_type:
                        extra_payload["model_type"] = str(model_type)
                    if forecast_length:
                        extra_payload["forecast_length"] = str(forecast_length)

                # Special handling for API key management endpoints
                elif APIEndpoints.is_api_key_management_endpoint(endpoint):
                    api_key_details = getattr(
                        request.state, "api_key_details", None
                    )
                    logger.debug(
                        f"API key management endpoint detected: {endpoint}"
                    )
                    logger.debug(
                        f"API key details from request.state: {api_key_details}"
                    )
                    # Add API key details to the payload
                    if api_key_details:
                        for key, value in api_key_details.items():
                            if value is not None:
                                extra_payload[key] = str(value)
                                logger.debug(
                                    f"Added to payload: {key} = {value}"
                                )
                    else:
                        logger.debug(
                            "No API key details found in request.state"
                        )

        return event_type, extra_payload
