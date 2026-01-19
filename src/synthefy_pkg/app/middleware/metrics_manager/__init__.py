"""
Metrics manager middleware for FastAPI.
"""

import time
from typing import Optional

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from synthefy_pkg.app.middleware.metrics_manager.bigquery import (
    BigQueryMetricsManager,
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to initialize and handle metrics recording."""

    def __init__(self, app, enable_metrics: bool = True):
        super().__init__(app)
        self.enable_metrics = enable_metrics
        self.metrics_manager: Optional[BigQueryMetricsManager] = None

        # Initialize metrics manager if enabled
        if self.enable_metrics:
            self._initialize_manager()

    def _initialize_manager(self) -> None:
        """Initialize the BigQuery metrics manager."""
        try:
            logger.info("Initializing BigQuery metrics manager...")
            self.metrics_manager = BigQueryMetricsManager.initialize()
            logger.info("Metrics manager initialized successfully")
            logger.info(f"Config: {self.metrics_manager.get_config()}")
        except Exception as e:
            logger.warning(f"Metrics manager initialization failed: {e}")
            logger.info("Middleware will continue without metrics")
            self.metrics_manager = None

    async def dispatch(self, request: Request, call_next):
        """Process request and record metrics."""
        # Make metrics manager available on request state
        request.state.metrics_manager = self.metrics_manager

        # Record request start time
        start_time = time.time()

        # Process request
        response: Response = await call_next(request)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Record API usage if manager is available
        if self.metrics_manager and self.metrics_manager.is_ready():
            try:
                # Extract basic info from request
                user_id = getattr(request.state, "user_id", "anonymous")
                api_key = request.headers.get("x-api-key")
                endpoint = f"{request.method} {request.url.path}"
                dataset_name = request.path_params.get("dataset_name")
                correlation_id = request.headers.get("x-correlation-id")

                self.metrics_manager.record_api_usage(
                    user_id=user_id,
                    api_key=api_key,
                    endpoint=endpoint,
                    dataset_name=dataset_name,
                    processing_time_ms=processing_time_ms,
                    status_code=response.status_code,
                    correlation_id=correlation_id,
                )
            except Exception as e:
                logger.debug(f"Failed to record API usage: {e}")

        return response
