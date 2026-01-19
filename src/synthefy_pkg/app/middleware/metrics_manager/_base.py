"""
Simple base metrics manager interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from synthefy_pkg.app.middleware.api_endpoints import APIEventType


class BaseMetricsManager(ABC):
    """Simple abstract base class for metrics managers."""

    @abstractmethod
    def record_api_usage(
        self,
        user_id: str,
        api_key: Optional[str],
        endpoint: str,
        dataset_name: Optional[str],
        processing_time_ms: float,
        status_code: int,
        correlation_id: Optional[str] = None,
        event_type: str = APIEventType.API_CALL.value,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record API usage metrics."""
        pass

    @abstractmethod
    def record_metric(self, event_data: Dict[str, Any]) -> None:
        """Record a custom metric event."""
        pass
