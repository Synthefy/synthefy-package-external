"""
Centralized API endpoint definitions and metadata.

This module provides a unified way to define API endpoints and their associated metadata,
including tracking patterns, extraction patterns, and event types.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class APIEventType(Enum):
    """Enumeration of API event types for consistent categorization in usage tracking."""

    # General API events (fallback)
    API_REQUEST = "api_request"
    API_CALL = "api_call"  # Legacy compatibility

    # User authentication events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"

    # Business event types
    DATASET_UPLOAD = "dataset_upload"
    DATASET_CREATION = "dataset_creation"
    FORECAST_EXECUTION = "forecast_execution"
    STREAM_FORECAST_EXECUTION = "fm_stream_forecast_execution"
    STREAM_FORECAST_EXECUTION_V2 = "fm_stream_forecast_execution_v2"
    BACKTEST_EXECUTION = "backtest_execution"
    TRAINING_JOB_CREATION = "training_job_creation"
    POSTPROCESSING_REPORT_CREATION = "postprocessing_report_creation"
    SYNTHETIC_DATA_GENERATION = "synthetic_data_generation"
    SYNTHESIS_REQUEST = "synthesis_request"

    # API key management events
    API_KEY_CREATION = "api_key_creation"
    API_KEY_DELETION = "api_key_deletion"
    API_KEY_RETRIEVAL = "api_key_retrieval"

    # System health events
    HEALTH_CHECK = "health_check"


class EndpointCategory(Enum):
    """Categories for different types of API endpoints."""

    STREAM = "stream"
    FOUNDATION_MODELS = "foundation_models"
    SYNTHESIS = "synthesis"
    TRAINING = "training"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    SYNTHETIC_DATA = "synthetic_data"
    ANOMALY_DETECTION = "anomaly_detection"
    EXPLAIN = "explain"
    API_KEY_MANAGEMENT = "api_key_management"
    SYSTEM = "system"


@dataclass
class EndpointConfig:
    """Configuration for an API endpoint."""

    pattern: str
    category: EndpointCategory
    description: str
    dataset_name_extraction: Optional[str] = None
    user_id_extraction: Optional[str] = None
    event_type: Optional[str] = None
    operation: Optional[str] = None
    requires_dataset_name: bool = True
    requires_user_id: bool = True


class APIEndpoints:
    """Centralized API endpoint definitions and utilities."""

    # Individual endpoint patterns for easy reference and maintenance

    # Stream endpoints
    FORECAST_STREAM = r"^/api/forecast/[^/]+/stream"
    SYNTHESIS_STREAM_SIMPLE = r"^/api/synthesis/[^/]+/stream"
    SYNTHESIS_STREAM_USER_DATASET = r"^/api/synthesis/[^/]+/[^/]+/stream"
    SYNTHESIS_STREAM_USER_DATASET_TRAINING = (
        r"^/api/synthesis/[^/]+/[^/]+/[^/]+/stream"
    )
    PRETRAINED_ANOMALY_STREAM = r"^/api/pretrained_anomaly/[^/]+/stream"
    V2_ANOMALY_DETECTION_STREAM = r"^/api/v2/anomaly_detection/[^/]+/stream"

    # Foundation model endpoints
    FOUNDATION_MODELS_UPLOAD = r"^/api/foundation_models/upload"
    FOUNDATION_MODELS_FORECAST_STREAM = (
        r"^/api/foundation_models/forecast/stream"
    )
    FOUNDATION_MODELS_FORECAST_STREAM_V2 = (
        r"^/api/v2/foundation_models/forecast/stream"
    )
    FOUNDATION_MODELS_FORECAST = r"^/api/foundation_models/forecast/?$"
    FOUNDATION_MODELS_BACKTEST = r"^/api/foundation_models/forecast/backtest"

    # Synthesis and data generation endpoints
    SYNTHETIC_DATA_AGENT_GENERATE = (
        r"^/api/synthetic_data_agent/generate_synthetic_data/[^/]+/[^/]+"
    )

    # Other endpoints
    EXPLAIN = r"^/api/explain"
    PREPROCESS = r"^/api/preprocess/[^/]+/[^/]+"
    TRAIN = r"^/api/train/(?!status|stop|list)[^/]+$"
    POSTPROCESS_HTML = r"^/api/postprocess/html/[^/]+"

    # API key management endpoints
    USER_API_KEY_CREATE = r"^/api/user_api_key/?$"
    USER_API_KEY_DELETE = r"^/api/user_api_key/[^/]+$"
    USER_API_KEYS_LIST = r"^/api/user_api_keys/?$"

    # System endpoints
    HEALTH_CHECK = r"^/api/health$"

    # Dataset name extraction patterns
    DATASET_NAME_FORECAST_STREAM = (
        r"^/api/(?:forecast|pretrained_anomaly)/([^/]+)/stream"
    )
    DATASET_NAME_SYNTHESIS_STREAM_SIMPLE = r"^/api/synthesis/([^/]+)/stream"
    DATASET_NAME_SYNTHESIS_STREAM_USER_DATASET = (
        r"^/api/synthesis/[^/]+/([^/]+)/stream"
    )
    DATASET_NAME_SYNTHESIS_STREAM_USER_DATASET_TRAINING = (
        r"^/api/synthesis/[^/]+/([^/]+)/[^/]+/stream"
    )
    DATASET_NAME_V2_ANOMALY_DETECTION_STREAM = (
        r"^/api/v2/(?:anomaly_detection)/([^/]+)/stream"
    )
    DATASET_NAME_PREPROCESS = r"^/api/preprocess/[^/]+/([^/]+)"
    DATASET_NAME_TRAIN = r"^/api/train/(?!status|stop|list)([^/]+)$"
    DATASET_NAME_POSTPROCESS_HTML = r"^/api/postprocess/html/([^/]+)"
    DATASET_NAME_SYNTHETIC_DATA_AGENT_GENERATE = (
        r"^/api/synthetic_data_agent/generate_synthetic_data/[^/]+/([^/]+)"
    )

    # User ID extraction patterns
    USER_ID_SYNTHESIS_STREAM_USER_DATASET = (
        r"^/api/synthesis/([^/]+)/[^/]+/stream"
    )
    USER_ID_SYNTHESIS_STREAM_USER_DATASET_TRAINING = (
        r"^/api/synthesis/([^/]+)/[^/]+/[^/]+/stream"
    )
    USER_ID_PREPROCESS = r"^/api/preprocess/([^/]+)/[^/]+"
    USER_ID_SYNTHETIC_DATA_AGENT_GENERATE = (
        r"^/api/synthetic_data_agent/generate_synthetic_data/([^/]+)/[^/]+"
    )

    # Tracking patterns - endpoints that should be tracked for billing
    TRACKING_PATTERNS = [
        FORECAST_STREAM,
        SYNTHESIS_STREAM_SIMPLE,
        SYNTHESIS_STREAM_USER_DATASET,
        SYNTHESIS_STREAM_USER_DATASET_TRAINING,
        PRETRAINED_ANOMALY_STREAM,
        V2_ANOMALY_DETECTION_STREAM,
        FOUNDATION_MODELS_UPLOAD,
        FOUNDATION_MODELS_FORECAST_STREAM,
        FOUNDATION_MODELS_FORECAST_STREAM_V2,
        FOUNDATION_MODELS_FORECAST,
        FOUNDATION_MODELS_BACKTEST,
        SYNTHETIC_DATA_AGENT_GENERATE,
        EXPLAIN,
        PREPROCESS,
        TRAIN,
        POSTPROCESS_HTML,
        USER_API_KEY_CREATE,
        USER_API_KEY_DELETE,
        USER_API_KEYS_LIST,
        HEALTH_CHECK,
    ]

    # Dataset name extraction patterns
    DATASET_NAME_PATTERNS = [
        DATASET_NAME_FORECAST_STREAM,
        DATASET_NAME_SYNTHESIS_STREAM_SIMPLE,
        DATASET_NAME_SYNTHESIS_STREAM_USER_DATASET,
        DATASET_NAME_SYNTHESIS_STREAM_USER_DATASET_TRAINING,
        DATASET_NAME_V2_ANOMALY_DETECTION_STREAM,
        DATASET_NAME_PREPROCESS,
        DATASET_NAME_TRAIN,
        DATASET_NAME_POSTPROCESS_HTML,
        DATASET_NAME_SYNTHETIC_DATA_AGENT_GENERATE,
    ]

    # User ID extraction patterns
    USER_ID_PATTERNS = [
        USER_ID_SYNTHESIS_STREAM_USER_DATASET,
        USER_ID_SYNTHESIS_STREAM_USER_DATASET_TRAINING,
        USER_ID_PREPROCESS,
        USER_ID_SYNTHETIC_DATA_AGENT_GENERATE,
    ]

    # Endpoint configurations for event type determination
    ENDPOINT_CONFIGS = {
        "/api/v2/foundation_models/forecast/stream": EndpointConfig(
            pattern=FOUNDATION_MODELS_FORECAST_STREAM_V2,
            category=EndpointCategory.FOUNDATION_MODELS,
            description="Forecast execution endpoint",
            event_type=APIEventType.STREAM_FORECAST_EXECUTION_V2.value,
            operation="stream_forecast_v2",
            requires_dataset_name=False,  # Dataset name comes from request body
        ),
        "/api/foundation_models/forecast/stream": EndpointConfig(
            pattern=FOUNDATION_MODELS_FORECAST_STREAM,
            category=EndpointCategory.FOUNDATION_MODELS,
            description="Forecast execution endpoint",
            event_type=APIEventType.STREAM_FORECAST_EXECUTION.value,
            operation="stream_forecast",
            requires_dataset_name=False,  # Dataset name comes from request body
        ),
        "/api/foundation_models/upload": EndpointConfig(
            pattern=FOUNDATION_MODELS_UPLOAD,
            category=EndpointCategory.FOUNDATION_MODELS,
            description="Dataset upload endpoint",
            event_type=APIEventType.DATASET_UPLOAD.value,
            operation="upload",
            requires_dataset_name=False,  # Dataset name comes from form data
        ),
        "/api/foundation_models/forecast": EndpointConfig(
            pattern=FOUNDATION_MODELS_FORECAST,
            category=EndpointCategory.FOUNDATION_MODELS,
            description="Forecast execution endpoint",
            event_type=APIEventType.FORECAST_EXECUTION.value,
            operation="forecast",
            requires_dataset_name=False,  # Dataset name comes from request body
        ),
        "/api/foundation_models/forecast/backtest": EndpointConfig(
            pattern=FOUNDATION_MODELS_BACKTEST,
            category=EndpointCategory.FOUNDATION_MODELS,
            description="Backtest execution endpoint",
            event_type=APIEventType.BACKTEST_EXECUTION.value,
            operation="backtest",
            requires_dataset_name=False,  # Dataset name comes from request body
        ),
        "/api/preprocess": EndpointConfig(
            pattern=PREPROCESS,
            category=EndpointCategory.PREPROCESSING,
            description="Dataset creation endpoint",
            event_type=APIEventType.DATASET_CREATION.value,
            operation="preprocess",
            user_id_extraction=USER_ID_PREPROCESS,
        ),
        "/api/train": EndpointConfig(
            pattern=TRAIN,
            category=EndpointCategory.TRAINING,
            description="Training job creation endpoint",
            event_type=APIEventType.TRAINING_JOB_CREATION.value,
            operation="train",
        ),
        "/api/postprocess/html": EndpointConfig(
            pattern=POSTPROCESS_HTML,
            category=EndpointCategory.POSTPROCESSING,
            description="Postprocessing report creation endpoint",
            event_type=APIEventType.POSTPROCESSING_REPORT_CREATION.value,
            operation="postprocess_html",
        ),
        "/api/synthetic_data_agent/generate_synthetic_data": EndpointConfig(
            pattern=SYNTHETIC_DATA_AGENT_GENERATE,
            category=EndpointCategory.SYNTHETIC_DATA,
            description="Synthetic data generation endpoint",
            event_type=APIEventType.SYNTHETIC_DATA_GENERATION.value,
            operation="generate_synthetic_data",
            user_id_extraction=USER_ID_SYNTHETIC_DATA_AGENT_GENERATE,
        ),
        "/api/synthesis/stream": EndpointConfig(
            pattern=SYNTHESIS_STREAM_USER_DATASET_TRAINING,
            category=EndpointCategory.SYNTHESIS,
            description="Synthesis stream endpoint",
            event_type=APIEventType.SYNTHESIS_REQUEST.value,
            operation="synthesis",
            user_id_extraction=USER_ID_SYNTHESIS_STREAM_USER_DATASET_TRAINING,
        ),
        "/api/synthesis/simple": EndpointConfig(
            pattern=SYNTHESIS_STREAM_SIMPLE,
            category=EndpointCategory.SYNTHESIS,
            description="Simple synthesis stream endpoint",
            event_type=APIEventType.SYNTHESIS_REQUEST.value,
            operation="synthesis",
        ),
        "/api/synthesis/user_dataset": EndpointConfig(
            pattern=SYNTHESIS_STREAM_USER_DATASET,
            category=EndpointCategory.SYNTHESIS,
            description="Synthesis stream endpoint with user_id and dataset_name",
            event_type=APIEventType.SYNTHESIS_REQUEST.value,
            operation="synthesis",
            user_id_extraction=USER_ID_SYNTHESIS_STREAM_USER_DATASET,
        ),
        "/api/user_api_key": EndpointConfig(
            pattern=USER_API_KEY_CREATE,
            category=EndpointCategory.API_KEY_MANAGEMENT,
            description="API key creation endpoint",
            event_type=APIEventType.API_KEY_CREATION.value,
            operation="create_api_key",
            requires_dataset_name=False,
            requires_user_id=True,
        ),
        "/api/user_api_key/{api_key_id}": EndpointConfig(
            pattern=USER_API_KEY_DELETE,
            category=EndpointCategory.API_KEY_MANAGEMENT,
            description="API key deletion endpoint",
            event_type=APIEventType.API_KEY_DELETION.value,
            operation="delete_api_key",
            requires_dataset_name=False,
            requires_user_id=True,
        ),
        "/api/user_api_keys": EndpointConfig(
            pattern=USER_API_KEYS_LIST,
            category=EndpointCategory.API_KEY_MANAGEMENT,
            description="API keys list endpoint",
            event_type=APIEventType.API_KEY_RETRIEVAL.value,
            operation="list_api_keys",
            requires_dataset_name=False,
            requires_user_id=True,
        ),
        "/api/health": EndpointConfig(
            pattern=HEALTH_CHECK,
            category=EndpointCategory.SYSTEM,
            description="Health check endpoint",
            event_type=APIEventType.HEALTH_CHECK.value,
            operation="health_check",
            requires_dataset_name=False,
            requires_user_id=False,
        ),
    }

    @classmethod
    def should_track_endpoint(cls, path: str) -> bool:
        """Check if the endpoint should be tracked for billing."""
        for pattern in cls.TRACKING_PATTERNS:
            if re.match(pattern, path):
                return True
        return False

    @classmethod
    def should_bill_endpoint(cls, path: str) -> bool:
        """Check if the endpoint should be billed (only synthesis, backtest, and generate_synthetic_data)."""
        # Only bill for synthesis, backtest, and generate_synthetic_data endpoints
        synthesis_patterns = [
            cls.SYNTHESIS_STREAM_SIMPLE,
            cls.SYNTHESIS_STREAM_USER_DATASET,
            cls.SYNTHESIS_STREAM_USER_DATASET_TRAINING,
        ]

        backtest_patterns = [
            cls.FOUNDATION_MODELS_BACKTEST,
        ]

        generate_synthetic_data_patterns = [
            cls.SYNTHETIC_DATA_AGENT_GENERATE,
        ]

        # Check if path matches any of the billable patterns
        all_billable_patterns = (
            synthesis_patterns
            + backtest_patterns
            + generate_synthetic_data_patterns
        )

        for pattern in all_billable_patterns:
            if re.match(pattern, path):
                return True
        return False

    @classmethod
    def extract_dataset_name(cls, path: str) -> Optional[str]:
        """Extract dataset name from URL path if present."""
        for pattern in cls.DATASET_NAME_PATTERNS:
            match = re.match(pattern, path)
            if match:
                return match.group(1)

        # Special cases where dataset name cannot be extracted from path
        if path == "/api/foundation_models/upload":
            return None  # Dataset name comes from form data
        if path in [
            "/api/foundation_models/forecast/backtest",
            "/api/foundation_models/forecast",
        ]:
            return None  # Dataset name comes from request body

        return None

    @classmethod
    def extract_user_id_from_path(cls, path: str) -> Optional[str]:
        """Extract user_id from URL path if present."""
        for pattern in cls.USER_ID_PATTERNS:
            match = re.match(pattern, path)
            if match:
                return match.group(1)
        return None

    @classmethod
    def get_endpoint_config(cls, path: str) -> Optional[EndpointConfig]:
        """Get endpoint configuration for a given path."""
        for config in cls.ENDPOINT_CONFIGS.values():
            if re.match(config.pattern, path):
                return config
        return None

    @classmethod
    def get_event_type_and_operation(
        cls, path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get event type and operation for a given path."""
        config = cls.get_endpoint_config(path)
        if config:
            return config.event_type, config.operation
        return None, None

    @classmethod
    def extract_user_id_from_path_by_config(cls, path: str) -> Optional[str]:
        """Extract user_id from path using endpoint configuration."""
        config = cls.get_endpoint_config(path)
        if config and config.user_id_extraction:
            match = re.match(config.user_id_extraction, path)
            if match:
                return match.group(1)
        return None

    # Convenience methods for specific endpoint patterns
    @classmethod
    def is_synthetic_data_agent_endpoint(cls, path: str) -> bool:
        """Check if the path matches the synthetic data agent endpoint."""
        return bool(re.match(cls.SYNTHETIC_DATA_AGENT_GENERATE, path))

    @classmethod
    def is_foundation_models_forecast(cls, path: str) -> bool:
        """Check if the path matches the foundation models forecast endpoint."""
        return bool(re.match(cls.FOUNDATION_MODELS_FORECAST, path))

    @classmethod
    def is_foundation_models_backtest(cls, path: str) -> bool:
        """Check if the path matches the foundation models backtest endpoint."""
        return bool(re.match(cls.FOUNDATION_MODELS_BACKTEST, path))

    @classmethod
    def is_preprocess_endpoint(cls, path: str) -> bool:
        """Check if the path matches the preprocess endpoint."""
        return bool(re.match(cls.PREPROCESS, path))

    @classmethod
    def is_train_endpoint(cls, path: str) -> bool:
        """Check if the path matches the train endpoint."""
        return bool(re.match(cls.TRAIN, path))

    @classmethod
    def is_postprocess_html_endpoint(cls, path: str) -> bool:
        """Check if the path matches the postprocess html endpoint."""
        return bool(re.match(cls.POSTPROCESS_HTML, path))

    @classmethod
    def is_synthesis_stream_endpoint(cls, path: str) -> bool:
        """Check if the path matches any synthesis stream endpoint."""
        return any(
            [
                bool(re.match(cls.SYNTHESIS_STREAM_SIMPLE, path)),
                bool(re.match(cls.SYNTHESIS_STREAM_USER_DATASET, path)),
                bool(
                    re.match(cls.SYNTHESIS_STREAM_USER_DATASET_TRAINING, path)
                ),
            ]
        )

    @classmethod
    def is_dataset_creation_endpoint(cls, path: str) -> bool:
        """Check if the path matches a dataset creation endpoint that uses compound dataset naming."""
        # Currently only preprocess endpoint uses compound dataset naming
        return cls.is_preprocess_endpoint(path)

    @classmethod
    def is_api_key_create_endpoint(cls, path: str) -> bool:
        """Check if the path matches the API key creation endpoint."""
        return bool(re.match(cls.USER_API_KEY_CREATE, path))

    @classmethod
    def is_api_key_delete_endpoint(cls, path: str) -> bool:
        """Check if the path matches the API key deletion endpoint."""
        return bool(re.match(cls.USER_API_KEY_DELETE, path))

    @classmethod
    def is_api_keys_list_endpoint(cls, path: str) -> bool:
        """Check if the path matches the API keys list endpoint."""
        return bool(re.match(cls.USER_API_KEYS_LIST, path))

    @classmethod
    def is_api_key_management_endpoint(cls, path: str) -> bool:
        """Check if the path matches any API key management endpoint."""
        return any(
            [
                cls.is_api_key_create_endpoint(path),
                cls.is_api_key_delete_endpoint(path),
                cls.is_api_keys_list_endpoint(path),
            ]
        )

    @classmethod
    def is_health_check_endpoint(cls, path: str) -> bool:
        """Check if the path matches the health check endpoint."""
        return bool(re.match(cls.HEALTH_CHECK, path))

    @classmethod
    def is_system_endpoint(cls, path: str) -> bool:
        """Check if the path matches any system endpoint."""
        return any(
            [
                cls.is_health_check_endpoint(path),
            ]
        )

    @classmethod
    def get_pattern_by_name(cls, pattern_name: str) -> Optional[str]:
        """Get a specific pattern by its attribute name."""
        return getattr(cls, pattern_name, None)

    @classmethod
    def list_all_patterns(cls) -> Dict[str, str]:
        """Get all pattern attributes as a dictionary."""
        patterns = {}
        for attr_name in dir(cls):
            if attr_name.isupper() and not attr_name.startswith("_"):
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, str) and attr_value.startswith("^"):
                    patterns[attr_name] = attr_value
        return patterns
