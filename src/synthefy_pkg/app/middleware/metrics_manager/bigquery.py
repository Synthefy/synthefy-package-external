"""
Simple BigQuery metrics manager implementation.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from google.cloud import bigquery
from loguru import logger

from synthefy_pkg.app.config import SynthefyCredentialsSettings
from synthefy_pkg.app.middleware.api_endpoints import APIEventType
from synthefy_pkg.app.middleware.metrics_manager._base import BaseMetricsManager
from synthefy_pkg.app.utils.s3_utils import (
    download_file_from_s3,
    parse_s3_url,
)
from synthefy_pkg.utils.config_utils import load_yaml_config

SYNTHEFY_CREDENTIALS_SETTINGS = SynthefyCredentialsSettings()


class BigQueryMetricsManager(BaseMetricsManager):
    """Simple BigQuery metrics manager with class-based configuration."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str = "metrics_raw",
        table_id: str = "api_events_raw",
    ):
        """Initialize with basic config."""
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_full_id = (
            f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        )

        # Client state
        self._client: Optional[bigquery.Client] = None
        self._initialized = False
        self._config = None

    @classmethod
    def initialize(cls) -> "BigQueryMetricsManager":
        """
        Initialize BigQuery manager from environment variables.

        Environment Variables:
            GOOGLE_CLOUD_PROJECT: GCP project ID (required)
            BIGQUERY_DATASET_ID: Dataset ID (optional, defaults to 'metrics_raw')
            BIGQUERY_TABLE_ID: Table ID (optional, defaults to 'api_events_raw')

        Returns:
            Initialized BigQuery metrics manager

        Raises:
            ValueError: If GOOGLE_CLOUD_PROJECT is not set
            RuntimeError: If BigQuery libraries not available or client initialization fails
        """
        config = load_yaml_config(os.getenv("CONFIG_PATH", ""))
        config = config["api_usage_bigquery_metrics"]

        # Get environment variables
        project_id = config.get("project_id", "raimi-test-454417")
        if not project_id:
            msg = "Project ID is missing from the config file"
            raise ValueError(msg)

        # Get dataset_id from config
        dataset_id = config["dataset_id"]

        table_id = config.get("table_id", "api_events_raw")

        logger.info(
            f"Initializing BigQuery metrics manager with dataset_id: {dataset_id}"
        )

        local_gcp_credentials_path = Path(
            SYNTHEFY_CREDENTIALS_SETTINGS.local_credentials_path
        ).joinpath(SYNTHEFY_CREDENTIALS_SETTINGS.gcp_file_name)

        # Set the required environment variables
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
            local_gcp_credentials_path
        )
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        if not local_gcp_credentials_path.exists():
            try:
                s3_client = boto3.client("s3")
                s3_url = f"{SYNTHEFY_CREDENTIALS_SETTINGS.s3_credentials_path.rstrip('/')}/{SYNTHEFY_CREDENTIALS_SETTINGS.gcp_file_name}"
                bucket, key = parse_s3_url(s3_url)
                download_file_from_s3(
                    s3_client=s3_client,
                    bucket=bucket,
                    s3_key=key,
                    local_path=str(local_gcp_credentials_path),
                )
                logger.debug(
                    f"Downloaded credentials from s3: {local_gcp_credentials_path.parent}"
                )

            except Exception as e:
                logger.error(f"Failed to download credentials: {e}")
                raise e

        # Create instance
        manager = cls(project_id, dataset_id, table_id)

        # Initialize BigQuery client
        try:
            manager._client = bigquery.Client(project=project_id)
            manager._initialized = True
            manager._config = config
            logger.info(
                f"BigQuery metrics manager initialized for {manager.table_full_id}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BigQuery client: {e}")

        return manager

    def is_ready(self) -> bool:
        """Check if the manager is ready to record metrics."""
        return self._initialized and self._client is not None

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
        """Record API usage to BigQuery."""
        if not self.is_ready():
            logger.debug(
                "BigQuery metrics manager not ready, skipping API usage recording"
            )
            return

        # Type guard: ensure client is not None after is_ready() check
        assert self._client is not None

        # Create simple record
        payload_data = {
            "processing_time_ms": processing_time_ms,
            "status_code": status_code,
            "correlation_id": correlation_id,
            "api_key_hash": self._hash_api_key(api_key) if api_key else None,
        }

        # Add extra payload data if provided
        if extra_payload:
            payload_data.update(extra_payload)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "api_endpoint": endpoint,
            "dataset_id": dataset_name,
            "status": "success" if 200 <= status_code < 400 else "error",
            "payload": json.dumps(payload_data),
        }

        try:
            self._client.insert_rows_json(self.table_full_id, [row])
            logger.debug(f"API usage recorded for user {user_id}")
        except Exception as e:
            logger.debug(f"BigQuery insert failed: {e}")

    def record_metric(self, event_data: Dict[str, Any]) -> None:
        """Record custom metric to BigQuery."""
        if not self.is_ready():
            logger.debug(
                "BigQuery metrics manager not ready, skipping metric recording"
            )
            return

        # Type guard: ensure client is not None after is_ready() check
        assert self._client is not None

        # Add timestamp if missing
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Convert payload to JSON string if present
        if "payload" in event_data and event_data["payload"] is not None:
            event_data["payload"] = json.dumps(event_data["payload"])

        try:
            self._client.insert_rows_json(self.table_full_id, [event_data])
            logger.debug(f"Metric recorded: {event_data.get('event_type')}")
        except Exception as e:
            logger.debug(f"BigQuery insert failed: {e}")

    def record_log(
        self,
        level: str,
        message: str,
        module: str = "application",
        user_id: str = "system",
        correlation_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record log entry to BigQuery."""
        if not self.is_ready():
            logger.debug(
                "BigQuery metrics manager not ready, skipping log recording"
            )
            return

        # Type guard: ensure client is not None after is_ready() check
        assert self._client is not None

        # Create log record
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "application_log",
            "user_id": user_id,
            "status": self._get_log_status(level),
            "payload": {
                "level": level,
                "message": message,
                "module": module,
                "correlation_id": correlation_id,
                **(extra_data or {}),
            },
        }

        try:
            self._client.insert_rows_json(self.table_full_id, [log_data])
            logger.debug(f"Log recorded: {level} - {message[:50]}...")
        except Exception as e:
            logger.debug(f"BigQuery log insert failed: {e}")

    def _get_log_status(self, level: str) -> str:
        """Map log level to status."""
        if level in ["ERROR", "CRITICAL"]:
            return "error"
        elif level == "WARNING":
            return "warning"
        else:
            return "info"

    def test_connection(self) -> bool:
        """
        Test BigQuery connection.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.is_ready():
            logger.error(
                "BigQuery manager is not ready - client not initialized"
            )
            return False

        # Type guard: ensure client is not None after is_ready() check
        assert self._client is not None

        try:
            # Test basic connection
            list(self._client.list_datasets(max_results=1))
            logger.info("BigQuery connection test successful")
            return True
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            logger.error(f"Project ID: {self.project_id}")
            logger.error(
                "Check: 1) GOOGLE_CLOUD_PROJECT env var, 2) Authentication, 3) Project permissions"
            )
            return False

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "table_id": self.table_id,
            "table_full_id": self.table_full_id,
            "initialized": self._initialized,
            "ready": self.is_ready(),
        }

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for privacy."""
        import hashlib

        return hashlib.sha256(api_key.encode()).hexdigest()[:16]


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def test_bigquery_connection(project_id: str) -> bool:
    """Test BigQuery connection (standalone function)."""

    try:
        client = bigquery.Client(project=project_id)
        list(client.list_datasets(max_results=1))
        logger.info("BigQuery connection test successful")
        return True
    except Exception as e:
        logger.error(f"BigQuery connection test failed: {e}")
        return False


def simulate_api_calls(
    manager: BigQueryMetricsManager, num_calls: int = 100
) -> None:
    """Simulate API calls to test BigQuery integration."""
    import random
    import time

    if not manager or not manager.is_ready():
        logger.error("BigQuery manager is not ready for simulation")
        return

    logger.info(f"Starting simulation of {num_calls} API calls...")

    endpoints = [
        "/api/forecast/air_quality/stream",
        "/api/synthesis/ppg_data/stream",
        "/api/v2/anomaly_detection/ecg_data/stream",
        "/api/foundation_models/forecast/stream",
        "/api/train/weather_model",
        "/api/explain",
    ]

    user_ids = [f"user_{i}" for i in range(1, 11)]
    dataset_names = [
        "air_quality",
        "ppg_data",
        "ecg_data",
        "weather_data",
        "sensor_data",
    ]

    for i in range(num_calls):
        endpoint = random.choice(endpoints)
        user_id = random.choice(user_ids)
        dataset_name = (
            random.choice(dataset_names) if "stream" in endpoint else None
        )
        api_key = f"api_key_{random.randint(1000, 9999)}"
        processing_time_ms = random.uniform(50, 2000)
        status_code = random.choices(
            [200, 201, 400, 401, 500], weights=[0.8, 0.1, 0.05, 0.03, 0.02]
        )[0]
        correlation_id = f"corr_{random.randint(10000, 99999)}"

        manager.record_api_usage(
            user_id=user_id,
            api_key=api_key,
            endpoint=endpoint,
            dataset_name=dataset_name,
            processing_time_ms=processing_time_ms,
            status_code=status_code,
            correlation_id=correlation_id,
        )

        time.sleep(0.01)

        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i + 1}/{num_calls} API calls")

    logger.info(f"API simulation completed. Processed {num_calls} calls.")


def simulate_training_jobs(
    manager: BigQueryMetricsManager, num_jobs: int = 25
) -> None:
    """Simulate training job events."""
    import random
    import time

    if not manager or not manager.is_ready():
        logger.error("BigQuery manager is not ready for simulation")
        return

    logger.info(f"Starting simulation of {num_jobs} training jobs...")

    job_types = ["forecast", "synthesis", "anomaly_detection"]
    datasets = [
        "air_quality",
        "ppg_data",
        "ecg_data",
        "weather_data",
        "sensor_data",
    ]
    user_ids = [f"user_{i}" for i in range(1, 6)]

    for i in range(num_jobs):
        job_type = random.choice(job_types)
        dataset = random.choice(datasets)
        user_id = random.choice(user_ids)
        job_id = f"job_{random.randint(10000, 99999)}"

        # Job started
        manager.record_metric(
            {
                "event_type": "training_job_started",
                "user_id": user_id,
                "job_id": job_id,
                "dataset_id": dataset,
                "status": "started",
                "payload": {
                    "job_type": job_type,
                    "dataset": dataset,
                    "model_config": f"config_{job_type}_{dataset}",
                },
            }
        )

        time.sleep(0.1)

        # Job completion (90% success rate)
        success = random.random() > 0.1
        if success:
            manager.record_metric(
                {
                    "event_type": "training_job_completed",
                    "user_id": user_id,
                    "job_id": job_id,
                    "dataset_id": dataset,
                    "status": "completed",
                    "payload": {
                        "job_type": job_type,
                        "dataset": dataset,
                        "training_duration_minutes": random.randint(10, 120),
                        "final_loss": random.uniform(0.01, 0.5),
                    },
                }
            )
        else:
            manager.record_metric(
                {
                    "event_type": "training_job_failed",
                    "user_id": user_id,
                    "job_id": job_id,
                    "dataset_id": dataset,
                    "status": "failed",
                    "error_message": "Training job failed due to insufficient memory",
                    "error_code": "MEMORY_ERROR",
                    "payload": {
                        "job_type": job_type,
                        "dataset": dataset,
                        "failure_reason": "insufficient_memory",
                    },
                }
            )

        if (i + 1) % 5 == 0:
            logger.info(f"Processed {i + 1}/{num_jobs} training jobs")

    logger.info(
        f"Training job simulation completed. Processed {num_jobs} jobs."
    )


def simulate_user_events(
    manager: BigQueryMetricsManager, num_events: int = 50
) -> None:
    """Simulate user login/logout events."""
    import random
    import time

    if not manager or not manager.is_ready():
        logger.error("BigQuery manager is not ready for simulation")
        return

    logger.info(f"Starting simulation of {num_events} user events...")

    user_ids = [f"user_{i}" for i in range(1, 16)]

    for i in range(num_events):
        user_id = random.choice(user_ids)
        event_type = random.choice(
            [APIEventType.USER_LOGIN.value, APIEventType.USER_LOGOUT.value]
        )

        manager.record_metric(
            {
                "event_type": event_type,
                "user_id": user_id,
                "status": "success",
                "payload": {
                    "ip_address": f"192.168.1.{random.randint(1, 255)}",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            }
        )

        time.sleep(0.05)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{num_events} user events")

    logger.info(
        f"User events simulation completed. Processed {num_events} events."
    )


def run_comprehensive_test(
    project_id: str,
    dataset_id: str = "metrics_raw",
    table_id: str = "api_events_raw",
) -> None:
    """Run comprehensive test of the BigQuery metrics system."""
    logger.info("Starting comprehensive BigQuery metrics test...")

    # Create and initialize manager
    try:
        # Set environment variables for initialization
        import os

        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["BIGQUERY_DATASET_ID"] = dataset_id
        os.environ["BIGQUERY_TABLE_ID"] = table_id

        # Use the initialize method to properly create the manager
        manager = BigQueryMetricsManager.initialize()

        logger.info("BigQuery metrics manager initialized successfully")
        logger.info(f"Configuration: {manager.get_config()}")

    except Exception as e:
        logger.error(f"Failed to initialize BigQuery metrics manager: {e}")
        return

    # Test connection
    if not manager.test_connection():
        logger.error("BigQuery connection test failed. Aborting test.")
        return

    # Run simulations
    logger.info("Running API calls simulation...")
    simulate_api_calls(manager, 50)

    logger.info("Running training jobs simulation...")
    simulate_training_jobs(manager, 10)

    logger.info("Running user events simulation...")
    simulate_user_events(manager, 25)

    logger.info("Comprehensive test completed successfully!")


if __name__ == "__main__":
    # Test script when run directly
    import os

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset_id = os.getenv("BIGQUERY_DATASET_ID", "metrics_raw")
    table_id = os.getenv("BIGQUERY_TABLE_ID", "api_events_raw")

    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        logger.info("Please set GOOGLE_CLOUD_PROJECT to your GCP project ID")
        exit(1)

    run_comprehensive_test(project_id, dataset_id, table_id)
