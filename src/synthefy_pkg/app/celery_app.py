import logging
import os

from celery import Celery

# Set up logging
logger = logging.getLogger(__name__)

# Only create real Celery instance if USE_CELERY is true
if os.getenv("USE_CELERY", "false").lower() == "true":
    # Create real Celery instance
    celery_app = Celery(
        "synthefy",
        broker="redis://synthefy-redis:6379/0",
        backend="redis://synthefy-redis:6379/0",
        include=["synthefy_pkg.app.tasks", "examples.tstr_lightgbm"],
    )

    # Enhanced configuration with task persistence and reliability
    celery_app.conf.update(
        result_expires=86400,  # Results expire after 1 day - can adjust later
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_track_started=True,
        timezone="UTC",
        enable_utc=True,
        task_publish_retry=True,
        worker_send_task_events=True,
        task_send_sent_event=True,
        task_ignore_result=False,
        # Task persistence and reliability settings
        task_acks_late=True,  # Don't acknowledge until task completes
        worker_prefetch_multiplier=1,  # Process one task at a time
        task_reject_on_worker_lost=True,  # Requeue tasks if worker dies
        task_acks_on_failure_or_timeout=True,  # Handle failed/timed out tasks properly
        # Task timeout settings
        task_soft_time_limit=1800,  # 30 minutes soft limit (send SIGTERM)
        task_time_limit=1900,  # 31 minutes hard limit (send SIGKILL)
        # Redis specific settings for persistence
        broker_connection_retry=True,
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=10,
        # Worker settings
        worker_hijack_root_logger=False,
        worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
        worker_disable_rate_limits=False,
        # Task routing - all tasks go to default queue
        task_routes={},
        # Default queue
        task_default_queue="default",
        task_default_exchange="default",
        task_default_routing_key="default",
    )
else:
    # Create a mock Celery app that has the same interface but does nothing
    class MockAsyncResult:
        """Mock AsyncResult that mimics Celery's AsyncResult interface."""

        def __init__(self, task_id=None):
            self.task_id = task_id
            self.info = {}
            self.state = "PENDING"
            self.result = None
            logger.warning(
                f"MockAsyncResult created for task_id: {task_id} - This is NOT a real Celery result!"
            )

        def get(self, *args, **kwargs):
            logger.debug(
                f"MockAsyncResult.get() called for task_id: {self.task_id} - returning None (mock)"
            )
            return None

        def ready(self):
            logger.debug(
                f"MockAsyncResult.ready() called for task_id: {self.task_id} - returning True (mock)"
            )
            return True

        def successful(self):
            logger.debug(
                f"MockAsyncResult.successful() called for task_id: {self.task_id} - returning True (mock)"
            )
            return True

        def failed(self):
            logger.debug(
                f"MockAsyncResult.failed() called for task_id: {self.task_id} - returning False (mock)"
            )
            return False

        def wait(self, *args, **kwargs):
            logger.debug(
                f"MockAsyncResult.wait() called for task_id: {self.task_id} - returning None (mock)"
            )
            return None

    class MockControl:
        """Mock Celery control interface."""

        def revoke(self, task_id, terminate=False):
            logger.debug(
                f"MockControl.revoke() called for task_id: {task_id} (mock)"
            )
            return True

        def inspect(self):
            return MockInspect()

        def broadcast(self, command, reply=False):
            logger.debug(
                f"MockControl.broadcast() called with command: {command} (mock)"
            )
            return True

    class MockInspect:
        """Mock Celery inspect interface."""

        def ping(self):
            return {"mock_worker": "pong"}

        def active(self):
            return {}

        def stats(self):
            return {}

        def reserved(self):
            return {}

        def scheduled(self):
            return {}

    class MockCelery:
        """Mock Celery app that provides the same interface but runs tasks synchronously."""

        def __init__(self):
            self.control = MockControl()
            logger.warning(
                "MockCelery initialized - This is NOT a real Celery app! Tasks will run synchronously."
            )

        def task(self, *args, **kwargs):
            """Mock task decorator that just returns the function unchanged."""

            def decorator(func):
                logger.debug(
                    f"Mock task decorator applied to function: {func.__name__} - function will run synchronously"
                )
                # Return the function unchanged so it can be called directly
                return func

            return decorator

        def AsyncResult(self, task_id):
            """Return a mock AsyncResult."""
            logger.debug(
                f"MockCelery.AsyncResult() called for task_id: {task_id} - returning MockAsyncResult"
            )
            return MockAsyncResult(task_id)

        def delay(self, *args, **kwargs):
            """Mock delay method that just returns a mock task ID."""
            logger.debug(
                "MockCelery.delay() called - returning MockAsyncResult with mock_task_id"
            )
            return MockAsyncResult("mock_task_id")

        def apply_async(self, *args, **kwargs):
            """Mock apply_async method that just returns a mock task ID."""
            logger.debug(
                "MockCelery.apply_async() called - returning MockAsyncResult with mock_task_id"
            )
            return MockAsyncResult("mock_task_id")

    celery_app = MockCelery()
