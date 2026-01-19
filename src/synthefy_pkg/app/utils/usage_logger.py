import asyncio
import hashlib
import hmac
import json
import os
import threading
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import aiohttp
import requests
from loguru import logger

from synthefy_pkg.app.middleware.api_endpoints import APIEventType


@lru_cache(maxsize=1)
def _get_webhook_config() -> tuple[str, str]:
    """Get webhook configuration from environment variables.

    Cached using lru_cache to avoid reading from environment on every call.

    Returns:
        Tuple of (webhook_url, webhook_secret)
    """
    webhook_url = os.environ.get("API_USAGE_WEBHOOK_URL", "")
    webhook_secret = os.environ.get("API_USAGE_WEBHOOK_SECRET", "")

    # Verify webhook configuration
    if not webhook_url:
        logger.warning(
            "API_USAGE_WEBHOOK_URL not set - API usage tracking disabled"
        )
    if not webhook_secret:
        logger.warning(
            "API_USAGE_WEBHOOK_SECRET not set - API usage tracking may not be secure"
        )

    return webhook_url, webhook_secret


async def log_api_usage_async(
    user_id: str,
    api_key: Optional[str],
    endpoint: str,
    dataset_name: Optional[str],
    processing_time_ms: float,
    status_code: int,
) -> None:
    """
    Log API usage asynchronously by sending a webhook to the configured endpoint.

    This function is used for billing purposes and should only be called for successful responses
    (status codes 200-399). Failed requests should not be billed to customers.

    Args:
        user_id: The user ID associated with the request
        api_key: The API key used for the request (can be None)
        endpoint: The API endpoint that was called
        dataset_name: The dataset name if applicable (can be None)
        processing_time_ms: The time taken to process the request in milliseconds
        status_code: The HTTP status code of the response
    """
    # Get webhook configuration
    webhook_url, _ = _get_webhook_config()
    if not webhook_url:
        return

    try:
        # Create usage entry details for the webhook
        usage_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        webhook_payload_data = {
            "id": usage_id,
            "user_id": user_id,
            "api_key": api_key,
            "endpoint": endpoint,
            "dataset_name": dataset_name,
            "processing_time_ms": processing_time_ms,
            "status_code": status_code,
            "timestamp": timestamp,
        }

        await _send_webhook_event_async(webhook_payload_data)
    except Exception as e:
        logger.error(
            f"Failed to initiate asynchronous API usage logging: {str(e)}"
        )


def log_api_usage(
    user_id: str,
    api_key: Optional[str],
    endpoint: str,
    dataset_name: Optional[str],
    processing_time_ms: float,
    status_code: int,
) -> None:
    """
    Log API usage by sending a webhook (synchronous version).
    For use in non-async contexts only.

    Args:
        user_id: The user ID associated with the request
        api_key: The API key used for the request (can be None)
        endpoint: The API endpoint that was called
        dataset_name: The dataset name if applicable (can be None)
        processing_time_ms: The time taken to process the request in milliseconds
        status_code: The HTTP status code of the response
    """
    # Get webhook configuration
    webhook_url, _ = _get_webhook_config()
    if not webhook_url:
        return

    try:
        # Create usage entry details for the webhook
        usage_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        webhook_payload_data = {
            "id": usage_id,
            "user_id": user_id,
            "api_key": api_key,
            "endpoint": endpoint,
            "dataset_name": dataset_name,
            "processing_time_ms": processing_time_ms,
            "status_code": status_code,
            "timestamp": timestamp,
        }

        # Run asynchronously but don't block or wait for completion
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send_webhook_event_async(webhook_payload_data))
        loop.close()
    except Exception as e:
        logger.error(f"Failed to initiate API usage logging: {str(e)}")


async def _send_webhook_event_async(event_data: Dict[str, Any]) -> bool:
    """
    Send a webhook directly for a usage event (async version).
    """
    # Get webhook configuration
    webhook_url, webhook_secret = _get_webhook_config()
    if not webhook_url:
        logger.debug("Skipping webhook send - URL not configured")
        return False

    try:
        # Prepare webhook payload structure for the receiving app
        payload = {
            "event_type": APIEventType.API_REQUEST.value,
            "log_id": event_data["id"],
            "user_id": event_data["user_id"],
            "api_key": event_data["api_key"],
            "endpoint": event_data["endpoint"],
            "dataset_name": event_data.get("dataset_name"),
            "processing_time_ms": event_data["processing_time_ms"],
            "status_code": event_data["status_code"],
            "timestamp": event_data["timestamp"],
        }

        # Generate JSON string for consistent signature calculation
        payload_str = json.dumps(payload)

        # Sign the webhook payload using HMAC SHA-256
        signature = hmac.new(
            webhook_secret.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Set up headers with signature
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Important: use data=payload_str to ensure consistent payload between signature and request
                async with session.post(
                    webhook_url,
                    data=payload_str,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15.0),
                ) as response:
                    if response.status == 200:
                        logger.debug(
                            f"Successfully sent API usage webhook for {event_data['user_id']}"
                        )
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(
                            f"Webhook sending failed: {response.status} - {response_text}"
                        )
                        return False
        except aiohttp.ClientError as e:
            logger.error(f"Network error sending webhook: {str(e)}")
            return False
    except Exception as e:
        logger.error(
            f"Error sending webhook for event ID: {event_data.get('id', 'N/A')}: {str(e)}"
        )
        return False


def _send_webhook_event(event_data: Dict[str, Any]) -> bool:
    """
    Send a webhook directly for a usage event (non-blocking wrapper around async version).

    Returns:
        bool: Always returns True to indicate the process has been started.
              Check logs for actual webhook sending status.
    """

    # Create a task to run the async version in a separate thread/event loop
    async def _run_async_webhook():
        try:
            return await _send_webhook_event_async(event_data)
        except Exception as e:
            logger.error(f"Error in async webhook execution: {str(e)}")
            return False

    # Run the async function in a new event loop without blocking
    def _run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_async_webhook())
        finally:
            loop.close()

    # Start the webhook sending process in a separate thread without blocking
    threading_thread = threading.Thread(target=_run_in_thread)
    threading_thread.daemon = (
        True  # Allow the program to exit even if this thread is running
    )
    threading_thread.start()

    # Return immediately without waiting for the webhook to be sent
    return True
