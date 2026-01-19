from datetime import datetime, timezone

import requests
from loguru import logger

COMPILE = True


def check_codebase_expired():
    # Define the threshold date
    threshold_date = datetime(2025, 3, 1, tzinfo=timezone.utc)

    # Query a time server for the current time
    try:
        response = requests.get("http://worldtimeapi.org/api/timezone/Etc/UTC")
        data = response.json()
        current_datetime = datetime.fromisoformat(data["utc_datetime"])
    except:
        logger.error("Failed to query time server. Using local time.")
        current_datetime = datetime.now(timezone.utc)

    # Check if the current date is later than Nov 1, 2024
    if current_datetime > threshold_date:
        raise PermissionError(
            f"This codebase expired on {threshold_date.strftime('%b %d, %Y')} UTC."
        )
    else:
        logger.warning(
            f"This codebase will expire on {threshold_date.strftime('%b %d, %Y')} UTC. Current time is {current_datetime}."
        )
