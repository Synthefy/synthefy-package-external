import os

import requests
from loguru import logger

COMPILE = False
HUGGING_FACE_HUB_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")


def check_huggingface_access():
    try:
        headers = {}
        if HUGGING_FACE_HUB_TOKEN is not None:
            headers = {"Authorization": f"Bearer {HUGGING_FACE_HUB_TOKEN}"}
        response = requests.get("https://huggingface.co", headers=headers)
        if response.status_code == 200:
            logger.info(
                "Chronos/TimesFM access is available via huggingface.co"
            )

        else:
            raise ConnectionError("Access to huggingface.co is not available.")

    except requests.exceptions.RequestException:
        raise ConnectionError("Access to huggingface.co is not available.")
