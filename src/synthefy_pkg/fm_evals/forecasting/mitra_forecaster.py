from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from loguru import logger

from synthefy_pkg.fm_evals.forecasting.api.transport import (
    eval_batch_to_request_payload,
    response_to_forecast_output,
)
from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)

COMPILE = True


class MitraForecaster(BaseForecaster):
    """Skeleton Mitra forecaster that operates on EvalBatchFormat and returns
    ForecastOutputFormat.

    Replace the constructor and prediction implementation with your model loading
    and inference logic. Follows the BaseForecaster fit/_predict interface.
    """

    def __init__(
        self,
        multivariate: bool = False,
        future_leak: bool = False,
    ) -> None:
        super().__init__(name="Mitra Forecaster")
        # Base URL for Mitra server from env var
        env_url = os.getenv("MODEL_M_SERVER_URL", "http://localhost:8001")
        self.base_url = env_url.rstrip("/")

        self.multivariate = multivariate
        self.future_leak = future_leak

        self._setup_mitra_model()

        logger.info(
            "Initialized mForecaster name={} base_url={}",
            self.get_name(),
            self.base_url,
        )
        # TODO: Load model/weights here later

    def _setup_mitra_model(self) -> None:
        """Configure the remote Mitra server via POST /setup.

        Sends the current configuration flags to the server so it can
        initialize the forecaster accordingly.
        """
        url = f"{self.base_url}/setup"
        payload = {
            "metadata": bool(self.multivariate),
            "future_leak": bool(self.future_leak),
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                logger.info("m server setup OK: {}", resp.json())
        except Exception as exc:
            raise RuntimeError(
                f"m server setup failed: {type(exc).__name__}: {exc}"
            ) from exc

    def fit(self, batch: EvalBatchFormat) -> bool:
        """Optional training/fine-tuning step.

        Implement if your model requires fitting; otherwise you can keep this as a no-op
        that returns True to indicate readiness.
        """
        # TODO: Implement training or adapter loading if needed
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """Call the Mitra server with the full batch and return its forecasts."""
        # Build request payload using shared marshalling helper
        req_json = eval_batch_to_request_payload(batch)

        url = f"{self.base_url}/forecast"
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=req_json)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"m server request failed: {type(exc).__name__}: {exc}"
            ) from exc

        # Parse response using shared marshalling helper
        return response_to_forecast_output(data)
