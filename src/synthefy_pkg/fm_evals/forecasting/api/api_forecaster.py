import os
from functools import partial
from typing import Optional

import httpx
from loguru import logger
from synthefy.data_models import ForecastV2Request, ForecastV2Response

from synthefy_pkg.fm_evals.forecasting.api.transport import (
    eval_batch_to_request_payload,
    forecasts_to_response,
    payload_to_eval_batch,
    response_to_forecast_output,
)
from synthefy_pkg.fm_evals.forecasting.autoarima_forecaster import (
    AutoARIMAForecaster,
)
from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.boosting_forecaster import (
    TabPFNBoostingForecaster,
)
from synthefy_pkg.fm_evals.forecasting.mitra_forecaster import (
    MitraForecaster,
)
from synthefy_pkg.fm_evals.forecasting.prophet_forecaster import (
    ProphetForecaster,
)
from synthefy_pkg.fm_evals.forecasting.routing_forecaster import (
    RoutingForecaster,
)
from synthefy_pkg.fm_evals.forecasting.sarimax_forecaster import (
    SARIMAXForecaster,
)
from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
    TabPFNMultivariateForecaster,
    TabPFNUnivariateForecaster,
)
from synthefy_pkg.fm_evals.forecasting.toto_forecaster import (
    TotoForecaster,
    TotoUnivariateForecaster,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)

COMPILE = True
DEFAULT_FORECASTING_API_PORT = 8018
SUPPORTED_MODELS = {
    "synthefy_tuv": partial(TabPFNUnivariateForecaster),
    "synthefy_tmv": partial(
        TabPFNMultivariateForecaster,
        future_leak=False,
        individual_correlate_timestamps=False,
        num_covariate_lags=0,
    ),
    "sfm-moe-v0": partial(TabPFNBoostingForecaster, boosting_models=["toto"]),
    "sfm-moe-v1": partial(
        RoutingForecaster,
        model_options=[
            "tabpfn_multivariate",
            "tabpfn_multivariate_with_lags",
            "tabpfn_boosting_on_toto",
            "tabpfn_boosting_on_prophet",
            "chronos",
            "prophet",
            "toto_univariate",
            "toto_multivariate",
        ],
    ),
    "sfm-moe-v2": partial(
        RoutingForecaster,
        model_options=[
            "tabpfn_multivariate",
            "tabpfn_multivariate_with_lags",
            "tabpfn_boosting_on_toto",
            "tabpfn_boosting_on_prophet",
            "chronos",
            "prophet",
            "toto_univariate",
            "toto_multivariate",
            "mitra_boosting_on_toto",
            "mitra_boosting_on_prophet",
            "mitra_boosting_on_chronos",
        ],
    ),
    "synthefy_ttu": partial(TotoUnivariateForecaster),
    "synthefy_ttm": partial(TotoForecaster),
    "synthefy_mtu": partial(
        MitraForecaster, multivariate=False, future_leak=False
    ),
    "synthefy_mtm": partial(
        MitraForecaster, multivariate=True, future_leak=False
    ),
    "synthefy_mtf": partial(
        MitraForecaster, multivariate=True, future_leak=True
    ),
    "arima_univariate": partial(AutoARIMAForecaster),
    # Note: arima_multivariate is handled as a special case in _initialize_forecaster
    # because it requires seasonal_period to be passed at initialization
    "prophet_univariate": partial(ProphetForecaster),
}


class ForecastV2APIAdapter:
    """
    API Forecaster that acts as a wrapper/adapter for fm-evals forecasting models.

    This class handles the conversion between API request/response formats and
    the internal EvalBatchFormat used by fm-evals models.
    """

    def __init__(self, model_name: str, *args, **kwargs):
        """
        Initialize the API Forecaster with a specific model.

        Args:
            model_name: Name of the forecasting model to use
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments (e.g., seasonal_period for arima_multivariate)
        """
        self.model_name = model_name
        self.forecaster: Optional[BaseForecaster] = None
        self._initialize_forecaster(model_name, *args, **kwargs)

    def _initialize_forecaster(self, model_name: str, *args, **kwargs) -> None:
        """
        Initialize the appropriate forecaster based on the model name.

        This method should be implemented to handle different model types.
        For now, it's a placeholder that logs the model name.
        """
        logger.info(
            f"Attempting Initializing forecaster for model: {self.model_name}"
        )

        # if model_name == "arima_multivariate":
        #     if "seasonal_period" not in kwargs:
        #         raise ValueError(
        #             "seasonal_period is required for arima_multivariate model"
        #         )
        #     # TODO: Consider supporting more kwargs in the future
        #     self.forecaster = SARIMAXForecaster(
        #         seasonal_period=kwargs["seasonal_period"]
        #     )
        #     return

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported")

        # Only models that expect kwargs should be passed them
        # For now, only arima_multivariate needs kwargs
        self.forecaster = SUPPORTED_MODELS[model_name]()

    def predict(self, request: ForecastV2Request) -> ForecastV2Response:
        """
        Generate forecasts using the initialized model.

        Args:
            request: ForecasV2tRequest containing the input data

        Returns:
            ForecastResponse containing the generated forecasts

        Raises:
            RuntimeError: If the forecaster is not properly initialized
            ValueError: If the request data is invalid
        """
        if self.forecaster is None:
            raise RuntimeError(
                f"Forecaster for model {self.model_name} is not initialized"
            )

        try:
            # Step 1: Convert API request to EvalBatchFormat
            logger.debug(
                f"Converting request to EvalBatchFormat for {len(request.samples)} samples"
            )
            eval_batch = payload_to_eval_batch(request.samples)

            # Step 2: Call the forecasting model's fit() and _predict() methods
            logger.debug(f"Fitting and predicting with model {self.model_name}")
            self.forecaster.fit(eval_batch)
            predictions = self.forecaster._predict(eval_batch)

            # Step 3: Convert predictions back to API response format
            logger.debug("Converting predictions to API response format")
            response = forecasts_to_response(predictions, self.model_name)

            return response

        except Exception as e:
            logger.error(f"Prediction failed for model {self.model_name}: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")


class APIForecaster(BaseForecaster):
    """
    Forecaster that calls a remote forecasting API.

    This allows running evaluations against deployed forecasting services,
    enabling comparison between API-served models and local models.
    """

    def __init__(
        self,
        model_name: str,
        server_url: Optional[str] = None,
        timeout: float = 600.0,
    ):
        """
        Initialize the APIForecaster.

        Args:
            model_name: The model alias to use when calling the API
            server_url: Base URL of the forecasting API
                (e.g., "http://localhost:8018").
                If not provided, constructs local URL using FORECASTING_API_PORT
                (default: DEFAULT_FORECASTING_API_PORT).
            timeout: Request timeout in seconds. Default is 600s (10 minutes).
        """
        super().__init__(f"api_{model_name}")
        self.model_name = model_name
        # Use FORECASTING_API_PORT for local development, matching pattern used elsewhere
        port = os.getenv(
            "FORECASTING_API_PORT", str(DEFAULT_FORECASTING_API_PORT)
        )
        local_url = f"http://localhost:{port}"
        self.server_url = server_url or local_url
        self.timeout = timeout

        # Construct the forecast endpoint URL
        self.forecast_url = f"{self.server_url.rstrip('/')}/v2/forecast"

        logger.info(
            f"APIForecaster initialized: model={model_name}, "
            f"url={self.forecast_url}"
        )

        # Perform health check
        self._health_check()

    def _health_check(self) -> None:
        """Check if the API is reachable."""
        health_url = f"{self.server_url.rstrip('/')}/healthz"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(health_url)
                resp.raise_for_status()
                logger.info(f"API health check passed: {health_url}")
        except httpx.HTTPError as e:
            logger.warning(
                f"API health check failed: {health_url} - {e}. "
                "Proceeding anyway, but requests may fail."
            )

    def fit(self, batch: EvalBatchFormat) -> bool:
        """
        Fit method for API forecaster.

        Since the API handles model state internally, this is a no-op
        that always returns True.

        Args:
            batch: Batch of evaluation samples (not used)

        Returns:
            True always
        """
        # API forecasters don't require local fitting
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """
        Generate forecasts by calling the remote API.

        Args:
            batch: Batch of evaluation samples

        Returns:
            Forecast output with predictions from the API
        """
        # Convert EvalBatchFormat to API request payload
        payload = eval_batch_to_request_payload(batch)
        payload["model"] = self.model_name

        logger.debug(
            f"Calling API: {self.forecast_url} with model={self.model_name}, "
            f"batch_size={batch.batch_size}, num_correlates={batch.num_correlates}"
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.forecast_url, json=payload)
                resp.raise_for_status()
                response_json = resp.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"API request failed with status {e.response.status_code}: "
                f"{e.response.text}"
            )
            raise RuntimeError(
                f"API request failed: {e.response.status_code} - "
                f"{e.response.text}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"API request error: {e}")
            raise RuntimeError(f"API request error: {e}") from e

        # Convert API response to ForecastOutputFormat
        forecast_output = response_to_forecast_output(response_json)

        logger.debug(
            f"API returned {len(forecast_output.forecasts)} forecast rows"
        )

        return forecast_output
