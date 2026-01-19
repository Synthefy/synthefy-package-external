import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from synthefy_pkg.fm_evals import eval_utils as eu
from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class TimeSeriesPrediction(BaseModel):
    """Time series prediction model."""

    explanation: str = Field(
        description="Explanation of the seasonality, trend, and other patterns in the history that helps forecast the future"
    )
    future_values: List[float] = Field(
        description="List of future values. The length of the list should be equal to the forecast_length."
    )


class LLMForecaster(BaseForecaster, ABC):
    """Forecaster wrapper around LLM-based time series forecasting."""

    def __init__(
        self,
        temperature: float = 0.0,
        max_retries: int = 2,
        name: str = "LLMForecaster",
        **kwargs,
    ):
        super().__init__(name)
        self.temperature = temperature
        self.max_retries = max_retries
        self.kwargs = kwargs

        # Initialize LLM based on model name
        self.llm = self._initialize_llm()

        # Initialize output parser
        self.output_parser = PydanticOutputParser(
            pydantic_object=TimeSeriesPrediction
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert analyst of time series data and trends. Given the history, predict future values.\n{format_instructions}",
                ),
                (
                    "human",
                    "forecast_length: {forecast_length}\nhistory: {history}",
                ),
            ]
        )

        # One entry per (batch, correlate) pair – stores the fitted history data
        self.histories: List[Optional[List[float]]] = []
        self.fitted_sample_ids: set[str] = set()
        self.B = 0
        self.NC = 0

    @abstractmethod
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the appropriate LLM. Must be implemented by subclasses."""
        pass

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_history(sample: SingleEvalSample) -> List[float]:
        """Extract history values from a `SingleEvalSample`."""
        return sample.history_values.tolist()

    def _forecast_time_series(
        self, history: List[float], forecast_length: int
    ) -> TimeSeriesPrediction:
        """Forecast time series values based on historical data."""
        chain = self.prompt | self.llm | self.output_parser

        try:
            result = chain.invoke(
                {
                    "history": history,
                    "forecast_length": forecast_length,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            )

            # Ensure we have the right number of forecast values
            if len(result.future_values) < forecast_length:
                # Pad with the last value if we don't have enough
                last_value = (
                    result.future_values[-1]
                    if result.future_values
                    else history[-1]
                    if history
                    else 0.0
                )
                result.future_values.extend(
                    [last_value] * (forecast_length - len(result.future_values))
                )
            elif len(result.future_values) > forecast_length:
                # Truncate if we have too many
                result.future_values = result.future_values[:forecast_length]

            return result
        except Exception as e:
            logger.warning(f"LLM forecasting error: {e}")
            # Fallback: use simple trend continuation
            if len(history) >= 2:
                trend = (history[-1] - history[-2]) / len(history)
                fallback_values = [
                    history[-1] + trend * (i + 1)
                    for i in range(forecast_length)
                ]
            else:
                fallback_values = [
                    history[-1] if history else 0.0
                ] * forecast_length

            return TimeSeriesPrediction(
                explanation="Fallback prediction due to LLM error",
                future_values=fallback_values,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, batch: EvalBatchFormat) -> bool:  # type: ignore[override]
        """Store history data for each sample in the batch."""
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.histories = []
        self.fitted_sample_ids = set()

        for i in tqdm(range(self.B), desc="Fitting ll"):
            for j in range(self.NC):
                sample = batch[i, j]
                if not sample.forecast:
                    self.histories.append(None)
                    continue

                history = self._prepare_history(sample)
                if len(history) < 2:
                    logger.warning(
                        f"LLMForecaster: Not enough data to fit for sample {i}, correlate {j} (sample_id={sample.sample_id})"
                    )
                    self.histories.append(None)
                    continue

                self.histories.append(history)
                self.fitted_sample_ids.add(str(sample.sample_id))

        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:  # type: ignore[override]
        """Generate forecasts using the fitted LLM model."""
        if not self.histories:
            raise ValueError("LLMForecaster model not fitted yet")

        B = batch.batch_size
        NC = batch.num_correlates
        idx = 0  # pointer into self.histories
        forecasts: List[List[SingleSampleForecast]] = []

        for i in tqdm(range(B), desc="Predicting ll"):
            row: List[SingleSampleForecast] = []
            for j in range(NC):
                sample = batch[i, j]

                if not sample.forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    idx += 1
                    continue

                item_id = str(sample.sample_id)
                history = self.histories[idx]

                if item_id not in self.fitted_sample_ids or history is None:
                    logger.warning(
                        f"LLMForecaster: No fitted history for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    pred_vals = np.full(
                        sample.target_timestamps.shape, np.nan, dtype=np.float32
                    )
                else:
                    forecast_length = len(sample.target_timestamps)
                    prediction = self._forecast_time_series(
                        history, forecast_length
                    )
                    pred_vals = np.array(
                        prediction.future_values, dtype=np.float32
                    )

                row.append(
                    SingleSampleForecast(
                        sample_id=sample.sample_id,
                        timestamps=sample.target_timestamps,
                        values=pred_vals,
                        model_name=self.name,
                    )
                )
                idx += 1
            forecasts.append(row)
        return ForecastOutputFormat(forecasts)


class GeminiForecaster(LLMForecaster):
    """Concrete implementation of LLMForecaster for Google Gemini models."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_retries: int = 2,
        **kwargs,
    ):
        self.model_name = model_name
        display_name = eu.MAP_NAME_TO_DISPLAY_NAME[model_name]
        super().__init__(
            temperature=temperature,
            max_retries=max_retries,
            name=display_name,
            **kwargs,
        )

    def _initialize_llm(self) -> BaseChatModel:
        """Initialize Google Generative AI (Gemini) model."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini provider"
            )

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_retries=self.max_retries,
            google_api_key=api_key,
            **self.kwargs,
        )


class AzureOpenAIForecaster(LLMForecaster):
    """Concrete implementation of LLMForecaster for Azure OpenAI models."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_retries: int = 2,
        **kwargs,
    ):
        self.model_name = model_name
        display_name = eu.MAP_NAME_TO_DISPLAY_NAME[model_name]
        super().__init__(
            temperature=temperature,
            max_retries=max_retries,
            name=display_name,
            **kwargs,
        )

    def _initialize_llm(self) -> BaseChatModel:
        """Initialize Azure OpenAI model."""
        # Get configuration from kwargs or environment variables
        api_key = self.kwargs.get("openai_api_key") or os.getenv(
            "AZURE_OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY environment variable or openai_api_key parameter is required for Azure OpenAI provider"
            )

        return AzureChatOpenAI(
            azure_endpoint="https://synthefy-synthesis.openai.azure.com/",
            api_key=api_key,  # type: ignore
            model=self.model_name,
            azure_deployment="SynthefyGPT4o",
            api_version="2024-02-15-preview",
            temperature=self.temperature,
            max_retries=self.max_retries,
        )
