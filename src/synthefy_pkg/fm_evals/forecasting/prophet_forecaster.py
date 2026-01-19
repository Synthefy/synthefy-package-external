import numpy as np
import pandas as pd
from loguru import logger
from prophet import Prophet
from tqdm import tqdm

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


class ProphetForecaster(BaseForecaster):
    def __init__(self):
        super().__init__("ProphetForecaster")
        self.models = []  # One model per (batch, correlate) pair
        self.fitted_sample_ids = set()
        self.B = 0
        self.NC = 0

    def fit(self, batch: EvalBatchFormat):
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.models = []
        self.fitted_sample_ids = set()
        for i in tqdm(range(self.B), desc="Fitting pr"):
            for j in range(self.NC):
                sample = batch[i, j]
                if not sample.forecast:
                    self.models.append(None)
                    continue
                item_id = str(sample.sample_id)
                self.fitted_sample_ids.add(item_id)
                # Prepare data for Prophet (requires columns ds and y)
                train_df = pd.DataFrame(
                    {
                        "ds": pd.to_datetime(sample.history_timestamps),
                        "y": sample.history_values,
                    }
                )
                train_df = train_df.dropna(subset=["y"])
                if len(train_df) < 2:
                    logger.warning(
                        f"pr: Not enough data to fit for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    self.models.append(None)
                    continue
                model = Prophet(
                    yearly_seasonality="auto",
                    weekly_seasonality="auto",
                    daily_seasonality="auto",
                    seasonality_mode="multiplicative",
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    changepoint_range=0.8,
                )
                try:
                    model.fit(train_df)
                    self.models.append(model)
                except Exception as e:
                    logger.warning(
                        f"pr: Error fitting pr model for sample {i}, correlate {j} (sample_id={item_id}): {e}"
                    )
                    self.models.append(None)
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        if not self.models:
            raise ValueError("prForecaster model not fitted yet")
        B = batch.batch_size
        NC = batch.num_correlates
        T = batch.target_length
        idx = 0
        forecasts = []
        for i in tqdm(range(B), desc="Predicting pr"):
            row = []
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
                if (
                    item_id not in self.fitted_sample_ids
                    or self.models[idx] is None
                ):
                    logger.warning(
                        f"prForecaster: No fitted model for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    pred = np.full((T,), np.nan, dtype=np.float32)
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=sample.target_timestamps,
                            values=pred,
                            model_name=self.name,
                        )
                    )
                    idx += 1
                    continue
                model = self.models[idx]
                future_df = pd.DataFrame(
                    {"ds": pd.to_datetime(sample.target_timestamps)}
                )
                try:
                    forecast = model.predict(future_df)
                    pred = forecast["yhat"].values.astype(np.float32)
                except Exception as e:
                    logger.warning(
                        f"prForecaster: Error predicting for sample {i}, correlate {j} (sample_id={item_id}): {e}"
                    )
                    pred = np.full(
                        sample.target_timestamps.shape, np.nan, dtype=np.float32
                    )
                row.append(
                    SingleSampleForecast(
                        sample_id=sample.sample_id,
                        timestamps=sample.target_timestamps,
                        values=pred,
                        model_name=self.name,
                    )
                )
                idx += 1
            forecasts.append(row)
        return ForecastOutputFormat(forecasts)
