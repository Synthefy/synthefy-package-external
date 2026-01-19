from typing import Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA as _AutoARIMA
from tqdm import tqdm

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.utils import truncate_eval_batch
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class AutoARIMAForecaster(BaseForecaster):
    """Forecaster wrapper around Nixtla's statsforecast AutoARIMA model."""

    def __init__(self):
        super().__init__("AutoARIMAForecaster")
        # One entry per (batch, correlate) pair – can be None if model not fitted
        self.models: List[Optional[Any]] = []
        self.fitted_sample_ids: set[str] = set()
        self.B = 0
        self.NC = 0

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_df(sample: SingleEvalSample) -> tuple[pd.DataFrame, str, str]:
        """Convert a `SingleEvalSample` to dataframe expected by StatsForecast."""
        item_id = str(sample.sample_id)
        ts = pd.to_datetime(sample.history_timestamps)
        # Infer frequency – fallback to daily
        freq = pd.infer_freq(ts)
        if freq is None:
            freq = "D"
        df = pd.DataFrame(
            {
                "unique_id": item_id,
                "ds": ts,
                "y": sample.history_values,
            }
        ).dropna(subset=["y"])
        return df, freq, item_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, batch: EvalBatchFormat):  # type: ignore[override]
        if StatsForecast is None or _AutoARIMA is None:
            raise ImportError(
                "statsforecast package is required for AutoARIMAForecaster but is not installed."
            )

        # batch = truncate_eval_batch(batch, max_context_length=334)

        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.models = []
        self.fitted_sample_ids = set()

        for i in tqdm(range(self.B), desc="Fitting au"):
            for j in range(self.NC):
                sample = batch[i, j]
                if not sample.forecast:
                    self.models.append(None)
                    continue

                df, freq, item_id = self._prepare_df(sample)
                if len(df) < 2:
                    logger.warning(
                        f"AutoARIMAForecaster: Not enough data to fit for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    self.models.append(None)
                    continue

                model = _AutoARIMA()  # default hyper-parameters
                sf = StatsForecast(models=[model], freq=freq, n_jobs=1)
                sf.fit(df)
                self.models.append(sf)
                self.fitted_sample_ids.add(item_id)
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:  # type: ignore[override]
        if not self.models:
            raise ValueError("AutoARIMAForecaster model not fitted yet")

        # batch = truncate_eval_batch(batch, max_context_length=334)

        B = batch.batch_size
        NC = batch.num_correlates
        idx = 0  # pointer into self.models
        forecasts: List[List[SingleSampleForecast]] = []

        for i in tqdm(range(B), desc="Predicting au"):
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
                model_wrapper = self.models[idx]
                if (
                    item_id not in self.fitted_sample_ids
                    or model_wrapper is None
                ):
                    logger.warning(
                        f"AutoARIMAForecaster: No fitted model for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    pred_vals = np.full(
                        sample.target_timestamps.shape, np.nan, dtype=np.float32
                    )
                else:
                    h = len(sample.target_timestamps)
                    pred_df = model_wrapper.predict(h=h)  # type: ignore[arg-type]
                    # column 2 holds the point forecasts (after unique_id, ds)
                    pred_vals = pred_df["AutoARIMA"].values.astype(np.float32)

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
