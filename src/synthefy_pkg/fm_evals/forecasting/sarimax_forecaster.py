import numpy as np
from loguru import logger
from pmdarima import auto_arima
from tqdm import tqdm

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.utils import (
    fill_nan_values,
    truncate_eval_batch,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class SARIMAXForecaster(BaseForecaster):
    def __init__(self, seasonal_period: int, future_leaked: bool = False):
        if future_leaked:
            super().__init__("SARIMAX Forecaster")
        else:
            super().__init__("SARIMA Forecaster")
        self.model = None
        self.future_leaked = future_leaked
        self.seasonal_period = seasonal_period

    def fit(self, batch: EvalBatchFormat):
        # Fitting is way too slow with too large of a context length
        batch = truncate_eval_batch(batch, max_context_length=500)

        B = batch.batch_size

        self.y_fit = []
        self.y_pred = []
        self.exog_fit = []
        self.exog_pred = []
        self.models = []

        for b in tqdm(
            range(B), desc=f"Fitting {self.name} models"
        ):  # For each element of the batch
            y_fit = []
            y_pred = []
            exog_fit = []
            exog_pred = []
            models = []  # We need one model per target

            for nc in range(batch.num_correlates):  # For each possible target
                if not batch[b, nc].forecast:
                    y_fit.append(None)
                    y_pred.append(None)
                    exog_fit.append(None)
                    exog_pred.append(None)
                    models.append(None)
                    continue

                # Build the target
                local_y_fit = fill_nan_values(batch[b, nc].history_values)
                local_y_pred = fill_nan_values(batch[b, nc].target_values)
                y_fit.append(local_y_fit)
                y_pred.append(local_y_pred)

                # Gather all exogenous variables that are not the target and allowed to be leaked
                local_exog_fit = [
                    fill_nan_values(batch[b, m_idx].history_values).reshape(
                        -1, 1
                    )
                    for m_idx in range(batch.num_correlates)
                    if m_idx != nc
                    and (self.future_leaked or batch[b, m_idx].leak_target)
                ]
                local_exog_pred = [
                    fill_nan_values(batch[b, m_idx].target_values).reshape(
                        -1, 1
                    )
                    for m_idx in range(batch.num_correlates)
                    if m_idx != nc
                    and (self.future_leaked or batch[b, m_idx].leak_target)
                ]

                if len(local_exog_fit) > 0:
                    local_exog_fit = np.concatenate(local_exog_fit, axis=1)
                    local_exog_pred = np.concatenate(local_exog_pred, axis=1)

                    exog_fit.append(local_exog_fit)
                    exog_pred.append(local_exog_pred)
                else:
                    exog_fit.append(None)
                    exog_pred.append(None)

                model = auto_arima(
                    y=local_y_fit,
                    X=local_exog_fit if len(local_exog_fit) > 0 else None,
                    seasonal=True,
                    m=self.seasonal_period,
                    stepwise=False,
                    trace=False,
                    error_action="ignore",
                    suppress_warnings=True,
                    # Reduced search space for faster fitting
                    max_p=1,  # Reduced from 5
                    max_q=1,  # Reduced from 5
                    max_d=1,  # Reduced from 2
                    max_P=1,  # Reduced from 2
                    max_Q=1,  # Reduced from 2
                    max_D=1,  # Reduced from 1
                    # Early stopping parameters
                    maxiter=50,  # Limit maximum iterations
                    seasonal_test="ch",  # Faster seasonal test
                    information_criterion="aic",  # Use AIC for model selection
                    start_p=0,
                    start_q=0,
                    start_P=0,
                    start_Q=0,
                    n_jobs=64,
                )

                models.append(model)

            self.y_fit.append(y_fit)
            self.y_pred.append(y_pred)
            self.exog_fit.append(exog_fit)
            self.exog_pred.append(exog_pred)
            self.models.append(models)
        return True

    def _predict(self, batch: EvalBatchFormat):
        B = batch.batch_size
        NC = batch.num_correlates

        all_samples = []
        for b in tqdm(range(B)):
            row = []
            for nc in range(NC):
                sample = batch[b, nc]
                if not batch[b, nc].forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    continue

                n_periods = len(sample.target_timestamps)

                y_pred = self.models[b][nc].predict(
                    n_periods=n_periods, X=self.exog_pred[b][nc]
                )

                row.append(
                    SingleSampleForecast(
                        sample_id=sample.sample_id,
                        timestamps=sample.target_timestamps,
                        values=y_pred,
                        model_name=self.name,
                    )
                )

            all_samples.append(row)

        return ForecastOutputFormat(all_samples)
