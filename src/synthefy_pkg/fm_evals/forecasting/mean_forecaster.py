import numpy as np

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.utils import fill_nan_values
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class MeanForecaster(BaseForecaster):
    def __init__(self):
        super().__init__("Mean Forecaster")

    def fit(self, batch: EvalBatchFormat):
        B = batch.batch_size
        NC = batch.num_correlates

        self.means = []

        for b in range(B):
            local_means = []
            for nc in range(NC):
                if not batch[b, nc].forecast:
                    local_means.append(None)
                    continue

                y_fit = fill_nan_values(batch[b, nc].history_values)
                local_means.append(np.nanmean(y_fit))

            self.means.append(local_means)

    def _predict(self, batch: EvalBatchFormat):
        B = batch.batch_size
        NC = batch.num_correlates

        all_forecasts = []
        for b in range(B):
            row = []
            for nc in range(NC):
                sample = batch[b, nc]
                if not sample.forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    continue

                row.append(
                    SingleSampleForecast(
                        sample_id=sample.sample_id,
                        timestamps=sample.target_timestamps,
                        values=self.means[b][nc]
                        * np.ones(sample.target_timestamps.shape),
                        model_name=self.name,
                    )
                )
            all_forecasts.append(row)
        return ForecastOutputFormat(all_forecasts)
