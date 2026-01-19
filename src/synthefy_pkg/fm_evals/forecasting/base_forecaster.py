from abc import ABC, abstractmethod

import numpy as np

from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)
from synthefy_pkg.fm_evals.formats.metrics import (
    SUPPORTED_METRICS,
    ForecastMetrics,
)
from synthefy_pkg.fm_evals.metrics.compute_metrics import compute_sample_metrics

COMPILE = True


class BaseForecaster(ABC):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def fit(self, batch: EvalBatchFormat) -> bool:
        pass

    @abstractmethod
    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """Protected method that subclasses must implement for actual prediction logic."""
        pass

    def predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """
        Public predict method that automatically computes metrics after prediction.
        Subclasses should not override this method.
        """
        # Call the subclass's prediction logic
        forecast_output = self._predict(batch)

        # Automatically compute metrics on the forecast output
        forecast_output = self._compute_metrics(batch, forecast_output)

        return forecast_output

    @staticmethod
    def _compute_metrics(
        batch: EvalBatchFormat, forecast_output: ForecastOutputFormat
    ):
        """
        Compute metrics on the forecast output.
        Can be overridden by subclasses to customize metric computation.
        """
        # Initialize collection dictionaries for each metric
        metric_collections = {name: [] for name in SUPPORTED_METRICS}

        for b in range(batch.batch_size):
            for nc in range(batch.num_correlates):
                eval_sample = batch[b, nc]
                if not eval_sample.forecast:
                    continue

                forecast_sample = forecast_output.forecasts[b][nc]
                if isinstance(forecast_sample, list):
                    continue

                # Compute metrics for this sample
                sample_metrics = compute_sample_metrics(
                    eval_sample, forecast_sample
                )

                # Assign metrics to the forecast sample
                forecast_sample.metrics = sample_metrics

                # Collect values for batch-level aggregation
                for metric_name in SUPPORTED_METRICS:
                    value = getattr(sample_metrics, metric_name)
                    if value is not None and not np.isnan(value):
                        metric_collections[metric_name].append(value)

        # Compute batch-level aggregated metrics
        batch_metrics_dict = {}

        for metric_name in SUPPORTED_METRICS:
            values = metric_collections[metric_name]
            if values:
                # For median metrics, use median; for others, use mean
                if metric_name.startswith("median_"):
                    agg_value = float(np.nanmedian(values))
                else:
                    agg_value = float(np.nanmean(values))
            else:
                agg_value = float("nan")

            batch_metrics_dict[metric_name] = agg_value

        # Create and assign batch-level metrics
        batch_metrics = ForecastMetrics(
            sample_id="batch_aggregated", **batch_metrics_dict
        )
        forecast_output.metrics = batch_metrics

        return forecast_output
