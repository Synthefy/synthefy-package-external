# Disable logging for prophet
import logging
from multiprocessing import Pool
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from prophet import Prophet

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.baselines.chronos_baseline import (
    NUM_FORECASTS_PROBABILISTIC,
)

logging.getLogger("cmdstanpy").disabled = True

COMPILE = True


class ProphetBaseline:
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        batch_size: int,
        num_channels: int,
        use_probabilistic_forecast: bool,
        fmv2_prepare_fn: Optional[Callable] = None,
    ):
        self.seq_len: int = seq_len
        self.pred_len: int = pred_len

        self.date_range = pd.date_range(
            start="2020-01-01", periods=self.seq_len, freq="h"
        )

        self.batch_size: int = batch_size
        self.num_channels: int = num_channels
        self.use_probabilistic_forecast: bool = use_probabilistic_forecast
        self.fmv2_prepare_fn: Optional[Callable] = fmv2_prepare_fn

    def prepare_fn(self, batch: dict, *args, **kwargs) -> dict:
        assert batch["timeseries_full"].shape[2] == self.seq_len, (
            f"batch['timeseries_full'].shape[2]: {batch['timeseries_full'].shape[2]} != {self.seq_len}"
        )
        forecast_input = {
            "history": batch["timeseries_full"][:, :, : -self.pred_len]
            .cpu()
            .permute(0, 2, 1),
            "forecast": batch["timeseries_full"][:, :, -self.pred_len :]
            .cpu()
            .permute(0, 2, 1),
            "history_ds": self.date_range[: -self.pred_len],
            "forecast_ds": self.date_range[-self.pred_len :],
        }

        return forecast_input

    def _forecast_with_prophet(
        self, forecast_input: dict, batch_idx: int, channel_idx: int
    ) -> np.ndarray:
        """
        Function to apply across the timeseries axis.
        """
        # Forecast input is shape (batch_size, channel_dim, time_series_length)
        history = forecast_input["history"]
        history_ds = forecast_input["history_ds"]
        forecast_ds = forecast_input["forecast_ds"]

        history_ts = history[batch_idx, :, channel_idx]
        hist_df = pd.DataFrame(data={"ds": history_ds, "y": history_ts})

        if self.use_probabilistic_forecast:
            model = Prophet(
                uncertainty_samples=NUM_FORECASTS_PROBABILISTIC,
                interval_width=0.99,
            )
        else:
            model = Prophet()
        model.fit(hist_df)

        forecast_df = pd.DataFrame(data={"ds": forecast_ds})

        if self.use_probabilistic_forecast:
            posterior_samples = model.predictive_samples(forecast_df)
            # (pred_length, NUM_FORECASTS_PROBABILISTIC)
            return posterior_samples["yhat"]
        else:
            forecast = model.predict(forecast_df)
            # (pred_length,)
            return forecast["yhat"].to_numpy()

    def synthesis_function(self, batch: dict, synthesizer: Any) -> dict:
        if self.fmv2_prepare_fn is not None:
            forecast_input = self.fmv2_prepare_fn(batch)
            forecast_input["history_ds"] = self.date_range[: -self.pred_len]
            forecast_input["forecast_ds"] = self.date_range[-self.pred_len :]
        else:
            forecast_input = self.prepare_fn(batch)

        # We need to process the history as a numpy array
        forecast_input["history"] = forecast_input["history"].cpu().numpy()

        # In the last iteration, we may not get a full batch
        effective_batch_size = forecast_input["history"].shape[0]

        # Generate all indices for the first two dimensions
        inputs = [
            (forecast_input, i, j)
            for i in range(effective_batch_size)
            for j in range(self.num_channels)
        ]

        results = [self._forecast_with_prophet(*input) for input in inputs]

        # Step 1:
        # results is a list, with length batch_size * num_channels, where each element is an np.ndarray of length pred_len
        # Turn this into a single np.ndarray of shape (batch_dim, channel_dim, forecast_len)
        # or in the probabilistic case, shape (batch_dim, channel_dim, forecast_len, NUM_FORECASTS_PROBABILISTIC)

        # Step 2:
        # The history is shape (batch_dim, hist_len, channel_dim)
        # The output we need is the same shape as results, so we
        # transpose the history to shape (batch_dim, channel_dim, hist_len) and concatenate
        # along axis 2 (hist_len + forecast_len)
        # yielding shape (batch_dim, channel_dim, hist_len + forecast_len)
        if self.use_probabilistic_forecast:
            # Step 1:
            results = np.array(results).reshape(
                effective_batch_size, self.num_channels, self.pred_len, -1
            )
            assert results.shape[-1] == NUM_FORECASTS_PROBABILISTIC

            # Step 2:
            concatenated_results = np.concatenate(
                (
                    np.expand_dims(
                        forecast_input["history"].transpose(0, 2, 1), axis=-1
                    ).repeat(results.shape[-1], axis=-1),
                    results,
                ),
                axis=2,
            )
            concatenated_results = concatenated_results.transpose(0, 3, 1, 2)
        else:
            # Step 1:
            results = np.array(results).reshape(
                effective_batch_size, self.num_channels, self.pred_len
            )

            # Step 2:
            concatenated_results = np.concatenate(
                (forecast_input["history"].transpose(0, 2, 1), results), axis=2
            )

        if self.fmv2_prepare_fn is None:
            dataset_dict = {
                "timeseries": concatenated_results,
                "discrete_conditions": batch["discrete_label_embedding"]
                .float()
                .cpu(),
                "continuous_conditions": batch["continuous_label_embedding"]
                .float()
                .cpu(),
            }

        else:
            dataset_dict = {
                "timeseries": concatenated_results,
                "discrete_conditions": forecast_input[
                    "full_discrete_conditions"
                ]
                .float()
                .cpu(),
                "continuous_conditions": forecast_input[
                    "full_continuous_conditions"
                ]
                .float()
                .cpu(),
            }

        return dataset_dict
