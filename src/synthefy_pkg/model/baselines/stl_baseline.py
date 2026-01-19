import warnings
from multiprocessing import Pool
from typing import Callable, Optional

import numpy as np
import torch
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

warnings.simplefilter("ignore", ConvergenceWarning)

COMPILE = True


class STLBaseline:
    def __init__(
        self,
        seq_len,
        pred_len,
        batch_size,
        num_channels,
        fmv2_prepare_fn: Optional[Callable] = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.fmv2_prepare_fn: Optional[Callable] = fmv2_prepare_fn

    def prepare_fn(self, batch, *args, **kwargs):
        forecast_input = {
            "history": batch["timeseries_full"][:, :, : -self.pred_len]
            .cpu()
            .permute(0, 2, 1),
            "forecast": batch["timeseries_full"][:, :, -self.pred_len :]
            .cpu()
            .permute(0, 2, 1),
        }

        return forecast_input

    def _forecast_with_stl(self, forecast_input, batch_idx, channel_idx):
        """
        Function to apply across the timeseries axis.
        """
        # Forecast input is shape (batch_size, timeseries_length, channel_idx)
        history = forecast_input["history"]

        history_ts = history[batch_idx, :, channel_idx]

        # Order of ARIMA selects order of autoregressive, difference, and moving
        # Average components
        stlf = STLForecast(
            history_ts,
            ARIMA,
            period=self.seq_len + self.pred_len,
            model_kwargs=dict(order=(1, 1, 0), trend="t"),
        )
        stlf_res = stlf.fit()

        forecast = stlf_res.forecast(self.pred_len)

        return forecast

    def synthesis_function(self, batch, synthesizer):
        if self.fmv2_prepare_fn is not None:
            forecast_input = self.fmv2_prepare_fn(batch)
        else:
            forecast_input = self.prepare_fn(batch)

        # We need to process the history as a numpy array
        for key, value in forecast_input.items():
            if isinstance(value, torch.Tensor):
                forecast_input[key] = value.cpu()

        forecast_input["history"] = forecast_input["history"].cpu().numpy()

        # In the last iteration, we may not get a full batch
        effective_batch_size = forecast_input["history"].shape[0]

        # Generate all indices for the first two dimensions
        inputs = [
            (forecast_input, i, j)
            for i in range(effective_batch_size)
            for j in range(self.num_channels)
        ]

        results = [self._forecast_with_stl(*input) for input in inputs]

        # results is a list of length batch_size * num_channels of np arrays of length pred_len
        # We need shape (batch_dim, channel_dim, forecast_len)
        results = np.array(results).reshape(
            effective_batch_size, self.num_channels, self.pred_len
        )

        # Results is shape (batch_dim, channel_dim, forecast_len)
        # the history is shape (batch_dim, seq_len, channel_dim)
        # The output we need is the same shape as results, so we
        # Transpose the history to shape (batch_dim, channel_dim, seq_len) and concatenate
        # along axis 2 with results.
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
