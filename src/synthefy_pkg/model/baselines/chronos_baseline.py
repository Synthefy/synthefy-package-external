import os
from typing import Any, Callable

import torch
from chronos import ChronosPipeline

COMPILE = False

NUM_FORECASTS_PROBABILISTIC = 100
NUM_FORECASTS_DETERMINISTIC = 1

HUGGING_FACE_HUB_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")


class ChronosBaseline:
    def __init__(
        self,
        device: str,
        pred_len: int,
        prepare_fn: Callable,
        use_probabilistic_forecast: bool,
    ):
        self.device: str = device
        self.chronos_pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            use_auth_token=HUGGING_FACE_HUB_TOKEN,
        )
        self.pred_len: int = pred_len
        self.prepare_fn: Callable = prepare_fn
        self.use_probabilistic_forecast: bool = use_probabilistic_forecast

    def synthesis_function(self, batch: dict, synthesizer: Any):
        batch = self.prepare_fn(batch)
        batch_size, hist_len, num_channels = batch["history"].shape

        # (batch_dim, hist_len, num_channels) > (batch_dim, num_channels, hist_len)
        hist_data = batch["history"].cpu().permute(0, 2, 1)
        # (batch_dim, num_channels, hist_len) > (batch_dim * num_channels, hist_len)
        hist_data_reshaped = hist_data.reshape(-1, hist_len)

        num_samples = (
            NUM_FORECASTS_PROBABILISTIC
            if self.use_probabilistic_forecast
            else NUM_FORECASTS_DETERMINISTIC
        )
        forecast = self.chronos_pipeline.predict(
            context=hist_data_reshaped,
            prediction_length=self.pred_len,
            num_samples=num_samples,
        )

        forecast = forecast.to(self.device)

        # If we are using a probabilistic forecast, we need to reshape the forecast and hist_data to match the shape of the forecast
        if self.use_probabilistic_forecast:
            # (batch_dim * num_channels, num_samples, pred_len) > (batch_dim * num_channels, pred_len, num_samples)
            forecast = forecast.permute(0, 2, 1)
            # (batch_dim * num_channels, pred_len, num_samples) > (batch_dim, num_channels, pred_len, num_samples)
            forecast = forecast.reshape(
                batch_size, num_channels, self.pred_len, -1
            )
            # (batch_dim, num_channels, pred_len, num_samples) > (batch_dim, num_samples, num_channels, pred_len)
            forecast = forecast.permute(0, 3, 1, 2)

            # Note: using the original hist_data, and expanding it (unsqueeze and repeat) to match the shape of forecast
            # (batch_dim, num_channels, hist_len) > (batch_dim, num_samples, num_channels, hist_len)
            hist_data = (
                hist_data.to(self.device)
                .unsqueeze(1)
                .repeat(1, NUM_FORECASTS_PROBABILISTIC, 1, 1)
            )

            # (batch_dim, num_samples, num_channels, hist_len) + (batch_dim, num_samples, num_channels, pred_len) > (batch_dim, num_samples, num_channels, hist_len + pred_len)
            concatenated = torch.cat((hist_data, forecast), dim=-1)
        else:
            # (batch_dim * num_channels, pred_len)
            forecast = torch.squeeze(forecast, dim=1)
            # (batch_dim * num_channels, pred_len) > (batch_dim, num_channels, pred_len)
            forecast = forecast.reshape(batch_size, num_channels, self.pred_len)

            hist_data = hist_data.to(self.device)

            # (batch_dim, num_channels, hist_len) + (batch_dim, num_channels, pred_len) > (batch_dim, num_channels, hist_len + pred_len)
            concatenated = torch.cat((hist_data, forecast), dim=-1)

        return {
            "timeseries": concatenated.cpu().numpy(),
            "discrete_conditions": batch["full_discrete_conditions"]
            .cpu()
            .numpy(),
            "continuous_conditions": batch["full_continuous_conditions"]
            .cpu()
            .numpy(),
        }
