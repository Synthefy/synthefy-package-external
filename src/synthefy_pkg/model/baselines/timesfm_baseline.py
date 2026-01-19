import numpy as np
import timesfm
import torch

COMPILE = False


class TimesfmBaseline:
    def __init__(
        self, device, batch_size, pred_len, prepare_fn, use_timesfm2=False
    ):
        # Set the device to either cpu or gpu
        # set per core batch size
        # horizon_len = pred_lens
        backend = "gpu" if device == "cuda" else "cpu"

        if use_timesfm2:
            self.tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=batch_size,
                    horizon_len=pred_len,
                    # These values fixed based on: https://huggingface.co/google/timesfm-2.0-500m-pytorch
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=50,
                    model_dims=1280,
                    # Max context length for TimesFM-2 is 2048; set it here explicitly
                    context_len=2048,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
                ),
            )
        else:
            self.tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=batch_size,
                    horizon_len=pred_len,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                ),
            )

        self.device = device
        self.prepare_fn = prepare_fn
        self.batch_size = batch_size
        self.pred_len = pred_len

    def synthesis_function(self, batch, synthesizer):
        batch = self.prepare_fn(batch)

        effective_batch_size, hist_len, num_channels = batch["history"].shape

        # Reshape to (batch_size, num_channels, hist_len)
        # And flatten to (batch_size * num_channels, hist_len)
        forecast_input = (
            batch["history"].cpu().permute(0, 2, 1).reshape(-1, hist_len)
        )

        # Set the frequency to 0 for all batch * num_channels
        # Timesfm supports 3 frequencies. For now we fix the frequency to 0 for high frequency data
        frequency_input = [0] * len(forecast_input)

        point_forecast, experimental_quantile_forecast = self.tfm.forecast(
            forecast_input,
            freq=frequency_input,
        )

        # Timesfm processes each channel independently, meaning point_forecast is a 2d
        # array of shape (effective_batch_size * num_channels, pred_len)
        # Before concatenating with the history, we need to reshape this to (effective_batch_size, num_channels, pred_len)
        forecast = point_forecast.reshape(
            effective_batch_size, num_channels, self.pred_len
        )

        # Output needs to be in the shape (batch_dim, channel_dim, seq_len)
        # forecast is is (batch_dim, channel_dim, seq_len)
        # batch[history] is (batch_dim, seq_len, channel_dim)
        concatenated = np.concatenate(
            [batch["history"].cpu().numpy().transpose(0, 2, 1), forecast],
            axis=2,
        )

        return {
            "timeseries": concatenated,
            "discrete_conditions": batch["full_discrete_conditions"]
            .float()
            .cpu(),
            "continuous_conditions": batch["full_continuous_conditions"]
            .float()
            .cpu(),
        }
