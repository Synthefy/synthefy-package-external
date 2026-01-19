import json
import os
from enum import Enum, auto

import torch
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.experiments.synthesis_experiment import SynthesisExperiment
from synthefy_pkg.model.baselines.chronos_baseline import (
    NUM_FORECASTS_DETERMINISTIC,
)
from synthefy_pkg.utils.basic_utils import ENDC, OKYELLOW
from synthefy_pkg.utils.synthesis_utils import load_synthesis_model

COMPILE = True

# For an unkown reason, setting this to 100 or 128 causes the model to crash (not CUDA OOM)
# 64 sometimes causes CUDA OOMs.
NUM_FORECASTS_PROBABILISTIC = 32
SYNTHEFY_DATASETS_BASE = (
    os.getenv("SYNTHEFY_DATASETS_BASE") or ""
)  # Ensure string type for path joining


class MetadataStrategy(Enum):
    """Strategy for handling metadata in the forecast window

    NAIVE : Leave GT metadata in the forecast window
    DELETE : Delete the metadata in the forecast window
    REPEAT : Copy forward the last metadata step from history into the forecast window
    REPEAT_WINDOW : Copy forward the last metadata window from the history into the forecast window
    """

    NAIVE = auto()
    DELETE = auto()
    REPEAT = auto()
    REPEAT_WINDOW = auto()


class ForecastViaDiffusionBaseline:
    def __init__(
        self,
        config,
        model_checkpoint_path,
        use_probabilistic_forecast: bool,
        metadata_strategy: MetadataStrategy = MetadataStrategy.NAIVE,
        require_time_invariant_metadata: bool = True,
    ):
        """
        Args:
            config: Configuration object containing model and training parameters
            model_checkpoint_path: Path to the trained model checkpoint
            use_probabilistic_forecast: Whether to generate probabilistic forecasts
            metadata_strategy: Strategy for handling metadata in the forecast window
            require_time_invariant_metadata: If True, validates that metadata values remain constant
                across the entire time window. This is important to avoid information leakage.
                Should be used only in conjuction with metadata_strategy=MetadataStrategy.NAIVE
        """
        if not SYNTHEFY_DATASETS_BASE:
            raise ValueError(
                "SYNTHEFY_DATASETS_BASE environment variable must be set"
            )

        experiment = SynthesisExperiment(config)

        model, _, _ = load_synthesis_model(
            experiment.configuration, model_checkpoint_path
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.synthesizer_model = (
            model.denoiser_model
        )  # typically DiffusionTransformer
        self.use_probabilistic_forecast: bool = use_probabilistic_forecast

        self.history_length = (
            experiment.configuration.dataset_config.time_series_length
            - experiment.configuration.dataset_config.forecast_length
        )

        self.metadata_strategy = metadata_strategy
        self.require_time_invariant_metadata = require_time_invariant_metadata

        path = os.path.join(
            SYNTHEFY_DATASETS_BASE,
            experiment.configuration.dataset_name,
            "colnames.json",
        )
        assert os.path.exists(path), (
            f"Could not find {path}; try rerunning preprocessing."
        )
        with open(path, "r") as f:
            colnames = json.load(f)

        self.relevant_cols_idxs = [
            index
            for index, c in enumerate(colnames["continuous_colnames"])
            if c.startswith("timestamps_")
        ]
        logger.info(f"{colnames=}")

    def synthesis_function(
        self,
        batch,
        synthesizer,
    ):
        if self.use_probabilistic_forecast:
            num_forecasts = NUM_FORECASTS_PROBABILISTIC
        else:
            num_forecasts = NUM_FORECASTS_DETERMINISTIC

        T, Alpha, Alpha_bar, Sigma = (
            self.synthesizer_model.diffusion_hyperparameters["T"],
            self.synthesizer_model.diffusion_hyperparameters["Alpha"],
            self.synthesizer_model.diffusion_hyperparameters["Alpha_bar"],
            self.synthesizer_model.diffusion_hyperparameters["Sigma"],
        )
        device = self.synthesizer_model.device
        Alpha = Alpha.to(device)
        Alpha_bar = Alpha_bar.to(device)
        Sigma = Sigma.to(device)

        input_ = self.synthesizer_model.prepare_training_input(batch)

        # (batch, time_series_len, num_discrete_conditions)
        discrete_cond_input = input_["discrete_cond_input"]

        # (batch, time_series_len, num_continuous_labels)
        continuous_cond_input = input_["continuous_cond_input"]

        # (batch, channels, time_series_len)
        sample = input_["sample"]
        NUM_BATCHES, NUM_CHANNELS, TIME_SERIES_LEN = sample.shape

        x = torch.randn_like(sample).to(device)

        # (num_forecasts * batch_dim, channels, forecast_len)
        history = sample[:, :, : self.history_length]

        logger.info(
            OKYELLOW
            + f"Using Metadata Strategy {self.metadata_strategy=}"
            + ENDC
        )

        continuous_conditions_non_timestamp_cols_idxs = [
            i
            for i in range(continuous_cond_input.shape[2])
            if i not in self.relevant_cols_idxs
        ]

        if self.metadata_strategy is MetadataStrategy.NAIVE:
            # Keep metadata as-is, no modification needed
            if self.require_time_invariant_metadata:
                # Assert discrete conditions are time invariant (same across all timesteps)
                # discrete_cond_input shape: (batch, sequence_len, num_discrete_condtions)
                first_timestep_discrete = discrete_cond_input[
                    :, 0, :
                ].unsqueeze(1)
                assert torch.all(
                    discrete_cond_input == first_timestep_discrete
                ), "Discrete conditions not time invariant"

                # Assert continuous conditions are time invariant (same across all timesteps)
                # Excluding the timeseries cols
                # continuous_cond_input shape: (batch, sequence_len, num_continous_condtions)
                first_timestep_continuous = continuous_cond_input[
                    :, 0, continuous_conditions_non_timestamp_cols_idxs
                ].unsqueeze(1)
                assert torch.all(
                    continuous_cond_input[
                        :, :, continuous_conditions_non_timestamp_cols_idxs
                    ]
                    == first_timestep_continuous
                ), "Continuous conditions not time invariant"

        elif self.metadata_strategy is MetadataStrategy.DELETE:
            discrete_cond_input[:, self.history_length :, :] = 0

            # Zero out only the non-timeseries cols
            continuous_cond_input[
                :,
                self.history_length :,
                continuous_conditions_non_timestamp_cols_idxs,
            ] = 0

        elif self.metadata_strategy is MetadataStrategy.REPEAT:
            # Repeat the last history timestep's metadata for all forecast timesteps
            discrete_cond_input[:, self.history_length :, :] = (
                discrete_cond_input[
                    :, self.history_length - 1 : self.history_length, :
                ].repeat(1, TIME_SERIES_LEN - self.history_length, 1)
            )

            continuous_cond_input[
                :,
                self.history_length :,
                continuous_conditions_non_timestamp_cols_idxs,
            ] = continuous_cond_input[
                :,
                self.history_length - 1 : self.history_length,
                continuous_conditions_non_timestamp_cols_idxs,
            ].repeat(1, TIME_SERIES_LEN - self.history_length, 1)

        elif self.metadata_strategy is MetadataStrategy.REPEAT_WINDOW:
            # Calculate how many full windows of history we need to repeat
            forecast_length = sample.shape[2] - self.history_length
            num_full_repeats = forecast_length // self.history_length
            remaining_length = forecast_length % self.history_length

            # For each full window, copy the entire history
            for i in range(num_full_repeats):
                start_idx = self.history_length + (i * self.history_length)
                end_idx = start_idx + self.history_length
                discrete_cond_input[:, start_idx:end_idx, :] = (
                    discrete_cond_input[:, : self.history_length, :]
                )

                continuous_cond_input[
                    :,
                    start_idx:end_idx,
                    continuous_conditions_non_timestamp_cols_idxs,
                ] = continuous_cond_input[
                    :,
                    : self.history_length,
                    continuous_conditions_non_timestamp_cols_idxs,
                ]

            # Handle any remaining timesteps by copying the start of the history
            if remaining_length > 0:
                start_idx = self.history_length + (
                    num_full_repeats * self.history_length
                )
                end_idx = start_idx + remaining_length
                discrete_cond_input[:, start_idx:end_idx, :] = (
                    discrete_cond_input[:, :remaining_length, :]
                )

                continuous_cond_input[
                    :,
                    start_idx:end_idx,
                    continuous_conditions_non_timestamp_cols_idxs,
                ] = continuous_cond_input[
                    :,
                    :remaining_length,
                    continuous_conditions_non_timestamp_cols_idxs,
                ]

        history = [history] * num_forecasts
        x = [x] * num_forecasts
        discrete_cond_input = [discrete_cond_input] * num_forecasts
        continuous_cond_input = [continuous_cond_input] * num_forecasts

        # num_forecasts * (batch_dim, channels, forecast_len) > (num_forecasts * batch_dim, channels, forecast_len)
        history = torch.concatenate(history, dim=0)

        # num_forecasts * (batch_dim, channels, forecast_len) > (num_forecasts * batch_dim, channels, forecast_len)
        x = torch.concatenate(x, dim=0)

        x = torch.randn_like(x)
        discrete_cond_input = torch.concatenate(discrete_cond_input, dim=0)
        continuous_cond_input = torch.concatenate(continuous_cond_input, dim=0)

        B = x.shape[0]

        with torch.no_grad():
            for t in tqdm(
                range(T - 1, -1, -1), desc="Diffusion Steps", total=T
            ):
                diffusion_steps = torch.LongTensor(
                    [
                        t,
                    ]
                    * B
                ).to(device)
                synthesis_input = {
                    "noisy_sample": x,
                    "discrete_cond_input": discrete_cond_input,
                    "continuous_cond_input": continuous_cond_input,
                    "diffusion_step": diffusion_steps,
                }

                epsilon_theta = self.synthesizer_model(synthesis_input)

                # get the sample estimate from the noisy sample and the noise estimate
                x0_est = get_sample_est_from_noisy_sample(
                    x, epsilon_theta, Alpha_bar[t]
                )
                # enforce history
                x0_est[:, :, : self.history_length] = history

                # control_param = ((1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])) * (
                #     1 - Alpha_bar[t] / Alpha_bar[t - 1]
                # )  # DDPM
                control_param = 0
                if t > 0:
                    noise = torch.randn_like(x).to(device)
                    x = (
                        (Alpha_bar[t - 1] ** 0.5) * x0_est
                        + (1.0 - Alpha_bar[t - 1] - control_param) ** 0.5
                        * epsilon_theta
                        + noise * (control_param**0.5)
                    )
                else:
                    x = x0_est

        synthesized_timeseries = self.synthesizer_model.prepare_output(x)
        discrete_conditions = (
            batch["discrete_label_embedding"].detach().cpu().numpy()
        )
        continuous_conditions = (
            batch["continuous_label_embedding"].detach().cpu().numpy()
        )

        if self.use_probabilistic_forecast:
            NUM_FORECASTS_TIMES_NUM_BATCHES, CHANNELS, TIME_SERIES_LEN = (
                synthesized_timeseries.shape
            )
            # (batch_dim, num_samples, CHANNELS, TIME_SERIES_LEN) > (num_forecasts, batch_dim, num_channels, time_series_len)
            synthesized_timeseries = synthesized_timeseries.reshape(
                num_forecasts, -1, CHANNELS, TIME_SERIES_LEN
            )
            synthesized_timeseries = synthesized_timeseries.transpose(
                1, 0, 2, 3
            )

        dataset_dict = {
            "timeseries": synthesized_timeseries,  # (num_forecasts, batch_dim, num_channels, time_series_len)
            "discrete_conditions": discrete_conditions,  # (batch, num_discrete_conditions)
            "continuous_conditions": continuous_conditions,  # (batch, time_series_len, num_continuous_labels)
        }
        return dataset_dict


def get_sample_est_from_noisy_sample(
    noisy_sample, noise_est, current_alpha_bar
):
    sample_est = (1 / (current_alpha_bar**0.5 + 1e-8)) * (
        noisy_sample - noise_est * (1 - current_alpha_bar) ** 0.5
    )
    return sample_est
