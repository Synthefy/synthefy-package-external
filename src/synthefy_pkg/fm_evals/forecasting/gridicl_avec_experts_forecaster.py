from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

# Lazy import to avoid circular import with regressor.py
if TYPE_CHECKING:
    import synthefy_pkg.model.trainers.timeseries_decoder_forecasting_foundation_trainer as tdf
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.data.otf_synthetic_dataloader import OTFSyntheticDataloader
from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.utils import truncate_eval_batch
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)
from synthefy_pkg.model.architectures.tabicl.icl_inference.regressor import (
    TabICLGridRegressor,
)
from synthefy_pkg.prior.observation import outlier_removing, standard_scaling

COMPILE = True


def process_features(X):
    """Process inputs through outlier removal, shuffling, scaling, and padding to max features.

    Parameters
    ----------
    X : Tensor
        Feature tensor of shape (T, H).

    Returns
    -------
    Tensor
        Normalized feature tensor (T, H).
    """
    X, mean, std = standard_scaling(X, return_mean_std=True)
    return X, mean, std


class GridICLAvecExpertsForecaster(BaseForecaster):
    def __init__(
        self,
        model_checkpoint_path: str,
        config: Union[str, Configuration, dict],
        history_length: int,
        forecast_length: int = 1,
        name: str = "",
        use_aux_features: bool = False,
        trainer: Optional[
            "tdf.TimeSeriesDecoderForecastingFoundationTrainer"
        ] = None,
    ):
        super().__init__(name)
        self.history_length = history_length
        self.forecast_length = forecast_length

        # Lazy import to avoid circular dependency
        raise NotImplementedError(
            "FoundationForecastExperiment has been removed. This class needs to be updated to use an alternative experiment class."
        )
        # from synthefy_pkg.experiments.foundation_forecast_experiment import (
        #     FoundationForecastExperiment,
        # )
        # experiment = FoundationForecastExperiment(config_source=config)
        # experiment.configuration.dataset_config.forecast_length = (
        #     self.forecast_length
        # )
        # experiment.configuration.prior_config.add_both_time_stamps_and_features = False
        # experiment.configuration.dataset_config.mixed_real_synthetic_sampling = False
        # experiment.configuration.prior_config.scm_used_sampler[
        #     "choice_values"
        # ] = ["ts"]
        # experiment.configuration.tabicl_config.external_forecasts_to_use = [
        #     "toto_univariate"
        # ]
        # self.experiment = experiment

        # Load model using trainer if available, otherwise from checkpoint
        if trainer is not None:
            # trainer passed in at validation time will already be in eval mode
            assert isinstance(config, Configuration), (
                "config must be a Configuration object"
            )
            model = trainer
            # Dummy dataloader, data will be loaded from the batch
            self.experiment.data_loader = OTFSyntheticDataloader(config)
        else:
            # Load model from experiment
            model = self.experiment._setup_inference(model_checkpoint_path)
        model.eval()
        model.to("cuda")
        self.model = model

        self.num_timestamp_features = len(
            self.experiment.configuration.prior_config.add_synthetic_timestamps
        )

        assert self.name in [
            "gridicl_experts_univariate",
            "gridicl_experts_multivariate",
            "gridicl_experts_future_leaked",
        ], "Invalid model name"

    def fit(self, batch: EvalBatchFormat, disable_tqdm: bool = False) -> bool:
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        B = batch.batch_size
        NC = batch.num_correlates
        X = []
        y = []
        forecasts = []

        if (
            batch[-1, 0].history_values.shape[0] > 1_000
        ):  # Artificial max context length
            batch = truncate_eval_batch(batch, 1_000)
            logger.info(
                f"Truncated batch to {batch[-1, 0].history_values.shape[0]} context length"
            )

        for i in range(B):
            # Infer lengths from the sample instead of the lengths that have been set so far
            effective_history_length = batch[i, 0].history_values.shape[0]
            effective_forecast_length = batch[i, 0].target_values.shape[0]

            self.history_length = effective_history_length
            self.forecast_length = effective_forecast_length

            self.experiment.configuration.dataset_config.forecast_length = (
                self.forecast_length
            )
            logger.info(
                f"Effective history length: {effective_history_length}, effective forecast length: {effective_forecast_length}"
            )

            per_sample_X = []
            per_sample_y = None
            for j in range(NC):
                sample = batch[i, j]
                correlate = np.concatenate(
                    [sample.history_values, sample.target_values], axis=0
                )
                if sample.forecast:
                    per_sample_y = correlate
                else:
                    per_sample_X.append(correlate)
            if len(per_sample_X) == 0:
                per_sample_X = np.zeros(
                    (
                        self.history_length + self.forecast_length,
                        self.model.config.prior_config.max_features,
                    )
                )
            else:
                per_sample_X = np.stack(per_sample_X, axis=1)

            assert per_sample_y is not None, "per_sample_y is None"
            train_size = per_sample_X.shape[0] - self.forecast_length
            dict_of_data = {
                "X": per_sample_X[
                    :, : self.model.config.prior_config.max_features
                ],
                "y": per_sample_y,
                "train_size": train_size,
            }
            _, y_mean_history, y_std_history = process_features(
                torch.tensor(np.expand_dims(per_sample_y[:train_size], axis=-1))
            )
            assert isinstance(
                self.experiment.data_loader, OTFSyntheticDataloader
            ), "data_loader is not an OTFSyntheticDataloader"

            X, y, d, _, _, _ = (
                self.experiment.data_loader.train_dataset.get_batch(  # type: ignore
                    batch_size=1, from_csv=True, dict_of_data=dict_of_data
                )
            )  # type: ignore

            train_batch = {
                "X": X.to("cuda"),
                "y": y.to("cuda"),
                "d": d.to("cuda"),
                "train_sizes": torch.tensor([train_size]).to("cuda"),
            }

            y_mean_history_np = y_mean_history.cpu().numpy()
            y_std_history_np = y_std_history.cpu().numpy()

            if self.name == "gridicl_experts_univariate":
                decoder_input = self.model.decoder_model.prepare_training_input_only_target_prediction(
                    train_batch, task="univariate", train=False
                )
                with torch.no_grad():
                    output_dict = self.model.decoder_model(decoder_input)

            elif self.name == "gridicl_experts_multivariate":
                decoder_input = self.model.decoder_model.prepare_training_input_only_target_prediction(
                    train_batch, task="multivariate", train=False
                )
                with torch.no_grad():
                    output_dict = self.model.decoder_model(decoder_input)

            elif self.name == "gridicl_experts_future_leaked":
                decoder_input = self.model.decoder_model.prepare_training_input_only_target_prediction(
                    train_batch, task="future_leaked", train=False
                )
                with torch.no_grad():
                    output_dict = self.model.decoder_model(decoder_input)

            prediction_multivariate = output_dict["logits_multivariate"]
            prediction = self.model.decoder_model.distribution.mean(
                prediction_multivariate
            )

            unnormalized_pred_forecast = (
                prediction.cpu() * y_std_history_np + y_mean_history_np
            )
            row = []
            for j in range(NC):
                if batch[i, j].forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=batch[i, j].sample_id,
                            timestamps=batch[i, j].target_timestamps,
                            values=unnormalized_pred_forecast,
                            model_name=self.name,
                        )
                    )
                else:
                    row.append(
                        SingleSampleForecast(
                            sample_id=batch[i, j].sample_id,
                            timestamps=batch[i, j].target_timestamps,
                            values=np.zeros_like(batch[i, j].target_timestamps),
                            model_name=self.name,
                        )
                    )

            forecasts.append(row)
        return ForecastOutputFormat(forecasts)
