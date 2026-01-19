from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    import synthefy_pkg.model.trainers.timeseries_decoder_forecasting_foundation_trainer as tdf
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
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
    X = outlier_removing(X, threshold=4)
    X, mean, std = standard_scaling(X, return_mean_std=True)
    return X, mean, std


class GridICLForecaster(BaseForecaster):
    def __init__(
        self,
        model_checkpoint_path: str,
        history_length: int,
        forecast_length: int,
        name: str = "",
        use_aux_features: bool = False,
        trainer: Optional[
            "tdf.TimeSeriesDecoderForecastingFoundationTrainer"
        ] = None,
        train_time_config: Optional[Configuration] = None,
    ):
        super().__init__(name)
        if not model_checkpoint_path and isinstance(
            train_time_config, Configuration
        ):
            self.config = train_time_config

        else:
            ckpt = torch.load(model_checkpoint_path)
            ckpt["hyper_parameters"]["dataset_config"][
                "curriculum_config_path"
            ] = "src/synthefy_pkg/prior/config/curriculum_configs/config_adaptive_curr.yaml"
            ckpt["hyper_parameters"]["dataset_config"]["prior_config_path"] = (
                "src/synthefy_pkg/prior/config/synthetic_configs/config_medium_series.yaml"
            )
            self.config = Configuration(config=ckpt["hyper_parameters"])

            # TODO: remove this and test if eval bench is working
            self.config.dataset_config.num_correlates = 49

        self.history_length = history_length
        self.forecast_length = forecast_length
        if model_checkpoint_path:
            self.model = TabICLGridRegressor(
                config=self.config,
                model_path=model_checkpoint_path,
            )
        else:
            self.model = TabICLGridRegressor(
                config=self.config,
                model_path="",
                trainer=trainer,
            )

        self.num_timestamp_features = len(
            self.config.prior_config.add_synthetic_timestamps
        )

        assert self.name in [
            "gridicl_univariate",
            "gridicl_multivariate",
            "gridicl_future_leaked",
        ], "Invalid model name"

    def get_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        dt = pd.to_datetime(timestamps)
        minutes = dt.minute.values.astype(np.float32)
        hours = dt.hour.values.astype(np.float32)
        days = dt.day.values.astype(np.float32)
        months = dt.month.values.astype(np.float32)
        years = dt.year.values.astype(np.float32)
        return np.stack(
            (
                minutes / 60,
                hours / 24,
                days / 30,
                months / 12,
                years / 1000,
            ),
            axis=1,
        )

    def fit(self, batch: EvalBatchFormat, disable_tqdm: bool = False):
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        X = []
        y = []
        for i in tqdm(
            range(self.B),
            desc="Fitting gr",
            disable=disable_tqdm,
        ):
            history_time_features = None
            per_sample_X = []
            per_sample_y = None
            for j in range(self.NC):
                sample = batch[i, j]
                history_timestamps = sample.history_timestamps
                history_values = sample.history_values

                # convert history timestamps into minutes, hours, days, months, years
                if j == 0:
                    history_time_features = self.get_time_features(
                        history_timestamps
                    )

                if not sample.forecast:
                    per_sample_X.append(history_values)
                else:
                    per_sample_y = history_values
            if len(per_sample_X) == 0:
                per_sample_X = history_time_features
            else:
                per_sample_X = np.stack(per_sample_X, axis=1)
                assert history_time_features is not None, (
                    "History time features should not be None"
                )
                assert per_sample_X is not None, (
                    "Per sample X should not be None"
                )
                per_sample_X = np.concatenate(
                    [history_time_features, per_sample_X], axis=1
                )

            X.append(per_sample_X)
            y.append(per_sample_y)

        self.X = np.stack(X, axis=0)
        self.y = np.stack(y, axis=0)

        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        B = batch.batch_size
        NC = batch.num_correlates

        forecasts = []
        for i in range(B):
            X_train = self.X[i]
            y_train = self.y[i]

            test_X_features = []
            per_sample_y = None
            num_targets = 0
            for j in range(NC):
                if not batch[i, j].forecast:
                    test_X = batch[i, j].target_values
                    test_X_features.append(test_X)
                else:
                    test_y = batch[i, j].target_values
                    per_sample_y = test_y
                    num_targets += 1

            assert num_targets == 1, (
                "Number of targets should be 1, gridicl only supports one target for now"
            )
            test_time_features = self.get_time_features(
                batch[i, 0].target_timestamps
            )
            if len(test_X_features) == 0:
                X_test = test_time_features
            else:
                test_X_features = np.stack(test_X_features, axis=1)
                X_test = np.concatenate(
                    [test_time_features, test_X_features], axis=1
                )
            assert per_sample_y is not None, "Test y should not be None"
            y_test = per_sample_y

            train_timestamp_features = torch.tensor(
                X_train[:, : self.num_timestamp_features]
            )
            processed_train_X, mean_X, std_X = process_features(
                torch.tensor(X_train)[:, self.num_timestamp_features :]
            )
            X_train = torch.cat(
                [train_timestamp_features, processed_train_X], dim=1
            )
            processed_train_y, mean_train_y, std_train_y = process_features(
                torch.tensor(np.expand_dims(y_train, axis=-1))
            )
            y_train = processed_train_y.squeeze(-1)

            test_timestamp_features = torch.tensor(
                X_test[:, : self.num_timestamp_features]
            )
            # remove outliers
            processed_test_X = outlier_removing(
                torch.tensor(X_test)[:, self.num_timestamp_features :],
                threshold=4,
            )
            # standardize
            processed_test_X = (
                processed_test_X - mean_X.unsqueeze(0)
            ) / std_X.unsqueeze(0)
            # clip
            processed_test_X = torch.clip(processed_test_X, min=-100, max=100)
            # append to timestamp features
            X_test = torch.cat(
                [test_timestamp_features, processed_test_X], dim=1
            )
            # remove outliers
            processed_test_y = outlier_removing(
                torch.tensor(np.expand_dims(y_test, axis=-1)), threshold=4
            )
            # standardize
            processed_test_y = (
                processed_test_y - mean_train_y.unsqueeze(0)
            ) / std_train_y.unsqueeze(0)
            # clip
            processed_test_y = torch.clip(processed_test_y, min=-100, max=100)
            y_test = processed_test_y.squeeze(-1)

            X_overall = torch.cat([X_train, X_test], dim=0)
            y_overall = torch.cat([y_train, y_test], dim=0)

            X_overall = X_overall.unsqueeze(0)
            y_overall = y_overall.unsqueeze(0)

            if self.name == "gridicl_univariate":
                d = torch.tensor(
                    [self.num_timestamp_features] * X_overall.shape[0]
                )
                train_sizes = torch.tensor(
                    [self.history_length] * X_overall.shape[0]
                )

                target_mask = (
                    torch.zeros(
                        X_overall.shape[0],
                        1 + self.num_timestamp_features,
                        X_overall.shape[1],
                    )
                    .to(X_overall.device)
                    .bool()
                )
                target_mask[:, d, -self.forecast_length :] = True

                X_full = [
                    X_overall[i].clone() for i in range(X_overall.shape[0])
                ]
                for k, d_i in enumerate(d):
                    X_full[k] = torch.cat(
                        [
                            X_full[k][:, : self.num_timestamp_features],
                            y_overall[k].unsqueeze(-1).squeeze(0),
                        ],
                        dim=-1,
                    )
                X_full = torch.stack(X_full, dim=0)
                y_full = y_overall.clone()
                target = X_full[target_mask.bool().transpose(1, 2)].clone()
                X_full_masked = X_full.clone()
                X_full_masked[target_mask.bool().transpose(1, 2)] = -100

                decoder_input = {
                    "values": X_overall.to(self.model.device).float(),
                    "values_full": X_full.to(self.model.device).float(),
                    "values_masked": X_full_masked.to(
                        self.model.device
                    ).float(),
                    "y_full": y_full.to(self.model.device).float(),
                    "target": target.to(self.model.device).float(),
                    "train_sizes": train_sizes.to(self.model.device).float(),
                    "useful_features": d.to(self.model.device).float(),
                    "target_mask": target_mask.bool().to(self.model.device),
                }

            elif self.name == "gridicl_multivariate":
                d = torch.tensor([X_overall.shape[2]] * X_overall.shape[0])
                train_sizes = torch.tensor(
                    [self.history_length] * X_overall.shape[0]
                )

                target_mask = (
                    torch.zeros(
                        X_overall.shape[0],
                        X_overall.shape[2] + 1,
                        X_overall.shape[1],
                    )
                    .to(X_overall.device)
                    .bool()
                )
                target_mask[
                    :, self.num_timestamp_features :, -self.forecast_length :
                ] = True

                X_full = [
                    X_overall[i].clone() for i in range(X_overall.shape[0])
                ]
                for k, d_i in enumerate(d):
                    X_full[k] = torch.cat(
                        [
                            X_full[k][:, : X_overall.shape[2]],
                            y_overall[k].unsqueeze(-1).squeeze(0),
                        ],
                        dim=-1,
                    )
                X_full = torch.stack(X_full, dim=0)
                y_full = y_overall.clone()
                target = X_full[target_mask.bool().transpose(1, 2)].clone()
                X_full_masked = X_full.clone()
                X_full_masked[target_mask.bool().transpose(1, 2)] = -100

                decoder_input = {
                    "values": X_overall.to(self.model.device).float(),
                    "values_full": X_full.to(self.model.device).float(),
                    "values_masked": X_full_masked.to(
                        self.model.device
                    ).float(),
                    "y_full": y_full.to(self.model.device).float(),
                    "target": target.to(self.model.device).float(),
                    "train_sizes": train_sizes.to(self.model.device).float(),
                    "useful_features": d.to(self.model.device).float(),
                    "target_mask": target_mask.bool().to(self.model.device),
                }

            elif self.name == "gridicl_future_leaked":
                d = torch.tensor([X_overall.shape[2]] * X_overall.shape[0])
                train_sizes = torch.tensor(
                    [self.history_length] * X_overall.shape[0]
                )

                target_mask = (
                    torch.zeros(
                        X_overall.shape[0],
                        X_overall.shape[2] + 1,
                        X_overall.shape[1],
                    )
                    .to(X_overall.device)
                    .bool()
                )
                target_mask[:, d, -self.forecast_length :] = True

                X_full = [
                    X_overall[i].clone() for i in range(X_overall.shape[0])
                ]
                for k, d_i in enumerate(d):
                    X_full[k] = torch.cat(
                        [
                            X_full[k][:, : X_overall.shape[2]],
                            y_overall[k].unsqueeze(-1).squeeze(0),
                        ],
                        dim=-1,
                    )
                X_full = torch.stack(X_full, dim=0)
                y_full = y_overall.clone()
                target = X_full[target_mask.bool().transpose(1, 2)].clone()
                X_full_masked = X_full.clone()
                X_full_masked[target_mask.bool().transpose(1, 2)] = -100

                decoder_input = {
                    "values": X_overall.to(self.model.device).float(),
                    "values_full": X_full.to(self.model.device).float(),
                    "values_masked": X_full_masked.to(
                        self.model.device
                    ).float(),
                    "y_full": y_full.to(self.model.device).float(),
                    "target": target.to(self.model.device).float(),
                    "train_sizes": train_sizes.to(self.model.device).float(),
                    "useful_features": d.to(self.model.device).float(),
                    "target_mask": target_mask.bool().to(self.model.device),
                }

            assert self.model.model_ is not None, "Model is not loaded"

            with torch.no_grad():
                output = self.model.model_.decoder_model(decoder_input)

            pred = output["prediction_multivariate"]
            if self.name == "gridicl_multivariate":
                pred = pred[-self.forecast_length :]
            pred_forecast = self.model.model_.decoder_model.distribution.mean(
                pred
            )

            unnormalized_pred_forecast = (
                pred_forecast.cpu() * std_train_y.cpu() + mean_train_y.cpu()
            )
            unnormalized_pred_forecast = unnormalized_pred_forecast.numpy()

            # target is always the last column

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
