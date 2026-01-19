import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.model.architectures.tabicl.inference_config import (
    InferenceConfig,
)
from synthefy_pkg.model.architectures.tabicl.multilayer_tabicl import (
    MultilayerTabICL,
)
from synthefy_pkg.model.architectures.tabicl.tabicl import TabICL
from synthefy_pkg.model.foundation_model.distributional_forecasting_utils import (
    FullSupportBarDistribution,
)
from synthefy_pkg.model.foundation_model.utils import generate_target_mask

batch_idx = 0


def retrieve_tabicl_args(config: Configuration, model_name: str):
    arguments = {
        "max_classes_or_dim": config.tabicl_config.max_classes,
        "embed_dim": config.tabicl_config.embed_dim,
        "col_num_blocks": config.tabicl_config.col_num_blocks,
        "col_nhead": config.tabicl_config.col_nhead,
        "col_num_inds": config.tabicl_config.col_num_inds,
        "row_num_blocks": config.tabicl_config.row_num_blocks,
        "row_nhead": config.tabicl_config.row_nhead,
        "row_num_cls": config.tabicl_config.row_num_cls,
        "row_rope_base": config.tabicl_config.row_rope_base,
        "icl_num_blocks": config.tabicl_config.icl_num_blocks,
        "icl_nhead": config.tabicl_config.icl_nhead,
        "ff_factor": config.tabicl_config.ff_factor,
        "dropout": config.tabicl_config.dropout,
        "activation": config.tabicl_config.activation,
        "norm_first": config.tabicl_config.norm_first,
        "is_regression": config.tabicl_config.is_regression,
        "preserve_col_order": config.tabicl_config.preserve_col_order,
        "weight_range": config.tabicl_config.weight_range,
        "use_time_mask": config.tabicl_config.use_time_mask,
        "full_reg_decoder_config": config.token_decoder_config,
        "embedder_name": config.tabicl_config.external_column_embedder,
        "embedder_config": config.tabicl_config.external_column_embedder_config,
        "embedder_checkpoint": config.tabicl_config.external_column_embedder_checkpoint,
        "train_as_univariate_forecast": config.tabicl_config.train_as_univariate_forecast,
        "time_mask_type": config.tabicl_config.time_mask_type,
        "time_mask_mixing_probs": config.tabicl_config.time_mask_mixing_probs,
    }
    if model_name == "multilayer_tabicl":
        arguments.update(
            {
                "embed_col_num_blocks": config.tabicl_config.embed_col_num_blocks,
                "embed_row_num_blocks": config.tabicl_config.embed_row_num_blocks,
                "num_layers": config.tabicl_config.num_layers,
                "skip_col_embedding": config.tabicl_config.skip_col_embedding,
            }
        )
    return arguments


class TabICLModel(nn.Module):
    """Wrapper around TabICL to match V3E interface."""

    def __init__(
        self,
        config: Configuration,
    ):
        super().__init__()
        self.config = config
        # Forecasting configuration
        self.horizon_len = config.dataset_config.forecast_length
        self.context_len = (
            config.dataset_config.time_series_length - self.horizon_len
        )
        self.time_series_length = config.dataset_config.time_series_length
        self.target_masking_schemes = (
            config.foundation_model_config.masking_schemes
        )

        # Dataset and device specific configuration
        self.device = config.device
        self.batch_size = config.dataset_config.batch_size
        self.window_size = config.dataset_config.time_series_length
        self.dataset_config = config.dataset_config
        self.use_metadata = config.foundation_model_config.use_metadata
        self.num_correlates = config.dataset_config.num_correlates
        self.is_regression = config.tabicl_config.is_regression
        self.is_synthetic = config.dataset_config.using_synthetic_data
        self.is_tabular = config.dataset_config.is_tabular
        self.is_only_target_prediction = (
            config.dataset_config.is_only_target_prediction
        )
        self.bound_output_scale = (
            config.foundation_model_config.bound_output_scale
        )
        self.random_train_mask_ratio = (
            config.tabicl_config.random_train_mask_ratio
        )
        self.mask_mixing_rates = (
            config.foundation_model_config.mask_mixing_rates
        )
        self.last_row_masking = config.tabicl_config.last_row_masking

        self.use_full_reg = config.tabicl_config.use_full_reg
        token_decoder_config = None
        if self.use_full_reg:
            # set the token_decoder_config for full regression, set output dim to (the number of bins if using bins, otherwise 1) * forecast length
            token_decoder_config = config.token_decoder_config
            assert token_decoder_config is not None
            token_decoder_config.output_dim = (
                1
                if not config.foundation_model_config.generate_probabilistic_forecast_using_bins
                else config.foundation_model_config.num_bins
            ) * token_decoder_config.token_forecast_length
        self.external_column_embedder = (
            config.tabicl_config.external_column_embedder
        )
        self.external_column_embedder_config = (
            config.tabicl_config.external_column_embedder_config
        )
        self.external_column_embedder_checkpoint = (
            config.tabicl_config.external_column_embedder_checkpoint
        )

        tabicl_args = retrieve_tabicl_args(
            config, config.foundation_model_config.model_name
        )

        if config.foundation_model_config.model_name == "tabicl":
            self.model = TabICL(**tabicl_args)
        elif config.foundation_model_config.model_name == "multilayer_tabicl":
            self.model = MultilayerTabICL(**tabicl_args)

        # Inference configuration
        init_config = {
            "COL_CONFIG": {
                "device": self.device,
                "use_amp": False,
                "verbose": False,
            },
            "ROW_CONFIG": {
                "device": self.device,
                "use_amp": False,
                "verbose": False,
            },
            "ICL_CONFIG": {
                "device": self.device,
                "use_amp": False,
                "verbose": False,
            },
        }
        # If None, default settings in InferenceConfig
        self.inference_config = InferenceConfig()
        self.inference_config.update_from_dict(init_config)

        if self.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
            # logger.info(
            #     "Learning a model for probabilistic forecasts using bins"
            # )
            num_bins = self.config.foundation_model_config.num_bins

            # Create num_bins+1 borders that are gaussian distributed from -100 to 100
            normal = torch.distributions.Normal(0, 1)
            # Get evenly spaced points in the CDF of a standard normal
            quantiles = torch.linspace(0.001, 0.999, num_bins + 1)
            # Transform to standard normal quantiles
            borders = normal.icdf(quantiles)
            # Scale to [-100, 100] range
            borders = (
                borders
                * self.config.foundation_model_config.absolute_max_bar_value
                / borders.abs().max()
            )
            self.distribution = FullSupportBarDistribution(borders)

        self.external_forecasts_to_use = (
            config.tabicl_config.external_forecasts_to_use
        )

        self.external_forecasting_models = {}
        if "toto_univariate" in self.external_forecasts_to_use:
            logger.info("Using Toto Univariate for external forecasting")
            from synthefy_pkg.fm_evals.forecasting.toto_forecaster import (
                TotoUnivariateForecaster,
            )

            self.external_forecasting_models["toto_univariate"] = (
                TotoUnivariateForecaster(server_url="http://localhost:50787")
            )

        if "tabpfn_univariate" in self.external_forecasts_to_use:
            logger.info("Using TabPFN Univariate for external forecasting")
            from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
                TabPFNUnivariateForecaster,
            )

            self.external_forecasting_models["tabpfn_univariate"] = (
                TabPFNUnivariateForecaster()
            )

        if "tabpfn_multivariate" in self.external_forecasts_to_use:
            logger.info("Using TabPFN Multivariate for external forecasting")
            from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
                TabPFNMultivariateForecaster,
            )

            self.external_forecasting_models["tabpfn_multivariate"] = (
                TabPFNMultivariateForecaster()
            )

        if "tabpfn_future_leaked" in self.external_forecasts_to_use:
            logger.info("Using TabPFN Future Leaked for external forecasting")
            from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
                TabPFNMultivariateForecaster,
            )

            self.external_forecasting_models["tabpfn_future_leaked"] = (
                TabPFNMultivariateForecaster(future_leak=True)
            )

        self.tasks = config.foundation_model_config.tasks

    def forward(
        self,
        decoder_input: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Forward pass matching V3E interface.

        Parameters
        ----------
        decoder_input : Dict[str, Tensor]
            Dictionary containing:
            - timestamps: Tensor of shape (B, T, timestamp_features)
            - descriptions: Tensor of shape (B, T, description_features)
            - continuous: Tensor of shape (B, T, 1)
            - continuous_tokens: Tensor of shape (B, T, token_dims)
            - mask: Tensor of shape (B, T)
            - target_mask: Tensor of shape (B, T)
            - target: Tensor of shape (B, T, 1)

        inference_config : Optional[InferenceConfig]
            Configuration for inference behavior

        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing:
            - prediction: Tensor of shape (B, T, 1)
        """
        X = (
            decoder_input["values_masked"]
            if self.use_full_reg
            else decoder_input["values"]
        )
        y = (
            decoder_input["y_full"]
            if self.use_full_reg
            else decoder_input["target_partial"]
        )
        d = decoder_input["useful_features"] + int(self.use_full_reg)
        target_mask = decoder_input["target_mask"]
        mask = decoder_input["mask"]
        if self.training:
            out = self.model(
                X=X,
                y_train=y,
                d=d,
                target_mask=target_mask,
                mask=mask,
            )
        else:
            out = self.model(
                X=X,
                y_train=y,
                d=d,
                inference_config=self.inference_config,
                target_mask=target_mask,
                mask=mask,
            )
        if self.use_full_reg:
            # TODO: this is a hack to get the logits for the univariate and multivariate forecasts
            # it is the first if that we are always using now
            if self.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
                pred_univariate = out["univariate_forecast"]
                pred_multivariate = out["multivariate_forecast"]
                logits_univariate = pred_univariate
                logits_multivariate = pred_multivariate
            else:
                raise NotImplementedError(
                    "Not implemented for timeseries forecasting model with regression loss"
                )
        elif not self.config.dataset_config.is_regression:
            # TODO: this needs to be fixed for a pure tabular classification model, however, we don't use it for now
            raise NotImplementedError(
                "Not implemented for a pure tabular classification model"
            )
        elif self.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
            raise NotImplementedError(
                "Not implemented for a pure tabular regression with bins model"
            )
        else:  # directly regress the values as point forecasts
            raise NotImplementedError(
                "Not implemented for a pure tabular regression model"
            )

        return {
            "prediction_univariate": pred_univariate,
            "prediction_multivariate": pred_multivariate,
            "logits_univariate": logits_univariate,
            "logits_multivariate": logits_multivariate,
            "target_mask": target_mask,
            "logits": logits_multivariate,
            "prediction": pred_multivariate,
        }

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        self.model = self.model.train(mode)
        return super().train(mode)

    def eval(self):
        self.model = self.model.eval()
        return super().eval()

    def get_external_forecast(
        self, y_partial: Tensor, prediction_length: int
    ) -> Tensor:
        random_idx = int(
            torch.randint(0, len(self.external_forecasts_to_use), (1,)).item()
        )
        model_name = self.external_forecasts_to_use[random_idx]
        external_forecasting_model = self.external_forecasting_models[
            model_name
        ]

        # Convert tensor to numpy and prepare for TotoForecaster
        # Convert y_partial to numpy array
        y_np = y_partial.detach().cpu().numpy()
        y_np = np.expand_dims(y_np, axis=1)

        history_length = y_partial.shape[1]
        time_series_length = prediction_length + history_length
        timestamps = pd.date_range(
            start="2010-01-01", periods=time_series_length, freq="W", tz="UTC"
        )
        # convert to np.datetime64
        timestamps = np.array(timestamps, dtype=np.datetime64)
        # Create dummy timestamps for TotoForecaster
        batch_size, num_features, seq_len = y_np.shape

        # repeat the timestamps for each feature
        timestamps = np.expand_dims(timestamps, axis=0).repeat(
            batch_size, axis=0
        )
        timestamps = np.expand_dims(timestamps, axis=1).repeat(
            num_features, axis=1
        )

        # Create EvalBatchFormat for TotoForecaster
        # sample_ids should be (batch_size, num_correlates)
        sample_ids = (
            np.arange(0, batch_size).reshape(-1, 1).repeat(num_features, axis=1)
        )

        batch = EvalBatchFormat.from_arrays(
            sample_ids=sample_ids,
            history_timestamps=timestamps[..., :history_length],
            history_values=y_np,
            target_timestamps=timestamps[..., -prediction_length:],
            target_values=np.zeros(
                (batch_size, num_features, prediction_length)
            ),
        )

        # Get forecast from TotoForecaster
        external_forecasting_model.fit(batch)
        if model_name == "toto_univariate":
            forecast_output = external_forecasting_model.predict_for_training(
                batch, num_samples=10
            )
        else:
            forecast_output = external_forecasting_model.predict(batch)

        # Convert back to tensor
        forecast_values = []
        for b in range(batch_size):
            per_sample_forecast = []
            for nc in range(num_features):
                per_sample_forecast.append(forecast_output[b, nc].values)
            forecast_values.append(np.array(per_sample_forecast))
        forecast_values = np.array(forecast_values)

        y_forecast = torch.tensor(
            np.array(forecast_values),
            device=y_partial.device,
            dtype=y_partial.dtype,
        ).squeeze(1)
        return y_forecast.to(y_partial.device)

    def get_mask_and_useful_features_for_specific_task(
        self,
        task: str,
        input_size: Tuple[int, int, int],
        d: Tensor,
        train_sizes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        target_mask = torch.zeros(
            input_size[0], input_size[2], input_size[1], dtype=torch.bool
        )
        if task == "multivariate":
            num_timestamps = len(
                self.config.prior_config.add_synthetic_timestamps
            )
            for idx in range(input_size[0]):
                target_mask[
                    idx, num_timestamps : d[idx], train_sizes[idx] :
                ] = True
                if len(self.external_forecasts_to_use) > 0:
                    target_mask[idx, d[idx] + 2, train_sizes[idx] :] = True
                    d[idx] += 2
                else:
                    target_mask[idx, d[idx], train_sizes[idx] :] = True
        elif task == "univariate":
            num_timestamps = len(
                self.config.prior_config.add_synthetic_timestamps
            )
            for idx in range(input_size[0]):
                if len(self.external_forecasts_to_use) > 0:
                    target_mask[idx, num_timestamps + 2, train_sizes[idx] :] = (
                        True
                    )
                    d[idx] = num_timestamps + 2
                else:
                    target_mask[idx, num_timestamps, train_sizes[idx] :] = True
                    d[idx] = num_timestamps
        elif task == "future_leaked":
            num_timestamps = len(
                self.config.prior_config.add_synthetic_timestamps
            )
            for idx in range(input_size[0]):
                if len(self.external_forecasts_to_use) > 0:
                    target_mask[idx, d[idx] + 2, train_sizes[idx] :] = True
                    d[idx] += 2
                else:
                    target_mask[idx, d[idx], train_sizes[idx] :] = True
        return target_mask, d

    def prepare_training_input_only_target_prediction(
        self,
        train_batch: Dict[str, Tensor],
        task: str = "multivariate",
        train: bool = True,
        *args,
        **kwargs,
    ):
        if train:
            # randomly choose a task from the list of tasks
            task = random.choice(self.tasks)

        # print(train_batch["d"])
        y_full = train_batch["y"]  # b, t, 1
        train_sizes = train_batch["train_sizes"]
        y = y_full[:, : train_sizes[0]]

        external_forecaster_context_length = int(0.8 * train_sizes[0])
        y_partial = y_full[:, :external_forecaster_context_length]
        max_seq_len = y_full.shape[1]
        y_forecast = self.get_external_forecast(
            y_partial, int(max_seq_len - external_forecaster_context_length)
        )
        forecast_correlate1 = torch.cat([y_partial, y_forecast], dim=-1)
        y_partial = y_full[:, : train_sizes[0]]
        y_forecast = self.get_external_forecast(
            y_partial, int(max_seq_len - train_sizes[0])
        )
        forecast_correlate2 = torch.cat([y_partial, y_forecast], dim=-1)
        forecast_correlate = torch.stack(
            [forecast_correlate1, forecast_correlate2], dim=-1
        )

        X = train_batch["X"]  # b, t, nc
        X_full = [X[i].clone() for i in range(X.shape[0])]

        # append y to the end of d
        for i, d_i in enumerate(train_batch["d"]):
            X_full[i] = torch.cat(
                [
                    X_full[i][:, :d_i],
                    forecast_correlate[i],
                    train_batch["y"][i].unsqueeze(-1).squeeze(0),
                    X_full[i][:, d_i:],
                ],
                dim=-1,
            )
        X_full = torch.stack(X_full, dim=0)  # b, t, nc+1

        d = train_batch["d"].clone()

        input_size: Tuple[int, int, int] = (
            X_full.shape[0],
            X_full.shape[1],
            X_full.shape[2],
        )
        mask, d = self.get_mask_and_useful_features_for_specific_task(
            task, input_size, d, train_sizes
        )
        mask = mask.to(self.device)
        d = d.to(self.device)

        X_full_uni = torch.zeros_like(X_full)
        if task == "univariate":
            num_timestamps = len(
                self.config.prior_config.add_synthetic_timestamps
            )
            for i in range(X_full.shape[0]):
                X_full_uni[i, :, :num_timestamps] = X_full[
                    i, :, :num_timestamps
                ]
                X_full_uni[i, :, num_timestamps : num_timestamps + 2] = (
                    forecast_correlate[i]
                )
                X_full_uni[i, :, num_timestamps + 2] = train_batch["y"][i]
            X_full = X_full_uni

        target_mask = torch.zeros_like(mask, dtype=torch.bool).to(X_full.device)
        for idx in range(train_batch["d"].shape[0]):
            target_mask[idx, d[idx], train_sizes[idx] :] = True

        if self.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
            y_full = y_full.unsqueeze(-1)
        if self.use_full_reg:
            target = (
                X_full[target_mask.bool().transpose(1, 2)].unsqueeze(-1).clone()
            )
            X_full_masked = X_full.clone()
            X_full_masked[mask.bool().transpose(1, 2)] = -100
        else:
            raise NotImplementedError(
                "Not implemented for a pure tabular classification model"
            )

        # if not train:
        #     if task == "univariate":
        #         num_timestamps = len(self.config.prior_config.add_synthetic_timestamps)
        #         # d should be num_timestamps + 2
        #         assert torch.all(d == num_timestamps + 2)
        #         # final_target_mask should be the same as target_mask
        #         assert torch.all(final_target_mask == target_mask)
        #         # none of the first 8 features should be fully zero (5 timestamps + 2 forecast + 1 target)
        #         for bid in range(X_full.shape[0]):
        #             for fid in range(X_full.shape[2]):
        #                 if fid > num_timestamps + 2 and torch.all(X_full[bid, :, fid] != 0):
        #                     raise ValueError(f"Feature {fid} is non zero for batch {bid}")

        #         # history of columns 7 and 8 should be the same
        #         for bid in range(X_full.shape[0]):
        #             if not torch.all(X_full[bid, :train_sizes[bid], num_timestamps+1] == X_full[bid, :train_sizes[bid], num_timestamps+2]):
        #                 raise ValueError(f"History of columns 6 and 7 are not the same for batch {bid}")

        #         # 0.8 * history of columns 5 and 6 should be the same
        #         for bid in range(X_full.shape[0]):
        #             if not torch.all(X_full[bid, :int(0.8 * train_sizes[bid]), num_timestamps+1] == X_full[bid, :int(0.8 * train_sizes[bid]), num_timestamps]):
        #                 raise ValueError(f"History of columns 5 and 6 are not the same for batch {bid}")

        #         # 0.8 * history of columns 5 and 6 should be the same
        #         for bid in range(X_full.shape[0]):
        #             for fid in range(X_full.shape[2]):
        #                 if fid < num_timestamps:
        #                     assert not torch.any(target_mask[bid, fid, :])
        #                 elif fid >= num_timestamps and fid < num_timestamps+2:
        #                     assert not torch.any(target_mask[bid, fid, :])
        #                 elif fid == num_timestamps+2:
        #                     assert not torch.any(target_mask[bid, fid, :train_sizes[bid]])
        #                     assert torch.all(target_mask[bid, fid, train_sizes[bid]:])

        #     if task == "multivariate":
        #         # d should be 2 more than train_batch["d"]
        #         assert torch.all(d == train_batch["d"] + 2)

        #         # history of columns 7 and 8 should be the same
        #         for bid in range(X_full.shape[0]):
        #             if not torch.all(X_full[bid, :train_sizes[bid], d[bid]-1] == X_full[bid, :train_sizes[bid], d[bid]]):
        #                 raise ValueError(f"History of columns 6 and 7 are not the same for batch {bid}")

        #         # 0.8 * history of columns 5 and 6 should be the same
        #         for bid in range(X_full.shape[0]):
        #             if not torch.all(X_full[bid, :int(0.8 * train_sizes[bid]), d[bid]-2] == X_full[bid, :int(0.8 * train_sizes[bid]), d[bid]-1]):
        #                 raise ValueError(f"History of columns 5 and 6 are not the same for batch {bid}")

        #             for fid in range(X_full.shape[2]):
        #                 if fid < num_timestamps:
        #                     assert not torch.any(target_mask[bid, fid, :])
        #                 elif fid >= num_timestamps and fid < d[bid]-2:
        #                     assert not torch.any(target_mask[bid, fid, :train_sizes[bid]])
        #                     assert torch.all(target_mask[bid, fid, train_sizes[bid]:])
        #                 elif fid >= d[bid]-2 and fid < d[bid]:
        #                     assert not torch.any(target_mask[bid, fid, :])
        #                 elif fid == d[bid]:
        #                     assert not torch.any(target_mask[bid, fid, :train_sizes[bid]])
        #                     assert torch.all(target_mask[bid, fid, train_sizes[bid]:])

        #     if task == "future_leaked":
        #         # d should be 2 more than train_batch["d"]
        #         assert torch.all(d == train_batch["d"] + 2)

        #         # history of columns 7 and 8 should be the same
        #         for bid in range(X_full.shape[0]):
        #             if not torch.all(X_full[bid, :train_sizes[bid], d[bid]-1] == X_full[bid, :train_sizes[bid], d[bid]]):
        #                 raise ValueError(f"History of columns 6 and 7 are not the same for batch {bid}")

        #         # 0.8 * history of columns 5 and 6 should be the same
        #         for bid in range(X_full.shape[0]):
        #             if not torch.all(X_full[bid, :int(0.8 * train_sizes[bid]), d[bid]-2] == X_full[bid, :int(0.8 * train_sizes[bid]), d[bid]-1]):
        #                 raise ValueError(f"History of columns 5 and 6 are not the same for batch {bid}")

        #             for fid in range(X_full.shape[2]):
        #                 if fid < num_timestamps:
        #                     assert not torch.any(target_mask[bid, fid, :])
        #                 elif fid >= num_timestamps and fid < d[bid]-2:
        #                     assert not torch.any(target_mask[bid, fid, :])
        #                 elif fid >= d[bid]-2 and fid < d[bid]:
        #                     assert not torch.any(target_mask[bid, fid, :])
        #                 elif fid == d[bid]:
        #                     assert not torch.any(target_mask[bid, fid, :train_sizes[bid]])
        #                     assert torch.all(target_mask[bid, fid, train_sizes[bid]:])

        #     target_mask_reshaped = target_mask.transpose(1, 2)
        #     assert torch.unique(X_full_masked[target_mask_reshaped]) == -100

        return {
            "values": X,
            "values_full": X_full,
            "values_masked": X_full_masked,
            "target_partial": y,
            "y_full": y_full,
            "target": target,
            "train_sizes": train_batch["train_sizes"],
            "useful_features": d,
            "target_mask": target_mask.bool(),
            "mask": mask.bool(),
            "external_forecast": y_forecast,
            "task": task,
        }

    def prepare_training_input_synthetic(
        self, train_batch: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        """Prepare input for training/inference following V3E format."""
        X = train_batch["X"]  # b, t, nc
        X_full = [X[i].clone() for i in range(X.shape[0])]

        # append y to the end of d
        for i, d_i in enumerate(train_batch["d"]):
            X_full[i] = torch.cat(
                [
                    X_full[i][:, :d_i],
                    train_batch["y"][i].unsqueeze(-1).squeeze(0),
                    X_full[i][:, d_i:],
                ],
                dim=-1,
            )
        X_full = torch.stack(X_full, dim=0)  # b, t, nc+1
        y_full = train_batch["y"]  # b, t, 1
        d = train_batch["d"]  # b, 1
        batch_size = train_batch["X"].shape[0]
        sequence_length = train_batch["X"].shape[1]

        train_sizes = train_batch["train_sizes"]
        train_sizes[:] = torch.min(
            train_sizes
        )  # TODO: when we fix train sizes bugs this will need to be changed
        # if self.use_full_reg:  # no train-test split for full regression
        #     train_sizes[:] = sequence_length

        y = y_full[:, : train_sizes[0]]
        target_mask = self._generate_target_mask(
            batch_size, sequence_length, train_sizes, d
        )

        # give dummy timestamps of all ones for tabular, sequence for time series
        tabular_flags = train_batch["series_flags"]
        timestamps = torch.ones(
            (batch_size, sequence_length, 1), device=self.device
        ).long()
        timestamps[tabular_flags] = (
            torch.arange(sequence_length, device=self.device)
            .unsqueeze(0)
            .repeat(int(tabular_flags.int().sum().item()), 1)
            .unsqueeze(-1)
        )
        if self.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
            y_full = y_full.unsqueeze(-1)
        if self.use_full_reg:
            target = (
                X_full[target_mask.bool().transpose(1, 2)].unsqueeze(-1).clone()
            )
            X_full_masked = X_full.clone()
            X_full_masked[target_mask.bool().transpose(1, 2)] = -100
        else:
            if self.target_masking_schemes[0] == "random_last":
                y = y_full.clone()
                y[target_mask.bool()] = -100
                target = y_full
                X_full_masked = X_full.clone()
            else:
                target = y_full
                X_full_masked = X_full.clone()
            if self.last_row_masking:
                row_mask = (
                    target_mask.clone().unsqueeze(-1).repeat(1, 1, X.shape[2])
                )
                row_mask[
                    :, :, : self.config.foundation_model_config.row_mask_min
                ] = 0
                X[row_mask.bool()] = -100

        return {
            "timestamps": timestamps,
            "values": X,
            "values_full": X_full,
            "values_masked": X_full_masked,
            "target_partial": y,
            "y_full": y_full,
            "target": target,
            "train_sizes": train_batch["train_sizes"],
            "useful_features": d,
            "target_mask": target_mask.bool(),
        }

    def prepare_training_input_real(
        self, train_batch: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        """Prepare input for training/inference following V3E format."""
        train_sizes = train_batch["train_sizes"]
        if "d" in train_batch:
            ds = train_batch["d"]
        else:
            ds = (
                torch.ones((train_batch["values"].shape[0],))
                * train_batch["values"].shape[2]
            )
        batch_size = train_batch["values"].shape[0]
        sequence_length = train_batch["values"].shape[2]

        input_values = train_batch[
            "values"
        ].float()  # TODO: try adding timestamps to the input later
        # input values is now b, nc+1 , t

        X = rearrange(input_values, "b nc t -> b t nc")
        y_full = input_values[:, -1]

        # if train sizes were different, this code wouldn't work
        # y = torch.stack([y[i, :train_sizes[i]] for i in range(batch_size)], dim=0)
        y = y_full[:, : train_sizes[0]]
        # # Create a mask for the valid indices
        # mask = torch.arange(y.size(1), device=y.device)[None, :] < ds[:, None]
        # y_full = torch.where(mask, y, torch.zeros_like(y))

        if "target_mask" in train_batch:
            target_mask = train_batch["target_mask"]
        else:
            target_mask = self._generate_target_mask(
                batch_size, sequence_length, train_sizes, ds
            )

        # give dummy timestamps of all ones for tabular, sequence for time series
        tabular_flags = train_batch["series_flags"]
        timestamps = torch.ones(
            (batch_size, sequence_length, 1), device=self.device
        )
        timestamps[tabular_flags] = (
            train_batch["timestamps"]
            if "timestamps" in train_batch
            else (
                torch.arange(sequence_length, device=self.device)
                .unsqueeze(0)
                .repeat(int(tabular_flags.int().sum().item()), 1)
                .unsqueeze(-1)
            )
        )

        return {
            "timestamps": timestamps,
            "values": X[..., :-1],
            "values_masked": input_values.transpose(1, 2),
            "values_full": input_values,
            "y_full": y_full,
            "target_partial": y,
            "target": y_full.unsqueeze(-1),
            "train_sizes": train_batch["train_sizes"],
            "useful_features": ds,
            "target_mask": target_mask.bool(),
        }

    def prepare_training_input(
        self, train_batch: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        self.device = next(self.parameters()).device
        for k, v in train_batch.items():
            if isinstance(v, Tensor):
                train_batch[k] = v.to(self.device)

        if self.is_synthetic:
            return self.prepare_training_input_synthetic(
                train_batch, *args, **kwargs
            )
        elif self.is_only_target_prediction:
            return self.prepare_training_input_only_target_prediction(
                train_batch, *args, **kwargs
            )
        else:
            return self.prepare_training_input_real(
                train_batch, *args, **kwargs
            )

    def autoregressive_forecast(
        self,
        batch,
        synthesizer,
        history_length: Optional[int] = None,
        forecast_length: Optional[int] = None,
        use_ground_truth_for_next_step: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        """Perform autoregressive forecasting on the target column."""
        if history_length is None:
            history_length = (
                self.config.dataset_config.time_series_length
                - self.config.dataset_config.forecast_length
            )
        if forecast_length is None:
            forecast_length = self.config.dataset_config.forecast_length

        decoder_input = self.prepare_training_input(batch)

        # Initialize with history from target column
        history = decoder_input["values"][
            ..., :history_length, -1:
        ]  # [batch, time, 1]
        forecast_outputs = []

        for step in tqdm(range(forecast_length), desc="Forecasting"):
            # Forward pass
            output = self.forward({"values": history})["prediction"]
            forecast_outputs.append(output[..., -1, :])

            if step < forecast_length - 1:
                # Update history with new prediction
                if use_ground_truth_for_next_step:
                    next_value = decoder_input["values"][
                        ...,
                        history_length + step : history_length + step + 1,
                        -1:,
                    ]
                else:
                    next_value = output[..., -1:, :]

                history = torch.cat([history, next_value], dim=1)

        return torch.stack(forecast_outputs, dim=-1)

    def _generate_target_mask(
        self,
        batch_size: int,
        seq_len: int,
        train_sizes: Tensor,
        corr_dims: Tensor,
    ) -> torch.Tensor:
        """
        Generate a mask for the target sequence.
        During training, mask out a random length from the last column.
        """

        # # Create sequence indices and expand to match batch dimension
        # seq_indices = torch.arange(seq_len, device=self.device).unsqueeze(
        #     0
        # )  # [1, seq_len]
        # seq_indices = seq_indices.repeat(batch_size, 1)
        # # Compare and create mask
        # mask_last = (
        #     seq_indices >= train_sizes.unsqueeze(1)
        # ).float()  # [batch_size, seq_len]

        # if self.random_train_mask_ratio > 0:
        #     mask_last = mask_last + (
        #         torch.rand_like(mask_last) < self.random_train_mask_ratio
        #     )
        #     mask_last[mask_last > 1] = 1
        # return mask_last
        full_target_mask_raw, masks_applied_dict = generate_target_mask(
            mask_mixing_rates=self.config.foundation_model_config.mask_mixing_rates,
            target_masking_schemes=self.config.foundation_model_config.masking_schemes,
            target_filtering_schemes=self.config.foundation_model_config.target_filtering_schemes,
            block_target_mask_mean=self.config.foundation_model_config.block_target_mask_mean,
            block_target_mask_range=self.config.foundation_model_config.block_target_mask_range,
            block_mask_num=self.config.foundation_model_config.block_mask_num,
            block_mask_every=self.config.foundation_model_config.block_mask_every,
            row_mask_ratio=self.config.foundation_model_config.row_mask_ratio,
            row_use_train_test=self.config.foundation_model_config.row_use_lengths,
            row_mask_min=self.config.foundation_model_config.row_mask_min,
            target_mask_ratio=self.config.training_config.target_mask_ratio,
            time_series_length=seq_len,
            batch_size=batch_size,
            num_correlates=self.config.dataset_config.num_correlates,
            device=self.device,
            train_sizes=train_sizes,
            seq_lens=torch.tensor([seq_len], device=self.device).repeat(
                batch_size, 1
            ),
            time_columns=self.config.dataset_config.time_columns,
            num_actual_correlates=corr_dims,
        )  # b, t, nc if not train_test_last, b, t if train_test_last
        assert full_target_mask_raw is not None, (
            "generate_target_mask should never return None"
        )
        assert not (
            (not self.use_full_reg)
            and (
                "train_test_last"
                != self.config.foundation_model_config.masking_schemes[0]
                and "random_last"
                != self.config.foundation_model_config.masking_schemes[0]
            )
        ), "only train_test_last, rand_last is supported for full regression"
        if (
            "train_test_last"
            not in self.config.foundation_model_config.masking_schemes
        ):
            full_target_mask_shaped = rearrange(
                full_target_mask_raw,
                "b (nc t) -> b nc t",
                nc=self.config.dataset_config.num_correlates,
            )
        else:
            # For train_test_last scheme, full_target_mask_raw is already in the right shape
            full_target_mask_shaped = full_target_mask_raw

        if self.use_full_reg:
            # apply masking using the dimensions d
            # Create a mask for all batch items and features at once
            # For each batch item i, mask the first corr_dims[i] elements for all features
            mask_indices = (
                torch.arange(
                    full_target_mask_shaped.shape[1], device=self.device
                )
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            corr_dims_expanded = (
                corr_dims.unsqueeze(1).unsqueeze(-1).to(self.device)
            )

            # Create boolean mask: True where we want to keep values, False where we want to mask
            keep_mask = mask_indices < corr_dims_expanded

            # Apply the mask: set to 0 where keep_mask is False
            full_target_mask = torch.where(
                keep_mask,
                full_target_mask_shaped,
                torch.zeros_like(full_target_mask_shaped),
            )
        else:
            full_target_mask = full_target_mask_shaped

        return full_target_mask
