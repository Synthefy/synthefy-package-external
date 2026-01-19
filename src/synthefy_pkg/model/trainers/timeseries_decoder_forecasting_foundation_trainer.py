import copy
import imp
import os
from dataclasses import asdict
from typing import Dict, Optional, Tuple, Type, Union

import lightning as L
import mlflow
import numpy as np
import torch
from einops import rearrange
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
from matplotlib import pyplot as plt

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster

# NOTE: GridICLForecaster and GridICLAvecExpertsForecaster are imported lazily
# inside methods to avoid circular imports with regressor.py
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)
from synthefy_pkg.model.load_models import load_timeseries_decoder
from synthefy_pkg.model.utils.lr_scheduler import get_lr_scheduler
from synthefy_pkg.plot_utils.ldm_plot import generate_tsdiffusion_plots
from synthefy_pkg.postprocessing.utils import (
    plot_autoregressive_loss,
    plot_learning_curve,
)
from synthefy_pkg.utils.sagemaker_utils import get_sagemaker_logger

COMPILE = False


class TimeSeriesDecoderForecastingFoundationTrainer(L.LightningModule):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.training_config = config.training_config

        # TODO: refactor exec config so .asdict works directly for all nested configs
        hparams = self.add_config_to_hparams(
            [
                "dataset_config",
                "training_config",
                "foundation_model_config",
                "metadata_encoder_config",
                "prior_config",
                "tabicl_config",
                "token_decoder_config",
                "curriculum_config",
            ],
        )

        self.save_hyperparameters(hparams)
        self.sagemaker_logger = get_sagemaker_logger()

        self.decoder_model = load_timeseries_decoder(
            config=self.config
        )  # typically synthefy_forecasting_model_v1

        self.foundation_model = True

        self.eval_bench_model_name: Optional[str] = None
        if self.config.dataset_config.run_val_with_eval_bench:
            if (
                hasattr(self.config, "tabicl_config")
                and self.config.tabicl_config.use_full_reg
            ):
                if len(self.config.tabicl_config.external_forecasts_to_use) > 0:
                    self.eval_bench_model_name = "gridicl_experts_multivariate"
                else:
                    self.eval_bench_model_name = "gridicl_multivariate"

    def add_config_to_hparams(self, names: list[str]) -> dict:
        hparams = {}
        hparams["execution_config"] = asdict(self.config)
        for name in names:
            if (
                hasattr(self.config, name)
                and getattr(self.config, name) is not None
            ):
                temp_config = copy.deepcopy(getattr(self.config, name))
                if hasattr(temp_config, "config"):
                    delattr(temp_config, "config")
                hparams[name] = asdict(temp_config)
        return hparams

    def _get_mlflow_run_id(self) -> str | None:
        """Helper to get the run_id from MLFlowLogger."""
        if self.trainer and self.trainer.loggers:
            for logger_instance in self.trainer.loggers:
                if isinstance(logger_instance, MLFlowLogger):
                    return logger_instance.run_id
        return None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        # TODO: I don't like selection based on dataloader name.
        is_foundation_dataset = self.config.dataset_config.dataloader_name in [
            "ShardedDataloaderV1",
            "V3ShardedDataloader",
            "OTFSyntheticDataloader",
            "LoadSavedSyntheticDataloader",
        ]

        decoder_input = self.decoder_model.prepare_training_input(
            batch,
            log_dir=self.log_dir,
            is_foundation_dataset=is_foundation_dataset,
        )
        output_dict = self.decoder_model(decoder_input)
        return decoder_input, output_dict

    def calculate_loss(
        self,
        decoder_input: Dict,
        output_dict: Dict,
    ) -> dict[str, torch.Tensor]:
        loss_dict = {}

        if not self.decoder_model.config.dataset_config.is_regression:
            true = decoder_input["target"].long()
            prediction = output_dict["logits"]
            target_mask = decoder_input["target_mask"]
            pshape = prediction.shape
            if torch.isnan(prediction).sum() > 0:
                logger.warning(
                    f"NaN in prediction: {torch.isnan(prediction).sum()}"
                )
            prediction[torch.isnan(prediction)] = -10
            prediction, true = (
                prediction.flatten(end_dim=-2)[target_mask.flatten() == 1],
                true.flatten()[target_mask.flatten() == 1],
            )
            denoiser_loss = torch.nn.functional.cross_entropy(prediction, true)
            # print(f"prediction: {prediction.min().item()}, {prediction.max().item()}, {pshape}")
            if torch.isnan(denoiser_loss):
                logger.error(
                    f"Input: {torch.isnan(decoder_input['target'].sum())}"
                )
                logger.error(
                    f"target mask: {torch.min(torch.where(target_mask == 1, 1, 0))}"
                )
                logger.error(f"prediction: {torch.isnan(prediction).sum()}")
                logger.error(
                    f"prediction: {torch.isnan(prediction.reshape(pshape[0], -1, pshape[-1])).nonzero()}"
                )
                logger.error(
                    f"decoder input: {torch.isnan(decoder_input['target']).sum()}"
                )
                logger.error(f"Denoiser loss is NaN: {denoiser_loss}")
                raise ValueError("Denoiser loss is NaN")
        elif self.decoder_model.config.foundation_model_config.generate_point_forecast:
            if len(output_dict["prediction"].shape) == 1:
                denoiser_loss = torch.nn.functional.mse_loss(
                    output_dict["prediction"].unsqueeze(-1),
                    decoder_input["target"],
                    reduction="mean",
                )
            else:
                denoiser_loss = (
                    torch.nn.functional.mse_loss(
                        output_dict["prediction"]
                        * output_dict["target_mask"].unsqueeze(-1),
                        decoder_input["target"]
                        * output_dict["target_mask"].unsqueeze(-1),
                        reduction="mean",
                    )
                    * torch.prod(
                        torch.tensor(
                            output_dict["prediction"].shape,
                            device=output_dict["prediction"].device,
                        )
                    )
                    / max(1, output_dict["target_mask"].sum())
                )
        elif self.decoder_model.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
            target = decoder_input["target"][..., 0]
            if "tabicl" in self.config.foundation_model_config.model_name:
                prediction_univariate = output_dict["logits_univariate"]
                prediction_multivariate = output_dict["logits_multivariate"]
                target_mask = decoder_input["target_mask"]

                multivariate_loss = self.get_multivariate_tabicl_loss(
                    prediction_multivariate, target, target_mask
                )

                if self.config.tabicl_config.train_as_univariate_forecast:
                    univariate_loss = self.get_multivariate_tabicl_loss(
                        prediction_univariate, target, target_mask
                    )
                    # print(f"univariate_loss: {univariate_loss}, multivariate_loss: {multivariate_loss}")
                    denoiser_loss = multivariate_loss + univariate_loss
                    loss_dict["uni_loss"] = univariate_loss
                    loss_dict["multi_loss"] = multivariate_loss
                else:
                    denoiser_loss = multivariate_loss

                loss_dict["multi_loss"] = multivariate_loss
            else:
                prediction = output_dict["logits"]
                target_mask = decoder_input["target_mask"]
                denoiser_loss = self.get_multivariate_tabicl_loss(
                    prediction, target, target_mask
                )

        loss_dict["loss"] = denoiser_loss

        return loss_dict

    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(
            self.decoder_model.parameters(),
            lr=self.training_config.learning_rate,
        )

        lr_scheduler = get_lr_scheduler(
            optimizer=adam_optimizer,
            lr_scheduler_config=self.training_config.lr_scheduler_config,
        )

        if lr_scheduler is None:
            return {
                "optimizer": adam_optimizer,
            }

        return {
            "optimizer": adam_optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        decoder_input, output_dict = self.forward(batch)
        loss_dict = self.calculate_loss(decoder_input, output_dict)

        # Extract the main loss for return
        loss = loss_dict["loss"]

        if not self.config.dataset_config.is_regression:
            prediction = output_dict["logits"]
            target_mask = decoder_input["target_mask"]
            # compute and log accuracy
            prediction_int = prediction.argmax(dim=-1).flatten()[
                target_mask.flatten() == 1
            ]
            target = (
                decoder_input["target"]
                .long()
                .squeeze(-1)
                .flatten()[target_mask.flatten() == 1]
            )
            accuracy = (prediction_int == target).float().mean()
            self.log(
                "acc",
                accuracy,
                sync_dist=False,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        # Log all metrics from loss_dict
        for metric_name, metric_value in loss_dict.items():
            self.log(
                f"train_{metric_name}",
                metric_value,
                sync_dist=False,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        # Additional SageMaker logging if available
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric("train_loss", loss.item())
            optimizer = self.optimizers()
            if isinstance(optimizer, torch.optim.Optimizer):
                lr = optimizer.param_groups[0]["lr"]
                self.sagemaker_logger.log_metric("learning_rate", lr)

        return loss

    def validation_step(
        self,
        batch: Union[Dict[str, torch.Tensor], EvalBatchFormat],
        batch_idx: int,
        *args,
        **kwargs,
    ):
        # TODO: support multiple dataloaders with dataloader_idx
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.validation_step
        compute_ar_loss = kwargs.get("compute_ar_loss", True)

        # Handle different batch types
        if isinstance(batch, EvalBatchFormat):
            predictions = self._run_eval_bench(batch, batch_idx, **kwargs)
            assert (
                isinstance(predictions, ForecastOutputFormat)
                and predictions.metrics is not None
            ), (
                f"Eval bench should return EvalBatchFormat, instead received {type(predictions)} ({predictions})"
            )
            # Log eval bench metrics if available
            log_dict = {}
            if predictions is not None and predictions.metrics is not None:
                log_dict = {
                    "nmae": predictions.metrics.nmae,
                    "mape": predictions.metrics.mape,
                }
            metric_or_loss = torch.tensor(
                predictions.metrics.nmae, device=self.device
            )
            batch_size = batch.batch_size

        else:
            # Normal validation flow for Dict[str, torch.Tensor] batches
            decoder_input, output_dict = self.forward(batch)
            log_dict = self.calculate_loss(decoder_input, output_dict)

            # Extract the main loss for return
            metric_or_loss = log_dict["loss"]
            batch_size = None  # will be inferred automatically

        # Log all metrics from log_dict
        for metric_name, metric_value in log_dict.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        # Additional SageMaker logging if available
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric("val_loss", metric_or_loss.item())

        # Compute AR loss if both compute_ar_loss is True
        # and we are in the first num_ar_batches of validation
        if (
            not isinstance(batch, EvalBatchFormat)
            and compute_ar_loss
            and batch_idx < self.training_config.num_ar_batches
            and not self.trainer.sanity_checking
        ):
            if "tabicl" in self.config.foundation_model_config.model_name:
                self._compute_ar_loss_for_tabicl(
                    batch, decoder_input, output_dict, batch_idx
                )
            else:
                self._compute_ar_loss(batch, decoder_input)

        return metric_or_loss

    def on_validation_start(self):
        # Only set up eval bench if explicitly requested and not during sanity check
        if (
            self.config.dataset_config.run_val_with_eval_bench
            and self.eval_bench_model_name is not None
            and not self.trainer.sanity_checking
        ):
            self._eval_bench_setup()

    def _eval_bench_setup(self):
        # Lazy imports to avoid circular imports with regressor.py
        from synthefy_pkg.fm_evals.forecasting.gridicl_avec_experts_forecaster import (
            GridICLAvecExpertsForecaster,
        )
        from synthefy_pkg.fm_evals.forecasting.gridicl_forecaster import (
            GridICLForecaster,
        )

        if self.eval_bench_model_name is None:
            raise Exception("Ensure that the eval_bench_model_name is set.")
        assert self.eval_bench_model_name in [
            "gridicl_multivariate",
            "gridicl_experts_multivariate",
        ], (
            "Only GridICLForecaster and GridICLAvecExpertsForecaster are supported for now."
        )
        forecast_length = self.config.dataset_config.forecast_length
        history_length = (
            self.config.dataset_config.time_series_length - forecast_length
        )
        if self.eval_bench_model_name == "gridicl_experts_multivariate":
            temp_config = copy.deepcopy(self.config)
            self.eval_bench_model = GridICLAvecExpertsForecaster(
                model_checkpoint_path="",
                config=temp_config,
                history_length=history_length,
                forecast_length=forecast_length,
                name="gridicl_experts_multivariate",
                trainer=self,
            )
        elif self.eval_bench_model_name == "gridicl_multivariate":
            # Standard GridICLForecaster initialization
            self.eval_bench_model = GridICLForecaster(
                model_checkpoint_path="",
                history_length=history_length,
                forecast_length=forecast_length,
                name="gridicl_multivariate",
                trainer=self,
                train_time_config=self.config,
            )

    def _run_eval_bench(self, batch: EvalBatchFormat, batch_idx: int, **kwargs):
        """Run evaluation benchmark on a batch."""
        # Only run eval bench if explicitly configured
        if not (
            self.config.dataset_config.run_val_with_eval_bench
            and hasattr(self, "eval_bench_model")
            and self.eval_bench_model is not None
        ):
            logger.warning("Eval bench is not configured, skipping batch")
            return None

        try:
            self.eval_bench_model.fit(batch, disable_tqdm=True)
            predictions = self.eval_bench_model.predict(batch)
            return predictions
        except Exception as e:
            logger.error(f"Error running eval bench: {e}")
            return None

    @rank_zero_only
    def _compute_ar_loss_for_tabicl(
        self,
        batch: Dict[str, torch.Tensor],
        decoder_input: Dict[str, torch.Tensor],
        output_dict: Dict[str, torch.Tensor],
        batch_idx: int,
        dir: str = "",
    ):
        # Create a single figure with 3 subplots
        tasks = self.decoder_model.tasks
        fig, axes = plt.subplots(1, len(tasks), figsize=(18, 6))

        mses = {task: [] for task in tasks}
        mses["external_forecast"] = []

        for i, task in enumerate(tasks):
            decoder_input = self.decoder_model.prepare_training_input_only_target_prediction(
                batch, task=task, train=False
            )
            # from IPython import embed; embed()
            with torch.no_grad():
                output_dict = self.decoder_model(decoder_input)

            target = decoder_input["target"][..., 0]
            prediction_multivariate = output_dict["logits_multivariate"]
            target_mask = decoder_input["target_mask"]
            nll_loss = self.get_multivariate_tabicl_loss(
                prediction_multivariate, target, target_mask
            )
            prediction = self.decoder_model.distribution.mean(
                prediction_multivariate
            )

            y_full = decoder_input["y_full"][0, :, 0]
            forecast_length = prediction.shape[0]
            history = y_full[:-forecast_length]
            true_forecast = y_full[-forecast_length:]
            pred_forecast = prediction

            gt = torch.cat([history, true_forecast], dim=0)
            pred = torch.cat([history, pred_forecast], dim=0)
            mae_loss = torch.nn.functional.l1_loss(
                pred[-forecast_length:], gt[-forecast_length:], reduction="mean"
            )
            mses[task].append(mae_loss)
            # Get external forecast if available
            external_forecast = None
            external_forecast_mae = None
            if "external_forecast" in decoder_input:
                external_forecast = decoder_input["external_forecast"][
                    0
                ]  # Take first batch, first channel
                external_forecast_length = external_forecast.shape[0]
                external_forecast = torch.cat(
                    [gt[:-external_forecast_length], external_forecast], dim=0
                )
                external_forecast_mae = torch.nn.functional.l1_loss(
                    external_forecast[-forecast_length:],
                    gt[-forecast_length:],
                    reduction="mean",
                )
                mses["external_forecast"].append(external_forecast_mae)

            # Plot on the corresponding subplot
            ax = axes[i]
            ax.plot(gt.cpu().numpy(), label="ground truth", color="blue")
            ax.plot(pred.cpu().numpy(), label="prediction", color="red")

            if (
                external_forecast is not None
                and external_forecast_mae is not None
            ):
                ax.plot(
                    external_forecast.cpu().numpy(),
                    label="external forecast",
                    linestyle="--",
                    alpha=0.7,
                    color="green",
                )
                ef_mae_value = external_forecast_mae.item()
                ax.set_title(
                    f"{task.upper()}\nAR MAE: {round(mae_loss.item(), 3)}, NLL: {round(nll_loss.item(), 3)}\nEF MAE: {round(ef_mae_value, 3)}"
                )
            else:
                ax.set_title(
                    f"{task.upper()}\nAR MAE: {round(mae_loss.item(), 3)}, NLL: {round(nll_loss.item(), 3)}"
                )

            ax.legend()
            ax.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        if dir == "":
            save_loc = os.path.join(
                self.log_dir,
                f"train_epoch_{self.current_epoch}_batch_{batch_idx}_all_tasks.png",
            )
        else:
            save_loc = os.path.join(
                dir,
                f"train_epoch_{self.current_epoch}_batch_{batch_idx}_all_tasks.png",
            )  # TODO: add task name to the filename
        if batch_idx == 0:
            print(f"Saving combined AR loss plot to {save_loc}")
        plt.savefig(save_loc, dpi=150, bbox_inches="tight")
        plt.close()

        # print(f"MSEs: {mses}")
        return mses

    @rank_zero_only
    def _compute_ar_loss(
        self,
        batch: Dict[str, torch.Tensor],
        decoder_input: Dict[str, torch.Tensor],
    ):
        use_text_description_mask = (
            self.config.training_config.description_mask_ratio != 0
        )
        autoregressive_forecast = self.decoder_model.autoregressive_forecast(
            batch,
            self.decoder_model,
            use_text_description_mask=use_text_description_mask,
        )
        batch_size = decoder_input["continuous"].shape[0]

        ts = decoder_input["continuous"].reshape(
            batch_size,
            -1,
            self.config.dataset_config.time_series_length,
            1,
        )[..., 0]
        history = ts[
            ...,
            : self.config.dataset_config.time_series_length
            - self.config.dataset_config.forecast_length,
        ].permute(0, 2, 1)
        forecast = ts[
            ...,
            self.config.dataset_config.time_series_length
            - self.config.dataset_config.forecast_length :,
        ].permute(0, 2, 1)
        decoder_input["history"] = history
        decoder_input["forecast"] = forecast
        autoregressive_forecast = autoregressive_forecast.permute(0, 2, 1)

        # Compute MSE between ground truth and autoregressive forecast
        # Ground truth and AR forecast shape: [batch_size, TS, num_correlates]
        mask = decoder_input["mask"]
        # Mask is of shape [batch_size, (nc x TS)]
        mask = rearrange(
            mask,
            "b (nc ts) -> b nc ts",
            nc=self.config.dataset_config.num_correlates,
        )
        forecast_mask = mask[..., -self.config.dataset_config.forecast_length :]
        forecast_mask = rearrange(forecast_mask, "b nc ts -> b ts nc")

        mse_per_sample = torch.mean(
            (decoder_input["forecast"] - autoregressive_forecast) ** 2
            * (~forecast_mask),
            dim=1,
        )

        # Log individual batch MSE (optional)
        batch_avg_mse = mse_per_sample.mean()
        self.log(
            "val_autoregressive_mse",  # Simplified name since Lightning adds _epoch
            batch_avg_mse,
            sync_dist=False,
            on_step=False,  # Don't log step values
            on_epoch=True,  # Let Lightning compute the epoch average
            prog_bar=True,  # Show in progress bar
        )

        # Additional SageMaker logging if available
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric(
                "val_autoregressive_mse", batch_avg_mse.item()
            )

        self.plot(
            decoder_input,
            autoregressive_forecast,
            train=False,
            experiment_name=self.config.experiment_name,
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", self.config.mlflow_tracking_uri
            ),
        )

    def test_step(
        self,
        batch: Union[Dict[str, torch.Tensor], EvalBatchFormat],
        batch_idx: int,
        *args,
        **kwargs,
    ):
        # Handle different batch types
        if isinstance(batch, EvalBatchFormat):
            predictions = self._run_eval_bench(batch, batch_idx, **kwargs)
            assert (
                isinstance(predictions, ForecastOutputFormat)
                and predictions.metrics is not None
            ), "Eval bench should return EvalBatchFormat"
            metric_or_loss = torch.tensor(
                predictions.metrics.nmae, device=self.device
            )
            batch_size = batch.batch_size
            log_dict = {
                "nmae": predictions.metrics.nmae,
                "mape": predictions.metrics.mape,
            }

        else:
            # Normal test flow for Dict[str, torch.Tensor] batches
            decoder_input, output_dict = self.forward(batch)
            log_dict = self.calculate_loss(decoder_input, output_dict)

            # Extract the main loss for return
            metric_or_loss = log_dict["loss"]
            batch_size = None  # will be inferred automatically

        # Log all metrics from log_dict
        for metric_name, metric_value in log_dict.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        # Additional SageMaker logging if available
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric("test_loss", metric_or_loss.item())

        return metric_or_loss

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        # Plot learning curve after each test epoch
        self._plot_learning_curve()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        # Plot learning curve after each validation epoch
        self._plot_learning_curve()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        # Plot learning curve after each training epoch
        self._plot_learning_curve()

    @rank_zero_only
    def _plot_learning_curve(self) -> None:
        """
        Internal helper to invoke plot_learning_curve once per epoch.
        """
        if self.training_config.num_devices > 1:
            return

        if self.trainer.current_epoch <= 1:
            logger.warning(
                "Skipping learning curve plot for epoch 1 because it's too early"
            )
            return

        if self.trainer.log_dir:
            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI", self.config.mlflow_tracking_uri
            )
            mlflow_run_id = self._get_mlflow_run_id()
            try:
                plot_learning_curve(
                    input_logs_dir=os.path.dirname(self.trainer.log_dir),
                    output_fig_path=os.path.join(
                        self.trainer.log_dir, "learning_curve.png"
                    ),
                    run_name=self.config.run_name,
                    dataset_name=self.config.dataset_name,
                    log_to_mlflow=self.config.use_mlflow,
                    experiment_name=self.config.experiment_name,
                    tracking_uri=tracking_uri,
                    run_id=mlflow_run_id,
                )
            except Exception as e:
                logger.error(f"Error plotting learning curve: {e}")

            try:
                # TODO: support autoregressive plots for TabICL
                if (
                    "tabicl"
                    not in self.config.foundation_model_config.model_name
                ):
                    plot_autoregressive_loss(
                        input_logs_dir=os.path.dirname(self.trainer.log_dir),
                        output_fig_path=os.path.join(
                            self.trainer.log_dir, "autoregressive_loss.png"
                        ),
                        run_name=self.config.run_name,
                        dataset_name=self.config.dataset_name,
                        log_to_mlflow=self.config.use_mlflow,
                        experiment_name=self.config.experiment_name,
                        tracking_uri=tracking_uri,
                        run_id=mlflow_run_id,
                    )
            except Exception as e:
                logger.error(f"Error plotting autoregressive loss: {e}")
        else:
            logger.warning(
                "Cannot plot learning curve: trainer.log_dir is None"
            )

    @rank_zero_only
    def plot(
        self,
        decoder_input,
        prediction,
        train=False,
        experiment_name: str = "",
        tracking_uri: str = "",
    ) -> None:
        if self.training_config.num_devices > 1:
            return
        history = decoder_input["history"]
        forecast = decoder_input["forecast"]
        forecast_prediction = prediction
        original_timeseries = torch.cat([history, forecast], dim=1)
        synthesized_timeseries = torch.cat(
            [history, forecast_prediction], dim=1
        )

        original_timeseries = (
            original_timeseries.permute(0, 2, 1).cpu().detach().numpy()
        )
        synthesized_timeseries = (
            synthesized_timeseries.permute(0, 2, 1).cpu().detach().numpy()
        )

        sample_idx = 0
        save_path = (
            os.path.join(
                self.log_dir,
                f"train_epoch_{self.current_epoch}_batch_{sample_idx}.png",
            )
            if train
            else os.path.join(
                self.log_dir,
                f"epoch_{self.current_epoch}_batch_{sample_idx}.png",
            )
        )

        # Generate plot and handle MLflow logging
        try:
            generate_tsdiffusion_plots(
                config=self.config,
                save_path=save_path,
                synthesized_timeseries=synthesized_timeseries,
                actual_timeseries=original_timeseries,
                task="forecast",
            )

            # Log forecasting plot to MLflow if enabled and plot was successfully created
            if self.config.use_mlflow and os.path.exists(save_path):
                mlflow_run_id = self._get_mlflow_run_id()
                if experiment_name and tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_experiment(experiment_name)
                mlflow.log_artifact(
                    save_path,
                    f"plots/epoch_{self.current_epoch}",
                    run_id=mlflow_run_id,
                )
        except Exception as e:
            logger.error(f"Failed to generate or log forecasting plot: {e}")

    def get_multivariate_tabicl_loss(self, prediction, target, target_mask):
        # prevent occasional nans (TODO: find out why)
        if torch.isnan(prediction).sum() > 0:
            logger.warning(
                f"NaN in prediction: {torch.isnan(prediction).sum()}"
            )
        prediction[torch.isnan(prediction)] = -10

        if len(prediction.shape) == 3:
            prediction = rearrange(prediction, "b t c -> t b c")
            target = rearrange(target, "b t -> t b")

        denoiser_loss = self.decoder_model.distribution(
            logits=prediction,
            y=target,
        )
        if len(denoiser_loss.shape) == 1:
            denoiser_loss = denoiser_loss.mean()
        else:
            denoiser_loss = denoiser_loss[
                rearrange(target_mask, "b t -> t b") == 1
            ].mean()
        return denoiser_loss
