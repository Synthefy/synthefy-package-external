import os
from typing import Dict, Tuple

import lightning as L
import torch

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.load_models import load_timeseries_decoder
from synthefy_pkg.model.utils.lr_scheduler import get_lr_scheduler
from synthefy_pkg.plot_utils.ldm_plot import generate_tsdiffusion_plots
from synthefy_pkg.postprocessing.utils import plot_learning_curve
from synthefy_pkg.utils.sagemaker_utils import get_sagemaker_logger

COMPILE = False


class TimeSeriesDecoderForecastingTrainer(L.LightningModule):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.training_config = config.training_config
        self.sagemaker_logger = get_sagemaker_logger()

        self.decoder_model = load_timeseries_decoder(
            config=self.config
        )  # typically synthefy_forecasting_model_v1

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, torch.Tensor]:
        # TODO: I don't like selection based on dataloader name.
        if (
            self.config.dataset_config.dataloader_name == "ShardedDataloaderV1"
            or self.config.dataset_config.dataloader_name
            == "V3ShardedDataloader"
        ):
            decoder_input = self.decoder_model.prepare_training_input(
                batch, log_dir=self.log_dir, is_foundation_dataset=True
            )
        elif (
            self.config.dataset_config.dataloader_name
            == "ForecastingDataLoader"
        ):
            decoder_input = self.decoder_model.prepare_training_input(
                batch, log_dir=self.log_dir, is_foundation_dataset=False
            )
        prediction = self.decoder_model(decoder_input)
        return decoder_input, prediction

    def calculate_loss(
        self, decoder_input: Dict, prediction: torch.Tensor
    ) -> torch.Tensor:
        denoiser_loss = torch.nn.functional.mse_loss(
            prediction, decoder_input["forecast"], reduction="mean"
        )
        return denoiser_loss

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
        for key, value in batch.items():
            # if the value is a list it can't be assigned to a device (assuming not a list of tensors)
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.training_config.device)
        batch["epoch"] = torch.tensor(
            self.current_epoch, device=self.training_config.device
        )
        decoder_input, prediction = self.forward(batch)
        loss = self.calculate_loss(decoder_input, prediction)

        # Log to Lightning
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
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
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.training_config.device)
        batch["epoch"] = torch.tensor(
            self.current_epoch, device=self.training_config.device
        )
        decoder_input, prediction = self.forward(batch)

        loss = self.calculate_loss(decoder_input, prediction)

        # Log to Lightning
        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.current_epoch > 1:
            try:
                if self.trainer.log_dir is not None:
                    plot_learning_curve(
                        input_logs_dir=os.path.dirname(self.trainer.log_dir),
                        output_fig_path=os.path.join(
                            self.trainer.log_dir, "learning_curve.png"
                        ),
                        run_name=self.config.run_name,
                        dataset_name=self.config.dataset_name,
                    )
                else:
                    print("Cannot plot learning curve: trainer.log_dir is None")
            except Exception as e:
                print(f"Error plotting learning curve: {e}")

        if batch_idx == 0:
            self.plot(decoder_input, prediction, train=False)
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.training_config.device)
        batch["epoch"] = torch.tensor(
            self.current_epoch, device=self.training_config.device
        )
        decoder_input, prediction = self.forward(batch)
        loss = self.calculate_loss(decoder_input, prediction)

        # Log to Lightning
        self.log(
            "test_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Additional SageMaker logging if available
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric("test_loss", loss.item())

        return loss

    def on_validation_epoch_end(self) -> None:
        if (
            hasattr(self, "trainer")
            and self.trainer
            and hasattr(self.trainer, "callback_metrics")
            and "val_loss" in self.trainer.callback_metrics
            and isinstance(
                self.trainer.callback_metrics["val_loss"], torch.Tensor
            )
        ):
            final_val_loss = self.trainer.callback_metrics["val_loss"].item()

            # Log the final aggregated value once per epoch to SageMaker
            if self.sagemaker_logger and self.trainer.is_global_zero:
                self.sagemaker_logger.log_metric("val_loss", final_val_loss)

        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

    def plot(self, decoder_input, prediction, train=False) -> None:
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
        generate_tsdiffusion_plots(
            config=self.config,
            save_path=save_path,
            synthesized_timeseries=synthesized_timeseries,
            actual_timeseries=original_timeseries,
            task="forecast",
        )
