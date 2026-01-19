import os
from typing import Dict, Tuple

import lightning as L
import torch
from loguru import logger

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.utils.lr_scheduler import get_lr_scheduler
from synthefy_pkg.plot_utils.ldm_plot import generate_tsdiffusion_plots
from synthefy_pkg.postprocessing.utils import plot_learning_curve
from synthefy_pkg.utils.sagemaker_utils import get_sagemaker_logger

COMPILE = True


class TimeSeriesDiffusionModelTrainer(L.LightningModule):
    def __init__(self, config: Configuration, diffusion_model: torch.nn.Module):
        super().__init__()
        self.config = config

        self.log_dir = config.log_dir
        self.dataset_config = config.dataset_config
        self.denoiser_config = config.denoiser_config
        self.training_config = config.training_config

        self.denoiser_model = diffusion_model

        self.sagemaker_logger = get_sagemaker_logger()

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, torch.Tensor]:
        denoiser_input = self.denoiser_model.prepare_training_input(batch)
        noise_est = self.denoiser_model(denoiser_input)
        return denoiser_input, noise_est

    def calculate_loss(
        self, denoiser_input: Dict, noise_est: torch.Tensor
    ) -> torch.Tensor:
        denoiser_loss = torch.nn.functional.mse_loss(
            noise_est, denoiser_input["noise"], reduction="mean"
        )
        return denoiser_loss

    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(
            self.denoiser_model.parameters(),
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
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)
        denoiser_loss = self.calculate_loss(denoiser_input, noise_est)

        # Log to Lightning
        self.log(
            "train_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Add SageMaker logging
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric("train_loss", denoiser_loss.item())
            optimizer = self.optimizers()
            if isinstance(optimizer, torch.optim.Optimizer):
                lr = optimizer.param_groups[0]["lr"]
                self.sagemaker_logger.log_metric("learning_rate", lr)

        return denoiser_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)
        denoiser_loss = self.calculate_loss(denoiser_input, noise_est)

        # Log to Lightning
        self.log(
            "val_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.current_epoch > 1:
            log_dir = self.trainer.log_dir
            if log_dir is not None:
                plot_learning_curve(
                    input_logs_dir=os.path.dirname(log_dir),
                    output_fig_path=os.path.join(log_dir, "learning_curve.png"),
                    run_name=self.config.run_name,
                    dataset_name=self.config.dataset_name,
                )

        # Add SageMaker logging
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric("val_loss", denoiser_loss.item())

        if batch_idx == 0:
            self.plot(batch)
        return denoiser_loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)

        denoiser_loss = torch.nn.functional.mse_loss(
            noise_est, denoiser_input["noise"], reduction="mean"
        )
        self.log(
            "test_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Add SageMaker logging
        if (
            self.sagemaker_logger
            and self.trainer.is_global_zero
            and self.trainer.is_last_batch
        ):
            self.sagemaker_logger.log_metric("test_loss", denoiser_loss.item())

        return denoiser_loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

    def plot(self, batch: Dict[str, torch.Tensor], train=False) -> None:
        with torch.no_grad():
            dict_ = get_synthesis_via_diffusion(
                batch=batch,
                synthesizer=self.denoiser_model,
            )
            sampled_input = self.denoiser_model.prepare_training_input(batch)
            original_timeseries = sampled_input["sample"]
            original_timeseries = self.denoiser_model.prepare_output(
                original_timeseries
            )
            synthesized_timeseries = dict_["timeseries"]

            save_path = os.path.join(
                self.config.base_path,
                self.config.save_path,
                self.config.dataset_config.dataset_name,
                self.config.experiment_name,
                self.config.run_name,
                (
                    f"train_epoch_{self.current_epoch}_batch_0.png"
                    if train
                    else f"epoch_{self.current_epoch}_batch_0.png"
                ),
            )
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            generate_tsdiffusion_plots(
                config=self.config,
                save_path=save_path,
                synthesized_timeseries=synthesized_timeseries,
                actual_timeseries=original_timeseries,
                labels=None,
            )


def get_synthesis_via_diffusion(
    batch, synthesizer, similarity_guidance_dict=None
):
    """
    Synthesizes data using the diffusion model.

    Args:
        batch (dict): The input batch containing the data to be synthesized.
        synthesizer (Synthesizer): The synthesizer object used for synthesis.
        similarity_guidance_dict (dict, optional): A dictionary containing similarity guidance information. Defaults to None.

    Returns:
        dict: A dictionary containing the synthesized data and associated conditions.
    """
    T, Alpha, Alpha_bar, Sigma = (
        synthesizer.diffusion_hyperparameters["T"],
        synthesizer.diffusion_hyperparameters["Alpha"],
        synthesizer.diffusion_hyperparameters["Alpha_bar"],
        synthesizer.diffusion_hyperparameters["Sigma"],
    )
    device = synthesizer.device
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)

    input_ = synthesizer.prepare_training_input(batch)
    discrete_cond_input = input_["discrete_cond_input"]
    continuous_cond_input = input_["continuous_cond_input"]
    sample = input_["sample"]
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
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

            epsilon_theta = synthesizer(synthesis_input)
            x = (
                x
                - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
            ) / torch.sqrt(Alpha[t])
            noise = torch.randn_like(x).to(device)
            if t > 0:
                x = x + Sigma[t] * noise

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = (
        batch["discrete_label_embedding"].detach().cpu().numpy()
    )
    continuous_conditions = (
        batch["continuous_label_embedding"].detach().cpu().numpy()
    )

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


def forecast_via_diffusion(batch, synthesizer, similarity_guidance_dict=None):
    T, Alpha, Alpha_bar, Sigma = (
        synthesizer.diffusion_hyperparameters["T"],
        synthesizer.diffusion_hyperparameters["Alpha"],
        synthesizer.diffusion_hyperparameters["Alpha_bar"],
        synthesizer.diffusion_hyperparameters["Sigma"],
    )
    device = synthesizer.device
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)

    input_ = synthesizer.prepare_training_input(batch)
    discrete_cond_input = input_["discrete_cond_input"]
    continuous_cond_input = input_["continuous_cond_input"]
    timestamp_cond_input = input_["timestamp_cond_input"]

    time_series_length = input_["noise"].shape[-1]
    sample = input_["sample"]
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            x[:, :, :time_series_length] = sample[
                :, :, :time_series_length
            ]  # assigin the observed part of the time series
            print(t)
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
                "timestamp_cond_input": timestamp_cond_input,
                "diffusion_step": diffusion_steps,
            }

            epsilon_theta = synthesizer(synthesis_input)
            x = (
                x
                - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
            ) / torch.sqrt(Alpha[t])
            noise = torch.randn_like(x).to(device)
            if t > 0:
                x = x + Sigma[t] * noise

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = (
        batch["discrete_label_embedding"].detach().cpu().numpy()
    )
    continuous_conditions = (
        batch["continuous_label_embedding"].detach().cpu().numpy()
    )

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict
