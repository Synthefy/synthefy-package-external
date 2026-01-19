import os
from typing import Dict, Optional

import lightning as L
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from lightning.fabric.utilities.rank_zero import rank_zero_only
from loguru import logger
from matplotlib.axes import Axes

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with 'pip install mlflow' to enable MLflow logging.")


class PredictionTargetPlotCallback(L.Callback):
    """
    A Lightning callback that periodically saves plots comparing model predictions to targets.

    The plots show predictions in red and targets in green, making it easy to visualize
    how well the model is performing during training.
    """

    def __init__(
        self,
        plot_every_n_steps: int = 1000,
        max_samples_per_plot: int = 10,
        save_dir: Optional[str] = None,
        log_to_mlflow: bool = False,
        tracking_uri: str = "",
        run_id: str | None = None,
        experiment_name: str = "",
    ):
        """
        Initialize the callback.

        Args:
            plot_every_n_steps: How often to generate plots (every N steps)
            max_samples_per_plot: Maximum number of samples to plot per figure
            save_dir: Directory to save plots (if None, uses trainer's log_dir)
            log_to_mlflow: Whether to log plots as MLflow artifacts
            tracking_uri: MLflow tracking URI
            run_id: Run ID for logging to MLflow
            experiment_name: Name of the MLflow experiment
        """
        super().__init__()
        self.plot_every_n_steps = plot_every_n_steps
        self.max_samples_per_plot = max_samples_per_plot
        self.save_dir = save_dir
        self.log_to_mlflow = log_to_mlflow
        self.tracking_uri = tracking_uri
        self.run_id = run_id
        self.experiment_name = experiment_name

        if self.log_to_mlflow and not MLFLOW_AVAILABLE:
            logger.warning("MLflow logging requested but MLflow is not available. Install with 'pip install mlflow'.")
            self.log_to_mlflow = False

    def _get_save_dir(self, trainer: L.Trainer) -> str:
        """Get the directory to save plots."""
        if self.save_dir:
            return self.save_dir
        elif trainer.log_dir:
            return trainer.log_dir
        else:
            return "."

    def _extract_predictions_and_targets(
        self, pl_module: L.LightningModule, batch: Dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Extract predictions and targets from the model output.

        Returns:
            tuple: (predictions, targets, plot_type)
        """
        # Get model predictions
        decoder_input, output_dict = pl_module.forward(batch)

        # Extract predictions based on model type and configuration
        if (
            hasattr(pl_module, "foundation_model")
            and pl_module.foundation_model
        ):
            is_full_reg = (
                hasattr(pl_module.decoder_model.config, "tabicl_config")
                and pl_module.decoder_model.config.tabicl_config.use_full_reg
            )
            target_mask = decoder_input["target_mask"].detach().cpu().numpy()
            corr_dims = decoder_input["useful_features"].detach().cpu().numpy()

            if pl_module.decoder_model.config.foundation_model_config.generate_point_forecast:
                predictions = output_dict["prediction"].detach().cpu().numpy()
                targets = decoder_input["target"].detach().cpu().numpy()
                losses = np.concatenate(
                    [
                        -np.ones_like(
                            output_dict["loss"].detach().cpu().numpy()
                        ),
                        output_dict["loss"].detach().cpu().numpy(),
                        np.abs(predictions - targets),
                    ],
                    axis=1,
                )
                plot_type = "point_forecast"
            elif pl_module.decoder_model.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
                # For probabilistic forecasts, use the mean of the distribution
                logits = output_dict["logits"]
                if len(logits.shape) == 3:
                    logits = rearrange(logits, "b t c -> t b c")
                predictions = (
                    pl_module.decoder_model.distribution.mean(logits)
                    .detach()
                    .cpu()
                    .numpy()
                )
                if not is_full_reg:
                    target = rearrange(
                        decoder_input["target"][..., 0], "b t -> t b"
                    )
                else:
                    target = decoder_input["target"]
                denoiser_loss = (
                    pl_module.decoder_model.distribution(
                        logits=logits,
                        y=target,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                targets = decoder_input["target"][..., 0].detach().cpu().numpy()
                if is_full_reg:
                    losses = np.stack(
                        [
                            denoiser_loss,
                            np.square(predictions - targets),
                            np.abs(predictions - targets),
                        ],
                        axis=1,
                    )
                else:
                    predicted_targets = targets[target_mask == 1].reshape(
                        predictions.shape[1], -1
                    )
                    used_predictions = (
                        predictions[target_mask.transpose(1, 0) == 1]
                        .reshape(-1, predictions.shape[1])
                        .transpose(1, 0)
                    )
                    valid_loss_values = (
                        denoiser_loss[target_mask.transpose(1, 0) == 1]
                        .reshape(-1, predictions.shape[1])
                        .transpose(1, 0)
                    )
                    losses = np.stack(
                        [
                            valid_loss_values,
                            np.square(used_predictions - predicted_targets),
                            np.abs(used_predictions - predicted_targets),
                        ],
                        axis=1,
                    ).mean(axis=-1)
                    logger.info(
                        f"nll loss: {losses[: min(len(targets), self.max_samples_per_plot), 0]}, mse loss: {losses[: min(len(targets), self.max_samples_per_plot), 1]}, mae loss: {losses[: min(len(targets), self.max_samples_per_plot), 2]}, variance: {np.var(predicted_targets, axis=-1)[: min(len(targets), self.max_samples_per_plot)]}"
                    )
                plot_type = "probabilistic_forecast"
            else:
                # For classification tasks
                predictions = (
                    output_dict["logits"].argmax(dim=-1).detach().cpu().numpy()
                )
                targets = decoder_input["target"].long().detach().cpu().numpy()
                losses = np.stack(
                    [
                        -np.ones_like(
                            output_dict["loss"].detach().cpu().numpy()
                        ),
                        np.mean(predictions - targets, axis=0),
                        np.abs(predictions - targets),
                    ],
                    axis=1,
                )
                plot_type = "classification"
            if is_full_reg:
                values_full = decoder_input["values_full"].clone()
                targets = values_full.transpose(2, 1).detach().cpu().numpy()
                plot_type = "full_reg"

        else:
            # For non-foundation models, assume direct prediction
            predictions = output_dict.detach().cpu().numpy()
            targets = decoder_input["target"].detach().cpu().numpy()
            losses = np.concatenate(
                [
                    -np.ones_like(output_dict["loss"].detach().cpu().numpy()),
                    output_dict["loss"].detach().cpu().numpy(),
                    np.abs(predictions - targets),
                ],
                axis=0,
            )
            plot_type = "direct_prediction"
            corr_dims = np.ones(targets.shape[0]) * targets.shape[1]
        return predictions, targets, target_mask, corr_dims, losses, plot_type

    def _create_comparison_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
        corr_dims: np.ndarray,
        losses: np.ndarray,
        plot_type: str,
        epoch: int,
        save_path: str,
    ) -> None:
        """
        Create and save a comparison plot of predictions vs targets.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            plot_type: Type of prediction (for title)
            epoch: Current epoch number
            save_path: Path to save the plot
        """
        if plot_type == "full_reg":
            num_samples = min(len(targets), self.max_samples_per_plot)
            fig, axes = plt.subplots(
                num_samples, 1, figsize=(12, 4 * num_samples), sharex=True
            )
            target_mask = target_mask.astype(int)
            start_indices = []
            end_indices = [0]
            remainder = 0
            for i, d_i in zip(range(target_mask.shape[0]), corr_dims):
                start_indices.append(
                    end_indices[-1]
                    + np.sum(target_mask[i, : d_i - 1])
                    + remainder
                )
                end_indices.append(
                    start_indices[-1] + np.sum(target_mask[i, d_i - 1])
                )
                remainder = np.sum(target_mask[i, d_i:])
            end_indices = end_indices[1:]
            losses = np.stack(
                [
                    np.mean(losses[s:e], axis=0)
                    for s, e in zip(start_indices, end_indices)
                ],
                axis=0,
            )
            for i in range(num_samples):
                current_ax: Axes = axes[i]

                # Get predictions where target mask is true
                corresponding_predictions = predictions[
                    start_indices[i] : end_indices[i]
                ]
                d_i = corr_dims[i]
                corresponding_indices = np.where(target_mask[i, d_i - 1])[0]

                # Create scatterplot for values where target mask is true
                if len(corresponding_indices) > 0:
                    current_ax.scatter(
                        corresponding_indices,
                        corresponding_predictions,
                        label="Prediction (masked)",
                        color="red",
                        alpha=0.7,
                        s=10,
                        marker=markers.MarkerStyle("."),
                    )

                # now plot the target values
                target = targets[i, d_i - 1]
                time_steps = np.arange(len(target))
                current_ax.plot(
                    time_steps,
                    target,
                    label="Target",
                    color="green",
                    linewidth=2,
                    alpha=0.8,
                )

                # Add loss annotation
                if i < len(losses):
                    loss_value = losses[i]
                    current_ax.text(
                        0.02,
                        0.98,
                        f"Dist Loss: {loss_value[0]:.4f}",
                        transform=current_ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round", facecolor="white", alpha=0.8
                        ),
                    )
                    current_ax.text(
                        0.02,
                        0.90,
                        f"MSE Loss: {loss_value[1]:.4f}",
                        transform=current_ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round", facecolor="white", alpha=0.8
                        ),
                    )
                    current_ax.text(
                        0.02,
                        0.82,
                        f"MAE Loss: {loss_value[2]:.4f}",
                        transform=current_ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round", facecolor="white", alpha=0.8
                        ),
                    )

                current_ax.set_title(
                    f"Sample {i + 1} - {plot_type.replace('_', ' ').title()}"
                )
                current_ax.set_xlabel("Time Steps")
                current_ax.set_ylabel("Value")
                current_ax.legend()
                current_ax.grid(True, alpha=0.3)

            # Add overall title and save
            fig.suptitle(
                f"Predictions vs Targets - Epoch {epoch}", fontsize=16, y=0.98
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.trace(
                f"Saved prediction-target comparison plot to {save_path}"
            )

            # Log to MLflow if enabled
            if self.log_to_mlflow and MLFLOW_AVAILABLE:
                try:
                    # Set tracking URI and experiment if provided
                    if self.experiment_name and self.tracking_uri:
                        mlflow.set_tracking_uri(self.tracking_uri)
                        mlflow.set_experiment(self.experiment_name)
                    mlflow.log_artifact(save_path, artifact_path="prediction_plots", run_id=self.run_id)
                    logger.trace("Logged prediction plot to MLflow as artifact")
                except Exception as e:
                    logger.exception(f"Failed to log plot to MLflow: {e}")

            return

        # Ensure we have the same number of samples
        num_samples = min(
            len(predictions), len(targets), self.max_samples_per_plot
        )

        if num_samples == 0:
            logger.warning("No samples to plot")
            return

        # Create subplots
        fig, axes = plt.subplots(
            num_samples, 1, figsize=(12, 4 * num_samples), sharex=True
        )

        # Handle single sample case
        if num_samples == 1:
            axes = [axes]
        elif num_samples == 0:
            return

        used_predictions = (
            predictions[target_mask.transpose(1, 0) == 1]
            .reshape(-1, predictions.shape[1])
            .transpose(1, 0)
        )
        for i in range(num_samples):
            ax: Axes = axes[i]  # type: ignore

            # Get the data for this sample
            pred = used_predictions[i]

            target = targets[i][len(targets[i]) // 2 :]
            pred = np.concatenate(
                [target[: len(target) - len(pred)], pred], axis=0
            )
            # Add loss annotation
            loss_value = losses[i]
            ax.text(
                0.02,
                0.98,
                f"Dist Loss: {loss_value[0]:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.text(
                0.02,
                0.90,
                f"MSE Loss: {loss_value[1]:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.text(
                0.02,
                0.82,
                f"MAE Loss: {loss_value[2]:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Handle different shapes
            if len(pred.shape) == 1:
                # 1D time series
                time_steps = range(len(pred))
                ax.plot(
                    time_steps,
                    target,
                    label="Target",
                    color="green",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.plot(
                    time_steps,
                    pred,
                    label="Prediction",
                    color="red",
                    linewidth=2,
                    alpha=0.8,
                    linestyle="--",
                )
                # create a vertical line at len(pred) - len(target)
                ax.axvline(
                    x=len(pred) - len(target),
                    color="black",
                    linestyle="--",
                    linewidth=2,
                )
            elif len(pred.shape) == 2:
                # 2D: (time_steps, features)
                time_steps = range(pred.shape[0])
                for j in range(min(pred.shape[1], 3)):  # Plot first 3 features
                    ax.plot(
                        time_steps,
                        target[:, j],
                        label=f"Target (feature {j})",
                        color="green",
                        linewidth=2,
                        alpha=0.8,
                    )
                    ax.plot(
                        time_steps,
                        pred[:, j],
                        label=f"Prediction (feature {j})",
                        color="red",
                        linewidth=1,
                        alpha=0.8,
                    )
            else:
                # For higher dimensions, flatten or take first dimension
                pred_flat = pred.flatten()
                target_flat = target.flatten()
                time_steps = range(len(pred_flat))
                ax.plot(
                    time_steps,
                    target_flat,
                    label="Target",
                    color="green",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.plot(
                    time_steps,
                    pred_flat,
                    label="Prediction",
                    color="red",
                    linewidth=2,
                    alpha=0.8,
                    linestyle="--",
                )

            ax.set_title(
                f"Sample {i + 1} - {plot_type.replace('_', ' ').title()}"
            )
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle(
            f"Predictions vs Targets - Epoch {epoch}", fontsize=16, y=0.98
        )
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.trace(f"Saved prediction-target comparison plot to {save_path}")

        # Log to MLflow if enabled
        if self.log_to_mlflow and MLFLOW_AVAILABLE:
            try:
                # Set tracking URI and experiment if provided
                if self.experiment_name and self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                    mlflow.set_experiment(self.experiment_name)
                mlflow.log_artifact(save_path, artifact_path="prediction_plots", run_id=self.run_id)
                logger.trace("Logged prediction plot to MLflow as artifact")
            except Exception as e:
                logger.exception(f"Failed to log plot to MLflow: {e}")

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        """Generate plots at the end of training epochs (optional)."""
        # Only plot every N epochs and only after some training has occurred

        if trainer.global_step % self.plot_every_n_steps == 0:
            # Get a batch from training dataloader
            try:
                assert trainer.train_dataloader is not None
                batch = next(iter(trainer.train_dataloader))
            except StopIteration:
                logger.warning("No training data available for plotting")
                return

            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(pl_module.device)

            # Set model to eval mode
            pl_module.eval()

            with torch.no_grad():
                # Extract predictions and targets
                (
                    predictions,
                    targets,
                    target_mask,
                    corr_dims,
                    losses,
                    plot_type,
                ) = self._extract_predictions_and_targets(pl_module, batch)

                # Create save path
                save_dir = self._get_save_dir(trainer)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir,
                    f"train_prediction_target_comparison_step_{trainer.global_step}.png",
                )

                # Create and save the plot
                self._create_comparison_plot(
                    predictions,
                    targets,
                    target_mask,
                    corr_dims,
                    losses,
                    plot_type,
                    trainer.current_epoch,
                    save_path,
                )

            # Restore training mode
            pl_module.train()
