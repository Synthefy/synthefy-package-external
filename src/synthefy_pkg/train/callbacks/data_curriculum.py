import copy
import json
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Optional

import lightning as L
import mlflow
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.model.architectures.tabicl_wrapper import TabICLModel
from synthefy_pkg.model.foundation_model.base_foundation_forecasting_model import (
    BaseFoundationForecastingModel,
)
from synthefy_pkg.model.trainers.timeseries_decoder_forecasting_foundation_trainer import (
    TimeSeriesDecoderForecastingFoundationTrainer,
)
from synthefy_pkg.prior.curriculum.curriculum import Curriculum
from synthefy_pkg.utils.curriculum_logging import (
    get_mlflow_run_id_from_trainer,
    log_curriculum_parameters_to_mlflow,
    log_curriculum_parameters_to_mlflow_with_retry,
)


class DataCurriculumCallback(L.Callback):
    def __init__(
        self,
        dataset_generator,
        curriculum_manager: Optional[Curriculum] = None,
        unique_run_id: str = "default",
        update_frequency: int = 1000,
        config_file_path: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_generator = dataset_generator
        self.curriculum_manager = curriculum_manager
        self.update_frequency = update_frequency
        self.unique_run_id = unique_run_id
        self.running_cumulative_loss = 0

        # Create a temporary file to store the shared configuration
        if config_file_path is None:
            # Create a temporary file in a location accessible by all processes
            temp_dir = Path(tempfile.gettempdir()) / "synthefy_curriculum"
            temp_dir.mkdir(exist_ok=True)
            self.config_file_path = (
                temp_dir / f"curriculum_config_{self.unique_run_id}.pkl"
            )
        else:
            self.config_file_path = Path(config_file_path)

        self.update_values(0, 0, 0, 100, None)

        # Initialize the config file with the current configuration
        valtest_copy = copy.deepcopy(self.dataset_generator.config.prior_config)
        valtest_copy.run_id = valtest_copy.run_id + "_valtest"
        self._save_config_to_file(
            valtest_copy,
            override_path=str(self.config_file_path).replace(
                ".pkl", "_valtest.pkl"
            ),
        )

    def _save_config_to_file(self, config, override_path: Optional[str] = None):
        """Save configuration to a file that can be read by worker processes."""
        try:
            # Convert config to a serializable format
            logger.debug(
                f"Saving curriculum config to {self.config_file_path} with config {config.min_train_size} {config.max_train_size}"
            )
            with open(
                override_path
                if override_path is not None
                else self.config_file_path,
                "wb",
            ) as f:
                pickle.dump(config, f)

            logger.debug(f"Saved curriculum config to {self.config_file_path}")
        except Exception as e:
            logger.exception(f"Failed to save curriculum config: {e}")

    def _load_config_from_file(self):
        """Load configuration from the shared file."""
        try:
            if self.config_file_path.exists():
                with open(self.config_file_path, "rb") as f:
                    config = pickle.load(f)

                # Update the config object with the loaded values
                self.dataset_generator.update_config(config)

                logger.debug(
                    f"Loaded curriculum config from {self.config_file_path}"
                )
                return True
            return False
        except Exception as e:
            logger.exception(f"Failed to load curriculum config: {e}")
            return False

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: TimeSeriesDecoderForecastingFoundationTrainer,
        outputs,
        batch,
        batch_idx,
    ):
        assert isinstance(outputs, dict)
        assert "loss" in outputs
        self.running_cumulative_loss += float(outputs["loss"])
        if self.curriculum_manager is not None and (
            trainer.global_step == 1
            or trainer.global_step % self.update_frequency == 0
        ):
            updated_values = self.update_values(
                trainer.global_step,
                trainer.current_epoch,
                batch_idx,
                self.running_cumulative_loss / self.update_frequency,
                pl_module,
            )

            # Log curriculum parameters to MLflow (only on main process)
            if trainer.is_global_zero:
                # Try to get MLflow run ID from trainer first
                run_id = get_mlflow_run_id_from_trainer(trainer)
                if run_id:
                    logger.debug(f"Using MLflow run ID: {run_id}")
                    log_curriculum_parameters_to_mlflow(
                        updated_values, trainer.global_step, run_id=run_id
                    )
                else:
                    logger.trace("No MLflow run ID found, using retry approach")
                    # Fall back to retry approach
                    log_curriculum_parameters_to_mlflow_with_retry(
                        updated_values, trainer.global_step
                    )

            # Add curriculum parameters to outputs for Lightning logging
            for k in updated_values:
                if isinstance(updated_values[k], dict):
                    for k1 in updated_values[k]:
                        outputs[k + "_" + k1] = updated_values[k][k1]
                else:
                    outputs[k] = updated_values[k]
            self.running_cumulative_loss = 0

    def update_values(
        self,
        global_step: int,
        epoch: int,
        batch_idx: int,
        loss: float,
        pl_module: Optional[
            TimeSeriesDecoderForecastingFoundationTrainer
        ] = None,
    ):
        update_kwargs = {
            "global_step": global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "loss": loss,
        }
        assert self.curriculum_manager is not None
        updated_configs, updated_values = self.curriculum_manager(
            **update_kwargs
        )
        # update internal config
        self.dataset_generator.update_config(updated_configs[0])
        # Save the updated configuration to the shared file
        self._save_config_to_file(updated_configs[0])

        # update the assignment features TODO: we only handle mask mixing rates for now
        self.update_assignment_features(pl_module, updated_configs[1])

        logger.debug(
            f"Updated dataset generator config: {updated_values}, assignment features: {updated_configs[1]}"
        )
        return updated_values

    def update_assignment_features(
        self,
        pl_module: Optional[TimeSeriesDecoderForecastingFoundationTrainer],
        assignment_features: DictConfig,
    ):
        if pl_module is not None:
            if hasattr(assignment_features, "mask_mixing_rates"):
                pl_module.config.foundation_model_config.mask_mixing_rates = (
                    assignment_features.mask_mixing_rates
                )
                if isinstance(
                    pl_module.decoder_model, BaseFoundationForecastingModel
                ) or isinstance(pl_module.decoder_model, TabICLModel):
                    pl_module.decoder_model.config.foundation_model_config.mask_mixing_rates = assignment_features.mask_mixing_rates
                else:
                    raise ValueError(
                        f"Model type {type(pl_module.decoder_model)} not supported"
                    )

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        """Load the latest configuration at the start of each epoch."""
        # This ensures worker processes pick up any configuration changes
        self._load_config_from_file()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Clean up the temporary config file when training ends."""
        try:
            if self.config_file_path.exists():
                self.config_file_path.unlink()
                logger.debug(
                    f"Cleaned up curriculum config file: {self.config_file_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to clean up curriculum config file: {e}")
