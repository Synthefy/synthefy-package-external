import os
from datetime import datetime
from typing import List, Union

import lightning as L
import mlflow
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from loguru import logger

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.prior.curriculum.curriculum import get_curriculum_manager
from synthefy_pkg.train.callbacks.check_test_loss import CheckTestLossCallback
from synthefy_pkg.train.callbacks.data_curriculum import DataCurriculumCallback
from synthefy_pkg.train.callbacks.global_step_checkpoint import (
    GlobalStepCheckpointCallback,
)
from synthefy_pkg.train.callbacks.log_file_artifact_callback import (
    LogFileArtifactCallback,
)
from synthefy_pkg.train.callbacks.mlflow_system_monitor_callback import (
    MLFlowSystemMonitorCallback,
)
from synthefy_pkg.train.callbacks.prediction_target_plot_callback import (
    PredictionTargetPlotCallback,
)
from synthefy_pkg.utils.basic_utils import ENDC, OKBLUE, get_num_devices

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

COMPILE = False


def is_mlflow_server_running(tracking_uri: str) -> bool:
    try:
        response = requests.get(f"{tracking_uri}/health", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def validate_mlflow_experiment(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(
            f"Mlflow experiment {experiment_name} not found. Creating new experiment."
        )
        mlflow.create_experiment(experiment_name)
    elif experiment.lifecycle_stage == "deleted":
        logger.info(
            f"Mlflow experiment {experiment_name} is deleted. Restoring it."
        )
        mlflow.MlflowClient().restore_experiment(experiment.experiment_id)


class ModelTrain:
    def __init__(
        self,
        config: Configuration,
        dataset_generator,
        model_trainer,
        start_epoch=0,
        global_step=0,
        use_test_loss_callback: bool = True,
    ) -> None:
        self.config = config
        self.dataset_config = config.dataset_config
        self.dataset_generator = dataset_generator
        self.model_trainer: L.LightningModule = model_trainer
        self.start_epoch = start_epoch
        self.global_step = global_step
        self.training_config = config.training_config
        self.use_test_loss_callback = use_test_loss_callback
        self.make_artifact_directory()

    def make_artifact_directory(self):
        self.log_dir = self.config.get_log_dir()
        self.save_path = os.path.join(self.log_dir, "checkpoints")
        logger.info(f"Model save path: {self.save_path}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        total_epochs = self.config.training_config.max_epochs
        remaining_epochs = total_epochs - self.start_epoch

        logger.info(
            f"{OKBLUE}Training started. Save path: {self.save_path}{ENDC}"
        )
        logger.info(
            f"Starting from epoch {self.start_epoch}, global step {self.global_step}"
        )

        if self.config.training_config.use_early_stopping:
            logger.info(
                f"Early stopping enabled with patience {self.training_config.patience}"
            )
        elif remaining_epochs != total_epochs:
            logger.info(f"Training for remaing {remaining_epochs} epochs")
        else:
            logger.info(f"Training for {total_epochs} epochs")

        # Check validation data availability and set up appropriate checkpoint callback
        # if no length, assume it's an infinite dataset
        has_validation_data = (
            len(self.dataset_generator.val_dataloader()) > 0
            if (
                hasattr(self.dataset_generator.val_dataloader(), "dataset")
                and "__len__"
                in dir(self.dataset_generator.val_dataloader().dataset)
            )
            else True
        )

        checkpoint_kwargs = {
            "dirpath": self.save_path,
            "every_n_epochs": 1,
            "enable_version_counter": False,
            "save_on_train_epoch_end": True,
        }

        save_all_checkpoints = self.training_config.save_all_checkpoints

        if has_validation_data and not save_all_checkpoints:
            checkpoint_kwargs.update(
                {
                    "monitor": self.config.save_key,
                    "filename": "best_model",
                    "mode": "min",
                }
            )
            logger.info(
                f"Validation data detected. Using validation metric {self.config.save_key} for checkpointing."
            )
        elif has_validation_data and save_all_checkpoints:
            if self.training_config.save_checkpoint_every_n_epochs > 0:
                logger.info(
                    f"Saving checkpoints every {self.training_config.save_checkpoint_every_n_epochs} epochs"
                )
                assert (
                    self.training_config.save_checkpoint_every_n_steps == -1
                ), (
                    "save_checkpoint_every_n_steps should be -1 when save_checkpoint_every_n_epochs is set"
                )
                checkpoint_kwargs.update(
                    {
                        "filename": "epoch_{epoch:03d}",  # Unique filename per epoch
                        "save_top_k": -1,  # Save all, but only at specified intervals
                        "every_n_epochs": self.training_config.save_checkpoint_every_n_epochs,  # Save every 5 epochs
                    }
                )
            elif self.training_config.save_checkpoint_every_n_steps > 0:
                logger.info(
                    f"Saving checkpoints every {self.training_config.save_checkpoint_every_n_steps} steps"
                )
                assert (
                    self.training_config.save_checkpoint_every_n_epochs == -1
                ), (
                    "save_checkpoint_every_n_epochs should be -1 when save_checkpoint_every_n_steps is set"
                )
            else:
                raise ValueError(
                    "Neither save_checkpoint_every_n_epochs nor save_checkpoint_every_n_steps are set. Please set one."
                )
            logger.info("Validation data detected.")
        else:
            checkpoint_kwargs.update(
                {
                    "filename": "best_model",
                    "save_top_k": 1,
                }
            )
            logger.info(
                "No validation data detected. Will be saving the last model."
            )

        checkpoint_callbacks = []
        if self.config.training_config.save_checkpoint_every_n_steps > 0:
            checkpoint_callbacks.append(
                GlobalStepCheckpointCallback(
                    self.save_path,
                    self.config.training_config.save_checkpoint_every_n_steps,
                    self.config.training_config.pred_plot_freq,
                    self.config.training_config.num_ar_batches,
                    self.dataset_generator,
                )
            )
        else:
            checkpoint_callbacks.append(ModelCheckpoint(**checkpoint_kwargs))

        # Add periodic checkpoint callback if configured (saves every N epochs regardless of performance)
        if self.config.training_config.save_periodic_checkpoint_every_n_epochs > 0:
            periodic_checkpoint_kwargs = {
                "dirpath": self.save_path,
                "filename": "periodic_epoch_{epoch:03d}",
                "every_n_epochs": self.config.training_config.save_periodic_checkpoint_every_n_epochs,
                "save_top_k": -1,  # Save all periodic checkpoints
                "enable_version_counter": False,
                "save_on_train_epoch_end": True,
            }
            checkpoint_callbacks.append(ModelCheckpoint(**periodic_checkpoint_kwargs))
            logger.info(
                f"Added periodic checkpoint saver: every {self.config.training_config.save_periodic_checkpoint_every_n_epochs} epochs"
            )

        if len(self.config.dataset_config.curriculum_config_path) > 0:
            checkpoint_callbacks.append(
                DataCurriculumCallback(
                    self.dataset_generator,
                    get_curriculum_manager(
                        self.config.curriculum_config,
                        self.config.prior_config,
                        self.config.assignment_config
                        if hasattr(self.config, "assignment_config")
                        else None,
                    ),
                    unique_run_id=(
                        self.config.experiment_name
                        + "_"
                        + self.config.run_name
                        + "_"
                        + datetime.now().strftime("%Y%m%d%H%M%S")
                    ),
                    update_frequency=self.config.curriculum_config.update_frequency,
                )
            )

        # Early stopping
        if self.training_config.use_early_stopping:
            if self.training_config.patience is None:
                raise ValueError(
                    "patience must be set if use_early_stopping is True"
                )
            early_stopping = EarlyStopping(
                monitor=self.config.save_key,
                min_delta=0.00,
                patience=self.training_config.patience,
                verbose=False,
                mode="min",
            )
            checkpoint_callbacks.append(early_stopping)

        if self.use_test_loss_callback:
            test_mse_callback = CheckTestLossCallback(
                test_dataloader=self.dataset_generator.test_dataloader(),
                check_test_every_n_epoch=self.config.training_config.check_test_every_n_epoch,
            )
            checkpoint_callbacks.append(test_mse_callback)

        # Add Learning Rate Monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks.append(lr_monitor)

        L.seed_everything(self.config.seed)

        kwargs = {}
        if self.training_config.strategy != "None":
            kwargs["strategy"] = self.training_config.strategy
        if hasattr(self.config, "model_checkpoint_path"):
            pass
            # if self.config.model_checkpoint_path != "":
            # pl_model = model_type.load_from_checkpoint(
            #     config.model_checkpoint_path,
            #     config=config,
            #     scaler=pl_dataloader.scaler if hasattr(pl_dataloader, "scaler") else None,
            # )
            # logger.info(OKBLUE + "model loaded from checkpoint" + ENDC)
            # pl_model.log_dir = log_dir
        torch.set_float32_matmul_precision("high")

        # Tell Lightning where the tracking server is (fallback to localhost for Mode A)
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", self.config.mlflow_tracking_uri
        )

        loggers: list[Union[CSVLogger, MLFlowLogger]] = [
            CSVLogger(save_dir=self.save_path)
        ]

        if self.config.use_mlflow and is_mlflow_server_running(tracking_uri):
            validate_mlflow_experiment(
                tracking_uri, self.config.experiment_name
            )

            mlflow_logger = MLFlowLogger(
                experiment_name=self.config.experiment_name,
                run_name=self.config.run_name,
                tracking_uri=tracking_uri,  # UI at this address
                log_model=True,  # logs .ckpt as artifacts
            )
            loggers.append(mlflow_logger)
            logger.info("Using MLFlow logger")
        elif self.config.use_mlflow:
            logger.error(
                f"MLFlow server is not running at {tracking_uri}. Disabling MLFlow logging."
            )
            self.config.use_mlflow = False

        # Callbacks that interface with MLFlow
        if self.config.training_config.pred_plot_freq > 0:
            os.makedirs(
                os.path.join(self.save_path, "training_plots"), exist_ok=True
            )
            checkpoint_callbacks.append(
                PredictionTargetPlotCallback(
                    plot_every_n_steps=self.config.training_config.pred_plot_freq,
                    max_samples_per_plot=self.config.training_config.max_samples_per_plot,
                    save_dir=os.path.join(self.save_path, "training_plots"),
                    log_to_mlflow=self.config.use_mlflow,
                    tracking_uri=tracking_uri if self.config.use_mlflow else "",
                    run_id=mlflow_logger.run_id if self.config.use_mlflow else None,
                    experiment_name=self.config.experiment_name,
                )
            )

        if self.config.use_mlflow and is_mlflow_server_running(tracking_uri):
            log_file_callback = LogFileArtifactCallback(
                config=self.config,
                log_every_n_steps=self.config.training_config.push_logs_every_n_steps,
                tracking_uri=tracking_uri,
                run_id=mlflow_logger.run_id,
                experiment_name=self.config.experiment_name,
            )
            checkpoint_callbacks.append(log_file_callback)

        if self.config.use_mlflow and is_mlflow_server_running(tracking_uri):
            system_monitor_callback = MLFlowSystemMonitorCallback(
                config=self.config,
                tracking_uri=tracking_uri,
                run_id=mlflow_logger.run_id,
                experiment_name=self.config.experiment_name,
            )
            checkpoint_callbacks.append(system_monitor_callback)

        trainer = L.Trainer(
            accelerator=self.training_config.device,
            devices=get_num_devices(self.training_config.num_devices),
            max_epochs=remaining_epochs,  # Use remaining_epochs instead of max_epochs
            check_val_every_n_epoch=(
                self.training_config.check_val_every_n_epoch
                if self.training_config.check_val_every_n_epoch is not None
                and self.training_config.check_val_every_n_epoch > 0
                else None
            ),
            val_check_interval=self.training_config.val_check_interval,
            log_every_n_steps=self.training_config.log_every_n_steps,
            default_root_dir=self.save_path,
            callbacks=checkpoint_callbacks,
            logger=loggers,
            precision=self.training_config.precision,
            num_sanity_val_steps=0,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            gradient_clip_val=self.training_config.gradient_clipping,
            **kwargs,
        )

        # Evaluate on validation set before training starts
        if self.config.run_validation_before_training:
            if hasattr(self.model_trainer, "decoder_model"):
                self.model_trainer.decoder_model.eval()
            if hasattr(self.model_trainer, "denoiser_model"):
                self.model_trainer.denoiser_model.eval()
            trainer.validate(
                self.model_trainer,
                self.dataset_generator.val_dataloader(),
            )

        # Fit the model
        if hasattr(self.model_trainer, "decoder_model"):
            self.model_trainer.decoder_model.train()
        if hasattr(self.model_trainer, "denoiser_model"):
            self.model_trainer.denoiser_model.train()
        trainer.fit(
            self.model_trainer,
            self.dataset_generator.train_dataloader(),
            self.dataset_generator.val_dataloader(),
        )

        if hasattr(self.model_trainer, "decoder_model"):
            self.model_trainer.decoder_model.eval()
        if hasattr(self.model_trainer, "denoiser_model"):
            self.model_trainer.denoiser_model.eval()
        trainer.test(
            self.model_trainer,
            self.dataset_generator.test_dataloader(),
            ckpt_path="best",
        )

        # reset this back to train mode
        if hasattr(self.model_trainer, "decoder_model"):
            self.model_trainer.decoder_model.train()
        if hasattr(self.model_trainer, "denoiser_model"):
            self.model_trainer.denoiser_model.train()
