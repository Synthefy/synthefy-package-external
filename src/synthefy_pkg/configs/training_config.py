from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.base_config import BaseConfig
from synthefy_pkg.configs.lr_configs import get_lr_config


@dataclass
class TrainingConfig(BaseConfig):
    # General
    max_epochs: int = 100

    # Loss and learning rate parameters
    learning_rate: float = 1e-4
    auto_lr_find: bool = True
    patience: Optional[int] = None
    gradient_clipping: float = 0.0
    use_early_stopping: bool = False
    strategy: str = "auto"
    device: str = "cuda"
    num_devices: int = 1
    temperature: Optional[float] = None
    use_clip_loss: bool = False
    use_condition_scl_loss: bool = False
    use_timeseries_scl_loss: bool = False
    train_for_classification: bool = False
    save_checkpoint_every_n_epochs: int = -1
    precision: Union[Literal["16-mixed"], Literal[32]] = 32

    # Foundation model parameters:
    num_ar_batches: int = 0
    description_mask_ratio: float = 0.0
    target_mask_ratio: float = 0.5
    pred_plot_freq: int = -1
    # Plotting parameters
    n_plots: int = 4
    max_samples_per_plot: int = 5

    # Logging parameters
    check_val_every_n_epoch: int = 1
    check_test_every_n_epoch: int = 1
    val_check_interval: float = 1.0
    log_every_n_steps: int = 1
    push_logs_every_n_steps: int = 500  # push logs to MLFlow every n steps
    # lib default is DEBUG
    logger_level: Literal[
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "EXCEPTION",
        "CRITICAL",
    ] = "DEBUG"

    # Checkpointing parameters
    save_all_checkpoints: bool = False
    save_checkpoint_every_n_steps: int = -1
    # Periodic checkpoint saving (saves every N epochs regardless of performance)
    save_periodic_checkpoint_every_n_epochs: int = -1

    # Nested configs
    synthetic_dataset_length: int = 0  # required for lr scheduler
    lr_scheduler_config = None

    strict_load: bool = True  # For checkpoint loading, see synthesis_utils.py

    def __post_init__(self):
        super().__post_init__()

        if (
            isinstance(self.config, DictConfig)
            and "lr_scheduler" in self.config
        ):
            # Check if max_steps field exists in scheduler config
            if "max_steps" in self.config["lr_scheduler"]:
                # Try existing methods first
                if (
                    self.synthetic_dataset_length > 0
                    and self.config["lr_scheduler"]["max_steps"] < 0
                ):
                    # OTF synthetic data
                    self.config["lr_scheduler"]["max_steps"] = (
                        self.synthetic_dataset_length * 0.5 * self.max_epochs
                    )
                    logger.info(
                        f"Setting scheduler max_steps to {self.config['lr_scheduler']['max_steps']} (synthetic dataset)"
                    )
                    self.lr_scheduler_config = get_lr_config(
                        self.config["lr_scheduler"]
                    )
                elif self.config["lr_scheduler"]["max_steps"] > 0:
                    # Already set to a positive value
                    logger.info(
                        f"Using pre-configured scheduler max_steps: {self.config['lr_scheduler']['max_steps']}"
                    )
                    self.lr_scheduler_config = get_lr_config(
                        self.config["lr_scheduler"]
                    )
                elif self.config["lr_scheduler"]["max_steps"] < 0:
                    raise ValueError(
                        "max_steps must be set to a positive value"
                    )
                    # Defer lr_scheduler setup - will be handled by ModelTrain after trainer is available
                    # logger.info("Deferring lr_scheduler setup until trainer stepping batches are known")
                    # self.lr_scheduler_config = None
                    # # Store config for later use
                    # self._deferred_lr_config = self.config["lr_scheduler"].copy()
                else:
                    # max_steps is 0, setup normally
                    self.lr_scheduler_config = get_lr_config(
                        self.config["lr_scheduler"]
                    )
            else:
                # No max_steps field, setup normally
                self.lr_scheduler_config = get_lr_config(
                    self.config["lr_scheduler"]
                )

        self._validate_config()

    def setup_deferred_lr_scheduler(self, max_steps: int):
        """TODO: Setup the lr_scheduler with calculated max_steps."""
        # if hasattr(self, '_deferred_lr_config'):
        #     self._deferred_lr_config['max_steps'] = max_steps
        #     self.lr_scheduler_config = get_lr_config(self._deferred_lr_config)
        #     logger.info(f"Setup deferred lr_scheduler with max_steps: {max_steps}")
        # else:
        #     logger.warning("No deferred lr_config found to setup")

    def _validate_config(self) -> None:
        if self.device == "cuda":
            if not torch.cuda.is_available():
                self.device = "cpu"
                logger.error("cuda is not available - using cpu")

        # If patience is set in config, set max_epochs to a large number
        if self.patience is not None:
            self.use_early_stopping = True
            self.max_epochs = max(self.max_epochs, 1000)
            logger.warning(
                f"Patience is set in config, setting max_epochs to {self.max_epochs}"
            )

        if self.num_devices > 1 and self.save_checkpoint_every_n_steps > 0:
            raise ValueError(
                "GlobalStepCheckpointCallback with multiple devices is not supported: Cannot run custom validation loop in a distributed setup."
            )

        if self.logger_level not in [
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "EXCEPTION",
            "CRITICAL",
        ]:
            logger.critical(
                f"Invalid logger level: {self.logger_level}, setting to DEBUG"
            )
            self.logger_level = "DEBUG"

        # Assert that precision is either 16-mixed or 32
        assert self.precision in [
            "16-mixed",
            32,
        ], f"precision must be one of ['16-mixed', 32], got {self.precision}"
