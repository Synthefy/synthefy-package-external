import os
import shutil
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import yaml
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.utils.sagemaker_utils import is_running_in_sagemaker

CONFIGS = [
    "dataset_config",
    "metadata_encoder_config",
    "denoiser_config",
    "training_config",
    "execution_config",
]

COMPILE = False


# create abstract class for experiments
class Experiment(ABC):
    configuration: Configuration
    training_runner: Any

    def __init__(self, config_source: Union[str, Dict[str, Any], Configuration]):
        if isinstance(config_source, str):
            with open(config_source, "r") as file:
                config = yaml.safe_load(file)
        else:
            config = config_source

        if isinstance(config, dict):

            config["dataset_config"] = config.get("dataset_config", {})
            config["metadata_encoder_config"] = config.get(
                "metadata_encoder_config", {}
            )
            config["decoder_config"] = config.get("denoiser_config", {})
            config["training_config"] = config.get("training_config", {})
            config["execution_config"] = config.get("execution_config", {})

            # Ensure check_val_every_n_epoch is not greater than max_epochs
            if (
                "training_config" in config
                and "check_val_every_n_epoch" in config["training_config"]
                and "max_epochs" in config["training_config"]
            ):
                if config["training_config"]["check_val_every_n_epoch"] is None:
                    if (
                        config["training_config"]["val_check_interval"] is None
                        or config["training_config"]["val_check_interval"] < 0
                    ):
                        raise ValueError(
                            "val_check_interval must be set if check_val_every_n_epoch is None.  If you want to disable validation set `limit_val_batches` to 0.0"
                        )
                elif (
                    config["training_config"]["check_val_every_n_epoch"]
                    > config["training_config"]["max_epochs"]
                ):
                    config["training_config"]["check_val_every_n_epoch"] = 1
                    logger.info(
                        "check_val_every_n_epoch was greater than max_epochs, setting to 1"
                    )

            if not isinstance(config, DictConfig):
                config = DictConfig(config)

            self.configuration = Configuration(
                config=config,
            )

        elif isinstance(config, Configuration):
            self.configuration = config

        # Check if logging has already been configured
        if self.configuration.training_config.logger_level and not os.getenv("SYNTHEFY_LOGGER_IS_CONFIGURED"):
            # Remove existing handlers and add new one with configured level
            logger.remove()
            logger.add(
                sys.stderr,
                level=self.configuration.training_config.logger_level,
            )
            logger.add(
                os.path.join(
                    self.configuration.get_log_dir(),
                    f"run_logs_{self.configuration.training_config.logger_level}.log",
                ),
                level=self.configuration.training_config.logger_level,
            )
            logger.add(
                os.path.join(
                    self.configuration.get_log_dir(), "run_logs_TRACE.log"
                ),
                level="TRACE",
            )
            logger.critical(
                f"Setting logger level to {self.configuration.training_config.logger_level}"
            )
            # Set flag to prevent reconfiguration (eg. when loading a checkpoint of a different model)
            os.environ["SYNTHEFY_LOGGER_IS_CONFIGURED"] = "true"

        # Useful to check GPU usage stats in MLFlow
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        self._setup()

    def _setup_sagemaker_checkpoint_callback(self):
        """Sets up checkpoint saving callback for SageMaker training"""
        if not is_running_in_sagemaker():
            logger.info(
                "Not running in SageMaker, skipping checkpoint callback"
            )
            return

        def sagemaker_checkpoint_callback(trainer):
            model_dir = os.environ.get("SM_MODEL_DIR")
            if not model_dir:
                return

            checkpoints_dir = os.path.join(
                self.training_runner.log_dir, "checkpoints"
            )
            if not os.path.exists(checkpoints_dir):
                return

            checkpoints = [
                f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")
            ]
            if not checkpoints:
                return

            latest_checkpoint = sorted(checkpoints)[-1]
            src = os.path.join(checkpoints_dir, latest_checkpoint)
            dst = os.path.join(model_dir, "model.ckpt")

            try:
                shutil.copy2(src, dst)
                logger.info(f"Successfully copied checkpoint to {dst}")
            except Exception as e:
                logger.error(f"Error copying checkpoint: {str(e)}")

        self.training_runner.checkpoint_callbacks.append(
            sagemaker_checkpoint_callback
        )

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def generate_synthetic_data(
        self,
        model_checkpoint_path: str,
        splits: List[str] = ["test"],
    ):
        pass
