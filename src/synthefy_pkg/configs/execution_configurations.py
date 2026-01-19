import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.dataset_configs import DatasetConfig
from synthefy_pkg.configs.decoder_config import DecoderConfig
from synthefy_pkg.configs.denoiser_configs import (
    DenoiserConfig,
    MetadataEncoderConfig,
)
from synthefy_pkg.configs.encoder_configs import CLTSPV3Config
from synthefy_pkg.configs.foundation_model_config import FoundationModelConfig
from synthefy_pkg.configs.patchtst_config import PatchTSTConfig
from synthefy_pkg.configs.tabicl_config import (
    AssignmentConfig,
    TabICLCurriculumConfig,
    TabICLModelConfig,
    TabICLPriorConfig,
    TokenDecoderConfig,
)
from synthefy_pkg.configs.training_config import TrainingConfig

DEFAULT_SEED = 42
SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

# MLFLOW_FOLDER = os.getenv("SYNTHEFY_MLFLOW_FOLDER")
assert SYNTHEFY_DATASETS_BASE is not None
# assert MLFLOW_FOLDER is not None

COMPILE = True


@dataclass
class Configuration:
    # Paths
    log_dir: str = SYNTHEFY_DATASETS_BASE
    base_path: str = SYNTHEFY_DATASETS_BASE
    save_path: str = ""
    generation_save_path: str = ""
    # mlflow_folder: str = MLFLOW_FOLDER
    seed: int = DEFAULT_SEED
    inference_seed: int = DEFAULT_SEED
    device: str = "cuda"
    num_workers: int = (
        8  # DEPR: use dataset_config.num_workers instead # TODO remove
    )
    dataset_name: str = ""
    denoiser_name: str = "csdi_timeseries_denoiser_v1"
    model_file: str = "timeseries_diffusion"
    model_name: str = "TimeSeriesDiffusionModelTrainer"
    cltsp_name: str = "cltsp_v1"
    cltsp_model_file: str = "cltsp"
    cltsp_model_name: str = "CLTSPTrainer"
    save_key: str = "val_loss"
    should_compile_torch: bool = True
    autoencoder_checkpoint_path: str = "none"
    batch_size: int = (
        1  # DEPR: use dataset_config.batch_size instead # TODO remove
    )
    # Effective batch size = batch_size * accumulate_grad_batches
    # Gradients are normalized by accumulate_grad_batches internally
    accumulate_grad_batches: int = 1
    n_plots: int = 5
    train_test_split: float = 0.8
    run_name: str = ""
    experiment_name: str = ""
    remove_outliers_from_plot: bool = False
    encoder_config = None
    denoiser_config = None
    run_validation_before_training: bool = False
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "http://localhost:5000"

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        config_filepath: Optional[str] = None,
    ):
        if config is None and config_filepath is None:
            logger.info(
                "no config or config_filepath provided, using default config"
            )
            config = DictConfig(
                {
                    "foundation_model_config": None,
                    "training_config": None,
                    "dataset_config": None,
                    "execution_config": None,
                }
            )

            # raise ValueError("config or config_filepath must be provided")

        if config is None and config_filepath is not None:
            with open(config_filepath, "r") as file:
                config = yaml.safe_load(file)
            config = DictConfig(config)

        assert config is not None

        # assign all the keys from the config to this class, excluding subclasses
        for key in self.__dir__():
            if key in config and key.find("_config") == -1:
                setattr(self, key, config[key])

        self.checkpoint_path = config.get("checkpoint_path", "")
        self.task = config.get("task", "")
        self.dataset_config = DatasetConfig(config=config["dataset_config"])
        self.trainer = config.get("trainer", "default_trainer")

        # Dispatch the configuration to the appropriate model
        if "foundation_model_config" in config:
            self.foundation_model_config = FoundationModelConfig(
                config=config["foundation_model_config"]
            )
            if self.foundation_model_config.use_metadata:
                self.metadata_encoder_config = MetadataEncoderConfig(
                    config["metadata_encoder_config"]
                )

        elif "patchtst_config" in config:
            self.patchtst_config = PatchTSTConfig(config["patchtst_config"])
        elif "encoder_config" in config:
            self.encoder_config = CLTSPV3Config(config["encoder_config"])
        else:
            self.metadata_encoder_config = MetadataEncoderConfig(
                config["metadata_encoder_config"]
            )
            self.denoiser_config = DenoiserConfig(
                config=config.get("denoiser_config", {}),
                metadata_encoder_config=self.metadata_encoder_config,
            )

        if "decoder_config" in config:
            self.decoder_config = DecoderConfig(config=config["decoder_config"])
        else:
            self.decoder_config = None

        if (
            "tabicl_config" in config
            and "use_full_reg" in config["tabicl_config"]
            and config["tabicl_config"]["use_full_reg"]
        ):
            if (
                "token_decoder_config" not in config
                or config["token_decoder_config"] is None
            ):
                self.token_decoder_config = TokenDecoderConfig()
            else:
                self.token_decoder_config = TokenDecoderConfig(
                    **config["token_decoder_config"]
                )
        else:
            self.token_decoder_config = None

        # tabicl parameter override foundation model ones
        if "tabicl_config" in config:
            config["tabicl_config"]["device"] = config.get("device", "cuda:0")
            self.tabicl_config = TabICLModelConfig(**config["tabicl_config"])
            if not self.tabicl_config.use_full_reg:
                if (
                    self.foundation_model_config.masking_schemes[0]
                    != "train_test_last"
                ):
                    logger.warning(
                        f"Only valid masking scheme for TabICL is train_test_last, but got {self.foundation_model_config.masking_schemes}, resetting to train_test_last"
                    )
                    self.foundation_model_config.masking_schemes = [
                        "train_test_last"
                    ]
        if len(self.dataset_config.curriculum_config_path) > 0:
            with open(self.dataset_config.curriculum_config_path, "r") as file:
                curriculum_config = yaml.safe_load(file)
            # override prior config with any values in the config
            if "curriculum_config" in config:
                for key, value in config["curriculum_config"].items():
                    curriculum_config[key] = value
            self.curriculum_config = TabICLCurriculumConfig(**curriculum_config)
            if (
                "assignment_features" in curriculum_config
            ):  # used for helping with curriculum
                self.assignment_config = AssignmentConfig()

        if len(self.dataset_config.prior_config_path) > 0:
            with open(self.dataset_config.prior_config_path, "r") as file:
                prior_config = yaml.safe_load(file)
            # override prior config with any values in the config
            if "prior_config" in config:
                for key, value in config["prior_config"].items():
                    prior_config[key] = value
            # prior_config = DictConfig(prior_config)

            # Remove config_path from prior_config if it's None to avoid TypeError
            if (
                "config_path" in prior_config
                and prior_config["config_path"] is None
            ):
                del prior_config["config_path"]

            self.prior_config = TabICLPriorConfig(**prior_config)
            if self.dataset_config.curriculum_config_path:
                self.prior_config.check_for_updates_freq = (
                    self.curriculum_config.update_frequency
                )
            self.prior_config.batch_size = self.dataset_config.batch_size
            self.prior_config.seed = self.seed
            self.prior_config.device = "cpu"  # config.get("device", "cuda:0")
            self.prior_config.prior_device = (
                "cpu"  # config.get("device", "cuda:0")
            )
            # ensure parameters are matched together
            self.prior_config.is_regression = self.dataset_config.is_regression

            if "tabicl_config" in config:
                assert self.foundation_model_config is not None, (
                    "foundation_model_config is required for TabICL training"
                )
                self.tabicl_config.is_regression = (
                    self.prior_config.is_regression
                )
                if self.dataset_config.is_regression:
                    if self.foundation_model_config.generate_probabilistic_forecast_using_bins:
                        self.tabicl_config.max_classes = (
                            self.foundation_model_config.num_bins
                        )
                    else:
                        self.tabicl_config.max_classes = (
                            self.dataset_config.num_channels
                        )
                else:
                    self.tabicl_config.max_classes = (
                        self.prior_config.max_classes
                    )

            # Duplicate the dataset length to the training config for LR scheduler
            config["training_config"]["synthetic_dataset_length"] = (
                self.prior_config.dataset_length
            )

        self.training_config = TrainingConfig(config=config["training_config"])
        self.dataset_name = self.dataset_config.dataset_name

        execution_config = config["execution_config"]

        if hasattr(self, "prior_config"):
            if (
                self.prior_config.dataset_length
                / (
                    self.training_config.num_devices
                    * self.dataset_config.num_workers
                )
                <= 1
            ):
                raise ValueError(
                    f"dataset_length {self.prior_config.dataset_length} is too small for {self.training_config.num_devices} devices and {self.dataset_config.num_workers} workers"
                )
            if (
                self.prior_config.dataset_length
                % (
                    self.training_config.num_devices
                    * self.dataset_config.num_workers
                )
                != 0
            ):
                raise ValueError(
                    f"dataset_length {self.prior_config.dataset_length} is not divisible by {self.training_config.num_devices} devices and {self.dataset_config.num_workers} workers"
                )
            if self.training_config.num_devices > 1:
                if self.dataset_config.run_val_with_eval_bench:
                    raise ValueError(
                        "Running eval bench with multiple devices is not supported: EvalBench is not a torch dataset."
                    )

        if execution_config is not None:
            if isinstance(execution_config, str):
                with open(execution_config, "r") as file:
                    execution_config = yaml.safe_load(file)

            for key, value in execution_config.items():
                try:
                    setattr(self, key, value)
                except AttributeError:
                    logger.warning(
                        f"Attribute {key} not found in Configuration"
                    )

        # run name needs to be set after execution config is filled:
        if hasattr(self, "prior_config"):
            self.prior_config.run_id = self.run_name

        self._validate_config()

    def _validate_config(self) -> None:
        assert self.save_path != "", "save_path must be set"
        assert self.run_name != "", "run_name must be set"
        assert self.dataset_name != "", "dataset_name must be set"
        assert self.experiment_name != "", "experiment_name must be set"
        assert self.trainer in [
            "foundation_model",
            "forecasting_model",
            "synthesis_model",
            "metadata_encoder",
            "default_trainer",
        ], (
            f"trainer must be one of ['foundation_model', 'forecasting_model', 'synthesis_model', 'metadata_encoder', 'default_trainer'], got {self.trainer}"
        )

        if self.device == "cuda":
            if not torch.cuda.is_available():
                self.device = "cpu"
                logger.error("cuda is not available - using cpu")

    def get_log_dir(self) -> str:
        assert SYNTHEFY_DATASETS_BASE is not None, (
            "SYNTHEFY_DATASETS_BASE is not set"
        )
        return os.path.join(
            SYNTHEFY_DATASETS_BASE,
            self.save_path,
            self.dataset_name,
            self.experiment_name,
            self.run_name,
        )

    def get_save_dir(self, base_path: str) -> str:
        return os.path.join(
            base_path,
            self.generation_save_path,
            self.dataset_name,
            self.experiment_name,
            self.run_name,
        )

    def get_lightning_logs_path(self, base_path: str) -> str:
        assert self.dataset_config is not None, "dataset_config is required"
        return os.path.join(
            base_path,
            self.save_path,
            self.dataset_config.dataset_name,
            self.experiment_name,
            self.run_name,
            "checkpoints",
            "lightning_logs",
        )
