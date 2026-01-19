import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.base_config import BaseConfig

COMPILE = False
SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))

assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


@dataclass
class DatasetConfig(BaseConfig):
    forecast_length: int = -1
    batch_size: int = 512
    dataset_name: str = ""
    device: str = "cuda"
    experiment: str = "diffusion"
    input_condition_max_norm: float = 1.0
    log_dir: str = str(DATASETS_BASE)
    num_channels: int = 1
    num_conditions: int = 4
    num_input_features: int = 1
    num_output_features: int = 1
    num_workers: int = 8
    per_condition_embedding_dim: int = 1
    required_time_series_length: int = 64
    seed: int = 42
    time_series_length: int = 96
    train_test_split: float = 0.8
    use_amplitude: bool = True
    use_frequency: bool = True
    use_phase: bool = True
    use_type: bool = True
    num_timestamp_labels: int = 0
    use_timestamp: Union[bool, None] = None
    use_metadata: bool = True
    write_on_disk_cache: bool = False
    dataloader_name: str = "ForecastingDataLoader"
    text_embedding_dim: int = 384
    num_timestamp_features: int = 11
    dataset_description_start_idx: int = 0
    time_columns: int = 0
    mixed_real_synthetic_sampling: bool = False
    train_as_univariate_model: bool = False
    is_only_target_prediction: bool = False
    is_tabular: bool = False

    # Conditional parameters
    # discrete_conditions: ClassVar[list[str]] = ["type"]
    # continuous_conditions: ClassVar[list[str]] = ["amplitude", "phase", "frequency"]
    discrete_condition_embedding_dim: int = 128
    num_discrete_categories_for_encoder: int = 1
    num_discrete_conditions = 4
    num_discrete_labels = 10  # only set this if somi says so only for backward compatibility for timeseries synthesis repo.
    num_continuous_labels = 75

    # Constraints parameters: used only in synthesis API
    use_constraints: bool = False
    projection_during_synthesis: str = "clipping"
    selectively_denoise: bool = False
    gamma_choice: Optional[str] = None
    extract_equality_constraints_from_windows: bool = False
    constraints: List[str] = field(default_factory=lambda: [])
    predetermined_constraint_values: Dict[str, float] = field(
        default_factory=lambda: {}
    )
    user_provided_constraints: Optional[
        Dict[str, Dict[str, Union[int, float]]]
    ] = None

    # Foundation model only
    use_sharded_dataset: bool = False
    num_correlates: int = 50
    num_datasets: int = 10000
    use_relation_shards: bool = False  # For V1 Sharded Dataset Only
    relational_sampling_strategy: Optional[str] = None  # For V3 Format
    relational_sampling_data_location: Optional[str] = None  # For V3 Format
    v3_data_paths: List[str] = field(default_factory=lambda: [])
    use_window_counts: bool = True
    using_synthetic_data: bool = False
    prior_config_path: str = ""
    is_regression: bool = True
    curriculum_config_path: str = ""

    # Foundation model indices, these should get set by yaml
    dataset_description_end_idx: int = 0
    timestamp_start_idx: int = 0
    timestamp_end_idx: int = 0
    continuous_start_idx: int = 0
    continuous_end_idx: int = 0
    metadata_length: int = 0

    run_val_with_eval_bench: bool = False
    eval_bench_dataset_name: str = (
        ""  # Not utilized, we default to SyntheticMediumLagDataloader
    )
    eval_bench_data_path: str = ""
    num_eval_bench_batches: int = 64

    def __post_init__(self):
        self.load_config()
        assert isinstance(self.config, DictConfig), (
            "config must be a DictConfig"
        )
        self.set_constraint_parameters(self.config)
        self.load_values_from_config()

        self._validate_config()

    def set_constraint_parameters(self, config: DictConfig) -> None:
        self.use_constraints = config.get("use_constraints", False)
        self.user_provided_constraints = None
        if (
            config.get("extract_equality_constraints_from_windows", None)
            is None
        ):
            logger.warning(
                "Using default value False for extract_equality_constraints_from_windows, since wasn't provided"
            )
        self.extract_equality_constraints_from_windows = config.get(
            "extract_equality_constraints_from_windows", False
        )
        self.constraints = config.get("constraints", [])
        if (
            config.get("predetermined_constraint_values", None) is None
            and not self.extract_equality_constraints_from_windows
        ):
            logger.warning(
                "Will use predetermined_constraint_values from saved constraints from preprocessing"
            )
        self.predetermined_constraint_values = config.get(
            "predetermined_constraint_values", {}
        )
        if config.get("projection_during_synthesis", None) is None:
            logger.warning(
                "Using default value 'clipping' for projection_during_synthesis, since wasn't provided"
            )
        self.projection_during_synthesis = config.get(
            "projection_during_synthesis", "clipping"
        )
        self.selectively_denoise = config.get("selectively_denoise", False)
        self.gamma_choice = config.get("gamma_choice", None)

    def _validate_use_timestamp(self) -> None:
        if self.use_timestamp is None:
            # Infer use_timestamp from num_timestamp_labels
            # Essentially default to True if we have timestamp labels
            self.use_timestamp = self.num_timestamp_labels > 0
            logger.warning(
                f"use_timestamp inferred from num_timestamp_labels: {self.use_timestamp}"
            )

        else:
            if self.use_timestamp and self.num_timestamp_labels == 0:
                raise ValueError(
                    "num_timestamp_labels must be greater than 0 if use_timestamp is True"
                )
            elif not self.use_timestamp and self.num_timestamp_labels > 0:
                logger.warning(
                    "num_timestamp_labels is greater than 0 but use_timestamp is False. Make sure this is intentional."
                )

    def _validate_config(self) -> None:
        self._validate_use_timestamp()

        if self.dataset_name == "":
            raise ValueError("dataset_name must be provided")
        if self.device == "cuda":
            if not torch.cuda.is_available():
                self.device = "cpu"
                logger.error("cuda is not available - using cpu")
