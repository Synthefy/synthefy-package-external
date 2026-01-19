"""Define configuration system for TabICL training."""

import argparse
import dataclasses
import os
from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import yaml
from loguru import logger

from synthefy_pkg.prior.activations import (
    get_activations,
    get_activations_by_name,
)


@dataclass
class TabICLBaseConfig:
    """Abstract Configuration class for TabICL components ."""

    def __init__(self, **kwargs):
        """Initialize TabICLConfig from a YAML file or keyword arguments.

        Args:
            config_path: Optional path to a YAML configuration file
            **kwargs: Optional keyword arguments to override default values
        """
        # First, initialize all attributes with their default values
        for field_name, field_def in self.__dataclass_fields__.items():
            if field_def.default_factory is not dataclasses.MISSING:
                setattr(self, field_name, field_def.default_factory())
            elif field_def.default is not dataclasses.MISSING:
                setattr(self, field_name, field_def.default)

        if "config_path" in kwargs:
            config_path = kwargs["config_path"]
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path}"
                )
            if not config_path.endswith((".yaml", ".yml")):
                raise ValueError(
                    f"Configuration file must be a YAML file: {config_path}"
                )

            # Load values from YAML and override defaults
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                for key, value in yaml_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Validate parameters after initialization
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate configuration parameters.

        This is a placeholder for custom validation rules.
        Override this method to add specific validation logic.
        """
        for params in []:
            if not isinstance(getattr(self, params), (int, float)):
                raise ValueError(f"{params} must be a number")
            if isinstance(getattr(self, params), float) and (
                getattr(self, params) <= 0 or getattr(self, params) >= 1
            ):
                raise ValueError(f"{params} must be a number between 0 and 1")

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class TabICLTrainingConfig(TabICLBaseConfig):
    """Configuration class for TabICL training ."""

    # Debug params
    disable_print: bool = True

    # Wandb Config
    wandb_log: bool = False
    wandb_project: str = "TabICL"
    wandb_name: Optional[str] = None
    wandb_id: Optional[str] = None
    wandb_dir: Optional[str] = None
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"

    # Training Config
    device: str = "cuda:4"  # Example: Using GPU 4 for model training. Do not use this for genload.py, use prior_device instead
    dtype: Literal["float16", "float32"] = "float32"
    np_seed: Optional[int] = 42
    torch_seed: Optional[int] = 42
    max_steps: int = 60000
    batch_size: int = 512
    micro_batch_size: int = 8

    # Optimization Config
    lr: float = 0.0001  # 1e-4
    scheduler: Literal[
        "cosine_warmup", "cosine_with_restarts", "polynomial_decay"
    ] = "cosine_warmup"
    warmup_proportion: float = 0.2
    warmup_steps: int = 2000
    gradient_clipping: float = 1.0
    weight_decay: float = 0
    cosine_num_cycles: int = 1
    cosine_amplitude_decay: float = 1.0
    cosine_lr_end: float = 0
    poly_decay_lr_end: float = 0.0000001  # 1e-7
    poly_decay_power: float = 1.0

    # Checkpointing
    checkpoint_dir: Optional[str] = (
        None  # /workspace/data/tabicl_exps/tabicl_tabular_cls/  # Example checkpoint directory
    )
    save_temp_every: int = 50
    save_perm_every: int = 5000
    max_checkpoints: int = 5
    checkpoint_path: Optional[str] = None
    only_load_model: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call parent's __init__
        # Additional initialization if needed


@dataclass
class TabICLPriorConfig(TabICLBaseConfig):
    """Configuration class for TabICL training using dataclass for better type safety and organization.

    This configuration class manages all parameters for synthetic data generation and prior model training
    in the TabICL (Tabular In-Context Learning) framework. Parameters are organized into several categories
    with their relationships and code correspondences documented in prior/prior_README.md.
    """

    config_path: Optional[str] = None

    # Intended only for SavePriorDataset access, NEVER set these in the yamls
    # Will results in unpredictable nested parallelism during DDP
    n_jobs: int = 1
    num_threads_per_generate: int = 1

    # Prior Dataset Config
    run_id: str = "default"
    prior_dir: Optional[str] = None  # (
    # "/workspace/data/tabicl_prior_data/tabicl_tabular_cls_lag_test/" # null for generating on the fly
    # )
    batch_size: int = 512  # this should get overridden by the training config
    np_seed: int = (
        42  # DEPRECATED: this should get overridden by the training config
    )
    torch_seed: int = (
        42  # DEPRECATED: this should get overridden by the training config
    )
    seed: int = 42  # this should get overridden by the training config
    device: str = "cuda:4"  # this should get overridden by the training config
    dataset_length: int = 512  # Total number of tables. Preferably a multiple of the number of workers.
    load_prior_start: int = 0
    delete_after_load: bool = False
    batch_size_per_gp: int = 4
    min_features: int = 5
    max_features: int = 100
    n_features: int = 5
    max_classes: int = 10
    min_seq_len: Optional[int] = None
    max_seq_len: int = 1024
    log_seq_len: bool = False
    seq_len_per_gp: bool = False
    min_train_size: Union[int, float] = 0.1
    max_train_size: Union[int, float] = 0.9
    replay_small: bool = False
    prior_type: Literal["mlp_scm", "tree_scm", "mix_scm", "dummy"] = "mix_scm"
    disable_print: bool = True
    prior_device: str = "cpu"  # keeping this so that the prior device can be CPU while the other is cuda, but use cuda_visible_devices to handle device management
    is_regression: bool = False
    add_synthetic_timestamps: List[str] = field(
        default_factory=lambda: []  # "minutely", "hourly", "daily", "monthly"
    )
    add_time_stamps_as_features: bool = False
    add_both_time_stamps_and_features: bool = False
    check_for_updates_freq: int = -1  # should be set by curriculum if used
    check_for_curriculum_config: bool = True  # should be set only when creating validation, default true because of tests
    tabular_dataset_rate: float = (
        0.0  # how often to create a tabular dataset (default never)
    )
    dataset_has_timestamp_rate: float = (
        1.0  # how often to add timestamps to a series dataset (default always)
    )

    # observation function parameters (fixed SCM prior)
    row_missing_prob: float = 0.0
    column_has_missing_prob: float = 0.0
    dataset_has_lag: float = 0.00
    min_lag: int = 0
    max_lag: float = 0.2
    exclude_inputs: Literal["allow", "exclude", "ensure"] = "allow"
    univariate_prob: float = 0.0
    num_flat_range: tuple[int, int] = (7, 20)
    flatten_prob: float = 0.0
    respect_ancestry_for_lag: bool = False
    use_input_as_target: bool = False

    # SCM Prior Config - Fixed
    scm_use_layer_operator: bool = False
    scm_mix_probs: tuple[float, float] = (
        0.7,
        0.3,
    )  # Mix probabilities for MLP vs Tree SCM
    scm_tree_model: str = "xgboost"  # Tree model type for TreeSCM
    scm_tree_depth_lambda: float = 0.5  # Lambda for tree depth sampling
    scm_tree_n_estimators_lambda: float = (
        0.5  # Lambda for number of estimators sampling
    )
    scm_balanced: bool = False  # Whether to balance classes
    scm_multiclass_ordered_prob: float = (
        0.0  # Probability of ordered multiclass
    )
    scm_cat_prob: float = 0.2  # Probability of categorical features
    scm_max_categories: float = float("inf")  # Maximum number of categories
    scm_scale_by_max_features: bool = False  # Whether to scale by max features
    scm_permute_features: bool = True  # Whether to permute features
    scm_permute_labels: bool = True  # Whether to permute labels
    scm_override_activations: list[str] = field(
        default_factory=lambda: []
    )  # Whether to override the activations
    scm_sigmoid_mixed_sampling_rate: int = 7  # negative samples more for the first, positive samples more from later ones
    scm_mixed_names: List[str] = field(
        default_factory=lambda: [
            "fourier",
            "fourier_frequency_domain",
            "fourier_both",
            "arima",
            "impulse_periodic",
            "wiener",
            "ornstein_uhlenbeck",
            "linear_trend",
            "box_periodic",
            "wavelet_periodic",
            "wedge_periodic",
            "step",
            "box",
            "wavelet",
            "wedge",
            "piecewise_splines",
            "impulse",
            "normal",
            "uniform",
        ]
    )
    scm_diverse_activation_names: List[str] = field(
        default_factory=lambda: [
            "tanh",
            "leaky_relu",
            "elu",
            "identity",
            "selu",
            "silu",
            "relu",
            "softplus",
            "relu6",
            "hardtanh",
            "sign",
            "rbf",
            "exp",
            "sqrt_abs",
            "unit_interval_indicator",
            "sine",
            "square",
            "abs",
            "random_function",
        ]
    )

    # layer operator parameters
    # normalization is fixed since for preventing blowup
    scm_normalization_type: str = (
        "none"  # "none", "z_score", "min_max", "robust", "batch"
    )
    scm_normalization_minimum_magnitude: float = 0.0
    scm_normalization_maximum_magnitude: float = -1.0
    scm_normalization_apply_probability: float = 0.0

    # scm layer kernel parameters
    scm_kernel_direction: str = "history"  # "history", "future", "mixed"
    scm_kernel_size_min: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 1,
            "min": 1,
            "round": True,
        }
    )
    scm_kernel_size_max: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 1,
            "min": 1,
            "round": True,
        }
    )
    scm_kernel_sigma: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 0.5,
            "min": 0.0,
        }
    )
    scm_kernel_type: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [
                "uniform"
            ],  # "gaussian", "laplacian", "sobel", "recent_exponent", "mixed"],
        }
    )

    # functional lag parameters
    scm_functional_lag_min: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 0.05,
            "min": 0.01,
        }
    )
    scm_functional_lag_max: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 0.1,
            "min": 0.01,
        }
    )
    scm_functional_lag_rate: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 0.5,
            "min": 0.0,
        }
    )
    scm_use_functional_lag: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_functional_lag_variance: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 0.5,
            "min": 0.0,
        }
    )
    scm_layer_has_functional_lag_rate: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 1.0,
            "min": 0.5,
        }
    )
    scm_node_has_functional_lag_rate: Dict[str, Any] = field(  # TODO: not used
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 1.0,
            "min": 0.5,
        }
    )

    # SCM Prior Config - Sampled Hyperparameters
    scm_multiclass_type: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": ["value", "rank"],
        }
    )
    scm_mlp_activations: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice_mixed",
            "choice_values": get_activations(
                random=True, scale=True, diverse=True
            ),
        }
    )
    scm_block_wise_dropout: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_mlp_dropout_prob: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_beta",
            "scale": 0.9,
            "min": 0.1,
            "max": 5.0,
        }
    )
    scm_is_causal: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_num_causes: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 12,
            "min_mean": 1,
            "round": True,
            "lower_bound": 1,
        }
    )
    scm_y_is_effect: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_in_clique: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_sort_features: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_num_layers: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 6,
            "min_mean": 1,
            "round": True,
            "lower_bound": 2,
        }
    )
    scm_hidden_dim: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 130,
            "min_mean": 5,
            "round": True,
            "lower_bound": 4,
        }
    )
    scm_init_std: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 10.0,
            "min_mean": 0.01,
            "round": False,
            "lower_bound": 0.0,
        }
    )
    scm_noise_std: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 0.3,
            "min_mean": 0.0001,
            "round": False,
            "lower_bound": 0.0,
        }
    )
    scm_noise_type: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": ["gaussian"],  # "ts", "mixed"
        }
    )
    scm_mixed_noise_ratio: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "max": 0.5,
            "min": 0.0,
        }
    )
    scm_used_sampler: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": ["ts"],  # "ts", "tabular", "real", "mixed"
        }
    )
    scm_sampling: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": ["mixed_series"],
        }
    )
    scm_ts_noise_sampling: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": ["normal"],
        }
    )
    scm_pre_sample_cause_stats: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_pre_sample_noise_std: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        }
    )
    scm_counterfactual_type: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [
                "random_normal",
                "random_uniform",
                "zero",
                "additive_norm",
            ],
        }
    )
    scm_counterfactual_num_changes: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 10,
            "min_mean": 1,
            "round": True,
            "lower_bound": 1,
        }
    )
    scm_counterfactual_num_samples: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 0.1,
            "min_mean": 0.1,
            "round": True,
            "lower_bound": 0,
        }
    )

    scm_counterfactual_enabled: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "meta_choice",
            "choice_values": [False],
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call parent's __init__
        # Additional initialization if needed

    def __post_init__(self, **kwargs):
        if self.config_path is not None:
            # Properly load and set values instead of trying to reassign self
            with open(self.config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                print(yaml_config)
                for key, value in yaml_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

    def get_scm_fixed_hp(self) -> Dict[str, Any]:
        """Get fixed hyperparameters for SCM priors."""
        return {
            "mix_probs": self.scm_mix_probs,
            "tree_model": self.scm_tree_model,
            "tree_depth_lambda": self.scm_tree_depth_lambda,
            "tree_n_estimators_lambda": self.scm_tree_n_estimators_lambda,
            "balanced": self.scm_balanced,
            "multiclass_ordered_prob": self.scm_multiclass_ordered_prob,
            "cat_prob": self.scm_cat_prob,
            "max_categories": self.scm_max_categories,
            "scale_by_max_features": self.scm_scale_by_max_features,
            "permute_features": self.scm_permute_features,
            "permute_labels": self.scm_permute_labels,
            "sigmoid_mixed_sampling_rate": self.scm_sigmoid_mixed_sampling_rate,
            "sampling_mixed_names": self.scm_mixed_names,
            "mixed_names": self.scm_mixed_names,
            "univariate_prob": self.univariate_prob,
            "num_flat_range": self.num_flat_range,
            "flatten_prob": self.flatten_prob,
            "use_layer_operator": self.scm_use_layer_operator,
            "normalization_type": self.scm_normalization_type,
            "normalization_minimum_magnitude": self.scm_normalization_minimum_magnitude,
            "normalization_maximum_magnitude": self.scm_normalization_maximum_magnitude,
            "normalization_apply_probability": self.scm_normalization_apply_probability,
        }

    def get_scm_sampled_hp(self) -> Dict[str, Any]:
        """Get sampled hyperparameters for SCM priors."""
        if len(self.scm_override_activations) > 0:
            logger.info(
                f"Overriding activations to {self.scm_override_activations} from {self.scm_mlp_activations}"
            )
        if len(self.scm_override_activations) > 0:
            diverse_names_without_diverse = (
                [a for a in self.scm_override_activations if a != "diverse"]
                if len(self.scm_diverse_activation_names) == 0
                else self.scm_diverse_activation_names
            )
            self.scm_mlp_activations = {
                "distribution": "meta_choice_mixed",
                "choice_values": [
                    get_activations_by_name(
                        a,
                        random=True,
                        scale=True,
                        diverse=True,
                        diverse_names=diverse_names_without_diverse,
                    )
                    for a in self.scm_override_activations
                ],
            }

        return {
            "multiclass_type": self.scm_multiclass_type,
            "mlp_activations": self.scm_mlp_activations,
            "block_wise_dropout": self.scm_block_wise_dropout,
            "mlp_dropout_prob": self.scm_mlp_dropout_prob,
            "is_causal": self.scm_is_causal,
            "num_causes": self.scm_num_causes,
            "y_is_effect": self.scm_y_is_effect,
            "in_clique": self.scm_in_clique,
            "sort_features": self.scm_sort_features,
            "num_layers": self.scm_num_layers,
            "hidden_dim": self.scm_hidden_dim,
            "init_std": self.scm_init_std,
            "noise_std": self.scm_noise_std,
            "noise_type": self.scm_noise_type,
            "mixed_noise_ratio": self.scm_mixed_noise_ratio,
            "used_sampler": self.scm_used_sampler,
            "sampling": self.scm_sampling,
            "ts_noise_sampling": self.scm_ts_noise_sampling,
            "pre_sample_cause_stats": self.scm_pre_sample_cause_stats,
            "pre_sample_noise_std": self.scm_pre_sample_noise_std,
            "counterfactual_num_changes": self.scm_counterfactual_num_changes,
            "counterfactual_num_samples": self.scm_counterfactual_num_samples,
            "counterfactual_type": self.scm_counterfactual_type,
            "counterfactual_enabled": self.scm_counterfactual_enabled,
            "kernel_size_min": self.scm_kernel_size_min,
            "kernel_size_max": self.scm_kernel_size_max,
            "kernel_type": self.scm_kernel_type,
            "functional_lag_min": self.scm_functional_lag_min,
            "functional_lag_max": self.scm_functional_lag_max,
            "use_functional_lag": self.scm_use_functional_lag,
            "functional_lag_variance": self.scm_functional_lag_variance,
            "functional_lag_rate": self.scm_functional_lag_rate,
            "layer_has_functional_lag_rate": self.scm_layer_has_functional_lag_rate,
            "node_has_funcitonal_lag_rate": self.scm_node_has_functional_lag_rate,
        }


@dataclass
class TokenDecoderConfig(TabICLBaseConfig):
    """Configuration class for TabICL model."""

    decoder_type: str = "linear"  # "linear", "mlp"
    num_layers: int = 1
    hidden_dim: int = 128
    dropout: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"
    token_forecast_length: int = 1
    model_dim: int = 128  # should be set after initialization
    output_dim: int = 10  # should be set after initialization
    input_expert_module_paths: List[str] = field(default_factory=lambda: [])
    embed_expert_module_paths: List[str] = field(default_factory=lambda: [])
    input_expert_module_names: List[str] = field(default_factory=lambda: [])
    embed_expert_module_names: List[str] = field(default_factory=lambda: [])
    # RowColumnDecoder specific parameters
    norm_first: bool = True
    weight_range: tuple[float, float] = (0, 0)
    row_num_blocks: int = 3
    row_nhead: int = 8
    row_num_cls: int = 4
    col_num_blocks: int = 3
    col_nhead: int = 4
    col_num_cls: int = 4
    rope_base: float = 100000
    ff_factor: float = 2.0
    # mixture of experts specific parameters
    world_size: int = 1
    # TODO: Does not handle routing to different GPUs yet
    # TODO: thus, n_routed_experts = n_activated_experts, world_size = 1
    n_routed_experts: int = 8
    n_activated_experts: int = 8
    rank: int = 1
    moe_inter_dim: int = 1
    bias_update_rate: float = 0.01
    topk: int = 4
    score_func: str = "softmax"
    route_scale: float = 1.0

    def __post_init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class TabICLModelConfig(TabICLBaseConfig):
    """Configuration class for TabICL model."""

    # Model Architecture Config
    amp: bool = True
    model_compile: bool = False

    # Column Embedding Config
    embed_dim: int = 128
    col_num_blocks: int = 3
    col_nhead: int = 4
    col_num_inds: int = 128
    freeze_col: bool = False
    embedder_name: str = "linear"
    use_time_mask: bool = False
    time_mask_type: str = (
        "all"  # history, future, all, self, mixed (history, all, self)
    )
    time_mask_mixing_probs: List[float] = field(
        default_factory=lambda: [0.6, 0.3, 0.1]
    )

    # Row Interaction Config
    row_num_blocks: int = 3
    row_nhead: int = 8
    row_num_cls: int = 4
    row_rope_base: float = 100000
    freeze_row: bool = False

    # ICL Config
    icl_num_blocks: int = 12
    icl_nhead: int = 4
    freeze_icl: bool = False
    random_train_mask_ratio: float = 0.0
    last_row_masking: bool = False

    # Shared Architecture Config
    ff_factor: int = 2
    dropout: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"
    norm_first: bool = True
    is_regression: bool = False
    max_classes: int = 10
    weight_range: tuple[float, float] = (0, 0)
    preserve_col_order: bool = False

    # Multilayer TabICL Config
    embed_col_num_blocks: int = 1
    embed_row_num_blocks: int = 1
    num_layers: int = 2
    skip_col_embedding: bool = False
    use_full_reg: bool = False
    full_reg_decoder_config: Optional[TokenDecoderConfig] = None

    # Embedder config
    external_column_embedder: str = "sfm_v3e"
    external_column_embedder_config: str = (
        "examples/configs/foundation_model_configs/config_syn_uni_sfm.yaml"
    )
    # external_column_embedder_checkpoint: str = "/mnt/workspace3/synthefy_data/training_logs/covid_deaths/MLFlow_Plotting_Test/chronos_gift_pretrain_bs8_lr1e-5_1024_window_counts_block_mask_8_48_1/checkpoints/epoch_epoch=250.ckpt"
    external_column_embedder_checkpoint: str = ""
    train_as_univariate_forecast: bool = False

    external_forecasts_to_use: List[str] = field(
        default_factory=lambda: ["toto_univariate"]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call parent's __init__
        # Additional initialization if needed


@dataclass
class TabICLCurriculumConfig(TabICLBaseConfig):
    """Configuration class for TabICL model."""

    curriculum_type: str = "linear"
    curriculum_features: Dict[str, Any] = field(
        default_factory=lambda: {}
    )  # handles changes to the prior config
    assignment_features: Dict[str, Any] = field(
        default_factory=lambda: {}
    )  # handles changes to any other attribute (TODO: only supports masking for now)
    max_epochs: int = -1  # set > -1 to use
    max_steps: int = -1  # set > -1 to use
    min_loss: float = 0.0
    max_loss: float = 1.0
    floor_ratio: bool = False
    update_frequency: int = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call parent's __init__
        # Additional initialization if needed


@dataclass
class AssignmentConfig(TabICLBaseConfig):
    """Configuration class for assigning values to the model.
    These names overlap with those they are meant to assign,
    each assignment is handled separately"""

    mask_mixing_rates: List[float] = field(default_factory=lambda: [1.0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call parent's __init__
        # Additional initialization if needed
