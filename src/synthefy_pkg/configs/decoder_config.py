"""Define configuration system for TabICL training."""

import argparse
import dataclasses
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal

import yaml


@dataclass
class DecoderConfig:
    """Abstract Configuration class for TabICL components ."""

    decoder_type: Literal["moe", "row_column", "linear", "mlp"] = "linear"
    input_expert_module_names: List[str] = field(default_factory=lambda: [])
    embed_expert_module_names: List[str] = field(default_factory=lambda: [])
    world_size: int = 1
    n_routed_experts: int = 8
    n_activated_experts: int = 8
    rank: int = 1
    moe_inter_dim: int = 256
    bias_update_rate: float = 0.01
    topk: int = 4
    score_func: str = "softmax"
    row_num_blocks: int = 3
    row_nhead: int = 8
    row_num_cls: int = 4
    row_rope_base: float = 10000
    icl_num_blocks: int = 12
    icl_nhead: int = 4

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
