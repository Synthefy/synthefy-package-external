from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import yaml
from loguru import logger
from omegaconf import DictConfig

COMPILE = True


@dataclass
class CltspV1Config:
    d_model: int = 512
    d_keys: int = 16
    d_values: int = 16
    n_heads: int = 16
    latent_dim: int = 256
    num_encoder_layers: int = 6
    num_compression_layers: int = 2
    d_ff: int = 2048
    dropout: float = 0.05
    activation: str = "gelu"
    num_positive_samples: int = 8
    discrete_condition_embedding_dim: int = 128


@dataclass
class CLTSPV3Config:
    d_model: int = 128
    d_keys: int = 8
    d_values: int = 8
    n_heads: int = 8
    num_encoder_layers: int = 8
    num_compression_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.05
    activation: str = "gelu"
    num_positive_samples: int = 2
    discrete_condition_embedding_dim: int = 128
    supervised_contrastive_learning: Optional[Dict[str, Any]] = None

    def __init__(self, config: Optional[Union[str, DictConfig]] = None):
        """
        config can be either path to a yaml file or a dictionary
        """
        if config is not None:
            if isinstance(config, str):
                with open(config, "r") as file:
                    config = yaml.safe_load(file)

            assert isinstance(config, (dict, DictConfig)), (
                "config must be a dictionary"
            )
            for key, value in config.items():
                try:
                    setattr(self, str(key), value)
                except AttributeError:
                    logger.warning(f"Attribute {key} not found in CLTSPConfig")

        self.validate_config()

    def validate_config(self) -> None:
        pass
