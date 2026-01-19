from dataclasses import dataclass
from typing import Dict, Optional, Union

import yaml
from loguru import logger
from omegaconf import DictConfig

COMPILE = True


@dataclass
class DiffusionConfig:
    num_diffusion_steps: int = 200
    beta_start: float = 0.0001
    beta_end: float = 0.1  # for SSSD, 0.1 for CSDI
    use_timeseries_full: bool = True
    use_classifier_guided_sampling: bool = False
    use_ddim_sampling: bool = True
    use_classifier_free_guidance: bool = False
    use_classifier_free_guidance_training: bool = False
    unconditional_probability: float = 0.5

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
                    logger.warning(
                        f"Attribute {key} not found in DiffusionConfig"
                    )
        self.validate_config()

    def validate_config(self) -> None:
        pass


@dataclass
class MetadataEncoderConfig:
    channels: int = 256
    n_heads: int = 8
    num_encoder_layers: int = 2
    metadata_pretrain_epochs: int = 0

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
                    logger.warning(
                        f"Attribute {key} not found in MetadataEncoderConfig"
                    )
        self.validate_config()

    def validate_config(self) -> None:
        pass


@dataclass
class DenoiserConfig:
    stride: int
    d_layers: int
    d_model: int
    factor: int
    dropout: float
    output_attention: bool
    d_model: int
    d_ff: int
    activation: str
    e_layers: int
    denoiser_name: str = ""
    positional_embedding_dim: int = 128
    channel_embedding_dim: int = 16
    channels: int = 256
    n_heads: int = 16
    n_layers: int = 16
    dropout_pos_enc: float = 0.2
    use_cltsp: bool = False
    pretrained_loc: str = "/"
    T: int = 200
    beta_0: float = 0.0001
    beta_T: float = 0.1
    patch_len: int = 16

    # Only used for synthefy_forecasting_model_v2
    use_metadata: bool = True
    ratio: float = 0.5

    # for datasets that are periodic
    use_periodic_projection: bool = False

    use_probabilistic_forecast: bool = False

    def __init__(
        self,
        config: Optional[Union[str, DictConfig]] = None,
        metadata_encoder_config: MetadataEncoderConfig = MetadataEncoderConfig(),
    ):
        """
        config can be either path to a yaml file or a dictionary
        """
        self.metadata_encoder_config = metadata_encoder_config
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
                    logger.warning(
                        f"Attribute {key} not found in DenoiserConfig"
                    )

        self.validate_config()

    def validate_config(self) -> None:
        pass
