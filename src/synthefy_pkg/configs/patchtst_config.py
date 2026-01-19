from dataclasses import dataclass
from typing import Optional, Union, List

import yaml
from loguru import logger
from omegaconf import DictConfig

COMPILE = True

@dataclass
class PatchTSTConfig:
    num_input_channels: int = 1,
    context_length: int = 32,
    distribution_output: str = "student_t",
    loss: str = "mse",
    # PatchTST arguments
    patch_length: int = 1,
    patch_stride: int = 1,
    # Transformer architecture configuration
    num_hidden_layers: int = 3,
    d_model: int = 128,
    num_attention_heads: int = 4,
    share_embedding: bool = True,
    channel_attention: bool = False,
    ffn_dim: int = 512,
    norm_type: str = "batchnorm",
    norm_eps: float = 1e-05,
    attention_dropout: float = 0.0,
    positional_dropout: float = 0.0,
    path_dropout: float = 0.0,
    ff_dropout: float = 0.0,
    bias: bool = True,
    activation_function: str = "gelu",
    pre_norm: bool = True,
    positional_encoding_type: str = "sincos",
    use_cls_token: bool = False,
    init_std: float = 0.02,
    share_projection: bool = True,
    scaling: Optional[Union[str, bool]] = "std",
    # mask pretraining
    do_mask_input: Optional[bool] = None,
    mask_type: str = "random",
    random_mask_ratio: float = 0.5,
    num_forecast_mask_patches: Optional[Union[List[int], int]] = [2],
    channel_consistent_masking: Optional[bool] = False,
    unmasked_channel_indices: Optional[List[int]] = None,
    mask_value: int = 0,
    # head
    pooling_type: str = "mean",
    head_dropout: float = 0.0,
    prediction_length: int = 24,
    num_targets: int = 1,
    output_range: Optional[List] = None,
    # distribution head
    num_parallel_samples: int = 100,
    
    def __init__(
        self,
        config: Optional[Union[str, DictConfig]] = None,
    ):
        """
        config can be either path to a yaml file or a dictionary
        """
        if config is not None:
            if isinstance(config, str):
                with open(config, "r") as file:
                    config = yaml.safe_load(file)

            for key, value in config.items():
                try:
                    setattr(self, str(key), value)
                except AttributeError:
                    logger.warning(f"Attribute {key} not found in patchtst_config")

        self.validate_config()

    def validate_config(self) -> None:
        pass