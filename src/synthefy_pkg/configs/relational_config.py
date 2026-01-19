from dataclasses import dataclass, field
from typing import Optional, Union

import yaml
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.dataset_configs import DatasetConfig

COMPILE = False


@dataclass
class SubDatasetConfig:
    device: str = "cuda:0"
    num_workers: int = 16
    batch_size: int = 128

    def __init__(
        self,
        config: Optional[Union[str, DictConfig]] = None,
    ):
        """
        A dummy configuration that allows you to have a dataset_config as a subdirectory
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
                        f"Attribute {key} not found in foundation_model_config"
                    )

            self.dataset_config = DatasetConfig(config=config)

        self.validate_config()

    def validate_config(self) -> None:
        pass


@dataclass
class RelationConfig:
    output_subdir: str = "output/relations"
    splits: list[str] = field(default_factory=lambda: ["train"])
    batch_size: int = 128
    window_size: int = 128
    device: str = "cuda:0"
    dataset_dir: str = ""
    num_workers: int = 16
    shard_size: int = 20
    dataset_chunk_size: int = 128
    dataset_relation_types: list[str] = field(
        default_factory=lambda: ["text_cosine"]
    )
    dataset_scaling_lambdas: list[float] = field(default_factory=lambda: [1])
    dataset_combine_operation: str = "sum"
    window_relation_types: list[str] = field(
        default_factory=lambda: ["time_overlap"]
    )
    window_scaling_lambdas: list[float] = field(default_factory=lambda: [1])
    window_combine_operation: str = "sum"
    window_max_batches: int = 20
    reduced_series_embedding_dim: int = 2
    reduced_dataset_embedding_dim: int = 2
    series_embedding_methods: list[str] = field(default_factory=lambda: [])
    series_reduce_embed_method: str = "none"
    dataset_reduce_embed_method: str = "none"
    series_embedding_dims: list[int] = field(default_factory=lambda: [])

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

            assert isinstance(config, (dict, DictConfig)), (
                "config must be a dictionary"
            )
            for key, value in config.items():
                try:
                    setattr(self, str(key), value)
                except AttributeError:
                    logger.warning(
                        f"Attribute {key} not found in foundation_model_config"
                    )

            # Only try to create dataset_config if it exists in config
            if "dataset_loader_config" in config:
                self.window_loader_config = SubDatasetConfig(
                    config["window_loader_config"]
                )
                self.inner_loop_window_loader_config = SubDatasetConfig(
                    config["inner_loop_window_loader_config"]
                )
                self.dataset_loader_config = SubDatasetConfig(
                    config["dataset_loader_config"]
                )
            else:
                raise ValueError("dataset_config must be provided")

        self.validate_config()

    def validate_config(self) -> None:
        assert self.batch_size > 0
        assert self.window_size > 0
