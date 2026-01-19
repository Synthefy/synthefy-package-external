# Individual configuration classes for each learning rate scheduler since they have different parameters
from dataclasses import dataclass
from typing import Optional, Union

import yaml
from loguru import logger
from omegaconf import DictConfig


def get_lr_config(config: DictConfig):
    if "scheduler_name" not in config:
        raise ValueError("scheduler_name must be set")

    if config["scheduler_name"] == "exponential":
        return ExponentialLRConfig(config)
    elif config["scheduler_name"] == "reduce_lr_on_plateau":
        return ReduceLROnPlateauConfig(config)
    elif config["scheduler_name"] == "cyclic":
        return CyclicLRConfig(config)
    elif config["scheduler_name"] == "constant":
        return ConstantLRConfig(config)
    elif config["scheduler_name"] == "linear_warmup":
        return LinearWarmupLRConfig(config)
    elif config["scheduler_name"] == "cosine_warmup":
        return CosineWarmupLRConfig(config)
    elif config["scheduler_name"] == "cosine_with_restarts":
        return CosineWithRestartsLRConfig(config)
    elif config["scheduler_name"] == "polynomial_decay_warmup":
        return PolynomialDecayWarmupLRConfig(config)
    else:
        raise ValueError(f"Unknown scheduler_name: {config['scheduler_name']}")


@dataclass
class ExponentialLRConfig:
    gamma: float = 1.0  # default to constant
    step_size: int = 1
    scheduler_name: str = "exponential"

    def __init__(self, config: Optional[Union[str, DictConfig]] = None):
        if config is not None:
            if isinstance(config, str):
                with open(config, "r") as file:
                    config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )
        for key, value in config.items():
            try:
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(f"Attribute {key} not found in TrainingConfig")

        self.validate_config()

    def validate_config(self):
        if self.gamma <= 0:
            raise ValueError("gamma must be greater than 0")
        if self.step_size <= 0:
            raise ValueError("step_size must be greater than 0")


@dataclass
class ReduceLROnPlateauConfig:
    factor: float = 0.1
    patience: int = 10
    scheduler_name: str = "reduce_lr_on_plateau"

    def __init__(self, config: Optional[Union[str, DictConfig]] = None):
        if config is not None:
            if isinstance(config, str):
                with open(config, "r") as file:
                    config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )
        for key, value in config.items():
            try:
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(
                    f"Attribute {key} not found in ReduceLROnPlateauConfig"
                )

        self.validate_config()

    def validate_config(self):
        if self.factor <= 0:
            raise ValueError("factor must be greater than 0")
        if self.patience <= 0:
            raise ValueError("patience must be greater than 0")


@dataclass
class CyclicLRConfig:
    base_lr: float = 5e-5
    max_lr: float = 1e-4
    step_size_up: int = 10
    step_size_down: int = 10
    scheduler_name: str = "cyclic"

    def __init__(self, config: Optional[Union[str, DictConfig]] = None):
        if config is not None:
            if isinstance(config, str):
                with open(config, "r") as file:
                    config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )
        for key, value in config.items():
            try:
                if key == "base_lr":
                    value = float(value)
                elif key == "max_lr":
                    value = float(value)
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(f"Attribute {key} not found in CyclicLRConfig")

        self.validate_config()

    def validate_config(self):
        if self.base_lr <= 0:
            raise ValueError("base_lr must be greater than 0")
        if self.max_lr <= 0:
            raise ValueError("max_lr must be greater than 0")
        if self.step_size_up <= 0:
            raise ValueError("step_size_up must be greater than 0")
        if self.step_size_down <= 0:
            raise ValueError("step_size_down must be greater than 0")


@dataclass
class ConstantLRConfig:
    scheduler_name: str = "constant"

    def __init__(self, config: Union[str, DictConfig]):
        if isinstance(config, str):
            with open(config, "r") as file:
                config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )

        for key, value in config.items():
            try:
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(f"Attribute {key} not found in ConstantLRConfig")

    def validate_config(self):
        pass  # No parameters to validate


@dataclass
class LinearWarmupLRConfig:
    warmup_steps: int = 0
    warmup_proportion: float = -1.0
    max_steps: int = 1000
    scheduler_name: str = "linear_warmup"

    def __init__(self, config: Union[str, DictConfig]):
        if isinstance(config, str):
            with open(config, "r") as file:
                config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )

        for key, value in config.items():
            try:
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(
                    f"Attribute {key} not found in LinearWarmupLRConfig"
                )

        self.validate_config()

    def validate_config(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        if self.warmup_proportion >= 0 and self.warmup_steps > 0:
            logger.warning(
                "Both warmup_proportion and warmup_steps are set. Using warmup_proportion."
            )


@dataclass
class CosineWarmupLRConfig:
    warmup_steps: int = 0
    warmup_proportion: float = -1.0
    max_steps: int = 1000
    scheduler_name: str = "cosine_warmup"

    def __init__(self, config: Union[str, DictConfig]):
        if isinstance(config, str):
            with open(config, "r") as file:
                config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )

        for key, value in config.items():
            try:
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(
                    f"Attribute {key} not found in CosineWarmupLRConfig"
                )

        self.validate_config()

    def validate_config(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        if self.warmup_proportion >= 0 and self.warmup_steps > 0:
            logger.warning(
                "Both warmup_proportion and warmup_steps are set. Using warmup_proportion."
            )


@dataclass
class CosineWithRestartsLRConfig:
    warmup_steps: int = 0
    warmup_proportion: float = -1.0
    max_steps: int = 1000
    cosine_num_cycles: int = 1
    cosine_amplitude_decay: float = 1.0
    cosine_lr_end: float = 0.0
    scheduler_name: str = "cosine_with_restarts"

    def __init__(self, config: Union[str, DictConfig]):
        if isinstance(config, str):
            with open(config, "r") as file:
                config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )

        for key, value in config.items():
            try:
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(
                    f"Attribute {key} not found in CosineWithRestartsLRConfig"
                )

        self.validate_config()

    def validate_config(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        if self.cosine_num_cycles <= 0:
            raise ValueError("cosine_num_cycles must be greater than 0")
        if self.cosine_amplitude_decay <= 0:
            raise ValueError("cosine_amplitude_decay must be greater than 0")
        if self.warmup_proportion >= 0 and self.warmup_steps > 0:
            logger.warning(
                "Both warmup_proportion and warmup_steps are set. Using warmup_proportion."
            )


@dataclass
class PolynomialDecayWarmupLRConfig:
    warmup_steps: int = 0
    warmup_proportion: float = -1.0
    max_steps: int = 1000
    poly_decay_lr_end: float = 0.0
    poly_decay_power: float = 1.0
    scheduler_name: str = "polynomial_decay_warmup"

    def __init__(self, config: Union[str, DictConfig]):
        if isinstance(config, str):
            with open(config, "r") as file:
                config = yaml.safe_load(file)

        assert isinstance(config, DictConfig) or isinstance(config, dict), (
            "config must be a DictConfig after loading"
        )

        for key, value in config.items():
            try:
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(
                    f"Attribute {key} not found in PolynomialDecayWarmupLRConfig"
                )

        self.validate_config()

    def validate_config(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        if self.poly_decay_power <= 0:
            raise ValueError("poly_decay_power must be greater than 0")
        if self.warmup_proportion >= 0 and self.warmup_steps > 0:
            logger.warning(
                "Both warmup_proportion and warmup_steps are set. Using warmup_proportion."
            )
