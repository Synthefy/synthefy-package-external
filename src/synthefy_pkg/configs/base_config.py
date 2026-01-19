from dataclasses import dataclass, field
from typing import Optional, Union

import yaml
from loguru import logger
from omegaconf import DictConfig

COMPILE=True

@dataclass
class BaseConfig:
    config: Optional[Union[str, DictConfig]] = None

    def load_config(self):
        """
        config can be either path to a yaml file or a dictionary.

        Values in the config will override the default values.
        """
        if self.config is None:
            return

        if isinstance(self.config, str):
            with open(self.config, "r") as file:
                self.config = yaml.safe_load(file)

        if isinstance(self.config, dict):
            self.config = DictConfig(self.config)

    def load_values_from_config(self):
        if self.config is None:
            return

        assert isinstance(self.config, DictConfig), "config must be a dictionary"

        for key, value in self.config.items():
            try:
                if key == "learning_rate":
                    value = float(value)
                setattr(self, str(key), value)
            except AttributeError:
                logger.warning(
                    f"Attribute {key} not found in TrainingConfig"
                )

    def __post_init__(self):
        # Override if you need some other initialization behavior
        # Use super() to call the parent class's __post_init__ in subclasses
        self.load_config()
        self.load_values_from_config()
