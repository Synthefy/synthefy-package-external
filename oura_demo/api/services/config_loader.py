"""
ConfigLoader for dynamic config loading by dataset_name.

Maps dataset names to their corresponding preprocessing and synthesis config files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger
from models import DatasetName, RequiredColumns

# Base path for configs - relative to synthefy-package root
SYNTHEFY_PACKAGE_BASE = os.getenv(
    "SYNTHEFY_PACKAGE_BASE",
    str(
        Path(__file__).parent.parent.parent.parent
    ),  # oura_demo -> synthefy-package
)
CONFIGS_BASE = Path(SYNTHEFY_PACKAGE_BASE) / "examples" / "configs"

# Mapping of dataset_name -> config file paths
DATASET_CONFIG_MAPPING: Dict[str, Dict[str, str]] = {
    DatasetName.OURA.value: {
        "preprocess": "preprocessing_configs/config_oura_preprocessing.json",
        "synthesis": "synthesis_configs/config_oura_synthesis.yaml",
    },
    DatasetName.OURA_SUBSET.value: {
        "preprocess": "preprocessing_configs/config_oura_subset_preprocessing.json",
        "synthesis": "synthesis_configs/config_oura_subset_synthesis.yaml",
    },
    DatasetName.PPG.value: {
        "preprocess": "preprocessing_configs/config_ppg_preprocessing.json",
        "synthesis": "synthesis_configs/config_ppg_synthesis.yaml",
    },
}


class ConfigLoader:
    """Load preprocessing and synthesis configs by dataset_name."""

    def __init__(self, dataset_name: str):
        """Initialize ConfigLoader for a specific dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "oura", "oura_subset", "ppg")

        Raises:
            ValueError: If dataset_name is not in DATASET_CONFIG_MAPPING
        """
        if dataset_name not in DATASET_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Valid options: {list(DATASET_CONFIG_MAPPING.keys())}"
            )

        self.dataset_name = dataset_name
        self._config_paths = DATASET_CONFIG_MAPPING[dataset_name]
        self._preprocess_config: Dict[str, Any] = {}
        self._synthesis_config: Dict[str, Any] = {}

        self._load_configs()

    def _load_configs(self) -> None:
        """Load preprocessing and synthesis configs from files."""
        # Load preprocessing config
        preprocess_path = CONFIGS_BASE / self._config_paths["preprocess"]
        logger.info(f"Loading preprocessing config from: {preprocess_path}")
        with open(preprocess_path) as f:
            self._preprocess_config = json.load(f)

        # Load synthesis config
        synthesis_path = CONFIGS_BASE / self._config_paths["synthesis"]
        logger.info(f"Loading synthesis config from: {synthesis_path}")
        with open(synthesis_path) as f:
            self._synthesis_config = yaml.safe_load(f)

    @property
    def preprocess_config(self) -> Dict[str, Any]:
        """Get the preprocessing config dictionary."""
        return self._preprocess_config

    @property
    def synthesis_config(self) -> Dict[str, Any]:
        """Get the synthesis config dictionary."""
        return self._synthesis_config

    @property
    def preprocess_config_path(self) -> str:
        """Get the full path to the preprocessing config file."""
        return str(CONFIGS_BASE / self._config_paths["preprocess"])

    @property
    def synthesis_config_path(self) -> str:
        """Get the full path to the synthesis config file."""
        return str(CONFIGS_BASE / self._config_paths["synthesis"])

    def get_required_columns(self) -> RequiredColumns:
        """Return all required columns from preprocessing config.

        Returns:
            RequiredColumns object with timeseries, discrete, continuous, and group_labels
        """
        return RequiredColumns(
            timeseries=self._preprocess_config.get("timeseries", {}).get(
                "cols", []
            ),
            discrete=self._preprocess_config.get("discrete", {}).get(
                "cols", []
            ),
            continuous=self._preprocess_config.get("continuous", {}).get(
                "cols", []
            ),
            group_labels=self._preprocess_config.get("group_labels", {}).get(
                "cols", []
            ),
        )

    def get_window_size(self) -> int:
        """Get the window size from preprocessing config.

        Returns:
            Window size as integer
        """
        return self._preprocess_config.get("window_size", 96)

    def get_num_channels(self) -> int:
        """Get the number of time series channels from synthesis config.

        Returns:
            Number of channels as integer
        """
        return self._synthesis_config.get("dataset_config", {}).get(
            "num_channels", 1
        )

    def get_all_required_column_names(self) -> List[str]:
        """Get flat list of all required column names (excluding group_labels).

        Returns:
            List of all required column names
        """
        required = self.get_required_columns()
        return required.timeseries + required.discrete + required.continuous

    def validate_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Validate a list of columns against the config requirements.

        Args:
            columns: List of column names from uploaded data

        Returns:
            Dictionary with validation results:
                - valid: bool
                - missing_columns: List[str]
                - extra_columns: List[str]
        """
        required_cols = set(self.get_all_required_column_names())
        group_labels = set(self.get_required_columns().group_labels)
        provided_cols = set(columns)

        missing = list(required_cols - provided_cols)
        extra = list(provided_cols - required_cols - group_labels)

        return {
            "valid": len(missing) == 0,
            "missing_columns": sorted(missing),
            "extra_columns": sorted(extra),
        }


# Cache for ConfigLoader instances to avoid reloading configs
_config_cache: Dict[str, ConfigLoader] = {}


def get_config_loader(dataset_name: str) -> ConfigLoader:
    """Get a ConfigLoader instance (cached).

    Args:
        dataset_name: Name of the dataset

    Returns:
        ConfigLoader instance for the dataset
    """
    if dataset_name not in _config_cache:
        _config_cache[dataset_name] = ConfigLoader(dataset_name)
    return _config_cache[dataset_name]
