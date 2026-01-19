import json
import os
from typing import Dict, List

from dotenv import load_dotenv

COMPILE = False

SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")
assert SYNTHEFY_PACKAGE_BASE is not None
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))
SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


def load_column_names(dataset_name: str) -> Dict[str, List[str]]:
    """
    Load column names from the colnames.json file.
    Parameters:
        - dataset_name (str): Name of the dataset directory
    Returns:
        - Dict[str, List[str]]: Dictionary containing column names for different window types
    """
    try:
        colnames_path = os.path.join(
            str(SYNTHEFY_DATASETS_BASE), dataset_name, "colnames.json"
        )
        with open(colnames_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(
            f"Colnames file doesn't exist for dataset: {dataset_name}: {colnames_path}"
        )


def load_timeseries_colnames(dataset_name: str) -> List[str]:
    """
    Load the timeseries column names from the dataset directory.
    """
    colnames_data = load_column_names(dataset_name)
    channel_names = colnames_data["timeseries_colnames"]
    return channel_names
