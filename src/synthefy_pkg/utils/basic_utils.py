import os
import pickle
import random
import shutil
import subprocess
import time
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from loguru import logger

from synthefy_pkg.configs.execution_configurations import Configuration

OKRED = "\033[91m"
OKBLUE = "\033[94m"
ENDC = "\033[0m"
OKGREEN = "\033[92m"
OKYELLOW = "\033[93m"

COMPILE = False


def get_dataloader_purpose(pl_dataloader):
    print(OKYELLOW + "checking dataloader" + ENDC)
    for batch in pl_dataloader.train_dataloader():
        for key, val in batch.items():
            print(key, val.shape)
        break
    print(OKYELLOW + "dataloader check over" + ENDC)


def get_classifier_config(config):
    return config.classifier_config


def get_gan_config(config):
    if config.gan_name == "p2p":
        gan_config = config.p2p_config
    elif config.gan_name == "p2p_v1":
        gan_config = config.p2p_v1_config
    elif config.gan_name == "wavegan":
        gan_config = config.wavegan_config
    elif config.gan_name == "wavegan_v1":
        gan_config = config.wavegan_v1_config
    else:
        raise ValueError("gan name not recognized")
    return gan_config


def get_autoencoder_config(config):
    if config.autoencoder_name == "informer_autoencoder":
        autoencoder_config = config.informer_autoencoder_config
    elif config.autoencoder_name == "csdi_autoencoder":
        autoencoder_config = config.csdi_autoencoder_config
    else:
        raise ValueError("autoencoder name not recognized")
    return autoencoder_config


def get_cltsp_config(config):
    return config.encoder_config


def get_dataset_config(config):
    # Show a warning that this deprecated function is being called:
    print(
        OKRED
        + "Warning: get_dataset_config is deprecated. Please use get_dataset_config_from_edict instead."
        + ENDC
    )
    if config.dataset_name == "electricity":
        dataset_config = config.electricity_dataset
    elif config.dataset_name == "electricity_conditional":
        dataset_config = config.electricity_conditional_dataset
    elif config.dataset_name == "electricity_comparison":
        dataset_config = config.electricity_comparison_dataset
    elif config.dataset_name == "traffic":
        dataset_config = config.traffic_dataset
    elif config.dataset_name == "air_quality":
        dataset_config = config.air_quality_dataset
    elif config.dataset_name == "twamp_one_month":
        dataset_config = config.twamp_one_month_dataset
    else:
        raise ValueError(f"dataset name not recognized: {config.dataset_name=}")
    return dataset_config


def seed_everything(seed=42, worker_id=-1, seed_by_device=False):
    """
    Set seeds for reproducibility across different processes.

    RECOMMENDATION:
    - For standard datasets, use the same seed for all workers and devices
    - For procedurally generated datasets, use a different seed for each process and worker

    Parameters
    ----------
    seed : int, default=42
        Base seed value
    worker_id : int, default=-1
        Worker ID to add to seed (for DataLoader workers). If -1, all workers will have the same seed.
    seed_by_device : bool, default=False
        If True, adds device rank to seed (for distributed training)

    Examples:
    For 2 devices, 16 workers:
    """
    final_seed = seed

    # Add device rank if requested
    if seed_by_device:
        device_rank = 0
        if dist.is_available() and dist.is_initialized():
            device_rank = dist.get_rank()
        final_seed += device_rank * 1000

    # Add worker rank if requested
    if worker_id > 0:
        final_seed += worker_id

    os.environ["PYTHONHASHSEED"] = str(final_seed)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def load_pickle_from_path(
    filepath: str,
    raise_error_if_not_exists: bool = True,
    default_return_value: Any = None,
) -> Dict[Any, Any]:
    if not os.path.exists(filepath):
        if raise_error_if_not_exists:
            raise ValueError(f"File not found: {filepath}")
        else:
            logger.error(
                f"File not found: {filepath}. Returning default value: {default_return_value}"
            )
            return default_return_value
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data


def check_missing_cols(required_cols: List[str], df: pd.DataFrame) -> None:
    """
    Check if the required columns are present in the dataframe
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def get_num_devices(requested_devices):
    if torch.cuda.is_available():
        if requested_devices > torch.cuda.device_count():
            logger.warning(
                f"Requested {requested_devices} devices, but only {torch.cuda.device_count()} are available. Using {torch.cuda.device_count()} devices."
            )
        return min(requested_devices, torch.cuda.device_count())
    return 1


def clean_lightning_logs(config_filepath: str, seconds: int = 5) -> None:
    """
    Warns the user about deleting lightning logs and removes them after a delay.

    Args:
        config_filepath: Path to the configuration file
        seconds: Number of seconds to wait before deletion
    """
    SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE", "")
    configuration = Configuration(config_filepath=config_filepath)
    lightning_logs_path = configuration.get_lightning_logs_path(
        SYNTHEFY_DATASETS_BASE
    )

    if (
        os.path.exists(lightning_logs_path)
        and len(os.listdir(lightning_logs_path)) > 0
    ):
        logger.warning(
            f"No model checkpoint path provided. The current lightning logs will be deleted if this run is not interrupted within {seconds} seconds. Press CTRL+C if you don't want this to happen!"
        )

        time.sleep(seconds)
        shutil.rmtree(lightning_logs_path)


def get_system_specs() -> Dict[str, str]:
    """
    Get the current system's specifications.

    Returns:
        dict: A dictionary containing the current system's CUDA version, GPU type, and driver version

    Raises:
        ValueError: If any of the system specs cannot be determined
    """
    system_specs = {}

    # Get CUDA version
    try:
        current_cuda = subprocess.check_output(["nvcc", "--version"]).decode(
            "utf-8"
        )
        system_specs["cuda"] = (
            current_cuda.split("release")[1].split(",")[0].strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Failed to determine CUDA version")

    # Get GPU type
    try:
        system_specs["gpu"] = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv"]
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Failed to determine GPU type")

    # Get driver version
    try:
        system_specs["driver"] = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Failed to determine driver version")

    return system_specs


def compare_system_specs(path: str):
    """
    Check if the current system's specs match the expected specs for the given key.

    Args:
        key (str): The key to look up in SPECS dictionary (e.g., 'gh', 'wr', 'sb')

    Returns:
        bool: True if specs match or warnings were issued, False if key is unknown
    """

    SPECS = {
        "gh": {"cuda": "12.4", "gpu": "Tesla T4", "driver": "550.127.05"},
        "wr": {"cuda": "12.7", "gpu": "NVIDIA A100", "driver": "565.57.01"},
        "sb": {"cuda": "12.4", "gpu": "NVIDIA A100", "driver": "550.144.03"},
    }

    # Check if we're using a base file with environment-specific metrics
    base_filename = os.path.basename(path)
    key = None

    # Extract environment key (gh, wr, sb) from filename if present
    if "_gh." in base_filename:
        key = "gh"
    elif "_wr." in base_filename:
        key = "wr"
    elif "_sb." in base_filename:
        key = "sb"
    else:
        logger.warning(
            "No key could be extracted from the base file name. Skipping system specs comparison."
        )
        return False

    expected_specs = SPECS[key]
    current_specs = get_system_specs()

    # Check CUDA version
    if "cuda" not in current_specs:
        logger.warning(
            f"CUDA version not detected, expected {expected_specs['cuda']}"
        )
    elif current_specs["cuda"] != expected_specs["cuda"]:
        logger.warning(
            f"CUDA version mismatch: expected {expected_specs['cuda']}, got {current_specs['cuda']}"
        )

    # Check GPU type
    if "gpu" not in current_specs:
        logger.warning(f"GPU not detected, expected {expected_specs['gpu']}")
    elif expected_specs["gpu"] not in current_specs["gpu"]:
        logger.warning(
            f"GPU mismatch: expected {expected_specs['gpu']}, got {current_specs['gpu']}"
        )

    # Check driver version
    if "driver" not in current_specs:
        logger.warning(
            f"Driver version not detected, expected {expected_specs['driver']}"
        )
    elif expected_specs["driver"] not in current_specs["driver"]:
        logger.warning(
            f"Driver version mismatch: expected {expected_specs['driver']}, got {current_specs['driver']}"
        )

    return True
