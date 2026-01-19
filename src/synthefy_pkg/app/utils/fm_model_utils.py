import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import boto3
import yaml
from loguru import logger

from synthefy_pkg.app.config import SynthefyFoundationModelSettings
from synthefy_pkg.app.data_models import (
    ConfidenceInterval,
    SynthefyFoundationModelMetadata,
    SynthefyFoundationModelTypeMetadata,
    get_model_card_as_dict,
)
from synthefy_pkg.app.utils.api_utils import cleanup_local_directories
from synthefy_pkg.app.utils.s3_utils import (
    download_directory_from_s3,
    parse_s3_url,
    read_s3_yaml_file,
)
from synthefy_pkg.utils.config_utils import load_yaml_config

SYNTHEFY_FM_SETTINGS = SynthefyFoundationModelSettings()
MODEL_CARD_FILENAME = "model_card.yaml"
MODEL_CONFIG_FILENAME = "model_config.yaml"
FOUNDATION_MODEL_METADATA_FILENAME = "synthefy_model_metadata.yaml"


def build_s3_full_path(
    metadata: SynthefyFoundationModelMetadata, model_name: str
) -> str:
    """
    Build the full s3 path for a model.
    """
    foundation_model_metadata = getattr(metadata, model_name)
    model_dir_s3_url = metadata.s3_url
    if model_dir_s3_url.endswith("//"):
        model_dir_s3_url = model_dir_s3_url[:-1]
    elif not model_dir_s3_url.endswith("/"):
        model_dir_s3_url += "/"
    date = foundation_model_metadata.date
    version = foundation_model_metadata.model_version
    return f"{model_dir_s3_url}{version}_{date}/"


def get_available_models(model_dir: str | Path) -> List[Path]:
    """
    Get all available models in the local model directory.
    """
    model_dir = Path(model_dir).expanduser()
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        return []
    return [
        model_dir / model_name
        for model_name in model_dir.iterdir()
        if model_name.is_dir()
    ]


def download_local_ckpt_file(
    source_file: str | Path, destination_dir: str | Path
) -> Path:
    """
    Copy a single .ckpt file to destination directory.
    """
    source_file = Path(source_file)
    destination_dir = Path(destination_dir).expanduser()

    print(f"Attempting to copy from: {source_file}")
    print(f"Destination directory: {destination_dir}")

    if not source_file.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_file}")

    if not source_file.suffix == ".ckpt":
        raise ValueError(f"Source file is not a .ckpt file: {source_file}")

    # Copy the file to destination directory
    dest_file = destination_dir / source_file.stem

    # Create destination directory if it doesn't exist
    dest_file.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, dest_file)
    logger.info(f"Successfully copied: {source_file} -> {dest_file}")
    return dest_file


def get_model_ckpt_and_config_path(
    model_dir: Path,
    ckpt_filename: str = "*.ckpt",
    config_filename: str = "*.yaml",
) -> Tuple[Path | None, Path | None]:
    """
    Get the model checkpoint and config path from the model directory.
    """
    model_checkpoint_path = list(model_dir.glob(ckpt_filename))
    model_config_path = list(model_dir.glob(config_filename))

    if not model_checkpoint_path:
        cleanup_local_directories([str(model_dir)])
        logger.error(f"No .ckpt file found in {model_dir}")
        return None, None
    if not model_config_path:
        cleanup_local_directories([str(model_dir)])
        logger.error(f"No .yaml file found in {model_dir}")
        return None, None

    return model_checkpoint_path[0], model_config_path[0]


def get_foundation_model_info(
    local_model_dir: str | Path,
    foundation_model_path: str | None = None,
) -> SynthefyFoundationModelMetadata:
    """
    Get the model info from the model directory such as .ckpt and .yaml files.

    Args:
        local_model_dir: The directory containing the model.

    Returns:
        SynthefyFoundationModelMetadata

        Example:
        SynthefyFoundationModelMetadata(
            forecasting=SynthefyFoundationModelTypeMetadata(
                model_version="v1",
                date="2021-01-01",
                local_checkpoint_path=Path("path/to/checkpoint.ckpt"),
                local_config_path=Path("path/to/config.yaml"),
                available=True,
            ),
            synthesis=SynthefyFoundationModelTypeMetadata(
                model_version="v1",
                date="2021-01-01",
                local_checkpoint_path=Path("path/to/checkpoint.ckpt"),
                local_config_path=Path("path/to/config.yaml"),
                available=True,
            ),
        )
    """
    is_model_downloaded = False
    local_model_root_dir = Path(local_model_dir).expanduser()
    available_model_dirs = get_available_models(local_model_root_dir)
    if available_model_dirs:
        foundation_model_metadata = {}
        for model_dir_path in available_model_dirs:
            model_checkpoint_path, model_config_path = (
                get_model_ckpt_and_config_path(model_dir_path)
            )
            if model_checkpoint_path and model_config_path:
                is_model_downloaded = True
                model_card_yaml_path = (
                    Path(model_dir_path) / MODEL_CARD_FILENAME
                )
                if model_card_yaml_path.exists():
                    with open(model_card_yaml_path, "r") as yaml_file:
                        model_card = yaml.safe_load(yaml_file)
                foundation_model_metadata[model_dir_path.name] = {
                    "model_version": model_card.get("model_version"),
                    "date": model_card.get("date"),
                    "available": model_card.get("available"),
                    "local_checkpoint_path": model_card.get(
                        "local_checkpoint_path"
                    ),
                    "local_config_path": model_card.get("local_config_path"),
                }
        return SynthefyFoundationModelMetadata(**foundation_model_metadata)

    if not is_model_downloaded:
        s3_client = boto3.client("s3")
        foundation_model_metadata = get_model_metadata(
            foundation_model_path=foundation_model_path
        )

        for available_model_name in foundation_model_metadata.available_models:
            foundation_model_s3_url = build_s3_full_path(
                foundation_model_metadata, available_model_name
            )

            local_model_dir = local_model_root_dir / available_model_name
            local_model_dir.mkdir(parents=True, exist_ok=True)
            try:
                bucket, key = parse_s3_url(s3_url=foundation_model_s3_url)
                # Download the model from s3
                download_directory_from_s3(
                    s3_client=s3_client,
                    bucket=bucket,
                    s3_dir=key,
                    local_dir=str(local_model_dir),
                )
                logger.debug(f"Downloaded model from s3: {local_model_dir}")
                model_checkpoint_path, model_config_path = (
                    get_model_ckpt_and_config_path(
                        model_dir=local_model_dir,
                        config_filename=MODEL_CONFIG_FILENAME,
                    )
                )
                setattr(
                    getattr(foundation_model_metadata, available_model_name),
                    "local_checkpoint_path",
                    str(model_checkpoint_path),
                )
                setattr(
                    getattr(foundation_model_metadata, available_model_name),
                    "local_config_path",
                    str(model_config_path),
                )
                model_card = get_model_card_as_dict(
                    foundation_model_metadata, available_model_name
                )
                model_card_yaml_path = local_model_dir / MODEL_CARD_FILENAME
                with open(model_card_yaml_path, "w") as yaml_file:
                    yaml.dump(
                        model_card,
                        yaml_file,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                logger.info(f"Model card saved as YAML: {model_card_yaml_path}")
            except Exception as e:
                logger.error(f"Error downloading model from s3: {e}")
                # Clean up the local model directory if it exists
                cleanup_local_directories([str(local_model_dir)])
                raise e

    return foundation_model_metadata


def get_model_metadata(
    foundation_model_path: str | None = None,
) -> SynthefyFoundationModelMetadata:
    """
    Get the metadata for a model from the s3 - dev or prod.

    Args:
        model_name: Name of the model type ('forecasting' or 'synthesis')

    Returns:
        FoundationModelTypeMetadata: Metadata for the specified model type
    """
    try:
        if not foundation_model_path:
            synthefy_config = load_yaml_config(
                os.environ.get("CONFIG_PATH", ""),
            )
            foundation_model_path = synthefy_config.get(
                "synthefy_foundation_model_path"
            )
    except Exception as e:
        logger.error(f"Error loading synthefy config: {e}")
        raise e

    if not foundation_model_path:
        msg = "Model path is not set in the config"
        raise ValueError(msg)

    foundation_model_metadata_path = (
        foundation_model_path + FOUNDATION_MODEL_METADATA_FILENAME
    )
    try:
        model_metadata_dict = read_s3_yaml_file(foundation_model_metadata_path)
    except Exception:
        msg = "Fail to locate the foundation model metadata path from S3"
        raise ValueError(msg)

    # adding s3_url to the model metadata
    model_metadata_dict["s3_url"] = foundation_model_path
    return SynthefyFoundationModelMetadata(**model_metadata_dict)


def post_process_forecast_values(
    forecast_values: List[float],
    confidence_intervals: List[ConfidenceInterval],
    univariate_confidence_intervals: List[ConfidenceInterval],
    ground_truth_values: List[float | None],
    future_time_stamps: List[datetime],
) -> Tuple[
    List[float],
    List[float | None],
    List[ConfidenceInterval],
    List[ConfidenceInterval],
]:
    num_future_time_stamps = len(future_time_stamps)

    forecast_values = forecast_values[:num_future_time_stamps]
    confidence_intervals = confidence_intervals[:num_future_time_stamps]
    univariate_confidence_intervals = univariate_confidence_intervals[
        :num_future_time_stamps
    ]
    ground_truth_values = ground_truth_values[:num_future_time_stamps]

    return (
        forecast_values,
        ground_truth_values,
        confidence_intervals,
        univariate_confidence_intervals,
    )
