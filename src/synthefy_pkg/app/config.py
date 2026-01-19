import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

COMPILE = True


class DispatcherSettings(BaseSettings):
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )


class SynthefySettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    preprocessed_data_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESSED_DATA_PATH"}
    )
    synthesis_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_CONFIG_PATH"}
    )
    forecast_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_FORECAST_CONFIG_PATH"}
    )
    synthesis_model_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_MODEL_PATH"}
    )
    forecast_model_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_FORECAST_MODEL_PATH"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})


class SynthesisSettings(BaseSettings):
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    synthesis_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_CONFIG_PATH"}
    )
    synthesis_model_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_MODEL_PATH"}
    )
    show_gt_synthesis_timeseries: bool = Field(
        json_schema_extra={"env": "SYNTHEFY_SHOW_GT_SYNTHESIS_TIMESERIES"}
    )
    return_only_synthetic_in_streaming_response: bool = Field(
        default=True,
        json_schema_extra={
            "env": "SYNTHEFY_RETURN_ONLY_SYNTHETIC_IN_STREAMING_RESPONSE"
        },
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})


class SearchSettings(BaseSettings):
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})


class ForecastSettings(BaseSettings):
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    forecast_model_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_FORECAST_MODEL_PATH"}
    )
    forecast_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_FORECAST_CONFIG_PATH"}
    )
    show_gt_forecast_timeseries: bool = Field(
        json_schema_extra={"env": "SYNTHEFY_SHOW_GT_FORECAST_TIMESERIES"}
    )
    only_include_forecast_in_streaming_response: bool = Field(
        default=True,
        json_schema_extra={
            "env": "SYNTHEFY_ONLY_INCLUDE_FORECAST_IN_STREAMING_RESPONSE"
        },
    )
    return_only_synthetic_in_streaming_response: bool = Field(
        default=True,
        json_schema_extra={
            "env": "SYNTHEFY_RETURN_ONLY_SYNTHETIC_IN_STREAMING_RESPONSE"
        },
    )

    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})

    model_to_use: str = Field(
        default="sfmv2",  # acceptable values: "sfmv2", "tabpfnv1"
        json_schema_extra={"env": "SYNTHEFY_FORECAST_MODEL_TO_USE"},
    )


class ViewSettings(BaseSettings):
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})

    window_naming_config: Dict[str, str] = Field(
        default={
            "pretrained_anomaly_prefix": "Failure",
            "synthesis_prefix": "Synthesis",
            "forecast_prefix": "Forecast",
            "search_prefix": "Similar Window",
            "view_prefix": "Window",
        },
        json_schema_extra={"env": "WINDOW_NAMING_CONFIG"},
    )


class PreTrainedAnomalySettings(BaseSettings):
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})
    anomaly_threshold: float = Field(
        default=2.5, json_schema_extra={"env": "ANOMALY_THRESHOLD"}
    )
    window_naming_config: Dict[str, str] = Field(
        default={
            "pretrained_anomaly_prefix": "Failure",
            "synthesis_prefix": "Synthesis",
            "forecast_prefix": "Forecast",
            "search_prefix": "Similar Window",
            "view_prefix": "Window",
        },
        json_schema_extra={"env": "WINDOW_NAMING_CONFIG"},
    )


class FoundationModelApiSettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    metadata_datasets_bucket: str = Field(
        json_schema_extra={"env": "SYNTHEFY_METADATA_DATASETS_BUCKET"}
    )


class MetadataEmbeddingSettings(BaseSettings):
    metadata_index_host: str


class MetadataRagSettings(BaseSettings):
    use_llm_generation_in_metadata_recommendations: Optional[int] = Field(
        default=1,
        json_schema_extra={
            "env": "USE_LLM_GENERATION_IN_METADATA_RECOMMENDATIONS"
        },
    )
    number_of_vectors_to_retrieve_before_generation_scale_factor: Optional[
        int
    ] = Field(
        default=2,
        json_schema_extra={
            "env": "NUMBER_OF_VECTORS_TO_RETRIEVE_BEFORE_GENERATION_SCALE_FACTOR"
        },
    )


class PostprocessSettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    dataset_name: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASET_NAME"}
    )
    synthesis_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_CONFIG_PATH"}
    )
    forecast_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_FORECAST_CONFIG_PATH"}
    )
    preprocessed_data_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESSED_DATA_PATH"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})


class PostPreProcessSettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    dataset_name: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASET_NAME"}
    )
    preprocessed_data_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESSED_DATA_PATH"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})


class PreprocessSettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    dataset_name: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASET_NAME"}
    )
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})

    max_file_size: int = Field(
        default=1024 * 1024 * 1024,  # 1GB
        json_schema_extra={"env": "SYNTHEFY_MAX_FILE_SIZE"},
    )


class TrainSettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    synthesis_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_CONFIG_PATH"}
    )
    forecast_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_FORECAST_CONFIG_PATH"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})


class SummarizerSettings(BaseSettings):
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    dataset_name: str = Field(..., description="Name of the dataset")
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    json_save_path: str = Field(
        default=os.path.join(os.getenv("SYNTHEFY_PACKAGE_BASE", ""), "tmp"),
        json_schema_extra={"env": None},
    )


class DataRetrievalSettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})


class APISettings(BaseSettings):
    DATABASE_URL: Optional[str] = Field(
        default="sqlite:///:memory:", validation_alias="DATABASE_URL"
    )


class SynthefyCredentialsSettings(BaseSettings):
    s3_credentials_path: str = Field(
        default="s3://synthefy-core/credentials",
        json_schema_extra={"env": "SYNTHEFY_S3_CREDENTIALS_PATH"},
    )
    local_credentials_path: str = Field(
        default=f"{Path.home()}/.synthefy/credentials",
        json_schema_extra={"env": "SYNTHEFY_CREDENTIALS_PATH"},
    )
    gcp_file_name: str = Field(
        default="gcp.json",
        json_schema_extra={"env": "SYNTHEFY_GCP_FILE_NAME"},
    )

    @field_validator("local_credentials_path", mode="after")
    @classmethod
    def create_local_credentials_path(cls, v: str) -> str:
        """Create the local credentials directory if it doesn't exist."""
        expanded_path = os.path.expanduser(v)  # Expands ~ to /home/user
        # Create parent directory for the credentials file
        parent_dir = os.path.dirname(expanded_path)
        try:
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True, mode=0o755)
        except PermissionError as e:
            # If we can't create in the default location, try a writable location
            logger.warning(
                f"Permission denied creating {parent_dir}: {e}. "
                f"Using fallback location in user's home directory."
            )
            # Use a cache directory in the user's home that they can write to
            fallback_dir = os.path.join(Path.home(), ".cache", "synthefy", "credentials")
            try:
                os.makedirs(fallback_dir, exist_ok=True, mode=0o755)
                fallback_path = os.path.join(fallback_dir, os.path.basename(expanded_path))
                logger.info(f"Using fallback credentials path: {fallback_path}")
                return fallback_path
            except Exception as fallback_error:
                logger.error(
                    f"Failed to create fallback path {fallback_dir}: {fallback_error}. "
                    f"Please fix permissions on {parent_dir} or set SYNTHEFY_CREDENTIALS_PATH environment variable."
                )
                raise
        return expanded_path  # Return the expanded path


class SynthefyFoundationModelSettings(BaseSettings):
    local_model_path: str = Field(
        default=f"{Path.home()}/.synthefy/model/synthefy_foundation_model",
        json_schema_extra={"env": "SYNTHEFY_FOUNDATION_MODEL_LOCAL_PATH"},
    )
    available_models: Dict[str, str] = Field(
        default={"default": "Synthefy's Foundation Model V1.0"},
        json_schema_extra={"env": "SYNTHEFY_FOUNDATION_MODEL_AVAILABLE_MODELS"},
    )

    @field_validator("local_model_path", mode="after")
    @classmethod
    def create_model_path(cls, v: str) -> str:
        """Create the local model path if it doesn't exist."""
        expanded_path = os.path.expanduser(v)  # Expands ~ to /home/user
        try:
            # Try to create parent directories first
            parent_dir = os.path.dirname(expanded_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True, mode=0o755)
            os.makedirs(
                expanded_path, exist_ok=True, mode=0o755
            )  # Creates directory if it doesn't exist
        except PermissionError as e:
            # If we can't create in the default location, try a writable location
            logger.warning(
                f"Permission denied creating {expanded_path}: {e}. "
                f"Using fallback location in user's home directory."
            )
            # Use a cache directory in the user's home that they can write to
            fallback_path = os.path.join(
                Path.home(), ".cache", "synthefy", "model", "synthefy_foundation_model"
            )
            try:
                os.makedirs(fallback_path, exist_ok=True, mode=0o755)
                logger.info(f"Using fallback model path: {fallback_path}")
                return fallback_path
            except Exception as fallback_error:
                logger.error(
                    f"Failed to create fallback path {fallback_path}: {fallback_error}. "
                    f"Please fix permissions on {expanded_path} or set SYNTHEFY_FOUNDATION_MODEL_LOCAL_PATH environment variable."
                )
                raise
        return expanded_path  # Return the expanded path, not the original


class SupabaseSettings(BaseSettings):
    SUPABASE_URL: str = ""  # type: ignore
    SUPABASE_API_KEY: str = ""  # type: ignore


class SyntheticDataAgentSettings(BaseSettings):
    bucket_name: str = Field(json_schema_extra={"env": "SYNTHEFY_BUCKET_NAME"})
    dataset_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_DATASETS_BASE"}
    )
    json_save_path: str = Field(json_schema_extra={"env": "JSON_SAVE_PATH"})
    preprocess_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESS_CONFIG_PATH"}
    )
    preprocessed_data_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_PREPROCESSED_DATA_PATH"}
    )
    synthesis_config_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_CONFIG_PATH"}
    )
    synthesis_model_path: str = Field(
        json_schema_extra={"env": "SYNTHEFY_SYNTHESIS_MODEL_PATH"}
    )
