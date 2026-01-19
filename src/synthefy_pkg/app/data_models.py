import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
import yaml
from isodate import parse_duration
from pydantic import (
    BaseModel,
    Field,
    RootModel,
    computed_field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from synthefy_pkg.anomaly_detection.synthefy_anomaly_detector_v2 import (
    AnomalyMetadata,
)

COMPILE = True


class TimeFrequency(BaseModel):
    """Represents a time frequency with a numeric value and unit."""

    value: int = Field(gt=0, description="The numeric part of the frequency")
    unit: str = Field(
        description="The unit of time (e.g., 'minute', 'hour', 'day')"
    )

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        valid_units = {
            "nanosecond",
            "microsecond",
            "millisecond",
            "second",
            "minute",
            "hour",
            "day",
            "week",
            "month",
            "quarter",
            "year",
        }
        if v not in valid_units:
            raise ValueError(
                f"Invalid unit '{v}'. Must be one of: {', '.join(sorted(valid_units))}"
            )
        return v

    def __str__(self) -> str:
        unit_name = self.unit if self.value == 1 else f"{self.unit}s"
        return f"{self.value} {unit_name}"


class DynamicTimeSeriesData(RootModel):
    root: Dict[
        str,
        Union[
            Dict[
                Union[str, int], Union[float, int, str, None]
            ],  # For regular time series data
            str,  # For JSON-encoded constraints
        ],
    ] = Field(
        description="Dictionary containing nested key-value pairs for time series data, "
        "where outer keys are column names and inner keys are row indices. "
        "Special key '_constraints_' accepts a JSON-encoded string containing synthesis constraints."
    )


#################### Shared Data Models ####################


class TimeStamps(BaseModel):
    name: str
    values: List[Any] = Field(..., description="List of timestamp values")


class TimeStampsRange(BaseModel):
    name: str
    min_time: str  # isoformat
    max_time: str  # isoformat
    interval: Optional[str]  # isoformat
    length: Optional[int]


class OneTimeSeries(BaseModel):
    name: str
    values: List[float | None]


class OneDiscreteMetaData(BaseModel):
    name: str
    values: List[Union[str, int, float, None]] = Field(
        default_factory=list,
        description="List of string, integer, or float values for discrete metadata",
    )


class OneContinuousMetaData(BaseModel):
    name: str
    values: List[float | None] = Field(
        default_factory=list,
        description="List of float values for continuous metadata",
    )


class MetaData(BaseModel):
    discrete_conditions: List[OneDiscreteMetaData] = Field(
        default_factory=list, description="List of discrete metadata conditions"
    )
    continuous_conditions: List[OneContinuousMetaData] = Field(
        default_factory=list,
        description="List of continuous metadata conditions",
    )


# New OneUI Data Model
# TODO - move forecast timestamp and anomaly timestamp to this class as optional.
class SynthefyTimeSeriesWindow(BaseModel):
    id: int = Field(default=0)
    name: str = Field(default="Window")
    timeseries_data: List[OneTimeSeries]
    metadata: MetaData
    timestamps: Optional[TimeStamps] = None
    text: Optional[str] = Field(default="")

    @field_validator("text")
    @classmethod
    def validate_markdown(cls, v):
        if v:
            markdown_patterns = [
                r"\*\*.+?\*\*",  # Bold (**bold**)
                r"\*.+?\*",  # Italic (*italic*)
                r"#{1,6}\s.+",  # Headers (# Header)
                r"`[^`\n]+`",  # Inline code (`code`)
                r"```[\s\S]+?```",  # Code blocks (```code```)
                r"\[.+?\]\(.+?\)",  # Links [text](url)
                r"^\s*[-*+]\s",  # Unordered lists (- item)
                r"^\s*\d+\.\s",  # Ordered lists (1. item)
                r"^>\s",  # Blockquotes (> quote)
            ]

            # Check if any Markdown pattern exists
            has_markdown = any(
                re.search(pattern, v, re.MULTILINE)
                for pattern in markdown_patterns
            )

            if has_markdown:
                return v  # It's valid Markdown, return as is

            # If text contains some Markdown but is incorrectly formatted, raise an error
            if any(
                re.search(pattern, v, re.MULTILINE)
                for pattern in markdown_patterns
            ):
                raise ValueError(
                    "Text contains improperly formatted Markdown elements."
                )

        return v  # Return plain text if no Markdown is found

    # TODO add below later.
    # anomaly_timestamps: Optional[TimeStamps] = []
    # forecast_timestamps: Optional[TimeStamps] = []


# MetaDataRange for view, search
class OneDiscreteMetaDataRange(BaseModel):
    name: str
    options: List[Union[str, int, float]]

    @field_validator("options")
    @classmethod
    def validate_options_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("options must contain at least one value")
        return v


class OneContinuousMetaDataRange(BaseModel):
    name: str
    min_val: Union[int, float]
    max_val: Union[int, float]


class MetaDataRange(BaseModel):
    discrete_conditions: List[OneDiscreteMetaDataRange] = []
    continuous_conditions: List[OneContinuousMetaDataRange] = []


# Misc
class InfoContainer(BaseModel):
    name: str
    type: str
    value: Any


# New Shared Data Models
class WindowsSelectionOptions(str, Enum):
    TRAIN = "TRAIN"
    VAL = "VAL"
    TEST = "TEST"
    UPLOADED_DATASET = "UPLOADED_DATASET"
    CURRENT_VIEW_WINDOWS = "CURRENT_VIEW_WINDOWS"


class SelectedAction(str, Enum):
    SYNTHESIS = "SYNTHESIS"
    VIEW = "VIEW"
    SEARCH = "SEARCH"
    FORECAST = "FORECAST"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"


class SynthefyTimeSeriesModelType(str, Enum):
    FOUNDATION_MODEL_SYNTHESIS = "synthefy-synthesis"
    FOUNDATION_MODEL_FORECASTING = "synthefy-forecasting"
    DEFAULT = "default"


class SelectedWindows(BaseModel):
    window_type: WindowsSelectionOptions
    window_indices: List[int] = [0]


#################### End Shared Data Models ####################


#################### Default UI Setup Data Models ####################
class SynthefyDefaultSetupOptions(BaseModel):
    windows: List[SynthefyTimeSeriesWindow]
    metadata_range: MetaDataRange  # Options for the metadata
    timestamps_range: TimeStampsRange  # Options for the timestamps
    text: List[
        str
    ] = []  # highest priority - can overwrite timestamp/metadata range


#################### End Default UI Setup Data Models ####################


class ConstraintType(str, Enum):
    MIN = "min"
    MAX = "max"
    ARGMAX = "argmax"
    ARGMIN = "argmin"


class OneConstraint(BaseModel):
    channel_name: str
    constraint_name: ConstraintType
    constraint_value: Union[int, float]


class ProjectionType(str, Enum):
    CLIPPING = "clipping"
    STRICT = "strict"


class SynthesisConstraints(BaseModel):
    constraints: List[OneConstraint] = []
    projection_during_synthesis: ProjectionType = ProjectionType.CLIPPING

    @model_validator(mode="after")  # pyright: ignore
    def validate_constraints(self) -> "SynthesisConstraints":
        if not self.constraints:
            return self

        if self.projection_during_synthesis == "clipping":
            invalid_constraints = [
                c
                for c in self.constraints
                if c.constraint_name
                not in [ConstraintType.MIN, ConstraintType.MAX]
            ]
            if invalid_constraints:
                raise ValueError(
                    "When projection_during_synthesis='clipping', only 'min' and 'max' "
                    "constraints are supported. Found invalid constraints: "
                    + ", ".join(
                        f"'{c.constraint_name}'" for c in invalid_constraints
                    )
                )
        return self


#################### SynthefyRequest Data Models ####################
class SynthefyRequest(BaseModel):
    windows: List[SynthefyTimeSeriesWindow]  # always shows the X windows

    # selected windows works as follows:
    # user can select any of the currently displayed windows (by index)
    # They can also select the entire dataset from train/val/test
    # or they can select "UPLOADED_DATASET" (previously CSV/JSON uploaded dataset)
    # If they select train/test/val or the uploaded dataset, they also have the option to select the indices of the windows they want to use
    selected_windows: SelectedWindows = SelectedWindows(
        window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS
    )
    # Check box next to chat bar.
    # User can check for each window or pick "UPLOADED_DATASET".

    # for search, (and forecast?), only 1 selected window is allowed
    n_synthesis_windows: int = 1
    n_view_windows: int = 5
    n_anomalies: int = 5
    top_n_search_windows: int = 5
    n_forecast_windows: int = 1

    selected_action: SelectedAction

    text: str = ""
    synthesis_constraints: Optional[SynthesisConstraints] = None


#################### End SynthefyRequest Data Models ####################


#################### SynthefyResponse Data Models ####################
class SynthefyResponse(BaseModel):
    windows: List[SynthefyTimeSeriesWindow]
    anomaly_timestamps: List[TimeStamps] = []  # red vertical line
    forecast_timestamps: List[TimeStamps] = []  # grey vertical line
    combined_text: str = Field(
        default="", description="Markdown-formatted string"
    )


#################### End SynthefyResponse Data Models ####################


################################### BELOW ARE OLD DATA MODELS ###########################


#################### View Data Models ####################
class ViewRequest(BaseModel):
    metadata_range: Optional[MetaDataRange] = None
    timestamps_range: Optional[TimeStampsRange] = (
        None  # one of the above 2 must be defined
    )
    # note - text overwrites timestamp/metadata since timestamp, metadata range are filled by default
    text: Optional[str] = Field(default="")
    search_set: Optional[
        List[Literal["train", "val", "test", "current_uploaded_dataset"]]
    ] = [
        "train",
        "val",
        "test",
    ]
    n_windows: int = 1


class ViewResponse(BaseModel):
    x_axis: List[TimeStamps]
    timeseries_data: List[List[OneTimeSeries]]
    metadata: List[MetaData]
    info_containers: Optional[List[InfoContainer]] = None
    text: Optional[str] = Field(default="")


#################### End View Data Models ####################


#################### Synthesis Data Models ####################
class SynthesisRequestOptions(BaseModel):
    metadata_range: MetaDataRange
    timestamps_range: Optional[TimeStampsRange]
    text: Optional[str] = (
        ""  # overwrites timestamp/metadata since timestamp, metadata range are filled by default
    )


class SynthesisRequest(BaseModel):
    input_timeseries: List[OneTimeSeries]
    metadata: MetaData
    timestamps: Optional[TimeStamps] = None
    # note - text overwrites metadata since metadata are filled by default
    text: Optional[str] = Field(default="")
    n_synthesis_windows: int = 1


class SynthesisResponse(BaseModel):
    x_axis: TimeStamps
    timeseries_data: List[OneTimeSeries]
    metadata: MetaData
    info_containers: Optional[List[InfoContainer]] = None
    text: Optional[str] = Field(default="")


#################### End Synthesis Data Models ####################


#################### Search Data Models ####################
class SearchRequest(BaseModel):
    search_query: List[
        OneTimeSeries
    ]  # required for demo purposes, optional later
    search_metadata: MetaData
    search_timestamps: Optional[TimeStamps] = None
    metadata_ranges: Optional[MetaDataRange] = None
    timestamps_range: Optional[TimeStampsRange] = (
        None  # range of timestamps to search over
    )
    n_closest: int = 1
    text: Optional[str] = Field(default="")
    # TODO - support search over synthesized data
    search_set: List[str] = ["train", "val", "test"]


class SearchResponse(BaseModel):
    x_axis: List[TimeStamps]
    timeseries_data: List[List[OneTimeSeries]]
    metadata: List[MetaData]
    info_containers: Optional[List[InfoContainer]] = None
    text: Optional[str] = Field(default="")


#################### End Search Data Models ####################


#################### Forecast Data Models ####################
class ForecastRequest(BaseModel):
    past_metadata: MetaData
    past_timeseries: List[OneTimeSeries]
    past_timestamps: Optional[TimeStamps]
    text: Optional[str] = Field(default="")


class ForecastResponse(BaseModel):
    x_axis: TimeStamps
    timeseries_data: List[OneTimeSeries]
    metadata: MetaData
    info_containers: Optional[List[InfoContainer]] = None
    start_of_forecast_timestamp: TimeStamps
    text: Optional[str] = Field(default="")


#################### End Forecast Data Models #####################


#################### Zero Shot Anomaly Data Models #####################
class ZeroShotAnomalyRequest(BaseModel):
    input_timeseries: List[OneTimeSeries]
    n_anomalies: int = 10


class ZeroShotAnomalyResponse(BaseModel):
    anomaly_timestamps: TimeStamps


#################### End Zero Shot Anomaly Data Models #####################


class PreTrainedAnomalyRequest(BaseModel):
    input_timeseries: List[OneTimeSeries]
    metadata: MetaData
    timestamps: TimeStamps
    n_anomalies: int = 10


class PreTrainedAnomalyResponse(BaseModel):
    x_axis: TimeStamps
    timeseries_data: List[OneTimeSeries]
    metadata: MetaData
    anomaly_timestamps: TimeStamps = []  # type: ignore
    info_containers: Optional[List[InfoContainer]] = None
    text: Optional[str] = Field(default="")


#################### End Pre Trained Anomaly Data Models #####################
#################### End Pre Trained Anomaly Data Models #####################


#################### Dataset Names Data Models #######################
class DatasetInfo(BaseModel):
    name: str
    last_modified: str = Field(description="ISO format datetime string")


class RetrievePreprocessedDatasetNamesRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID to fetch preprocessed datasets for"
    )


class RetrievePreprocessedDatasetNamesResponse(BaseModel):
    dataset_names: List[DatasetInfo] = Field(
        ..., description="List of preprocessed datasets with their metadata"
    )


#################### End Dataset Names Data Models #######################


#################### Train Job Data Models #######################
class RetrieveTrainJobIDsRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID to fetch training job IDs for"
    )
    dataset_name: str = Field(
        ..., description="Name of the preprocessed dataset"
    )


class RetrieveTrainJobIDsResponse(BaseModel):
    synthesis_train_job_ids: List[str] = Field(
        default_factory=list, description="List of synthesis training job IDs"
    )
    forecast_train_job_ids: List[str] = Field(
        default_factory=list, description="List of forecast training job IDs"
    )


class TrainingJobInfo(BaseModel):
    training_job_id: str
    job_type: Literal["synthesis", "forecast", "embedding"]
    dataset_name: str


class ListAllTrainingJobsRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID to fetch all training jobs for"
    )


class ListAllTrainingJobsResponse(BaseModel):
    training_jobs: List[TrainingJobInfo] = Field(
        ..., description="List of all training jobs"
    )


#################### End Train Job Data Models #######################


#################### Delete Dataset Data Models #####################


class DeletePreprocessedDatasetRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    dataset_name: str = Field(..., description="Preprocessed dataset name")


class DeletePreprocessedDatasetResponse(BaseModel):
    status: str = Field(..., description="Status of the deletion operation")
    message: str = Field(..., description="Detailed message about the deletion")
    deleted_dataset_path: str = Field(
        ..., description="Path of the deleted preprocessed dataset"
    )


class DeleteTrainingJobsRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    dataset_name: str = Field(..., description="Preprocessed dataset name")
    training_job_ids: List[str] = Field(
        ..., description="List of training job IDs to delete"
    )


class DeleteTrainingJobsResponse(BaseModel):
    status: str = Field(..., description="Status of the deletion operation")
    message: str = Field(..., description="Detailed message about the deletion")


#################### End Delete Dataset Data Models #####################


#################### AWS CREDENTIALS Data Models ####################


class AWSInfo(BaseModel):
    bucket_name: str = Field(description="Name of the bucket")
    user_id: str = Field(description="Customer ID")
    dataset_name: str = Field(description="Name of the dataset")


class S3Source(BaseModel):
    bucket_name: str
    key: str
    region: str
    access_key_id: str
    secret_access_key: str


class DataSource(str, Enum):
    UPLOAD = "upload"
    S3 = "s3"


#################### End AWS CREDENTIALS Data Models ####################


#################### Summarize Data Models ####################


class SummaryTimeSeriesItem(BaseModel):
    column_name: str = Field(alias="Column Name")
    range: str = Field(alias="Range")
    missing_percentage: str = Field(alias="Missing Percentage")
    outliers: str = Field(alias="% Outliers (>3.0 SD)")

    class Config:
        populate_by_name = True


class SummaryTimeRangeItem(BaseModel):
    min_time: str = Field(alias="Min Time")
    max_time: str = Field(alias="Max Time")
    time_interval: Optional[str] = Field(alias="Time Interval", default=None)

    class Config:
        populate_by_name = True


class SummaryMetadataItem(BaseModel):
    column_name: str = Field(alias="Column Name")
    type: str = Field(alias="Type")
    range: str = Field(alias="Range")
    outliers: str = Field(alias="% Outliers (>3.0 SD)")

    class Config:
        populate_by_name = True


class SummarySampleCount(BaseModel):
    metric: str = Field(alias="Metric")
    counts: int = Field(alias="Counts")

    class Config:
        populate_by_name = True


class SummarizeRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    data_source: DataSource = Field(
        default=DataSource.UPLOAD,
        description="Source of the data - either local upload or S3",
    )
    s3_source: Optional[S3Source] = Field(
        None,
        description="S3 source configuration, required when data_source is 's3'",
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Configuration dictionary for data summarization"
    )
    group_cols: Optional[str] = Field(
        None,
        description="Comma-separated list of column names for grouping. Takes precedence over config group_labels if both are provided.",
    )
    user_id: str = Field(..., description="User ID for organizing files in S3")
    skip_plots: bool = Field(
        default=False,
        description="Whether to skip generating plots in the summary",
    )

    @field_validator("s3_source")
    def validate_s3_source(cls, v, values):
        if values.data.get("data_source") == DataSource.S3 and v is None:
            raise ValueError("s3_source is required when data_source is 's3'")
        return v

    @model_validator(mode="after")
    def validate_grouping_options(self):
        if self.group_cols is None and self.config is None:
            raise ValueError(
                "At least one of group_cols or config must be provided"
            )
        return self


class SummarizeResponse(BaseModel):
    time_series_summary: Optional[List[SummaryTimeSeriesItem]] = None
    time_range: Optional[List[SummaryTimeRangeItem]] = None
    metadata_summary: Optional[List[SummaryMetadataItem]] = None
    columns_by_type: Optional[Dict[str, int]] = None
    sample_counts: Optional[List[SummarySampleCount]] = None
    html_url: Optional[str] = None

    class Config:
        populate_by_name = True


#################### End Summarize Data Models ####################


class RetrieveSummaryReportRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID requesting the dataset summary"
    )


class RetrieveSummaryReportResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the request (success/error)"
    )
    message: str = Field(..., description="Success/error message")
    presigned_url: Optional[str] = Field(
        None, description="S3 presigned URL to download the summary HTML"
    )


class PreprocessingConfig(BaseModel):
    window_size: int
    stride: int
    timeseries: Dict[str, Any]

    # Optional fields
    timestamps_col: List[str] = Field(default_factory=list)
    group_labels: Dict[str, Any] = Field(default_factory=dict)
    discrete: Dict[str, Any] = Field(default_factory=dict)
    continuous: Dict[str, Any] = Field(default_factory=dict)
    timestamps: Dict[str, Any] = Field(
        default={"timestamps_features": ["Y", "M", "D"], "scalers": {}},
        description="Configuration for timestamp features and their scalers",
    )
    window_from_beginning: bool = Field(
        default=False,
        description="Whether to start the window from the beginning",
    )
    shuffle: bool = Field(
        default=False, description="Whether to shuffle the windows"
    )
    train_val_split: List[float] = Field(
        default=[0.8, 0.8], description="Train/val/test split ratios"
    )
    use_label_col_as_discrete_metadata: bool = Field(
        default=True,
        description="Whether to use label column as discrete metadata",
    )
    search_encoder: Optional[str] = Field(
        default=None, description="Type of encoder to use for search"
    )
    channel_label: int = Field(
        default=0, description="Channel label for Fourier feature extraction"
    )
    add_fourier: bool = Field(
        default=False, description="Whether to add Fourier features"
    )
    fourier_params: Dict[str, int] = Field(
        default={"num_coefficients": 25, "label_num": 1},
        description="Parameters for Fourier feature extraction",
    )


class PreprocessRequest(BaseModel):
    dataset_name: str
    data_source: DataSource
    config: Dict[str, Any]
    s3_source: Optional[S3Source] = None
    aws_upload_info: Optional[AWSInfo] = None

    @field_validator("s3_source")
    def validate_s3_source(cls, v, values):
        if values.data.get("data_source") == DataSource.S3 and v is None:
            raise ValueError("s3_source is required when data_source is 's3'")
        return v

    class Config:
        use_enum_values = True


class PostprocessRequest(BaseModel):
    user_id: str
    job_id: str
    splits: List[str] = ["test"]


class PostPreProcessRequest(BaseModel):
    user_id: str
    jsd_threshold: float = Field(
        default=0.3, description="Threshold for Jensen-Shannon distance"
    )
    emd_threshold: float = Field(
        default=0.3, description="Threshold for Earth Mover's distance"
    )
    use_scaled_data: bool = Field(
        default=True, description="Whether to use scaled data"
    )
    pairwise_corr_figures: bool = Field(
        default=False,
        description="Whether to generate pairwise correlation figures",
    )
    downsample_factor: int = Field(
        default=50,
        description="Factor by which to downsample data for plotting efficiency",
    )


class PostPreProcessResponse(BaseModel):
    status: str
    message: str
    presigned_url: Optional[str] = None


class DatasetConfig(BaseModel):
    time_series_length: int = Field(
        description="Window size for time series data"
    )
    num_channels: int = Field(description="Number of time series features")
    num_discrete_conditions: int = Field(
        description="Number of discrete features"
    )
    num_continuous_labels: int = Field(
        description="Number of continuous features"
    )
    num_timestamp_labels: int = Field(
        description="Number of timestamp features"
    )


class PreprocessResponse(BaseModel):
    status: str
    message: str
    output_path: str
    dataset_config: DatasetConfig


class PostprocessResponse(BaseModel):
    status: str
    message: str
    presigned_url: Optional[str] = None


# DEFAULT TRAINING CONFIGS
class GetTrainConfigRequest(BaseModel):
    user_id: str
    task: Literal["synthesis", "forecast"]
    training_model_name: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class GetTrainConfigResponse(BaseModel):
    config: Dict[str, Any]
    message: str = "Successfully retrieved training configuration"


# SYNTHESIS TRAINING
class DatasetModelConfig(BaseModel):
    dataset_name: str
    num_channels: int
    time_series_length: int
    forecast_length: int
    required_time_series_length: int
    num_discrete_conditions: int
    num_discrete_labels: int = Field(default=1)
    num_continuous_labels: int
    discrete_condition_embedding_dim: int = Field(default=64)
    latent_dim: int = Field(default=64)
    batch_size: int = Field(default=16)
    use_metadata: bool = Field(default=True)


class DenoiserModelConfig(BaseModel):
    positional_embedding_dim: int = Field(default=64)
    channel_embedding_dim: int = Field(default=16)
    channels: int = Field(default=128)
    n_heads: int = Field(default=8)
    n_layers: int = Field(default=6)
    dropout_pos_enc: float = Field(default=0.2)
    use_metadata: bool = Field(default=True)


class ForecastingDenoiserModelConfig(BaseModel):
    denoiser_name: str = Field(default="synthefy_forecasting_model_v1")
    d_model: int = Field(default=256)
    patch_len: int = Field(default=16)
    stride: int = Field(default=8)
    dropout: float = Field(default=0.1)
    e_layers: int = Field(default=6)
    d_layers: int = Field(default=1)
    d_ff: int = Field(default=512)
    n_heads: int = Field(default=8)
    activation: str = Field(default="gelu")
    factor: int = Field(default=3)
    output_attention: bool = Field(default=True)
    use_metadata: bool = Field(default=True)


class MetadataEncoderConfig(BaseModel):
    channels: int = Field(default=128)
    n_heads: int = Field(default=8)
    num_encoder_layers: int = Field(default=2)


class ExecutionConfig(BaseModel):
    save_path: str = Field(default="training_logs")
    run_name: str
    generation_save_path: str = Field(default="generation_logs")
    experiment_name: str = Field(default="Time_Series_Diffusion_Training")


class TrainingConfig(BaseModel):
    max_epochs: int = Field(default=30)
    learning_rate: float = Field(default=1e-4)
    n_plots: int = Field(default=4)
    auto_lr_find: bool = Field(default=True)
    check_val_every_n_epoch: int = Field(default=10)
    log_every_n_steps: int = Field(default=1)
    num_devices: int = Field(default=1)
    strategy: str = Field(default="auto")


# TODO: Add model config to train config and use this as final validation in train_service
class FullTrainConfig(BaseModel):
    device: str = Field(
        default="cuda", description="Device to use for training"
    )
    task: str = Field(default="synthesis", description="Task to perform")
    dataset_config: DatasetModelConfig = Field(
        default_factory=DatasetModelConfig  # type: ignore
    )
    denoiser_config: DenoiserModelConfig = Field(
        default_factory=DenoiserModelConfig  # type: ignore
    )
    metadata_encoder_config: MetadataEncoderConfig = Field(
        default_factory=MetadataEncoderConfig  # type: ignore
    )
    execution_config: ExecutionConfig = Field(
        default_factory=ExecutionConfig  # type: ignore
    )
    training_config: TrainingConfig = Field(
        default_factory=TrainingConfig  # type: ignore
    )


class TrainRequest(BaseModel):
    run_from_checkpoint: bool = Field(default=True)
    dataset_name: str = Field(description="Name of the dataset")
    config: Dict[str, Any] = Field(
        description="Training configuration dictionary"
    )


class TrainResponse(BaseModel):
    status: str
    message: str
    training_job_name: str


# TODO - support Lists of training jobs
class TrainStatusRequest(BaseModel):
    training_job_name: str
    user_id: str = Field(..., description="User ID for authorization")


class TrainStatusResponse(BaseModel):
    status: str
    metrics: Dict[str, Any]
    timestamps: Optional[List[str]]
    secondary_status: Optional[str]
    job_name: str
    last_update: str


class TrainStopRequest(BaseModel):
    training_job_name: str
    user_id: str = Field(..., description="User ID for authorization")


class TrainStopResponse(BaseModel):
    response: Dict[str, Any]


class TrainingTaskType(str, Enum):
    """Enum representing valid training job types."""

    SYNTHESIS = "synthesis"
    FORECAST = "forecast"
    EMBEDDING = "embedding"


class SageMakerTrainingJobStatus(str, Enum):
    """Enum representing valid SageMaker training job statuses."""

    COMPLETED = "Completed"
    INPROGRESS = "InProgress"
    STOPPED = "Stopped"
    FAILED = "Failed"
    STOPPING = "Stopping"
    PENDING = "Pending"


class TrainListJobsRequest(BaseModel):
    action: Literal["list_training_jobs"] = Field(
        description="Action to perform - must be 'list_training_jobs'"
    )
    client_id: Optional[Union[str, List[str]]] = Field(
        description="Client ID(s) to filter training jobs", default=None
    )
    max_results: Optional[int] = Field(
        description="Maximum number of results to return (0 for all)",
        default=None,
    )
    # Keep these as optional for backward compatibility
    status: Optional[SageMakerTrainingJobStatus] = Field(
        description="Filter sagemaker training jobs by specific status",
        default=None,
    )
    dataset_name: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Dataset name(s) to filter sagemaker training jobs",
    )
    task: Optional[Union[TrainingTaskType, List[TrainingTaskType]]] = Field(
        default=None, description="Job type(s) to filter"
    )


class TrainJobInfo(BaseModel):
    """Information about a SageMaker training job."""

    dataset_name: str
    task: str
    TrainingJobName: str
    TrainingJobArn: str
    CreationTime: datetime
    TrainingEndTime: Optional[datetime] = None
    LastModifiedTime: datetime
    TrainingJobStatus: str
    HyperParameters: Optional[Dict[str, str]] = None


class TrainListJobsResponse(BaseModel):
    training_jobs: List[TrainJobInfo] = Field(
        description="List of sagemaker training jobs matching the filter criteria"
    )
    total_count: int = Field(description="Total number of jobs found")
    truncated: bool = Field(description="Whether the results are truncated")
    execution_time_seconds: float = Field(
        description="Time taken to fetch jobs"
    )
    timeout: Optional[bool] = Field(
        default=None, description="Whether the operation timed out"
    )


class SynthefyAgentSetupRequest(BaseModel):
    user_id: str
    dataset_name: str
    synthesis_training_job_id: Optional[str] = None
    forecast_training_job_id: Optional[str] = None


class ModelAPIAccessRequest(BaseModel):
    user_id: str
    dataset_name: str
    synthesis_training_job_id: Optional[str] = None
    forecast_training_job_id: Optional[str] = None
    task: str = "synthesis"


class ModelAPIAccessResponse(BaseModel):
    status: str
    formatted_python_code: str
    dataset_link: Optional[str] = None


# Pretrained Anomaly v2
class PreTrainedAnomalyV2Request(BaseModel):
    root: Dict[str, Dict[Union[str, int], Union[float, int, str, None]]] = (
        Field(
            description="Dictionary containing nested key-value pairs for time series data, where outer keys are column names and inner keys are row indices"
        )
    )
    num_anomalies_limit: int = Field(
        default=50, description="Number of top anomalies to detect"
    )
    min_anomaly_score: Optional[float] = Field(
        default=None, description="Minimum anomaly score to consider"
    )
    n_jobs: int = Field(
        default=-1, description="Number of parallel jobs to run"
    )


class PreTrainedAnomalyV2Response(BaseModel):
    results: Dict[str, Dict[str, Dict[str, List[AnomalyMetadata]]]] = Field(
        description="Dictionary containing anomaly metadata for each kpi: kpi_name -> anomaly_type -> group_key -> List[AnomalyMetadata]"
    )
    concurrent_results: Dict[str, Dict[str, Any]] = Field(
        description="Dictionary mapping timestamps to clusters of concurrent anomalies across multiple KPIs."
        "\n- concurrent anomalies are anomalies that occur across multiple KPIs within a specified time window."
        "\n- Each cluster contains:"
        "\n- timestamp: ISO format timestamp of the cluster"
        "\n- anomalies: List of concurrent anomalies with their metadata"
        "\n- distinct_kpis: Number of different KPIs involved"
        "\n- total_score: Sum of anomaly scores in the cluster"
        "\n- kpis_involved: List of KPI names involved in the concurrent anomalies"
    )


#################### End Pre Trained Anomaly v2 Data Models #####################


class RetrievePostprocessReportRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID requesting the postprocess report"
    )
    job_id: str = Field(..., description="Training job ID")


class RetrievePostprocessReportResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the request (success/error)"
    )
    message: str = Field(..., description="Success/error message")
    presigned_url: str | None = Field(
        ...,
        description="S3 presigned URL to download the postprocess report HTML",
    )


class RetrievePostPreProcessReportRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID requesting the postpreprocess report"
    )


class RetrievePostPreProcessReportResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the request (success/error)"
    )
    message: str = Field(..., description="Success/error message")
    presigned_urls: List[str] = Field(
        ...,
        description="S3 presigned URLs to download the postpreprocess report HTML",
    )


#################### Synthetic Data Agent Data Models #####################


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PerturbationsOrExactValues(str, Enum):
    PERTURBATIONS = "perturbations"
    EXACT_VALUES = "exact_values"


class PerturbationType(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


def validate_grid_or_perturbation_values(model, name: str) -> Any:
    """
    Validate that either all grid values or all perturbation values are provided,
    but not both or neither.

    Currently, we only support perturbations see flag - will support both in the future:
    ONLY_PERTURBATIONS = True
    """
    ONLY_PERTURBATIONS = True

    grid_values = [model.min_val, model.max_val, model.step]
    perturbation_values = [model.perturbation_type, model.perturbation_value]

    # Check if grid values are all present or all None
    grid_present = all(v is not None for v in grid_values)
    grid_absent = all(v is None for v in grid_values)

    # Check if perturbation values are all present or all None
    pert_present = all(v is not None for v in perturbation_values)
    pert_absent = all(v is None for v in perturbation_values)

    if ONLY_PERTURBATIONS:
        if not pert_present:
            raise ValueError(
                "Either provide all perturbation values (perturbation_type, perturbation_value), "
                "but not both or neither"
            )
    else:
        if not (
            (grid_present and pert_absent) or (grid_absent and pert_present)
        ):
            raise ValueError(
                "Either provide all grid values (min_val, max_val, step) "
                "or all perturbation values (perturbation_type, perturbation_value), "
                "but not both or neither"
            )

    # Special check for TimeStampVariation
    if name == "TimeStampVariation" and model.perturbation_type in [
        PerturbationType.MULTIPLY,
        PerturbationType.DIVIDE,
    ]:
        raise ValueError(
            "TimeStampVariation does not support 'multiply' or 'divide' perturbation types. "
            "Please use 'add' or 'subtract' only."
        )
    return model


class DiscreteVariation(BaseModel):
    name: str
    options: List[Union[str, int, float]]

    @field_validator("options")
    @classmethod
    def validate_options_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("options must contain at least one value")
        return v


class ContinuousVariation(BaseModel):
    """
    Only grid values or perturbation values can be provided, not both or neither
    """

    name: str

    # grid of values
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None

    # perturbations
    perturbation_type: Optional[PerturbationType] = None
    perturbation_value: Optional[Union[int, float]] = None

    @model_validator(mode="after")
    def validate_grid_or_perturbation(cls, model) -> Self:
        return validate_grid_or_perturbation_values(
            model, "ContinuousVariation"
        )


class TimeStampVariation(BaseModel):
    """
    Only grid values or perturbation values can be provided, not both or neither
    """

    name: str

    # grid of timestamps
    min_val: Optional[str] = None
    max_val: Optional[str] = None
    step: Optional[str] = None

    # perturbations
    perturbation_type: Optional[PerturbationType] = None
    perturbation_value: Optional[Union[int, float]] = None

    @model_validator(mode="after")
    def validate_grid_or_perturbation(cls, model) -> Self:
        return validate_grid_or_perturbation_values(model, "TimeStampVariation")


class MetaDataGridSample(BaseModel):
    group_labels_combinations: Dict[str, List[str]] | None
    metadata_range: MetaDataRange | None
    timestamps_range: TimeStampsRange | None


class MetaDataGrid(BaseModel):
    discrete_conditions_to_change: List[DiscreteVariation] | None
    continuous_conditions_to_change: List[ContinuousVariation] | None
    timestamps_conditions_to_change: List[TimeStampVariation] | None


class WindowFilters(BaseModel):
    # note this one is tricky if use_label_cols_as_discrete_metadata=False
    group_label_cols: Optional[OneDiscreteMetaDataRange] = None
    metadata_range: MetaDataRange
    timestamps_range: Optional[TimeStampsRange] = None
    # can also filter on timeseries values
    timeseries_range: List[OneContinuousMetaDataRange] = []


class GenerateCombinationsRequest(BaseModel):
    split_type: SplitType
    window_filters: WindowFilters
    metadata_grid: MetaDataGrid


class MetaDataVariation(BaseModel):
    """
    Unified modification that can be applied to full data or specific time ranges.

    Two types of modifications:
    1. 'full': applies modification to the whole data (no timestamps specified)
    2. 'range': applies to specified time range (has min_timestamp and/or max_timestamp)

    The type is automatically determined based on presence of timestamp fields:
    - If both min_timestamp and max_timestamp are None: 'full' modification
    - If one or both timestamps are present: 'range' modification
    """

    name: str
    value: Union[str, int, float]
    perturbation_or_exact_value: Literal["perturbation", "exact_value"]
    perturbation_type: Optional[PerturbationType] = None
    order: int = Field(
        default=0,
        description="Order of application (lower numbers applied first)",
    )

    # Range specification (optional)
    min_timestamp: Optional[str] = Field(
        default=None,
        description="Start timestamp for range modification (ISO format). If None, applies from beginning of data.",
    )
    max_timestamp: Optional[str] = Field(
        default=None,
        description="End timestamp for range modification (ISO format). If None, applies to end of data.",
    )

    @model_validator(mode="after")
    def validate_perturbation_or_exact_value(cls, model) -> Self:
        if (
            model.perturbation_or_exact_value == "perturbation"
            and model.perturbation_type is None
        ):
            raise ValueError(
                "perturbation_or_exact_value is 'perturbation' but perturbation_type is None"
            )
        if (
            model.perturbation_or_exact_value == "exact_value"
            and model.perturbation_type is not None
        ):
            raise ValueError(
                "perturbation_or_exact_value is 'exact_value' but perturbation_type is not None"
            )
        return model

    @field_validator("min_timestamp", "max_timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(
                f"Timestamp '{v}' is not in valid ISO format. "
                f"Supported formats: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS.ffffff "
                f"(with optional microseconds and timezone offset)"
            )

    @model_validator(mode="after")
    def validate_timestamp_range(cls, model) -> Self:
        if model.min_timestamp and model.max_timestamp:
            try:
                min_dt = datetime.fromisoformat(model.min_timestamp)
                max_dt = datetime.fromisoformat(model.max_timestamp)
                if min_dt >= max_dt:
                    raise ValueError(
                        f"min_timestamp ({model.min_timestamp}) must be before max_timestamp ({model.max_timestamp})"
                    )
            except ValueError as e:
                if "min_timestamp" in str(e) and "max_timestamp" in str(e):
                    raise e
                # Re-raise validation errors from datetime parsing
                raise ValueError(f"Invalid timestamp format: {e}")
        return model

    @property
    def modification_type(self) -> Literal["full", "range"]:
        """
        Determine if this is a 'full' or 'range' modification based on timestamp presence.
        """
        if self.min_timestamp is None and self.max_timestamp is None:
            return "full"
        else:
            return "range"


class GridCombinations(BaseModel):
    # each dict is the metadata:value
    combinations: List[List[MetaDataVariation]]
    num_windows_satisfying_conditions: int
    # this is num_windows_satisfying_conditions * num_combinations
    num_windows_to_generate: int


class SyntheticDataGenerationRequest(BaseModel):
    run_name: str
    previous_grid_combinations_output: GridCombinations
    synthesis_training_job_id: str
    split_type: SplitType
    window_filters: WindowFilters
    metadata_grid: MetaDataGrid
    window_start_idx: int
    window_inclusive_end_idx: int
    synthesis_constraints: Optional[SynthesisConstraints] = None


class SyntheticDataAgentRunInfo(BaseModel):
    run_name: str
    last_modified: str = Field(description="ISO format datetime string")


class ListSyntheticDataAgentRunsRequest(BaseModel):
    user_id: str = Field(..., description="User ID to fetch synthetic data for")
    dataset_name: str = Field(..., description="Name of the dataset")


class ListSyntheticDataAgentRunsResponse(BaseModel):
    synthetic_data_runs: List[SyntheticDataAgentRunInfo] = Field(
        ..., description="List of synthetic data runs with their metadata"
    )


class RetrieveSyntheticDataAgentRunRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID requesting the synthetic data"
    )
    dataset_name: str = Field(..., description="Name of the dataset")
    run_name: str = Field(..., description="Name of the synthetic data run")


class RetrieveSyntheticDataAgentRunResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the request (success/error)"
    )
    message: str = Field(..., description="Success/error message")
    presigned_url: str = Field(
        ..., description="S3 presigned URL to download the synthetic data"
    )


#################### End Synthetic Data Agent Data Models #####################


#################### Retrieve Training Config Model Data Models #####################
class RetrieveTrainingConfigModelRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID requesting the training config"
    )
    dataset_name: str = Field(..., description="Name of the dataset")
    training_job_id: str = Field(..., description="Training job ID")

    @field_validator("user_id", "dataset_name", "training_job_id")
    @classmethod
    def check_not_empty(cls, v: str, info) -> str:
        if not v:
            raise ValueError(f"{info.field_name} cannot be empty")
        return v


class RetrieveTrainingConfigModelResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the request (success/error)"
    )
    message: str = Field(..., description="Success/error message")
    presigned_url: Optional[str] = Field(
        None,
        description="S3 presigned URL to download the training config YAML",
    )


#################### End Retrieve Training Config Model Data Models #####################


#################### Retrieve Preprocessing Config Data Models #####################
class RetrievePreprocessingConfigRequest(BaseModel):
    user_id: str = Field(
        ..., description="User ID requesting the preprocessing config"
    )
    dataset_name: str = Field(..., description="Name of the dataset")

    @field_validator("user_id", "dataset_name")
    @classmethod
    def check_not_empty(cls, v: str, info) -> str:
        if not v:
            raise ValueError(f"{info.field_name} cannot be empty")
        return v


class RetrievePreprocessingConfigResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the request (success/error)"
    )
    message: str = Field(..., description="Success/error message")
    presigned_url: Optional[str] = Field(
        None,
        description="S3 presigned URL to download the preprocessing config JSON",
    )


#################### End Retrieve Preprocessing Config Data Models #####################


#################### Synthefy Agent Data Models #####################
class SynthefyAgentFilterRequest(BaseModel):
    user_id: str
    dataset_name: str
    window_filters: WindowFilters
    split: Literal["train", "val", "test"]
    n_windows: Optional[int] = 5

    # in case the model files are gone somehow, allow redownloading.
    synthesis_training_job_id: Optional[str] = None
    forecast_training_job_id: Optional[str] = None

    @field_validator("n_windows")
    @classmethod
    def validate_n_windows(cls, v: int) -> int:
        if v > 10:
            raise ValueError("n_windows cannot be greater than 10")
        return v


#################### End Synthefy Agent Data Models #####################


#################### Begin Foundation Data Models #####################


class StatusCode(int, Enum):
    ok = 200
    created = 201
    bad_request = 400
    not_found = 404
    unprocessable_entity = 422
    internal_server_error = 500


class CategoricalFeatureValues(BaseModel):
    """Model representing a categorical feature and its distinct values."""

    feature_name: str
    values: List[str | int | float]


class CategoricalFeaturesResponse(BaseModel):
    """Response model for categorical features endpoint."""

    status: StatusCode
    message: str
    categorical_features: List[CategoricalFeatureValues] = []
    file_path_key: str


class CovariateGridRequest(BaseModel):
    """Request model for covariate grid generation."""

    available_covariates: List[str] = Field(
        ..., description="List of available covariate column names"
    )
    max_combinations: int = Field(
        default=30,
        description="Maximum number of covariate combinations to return",
        ge=1,
        le=40,
    )

    @field_validator("available_covariates")
    @classmethod
    def validate_available_covariates(cls, v):
        if len(v) == 0:
            raise ValueError("available_covariates cannot be empty")
        return v


class CovariateGridResponse(BaseModel):
    """Response model for covariate grid generation."""

    status: StatusCode
    message: str
    covariate_grid: List[List[str]]
    total_combinations: int


class SupportedAggregationFunctions(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    MIN = "min"
    MAX = "max"


class GroupLabelColumnFilters(BaseModel):
    """
    Represents a set of filters used to select specific group-label combinations
    for column-based data queries.

    Each filter is a dictionary where the key is a group label (e.g. 'region', 'product_type'),
    and the value is a list of accepted string values for that label.

    Example:
        filters = GroupLabelColumnFilters(
            filter=[
                {"region": ["North", "South"]},
                {"product_type": ["Electronics", "Clothing"]}
            ]
        )

        # This means:
        # - Select data where 'region' is either 'North' or 'South'
        # - AND where 'product_type' is either 'Electronics' or 'Clothing'
    """

    filter: List[Dict[str, List[str | int | float]]]
    aggregation_func: SupportedAggregationFunctions = Field(
        default=SupportedAggregationFunctions.SUM,
        description="Aggregation function to use (default: 'sum')",
    )

    @field_validator("filter")
    @classmethod
    def validate_filter_not_empty(cls, v):
        # Allow empty filter list for cases where only aggregation_func is provided
        if not v:
            return v

        for item in v:
            if not item:
                raise ValueError("filter must not contain empty dictionaries")

        return v


class ReducedMetadata(BaseModel):
    description: str = Field(description="Description of the metadata")
    name: str = Field(description="Name of the metadata")
    dataType: str = Field(description="Data type of the metadata")
    startDate: str = Field(description="Start date of the metadata")
    databaseName: str = Field(description="Database name of the metadata")
    frequency: str = Field(description="Frequency of the data")


class HaverMetadataAccessInfo(BaseModel):
    db_path_info: Optional[str] = None
    data_source: Literal["haver"]
    description: str
    start_date: int = Field(description="Start date of the data")
    database_name: Optional[str] = Field(
        None, description="Name of the database"
    )
    name: Optional[str] = Field(None, description="Name of the series")
    file_name: Optional[str] = Field(None, description="Name of the file")

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if not (
            (self.file_name is not None)
            or (self.database_name is not None and self.name is not None)
        ):
            raise ValueError(
                "Must provide either file_name or both database_name and name"
            )
        return self


class TimePeriod(BaseModel):
    """Time period configuration for metadata access."""

    min_timestamp: str
    forecast_timestamp: str


class WeatherParameters(BaseModel):
    temperature: bool = False
    uv_index: bool = False
    wind_speed: bool = False
    wind_degree: bool = False
    windchill: bool = False
    windgust: bool = False
    precip: bool = False
    humidity: bool = False
    visibility: bool = False
    pressure: bool = False
    cloudcover: bool = False
    heatindex: bool = False
    dewpoint: bool = False
    feelslike: bool = False
    chanceofrain: bool = False
    chanceofremdry: bool = False
    chanceofwindy: bool = False
    chanceofovercast: bool = False
    chanceofsunshine: bool = False
    chanceoffrost: bool = False
    chanceofhightemp: bool = False
    chanceoffog: bool = False
    chanceofsnow: bool = False
    chanceofthunder: bool = False


class WeatherStackLocation(BaseModel):
    name: str
    latitude: float
    longitude: float
    country_code: Optional[str] = None
    admin1_code: Optional[str] = None
    population: Optional[int] = None


class PredictHQEventCategories(BaseModel):
    """Event categories configuration for PredictHQ API."""

    academic: bool = False
    concerts: bool = False
    conferences: bool = False
    expos: bool = False
    festivals: bool = False
    performing_arts: bool = False
    sports: bool = False
    community: bool = False
    daylight_savings: bool = False
    observances: bool = False
    politics: bool = False
    public_holidays: bool = False
    school_holidays: bool = False
    severe_weather: bool = False


class PredictHQAggregationConfig(BaseModel):
    """Configuration for how to aggregate event data into time series."""

    method: Literal["count", "impact_score", "category_indicators"] = "count"
    time_granularity: Literal["daily", "weekly", "monthly"] = "daily"
    min_phq_rank: Optional[int] = None
    include_attendance: bool = False
    include_categories: bool = False


class PredictHQLocationConfig(BaseModel):
    """Location configuration for PredictHQ API queries."""

    # Option 1: Lat/Lng with radius
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = None

    # Option 2: Place ID (compatible with existing location search)
    place_id: Optional[str] = None

    # Option 3: Country code
    country_code: Optional[str] = None

    # Display name for UI
    location_name: str = "Unknown Location"


class PredictHQMetadataAccessInfo(BaseModel):
    """Access information for PredictHQ event data."""

    data_source: Literal["predicthq"] = "predicthq"

    # Location configuration
    location_config: PredictHQLocationConfig

    # Time period configuration
    time_period: TimePeriod

    # Event filtering
    event_categories: PredictHQEventCategories

    # Aggregation configuration
    aggregation_config: PredictHQAggregationConfig

    # Display name
    name: str = "PredictHQ Events"

    # API configuration
    api_key: Optional[str] = (
        None  # Will use environment variable if not provided
    )


class WeatherMetadataAccessInfo(BaseModel):
    db_path_info: Optional[str] = None
    data_source: Literal["weather"]
    name: str
    description: str
    file_name: Optional[str] = None
    location_data: WeatherStackLocation
    weather_parameters: WeatherParameters
    units: Literal["m", "s", "f"]
    frequency: str
    time_period: TimePeriod
    aggregate_intervals: bool = Field(
        default=False,
        description="Only related to frequency larger than or equal todaily. "
        "If True, aggregate weather data over intervals e.g. monthly, yearly, etc. any freqency larger than daily will be aggregated. "
        "First we fetch daily data for the entire time period, then we aggregate the data over the intervals. "
        "When False, only the exact dates in target_timestamps will be fetched from the API, making it more efficient. "
        "target_timestamps are the timestamps from user's uploaded data. ",
    )

    @model_validator(mode="after")
    def validate_location_coordinates(self) -> Self:
        """Validate that latitude and longitude are valid for weather data processing."""
        lat = self.location_data.latitude
        lon = self.location_data.longitude

        # Check for coordinates out of valid range
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError(
                f"Invalid latitude ({lat}) or longitude ({lon}) for WeatherStack metadata. "
                f"Coordinates must be within valid ranges: latitude [-90, 90], longitude [-180, 180]."
            )
        return self


class FMDataPointModification(BaseModel):
    timestamp: str
    modification_dict: Dict[str, Union[int, float, str]]

    @field_validator("modification_dict")
    @classmethod
    def validate_modification_dict(cls, v: Dict[str, Union[int, float, str]]):
        if not v:
            raise ValueError("modification_dict must not be empty")
        return v


class FoundationModelConfig(BaseModel):
    """Configuration for file upload and processing"""

    file_path_key: str
    model_type: SynthefyTimeSeriesModelType | str = (
        SynthefyTimeSeriesModelType.DEFAULT
    )
    forecast_length: int
    timeseries_columns: List[str]
    covariate_columns: List[str] = Field(
        description="Columns that are used as covariates for the forecast",
        default_factory=list,
    )
    covariate_columns_to_leak: List[str] = Field(
        description="Columns that are used as covariates for the forecast",
        default_factory=list,
    )
    min_timestamp: str
    forecasting_timestamp: str
    timestamp_column: str
    metadata_info_combined: Optional[
        List[
            HaverMetadataAccessInfo
            | WeatherMetadataAccessInfo
            | PredictHQMetadataAccessInfo
        ]
    ] = None
    metadata_dataframes_leak_idxs: Optional[List[int]] = None

    # point modifications are singular timestampped modifications to the data
    point_modifications: Optional[List[FMDataPointModification]] = None
    # unified modifications that can be applied to full data or specific time ranges
    full_and_range_modifications: Optional[List[MetaDataVariation]] = None

    do_llm_explanation: bool = Field(default=False)
    do_shap_analysis: bool = Field(default=False)

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    @field_validator("min_timestamp", "forecasting_timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(
                f"Timestamp '{v}' is not in valid ISO format. "
                f"Supported formats: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS.ffffff "
                f"(with optional microseconds and timezone offset)"
            )


class MetadataDataFrame(BaseModel):
    df: pd.DataFrame
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    timestamp_key: Optional[str] = None
    feature_names: Optional[List[str]] = None

    model_config = {"arbitrary_types_allowed": True}


class SynthefyDatabaseMetadataSearchRequest(BaseModel):
    data_source: Literal["haver"]
    regex: Optional[str] = None
    date_range: Optional[str] = None

    # This allows complex searching through pd.query() or SQL query when the data is in a DB.
    # Note: we then need a separate API to provide the columns/headers so we know what to query upon.
    query: Optional[str] = None

    # Can add more as we move on.


class HaverDatasetMatch(BaseModel):
    # includes minimal set of info to pull the data
    # Ex: haver: access_info = {"databaseName": "US", "name": "USGDP", "description": "..", "start_date": 90}
    access_info: HaverMetadataAccessInfo

    # includes the path to the leaf dataset where the match is found
    # Ex: haver/US/GDP/
    # This db_path_info can then be used to search any of the directories it contains.
    db_path_info: Optional[str] = None


class SynthefyDatabaseMetadataSearchResponse(BaseModel):
    # this should include a reference to the data in the tree structure
    # in case we want to then search for similar datasets to the one retrieved from search.
    matches: List[HaverDatasetMatch] = Field(
        default_factory=list,
        description="List of matches from the search",
    )


class SynthefyDatabaseDirectorySearchRequest(BaseModel):
    # If None - just show the available folders at: synthefy-foundation-model-metadata-datasets/univariate/
    # Otherwise, we will show the files and folders available at the current directory
    directory_to_search: Optional[str] = None


class SynthefyDatabaseDirectorySearchResponse(BaseModel):
    # Shows the directories that we can further traverse
    directories: List[str] = Field(
        default_factory=list,
        description="List of directories that we can further traverse",
    )
    # Shows the datasets that exist at the current level.
    datasets: List[HaverDatasetMatch] = Field(
        default_factory=list,
        description="List of datasets that exist at the current level",
    )


class UploadResponse(BaseModel):
    original_file_key: str | None
    status: StatusCode
    dataset_columns: List[str]
    timestamp_columns: List[str] = Field(
        default_factory=list,
        description="Columns that resemble timestamps in the dataset",
    )
    time_frequency: Optional[TimeFrequency] = Field(
        default=None,
        description="Detected time frequency of the data (e.g. TimeFrequency(value=1, unit='day'), TimeFrequency(value=15, unit='minute'))",
    )
    message: Optional[str]


class ApiForecastRequest(BaseModel):
    """Request model for forecast generation that includes user identification"""

    user_id: Optional[str] = None
    config: FoundationModelConfig


class ConfidenceInterval(BaseModel):
    """Model for confidence interval bounds"""

    lower: float
    upper: float


class ForecastGroup(BaseModel):
    """Model for a group of forecasts with their identifiers"""

    target_column: str
    forecasts: List[float]  # Forecast values
    confidence_intervals: List[
        ConfidenceInterval
    ]  # Confidence intervals for each forecast
    univariate_forecasts: List[float]
    univariate_confidence_intervals: List[
        ConfidenceInterval
    ]  # Confidence intervals for eaWch univariate forecast
    ground_truth: List[float | None] = Field(
        default_factory=list,
        description="Ground truth values for the forecast period, None for timestamps beyond available data",
    )
    full_targets: List[float] = Field(
        default_factory=list,
        description="Full target values for the forecast period",
    )
    explanation: Optional[str] = None
    shap_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="SHAP analysis results including feature importance and SHAP values",
    )


class HistoricalDataset(BaseModel):
    """Model for historical time series data"""

    timestamps: List[str] = Field(
        description="Historical timestamps in string format"
    )
    values: Dict[str, List[float]] = Field(
        description="Dictionary mapping target column names to their historical values"
    )
    target_columns: List[str] = Field(
        description="List of target column names in the historical dataset"
    )


class ForecastDataset(BaseModel):
    """Model for the forecast dataset"""

    timestamps: List[str]  # Forecast timestamps
    values: List[ForecastGroup]  # Groups of forecast values


class ApiForecastResponse(BaseModel):
    """Response model for forecast generation"""

    status: StatusCode
    message: str
    dataset: ForecastDataset | None = None
    historical_dataset: HistoricalDataset | None = None
    warnings: List[str] | None = Field(
        default=None,
        description="List of warning messages for the client",
    )


class BacktestInfo(BaseModel):
    """Information about a backtest result."""

    s3_key: str = Field(..., description="S3 key of the backtest result file")
    download_url: str = Field(
        ..., description="Presigned URL for downloading the backtest result"
    )
    execution_datetime: str = Field(
        ..., description="ISO format datetime when the backtest was executed"
    )
    dataset_name: str = Field(
        ..., description="Name of the dataset used for the backtest"
    )
    file_size_bytes: Optional[int] = Field(
        default=None, description="Size of the backtest file in bytes"
    )


class ListBacktestsRequest(BaseModel):
    """Request model for listing available backtests."""

    user_id: str = Field(..., description="User ID to filter backtests")
    dataset_name: Optional[str] = Field(
        default=None, description="Optional dataset name to filter backtests"
    )
    # Pagination fields
    page: int = Field(default=1, description="Page number (1-based)", ge=1)
    page_size: int = Field(
        default=20, description="Number of backtests per page", ge=1, le=100
    )
    sort_order: str = Field(
        default="desc",
        description="Sort order for execution datetime: 'asc' or 'desc'",
        pattern="^(asc|desc)$",
    )


class ListBacktestsResponse(BaseModel):
    """Response model for listing available backtests."""

    status: StatusCode
    message: str
    backtests: List[BacktestInfo] = Field(
        default_factory=list, description="List of available backtests"
    )
    # Pagination metadata
    pagination: "PaginationInfo" = Field(
        ..., description="Pagination information"
    )


class PaginationInfo(BaseModel):
    """Pagination information for list responses."""

    current_page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(
        ..., description="Whether there are previous pages"
    )


# Add this after the ListBacktestsResponse class to resolve forward reference
ListBacktestsResponse.model_rebuild()


class BacktestAPIForecastRequest(BaseModel):
    user_id: Optional[str] = None
    # Note - the forecasting_timestamp in config will be treated as the last forecasting timestamp
    # Note - the min_timestamp in config will be treated start of the first window (test_X)
    #        and the first forecast_timestamp will be at (min_timestamp + window_size)
    config: FoundationModelConfig

    group_filters: GroupLabelColumnFilters

    # stride is the step size for the rolling windows.
    # This should be an isoformat string that can be converted to datetime.
    stride: Optional[str] = None

    # Grid search over different covariate combinations with leaking options
    # e.g., [{}, {"Holiday_Flag": false}, {"Temperature": false}, {"Holiday_Flag": false, "CPI": false}, {"Holiday_Flag": true, "Temperature": false, "Fuel_Price": true}]
    # Keys are covariate column names, values are booleans indicating whether to leak them into the future
    # If None, uses config.covariate_columns as single combination
    covariate_grid: Optional[List[Dict[str, bool]]] = None

    # TODO add validation for stride
    @field_validator("stride")
    def validate_stride(cls, v):
        if v is not None:
            try:
                # Parse the ISO 8601 duration string
                parse_duration(v)
            except ValueError:
                raise ValueError(
                    "Stride must be in ISO 8601 duration format (e.g., 'P28D' for 28 days)"
                )
        return v

    @field_validator("covariate_grid")
    def validate_covariate_grid(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("covariate_grid cannot be empty if provided")
        return v


class BacktestAPIForecastResponse(BaseModel):
    status: StatusCode
    message: str
    presigned_url: str | None
    task_id: str | None
    warnings: List[str] | None = Field(
        default=None,
        description="List of warning messages for the client",
    )


class FoundationModelForecastStreamRequest(BaseModel):
    # --------------- User uploaded data ---------------
    # TODO future - can be None if no timestamps; isoformat string from pd.timestamps
    historical_timestamps: List[str]

    # from df.to_dict(orient='list')
    historical_timeseries_data: Dict[str, List[Any]]
    targets: List[str]  # must be present as keys in historical_timeseries_data
    # must be present in historical_timeseries_data if provided
    covariates: List[str] = Field(default_factory=list)

    model_type: SynthefyTimeSeriesModelType = Field(
        default=SynthefyTimeSeriesModelType.DEFAULT,
        description="Model type to use for forecasting and synthesis.",
    )

    # --------------- End user uploaded data ---------------

    # --------------- Synthefy Database context ---------------
    synthefy_metadata_info_combined: (
        List[
            HaverMetadataAccessInfo
            | WeatherMetadataAccessInfo
            | PredictHQMetadataAccessInfo
        ]
        | None
    ) = None
    # Must be a subset of synthefy_metadata_list; these will be leaked into the future_df
    synthefy_metadata_leak_idxs: Optional[List[int]] = None
    # --------------- End Synthefy Database context ---------------

    # --------------- Data for Forecasting ---------------
    # the timestamps for which we want to predict the targets' values
    forecast_timestamps: List[str]
    # from df.to_dict(orient='list'); future metadata that will be used
    future_timeseries_data: Dict[str, List[Any]] | None = None
    # --------------- End Data for Forecasting ---------------

    # Dict used to add constant context (will be same for each timestamp/repeated for the dfs)
    static_context: Dict[str, float | int | str] | None = None
    prompt: str | None = None  # Prompt/description of the task/data/etc

    quantiles: List[float] | None = None  # which quantiles to return

    do_llm_explanation: bool = Field(default=False)
    do_shap_analysis: bool = Field(default=False)

    model_config = {"protected_namespaces": ()}

    @field_validator("historical_timestamps", "forecast_timestamps")
    def validate_timestamps_individual(cls, v, info):
        if v is not None:
            try:
                timestamps = pd.to_datetime(v)
            except ValueError:
                raise ValueError(
                    f"{info.field_name} must be parseable as datetime"
                )
            if len(v) != len(set(v)):
                raise ValueError(
                    f"{info.field_name} must not contain duplicate timestamps"
                )
            if not timestamps.is_monotonic_increasing:
                raise ValueError(
                    f"{info.field_name} must be monotonically increasing"
                )
        return v

    @field_validator("quantiles")
    def validate_quantiles(cls, v):
        allowed_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if v is not None:
            if len(v) != 2:
                raise ValueError("Quantiles must be a list of 2 floats")
            if not all(0 <= q <= 1 for q in v):
                raise ValueError("All quantiles must be between 0 and 1")
            if len(v) != len(set(v)):
                raise ValueError("Quantiles must not contain duplicates")
            if not all(q1 < q2 for q1, q2 in zip(v, v[1:])):
                raise ValueError("Quantiles must be in ascending order")
            if not all(q in allowed_quantiles for q in v):
                raise ValueError(
                    "Quantiles must be one of the following: "
                    + str(allowed_quantiles)
                )
        return v

    @field_validator("model_type")
    def validate_model_type(cls, v):
        allowed_model_types = {
            SynthefyTimeSeriesModelType.DEFAULT,
            SynthefyTimeSeriesModelType.FOUNDATION_MODEL_FORECASTING,
        }
        if v not in allowed_model_types:
            msg = (
                "Invalid model_type, please reach out to synthefy support team."
            )
            raise ValueError(msg)
        return v

    @property
    def historical_df(self) -> pd.DataFrame:
        """Convert historical_timeseries_data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the historical timeseries data with timestamps as index

        Raises:
            ValueError: If timestamps length doesn't match data length or if DataFrame conversion fails
        """
        try:
            # Create DataFrame from the dictionary
            df = pd.DataFrame(self.historical_timeseries_data)

            # Validate lengths match
            if len(self.historical_timestamps) != len(df):
                raise ValueError(
                    f"Length of historical_timestamps ({len(self.historical_timestamps)}) "
                    f"must match length of historical_timeseries_data ({len(df)})"
                )

            # Set timestamps as index
            df["timestamp"] = pd.to_datetime(self.historical_timestamps).values

            return df

        except Exception as e:
            raise ValueError(
                f"Failed to convert historical data to DataFrame: {str(e)}"
            )

    @property
    def future_df(self) -> pd.DataFrame:
        """Convert future_timeseries_data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the future timeseries data with timestamps as index

        Raises:
            ValueError: If timestamps length doesn't match data length or if DataFrame conversion fails
        """
        try:
            # Create DataFrame from the dictionary
            df = pd.DataFrame(self.future_timeseries_data)

            if len(df) > len(self.forecast_timestamps):
                raise ValueError(
                    f"Length of future_timeseries_data ({len(df)}) "
                    f"must be less than or equal to length of forecast_timestamps ({len(self.forecast_timestamps)})"
                )

            # only set the timestamps for the future data since the timestamps can be larger than the amount of data
            df["timestamp"] = pd.to_datetime(self.forecast_timestamps).values[
                : len(df)
            ]
            return df

        except Exception as e:
            raise ValueError(
                f"Failed to convert historical data to DataFrame: {str(e)}"
            )


class FoundationModelForecastStreamResponse(BaseModel):
    # includes columns: timestamp_col (whatever the timestamp_col was in the request),
    # column: "forecast" includes the forecasts
    # column "forecast_{quantile}" includes the quantiles
    forecast_timestamps: List[str] | None  # isoformat string from pd.timestamps

    # includes only the targets and their values.
    forecast: Dict[str, List[Union[int, float]]]
    # keys are {target_col}_{quantile}
    forecast_quantiles: Dict[str, List[Union[int, float]]]
    shap_analysis: Dict[str, Dict[str, Any]] | None = None
    warnings: List[str] | None = Field(
        default=None,
        description="List of warning messages for the client",
    )


class FoundationModelChatRequest(BaseModel):
    user_id: Optional[str] = None
    thread_id: str
    prompt: str
    # list the full columns from the raw df in case the user has filtered already.
    dataset_columns: List[str]
    timestamp_columns: List[str] = Field(
        default_factory=list,
        description="Columns that resemble timestamps in the dataset",
    )
    categorical_features: List[CategoricalFeatureValues] = Field(
        default_factory=list,
        description="Categorical features and their possible values from the dataset",
    )
    number_of_recommended_metadata_datasets: Optional[int] = Field(
        None, description="Number of recommended metadata datasets"
    )
    config: FoundationModelConfig


class UIKnobs(BaseModel):
    target_columns: Optional[List[str]] = Field(
        None, description="Timeseries target columns"
    )
    timestamp_column: Optional[str] = Field(
        None, description="Timestamp column"
    )
    covariates: Optional[List[str]] = Field(
        None, description="Covariate columns"
    )
    min_timestamp: Optional[str] = Field(
        None, description="Minimum timestamp (ISO format)"
    )
    forecast_timestamp: Optional[str] = Field(
        None, description="Forecast timestamp (last timestamp for history)"
    )
    group_filters: Optional[GroupLabelColumnFilters] = Field(
        None, description="Group filters or filter columns"
    )
    forecast_horizon: Optional[int] = Field(
        None, description="Forecast horizon (with units, e.g., '30d')"
    )
    # backtest_filters: Optional[Dict[str, Any]] = Field(
    #     None, description="Filters for backtesting"
    # )
    backtest_stride: Optional[str] = Field(
        None, description="Stride for backtesting (ISOformat string)"
    )
    leak_metadata: Optional[bool] = Field(
        None, description="Whether to leak metadata into the future context."
    )


#################### End Foundation Data Models #####################


#################### Begin Explain Data Models #####################


class LLMExplanationRequest(BaseModel):
    userid: str
    timeseries_to_summarize: List[OneTimeSeries]
    filters_applied: Optional[Dict[str, List[str]]] = None
    other_timeseries: Optional[List[OneTimeSeries]] = None
    timestamps: Optional[TimeStamps] = None
    text: Optional[str] = None


class LLMExplanationResponse(BaseModel):
    explanation: str
    status: str


#################### End Explain Data Models #####################


#################### Begin Location Search Data Models #####################


class LocationSearchRequest(BaseModel):
    """Request model for location search."""

    search: str = Field(
        ..., description="Search string to filter locations by name"
    )


#################### End Location Search Data Models #####################


#################### Begin Metadata Visualization Data Models #####################


class MetadataVisualizationRequest(BaseModel):
    """Request model for metadata dataset visualization."""

    metadata_access_info: (
        HaverMetadataAccessInfo
        | WeatherMetadataAccessInfo
        | PredictHQMetadataAccessInfo
    ) = Field(..., discriminator="data_source")

    target_timestamps: List[str] = Field(
        default=None,
        description="List of exact timestamps in ISO format (e.g., ['2023-01-01T14:30:00']) to fetch data for. "
        "This is only applicable for WeatherStack metadata access info.",
    )


class MetadataVisualizationItem(BaseModel):
    """Individual metadata dataset visualization item."""

    id: str  # This is where a unique ID for each dataset would go
    display_name: str
    download_url: str | None = Field(
        default=None,
        description="Pre-signed URL for downloading the dataset in parquet format",
    )
    json_download_url: str | None = Field(
        default=None,
        description="Pre-signed URL for downloading the dataset in JSON format",
    )


class MetadataVisualizationResponse(BaseModel):
    """Response model for metadata dataset visualization."""

    status: StatusCode
    message: str
    datasets: List[MetadataVisualizationItem] = Field(
        default_factory=list,
        description="List of processed datasets with their URLs and display names",
    )


#################### End Metadata Visualization Data Models #####################


class SynthefyFoundationModelTypeMetadata(BaseModel):
    """Metadata for a specific foundation model type (e.g., forecasting, synthesis)."""

    model_config = {"protected_namespaces": ()}

    model_version: Optional[str] = Field(
        default=None, description="Version of the model (e.g., 'v3e')"
    )
    date: Optional[str] = Field(
        default=None,
        description="Date when the model was created or updated (ISO format)",
    )
    available: bool = Field(
        default=False,
        description="Whether the model is currently available for use",
    )
    local_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Local path to the model checkpoint file",
    )
    local_config_path: Optional[str] = Field(
        default=None,
        description="Local path to the model config file",
    )


class SynthefyFoundationModelMetadata(BaseModel):
    """Complete metadata for all foundation model types."""

    s3_url: str = Field(
        default=None,
        description="S3 URL for the foundation model metadata",
    )
    forecasting: SynthefyFoundationModelTypeMetadata = Field(
        default_factory=lambda: SynthefyFoundationModelTypeMetadata(),
        description="Metadata for forecasting models",
    )
    synthesis: SynthefyFoundationModelTypeMetadata = Field(
        default_factory=lambda: SynthefyFoundationModelTypeMetadata(),
        description="Metadata for synthesis models",
    )

    @field_validator("forecasting", "synthesis")
    @classmethod
    def validate_model_metadata(
        cls, v: SynthefyFoundationModelTypeMetadata
    ) -> SynthefyFoundationModelTypeMetadata:
        """Validate that model metadata is properly formatted."""
        if v.date is not None:
            try:
                datetime.fromisoformat(v.date)
            except ValueError:
                raise ValueError(f"Date '{v.date}' is not in valid ISO format")
        return v

    @property
    def available_models(self) -> List[str]:
        """Get the available models."""
        available = []
        if self.forecasting.available:
            available.append("forecasting")
        if self.synthesis.available:
            available.append("synthesis")
        return available

def get_model_card_as_dict(
    model_metadata: SynthefyFoundationModelMetadata, model_name: str
) -> Dict[str, str | datetime | bool]:
    """Convert the SynthefyFoundationModelMetadata to a dictionary.

    Cannot be a method because of Pydantic validation.

    Returns:
        Dict[str, Any]: Dictionary representation of the metadata
    """
    return {
        "model_type": f"synthefy-{model_name}",
        "model_version": getattr(model_metadata, model_name).model_version,
        "date": getattr(model_metadata, model_name).date,
        "available": getattr(model_metadata, model_name).available,
        "local_checkpoint_path": getattr(
            model_metadata, model_name
        ).local_checkpoint_path,
        "local_config_path": getattr(model_metadata, model_name).local_config_path,
        "s3_url": model_metadata.s3_url,
    }
