"""
Pydantic data models for the Oura Demo API.

Simplified models for:
- Config loading and validation
- File upload and data parsing
- LLM modification requests (with code execution)
- Synthesis inference
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Enums
# =============================================================================


class DatasetName(str, Enum):
    """Supported dataset names for config loading."""

    OURA = "oura"
    OURA_SUBSET = "oura_subset"
    PPG = "ppg"


class FileType(str, Enum):
    """Supported file types for upload."""

    PARQUET = "parquet"
    CSV = "csv"


class ModelType(str, Enum):
    """Model architecture type for synthesis.

    - STANDARD: Original patched diffusion transformer
    - FLEXIBLE: Flexible patched diffusion transformer (better metadata handling)
    """

    STANDARD = "standard"
    FLEXIBLE = "flexible"


class TaskType(str, Enum):
    """Task type for the synthesis experiment.

    - SYNTHESIS: Generate synthetic data from scratch
    - FORECAST: Generate forecast continuation of input data
    """

    SYNTHESIS = "synthesis"
    FORECAST = "forecast"


# =============================================================================
# DataFrame-like Data Structure
# =============================================================================


class DataFrameModel(BaseModel):
    """DataFrame-like structure: column-oriented dict with lists.

    Example:
        {
            "average_hrv": [50.0, 52.0, 48.0, ...],
            "lowest_heart_rate": [55, 54, 56, ...],
            "gender_male": [1, 1, 1, ...],
        }
    """

    columns: Dict[str, List[Union[float, int, str, None]]] = Field(
        description="Column name to list of values mapping"
    )

    @field_validator("columns")
    @classmethod
    def validate_equal_lengths(cls, v: Dict[str, List]) -> Dict[str, List]:
        """Ensure all columns have the same length."""
        if not v:
            return v
        lengths = {col: len(vals) for col, vals in v.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"All columns must have equal length. Got: {lengths}"
            )
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.columns)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "DataFrameModel":
        """Create from pandas DataFrame.

        Converts NaN/Inf values to None for JSON serialization.
        """
        columns_dict = {}
        for col in df.columns:
            values = df[col].tolist()
            # Convert NaN and Inf to None for JSON serialization
            clean_values = []
            for v in values:
                if v is None:
                    clean_values.append(None)
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean_values.append(None)
                else:
                    clean_values.append(v)
            columns_dict[col] = clean_values

        return cls(columns=columns_dict)

    @property
    def num_rows(self) -> int:
        """Number of rows in the data."""
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    @property
    def column_names(self) -> List[str]:
        """List of column names."""
        return list(self.columns.keys())


# =============================================================================
# Config Models
# =============================================================================


class RequiredColumns(BaseModel):
    """Required columns extracted from preprocessing config."""

    timeseries: List[str] = Field(
        default_factory=list,
        description="Time series columns to be synthesized",
    )
    discrete: List[str] = Field(
        default_factory=list,
        description="Discrete metadata columns",
    )
    continuous: List[str] = Field(
        default_factory=list,
        description="Continuous metadata columns",
    )
    group_labels: List[str] = Field(
        default_factory=list,
        description="Group label columns",
    )


class ConfigResponse(BaseModel):
    """Response model for GET /api/config/{dataset_name}."""

    dataset_name: DatasetName
    required_columns: RequiredColumns
    window_size: int = Field(gt=0, description="Window size for synthesis")
    num_channels: int = Field(
        gt=0, description="Number of time series channels"
    )
    available_datasets: List[str] = Field(
        default_factory=lambda: [d.value for d in DatasetName],
        description="List of all available dataset names",
    )


# =============================================================================
# Validation Models
# =============================================================================


class ColumnValidationResult(BaseModel):
    """Result of validating uploaded data columns against config."""

    valid: bool = Field(description="Whether validation passed")
    missing_columns: List[str] = Field(
        default_factory=list,
        description="Required columns missing from uploaded data",
    )
    extra_columns: List[str] = Field(
        default_factory=list,
        description="Extra columns not in config",
    )


# =============================================================================
# Upload Models
# =============================================================================


class UploadResponse(BaseModel):
    """Response model for POST /api/upload/{dataset_name}."""

    data: DataFrameModel = Field(description="Uploaded data")
    window_size: int = Field(
        gt=0, description="Expected window size from config"
    )
    validation: ColumnValidationResult = Field(
        description="Column validation result"
    )
    file_type: FileType = Field(description="Detected file type")


# =============================================================================
# LLM Modification Models
# =============================================================================


class LLMModifyRequest(BaseModel):
    """Request model for POST /api/llm/modify.

    Stateless: includes dataset_name for column context.
    LLM will execute code to modify the data directly.
    """

    dataset_name: DatasetName = Field(
        description="Dataset name for column context (timeseries vs metadata)"
    )
    data: DataFrameModel = Field(description="Current data to modify")
    user_query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural language modification request",
    )

    @field_validator("user_query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_query cannot be empty")
        return v.strip()


class LLMModifyResponse(BaseModel):
    """Response model for POST /api/llm/modify."""

    modified_data: DataFrameModel = Field(description="Modified data")
    code_executed: str = Field(
        description="Python code that was executed to modify data"
    )
    explanation: str = Field(description="LLM explanation of modifications")


# =============================================================================
# Synthesis Models
# =============================================================================


class OneTimeSeries(BaseModel):
    """A single time series with name and values."""

    name: str = Field(description="Name of the time series column")
    values: List[Optional[float]] = Field(description="Time series values")


class SynthesisRequest(BaseModel):
    """Request model for POST /api/synthesize.

    Stateless: includes dataset_name to load correct configs.
    Input data is always exactly 1 window.
    """

    dataset_name: DatasetName = Field(
        description="Dataset name to load synthesis/preprocessing configs"
    )
    data: DataFrameModel = Field(description="Input data (1 window)")
    model_type: ModelType = Field(
        default=ModelType.FLEXIBLE,
        description="Model architecture: 'standard' or 'flexible' (default: flexible)",
    )
    task_type: TaskType = Field(
        default=TaskType.SYNTHESIS,
        description="Task type: 'synthesis' or 'forecast' (default: synthesis)",
    )
    num_samples: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Number of synthesis runs to average (1-100, default: 2)",
    )
    ground_truth_prefix_length: int = Field(
        default=0,
        ge=0,
        description="For synthesis task: number of initial points to keep as ground truth (0 = disabled)",
    )
    forecast_length: int = Field(
        default=96,
        ge=1,
        description="For forecast task: number of time steps to forecast (default: 96)",
    )


class SynthesisResponse(BaseModel):
    """Response model for POST /api/synthesize."""

    original_timeseries: List[OneTimeSeries] = Field(
        description="Original time series input"
    )
    synthetic_timeseries: List[OneTimeSeries] = Field(
        description="Synthesized time series output"
    )
    window_size: int = Field(gt=0, description="Window size")
    num_channels: int = Field(gt=0, description="Number of channels")
    dataset_name: DatasetName = Field(description="Dataset used")
    model_type: ModelType = Field(description="Model architecture used")
    task_type: TaskType = Field(description="Task type used")
    forecast_horizon: int = Field(
        default=96,
        gt=0,
        description="Number of time steps in the forecast horizon",
    )
