"""
Dataset Processing Pipeline

This module implements a robust data processing pipeline designed to efficiently
handle time series data from structured parquet files. The pipeline takes configuration
files that define dataset parameters and transforms the data into standardized
formats optimized for time series analysis and machine learning.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from synthefy_pkg.utils.llm_utils import process_dataset_description

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""

    def default(self, obj):
        # Convert numpy scalars to Python standard types
        if isinstance(obj, np.number):
            return obj.item()
        # Convert numpy arrays to lists
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Convert numpy booleans to Python booleans
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Return None for numpy voids
        elif isinstance(obj, np.void):
            return None
        # Let the parent class handle everything else
        return super(NumpyEncoder, self).default(obj)


class ConfigParser:
    """Validates and processes JSON configuration files"""

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the configuration parser.

        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict:
        """Load the configuration file"""
        logger.info(f"Loading configuration from {self.config_path}")
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_config(self) -> None:
        """Validate the configuration schema"""
        required_fields = ["dataset_name", "filename"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(
                    f"Required field '{field}' missing in configuration"
                )

        # Validate timestamp column exists
        if "timestamps_col" not in self.config:
            raise ValueError("No timestamp column specified in configuration")

        # Convert timestamps_col to list if it's a string
        if isinstance(self.config["timestamps_col"], str):
            self.config["timestamps_col"] = [self.config["timestamps_col"]]

        # Validate group labels
        if "group_labels" not in self.config:
            logger.warning("No group labels specified. Using single group.")
            self.config["group_labels"] = []
        # Convert group_labels from old format if needed
        elif (
            isinstance(self.config["group_labels"], dict)
            and "cols" in self.config["group_labels"]
        ):
            self.config["group_labels"] = self.config["group_labels"]["cols"]

        # Validate timeseries columns
        if "timeseries" not in self.config:
            raise ValueError("No timeseries columns specified")
        # Convert timeseries from old format if needed
        elif (
            isinstance(self.config["timeseries"], dict)
            and "cols" in self.config["timeseries"]
        ):
            self.config["timeseries"] = self.config["timeseries"]["cols"]

        # Ensure other fields have defaults if missing
        if "continuous" not in self.config:
            self.config["continuous"] = []
        # Convert continuous from old format if needed
        elif (
            isinstance(self.config["continuous"], dict)
            and "cols" in self.config["continuous"]
        ):
            self.config["continuous"] = self.config["continuous"]["cols"]

        if "tags" not in self.config:
            self.config["tags"] = []

        if "domain" not in self.config:
            self.config["domain"] = ""

        if "description" not in self.config:
            self.config["description"] = ""

    def get_config(self) -> Dict:
        """Get the validated configuration"""
        return self.config


class DataLoader:
    """Imports and validates parquet files"""

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the data loader.

        Args:
            file_path: Path to the parquet file
        """
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        """Load data from parquet file"""
        logger.info(f"Loading data from {self.file_path}")
        try:
            df = pd.read_parquet(self.file_path)
            logger.info(f"Loaded data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def validate_columns(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> None:
        """Validate that required columns exist in the dataframe"""
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")


class DataTransformer:
    """Processes and restructures data according to configuration"""

    def __init__(self, config: Dict):
        """
        Initialize the data transformer.

        Args:
            config: Validated configuration dictionary
        """
        self.config = config
        self.timestamp_columns = self.config["timestamps_col"]
        self.group_label_columns = self.config["group_labels"]
        self.timeseries_columns = self.config["timeseries"]
        self.continuous_columns = self.config["continuous"]

    def transform_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Transform data according to configuration.

        Args:
            df: Input dataframe

        Returns:
            Dictionary of transformed dataframes by group
        """
        logger.info("Transforming data")

        # Validate required columns
        required_columns = (
            self.timestamp_columns
            + self.group_label_columns
            + self.timeseries_columns
            + self.continuous_columns
        )

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")

        # Group data by group labels if any
        if self.group_label_columns:
            grouped_data = {}
            for group_name, group_df in df.groupby(self.group_label_columns):
                # Handle single column or multiple column grouping
                if not isinstance(group_name, tuple):
                    group_name = (group_name,)

                # Create a string identifier for this group
                group_id = "_".join(str(value) for value in group_name)

                # Store the grouped dataframe
                grouped_data[group_id] = group_df

            logger.info(
                f"Created {len(grouped_data)} groups based on {self.group_label_columns}"
            )
            return grouped_data
        else:
            # If no grouping, return the whole dataframe as a single group
            return {"0": df}


class MetadataGenerator:
    """Creates detailed metadata files for each dataset"""

    def __init__(self, config: Dict):
        """
        Initialize the metadata generator.

        Args:
            config: Validated configuration dictionary
        """
        self.config = config

    def generate_metadata(
        self,
        df: pd.DataFrame,
        dataset_id: int,
        group_id: str,
        processed_description: Optional[str] = None,
        is_metadata: bool = False,
    ) -> Dict:
        """
        Generate metadata for a dataframe.

        Args:
            df: Input dataframe
            dataset_id: Dataset identifier
            group_id: Group identifier
            is_metadata: Whether this is metadata or target data

        Returns:
            Metadata dictionary
        """
        dataset_name = self.config["dataset_name"]
        if is_metadata:
            dataset_name = f"{dataset_name}_metadata"

        # Determine frequency if timestamp columns exist
        frequency = "unknown"
        timezone = "UTC"
        timezone_guessed = True

        # Extract basic stats
        timestamp_col = self.config["timestamps_col"][
            0
        ]  # Use first timestamp column

        try:
            start_date = df[timestamp_col].min().isoformat()
            end_date = df[timestamp_col].max().isoformat()

            # Try to infer frequency
            if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                sorted_timestamps = df[timestamp_col].sort_values()
                diffs = sorted_timestamps.diff().dropna()
                if not diffs.empty:
                    # Get the most common difference
                    most_common_diff = diffs.value_counts().idxmax()
                    # Convert to human-readable frequency
                    if most_common_diff == pd.Timedelta(seconds=1):
                        frequency = "1S"  # 1 second
                    elif most_common_diff == pd.Timedelta(minutes=1):
                        frequency = "1T"  # 1 minute
                    elif most_common_diff == pd.Timedelta(hours=1):
                        frequency = "1H"  # 1 hour
                    elif most_common_diff == pd.Timedelta(days=1):
                        frequency = "1D"  # 1 day
                    else:
                        frequency = str(most_common_diff)
        except Exception as e:
            logger.warning(f"Could not extract timestamp info: {e}")
            start_date = "unknown"
            end_date = "unknown"

        # Generate column metadata
        columns_metadata = []
        for col in df.columns:
            if col in self.config["timestamps_col"]:
                continue
            elif col in self.config["timeseries"]:
                column_type = "timeseries"
            elif col in self.config["continuous"]:
                column_type = "continuous"
            else:
                raise ValueError(
                    f"Column {col} is not a timestamp, timeseries, or continuous column"
                )

            if processed_description is None:
                description_value = (
                    f"{group_id}:{col}" if column_type == "timeseries" else col
                )
            else:
                description_value = processed_description

            column_meta = {
                "id": (
                    f"{group_id}:{col}" if column_type == "timeseries" else col
                ),
                "title": (
                    f"{group_id}:{col}" if column_type == "timeseries" else col
                ),
                "column_id": col,
                "description": description_value,
                "is_metadata": "yes" if is_metadata else "no",
                "type": column_type,
                "units": "",
            }
            columns_metadata.append(column_meta)

        # Construct metadata
        metadata = {
            "title": dataset_name,
            "id": f"{dataset_name}_{dataset_id}",
            "timestamp_columns": self.config["timestamps_col"],
            "frequency": frequency,
            "tags": self.config.get("tags", []),
            "domain": self.config.get("domain", ""),
            "timezone": timezone,
            "timezone_guessed": timezone_guessed,
            "target_column": (
                self.config["timeseries"][0]
                if self.config["timeseries"]
                else ""
            ),
            "start_date": start_date,
            "end_date": end_date,
            "length": len(df),
            "size": df.memory_usage(deep=True).sum(),
            "num_columns": len(df.columns),
            "num_continuous_columns": len(self.config["continuous"]),
            "num_discrete_columns": 0,  # Not implemented in Phase 1
            "num_text_columns": 0,  # Not implemented in Phase 1
            "num_metadata_columns": (
                len(self.config["continuous"]) if is_metadata else 0
            ),
            "columns": columns_metadata,
        }

        return metadata


class OutputManager:
    """Handles file writing and directory structure creation"""

    def __init__(self, base_path: Union[str, Path], config: Dict):
        """
        Initialize the output manager.

        Args:
            base_path: Base path where output will be written
            config: Validated configuration dictionary
        """
        self.base_path = Path(base_path)
        self.config = config
        self.dataset_name = config["dataset_name"]
        self.folder_counter = (
            0  # Add a counter for consecutive folder numbering
        )

    def create_directory_structure(self, is_metadata: bool = False) -> None:
        """
        Create the directory structure.

        Args:
            is_metadata: Whether to create metadata or target directories
        """
        dataset_path = self.base_path / self.dataset_name
        if is_metadata:
            dataset_path = self.base_path / f"{self.dataset_name}_metadata"

        if not dataset_path.exists():
            logger.info(f"Creating directory {dataset_path}")
            dataset_path.mkdir(parents=True, exist_ok=True)

    def save_data(
        self,
        group_id: str,
        df: pd.DataFrame,
        metadata: Dict,
        is_metadata: bool = False,
    ) -> None:
        """
        Save data and metadata for a group.

        Args:
            group_id: Group identifier
            df: Dataframe to save
            metadata: Metadata to save
            is_metadata: Whether this is metadata or target data
        """
        # Determine paths
        dataset_dir = self.dataset_name
        if is_metadata:
            dataset_dir = f"{self.dataset_name}_metadata"

        group_path = self.base_path / dataset_dir / f"{dataset_dir}_{group_id}"
        group_path.mkdir(parents=True, exist_ok=True)

        # Save parquet file
        parquet_path = group_path / f"{dataset_dir}_{group_id}.parquet"
        logger.info(f"Saving parquet to {parquet_path}")
        df.to_parquet(parquet_path, index=False)

        # Save metadata
        metadata_path = group_path / f"{dataset_dir}_{group_id}_metadata.json"
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, cls=NumpyEncoder)


class Pipeline:
    """Main pipeline controller"""

    def __init__(
        self, config_path: Union[str, Path], base_path: Union[str, Path]
    ):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to the configuration file
            base_path: Base path for output
        """
        self.config_parser = ConfigParser(config_path)
        self.config = self.config_parser.get_config()
        self.base_path = Path(base_path)

        # Initialize components
        self.data_loader = DataLoader(self.config["filename"])
        self.data_transformer = DataTransformer(self.config)
        self.metadata_generator = MetadataGenerator(self.config)
        self.output_manager = OutputManager(base_path, self.config)

    def run(self) -> None:
        """Run the pipeline"""
        logger.info(
            f"Starting pipeline for dataset {self.config['dataset_name']}"
        )

        # Load data
        df = self.data_loader.load_data()

        # Ensure the base output directories exist
        self.output_manager.create_directory_structure(is_metadata=False)
        self.output_manager.create_directory_structure(is_metadata=True)

        # Process and transform the data
        grouped_data = self.data_transformer.transform_data(df)

        # Process target columns
        target_id = 0
        metadata_id = 0

        dataset_description = self.config.get("description", None)
        use_llm = self.config.get("use_llm", False)

        # Process dataset description with LLM if enabled
        processed_description = None
        if dataset_description and use_llm:
            logger.info(
                "Using LLM to extract structured information from dataset description"
            )
            extracted_description_details = process_dataset_description(
                dataset_description
            )
            processed_description = extracted_description_details[
                "processed_description"
            ]

        for group_id, group_df in grouped_data.items():
            # Create target dataframe with timestamp and target columns
            for ts_col in self.config["timeseries"]:
                target_pair_df = group_df[
                    [self.config["timestamps_col"][0], ts_col]
                ]
                # Generate metadata for target data
                target_metadata = self.metadata_generator.generate_metadata(
                    target_pair_df,
                    target_id,
                    group_id,
                    processed_description=processed_description,
                    is_metadata=False,
                )

                # Save target data and metadata
                self.output_manager.save_data(
                    f"{target_id}",
                    target_pair_df,
                    target_metadata,
                    is_metadata=False,
                )
                target_id += 1

            # Process metadata columns if any
            if self.config["continuous"]:
                for cont_col in self.config["continuous"]:
                    metadata_pair_df = group_df[
                        [self.config["timestamps_col"][0], cont_col]
                    ]

                    # Generate metadata for metadata columns
                    metadata_metadata = (
                        self.metadata_generator.generate_metadata(
                            metadata_pair_df,
                            metadata_id,
                            group_id,
                            processed_description=processed_description,
                            is_metadata=True,
                        )
                    )

                    # Save metadata data and metadata
                    self.output_manager.save_data(
                        f"{metadata_id}",
                        metadata_pair_df,
                        metadata_metadata,
                        is_metadata=True,
                    )
                    metadata_id += 1

        logger.info(
            f"Pipeline completed for dataset {self.config['dataset_name']}"
        )


def process_dataset(
    config_path: Union[str, Path], base_path: Union[str, Path]
) -> None:
    """
    Process a dataset according to the specified configuration.

    Args:
        config_path: Path to the configuration file
        base_path: Base path for output
    """
    try:
        pipeline = Pipeline(config_path, base_path)
        pipeline.run()
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process dataset based on configuration file"
    )
    parser.add_argument("config_path", help="Path to configuration file")
    parser.add_argument("base_path", help="Base path for output")

    args = parser.parse_args()

    process_dataset(args.config_path, args.base_path)
