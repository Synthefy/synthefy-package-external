import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def get_col_type_timestamp_or_continuous(
    col_type: str, v1: bool, is_metadata: Optional[str]
):
    if col_type != "continuous":
        return col_type

    if not v1:
        return "timeseries"
    else:
        if is_metadata == "yes":
            return "continuous"
        else:
            return "timeseries"


def get_columns_info_dict_from_metadata_json(
    metadata: Dict[str, Any], v1: bool = True
) -> Dict[str, Any]:
    """Extract columns info dict from metadata, organizing columns by their types.

    This function processes the metadata dictionary and creates a nested structure
    organizing columns by their types. Timestamp columns are excluded from the output.
    For each column, it extracts the description and time range information if available.

    Args:
        metadata: Dictionary of metadata for this data source, expected to have a
                 "columns" key containing a list of column information dictionaries.
                 Each column dictionary should have:
                 - "type": str (column type, e.g., "continuous", "text")
                 - "column_id": str (name of the column)
                 - "description": str (description of the column)
                 - "start_time": Optional[pd.Timestamp] (start of valid data range)
                 - "end_time": Optional[pd.Timestamp] (end of valid data range)

    Returns:
        Dictionary organized by column types, with format:
        {
            "column_type1": {
                "column_name1": {
                    "description": str,
                    "start_time": Optional[pd.Timestamp],
                    "end_time": Optional[pd.Timestamp]
                },
                "column_name2": {...}
            },
            "column_type2": {...}
        }
        Note: "timestamp" type columns are excluded from the output.

    Example:
        >>> metadata = {
        ...     "columns": [
        ...         {
        ...             "type": "continuous",
        ...             "column_id": "temperature",
        ...             "description": "Temperature in Celsius",
        ...             "start_time": pd.Timestamp("2023-01-01"),
        ...             "end_time": pd.Timestamp("2023-12-31")
        ...         }
        ...     ]
        ... }
        >>> result = get_v2_columns_info_dict_from_metadata_json(metadata)
        >>> print(result)
        {
            "continuous": {
                "temperature": {
                    "description": "Temperature in Celsius",
                    "start_time": Timestamp("2023-01-01 00:00:00"),
                    "end_time": Timestamp("2023-12-31 00:00:00")
                }
            }
        }
    """
    info_dict = {}
    for column_info_dict in metadata["columns"]:
        col_type = get_col_type_timestamp_or_continuous(
            column_info_dict["type"],
            v1,
            column_info_dict.get("is_metadata", None),
        )
        if col_type == "timestamp":
            continue

        if col_type not in info_dict:
            info_dict[col_type] = {}

        # Convert timestamps to pandas datetime if they exist
        start_time = pd.to_datetime(
            column_info_dict.get("start_time", None), utc=True
        )
        end_time = pd.to_datetime(
            column_info_dict.get("end_time", None), utc=True
        )

        info_dict[col_type][column_info_dict["column_id"]] = {
            # TODO: for now keep using title, but later switch to description
            "description": column_info_dict.get("title", ""),
            "start_time": start_time,
            "end_time": end_time,
        }

    return info_dict


def load_dataframe_from_directory(
    data_dir: str,
) -> pd.DataFrame:
    """Load dataframe from a directory.

    Loads data from a parquet file in the specified directory.

    Args:
        data_dir: Path to the directory containing data

    Returns:
        pandas.DataFrame containing the loaded data

    Raises:
        ValueError: If no parquet file or multiple parquet files are found in the directory.
    """
    # Load data from parquet file
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if len(parquet_files) != 1:
        raise ValueError(
            f"Expected 1 parquet file in {data_dir}, found {len(parquet_files)} parquet files"
        )

    parquet_path = parquet_files[0]
    df = pd.read_parquet(parquet_path)
    # logger.info(f"Loaded {len(df)} rows from {parquet_path}")

    return df


def load_metadata_from_directory(
    data_dir: str,
) -> Dict[str, Any]:
    """Load metadata from a directory.

    Loads metadata from a JSON file in the specified directory.

    Args:
        data_dir: Path to the directory containing metadata

    Returns:
        Dictionary containing the loaded metadata

    Raises:
        ValueError: If no JSON file or multiple JSON files are found in the directory.
    """
    metadata_files = glob.glob(os.path.join(data_dir, "*.json"))
    if len(metadata_files) != 1:
        raise ValueError(
            f"Expected 1 metadata file in {data_dir}, found {len(metadata_files)} metadata files"
        )

    metadata = {}
    metadata_path = metadata_files[0]
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    # logger.info(f"Loaded metadata from {metadata_path}")

    return metadata


def load_data_from_directory(
    data_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data from a directory.

    Loads data from a parquet file and reads metadata if available.

    Args:
        data_dir: Path to the directory containing data

    Returns:
        tuple: (pandas.DataFrame, dict) containing the loaded DataFrame and metadata
    """
    df = load_dataframe_from_directory(data_dir)
    metadata = load_metadata_from_directory(data_dir)

    return df, metadata
