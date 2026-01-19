import asyncio
import json
import os
import pickle
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from fastapi import Depends, Header, HTTPException, UploadFile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger
from pydantic import BaseModel
from scipy import stats
from sqlalchemy.orm import Session

from synthefy_pkg.app.dao.user_api_keys import validate_api_key
from synthefy_pkg.app.data_models import (
    ConstraintType,
    DynamicTimeSeriesData,
    ForecastResponse,
    MetaData,
    MetaDataVariation,
    OneConstraint,
    OneContinuousMetaData,
    OneContinuousMetaDataRange,
    OneDiscreteMetaData,
    OneDiscreteMetaDataRange,
    OneTimeSeries,
    PerturbationType,
    PostPreProcessRequest,
    PostPreProcessResponse,
    PostprocessRequest,
    PostprocessResponse,
    PreTrainedAnomalyResponse,
    PreTrainedAnomalyV2Request,
    PreTrainedAnomalyV2Response,
    ProjectionType,
    SearchRequest,
    SearchResponse,
    SelectedAction,
    SelectedWindows,
    SynthefyRequest,
    SynthefyResponse,
    SynthefyTimeSeriesWindow,
    SynthesisConstraints,
    SynthesisResponse,
    TimeFrequency,
    TimeStamps,
    TimeStampsRange,
    WindowFilters,
    WindowsSelectionOptions,
)
from synthefy_pkg.app.db import get_db
from synthefy_pkg.app.utils.llm_utils import (
    MetaDataToParse,
    extract_metadata_from_query,
    llm,
)
from synthefy_pkg.utils.config_utils import load_yaml_config

COMPILE = True

# Time frequency detection constants
MICROSECOND_PRECISION = 6  # Decimal places for microsecond precision in seconds


def save_request(
    request: Union[
        SynthefyRequest,
        DynamicTimeSeriesData,
        PreTrainedAnomalyV2Request,
        PostprocessRequest,
        PostPreProcessRequest,
    ],
    endpoint: str,
    json_save_path: str = "/tmp",
):
    # Ensure the directory exists
    os.makedirs(json_save_path, exist_ok=True)
    with open(
        os.path.join(json_save_path, f"{endpoint}_request.json"), "w"
    ) as f:
        f.write(request.model_dump_json())


def save_response(
    response: Union[
        SynthefyResponse,
        PreTrainedAnomalyV2Response,
        DynamicTimeSeriesData,
        PostprocessResponse,
        PostPreProcessResponse,
    ],
    endpoint: str,
    json_save_path: str = "/tmp",
) -> None:
    # Ensure the directory exists
    os.makedirs(json_save_path, exist_ok=True)
    with open(
        os.path.join(json_save_path, f"{endpoint}_response.json"), "w"
    ) as f:
        f.write(response.model_dump_json())


def extract_col_names(
    request: SynthefyRequest,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    timestamps_colname = (
        [request.windows[0].timestamps.name]
        if request.windows[0].timestamps is not None
        else ["index"]
    )
    timeseries_list = request.windows[0].timeseries_data
    timeseries_colnames = [timeseries.name for timeseries in timeseries_list]
    metadata = request.windows[0].metadata
    discrete_colnames = [
        condition.name for condition in metadata.discrete_conditions
    ]
    continuous_colnames = [
        condition.name for condition in metadata.continuous_conditions
    ]
    logger.info(
        f"Extracted colnames: {timeseries_colnames=}, {continuous_colnames=}, {discrete_colnames=}, {timestamps_colname=}"
    )
    return (
        timeseries_colnames,
        continuous_colnames,
        discrete_colnames,
        timestamps_colname,
    )


def update_metadata_from_query(request: SynthefyRequest) -> SynthefyRequest:
    """
    Update the metadata in the request with the metadata from the query.
    If the text prompt has (M) less indices than the original metadata (N), we fill in the first N-M indices with the text prompt values.

    """
    if request.text is None or len(request.text) == 0:
        return request
    (
        timeseries_colnames,
        continuous_colnames,
        discrete_colnames,
        timestamps_colname,
    ) = extract_col_names(request)
    parsed_metadata: MetaDataToParse = extract_metadata_from_query(
        request.text,
        timeseries_colnames,
        continuous_colnames,
        discrete_colnames,
        timestamps_colname,
    )

    request.n_forecast_windows = parsed_metadata.num_examples  # both for now
    request.n_synthesis_windows = parsed_metadata.num_examples  # both for now

    for window_idx in request.selected_windows.window_indices:
        for parsed_condition in parsed_metadata.discrete_conditions:
            cond_idx = discrete_colnames.index(parsed_condition.name)
            original_values = (
                request.windows[window_idx]
                .metadata.discrete_conditions[cond_idx]
                .values
            )
            if parsed_condition.name.startswith("Label-Tuple-"):
                if len(parsed_condition.values) != 1:
                    logger.error(
                        "Label-Tuple can only have 1 value - setting all values to the first value"
                    )
                parsed_condition.values = [parsed_condition.values[0]] * len(
                    original_values
                )  # can only update with the same values.

            else:
                if len(parsed_condition.values) != len(original_values):
                    if len(parsed_condition.values) < len(original_values):
                        parsed_condition.values.extend(
                            original_values[len(parsed_condition.values) :]
                        )
                    else:
                        parsed_condition.values = parsed_condition.values[
                            : len(original_values)
                        ]
            request.windows[window_idx].metadata.discrete_conditions[
                cond_idx
            ] = parsed_condition
            logger.info(
                f"Updated {parsed_condition.name} with {parsed_condition.values}"
            )
        for parsed_condition in parsed_metadata.continuous_conditions:
            cond_idx = continuous_colnames.index(parsed_condition.name)
            original_values = (
                request.windows[window_idx]
                .metadata.continuous_conditions[cond_idx]
                .values
            )
            if len(parsed_condition.values) != len(original_values):
                if len(parsed_condition.values) < len(original_values):
                    parsed_condition.values.extend(
                        original_values[len(parsed_condition.values) :]
                    )
                else:
                    parsed_condition.values = parsed_condition.values[
                        : len(original_values)
                    ]
            request.windows[window_idx].metadata.continuous_conditions[
                cond_idx
            ] = parsed_condition
            logger.info(
                f"Updated {parsed_condition.name} with {parsed_condition.values}"
            )

    # TODO - add timestamps, right now we don't support it

    return request


async def generate_ts_summary(
    query: str | None, org_ts: List, out_ts: List
) -> str:
    if not query:
        return ""
    prompt_template = """
    I have an original time series: {org_timeseries}.
    The user provided the following request: "{user_prompt}".
    My model then generated the following time series: {out_timeseries}.

    Can you analyze the changes in the output time series compared to the original one?
    Please explain these changes in the context of the user's request to help them better understand the generated
    output. The output should be in markdown format and it should directly answers the user's query.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke(
        {
            "user_prompt": query,
            "org_timeseries": org_ts,
            "out_timeseries": out_ts,
        }
    )
    return response


async def generate_combined_summary(user_prompt: str, summaries: List) -> str:
    if not user_prompt:
        return ""
    prompt_template = """
    Below is a collection of summaries describing the changes in the time series.
    Your task is to generate a **comprehensive and well-structured summary** that captures
    the overall transformation of the time series in **Markdown format**.

    **Summaries:**
    {window_summaries}

    Please generate a **concise, coherent, and insightful** summary that highlights the key trends,
    patterns, and changes observed. The summary should clearly convey how the time series evolved
    in response to the user's query: **"{user_prompt}"**.

    Structure the summary using the following sections:

    ---
    ## Key Insights
    *[Provide a brief 1-2 sentence summary capturing the most significant transformation in the time series.
    This should immediately highlight the core observation.]*

    ## Detailed Analysis
    *[Expand on the key insights by explaining the key trends, fluctuations, and changes observed in the time series.
    Ensure the explanation is logically structured and directly addresses the user's query.]*

    ## Key Trends and Patterns
    *[Highlight specific trends, patterns, or anomalies observed in the data.
    Use Markdown formatting (`##`, `###`) for clear organization where needed.]*
    ---

    Ensure that the summary is **structured, easy to follow, and free of redundant information**.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke(
        {"window_summaries": summaries, "user_prompt": user_prompt}
    )
    return response


def convert_discrete_metadata_range_to_label_tuple_range(
    discrete_metadata_range: List[OneDiscreteMetaDataRange],
    group_labels_combinations: Dict[str, Any],
    group_label_cols: List[str],
) -> List[OneDiscreteMetaDataRange]:
    """
    discrete_metadata: list of OneDiscreteMetaDataRange
    group_labels_combinations: dict of group labels combinations
    group_label_cols: list of group label columns

    return: list of OneDiscreteMetaDataRange with label tuples

    Removes the original discrete metadata range that is a part of the group_label_cols
    Adds a new discrete metadata range with the group labels combined so the tuple
    will always be together.
    """
    if not group_label_cols or not group_labels_combinations:
        return discrete_metadata_range

    if any("-" in col for col in group_label_cols):
        raise ValueError("Group-label cols cannot include '-' in the name.")

    metadata_dict = {
        metadata.name: metadata.options for metadata in discrete_metadata_range
    }
    # Ensure we have all required columns
    if not all(col in metadata_dict for col in group_label_cols):
        missing_cols = set(group_label_cols) - set(metadata_dict.keys())
        raise ValueError(f"Missing metadata for columns: {missing_cols}")

    # Remove the discrete metadata range that is part of the group_label_cols
    discrete_metadata_range = [
        metadata
        for metadata in discrete_metadata_range
        if metadata.name not in group_label_cols
    ]

    # add the concatenated group labels as discrete metadata range
    # insert it at 0 location
    group_name = list(group_labels_combinations.keys())[0]
    allowed_combinations = group_labels_combinations[group_name]
    group_labels_range = OneDiscreteMetaDataRange(
        name=f"Label-Tuple-{group_name}", options=allowed_combinations
    )
    discrete_metadata_range.insert(0, group_labels_range)

    return discrete_metadata_range


def convert_discrete_metadata_to_label_tuple(
    discrete_metadata: List[OneDiscreteMetaData],
    group_labels_combinations: Dict[str, Any],
    group_label_cols: List[str],
) -> List[OneDiscreteMetaData]:
    """
    discrete_metadata: list of OneDiscreteMetaData
    group_labels_combinations: dict of group labels combinations
    group_label_cols: list of group label columns

    return: list of OneDiscreteMetaData with label tuples

    Removes the original discrete metadata that is a part of the group_label_cols
    Adds a new discrete metadata with the group labels combined so the tuple
    will always be together.
    """
    if not group_label_cols or not group_labels_combinations:
        return discrete_metadata

    if any("-" in col for col in group_label_cols):
        raise ValueError("Group-label cols cannot include '-' in the name.")

    metadata_dict = {
        metadata.name: metadata.values for metadata in discrete_metadata
    }
    # Ensure we have all required columns
    if not all(col in metadata_dict for col in group_label_cols):
        missing_cols = set(group_label_cols) - set(metadata_dict.keys())
        if missing_cols == set(
            list(group_labels_combinations.keys())[0].split("-")
        ):
            # no need to raise error since use_discrete_metadata_as_label_tuple is false. may be dangerous.
            return discrete_metadata
        else:
            raise ValueError(f"Missing metadata for columns: {missing_cols}")

    # Get the number of values (all grouped columns have the same length)
    num_values = len(metadata_dict[group_label_cols[0]])  # window size

    # Combine values in the order specified by group_label_cols
    group_values = [
        "-".join(str(metadata_dict[col][i]) for col in group_label_cols)
        for i in range(num_values)
    ]

    # Validate that all combined values are in the allowed combinations
    group_name = list(group_labels_combinations.keys())[0]
    allowed_combinations = group_labels_combinations[group_name]
    invalid_combinations = set(group_values) - set(allowed_combinations)
    if invalid_combinations:
        raise ValueError(f"Invalid combinations found: {invalid_combinations}")

    new_discrete_metadata = OneDiscreteMetaData(
        name=f"Label-Tuple-{group_name}", values=list(group_values)
    )

    # Remove the discrete metadata that is part of the group_label_cols
    discrete_metadata = [
        metadata
        for metadata in discrete_metadata
        if metadata.name not in group_label_cols
    ]

    # add the concatenated group labels as discrete metadata
    # insert it at 0 location
    discrete_metadata.insert(0, new_discrete_metadata)

    return discrete_metadata


def convert_label_tuple_to_discrete_metadata(
    request: SynthefyRequest,
) -> SynthefyRequest:
    """
    input: SynthefyRequest
    output: SynthefyRequest
    description: Converts the label tuple to discrete metadata.
    This is called after each request in each service.
    """
    for window in request.windows:
        if window.metadata.discrete_conditions:
            # find the index that started with "Label-Tuple-"
            for idx, condition in enumerate(
                window.metadata.discrete_conditions
            ):
                if condition.name.startswith("Label-Tuple-"):
                    group_label_cols = condition.name.split("-")[2:]
                    group_labels_values = {col: [] for col in group_label_cols}
                    if len(condition.values) == 0:
                        raise ValueError("Group label values cannot be empty.")
                    for value in condition.values:
                        split_value = str(value).split("-")
                        if len(split_value) != len(group_label_cols):
                            if 0:
                                # OPANGA ONLY TEMP FIX
                                split_value = [split_value[0]] + [
                                    "-".join(split_value[1:])
                                ]
                            else:
                                raise ValueError(
                                    "Group label cols and values must be the same length."
                                )
                        for i, col in enumerate(group_label_cols):
                            group_labels_values[col].append(split_value[i])

                    metadata_to_add = [
                        OneDiscreteMetaData(
                            name=col,
                            values=group_labels_values[col],
                        )
                        for col in group_label_cols
                    ]

                    window.metadata.discrete_conditions.pop(idx)
                    if len(window.metadata.discrete_conditions) == 0:
                        window.metadata.discrete_conditions = metadata_to_add
                        # Replace extendleft with insert operations
                    else:
                        for i, meta in enumerate(reversed(metadata_to_add)):
                            window.metadata.discrete_conditions.insert(0, meta)
                    break  # can only have 1 label tuple

    return request


def trim_response_to_forecast_window(
    response: DynamicTimeSeriesData, window_size: int, forecast_length: int
) -> DynamicTimeSeriesData:
    """Trim response to only include the forecast window by removing historical data.

    Args:
        response: The response object containing the forecast data
        window_size: Size of the original window
        forecast_length: Length of the forecast period

    Returns:
        dict: A new response object containing only the forecast window data
    """
    if forecast_length <= 0:
        return response

    trimmed_response = response.model_copy(deep=True)

    for k, v in trimmed_response.root.items():
        values_list = list(v.values())  # pyright: ignore
        start_idx = window_size - forecast_length
        # Create new dict with sequential indices starting from 0
        trimmed_response.root[k] = {
            i: values_list[i + start_idx]
            for i in range(len(values_list) - start_idx)
        }

    return trimmed_response


def convert_synthefy_response_to_dynamic_time_series_data(
    synthefy_response: SynthefyResponse,
    return_only_synthetic: bool = False,
    suffix_label: str = "_synthetic",
) -> DynamicTimeSeriesData:
    """
    Convert a SynthefyResponse to DynamicTimeSeriesData.

    Args:
        synthefy_response: Response containing windows with timeseries, metadata, and timestamps
        return_only_synthetic: If True, only return the synthetic timeseries data
        suffix_label: For forecasting, suffix is `_forecast`, default is `_synthetic`.

    Returns:
        DynamicTimeSeriesData with all data combined into a single dataframe-like structure.
        A window_idx column is added only if there are multiple windows.

    Raises:
        ValueError: If the response contains windows with inconsistent structures
    """
    if not synthefy_response.windows:
        raise ValueError("Response contains no windows")

    # Initialize data dictionary
    combined_data = {}
    first_window = synthefy_response.windows[0]

    window_size = len(first_window.timeseries_data[0].values)

    # Get all expected column names from the first window
    expected_keys = set(
        [ts.name for ts in first_window.timeseries_data]
        + [first_window.timestamps.name]  # pyright: ignore
        + [c.name for c in first_window.metadata.discrete_conditions]
        + [c.name for c in first_window.metadata.continuous_conditions]
    )
    # Add window_idx if multiple windows
    if len(synthefy_response.windows) > 1:
        expected_keys.add("window_idx")
        combined_data["window_idx"] = {}

    # Initialize all columns
    for key in expected_keys:
        if key != "window_idx":  # window_idx already initialized if needed
            if (
                not return_only_synthetic
                or key.endswith(suffix_label)
                or key == first_window.timestamps.name  # pyright: ignore
            ):
                combined_data[key] = {}

    if len(synthefy_response.anomaly_timestamps) > 0:
        combined_data["is_anomaly"] = {}

    # Process all windows
    total_len = 0
    for window_idx, window in enumerate(synthefy_response.windows):
        # Validate window structure
        if window.timestamps is None:
            raise ValueError("Window contains no timestamps")
        current_keys = set(
            [ts.name for ts in window.timeseries_data]
            + [window.timestamps.name]
            + [c.name for c in window.metadata.discrete_conditions]
            + [c.name for c in window.metadata.continuous_conditions]
        )
        if current_keys != expected_keys - {"window_idx"}:
            raise ValueError("Inconsistent data structure across windows")

        if len(window.timestamps.values) != window_size:
            raise ValueError("Window size mismatch")

        # Add data from this window
        if len(synthefy_response.windows) > 1:
            combined_data["window_idx"].update(
                dict(
                    zip(
                        range(total_len, total_len + window_size),
                        [window_idx] * window_size,
                    )
                )
            )

        # always add the timestamp data
        combined_data[window.timestamps.name].update(
            dict(
                zip(
                    range(total_len, total_len + window_size),
                    window.timestamps.values,
                )
            )
        )
        if return_only_synthetic:
            for ts in window.timeseries_data:
                if ts.name.endswith(suffix_label):
                    combined_data[ts.name].update(
                        dict(
                            zip(
                                range(total_len, total_len + window_size),
                                ts.values,
                            )
                        )
                    )
            # add timestamp data too

        else:
            for ts in window.timeseries_data:
                combined_data[ts.name].update(
                    dict(
                        zip(
                            range(total_len, total_len + window_size), ts.values
                        )
                    )
                )

            for condition in window.metadata.discrete_conditions:
                combined_data[condition.name].update(
                    dict(
                        zip(
                            range(total_len, total_len + window_size),
                            condition.values,
                        )
                    )
                )

            for condition in window.metadata.continuous_conditions:
                combined_data[condition.name].update(
                    dict(
                        zip(
                            range(total_len, total_len + window_size),
                            condition.values,
                        )
                    )
                )

            if len(synthefy_response.anomaly_timestamps) > 0:
                # Add true if the anomaly time stamp is same as timestamp column otherwise add false
                anomaly_timestamps = synthefy_response.anomaly_timestamps[
                    window_idx
                ].values
                is_anomaly = [0] * len(combined_data[window.timestamps.name])

                is_anomaly = [
                    1 if ts in anomaly_timestamps else is_anomaly[i]
                    for i, ts in enumerate(window.timestamps.values)
                ]

                combined_data["is_anomaly"].update(
                    dict(
                        zip(
                            range(total_len, total_len + window_size),
                            is_anomaly,
                        )
                    )
                )
        total_len += window_size

    return DynamicTimeSeriesData(root=combined_data)


def convert_dynamic_time_series_data_to_synthefy_request(
    dynamic_time_series_data: DynamicTimeSeriesData,
    group_label_cols: List[str],
    timeseries_colnames: List[str],
    continuous_colnames: List[str],
    discrete_colnames: List[str],
    timestamps_colname: List[str],
    window_size: int,
    selected_action: SelectedAction,
) -> SynthefyRequest:
    """
    Convert DynamicTimeSeriesData to SynthefyRequest. Input data should already be windowed.

    Args:
        dynamic_time_series_data: Input data containing all columns
        timeseries_colnames: Names of timeseries columns
        continuous_colnames: Names of continuous metadata columns
        discrete_colnames: Names of discrete metadata columns
        timestamps_colname: Name of the timestamp column
        window_size: Expected size of the window

    Returns:
        SynthefyRequest with the data

    Raises:
        ValueError: If input validation fails or data is inconsistent
        KeyError: If required columns are missing from the data
    """
    if window_size < 1:
        raise ValueError("Window size must be positive")

    data = dynamic_time_series_data.root

    # Validate all columns exist in data
    all_required_cols = (
        set(
            continuous_colnames
            + discrete_colnames
            + timestamps_colname
            + group_label_cols
        )
        if selected_action == SelectedAction.SYNTHESIS
        else set(
            timeseries_colnames
            + continuous_colnames
            + discrete_colnames
            + timestamps_colname
            + group_label_cols
        )
    )
    timestamps_colname_str = (
        timestamps_colname[0]
        if len(timestamps_colname) == 1 and isinstance(timestamps_colname, list)
        else None
    )

    # Create column type mapping
    if (
        timestamps_colname_str != []
        and timestamps_colname_str is not None
        and timestamps_colname_str != ""
    ):
        column_types = {
            **{col: "timeseries" for col in timeseries_colnames},
            **{col: "continuous metadata" for col in continuous_colnames},
            **{col: "discrete metadata" for col in discrete_colnames},
            **{col: "group label" for col in group_label_cols},
            **{timestamps_colname_str: "timestamp"},
        }
    else:
        column_types = {
            **{col: "timeseries" for col in timeseries_colnames},
            **{col: "continuous metadata" for col in continuous_colnames},
            **{col: "discrete metadata" for col in discrete_colnames},
            **{col: "group label" for col in group_label_cols},
        }

    # Check for missing columns
    missing_cols = all_required_cols - set(data.keys())
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate no null values in any columns
    null_error_message = ""
    for col in data.keys():
        # Skip validation for the constraints field
        if col == "_constraints_":
            continue
        if any(pd.isna(val) for val in data[col].values()):  # pyright: ignore
            col_type = column_types.get(col, "unknown")
            null_error_message += f"Column '{col}' ({col_type}) contains null values, which is not allowed.\n"
    if null_error_message != "":
        logger.error(null_error_message)
        raise ValueError(null_error_message)

    # Continue with existing validation...
    inconsistent_cols = [
        col
        for col, values in data.items()
        if col != "_constraints_" and len(values) != window_size
    ]
    if inconsistent_cols:
        raise ValueError(
            f"Inconsistent data lengths found in columns: {inconsistent_cols}. "
            f"All columns must have {window_size} rows."
        )

    # Validate data types
    try:
        # Check timeseries columns can be converted to float
        for col in timeseries_colnames:
            if col in data:
                _ = [float(x) for x in data[col].values()]  # pyright: ignore

        # Check continuous columns can be converted to float
        for col in continuous_colnames:
            _ = [float(x) for x in data[col].values()]  # pyright: ignore

        # Check discrete columns can be converted to string
        for col in discrete_colnames:
            _ = [str(x) for x in data[col].values()]  # pyright: ignore

    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Data type conversion failed: {str(e)}. "
            "Timeseries and continuous columns must be numeric, "
            "discrete columns must be convertible to string."
        )
    try:
        # if synthesis, just add fake data (None)
        if isinstance(selected_action, type(SelectedAction.SYNTHESIS)):
            timeseries_data = []
            for col in timeseries_colnames:
                if col in data:
                    timeseries_data.append(
                        OneTimeSeries(
                            name=col,
                            values=[
                                float(x)  # pyright: ignore
                                for x in data[col].values()  # pyright: ignore
                            ],
                        )
                    )
                else:
                    timeseries_data.append(
                        OneTimeSeries(
                            name=col, values=[-1] * window_size
                        )  # fake data
                    )
        else:
            # Create timeseries data
            timeseries_data = [
                OneTimeSeries(
                    name=col,
                    values=[
                        float(x)  # pyright: ignore
                        for x in data[col].values()  # pyright: ignore
                    ],
                )
                for col in timeseries_colnames
            ]

        # Create metadata
        discrete_conditions = [
            OneDiscreteMetaData(
                name=col,
                values=[str(x) for x in data[col].values()],  # pyright: ignore
            )
            for col in discrete_colnames + group_label_cols
        ]

        continuous_conditions = [
            OneContinuousMetaData(
                name=col,
                values=[
                    float(x)  # pyright: ignore
                    for x in data[col].values()  # pyright: ignore
                ],
            )
            for col in continuous_colnames
        ]

        metadata = MetaData(
            discrete_conditions=discrete_conditions,
            continuous_conditions=continuous_conditions,
        )

        # Create timestamps
        timestamps = (
            TimeStamps(
                name=timestamps_colname_str,
                values=list(
                    data[timestamps_colname_str].values()  # pyright: ignore
                ),
            )
            if timestamps_colname_str
            else None
        )

        # Create window
        window = SynthefyTimeSeriesWindow(
            id=0,
            name="Window 0",
            timeseries_data=timeseries_data,
            metadata=metadata,
            timestamps=timestamps,
        )

        # Extract constraints if they exist in the root data with special key
        synthesis_constraints = None
        if "_constraints_" in data:
            try:
                # Parse the constraints JSON string if it's a string
                constraints_data = data["_constraints_"]
                if isinstance(constraints_data, str):
                    try:
                        constraints_data = json.loads(constraints_data)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid JSON in _constraints_: {constraints_data}"
                        )
                        constraints_data = {}

                # If constraints_data is empty, return None
                if not constraints_data:
                    synthesis_constraints = None
                else:
                    # Extract and remove projection_type
                    projection_type = constraints_data.pop(
                        "_projection_type_", "clipping"
                    )

                    constraints_list = []

                    # Process remaining constraints
                    for channel_name, constraints in constraints_data.items():
                        if not isinstance(constraints, dict):
                            raise ValueError(
                                f"Invalid constraints format for {channel_name}: {constraints}"
                            )

                        for (
                            constraint_type,
                            constraint_value,
                        ) in constraints.items():
                            # Validate constraint type
                            try:
                                constraint_type_enum = ConstraintType(
                                    constraint_type
                                )
                            except ValueError:
                                logger.warning(
                                    f"Invalid constraint type: {constraint_type}"
                                )
                                continue

                            # Validate constraint value
                            if not isinstance(
                                constraint_value, (int, float, str)
                            ):
                                raise ValueError(
                                    f"Invalid constraint value type for {channel_name}:{constraint_type}: "
                                    f"{type(constraint_value).__name__}"
                                )

                            try:
                                constraint_value_float = float(constraint_value)
                            except (ValueError, TypeError):
                                raise ValueError(
                                    f"Invalid constraint value for {channel_name}:{constraint_type}: {constraint_value}"
                                )

                            # Validate constraint type compatibility with projection type
                            if (
                                projection_type == "clipping"
                                and constraint_type_enum
                                not in [ConstraintType.MIN, ConstraintType.MAX]
                            ):
                                raise ValueError(
                                    "When projection_during_synthesis='clipping', only 'min' and 'max' "
                                    "constraints are supported"
                                )

                            # Add valid constraint to the list
                            constraints_list.append(
                                OneConstraint(
                                    channel_name=str(channel_name),
                                    constraint_name=constraint_type_enum,
                                    constraint_value=constraint_value_float,
                                )
                            )

                    # Create synthesis_constraints object only if we have valid constraints
                    if constraints_list:
                        synthesis_constraints = SynthesisConstraints(
                            constraints=constraints_list,
                            projection_during_synthesis=ProjectionType(
                                projection_type
                            ),
                        )

            except ValueError as e:
                # Re-raise ValueError for invalid constraint values or combinations
                logger.error(
                    f"Validation error while processing constraints: {str(e)}"
                )
                raise ValueError(f"Failed to create window: {str(e)}")
            except Exception as e:
                # Convert other exceptions to ValueError to provide clear feedback
                logger.error(
                    f"Unexpected error while processing constraints: {type(e).__name__}: {str(e)}"
                )
                raise ValueError(f"Error processing constraints: {str(e)}")

    except Exception as e:
        logger.error(f"Error creating window: {str(e)}")
        raise ValueError(f"Failed to create window: {str(e)}")

    return SynthefyRequest(
        windows=[window],
        selected_windows=SelectedWindows(
            window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
            window_indices=[0],
        ),
        selected_action=selected_action,
        n_synthesis_windows=1,
        n_view_windows=5,
        n_anomalies=5,
        top_n_search_windows=5,
        n_forecast_windows=1,
        text="",
        synthesis_constraints=synthesis_constraints,
    )


def timeseries_to_dataframe(
    timeseries: List[OneTimeSeries], timestamps: Optional[TimeStamps] = None
) -> pd.DataFrame:
    data = {}
    for channel in timeseries:
        data[channel.name] = channel.values

    if timestamps is not None:
        data[timestamps.name] = timestamps.values

    df = pd.DataFrame(data)
    return df


def metadata_to_dataframe(
    metadata: MetaData,
    timeseries: Optional[List[OneTimeSeries]] = None,
    timestamps: Optional[TimeStamps] = None,
) -> pd.DataFrame:
    data = {}
    # Process discrete conditions
    for condition in metadata.discrete_conditions:
        data[condition.name] = condition.values
    # Process continuous conditions
    for condition in metadata.continuous_conditions:
        data[condition.name] = condition.values

    if timeseries is not None:
        for channel in timeseries:
            data[channel.name] = channel.values

    if timestamps is not None:
        data[timestamps.name] = timestamps.values

    df = pd.DataFrame(data)
    return df


def convert_str_to_isoformat(
    input: Union[pd.Timestamp, datetime, np.datetime64, str],
) -> str:
    return pd.Timestamp(input).isoformat()


def convert_list_to_isoformat(
    input: Sequence[Union[pd.Timestamp, datetime, np.datetime64, str]],
) -> List[str]:
    # convert to pd.Timestamp
    return [convert_str_to_isoformat(value) for value in input]


def array_to_timeseries(
    timeseries_window: np.ndarray,
    channel_names: List[str],
) -> List[OneTimeSeries]:
    """
    Convert each col of the given timeseries window into OneTimeSeries list.
    Parameters:
        - timeseries_window: 2D np array, one window (window_size, num_channels)
        - channel_names: timeseries col names
    Return:
        - List[OneTimeSeries]: List of OneTimeSeries instances
    """
    return [
        OneTimeSeries(
            name=channel_names[channel_idx],
            values=list(timeseries_window[:, channel_idx]),
        )
        for channel_idx in range(timeseries_window.shape[1])
    ]


def array_to_continuous(
    continuous_window: np.ndarray,
    continuous_col_names: List[str],
) -> List[OneContinuousMetaData]:
    """
    Convert each col of the given continuous condition np array into OneContinuousMetaData list.
    Parameters:
        - continuous_window: 2D np array, one window (window_size, num_continuous_features)
        - continuous_col_names: list with continuous feature names
    Return:
        - List[OneContinuousMetaData]: List of OneContinuousMetaData instances
    """
    return [
        OneContinuousMetaData(
            name=continuous_col_names[continuous_condition_idx],
            values=list(continuous_window[:, continuous_condition_idx]),
        )
        for continuous_condition_idx in range(continuous_window.shape[1])
    ]


def inverse_transform_discrete(
    discrete_windows: np.ndarray,
    encoders: Dict[str, Any],
) -> Tuple[List[str], np.ndarray]:
    """
    Inverse transform encoded discrete cols for UI.
    Returns empty if no encoders provided to prevent larger downstream changes.
    Parameteres:
        - discrete_windows: encoded discrete windows, 3D np.array
        - encoders: encoders of struct {encoder_type: encoder}
    Return:
        - decoded col names list and decoded discrete conditions 3D array
        - [] and np.array([]) if encoders is empty

    """
    if not encoders:
        logger.debug("No encoders provided for inverse transformation")
        return [], np.array([])
    decoded_arrays_list = []
    final_col_names = []
    start_col_ind = 0
    logger.info(f"{discrete_windows.shape=}")
    for encoder_type, encoder in encoders.items():
        logger.info(f"Inverse transforming {encoder_type}...")

        final_col_names.extend(list(encoder.feature_names_in_))
        encoded_cols = encoder.get_feature_names_out()
        logger.info(f"{encoder_type} cols num: {len(encoded_cols)}")
        end_col_ind = start_col_ind + len(encoded_cols)
        sliced_discrete_windows = discrete_windows[
            :, :, start_col_ind:end_col_ind
        ]
        start_col_ind = end_col_ind

        windows_2d = sliced_discrete_windows.reshape(-1, len(encoded_cols))
        decoded_2d = encoder.inverse_transform(windows_2d)
        # Reshape the 2D array back to the original 3D shape
        decoded_3d = decoded_2d.reshape(
            sliced_discrete_windows.shape[0],  # Number of windows
            sliced_discrete_windows.shape[1],  # Window size
            -1,  # Number of original features (before one-hot encoding)
        )
        decoded_arrays_list.append(decoded_3d)

    decoded_array = np.concatenate(decoded_arrays_list, axis=2)

    return final_col_names, decoded_array


def array_to_discrete(
    discrete_window: np.ndarray,
    discrete_col_names: List[str],
) -> List[OneDiscreteMetaData]:
    """
    Convert each col of the given discrete condition np array into OneDiscreteMetaData list.
    Parameters:
        - discrete_window: 2D np array, one window (window_size, num_discrete_features)
        - discrete_col_names: list with discrete feature names
    Return:
        - List[OneDiscreteMetaData]: List of OneDiscreteMetaData instances
    """
    discrete_list = []
    if not discrete_col_names:
        discrete_col_names = [
            f"discrete_condition_{i}" for i in range(discrete_window.shape[1])
        ]

    for discrete_condition_idx in range(discrete_window.shape[1]):
        discrete_list.append(
            OneDiscreteMetaData(
                name=discrete_col_names[discrete_condition_idx],
                values=list(discrete_window[:, discrete_condition_idx]),
            )
        )
    return discrete_list


def array_to_timestamps(
    timestamps_window: np.ndarray, timestamps_col_names: List[str]
) -> TimeStamps:
    # only support 1 timestamp column for now
    return TimeStamps(
        name=timestamps_col_names[0],
        values=list(timestamps_window[:, 0]),
    )


def load_preprocessed_dataset_windows(
    dataset_path: str,
    train_test_val: str,
    window_type: List[str],
    window_indices: List[int],
) -> Dict[str, np.ndarray]:
    """
    inputs:
        dataset_path: path to the datasets output folder
        train_test_val: "train"/"val"/"test"
        window_indices: indices of the windows to load
        window_type: type of the windows to load options: ["timeseries", "continuous", "discrete", "timestamps"]
    outputs:
        Dict[str, np.ndarray]: windows of interest
    """
    ret_dict = {}

    if "timeseries" in window_type:
        ts = np.load(
            os.path.join(
                dataset_path, f"{train_test_val}_timeseries_windows.npy"
            )
        )[window_indices]
        ret_dict["timeseries"] = ts
    if "discrete" in window_type:
        discrete = np.load(
            os.path.join(dataset_path, f"{train_test_val}_discrete_windows.npy")
        )[window_indices]
        ret_dict["discrete"] = discrete
    if "continuous" in window_type:
        continuous = np.load(
            os.path.join(
                dataset_path, f"{train_test_val}_continuous_windows.npy"
            )
        )[window_indices]
        ret_dict["continuous"] = continuous
    if "timestamps" in window_type:
        timestamps = np.load(
            os.path.join(
                dataset_path, f"{train_test_val}_timestamps_windows.npy"
            )
        )[window_indices]
        ret_dict["timestamps"] = timestamps

    return ret_dict


# TODO - for later -- what was this for??
def get_windows_from_request(
    request: SynthefyRequest,
    datasets_output_path: str,
) -> None:  # List[SynthefyTimeSeriesWindow]:
    """
    inputs:
        request: SynthefyRequest object
        datasets_output_path: path to the datasets output folder - for reading train/val/test datasets
    outputs:
        List[SynthefyTimeSeriesWindow]: List of SynthefyTimeSeriesWindow objects
    description:
        This function is used to get the windows that the user is interested in from the request object.
        The options for selections are as follows:

        User can always select one or more windows by index.
        User can select subset of the currently displayed windows (CURRENT_VIEW_WINDOWS)
        User can select subset of train/val/test datasets (TRAIN/VAL/TEST)
        User can select "UPLOADED_DATASET" (which corresponds to the latest uploaded dataset)

    """
    pass
    # # Right now. we will only support 1 index for the selected_windows.window_indices
    # if len(request.selected_windows.window_indices) != 1:
    #     raise ValueError("Currently we only support 1 window.")

    # if (
    #     request.selected_windows.window_type
    #     == WindowsSelectionOptions.CURRENT_VIEW_WINDOWS
    # ):
    #     windows = [request.windows[i] for i in request.selected_windows.window_indices]
    #     return convert_request_windows_to_arrays(windows)

    # elif (
    #     request.selected_windows.window_type == WindowsSelectionOptions.TRAIN
    #     or request.selected_windows.window_type == WindowsSelectionOptions.VAL
    #     or request.selected_windows.window_type == WindowsSelectionOptions.TEST
    # ):
    #     windows = load_preprocessed_dataset_windows(
    #         datasets_output_path, request.selected_windows.window_indices
    #     )
    #     return convert_request_windows_to_arrays(windows)

    # elif (
    #     request.selected_windows.window_type == WindowsSelectionOptions.UPLOADED_DATASET
    # ):
    #     # TODO - this needs to be put into the preprocessed dataset folder with current_uploaded_dataset_* prefix?
    #     # Then this if can be removed, and uploaded dataset can be used in the same way as train/val/test datasets
    #     pass

    # else:
    #     raise ValueError(f"Invalid window type: {request.selected_windows.window_type}")


def create_window_name_from_group_labels(
    discrete_conditions: List[OneDiscreteMetaData],
    group_label_cols: List[str],
    default_name: str = "Window",
) -> str:
    """
    Create a window name from the group label values in the discrete_conditions.

    Parameters:
        discrete_conditions: List[OneDiscreteMetaData] - discrete metadata for the window.
        group_label_cols: List[str] - the group label column names as defined in the preprocess config.

    Returns:
        A string in the format "col1=val1,col2=val2,..." if any group label columns are found,
        or "Window" if none are found.
    """
    group_labels = []
    for condition in discrete_conditions:
        if condition.name in group_label_cols:
            if condition.values:
                group_labels.append(f"{condition.name}={condition.values[0]}")
    return ",".join(group_labels) if group_labels else default_name


def convert_response_to_synthefy_window_and_text(
    dataset_name: str,
    response: Union[
        SynthesisResponse,
        SearchResponse,
        ForecastResponse,
    ],
    response_id: int = 0,
    streaming: bool = False,
) -> Any:
    """Convert the response to a SynthefyTimeSeriesWindow object and the text."""
    # Load config and labels config
    window_naming = get_window_naming_config()
    labels_config = get_labels_description(dataset_name)
    group_label_cols = labels_config.get("group_label_cols", [])

    # get default prefix for the window_name
    if isinstance(response, SearchResponse):
        prefix = window_naming.get("search_prefix", "Window")
    elif isinstance(response, SynthesisResponse):
        prefix = window_naming.get("synthesis_prefix", "Window")
    elif isinstance(response, ForecastResponse):
        prefix = window_naming.get("forecast_prefix", "Window")
    else:
        raise ValueError(f"Invalid response type: {type(response)}")

    if isinstance(response, SearchResponse):
        windows = [
            SynthefyTimeSeriesWindow(
                id=i,
                name=(
                    create_window_name_from_group_labels(
                        response.metadata[i].discrete_conditions,
                        group_label_cols,
                        f"{prefix} {i}",
                    )
                    if i != 0
                    else "Search Query"
                ),
                timeseries_data=response.timeseries_data[i],
                metadata=MetaData(
                    continuous_conditions=response.metadata[
                        i
                    ].continuous_conditions,
                    discrete_conditions=(
                        convert_discrete_metadata_to_label_tuple(
                            response.metadata[i].discrete_conditions,
                            labels_config["group_labels_combinations"],
                            labels_config["group_label_cols"],
                        )
                        if not streaming
                        else response.metadata[i].discrete_conditions
                    ),
                ),
                timestamps=response.x_axis[i],
                text=(
                    f"Closest window {i} to your query window."
                    if response.text
                    else ""
                ),
            )
            for i in range(len(response.x_axis))
        ]
        forecast_timestamp = []
        anomaly_timestamps = []
        return windows, forecast_timestamp, anomaly_timestamps
    else:
        window = SynthefyTimeSeriesWindow(
            id=response_id,
            name=create_window_name_from_group_labels(
                response.metadata.discrete_conditions,
                group_label_cols,
                f"{prefix} {response_id}",
            ),
            timeseries_data=response.timeseries_data,
            metadata=MetaData(
                continuous_conditions=response.metadata.continuous_conditions,
                discrete_conditions=(
                    convert_discrete_metadata_to_label_tuple(
                        response.metadata.discrete_conditions,
                        labels_config["group_labels_combinations"],
                        labels_config["group_label_cols"],
                    )
                    if not streaming
                    else response.metadata.discrete_conditions
                ),
            ),
            timestamps=response.x_axis,
            text=response.text if response.text else "",
        )

    # return order is window, text, forecast_timestamp, anomaly_timestamps
    if isinstance(response, ForecastResponse):
        return window, response.start_of_forecast_timestamp, []
    elif isinstance(response, PreTrainedAnomalyResponse):
        return window, [], response.anomaly_timestamps
    else:
        return window, [], []


async def create_synthefy_response_from_other_types(
    dataset_name: str,
    responses: Union[
        List[SynthesisResponse], List[SearchResponse], List[ForecastResponse]
    ],
    streaming: bool = False,
    text: str = "",
) -> SynthefyResponse:
    windows, forecast_timestamp_list, anomaly_timestamps_list = [], [], []
    if isinstance(responses[0], SearchResponse):
        windows, forecast_timestamp_list, anomaly_timestamps_list = (
            convert_response_to_synthefy_window_and_text(
                dataset_name, responses[0], streaming=streaming
            )
        )  # only 1 in search.
        return SynthefyResponse(
            windows=windows,
            forecast_timestamps=forecast_timestamp_list,
            anomaly_timestamps=anomaly_timestamps_list,
            combined_text="Closest windows to your query window.",
        )
    else:
        for response_id, response in enumerate(responses):
            window, forecast_timestamp, anomaly_timestamps = (
                convert_response_to_synthefy_window_and_text(
                    dataset_name, response, response_id, streaming=streaming
                )
            )
            windows.append(window)
            forecast_timestamp_list.append(forecast_timestamp)
            anomaly_timestamps_list.append(anomaly_timestamps)

        # TODO remove this bad code...
        if any(
            isinstance(timestamp, TimeStamps)
            for timestamp in forecast_timestamp_list
        ):
            new_forecast_timestamp_list = []
            for timestamp in forecast_timestamp_list:
                if timestamp == []:
                    new_forecast_timestamp_list.append(
                        TimeStamps(name="time", values=[])
                    )
                else:
                    new_forecast_timestamp_list.append(timestamp)
            forecast_timestamp_list = new_forecast_timestamp_list
        else:
            forecast_timestamp_list = []
        if any(
            isinstance(timestamp, TimeStamps)
            for timestamp in anomaly_timestamps_list
        ):
            new_anomaly_timestamps_list = []
            for timestamp in anomaly_timestamps_list:
                if timestamp == []:
                    new_anomaly_timestamps_list.append(
                        TimeStamps(name="time", values=[])
                    )
                else:
                    new_anomaly_timestamps_list.append(timestamp)
            anomaly_timestamps_list = new_anomaly_timestamps_list
        else:
            anomaly_timestamps_list = []
        # TODO end of bad code...

        summaries = [
            {"window_name": window.name, "summary": window.text}
            for window in windows
        ]
        combined_text = await generate_combined_summary(text, summaries)
        return SynthefyResponse(
            windows=windows,
            combined_text=combined_text,
            forecast_timestamps=forecast_timestamp_list,
            anomaly_timestamps=anomaly_timestamps_list,
        )


def convert_synthefy_request_to_search_request(
    synthefy_request: SynthefyRequest,
) -> SearchRequest:
    if len(synthefy_request.selected_windows.window_indices) > 1:
        raise ValueError(
            "You can only search for one similar window at a time."
        )
    if (
        synthefy_request.selected_windows.window_type
        != WindowsSelectionOptions.CURRENT_VIEW_WINDOWS
    ):
        raise ValueError("You can only search for the current view windows.")

    # search query is current view window @ index X by default.
    search_query = synthefy_request.windows[
        synthefy_request.selected_windows.window_indices[0]
    ].timeseries_data
    search_metadata = synthefy_request.windows[
        synthefy_request.selected_windows.window_indices[0]
    ].metadata
    search_timestamps = synthefy_request.windows[
        synthefy_request.selected_windows.window_indices[0]
    ].timestamps

    # once we support more than the current view window, we will need to add more logic here
    if (
        synthefy_request.selected_windows.window_type
        == WindowsSelectionOptions.TRAIN
    ):
        # load the train window at index X for the search query
        raise ValueError("Not Implemented Yet.")
    elif (
        synthefy_request.selected_windows.window_type
        == WindowsSelectionOptions.VAL
    ):
        # load the val window at index X for the search query
        raise ValueError("Not Implemented Yet.")
    elif (
        synthefy_request.selected_windows.window_type
        == WindowsSelectionOptions.TEST
    ):
        # load the test window at index X for the search query
        raise ValueError("Not Implemented Yet.")
    elif (
        synthefy_request.selected_windows.window_type
        == WindowsSelectionOptions.UPLOADED_DATASET
    ):
        # load the uploaded dataset at index X for the search query
        raise ValueError("Not Implemented Yet.")

    return SearchRequest(
        search_query=search_query,
        search_metadata=search_metadata,
        search_timestamps=search_timestamps,
        metadata_ranges=None,
        timestamps_range=None,
        n_closest=synthefy_request.top_n_search_windows,
        text=synthefy_request.text,
    )


def delete_gt_real_timeseries_windows(request: SynthefyRequest):
    """
    If all timeseries in the window have _synthesis in it, we should delete the window
    since it was the result of forecast/synthesis "GT/real values" from the previous step.

    Note: It does this inplace and overwrites the request.windows
    Note: If all timseries names end in "_query", we will remove "_query" from the name.
          as it was output from search API previously.
    """

    # TODO - create more special endings instead of _synthetic and _query to avoid removing real data.
    # or read the real timeseries names from the dataset.
    new_windows = []
    for window in request.windows:
        new_window_timeseries = []

        # this if handles search api output along with synthesis/forecast output when no real data is returned.
        if (
            all(
                timeseries.name.endswith("_query")
                for timeseries in window.timeseries_data
            )
            or all(
                timeseries.name.endswith("_synthetic")
                for timeseries in window.timeseries_data
            )
            or all(
                timeseries.name.endswith("_forecast")
                for timeseries in window.timeseries_data
            )
        ):
            for timeseries in window.timeseries_data:
                # Determine suffix type and remove it
                if timeseries.name.endswith("_query"):
                    new_name = timeseries.name[:-6]  # remove _query
                elif timeseries.name.endswith("_forecast"):
                    new_name = timeseries.name[:-9]  # remove _forecast
                elif timeseries.name.endswith("_synthetic"):
                    new_name = timeseries.name[:-10]  # remove _synthetic
                else:
                    new_name = timeseries.name
                new_window_timeseries.append(
                    OneTimeSeries(
                        name=new_name,
                        values=timeseries.values,
                    )
                )
        else:
            # this handles when synthesis/forecast returns real data along with synthetic data.
            for timeseries in window.timeseries_data:
                if (
                    not timeseries.name.endswith("_synthetic")
                    and not timeseries.name.endswith("_query")
                    and not timeseries.name.endswith("_forecast")
                ):
                    new_window_timeseries.append(timeseries)

        if len(new_window_timeseries) > 0:
            new_windows.append(
                SynthefyTimeSeriesWindow(
                    timeseries_data=new_window_timeseries,
                    metadata=window.metadata,
                    timestamps=window.timestamps,
                )
            )
        else:
            raise ValueError(
                "No timeseries left after dropping -- this should never happen in prod/UI. Please check the code."
            )

    request.windows = new_windows
    return request


def s3_prefix_exists(s3_client, bucket_name: str, prefix: str) -> bool:
    """
    Check if any objects exist under the given S3 prefix
    """
    response = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=prefix, MaxKeys=1
    )
    return "Contents" in response


def delete_s3_objects(s3_client, bucket: str, prefix: str):
    """Delete all objects under a prefix in an S3 bucket."""
    paginator = s3_client.get_paginator("list_objects_v2")
    objects_to_delete = []

    # Collect all objects
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            objects_to_delete.extend(
                [{"Key": obj["Key"]} for obj in page["Contents"]]
            )

    # Delete in batches of 1000 (S3 limit)
    if objects_to_delete:
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i : i + 1000]
            s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        logger.info(
            f"Deleted objects under prefix '{prefix}' in bucket '{bucket}'."
        )


def api_key_required(
    x_api_key: str | None = Header(None), db: Session = Depends(get_db)
) -> str | None:
    if os.environ.get("API_KEY_AUTH_ENABLED", "false") == "false":
        # Skip API key-based authentication when API_KEY_AUTH_ENABLED is not set or set to false.
        return

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    user_id = validate_api_key(db, x_api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user_id


def check_scale_by_metadata_used(scalers: Dict[str, Any]) -> bool:
    """
    Check if any of the scalers have the scale_by_discrete_metadata flag set.
    """
    for k, v in scalers.items():
        if v[0].get("tuple", None) is not None:
            logger.info(f"Scaler {k} has scale_by_discrete_metadata flag set.")
            return True

    return False


def get_window_naming_config() -> Dict[str, str]:
    """Get window naming configuration from loaded api config.

    Returns:
        Dict containing window naming prefixes
    """
    api_config_path = os.getenv("SYNTHEFY_CONFIG_PATH")
    assert api_config_path is not None
    api_config = load_yaml_config(api_config_path)

    default_window_naming_config = {
        "pretrained_anomaly_prefix": "Failure",
        "synthesis_prefix": "Synthesis",
        "forecast_prefix": "Forecast",
        "search_prefix": "Similar Window",
        "view_prefix": "Window",
    }
    window_naming_config = api_config.get("window_naming_config", {})
    for key, default_value in default_window_naming_config.items():
        if key not in window_naming_config:
            logger.warning(
                f"Window naming config missing {key}, using default - {default_value}"
            )
            window_naming_config[key] = default_value
    return window_naming_config


def get_labels_description(dataset_name: str) -> Dict[str, Any]:
    """Load `labels_description.pkl`.

    Args:
        dataset_name: Name of the dataset dir

    Returns:
        Dict containing labels configuration
    """
    datasets_base = os.getenv("SYNTHEFY_DATASETS_BASE")
    assert datasets_base is not None
    file_path = os.path.join(
        datasets_base, dataset_name, "labels_description.pkl"
    )
    with open(file_path, "rb") as f:
        labels_description = pickle.load(f)
    return {
        "group_labels_combinations": labels_description.get(
            "labels_description", {}
        ).get("group_labels_combinations", {}),
        "group_label_cols": labels_description.get("group_label_cols", []),
    }


def get_settings(
    settings_class: Type[BaseModel], dataset_name: Optional[str] = None
) -> Any:
    """Get settings from api config file for a specific settings class.

    Args:
        settings_class: The settings class to instantiate (must be a Pydantic model)
        dataset_name: Optional dataset name for path substitution

    Returns:
        Instance of the specified settings class

    Raises:
        RuntimeError: If api config file cannot be loaded or SYNTHEFY_CONFIG_PATH not set
    """
    config_path = os.getenv("SYNTHEFY_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("SYNTHEFY_CONFIG_PATH environment variable not set")

    api_config = load_yaml_config(config_path)
    if dataset_name is not None:
        api_config["dataset_name"] = dataset_name

    # Extract fields defined in the settings class
    settings_dict = {
        field: api_config.get(field)
        for field in settings_class.model_fields
        if field in api_config
    }

    # No need for manual environment variable substitution since it's handled by YAML loader
    if dataset_name is not None:
        settings_dict = {
            k: (
                v.replace("${dataset_name}", dataset_name)
                if isinstance(v, str)
                else v
            )
            for k, v in settings_dict.items()
        }

    return settings_class(**settings_dict)


def get_user_tmp_dir(user_id: str, dataset_name: Optional[str] = None) -> str:
    """Generate a temporary directory path for a user and optionally a dataset.

    Args:
        user_id: User identifier
        dataset_name: Optional dataset name for dataset-specific directories

    Returns:
        str: Path to the user's temporary directory
    """
    user_id = user_id.replace('"', "")
    if dataset_name:
        tmp_dir = os.path.join("/tmp", user_id, dataset_name)
    else:
        tmp_dir = os.path.join("/tmp", user_id)

    os.makedirs(tmp_dir, exist_ok=True)
    logger.debug(f"Created temporary directory: {tmp_dir}")
    return tmp_dir


def handle_file_upload(file: UploadFile, tmp_dir: str) -> str:
    """
    Handle file upload and save to temporary directory.

    Args:
        file: Uploaded file
        tmp_dir: Temporary directory to save file

    Returns:
        Path to saved file

    Raises:
        HTTPException: If file save operation fails
    """
    if not file.filename:
        raise ValueError("Filename cannot be None")
    file_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(file_path, "wb") as dest_file:
            shutil.copyfileobj(file.file, dest_file)
        return file_path
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {str(e)}"
        )


def cleanup_tmp_dir(tmp_dir: str) -> None:
    """
    Recursively remove a temporary directory and all its contents (equivalent to rm -rf).

    Args:
        tmp_dir: Path to the temporary directory to be completely removed

    Note:
        This will delete the directory itself along with everything inside it.
        Use with caution as the deletion cannot be undone.
    """
    try:
        shutil.rmtree(tmp_dir)
        logger.info(f"Cleaned up temporary directory: {tmp_dir}")
    except Exception as e:
        logger.warning(f"Error cleaning up temporary files: {e}")


def cleanup_local_directories(paths: List[str]) -> None:
    """
    Clean up local directories and files.

    Args:
        paths: List of paths to clean up

    Note:
        - Handles both files and directories
        - Logs warnings for non-existent paths
        - Logs errors for cleanup failures
        - Does not raise exceptions
    """
    if not paths:
        logger.warning("No cleanup paths found")
        return

    logger.info(f"Cleaning up local directories: {paths=}")
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            else:
                logger.warning(f"Path does not exist: {path}")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup of {path}: {cleanup_error}")


def filter_window_dataframe_group_labels(
    df: pd.DataFrame,
    group_label_cols_filters: OneDiscreteMetaDataRange | None = None,
) -> List[int]:
    """
    Filter the dataframe by the group labels columns.

    Inputs:
        df: pd.DataFrame
        group_label_cols_filters: OneDiscreteMetaDataRange | None
            ex: OneDiscreteMetaDataRange(name='group_label_1-group_label_2', options=['EU-dev1', 'NA-dev2'])
            # note - support None here just so we don't have to check before calling this function
    Returns:
        List[int]: List of window indices that match the filter criteria. [] if no windows match the filter criteria.

    Edge Cases:
        return [] if none of the filters are present in the dataframe
    """
    # split by "-"
    if group_label_cols_filters is None:
        return list(df["window_idx"].unique())

    # get the column names (just from first item)
    group_label_col_names = group_label_cols_filters.name.split("-")

    # Find the unique values in the dataframe for the group_label_col_names
    df_tmp = df[group_label_col_names + ["window_idx"]].drop_duplicates()
    if df_tmp.empty:
        return []

    # Create combined keys for each row
    df_tmp["combined_key"] = (
        df_tmp[group_label_col_names].astype(str).agg("-".join, axis=1)
    )

    # Filter rows where the combined key is in the valid options
    valid_window_indices = df_tmp[
        df_tmp["combined_key"].isin(group_label_cols_filters.options)
    ]["window_idx"].tolist()

    return valid_window_indices


def filter_window_dataframe_continuous_conditions(
    df: pd.DataFrame,
    continuous_conditions_filters: List[OneContinuousMetaDataRange] = [],
) -> List[int]:
    """
    Filter the dataframe by the continuous conditions.

    Inputs:
        df: pd.DataFrame -> must include columns: window_idx, continuous_1, continuous_2, ...
        continuous_conditions: List[OneContinuousMetaDataRange]
            ex: OneContinuousMetaDataRange(name='continuous_1', min_val=0, max_val=10)

    Returns:
        List[int]: List of window indices that match the filter criteria. [] if no windows match the filter criteria.

    Edge Cases:
        return [] if none of the filters are present in the dataframe
    """
    if not continuous_conditions_filters:
        logger.warning(
            "No continuous conditions filters provided - returning all windows"
        )
        return list(df["window_idx"].unique())

    # Extract column names and their corresponding min/max values
    condition_cols = [
        c.name for c in continuous_conditions_filters if c.name in df.columns
    ]
    if not condition_cols:
        return []

    min_vals = {
        c.name: c.min_val
        for c in continuous_conditions_filters
        if c.name in df.columns
    }
    max_vals = {
        c.name: c.max_val
        for c in continuous_conditions_filters
        if c.name in df.columns
    }

    # Calculate min/max for all conditions at once
    window_stats = df.groupby("window_idx")[condition_cols].agg(["min", "max"])

    # Create masks for min and max conditions
    min_mask = pd.DataFrame(
        {
            col: window_stats[(col, "min")] >= min_vals[col]
            for col in condition_cols
        }
    )
    max_mask = pd.DataFrame(
        {
            col: window_stats[(col, "max")] <= max_vals[col]
            for col in condition_cols
        }
    )

    # Get windows where all conditions are met
    valid_windows = min_mask.all(axis=1) & max_mask.all(axis=1)
    logger.info(
        f"Filtered based on continuous conditions to {len(valid_windows[valid_windows].index)} windows"
    )
    return list(valid_windows[valid_windows].index)


def filter_window_dataframe_discrete_conditions(
    df: pd.DataFrame,
    discrete_conditions_filters: List[OneDiscreteMetaDataRange],
) -> List[int]:
    """
    Filter the dataframe by the discrete conditions using vectorized operations.

    Inputs:
        df: pd.DataFrame -> must include columns: window_idx, discrete_1, discrete_2, ...
        discrete_conditions: List[OneDiscreteMetaDataRange]
            ex: [OneDiscreteMetaDataRange(name='discrete_1', options=['A', 'C']),
                 OneDiscreteMetaDataRange(name='discrete_2', options=['B', 'D'])]

    Returns:
        List[int]: List of window indices that match the filter criteria. [] if no windows match the filter criteria.

    Edge Cases:
        return [] if none of the filters are present in the dataframe
    """
    if not discrete_conditions_filters:
        logger.warning(
            "No discrete conditions filters provided - returning all windows"
        )
        return list(df["window_idx"].unique())

    # Extract column names that exist in the dataframe
    condition_cols = [
        c.name for c in discrete_conditions_filters if c.name in df.columns
    ]
    if not condition_cols:
        return []

    # Create masks for each condition
    masks = []
    for condition in discrete_conditions_filters:
        if condition.name in df.columns:
            # Try both with and without string conversion
            mask = df[condition.name].isin(condition.options)
            # Group by window_idx and check if all values satisfy the condition
            window_mask = mask.groupby(df["window_idx"]).all()
            masks.append(window_mask)

    # Combine all masks with AND operation
    final_mask = pd.concat(masks, axis=1).all(axis=1)

    # Get window indices where all conditions are satisfied
    logger.info(
        f"Filtered based on discrete conditions to {len(final_mask[final_mask].index)} windows"
    )
    return list(final_mask[final_mask].index)


def filter_window_dataframe_timestamp_conditions(
    df: pd.DataFrame, timestamp_conditions: Optional[TimeStampsRange] = None
) -> List[int]:
    """Filter the dataframe by the timestamp conditions."""
    # TODO - implement this - for now just return all windows
    logger.warning(
        f"Not implemented: Filtering dataframe with timestamp conditions: {timestamp_conditions} - returning all windows"
    )

    if timestamp_conditions is None:
        return list(df.window_idx.unique())

    return list(df.window_idx.unique())


async def filter_window_dataframe_by_window_filters(
    df: pd.DataFrame, window_filters: WindowFilters
) -> pd.DataFrame:
    """
    Filter the dataframe by applying all window filters sequentially and asynchronously.
    Resets the window_idx to be 0:N_windows_after_filtering

    Args:
        df: Input dataframe containing window data
        window_filters: WindowFilters object containing filtering criteria

    Returns:
        pd.DataFrame: Dataframe with only the windows that satisfy all filter conditions

    Note:
        Runs filtering operations sequentially but asynchronously to avoid blocking.
        Each filtering function handles None/empty cases by returning all window indices.
    """
    logger.info(f"Filtering dataframe with window filters: {window_filters}")

    async def run_filter_in_thread(filter_func, *args):
        return await asyncio.to_thread(filter_func, *args)

    # Run filtering operations sequentially but asynchronously
    valid_window_idxs = set(df["window_idx"].unique())

    # Group labels filter
    group_labels_results = await run_filter_in_thread(
        filter_window_dataframe_group_labels,
        df,
        window_filters.group_label_cols,
    )
    valid_window_idxs &= set(group_labels_results)

    # Continuous conditions filter
    continuous_results = await run_filter_in_thread(
        filter_window_dataframe_continuous_conditions,
        df,
        window_filters.metadata_range.continuous_conditions,
    )
    valid_window_idxs &= set(continuous_results)

    # Discrete conditions filter
    discrete_results = await run_filter_in_thread(
        filter_window_dataframe_discrete_conditions,
        df,
        window_filters.metadata_range.discrete_conditions,
    )
    valid_window_idxs &= set(discrete_results)

    # Timestamp conditions filter
    timestamp_results = await run_filter_in_thread(
        filter_window_dataframe_timestamp_conditions,
        df,
        window_filters.timestamps_range,
    )
    valid_window_idxs &= set(timestamp_results)

    # Timeseries range filter
    timeseries_results = await run_filter_in_thread(
        filter_window_dataframe_continuous_conditions,
        df,
        window_filters.timeseries_range,
    )
    valid_window_idxs &= set(timeseries_results)

    valid_window_idxs = sorted(list(valid_window_idxs))
    original_num_windows = df["window_idx"].nunique()

    # Final filtering in thread to avoid blocking
    filtered_df = await asyncio.to_thread(
        lambda: df[df["window_idx"].isin(valid_window_idxs)]
    )

    filtered_num_windows = filtered_df["window_idx"].nunique()
    logger.info(
        f"Filtered dataframe from {original_num_windows} windows to {filtered_num_windows} windows"
    )
    # get the window_size by looking at the count of the first window_idx
    if filtered_num_windows > 0:
        window_size = len(
            filtered_df[filtered_df["window_idx"] == valid_window_idxs[0]]
        )
        filtered_df.loc[:, "window_idx"] = np.repeat(
            np.arange(int(len(filtered_df) / window_size)), window_size
        )

    return filtered_df


async def apply_perturbation_variation_to_dataframe(
    df: pd.DataFrame,
    col_name: str,
    perturbation_amount: Union[float, int],
    perturbation_type: PerturbationType,
) -> pd.DataFrame:
    """
    Apply a perturbation to the dataframe.
    supports add, subtract, multiply, divide on the numeric column values.

    Inputs:
        df: pd.DataFrame
        col_name: str
        perturbation_amount: Union[float, int]
        perturbation_type: PerturbationType

    Returns:
        pd.DataFrame: Dataframe with the perturbation applied

    Edge Cases:
        raise ValueError if the column is not numeric
    """

    current_values = df[col_name]

    # Convert to numeric type first, then check
    try:
        current_values = pd.to_numeric(current_values)
    except Exception:
        raise ValueError(
            f"Column {col_name} contains non-numeric values - cannot apply perturbation: {perturbation_type} with amount: {perturbation_amount}"
        )

    # apply the perturbation
    if perturbation_type == PerturbationType.ADD:
        new_values = current_values + perturbation_amount
    elif perturbation_type == PerturbationType.SUBTRACT:
        new_values = current_values - perturbation_amount
    elif perturbation_type == PerturbationType.MULTIPLY:
        new_values = current_values * perturbation_amount
    elif perturbation_type == PerturbationType.DIVIDE:
        new_values = current_values / perturbation_amount
    # update the dataframe
    df.loc[:, col_name] = new_values
    return df


async def apply_exact_value_variation_to_dataframe(
    df: pd.DataFrame,
    col_name: str,
    value: Union[str, int, float],
) -> pd.DataFrame:
    """
    Apply an exact value to the dataframe.

    Inputs:
        df: pd.DataFrame
        col_name: str
        value: Union[str, int, float]

    Returns:
        pd.DataFrame: Dataframe with the exact value applied

    Edge Cases:
        raise ValueError if the value is not the same type as the column
    """

    # check to make sure the type of the value is the same as the type of the column
    # Check if value type matches column type, excluding NaN values
    valid_values = df[col_name].dropna()
    if len(valid_values) == 0 or not all(
        isinstance(value, type(val)) for val in valid_values
    ):
        raise ValueError(
            f"Value {value} is not the same type as the column {col_name}: {type(value)=} != {type(valid_values.iloc[0])=}"
        )

    # update the dataframe
    df.loc[:, col_name] = value
    return df


async def apply_metadata_variation(
    df: pd.DataFrame,
    metadata_variation: MetaDataVariation,
) -> pd.DataFrame:
    """Apply a metadata variation to the dataframe.

    Inputs:
        df: pd.DataFrame
        metadata_variation: MetaDataVariation

        MetadataVariation:
            name: str
            value: Union[str, int, float]
            perturbation_or_exact_value: Literal["perturbation", "exact_value"]
            perturbation_type: Optional[PerturbationType] = None

    Returns:
        pd.DataFrame: Dataframe with the metadata variation applied
    """
    col_name = metadata_variation.name

    # get the perturbation_or_exact_value
    perturbation_or_exact_value = metadata_variation.perturbation_or_exact_value
    if (
        perturbation_or_exact_value == "perturbation"
        and metadata_variation.perturbation_type is not None
    ):
        perturbation_type = metadata_variation.perturbation_type
        perturbation_amount = metadata_variation.value
        if not isinstance(perturbation_amount, (float, int)):
            raise ValueError(
                f"Perturbation amount {perturbation_amount} is not a float or int "
                f"- cannot apply perturbation: {perturbation_type} with amount: {perturbation_amount}"
            )
        # apply the perturbation to the dataframe
        df = await apply_perturbation_variation_to_dataframe(
            df=df,
            col_name=col_name,
            perturbation_amount=perturbation_amount,
            perturbation_type=perturbation_type,
        )
    else:
        # apply the exact value to the dataframe
        value = metadata_variation.value
        df = await apply_exact_value_variation_to_dataframe(
            df=df, col_name=col_name, value=value
        )
    return df


async def apply_metadata_variations(
    df: pd.DataFrame,
    metadata_variations: List[List[MetaDataVariation]],
    window_start_idx: int,
    window_inclusive_end_idx: int,
    window_size: int,
) -> pd.DataFrame:
    """
    Apply metadata variations to the dataframe.

    Note: we proactively reset the window_idx to be 0:N_windows_idxs

    Inputs:
        df: pd.DataFrame -> must include column: window_idx
        metadata_variations: List[List[MetadataVariation]]
        window_start_idx: int
        window_inclusive_end_idx: int
        window_size: int
        structure of metadata_variations:
        class MetaDataVariation(BaseModel):
            name: str
            value: Union[str, int, float]
            perturbation_or_exact_value: Literal["perturbation", "exact_value"]
            perturbation_type: Optional[PerturbationType] = None


    Returns:
        pd.DataFrame: Dataframe with metadata variations applied
    """
    if len(metadata_variations) == 0:
        return df

    # reset the window_idx so we can use window_start_idx and window_inclusive_end_idx
    df.loc[:, "window_idx"] = np.repeat(
        np.arange(int(len(df) / window_size)), window_size
    )

    # Sort the window_idxs in the dataframe and select the first
    df = df[
        df["window_idx"].isin(
            range(window_start_idx, window_inclusive_end_idx + 1)
        )
    ]

    # loop over the metadata_variations and insert the new variations into the dataframe
    # modify the window_idx to be the window_idx + the index of the metadata_variation

    df_list = []

    for metadata_variation_idx, metadata_variation_list in enumerate(
        metadata_variations
    ):
        df_tmp = df.copy()
        # loop over the metadata_variation
        for variation in metadata_variation_list:
            # apply the variation to the dataframe
            df_tmp = await apply_metadata_variation(df_tmp, variation)
        df_list.append(df_tmp)

    # TODO later - figure out window_idx for the final dataframe

    final_df = pd.concat(df_list)
    final_df["window_idx"] = np.repeat(
        np.arange(int(len(final_df) / window_size)), window_size
    )

    final_df = final_df.reset_index(drop=True)

    return final_df


def get_train_config_file_name(
    dataset_name: str,
    task: str,
    job_name: Optional[str] = None,
) -> str:
    """
    Generates the standardized training configuration file name.
    Inputs:
        dataset_name: str
        job_name: Optional[str]
        task: str ("forecast" or "synthesis")
    Returns:
        str: The standardized training configuration file name
    """
    task_type_str = "forecasting" if task == "forecast" else "synthesis"

    job_part = f"_{job_name}" if job_name else ""
    return f"config_{dataset_name}{job_part}_{task_type_str}.yaml"


def detect_time_frequency(
    timestamps: Union[
        pd.Series,
        pd.DatetimeIndex,
        Sequence[Union[pd.Timestamp, datetime, np.datetime64, str]],
    ],
) -> Optional[TimeFrequency]:
    """
    Detect the time frequency of the data based on pandas frequency inference.
    Falls back to most common interval detection for irregular data.

    Args:
        timestamps: Series, DatetimeIndex, or sequence of timestamps to analyze

    Returns:
        TimeFrequency: Detected frequency (e.g., TimeFrequency(value=1, unit='day'), TimeFrequency(value=15, unit='minute'), etc.)
                      or None if frequency cannot be determined
    """

    try:
        # Convert to pandas Series if it's a sequence or DatetimeIndex
        if isinstance(
            timestamps, (Sequence, pd.DatetimeIndex)
        ) and not isinstance(timestamps, (str, pd.Series)):
            timestamps = pd.Series(timestamps)
        elif not isinstance(timestamps, pd.Series):
            # Convert single values or other types to Series
            timestamps = pd.Series([timestamps])

        # If empty, return None
        if len(timestamps) == 0:
            return None

        try:
            # First try to parse with default parser
            parsed_timestamps = pd.to_datetime(timestamps, errors="coerce")

            # Ensure parsed_timestamps is always a Series for consistent API
            if not isinstance(parsed_timestamps, pd.Series):
                parsed_timestamps = pd.Series(parsed_timestamps)

            # If we have any NaT values, try individual parsing (pandas often fails on mixed precision collectively)
            if parsed_timestamps.isna().any():
                # Try parsing each timestamp individually to handle mixed precision
                individual_parsed = []
                for ts in timestamps:
                    try:
                        individual_parsed.append(
                            pd.to_datetime(ts, errors="coerce")
                        )
                    except (ValueError, TypeError, pd.errors.ParserError):
                        individual_parsed.append(pd.NaT)
                parsed_timestamps = pd.Series(individual_parsed)

                # If still have NaT values, try cleaning separators
                if (
                    parsed_timestamps.isna().any()
                    and hasattr(timestamps, "dtype")
                    and timestamps.dtype == "object"
                ):
                    cleaned_dates = (
                        timestamps.astype(str)
                        .str.replace("/", "-")
                        .str.replace(
                            r"(\d{4}-\d{2}-\d{2})-(\d{2}:\d{2}:\d{2})",
                            r"\1 \2",
                            regex=True,
                        )  # Fix corrupted dates
                    )
                    parsed_timestamps = pd.to_datetime(
                        cleaned_dates, errors="coerce"
                    )
        except Exception:
            logger.exception("Error converting timestamps to datetime")
            return None

        # If any timestamp is still invalid (NaT), return None
        if parsed_timestamps.isna().any():
            return None

        # Sort by timestamp
        unique_timestamps = parsed_timestamps.drop_duplicates().sort_values()

        # Use pandas' built-in frequency inference
        freq = pd.infer_freq(unique_timestamps)

        # Map pandas frequency strings to human-readable format
        freq_map = {
            "A": "year",
            "A-DEC": "year",
            "A-JAN": "year",
            "Q": "quarter",
            "Q-DEC": "quarter",
            "Q-JAN": "quarter",
            "M": "month",
            "W": "week",
            "W-SUN": "week",
            "W-MON": "week",
            "D": "day",
            "H": "hour",
            "T": "minute",
            "min": "minute",
            "S": "second",
            "ms": "millisecond",
            "us": "microsecond",
            "ns": "nanosecond",
        }

        if freq:
            # Extract numeric part and unit from frequency string
            # Handle patterns like "3H", "15T", "2D", or just "H", "T", "D"
            match = re.match(r"(\d+)?([A-Za-z]+)(?:-[A-Za-z]+)?", freq)
            if match:
                # Get numeric part (default to 1 if not specified)
                numeric_part = int(match.group(1)) if match.group(1) else 1
                # Get the base frequency unit
                base_freq = match.group(2)
                unit = freq_map.get(base_freq)
                if unit:
                    return TimeFrequency(value=numeric_part, unit=unit)

            # Fallback: try exact match for complex frequencies
            if freq in freq_map:
                return TimeFrequency(value=1, unit=freq_map[freq])

        # If pandas inference failed, fall back to most common interval detection
        # This handles irregular data like the forecast endpoint does
        logger.debug(
            "Pandas frequency inference failed, trying most common interval detection"
        )
        return _detect_frequency_from_mode(unique_timestamps)

    except Exception:
        logger.exception("Error detecting frequency")
        return None


def _detect_frequency_from_mode(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
) -> Optional[TimeFrequency]:
    """
    Detect frequency by finding the most common time interval.
    Fallback for when pandas frequency inference fails on irregular data.

    Args:
        timestamps: Series or DatetimeIndex of datetime timestamps (should be sorted and unique)

    Returns:
        TimeFrequency: Frequency object or None
    """
    try:
        # Convert to Series if DatetimeIndex to ensure consistent .diff() behavior
        if isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.Series(timestamps)

        # Calculate time differences and find most common interval
        time_diffs = timestamps.diff().dropna()
        if len(time_diffs) == 0:
            return None

        # Use mode to find most common interval (same as _calculate_time_delta)
        # Convert to seconds - time_diffs is a Series of Timedelta objects
        # Use the .dt accessor to get total_seconds
        time_diffs_seconds = time_diffs.dt.total_seconds()

        # Round to microseconds for consistent mode detection
        rounded_seconds = np.round(time_diffs_seconds, MICROSECOND_PRECISION)
        mode_diff_seconds = stats.mode(rounded_seconds, keepdims=False)[0]

        # Convert to human-readable frequency with numeric part
        if mode_diff_seconds >= 31536000:  # >= 1 year
            years = int(round(mode_diff_seconds / 31536000))
            return TimeFrequency(value=years, unit="year")
        elif mode_diff_seconds >= 2629746:  # >= 1 month
            months = int(round(mode_diff_seconds / 2629746))
            return TimeFrequency(value=months, unit="month")
        elif mode_diff_seconds >= 604800:  # >= 1 week
            weeks = int(round(mode_diff_seconds / 604800))
            return TimeFrequency(value=weeks, unit="week")
        elif mode_diff_seconds >= 86400:  # >= 1 day
            days = int(round(mode_diff_seconds / 86400))
            return TimeFrequency(value=days, unit="day")
        elif mode_diff_seconds >= 3600:  # >= 1 hour
            hours = int(round(mode_diff_seconds / 3600))
            return TimeFrequency(value=hours, unit="hour")
        elif mode_diff_seconds >= 60:  # >= 1 minute
            minutes = int(round(mode_diff_seconds / 60))
            return TimeFrequency(value=minutes, unit="minute")
        elif mode_diff_seconds >= 1:  # >= 1 second
            seconds = int(round(mode_diff_seconds))
            return TimeFrequency(value=seconds, unit="second")
        else:  # < 1 second (milliseconds)
            milliseconds = int(round(mode_diff_seconds * 1000))
            if milliseconds == 0:
                milliseconds = 1  # Ensure we don't return 0 milliseconds
            return TimeFrequency(value=milliseconds, unit="millisecond")

    except Exception:
        logger.exception("Error in fallback frequency detection")
        return None


def format_timestamp_with_optional_fractional_seconds(
    timestamp: pd.Timestamp,
) -> str:
    """
    Format a pandas timestamp to ISO format string with optional fractional seconds.

    This function preserves fractional second precision when present, including
    both milliseconds and microseconds, but doesn't add ".000000" when there
    are no fractional seconds.

    Args:
        timestamp: pandas Timestamp object

    Returns:
        ISO formatted string with optional fractional seconds

    Examples:
        "2010-02-05T00:00:00"         -> "2010-02-05T00:00:00"
        "2010-02-05T00:00:00.142000"  -> "2010-02-05T00:00:00.142"
        "2010-02-05T00:00:00.142567"  -> "2010-02-05T00:00:00.142567"
    """
    # Convert to datetime object if it's a pandas Timestamp
    if hasattr(timestamp, "to_pydatetime"):
        dt = timestamp.to_pydatetime()
    else:
        dt = timestamp

    # Get the base ISO format without microseconds
    base_format = dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Check if there are microseconds
    if dt.microsecond > 0:
        # Format microseconds and remove trailing zeros
        microseconds_str = f".{dt.microsecond:06d}".rstrip("0")
        return base_format + microseconds_str
    else:
        return base_format
