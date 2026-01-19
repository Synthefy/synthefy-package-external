import os
from unittest.mock import patch

import pytest
from loguru import logger

from synthefy_pkg.app.utils.llm_utils import (
    MetaDataToParse,
    OneContinuousMetaData,
    OneDiscreteMetaData,
    extract_metadata_from_query,
    validate_metadata,
)

skip_in_ci_llm = pytest.mark.skipif(
    os.environ.get("SKIP_LLM_TESTS") == "true",
    reason="Test skipped based on environment variable",
)


# Sample data for testing
timeseries_columns = ["timeseries_1", "packet_loss"]
continuous_columns = ["temperature", "pressure", "humidity"]
discrete_columns = ["device_status", "error_code", "priority_levels", "status"]
timestamps_col = ["timestamp"]


# def test_extract_metadata_from_query_valid():
#     query = (
#         "Show me a forecast for timeseries_1 with device_status with values ['active', 'inactive'], "
#         "error_codes with values [100, 101, 102], priority_levels with values ['high', 'medium', 'low'], "
#         "and temperature = 22.5, 23.0, 21.8."
#     )

#     metadata = extract_metadata_from_query(
#         query,
#         timeseries_columns=timeseries_columns,
#         continuous_columns=continuous_columns,
#         discrete_columns=discrete_columns,
#         timestamps_col=timestamps_col,
#     )

#     assert isinstance(metadata, MetaDataToParse)
#     assert len(metadata.continuous_conditions) == 1
#     assert any(
#         cond.name == "temperature" and cond.values == [22.5, 23.0, 21.8]
#         for cond in metadata.continuous_conditions
#     )

#     assert len(metadata.discrete_conditions) == 3
#     assert any(
#         cond.name == "device_status" and cond.values == ["active", "inactive"]
#         for cond in metadata.discrete_conditions
#     )
#     assert any(
#         cond.name == "error_code" and cond.values == [100, 101, 102]
#         for cond in metadata.discrete_conditions
#     )
#     assert any(
#         cond.name == "priority_levels"
#         and cond.values == ["high", "medium", "low"]
#         for cond in metadata.discrete_conditions
#     )

#     assert (
#         len(metadata.timestamps) == 0
#     )  # Assuming timestamps are not set in this query
#     assert metadata.num_examples == 1


@skip_in_ci_llm
def test_extract_metadata_from_query_multiple_examples():
    query = (
        "Synthesize 5 examples of packet_loss with temperature values [22.5, 23.0, 21.8] "
        "and status values ['ok', 'fail']."
    )

    metadata = extract_metadata_from_query(
        query,
        timeseries_columns=timeseries_columns,
        continuous_columns=continuous_columns,
        discrete_columns=discrete_columns,
        timestamps_col=timestamps_col,
    )

    assert isinstance(metadata, MetaDataToParse)
    assert len(metadata.continuous_conditions) == 1
    assert any(
        cond.name == "temperature" and cond.values == [22.5, 23.0, 21.8]
        for cond in metadata.continuous_conditions
    ), (
        f"Expected continuous conditions to include 'temperature' with values [22.5, 23.0, 21.8], but got {metadata.continuous_conditions}"
    )

    assert len(metadata.discrete_conditions) == 1
    assert any(
        cond.name == "status" and cond.values == ["ok", "fail"]
        for cond in metadata.discrete_conditions
    ), (
        f"Expected discrete conditions to include 'status' with values ['ok', 'fail'], but got {metadata.discrete_conditions}"
    )

    assert (
        len(metadata.timestamps) == 0
    )  # Assuming timestamps are not set in this query
    assert metadata.num_examples == 5


# def test_extract_metadata_from_query_with_timestamps():
#     query = (
#         "Generate 3 examples for timeseries_2 with humidity values [45, 50, 55] and status values ['active'], "
#         "including timestamps."
#     )

#     metadata = extract_metadata_from_query(
#         query,
#         timeseries_columns=timeseries_columns,
#         continuous_columns=continuous_columns,
#         discrete_columns=discrete_columns,
#         timestamps_col=timestamps_col,
#     )

#     assert isinstance(metadata, MetaDataToParse)
#     assert len(metadata.continuous_conditions) == 1
#     assert metadata.continuous_conditions[0].name == "humidity"
#     assert metadata.continuous_conditions[0].values == [45, 50, 55]

#     assert len(metadata.discrete_conditions) == 1
#     assert metadata.discrete_conditions[0].name == "status"
#     assert metadata.discrete_conditions[0].values == ["active"]

#     assert len(metadata.timestamps) == 1
#     # Assuming the LLM returns standardized timestamp formats
#     assert all(
#         isinstance(ts.value, str) for ts in metadata.timestamps[0].values
#     )
#     assert metadata.num_examples == 3


def test_extract_metadata_from_query_invalid_query():
    query = "This is an invalid query that does not follow the expected format."

    metadata = extract_metadata_from_query(
        query,
        timeseries_columns=timeseries_columns,
        continuous_columns=continuous_columns,
        discrete_columns=discrete_columns,
        timestamps_col=timestamps_col,
    )

    # Expect default MetaData due to parsing error
    assert isinstance(metadata, MetaDataToParse)
    assert metadata.continuous_conditions == []
    assert metadata.discrete_conditions == []
    assert metadata.timestamps == []
    assert metadata.num_examples == 1


def test_extract_metadata_from_query_empty_query():
    query = ""

    metadata = extract_metadata_from_query(
        query,
        timeseries_columns=timeseries_columns,
        continuous_columns=continuous_columns,
        discrete_columns=discrete_columns,
        timestamps_col=timestamps_col,
    )

    # Expect default MetaData as no query provided
    assert isinstance(metadata, MetaDataToParse)
    assert metadata.continuous_conditions == []
    assert metadata.discrete_conditions == []
    assert metadata.timestamps == []
    assert metadata.num_examples == 1


def test_validate_metadata_valid_conditions():
    metadata = MetaDataToParse(
        continuous_conditions=[
            OneContinuousMetaData(name="temperature", values=[22.5, 23.0]),
            OneContinuousMetaData(name="humidity", values=[45.0, 50.0]),
        ],
        discrete_conditions=[
            OneDiscreteMetaData(name="status", values=["active", "inactive"]),
            OneDiscreteMetaData(name="error_code", values=[100, 101]),
        ],
    )
    continuous_columns = ["temperature", "humidity", "pressure"]
    discrete_columns = ["status", "error_code", "priority"]

    validate_metadata(metadata, continuous_columns, discrete_columns)

    assert len(metadata.continuous_conditions) == 2
    assert len(metadata.discrete_conditions) == 2


def test_validate_metadata_remove_invalid_conditions():
    metadata = MetaDataToParse(
        continuous_conditions=[
            OneContinuousMetaData(name="temperature", values=[22.5, 23.0]),
            OneContinuousMetaData(name="invalid_continuous", values=[1.0, 2.0]),
        ],
        discrete_conditions=[
            OneDiscreteMetaData(name="status", values=["active", "inactive"]),
            OneDiscreteMetaData(name="invalid_discrete", values=["a", "b"]),
        ],
    )
    continuous_columns = ["temperature", "humidity"]
    discrete_columns = ["status", "error_code"]

    validate_metadata(metadata, continuous_columns, discrete_columns)

    assert len(metadata.continuous_conditions) == 1
    assert metadata.continuous_conditions[0].name == "temperature"
    assert len(metadata.discrete_conditions) == 1
    assert metadata.discrete_conditions[0].name == "status"


def test_validate_metadata_all_invalid_conditions():
    metadata = MetaDataToParse(
        continuous_conditions=[
            OneContinuousMetaData(
                name="invalid_continuous1", values=[1.0, 2.0]
            ),
            OneContinuousMetaData(
                name="invalid_continuous2", values=[3.0, 4.0]
            ),
        ],
        discrete_conditions=[
            OneDiscreteMetaData(name="invalid_discrete1", values=["a", "b"]),
            OneDiscreteMetaData(name="invalid_discrete2", values=[1, 2]),
        ],
    )
    continuous_columns = ["temperature", "humidity"]
    discrete_columns = ["status", "error_code"]

    validate_metadata(metadata, continuous_columns, discrete_columns)

    assert len(metadata.continuous_conditions) == 0
    assert len(metadata.discrete_conditions) == 0


def test_validate_metadata_empty_conditions():
    metadata = MetaDataToParse()
    continuous_columns = ["temperature", "humidity"]
    discrete_columns = ["status", "error_code"]

    validate_metadata(metadata, continuous_columns, discrete_columns)

    assert len(metadata.continuous_conditions) == 0
    assert len(metadata.discrete_conditions) == 0
