import copy
import os
import sys
from collections import Counter
from datetime import datetime

import pandas as pd
import pytest

from synthefy_pkg.loaders.randomized_loader import RandomLoader


@pytest.fixture
def rand_loader():
    return RandomLoader("")  # You'll need to provide a valid config path


def test_rename_index(rand_loader):
    loader = rand_loader

    # Test with DatetimeIndex
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)
    result = rand_loader._rename_index(df)

    assert "timestamp" in result.columns
    assert isinstance(result.index, pd.RangeIndex)
    assert (result["timestamp"] == dates).all()

    # Test with non-DatetimeIndex
    df = pd.DataFrame({"value": [1, 2, 3]})
    result = loader._rename_index(df)

    assert "timestamp" not in result.columns
    assert result.equals(df)


def test_get_column_statistics(rand_loader):
    loader = rand_loader

    # Create test DataFrame with timestamp column
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({"value": [1, 2, 3], "timestamp": dates})

    stats = loader._get_column_statistics(df)

    # Test return value structure and content
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"prefill_length", "start_time", "end_time"}
    assert all(isinstance(v, str) for v in stats.values())

    # Test actual values
    assert stats["prefill_length"] == "3"
    assert stats["start_time"] == str(dates[0])
    assert stats["end_time"] == str(dates[-1])

    # Test with empty DataFrame
    empty_df = pd.DataFrame({"timestamp": []})
    stats = loader._get_column_statistics(empty_df)
    assert stats["prefill_length"] == "0"
    assert stats["start_time"] == str(pd.Timestamp.min)
    assert stats["end_time"] == str(pd.Timestamp.min)

    # Test with DataFrame missing timestamp column
    invalid_df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(KeyError):
        loader._get_column_statistics(invalid_df)


def test_rename_timestamp_duplicate_column(rand_loader):
    loader = rand_loader

    # Create test DataFrame
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({"value": [1, 2, 3], "timestamp": dates})

    # Create test metadata
    meta = {
        "columns": [
            {
                "column_id": "value",
                "title": "Test Values",
                "description": "Test data",
            }
        ],
        "timestamps_columns": ["timestamp"],
    }

    # Test with seen_count = 1
    df_result, meta_result = loader._rename_timestamp_duplicate_column(
        df.copy(), meta.copy(), 1
    )

    # Check DataFrame renaming
    assert "value_1" in df_result.columns
    assert "value_1_timestamp" in df_result.columns
    assert "timestamp" not in df_result.columns

    # Check metadata updates
    assert len(meta_result["columns"]) == 2
    timestamp_meta = meta_result["columns"][1]
    assert timestamp_meta["column_id"] == "value_1_timestamp"
    assert timestamp_meta["type"] == "timestamp"
    assert timestamp_meta["prefill_length"] == "3"
    assert timestamp_meta["start_time"] == str(dates[0])
    assert timestamp_meta["end_time"] == str(dates[-1])

    # Test with non-dict metadata columns
    meta_non_dict = {"columns": ["value"], "timestamps_columns": ["timestamp"]}
    df_result, meta_result = loader._rename_timestamp_duplicate_column(
        df.copy(), meta_non_dict.copy(), 2
    )
    assert len(meta_result["columns"]) == 2

    # Test error cases
    with pytest.raises(KeyError):
        loader._rename_timestamp_duplicate_column(df, {"wrong_key": []}, 1)

    with pytest.raises(IndexError):
        loader._rename_timestamp_duplicate_column(
            pd.DataFrame(), {"columns": []}, 1
        )


def test_load_subset(rand_loader):
    loader = rand_loader

    # Mock the data_generator attribute that would normally come from SimpleLoader
    dates = pd.date_range("2023-01-01", periods=3)
    mock_data = [
        (
            "dataset1",
            pd.DataFrame({"value1": [1, 2, 3], "timestamp": dates}),
            {
                "columns": [{"column_id": "value1"}],
                "timestamps_columns": ["timestamp"],
            },
        ),
        (
            "dataset2",
            pd.DataFrame({"value2": [4, 5, 6], "timestamp": dates}),
            {
                "columns": [{"column_id": "value2"}],
                "timestamps_columns": ["timestamp"],
            },
        ),
    ]

    data_generator = mock_data

    result = loader._load_subset(data_generator)

    # Check the structure of the result
    assert len(result) == 2
    assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    # Check that each DataFrame has a timestamp column
    for _, df, _ in result:
        assert "timestamp" in df.columns
        assert isinstance(df.index, pd.RangeIndex)

    # Test with empty inputs
    empty_result = loader._load_subset([])
    assert len(empty_result) == 0


def test_merge_variate(rand_loader):
    loader = rand_loader

    # Create initial result DataFrame and metadata
    dates = pd.date_range("2023-01-01", periods=5)
    result_df = pd.DataFrame(
        {"existing_value": [1, 2, 3, 4, 5], "existing_value_1_timestamp": dates}
    )
    result_meta = {
        "columns": [
            {
                "column_id": "existing_value",
                "title": "Existing Values",
                "description": "Initial data",
            },
            {
                "column_id": "existing_value_1_timestamp",
                "title": "Timestamps",
                "type": "timestamp",
            },
        ],
        "timestamps_columns": ["timestamp"],
    }

    # Create new data to merge
    new_dates = pd.date_range("2023-01-02", periods=3)  # Shorter series
    new_data = pd.DataFrame({"new_value": [10, 20, 30], "timestamp": new_dates})
    new_meta = {
        "columns": [
            {
                "column_id": "new_value",
                "title": "New Values",
                "description": "New data",
            }
        ],
        "timestamps_columns": ["timestamp"],
    }

    series_seen_count = Counter()

    # Test the merge
    result_df, result_meta = loader._merge_variate(
        result_df,
        result_meta,
        "new_series",
        new_data,
        new_meta,
        series_seen_count,
    )

    # Test DataFrame structure
    assert len(result_df) == 5  # Should maintain original length
    assert "new_value_1" in result_df.columns
    assert "new_value_1_timestamp" in result_df.columns

    # Test forward fill behavior
    assert (
        result_df["new_value_1"].iloc[3] == result_df["new_value_1"].iloc[2]
    )  # Should forward fill

    # Test metadata structure
    assert (
        len(result_meta["columns"]) == 4
    )  # Original 2 + new value + new timestamp
    new_columns = [col["column_id"] for col in result_meta["columns"]]
    assert "new_value_1" in new_columns
    assert "new_value_1_timestamp" in new_columns

    # Test multiple merges (duplicate handling)
    second_data = pd.DataFrame(
        {"new_value": [40, 50, 60], "timestamp": new_dates}
    )

    result_df, result_meta = loader._merge_variate(
        result_df,
        result_meta,
        "new_series",
        second_data,
        copy.deepcopy(new_meta),
        series_seen_count,
    )

    # Test duplicate column handling
    assert "new_value_2" in result_df.columns
    assert "new_value_2_timestamp" in result_df.columns
    assert len(result_meta["columns"]) == 6  # Should now have 6 columns total

    # Test error cases
    with pytest.raises(KeyError):
        # Test with missing timestamp column
        bad_data = pd.DataFrame({"value": [1, 2, 3]})
        loader._merge_variate(
            result_df,
            result_meta,
            "bad_series",
            bad_data,
            {"columns": [{"column_id": "value"}]},
            series_seen_count,
        )

    # Test with empty DataFrame
    empty_data = pd.DataFrame({"value": [], "timestamp": []})
    empty_meta = {"columns": [{"column_id": "value"}], "timestamps_columns": []}
    result_df_empty, result_meta_empty = loader._merge_variate(
        result_df,
        copy.deepcopy(result_meta),
        "empty_series",
        empty_data,
        empty_meta,
        Counter(),
    )
    assert len(result_df_empty) == len(
        result_df
    )  # Should maintain original length

    # Test metadata column stats
    timestamp_meta = next(
        col
        for col in result_meta["columns"]
        if col["column_id"] == "new_value_1_timestamp"
    )
    assert "start_time" in timestamp_meta
    assert "end_time" in timestamp_meta
    assert "prefill_length" in timestamp_meta


def test_merge_variate_edge_cases(rand_loader):
    loader = rand_loader

    # Test with single-row DataFrames
    dates = pd.date_range("2023-01-01", periods=1)
    result_df = pd.DataFrame({"value": [1], "value_1_timestamp": dates})
    result_meta = {
        "columns": [{"column_id": "value"}],
        "timestamps_columns": ["timestamp"],
    }

    new_data = pd.DataFrame({"new_value": [10], "timestamp": dates})
    new_meta = {
        "columns": [{"column_id": "new_value"}],
        "timestamps_columns": ["timestamp"],
    }

    result_df, result_meta = loader._merge_variate(
        result_df, result_meta, "new_series", new_data, new_meta, Counter()
    )

    assert len(result_df) == 1
    assert "new_value_1" in result_df.columns
