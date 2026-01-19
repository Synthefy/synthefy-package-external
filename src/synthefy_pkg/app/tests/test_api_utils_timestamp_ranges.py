import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from synthefy_pkg.app.utils.timestamp_utils import get_timestamp_range


class TestGetTimestampRange:
    """
    Tests for the get_timestamp_range utility function.

    Note: This test uses an isolated copy of the function in timestamp_range_util.py
    to avoid database connection dependency issues during testing. The implementation
    is identical to the one in api_utils.py.
    """

    @pytest.fixture
    def datetime_df(self):
        """DataFrame with a column of datetime objects."""
        dates = [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
            datetime(2022, 1, 4),
            datetime(2022, 1, 5),
        ]
        return pd.DataFrame({"timestamp": dates, "value": range(5)})

    @pytest.fixture
    def string_df(self):
        """DataFrame with a column of string timestamps."""
        dates = [
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
        ]
        return pd.DataFrame({"timestamp": dates, "value": range(5)})

    @pytest.fixture
    def mixed_df(self):
        """DataFrame with a mixed column of datetime and strings."""
        dates = [
            datetime(2022, 1, 1),
            "2022-01-02",
            datetime(2022, 1, 3),
            "2022-01-04",
            datetime(2022, 1, 5),
        ]
        return pd.DataFrame({"timestamp": dates, "value": range(5)})

    @pytest.fixture
    def df_with_nulls(self):
        """DataFrame with null values in timestamp column."""
        dates = [
            datetime(2022, 1, 1),
            None,
            datetime(2022, 1, 3),
            None,
            datetime(2022, 1, 5),
        ]
        return pd.DataFrame({"timestamp": dates, "value": range(5)})

    @pytest.fixture
    def df_all_nulls(self):
        """DataFrame with all null values in timestamp column."""
        dates = [None, None, None]
        return pd.DataFrame({"timestamp": dates, "value": range(3)})

    @pytest.fixture
    def df_non_null_becomes_null(self):
        """DataFrame where non-null values become null after conversion."""
        # These values will all fail to convert to datetime
        dates = ["invalid date", "not a timestamp", "unparseable"]
        return pd.DataFrame({"timestamp": dates, "value": range(3)})

    def test_datetime_column(self, datetime_df):
        """Test with a column that's already in datetime format."""
        min_time, max_time = get_timestamp_range(datetime_df, "timestamp")
        assert min_time == datetime(2022, 1, 1)
        assert max_time == datetime(2022, 1, 5)

    def test_string_column(self, string_df):
        """Test with a column of string timestamps that need conversion."""
        min_time, max_time = get_timestamp_range(string_df, "timestamp")
        assert min_time == pd.Timestamp("2022-01-01")
        assert max_time == pd.Timestamp("2022-01-05")

    def test_string_column_no_conversion(self, string_df):
        """Test with a column of string timestamps with conversion disabled."""
        min_time, max_time = get_timestamp_range(
            string_df, "timestamp", convert_to_datetime=False
        )
        assert min_time == "2022-01-01"
        assert max_time == "2022-01-05"

    def test_mixed_column(self, mixed_df):
        """Test with a mixed column that pandas can automatically convert."""
        min_time, max_time = get_timestamp_range(mixed_df, "timestamp")
        assert min_time == pd.Timestamp("2022-01-01")
        assert max_time == pd.Timestamp("2022-01-05")

    def test_column_not_found(self, datetime_df):
        """Test with a column that doesn't exist in the dataframe."""
        with pytest.raises(
            ValueError, match="Column 'not_a_column' not found in dataframe"
        ):
            get_timestamp_range(datetime_df, "not_a_column")

    def test_with_nulls(self, df_with_nulls):
        """Test with a column containing some null values."""
        min_time, max_time = get_timestamp_range(df_with_nulls, "timestamp")
        assert min_time == datetime(2022, 1, 1)
        assert max_time == datetime(2022, 1, 5)

    def test_all_nulls(self, df_all_nulls):
        """Test with a column containing only null values."""
        with pytest.raises(
            ValueError, match="Column 'timestamp' contains only null values"
        ):
            get_timestamp_range(df_all_nulls, "timestamp")

    def test_empty_dataframe(self):
        """Test with an empty dataframe."""
        empty_df = pd.DataFrame({"timestamp": []})
        with pytest.raises(ValueError, match="Column 'timestamp' is empty"):
            get_timestamp_range(empty_df, "timestamp")

    def test_non_orderable_values(self):
        """Test with values that can't be ordered."""
        df = pd.DataFrame({"timestamp": [{"a": 1}, {"b": 2}, {"c": 3}]})
        with pytest.raises(
            ValueError, match="Failed to convert column 'timestamp' to datetime"
        ):
            get_timestamp_range(df, "timestamp")

    def test_timezone_awareness(self):
        """Test with timezone-aware datetimes."""
        dates = [
            pd.Timestamp("2022-01-01", tz="UTC"),
            pd.Timestamp("2022-01-02", tz="UTC"),
            pd.Timestamp("2022-01-03", tz="UTC"),
        ]
        df = pd.DataFrame({"timestamp": dates})

        # Use a separate test dataframe with all timezone-aware timestamps
        min_time, max_time = get_timestamp_range(df, "timestamp")

        # Ensure the result timestamps have timezone info
        assert min_time.tzinfo is not None
        assert max_time.tzinfo is not None

        # Check the values (using timezone-aware comparisons)
        assert min_time == pd.Timestamp("2022-01-01", tz="UTC")
        assert max_time == pd.Timestamp("2022-01-03", tz="UTC")
