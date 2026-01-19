import os
import time
from multiprocessing import Pool
from random import Random
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from synthefy_pkg.app.utils.smart_approximation_utils import (
    DEFAULT_NULL_THRESHOLD,
    SmartApproximationUtils,
)


class TestSmartApproximationUtils:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe with various column types for testing."""
        # Create a dataframe with datetime, numeric, and string columns
        df = pd.DataFrame(
            {
                "datetime_col": pd.date_range(
                    start="2023-01-01", periods=100, freq="H"
                ),
                "datetime_with_nulls": pd.date_range(
                    start="2023-01-01", periods=100, freq="H"
                ),
                "numeric_col": range(100),
                "random_numeric": [
                    Random(42).random() for _ in range(100)
                ],  # 100 reproducible random values with seed 42
                "string_col": ["value_" + str(i) for i in range(100)],
                "mixed_col": ["2023-01-01"] * 50
                + list(range(50)),  # Mixed types
                "high_null_datetime": pd.date_range(
                    start="2023-01-01", periods=100, freq="H"
                ),
                "non_monotonic_datetime": pd.date_range(
                    start="2023-01-01", periods=100, freq="H"
                ).tolist(),
            }
        )

        # Add nulls to some columns
        df.loc[10:20, "datetime_with_nulls"] = None
        df.loc[0:90, "high_null_datetime"] = None  # 90% nulls

        # Make the non-monotonic datetime column non-monotonic
        df["non_monotonic_datetime"] = (
            df["non_monotonic_datetime"].sample(frac=1).values
        )

        return df

    def test_check_column_for_timestamp_index_valid(self, sample_dataframe):
        """Test check_column_for_timestamp_index with a valid datetime column."""
        col_name = "datetime_col"
        series = sample_dataframe[col_name]
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result == col_name
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_with_nulls(
        self, sample_dataframe
    ):
        """Test check_column_for_timestamp_index with a datetime column containing some nulls."""
        col_name = "datetime_with_nulls"
        series = sample_dataframe[col_name]
        null_threshold = 0.2  # Allow up to 20% nulls

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is not None
        assert result == col_name
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_high_nulls(
        self, sample_dataframe
    ):
        """Test check_column_for_timestamp_index with a datetime column containing too many nulls."""
        col_name = "high_null_datetime"
        series = sample_dataframe[col_name]
        null_threshold = 0.1  # Only allow 10% nulls

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_numeric_skip(
        self, sample_dataframe
    ):
        """Test that numeric columns are skipped to prevent epoch timestamp misinterpretation."""
        col_name = "numeric_col"
        series = sample_dataframe[col_name]
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_float_skip(self):
        """Test that float columns are skipped."""
        col_name = "float_col"
        series = pd.Series([1.5, 2.5, 3.5])
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_string_dates(self):
        """Test conversion of string dates to datetime."""
        col_name = "string_dates"
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result == col_name
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_invalid_strings(self):
        """Test handling of non-datetime strings."""
        col_name = "invalid_strings"
        series = pd.Series(
            ["not a date", "also not a date", "still not a date"]
        )
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_mixed_strings(self):
        """Test handling of mixed valid/invalid datetime strings."""
        col_name = "mixed_strings"
        series = pd.Series(["2023-01-01", "not a date", "2023-01-03"])
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_null_threshold(self):
        """Test different null thresholds."""
        col_name = "datetime_with_nulls"
        series = pd.Series(
            [
                pd.Timestamp("2023-01-01"),
                None,
                pd.Timestamp("2023-01-03"),
                None,
                pd.Timestamp("2023-01-05"),
            ]
        )

        # With 40% nulls (2/5)
        # Should pass with 0.5 threshold
        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, 0.5)
            )
        )
        assert result == col_name
        assert len(messages) > 0

        # Should fail with 0.3 threshold
        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, 0.3)
            )
        )
        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_various_datetime_formats(self):
        """Test handling of various datetime string formats."""
        col_name = "various_formats"
        series = pd.Series(
            [
                "2023-01-01",  # ISO format
                "01/02/2023",  # US format
                "2023-01-03 14:30:00",  # ISO with time
                "January 4, 2023",  # Natural language
                "2023-01-05T16:30:00Z",  # ISO with timezone
            ]
        )
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None  # Can't handle diff types
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_already_datetime(self):
        """Test handling of series that's already datetime type."""
        col_name = "datetime_series"
        series = pd.Series(
            [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-03"),
            ]
        )
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result == col_name
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_error_handling(self):
        """Test error handling for problematic data."""
        col_name = "problematic_data"
        # Create a series that will cause issues during datetime conversion
        series = pd.Series(
            [1, "string", pd.Timestamp("2023-01-01")]
        )  # Mixed types
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_edge_cases(self):
        """Test edge cases and boundary conditions."""
        null_threshold = DEFAULT_NULL_THRESHOLD

        # Test with single value
        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                ("single", pd.Series(["2023-01-01"]), null_threshold)
            )
        )
        assert result == "single"
        assert len(messages) > 0

        # Test with very large timestamps
        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (
                    "future",
                    pd.Series(["2030-12-31", "2031-01-01"]),
                    null_threshold,
                )
            )
        )
        assert result == "future"
        assert len(messages) > 0

        # Test with very old timestamps
        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (
                    "past",
                    pd.Series(["1990-01-01", "1990-01-02"]),
                    null_threshold,
                )
            )
        )
        assert result == "past"
        assert len(messages) > 0

    def test_find_timestamp_index_candidates(self, sample_dataframe):
        """Test find_timestamp_index_candidates with a dataframe containing valid candidates."""
        # Call the method
        result, error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates(
                sample_dataframe, null_threshold=0.2
            )
        )

        # Verify the results - should include datetime columns that meet criteria
        assert "datetime_col" in result
        assert "datetime_with_nulls" in result

        # Verify columns that shouldn't be included
        assert "high_null_datetime" not in result
        assert "numeric_col" not in result
        assert "random_numeric" not in result
        assert "string_col" not in result
        assert "mixed_col" not in result
        assert "non_monotonic_datetime" in result

        # Verify we got the expected number of candidates
        assert len(result) == 3

    def test_find_timestamp_index_candidates_some_valid(self, sample_dataframe):
        """Test find_timestamp_index_candidates with a dataframe containing no valid candidates."""
        # Create a dataframe with some valid timestamp columns
        df_some_timestamps = pd.DataFrame(
            {
                "numeric_col1": range(100),
                "numeric_col2": range(100, 200),
                "string_col": ["value_" + str(i) for i in range(100)],
            }
        )

        # Call the method
        result, error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates(
                df_some_timestamps
            )
        )
        assert "numeric_col1" not in result
        assert "numeric_col2" not in result
        assert "string_col" not in result
        assert len(result) == 0

    def test_check_column_for_timestamp_index_empty_series(self):
        """Test check_column_for_timestamp_index with an empty series."""
        col_name = "empty_col"
        series = pd.Series([], dtype="object")
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_all_nulls(self):
        """Test check_column_for_timestamp_index with a series containing all nulls."""
        col_name = "all_nulls"
        series = pd.Series([None] * 100)
        null_threshold = DEFAULT_NULL_THRESHOLD

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result is None
        assert len(messages) > 0

    def test_check_column_for_timestamp_index_string_dates_non_unique(self):
        """Test check_column_for_timestamp_index with string dates that can be converted."""
        col_name = "string_dates"
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"] * 10)
        null_threshold = DEFAULT_NULL_THRESHOLD

        # This should return None because while function checks is_datetime64_any_dtype
        # and attempts to convert string dates, the resulting series does not have unique values
        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        assert result == "string_dates"
        assert len(messages) > 0

    @patch("synthefy_pkg.app.utils.smart_approximation_utils.Pool")
    def test_find_timestamp_index_candidates_exception_handling(
        self, mock_pool, sample_dataframe
    ):
        """Test find_timestamp_index_candidates handles exceptions properly."""
        # Setup the mock pool to raise an exception
        mock_pool.return_value.__enter__.side_effect = Exception(
            "Test exception"
        )

        # Call the method and expect it to handle the exception
        result, error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates_parallel(
                sample_dataframe, null_threshold=0.1
            )
        )

        # Should return an empty list when an exception occurs
        assert "datetime_col" in result
        assert "numeric_col" not in result
        assert "non_monotonic_datetime" in result
        assert len(result) == 2

    def test_find_timestamp_index_candidates_parallel(self, sample_dataframe):
        """Test find_timestamp_index_candidates_parallel with a dataframe containing valid candidates."""
        # Call the parallel method
        result, error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates_parallel(
                sample_dataframe, null_threshold=0.2
            )
        )

        # Verify the results - should include datetime columns that meet criteria
        assert "datetime_col" in result
        assert "datetime_with_nulls" in result

        # Verify columns that shouldn't be included
        assert "high_null_datetime" not in result
        assert "numeric_col" not in result
        assert "random_numeric" not in result
        assert "string_col" not in result
        assert "mixed_col" not in result
        assert "non_monotonic_datetime" in result

        # Verify we got the expected number of candidates
        assert len(result) == 3  # Now we should have 3 instead of 4

    def test_parallel_vs_sequential_results_match(self, sample_dataframe):
        """Test that parallel and sequential implementations return the same results."""
        parallel_result, parallel_error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates_parallel(
                sample_dataframe, null_threshold=0.2
            )
        )
        sequential_result, sequential_error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates(
                sample_dataframe, null_threshold=0.2
            )
        )

        # Results might be in different order, so convert to sets for comparison
        assert set(parallel_result) == set(sequential_result)

    @pytest.mark.skipif(
        os.environ.get("RUN_PERFORMANCE_TESTS") != "1",
        reason="Performance test - run only when RUN_PERFORMANCE_TESTS=1",
    )
    def test_performance_comparison(self, sample_dataframe, capsys):
        """
        Test the performance difference between parallel and sequential implementations.
        This test is marked as optional since it's primarily for benchmarking.
        """

        # Create a larger dataframe to better demonstrate performance differences
        large_df = pd.concat([sample_dataframe] * 10000, ignore_index=True)

        # Time the sequential version
        start_time = time.time()
        sequential_result, sequential_error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates(
                large_df, null_threshold=0.2
            )
        )
        sequential_time = time.time() - start_time

        # Time the parallel version
        start_time = time.time()
        parallel_result, parallel_error_messages = (
            SmartApproximationUtils.find_timestamp_index_candidates_parallel(
                large_df, null_threshold=0.2
            )
        )
        parallel_time = time.time() - start_time

        # Print performance metrics
        speedup = (
            sequential_time / parallel_time
            if parallel_time > 0
            else float("inf")
        )

        capsys.disabled()
        print("\nPerformance comparison:")
        print(f"Sequential time: {sequential_time:.4f} seconds")
        print(f"Parallel time: {parallel_time:.4f} seconds")
        print(f"Speedup factor: {speedup:.2f}x")

        # Verify results match
        assert set(sequential_result) == set(parallel_result)

    @pytest.fixture
    def categorical_test_df(self):
        """Create a sample dataframe with various column types for testing categorical detection."""
        return pd.DataFrame(
            {
                # Categorical columns (should be detected)
                "explicit_category": pd.Series(
                    ["A", "B", "C", "A", "B"], dtype="category"
                ),
                "low_cardinality_object": ["X", "Y", "Z", "X", "Y"],
                "low_cardinality_numeric": [1, 2, 3, 1, 2],
                # Not categorical (should be excluded)
                "high_cardinality_numeric": list(
                    range(5)
                ),  # Use only 5 values but still test separately with higher max_unique
                "high_cardinality_object": [f"val_{i}" for i in range(5)],
                "empty_column": [None, None, None, None, None],
                # Datetime-like columns (should be excluded)
                "datetime_col": pd.date_range("2023-01-01", periods=5),
                "date_strings": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
                "mixed_date_strings": [
                    "2023-01-01",
                    "2023-01-02",
                    "not a date",
                    "2023-01-04",
                    "2023-01-05",
                ],
                # Edge cases
                "numeric_looking_strings": ["123", "456", "789", "123", "456"],
                "mixed_types_low_card": [
                    1,
                    "2",
                    3,
                    1,
                    "2",
                ],  # Mixed types with low cardinality
            }
        )

    def test_get_improved_categorical_values_basic(self, categorical_test_df):
        """Test that get_categorical_features correctly identifies categorical columns."""
        result = SmartApproximationUtils.get_categorical_features(
            categorical_test_df
        )

        # Should include these columns
        assert "explicit_category" in result
        assert "low_cardinality_object" in result
        assert "low_cardinality_numeric" in result
        assert "numeric_looking_strings" in result
        assert "mixed_types_low_card" in result
        assert "high_cardinality_numeric" in result
        assert "high_cardinality_object" in result
        assert "date_strings" in result

        # Should exclude these columns
        assert "empty_column" not in result
        assert "datetime_col" not in result

        # Check actual values
        assert set(result["explicit_category"]) == {"A", "B", "C"}
        assert set(result["low_cardinality_object"]) == {"X", "Y", "Z"}
        assert set(result["low_cardinality_numeric"]) == {1, 2, 3}

    def test_get_improved_categorical_values_empty_df(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        result = SmartApproximationUtils.get_categorical_features(empty_df)
        assert result == {}  # Should return empty dict for empty dataframe

    def test_get_improved_categorical_values_max_unique(
        self, categorical_test_df
    ):
        """Test that max_unique_values parameter works correctly."""
        # Setting a lower threshold should exclude more columns
        result = SmartApproximationUtils.get_categorical_features(
            categorical_test_df, max_unique_values=2
        )

        # Columns with <= 2 unique values should be included
        assert "low_cardinality_numeric" not in result  # Has 3 unique values

        # Now create a new dataframe with more unique values for testing larger thresholds
        high_card_df = pd.DataFrame(
            {
                "high_cardinality_col": list(range(30))[
                    :5
                ],  # First 5 elements to keep length consistent
                "dummy_col": [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],  # Just to maintain DataFrame shape
            }
        )

        # Setting a higher threshold should include columns with many unique values
        result = SmartApproximationUtils.get_categorical_features(
            high_card_df, max_unique_values=30
        )

        # Column with high cardinality but under threshold should be included
        assert "high_cardinality_col" in result

    def test_get_improved_categorical_values_numeric_like(self):
        """Test handling of numeric-like string columns."""
        df = pd.DataFrame(
            {
                "all_numeric_strings": ["100", "200", "300", "400", "500"],
                "mostly_numeric_strings": [
                    "100",
                    "200",
                    "not_num",
                    "400",
                    "500",
                ],
            }
        )

        # With default settings, both should be treated as categorical
        # since they are object type with low cardinality
        result = SmartApproximationUtils.get_categorical_features(df)

        assert "all_numeric_strings" in result
        assert "mostly_numeric_strings" in result

        # Create a test case with high cardinality numeric strings
        df_high_card = pd.DataFrame(
            {
                "high_card_numeric_strings": [str(i) for i in range(30)][
                    :5
                ],  # First 5 elements
                "dummy_col": [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],  # Just to maintain DataFrame shape
            }
        )

        # All numeric should pass through if low cardinality
        result = SmartApproximationUtils.get_categorical_features(df_high_card)
        assert "high_card_numeric_strings" in result

        # But with a higher threshold and many more values, it should be excluded
        result = SmartApproximationUtils.get_categorical_features(
            df_high_card, max_unique_values=4
        )
        assert "high_card_numeric_strings" not in result

    def test_get_improved_categorical_values_mixed_nulls(self):
        """Test handling of columns with null values mixed in."""
        df = pd.DataFrame(
            {
                "with_nulls": ["A", None, "B", "C", None],
                "mostly_nulls": [None, None, None, "X", None],
                "all_nulls": [None, None, None, None, None],
            }
        )

        result = SmartApproximationUtils.get_categorical_features(df)

        # Column with some nulls should be included
        assert "with_nulls" in result
        assert set(result["with_nulls"]) == {"A", "B", "C"}

        # Column with mostly nulls but some valid values should be included
        assert "mostly_nulls" in result
        assert set(result["mostly_nulls"]) == {"X"}

        # Column with all nulls should be excluded (empty after dropping nulls)
        assert "all_nulls" not in result

    def test_check_column_for_timestamp_index_messages_format(self):
        """Test that check_column_for_timestamp_index returns properly formatted messages."""
        col_name = "test_col"
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        null_threshold = 0.1

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        # Verify result
        assert result == col_name

        # Verify messages format
        assert isinstance(messages, list)
        assert len(messages) > 0

        # Verify each message is a tuple with (level, message)
        for message in messages:
            assert isinstance(message, tuple)
            assert len(message) == 2
            level, msg_text = message
            assert level in ["debug", "info", "warning", "error"]
            assert isinstance(msg_text, str)
            assert len(msg_text) > 0

    def test_check_column_for_timestamp_index_messages_content(self):
        """Test that check_column_for_timestamp_index returns appropriate message content."""
        col_name = "test_col"
        series = pd.Series([1, 2, 3])  # Numeric column that should be skipped
        null_threshold = 0.1

        result, messages = (
            SmartApproximationUtils.check_column_for_timestamp_index(
                (col_name, series, null_threshold)
            )
        )

        # Verify result
        assert result is None

        # Verify messages contain expected content
        assert len(messages) > 0

        # Should contain debug message about checking the column
        debug_messages = [msg for level, msg in messages if level == "debug"]
        assert any("Checking column" in msg for msg in debug_messages)

        # Should contain debug message about skipping numeric column
        assert any("Skipping numeric column" in msg for msg in debug_messages)
