import json
import math
import os
from datetime import datetime
from typing import Dict, List, Union, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta
from fastapi import HTTPException

from synthefy_pkg.app.data_models import (
    BacktestInfo,
    CovariateGridRequest,
    CovariateGridResponse,
    GroupLabelColumnFilters,
    HaverMetadataAccessInfo,
    ListBacktestsResponse,
    MetadataDataFrame,
    PaginationInfo,
    StatusCode,
    SupportedAggregationFunctions,
    TimePeriod,
    WeatherMetadataAccessInfo,
    WeatherParameters,
    WeatherStackLocation,
)
from synthefy_pkg.app.routers.foundation_models import (
    FilterRequest,
    ListBacktestsRequest,
    _calculate_time_delta,
    _combine_metadata_dataframes,
    _compute_forecast_metrics,
    _compute_forecast_windows,
    _generate_future_timestamps,
    _generate_model_forecast,
    _generate_response_message,
    _load_dataset_file,
    _merge_user_and_metadata_dataframes,
    _process_timestamps,
    _read_and_filter_dataset,
    _save_filtered_dataset_as_json_blob,
    _setup_forecast_environment,
    _validate_required_columns,
    find_best_covariate_combination,
    generate_covariate_combinations,
    generate_covariate_grid_endpoint,
    list_available_backtests,
    run_filter_core_code,
)
from synthefy_pkg.app.utils.external_metadata_utils import (
    prepare_metadata_dataframes,
    weatherstack_to_df,
)
from synthefy_pkg.app.utils.filter_utils import FilterUtils


@pytest.fixture
def sample_config():
    """Create a sample forecast config."""
    return MagicMock(
        file_path_key="user123/dataset/sample.csv",
        timestamp_column="date",
        timeseries_columns=["value1", "value2"],
        min_timestamp="2023-01-01",
        forecasting_timestamp="2023-01-10",
        forecast_length=5,
        model_type="tabpfn",
        metadata_dataframe_keys=["meta1", "meta2"],
    )


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "value2": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    )


class TestSetupForecastEnvironment:
    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    def test_setup_forecast_environment_success(self, mock_get_settings):
        """Test successful environment setup."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_get_settings.return_value = mock_settings

        # Act
        settings, bucket_name = _setup_forecast_environment()

        # Assert
        assert settings == mock_settings
        assert bucket_name == "test-bucket"
        mock_get_settings.assert_called_once()

    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    def test_setup_forecast_environment_missing_bucket(self, mock_get_settings):
        """Test environment setup with missing bucket."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.bucket_name = None
        mock_get_settings.return_value = mock_settings

        # Act & Assert
        with pytest.raises(ValueError, match="S3 bucket name not configured"):
            _setup_forecast_environment()


class TestLoadDatasetFile:
    def test_load_csv_file(self, tmp_path):
        """Test loading a CSV file."""
        # Arrange
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_csv(csv_file, index=False)

        # Act
        result = _load_dataset_file(csv_file, "test.csv")

        # Assert
        assert result is not None
        assert result.shape == (3, 2)
        assert list(result.columns) == ["col1", "col2"]

    def test_load_parquet_file(self, tmp_path):
        """Test loading a Parquet file."""
        # Arrange
        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_parquet(parquet_file, index=False)

        # Act
        result = _load_dataset_file(parquet_file, "test.parquet")

        # Assert
        assert result is not None
        assert result.shape == (3, 2)
        assert list(result.columns) == ["col1", "col2"]

    def test_unsupported_file_format(self, tmp_path):
        """Test handling unsupported file format."""
        # Arrange
        txt_file = tmp_path / "test.txt"
        with open(txt_file, "w") as f:
            f.write("This is a test file")

        # Act
        result = _load_dataset_file(txt_file, "test.txt")

        # Assert
        assert result is None

    def test_file_not_found(self):
        """Test handling file not found."""
        # Act
        result = _load_dataset_file("nonexistent.csv", "nonexistent.csv")

        # Assert
        assert result is None


class TestValidateRequiredColumns:
    def test_all_columns_present(self, sample_df):
        """Test validation when all required columns are present."""
        # Arrange
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.covariate_columns = []
        # Act
        result = _validate_required_columns(sample_df, config)

        # Assert
        assert result is True

    def test_missing_timestamp_column(self, sample_df):
        """Test validation when timestamp column is missing."""
        # Arrange
        config = MagicMock()
        config.timestamp_column = "timestamp"
        config.timeseries_columns = ["value1", "value2"]
        config.covariate_columns = []
        # Act
        result = _validate_required_columns(sample_df, config)

        # Assert
        assert result is False

    def test_missing_timeseries_column(self, sample_df):
        """Test validation when a timeseries column is missing."""
        # Arrange
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "missing_column"]
        config.covariate_columns = []
        # Act
        result = _validate_required_columns(sample_df, config)

        # Assert
        assert result is False


class TestCalculateTimeDelta:
    def test_regular_intervals(self):
        """Test calculating time delta for regular intervals."""
        # Arrange
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert time_delta == pd.Timedelta(days=1)

    def test_hourly_intervals(self):
        """Test calculating time delta for hourly intervals."""
        # Arrange
        dates = pd.date_range(start="2023-01-01", periods=5, freq="H")
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert time_delta == pd.Timedelta(hours=1)

    def test_irregular_intervals(self):
        """Test calculating time delta for irregular intervals."""
        # Arrange
        # Mostly hourly but one 2-hour gap
        dates = [
            pd.Timestamp("2023-01-01 00:00:00"),
            pd.Timestamp("2023-01-01 01:00:00"),
            pd.Timestamp("2023-01-01 03:00:00"),  # 2-hour gap
            pd.Timestamp("2023-01-01 04:00:00"),
            pd.Timestamp("2023-01-01 05:00:00"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert time_delta == pd.Timedelta(
            hours=1
        )  # Should return the most common interval

    def test_monthly_intervals(self):
        """Test calculating time delta for monthly intervals."""
        # Arrange
        dates = pd.date_range(start="2023-01-15", periods=5, freq="M")
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, relativedelta)
        assert time_delta.months == 1

    def test_quarterly_intervals(self):
        """Test calculating time delta for quarterly intervals."""
        # Arrange
        dates = [
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-04-15"),
            pd.Timestamp("2023-07-15"),
            pd.Timestamp("2023-10-15"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, relativedelta)
        assert time_delta.months == 3

    def test_semi_annual_intervals(self):
        """Test calculating time delta for semi-annual intervals."""
        # Arrange
        dates = [
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-07-15"),
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-07-15"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, relativedelta)
        assert time_delta.months == 6

    def test_yearly_intervals(self):
        """Test calculating time delta for yearly intervals."""
        # Arrange
        dates = [
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2025-01-15"),
            pd.Timestamp("2026-01-15"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, relativedelta)
        assert time_delta.years == 1

    def test_monthly_intervals_with_inconsistent_days(self):
        """Test monthly intervals when days of month are not consistent."""
        # Arrange
        dates = [
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-02-14"),  # Different day
            pd.Timestamp("2023-03-15"),
            pd.Timestamp("2023-04-15"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        # Should fall back to seconds-based approach since days are not consistent
        assert isinstance(time_delta, pd.Timedelta)
        assert time_delta.days > 0  # Should be approximately 30 days

    def test_yearly_intervals_with_inconsistent_dates(self):
        """Test yearly intervals when month-day combinations are not consistent."""
        # Arrange
        dates = [
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2024-02-15"),  # Different month
            pd.Timestamp("2025-01-15"),
            pd.Timestamp("2026-01-15"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        # Should fall back to seconds-based approach since dates are not consistent
        assert isinstance(time_delta, pd.Timedelta)
        assert time_delta.days > 0  # Should be approximately 365 days

    def test_sub_second_intervals_microsecond_precision(self):
        """Test calculating time delta for sub-second intervals with microsecond precision."""
        # Arrange - create timestamps with 500 millisecond intervals
        dates = [
            pd.Timestamp("2023-01-01 00:00:00.000"),
            pd.Timestamp("2023-01-01 00:00:00.500"),  # 500ms later
            pd.Timestamp("2023-01-01 00:00:01.000"),  # 500ms later
            pd.Timestamp("2023-01-01 00:00:01.500"),  # 500ms later
            pd.Timestamp("2023-01-01 00:00:02.000"),  # 500ms later
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, pd.Timedelta)
        assert time_delta.total_seconds() == 0.5  # 500 milliseconds
        assert time_delta.microseconds == 500000  # 500000 microseconds

    def test_sub_second_intervals_mixed_precision(self):
        """Test calculating time delta for mixed sub-second intervals."""
        # Arrange - create timestamps with varying sub-second intervals
        dates = [
            pd.Timestamp("2023-01-01 00:00:00.000"),
            pd.Timestamp("2023-01-01 00:00:00.100"),  # 100ms later
            pd.Timestamp("2023-01-01 00:00:00.250"),  # 150ms later
            pd.Timestamp("2023-01-01 00:00:00.350"),  # 100ms later
            pd.Timestamp("2023-01-01 00:00:00.450"),  # 100ms later
            pd.Timestamp("2023-01-01 00:00:00.550"),  # 100ms later
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5, 6]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, pd.Timedelta)
        # Should return the most common interval (100ms)
        assert time_delta.total_seconds() == 0.1  # 100 milliseconds
        assert time_delta.microseconds == 100000  # 100000 microseconds

    def test_sub_second_intervals_single_microsecond(self):
        """Test calculating time delta for single microsecond intervals."""
        # Arrange - create timestamps with 1 microsecond intervals
        dates = [
            pd.Timestamp("2023-01-01 00:00:00.000001"),
            pd.Timestamp("2023-01-01 00:00:00.000002"),  # 1µs later
            pd.Timestamp("2023-01-01 00:00:00.000003"),  # 1µs later
            pd.Timestamp("2023-01-01 00:00:00.000004"),  # 1µs later
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, pd.Timedelta)
        assert time_delta.total_seconds() == 0.000001  # 1 microsecond
        assert time_delta.microseconds == 1

    def test_sub_second_intervals_boundary_case(self):
        """Test calculating time delta for boundary case near 1 second."""
        # Arrange - create timestamps with intervals just under 1 second
        dates = [
            pd.Timestamp("2023-01-01 00:00:00.000"),
            pd.Timestamp("2023-01-01 00:00:00.999"),  # 999ms later (< 1s)
            pd.Timestamp("2023-01-01 00:00:01.998"),  # 999ms later (< 1s)
            pd.Timestamp("2023-01-01 00:00:02.997"),  # 999ms later (< 1s)
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})

        # Act
        time_delta = _calculate_time_delta(df, "date")

        # Assert
        assert isinstance(time_delta, pd.Timedelta)
        assert time_delta.total_seconds() == 0.999  # 999 milliseconds
        assert time_delta.microseconds == 999000  # 999000 microseconds


class TestGenerateFutureTimestamps:
    def test_multiple_data_points(self):
        """Test generating future timestamps with multiple data points."""
        # Arrange
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5]})
        forecasting_timestamp = pd.Timestamp("2023-01-05")
        time_delta = pd.Timedelta(days=1)
        forecast_length = 3

        # Act
        future_timestamps = _generate_future_timestamps(
            df, forecasting_timestamp, time_delta, forecast_length
        )

        # Assert
        assert len(future_timestamps) == forecast_length
        assert future_timestamps[0] == datetime(2023, 1, 6)
        assert future_timestamps[1] == datetime(2023, 1, 7)
        assert future_timestamps[2] == datetime(2023, 1, 8)

    def test_single_data_point(self):
        """Test generating future timestamps with a single data point."""
        # Arrange
        df = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "value": [1]})
        forecasting_timestamp = pd.Timestamp("2023-01-01")
        time_delta = pd.Timedelta(days=1)
        forecast_length = 3

        # Act
        future_timestamps = _generate_future_timestamps(
            df, forecasting_timestamp, time_delta, forecast_length
        )

        # Assert
        assert len(future_timestamps) == forecast_length
        assert future_timestamps[0] == datetime(2023, 1, 2)
        assert future_timestamps[1] == datetime(2023, 1, 3)
        assert future_timestamps[2] == datetime(2023, 1, 4)

    def test_empty_dataframe(self):
        """Test generating future timestamps with an empty dataframe."""
        # Arrange
        df = pd.DataFrame({"date": [], "value": []})
        forecasting_timestamp = pd.Timestamp("2023-01-01")
        time_delta = pd.Timedelta(days=1)
        forecast_length = 3

        # Act
        future_timestamps = _generate_future_timestamps(
            df, forecasting_timestamp, time_delta, forecast_length
        )

        # Assert
        assert len(future_timestamps) == forecast_length
        assert future_timestamps[0] == datetime(2023, 1, 2)
        assert future_timestamps[1] == datetime(2023, 1, 3)
        assert future_timestamps[2] == datetime(2023, 1, 4)

    def test_monthly_pattern(self):
        """Test generating future timestamps with monthly pattern."""
        # Arrange
        dates = pd.date_range(start="2023-01-15", periods=3, freq="M")
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3]})
        forecasting_timestamp = pd.Timestamp("2023-03-15")
        time_delta = relativedelta(months=1)
        forecast_length = 3

        # Act
        future_timestamps = _generate_future_timestamps(
            df,
            forecasting_timestamp,
            time_delta,
            forecast_length,
        )

        # Assert
        assert len(future_timestamps) == forecast_length
        assert future_timestamps[0] == datetime(2023, 4, 15)
        assert future_timestamps[1] == datetime(2023, 5, 15)
        assert future_timestamps[2] == datetime(2023, 6, 15)

    def test_monthly_pattern_with_year_rollover(self):
        """Test generating future timestamps with monthly pattern crossing year boundary."""
        # Arrange
        dates = pd.date_range(
            start="2023-10-15", periods=2, freq="MS"
        ) + pd.Timedelta(days=14)
        df = pd.DataFrame({"date": dates, "value": [1, 2]})
        forecasting_timestamp = pd.Timestamp("2024-01-15")
        time_delta = relativedelta(months=1)
        forecast_length = 3

        # Act
        future_timestamps = _generate_future_timestamps(
            df, forecasting_timestamp, time_delta, forecast_length
        )

        # Assert
        assert len(future_timestamps) == forecast_length
        assert future_timestamps[0] == datetime(2024, 2, 15)
        assert future_timestamps[1] == datetime(2024, 3, 15)
        assert future_timestamps[2] == datetime(2024, 4, 15)

    def test_monthly_pattern_with_month_end(self):
        """Test generating future timestamps with monthly pattern on month end."""
        # Arrange
        dates = [
            pd.Timestamp("2023-01-31"),
            pd.Timestamp("2023-02-28"),
            pd.Timestamp("2023-03-31"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3]})
        forecasting_timestamp = pd.Timestamp("2023-03-31")
        time_delta = relativedelta(months=1)
        forecast_length = 3

        # Act
        future_timestamps = _generate_future_timestamps(
            df, forecasting_timestamp, time_delta, forecast_length
        )

        # Assert
        assert len(future_timestamps) == forecast_length
        assert future_timestamps[0] == datetime(2023, 4, 30)
        assert future_timestamps[1] == datetime(2023, 5, 31)
        assert future_timestamps[2] == datetime(2023, 6, 30)

    def test_quarterly_pattern(self):
        """Test generating future timestamps with quarterly pattern."""
        # Arrange
        dates = [
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-04-15"),
            pd.Timestamp("2023-07-15"),
        ]
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3]})
        forecasting_timestamp = pd.Timestamp("2023-07-15")
        time_delta = relativedelta(months=3)
        forecast_length = 3

        # Act
        future_timestamps = _generate_future_timestamps(
            df, forecasting_timestamp, time_delta, forecast_length
        )

        # Assert
        assert len(future_timestamps) == forecast_length
        assert future_timestamps[0] == datetime(2023, 10, 15)
        assert future_timestamps[1] == datetime(2024, 1, 15)
        assert future_timestamps[2] == datetime(2024, 4, 15)


class TestProcessTimestamps:
    def test_process_timestamps_basic(self, sample_df):
        """Test processing timestamps with standard daily data."""
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-03"
        config.forecasting_timestamp = "2023-01-08"
        config.forecast_length = 3
        # Shuffle the sample_df
        sample_df = sample_df.sample(frac=1).reset_index(drop=True)

        df, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(sample_df, config)
        )

        # df: from 2023-01-03 to 2023-01-10 (inclusive)
        assert (
            len(df) == 8
        )  # since we don't clip after the forecasting_timestamp to enable leaking
        assert df.iloc[0]["date"] == pd.Timestamp("2023-01-03")
        assert df.iloc[-1]["date"] == pd.Timestamp("2023-01-10")
        assert df["date"].is_monotonic_increasing

        # ground_truth_df: from 2023-01-03 to 2023-01-10 (max 8 rows)
        assert len(ground_truth_df) == 8
        assert ground_truth_df.iloc[0]["date"] == pd.Timestamp("2023-01-03")
        assert ground_truth_df.iloc[-1]["date"] == pd.Timestamp("2023-01-10")
        # New: Ensure ground_truth_df is sorted by date
        assert ground_truth_df["date"].is_monotonic_increasing

        assert forecasting_timestamp == pd.Timestamp("2023-01-08")
        assert len(future_timestamps) == 3
        assert all(isinstance(ts, datetime) for ts in future_timestamps)
        # Should be 2023-01-09, 2023-01-10, 2023-01-11
        assert future_timestamps[0] == datetime(2023, 1, 9)
        assert future_timestamps[-1] == datetime(2023, 1, 11)

    def test_process_timestamps_empty_result(self, sample_df):
        """Test processing timestamps with filtering that results in empty dataframe."""
        # Arrange
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-20"  # After all data in sample_df
        config.forecasting_timestamp = "2023-01-25"
        config.forecast_length = 3

        # Act
        df, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(sample_df, config)
        )

        # Assert
        assert len(df) == 0  # No data matches the filter
        assert len(ground_truth_df) == 0  # No ground truth data
        assert forecasting_timestamp == pd.Timestamp("2023-01-25")
        assert len(future_timestamps) == 3
        assert all(isinstance(ts, datetime) for ts in future_timestamps)

    def test_process_timestamps_unsorted_input(self, sample_df):
        """Test that input is sorted by timestamp."""
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-01"
        config.forecasting_timestamp = "2023-01-05"
        config.forecast_length = 2
        # Shuffle the sample_df
        sample_df = sample_df.sample(frac=1).reset_index(drop=True)
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(sample_df, config)
        )
        assert df_out["date"].is_monotonic_increasing
        assert ground_truth_df["date"].is_monotonic_increasing

    def test_process_timestamps_timezone_aware(self):
        """Test with timezone-aware timestamps."""
        tz = "UTC"
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D", tz=tz)
        df = pd.DataFrame(
            {
                "date": dates,
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-01"
        config.forecasting_timestamp = "2023-01-03"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        # Output should be tz-naive
        assert df_out["date"].dt.tz is None
        assert ground_truth_df["date"].dt.tz is None
        assert forecasting_timestamp.tzinfo is None
        assert all(ts.tzinfo is None for ts in future_timestamps)

    def test_process_timestamps_with_relativedelta(self, sample_df):
        """Test with daily data that should result in daily future timestamps."""
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-01"
        config.forecasting_timestamp = "2023-01-05"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(sample_df, config)
        )
        # Should use daily intervals for future_timestamps since data is daily
        assert future_timestamps[0].day == 6  # 2023-01-06
        assert future_timestamps[1].day == 7  # 2023-01-07

    def test_process_timestamps_nonstandard_column(self):
        """Test with a nonstandard timestamp column name."""
        df = pd.DataFrame(
            {
                "ts": pd.date_range("2023-01-01", periods=4, freq="D"),
                "value1": [1, 2, 3, 4],
                "value2": [10, 20, 30, 40],
            }
        )
        config = MagicMock()
        config.timestamp_column = "ts"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-02"
        config.forecasting_timestamp = "2023-01-03"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        assert df_out.iloc[0]["ts"] == pd.Timestamp("2023-01-02")
        assert ground_truth_df.iloc[0]["ts"] == pd.Timestamp("2023-01-02")
        assert len(future_timestamps) == 2

    def test_process_timestamps_min_max_at_boundaries(self):
        """Test when min_timestamp and forecasting_timestamp are at the boundaries of the data."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-01"
        config.forecasting_timestamp = "2023-01-05"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        assert df_out.iloc[0]["date"] == pd.Timestamp("2023-01-01")
        assert df_out.iloc[-1]["date"] == pd.Timestamp("2023-01-05")
        assert ground_truth_df.iloc[0]["date"] == pd.Timestamp("2023-01-01")
        assert ground_truth_df.iloc[-1]["date"] == pd.Timestamp("2023-01-05")
        assert len(future_timestamps) == 2

    def test_process_timestamps_all_rows_filtered_out(self):
        """Test when all rows are filtered out by min_timestamp and forecasting_timestamp."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2024-01-01"
        config.forecasting_timestamp = "2024-01-05"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        assert len(df_out) == 0
        assert len(ground_truth_df) == 0
        assert len(future_timestamps) == 2

    def test_process_timestamps_df_tzaware_config_naive(self):
        """Test DataFrame with tz-aware timestamps, config with naive timestamps."""
        tz = "UTC"
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D", tz=tz)
        df = pd.DataFrame(
            {
                "date": dates,
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-02"  # naive
        config.forecasting_timestamp = "2023-01-04"  # naive
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        # Output should be tz-naive
        assert df_out["date"].dt.tz is None
        assert ground_truth_df["date"].dt.tz is None
        assert forecasting_timestamp.tzinfo is None
        assert all(ts.tzinfo is None for ts in future_timestamps)
        # Filtering should work
        assert df_out.iloc[0]["date"] == pd.Timestamp("2023-01-02")
        assert df_out.iloc[-1]["date"] == pd.Timestamp("2023-01-05")
        assert len(future_timestamps) == 2

    def test_process_timestamps_df_naive_config_tzaware(self):
        """Test DataFrame with naive timestamps, config with tz-aware timestamps."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        # Provide tz-aware strings
        config.min_timestamp = "2023-01-02T00:00:00+00:00"
        config.forecasting_timestamp = "2023-01-04T00:00:00+00:00"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        # Output should be tz-naive
        assert df_out["date"].dt.tz is None
        assert ground_truth_df["date"].dt.tz is None
        assert forecasting_timestamp.tzinfo is None
        assert all(ts.tzinfo is None for ts in future_timestamps)
        # Filtering should work
        assert df_out.iloc[0]["date"] == pd.Timestamp("2023-01-02")
        assert df_out.iloc[-1]["date"] == pd.Timestamp("2023-01-05")
        assert len(future_timestamps) == 2

    def test_process_timestamps_both_tzaware_same_tz(self):
        """Test both DataFrame and config with tz-aware timestamps (same tz)."""
        tz = "UTC"
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D", tz=tz)
        df = pd.DataFrame(
            {
                "date": dates,
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-02T00:00:00+00:00"
        config.forecasting_timestamp = "2023-01-04T00:00:00+00:00"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        # Output should be tz-naive
        assert df_out["date"].dt.tz is None
        assert ground_truth_df["date"].dt.tz is None
        assert forecasting_timestamp.tzinfo is None
        assert all(ts.tzinfo is None for ts in future_timestamps)
        # Filtering should work
        assert df_out.iloc[0]["date"] == pd.Timestamp("2023-01-02")
        assert df_out.iloc[-1]["date"] == pd.Timestamp("2023-01-05")
        assert len(future_timestamps) == 2

    def test_process_timestamps_both_tzaware_different_tz(self):
        """Test both DataFrame and config with tz-aware timestamps (different tz)."""
        tz_df = "UTC"
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D", tz=tz_df)
        df = pd.DataFrame(
            {
                "date": dates,
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        # Config timestamps in a different timezone
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = pd.Timestamp(
            "2023-01-02T01:00:00+01:00"
        ).isoformat()
        config.forecasting_timestamp = pd.Timestamp(
            "2023-01-04T01:00:00+01:00"
        ).isoformat()
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        # Output should be tz-naive
        assert df_out["date"].dt.tz is None
        assert ground_truth_df["date"].dt.tz is None
        assert forecasting_timestamp.tzinfo is None
        assert all(ts.tzinfo is None for ts in future_timestamps)
        # Filtering should work (should match UTC times)
        assert df_out.iloc[0]["date"] == pd.Timestamp("2023-01-02")
        assert df_out.iloc[-1]["date"] == pd.Timestamp("2023-01-05")
        assert len(future_timestamps) == 2

    def test_process_timestamps_both_naive(self):
        """Test both DataFrame and config with naive timestamps."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )
        config = MagicMock()
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]
        config.min_timestamp = "2023-01-02"
        config.forecasting_timestamp = "2023-01-04"
        config.forecast_length = 2
        df_out, ground_truth_df, forecasting_timestamp, future_timestamps = (
            _process_timestamps(df, config)
        )
        # Output should be tz-naive
        assert df_out["date"].dt.tz is None
        assert ground_truth_df["date"].dt.tz is None
        assert forecasting_timestamp.tzinfo is None
        assert all(ts.tzinfo is None for ts in future_timestamps)
        # Filtering should work
        assert df_out.iloc[0]["date"] == pd.Timestamp("2023-01-02")
        assert df_out.iloc[-1]["date"] == pd.Timestamp("2023-01-05")
        assert len(future_timestamps) == 2

    def test_process_timestamps_string_input(self):
        """Test processing timestamps when input column contains string values.

        This test ensures that the pd.to_datetime conversion is working properly
        and would catch the bug if that conversion line is missing.
        """
        # Create a DataFrame with string timestamps (not datetime objects)
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2023-01-01 10:00:00",
                    "2023-01-01 11:00:00",
                    "2023-01-01 12:00:00",
                    "2023-01-01 13:00:00",
                    "2023-01-01 14:00:00",
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )

        # Ensure the timestamp column is string type, not datetime
        assert df["timestamp"].dtype == "object"

        config = MagicMock()
        config.timestamp_column = "timestamp"
        config.min_timestamp = "2023-01-01 10:00:00"
        config.forecasting_timestamp = "2023-01-01 13:00:00"
        config.forecast_length = 2
        config.timeseries_columns = ["value"]

        # This should work even with string timestamps due to pd.to_datetime conversion
        (
            filtered_df,
            ground_truth_df,
            forecasting_timestamp,
            future_timestamps,
        ) = _process_timestamps(df, config)

        # Verify that the function successfully processed string timestamps
        assert len(filtered_df) == 5
        assert len(ground_truth_df) == 5
        assert forecasting_timestamp == pd.Timestamp("2023-01-01 13:00:00")
        assert len(future_timestamps) == 2

        # Verify that the timestamp columns are now datetime type
        assert pd.api.types.is_datetime64_any_dtype(filtered_df["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(
            ground_truth_df["timestamp"]
        )

        # Verify the timestamps are timezone-naive
        assert filtered_df["timestamp"].dt.tz is None
        assert ground_truth_df["timestamp"].dt.tz is None

    def test_process_timestamps_iso_string_format(self):
        """Test processing timestamps with ISO string format.

        This test ensures that the pd.to_datetime conversion handles ISO format.
        """
        # Create a DataFrame with ISO string timestamp format
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2023-01-01T10:00:00",
                    "2023-01-01T11:00:00",
                    "2023-01-01T12:00:00",
                    "2023-01-01T13:00:00",
                    "2023-01-01T14:00:00",
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )

        config = MagicMock()
        config.timestamp_column = "timestamp"
        config.min_timestamp = "2023-01-01 10:00:00"
        config.forecasting_timestamp = "2023-01-01 13:00:00"
        config.forecast_length = 2
        config.timeseries_columns = ["value"]

        # This should work with ISO format strings
        (
            filtered_df,
            ground_truth_df,
            forecasting_timestamp,
            future_timestamps,
        ) = _process_timestamps(df, config)

        # Verify successful processing
        assert len(filtered_df) == 5
        assert len(ground_truth_df) == 5
        assert forecasting_timestamp == pd.Timestamp("2023-01-01 13:00:00")
        assert len(future_timestamps) == 2

        # Verify timestamps are properly converted
        assert pd.api.types.is_datetime64_any_dtype(filtered_df["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(
            ground_truth_df["timestamp"]
        )


@patch(
    "synthefy_pkg.app.routers.foundation_models.FoundationModelService.get_model"
)
class TestGenerateModelForecast:
    @pytest.mark.asyncio
    async def test_generate_forecast(self, mock_get_model, sample_df):
        """Test generating model forecast."""
        # Arrange
        config = MagicMock()
        config.model_type = "tabpfn"
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]

        user_id = "test_user"
        forecasting_timestamp = pd.Timestamp("2023-01-08")
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        ground_truth_df = sample_df.copy()
        metadata_dataframes = []

        mock_model = MagicMock()
        mock_forecast_df = MagicMock()
        mock_model.predict.return_value = mock_forecast_df
        mock_get_model.return_value = mock_model

        # Act
        result = _generate_model_forecast(
            sample_df,
            config,
            user_id,
            forecasting_timestamp,
            future_timestamps,
            ground_truth_df,
            metadata_dataframes,
        )

        # Assert
        assert result == mock_forecast_df
        mock_get_model.assert_called_once_with(
            model_type="tabpfn", user_id=user_id
        )
        mock_model.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_forecast_exception(self, mock_get_model, sample_df):
        """Test error handling during forecast generation."""
        # Arrange
        config = MagicMock()
        config.model_type = "tabpfn"
        config.timestamp_column = "date"
        config.timeseries_columns = ["value1", "value2"]

        user_id = "test_user"
        forecasting_timestamp = pd.Timestamp("2023-01-08")
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        ground_truth_df = sample_df.copy()
        metadata_dataframes = []

        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("Test prediction error")
        mock_get_model.return_value = mock_model

        # Act & Assert
        with pytest.raises(HTTPException, match="Test prediction error"):
            _generate_model_forecast(
                sample_df,
                config,
                user_id,
                forecasting_timestamp,
                future_timestamps,
                ground_truth_df,
                metadata_dataframes,
            )


class TestSaveFilteredDatasetAsJsonBlob:
    """Tests for the _save_filtered_dataset_as_json_blob function."""

    @pytest.mark.asyncio
    async def test_save_filtered_dataset_as_json_blob_basic(self, tmp_path):
        """Test basic functionality of saving DataFrame as JSON blob."""
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [10.5, 20.5, 30.5],
            }
        )
        temp_dir = str(tmp_path)
        dataset_name = "test_dataset"
        output_path = None

        try:
            # Act
            output_path = await _save_filtered_dataset_as_json_blob(
                df, temp_dir, dataset_name
            )

            # Assert
            assert os.path.exists(output_path)
            assert output_path == os.path.join(
                temp_dir, "test_dataset_filtered.json"
            )

            # Verify the JSON content
            with open(output_path, "r") as f:
                json_data = json.load(f)

            assert isinstance(json_data, list)
            assert len(json_data) == 3

            # Check first record's structure
            first_record = json_data[0]
            assert "id" in first_record
            assert "name" in first_record
            assert "value" in first_record
            assert first_record["id"] == 1
            assert first_record["name"] == "Alice"
            assert first_record["value"] == 10.5
        finally:
            # Clean up the file even if test fails
            if output_path and os.path.exists(output_path):
                os.remove(output_path)

    @pytest.mark.asyncio
    async def test_save_filtered_dataset_as_json_blob_with_dates(
        self, tmp_path
    ):
        """Test saving DataFrame with date columns as JSON blob."""
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=3, freq="D"),
                "value": [10.5, 20.5, 30.5],
            }
        )
        temp_dir = str(tmp_path)
        dataset_name = "date_dataset"
        output_path = None

        try:
            # Act
            output_path = await _save_filtered_dataset_as_json_blob(
                df, temp_dir, dataset_name
            )

            # Assert
            assert os.path.exists(output_path)

            # Verify the JSON content
            with open(output_path, "r") as f:
                json_data = json.load(f)

            assert isinstance(json_data, list)
            assert len(json_data) == 3

            # Verify dates were properly converted to ISO format strings
            assert "2023-01-01" in json_data[0]["date"]
            assert "2023-01-02" in json_data[1]["date"]
            assert "2023-01-03" in json_data[2]["date"]
        finally:
            # Clean up the file even if test fails
            if output_path and os.path.exists(output_path):
                os.remove(output_path)

    @pytest.mark.asyncio
    async def test_save_filtered_dataset_as_json_blob_empty_df(self, tmp_path):
        """Test saving an empty DataFrame as JSON blob."""
        # Arrange
        df = pd.DataFrame(columns=["col1", "col2", "col3"])
        temp_dir = str(tmp_path)
        dataset_name = "empty_dataset"
        output_path = None

        try:
            # Act
            output_path = await _save_filtered_dataset_as_json_blob(
                df, temp_dir, dataset_name
            )

            # Assert
            assert os.path.exists(output_path)

            # Verify the JSON content
            with open(output_path, "r") as f:
                json_data = json.load(f)

            assert isinstance(json_data, list)
            assert len(json_data) == 0
        finally:
            # Clean up the file even if test fails
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


class TestComputeForecastWindows:
    def test_basic_daily_stride(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 3)
        max_ts = datetime(2023, 1, 10)
        stride = "P2D"  # 2 days
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )

        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0] == (min_ts, start_forecast_ts)
        assert windows[-1][1] == max_ts
        assert all(w[0] < w[1] for w in windows)
        assert len(windows) == 5  # Should have 4 windows

    def test_monthly_stride(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 2, 1)
        max_ts = datetime(2023, 4, 1)
        stride = "P1M"  # 1 month
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0] == (min_ts, start_forecast_ts)
        assert windows[-1][1] == max_ts
        assert len(windows) == 3  # Jan-Feb, Feb-Mar, Mar-Apr

    def test_yearly_stride(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2024, 1, 1)
        max_ts = datetime(2025, 1, 1)
        stride = "P1Y"  # 1 year
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0] == (min_ts, start_forecast_ts)
        assert windows[-1][1] == max_ts
        assert len(windows) == 2  # 2023-2024, 2024-2025

    def test_complex_stride(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 2, 16)
        max_ts = datetime(2023, 4, 15)
        stride = "P1M15D"  # 1 month and 15 days
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0] == (min_ts, start_forecast_ts)
        assert windows[-1][1] == max_ts
        assert all(w[0] < w[1] for w in windows)

    def test_stride_larger_than_range(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 5)
        max_ts = datetime(2023, 1, 5)
        stride = "P10D"  # 10 days
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows == [(min_ts, start_forecast_ts)]
        assert delta == relativedelta(days=10)

    def test_exact_fit_stride(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 2)
        max_ts = datetime(2023, 1, 4)
        stride = "P1D"  # 1 day
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert len(windows) == 3
        assert windows[0] == (min_ts, start_forecast_ts)
        assert windows[-1][1] == max_ts

    def test_zero_length_range(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 1)
        max_ts = datetime(2023, 1, 1)
        stride = "P1D"
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows == [(min_ts, start_forecast_ts)]
        assert delta == relativedelta(days=1)

    def test_invalid_stride(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 5)
        max_ts = datetime(2023, 1, 5)
        with pytest.raises(Exception):
            _compute_forecast_windows(
                min_ts, start_forecast_ts, max_ts, "not_a_duration"
            )

    def test_negative_stride(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 5)
        max_ts = datetime(2023, 1, 5)
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, "-P1D"
        )
        assert error_message is not None
        assert windows is None
        assert delta is None

    def test_year_rollover(self):
        min_ts = datetime(2023, 12, 1)
        start_forecast_ts = datetime(2024, 1, 1)
        max_ts = datetime(2024, 2, 1)
        stride = "P1M"  # 1 month
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0] == (min_ts, start_forecast_ts)
        assert windows[-1][1] == max_ts
        assert len(windows) == 2  # Dec-Jan, Jan-Feb

    def test_leap_year(self):
        min_ts = datetime(2024, 2, 1)
        start_forecast_ts = datetime(2024, 3, 1)
        max_ts = datetime(2024, 3, 1)
        stride = "P1M"  # 1 month
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0] == (min_ts, start_forecast_ts)
        assert len(windows) == 1  # Single window due to leap year February

    def test_microsecond_precision(self):
        min_ts = datetime(2023, 1, 1, 0, 0, 0, 500000)
        start_forecast_ts = datetime(2023, 1, 1, 0, 0, 1, 0)
        max_ts = datetime(2023, 1, 1, 0, 0, 1, 500000)
        stride = "PT1S"  # 1 second
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0][0] == min_ts
        assert windows[-1][1] == start_forecast_ts
        assert all(w[0] < w[1] for w in windows)
        assert len(windows) == 1

    def test_mixed_time_units(self):
        min_ts = datetime(2023, 1, 1, 12, 0)
        start_forecast_ts = datetime(2023, 1, 2, 0, 0)
        max_ts = datetime(2023, 1, 2, 12, 0)
        stride = "PT12H"  # 12 hours
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )
        assert error_message is None
        assert windows is not None
        assert delta is not None
        assert windows[0] == (min_ts, start_forecast_ts)
        assert windows[-1][1] == max_ts
        assert len(windows) == 2  # Two 12-hour windows

    def test_max_windows_limit(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(
            2023, 1, 1, 0, 0, 0, 100000
        )  # 100 milliseconds
        max_ts = datetime(2023, 1, 1, 0, 0, 0, 350000)  # 100 milliseconds
        stride = "PT0.001S"  # 1 millisecond
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, stride
        )

        assert error_message is not None
        assert windows is None
        assert delta is None

    def test_zero_delta(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 5)
        max_ts = datetime(2023, 1, 5)
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, "PT0S"
        )
        assert error_message is not None
        assert windows is None
        assert delta is None

    def test_zero_delta_complex(self):
        min_ts = datetime(2023, 1, 1)
        start_forecast_ts = datetime(2023, 1, 5)
        max_ts = datetime(2023, 1, 5)
        windows, delta, error_message = _compute_forecast_windows(
            min_ts, start_forecast_ts, max_ts, "P0Y0M0DT0H0M0S"
        )
        assert error_message is not None
        assert windows is None
        assert delta is None


class TestComputeForecastMetrics:
    def test_all_valid(self):
        actuals: List[float | None] = [1.0, 2.0, 3.0]
        forecasts = [1.1, 1.9, 3.2]
        metrics = _compute_forecast_metrics(actuals, forecasts)
        expected_rmse = np.sqrt(
            np.mean([(1.0 - 1.1) ** 2, (2.0 - 1.9) ** 2, (3.0 - 3.2) ** 2])
        )
        expected_mae = np.mean([abs(1.0 - 1.1), abs(2.0 - 1.9), abs(3.0 - 3.2)])
        expected_nmae = expected_mae / np.mean(np.abs([1.0, 2.0, 3.0]))
        assert math.isclose(metrics["rmse"], expected_rmse, rel_tol=1e-6)
        assert math.isclose(metrics["nmae"], expected_nmae, rel_tol=1e-6)

    def test_some_none(self):
        actuals: List[float | None] = [1.0, None, 3.0]
        forecasts = [1.1, 2.0, 3.2]
        metrics = _compute_forecast_metrics(actuals, forecasts)
        # Only valid pairs should be used
        valid_actuals = [1.0, 3.0]
        expected_rmse = np.sqrt(np.mean([(1.0 - 1.1) ** 2, (3.0 - 3.2) ** 2]))
        expected_mae = np.mean([abs(1.0 - 1.1), abs(3.0 - 3.2)])
        expected_nmae = expected_mae / np.mean(np.abs(valid_actuals))
        assert math.isclose(metrics["rmse"], expected_rmse, rel_tol=1e-6)
        assert math.isclose(metrics["nmae"], expected_nmae, rel_tol=1e-6)

    def test_all_none(self):
        actuals: List[float | None] = [None, None]
        forecasts = [1.0, 2.0]
        metrics = _compute_forecast_metrics(actuals, forecasts)
        assert math.isnan(metrics["rmse"])
        assert math.isnan(metrics["nmae"])

    def test_zero_mean_actual(self):
        actuals: List[float | None] = [0.0, 0.0, 0.0]
        forecasts = [1.0, 2.0, 3.0]
        metrics = _compute_forecast_metrics(actuals, forecasts)
        expected_rmse = np.sqrt(
            np.mean([(0.0 - 1.0) ** 2, (0.0 - 2.0) ** 2, (0.0 - 3.0) ** 2])
        )
        assert math.isclose(metrics["rmse"], expected_rmse, rel_tol=1e-6)
        assert math.isnan(metrics["nmae"])

    def test_include_all_metrics(self):
        actuals: List[float | None] = [2.0, 4.0, 6.0]
        forecasts = [1.0, 5.0, 7.0]
        metrics = _compute_forecast_metrics(
            actuals, forecasts, include_all_metrics=True
        )
        expected_mae = np.mean([abs(2.0 - 1.0), abs(4.0 - 5.0), abs(6.0 - 7.0)])
        expected_mse = np.mean(
            [(2.0 - 1.0) ** 2, (4.0 - 5.0) ** 2, (6.0 - 7.0) ** 2]
        )
        expected_rmse = np.sqrt(expected_mse)
        expected_nmae = expected_mae / np.mean(np.abs([2.0, 4.0, 6.0]))
        expected_residual = np.mean([2.0 - 1.0, 4.0 - 5.0, 6.0 - 7.0])
        expected_mape = (
            np.mean(
                [
                    abs(2.0 - 1.0) / 2.0,
                    abs(4.0 - 5.0) / 4.0,
                    abs(6.0 - 7.0) / 6.0,
                ]
            )
            * 100
        )
        expected_smape = (
            np.mean(
                [
                    2 * abs(1.0 - 2.0) / (abs(2.0) + abs(1.0)),
                    2 * abs(5.0 - 4.0) / (abs(4.0) + abs(5.0)),
                    2 * abs(7.0 - 6.0) / (abs(6.0) + abs(7.0)),
                ]
            )
            * 100
        )
        assert math.isclose(metrics["mae"], expected_mae, rel_tol=1e-6)
        assert math.isclose(metrics["mse"], expected_mse, rel_tol=1e-6)
        assert math.isclose(metrics["rmse"], expected_rmse, rel_tol=1e-6)
        assert math.isclose(metrics["nmae"], expected_nmae, rel_tol=1e-6)
        assert math.isclose(
            metrics["residual"], expected_residual, rel_tol=1e-6
        )
        assert math.isclose(metrics["mape"], expected_mape, rel_tol=1e-6)
        assert math.isclose(metrics["smape"], expected_smape, rel_tol=1e-6)


class TestMetadataDataFramePreparation:
    """Test class for metadata dataframe preparation."""

    @pytest.fixture
    def sample_haver_metadata_info(self):
        return HaverMetadataAccessInfo(
            data_source="haver",
            database_name="test_db",
            name="test_series",
            description="Test Description",
            start_date=20230101,
            file_name=None,
        )

    @pytest.fixture
    def sample_weather_metadata_info(self):
        return WeatherMetadataAccessInfo(
            data_source="weather",
            name="Test City",
            description="Test City Description",
            file_name="test_city",
            location_data=WeatherStackLocation(
                name="Test City",
                latitude=40.7,
                longitude=-74.0,
                country_code="US",
            ),
            weather_parameters=WeatherParameters(temperature=True),
            units="m",
            frequency="day",
            time_period=TimePeriod(
                min_timestamp="2023-01-01T00:00:00",
                forecast_timestamp="2023-01-02T00:00:00",
            ),
        )

    @pytest.mark.asyncio
    async def test_prepare_metadata_dataframes_empty(self):
        """Test preparing metadata dataframes with no input."""
        result, leak_idxs = await prepare_metadata_dataframes(
            None, "test-bucket", ["2023-01-01T00:00:00", "2023-01-02T00:00:00"]
        )
        assert len(result) == 0
        assert len(leak_idxs) == 0

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
    )
    async def test_prepare_metadata_dataframes_haver(
        self, mock_process_haver, sample_haver_metadata_info
    ):
        """Test preparing metadata dataframes with Haver data."""
        mock_df = pd.DataFrame({"date": ["2023-01-01"], "value": [100]})
        mock_process_haver.return_value = [
            MetadataDataFrame(
                df=mock_df,
                metadata_json={"timestamp_columns": ["date"]},
                description="Test",
                timestamp_key="date",
            )
        ]

        result, leak_idxs = await prepare_metadata_dataframes(
            [sample_haver_metadata_info],
            "test-bucket",
            ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
            leak_idxs=[0],
        )

        assert len(result) == 1
        assert result[0].df.equals(mock_df)
        assert leak_idxs == [0]

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.utils.external_metadata_utils.process_weatherstack_metadata_access_info"
    )
    async def test_prepare_metadata_dataframes_weather(
        self, mock_process_weather, sample_weather_metadata_info
    ):
        """Test preparing metadata dataframes with weather data."""
        mock_df = pd.DataFrame({"date": ["2023-01-01"], "value": [25]})
        mock_process_weather.return_value = [
            MetadataDataFrame(
                df=mock_df,
                metadata_json={"timestamp_columns": ["date"]},
                description="Temperature",
                timestamp_key="date",
            )
        ]

        result, leak_idxs = await prepare_metadata_dataframes(
            [sample_weather_metadata_info],
            "test-bucket",
            ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
            leak_idxs=[0],
        )

        assert len(result) == 1
        assert result[0].df.equals(mock_df)
        assert leak_idxs == [0]

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
    )
    @patch(
        "synthefy_pkg.app.utils.external_metadata_utils.process_weatherstack_metadata_access_info"
    )
    async def test_prepare_metadata_dataframes_mixed(
        self,
        mock_process_weather,
        mock_process_haver,
        sample_haver_metadata_info,
        sample_weather_metadata_info,
    ):
        """Test preparing metadata dataframes with mixed data sources."""
        mock_haver_df = pd.DataFrame({"date": ["2023-01-01"], "value": [100]})
        mock_weather_df = pd.DataFrame({"date": ["2023-01-01"], "value": [25]})

        mock_process_haver.return_value = [
            MetadataDataFrame(
                df=mock_haver_df,
                metadata_json={"timestamp_columns": ["date"]},
                description="Haver Test",
                timestamp_key="date",
            )
        ]

        mock_process_weather.return_value = [
            MetadataDataFrame(
                df=mock_weather_df,
                metadata_json={"timestamp_columns": ["date"]},
                description="Weather Test",
                timestamp_key="date",
            )
        ]

        result, leak_idxs = await prepare_metadata_dataframes(
            [sample_haver_metadata_info, sample_weather_metadata_info],
            "test-bucket",
            ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
            leak_idxs=[0, 1],
        )

        assert len(result) == 2
        assert leak_idxs == [0, 1]

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
    )
    async def test_prepare_metadata_dataframes_with_failures(
        self, mock_process_haver, sample_haver_metadata_info
    ):
        """Test preparing metadata dataframes with some failures."""
        # First call succeeds, second fails
        mock_df = pd.DataFrame({"date": ["2023-01-01"], "value": [100]})
        mock_process_haver.side_effect = [
            [
                MetadataDataFrame(
                    df=mock_df,
                    metadata_json={"timestamp_columns": ["date"]},
                    description="Test",
                    timestamp_key="date",
                )
            ],
            None,  # Failure
        ]

        result, leak_idxs = await prepare_metadata_dataframes(
            [sample_haver_metadata_info, sample_haver_metadata_info],
            "test-bucket",
            ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
            leak_idxs=[0, 1],
        )

        assert len(result) == 1  # Only one succeeded
        assert leak_idxs == [0]  # Only the first index


class TestLeakIdxsBehavior:
    """Test class specifically for comprehensive leak_idxs behavior testing."""

    @pytest.fixture
    def haver_info_list(self):
        """Create a list of Haver metadata info for testing."""
        return [
            HaverMetadataAccessInfo(
                data_source="haver",
                database_name=f"db_{i}",
                name=f"series_{i}",
                description=f"Description {i}",
                start_date=20230101,
                file_name=None,
            )
            for i in range(5)
        ]

    @pytest.fixture
    def mock_haver_responses(self):
        """Mock responses for different success/failure patterns."""
        return {
            "success": [
                MetadataDataFrame(
                    df=pd.DataFrame({"date": ["2023-01-01"], "value": [100]}),
                    metadata_json={"timestamp_columns": ["date"]},
                    description="Success Response",
                    timestamp_key="date",
                )
            ],
            "failure": None,
        }

    @pytest.mark.asyncio
    async def test_leak_idxs_none_input(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs behavior when None is passed."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["success"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:2],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=None,
            )

            assert len(result) == 2
            assert leak_idxs == []  # Should return empty list when None input

    @pytest.mark.asyncio
    async def test_leak_idxs_empty_list_input(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs behavior when empty list is passed."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["success"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:2],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[],
            )

            assert len(result) == 2
            assert leak_idxs == []  # Should return empty list when empty input

    @pytest.mark.asyncio
    async def test_leak_idxs_all_success_sequential(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs with all successful processing - sequential mapping."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["success"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:4],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 2, 3],
            )

            assert len(result) == 4
            assert leak_idxs == [
                0,
                1,
                2,
                3,
            ]  # Should maintain same indices when all succeed

    @pytest.mark.asyncio
    async def test_leak_idxs_all_success_sparse(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs with sparse indices - all successful."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["success"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:5],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 2, 4],
            )

            assert len(result) == 5
            assert leak_idxs == [
                0,
                2,
                4,
            ]  # Should maintain same indices when all succeed

    @pytest.mark.asyncio
    async def test_leak_idxs_partial_failure_beginning(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs when first items fail."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # Pattern: fail, fail, success, success, success
            mock_process.side_effect = [
                mock_haver_responses["failure"],  # Index 0 fails
                mock_haver_responses["failure"],  # Index 1 fails
                mock_haver_responses[
                    "success"
                ],  # Index 2 succeeds -> new index 0
                mock_haver_responses[
                    "success"
                ],  # Index 3 succeeds -> new index 1
                mock_haver_responses[
                    "success"
                ],  # Index 4 succeeds -> new index 2
            ]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:5],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 2, 3, 4],
            )

            assert len(result) == 3  # Only 3 successful
            # Original indices [0,1,2,3,4] -> indices 0,1 failed, so [2,3,4] become [0,1,2]
            assert leak_idxs == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_leak_idxs_partial_failure_middle(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs when middle items fail."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # Pattern: success, fail, fail, success, success
            mock_process.side_effect = [
                mock_haver_responses[
                    "success"
                ],  # Index 0 succeeds -> new index 0
                mock_haver_responses["failure"],  # Index 1 fails
                mock_haver_responses["failure"],  # Index 2 fails
                mock_haver_responses[
                    "success"
                ],  # Index 3 succeeds -> new index 1
                mock_haver_responses[
                    "success"
                ],  # Index 4 succeeds -> new index 2
            ]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:5],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 2, 3, 4],
            )

            assert len(result) == 3  # Only 3 successful
            # Original indices [0,1,2,3,4] -> indices 1,2 failed, so [0,3,4] become [0,1,2]
            assert leak_idxs == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_leak_idxs_partial_failure_sparse_leak_indices(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs with sparse leak indices and failures."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # Pattern: success, fail, success, fail, success
            mock_process.side_effect = [
                mock_haver_responses[
                    "success"
                ],  # Index 0 succeeds -> new index 0
                mock_haver_responses["failure"],  # Index 1 fails
                mock_haver_responses[
                    "success"
                ],  # Index 2 succeeds -> new index 1
                mock_haver_responses["failure"],  # Index 3 fails
                mock_haver_responses[
                    "success"
                ],  # Index 4 succeeds -> new index 2
            ]

            # Only leak indices 0, 2, 4 (which all succeed)
            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:5],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 2, 4],
            )

            assert len(result) == 3  # Only 3 successful
            # Original leak indices [0,2,4] all succeed and become [0,1,2]
            assert leak_idxs == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_leak_idxs_partial_failure_some_leak_indices_fail(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs when some of the leak indices themselves fail."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # Pattern: success, fail, success, success, fail
            mock_process.side_effect = [
                mock_haver_responses[
                    "success"
                ],  # Index 0 succeeds -> new index 0
                mock_haver_responses["failure"],  # Index 1 fails
                mock_haver_responses[
                    "success"
                ],  # Index 2 succeeds -> new index 1
                mock_haver_responses[
                    "success"
                ],  # Index 3 succeeds -> new index 2
                mock_haver_responses["failure"],  # Index 4 fails
            ]

            # Leak indices include some that fail: [0,1,3,4] where 1 and 4 fail
            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:5],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 3, 4],
            )

            assert len(result) == 3  # Only 3 successful (indices 0, 2, 3)
            # Original leak indices [0,1,3,4] -> 1,4 failed, so only 0,3 remain
            # These become new indices [0,2] (since original 0->0, original 3->2)
            assert leak_idxs == [0, 2]

    @pytest.mark.asyncio
    async def test_leak_idxs_all_failure(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs when all processing fails."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["failure"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:3],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 2],
            )

            assert len(result) == 0  # All failed
            assert leak_idxs == []  # Should be empty when all fail

    @pytest.mark.asyncio
    async def test_leak_idxs_out_of_bounds(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs with indices beyond the input list."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["success"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:2],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 5, 10],  # 5,10 are out of bounds
            )

            assert len(result) == 2  # Only 2 inputs processed
            # Only indices 0,1 are valid, out-of-bounds indices should be ignored
            assert leak_idxs == [0, 1]

    @pytest.mark.asyncio
    async def test_leak_idxs_multiple_dataframes_per_metadata(
        self, haver_info_list
    ):
        """Test leak_idxs when each metadata returns multiple dataframes."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # Each metadata returns 2 dataframes
            multiple_df_response = [
                MetadataDataFrame(
                    df=pd.DataFrame({"date": ["2023-01-01"], "value": [i]}),
                    metadata_json={"timestamp_columns": ["date"]},
                    description=f"DF {i}",
                    timestamp_key="date",
                )
                for i in range(2)
            ]
            mock_process.return_value = multiple_df_response

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:2],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1],
            )

            assert len(result) == 4  # 2 metadata × 2 dataframes each = 4 total
            # Leak index 0 should map to dataframes [0,1], leak index 1 should map to [2,3]
            assert leak_idxs == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_leak_idxs_consistency_after_failures(
        self, haver_info_list, mock_haver_responses
    ):
        """Test that leak_idxs indices correctly correspond to result dataframes after failures."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # Pattern: success, fail, success
            mock_process.side_effect = [
                [
                    MetadataDataFrame(
                        df=pd.DataFrame(
                            {"date": ["2023-01-01"], "value": [100]}
                        ),
                        metadata_json={"timestamp_columns": ["date"]},
                        description="First Success",
                        timestamp_key="date",
                    )
                ],
                None,  # Failure
                [
                    MetadataDataFrame(
                        df=pd.DataFrame(
                            {"date": ["2023-01-01"], "value": [300]}
                        ),
                        metadata_json={"timestamp_columns": ["date"]},
                        description="Third Success",
                        timestamp_key="date",
                    )
                ],
            ]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:3],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 2],  # Skip index 1
            )

            assert len(result) == 2  # Only 2 successful
            assert leak_idxs == [0, 1]  # Remapped to new indices

            # Verify that the leak indices correctly map to the expected dataframes
            assert result[leak_idxs[0]].description == "First Success"
            assert result[leak_idxs[1]].description == "Third Success"

    @pytest.mark.asyncio
    async def test_leak_idxs_with_duplicate_indices(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs behavior with duplicate indices in input."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["success"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:3],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 0, 1, 1, 2],  # Duplicates
            )

            assert len(result) == 3
            # Should deduplicate and maintain order
            assert leak_idxs == [0, 1, 2]  # Each index appears only once

    @pytest.mark.asyncio
    async def test_leak_idxs_with_negative_indices(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs behavior with negative indices (should be ignored)."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            mock_process.return_value = mock_haver_responses["success"]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:3],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[-1, 0, 1, -5],  # Negative indices
            )

            assert len(result) == 3
            # Should ignore negative indices and only keep valid ones
            assert leak_idxs == [0, 1]

    @pytest.mark.asyncio
    async def test_leak_idxs_exception_during_processing(
        self, haver_info_list, mock_haver_responses
    ):
        """Test leak_idxs behavior when exceptions occur during processing."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # Pattern: success, exception, success
            mock_process.side_effect = [
                mock_haver_responses["success"],  # Index 0 succeeds
                Exception("Processing error"),  # Index 1 raises exception
                mock_haver_responses["success"],  # Index 2 succeeds
            ]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:3],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 2],
            )

            assert (
                len(result) == 2
            )  # Only 2 successful (exceptions are handled)
            # Original indices [0,1,2] -> index 1 failed with exception, so [0,2] become [0,1]
            assert leak_idxs == [0, 1]

    @pytest.mark.asyncio
    async def test_leak_idxs_mixed_weather_and_haver_failures(
        self, haver_info_list
    ):
        """Test leak_idxs with mixed data sources and failure patterns."""
        weather_info = WeatherMetadataAccessInfo(
            data_source="weather",
            name="Test City",
            description="Test Weather",
            file_name="test_weather",
            location_data=WeatherStackLocation(
                name="Test City",
                latitude=40.7,
                longitude=-74.0,
                country_code="US",
            ),
            weather_parameters=WeatherParameters(temperature=True),
            units="m",
            frequency="day",
            time_period=TimePeriod(
                min_timestamp="2023-01-01T00:00:00",
                forecast_timestamp="2023-01-02T00:00:00",
            ),
        )

        with (
            patch(
                "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
            ) as mock_haver,
            patch(
                "synthefy_pkg.app.utils.external_metadata_utils.process_weatherstack_metadata_access_info"
            ) as mock_weather,
        ):
            haver_df = MetadataDataFrame(
                df=pd.DataFrame({"date": ["2023-01-01"], "value": [100]}),
                metadata_json={"timestamp_columns": ["date"]},
                description="Haver Success",
                timestamp_key="date",
            )
            weather_df = MetadataDataFrame(
                df=pd.DataFrame({"date": ["2023-01-01"], "value": [25]}),
                metadata_json={"timestamp_columns": ["date"]},
                description="Weather Success",
                timestamp_key="date",
            )

            # Pattern: haver succeeds, weather fails, haver fails, weather succeeds
            mock_haver.side_effect = [
                [haver_df],  # Index 0 succeeds
                None,  # Index 2 fails
            ]
            mock_weather.side_effect = [
                None,  # Index 1 fails
                [weather_df],  # Index 3 succeeds
            ]

            mixed_metadata = [
                haver_info_list[0],
                weather_info,
                haver_info_list[1],
                weather_info,
            ]

            result, leak_idxs = await prepare_metadata_dataframes(
                mixed_metadata,
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 1, 2, 3],
            )

            assert len(result) == 2  # Only indices 0 and 3 succeeded
            assert leak_idxs == [0, 1]  # Remapped indices
            assert result[0].description == "Haver Success"
            assert result[1].description == "Weather Success"

    @pytest.mark.asyncio
    async def test_leak_idxs_uneven_dataframe_counts(self, haver_info_list):
        """Test leak_idxs when different metadata items return different numbers of dataframes."""
        with patch(
            "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
        ) as mock_process:
            # First metadata returns 1 dataframe, second returns 3 dataframes, third returns 2
            mock_process.side_effect = [
                [
                    MetadataDataFrame(  # Index 0 -> result index 0
                        df=pd.DataFrame(
                            {"date": ["2023-01-01"], "value": [100]}
                        ),
                        metadata_json={},
                        description="Single DF",
                        timestamp_key="date",
                    )
                ],
                [
                    MetadataDataFrame(  # Index 1 -> result indices 1,2,3
                        df=pd.DataFrame({"date": ["2023-01-01"], "value": [i]}),
                        metadata_json={},
                        description=f"Triple DF {i}",
                        timestamp_key="date",
                    )
                    for i in range(3)
                ],
                [
                    MetadataDataFrame(  # Index 2 -> result indices 4,5
                        df=pd.DataFrame(
                            {"date": ["2023-01-01"], "value": [200 + i]}
                        ),
                        metadata_json={},
                        description=f"Double DF {i}",
                        timestamp_key="date",
                    )
                    for i in range(2)
                ],
            ]

            result, leak_idxs = await prepare_metadata_dataframes(
                haver_info_list[:3],
                "test-bucket",
                ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
                leak_idxs=[0, 2],  # Skip index 1
            )

            assert len(result) == 6  # 1 + 3 + 2 = 6 total dataframes
            # Index 0 maps to result index 0, index 2 maps to result indices 4,5
            assert leak_idxs == [0, 4, 5]
            assert result[0].description == "Single DF"
            assert result[4].description == "Double DF 0"
            assert result[5].description == "Double DF 1"


class TestEdgeCasesAndErrorHandling:
    """Test class for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_invalid_metadata_info_type(self):
        """Test handling invalid metadata info type in prepare_metadata_dataframes."""
        # Test with empty list instead of invalid type to avoid type errors
        result, leak_idxs = await prepare_metadata_dataframes(
            [],
            "test-bucket",
            ["2023-01-01T00:00:00", "2023-01-02T00:00:00"],
            [],
        )

        # Should handle gracefully and return empty results
        assert len(result) == 0
        assert len(leak_idxs) == 0

    def test_weatherstack_to_df_with_null_values(self):
        """Test WeatherStack conversion with null values in response."""
        response = {
            "historical": {
                "2023-01-01": {
                    "hourly": [
                        {"time": "0", "temperature": None, "humidity": 80},
                        {"time": "300", "temperature": 22, "humidity": None},
                    ]
                }
            }
        }
        df = weatherstack_to_df(response)

        assert not df.empty
        assert len(df) == 2
        assert pd.isna(df["temperature"].iloc[0])
        assert pd.isna(df["humidity"].iloc[1])

    # TODO: Fix this test
    # @pytest.mark.asyncio
    # async def test_prepare_metadata_dataframes_partial_failures(self):
    #     """Test prepare_metadata_dataframes with partial failures and leak index adjustments."""
    #     haver_info = HaverMetadataAccessInfo(
    #         data_source="haver",
    #         database_name="test_db",
    #         name="test_series",
    #         description="Test",
    #         start_date=20230101,
    #         file_name=None,
    #     )
    #
    #     # Mock one success and one failure
    #     with patch(
    #         "synthefy_pkg.app.utils.external_metadata_utils.process_haver_metadata_access_info"
    #     ) as mock_process:
    #         mock_df = pd.DataFrame({"date": ["2023-01-01"], "value": [100]})
    #         mock_process.side_effect = [
    #             [
    #                 MetadataDataFrame(
    #                     df=mock_df, metadata_json={}, description="Test1"
    #                 )
    #             ],
    #             None,  # Failure
    #             [
    #                 MetadataDataFrame(
    #                     df=mock_df, metadata_json={}, description="Test2"
    #                 )
    #             ],
    #         ]
    #
    #         result, leak_idxs = await prepare_metadata_dataframes(
    #             [haver_info, haver_info, haver_info],
    #             "test-bucket",
    #             leak_idxs=[0, 1, 2],
    #         )
    #
    #         # Should have 2 successful results (indices 0 and 2 from original)
    #         assert len(result) == 2
    #         # Leak indices should be adjusted: original [0,1,2] -> new [0,1]
    #         # (since index 1 failed, the third item becomes index 1)
    #         assert leak_idxs == [0, 1]


class TestCovariateGridEndpointIntegration:
    """Integration tests for the covariate grid endpoint."""

    @pytest.mark.asyncio
    async def test_generate_covariate_grid_endpoint_integration(self):
        """Test the covariate grid endpoint with realistic data."""
        request = CovariateGridRequest(
            available_covariates=[
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
            ],
        )

        response = await generate_covariate_grid_endpoint(request)

        # Verify response structure
        assert isinstance(response, CovariateGridResponse)
        assert response.status == StatusCode.ok
        assert "Generated" in response.message
        assert [] in response.covariate_grid  # Empty set included

        # Verify all combinations are valid
        for combination in response.covariate_grid:
            assert isinstance(combination, list)
            for covariate in combination:
                assert isinstance(covariate, str)
                assert covariate in request.available_covariates

    @pytest.mark.asyncio
    async def test_generate_covariate_grid_endpoint_single_covariate(self):
        """Test the endpoint with a single covariate."""
        request = CovariateGridRequest(
            available_covariates=["temperature"],
        )

        response = await generate_covariate_grid_endpoint(request)

        assert response.status == StatusCode.ok
        assert response.total_combinations == 2
        assert [] in response.covariate_grid
        assert ["temperature"] in response.covariate_grid

    @pytest.mark.asyncio
    async def test_generate_covariate_grid_endpoint_max_combinations_edge_case(
        self,
    ):
        """Test the endpoint when max_combinations equals total combinations."""
        request = CovariateGridRequest(
            available_covariates=["a", "b"],
        )

        response = await generate_covariate_grid_endpoint(request)

        assert response.status == StatusCode.ok
        assert response.total_combinations == 4
        # Should include all possible combinations
        expected = [[], ["a"], ["b"], ["a", "b"]]
        assert all(combo in response.covariate_grid for combo in expected)


class TestCovariateGridFunctionIntegration:
    """Integration tests for covariate grid functions working together."""

    def test_generate_and_find_best_integration(self):
        """Test that generate_covariate_combinations output works with find_best_covariate_combination."""
        # Generate combinations
        covariates = ["temperature", "humidity", "pressure"]
        combinations = generate_covariate_combinations(covariates, 30)

        # Create mock results DataFrame
        results_data = []
        for i, combo in enumerate(combinations):
            combo_str = ", ".join(combo) if combo else "None"
            # Simulate that temperature alone is best
            rmse = 0.2 if combo == ["temperature"] else 0.5 + i * 0.1
            results_data.append(
                {"Covariates_Str": combo_str, "rmse": rmse, "nmae": rmse * 0.8}
            )

        results_df = pd.DataFrame(results_data)

        # Find best combination
        best_result = find_best_covariate_combination(results_df)

        # Check that the result is a DataFrame
        assert isinstance(best_result, pd.DataFrame)
        # Check that it's sorted by RMSE (best first)
        assert best_result.index[0] == "temperature"  # Best combination
        assert best_result["rmse_mean"].iloc[0] == 0.2  # Best RMSE
        assert len(best_result) == len(
            combinations
        )  # All combinations included

    def test_covariate_combinations_exponential_growth(self):
        """Test that the number of combinations grows exponentially as expected."""
        test_cases = [
            (1, 2),  # 2^1 = 2
            (2, 4),  # 2^2 = 4
            (3, 8),  # 2^3 = 8
            (4, 16),  # 2^4 = 16
            (5, 30),  # 2^5 = 32>30=30
        ]

        for num_covariates, expected_combinations in test_cases:
            covariates = [f"var_{i}" for i in range(num_covariates)]
            combinations = generate_covariate_combinations(covariates, 30)
            assert len(combinations) == expected_combinations

    def test_find_best_with_tied_performance(self):
        """Test find_best_covariate_combination behavior with tied RMSE values."""
        df = pd.DataFrame(
            {
                "Covariates_Str": ["None", "temperature", "humidity"],
                "rmse": [0.3, 0.3, 0.3],  # All tied
                "nmae": [0.2, 0.2, 0.2],
            }
        )

        result = find_best_covariate_combination(df)

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should have all combinations with tied RMSE
        assert result["rmse_mean"].iloc[0] == 0.3
        assert len(result) == 3  # All combinations present
        # The exact choice among tied values is implementation-dependent
        assert result.index[0] in ["None", "temperature", "humidity"]

    def test_find_best_with_nan_values(self):
        """Test find_best_covariate_combination handling of NaN values."""
        df = pd.DataFrame(
            {
                "Covariates_Str": ["None", "temperature", "humidity"],
                "rmse": [0.3, np.nan, 0.5],
                "nmae": [0.2, 0.2, 0.4],
            }
        )

        result = find_best_covariate_combination(df)

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should pick "None" as it has the lowest non-NaN RMSE
        assert result.index[0] == "None"
        assert result["rmse_mean"].iloc[0] == 0.3


class TestFilterRequest:
    """Test class for FilterRequest model validation."""

    def test_valid_filter_request_with_filters(self):
        """Test creating a valid FilterRequest with filters."""
        filters = GroupLabelColumnFilters(
            filter=[{"category": ["A", "B"], "region": ["North"]}]
        )
        request = FilterRequest(
            user_id="test_user",
            file_key="path/to/file.parquet",
            timestamp_column="date",
            filters=filters,
        )

        assert request.user_id == "test_user"
        assert request.file_key == "path/to/file.parquet"
        assert request.timestamp_column == "date"
        assert request.filters == filters

    def test_valid_filter_request_with_empty_filters(self):
        """Test creating a valid FilterRequest with empty filters."""

        filters = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.SUM
        )
        request = FilterRequest(
            user_id="test_user",
            file_key="path/to/file.parquet",
            timestamp_column="date",
            filters=filters,
        )

        assert request.user_id == "test_user"
        assert request.file_key == "path/to/file.parquet"
        assert request.timestamp_column == "date"
        assert request.filters == filters

    def test_filter_request_invalid_file_extension(self):
        """Test that FilterRequest validates file extension."""

        filters = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.SUM
        )
        with pytest.raises(ValueError, match="file_key must end with .parquet"):
            FilterRequest(
                user_id="test_user",
                file_key="path/to/file.csv",
                timestamp_column="date",
                filters=filters,
            )

    def test_filter_request_none_user_id(self):
        """Test that FilterRequest validates user_id is not None."""

        filters = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.SUM
        )
        with pytest.raises(ValueError, match="user_id cannot be null or None"):
            FilterRequest(
                user_id=None,  # type: ignore
                file_key="path/to/file.parquet",
                timestamp_column="date",
                filters=filters,
            )

    def test_filter_request_none_file_key(self):
        """Test that FilterRequest validates file_key is not None."""

        filters = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.SUM
        )
        with pytest.raises(ValueError, match="file_key cannot be null or None"):
            FilterRequest(
                user_id="test_user",
                file_key=None,  # type: ignore
                timestamp_column="date",
                filters=filters,
            )


class TestFilterUtils:
    """Test class for FilterUtils with empty filters support."""

    def test_get_s3_key_with_none_filters(self):
        """Test FilterUtils.get_s3_key with empty filters."""
        result = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=GroupLabelColumnFilters(
                filter=[], aggregation_func=SupportedAggregationFunctions.SUM
            ),
        )
        expected = "test_user/foundation_models/test_dataset/unfiltered_agg=sum/data.parquet"
        assert result == expected

    def test_get_s3_key_with_filters(self):
        """Test FilterUtils.get_s3_key with actual filters."""
        filters = GroupLabelColumnFilters(
            filter=[{"category": ["A", "B"], "region": ["North"]}]
        )
        result = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters,
        )
        expected = "test_user/foundation_models/test_dataset/category=A+B/region=North/agg=sum/data.parquet"
        assert result == expected

    def test_get_metadata_key_with_none_filters(self):
        """Test FilterUtils.get_metadata_key with empty filters."""
        result = FilterUtils.get_metadata_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            filters=GroupLabelColumnFilters(
                filter=[], aggregation_func=SupportedAggregationFunctions.SUM
            ),
        )
        expected = "test_user/foundation_models/test_dataset/unfiltered_agg=sum/metadata.json"
        assert result == expected

    def test_get_metadata_key_with_filters(self):
        """Test FilterUtils.get_metadata_key with actual filters."""
        filters = GroupLabelColumnFilters(
            filter=[{"category": ["A", "B"], "region": ["North"]}]
        )
        result = FilterUtils.get_metadata_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            filters=filters,
        )
        expected = "test_user/foundation_models/test_dataset/category=A+B/region=North/agg=sum/metadata.json"
        assert result == expected

    def test_get_s3_key_with_different_aggregation_functions(self):
        """Test FilterUtils.get_s3_key with different aggregation functions."""
        # Test with MEAN aggregation
        filters_mean = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MEAN
        )
        result_mean = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters_mean,
        )
        expected_mean = "test_user/foundation_models/test_dataset/unfiltered_agg=mean/data.parquet"
        assert result_mean == expected_mean

        # Test with MODE aggregation
        filters_mode = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MODE
        )
        result_mode = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters_mode,
        )
        expected_mode = "test_user/foundation_models/test_dataset/unfiltered_agg=mode/data.parquet"
        assert result_mode == expected_mode

        # Test with MAX aggregation
        filters_max = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MAX
        )
        result_max = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters_max,
        )
        expected_max = "test_user/foundation_models/test_dataset/unfiltered_agg=max/data.parquet"
        assert result_max == expected_max

        # Test with MIN aggregation
        filters_min = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MIN
        )
        result_min = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters_min,
        )
        expected_min = "test_user/foundation_models/test_dataset/unfiltered_agg=min/data.parquet"
        assert result_min == expected_min

    def test_get_metadata_key_with_different_aggregation_functions(self):
        """Test FilterUtils.get_metadata_key with different aggregation functions."""
        # Test with MEAN aggregation
        filters_mean = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MEAN
        )
        result_mean = FilterUtils.get_metadata_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            filters=filters_mean,
        )
        expected_mean = "test_user/foundation_models/test_dataset/unfiltered_agg=mean/metadata.json"
        assert result_mean == expected_mean

        # Test with MODE aggregation
        filters_mode = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MODE
        )
        result_mode = FilterUtils.get_metadata_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            filters=filters_mode,
        )
        expected_mode = "test_user/foundation_models/test_dataset/unfiltered_agg=mode/metadata.json"
        assert result_mode == expected_mode

    def test_get_s3_key_with_filters_and_different_aggregation_functions(self):
        """Test FilterUtils.get_s3_key with filters and different aggregation functions."""
        # Test with MEAN aggregation and filters
        filters_mean = GroupLabelColumnFilters(
            filter=[{"category": ["A", "B"], "region": ["North"]}],
            aggregation_func=SupportedAggregationFunctions.MEAN,
        )
        result_mean = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters_mean,
        )
        expected_mean = "test_user/foundation_models/test_dataset/category=A+B/region=North/agg=mean/data.parquet"
        assert result_mean == expected_mean

        # Test with MAX aggregation and filters
        filters_max = GroupLabelColumnFilters(
            filter=[{"status": ["active"], "type": ["premium"]}],
            aggregation_func=SupportedAggregationFunctions.MAX,
        )
        result_max = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters_max,
        )
        expected_max = "test_user/foundation_models/test_dataset/status=active/type=premium/agg=max/data.parquet"
        assert result_max == expected_max

        # Test with MIN aggregation and complex filters
        filters_min = GroupLabelColumnFilters(
            filter=[
                {"category": ["A", "B", "C"], "region": ["North", "South"]}
            ],
            aggregation_func=SupportedAggregationFunctions.MIN,
        )
        result_min = FilterUtils.get_s3_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            ext="parquet",
            filters=filters_min,
        )
        expected_min = "test_user/foundation_models/test_dataset/category=A+B+C/region=North+South/agg=min/data.parquet"
        assert result_min == expected_min

    def test_get_metadata_key_with_filters_and_different_aggregation_functions(
        self,
    ):
        """Test FilterUtils.get_metadata_key with filters and different aggregation functions."""
        # Test with MODE aggregation and filters
        filters_mode = GroupLabelColumnFilters(
            filter=[{"category": ["A", "B"], "region": ["North"]}],
            aggregation_func=SupportedAggregationFunctions.MODE,
        )
        result_mode = FilterUtils.get_metadata_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            filters=filters_mode,
        )
        expected_mode = "test_user/foundation_models/test_dataset/category=A+B/region=North/agg=mode/metadata.json"
        assert result_mode == expected_mode

        # Test with MEAN aggregation and multiple filter conditions
        filters_mean = GroupLabelColumnFilters(
            filter=[
                {"type": ["A", "B"], "status": ["active"], "region": ["US"]}
            ],
            aggregation_func=SupportedAggregationFunctions.MEAN,
        )
        result_mean = FilterUtils.get_metadata_key(
            user_id="test_user",
            dataset_file_name="test_dataset",
            filters=filters_mean,
        )
        expected_mean = "test_user/foundation_models/test_dataset/region=US/status=active/type=A+B/agg=mean/metadata.json"
        assert result_mean == expected_mean

    def test_aggregation_func_consistency_between_s3_key_and_metadata_key(self):
        """Test that aggregation functions are consistent between S3 key and metadata key generation."""
        test_cases = [
            SupportedAggregationFunctions.SUM,
            SupportedAggregationFunctions.MEAN,
            SupportedAggregationFunctions.MODE,
            SupportedAggregationFunctions.MAX,
            SupportedAggregationFunctions.MIN,
        ]

        for agg_func in test_cases:
            # Test with empty filters
            filters_empty = GroupLabelColumnFilters(
                filter=[], aggregation_func=agg_func
            )
            s3_key = FilterUtils.get_s3_key(
                user_id="test_user",
                dataset_file_name="test_dataset",
                ext="parquet",
                filters=filters_empty,
            )
            metadata_key = FilterUtils.get_metadata_key(
                user_id="test_user",
                dataset_file_name="test_dataset",
                filters=filters_empty,
            )

            # Both should contain the same aggregation function
            assert f"agg={agg_func.value}" in s3_key
            assert f"agg={agg_func.value}" in metadata_key

            # Test with actual filters
            filters_with_data = GroupLabelColumnFilters(
                filter=[{"category": ["A"], "region": ["North"]}],
                aggregation_func=agg_func,
            )
            s3_key_with_filters = FilterUtils.get_s3_key(
                user_id="test_user",
                dataset_file_name="test_dataset",
                ext="parquet",
                filters=filters_with_data,
            )
            metadata_key_with_filters = FilterUtils.get_metadata_key(
                user_id="test_user",
                dataset_file_name="test_dataset",
                filters=filters_with_data,
            )

            # Both should contain the same aggregation function
            assert f"agg={agg_func.value}" in s3_key_with_filters
            assert f"agg={agg_func.value}" in metadata_key_with_filters

            # The paths should be identical except for the file extension
            s3_base_path = s3_key_with_filters.replace("/data.parquet", "")
            metadata_base_path = metadata_key_with_filters.replace(
                "/metadata.json", ""
            )
            assert s3_base_path == metadata_base_path


class TestReadAndFilterDataset:
    """Test class for _read_and_filter_dataset function."""

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_with_none_filters(self, tmp_path):
        """Test _read_and_filter_dataset returns entire dataset when filters is empty."""
        # Create test data
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "category": ["A", "B", "A", "C", "B"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Test with empty filters
        result = await _read_and_filter_dataset(
            str(temp_file),
            filters=GroupLabelColumnFilters(
                filter=[], aggregation_func=SupportedAggregationFunctions.SUM
            ),
            timestamp_column="date",
        )

        # Should return the entire dataset unchanged
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_with_filters(self, tmp_path):
        """Test _read_and_filter_dataset applies filters correctly."""
        # Create test data
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "category": ["A", "B", "A", "C", "B"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Create filters to only include category A
        filters = GroupLabelColumnFilters(filter=[{"category": ["A"]}])

        # Test with filters
        result = await _read_and_filter_dataset(
            str(temp_file), filters=filters, timestamp_column="date"
        )

        # Should only return rows with category A
        expected = df[df["category"] == "A"]
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_missing_filter_columns(
        self, tmp_path
    ):
        """Test _read_and_filter_dataset raises error for missing filter columns."""
        # Create test data
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "category": ["A", "B", "A", "C", "B"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Create filters with non-existent column
        filters = GroupLabelColumnFilters(
            filter=[{"nonexistent_column": ["A"]}]
        )

        # Test should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await _read_and_filter_dataset(
                str(temp_file), filters=filters, timestamp_column="date"
            )

        assert exc_info.value.status_code == 400
        assert "Filter columns not found in dataset" in str(
            exc_info.value.detail
        )

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_with_different_aggregation_functions(
        self, tmp_path
    ):
        """Test _read_and_filter_dataset with different aggregation functions and duplicate timestamps (numeric columns only)."""
        # Create test data with duplicate timestamps - only numeric columns
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-02",
                        "2023-01-03",
                    ]
                ),
                "value": [10, 20, 30, 40, 50],
                "amount": [100, 200, 300, 400, 500],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Test with SUM aggregation (empty filters)
        filters_sum = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.SUM
        )
        result_sum = await _read_and_filter_dataset(
            str(temp_file), filters=filters_sum, timestamp_column="date"
        )

        # Should aggregate duplicate timestamps using SUM
        expected_sum = (
            df.groupby("date")
            .agg(
                {
                    "value": "sum",
                    "amount": "sum",
                }
            )
            .reset_index()
        )
        pd.testing.assert_frame_equal(
            result_sum.sort_values("date").reset_index(drop=True),
            expected_sum.sort_values("date").reset_index(drop=True),
        )

        # Test with MEAN aggregation (empty filters)
        filters_mean = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MEAN
        )
        result_mean = await _read_and_filter_dataset(
            str(temp_file), filters=filters_mean, timestamp_column="date"
        )

        # Should aggregate duplicate timestamps using MEAN
        expected_mean = (
            df.groupby("date")
            .agg(
                {
                    "value": "mean",
                    "amount": "mean",
                }
            )
            .reset_index()
        )
        pd.testing.assert_frame_equal(
            result_mean.sort_values("date").reset_index(drop=True),
            expected_mean.sort_values("date").reset_index(drop=True),
        )

        # Test with MAX aggregation (empty filters)
        filters_max = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MAX
        )
        result_max = await _read_and_filter_dataset(
            str(temp_file), filters=filters_max, timestamp_column="date"
        )

        # Should aggregate duplicate timestamps using MAX
        expected_max = (
            df.groupby("date")
            .agg(
                {
                    "value": "max",
                    "amount": "max",
                }
            )
            .reset_index()
        )
        pd.testing.assert_frame_equal(
            result_max.sort_values("date").reset_index(drop=True),
            expected_max.sort_values("date").reset_index(drop=True),
        )

        # Test with MIN aggregation (empty filters)
        filters_min = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.MIN
        )
        result_min = await _read_and_filter_dataset(
            str(temp_file), filters=filters_min, timestamp_column="date"
        )

        # Should aggregate duplicate timestamps using MIN
        expected_min = (
            df.groupby("date")
            .agg(
                {
                    "value": "min",
                    "amount": "min",
                }
            )
            .reset_index()
        )
        pd.testing.assert_frame_equal(
            result_min.sort_values("date").reset_index(drop=True),
            expected_min.sort_values("date").reset_index(drop=True),
        )

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_duplicate_timestamps_with_non_numeric_columns_fails(
        self, tmp_path
    ):
        """Test _read_and_filter_dataset fails when duplicate timestamps exist with non-numeric columns."""
        # Create test data with duplicate timestamps and non-numeric columns with varying values
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-02",
                        "2023-01-03",
                    ]
                ),
                "category": [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ],  # Non-numeric column with varying values within groups
                "value": [10, 20, 30, 40, 50],
                "amount": [100, 200, 300, 400, 500],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Test with SUM aggregation (empty filters) - should fail
        filters_sum = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.SUM
        )

        with pytest.raises(HTTPException) as exc_info:
            await _read_and_filter_dataset(
                str(temp_file), filters=filters_sum, timestamp_column="date"
            )

        assert exc_info.value.status_code == 400
        assert (
            "duplicate timestamps and non-numeric columns with varying values"
            in str(exc_info.value.detail)
        )
        assert "category" in str(exc_info.value.detail)
        assert (
            "Aggregation strategy for non-numeric columns is not yet configurable"
            in str(exc_info.value.detail)
        )

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_duplicate_timestamps_with_consistent_non_numeric_columns_succeeds(
        self, tmp_path
    ):
        """Test _read_and_filter_dataset succeeds when duplicate timestamps exist with non-numeric columns that have consistent values."""
        # Create test data with duplicate timestamps and non-numeric columns with consistent values
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-02",
                        "2023-01-03",
                    ]
                ),
                "category": [
                    "A",
                    "A",
                    "B",
                    "B",
                    "C",
                ],  # Non-numeric column with consistent values within groups
                "value": [10, 20, 30, 40, 50],
                "amount": [100, 200, 300, 400, 500],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Test with SUM aggregation (empty filters) - should succeed
        filters_sum = GroupLabelColumnFilters(
            filter=[], aggregation_func=SupportedAggregationFunctions.SUM
        )

        result = await _read_and_filter_dataset(
            str(temp_file), filters=filters_sum, timestamp_column="date"
        )

        # Should have 3 rows (one for each unique timestamp)
        assert len(result) == 3
        # Check that aggregation worked correctly
        assert (
            result[result["date"] == pd.Timestamp("2023-01-01")]["value"].iloc[
                0
            ]
            == 30
        )  # 10 + 20
        assert (
            result[result["date"] == pd.Timestamp("2023-01-02")]["value"].iloc[
                0
            ]
            == 70
        )  # 30 + 40
        assert (
            result[result["date"] == pd.Timestamp("2023-01-03")]["value"].iloc[
                0
            ]
            == 50
        )  # Single value
        # Check that non-numeric columns were preserved correctly
        assert (
            result[result["date"] == pd.Timestamp("2023-01-01")][
                "category"
            ].iloc[0]
            == "A"
        )
        assert (
            result[result["date"] == pd.Timestamp("2023-01-02")][
                "category"
            ].iloc[0]
            == "B"
        )
        assert (
            result[result["date"] == pd.Timestamp("2023-01-03")][
                "category"
            ].iloc[0]
            == "C"
        )

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_with_filters_and_different_aggregation_functions(
        self, tmp_path
    ):
        """Test _read_and_filter_dataset with filters and different aggregation functions."""
        # Create test data with duplicate timestamps - keep it simple with mostly numeric data
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-02",
                        "2023-01-03",
                    ]
                ),
                "category": ["A", "A", "A", "A", "B"],
                "value": [10, 20, 30, 40, 50],
                "amount": [100, 200, 300, 400, 500],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Test with SUM aggregation and category filter (only category A, which has duplicate timestamps)
        filters_sum = GroupLabelColumnFilters(
            filter=[{"category": ["A"]}],
            aggregation_func=SupportedAggregationFunctions.SUM,
        )
        result_sum = await _read_and_filter_dataset(
            str(temp_file), filters=filters_sum, timestamp_column="date"
        )

        # Should filter to category A, then aggregate duplicate timestamps
        # After filtering by category=A, we have:
        # 2023-01-01: 2 rows -> values=[10,20], amounts=[100,200]
        # 2023-01-02: 2 rows -> values=[30,40], amounts=[300,400]

        # The first row should be the sum of 2023-01-01 duplicates
        # The second row should be the sum of 2023-01-02 duplicates
        assert len(result_sum) == 2
        assert result_sum.iloc[0]["value"] == 30  # 10 + 20
        assert result_sum.iloc[0]["amount"] == 300  # 100 + 200
        assert result_sum.iloc[1]["value"] == 70  # 30 + 40
        assert result_sum.iloc[1]["amount"] == 700  # 300 + 400

    @pytest.mark.asyncio
    async def test_read_and_filter_dataset_no_duplicate_timestamps(
        self, tmp_path
    ):
        """Test _read_and_filter_dataset when there are no duplicate timestamps."""
        # Create test data with unique timestamps
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "category": ["A", "B", "A", "C", "B"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Save to temporary parquet file
        temp_file = tmp_path / "test.parquet"
        df.to_parquet(temp_file, index=False)

        # Test with different aggregation functions (should not matter since no duplicates)
        for agg_func in [
            SupportedAggregationFunctions.SUM,
            SupportedAggregationFunctions.MEAN,
            SupportedAggregationFunctions.MAX,
            SupportedAggregationFunctions.MIN,
        ]:
            filters = GroupLabelColumnFilters(
                filter=[], aggregation_func=agg_func
            )
            result = await _read_and_filter_dataset(
                str(temp_file), filters=filters, timestamp_column="date"
            )

            # Should return the original dataset unchanged since no duplicates
            pd.testing.assert_frame_equal(result, df)


class TestRunFilterCoreCode:
    """Test class for run_filter_core_code function."""

    @pytest.fixture
    def mock_aioboto3_session(self):
        """Create a mock aioboto3 session."""
        return MagicMock()

    @pytest.fixture
    def sample_filter_request_with_filters(self):
        """Create a sample FilterRequest with filters."""
        filters = GroupLabelColumnFilters(filter=[{"category": ["A", "B"]}])
        return FilterRequest(
            user_id="test_user",
            file_key="test_user/dataset/test.parquet",
            timestamp_column="date",
            filters=filters,
        )

    @pytest.fixture
    def sample_filter_request_no_filters(self):
        """Create a sample FilterRequest without filters."""
        return FilterRequest(
            user_id="test_user",
            file_key="test_user/dataset/test.parquet",
            timestamp_column="date",
            filters=GroupLabelColumnFilters(
                filter=[], aggregation_func=SupportedAggregationFunctions.SUM
            ),
        )

    @pytest.mark.asyncio
    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    @patch("synthefy_pkg.app.routers.foundation_models.avalidate_file_exists")
    @patch("synthefy_pkg.app.routers.foundation_models._get_cached_dataset")
    @patch(
        "synthefy_pkg.app.routers.foundation_models._create_filtered_dataset"
    )
    async def test_run_filter_core_code_with_none_filters(
        self,
        mock_create_filtered_dataset,
        mock_get_cached_dataset,
        mock_validate_file_exists,
        mock_get_settings,
        sample_filter_request_no_filters,
        mock_aioboto3_session,
    ):
        """Test run_filter_core_code with empty filters."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_get_settings.return_value = mock_settings

        mock_validate_file_exists.return_value = (True, "File exists")
        mock_get_cached_dataset.return_value = None  # No cached dataset

        mock_response = MagicMock()
        mock_create_filtered_dataset.return_value = mock_response

        # Mock S3 client context
        mock_s3_client = MagicMock()
        mock_aioboto3_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )

        # Execute
        result = await run_filter_core_code(
            sample_filter_request_no_filters, mock_aioboto3_session
        )

        # Verify
        assert result == mock_response
        mock_validate_file_exists.assert_called_once()
        mock_get_cached_dataset.assert_called_once()
        mock_create_filtered_dataset.assert_called_once()

        # Check that the S3 key was generated with empty filters
        call_args = mock_get_cached_dataset.call_args[
            0
        ]  # Access positional args
        # The filters should be a GroupLabelColumnFilters object with empty filter list
        assert isinstance(call_args[6], GroupLabelColumnFilters)
        assert call_args[6].filter == []

    @pytest.mark.asyncio
    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    async def test_run_filter_core_code_missing_bucket_config(
        self,
        mock_get_settings,
        sample_filter_request_no_filters,
        mock_aioboto3_session,
    ):
        """Test run_filter_core_code raises error when bucket not configured."""
        # Setup mock with no bucket name
        mock_settings = MagicMock()
        mock_settings.bucket_name = None
        mock_get_settings.return_value = mock_settings

        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await run_filter_core_code(
                sample_filter_request_no_filters, mock_aioboto3_session
            )

        assert exc_info.value.status_code == 500
        assert "S3 bucket name not configured" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    @patch("synthefy_pkg.app.routers.foundation_models.avalidate_file_exists")
    async def test_run_filter_core_code_file_not_found(
        self,
        mock_validate_file_exists,
        mock_get_settings,
        sample_filter_request_no_filters,
        mock_aioboto3_session,
    ):
        """Test run_filter_core_code raises error when file not found."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_get_settings.return_value = mock_settings

        mock_validate_file_exists.return_value = (False, "File not found")

        # Mock S3 client context
        mock_s3_client = MagicMock()
        mock_aioboto3_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )

        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await run_filter_core_code(
                sample_filter_request_no_filters, mock_aioboto3_session
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "File not found"

    @pytest.mark.asyncio
    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    @patch("synthefy_pkg.app.routers.foundation_models.avalidate_file_exists")
    @patch("synthefy_pkg.app.routers.foundation_models._get_cached_dataset")
    @patch(
        "synthefy_pkg.app.routers.foundation_models._create_filtered_dataset"
    )
    async def test_run_filter_core_code_with_different_aggregation_functions(
        self,
        mock_create_filtered_dataset,
        mock_get_cached_dataset,
        mock_validate_file_exists,
        mock_get_settings,
        mock_aioboto3_session,
    ):
        """Test run_filter_core_code with different aggregation functions."""
        # Setup common mocks
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_get_settings.return_value = mock_settings

        mock_validate_file_exists.return_value = (True, "File exists")
        mock_get_cached_dataset.return_value = None  # No cached dataset

        mock_response = MagicMock()
        mock_create_filtered_dataset.return_value = mock_response

        # Mock S3 client context
        mock_s3_client = MagicMock()
        mock_aioboto3_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )

        # Test different aggregation functions
        test_agg_functions = [
            SupportedAggregationFunctions.SUM,
            SupportedAggregationFunctions.MEAN,
            SupportedAggregationFunctions.MAX,
            SupportedAggregationFunctions.MIN,
            SupportedAggregationFunctions.MODE,
        ]

        for agg_func in test_agg_functions:
            # Reset mocks for each test
            mock_get_cached_dataset.reset_mock()
            mock_create_filtered_dataset.reset_mock()

            # Create request with specific aggregation function
            filter_request = FilterRequest(
                user_id="test_user",
                file_key="test_user/dataset/test.parquet",
                timestamp_column="date",
                filters=GroupLabelColumnFilters(
                    filter=[], aggregation_func=agg_func
                ),
            )

            # Execute
            result = await run_filter_core_code(
                filter_request, mock_aioboto3_session
            )

            # Verify
            assert result == mock_response
            mock_validate_file_exists.assert_called()
            mock_get_cached_dataset.assert_called_once()
            mock_create_filtered_dataset.assert_called_once()

            # Check that the S3 key was generated with correct aggregation function
            call_args = mock_get_cached_dataset.call_args[
                0
            ]  # Access positional args
            filters_arg = call_args[6]  # filters parameter
            assert isinstance(filters_arg, GroupLabelColumnFilters)
            assert filters_arg.aggregation_func == agg_func
            assert filters_arg.filter == []

    @pytest.mark.asyncio
    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    @patch("synthefy_pkg.app.routers.foundation_models.avalidate_file_exists")
    @patch("synthefy_pkg.app.routers.foundation_models._get_cached_dataset")
    @patch(
        "synthefy_pkg.app.routers.foundation_models._create_filtered_dataset"
    )
    async def test_run_filter_core_code_with_filters_and_aggregation_functions(
        self,
        mock_create_filtered_dataset,
        mock_get_cached_dataset,
        mock_validate_file_exists,
        mock_get_settings,
        mock_aioboto3_session,
    ):
        """Test run_filter_core_code with actual filters and different aggregation functions."""
        # Setup common mocks
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_get_settings.return_value = mock_settings

        mock_validate_file_exists.return_value = (True, "File exists")
        mock_get_cached_dataset.return_value = None  # No cached dataset

        mock_response = MagicMock()
        mock_create_filtered_dataset.return_value = mock_response

        # Mock S3 client context
        mock_s3_client = MagicMock()
        mock_aioboto3_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )

        test_cases = [
            (SupportedAggregationFunctions.SUM, [{"category": ["A", "B"]}]),
            (SupportedAggregationFunctions.MEAN, [{"region": ["North"]}]),
            (
                SupportedAggregationFunctions.MAX,
                [{"status": ["active"], "type": ["premium"]}],
            ),
            (
                SupportedAggregationFunctions.MIN,
                [{"category": ["A"], "region": ["North", "South"]}],
            ),
        ]

        for agg_func, filter_conditions in test_cases:
            # Cast to proper type expected by GroupLabelColumnFilters
            typed_filter_conditions = cast(
                List[Dict[str, List[Union[str, int, float]]]], filter_conditions
            )
            # Reset mocks for each test
            mock_get_cached_dataset.reset_mock()
            mock_create_filtered_dataset.reset_mock()

            # Create request with specific aggregation function and filters
            filter_request = FilterRequest(
                user_id="test_user",
                file_key="test_user/dataset/test.parquet",
                timestamp_column="date",
                filters=GroupLabelColumnFilters(
                    filter=typed_filter_conditions, aggregation_func=agg_func
                ),
            )

            # Execute
            result = await run_filter_core_code(
                filter_request, mock_aioboto3_session
            )

            # Verify
            assert result == mock_response
            mock_validate_file_exists.assert_called()
            mock_get_cached_dataset.assert_called_once()
            mock_create_filtered_dataset.assert_called_once()

            # Check that the filters and aggregation function were passed correctly
            call_args = mock_get_cached_dataset.call_args[
                0
            ]  # Access positional args
            filters_arg = call_args[6]  # filters parameter
            assert isinstance(filters_arg, GroupLabelColumnFilters)
            assert filters_arg.aggregation_func == agg_func
            assert filters_arg.filter == typed_filter_conditions

    @pytest.mark.asyncio
    @patch("synthefy_pkg.app.routers.foundation_models.get_settings")
    @patch("synthefy_pkg.app.routers.foundation_models.avalidate_file_exists")
    @patch("synthefy_pkg.app.routers.foundation_models._get_cached_dataset")
    async def test_run_filter_core_code_cached_dataset_with_aggregation_functions(
        self,
        mock_get_cached_dataset,
        mock_validate_file_exists,
        mock_get_settings,
        mock_aioboto3_session,
    ):
        """Test run_filter_core_code when cached dataset exists with different aggregation functions."""
        # Setup common mocks
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_get_settings.return_value = mock_settings

        mock_validate_file_exists.return_value = (True, "File exists")

        # Mock a cached dataset response
        mock_cached_response = MagicMock()
        mock_get_cached_dataset.return_value = mock_cached_response

        # Mock S3 client context
        mock_s3_client = MagicMock()
        mock_aioboto3_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )

        # Test different aggregation functions with cached datasets
        for agg_func in [
            SupportedAggregationFunctions.SUM,
            SupportedAggregationFunctions.MEAN,
            SupportedAggregationFunctions.MAX,
        ]:
            # Reset mocks for each test
            mock_get_cached_dataset.reset_mock()
            mock_get_cached_dataset.return_value = mock_cached_response

            # Create request with specific aggregation function
            filter_request = FilterRequest(
                user_id="test_user",
                file_key="test_user/dataset/test.parquet",
                timestamp_column="date",
                filters=GroupLabelColumnFilters(
                    filter=[], aggregation_func=agg_func
                ),
            )

            # Execute
            result = await run_filter_core_code(
                filter_request, mock_aioboto3_session
            )

            # Verify that cached dataset was returned
            assert result == mock_cached_response
            mock_validate_file_exists.assert_called()
            mock_get_cached_dataset.assert_called_once()

            # Check that the S3 key was generated with correct aggregation function
            call_args = mock_get_cached_dataset.call_args[
                0
            ]  # Access positional args
            filters_arg = call_args[6]  # filters parameter
            assert isinstance(filters_arg, GroupLabelColumnFilters)
            assert filters_arg.aggregation_func == agg_func


class TestGenerateCovariateCombinations:
    """Comprehensive unit tests for generate_covariate_combinations function."""

    def test_empty_input(self):
        """Test with empty input list."""
        result = generate_covariate_combinations([], 30)
        assert result == [[]]  # Should only return empty combination

    def test_single_covariate(self):
        """Test with a single covariate."""
        result = generate_covariate_combinations(["temperature"], 30)
        expected = [[], ["temperature"]]
        assert result == expected

    def test_two_covariates(self):
        """Test with two covariates."""
        result = generate_covariate_combinations(
            ["temperature", "humidity"], 30
        )
        expected = [
            [],  # No covariates
            ["temperature"],  # Single covariates
            ["humidity"],
            ["temperature", "humidity"],  # All combinations
        ]
        assert result == expected

    def test_three_covariates(self):
        """Test with three covariates."""
        result = generate_covariate_combinations(
            ["temp", "humid", "pressure"], 30
        )
        expected = [
            [],
            ["temp"],
            ["humid"],
            ["pressure"],
            ["temp", "humid"],
            ["temp", "pressure"],
            ["humid", "pressure"],
            ["temp", "humid", "pressure"],
        ]
        assert result == expected

    def test_preserves_order(self):
        """Test that order of input covariates is preserved in combinations."""
        input_covariates = ["a", "b", "c"]
        result = generate_covariate_combinations(input_covariates, 30)

        # Check that combinations maintain the original order
        three_combo = ["a", "b", "c"]
        assert three_combo in result

        # Check two-element combinations maintain order
        ab_combo = ["a", "b"]
        ac_combo = ["a", "c"]
        bc_combo = ["b", "c"]
        assert ab_combo in result
        assert ac_combo in result
        assert bc_combo in result

    def test_no_duplicates_in_result(self):
        """Test that result contains no duplicate combinations."""
        result = generate_covariate_combinations(["x", "y", "z"], 30)
        # Convert to tuples for set comparison to check uniqueness
        result_tuples = [tuple(combo) for combo in result]
        assert len(result_tuples) == len(set(result_tuples))

    def test_correct_total_count(self):
        """Test that total combinations equals 2^n."""
        test_cases = [
            (0, 1),  # 2^0 = 1
            (1, 2),  # 2^1 = 2
            (2, 4),  # 2^2 = 4
            (3, 8),  # 2^3 = 8
            (4, 16),  # 2^4 = 16
        ]

        for n, expected_count in test_cases:
            covariates = [f"var{i}" for i in range(n)]
            result = generate_covariate_combinations(covariates, 30)
            assert len(result) == expected_count

    def test_empty_combination_always_first(self):
        """Test that empty combination is always the first element."""
        test_inputs = [
            [],
            ["a"],
            ["a", "b"],
            ["x", "y", "z", "w"],
        ]

        for input_covariates in test_inputs:
            result = generate_covariate_combinations(input_covariates, 30)
            assert result[0] == []

    def test_with_special_characters(self):
        """Test with covariate names containing special characters."""
        covariates = ["temp_avg", "humidity-level", "pressure.value"]
        result = generate_covariate_combinations(covariates, 30)

        # Should include all single covariates
        assert ["temp_avg"] in result
        assert ["humidity-level"] in result
        assert ["pressure.value"] in result

        # Should include the full combination
        assert ["temp_avg", "humidity-level", "pressure.value"] in result

    def test_with_numeric_strings(self):
        """Test with numeric string covariate names."""
        covariates = ["1", "2", "3"]
        result = generate_covariate_combinations(covariates, 30)

        assert len(result) == 8  # 2^3
        assert [] in result
        assert ["1", "2", "3"] in result

    def test_large_input_performance(self):
        """Test with larger input to ensure reasonable performance."""
        # Test with 10 covariates (will generate 2^10 = 1024 combinations)
        covariates = [f"covar_{i}" for i in range(10)]
        result = generate_covariate_combinations(covariates, 30)

        assert len(result) == 30  # 2^10
        assert [] in result  # Empty combination

    def test_result_types(self):
        """Test that result contains proper list types."""
        result = generate_covariate_combinations(["a", "b"], 30)

        # Result should be a list
        assert isinstance(result, list)

        # Each combination should be a list
        for combo in result:
            assert isinstance(combo, list)

        # Each element in combinations should be a string
        for combo in result:
            for item in combo:
                assert isinstance(item, str)


# Metadata merge functionality tests
class TestMergeUserAndMetadataDataframes:
    """Test cases for _merge_user_and_metadata_dataframes function."""

    def test_exact_timestamp_matches(self):
        """Test merge when user and metadata have exact timestamp matches."""
        # Create user DataFrame
        user_timestamps = pd.date_range("2023-01-01", periods=5, freq="D")
        user_df = pd.DataFrame(
            {"timestamp": user_timestamps, "user_value": [10, 20, 30, 40, 50]}
        )

        # Create metadata DataFrame with exact same timestamps
        metadata_df = pd.DataFrame(
            {
                "timestamp": user_timestamps,
                "metadata_value": [100, 200, 300, 400, 500],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        # Should have exact matches
        assert len(result_df) == 5
        assert common_count == 5
        assert list(result_df["user_value"]) == [10, 20, 30, 40, 50]
        assert list(result_df["metadata_value"]) == [100, 200, 300, 400, 500]

    def test_backward_fill_behavior(self):
        """Test that merge_asof correctly fills backwards (latest available metadata)."""
        # User timestamps: every day
        user_timestamps = pd.date_range("2023-01-01", periods=10, freq="D")
        user_df = pd.DataFrame(
            {"timestamp": user_timestamps, "user_value": list(range(10))}
        )

        # Metadata timestamps: every 3 days (sparser)
        metadata_timestamps = pd.date_range("2023-01-01", periods=4, freq="3D")
        metadata_df = pd.DataFrame(
            {
                "timestamp": metadata_timestamps,
                "metadata_value": [100, 200, 300, 400],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        # Should have 10 rows (all user timestamps)
        assert len(result_df) == 10
        assert common_count == 4  # Only 4 exact matches

        # Check backward fill behavior
        expected_metadata = [100, 100, 100, 200, 200, 200, 300, 300, 300, 400]
        assert list(result_df["metadata_value"]) == expected_metadata

    def test_no_metadata_before_user_data(self):
        """Test behavior when metadata starts after some user data."""
        # User data starts earlier
        user_timestamps = pd.date_range("2023-01-01", periods=5, freq="D")
        user_df = pd.DataFrame(
            {"timestamp": user_timestamps, "user_value": [10, 20, 30, 40, 50]}
        )

        # Metadata starts 2 days later
        metadata_timestamps = pd.date_range("2023-01-03", periods=3, freq="D")
        metadata_df = pd.DataFrame(
            {
                "timestamp": metadata_timestamps,
                "metadata_value": [300, 400, 500],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        # Should have 5 rows
        assert len(result_df) == 5
        assert common_count == 3  # Only last 3 match exactly

        # First two should have NaN, rest should have values
        assert pd.isna(result_df.loc[0, "metadata_value"])
        assert pd.isna(result_df.loc[1, "metadata_value"])
        assert result_df.loc[2, "metadata_value"] == 300
        assert result_df.loc[3, "metadata_value"] == 400
        assert result_df.loc[4, "metadata_value"] == 500

    def test_unsorted_input_data(self):
        """Test that function handles unsorted input data correctly."""
        # Create unsorted user data
        user_timestamps = [
            "2023-01-03",
            "2023-01-01",
            "2023-01-05",
            "2023-01-02",
            "2023-01-04",
        ]
        user_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(user_timestamps),
                "user_value": [30, 10, 50, 20, 40],
            }
        )

        # Create unsorted metadata
        metadata_timestamps = ["2023-01-02", "2023-01-04", "2023-01-01"]
        metadata_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(metadata_timestamps),
                "metadata_value": [200, 400, 100],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        # Should be sorted by timestamp in result
        assert len(result_df) == 5
        expected_timestamps = pd.to_datetime(
            [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ]
        )
        pd.testing.assert_series_equal(
            result_df["timestamp"].reset_index(drop=True),
            pd.Series(expected_timestamps),
            check_names=False,
        )

        # Check values are correctly aligned
        assert list(result_df["user_value"]) == [10, 20, 30, 40, 50]
        assert list(result_df["metadata_value"]) == [100, 200, 200, 400, 400]

    def test_different_timestamp_column_name(self):
        """Test using a different column name for timestamps."""
        user_timestamps = pd.date_range("2023-01-01", periods=3, freq="D")
        user_df = pd.DataFrame(
            {"date_col": user_timestamps, "user_value": [10, 20, 30]}
        )

        metadata_df = pd.DataFrame(
            {"timestamp": user_timestamps, "metadata_value": [100, 200, 300]}
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "date_col"
        )

        assert len(result_df) == 3
        assert common_count == 3
        assert "date_col" in result_df.columns
        assert list(result_df["metadata_value"]) == [100, 200, 300]

    def test_multiple_metadata_columns(self):
        """Test merging with multiple metadata columns."""
        user_timestamps = pd.date_range("2023-01-01", periods=3, freq="D")
        user_df = pd.DataFrame(
            {"timestamp": user_timestamps, "user_value": [10, 20, 30]}
        )

        metadata_df = pd.DataFrame(
            {
                "timestamp": user_timestamps,
                "temperature": [25.5, 26.0, 24.5],
                "humidity": [60, 65, 55],
                "pressure": [1013, 1015, 1012],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        assert len(result_df) == 3
        assert common_count == 3
        assert "temperature" in result_df.columns
        assert "humidity" in result_df.columns
        assert "pressure" in result_df.columns
        assert list(result_df["temperature"]) == [25.5, 26.0, 24.5]

    def test_timezone_handling(self):
        """Test that timezone-aware timestamps are handled correctly."""
        # Create timezone-aware timestamps
        user_timestamps = pd.date_range(
            "2023-01-01", periods=3, freq="D", tz="UTC"
        )
        user_df = pd.DataFrame(
            {"timestamp": user_timestamps, "user_value": [10, 20, 30]}
        )

        metadata_timestamps = pd.date_range(
            "2023-01-01", periods=3, freq="D", tz="US/Eastern"
        )
        metadata_df = pd.DataFrame(
            {
                "timestamp": metadata_timestamps,
                "metadata_value": [100, 200, 300],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        # Should still work and merge correctly
        assert len(result_df) == 3
        assert not result_df["metadata_value"].isna().any()

    def test_empty_dataframes(self):
        """Test behavior with empty DataFrames."""
        # Empty user DataFrame
        user_df = pd.DataFrame(columns=["timestamp", "user_value"])
        metadata_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "metadata_value": [100, 200, 300],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        assert len(result_df) == 0
        assert common_count == 0

    def test_intraday_timestamps(self):
        """Test with high-frequency intraday timestamps."""
        # User data: every hour
        user_timestamps = pd.date_range("2023-01-01", periods=24, freq="H")
        user_df = pd.DataFrame(
            {"timestamp": user_timestamps, "user_value": list(range(24))}
        )

        # Metadata: every 6 hours
        metadata_timestamps = pd.date_range("2023-01-01", periods=4, freq="6H")
        metadata_df = pd.DataFrame(
            {
                "timestamp": metadata_timestamps,
                "metadata_value": [100, 200, 300, 400],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, metadata_df, "timestamp"
        )

        assert len(result_df) == 24
        assert common_count == 4

        # Check that each 6-hour period gets the right metadata value
        for i in range(6):
            assert result_df.loc[i, "metadata_value"] == 100
        for i in range(6, 12):
            assert result_df.loc[i, "metadata_value"] == 200


class TestCombineMetadataDataframes:
    """Test cases for _combine_metadata_dataframes function."""

    def test_single_metadata_dataframe(self):
        """Test combining a single metadata DataFrame."""

        # Mock metadata object
        metadata_obj = MagicMock()
        metadata_obj.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "value": [100, 200, 300],
            }
        )
        metadata_obj.description = "Test weather data"
        metadata_obj.timestamp_key = "timestamp"
        metadata_obj.metadata_json = {"columns": [{"id": "temperature"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj]
        )

        assert combined_df is not None
        assert len(combined_df) == 3
        assert "temperature" in combined_df.columns
        assert columns == ["temperature"]
        assert description == "Test weather data"

    def test_multiple_metadata_dataframes(self):
        """Test combining multiple metadata DataFrames."""

        # First metadata object
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "value": [25.0, 26.0, 24.0],
            }
        )
        metadata_obj1.description = "Temperature data"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "temperature"}]}

        # Second metadata object
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "value": [60, 65, 55],
            }
        )
        metadata_obj2.description = "Humidity data"
        metadata_obj2.timestamp_key = "timestamp"
        metadata_obj2.metadata_json = {"columns": [{"id": "humidity"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert len(combined_df) == 3
        assert "temperature" in combined_df.columns
        assert "humidity" in combined_df.columns
        assert set(columns) == {"temperature", "humidity"}
        assert "Temperature data; Humidity data" in description

    def test_multiple_metadata_different_timestamps(self):
        """Test combining metadata DataFrames with different timestamps."""

        # First metadata object - daily data
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
                "value": [25.0, 26.0, 24.0, 27.0, 23.0],
            }
        )
        metadata_obj1.description = "Temperature data"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "temperature"}]}

        # Second metadata object - every 2 days
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-02", periods=3, freq="2D"),
                "value": [60, 65, 55],
            }
        )
        metadata_obj2.description = "Humidity data"
        metadata_obj2.timestamp_key = "timestamp"
        metadata_obj2.metadata_json = {"columns": [{"id": "humidity"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert len(combined_df) == 6  # Union of all timestamps
        assert "temperature" in combined_df.columns
        assert "humidity" in combined_df.columns
        assert set(columns) == {"temperature", "humidity"}

        # Check that NaN values exist where timestamps don't overlap
        humidity_series = combined_df["humidity"]
        assert pd.isna(
            humidity_series.iloc[0]
        )  # 2023-01-01 has no humidity data
        assert not pd.isna(
            humidity_series.iloc[1]
        )  # 2023-01-02 has humidity data
        assert pd.isna(
            humidity_series.iloc[2]
        )  # 2023-01-03 has no humidity data

    def test_multiple_metadata_no_timestamp_overlap(self):
        """Test combining metadata DataFrames with completely non-overlapping timestamps."""

        # First metadata object - January data
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "value": [25.0, 26.0, 24.0],
            }
        )
        metadata_obj1.description = "Temperature data"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "temperature"}]}

        # Second metadata object - February data (no overlap)
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-02-01", periods=3, freq="D"),
                "value": [60, 65, 55],
            }
        )
        metadata_obj2.description = "Humidity data"
        metadata_obj2.timestamp_key = "timestamp"
        metadata_obj2.metadata_json = {"columns": [{"id": "humidity"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert len(combined_df) == 6  # All timestamps from both sources
        assert "temperature" in combined_df.columns
        assert "humidity" in combined_df.columns

        # All temperature values should be NaN for February dates
        feb_data = combined_df[combined_df["timestamp"] >= "2023-02-01"]
        assert feb_data["temperature"].isna().all()

        # All humidity values should be NaN for January dates
        jan_data = combined_df[combined_df["timestamp"] < "2023-02-01"]
        assert jan_data["humidity"].isna().all()

    def test_multiple_metadata_sparse_timestamps(self):
        """Test combining metadata with very sparse, irregular timestamps."""

        # First metadata object - irregular timestamps
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2023-01-01", "2023-01-15", "2023-02-28"]
                ),
                "value": [25.0, 26.0, 24.0],
            }
        )
        metadata_obj1.description = "Temperature data"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "temperature"}]}

        # Second metadata object - different irregular timestamps
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2023-01-10", "2023-01-20", "2023-02-15"]
                ),
                "value": [60, 65, 55],
            }
        )
        metadata_obj2.description = "Humidity data"
        metadata_obj2.timestamp_key = "timestamp"
        metadata_obj2.metadata_json = {"columns": [{"id": "humidity"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert len(combined_df) == 6  # All unique timestamps
        assert "temperature" in combined_df.columns
        assert "humidity" in combined_df.columns

        # Check that only one value per timestamp is non-NaN per column
        temp_non_null_count = combined_df["temperature"].notna().sum()
        humidity_non_null_count = combined_df["humidity"].notna().sum()
        assert temp_non_null_count == 3
        assert humidity_non_null_count == 3

    def test_multiple_metadata_different_column_names(self):
        """Test combining metadata with different timestamp column names."""

        # First metadata object - uses 'timestamp'
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "value": [25.0, 26.0, 24.0],
            }
        )
        metadata_obj1.description = "Temperature data"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "temperature"}]}

        # Second metadata object - uses 'date' as timestamp column
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-02", periods=3, freq="D"),
                "value": [60, 65, 55],
            }
        )
        metadata_obj2.description = "Humidity data"
        metadata_obj2.timestamp_key = "date"  # Different timestamp column name
        metadata_obj2.metadata_json = {"columns": [{"id": "humidity"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert len(combined_df) == 4  # Union of timestamps
        assert "temperature" in combined_df.columns
        assert "humidity" in combined_df.columns
        assert (
            "timestamp" in combined_df.columns
        )  # Standardized timestamp column
        assert (
            "date" not in combined_df.columns
        )  # Should be renamed to timestamp

    def test_multiple_metadata_mixed_frequencies(self):
        """Test combining metadata with different frequencies (hourly vs daily)."""

        # First metadata object - daily data
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=2, freq="D"),
                "value": [25.0, 26.0],
            }
        )
        metadata_obj1.description = "Daily temperature"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "daily_temp"}]}

        # Second metadata object - hourly data
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=25, freq="H"),
                "value": list(range(25)),
            }
        )
        metadata_obj2.description = "Hourly wind speed"
        metadata_obj2.timestamp_key = "timestamp"
        metadata_obj2.metadata_json = {"columns": [{"id": "hourly_wind"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert len(combined_df) == 25  # Should have all hourly timestamps
        assert "daily_temp" in combined_df.columns
        assert "hourly_wind" in combined_df.columns

        # Daily temp should have many NaN values (only 2 non-null values)
        daily_temp_non_null = combined_df["daily_temp"].notna().sum()
        hourly_wind_non_null = combined_df["hourly_wind"].notna().sum()
        assert daily_temp_non_null == 2
        assert hourly_wind_non_null == 25

    def test_empty_metadata_dataframe_combination(self):
        """Test combining when one metadata DataFrame is empty."""

        # First metadata object - has data
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "value": [25.0, 26.0, 24.0],
            }
        )
        metadata_obj1.description = "Temperature data"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "temperature"}]}

        # Second metadata object - empty DataFrame
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(columns=["timestamp", "value"])
        metadata_obj2.description = "Empty humidity data"
        metadata_obj2.timestamp_key = "timestamp"
        metadata_obj2.metadata_json = {"columns": [{"id": "humidity"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert len(combined_df) == 3  # Only timestamps from first metadata
        assert "temperature" in combined_df.columns
        assert "humidity" in combined_df.columns
        # All humidity values should be NaN since second DataFrame was empty
        assert combined_df["humidity"].isna().all()

    def test_metadata_timezone_mismatch_combination(self):
        """Test combining metadata DataFrames with different timezones."""

        # First metadata object - UTC timezone
        metadata_obj1 = MagicMock()
        metadata_obj1.df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-01-01", periods=3, freq="D", tz="UTC"
                ),
                "value": [25.0, 26.0, 24.0],
            }
        )
        metadata_obj1.description = "UTC temperature"
        metadata_obj1.timestamp_key = "timestamp"
        metadata_obj1.metadata_json = {"columns": [{"id": "temperature"}]}

        # Second metadata object - Eastern timezone
        metadata_obj2 = MagicMock()
        metadata_obj2.df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-01-01", periods=3, freq="D", tz="US/Eastern"
                ),
                "value": [60, 65, 55],
            }
        )
        metadata_obj2.description = "Eastern humidity"
        metadata_obj2.timestamp_key = "timestamp"
        metadata_obj2.metadata_json = {"columns": [{"id": "humidity"}]}

        combined_df, columns, description = _combine_metadata_dataframes(
            [metadata_obj1, metadata_obj2]
        )

        assert combined_df is not None
        assert "temperature" in combined_df.columns
        assert "humidity" in combined_df.columns
        # Should handle timezone differences correctly
        assert not combined_df["temperature"].isna().all()
        assert not combined_df["humidity"].isna().all()


class TestGenerateResponseMessage:
    """Test cases for _generate_response_message function."""

    def test_user_data_with_matches(self):
        """Test message generation when user data exists with matches."""
        user_df = pd.DataFrame({"dummy": [1, 2, 3]})  # Mock user df
        original_row_count = 100
        final_row_count = 150
        common_timestamps_count = 80
        metadata_count = 50
        metadata_columns = ["temperature", "humidity"]

        message = _generate_response_message(
            user_df,
            original_row_count,
            final_row_count,
            common_timestamps_count,
            metadata_count,
            metadata_columns,
        )

        assert "100 rows" in message
        assert "150 rows" in message
        assert "80 exact matches" in message
        assert "50 timestamps" in message

    def test_user_data_no_matches(self):
        """Test message generation when user data exists but no matches."""
        user_df = pd.DataFrame({"dummy": [1, 2, 3]})
        original_row_count = 100
        final_row_count = 200
        common_timestamps_count = 0
        metadata_count = 100
        metadata_columns = ["temperature"]

        message = _generate_response_message(
            user_df,
            original_row_count,
            final_row_count,
            common_timestamps_count,
            metadata_count,
            metadata_columns,
        )

        assert "No exact timestamp matches" in message
        assert "NaN values" in message

    def test_metadata_only(self):
        """Test message generation for metadata-only requests."""

        message = _generate_response_message(
            None, None, 50, None, 50, ["temperature", "humidity"]
        )

        assert "Successfully fetched metadata with 2 columns" in message


class TestEndToEndIntegration:
    """Integration tests that test the complete metadata enrichment flow."""

    def test_weather_like_scenario(self):
        """Test a realistic weather data enrichment scenario."""
        # User data: daily sales data
        user_timestamps = pd.date_range("2023-01-01", periods=30, freq="D")
        user_df = pd.DataFrame(
            {
                "date": user_timestamps,
                "sales": np.random.randint(100, 1000, 30),
                "store_id": ["Store_A"] * 30,
            }
        )

        # Weather data: available every few days
        weather_timestamps = pd.date_range("2023-01-01", periods=10, freq="3D")
        weather_df = pd.DataFrame(
            {
                "timestamp": weather_timestamps,
                "temperature": np.random.uniform(20, 30, 10),
                "humidity": np.random.uniform(40, 80, 10),
                "precipitation": np.random.uniform(0, 5, 10),
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, weather_df, "date"
        )

        # Should have all user data rows
        assert len(result_df) == 30

        # Should have weather columns
        assert "temperature" in result_df.columns
        assert "humidity" in result_df.columns
        assert "precipitation" in result_df.columns

        # Should not have NaN in last few rows (weather data available)
        assert not pd.isna(result_df.loc[29, "temperature"])

        # Check that weather data is forward-filled appropriately
        # Days 1-3 should have same weather as day 1
        assert (
            result_df.loc[0, "temperature"]
            == result_df.loc[1, "temperature"]
            == result_df.loc[2, "temperature"]
        )

    def test_economic_indicators_scenario(self):
        """Test economic indicators enrichment scenario."""
        # User data: daily stock prices
        user_timestamps = pd.date_range("2023-01-01", periods=20, freq="D")
        user_df = pd.DataFrame(
            {
                "date": user_timestamps,
                "stock_price": np.random.uniform(100, 200, 20),
            }
        )

        # Economic data: monthly indicators
        econ_timestamps = pd.date_range(
            "2023-01-01", periods=1, freq="MS"
        )  # Start of month
        econ_df = pd.DataFrame(
            {
                "timestamp": econ_timestamps,
                "gdp_growth": [2.5],
                "unemployment_rate": [5.2],
                "inflation_rate": [3.1],
            }
        )

        result_df, common_count = _merge_user_and_metadata_dataframes(
            user_df, econ_df, "date"
        )

        # All days should get the same economic indicators (backward fill)
        assert len(result_df) == 20
        assert all(result_df["gdp_growth"] == 2.5)
        assert all(result_df["unemployment_rate"] == 5.2)
        assert all(result_df["inflation_rate"] == 3.1)


class TestListAvailableBacktests:
    """Test the list_available_backtests endpoint and related functions."""

    @pytest.fixture
    def mock_aioboto3_session(self):
        """Create a mock aioboto3 session."""
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )
        return mock_session

    @pytest.fixture
    def sample_s3_objects(self):
        """Create sample S3 objects for testing."""
        return [
            {
                "Key": "user123/foundation_models/dataset1/data.parquet_backtest_results_20241201_143022.zip",
                "Size": 1024000,
                "LastModified": datetime(2024, 12, 1, 14, 30, 22),
            },
            {
                "Key": "user123/foundation_models/dataset2/filter_path/data.parquet_backtest_results_20241202_093045.zip",
                "Size": 2048000,
                "LastModified": datetime(2024, 12, 2, 9, 30, 45),
            },
            {
                "Key": "user123/other_file.csv",
                "Size": 512000,
                "LastModified": datetime(2024, 12, 1, 10, 0, 0),
            },
            {
                "Key": "user123/foundation_models/dataset3/data.parquet_backtest_results_20241203_160000.zip",
                "Size": 1536000,
                "LastModified": datetime(2024, 12, 3, 16, 0, 0),
            },
        ]

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.routers.foundation_models._setup_forecast_environment"
    )
    @patch("synthefy_pkg.app.routers.foundation_models.acreate_presigned_url")
    async def test_list_available_backtests_success(
        self,
        mock_create_presigned_url,
        mock_setup_environment,
        mock_aioboto3_session,
        sample_s3_objects,
    ):
        """Test successful listing of backtests."""

        # Arrange
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_setup_environment.return_value = (mock_settings, "test-bucket")

        mock_create_presigned_url.return_value = (
            "https://test-presigned-url.com"
        )

        # Mock S3 paginator with async iterator
        mock_paginator = MagicMock()
        mock_page = {"Contents": sample_s3_objects}

        # Create an async iterator for the paginator
        async def async_iterator():
            yield mock_page

        mock_paginator.paginate.return_value = async_iterator()
        mock_aioboto3_session.client.return_value.__aenter__.return_value.get_paginator.return_value = mock_paginator

        request = ListBacktestsRequest(user_id="user123")

        # Act
        result = await list_available_backtests(request, mock_aioboto3_session)

        # Assert
        assert result.status == StatusCode.ok
        assert len(result.backtests) == 3  # Should find 3 backtest files
        assert (
            "Successfully retrieved page 1 of 1 (3 of 3 total backtest results)"
            in result.message
        )

        # Check that backtests are sorted by execution datetime (newest first)
        execution_datetimes = [bt.execution_datetime for bt in result.backtests]
        assert execution_datetimes == sorted(execution_datetimes, reverse=True)

        # Check that all backtests have required fields
        for backtest in result.backtests:
            assert backtest.s3_key.startswith("user123/")
            assert backtest.download_url == "https://test-presigned-url.com"
            assert backtest.execution_datetime is not None
            assert backtest.dataset_name is not None
            assert backtest.file_size_bytes is not None

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.routers.foundation_models._setup_forecast_environment"
    )
    @patch("synthefy_pkg.app.routers.foundation_models.acreate_presigned_url")
    async def test_list_available_backtests_with_dataset_filter(
        self,
        mock_create_presigned_url,
        mock_setup_environment,
        mock_aioboto3_session,
        sample_s3_objects,
    ):
        """Test listing backtests filtered by dataset name."""

        # Arrange
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_setup_environment.return_value = (mock_settings, "test-bucket")

        mock_create_presigned_url.return_value = (
            "https://test-presigned-url.com"
        )

        # Mock S3 paginator with async iterator
        mock_paginator = MagicMock()
        mock_page = {"Contents": sample_s3_objects}

        # Create an async iterator for the paginator
        async def async_iterator():
            yield mock_page

        mock_paginator.paginate.return_value = async_iterator()
        mock_aioboto3_session.client.return_value.__aenter__.return_value.get_paginator.return_value = mock_paginator

        request = ListBacktestsRequest(
            user_id="user123", dataset_name="dataset1"
        )

        # Act
        result = await list_available_backtests(request, mock_aioboto3_session)

        # Assert
        assert result.status == StatusCode.ok
        assert len(result.backtests) == 1  # Should find only dataset1
        assert result.backtests[0].dataset_name == "dataset1"

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.routers.foundation_models._setup_forecast_environment"
    )
    async def test_list_available_backtests_no_backtests_found(
        self,
        mock_setup_environment,
        mock_aioboto3_session,
    ):
        """Test listing backtests when no backtest files are found."""

        # Arrange
        mock_settings = MagicMock()
        mock_settings.bucket_name = "test-bucket"
        mock_setup_environment.return_value = (mock_settings, "test-bucket")

        # Mock S3 paginator with no backtest files
        mock_paginator = MagicMock()
        mock_page = {"Contents": []}

        # Create an async iterator for the paginator
        async def async_iterator():
            yield mock_page

        mock_paginator.paginate.return_value = async_iterator()
        mock_aioboto3_session.client.return_value.__aenter__.return_value.get_paginator.return_value = mock_paginator

        request = ListBacktestsRequest(user_id="user123")

        # Act
        result = await list_available_backtests(request, mock_aioboto3_session)

        # Assert
        assert result.status == StatusCode.ok
        assert len(result.backtests) == 0
        assert (
            "Successfully retrieved page 1 of 1 (0 of 0 total backtest results)"
            in result.message
        )

    @pytest.mark.asyncio
    @patch(
        "synthefy_pkg.app.routers.foundation_models._setup_forecast_environment"
    )
    async def test_list_available_backtests_exception_handling(
        self,
        mock_setup_environment,
        mock_aioboto3_session,
    ):
        """Test exception handling in list_available_backtests."""

        # Arrange
        mock_setup_environment.side_effect = Exception("Test exception")

        request = ListBacktestsRequest(user_id="user123")

        # Act
        result = await list_available_backtests(request, mock_aioboto3_session)

        # Assert
        assert result.status == StatusCode.internal_server_error
        assert "Failed to list backtests" in result.message
        assert len(result.backtests) == 0

    def test_extract_execution_datetime_from_key(self):
        """Test the _extract_execution_datetime_from_key function."""
        from synthefy_pkg.app.routers.foundation_models import (
            _extract_execution_datetime_from_key,
        )

        # Test valid timestamp
        s3_key = "user123/dataset1.parquet_backtest_results_20241201_143022.zip"
        result = _extract_execution_datetime_from_key(s3_key)
        assert result == "2024-12-01T14:30:22"

        # Test invalid timestamp format
        s3_key = (
            "user123/dataset1.parquet_backtest_results_invalid_timestamp.zip"
        )
        result = _extract_execution_datetime_from_key(s3_key)
        assert result is None

        # Test non-backtest file
        s3_key = "user123/regular_file.csv"
        result = _extract_execution_datetime_from_key(s3_key)
        assert result is None

        # Test edge case with different timestamp
        s3_key = "user123/dataset1.parquet_backtest_results_20240101_000000.zip"
        result = _extract_execution_datetime_from_key(s3_key)
        assert result == "2024-01-01T00:00:00"
