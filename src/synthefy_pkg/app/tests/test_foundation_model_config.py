import os
import tempfile

import pytest
from pydantic import ValidationError

from synthefy_pkg.app.config import SynthefyFoundationModelSettings
from synthefy_pkg.app.data_models import FoundationModelConfig


def test_valid_timestamps():
    """Test that valid ISO format timestamps are accepted."""
    config = FoundationModelConfig(
        file_path_key="some/path",
        forecast_length=10,
        timeseries_columns=["col1", "col2"],
        min_timestamp="2023-01-01T00:00:00",
        forecasting_timestamp="2023-01-02T00:00:00",
        timestamp_column="timestamp",
    )
    assert config.min_timestamp == "2023-01-01T00:00:00"
    assert config.forecasting_timestamp == "2023-01-02T00:00:00"


def test_invalid_min_timestamp():
    """Test that an invalid min_timestamp raises a ValidationError."""
    with pytest.raises(
        ValidationError,
        match="Timestamp 'invalid-timestamp' is not in valid ISO format\\. Supported formats: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS\\.ffffff \\(with optional microseconds and timezone offset\\)",
    ):
        FoundationModelConfig(
            file_path_key="some/path",
            forecast_length=10,
            timeseries_columns=["col1", "col2"],
            min_timestamp="invalid-timestamp",
            forecasting_timestamp="2023-01-02T00:00:00",
            timestamp_column="timestamp",
        )


def test_invalid_forecasting_timestamp():
    """Test that an invalid forecasting_timestamp raises a ValidationError."""
    with pytest.raises(
        ValidationError,
        match="Timestamp 'invalid-forecasting-timestamp' is not in valid ISO format\\. Supported formats: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS\\.ffffff \\(with optional microseconds and timezone offset\\)",
    ):
        FoundationModelConfig(
            file_path_key="some/path",
            forecast_length=10,
            timeseries_columns=["col1", "col2"],
            min_timestamp="2023-01-01T00:00:00",
            forecasting_timestamp="invalid-forecasting-timestamp",
            timestamp_column="timestamp",
        )


def test_timestamp_without_date():
    """Test that a timestamp without a date raises a ValidationError."""
    with pytest.raises(
        ValidationError,
        match="Timestamp '12:00:00' is not in valid ISO format\\. Supported formats: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS\\.ffffff \\(with optional microseconds and timezone offset\\)",
    ):
        FoundationModelConfig(
            file_path_key="some/path",
            forecast_length=10,
            timeseries_columns=["col1", "col2"],
            min_timestamp="12:00:00",
            forecasting_timestamp="2023-01-02T00:00:00",
            timestamp_column="timestamp",
        )


def test_synthefy_foundation_model_settings_creates_directory():
    """Test that SynthefyFoundationModelSettings creates the local model directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create settings with a temporary path
        settings = SynthefyFoundationModelSettings(
            local_model_path=os.path.join(temp_dir, "test_model")
        )

        # Check that the directory was created
        assert os.path.exists(settings.local_model_path)
        assert os.path.isdir(settings.local_model_path)


def test_nonsensical_month():
    """Test that a timestamp with a nonsensical month raises a ValidationError."""
    with pytest.raises(
        ValidationError,
        match="Timestamp '2023-13-01T00:00:00' is not in valid ISO format\\. Supported formats: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS\\.ffffff \\(with optional microseconds and timezone offset\\)",
    ):
        FoundationModelConfig(
            file_path_key="some/path",
            forecast_length=10,
            timeseries_columns=["col1", "col2"],
            min_timestamp="2023-13-01T00:00:00",
            forecasting_timestamp="2023-01-02T00:00:00",
            timestamp_column="timestamp",
        )
