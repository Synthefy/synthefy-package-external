import json
import os
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from synthefy_pkg.app.config import ForecastSettings
from synthefy_pkg.app.data_models import (
    MetaData,
    OneContinuousMetaData,
    OneDiscreteMetaData,
    OneTimeSeries,
    SelectedAction,
    SelectedWindows,
    SynthefyRequest,
    SynthefyTimeSeriesWindow,
    TimeStamps,
    WindowsSelectionOptions,
)
from synthefy_pkg.app.main import create_app
from synthefy_pkg.app.routers.forecast import get_forecast_service
from synthefy_pkg.app.services.forecast_service import (
    shift_to_zero_out_forecast_length,
)

skip_in_ci_or_twamp = pytest.mark.skipif(
    os.environ.get("SKIP_CERTAIN_TESTS") == "true"
    or os.environ.get("SKIP_TWAMP_TESTS") == "true",
    reason="Test skipped based on environment variable or Twamp test",
)


@pytest.fixture(scope="function")
def app():
    config_path = os.path.join(
        os.environ["SYNTHEFY_PACKAGE_BASE"],
        "examples/configs/api_configs/api_config_twamp_one_month.yaml",
    )  # type: ignore
    return create_app(config_path)


@pytest.fixture(scope="function")
def client(app):
    return TestClient(app)


@pytest.fixture(scope="function")
def twamp_stream_json():
    with open(
        os.path.join(
            os.environ["SYNTHEFY_PACKAGE_BASE"],
            "src/synthefy_pkg/app/tests/test_jsons/twamp_stream.json",
        ),  # type: ignore
        "r",
    ) as f:  # type: ignore
        return json.load(f)


mocked_settings_twamp_one_month = ForecastSettings(
    dataset_path=os.environ["SYNTHEFY_DATASETS_BASE"],
    preprocess_config_path=os.path.join(
        os.environ["SYNTHEFY_PACKAGE_BASE"],
        "examples/configs/preprocessing_configs/config_twamp_one_month_preprocessing.json",
    ),
    forecast_config_path=os.path.join(
        os.environ["SYNTHEFY_PACKAGE_BASE"],
        "examples/configs/forecast_configs/config_twamp_one_month_forecasting.yaml",
    ),
    forecast_model_path=os.path.join(
        os.environ["SYNTHEFY_DATASETS_BASE"],
        "training_logs/twamp_one_month/Time_Series_Diffusion_Training/synthefy_forecasting_model_v1_twamp_one_month/checkpoints/best_model.ckpt",
    ),
    show_gt_forecast_timeseries=True,
    only_include_forecast_in_streaming_response=True,
    return_only_synthetic_in_streaming_response=True,
    json_save_path="/tmp",
)


@pytest.fixture(scope="function")
def twamp_json():
    with open(
        os.path.join(
            os.environ["SYNTHEFY_PACKAGE_BASE"],
            "src/synthefy_pkg/app/tests/test_jsons/dispatch_twamp_one_month_request.json",
        ),  # type: ignore
        "r",
    ) as f:  # type: ignore
        return json.load(f)


@pytest.fixture(scope="function")
def forecast_service_twamp_one_month():
    # Set required environment variable
    os.environ["SYNTHEFY_CONFIG_PATH"] = os.path.join(
        str(os.getenv("SYNTHEFY_PACKAGE_BASE")),
        "examples/configs/api_configs/api_config_twamp_one_month.yaml",
    )
    service = get_forecast_service(dataset_name="twamp_one_month")
    return service


class TestShiftToZeroOutForecastLength:
    forecast_length = 32

    @pytest.fixture
    def basic_request(self):
        """Creates a basic request with all fields populated"""
        window_size = 96
        return SynthefyRequest(
            windows=[
                SynthefyTimeSeriesWindow(
                    timestamps=TimeStamps(
                        name="@timestamp",
                        values=[
                            (
                                datetime(2024, 3, 24) + timedelta(hours=i)
                            ).strftime("%Y-%m-%dT%H:00:00+00:00")
                            for i in range(96)
                        ],
                    ),
                    timeseries_data=[
                        OneTimeSeries(
                            name="test_series",
                            values=[float(i) for i in range(window_size)],
                        )
                    ],
                    metadata=MetaData(
                        discrete_conditions=[
                            OneDiscreteMetaData(
                                name="test_discrete",
                                values=[f"val_{i}" for i in range(window_size)],
                            )
                        ],
                        continuous_conditions=[
                            OneContinuousMetaData(
                                name="test_continuous",
                                values=[float(i) for i in range(window_size)],
                            )
                        ],
                    ),
                )
            ],
            selected_windows=SelectedWindows(
                window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
                window_indices=[0],
            ),
            selected_action=SelectedAction.FORECAST,
        )

    def test_basic_shift(self, basic_request):
        """Test basic shifting functionality with all fields populated"""
        window_size = 96

        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )
        window = shifted_request.windows[0]

        # Check timeseries values
        expected_ts_values = [
            float(i) for i in range(self.forecast_length, window_size)
        ] + [None] * self.forecast_length
        assert window.timeseries_data[0].values == expected_ts_values
        # Check timestamps
        expected_timestamps = [
            (datetime(2024, 3, 24) + timedelta(hours=i)).strftime(
                "%Y-%m-%dT%H:00:00+00:00"
            )
            for i in range(
                self.forecast_length, self.forecast_length + window_size
            )
        ]
        assert window.timestamps is not None
        assert len(window.timestamps.values) == window_size
        assert window.timestamps.values == expected_timestamps

        # Check discrete metadata
        expected_discrete = [
            f"val_{i}" for i in range(self.forecast_length, window_size)
        ] + [None for i in range(self.forecast_length)]
        assert (
            window.metadata.discrete_conditions[0].values == expected_discrete
        )

        # Check continuous metadata
        expected_continuous = [
            float(i) for i in range(self.forecast_length, window_size)
        ] + [None] * self.forecast_length
        assert (
            window.metadata.continuous_conditions[0].values
            == expected_continuous
        )

    def test_no_timestamps(self, basic_request):
        """Test shifting with no timestamps"""
        basic_request.windows[0].timestamps = None
        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )

        # Verify the shift still worked for other fields
        window = shifted_request.windows[0]
        assert window.timestamps is None
        assert (
            len(window.timeseries_data[0].values) == 96
        )  # Original length maintained

    def test_no_metadata(self, basic_request):
        """Test shifting with no metadata"""
        basic_request.windows[0].metadata = MetaData()
        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )

        window = shifted_request.windows[0]
        assert len(window.metadata.discrete_conditions) == 0
        assert len(window.metadata.continuous_conditions) == 0
        assert (
            len(window.timeseries_data[0].values) == 96
        )  # Original length maintained

    def test_empty_metadata_conditions(self, basic_request):
        """Test shifting with empty metadata conditions"""
        basic_request.windows[0].metadata.discrete_conditions = []
        basic_request.windows[0].metadata.continuous_conditions = []
        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )

        window = shifted_request.windows[0]
        assert len(window.metadata.discrete_conditions) == 0
        assert len(window.metadata.continuous_conditions) == 0
        assert (
            len(window.timeseries_data[0].values) == 96
        )  # Original length maintained

    def test_multiple_windows(self, basic_request):
        """Test shifting with multiple windows"""
        # Add a second window
        basic_request.windows.append(
            basic_request.windows[0].model_copy(deep=True)
        )
        basic_request.selected_windows.window_indices = [0, 1]

        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )

        assert len(shifted_request.windows) == 2
        # Verify both windows were shifted correctly
        for window in shifted_request.windows:
            assert len(window.timeseries_data[0].values) == 96
            assert (
                window.timeseries_data[0].values[-self.forecast_length :]
                == [None] * self.forecast_length
            )

    def test_multiple_timeseries(self, basic_request):
        """Test shifting with multiple timeseries in a window"""
        window_size = 96
        # Add a second timeseries
        basic_request.windows[0].timeseries_data.append(
            OneTimeSeries(
                name="test_series_2",
                values=[float(i) for i in range(window_size, window_size * 2)],
            )
        )

        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )

        window = shifted_request.windows[0]
        assert len(window.timeseries_data) == 2
        for ts in window.timeseries_data:
            assert len(ts.values) == window_size
            assert (
                ts.values[-self.forecast_length :]
                == [None] * self.forecast_length
            )

    def test_data_integrity(self, basic_request):
        """Test that the original request is not modified"""
        original_values = (
            basic_request.windows[0].timeseries_data[0].values.copy()
        )
        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )

        assert (
            basic_request.windows[0].timeseries_data[0].values
            == original_values
        )
        assert (
            shifted_request.windows[0].timeseries_data[0].values
            != original_values
        )

    def test_length_validation(self, basic_request):
        """Test that all arrays maintain the correct length after shifting"""
        window_size = 96
        shifted_request = shift_to_zero_out_forecast_length(
            basic_request, self.forecast_length
        )
        window = shifted_request.windows[0]

        assert window.timestamps is not None
        assert len(window.timestamps.values) == window_size
        assert len(window.timeseries_data[0].values) == window_size
        assert len(window.metadata.discrete_conditions[0].values) == window_size
        assert (
            len(window.metadata.continuous_conditions[0].values) == window_size
        )

    def test_timestamp_conversion(self):
        """Test shifting with different timestamp formats"""
        window_size = 96

        # Create timestamps in different formats
        base_dt = datetime(2024, 3, 24)
        unix_seconds = int(base_dt.timestamp())  # Unix timestamp in seconds
        unix_millis = unix_seconds * 1000  # Unix timestamp in milliseconds

        # Create request with different timestamp formats
        request = SynthefyRequest(
            windows=[
                SynthefyTimeSeriesWindow(
                    timestamps=TimeStamps(
                        name="@timestamp",
                        values=[
                            # Alternating between seconds and milliseconds format
                            (
                                unix_seconds + (i * 3600)
                                if i % 2 == 0
                                else unix_millis + (i * 3600 * 1000)
                            )
                            for i in range(window_size)
                        ],
                    ),
                    metadata=MetaData(),
                    timeseries_data=[
                        OneTimeSeries(
                            name="test_series",
                            values=[float(i) for i in range(window_size)],
                        )
                    ],
                ),
            ],
            selected_windows=SelectedWindows(
                window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
                window_indices=[0],
            ),
            selected_action=SelectedAction.FORECAST,
        )

        shifted_request = shift_to_zero_out_forecast_length(
            request, self.forecast_length
        )
        window = shifted_request.windows[0]
        # Verify all timestamps are properly converted to ISO format strings
        assert window.timestamps is not None
        for ts in window.timestamps.values:
            # Check if timestamp is in correct ISO format
            try:
                parsed_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                assert isinstance(parsed_ts, datetime)
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {ts}")

        # Verify timestamps are sequential and properly spaced
        for i in range(1, len(window.timestamps.values)):
            prev_ts = datetime.fromisoformat(
                window.timestamps.values[i - 1].replace("Z", "+00:00")
            )
            curr_ts = datetime.fromisoformat(
                window.timestamps.values[i].replace("Z", "+00:00")
            )
            # Check that timestamps are 1 hour apart
            assert (curr_ts - prev_ts) == timedelta(hours=1)

        # Verify first and last timestamps
        first_ts = datetime.fromisoformat(
            window.timestamps.values[0].replace("Z", "+00:00")
        )
        last_ts = datetime.fromisoformat(
            window.timestamps.values[-1].replace("Z", "+00:00")
        )
        assert first_ts == base_dt + timedelta(hours=self.forecast_length), (
            f"First timestamp should be {self.forecast_length} hours after base, got {first_ts - base_dt}"
        )
        assert last_ts == base_dt + timedelta(
            hours=self.forecast_length + window_size - 1
        ), (
            f"Last timestamp should be {self.forecast_length + window_size - 1} hours after base, got {last_ts - base_dt}"
        )
