import json
import os
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from synthefy_pkg.app.config import PreTrainedAnomalySettings
from synthefy_pkg.app.data_models import (
    DynamicTimeSeriesData,
    SelectedAction,
    SynthefyResponse,
    SynthefyTimeSeriesWindow,
)
from synthefy_pkg.app.main import create_app
from synthefy_pkg.app.routers.pretrained_anomaly import get_pretrained_anomaly_service
from synthefy_pkg.app.utils.api_utils import (
    convert_dynamic_time_series_data_to_synthefy_request,
    convert_label_tuple_to_discrete_metadata,
    convert_synthefy_response_to_dynamic_time_series_data,
    delete_gt_real_timeseries_windows,
)

skip_in_ci_or_twamp = pytest.mark.skipif(
    os.environ.get("SKIP_CERTAIN_TESTS") == "true"
    or os.environ.get("SKIP_TWAMP_TESTS") == "true",
    reason="Test skipped based on environment variable or Twamp test",
)


@pytest.fixture(scope="function")
def twamp_app():
    config_path = os.path.join(
        os.environ.get("SYNTHEFY_PACKAGE_BASE"),
        "examples/configs/api_configs/api_config_twamp_one_month.yaml",
    )
    return create_app(config_path)


@pytest.fixture(scope="function")
def twamp_client(twamp_app):
    from fastapi.testclient import TestClient

    return TestClient(twamp_app)


@pytest.fixture(scope="function")
def ativa_app():
    config_path = os.path.join(
        os.environ.get("SYNTHEFY_PACKAGE_BASE"),
        "examples/configs/api_configs/api_config_infovista_ativa.yaml",
    )
    return create_app(config_path)


@pytest.fixture(scope="function")
def ativa_client(ativa_app):
    from fastapi.testclient import TestClient

    return TestClient(ativa_app)


@pytest.fixture
def synthefy_request_json():
    """Loads a whole synthefy request from the dispatch json."""
    return json.load(
        open(
            os.path.join(
                os.environ.get("SYNTHEFY_PACKAGE_BASE"),
                "src/synthefy_pkg/app/tests/test_jsons/dispatch_twamp_one_month_request.json",
            ),
            "r",
        )
    )


mocked_settings_ativa = PreTrainedAnomalySettings(
    preprocess_config_path=os.path.join(
        os.environ["SYNTHEFY_PACKAGE_BASE"],
        "examples/configs/preprocessing_configs/config_ativa_preprocessing.json",
    ),
    json_save_path="/tmp",
    anomaly_threshold=3.0,
)


@pytest.fixture
def anomaly_service_ativa():
    service = get_pretrained_anomaly_service("ativa")
    return service


@pytest.fixture(scope="function")
def anomaly_stream_json():
    with open(
        os.path.join(
            os.environ.get("SYNTHEFY_PACKAGE_BASE"),
            "src/synthefy_pkg/app/tests/test_jsons/anomaly_stream.json",
        ),
        "r",
    ) as f:
        return json.load(f)


@pytest.mark.usefixtures("synthefy_request_json")
@skip_in_ci_or_twamp
class TestPretrainedAnomalyService:
    def test_end_to_end(self, synthefy_request_json, twamp_client):
        synthefy_request_json["selected_action"] = "ANOMALY_DETECTION"
        response = twamp_client.post(
            "/api/pretrained_anomaly/ativa", json=synthefy_request_json
        )

        assert response.status_code == 200
        response = SynthefyResponse(**response.json())

        # Make sure we receive the right types
        assert isinstance(response, SynthefyResponse)
        assert all(
            [
                isinstance(window, SynthefyTimeSeriesWindow)
                for window in response.windows
            ]
        )

        # Make sure we receive the right number of windows
        assert len(response.windows) == len(response.anomaly_timestamps) == 1


@skip_in_ci_or_twamp
class TestAnomalyServiceStream:
    def test_service_synthesis_stream_api(self, anomaly_stream_json, ativa_client):
        response = ativa_client.post(
            "/api/pretrained_anomaly/ativa/stream", json=anomaly_stream_json
        )
        assert response.status_code == 200
        dynamic_time_series_data = DynamicTimeSeriesData(**response.json())

        x = dynamic_time_series_data.model_dump_json()
        with open(
            f"/tmp/test_anomaly_stream.json",
            "w",
        ) as f:
            f.write(x)
        df = pd.read_json(f"/tmp/test_anomaly_stream.json")
        # Check df has column is_anomaly
        assert "is_anomaly" in df.columns

        # Check df has column is_anomaly with only 0 and 1
        assert df["is_anomaly"].isin([0, 1]).all()

    def test_service_synthesis_stream_individual_funcs(
        self, anomaly_stream_json, ativa_client, anomaly_service_ativa
    ):
        #

        dynamic_time_series_data = DynamicTimeSeriesData(root=anomaly_stream_json)

        request = convert_dynamic_time_series_data_to_synthefy_request(
            dynamic_time_series_data,
            anomaly_service_ativa.preprocess_config.get("group_labels", {}).get(
                "cols", []
            ),
            anomaly_service_ativa.preprocess_config.get("timeseries", {}).get(
                "cols", []
            ),
            anomaly_service_ativa.preprocess_config.get("continuous", {}).get(
                "cols", []
            ),
            anomaly_service_ativa.preprocess_config.get("discrete", {}).get("cols", []),
            anomaly_service_ativa.preprocess_config.get("timestamps_col", []),
            anomaly_service_ativa.preprocess_config.get("window_size", None),
            selected_action=SelectedAction.ANOMALY_DETECTION,
        )
        request = convert_label_tuple_to_discrete_metadata(request)
        request = delete_gt_real_timeseries_windows(request)
        synthefy_response = anomaly_service_ativa.pretrained_anomaly_detection(
            request, streaming=True
        )

        dynamic_time_series_data = (
            convert_synthefy_response_to_dynamic_time_series_data(
                synthefy_response, return_only_synthetic=False
            )
        )

        x = dynamic_time_series_data.model_dump_json()
        with open(
            f"/tmp/test_anomaly_stream.json",
            "w",
        ) as f:
            f.write(x)
        df = pd.read_json(f"/tmp/test_anomaly_stream.json")
        # Check df has column is_anomaly
        assert "is_anomaly" in df.columns

        # Check df has column is_anomaly with only 0 and 1
        assert df["is_anomaly"].isin([0, 1]).all()

        # Make sure all the anomaly timestamps in synthefy_response are 1 in the is_anomaly column and timestamps are in the correct order
        anomaly_timestamps = synthefy_response.anomaly_timestamps[0].values

        # Create a boolean mask for all anomaly timestamps at once
        anomaly_mask = df["timestamp"].isin(anomaly_timestamps)

        # Check all anomaly timestamps are marked with 1
        assert (df[anomaly_mask]["is_anomaly"] == 1).all()

        # Check all non-anomaly timestamps are marked with 0
        assert (df[~anomaly_mask]["is_anomaly"] == 0).all()
