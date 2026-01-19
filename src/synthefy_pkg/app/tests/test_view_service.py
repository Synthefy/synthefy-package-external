import json
import os

import pytest
from fastapi.testclient import TestClient

from synthefy_pkg.app.data_models import SynthefyRequest, SynthefyResponse
from synthefy_pkg.app.main import create_app


@pytest.fixture(scope="function")
def app():
    config_path = os.path.join(
        os.environ.get("SYNTHEFY_PACKAGE_BASE"),
        "examples/configs/api_configs/api_config_twamp_one_month.yaml",
    )
    return create_app(config_path)


@pytest.fixture(scope="function")
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


skip_in_ci = pytest.mark.skipif(
    os.environ.get("SKIP_CERTAIN_TESTS") == "true",
    reason="Test skipped based on environment variable",
)


@pytest.fixture(scope="function")
def twamp_json():
    with open(
        os.path.join(
            os.environ.get("SYNTHEFY_PACKAGE_BASE"),
            "src/synthefy_pkg/app/tests/test_jsons/dispatch_twamp_one_month_request.json",
        ),
        "r",
    ) as f:
        return json.load(f)


@skip_in_ci
def test_get_time_series_view(twamp_json, client):

    twamp_json["text"] = (
        "Show me where the mean counter.rtt_mean < .697"  # TODO timestamps not added yet.
    )
    request = SynthefyRequest(**twamp_json)
    response = client.post("/api/view/twamp_one_month", json=twamp_json)
    response = SynthefyResponse(**response.json())

    assert isinstance(response, SynthefyResponse)
    assert len(response.windows) <= twamp_json["n_view_windows"]
    assert len(response.windows) == 4

    # TODO add checks on the conditions.
    # for window in response.windows:
    #     assert window.timestamps.values[0] >= "2023-01-01T00:00:00Z"
    #     assert window.timestamps.values[-1] <= "2023-01-02T23:59:59Z"

    assert type(response.combined_text) == str
    assert response.forecast_timestamps == []
    assert response.anomaly_timestamps == []
