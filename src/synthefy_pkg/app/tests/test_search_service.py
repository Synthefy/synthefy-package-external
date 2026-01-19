import json
import os
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from synthefy_pkg.app.config import SearchSettings
from synthefy_pkg.app.data_models import SynthefyRequest, SynthefyResponse
from synthefy_pkg.app.main import create_app
from synthefy_pkg.app.routers.search import get_search_service

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

skip_in_ci = pytest.mark.skipif(
    os.environ.get("SKIP_CERTAIN_TESTS") == "true",
    reason="Test skipped based on environment variable",
)


@pytest.fixture(scope="function")
def app():
    # Set required environment variable
    os.environ["SYNTHEFY_CONFIG_PATH"] = os.path.join(
        os.environ.get("SYNTHEFY_PACKAGE_BASE"),
        "examples/configs/api_configs/api_config_twamp_one_month.yaml",
    )
    return create_app(os.environ["SYNTHEFY_CONFIG_PATH"])


@pytest.fixture(scope="function")
def client(app):
    return TestClient(app)


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


@pytest.fixture(scope="function")
def search_service_twamp_one_month():
    # Set required environment variable
    os.environ["SYNTHEFY_CONFIG_PATH"] = os.path.join(
        os.environ.get("SYNTHEFY_PACKAGE_BASE"),
        "examples/configs/api_configs/api_config_twamp_one_month.yaml",
    )
    return get_search_service(dataset_name = "twamp_one_month")


@skip_in_ci
@pytest.mark.usefixtures("search_service_twamp_one_month")
class TestSearchServiceTwampOneMonth:

    # TODO @Bekzat - redo this without default request. Also change the class name. There should be
    # separate classes and test for the preprocessing and end to end search.
    def test_twamp_one_month_request_preprocess(self, twamp_json):
        pass
        # request = search_service_twamp_one_month.get_default_search_request()
        # x_axis_name, df, n_closest, search_set = (
        #     search_service_twamp_one_month._preprocess_request(request)
        # )
        # assert request.timestamps_range.name == x_axis_name
        # assert len(request.search_query) == df.shape[1]
        # assert len(request.search_query[0].values) == df.shape[0]

    @pytest.mark.parametrize("selected_action, top_n_search_windows", [("SEARCH", 2)])
    @skip_in_ci
    def test_service_search(
        self, selected_action, top_n_search_windows, twamp_json, client
    ):
        # Prepare request
        twamp_json["selected_action"] = selected_action
        twamp_json["top_n_search_windows"] = top_n_search_windows
        twamp_json["selected_windows"] = {
            "window_type": "CURRENT_VIEW_WINDOWS",
            "window_indices": [0],
        }

        # Test API endpoint
        response = client.post(
            "/api/search/twamp_one_month",
            json=twamp_json,
            headers={"X-API-Key": "test_api_key"}
        )
        assert response.status_code == 200
        synthefy_response = SynthefyResponse(**response.json())

        # Verify response structure
        assert len(synthefy_response.windows) == top_n_search_windows + 1  # +1 for query window

        # Check window contents
        for window in synthefy_response.windows:
            assert window.timeseries_data is not None
            assert len(window.timeseries_data) > 0
            assert all(len(ts.values) > 0 for ts in window.timeseries_data)
