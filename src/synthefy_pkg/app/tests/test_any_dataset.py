import os

import pytest
from fastapi.testclient import TestClient

from synthefy_pkg.app.data_models import SynthefyResponse
from synthefy_pkg.app.main import create_app


@pytest.fixture(scope="function")
def app():
    config_path = os.path.join(
        os.environ.get("SYNTHEFY_PACKAGE_BASE"),
        "src/synthefy_pkg/app/services/configs/api_config_twamp_one_month.yaml",
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


@skip_in_ci
class TestAnyDataset:
    @skip_in_ci
    def test_setup_ui(self):
        pass
        # hit the endpoint
        # response = client.get("/api/default")
        # assert response.status_code == 200
        # assert response.json() == {"message": "Setup UI"}
