import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from synthefy_pkg.app.main import create_app


@pytest.fixture
def config_path():
    # Use a mock config path for testing
    return "mock_config_path"


@pytest.fixture
def client(config_path):
    # Create a test client with the mock config
    with patch.dict(os.environ, {"SYNTHEFY_CONFIG_PATH": config_path}):
        app = create_app(config_path)
        return TestClient(app)


class TestRouterLoading:
    def test_root_endpoint(self, client):
        """Test that the root endpoint works correctly."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Synthefy Platform API"}

    @patch("importlib.import_module")
    @patch("fastapi.FastAPI.include_router")
    def test_single_router_loading(
        self, mock_include_router, mock_import, config_path
    ):
        """Test loading a single router specified by SYNTHEFY_ROUTER."""
        # Mock the router module
        mock_router = MagicMock()
        mock_import.return_value = mock_router

        # Set env var to load only one router
        with patch.dict(
            os.environ,
            {
                "SYNTHEFY_CONFIG_PATH": config_path,
                "SYNTHEFY_ROUTER": "test_router",
            },
        ):
            create_app(config_path)

            # Verify router was imported correctly
            mock_import.assert_called_once_with(
                "synthefy_pkg.app.routers.test_router"
            )
            # Verify router was included in the app
            mock_include_router.assert_called_once_with(mock_router.router)

    @patch("importlib.import_module")
    def test_invalid_router_handling(self, mock_import, config_path):
        """Test handling of invalid router names."""
        # Mock import to raise ImportError
        mock_import.side_effect = ImportError("Router not found")

        # Set env var to load a non-existent router
        with patch.dict(
            os.environ,
            {
                "SYNTHEFY_CONFIG_PATH": config_path,
                "SYNTHEFY_ROUTER": "nonexistent_router",
            },
        ):
            # This should not raise an exception
            app = create_app(config_path)
            client = TestClient(app)

            # Root endpoint should still work
            response = client.get("/")
            assert response.status_code == 200

    @patch("importlib.import_module")
    @patch("fastapi.FastAPI.include_router")
    def test_all_routers_loading(
        self, mock_include_router, mock_import, config_path
    ):
        """Test that all routers are loaded when SYNTHEFY_ROUTER is not set."""
        # Mock the router module
        mock_router = MagicMock()
        mock_import.return_value = mock_router

        # Ensure SYNTHEFY_ROUTER is not set
        with patch.dict(
            os.environ, {"SYNTHEFY_CONFIG_PATH": config_path}, clear=True
        ):
            if "SYNTHEFY_ROUTER" in os.environ:
                del os.environ["SYNTHEFY_ROUTER"]

            create_app(config_path)

            # Verify multiple routers were imported
            # The number of calls should be equal to the number of routers in all_routers
            assert mock_import.call_count > 1

            # Verify all routers were included in the app
            assert mock_include_router.call_count == mock_import.call_count
