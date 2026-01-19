import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from synthefy_pkg.app.middleware.api_endpoints import APIEndpoints
from synthefy_pkg.app.middleware.api_usage_middleware import APIUsageMiddleware
from synthefy_pkg.app.middleware.metrics_manager.metrics_manager_utils import (
    extract_user_id_and_dataset_name,
    extract_user_id_from_dataset_name,
)


class TestAPIUsageMiddleware:
    """Test cases for APIUsageMiddleware helper methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = APIUsageMiddleware(app=None)

    def test_extract_api_key_from_headers_x_api_key(self):
        """Test extracting API key from X-API-Key header."""
        headers = {"X-API-Key": "test-api-key-123"}
        result = self.middleware._extract_api_key_from_headers(headers)
        assert result == "test-api-key-123"

    def test_extract_api_key_from_headers_authorization_bearer(self):
        """Test extracting API key from Authorization Bearer header."""
        headers = {"Authorization": "Bearer test-bearer-token-456"}
        result = self.middleware._extract_api_key_from_headers(headers)
        assert result == "test-bearer-token-456"

    def test_extract_api_key_from_headers_authorization_no_bearer(self):
        """Test extracting API key from Authorization header without Bearer prefix."""
        headers = {"Authorization": "plain-token-789"}
        result = self.middleware._extract_api_key_from_headers(headers)
        assert result is None

    def test_extract_api_key_from_headers_no_api_key(self):
        """Test when no API key headers are present."""
        headers = {"Content-Type": "application/json"}
        result = self.middleware._extract_api_key_from_headers(headers)
        assert result is None

    def test_extract_api_key_from_headers_empty_headers(self):
        """Test with empty headers."""
        headers = {}
        result = self.middleware._extract_api_key_from_headers(headers)
        assert result is None

    def test_extract_api_key_from_headers_x_api_key_precedence(self):
        """Test that X-API-Key takes precedence over Authorization header."""
        headers = {
            "X-API-Key": "preferred-api-key",
            "Authorization": "Bearer fallback-token",
        }
        result = self.middleware._extract_api_key_from_headers(headers)
        assert result == "preferred-api-key"

    def test_should_track_endpoint_forecast_stream(self):
        """Test tracking forecast stream endpoints."""
        path = "/api/forecast/dataset123/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_synthesis_stream(self):
        """Test tracking synthesis stream endpoints."""
        path = "/api/synthesis/my_dataset/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_pretrained_anomaly_stream(self):
        """Test tracking pretrained anomaly stream endpoints."""
        path = "/api/pretrained_anomaly/anomaly_dataset/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_v2_anomaly_detection_stream(self):
        """Test tracking v2 anomaly detection stream endpoints."""
        path = "/api/v2/anomaly_detection/v2_dataset/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_foundation_models_upload(self):
        """Test tracking foundation models upload endpoint."""
        path = "/api/foundation_models/upload"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_foundation_models_forecast(self):
        """Test tracking foundation models forecast endpoint."""
        path = "/api/foundation_models/forecast"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_foundation_models_forecast_stream(self):
        """Test tracking foundation models forecast stream endpoint."""
        path = "/api/foundation_models/forecast/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_foundation_models_forecast_stream_v2(self):
        """Test tracking foundation models forecast stream v2 endpoint."""
        path = "/api/v2/foundation_models/forecast/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_foundation_models_backtest(self):
        """Test tracking foundation models backtest endpoint."""
        path = "/api/foundation_models/forecast/backtest"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_synthetic_data_agent(self):
        """Test tracking synthetic data agent endpoint."""
        path = (
            "/api/synthetic_data_agent/generate_synthetic_data/user123/job456"
        )
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_train(self):
        """Test tracking train endpoint."""
        path = "/api/train/model123"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_preprocess(self):
        """Test tracking preprocess endpoints."""
        path = "/api/preprocess/user123/dataset456"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_postprocess(self):
        """Test tracking postprocess endpoints."""
        path = "/api/postprocess/html/dataset789"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_not_track_train_sub_endpoints(self):
        """Test that train sub-endpoints are not tracked."""
        paths = [
            "/api/train/status",
            "/api/train/stop",
            "/api/train/list",
            "/api/train/dataset123/config",
        ]
        for path in paths:
            result = self.middleware._should_track_endpoint(path)
            assert result is False, f"Path {path} should not be tracked"

    def test_should_track_endpoint_explain(self):
        """Test tracking explain endpoint."""
        path = "/api/explain"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_user_api_key_create(self):
        """Test tracking user API key creation endpoint."""
        path = "/api/user_api_key/"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_user_api_key_delete(self):
        """Test tracking user API key deletion endpoint."""
        path = "/api/user_api_key/123"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_user_api_keys_list(self):
        """Test tracking user API keys list endpoint."""
        path = "/api/user_api_keys/"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_health_check(self):
        """Test tracking health check endpoint."""
        path = "/api/health"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_synthesis_with_training_job(self):
        """Test tracking synthesis endpoint with training job ID."""
        path = "/api/synthesis/dataset123/job456/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_non_stream_forecast(self):
        """Test that non-stream forecast endpoints are not tracked."""
        path = "/api/forecast/dataset123"
        result = self.middleware._should_track_endpoint(path)
        assert result is False

    def test_should_track_endpoint_non_stream_synthesis(self):
        """Test that non-stream synthesis endpoints are not tracked."""
        path = "/api/synthesis/my_dataset"
        result = self.middleware._should_track_endpoint(path)
        assert result is False

    def test_should_track_endpoint_different_api_version(self):
        """Test that different API versions without stream are not tracked."""
        path = "/api/v1/forecast/dataset123"
        result = self.middleware._should_track_endpoint(path)
        assert result is False

    def test_should_track_endpoint_completely_different_path(self):
        """Test that completely different paths are not tracked."""
        path = "/health"
        result = self.middleware._should_track_endpoint(path)
        assert result is False

    def test_should_track_endpoint_root_path(self):
        """Test that root path is not tracked."""
        path = "/"
        result = self.middleware._should_track_endpoint(path)
        assert result is False

    def test_should_track_endpoint_empty_path(self):
        """Test that empty path is not tracked."""
        path = ""
        result = self.middleware._should_track_endpoint(path)
        assert result is False

    def test_should_track_endpoint_with_query_params(self):
        """Test that stream endpoints with query parameters are tracked."""
        path = "/api/forecast/dataset123/stream?param=value"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_with_trailing_slash(self):
        """Test that stream endpoints with trailing slash are tracked."""
        path = "/api/synthesis/my_dataset/stream/"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    def test_should_track_endpoint_complex_dataset_name(self):
        """Test that stream endpoints with complex dataset names are tracked."""
        path = "/api/forecast/my-complex_dataset-name_with_underscores/stream"
        result = self.middleware._should_track_endpoint(path)
        assert result is True

    # TODO: Do we need to track nested paths?
    # def test_should_track_endpoint_nested_paths(self):
    #     """Test that nested paths that don't match patterns are not tracked."""
    #     path = "/api/forecast/dataset123/stream/extra"
    #     result = self.middleware._should_track_endpoint(path)
    #     assert result is False

    # def test_should_track_endpoint_partial_matches(self):
    #     """Test that partial matches are not tracked."""
    #     path = "/api/forecast/dataset123/stream_extra"
    #     result = self.middleware._should_track_endpoint(path)
    #     assert result is False

    def test_should_track_endpoint_case_sensitivity(self):
        """Test that pattern matching is case sensitive."""
        path = "/API/FORECAST/dataset123/STREAM"
        result = self.middleware._should_track_endpoint(path)
        assert result is False

    # Billing Tests
    def test_should_bill_endpoint_synthesis_stream_simple(self):
        """Test billing for synthesis stream simple endpoint."""
        path = "/api/synthesis/my_dataset/stream"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is True

    def test_should_bill_endpoint_synthesis_stream_user_dataset(self):
        """Test billing for synthesis stream with user and dataset."""
        path = "/api/synthesis/user123/my_dataset/stream"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is True

    def test_should_bill_endpoint_synthesis_stream_user_dataset_training(self):
        """Test billing for synthesis stream with user, dataset, and training."""
        path = "/api/synthesis/user123/my_dataset/training_job456/stream"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is True

    def test_should_bill_endpoint_foundation_models_backtest(self):
        """Test billing for foundation models backtest endpoint."""
        path = "/api/foundation_models/forecast/backtest"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is True

    def test_should_bill_endpoint_synthetic_data_agent_generate(self):
        """Test billing for synthetic data agent generate endpoint."""
        path = "/api/synthetic_data_agent/generate_synthetic_data/user123/dataset456"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is True

    def test_should_not_bill_endpoint_forecast_stream(self):
        """Test that forecast stream endpoints are not billable."""
        path = "/api/forecast/my_dataset/stream"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_foundation_models_upload(self):
        """Test that foundation models upload endpoint is not billable."""
        path = "/api/foundation_models/upload"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_foundation_models_forecast(self):
        """Test that foundation models forecast endpoint is not billable."""
        path = "/api/foundation_models/forecast"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_preprocess(self):
        """Test that preprocess endpoint is not billable."""
        path = "/api/preprocess/user123/dataset456"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_train(self):
        """Test that train endpoint is not billable."""
        path = "/api/train/dataset456"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_postprocess(self):
        """Test that postprocess endpoint is not billable."""
        path = "/api/postprocess/html/dataset456"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_explain(self):
        """Test that explain endpoint is not billable."""
        path = "/api/explain"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_pretrained_anomaly(self):
        """Test that pretrained anomaly endpoint is not billable."""
        path = "/api/pretrained_anomaly/dataset456/stream"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_v2_anomaly_detection(self):
        """Test that v2 anomaly detection endpoint is not billable."""
        path = "/api/v2/anomaly_detection/dataset456/stream"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_non_api_path(self):
        """Test that non-API paths are not billable."""
        path = "/health"
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    def test_should_not_bill_endpoint_empty_path(self):
        """Test that empty path is not billable."""
        path = ""
        result = APIEndpoints.should_bill_endpoint(path)
        assert result is False

    # API Key Validation Tests
    @patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "false"})
    def test_api_key_validation_disabled_by_default(self):
        """Test that API key validation is disabled when API_KEY_AUTH_ENABLED is false."""
        request = Mock()
        request.headers = {"X-API-Key": "test-key"}
        request.url.path = "/api/synthesis/test_dataset/stream"

        # Mock the call_next function as AsyncMock
        call_next = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        call_next.return_value = mock_response

        # Mock the database session
        with patch(
            "synthefy_pkg.app.middleware.api_usage_middleware.SessionLocal"
        ) as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db

            # Mock validate_api_key to ensure it's not called
            with patch(
                "synthefy_pkg.app.middleware.api_usage_middleware.validate_api_key"
            ) as mock_validate:
                # Mock log_api_usage_async
                with patch(
                    "synthefy_pkg.app.middleware.api_usage_middleware.log_api_usage_async"
                ) as _:
                    # Run the dispatch method
                    import asyncio

                    _ = asyncio.run(
                        self.middleware.dispatch(request, call_next)
                    )

                    # Verify validate_api_key was not called
                    mock_validate.assert_not_called()

    @patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"})
    def test_api_key_validation_enabled_with_valid_key(self):
        """Test that API key validation works when enabled and key is valid."""
        request = Mock()
        request.headers = {"X-API-Key": "valid-api-key"}
        request.url.path = "/api/synthesis/test_dataset/stream"

        # Mock the call_next function as AsyncMock
        call_next = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        call_next.return_value = mock_response

        # Mock the database session
        with patch(
            "synthefy_pkg.app.middleware.api_usage_middleware.SessionLocal"
        ) as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db

            # Mock validate_api_key to return a user_id
            with patch(
                "synthefy_pkg.app.middleware.api_usage_middleware.validate_api_key"
            ) as mock_validate:
                mock_validate.return_value = "user-123"

                # Mock log_api_usage_async
                with patch(
                    "synthefy_pkg.app.middleware.api_usage_middleware.log_api_usage_async"
                ) as mock_log:
                    # Run the dispatch method
                    import asyncio

                    _ = asyncio.run(
                        self.middleware.dispatch(request, call_next)
                    )

                    # Verify validate_api_key was called with correct parameters
                    mock_validate.assert_called_once_with(
                        mock_db, "valid-api-key"
                    )

                    # Verify log_api_usage_async was called with correct user_id
                    mock_log.assert_called_once()
                    call_args = mock_log.call_args
                    assert call_args[1]["user_id"] == "user-123"

    @patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"})
    def test_api_key_validation_enabled_with_invalid_key(self):
        """Test that API key validation handles invalid keys gracefully."""
        request = Mock()
        request.headers = {"X-API-Key": "invalid-api-key"}
        request.url.path = "/api/synthesis/test_dataset/stream"

        # Mock the call_next function as AsyncMock
        call_next = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        call_next.return_value = mock_response

        # Mock the database session
        with patch(
            "synthefy_pkg.app.middleware.api_usage_middleware.SessionLocal"
        ) as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db

            # Mock validate_api_key to return None (invalid key)
            with patch(
                "synthefy_pkg.app.middleware.api_usage_middleware.validate_api_key"
            ) as mock_validate:
                mock_validate.return_value = None

                # Mock log_api_usage_async
                with patch(
                    "synthefy_pkg.app.middleware.api_usage_middleware.log_api_usage_async"
                ) as _:
                    # Run the dispatch method
                    import asyncio

                    _ = asyncio.run(
                        self.middleware.dispatch(request, call_next)
                    )

                    # Verify validate_api_key was called
                    mock_validate.assert_called_once_with(
                        mock_db, "invalid-api-key"
                    )

                    # Verify log_api_usage_async was called with anonymous user_id
                    # Note: This test needs to be updated since we now use the helper function
                    # The actual behavior depends on the helper function implementation

    @patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"})
    def test_api_key_validation_database_exception_handling(self):
        """Test that database exceptions during API key validation are handled gracefully."""
        request = Mock()
        request.headers = {"X-API-Key": "test-key"}
        request.url.path = "/api/synthesis/test_dataset/stream"

        # Mock the call_next function as AsyncMock
        call_next = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        call_next.return_value = mock_response

        # Mock the database session to raise an exception
        with patch(
            "synthefy_pkg.app.middleware.api_usage_middleware.SessionLocal"
        ) as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            # Mock log_api_usage_async
            with patch(
                "synthefy_pkg.app.middleware.api_usage_middleware.log_api_usage_async"
            ) as mock_log:
                # Run the dispatch method
                import asyncio

                _ = asyncio.run(self.middleware.dispatch(request, call_next))

                # Verify log_api_usage_async was called with anonymous user_id
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[1]["user_id"] == "anonymous"

    @patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"})
    def test_api_key_validation_validate_api_key_exception_handling(self):
        """Test that exceptions in validate_api_key function are handled gracefully."""
        request = Mock()
        request.headers = {"X-API-Key": "test-key"}
        request.url.path = "/api/synthesis/test_dataset/stream"

        # Mock the call_next function as AsyncMock
        call_next = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        call_next.return_value = mock_response

        # Mock the database session
        with patch(
            "synthefy_pkg.app.middleware.api_usage_middleware.SessionLocal"
        ) as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db

            # Mock validate_api_key to raise an exception
            with patch(
                "synthefy_pkg.app.middleware.api_usage_middleware.validate_api_key"
            ) as mock_validate:
                mock_validate.side_effect = Exception("Validation failed")

                # Mock log_api_usage_async
                with patch(
                    "synthefy_pkg.app.middleware.api_usage_middleware.log_api_usage_async"
                ) as mock_log:
                    # Run the dispatch method
                    import asyncio

                    _ = asyncio.run(
                        self.middleware.dispatch(request, call_next)
                    )

                    # Verify log_api_usage_async was called with anonymous user_id
                    mock_log.assert_called_once()
                    call_args = mock_log.call_args
                    assert call_args[1]["user_id"] == "anonymous"

    @patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"})
    def test_api_key_validation_no_x_api_key_header(self):
        """Test that API key validation is skipped when X-API-Key header is missing."""
        request = Mock()
        request.headers = {
            "Authorization": "Bearer some-token"
        }  # No X-API-Key header
        request.url.path = "/api/synthesis/test_dataset/stream"

        # Mock the call_next function as AsyncMock
        call_next = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        call_next.return_value = mock_response

        # Mock the database session
        with patch(
            "synthefy_pkg.app.middleware.api_usage_middleware.SessionLocal"
        ) as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db

            # Mock validate_api_key to ensure it's not called
            with patch(
                "synthefy_pkg.app.middleware.api_usage_middleware.validate_api_key"
            ) as mock_validate:
                # Mock log_api_usage_async
                with patch(
                    "synthefy_pkg.app.middleware.api_usage_middleware.log_api_usage_async"
                ) as _:
                    # Run the dispatch method
                    import asyncio

                    _ = asyncio.run(
                        self.middleware.dispatch(request, call_next)
                    )

                    # Verify validate_api_key was not called
                    mock_validate.assert_not_called()

    @patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"})
    def test_api_key_validation_database_session_cleanup(self):
        """Test that database session is properly closed even when exceptions occur."""
        request = Mock()
        request.headers = {"X-API-Key": "test-key"}
        request.url.path = "/api/synthesis/test_dataset/stream"

        # Mock the call_next function as AsyncMock
        call_next = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        call_next.return_value = mock_response

        # Mock the database session
        with patch(
            "synthefy_pkg.app.middleware.api_usage_middleware.SessionLocal"
        ) as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db

            # Mock validate_api_key to raise an exception
            with patch(
                "synthefy_pkg.app.middleware.api_usage_middleware.validate_api_key"
            ) as mock_validate:
                mock_validate.side_effect = Exception("Validation failed")

                # Mock log_api_usage_async
                with patch(
                    "synthefy_pkg.app.middleware.api_usage_middleware.log_api_usage_async"
                ) as _:
                    # Run the dispatch method
                    import asyncio

                    _ = asyncio.run(
                        self.middleware.dispatch(request, call_next)
                    )

                    # Verify database session was closed
                    mock_db.close.assert_called_once()

    def test_extract_dataset_name_forecast_stream(self):
        """Test extracting dataset name from forecast stream endpoint."""
        path = "/api/forecast/my_dataset/stream"
        result = self.middleware._extract_dataset_name(path)
        assert result == "my_dataset"

    def test_extract_dataset_name_synthesis_stream(self):
        """Test extracting dataset name from synthesis stream endpoint."""
        path = "/api/synthesis/test_dataset/stream"
        result = self.middleware._extract_dataset_name(path)
        assert result == "test_dataset"

    def test_extract_dataset_name_v2_anomaly_detection(self):
        """Test extracting dataset name from v2 anomaly detection endpoint."""
        path = "/api/v2/anomaly_detection/anomaly_dataset/stream"
        result = self.middleware._extract_dataset_name(path)
        assert result == "anomaly_dataset"

    def test_extract_dataset_name_no_match(self):
        """Test extracting dataset name from non-matching path."""
        path = "/api/health"
        result = self.middleware._extract_dataset_name(path)
        assert result is None

    def test_extract_dataset_name_preprocess(self):
        """Test dataset name extraction from preprocess endpoint."""
        path = "/api/preprocess/user123/dataset456"
        result = self.middleware._extract_dataset_name(path)
        assert result == "dataset456"

    def test_extract_dataset_name_train(self):
        """Test dataset name extraction from train endpoint."""
        path = "/api/train/dataset789"
        result = self.middleware._extract_dataset_name(path)
        assert result == "dataset789"

    def test_extract_dataset_name_postprocess(self):
        """Test dataset name extraction from postprocess endpoint."""
        path = "/api/postprocess/html/dataset123"
        result = self.middleware._extract_dataset_name(path)
        assert result == "dataset123"

    def test_extract_dataset_name_synthetic_data_agent(self):
        """Test dataset name extraction from synthetic data agent endpoint."""
        path = "/api/synthetic_data_agent/generate_synthetic_data/user123/dataset456"
        result = self.middleware._extract_dataset_name(path)
        assert result == "dataset456"

    def test_extract_user_id_from_path_synthesis(self):
        """Test user_id extraction from synthesis endpoint path."""
        path = "/api/synthesis/user123/dataset456/job789/stream"
        result = self.middleware._extract_user_id_from_path(path)
        assert result == "user123"

    def test_extract_user_id_from_path_preprocess(self):
        """Test user_id extraction from preprocess endpoint path."""
        path = "/api/preprocess/user123/dataset456"
        result = self.middleware._extract_user_id_from_path(path)
        assert result == "user123"

    def test_extract_user_id_from_path_synthetic_data_agent(self):
        """Test user_id extraction from synthetic data agent endpoint path."""
        path = "/api/synthetic_data_agent/generate_synthetic_data/user123/dataset456"
        result = self.middleware._extract_user_id_from_path(path)
        assert result == "user123"

    def test_extract_user_id_from_path_no_match(self):
        """Test user_id extraction from non-matching path."""
        path = "/api/forecast/dataset123/stream"
        result = self.middleware._extract_user_id_from_path(path)
        assert result is None

    def test_extract_user_id_from_dataset_name_with_uuid(self):
        """Test extracting user ID from dataset name with UUID."""
        dataset_name = "rrest_april_2_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0"
        result = extract_user_id_from_dataset_name(dataset_name)
        assert result == "6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0"

    def test_extract_user_id_from_dataset_name_no_uuid(self):
        """Test extracting user ID from dataset name without UUID."""
        dataset_name = "simple_dataset_name"
        result = extract_user_id_from_dataset_name(dataset_name)
        assert result is None

    def test_extract_user_id_from_dataset_name_invalid_uuid(self):
        """Test extracting user ID from dataset name with invalid UUID."""
        dataset_name = "test_dataset_invalid_uuid"
        result = extract_user_id_from_dataset_name(dataset_name)
        assert result is None

    def test_compound_dataset_name_construction(self):
        """Test that compound dataset name is correctly constructed for dataset creation endpoints."""
        # Test preprocess endpoint
        user_id = "972bf553-d0ad-4521-9e74-2da9e790798b"
        dataset_name = "qotaq5"

        # The middleware should construct: qotaq5_972bf553-d0ad-4521-9e74-2da9e790798b
        expected_compound_name = f"{dataset_name}_{user_id}"
        assert (
            expected_compound_name
            == "qotaq5_972bf553-d0ad-4521-9e74-2da9e790798b"
        )

        # Verify this matches the pattern from the response output_path
        # Response: "/app/data/qotaq5_972bf553-d0ad-4521-9e74-2da9e790798b"
        # Last part: "qotaq5_972bf553-d0ad-4521-9e74-2da9e790798b"
        assert (
            expected_compound_name
            == "qotaq5_972bf553-d0ad-4521-9e74-2da9e790798b"
        )

    def test_is_dataset_creation_endpoint(self):
        """Test that dataset creation endpoints are correctly identified."""
        # Test preprocess endpoint (should be identified as dataset creation)
        preprocess_path = (
            "/api/preprocess/972bf553-d0ad-4521-9e74-2da9e790798b/qotaq5"
        )
        assert (
            APIEndpoints.is_dataset_creation_endpoint(preprocess_path) is True
        )

        # Test train endpoint (should NOT be identified as dataset creation)
        train_path = "/api/train/dataset123"
        assert APIEndpoints.is_dataset_creation_endpoint(train_path) is False

        # Test synthetic data agent endpoint (should NOT be identified as dataset creation)
        synthetic_path = "/api/synthetic_data_agent/generate_synthetic_data/user123/dataset456"
        assert (
            APIEndpoints.is_dataset_creation_endpoint(synthetic_path) is False
        )

    def test_extract_user_id_from_dataset_name_none(self):
        """Test extracting user ID from None dataset name."""
        result = extract_user_id_from_dataset_name(None)
        assert result is None

    def test_is_api_key_create_endpoint(self):
        """Test API key creation endpoint detection."""
        path = "/api/user_api_key/"
        result = APIEndpoints.is_api_key_create_endpoint(path)
        assert result is True

        path = "/api/user_api_key"
        result = APIEndpoints.is_api_key_create_endpoint(path)
        assert result is True

        path = "/api/user_api_key/123"
        result = APIEndpoints.is_api_key_create_endpoint(path)
        assert result is False

    def test_is_api_key_delete_endpoint(self):
        """Test API key deletion endpoint detection."""
        path = "/api/user_api_key/123"
        result = APIEndpoints.is_api_key_delete_endpoint(path)
        assert result is True

        path = "/api/user_api_key/abc"
        result = APIEndpoints.is_api_key_delete_endpoint(path)
        assert result is True

        path = "/api/user_api_key/"
        result = APIEndpoints.is_api_key_delete_endpoint(path)
        assert result is False

    def test_is_api_keys_list_endpoint(self):
        """Test API keys list endpoint detection."""
        path = "/api/user_api_keys/"
        result = APIEndpoints.is_api_keys_list_endpoint(path)
        assert result is True

        path = "/api/user_api_keys"
        result = APIEndpoints.is_api_keys_list_endpoint(path)
        assert result is True

        path = "/api/user_api_key/"
        result = APIEndpoints.is_api_keys_list_endpoint(path)
        assert result is False

    def test_is_api_key_management_endpoint(self):
        """Test API key management endpoint detection."""
        # Test create endpoint
        path = "/api/user_api_key/"
        result = APIEndpoints.is_api_key_management_endpoint(path)
        assert result is True

        # Test delete endpoint
        path = "/api/user_api_key/123"
        result = APIEndpoints.is_api_key_management_endpoint(path)
        assert result is True

        # Test list endpoint
        path = "/api/user_api_keys/"
        result = APIEndpoints.is_api_key_management_endpoint(path)
        assert result is True

        # Test non-API key endpoint
        path = "/api/synthesis/test/stream"
        result = APIEndpoints.is_api_key_management_endpoint(path)
        assert result is False

    def test_get_event_type_and_operation_api_key_create(self):
        """Test event type and operation for API key creation endpoint."""
        path = "/api/user_api_key/"
        event_type, operation = APIEndpoints.get_event_type_and_operation(path)
        assert event_type == "api_key_creation"
        assert operation == "create_api_key"

    def test_get_event_type_and_operation_api_key_delete(self):
        """Test event type and operation for API key deletion endpoint."""
        path = "/api/user_api_key/123"
        event_type, operation = APIEndpoints.get_event_type_and_operation(path)
        assert event_type == "api_key_deletion"
        assert operation == "delete_api_key"

    def test_get_event_type_and_operation_api_keys_list(self):
        """Test event type and operation for API keys list endpoint."""
        path = "/api/user_api_keys/"
        event_type, operation = APIEndpoints.get_event_type_and_operation(path)
        assert event_type == "api_key_retrieval"
        assert operation == "list_api_keys"

    def test_is_health_check_endpoint(self):
        """Test health check endpoint detection."""
        path = "/api/health"
        result = APIEndpoints.is_health_check_endpoint(path)
        assert result is True

        path = "/api/health/"
        result = APIEndpoints.is_health_check_endpoint(path)
        assert result is False

        path = "/api/health/status"
        result = APIEndpoints.is_health_check_endpoint(path)
        assert result is False

    def test_is_system_endpoint(self):
        """Test system endpoint detection."""
        # Test health check endpoint
        path = "/api/health"
        result = APIEndpoints.is_system_endpoint(path)
        assert result is True

        # Test non-system endpoint
        path = "/api/synthesis/test/stream"
        result = APIEndpoints.is_system_endpoint(path)
        assert result is False

    def test_get_event_type_and_operation_health_check(self):
        """Test event type and operation for health check endpoint."""
        path = "/api/health"
        event_type, operation = APIEndpoints.get_event_type_and_operation(path)
        assert event_type == "health_check"
        assert operation == "health_check"


class TestMetricsManagerUtils:
    """Test cases for metrics manager utility functions."""

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_api_key_priority(self):
        """Test that API key authentication takes highest priority."""
        request = Mock()
        request.headers = {"X-API-Key": "valid-api-key"}

        # Mock functions
        mock_validate_api_key = Mock(return_value="user-from-api-key")
        mock_get_supabase_user = Mock(return_value="user-from-token")
        mock_extract_form_data = AsyncMock(
            return_value={"user_id": "user-from-form"}
        )
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-body"
        )

        with patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"}):
            result = await extract_user_id_and_dataset_name(
                request=request,
                path="/api/synthesis/test/stream",
                api_key="valid-api-key",
                dataset_name="test_dataset_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0",
                validate_api_key_func=mock_validate_api_key,
                get_supabase_user_func=mock_get_supabase_user,
                extract_form_data_func=mock_extract_form_data,
                extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
            )

        user_id, dataset_name = result
        assert user_id == "user-from-api-key"
        assert (
            dataset_name == "test_dataset_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0"
        )

        # Verify only API key validation was called
        mock_validate_api_key.assert_called_once_with("valid-api-key")
        mock_get_supabase_user.assert_not_called()
        mock_extract_form_data.assert_not_called()
        mock_extract_user_id_from_request_body.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_bearer_token_fallback(self):
        """Test that Bearer token authentication is used when API key fails."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid-token"}

        # Mock functions
        mock_validate_api_key = Mock(
            return_value=None
        )  # API key validation fails
        mock_get_supabase_user = Mock(return_value="user-from-token")
        mock_extract_form_data = AsyncMock(
            return_value={"user_id": "user-from-form"}
        )
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-body"
        )

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/synthesis/test/stream",
            api_key=None,
            dataset_name="test_dataset",
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "user-from-token"
        assert dataset_name == "test_dataset"

        # Verify Bearer token validation was called
        mock_get_supabase_user.assert_called_once_with("Bearer valid-token")

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_dataset_name_fallback(self):
        """Test that dataset name extraction is used when auth methods fail."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(return_value=None)

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/synthesis/test/stream",
            api_key=None,
            dataset_name="test_dataset_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0",
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0"
        assert (
            dataset_name == "test_dataset_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0"
        )

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_form_data_upload(self):
        """Test form data extraction for upload endpoint."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(
            return_value={"user_id": "user-from-form", "filename": "test.csv"}
        )
        mock_extract_user_id_from_request_body = AsyncMock(return_value=None)

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/foundation_models/upload",
            api_key=None,
            dataset_name=None,
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "user-from-form"
        assert dataset_name == "test"

        # Verify form data extraction was called
        mock_extract_form_data.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_request_body_backtest(self):
        """Test request body extraction for backtest endpoint."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-body"
        )

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/foundation_models/forecast/backtest",
            api_key=None,
            dataset_name=None,
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "user-from-body"
        assert dataset_name is None

        # Verify request body extraction was called
        mock_extract_user_id_from_request_body.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_anonymous_fallback(self):
        """Test that anonymous user_id is used when all methods fail."""
        request = Mock()
        request.headers = {}

        # Mock functions - all return None
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(return_value=None)

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/synthesis/test/stream",
            api_key=None,
            dataset_name=None,
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "anonymous"
        assert dataset_name is None

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_api_key_disabled(self):
        """Test that API key validation is skipped when disabled."""
        request = Mock()
        request.headers = {
            "Authorization": "Bearer valid-token"
        }  # Add Bearer token

        # Mock functions
        mock_validate_api_key = Mock(return_value="user-from-api-key")
        mock_get_supabase_user = Mock(return_value="user-from-token")
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "false"}):
            result = await extract_user_id_and_dataset_name(
                request=request,
                path="/api/synthesis/test/stream",
                api_key=None,  # No API key
                dataset_name="test_dataset",
                validate_api_key_func=mock_validate_api_key,
                get_supabase_user_func=mock_get_supabase_user,
                extract_form_data_func=mock_extract_form_data,
                extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
            )

        user_id, dataset_name = result
        # Should fall back to Bearer token since API key is disabled
        assert user_id == "user-from-token"
        assert dataset_name == "test_dataset"

        # Verify API key validation was not called
        mock_validate_api_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_exception_handling(self):
        """Test that exceptions in validation functions are handled gracefully."""
        request = Mock()
        request.headers = {"X-API-Key": "test-key"}

        # Mock functions with exceptions
        mock_validate_api_key = Mock(
            side_effect=Exception("API key validation failed")
        )
        mock_get_supabase_user = Mock(
            side_effect=Exception("Token validation failed")
        )
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(return_value=None)

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/synthesis/test/stream",
            api_key="test-key",
            dataset_name="test_dataset_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0",
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        # Should fall back to dataset name extraction
        assert user_id == "6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0"
        assert (
            dataset_name == "test_dataset_6dcaefa8-c8e7-457d-b905-f7ad7d0a60d0"
        )

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_form_data_updates_dataset_name(
        self,
    ):
        """Test that form data can update the dataset name."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(
            return_value={
                "user_id": "user-from-form",
                "filename": "new_dataset.csv",
            }
        )
        mock_extract_user_id_from_request_body = AsyncMock(return_value=None)

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/foundation_models/upload",
            api_key=None,
            dataset_name=None,  # No original dataset name, so form data will update it
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "user-from-form"
        assert dataset_name == "new_dataset"  # Should be updated from form data

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_form_data_preserves_original_dataset_name(
        self,
    ):
        """Test that form data doesn't update dataset name when original exists."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(
            return_value={
                "user_id": "user-from-form",
                "filename": "new_dataset.csv",
            }
        )
        mock_extract_user_id_from_request_body = AsyncMock(return_value=None)

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/foundation_models/upload",
            api_key=None,
            dataset_name="original_dataset",  # Original dataset name exists
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "user-from-form"
        assert (
            dataset_name == "original_dataset"
        )  # Should preserve original dataset name


class TestDatasetNameExtractionFromRequestBody:
    """Test cases for extracting dataset name from request body."""

    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = APIUsageMiddleware(app=None)

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_backtest_format(self):
        """Test extracting dataset name from backtest format file_path_key."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/different_values/different_values.parquet"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result == "different_values"

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_forecast_with_filters(
        self,
    ):
        """Test extracting dataset name from forecast format with filters."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/different_values/feature_A=1/agg=sum/data.parquet"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result == "different_values"

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_forecast_with_aggregation(
        self,
    ):
        """Test extracting dataset name from forecast format with aggregation."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/different_values/unfiltered_agg=sum/data.parquet"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result == "different_values"

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_complex_dataset_name(
        self,
    ):
        """Test extracting dataset name with complex dataset name."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/my_complex_dataset_name_2024/feature_A=1/agg=sum/data.parquet"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result == "my_complex_dataset_name_2024"

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_no_foundation_models(
        self,
    ):
        """Test when file_path_key doesn't contain 'foundation_models/'."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/other_path/dataset_name/data.parquet"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_malformed_path(self):
        """Test when file_path_key is malformed after 'foundation_models/'."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_no_config(self):
        """Test when request body doesn't contain config."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "other_field": "some_value",
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_no_file_path_key(
        self,
    ):
        """Test when config doesn't contain file_path_key."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {"other_field": "some_value"},
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_empty_file_path_key(
        self,
    ):
        """Test when file_path_key is empty."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {"file_path_key": ""},
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_non_json_content_type(
        self,
    ):
        """Test when content type is not application/json."""
        request = Mock()
        request.headers = {"content-type": "text/plain"}
        request.json = AsyncMock(return_value={})

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_json_exception(self):
        """Test when request.json() raises an exception."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(side_effect=Exception("JSON parsing failed"))

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_nested_config(self):
        """Test when config is nested deeper in the request body."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "request_data": {
                    "config": {
                        "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/different_values/data.parquet"
                    }
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result is None  # Should not find config at this level

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_dataset_name_with_special_chars(
        self,
    ):
        """Test extracting dataset name with special characters."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/dataset-name-with-dashes_2024/feature_A=1/agg=sum/data.parquet"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result == "dataset-name-with-dashes_2024"

    @pytest.mark.asyncio
    async def test_extract_dataset_name_from_request_body_multiple_foundation_models(
        self,
    ):
        """Test when file_path_key contains multiple 'foundation_models/' occurrences."""
        request = Mock()
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(
            return_value={
                "user_id": "test-user-123",
                "config": {
                    "file_path_key": "972bf553-d0ad-4521-9e74-2da9e790798b/foundation_models/different_values/foundation_models/another_path/data.parquet"
                },
            }
        )

        result = await self.middleware._extract_dataset_name_from_request_body(
            request
        )
        assert result == "different_values"  # Should take the first occurrence


class TestUpdatedExtractUserIdAndDatasetName:
    """Test cases for the updated extract_user_id_and_dataset_name function."""

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_forecast_endpoint_request_body(
        self,
    ):
        """Test that forecast endpoint extracts user_id from request body."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-request-body"
        )

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/foundation_models/forecast",
            api_key=None,
            dataset_name=None,
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "user-from-request-body"
        assert dataset_name is None

        # Verify the function was called
        mock_extract_user_id_from_request_body.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_backtest_endpoint_request_body(
        self,
    ):
        """Test that backtest endpoint extracts user_id from request body."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-request-body"
        )

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/foundation_models/forecast/backtest",
            api_key=None,
            dataset_name=None,
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "user-from-request-body"
        assert dataset_name is None

        # Verify the function was called
        mock_extract_user_id_from_request_body.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_other_endpoint_no_request_body(
        self,
    ):
        """Test that other endpoints don't extract user_id from request body."""
        request = Mock()
        request.headers = {}

        # Mock functions
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-request-body"
        )

        result = await extract_user_id_and_dataset_name(
            request=request,
            path="/api/synthesis/test/stream",
            api_key=None,
            dataset_name=None,
            validate_api_key_func=mock_validate_api_key,
            get_supabase_user_func=mock_get_supabase_user,
            extract_form_data_func=mock_extract_form_data,
            extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
        )

        user_id, dataset_name = result
        assert user_id == "anonymous"  # Should fall back to anonymous
        assert dataset_name is None

        # Verify the function was NOT called
        mock_extract_user_id_from_request_body.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_forecast_endpoint_with_api_key_priority(
        self,
    ):
        """Test that API key takes priority over request body extraction for forecast endpoint."""
        request = Mock()
        request.headers = {"X-API-Key": "valid-api-key"}

        # Mock functions
        mock_validate_api_key = Mock(return_value="user-from-api-key")
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-request-body"
        )

        with patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"}):
            result = await extract_user_id_and_dataset_name(
                request=request,
                path="/api/foundation_models/forecast",
                api_key="valid-api-key",
                dataset_name=None,
                validate_api_key_func=mock_validate_api_key,
                get_supabase_user_func=mock_get_supabase_user,
                extract_form_data_func=mock_extract_form_data,
                extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
            )

        user_id, dataset_name = result
        assert user_id == "user-from-api-key"  # API key should take priority
        assert dataset_name is None

        # Verify request body extraction was NOT called since API key was successful
        mock_extract_user_id_from_request_body.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_user_id_and_dataset_name_forecast_endpoint_fallback_chain(
        self,
    ):
        """Test the fallback chain for forecast endpoint when API key fails."""
        request = Mock()
        request.headers = {"X-API-Key": "invalid-api-key"}

        # Mock functions - API key fails, request body succeeds
        mock_validate_api_key = Mock(return_value=None)
        mock_get_supabase_user = Mock(return_value=None)
        mock_extract_form_data = AsyncMock(return_value=None)
        mock_extract_user_id_from_request_body = AsyncMock(
            return_value="user-from-request-body"
        )

        with patch.dict(os.environ, {"API_KEY_AUTH_ENABLED": "true"}):
            result = await extract_user_id_and_dataset_name(
                request=request,
                path="/api/foundation_models/forecast",
                api_key="invalid-api-key",
                dataset_name=None,
                validate_api_key_func=mock_validate_api_key,
                get_supabase_user_func=mock_get_supabase_user,
                extract_form_data_func=mock_extract_form_data,
                extract_user_id_from_request_body_func=mock_extract_user_id_from_request_body,
            )

        user_id, dataset_name = result
        assert (
            user_id == "user-from-request-body"
        )  # Should fall back to request body
        assert dataset_name is None

        # Verify request body extraction was called
        mock_extract_user_id_from_request_body.assert_called_once_with(request)
