import pytest
import pandas as pd
import json
from datetime import datetime
from synthefy_pkg.app.data_models import PreTrainedAnomalyV2Request, DynamicTimeSeriesData
from synthefy_pkg.app.services.pretrained_anomaly_v2_service import PreTrainedAnomalyV2Service

class TestPreTrainedAnomalyV2Service:
    @pytest.fixture
    def mock_config(self, tmp_path):
        config = {
            "timeseries": {
                "cols": ["value"]
            },
            "timestamps_col": ["timestamp"],
            "anomaly_detection": {
                "settings": {
                    "num_anomalies_limit": 50,
                    "min_anomaly_score": 0.7,
                    "n_jobs": -1
                }
            }
        }
        config_path = tmp_path / "mock_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return str(config_path)

    @pytest.fixture
    def service(self, mock_config):
        class MockSettings:
            preprocess_config_path = mock_config
        return PreTrainedAnomalyV2Service(MockSettings())

    @pytest.fixture
    def valid_request_data(self):
        return {
            "dataset_name": "test_dataset",
            "root": {
                "timestamp": {
                    0: "2023-01-01T00:00:00",
                    1: "2023-01-01T01:00:00",
                    2: "2023-01-01T02:00:00"
                },
                "value": {
                    0: 1.0,
                    1: 2.0,
                    2: 3.0
                }
            },
            "num_anomalies_limit": 10,
            "min_anomaly_score": 0.5,
            "n_jobs": 1
        }

    def test_valid_request_pretrained(self, service, valid_request_data):
        request = PreTrainedAnomalyV2Request(**valid_request_data)
        df = service._preprocess_request(request)
        
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"timestamp", "value"}
        assert len(df) == 3
        assert service.num_anomalies_limit == 10
        assert service.min_anomaly_score == 0.5
        assert service.n_jobs == 1

    def test_valid_request_dynamic(self, service, valid_request_data):
        # Create the nested dictionary structure expected by DynamicTimeSeriesData
        root_data = {
            "timestamp": valid_request_data['root']['timestamp'],  # Already in correct format
            "value": valid_request_data['root']['value']  # Already in correct format
        }
        
        request = DynamicTimeSeriesData(root=root_data)
        
        df = service._preprocess_request(request)
        
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"timestamp", "value"}
        assert len(df) == 3
        # Check that settings were loaded from config
        assert service.num_anomalies_limit == 50
        assert service.min_anomaly_score == 0.7
        assert service.n_jobs == -1

    def test_empty_root(self, service):
        request = PreTrainedAnomalyV2Request(
            dataset_name="test_dataset",
            root={},
            num_anomalies_limit=10,
            min_anomaly_score=0.5,
            n_jobs=1
        )
        with pytest.raises(ValueError, match="Failed to preprocess anomaly_v2 request: Root dictionary cannot be empty"):
            service._preprocess_request(request)

    def test_missing_required_timestamps_col(self, service):
        request = PreTrainedAnomalyV2Request(
            dataset_name="test_dataset",
            root={
                "timestamp_bla": {0: "2023-01-01T00:00:00"},
                "value": {0: 1, 1: 2}
            },
            num_anomalies_limit=10,
            min_anomaly_score=0.5,
            n_jobs=1
        )
        with pytest.raises(ValueError, match="Failed to preprocess anomaly_v2 request: .*Timestamps required for anomaly detection.*"):
            service._preprocess_request(request)

    def test_missing_required_columns(self, service):
        request = PreTrainedAnomalyV2Request(
            dataset_name="test_dataset",
            root={
                "timestamp": {0: "2023-01-01T00:00:00"},
                "some_column": {0: 1, 1: 2}
            },
            num_anomalies_limit=10,
            min_anomaly_score=0.5,
            n_jobs=1
        )
        with pytest.raises(ValueError, match="Failed to preprocess anomaly_v2 request: .*Missing required columns.*"):
            service._preprocess_request(request)

    def test_empty_dataframe(self, service):
        request = PreTrainedAnomalyV2Request(
            dataset_name="test_dataset",
            root={
                "timestamp": {},
                "value": {}
            },
            num_anomalies_limit=10,
            min_anomaly_score=0.5,
            n_jobs=1
        )
        with pytest.raises(ValueError, match="Failed to preprocess anomaly_v2 request: Converted DataFrame is empty"):
            service._preprocess_request(request)

    @pytest.mark.integration
    def test_full_anomaly_detection_flow(self, service, valid_request_data, mocker):
        mock_detector = mocker.patch('synthefy_pkg.app.services.pretrained_anomaly_v2_service.AnomalyDetector')
        mock_detector.return_value.detect_anomalies.return_value = (
            {
                "test_kpi": {
                    "anomaly_type_1": {
                        "group_1": []
                    }
                }
            },
            {}  # concurrent_results
        )

        request = PreTrainedAnomalyV2Request(**valid_request_data)
        response = service.pretrained_anomaly_detection(request)

        assert response.results == {
            "test_kpi": {
                "anomaly_type_1": {
                    "group_1": []
                }
            }
        }
        assert response.concurrent_results == {}
        
        mock_detector.return_value.detect_anomalies.assert_called_once_with(
            df=mocker.ANY,
            num_anomalies_limit=request.num_anomalies_limit,
            min_anomaly_score=request.min_anomaly_score,
            n_jobs=request.n_jobs,
        )

    def test_invalid_anomaly_detection_results(self, service, valid_request_data, mocker):
        mock_detector = mocker.patch('synthefy_pkg.app.services.pretrained_anomaly_v2_service.AnomalyDetector')
        mock_detector.return_value.detect_anomalies.return_value = ("invalid_results", {})

        request = PreTrainedAnomalyV2Request(**valid_request_data)
        with pytest.raises(ValueError, match="Anomaly detection results must be a dictionary"):
            service.pretrained_anomaly_detection(request)
