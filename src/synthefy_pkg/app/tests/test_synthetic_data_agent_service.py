import json
import os
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

import aioboto3
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from synthefy_pkg.app.config import SyntheticDataAgentSettings
from synthefy_pkg.app.data_models import (
    ContinuousVariation,
    DiscreteVariation,
    MetaDataGrid,
    MetaDataGridSample,
    MetaDataVariation,
    PerturbationType,
    TimeStampVariation,
)
from synthefy_pkg.app.main import create_app
from synthefy_pkg.app.routers.synthetic_data_agent import (
    get_synthetic_data_agent_service,
)
from synthefy_pkg.app.services.synthetic_data_agent_service import (
    SyntheticDataAgentService,
    count_metadata_perturbations,
)


@pytest.fixture(scope="function")
def app():
    config_path = os.path.join(
        os.environ["SYNTHEFY_PACKAGE_BASE"],
        "src/synthefy_pkg/app/services/configs/api_config_general_test.yaml",
    )
    return create_app(config_path)


@pytest.fixture(scope="function")
def client(app):
    return TestClient(app)


@pytest.fixture(scope="function")
def synthetic_data_agent_service():
    os.environ["SYNTHEFY_CONFIG_PATH"] = os.path.join(
        str(os.getenv("SYNTHEFY_PACKAGE_BASE")),
        "src/synthefy_pkg/app/services/configs/api_config_general_test.yaml",
    )

    service = get_synthetic_data_agent_service(
        user_id="test_user",
        dataset_name="test_dataset",
    )
    return service


@pytest.fixture
def mock_settings():
    return SyntheticDataAgentSettings(
        bucket_name="test-bucket",
        dataset_path=os.environ["SYNTHEFY_DATASETS_BASE"],
        json_save_path="/tmp",
        preprocessed_data_path="mocked_preprocessed_data_path",
        preprocess_config_path="mocked_preprocess_config_path",
        synthesis_config_path="mocked_synthesis_config_path",
        synthesis_model_path="mocked_synthesis_model_path",
    )


@pytest.fixture
def mock_aioboto3_session():
    with patch("aioboto3.Session") as mock_session:
        mock_s3_client = AsyncMock()
        # Setup the async context manager
        mock_session.client.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_s3_client),
            __aexit__=AsyncMock(return_value=None),
        )
        yield mock_session


@pytest.fixture
def service(mock_settings, mock_aioboto3_session):
    return SyntheticDataAgentService(
        user_id="test_user",
        dataset_name="test_dataset",
        settings=mock_settings,
        aioboto3_session=mock_aioboto3_session,
    )


class TestSyntheticDataAgentService:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment before each test"""
        self.test_user = "test_user"
        self.test_dataset = "test_dataset"

        # Test data
        self.full_labels_description = {
            "group_labels_combinations": {
                "subject-device": ["S1-phone", "S10-watch", "S2-tablet"]
            },
            "discrete_labels": {
                "category": {"A": 10, "B": 20, "C": 30},
                "status": {"active": 10, "inactive": 50},
            },
            "continuous_labels": {
                "temperature": {"min": 20.0, "max": 30.0, "mean": 25.0},
                "humidity": {"min": 40.0, "max": 80.0, "mean": 60.0},
            },
            "time_labels": {
                "@timestamp": {
                    "min": pd.Timestamp("2024-03-24 20:00:00+0000", tz="UTC"),
                    "max": pd.Timestamp("2024-04-23 23:30:00+0000", tz="UTC"),
                    "interval": pd.Timedelta("0 days 00:30:00"),
                }
            },
        }

        self.empty_labels_description = {
            "discrete_labels": {},
            "continuous_labels": {},
            "time_labels": {},
            "group_labels_combinations": [],
        }

        self.preprocess_config = {
            "window_size": 24,
            "use_label_col_as_discrete_metadata": True,
        }

    @pytest.mark.asyncio
    async def test_get_metadata_grid_sample_full_metadata(self, service):
        """Test metadata grid sample with full metadata"""
        with (
            patch("builtins.open"),
            patch("pickle.load") as mock_pickle_load,
            patch("json.load") as mock_json_load,
            patch(
                "synthefy_pkg.app.services.synthetic_data_agent_service.SyntheticDataAgentService._ensure_preprocessed_data_downloaded",
                return_value=True,
            ),
            patch(
                "synthefy_pkg.app.services.synthetic_data_agent_service.SyntheticDataAgentService._ensure_synthesis_model_downloaded",
                return_value=True,
            ),
        ):
            # Configure mock for labels_description
            mock_pickle_load.return_value = self.full_labels_description

            # Configure mock for config
            mock_json_load.return_value = self.preprocess_config

            # Execute
            result = await service.get_metadata_grid_sample()

            assert isinstance(result, MetaDataGridSample)
            # Assert
            assert (
                result.group_labels_combinations
                == self.full_labels_description["group_labels_combinations"]
            )

            assert result.metadata_range is not None
            # Verify discrete conditions
            discrete_conditions = result.metadata_range.discrete_conditions
            assert len(discrete_conditions) == 2  # Gender and subject
            assert discrete_conditions[0].name == "category"
            assert discrete_conditions[1].name == "status"
            assert discrete_conditions[0].options == ["A", "B", "C"]
            assert discrete_conditions[1].options == ["active", "inactive"]

            # Verify continuous conditions
            continuous_conditions = result.metadata_range.continuous_conditions
            assert (
                len(continuous_conditions) == 2
            )  # ACC_wrist_idx_0 and ACC_wrist_idx_1
            assert continuous_conditions[0].name == "temperature"
            assert continuous_conditions[0].min_val == 20.0
            assert continuous_conditions[0].max_val == 30.0
            assert continuous_conditions[1].name == "humidity"
            assert continuous_conditions[1].min_val == 40.0
            assert continuous_conditions[1].max_val == 80.0

            assert result.timestamps_range is not None
            assert (
                result.timestamps_range.min_time == "2024-03-24 20:00:00+00:00"
            )
            assert (
                result.timestamps_range.max_time == "2024-04-23 23:30:00+00:00"
            )
            assert result.timestamps_range.interval == "0 days 00:30:00"
            assert result.timestamps_range.length == 24

    @pytest.mark.asyncio
    async def test_get_metadata_grid_sample_empty_metadata(self, service):
        """Test metadata grid sample with empty metadata"""
        with (
            patch("builtins.open"),
            patch("pickle.load") as mock_pickle_load,
            patch("json.load") as mock_json_load,
            patch(
                "synthefy_pkg.app.services.synthetic_data_agent_service.SyntheticDataAgentService._ensure_preprocessed_data_downloaded",
                return_value=True,
            ),
            patch(
                "synthefy_pkg.app.services.synthetic_data_agent_service.SyntheticDataAgentService._ensure_synthesis_model_downloaded",
                return_value=True,
            ),
        ):
            # Configure mock for labels_description
            mock_pickle_load.return_value = self.empty_labels_description

            # Configure mock for config
            mock_json_load.return_value = self.preprocess_config

            # Execute
            result = await service.get_metadata_grid_sample()

            # Assert
            assert isinstance(result, MetaDataGridSample)
            assert result.group_labels_combinations is None

            assert result.metadata_range is not None
            assert len(result.metadata_range.discrete_conditions) == 0
            assert len(result.metadata_range.continuous_conditions) == 0

            assert result.timestamps_range is None

    @pytest.mark.asyncio
    async def test_get_metadata_grid_sample_partial_metadata(self, service):
        """Test metadata grid sample with partial metadata (only some fields populated)"""
        partial_labels_description = {
            "group_labels_combinations": {
                "subject-device": ["S1-phone", "S2-phone"]
            },
            "discrete_labels": {"category": {"A": 10, "B": 20}},
            "continuous_labels": {},
            "time_labels": {},
        }

        with (
            patch("builtins.open"),
            patch("pickle.load") as mock_pickle_load,
            patch("json.load") as mock_json_load,
            patch(
                "synthefy_pkg.app.services.synthetic_data_agent_service.SyntheticDataAgentService._ensure_preprocessed_data_downloaded",
                return_value=True,
            ),
            patch(
                "synthefy_pkg.app.services.synthetic_data_agent_service.SyntheticDataAgentService._ensure_synthesis_model_downloaded",
                return_value=True,
            ),
        ):
            mock_pickle_load.return_value = partial_labels_description
            mock_json_load.return_value = self.preprocess_config

            result = await service.get_metadata_grid_sample()

            assert isinstance(result, MetaDataGridSample)
            assert (
                result.group_labels_combinations
                == partial_labels_description["group_labels_combinations"]
            )
            assert result.metadata_range is not None
            assert len(result.metadata_range.discrete_conditions) == 1
            assert len(result.metadata_range.continuous_conditions) == 0
            assert result.timestamps_range is None


class TestCountMetadataPerturbations:
    """Tests for the count_metadata_perturbations helper function"""

    @pytest.mark.asyncio
    async def test_multiple_discrete_variations(self, service):
        """Test combinations with multiple discrete variations"""
        metadata_grid = MetaDataGrid(
            continuous_conditions_to_change=None,
            discrete_conditions_to_change=[
                DiscreteVariation(name="category", options=["A", "B"]),
                DiscreteVariation(
                    name="status", options=["active", "inactive"]
                ),
            ],
            timestamps_conditions_to_change=None,
        )

        result = await count_metadata_perturbations(metadata_grid)
        expected = [
            [
                MetaDataVariation(
                    order=0,
                    name="category",
                    value="A",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="active",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="category",
                    value="A",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="inactive",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="category",
                    value="B",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="active",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="category",
                    value="B",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="inactive",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
        ]
        assert result == expected
        assert len(result) == 4  # 2 categories * 2 statuses

    @pytest.mark.asyncio
    async def test_multiple_continuous_variations(self, service):
        """Test combinations with multiple continuous variations"""
        metadata_grid = MetaDataGrid(
            continuous_conditions_to_change=[
                ContinuousVariation(
                    name="heart_rate",
                    perturbation_type=PerturbationType.ADD,
                    perturbation_value=5.0,
                ),
                ContinuousVariation(
                    name="respiratory_rate",
                    perturbation_type=PerturbationType.MULTIPLY,
                    perturbation_value=1.1,
                ),
            ],
            discrete_conditions_to_change=None,
            timestamps_conditions_to_change=None,
        )

        result = await count_metadata_perturbations(metadata_grid)
        expected = [
            [
                MetaDataVariation(
                    order=0,
                    name="heart_rate",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=0,
                    name="respiratory_rate",
                    value=1.1,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
            ],
        ]
        assert result == expected
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_mixed_types(self):
        """Test combinations with mixed types of variations"""
        metadata_grid = MetaDataGrid(
            continuous_conditions_to_change=[
                ContinuousVariation(
                    name="heart_rate",
                    perturbation_type=PerturbationType.ADD,
                    perturbation_value=5.0,
                )
            ],
            discrete_conditions_to_change=[
                DiscreteVariation(name="category", options=["A", "B"])
            ],
            timestamps_conditions_to_change=[
                TimeStampVariation(
                    name="@timestamp",
                    perturbation_type=PerturbationType.ADD,
                    perturbation_value=3600,
                )
            ],
        )

        result = await count_metadata_perturbations(metadata_grid)
        expected = [
            [
                MetaDataVariation(
                    order=0,
                    name="heart_rate",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=0,
                    name="category",
                    value="A",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=0,
                    name="@timestamp",
                    value=3600,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="heart_rate",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=0,
                    name="category",
                    value="B",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=0,
                    name="@timestamp",
                    value=3600,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
        ]
        assert result == expected
        assert len(result) == 2  # 2 category options

    @pytest.mark.asyncio
    async def test_empty_grid(self):
        """Test combinations with empty metadata grid"""
        metadata_grid = MetaDataGrid(
            continuous_conditions_to_change=None,
            discrete_conditions_to_change=None,
            timestamps_conditions_to_change=None,
        )

        result = await count_metadata_perturbations(metadata_grid)
        assert result == []
