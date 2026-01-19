import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from synthefy_pkg.app.config import SynthefySettings
from synthefy_pkg.app.data_models import SynthefyDefaultSetupOptions
from synthefy_pkg.app.main import create_app

skip_in_ci = pytest.mark.skipif(
    os.environ.get("SKIP_CERTAIN_TESTS") == "true",
    reason="Test skipped based on environment variable",
)


@pytest.fixture(scope="function")
def app():
    config_path = os.path.join(
        os.environ.get("SYNTHEFY_PACKAGE_BASE"),
        "examples/configs/api_configs/api_config_twamp_one_month.yaml",
    )
    return create_app(config_path)


@pytest.fixture(scope="function")
def client(app):

    return TestClient(app)


@pytest.fixture
def mocked_settings_twamp_one_month():
    return SynthefySettings(
        dataset_path=os.environ["SYNTHEFY_DATASETS_BASE"],
        preprocess_config_path=os.path.join(
            os.environ["SYNTHEFY_PACKAGE_BASE"],
            "examples/configs/preprocessing_configs/config_twamp_one_month_preprocessing.json",
        ),
        synthesis_config_path=os.path.join(
            os.environ["SYNTHEFY_PACKAGE_BASE"],
            "examples/configs/synthesis_configs/config_twamp_one_month_synthesis.yaml",
        ),
        synthesis_model_path=os.path.join(
            os.environ["SYNTHEFY_DATASETS_BASE"],
            "training_logs/twamp_one_month/Time_Series_Diffusion_Training/twamp_one_month_100/checkpoints/best_model.ckpt",
        ),
        json_save_path="/tmp",
    )


@skip_in_ci
class TestGetDefaultSynthefyRequest:
    def test_get_default_synthesis_request_twamp_one_month(
        self, mocked_settings_twamp_one_month, client
    ):
        with patch(
            "synthefy_pkg.app.routers.setup_ui.get_synthefy_settings",
            return_value=mocked_settings_twamp_one_month,
        ):
            # get_response = _get_default_synthefy_request(
            #     settings=mocked_settings_twamp_one_month
            # )
            get_response = client.get("/api/default")

        assert get_response.status_code == 200

        preprocess_config = json.load(
            open(mocked_settings_twamp_one_month.preprocess_config_path)
        )
        discrete_col_names = preprocess_config.get("discrete", {}).get("cols", [])
        continuous_col_names = preprocess_config.get("continuous", {}).get("cols", [])
        timeseries_col_names = preprocess_config.get("timeseries", {}).get("cols", [])
        group_labels_col_names = preprocess_config.get("group_labels", {}).get(
            "cols", []
        )
        preprocess_config.get("timestamps_col", [])

        converted = SynthefyDefaultSetupOptions.model_validate(get_response.json())

        # test the 0'th window display
        for timeseries_data in converted.windows[0].timeseries_data:
            assert timeseries_data.name in timeseries_col_names

        # TODO - we need a discussion on this regarding use_label_col_as_discrete_metadata
        # for discrete_metadata in converted.window[0].metadata.discrete_conditions:
        #     assert discrete_metadata.name in discrete_col_names

        for continuous_metadata in converted.windows[0].metadata.continuous_conditions:
            assert continuous_metadata.name in continuous_col_names + [
                f"timestamps_feature_{i}" for i in ["Y", "M", "D", "H", "T", "S"]
            ]

        # Testing the Ranges + options
        # all lengths of metadata must be window_size
        assert len(converted.metadata_range.discrete_conditions) > 0
        assert len(converted.metadata_range.continuous_conditions) > 0
        for discrete_metadata in converted.metadata_range.discrete_conditions:
            # Label-Tuple-sto_kng-sran_node_id-sender_dscp
            if discrete_metadata.name.startswith("Label-Tuple-"):
                for discrete_md_name in discrete_metadata.name.split("-")[2:]:
                    assert (
                        discrete_md_name in discrete_col_names + group_labels_col_names
                    )
            else:
                assert (
                    discrete_metadata.name
                    in discrete_col_names + group_labels_col_names
                )
        for continuous_metadata in converted.metadata_range.continuous_conditions:
            assert continuous_metadata.name in continuous_col_names + [
                f"timestamps_feature_{i}" for i in ["Y", "M", "D", "H", "T", "S"]
            ]
        # turn this test off since it depends on the stride.
        # assert converted.timestamps_range == TimeStampsRange(
        #     name="@timestamp",
        #     min_time="2024-03-24 00:00:00+00:00",
        #     max_time="2024-04-23 23:30:00+00:00",
        #     interval="0 days 00:30:00",
        #     length=96,
        # )
        assert isinstance(converted.text, list)
        assert len(converted.text) == 0
