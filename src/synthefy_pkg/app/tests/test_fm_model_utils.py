import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from synthefy_pkg.app.data_models import (
    SynthefyFoundationModelMetadata,
)
from synthefy_pkg.app.utils.fm_model_utils import (
    get_available_models,
    get_foundation_model_info,
    get_model_ckpt_and_config_path,
    get_model_metadata,
)
from synthefy_pkg.app.utils.s3_utils import (
    parse_s3_url,
)

FORCASTING_MODEL_URL = "s3://synthefy-checkpoints/dev-checkpoints/synthefy-foundation-model/v3e_2025-07-02/"
SYNTHEFY_FOUNDATION_MODEL_PATH = (
    "s3://synthefy-checkpoints/dev-checkpoints/synthefy-foundation-model/"
)


def test_get_available_models():
    """Test get_available_models function."""
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    test_model_dir = Path(temp_dir) / "test_models"

    try:
        # Test 1: Directory doesn't exist - should create it and return empty list
        result = get_available_models(test_model_dir)
        assert result == []
        assert test_model_dir.exists()

        # Test 2: Create some model directories and test again
        model1 = test_model_dir / "model1"
        model2 = test_model_dir / "model2"
        model1.mkdir()
        model2.mkdir()

        result = get_available_models(test_model_dir)
        assert len(result) == 2
        assert model1 in result
        assert model2 in result

    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "foundation_model_path", [SYNTHEFY_FOUNDATION_MODEL_PATH]
)
def test_get_model_metadata(foundation_model_path: str):
    """Test get_model_metadata function."""
    os.environ["SYNTHEFY_FOUNDATION_MODEL_PATH"] = (
        SYNTHEFY_FOUNDATION_MODEL_PATH
    )
    foundation_model_metadata = get_model_metadata(
        foundation_model_path=foundation_model_path
    )
    assert (
        foundation_model_metadata.s3_url.split("/")[-2]
        == "synthefy-foundation-model"
    )


@pytest.mark.parametrize("model_url", [FORCASTING_MODEL_URL])
def test_model_exists(model_url: str):
    """Test model exists function."""
    bucket, key = parse_s3_url(s3_url=model_url)
    assert bucket == "synthefy-checkpoints"
    assert key.split("/")[-2] == model_url.split("/")[-2]


@pytest.mark.parametrize(
    "forecasting_model_url, foundation_model_path",
    [(FORCASTING_MODEL_URL, SYNTHEFY_FOUNDATION_MODEL_PATH)],
)
def test_get_foundation_model_info(
    forecasting_model_url: str, foundation_model_path: str
):
    """Test get_foundation_model_info function."""
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    test_model_dir = Path(temp_dir) / "test_models"

    try:
        result = get_foundation_model_info(
            local_model_dir=test_model_dir,
            foundation_model_path=foundation_model_path,
        )
        assert isinstance(result, SynthefyFoundationModelMetadata)
        forecast_metadata = result.forecasting
        forecasting_model_version = forecasting_model_url.split("/")[-2].split(
            "_"
        )[0]

        assert forecasting_model_version == forecast_metadata.model_version

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch("synthefy_pkg.app.utils.fm_model_utils.cleanup_local_directories")
def test_get_model_ckpt_and_config_path(mock_cleanup):
    """Test get_model_ckpt_and_config_path function."""
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    test_model_dir = Path(temp_dir) / "test_model"
    test_model_dir.mkdir(parents=True)

    try:
        # Test 1: Both files exist - should return both paths
        ckpt_file = test_model_dir / "model.ckpt"
        config_file = test_model_dir / "model_config.yaml"
        ckpt_file.touch()
        config_file.touch()

        ckpt_path, config_path = get_model_ckpt_and_config_path(test_model_dir)
        assert ckpt_path == ckpt_file
        assert config_path == config_file
        mock_cleanup.assert_not_called()

        # Test 2: if config_filename is provided, it should return the config file
        ckpt_file = test_model_dir / "model.ckpt"
        config_file = test_model_dir / "model_config.yaml"
        model_card_file = test_model_dir / "model_card.yaml"
        ckpt_file.touch()
        config_file.touch()
        model_card_file.touch()

        ckpt_path, config_path = get_model_ckpt_and_config_path(
            model_dir=test_model_dir, config_filename="model_config.yaml"
        )
        assert ckpt_path == ckpt_file
        assert config_path == config_file
        mock_cleanup.assert_not_called()

        # Test 3: Missing checkpoint file - should return None, None
        ckpt_file.unlink()
        model_card_file.unlink()
        ckpt_path, config_path = get_model_ckpt_and_config_path(test_model_dir)
        assert ckpt_path is None
        assert config_path is None
        mock_cleanup.assert_called_once()

        # Test 3: Missing config file - should return None, None
        # Reset mock and recreate ckpt file
        mock_cleanup.reset_mock()
        ckpt_file.touch()
        config_file.unlink()
        ckpt_path, config_path = get_model_ckpt_and_config_path(test_model_dir)
        assert ckpt_path is None
        assert config_path is None
        mock_cleanup.assert_called_once()

    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
