# Import the function to test
import unittest
from pathlib import Path

from synthefy_pkg.app.utils.fm_model_utils import build_s3_full_path


class TestBuildS3FullPath(unittest.TestCase):
    """Test cases for build_s3_full_path function."""

    def test_build_s3_full_path_basic(self):
        """Test basic functionality of build_s3_full_path."""

        # Create mock metadata
        class MockFoundationModelTypeMetadata:
            def __init__(self, model_version, date):
                self.model_version = model_version
                self.date = date

        class MockMetadata:
            def __init__(self, s3_url):
                self.s3_url = s3_url
                self.forecasting = MockFoundationModelTypeMetadata(
                    "v3e", "2024-01-15"
                )

        metadata = MockMetadata("s3://bucket/models/")  # type: ignore

        result = build_s3_full_path(metadata, "forecasting")  # type: ignore
        expected = "s3://bucket/models/v3e_2024-01-15/"

        self.assertEqual(result, expected)

    def test_build_s3_full_path_with_double_slash(self):
        """Test that double slashes at the end are handled correctly."""

        # Create mock metadata with double slash
        class MockFoundationModelTypeMetadata:
            def __init__(self, model_version, date):
                self.model_version = model_version
                self.date = date

        class MockMetadata:
            def __init__(self, s3_url):
                self.s3_url = s3_url
                self.forecasting = MockFoundationModelTypeMetadata(
                    "v3e", "2024-01-15"
                )

        metadata = MockMetadata("s3://bucket/models//")

        result = build_s3_full_path(metadata, "forecasting")  # type: ignore
        expected = "s3://bucket/models/v3e_2024-01-15/"

        self.assertEqual(result, expected)

    def test_build_s3_full_path_synthesis_model(self):
        """Test with synthesis model type."""

        # Create mock metadata
        class MockFoundationModelTypeMetadata:
            def __init__(self, model_version, date):
                self.model_version = model_version
                self.date = date

        class MockMetadata:
            def __init__(self, s3_url):
                self.s3_url = s3_url
                self.synthesis = MockFoundationModelTypeMetadata(
                    "v2a", "2024-02-20"
                )

        metadata = MockMetadata("s3://bucket/models/")  # type: ignore

        result = build_s3_full_path(metadata, "synthesis")  # type: ignore
        expected = "s3://bucket/models/v2a_2024-02-20/"

        self.assertEqual(result, expected)

    def test_build_s3_full_path_no_trailing_slash(self):
        """Test with s3_url that doesn't end with slash."""

        # Create mock metadata without trailing slash
        class MockFoundationModelTypeMetadata:
            def __init__(self, model_version, date):
                self.model_version = model_version
                self.date = date

        class MockMetadata:
            def __init__(self, s3_url):
                self.s3_url = s3_url
                self.forecasting = MockFoundationModelTypeMetadata(
                    "v3e", "2024-01-15"
                )

        metadata = MockMetadata("s3://bucket/models")  # type: ignore

        result = build_s3_full_path(metadata, "forecasting")  # type: ignore
        expected = "s3://bucket/models/v3e_2024-01-15/"

        self.assertEqual(result, expected)
