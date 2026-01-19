import json
import os
import re
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from synthefy_pkg.data.window_and_dataframe_utils import (
    convert_windows_to_dataframe,
    filter_continuous_features_by_pattern,
    load_dataset_files,
)
from synthefy_pkg.preprocessing.base_text_embedder import (
    EMBEDDED_COL_NAMES_PREFIX,
)
from synthefy_pkg.preprocessing.preprocess import (
    LAG_COL_FORMAT,
    TIMESTAMPS_FEATURES_PREFIX,
)


class TestLoadDatasetFiles:
    @pytest.fixture
    def mock_dataset_path(self):
        return "/mock/path/test_dataset"

    @pytest.fixture
    def mock_package_path(self):
        return "/mock/path/package"

    @pytest.fixture
    def mock_arrays(self):
        return {
            "timeseries": np.random.rand(10, 5, 20),  # (B, C, W)
            "discrete": np.random.randint(0, 5, (10, 20, 3)),  # (B, W, C)
            "continuous": np.random.rand(10, 20, 4),  # (B, W, C)
            "timestamps": np.arange(200).reshape(10, 20),  # (B, W)
            "text": np.random.rand(10, 20, 2),  # (B, W, C)
        }

    @pytest.fixture
    def mock_preprocess_config(self):
        return {
            "use_label_col_as_discrete_metadata": True,
            "group_labels": {"cols": ["group1", "group2"]},
        }

    @pytest.fixture
    def mock_group_label_dict(self):
        return {
            "train": {0: ("A", "X"), 1: ("B", "Y")},
            "val": {0: ("C", "Z")},
            "test": {0: ("D", "W")},
        }

    @pytest.mark.asyncio
    async def test_load_dataset_files_all_exist(
        self,
        mock_dataset_path,
        mock_package_path,
        mock_arrays,
        mock_preprocess_config,
    ):
        with (
            patch.dict(
                "os.environ",
                {
                    "SYNTHEFY_DATASETS_BASE": mock_dataset_path,
                    "SYNTHEFY_PACKAGE_BASE": mock_package_path,
                },
            ),
            patch("os.path.exists", return_value=True),
            patch("numpy.load") as mock_load,
            patch("json.load", return_value=mock_preprocess_config),
            patch("builtins.open", mock_open()),
        ):

            def mock_load_func(x, **kwargs):
                filename = os.path.basename(x)
                if "original_discrete" in filename:
                    return mock_arrays["discrete"]
                elif "timestamps_original" in filename:
                    return mock_arrays["timestamps"]
                elif "original_text" in filename:
                    return mock_arrays["text"]
                else:
                    return mock_arrays[filename.split("_")[1].split(".")[0]]

            mock_load.side_effect = mock_load_func

            result = await load_dataset_files(
                dataset_name="test_dataset", split="train"
            )

            assert len(result) == 7  # Now returns 7 items instead of 5
            assert all(isinstance(arr, np.ndarray) for arr in result[:5])
            assert (
                result[5] is None
            )  # group_label_dict should be None when use_label_col_as_discrete_metadata is True
            assert result[6] == mock_preprocess_config["group_labels"]["cols"]

    @pytest.mark.asyncio
    async def test_load_dataset_files_with_group_labels(
        self,
        mock_dataset_path,
        mock_package_path,
        mock_arrays,
        mock_preprocess_config,
        mock_group_label_dict,
    ):
        # Modify preprocess config to test group label loading
        mock_preprocess_config["use_label_col_as_discrete_metadata"] = False

        with (
            patch.dict(
                "os.environ",
                {
                    "SYNTHEFY_DATASETS_BASE": mock_dataset_path,
                    "SYNTHEFY_PACKAGE_BASE": mock_package_path,
                },
            ),
            patch("os.path.exists", return_value=True),
            patch("numpy.load") as mock_load,
            patch("json.load", return_value=mock_preprocess_config),
            patch("pickle.load", return_value=mock_group_label_dict),
            patch("builtins.open", mock_open()),
        ):

            def mock_load_func(x, **kwargs):
                filename = os.path.basename(x)
                if "original_discrete" in filename:
                    return mock_arrays["discrete"]
                elif "timestamps_original" in filename:
                    return mock_arrays["timestamps"]
                elif "original_text" in filename:
                    return mock_arrays["text"]
                else:
                    return mock_arrays[filename.split("_")[1].split(".")[0]]

            mock_load.side_effect = mock_load_func

            result = await load_dataset_files(
                dataset_name="test_dataset", split="train"
            )

            assert len(result) == 7
            assert all(isinstance(arr, np.ndarray) for arr in result[:5])
            assert result[5] == mock_group_label_dict
            assert result[6] == mock_preprocess_config["group_labels"]["cols"]

    @pytest.mark.asyncio
    async def test_load_dataset_files_missing_env_vars(self):
        """Test that the function raises an AssertionError when environment variables are missing."""
        test_cases = [
            # Missing both env vars
            {},
            # Missing SYNTHEFY_PACKAGE_BASE
            {"SYNTHEFY_DATASETS_BASE": "/mock/path"},
            # Missing SYNTHEFY_DATASETS_BASE
            {"SYNTHEFY_PACKAGE_BASE": "/mock/path"},
        ]

        for env_vars in test_cases:
            with (
                patch.dict("os.environ", env_vars, clear=True),
                patch(
                    "os.path.exists", return_value=False
                ),  # Prevent directory check from failing
            ):
                with pytest.raises(AssertionError) as exc_info:
                    await load_dataset_files(
                        dataset_name="test_dataset", split="train"
                    )
                assert (
                    "ENV SYNTHEFY_DATASETS_BASE or SYNTHEFY_PACKAGE_BASE is not set"
                    in str(exc_info.value)
                )

    @pytest.mark.asyncio
    async def test_load_dataset_files_with_preloaded_data(
        self,
        mock_dataset_path,
        mock_package_path,
        mock_arrays,
        mock_preprocess_config,
    ):
        preloaded_data = {
            "timeseries": mock_arrays["timeseries"],
            "discrete_original": mock_arrays["discrete"],
            "continuous": mock_arrays["continuous"],
            "timestamps_original": mock_arrays["timestamps"],
            "text": mock_arrays["text"],
            "group_label_to_split_idx_dict": {"train": {0: ("A", "X")}},
            "group_label_cols": ["group1", "group2"],
        }

        with (
            patch.dict(
                "os.environ",
                {
                    "SYNTHEFY_DATASETS_BASE": mock_dataset_path,
                    "SYNTHEFY_PACKAGE_BASE": mock_package_path,
                },
            ),
            patch("os.path.exists", return_value=True),
        ):
            result = await load_dataset_files(
                dataset_name="test_dataset", split="train", **preloaded_data
            )

            assert len(result) == 7
            assert np.array_equal(result[0], preloaded_data["timeseries"])
            assert np.array_equal(
                result[1], preloaded_data["discrete_original"]
            )
            assert np.array_equal(result[2], preloaded_data["continuous"])
            assert np.array_equal(
                result[3], preloaded_data["timestamps_original"]
            )
            assert np.array_equal(result[4], preloaded_data["text"])
            assert result[5] == preloaded_data["group_label_to_split_idx_dict"]
            assert result[6] == preloaded_data["group_label_cols"]

    @pytest.mark.asyncio
    async def test_load_dataset_files_partial_preloaded(
        self, mock_dataset_path, mock_arrays
    ):
        # Mock the environment variable before any imports or function calls
        with (
            patch.dict(
                "os.environ",
                {
                    "SYNTHEFY_DATASETS_BASE": mock_dataset_path,
                    "SYNTHEFY_PACKAGE_BASE": "/mock/package/path",
                },
                clear=True,  # This ensures no other env vars interfere
            ),
            patch("os.path.exists", return_value=True),
            patch("numpy.load") as mock_load,
            patch(
                "json.load",
                return_value={
                    "use_label_col_as_discrete_metadata": True,
                    "group_labels": {"cols": ["group1", "group2"]},
                },
            ),
            patch("builtins.open", mock_open()),
        ):
            # Rest of the test remains the same
            def mock_load_func(x, **kwargs):
                filename = os.path.basename(x)
                if "original_discrete" in filename:
                    return mock_arrays["discrete"]
                elif "timestamps_original" in filename:
                    return mock_arrays["timestamps"]
                elif "original_text" in filename:
                    return mock_arrays["text"]
                else:
                    return mock_arrays[filename.split("_")[1].split(".")[0]]

            mock_load.side_effect = mock_load_func

            partial_data = {
                "timeseries": mock_arrays["timeseries"],
                "continuous": mock_arrays["continuous"],
            }

            result = await load_dataset_files(
                dataset_name="test_dataset", split="train", **partial_data
            )

            expected_calls = [
                call(
                    os.path.join(
                        mock_dataset_path,
                        "test_dataset/train_original_discrete_windows.npy",
                    ),
                    allow_pickle=True,
                ),
                call(
                    os.path.join(
                        mock_dataset_path,
                        "test_dataset/train_timestamps_original.npy",
                    ),
                    allow_pickle=True,
                ),
                call(
                    os.path.join(
                        mock_dataset_path,
                        "test_dataset/train_original_text_conditions.npy",
                    ),
                    allow_pickle=True,
                ),
            ]
            mock_load.assert_has_calls(expected_calls, any_order=True)

            assert len(result) == 7  # Updated to check for 7 return values
            assert np.array_equal(result[0], mock_arrays["timeseries"])
            assert np.array_equal(result[1], mock_arrays["discrete"])
            assert np.array_equal(result[2], mock_arrays["continuous"])
            assert np.array_equal(result[3], mock_arrays["timestamps"])
            assert np.array_equal(result[4], mock_arrays["text"])
            assert (
                result[5] is None
            )  # group_label_dict should be None when use_label_col_as_discrete_metadata is True
            assert result[6] == [
                "group1",
                "group2",
            ]  # group_label_cols from mock config


class TestFilterContinuousFeaturesbyPattern:
    @pytest.fixture
    def sample_data(self):
        return {
            "continuous": np.random.rand(10, 9),  # 10 samples, 9 features
            "colnames": [
                "feature_1",
                f"{TIMESTAMPS_FEATURES_PREFIX}1",
                "feature_2",
                f"{TIMESTAMPS_FEATURES_PREFIX}2",
                "feature_3",
                f"{EMBEDDED_COL_NAMES_PREFIX}1",
                "feature_4",  # Added base column
                f"{LAG_COL_FORMAT.format(col='feature_1', window_size=1)}",
                f"{LAG_COL_FORMAT.format(col='feature_1', window_size=2)}",
            ],
        }

    def test_filter_timestamp_features(self, sample_data):
        continuous, filtered_colnames = filter_continuous_features_by_pattern(
            sample_data["continuous"],
            sample_data["colnames"],
            pattern=TIMESTAMPS_FEATURES_PREFIX,
            match_type="prefix",
        )

        assert continuous.shape[1] == 7  # Should remove 2 timestamp features
        assert len(filtered_colnames) == 7
        assert all(
            not col.startswith(TIMESTAMPS_FEATURES_PREFIX)
            for col in filtered_colnames
        )

    def test_filter_embedded_features(self, sample_data):
        continuous, filtered_colnames = filter_continuous_features_by_pattern(
            sample_data["continuous"],
            sample_data["colnames"],
            pattern=EMBEDDED_COL_NAMES_PREFIX,
            match_type="prefix",
        )

        assert continuous.shape[1] == 8  # Should remove 1 embedded feature
        assert len(filtered_colnames) == 8
        assert all(
            not col.startswith(EMBEDDED_COL_NAMES_PREFIX)
            for col in filtered_colnames
        )

    def test_filter_lag_features(self, sample_data):
        continuous, filtered_colnames = filter_continuous_features_by_pattern(
            sample_data["continuous"],
            sample_data["colnames"],
            pattern=LAG_COL_FORMAT,
            match_type="lag_format",
            timeseries_colnames=["feature_1", "feature_2", "feature_3"],
        )

        assert continuous.shape[1] == 7  # Should remove 2 lag features
        assert len(filtered_colnames) == 7
        assert all(
            not re.match(r".+_lag_\d+$", col) for col in filtered_colnames
        )

    def test_filter_lag_features_no_matching_base_cols(self, sample_data):
        """Test lag feature filtering when base columns don't exist in timeseries_colnames"""
        continuous, filtered_colnames = filter_continuous_features_by_pattern(
            sample_data["continuous"],
            sample_data["colnames"],
            pattern=LAG_COL_FORMAT,
            match_type="lag_format",
            timeseries_colnames=["other_feature"],  # No matching base columns
        )

        assert continuous.shape[1] == 9  # Should not remove any features
        assert len(filtered_colnames) == 9
        assert len([col for col in filtered_colnames if "_lag_" in col]) == 2

    def test_filter_no_matches(self):
        continuous = np.random.rand(10, 3)
        colnames = ["feature_1", "feature_2", "feature_3"]

        filtered_continuous, filtered_colnames = (
            filter_continuous_features_by_pattern(
                continuous,
                colnames,
                pattern="nonexistent_",
                match_type="prefix",
            )
        )

        assert np.array_equal(filtered_continuous, continuous)
        assert filtered_colnames == colnames

    def test_filter_empty_array(self):
        continuous = np.array([])
        colnames = []

        filtered_continuous, filtered_colnames = (
            filter_continuous_features_by_pattern(
                continuous, colnames, pattern="test_", match_type="prefix"
            )
        )

        assert len(filtered_continuous) == 0
        assert filtered_colnames == []

    def test_lag_format_missing_timeseries_colnames(self, sample_data):
        """Test that function raises ValueError when timeseries_colnames is missing for lag_format"""
        with pytest.raises(ValueError) as exc_info:
            filter_continuous_features_by_pattern(
                sample_data["continuous"],
                sample_data["colnames"],
                pattern=LAG_COL_FORMAT,
                match_type="lag_format",  # Requires timeseries_colnames
                timeseries_colnames=None,
            )
        assert (
            "timeseries_colnames is required when match_type is 'lag_format'"
            in str(exc_info.value)
        )

    def test_invalid_lag_format_columns(self, sample_data):
        """Test handling of columns that look like lag columns but don't match format exactly"""
        # Add some invalid lag format columns
        invalid_colnames = sample_data["colnames"] + [
            "feature_1_lag",  # Missing window size
            "feature_1_lag_abc",  # Non-numeric window size
            "_lag_1",  # Missing column name
        ]
        continuous = np.random.rand(10, len(invalid_colnames))

        filtered_continuous, filtered_colnames = (
            filter_continuous_features_by_pattern(
                continuous,
                invalid_colnames,
                pattern=LAG_COL_FORMAT,
                match_type="lag_format",
                timeseries_colnames=["feature_1"],
            )
        )

        # Should only match and remove valid lag format columns
        assert (
            len(filtered_colnames) == len(invalid_colnames) - 2
        )  # Only remove valid lag columns
        assert all(
            col in filtered_colnames
            for col in ["feature_1_lag", "feature_1_lag_abc", "_lag_1"]
        )


class TestConvertWindowsToDataframe:
    @pytest.fixture
    def mock_dataset_path(self):
        return "/mock/path"

    @pytest.fixture
    def mock_data(self):
        # Create a date range for 20 timestamps, repeated 10 times for each batch
        timestamp = pd.Timestamp("2024-01-01")
        timestamps = np.full((10, 20, 1), timestamp)

        return {
            "timeseries": np.random.rand(10, 5, 20),  # (B, C, W)
            "discrete_original": np.random.randint(
                0, 5, (10, 20, 3)
            ),  # (B, W, C)
            "continuous": np.random.rand(10, 20, 5),  # (B, W, C)
            "timestamps_original": timestamps,  # (B, W, C)
            "text": np.random.rand(10, 20, 2),  # (B, W, C)
            "group_label_to_split_idx_dict": {
                "train": {
                    0: ("A", "X"),
                    1: ("B", "Y"),
                    2: ("C", "Z"),
                    3: ("D", "W"),
                    4: ("E", "V"),
                    5: ("F", "U"),
                    6: ("G", "T"),
                    7: ("H", "S"),
                    8: ("I", "R"),
                    9: ("J", "Q"),
                },
            },
            "group_label_cols": ["group1", "group2"],
        }

    @pytest.fixture
    def mock_colnames(self):
        return {
            "timeseries_colnames": ["ts_1", "ts_2", "ts_3", "ts_4", "ts_5"],
            "original_discrete_colnames": [
                "disc_1",
                "disc_2",
                "disc_3",
            ],
            "continuous_colnames": [
                "cont_1",
                "cont_2",
                f"{TIMESTAMPS_FEATURES_PREFIX}1",
                "cont_3",
                f"{LAG_COL_FORMAT.format(col='ts_1', window_size=512)}",
            ],
            "original_text_colnames": [
                f"{EMBEDDED_COL_NAMES_PREFIX}1",
                f"{EMBEDDED_COL_NAMES_PREFIX}2",
            ],
            "timestamps_colnames": ["Timestamp_"],
            "group_label_cols": ["group1", "group2"],
        }

    @pytest.fixture
    def mock_preprocess_config(self):
        return {
            "use_label_col_as_discrete_metadata": True,  # Updated key name
            "group_labels": {"cols": ["group1", "group2"]},
        }

    @patch("synthefy_pkg.data.window_and_dataframe_utils.unscale_windows_dict")
    @patch("synthefy_pkg.data.window_and_dataframe_utils.json.load")
    @pytest.mark.asyncio
    async def test_convert_windows_basic(
        self,
        mock_json_load,
        mock_unscale,
        mock_data,
        mock_colnames,
        mock_dataset_path,
    ):
        # Setup mocks
        mock_json_load.return_value = mock_colnames
        mock_unscale.side_effect = (
            lambda windows_data_dict,
            window_type,
            dataset_name,
            original_discrete_windows: windows_data_dict
        )

        # Create a mock_open instance with some dummy content
        mock_file = mock_open(read_data=json.dumps(mock_colnames))

        with (
            patch.dict(
                "os.environ", {"SYNTHEFY_DATASETS_BASE": mock_dataset_path}
            ),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_file),
        ):
            # Call function
            df, _ = await convert_windows_to_dataframe(
                dataset_name="test_dataset", split="train", **mock_data
            )

            # Verify basic structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 200  # 10 batches * 20 windows
            assert "window_idx" in df.columns
            assert (
                len(df["window_idx"].unique()) == 10
            )  # Number of unique windows

            # Verify columns (excluding timestamp features)
            expected_columns = (
                mock_colnames["timeseries_colnames"]
                + mock_colnames["original_discrete_colnames"]
                + ["cont_1", "cont_2", "cont_3"]  # Filtered continuous columns
                + mock_colnames["timestamps_colnames"]
                + mock_colnames["original_text_colnames"]
                + ["window_idx"]
                + mock_colnames["group_label_cols"]
            )
            assert set(df.columns) == set(expected_columns)

    @patch("synthefy_pkg.data.window_and_dataframe_utils.unscale_windows_dict")
    @pytest.mark.asyncio
    async def test_convert_windows_no_optional_features(
        self, mock_unscale, mock_data, mock_dataset_path
    ):
        # Modified colnames with only required features
        minimal_colnames = {
            "timeseries_colnames": ["ts_1", "ts_2"],
            "continuous_colnames": [],
            "original_discrete_colnames": [],
            "original_text_colnames": [],
            "group_label_cols": [],
        }

        # Modify mock_data to only include required features
        minimal_data = {
            "timeseries": np.random.rand(5, 2, 10),  # (B, C, W)
            "discrete_original": np.zeros((5, 10, 0)),  # Empty discrete
            "continuous": np.zeros((5, 10, 0)),  # Empty continuous
            "timestamps_original": np.zeros((5, 0)),  # Empty timestamps
            "text": np.zeros((5, 10, 0)),  # Empty text
            "group_label_to_split_idx_dict": {"train": {}},
            "group_label_cols": [],
        }

        mock_unscale.side_effect = (
            lambda windows_data_dict,
            window_type,
            dataset_name,
            original_discrete_windows: windows_data_dict
        )

        with (
            patch.dict(
                "os.environ", {"SYNTHEFY_DATASETS_BASE": mock_dataset_path}
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                mock_open(read_data=json.dumps(minimal_colnames)),
            ),
        ):
            df, _ = await convert_windows_to_dataframe(
                dataset_name="test_dataset", split="train", **minimal_data
            )

            # Verify structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50  # 5 batches * 10 windows
            expected_columns = minimal_colnames["timeseries_colnames"] + [
                "window_idx"
            ]
            assert set(df.columns) == set(expected_columns)

    @patch("synthefy_pkg.data.window_and_dataframe_utils.unscale_windows_dict")
    @pytest.mark.asyncio
    async def test_convert_windows_with_timestamp_features(
        self,
        mock_unscale,
        mock_data,
        mock_colnames,
        mock_dataset_path,
    ):
        mock_unscale.side_effect = (
            lambda windows_data_dict,
            window_type,
            dataset_name,
            original_discrete_windows: windows_data_dict
        )

        with (
            patch.dict(
                "os.environ", {"SYNTHEFY_DATASETS_BASE": mock_dataset_path}
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open", mock_open(read_data=json.dumps(mock_colnames))
            ),
        ):
            df, _ = await convert_windows_to_dataframe(
                dataset_name="test_dataset", split="train", **mock_data
            )

            # Verify timestamp features are filtered out
            assert TIMESTAMPS_FEATURES_PREFIX not in df.columns
            assert "cont_1" in df.columns
            assert "cont_2" in df.columns
            assert "cont_3" in df.columns

    @patch("synthefy_pkg.data.window_and_dataframe_utils.load_dataset_files")
    @patch("synthefy_pkg.data.window_and_dataframe_utils.unscale_windows_dict")
    @pytest.mark.asyncio
    async def test_convert_windows_loads_files_when_no_data_provided(
        self,
        mock_unscale,
        mock_load_files,
        mock_data,
        mock_colnames,
        mock_dataset_path,
    ):
        mock_unscale.side_effect = (
            lambda windows_data_dict,
            window_type,
            dataset_name,
            original_discrete_windows: windows_data_dict
        )

        # Update mock_load_files to return colnames as well
        mock_load_files.return_value = (*tuple(mock_data.values()),)

        with (
            patch.dict(
                "os.environ", {"SYNTHEFY_DATASETS_BASE": mock_dataset_path}
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open", mock_open(read_data=json.dumps(mock_colnames))
            ),
        ):
            df, _ = await convert_windows_to_dataframe(
                dataset_name="test_dataset", split="train"
            )

            # Verify load_dataset_files was called
            mock_load_files.assert_called_once_with(
                dataset_name="test_dataset",
                split="train",
                timeseries=None,
                discrete_original=None,
                continuous=None,
                timestamps_original=None,
                text=None,
                group_label_to_split_idx_dict=None,
                group_label_cols=None,
            )

            # Verify the dataframe was created correctly
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 200  # 10 batches * 20 windows

    @patch("synthefy_pkg.data.window_and_dataframe_utils.unscale_windows_dict")
    @patch("synthefy_pkg.data.window_and_dataframe_utils.json.load")
    @pytest.mark.asyncio
    async def test_convert_windows_without_group_labels(
        self,
        mock_json_load,
        mock_unscale,
        mock_data,
        mock_colnames,
        mock_dataset_path,
    ):
        # remove group label cols from mock_colnames
        mock_colnames["group_label_cols"] = []
        mock_data["group_label_to_split_idx_dict"] = {"train": {}}
        mock_data["group_label_cols"] = []

        # Modify config to disable group labels
        config = {
            **mock_colnames,
            "use_label_col_as_discrete_metadata": False,
            "group_labels": {"cols": []},
        }

        mock_json_load.return_value = config
        mock_unscale.side_effect = (
            lambda windows_data_dict,
            window_type,
            dataset_name,
            original_discrete_windows: windows_data_dict
        )

        mock_file = mock_open(read_data=json.dumps(config))

        with (
            patch.dict(
                "os.environ", {"SYNTHEFY_DATASETS_BASE": mock_dataset_path}
            ),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_file),
        ):
            df, _ = await convert_windows_to_dataframe(
                dataset_name="test_dataset", split="train", **mock_data
            )

            # Verify group label columns are not included
            assert "group1" not in df.columns
            assert "group2" not in df.columns


if __name__ == "__main__":
    pytest.main([__file__])
