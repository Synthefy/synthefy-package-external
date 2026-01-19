import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
import yaml
from loguru import logger

from synthefy_pkg.preprocessing.fmv2_preprocess import (
    SCALING_FEATURES,
    TEXT_EMBEDDING_DIM,
    TIMESTAMPS_FEATURES,
)
from synthefy_pkg.preprocessing.fmv2_sharded_mix_and_split import (
    DataMixerAndSplitter,
    check_timestamp_leq,
    denormalize_timestamp_features,
)


class TestTimestampFunctions:
    @pytest.fixture
    def large_timestamp_array(self):
        """Fixture to create a large mock array with timestamps in a linear progression."""
        window_size = 40
        num_windows = 1000  # Large number of windows

        # Total feature dimension
        feature_dim = (
            len(SCALING_FEATURES)
            + window_size * len(TIMESTAMPS_FEATURES)
            + TEXT_EMBEDDING_DIM
            + window_size
            + 1
        )

        # Create mock data array
        mock_data = np.zeros((num_windows, feature_dim))

        # Create timestamp pattern: Linear progression from 2010-01-01 to 2022-12-31
        start_date = pd.Timestamp("2010-01-01")
        end_date = pd.Timestamp("2022-12-31")
        total_days = (end_date - start_date).days

        # Track expected timestamps for verification
        expected_timestamps = []

        # Feature indices mapping
        feature_indices = {
            name: i for i, name in enumerate(TIMESTAMPS_FEATURES)
        }

        # Starting index for the last timestamp features in each window
        start_feature_idx = len(SCALING_FEATURES) + (window_size - 1) * len(
            TIMESTAMPS_FEATURES
        )

        # Generate timestamps with linear progression
        for i in range(num_windows):
            # Calculate a timestamp that's linearly interpolated between start and end
            days_offset = int(i * total_days / num_windows)
            timestamp = start_date + pd.Timedelta(days=days_offset)
            expected_timestamps.append(timestamp)

            # Normalize year (1970-2030 range)
            mock_data[i, start_feature_idx + feature_indices["year"]] = (
                timestamp.year - 1970
            ) / (2030 - 1970)

            # Normalize month (1-12 range)
            mock_data[i, start_feature_idx + feature_indices["month"]] = (
                timestamp.month - 1
            ) / 11

            # Normalize day (1-31 range)
            mock_data[i, start_feature_idx + feature_indices["day"]] = (
                timestamp.day - 1
            ) / 30

            # Normalize hour (0-23 range)
            mock_data[i, start_feature_idx + feature_indices["hour"]] = (
                timestamp.hour / 23
            )

            # Normalize minute (0-59 range)
            mock_data[i, start_feature_idx + feature_indices["minute"]] = (
                timestamp.minute / 59
            )

            # Normalize second (0-59 range)
            mock_data[i, start_feature_idx + feature_indices["second"]] = (
                timestamp.second / 59
            )

            # Set other timestamp features
            mock_data[i, start_feature_idx + feature_indices["quarter"]] = (
                (timestamp.month - 1) // 3
            ) / 3
            mock_data[i, start_feature_idx + feature_indices["week"]] = (
                (timestamp.weekofyear - 1) / 52
                if hasattr(timestamp, "weekofyear")
                else timestamp.isocalendar()[1] / 53
            )
            mock_data[i, start_feature_idx + feature_indices["dayofweek"]] = (
                timestamp.dayofweek / 6
            )
            mock_data[i, start_feature_idx + feature_indices["holiday"]] = 0
            mock_data[i, start_feature_idx + feature_indices["timezone"]] = 0

        return mock_data, window_size, expected_timestamps

    def test_check_timestamp_leq_specific_cutoffs(self, large_timestamp_array):
        """Test check_timestamp_leq function with specific cutoff dates."""
        mock_data, window_size, expected_timestamps = large_timestamp_array

        cutoff_dates = [
            "2010-01-01",  # Should match only the first few windows
            "2016-06-15",  # Should match approximately half
            "2022-12-31",  # Should match all windows
        ]

        for cutoff_str in cutoff_dates:
            cutoff = pd.Timestamp(cutoff_str)

            # Test function execution
            result = check_timestamp_leq(mock_data, window_size, cutoff)

            # Calculate expected result
            expected_result = np.array(
                [ts <= cutoff for ts in expected_timestamps]
            )

            # Verify correctness
            assert np.array_equal(result, expected_result)

            # Additional statistics for debugging/info
            match_count = np.sum(result.astype(float))
            match_percent = match_count / len(result) * 100
            print(
                f"Cutoff {cutoff_str}: {match_count} windows ({match_percent:.2f}%) <= cutoff"
            )

    def test_check_timestamp_leq_midpoint(self, large_timestamp_array):
        """Test check_timestamp_leq function with a midpoint cutoff date."""
        mock_data, window_size, expected_timestamps = large_timestamp_array

        # Test with a mid-point cutoff that should match approximately half the windows
        mid_date_str = "2016-06-30"
        mid_date = pd.Timestamp(mid_date_str)
        mid_result = check_timestamp_leq(mock_data, window_size, mid_date)

        # The expected match count should be close to 50% of windows
        match_percent = np.sum(mid_result.astype(float)) / len(mid_result) * 100
        assert 45 <= match_percent <= 55, (
            f"Expected ~50% match rate, got {match_percent:.2f}%"
        )

    def test_check_timestamp_leq_random_cutoff(self, large_timestamp_array):
        """Test check_timestamp_leq function with a random cutoff date."""
        mock_data, window_size, expected_timestamps = large_timestamp_array

        # Choose a random timestamp from the middle of the array
        random_idx = np.random.randint(100, 900)
        random_idx = 800
        random_ts = expected_timestamps[random_idx]

        # Test with the random cutoff
        random_result = check_timestamp_leq(mock_data, window_size, random_ts)

        logger.info(f"Random cutoff: {random_ts}")
        logger.info(f"Random idx: {random_idx}")
        logger.info(f"Random result: {random_result[:100]}")

        # All windows before or at random_idx should be <= random_ts
        assert np.all(random_result[: random_idx + 1]), (
            "Windows before random cutoff should be True"
        )

        # All windows after random_idx should be > random_ts
        assert not np.any(random_result[random_idx + 1 :]), (
            "Windows after random cutoff should be False"
        )

    def test_check_timestamp_leq_performance(self, large_timestamp_array):
        """Test the performance of check_timestamp_leq with a large input array."""
        mock_data, window_size, expected_timestamps = large_timestamp_array

        # Define cutoff date for performance testing
        cutoff = pd.Timestamp("2016-06-15")

        # Measure performance
        start_time = time.time()
        check_timestamp_leq(mock_data, window_size, cutoff)
        elapsed = time.time() - start_time

        # Performance assertion - should complete in reasonable time
        # Adjust the threshold based on expected performance
        assert elapsed < 1.0, f"Function took too long: {elapsed:.4f} seconds"

        # Log performance information
        print(f"Processed {len(mock_data)} windows in {elapsed:.4f} seconds")


class TestDataMixerAndSplitter:
    @pytest.fixture
    def mock_dataset_structure(self):
        """Create a mock dataset structure for testing timestamp-based splitting."""
        # Create temp directory for test data
        test_dir = tempfile.mkdtemp()

        # Define window size and dataset structure
        window_size = 4
        num_scalars = 3

        # Create separate dataset directories
        dataset_dirs = [
            os.path.join(test_dir, "dataset_early_0"),
            os.path.join(test_dir, "dataset_mid_1"),
            os.path.join(test_dir, "dataset_late_2"),
        ]

        for dataset_dir in dataset_dirs:
            os.makedirs(dataset_dir, exist_ok=True)

        # Create mock data with different timestamp patterns for each dataset
        # Early dataset: timestamps from 2010-2015
        # Mid dataset: timestamps from 2015-2020
        # Late dataset: timestamps from 2020-2023
        date_ranges = [
            (pd.Timestamp("2010-01-01"), pd.Timestamp("2021-01-01")),
            (pd.Timestamp("1990-01-01"), pd.Timestamp("2022-01-01")),
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")),
        ]

        frequencies = ["weekly", "monthly", "daily"]

        # Track created files for cleanup
        created_files = []

        # For each dataset, create appropriate mockup data
        for i, (dataset_dir, (start_date, end_date), frequency) in enumerate(
            zip(dataset_dirs, date_ranges, frequencies)
        ):
            # the delta is based on the frequency
            if frequency == "daily":
                delta = pd.Timedelta(days=1)
            elif frequency == "weekly":
                delta = pd.Timedelta(weeks=1)
            elif frequency == "monthly":
                delta = pd.Timedelta(days=30)  # Approximate month length
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")

            # Calculate total duration and number of windows
            total_duration = end_date - start_date
            num_windows = int(total_duration / delta) - window_size

            # Calculate feature dimension
            feature_dim = (
                len(SCALING_FEATURES)
                + window_size * len(TIMESTAMPS_FEATURES)
                + 10
            )

            # Create timeseries data array
            timeseries = np.zeros((num_windows, feature_dim))

            # Fill in timestamp features for each window
            for win_idx in range(num_windows):
                timestamp = start_date + delta * win_idx
                # Linearly interpolate timestamps
                for ts in range(window_size):
                    # Add normalized timestamp features to the last position in each window
                    feature_indices = {
                        name: idx
                        for idx, name in enumerate(TIMESTAMPS_FEATURES)
                    }
                    start_feat_idx = len(SCALING_FEATURES) + ts * len(
                        TIMESTAMPS_FEATURES
                    )

                    # Normalize year
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["year"]
                    ] = (timestamp.year - 1970) / (2030 - 1970)

                    # Normalize month
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["month"]
                    ] = (timestamp.month - 1) / 11

                    # Normalize day
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["day"]
                    ] = (timestamp.day - 1) / 30

                    # Normalize other timestamp features
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["hour"]
                    ] = timestamp.hour / 23
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["minute"]
                    ] = timestamp.minute / 59
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["second"]
                    ] = timestamp.second / 59
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["quarter"]
                    ] = ((timestamp.month - 1) // 3) / 3
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["week"]
                    ] = (
                        (timestamp.weekofyear - 1) / 52
                        if hasattr(timestamp, "weekofyear")
                        else timestamp.isocalendar()[1] / 53
                    )
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["dayofweek"]
                    ] = timestamp.dayofweek / 6
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["holiday"]
                    ] = 0
                    timeseries[
                        win_idx, start_feat_idx + feature_indices["timezone"]
                    ] = 0

                    # Add some random data for the scalars
                    for j in range(num_scalars):
                        timeseries[win_idx, j] = np.random.randn()
                    timestamp = start_date + delta

            # Save preprocessed array
            timeseries_path = os.path.join(
                dataset_dir, "preprocessed_array.npy"
            )
            np.save(timeseries_path, timeseries)
            created_files.append(timeseries_path)

            # Create mock metadata
            dataset_name = os.path.basename(dataset_dir)
            metadata = {
                "columns": [{"title": f"Test Series {i}"}],
                "frequency": "daily",
                "length": num_windows,
            }

            metadata_path = os.path.join(
                dataset_dir, f"{dataset_name}_metadata.json"
            )
            with open(metadata_path, "w") as f:
                import json

                json.dump(metadata, f)
            created_files.append(metadata_path)

            # Create mock embedding
            embedding = np.random.randn(
                TEXT_EMBEDDING_DIM
            )  # Some random embedding
            embedding_path = os.path.join(
                dataset_dir, "description_embedding.npy"
            )
            np.save(embedding_path, embedding)
            created_files.append(embedding_path)

        # Create config file for mixer
        output_dir = os.path.join(test_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        config = {
            "input_base_dir": test_dir,
            "output_dir": output_dir,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "timestamp_split": "2018-01-01",  # This will put all early, some mid, and no late data in train
            "window_size": window_size,
            "num_scalars": num_scalars,
            "max_shard_size": 1000,
            "random_seed": 42,
            "selected_datasets": [
                {"dataset_name": "dataset_early_0", "usage": "pretrain"},
                {"dataset_name": "dataset_mid_1", "usage": "pretrain"},
                {"dataset_name": "dataset_late_2", "usage": "pretrain"},
            ],
        }

        config_path = os.path.join(test_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        created_files.append(config_path)

        yield test_dir, config_path, dataset_dirs

        # Cleanup
        shutil.rmtree(test_dir)

    def test_timestamp_based_split(self, mock_dataset_structure):
        """Test that timestamp-based splitting works correctly."""
        test_dir, config_path, dataset_dirs = mock_dataset_structure
        print(test_dir, config_path, dataset_dirs)
        # Initialize the mixer
        mixer = DataMixerAndSplitter(config_path=config_path)

        # Run the mix_and_split function
        total_rows, total_counts = mixer.mix_and_split("pretrain")

        # Verify output directories exist
        output_dir = os.path.join(
            os.path.dirname(config_path), "output", "pretrain"
        )
        assert os.path.exists(output_dir)

        # Load and check train data
        train_data_files = [
            f
            for f in os.listdir(output_dir)
            if f.startswith("train_timeseries") and f.endswith(".tar")
        ]
        print(
            total_rows,
            total_counts,
            len(train_data_files),
            [f for f in os.listdir(output_dir)],
        )
        assert len(train_data_files) > 0

        # Check that dataset_lookup.pkl exists
        assert os.path.exists(os.path.join(output_dir, "dataset_lookup.pkl"))

        # Load dataset indices to check distribution
        train_indices_file = os.path.join(
            output_dir, "train_dataset_indices_shard_0.json"
        )
        val_indices_file = os.path.join(
            output_dir, "val_dataset_indices_shard_0.json"
        )
        test_indices_file = os.path.join(
            output_dir, "test_dataset_indices_shard_0.json"
        )

        print(train_indices_file)
        assert os.path.exists(train_indices_file)
        assert os.path.exists(val_indices_file)
        assert os.path.exists(test_indices_file)

        # Optional: Check contents of split data if needed
        # This would require extracting and loading the tar files

        # Verify distribution makes sense for timestamp split
        # The 2018-01-01 cutoff should mean:
        # - dataset_early (2010-2015): all in train
        # - dataset_mid (2015-2020): some in train, some in val/test
        # - dataset_late (2020-2023): none in train, all in val/test
        assert total_counts["train"] > 0
        assert total_counts["val"] > 0
        assert total_counts["test"] > 0

        # Due to timestamp split, train should contain early data
        # and val/test should have more late data
        # Note: This is a basic check - would need to extract actual data for precise check
        assert total_counts["train"] > 0, (
            "Train set should not be empty with timestamp split"
        )

        # Load timestamps from original data to verify split by date
        timestamp_cutoff = pd.Timestamp("2018-01-01")
        window_size = 4

        # Create count tracking
        counts_by_dataset = {
            "early": {"train": 0, "val": 0, "test": 0, "total": 0},
            "mid": {"train": 0, "val": 0, "test": 0, "total": 0},
            "late": {"train": 0, "val": 0, "test": 0, "total": 0},
        }

        # Function to load and analyze original data (simplified)
        for dataset_dir, dataset_key in zip(
            dataset_dirs, ["early", "mid", "late"]
        ):
            # Load original data
            timeseries_path = os.path.join(
                dataset_dir, "preprocessed_array.npy"
            )
            timeseries = np.load(timeseries_path)

            # Check timestamps against cutoff
            window_timestamps = check_timestamp_leq(
                timeseries, window_size, timestamp_cutoff
            )

            # Count how many windows are before/after cutoff
            counts_by_dataset[dataset_key]["train"] = int(
                np.sum(window_timestamps.astype(float))
            )
            counts_by_dataset[dataset_key]["val_test"] = len(
                window_timestamps
            ) - int(np.sum(window_timestamps.astype(float)))
            counts_by_dataset[dataset_key]["total"] = len(window_timestamps)

        # Verify dataset_early has all/most windows in train
        assert (
            counts_by_dataset["early"]["train"]
            >= 0.9 * counts_by_dataset["early"]["total"]
        ), (
            "Early dataset should have ~100% of windows in train due to timestamp"
        )

        # Verify dataset_late has no/few windows in train
        assert (
            counts_by_dataset["late"]["train"]
            <= 0.1 * counts_by_dataset["late"]["total"]
        ), "Late dataset should have ~0% of windows in train due to timestamp"
