import json
import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from synthefy_pkg.preprocessing.fm_mix_and_split import (
    DataMixerAndSplitter,
    load_preprocessed_data,
    save_batch,
    shuffle_and_split_data,
)


class TestLoadPreprocessedData:
    def setup_method(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create test data
        self.test_timeseries = np.array([[1, 2, 3], [4, 5, 6]])
        self.test_metadata = np.array([{"a": 1}, {"b": 2}], dtype=object)
        self.test_scalers = [{"scaler1": "data"}, {"scaler2": "data"}]

        # Save test data to the temporary directory
        np.save(
            os.path.join(self.temp_dir.name, "timeseries.npy"),
            self.test_timeseries,
        )
        np.save(
            os.path.join(self.temp_dir.name, "metadata.npy"), self.test_metadata
        )
        with open(os.path.join(self.temp_dir.name, "scalers.pkl"), "wb") as f:
            pickle.dump(self.test_scalers, f)

    def teardown_method(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_load_preprocessed_data_success(self):
        # Test successful loading of data
        timeseries, metadata, scalers = load_preprocessed_data(
            self.temp_dir.name
        )

        assert np.array_equal(timeseries, self.test_timeseries)
        assert len(metadata) == len(self.test_metadata)
        assert scalers == self.test_scalers

    def test_load_preprocessed_data_missing_files(self):
        # Test error when files are missing
        # Create a new empty directory
        empty_dir = tempfile.TemporaryDirectory()

        with pytest.raises(FileNotFoundError) as excinfo:
            load_preprocessed_data(empty_dir.name)

        assert "Missing required files" in str(excinfo.value)
        empty_dir.cleanup()

    def test_load_preprocessed_data_corrupt_files(self):
        # Test error when files are corrupt
        corrupt_dir = tempfile.TemporaryDirectory()

        # Create corrupt files
        with open(os.path.join(corrupt_dir.name, "timeseries.npy"), "w") as f:
            f.write("not a numpy file")
        with open(os.path.join(corrupt_dir.name, "metadata.npy"), "w") as f:
            f.write("not a numpy file")
        with open(os.path.join(corrupt_dir.name, "scalers.pkl"), "w") as f:
            f.write("not a pickle file")

        with pytest.raises(FileNotFoundError) as excinfo:
            load_preprocessed_data(corrupt_dir.name)

        assert "Error loading data" in str(excinfo.value)
        corrupt_dir.cleanup()


class TestSaveBatch:
    def test_save_batch(self):
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test data
            timeseries = np.array([[1, 2, 3], [4, 5, 6]])
            metadata = np.array([{"a": 1}, {"b": 2}], dtype=object)

            # Call the function
            save_batch(temp_dir, "train", timeseries, metadata, 1)

            # Check that files were created
            assert os.path.exists(
                os.path.join(temp_dir, "train_timeseries_batch_1.npy")
            )
            assert os.path.exists(
                os.path.join(temp_dir, "train_metadata_batch_1.npy")
            )

            # Load the saved files and verify contents
            saved_timeseries = np.load(
                os.path.join(temp_dir, "train_timeseries_batch_1.npy"),
                allow_pickle=True,
            )
            saved_metadata = np.load(
                os.path.join(temp_dir, "train_metadata_batch_1.npy"),
                allow_pickle=True,
            )

            assert np.array_equal(saved_timeseries, timeseries)
            assert len(saved_metadata) == len(metadata)


class TestShuffleAndSplitData:
    def setup_method(self):
        # Create test data
        self.timeseries = np.array([[i, i + 1, i + 2] for i in range(100)])
        self.metadata = np.array([{"id": i} for i in range(100)], dtype=object)
        self.scalers = [{"scaler": f"data_{i}"} for i in range(100)]

    def test_shuffle_and_split_data_ratios(self):
        # Test with specific ratios
        train_ratio = 0.7
        val_ratio = 0.2

        train_data, val_data, test_data, shuffle_mapping = (
            shuffle_and_split_data(
                self.timeseries,
                self.metadata,
                self.scalers,
                train_ratio,
                val_ratio,
                random_seed=42,
            )
        )

        # Check split sizes
        assert len(train_data[0]) == int(100 * train_ratio)  # 70 samples
        assert len(val_data[0]) == int(100 * val_ratio)  # 20 samples
        assert len(test_data[0]) == 100 - len(train_data[0]) - len(
            val_data[0]
        )  # 10 samples

        # Check that all data is preserved (just reordered)
        all_timeseries = np.concatenate(
            [train_data[0], val_data[0], test_data[0]], axis=0
        )
        assert len(all_timeseries) == len(self.timeseries)

        # do the same for metadata:
        all_metadata = np.concatenate(
            [train_data[1], val_data[1], test_data[1]], axis=0
        )
        assert len(all_metadata) == len(self.metadata)

        # Check that scalers are properly split
        assert len(train_data[2]) == len(train_data[0])
        assert len(val_data[2]) == len(val_data[0])
        assert len(test_data[2]) == len(test_data[0])

        # Check that shuffle mapping was returned and has the correct structure
        assert shuffle_mapping is not None
        assert "train" in shuffle_mapping
        assert "val" in shuffle_mapping
        assert "test" in shuffle_mapping
        assert len(shuffle_mapping["train"]) == len(train_data[0])
        assert len(shuffle_mapping["val"]) == len(val_data[0])
        assert len(shuffle_mapping["test"]) == len(test_data[0])

    def test_no_shuffle_split_data(self):
        # Test with shuffle=False
        train_ratio = 0.7
        val_ratio = 0.2

        train_data, val_data, test_data, shuffle_mapping = (
            shuffle_and_split_data(
                self.timeseries,
                self.metadata,
                self.scalers,
                train_ratio,
                val_ratio,
                random_seed=42,
                shuffle=False,
            )
        )

        # Check split sizes
        assert len(train_data[0]) == int(100 * train_ratio)  # 70 samples
        assert len(val_data[0]) == int(100 * val_ratio)  # 20 samples
        assert len(test_data[0]) == 100 - len(train_data[0]) - len(
            val_data[0]
        )  # 10 samples

        # Check that data order is preserved (not shuffled)
        assert np.array_equal(train_data[0], self.timeseries[:70])
        assert np.array_equal(val_data[0], self.timeseries[70:90])
        assert np.array_equal(test_data[0], self.timeseries[90:])

        # Check metadata order is preserved
        assert np.array_equal(train_data[1], self.metadata[:70])
        assert np.array_equal(val_data[1], self.metadata[70:90])
        assert np.array_equal(test_data[1], self.metadata[90:])

        # Check scalers order is preserved
        assert train_data[2] == self.scalers[:70]
        assert val_data[2] == self.scalers[70:90]
        assert test_data[2] == self.scalers[90:]

        # Check that shuffle_mapping is None when shuffle=False
        assert shuffle_mapping is None

    def test_shuffle_and_split_data_reproducibility(self):
        # Test that using the same seed produces the same splits
        train_ratio = 0.8
        val_ratio = 0.1
        random_seed = 42

        # First run
        train_data1, val_data1, test_data1, shuffle_mapping1 = (
            shuffle_and_split_data(
                self.timeseries,
                self.metadata,
                self.scalers,
                train_ratio,
                val_ratio,
                random_seed=random_seed,
            )
        )

        # Second run with same parameters
        train_data2, val_data2, test_data2, shuffle_mapping2 = (
            shuffle_and_split_data(
                self.timeseries,
                self.metadata,
                self.scalers,
                train_ratio,
                val_ratio,
                random_seed=random_seed,
            )
        )

        # Check that train splits are identical
        assert np.array_equal(train_data1[0], train_data2[0])  # timeseries
        assert np.array_equal(train_data1[1], train_data2[1])  # metadata
        assert train_data1[2] == train_data2[2]  # scalers

        # Check that validation splits are identical
        assert np.array_equal(val_data1[0], val_data2[0])  # timeseries
        assert np.array_equal(val_data1[1], val_data2[1])  # metadata
        assert val_data1[2] == val_data2[2]  # scalers

        # Check that test splits are identical
        assert np.array_equal(test_data1[0], test_data2[0])  # timeseries
        assert np.array_equal(test_data1[1], test_data2[1])  # metadata
        assert test_data1[2] == test_data2[2]  # scalers

        # Check that shuffle mappings are identical
        assert shuffle_mapping1 is not None
        assert shuffle_mapping2 is not None
        for split in ["train", "val", "test"]:
            assert split in shuffle_mapping1
            assert split in shuffle_mapping2
            assert shuffle_mapping1[split] == shuffle_mapping2[split]


class TestDataMixerAndSplitter:
    def setup_method(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_base_dir = os.path.join(self.temp_dir.name, "input")
        self.output_dir = os.path.join(self.temp_dir.name, "output")

        # Create directories
        os.makedirs(self.input_base_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create test datasets
        self.create_test_dataset("dataset1", 50)
        self.create_test_dataset("dataset2", 30)

        # Create config file
        self.config_path = os.path.join(self.temp_dir.name, "config.yaml")
        config = {
            "input_base_dir": self.input_base_dir,
            "output_dir": self.output_dir,
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "random_seed": 42,
            "shuffle": True,
            "selected_datasets": [
                {"dataset_name": "dataset1", "usage": "pretrain"},
                {"dataset_name": "dataset2", "usage": "blind"},
            ],
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def teardown_method(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def create_test_dataset(self, name, n_samples):
        dataset_dir = os.path.join(self.input_base_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Create test data
        timeseries = np.array([[i, i + 1, i + 2] for i in range(n_samples)])
        metadata = np.array([{"id": i} for i in range(n_samples)], dtype=object)
        scalers = [{"scaler": f"data_{i}"} for i in range(n_samples)]

        # Save test data
        np.save(os.path.join(dataset_dir, "timeseries.npy"), timeseries)
        np.save(os.path.join(dataset_dir, "metadata.npy"), metadata)
        with open(os.path.join(dataset_dir, "scalers.pkl"), "wb") as f:
            pickle.dump(scalers, f)

    @patch("synthefy_pkg.preprocessing.fm_mix_and_split.logger")
    def test_mix_and_split_pretrain(self, mock_logger):
        # Test mixing and splitting for pretrain set
        mixer = DataMixerAndSplitter(config_path=self.config_path)
        mixer.mix_and_split(dataset_set="pretrain")

        # Check that output files were created
        pretrain_dir = os.path.join(self.output_dir, "pretrain")
        assert os.path.exists(
            os.path.join(pretrain_dir, "train_timeseries.npy")
        )
        assert os.path.exists(os.path.join(pretrain_dir, "val_timeseries.npy"))
        assert os.path.exists(os.path.join(pretrain_dir, "test_timeseries.npy"))
        assert os.path.exists(os.path.join(pretrain_dir, "train_metadata.npy"))
        assert os.path.exists(
            os.path.join(pretrain_dir, "train_dataset_indices.json")
        )
        assert os.path.exists(
            os.path.join(pretrain_dir, "val_dataset_indices.json")
        )
        assert os.path.exists(
            os.path.join(pretrain_dir, "test_dataset_indices.json")
        )
        assert os.path.exists(os.path.join(pretrain_dir, "manifest.yaml"))

        # Check that shuffle indices were saved for each split
        assert os.path.exists(
            os.path.join(pretrain_dir, "train_shuffle_indices.json")
        )
        assert os.path.exists(
            os.path.join(pretrain_dir, "val_shuffle_indices.json")
        )
        assert os.path.exists(
            os.path.join(pretrain_dir, "test_shuffle_indices.json")
        )

        # Check that the data was split correctly
        train_data = np.load(os.path.join(pretrain_dir, "train_timeseries.npy"))
        val_data = np.load(os.path.join(pretrain_dir, "val_timeseries.npy"))
        test_data = np.load(os.path.join(pretrain_dir, "test_timeseries.npy"))

        # Dataset1 has 50 samples, with 0.7/0.2/0.1 split should be ~35/10/5
        assert len(train_data) == int(50 * 0.7)
        assert len(val_data) == int(50 * 0.2)
        assert len(test_data) == 50 - len(train_data) - len(val_data)

        # Check that shuffle indices file contains the expected dataset
        with open(
            os.path.join(pretrain_dir, "train_shuffle_indices.json"), "r"
        ) as f:
            train_shuffle_indices = json.load(f)
        assert "dataset1" in train_shuffle_indices
        assert len(train_shuffle_indices["dataset1"]) == int(
            50 * 0.7
        )  # Should match train split size

    @patch("synthefy_pkg.preprocessing.fm_mix_and_split.logger")
    def test_mix_and_split_blind(self, mock_logger):
        # Test mixing and splitting for blind set
        mixer = DataMixerAndSplitter(config_path=self.config_path)
        mixer.mix_and_split(dataset_set="blind")

        # Check that output files were created
        blind_dir = os.path.join(self.output_dir, "blind")
        assert os.path.exists(os.path.join(blind_dir, "train_timeseries.npy"))
        assert os.path.exists(os.path.join(blind_dir, "val_timeseries.npy"))
        assert os.path.exists(os.path.join(blind_dir, "test_timeseries.npy"))

        # Check that shuffle indices were saved for each split
        assert os.path.exists(
            os.path.join(blind_dir, "train_shuffle_indices.json")
        )
        assert os.path.exists(
            os.path.join(blind_dir, "val_shuffle_indices.json")
        )
        assert os.path.exists(
            os.path.join(blind_dir, "test_shuffle_indices.json")
        )

        # Check that the data was split correctly
        train_data = np.load(os.path.join(blind_dir, "train_timeseries.npy"))
        val_data = np.load(os.path.join(blind_dir, "val_timeseries.npy"))
        test_data = np.load(os.path.join(blind_dir, "test_timeseries.npy"))

        # Dataset2 has 30 samples, with 0.7/0.2/0.1 split should be ~21/6/3
        assert len(train_data) == int(30 * 0.7)
        assert len(val_data) == int(30 * 0.2)
        assert len(test_data) == 30 - len(train_data) - len(val_data)

        # Check that shuffle indices file contains the expected dataset
        with open(
            os.path.join(blind_dir, "train_shuffle_indices.json"), "r"
        ) as f:
            train_shuffle_indices = json.load(f)
        assert "dataset2" in train_shuffle_indices
        assert len(train_shuffle_indices["dataset2"]) == int(
            30 * 0.7
        )  # Should match train split size

    @patch("synthefy_pkg.preprocessing.fm_mix_and_split.logger")
    def test_mix_and_split_no_shuffle(self, mock_logger):
        # Test mixing and splitting with shuffle=False
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml"
        ) as temp_config:
            # Create config with shuffle=False
            no_shuffle_config = {
                "input_base_dir": self.input_base_dir,
                "output_dir": self.output_dir,
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "random_seed": 42,
                "shuffle": False,
                "selected_datasets": [
                    {"dataset_name": "dataset1", "usage": "pretrain"},
                ],
            }
            yaml.dump(no_shuffle_config, temp_config)
            temp_config.flush()

            mixer = DataMixerAndSplitter(config_path=temp_config.name)
            mixer.mix_and_split(dataset_set="pretrain")

            # Check that output directory was created
            pretrain_dir = os.path.join(self.output_dir, "pretrain")

            # Check if shuffle indices files exist (they should be empty or not exist)
            for split in ["train", "val", "test"]:
                shuffle_indices_path = os.path.join(
                    pretrain_dir, f"{split}_shuffle_indices.json"
                )
                if os.path.exists(shuffle_indices_path):
                    with open(shuffle_indices_path, "r") as f:
                        shuffle_indices = json.load(f)
                    # Should be an empty dict since no shuffling was done
                    assert shuffle_indices == {}

    def test_init_with_invalid_ratios(self):
        # Test initialization with invalid ratios
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml"
        ) as temp_config:
            # Create config with invalid ratios
            invalid_config = {
                "input_base_dir": self.input_base_dir,
                "output_dir": self.output_dir,
                "train_ratio": 0.9,
                "val_ratio": 0.2,  # This makes train + val > 1
            }
            yaml.dump(invalid_config, temp_config)
            temp_config.flush()

            # Should raise an assertion error
            with pytest.raises(AssertionError):
                DataMixerAndSplitter(config_path=temp_config.name)

    def test_validate_config_dataset_usage_conflict(self):
        # Test validation that prevents a dataset from being used in both pretrain and other categories
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml"
        ) as temp_config:
            # Create config with the same dataset used for both pretrain and blind
            invalid_config = {
                "input_base_dir": self.input_base_dir,
                "output_dir": self.output_dir,
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "selected_datasets": [
                    {"dataset_name": "dataset1", "usage": "pretrain"},
                    {
                        "dataset_name": "dataset1",
                        "usage": "blind",
                    },  # Same dataset used for blind
                ],
            }
            yaml.dump(invalid_config, temp_config)
            temp_config.flush()

            # Should raise a ValueError
            with pytest.raises(ValueError) as excinfo:
                DataMixerAndSplitter(config_path=temp_config.name)

            # Check error message
            assert (
                "Dataset dataset1 is specified as pretrain and dataset1 is specified as blind"
                in str(excinfo.value)
            )

    def test_validate_config_override_check(self):
        # Test that the override flag allows conflicting dataset usage
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml"
        ) as temp_config:
            # Create config with the same dataset used for both pretrain and blind
            config = {
                "input_base_dir": self.input_base_dir,
                "output_dir": self.output_dir,
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "selected_datasets": [
                    {"dataset_name": "dataset1", "usage": "pretrain"},
                    {
                        "dataset_name": "dataset1",
                        "usage": "blind",
                    },  # Same dataset used for blind
                ],
            }
            yaml.dump(config, temp_config)
            temp_config.flush()

            # Should not raise an error when override is True
            mixer = DataMixerAndSplitter(
                config_path=temp_config.name,
                allow_pretrain_blind_overlap=True,
            )

            # Verify the config was loaded correctly
            assert len(mixer.config["selected_datasets"]) == 2
