import math
import os
import pickle
import tempfile
from typing import Generator

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from synthefy_pkg.data.foundation_dataloader import (
    ShardedTimeSeriesDataset,
    StandardTimeSeriesDataset,
    collate_fn,
)


class TestStandardTimeSeriesDataset:
    @pytest.fixture
    def sample_data_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory with sample data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample timeseries data
            timeseries_data = np.random.randn(
                10, 1, 512
            )  # 10 samples, 512 window size
            np.save(os.path.join(temp_dir, "timeseries.npy"), timeseries_data)

            # Create sample metadata
            metadata = np.random.randn(
                10, 1, 5120
            )  # 10 samples, 8 metadata features
            np.save(os.path.join(temp_dir, "metadata.npy"), metadata)

            # Create sample scalers
            scalers = {
                "feature_1": StandardScaler(),
                "feature_2": StandardScaler(),
            }
            with open(os.path.join(temp_dir, "scalers.pkl"), "wb") as f:
                pickle.dump(scalers, f)

            yield temp_dir

    @pytest.fixture
    def sample_data_dir_no_metadata(self) -> Generator[str, None, None]:
        """Create a temporary directory with sample data files but no metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample timeseries data
            timeseries_data = np.random.randn(
                10, 1, 512
            )  # 10 samples, 512 window size
            np.save(os.path.join(temp_dir, "timeseries.npy"), timeseries_data)

            # Create sample scalers
            scalers = {
                "feature_1": StandardScaler(),
                "feature_2": StandardScaler(),
            }
            with open(os.path.join(temp_dir, "scalers.pkl"), "wb") as f:
                pickle.dump(scalers, f)

            yield temp_dir

    def test_init_with_valid_data(self, sample_data_dir: str) -> None:
        """Test initialization with valid data files."""
        dataset = StandardTimeSeriesDataset(data_dir=sample_data_dir)

        assert dataset.total_samples == 10
        assert dataset.timeseries_data.shape == (10, 1, 512)
        assert dataset.metadata_data.shape == (10, 1, 5120)
        assert isinstance(dataset.scalers, dict)
        assert len(dataset.scalers) == 2

    def test_init_without_metadata(
        self, sample_data_dir_no_metadata: str
    ) -> None:
        """Test initialization without metadata file."""
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            StandardTimeSeriesDataset(data_dir=sample_data_dir_no_metadata)

    def test_init_without_scalers(self, sample_data_dir: str) -> None:
        """Test initialization without loading scalers."""
        # The StandardTimeSeriesDataset doesn't have a load_scalers parameter
        # It always tries to load scalers and raises FileNotFoundError if not found
        with pytest.raises(FileNotFoundError, match="Scalers file not found"):
            # Create a temp dir with only timeseries and metadata but no scalers
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy timeseries and metadata from sample_data_dir
                timeseries_data = np.load(
                    os.path.join(sample_data_dir, "timeseries.npy")
                )
                np.save(
                    os.path.join(temp_dir, "timeseries.npy"), timeseries_data
                )

                metadata_data = np.load(
                    os.path.join(sample_data_dir, "metadata.npy")
                )
                np.save(os.path.join(temp_dir, "metadata.npy"), metadata_data)

                # Initialize without scalers file
                StandardTimeSeriesDataset(data_dir=temp_dir)

    def test_init_missing_timeseries(self) -> None:
        """Test initialization with missing timeseries file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(
                FileNotFoundError, match="Timeseries file not found"
            ):
                StandardTimeSeriesDataset(data_dir=temp_dir)

    def test_getitem_valid_index(self, sample_data_dir: str) -> None:
        """Test __getitem__ with valid index."""
        dataset = StandardTimeSeriesDataset(data_dir=sample_data_dir)
        item = dataset[0]

        assert isinstance(item, dict)
        assert "timeseries" in item
        assert "metadata" in item
        assert "scalers" in item

        assert isinstance(item["timeseries"], torch.Tensor)
        assert item["timeseries"].shape == (1, 512)
        assert item["timeseries"].dtype == torch.float32

        assert isinstance(item["metadata"], torch.Tensor)
        assert item["metadata"].shape == (
            1,
            5120,
        )  # Fixed shape to match metadata dimensions
        assert item["metadata"].dtype == torch.float32

        assert isinstance(item["scalers"], dict)

    def test_getitem_invalid_index(self, sample_data_dir: str) -> None:
        """Test __getitem__ with invalid index."""
        dataset = StandardTimeSeriesDataset(data_dir=sample_data_dir)

        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[10]  # Index out of bounds

        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[-1]  # Negative index

    def test_dataloader_batching(self, sample_data_dir: str) -> None:
        """Test that the dataset works correctly with DataLoader batching."""
        dataset = StandardTimeSeriesDataset(data_dir=sample_data_dir)
        batch_size = 3
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=StandardTimeSeriesDataset.collate_fn,
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Check batch structure and shapes
        assert isinstance(batch, dict)
        assert "timeseries" in batch
        assert "metadata" in batch
        assert "scalers" in batch

        # Check tensor shapes
        assert batch["timeseries"].shape == (batch_size, 1, 512)
        assert batch["metadata"].shape == (batch_size, 1, 5120)

        # Check that scalers are preserved
        assert isinstance(batch["scalers"], list)
        assert len(batch["scalers"]) == batch_size
        assert isinstance(batch["scalers"][0], dict)

    def test_dataloader_with_multiple_workers(
        self, sample_data_dir: str
    ) -> None:
        """Test that the dataset works correctly with multiple worker processes."""
        import multiprocessing
        import os
        import tempfile

        # Skip test if running on a platform that doesn't support multiprocessing
        if (
            not hasattr(multiprocessing, "get_start_method")
            or multiprocessing.get_start_method() == "fork"
        ):
            pass  # Continue with test
        else:
            pytest.skip("Multiprocessing not supported in this environment")

        dataset = StandardTimeSeriesDataset(data_dir=sample_data_dir)
        batch_size = 2
        num_workers = 2

        # Create a temporary directory to store worker IDs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define a worker initialization function that records the worker ID
            def init_fn(worker_id):
                # Create a file for each worker
                with open(
                    os.path.join(temp_dir, f"worker_{worker_id}.txt"), "w"
                ) as f:
                    f.write(f"Worker {worker_id} initialized")

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=StandardTimeSeriesDataset.collate_fn,
                worker_init_fn=init_fn,
            )

            # Process all batches
            all_samples = []
            for batch in dataloader:
                # Check batch structure
                assert isinstance(batch, dict)
                assert "timeseries" in batch
                assert "metadata" in batch
                assert "scalers" in batch

                # Check tensor shapes
                assert batch["timeseries"].shape[0] <= batch_size
                assert batch["timeseries"].shape[1] == 1
                assert batch["timeseries"].shape[2] == 512
                assert (
                    batch["metadata"].shape[0] == batch["timeseries"].shape[0]
                )
                assert batch["metadata"].shape[1] == 1
                assert batch["metadata"].shape[2] == 5120

                # Check scalers
                assert isinstance(batch["scalers"], list)
                assert len(batch["scalers"]) == batch["timeseries"].shape[0]

                # Track number of processed samples
                all_samples.extend([t for t in batch["timeseries"]])

            # Verify we processed all samples
            assert len(all_samples) == len(dataset)

            # Check if multiple worker files were created
            worker_files = [
                f for f in os.listdir(temp_dir) if f.startswith("worker_")
            ]
            assert len(worker_files) > 1, (
                f"Expected multiple workers, but only found {worker_files}"
            )

            # Verify that we have files for the expected worker IDs
            worker_ids = set(
                int(f.split("_")[1].split(".")[0]) for f in worker_files
            )
            assert worker_ids == set(range(num_workers)), (
                f"Expected workers {set(range(num_workers))}, got {worker_ids}"
            )

    def test_init_with_filename_prefix(self, sample_data_dir: str) -> None:
        """Test initialization with a filename prefix."""
        # Create files with prefixes
        timeseries_data = np.random.randn(10, 1, 512)
        np.save(
            os.path.join(sample_data_dir, "train_timeseries.npy"),
            timeseries_data,
        )

        metadata = np.random.randn(10, 1, 5120)
        np.save(os.path.join(sample_data_dir, "train_metadata.npy"), metadata)

        scalers = {"feature_1": StandardScaler(), "feature_2": StandardScaler()}
        with open(
            os.path.join(sample_data_dir, "train_scalers.pkl"), "wb"
        ) as f:
            pickle.dump(scalers, f)

        # Test initialization with prefix
        dataset = StandardTimeSeriesDataset(
            data_dir=sample_data_dir, filename_prefix="train"
        )

        assert dataset.total_samples == 10
        assert dataset.timeseries_data.shape == (10, 1, 512)
        assert dataset.metadata_data.shape == (10, 1, 5120)
        assert isinstance(dataset.scalers, dict)
        assert len(dataset.scalers) == 2

    def test_train_val_test_dataloaders(self, sample_data_dir: str) -> None:
        """Test creating train/val/test dataloaders."""
        # Create train files
        train_timeseries = np.random.randn(10, 1, 512)
        np.save(
            os.path.join(sample_data_dir, "train_timeseries.npy"),
            train_timeseries,
        )
        train_metadata = np.random.randn(10, 1, 5120)
        np.save(
            os.path.join(sample_data_dir, "train_metadata.npy"), train_metadata
        )
        train_scalers = {
            "feature_1": StandardScaler(),
            "feature_2": StandardScaler(),
        }
        with open(
            os.path.join(sample_data_dir, "train_scalers.pkl"), "wb"
        ) as f:
            pickle.dump(train_scalers, f)

        # Create val files
        val_timeseries = np.random.randn(5, 1, 512)
        np.save(
            os.path.join(sample_data_dir, "val_timeseries.npy"), val_timeseries
        )
        val_metadata = np.random.randn(5, 1, 5120)
        np.save(os.path.join(sample_data_dir, "val_metadata.npy"), val_metadata)
        val_scalers = {
            "feature_1": StandardScaler(),
            "feature_2": StandardScaler(),
        }
        with open(os.path.join(sample_data_dir, "val_scalers.pkl"), "wb") as f:
            pickle.dump(val_scalers, f)

        # Create test files
        test_timeseries = np.random.randn(3, 1, 512)
        np.save(
            os.path.join(sample_data_dir, "test_timeseries.npy"),
            test_timeseries,
        )
        test_metadata = np.random.randn(3, 1, 5120)
        np.save(
            os.path.join(sample_data_dir, "test_metadata.npy"), test_metadata
        )
        test_scalers = {
            "feature_1": StandardScaler(),
            "feature_2": StandardScaler(),
        }
        with open(os.path.join(sample_data_dir, "test_scalers.pkl"), "wb") as f:
            pickle.dump(test_scalers, f)

        # Create datasets with different prefixes
        train_dataset = StandardTimeSeriesDataset(
            data_dir=sample_data_dir, filename_prefix="train"
        )
        val_dataset = StandardTimeSeriesDataset(
            data_dir=sample_data_dir, filename_prefix="val"
        )
        test_dataset = StandardTimeSeriesDataset(
            data_dir=sample_data_dir, filename_prefix="test"
        )

        # Check dataset sizes
        assert len(train_dataset) == 10
        assert len(val_dataset) == 5
        assert len(test_dataset) == 3

        # Create dataloaders
        batch_size = 2
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=StandardTimeSeriesDataset.collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=StandardTimeSeriesDataset.collate_fn,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=StandardTimeSeriesDataset.collate_fn,
        )

        # Check number of batches
        assert len(list(train_dataloader)) == math.ceil(10 / batch_size)
        assert len(list(val_dataloader)) == math.ceil(5 / batch_size)
        assert len(list(test_dataloader)) == math.ceil(3 / batch_size)

        # Check a batch from each dataloader
        train_batch = next(iter(train_dataloader))
        assert train_batch["timeseries"].shape[0] == batch_size
        assert train_batch["metadata"].shape[0] == batch_size

        val_batch = next(iter(val_dataloader))
        assert val_batch["timeseries"].shape[0] == batch_size
        assert val_batch["metadata"].shape[0] == batch_size

        test_batch = next(iter(test_dataloader))
        assert test_batch["timeseries"].shape[0] == batch_size
        assert test_batch["metadata"].shape[0] == batch_size

    def test_collate_fn(self, sample_data_dir: str) -> None:
        """Test the collate_fn static method."""
        dataset = StandardTimeSeriesDataset(data_dir=sample_data_dir)

        # Create a mock batch of items
        batch = [dataset[i] for i in range(3)]

        # Call the collate_fn
        collated = StandardTimeSeriesDataset.collate_fn(batch)

        # Verify the structure and shapes
        assert isinstance(collated, dict)
        assert "timeseries" in collated
        assert "metadata" in collated
        assert "scalers" in collated

        # Check tensor shapes
        assert collated["timeseries"].shape == (3, 1, 512)
        assert collated["metadata"].shape == (3, 1, 5120)

        # Check that scalers are preserved
        assert isinstance(collated["scalers"], list)
        assert len(collated["scalers"]) == 3
        assert all(isinstance(s, dict) for s in collated["scalers"])

        # Test with verbose flag
        collated_verbose = StandardTimeSeriesDataset.collate_fn(
            batch, verbose=True
        )
        assert (
            collated_verbose["timeseries"].shape == collated["timeseries"].shape
        )


class TestShardedTimeSeriesDataset:
    @pytest.fixture
    def sample_sharded_data_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory with sample sharded data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample sharded timeseries data
            for shard_idx in range(2):
                # Create timeseries data shards
                samples_per_shard = 5
                timeseries_data = np.random.randn(samples_per_shard, 1, 512)
                np.save(
                    os.path.join(
                        temp_dir, f"train_timeseries_shard_{shard_idx}.npy"
                    ),
                    timeseries_data,
                )

                # Create metadata shards
                metadata = np.random.randn(samples_per_shard, 1, 128)
                np.save(
                    os.path.join(
                        temp_dir, f"train_metadata_shard_{shard_idx}.npy"
                    ),
                    metadata,
                )

                # Create scaler shards - each shard has its own scaler file
                scaler_list = [
                    {
                        "feature_1": StandardScaler(),
                        "feature_2": StandardScaler(),
                    }
                    for _ in range(samples_per_shard)
                ]
                # Save scalers with .pkl extension
                with open(
                    os.path.join(
                        temp_dir, f"train_scalers_shard_{shard_idx}.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(scaler_list, f)

            # Create validation shard
            val_timeseries = np.random.randn(3, 1, 512)
            np.save(
                os.path.join(temp_dir, "val_timeseries_shard_0.npy"),
                val_timeseries,
            )
            val_metadata = np.random.randn(3, 1, 128)
            np.save(
                os.path.join(temp_dir, "val_metadata_shard_0.npy"), val_metadata
            )
            # Create validation scalers shard
            val_scalers = [
                {"feature_1": StandardScaler(), "feature_2": StandardScaler()}
                for _ in range(3)
            ]
            with open(
                os.path.join(temp_dir, "val_scalers_shard_0.pkl"), "wb"
            ) as f:
                pickle.dump(val_scalers, f)

            # Create test shard
            test_timeseries = np.random.randn(2, 1, 512)
            np.save(
                os.path.join(temp_dir, "test_timeseries_shard_0.npy"),
                test_timeseries,
            )
            test_metadata = np.random.randn(2, 1, 128)
            np.save(
                os.path.join(temp_dir, "test_metadata_shard_0.npy"),
                test_metadata,
            )
            # Create test scalers shard
            test_scalers = [
                {"feature_1": StandardScaler(), "feature_2": StandardScaler()}
                for _ in range(2)
            ]
            with open(
                os.path.join(temp_dir, "test_scalers_shard_0.pkl"), "wb"
            ) as f:
                pickle.dump(test_scalers, f)

            yield temp_dir

    def test_init_with_valid_data(self, sample_sharded_data_dir: str) -> None:
        """Test initialization with valid sharded data files."""
        dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )

        # Should have 10 samples (5 per shard × 2 shards)
        assert dataset.total_samples == 10
        assert len(dataset.shard_files) == 2
        assert len(dataset.metadata_shard_files) == 2
        assert len(dataset.scaler_shard_files) == 2

    def test_init_with_different_splits(
        self, sample_sharded_data_dir: str
    ) -> None:
        """Test initialization with different data splits."""
        train_dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )
        val_dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="val",
        )
        test_dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="test",
        )

        assert train_dataset.total_samples == 10  # 5 samples × 2 shards
        assert val_dataset.total_samples == 3  # 3 samples × 1 shard
        assert test_dataset.total_samples == 2  # 2 samples × 1 shard

    def test_init_missing_scalers_shard(self) -> None:
        """Test initialization with missing scaler shard file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create timeseries and metadata but no scalers
            timeseries_data = np.random.randn(5, 1, 512)
            np.save(
                os.path.join(temp_dir, "train_timeseries_shard_0.npy"),
                timeseries_data,
            )

            metadata = np.random.randn(5, 1, 128)
            np.save(
                os.path.join(temp_dir, "train_metadata_shard_0.npy"),
                metadata,
            )

            with pytest.raises(
                FileNotFoundError, match="Scalers file not found"
            ):
                ShardedTimeSeriesDataset(
                    data_dir=temp_dir,
                    split="train",
                )

    def test_init_missing_shard_files(self) -> None:
        """Test initialization with missing shard files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match="No shard files found"):
                ShardedTimeSeriesDataset(
                    data_dir=temp_dir,
                    split="train",
                )

    def test_init_missing_metadata_files(self) -> None:
        """Test initialization with missing metadata files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only timeseries data but no metadata
            timeseries_data = np.random.randn(5, 1, 512)
            np.save(
                os.path.join(temp_dir, "train_timeseries_shard_0.npy"),
                timeseries_data,
            )

            with pytest.raises(
                FileNotFoundError, match="Metadata file not found"
            ):
                ShardedTimeSeriesDataset(
                    data_dir=temp_dir,
                    split="train",
                )

    def test_getitem_valid_index(self, sample_sharded_data_dir: str) -> None:
        """Test __getitem__ with valid index."""
        dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )

        # Get an item from the first shard
        item = dataset[2]

        assert isinstance(item, dict)
        assert "timeseries" in item
        assert "metadata" in item
        assert "scalers" in item

        assert isinstance(item["timeseries"], torch.Tensor)
        assert item["timeseries"].shape == (1, 512)
        assert item["timeseries"].dtype == torch.float32

        assert isinstance(item["metadata"], torch.Tensor)
        assert item["metadata"].shape == (1, 128)
        assert item["metadata"].dtype == torch.float32

        assert isinstance(item["scalers"], dict)

        # Get an item from the second shard
        item = dataset[7]

        assert isinstance(item, dict)
        assert isinstance(item["timeseries"], torch.Tensor)
        assert item["timeseries"].shape == (1, 512)
        assert isinstance(item["metadata"], torch.Tensor)
        assert item["metadata"].shape == (1, 128)
        assert isinstance(item["scalers"], dict)

    def test_getitem_invalid_index(self, sample_sharded_data_dir: str) -> None:
        """Test __getitem__ with invalid index."""
        dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )

        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[10]  # Index out of bounds

        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[-1]  # Negative index

    def test_dataloader_batching(self, sample_sharded_data_dir: str) -> None:
        """Test that the dataset works correctly with DataLoader batching."""
        dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )

        batch_size = 3
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,  # Reuse the collate_fn
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Check batch structure and shapes
        assert isinstance(batch, dict)
        assert "timeseries" in batch
        assert "metadata" in batch
        assert "scalers" in batch

        # Check tensor shapes
        assert batch["timeseries"].shape == (batch_size, 1, 512)
        assert batch["metadata"].shape == (batch_size, 1, 128)

        # Check that scalers are preserved
        assert isinstance(batch["scalers"], list)
        assert len(batch["scalers"]) == batch_size
        assert all(isinstance(s, dict) for s in batch["scalers"])

    def test_cross_shard_batching(self, sample_sharded_data_dir: str) -> None:
        """Test that batches can span across multiple shards."""
        dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )

        # Create a dataloader with a batch size that will span shards
        # First shard has indices 0-4, second has 5-9
        batch_size = 6
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Important: no shuffle to ensure we get samples in order
            collate_fn=collate_fn,
        )

        # Get the first batch which should span both shards
        batch = next(iter(dataloader))

        # Check batch size
        assert batch["timeseries"].shape[0] == batch_size
        assert batch["metadata"].shape[0] == batch_size
        assert len(batch["scalers"]) == batch_size
        assert all(isinstance(s, dict) for s in batch["scalers"])

    def test_all_splits_with_dataloaders(
        self, sample_sharded_data_dir: str
    ) -> None:
        """Test creating dataloaders for all splits."""
        train_dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )
        val_dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="val",
        )
        test_dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="test",
        )

        batch_size = 2

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Check number of batches
        assert len(list(train_dataloader)) == math.ceil(10 / batch_size)
        assert len(list(val_dataloader)) == math.ceil(3 / batch_size)
        assert len(list(test_dataloader)) == math.ceil(2 / batch_size)

        # Check a batch from each dataloader
        train_batch = next(iter(train_dataloader))
        assert train_batch["timeseries"].shape[0] == batch_size
        assert train_batch["metadata"].shape[0] == batch_size
        assert len(train_batch["scalers"]) == batch_size
        assert all(isinstance(s, dict) for s in train_batch["scalers"])

        val_batch = next(iter(val_dataloader))
        assert val_batch["timeseries"].shape[0] == batch_size
        assert val_batch["metadata"].shape[0] == batch_size
        assert len(val_batch["scalers"]) == batch_size
        assert all(isinstance(s, dict) for s in val_batch["scalers"])

        test_batch = next(iter(test_dataloader))
        assert test_batch["timeseries"].shape[0] == batch_size
        assert test_batch["metadata"].shape[0] == batch_size
        assert len(test_batch["scalers"]) == batch_size
        assert all(isinstance(s, dict) for s in test_batch["scalers"])

    def test_sharded_scaler_loading(self, sample_sharded_data_dir: str) -> None:
        """Test that scalers are properly loaded from sharded files."""
        dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="train",
        )

        # Test loading from first shard (index 0-4)
        first_shard_item = dataset[2]
        assert isinstance(first_shard_item["scalers"], dict)
        assert "feature_1" in first_shard_item["scalers"]
        assert "feature_2" in first_shard_item["scalers"]
        assert isinstance(
            first_shard_item["scalers"]["feature_1"], StandardScaler
        )

        # Test loading from second shard (index 5-9)
        second_shard_item = dataset[7]
        assert isinstance(second_shard_item["scalers"], dict)
        assert "feature_1" in second_shard_item["scalers"]
        assert "feature_2" in second_shard_item["scalers"]
        assert isinstance(
            second_shard_item["scalers"]["feature_1"], StandardScaler
        )

        # Ensure getting the same index multiple times works correctly
        repeat_item = dataset[2]
        assert isinstance(repeat_item["scalers"], dict)
        assert "feature_1" in repeat_item["scalers"]

        # Test with different splits
        val_dataset = ShardedTimeSeriesDataset(
            data_dir=sample_sharded_data_dir,
            split="val",
        )
        val_item = val_dataset[1]
        assert isinstance(val_item["scalers"], dict)
        assert "feature_1" in val_item["scalers"]
        assert "feature_2" in val_item["scalers"]
