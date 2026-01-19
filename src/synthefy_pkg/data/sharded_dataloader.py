import io
import os
import pickle
import tarfile
from multiprocessing import Pool, cpu_count
from typing import List, Literal, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.configs.relational_config import SubDatasetConfig
from synthefy_pkg.data.data_gen_base import SynthesisBaseDataModule
from synthefy_pkg.data.shard_utils import ShardCache, ShardLookupTable


class ShardedDatasetV1(Dataset):
    def __init__(
        self,
        shards: List[str],
        num_correlates: int,
        relation_shards: Optional[List[str]] = None,
        shard_counts_path: Optional[str] = None,
        shard_cache_size: Optional[int] = None,
        relational_epsilon_rate: float = 0.2,
    ):
        self.shards = shards

        if relation_shards is not None:
            self.relation_shards = relation_shards
            self.use_relational_sampling = True
            self.relational_epsilon_rate = relational_epsilon_rate
        else:
            self.relation_shards = []
            self.use_relational_sampling = False
            self.relational_epsilon_rate = 0.0

        self.num_correlates = num_correlates

        shard_counts_path = shard_counts_path

        self.shard_lookup_table = ShardLookupTable(shards, shard_counts_path)

        self.shard_cache_size = (
            shard_cache_size
            if shard_cache_size is not None
            else len(self.shards)
        )
        self.shard_cache = ShardCache(self.shard_cache_size)

        if self.use_relational_sampling:
            self.relation_shard_lookup_table = ShardLookupTable(
                self.relation_shards, None
            )
            assert np.array_equal(
                self.relation_shard_lookup_table.cumulative_counts,
                self.shard_lookup_table.cumulative_counts,
            ), (
                f"relation shard lookup table and shard lookup table must have the same cumulative counts. Relation: {self.relation_shard_lookup_table.cumulative_counts} Shard: {self.shard_lookup_table.cumulative_counts}"
            )
            self.relation_shard_cache_size = (
                shard_cache_size
                if shard_cache_size is not None
                else len(self.relation_shards)
            )
            self.relation_shard_cache = ShardCache(
                self.relation_shard_cache_size
            )

    def __len__(self):
        return len(self.shard_lookup_table)

    def __getitem__(self, index: int):
        if self.use_relational_sampling:
            # lookup shard and local index for the relational data
            relation_shard_index, relation_local_index = (
                self.relation_shard_lookup_table.get_shard_and_local_index(
                    index
                )
            )
            shard_index, local_index = (
                self.shard_lookup_table.get_shard_and_local_index(index)
            )
            # print("retrieving", index, relation_shard_index, relation_local_index, len(self.relation_shards), "from dataset", shard_index, local_index, len(self.shards))
            relation_shard_path = self.relation_shards[relation_shard_index]

            # get the cached tarfile and members list
            relation_tar, relation_members = (
                self.relation_shard_cache.get_tarfile(relation_shard_path)
            )
            relation_member = relation_members[relation_local_index]

            # Get the actual numpy array data
            f = relation_tar.extractfile(relation_member)
            if f is None:
                raise ValueError(
                    f"Failed to extract file {relation_member.name} from tar"
                )
            relation_array_data = np.load(io.BytesIO(f.read()))
            relation_array_data = relation_array_data[
                0
            ]  # TODO: removes extra dim, not sure why that dim is there

            # Randomly select num_correlates from the relation array
            # TODO: Make this more efficient
            selected_indices = np.random.choice(
                relation_array_data,
                self.num_correlates - 1,
                replace=False,
            )
            selected_indices = [index] + selected_indices.tolist()

            indexes = []
            shard_indexes = []
            local_indexes = []

            output_timeseries = []
            for correlate_index in selected_indices:
                # Randomly select whether to use the relational data or not
                if np.random.rand() < self.relational_epsilon_rate:
                    # the "correlate" is replaced with a randomly selected shard
                    correlate_index = np.random.randint(
                        0, len(self.shard_lookup_table)
                    )

                # Lookup which shard and local index the sample is in
                shard_index, local_index = (
                    self.shard_lookup_table.get_shard_and_local_index(
                        correlate_index
                    )
                )
                shard_path = self.shards[shard_index]

                # Get the cached tarfile and members list
                tar, members = self.shard_cache.get_tarfile(shard_path)
                member = members[local_index]

                # Get the actual numpy array data
                f = tar.extractfile(member)
                if f is None:
                    raise ValueError(
                        f"Failed to extract file {member.name} from tar"
                    )
                array_data = np.load(io.BytesIO(f.read()))
                array_data = array_data[
                    0
                ]  # TODO: removes extra dim, not sure why that dim is there
                output_timeseries.append(torch.tensor(array_data))

                indexes.append(correlate_index)
                shard_indexes.append(shard_index)
                local_indexes.append(local_index)

            return (
                torch.stack(output_timeseries),
                torch.tensor(indexes),
                torch.tensor(shard_indexes),
                torch.tensor(local_indexes),
            )

        else:
            # Lookup which shard and local index the sample is in
            shard_index, local_index = (
                self.shard_lookup_table.get_shard_and_local_index(index)
            )
            shard_path = self.shards[shard_index]

            # Get the cached tarfile and members list
            tar, members = self.shard_cache.get_tarfile(shard_path)
            member = members[local_index]

            # Get the actual numpy array data
            f = tar.extractfile(member)
            if f is None:
                raise ValueError(
                    f"Failed to extract file {member.name} from tar"
                )
            array_data = np.load(io.BytesIO(f.read()))
            return torch.tensor(array_data), index, shard_index, local_index


class ShardedDataloaderV1(SynthesisBaseDataModule):
    def __init__(
        self,
        config: Configuration | SubDatasetConfig,
        name_specification: str = "_timeseries",
        data_dir: str = "",
    ):
        """
        Initialize the ShardedDataloaderV1.

        Args:
            config: Configuration object containing dataset settings
        """
        logger.info("Initializing ShardedDataloaderV1")

        # Join the dataset base path with the dataset name to get the full data directory
        if len(data_dir) == 0:
            if os.path.isabs(config.dataset_config.dataset_name):
                data_dir = config.dataset_config.dataset_name
            else:
                data_dir = os.path.join(
                    str(os.getenv("SYNTHEFY_DATASETS_BASE")),
                    config.dataset_config.dataset_name,
                )

        logger.info("Loading train dataset")
        train_shards = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith(f"train{name_specification}") and f.endswith(".tar")
        ]
        train_shards.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        # Get relation shards for training data
        if config.dataset_config.use_relation_shards:
            train_relation_shards = []
            relation_shards_dir = os.path.join(data_dir, "relation_shards")
            if os.path.exists(relation_shards_dir):
                train_relation_shards = [
                    os.path.join(relation_shards_dir, f)
                    for f in os.listdir(relation_shards_dir)
                    if f.startswith("train_window") and f.endswith(".tar")
                ]
                logger.info(
                    f"Found {len(train_relation_shards)} relation shards for training"
                )
                train_relation_shards.sort(
                    key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                val_relation_shards = [
                    os.path.join(relation_shards_dir, f)
                    for f in os.listdir(relation_shards_dir)
                    if f.startswith("val_window") and f.endswith(".tar")
                ]
                if len(val_relation_shards) > 0:
                    val_relation_shards.sort(
                        key=lambda x: int(x.split("_")[-1].split(".")[0])
                    )
                    logger.info(
                        f"Found {len(val_relation_shards)} relation shards for training"
                    )
                else:
                    val_relation_shards = None
                    logger.info("No relation shards found for validation")
                test_relation_shards = [
                    os.path.join(relation_shards_dir, f)
                    for f in os.listdir(relation_shards_dir)
                    if f.startswith("test_window") and f.endswith(".tar")
                ]
                if len(test_relation_shards) > 0:
                    test_relation_shards.sort(
                        key=lambda x: int(x.split("_")[-1].split(".")[0])
                    )
                    logger.info(
                        f"Found {len(test_relation_shards)} relation shards for training"
                    )
                else:
                    test_relation_shards = None
                    logger.info("No relation shards found for test")
            else:
                raise ValueError("Relation shards directory not found")
        else:
            train_relation_shards = None
            val_relation_shards = None
            test_relation_shards = None

        self.train_dataset = ShardedDatasetV1(
            shards=train_shards,
            relation_shards=train_relation_shards,
            num_correlates=config.dataset_config.num_correlates,
        )

        logger.info("Loading val dataset")
        val_shards = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith(f"val{name_specification}") and f.endswith(".tar")
        ]
        val_shards.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.val_dataset = ShardedDatasetV1(
            shards=val_shards,
            relation_shards=val_relation_shards,
            num_correlates=config.dataset_config.num_correlates,
        )

        logger.info("Loading test dataset")
        test_shards = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith(f"test{name_specification}") and f.endswith(".tar")
        ]
        test_shards.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.test_dataset = ShardedDatasetV1(
            shards=test_shards,
            relation_shards=test_relation_shards,
            num_correlates=config.dataset_config.num_correlates,
        )

        super().__init__(
            config, self.train_dataset, self.val_dataset, self.test_dataset
        )

        # Set batch size and other dataloader parameters from config
        self.batch_size = config.dataset_config.batch_size
        self.num_workers = config.dataset_config.num_workers
        self.num_correlates = config.dataset_config.num_correlates
        self.shuffle = True
        self.use_relation_shards = config.dataset_config.use_relation_shards

    def relational_sampling_collate_fn(self, batch):
        """
        Custom collate function to reshape batches from:
        [batch_size*num_correlates, window_size] to [batch_size, num_correlates, window_size]

        Args:
            batch: List of tensors, each of shape [window_size]

        Returns:
            Tensor of shape [batch_size, num_correlates, window_size]
        """
        # get out metainformation
        data, indices, shard_indices, local_indices = zip(*batch)
        # Stack all examples into a single tensor of shape [batch_size*num_correlates, window_size]
        # Stack all examples into a single tensor
        # Each batch item is already (num_correlates, window_size)
        stacked = torch.stack(
            data
        )  # Shape: [batch_size, num_correlates, window_size]

        # If the number of samples is less than the batch size, pad with nans
        if stacked.shape[0] < self.batch_size:
            valid_indices = torch.cat(
                [
                    torch.ones(
                        stacked.shape[0] * self.num_correlates,
                        device=stacked.device,
                        dtype=torch.bool,
                    ),
                    torch.zeros(
                        (self.batch_size - stacked.shape[0])
                        * self.num_correlates,
                        device=stacked.device,
                        dtype=torch.bool,
                    ),
                ]
            )
            stacked = torch.cat(
                [
                    stacked,
                    torch.full(
                        (
                            self.batch_size - stacked.shape[0],
                            *stacked.shape[1:],
                        ),
                        torch.nan,
                    ),
                ]
            )
        else:
            valid_indices = torch.ones(
                stacked.shape[0], device=stacked.device, dtype=torch.bool
            )

        return {
            "timeseries": stacked,
            "valid_indices": valid_indices,
            "window_indices": indices,
            "shard_indices": shard_indices,
            "shard_local_indices": local_indices,
        }

    def non_relational_sampling_collate_fn(self, batch):
        """
        Custom collate function to reshape batches from:
        [batch_size*num_correlates, window_size] to [batch_size, num_correlates, window_size]
        """
        """
        Custom collate function for non-relational sampling.
        
        Args:
            batch: List of tensors, each of shape [window_size]
            
        Returns:
            Dictionary with key "timeseries" containing tensor of shape [batch_size, num_correlates, window_size]
        """
        # Extract data from batch
        data, indices, shard_indices, local_indices = zip(*batch)

        # Stack all examples into a single tensor of shape [batch_size*num_correlates, window_size]
        stacked = torch.stack(data)

        # If the number of samples is less than the batch size, pad with nans
        if (
            stacked.shape[0] * stacked.shape[1]
            < self.batch_size * self.num_correlates
        ):
            valid_indices = torch.cat(
                [
                    torch.ones(
                        stacked.shape[0],
                        device=stacked.device,
                        dtype=torch.bool,
                    ),
                    torch.zeros(
                        (
                            self.batch_size * self.num_correlates
                            - stacked.shape[0]
                        ),
                        device=stacked.device,
                        dtype=torch.bool,
                    ),
                ]
            )
            stacked = torch.cat(
                [
                    stacked,
                    torch.full(
                        (
                            self.batch_size * self.num_correlates
                            - stacked.shape[0],
                            *stacked.shape[1:],
                        ),
                        torch.nan,
                    ),
                ]
            )
        else:
            valid_indices = torch.ones(
                stacked.shape[0], device=stacked.device, dtype=torch.bool
            )

        stacked = stacked.view(self.batch_size, self.num_correlates, -1)

        return {
            "timeseries": stacked,
            "valid_indices": valid_indices,
            "window_indices": indices,
            "shard_indices": shard_indices,
            "shard_local_indices": local_indices,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size
            if self.use_relation_shards
            else self.batch_size * self.num_correlates,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.relational_sampling_collate_fn
            if self.use_relation_shards
            else self.non_relational_sampling_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * self.num_correlates,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.non_relational_sampling_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * self.num_correlates,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.non_relational_sampling_collate_fn,
        )

    def get_all_dataloader(self):
        """
        Create a dataloader that combines all data from train, val, and test splits.

        Returns:
            DataLoader containing all data in sequential order
        """
        # Create a new dataset that combines all splits
        all_shards = (
            self.train_dataset.shards
            + self.val_dataset.shards
            + self.test_dataset.shards
        )

        all_dataset = ShardedDatasetV1(
            shards=all_shards,
            num_correlates=self.num_correlates,
        )

        return DataLoader(
            all_dataset,
            batch_size=self.batch_size * self.num_correlates,
            shuffle=False,  # Maintain sequential order
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.non_relational_sampling_collate_fn,
        )
