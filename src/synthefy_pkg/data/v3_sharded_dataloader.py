import io
import multiprocessing as mp
import os
import time
from typing import List, Literal, Optional

import holidays
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.data.data_gen_base import SynthesisBaseDataModule
from synthefy_pkg.data.shard_utils import (
    ShardCacheOnDisk,
    ShardLookupTable,
    alphanum_key,
    compute_on_disk_headers,
    glob_and_sort_shards,
)
from synthefy_pkg.preprocessing.fmv2_preprocess import (
    NORM_RANGES,
    TIMESTAMPS_FEATURES,
    convert_time_to_vector,
)
from synthefy_pkg.utils.basic_utils import seed_everything


class V3ShardedDataset(Dataset):
    def __init__(
        self,
        data_dirs: List[str],
        split: Literal["train", "val", "test"],
        length: int,
        context_length: int,
        use_window_counts: bool = False,
        relational_sampling_strategy: Optional[str] = None,
        relational_sampling_data_location: Optional[str] = None,
        num_correlates: Optional[int] = None,
        use_on_disk_cache: bool = True,
    ):
        # initialize lookup tables for each type of shard
        self.timestamps_shards = []
        self.values_shards = []
        self.text_embeddings_shards = []
        (
            self.timestamps_shards,
            self.values_shards,
            self.text_embeddings_shards,
        ) = glob_and_sort_shards(data_dirs)

        assert (
            len(self.timestamps_shards)
            == len(self.values_shards)
            == len(self.text_embeddings_shards)
        ), (
            f"Number of shards do not match: {len(self.timestamps_shards)} != {len(self.values_shards)} != {len(self.text_embeddings_shards)}"
        )

        self.timestamps_lookup_table = ShardLookupTable(
            shards=self.timestamps_shards,
            use_on_disk_cache=use_on_disk_cache,
        )
        self.values_lookup_table = ShardLookupTable(
            shards=self.values_shards,
            use_on_disk_cache=use_on_disk_cache,
        )
        self.text_embeddings_lookup_table = ShardLookupTable(
            shards=self.text_embeddings_shards,
            use_on_disk_cache=use_on_disk_cache,
        )

        assert (
            self.timestamps_lookup_table.cumulative_counts
            == self.values_lookup_table.cumulative_counts
        ).all()
        assert (
            self.timestamps_lookup_table.cumulative_counts
            == self.text_embeddings_lookup_table.cumulative_counts
        ).all()

        # initialize shard caches - TODO: Make this a single shard Cache
        self.timestamps_shard_cache = ShardCacheOnDisk()
        self.values_shard_cache = ShardCacheOnDisk()
        self.text_embeddings_shard_cache = ShardCacheOnDisk()

        data_dirs = sorted(data_dirs, key=alphanum_key)

        cutoff_date_mappings = []
        for data_dir in data_dirs:
            cutoff_date_mappings.append(
                np.load(os.path.join(data_dir, "cutoff_date_mapping.npy"))
            )
        self.cutoff_date_mapping = np.concatenate(cutoff_date_mappings)

        # Initialize window counts array
        self.use_window_counts = use_window_counts
        window_counts = []
        for data_dir in data_dirs:
            window_counts.append(
                np.load(os.path.join(data_dir, "window_counts.npy"))
            )
        self.window_counts = np.concatenate(window_counts)

        self.relational_sampling_strategy = relational_sampling_strategy
        self.relational_sampling_data_location = (
            relational_sampling_data_location
        )
        self.classes, self.relational_matrix = (
            self._load_relational_sampling_data(
                relational_sampling_strategy, relational_sampling_data_location
            )
        )

        self.split = split
        self.length = length
        self.context_length = context_length
        self.num_correlates = num_correlates

        # Cache valid indices
        if self.relational_sampling_strategy is not None:
            self._cache_valid_indices(relational=True)
        else:
            self._cache_valid_indices(relational=False)

        self.scaler_start_idx = 0
        self.scaler_end_idx = 3
        self.timestamp_start_idx = 3
        self.timestamp_end_idx = self.timestamp_start_idx + len(TIMESTAMPS_FEATURES) * self.context_length 
        self.continuous_start_idx = self.timestamp_end_idx
        self.continuous_end_idx = self.continuous_start_idx + self.context_length 

    def _load_relational_sampling_data(
        self,
        relational_sampling_strategy: Optional[str],
        relational_sampling_data_location: Optional[str],
    ):
        # Validation
        if relational_sampling_strategy is not None:
            # Check to make sure that the relational sampling data location is provided
            if relational_sampling_data_location is None:
                raise ValueError(
                    "relational_sampling_data_location must be provided when relational_sampling_strategy is not None"
                )
            # Check to make sure that the relational sampling data location is a valid path
            if not os.path.exists(relational_sampling_data_location):
                raise ValueError(
                    f"relational_sampling_data_location {relational_sampling_data_location} does not exist"
                )

            # Check to make sure that classes.npy and relational_matrix.npy exist in the relational sampling data location
            if not os.path.exists(
                os.path.join(relational_sampling_data_location, "classes.npy")
            ):
                raise ValueError(
                    f"classes.npy does not exist in {relational_sampling_data_location}"
                )
            if not os.path.exists(
                os.path.join(
                    relational_sampling_data_location, "relational_matrix.npy"
                )
            ):
                raise ValueError(
                    f"relational_matrix.npy does not exist in {relational_sampling_data_location}"
                )

            # Load the classes and relational matrix
            classes = np.load(
                os.path.join(relational_sampling_data_location, "classes.npy")
            )
            relational_matrix = np.load(
                os.path.join(
                    relational_sampling_data_location, "relational_matrix.npy"
                )
            )
            logger.info(
                f"Loaded relational information for {len(classes)} classes."
            )
            logger.info(
                f"From the relational matrix, we infer that the number of classes is {len(relational_matrix)}"
            )

            if (
                len(classes)
                != self.timestamps_lookup_table.cumulative_counts[-1]
            ):
                raise ValueError(
                    f"Number of datasets in relational sampling data ({len(classes)}) does not match number of datasets in timestamps lookup table ({self.timestamps_lookup_table.cumulative_counts[-1]})"
                )

            return classes, relational_matrix

        return None, None

    def __len__(self):
        return self.length

    def _cache_valid_indices(self, relational=False):
        start_time = time.time()
        # Cache the indices where cutoff date is valid for the split
        if self.split == "train":
            self.valid_date_indices = np.where(self.cutoff_date_mapping[:, 0])[
                0
            ]
        else:
            self.valid_date_indices = np.where(self.cutoff_date_mapping[:, 1])[
                0
            ]
        logger.info(
            f"Cached valid date indices in: {time.time() - start_time} seconds"
        )
 
        # Cache the indices where window count is long enough
        if self.split != "train":
            self.valid_window_indices = np.where(
                self.window_counts > self.context_length
            )[0]
        else:
            self.valid_window_indices = np.arange(len(self.window_counts))
        logger.info(
            f"Cached valid window indices in: {time.time() - start_time} seconds"
        )

        # Pre-compute the intersection of date and window indices (same for all classes)
        # Get the total length of the dataset for boolean mask creation
        total_length = len(self.cutoff_date_mapping)

        # Create boolean mask for date and window intersection
        valid_date_window_mask = np.zeros(total_length, dtype=bool)
        valid_date_window_mask[self.valid_date_indices] = True

        window_mask = np.zeros(total_length, dtype=bool)
        window_mask[self.valid_window_indices] = True

        valid_date_window_mask = valid_date_window_mask & window_mask

        if relational:
            assert self.relational_matrix is not None, (
                "Relational matrix must be provided to cache valid indices relationally"
            )
            assert self.classes is not None, (
                "Classes must be provided to cache valid indices relationally"
            )

            # Step 1: Use vectorized operation to create all class masks at once
            num_classes = len(self.relational_matrix)
            class_indices = np.arange(num_classes)[
                :, None
            ]  # Shape: (num_classes, 1)
            all_class_masks = (
                self.classes == class_indices
            )  # Broadcasting comparison
            logger.info(
                f"Created all class masks vectorized in: {time.time() - start_time} seconds"
            )

            # Step 2: Use vectorized operation to compute all intersections at once
            intersection_masks = (
                all_class_masks & valid_date_window_mask[None, :]
            )  # Broadcasting AND
            logger.info(
                f"Computed all intersections vectorized in: {time.time() - start_time} seconds"
            )

            # Convert boolean masks to indices only at the end
            self.valid_indices_dict = {}
            for class_idx in range(num_classes):
                self.valid_indices_dict[class_idx] = np.where(
                    intersection_masks[class_idx]
                )[0]
            logger.info(
                f"Cached valid indices dict in: {time.time() - start_time} seconds"
            )

        # For relational sampling, we might need the non-class-consrained version as well
        # This must be used for non-relational sampling
        self.valid_indices = np.where(valid_date_window_mask)[0]
        end_time = time.time()
        logger.info(f"Cached valid indices in: {end_time - start_time} seconds")

    def _select_random_dataset_index(self):
        # Randomly sample from the valid indices
        if self.use_window_counts:
            # Sample according to window counts distribution
            probabilities = self.window_counts[self.valid_indices]
            probabilities = (
                probabilities / probabilities.sum()
            )  # Normalize to probabilities
            dataset_index = np.random.choice(
                self.valid_indices, p=probabilities
            )
        else:
            dataset_index = np.random.choice(self.valid_indices)

        return dataset_index

    def _select_correlates_relationally(self):
        """
        Selects correlates relationally.
        - First selects a random dataset index
        - Then selects a random class from the probabilities of the selected dataset
        - Then selects a random dataset index from the class
        - Then repeats for the number of correlates

        Returns a list of tuples of (dataset_index, timestamp_data, value_data, text_embedding_data)
        """
        assert self.relational_sampling_strategy is not None, (
            "Relational sampling strategy must be provided to select correlates relationally"
        )
        assert self.relational_sampling_data_location is not None, (
            "Relational sampling data location must be provided to select correlates relationally"
        )
        assert self.classes is not None, (
            "Classes must be provided to select correlates relationally"
        )
        assert self.relational_matrix is not None, (
            "Relational matrix must be provided to select correlates relationally"
        )
        assert self.num_correlates is not None, (
            "Number of correlates must be provided to select correlates relationally"
        )

        correlates = []

        # First select a random dataset index
        valid_data = False
        while not valid_data:
            dataset_index = self._select_random_dataset_index()
            valid_data = self._is_sampled_dataset_valid(dataset_index)

        dataset_index, _, _, _ = valid_data
        correlates.append(valid_data)

        # Get the probabilities for the selected dataset
        probabilities = self.relational_matrix[
            self.classes[dataset_index]
        ]  # What other classes is this one allowed to sample from?

        dataset_indices = [dataset_index]

        for correlate_idx in range(1, self.num_correlates):
            valid_data = False
            while not valid_data:
                # Sample a class
                correlate_class = np.random.choice(
                    len(probabilities), p=probabilities
                )

                valid_indices = self.valid_indices_dict[correlate_class]

                if len(valid_indices) == 0:
                    # Instead of throwing an error if no valid dataset indices are found,
                    # we will just relax the relational sampling constraint.
                    # TODO: we should be very careful to monitor how often this happens when sampling relationally.
                    valid_indices = self.valid_indices

                # Sample a dataset index from this class
                if self.use_window_counts:
                    probabilities = self.window_counts[valid_indices]
                    probabilities = probabilities / probabilities.sum()
                    correlate_dataset_index = np.random.choice(
                        valid_indices, p=probabilities
                    )
                else:
                    correlate_dataset_index = np.random.choice(valid_indices)

                valid_data = self._is_sampled_dataset_valid(
                    correlate_dataset_index
                )
            correlates.append(valid_data)
            dataset_indices.append(correlate_dataset_index)

        return correlates, dataset_indices

    def _get_npy_data(self, shard_index, local_index):
        timestamp_shard_path = self.timestamps_shards[shard_index]
        value_shard_path = self.values_shards[shard_index]
        text_embedding_shard_path = self.text_embeddings_shards[shard_index]

        # Get the cached tarfile and members list
        timestamp_tar, timestamp_members = (
            self.timestamps_shard_cache.get_tarfile(timestamp_shard_path)
        )
        value_tar, value_members = self.values_shard_cache.get_tarfile(
            value_shard_path
        )
        text_embedding_tar, text_embedding_members = (
            self.text_embeddings_shard_cache.get_tarfile(
                text_embedding_shard_path
            )
        )

        try:
            # Get the actual numpy array data
            timestamp_data = np.load(
                io.BytesIO(
                    timestamp_tar.extractfile(
                        timestamp_members[local_index]
                    ).read()  # type: ignore
                ),
                allow_pickle=True,
            ).astype("datetime64[ns]")  # type: ignore
            value_data = np.load(
                io.BytesIO(
                    value_tar.extractfile(value_members[local_index]).read()  # type: ignore
                ),  # type: ignore
                allow_pickle=True,
            ).astype("float32")  # type: ignore
            text_embedding_data = np.load(
                io.BytesIO(
                    text_embedding_tar.extractfile(
                        text_embedding_members[local_index]
                    ).read()  # type: ignore
                )
            )  # type: ignore

            return timestamp_data, value_data, text_embedding_data
        finally:
            # Close the tarfiles to prevent memory leaks
            timestamp_tar.close()
            value_tar.close()
            text_embedding_tar.close()

    def _sample_valid_dataset(self):
        # index really doesn't matter in this dataset.
        while True:
            dataset_index = self._select_random_dataset_index()

            # Grab the timestamp and value numpy files for the sampled dataset
            shard_index, local_index = (
                self.timestamps_lookup_table.get_shard_and_local_index(
                    dataset_index
                )
            )

            timestamp_data, value_data, text_embedding_data = (
                self._get_npy_data(shard_index, local_index)
            )

            assert self.split == "train" or self.window_counts[
                dataset_index
            ] == len(timestamp_data), (
                f"Window count does not match length: {self.window_counts[dataset_index]} != {len(timestamp_data)}. This indicates a bug in V3 preprocessing."
            )

            # Check if value data is all NaN
            poor_window_flag = len(timestamp_data) != len(value_data)
            if len(value_data[~np.isnan(value_data)]) == 0 or poor_window_flag:
                # print(f"poor_window_flag: {poor_window_flag}")
                continue

            return (
                dataset_index,
                timestamp_data,
                value_data,
                text_embedding_data,
            )

    def _is_sampled_dataset_valid(self, dataset_index: int):
        """
        Checks if a dataset is valid.
        - Window count matches
        - Value data is not all NaN
        - Data is not too short
        - Data is not too long
        - Data is not too long

        If the dataset is not valid, returns False.
        If the dataset is valid, returns the index, timestamp, value, and text embedding data.
        """
        # Grab the timestamp and value numpy files for the sampled dataset
        shard_index, local_index = (
            self.timestamps_lookup_table.get_shard_and_local_index(
                dataset_index
            )
        )

        timestamp_data, value_data, text_embedding_data = self._get_npy_data(
            shard_index, local_index
        )

        assert self.split == "train" or self.window_counts[
            dataset_index
        ] == len(timestamp_data), (
            f"Window count does not match length: {self.window_counts[dataset_index]} != {len(timestamp_data)}. This indicates a bug in V3 preprocessing."
        )

        # Get length of data
        data_length = len(value_data)
        assert len(timestamp_data) == len(value_data), (
            "Timestamp and value data lengths do not match"
        )

        assert self.split == "train" or data_length >= self.context_length, (
            f"Data length is less than context length: {data_length} < {self.context_length} for split {self.split}"
        )

        assert data_length > 0, "Data length is 0"

        # Check if value data is all NaN
        poor_window_flag = len(timestamp_data) != len(value_data)
        if len(value_data[~np.isnan(value_data)]) == 0 or poor_window_flag:
            return False

        return (dataset_index, timestamp_data, value_data, text_embedding_data)

    def _pad_and_slice_data(
        self, timestamp_data: np.ndarray, value_data: np.ndarray
    ):
        """
        Pads and slices the data to the context length.
        """
        data_length = len(value_data)

        if data_length >= self.context_length:
            # Randomly select start index ensuring we have context_length elements
            start_idx = np.random.randint(
                0, data_length - self.context_length + 1
            )
            end_idx = start_idx + self.context_length

            # Slice the data
            timestamp_slice = timestamp_data[start_idx:end_idx]
            value_slice = value_data[start_idx:end_idx]

        else:
            # Need to pad - calculate padding amount
            start_idx = 0

            pad_length = self.context_length - data_length

            # Create padding arrays
            timestamp_pad = np.full(pad_length, np.nan, dtype=np.datetime64)
            value_pad = np.full(pad_length, np.nan)

            # Concatenate padding with data
            timestamp_slice = np.concatenate([timestamp_pad, timestamp_data])
            value_slice = np.concatenate([value_pad, value_data])

            assert len(value_slice.shape) == 1, "Value slice is not 1D"

        # Reshape value_slice to 2D if necessary
        value_slice_2d = value_slice.reshape(-1, 1)
        assert value_slice_2d.shape[0] > 0, "Value slice 2D is empty"

        return timestamp_slice, value_slice_2d, start_idx

    def _scale_data(self, value_slice_2d: np.ndarray):
        if np.isnan(value_slice_2d).all():
            value_slice = torch.tensor(value_slice_2d.reshape(-1))
            scalers = torch.tensor(
                [0.0, 1.0, 1.0]
            )  # zero mean, unit std, unit var
        else:
            # Step 2: Compute the scaling parameters using the imputed data
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.fit(value_slice_2d)

            # Step 3: Apply the scaling to the original data, preserving NaN values
            # Use the computed mean and scale to transform the original data
            value_slice_scaled = (value_slice_2d - scaler.mean_) / scaler.scale_  # type: ignore

            # Convert back to a tensor
            value_slice = torch.tensor(value_slice_scaled.reshape(-1))

            scalers = torch.tensor(
                [scaler.mean_[0], scaler.scale_[0], scaler.var_[0]]  # type: ignore
            )

        return value_slice, scalers

    def _get_timestamp_embeddings(self, timestamp_slice: np.ndarray):
        # Convert the timestamp to embeddings
        timestamp_embeddings = np.zeros(
            (len(timestamp_slice), len(TIMESTAMPS_FEATURES))
        )
        pd_timestamp_slice = pd.Series(timestamp_slice)
        us_holidays = holidays.country_holidays("US")
        timestamp_embeddings = convert_time_to_vector(
            timestamp_embeddings,
            TIMESTAMPS_FEATURES,
            NORM_RANGES,
            pd_timestamp_slice,
            us_holidays,
            0,
            np.array(pd_timestamp_slice.isna().values),
        )

        timestamp_embeddings = torch.tensor(timestamp_embeddings)
        return timestamp_embeddings

    def _get_single_item(self):
        # index really doesn't matter in this dataset.
        dataset_index, timestamp_data, value_data, text_embedding_data = (
            self._sample_valid_dataset()
        )

        # Pad and slice the data
        timestamp_slice, value_slice_2d, start_idx = self._pad_and_slice_data(
            timestamp_data, value_data
        )

        # scale the data
        value_slice, scalers = self._scale_data(value_slice_2d)

        # Get the timestamp embeddings
        timestamp_embeddings = self._get_timestamp_embeddings(timestamp_slice)

        output = torch.cat(
            [
                scalers,  # scalars
                timestamp_embeddings.flatten(),  # flattened timestamp embeddings
                torch.tensor(text_embedding_data),  # text embedding
                value_slice,  # value
                torch.tensor(
                    [dataset_index], dtype=torch.float32
                ),  # dataset index
            ]
        )

        return output, dataset_index, start_idx

    def _get_n_items_relationally(self):
        """
        Gets n items relationally.
        """
        correlates, dataset_indices = self._select_correlates_relationally()

        outputs = []
        start_idxs = []

        for (
            dataset_index,
            timestamp_data,
            value_data,
            text_embedding_data,
        ) in correlates:
            # Pad and slice the data
            timestamp_slice, value_slice_2d, start_idx = (
                self._pad_and_slice_data(timestamp_data, value_data)
            )
            start_idxs.append(start_idx)

            # Scale the data
            value_slice, scalers = self._scale_data(value_slice_2d)

            # Get the timestamp embeddings
            timestamp_embeddings = self._get_timestamp_embeddings(
                timestamp_slice
            )

            output = torch.cat(
                [
                    scalers,  # scalars
                    timestamp_embeddings.flatten(),  # flattened timestamp embeddings
                    torch.tensor(text_embedding_data),  # text embedding
                    value_slice,  # value
                    torch.tensor(
                        [dataset_index], dtype=torch.float32
                    ),  # dataset index
                ]
            )

            outputs.append(output)

        # Stack all outputs into a single tensor
        return torch.stack(outputs), dataset_indices, start_idxs

    def __getitem__(self, index: int):
        if self.relational_sampling_strategy is None:
            return self._get_single_item()
        else:
            return self._get_n_items_relationally()


class V3ShardedDataloader(SynthesisBaseDataModule):
    def __init__(self, config: Configuration):
        logger.info("Initializing V3ShardedDataloader")
        seed_everything(config.seed)
        self.config = config

        # Infer the data directory from the config
        # If it's an absolute path, don't append it to SYNTHEFY_DATASETS_BASE
        data_dir = config.dataset_config.dataset_name
        if os.path.isabs(data_dir):
            data_dir = data_dir
        else:
            data_dir = os.path.join(
                str(os.getenv("SYNTHEFY_DATASETS_BASE")), data_dir
            )

        if config.dataset_config.v3_data_paths:
            data_dirs = config.dataset_config.v3_data_paths
        else:
            data_dirs = [data_dir]

        self.batch_size = config.dataset_config.batch_size
        self.num_workers = config.dataset_config.num_workers
        self.context_length = config.dataset_config.time_series_length
        self.num_correlates = config.dataset_config.num_correlates
        self.use_window_counts = config.dataset_config.use_window_counts
        self.shuffle = True
        self.dataset_config = config.dataset_config
        if self.num_workers == 0:
            self.persistent_workers = False
        else:
            self.persistent_workers = True

        # Compute on-disk headers
        if self.dataset_config.write_on_disk_cache:
            logger.info("Writing on-disk headers")
            timestamps_shards, values_shards, text_embeddings_shards = (
                glob_and_sort_shards(data_dirs)
            )
            compute_on_disk_headers(
                timestamps_shards,
                use_multiprocessing=True,
                num_workers=self.num_workers,
            )
            compute_on_disk_headers(
                values_shards,
                use_multiprocessing=True,
                num_workers=self.num_workers,
            )
            compute_on_disk_headers(
                text_embeddings_shards,
                use_multiprocessing=True,
                num_workers=self.num_workers,
            )
            logger.info("On-disk headers computed and saved")
        else:
            logger.info(
                "Using pre-written on-disk headers. The dataloader will not confirm that the headers are correct. You should be certain that the headers are correct."
            )

        if config.dataset_config.relational_sampling_strategy is None:
            # Non-relational sampling
            self.dataset_length_multiplier = (
                self.batch_size * self.num_correlates
            )
            self.effective_batch_size = self.batch_size * self.num_correlates
        else:
            # Relational sampling
            self.dataset_length_multiplier = self.batch_size
            self.effective_batch_size = self.batch_size

        logger.info("Loading train dataset")
        self.train_dataset = V3ShardedDataset(
            data_dirs=data_dirs,
            split="train",
            length=config.foundation_model_config.train_epoch_length
            * self.dataset_length_multiplier,
            context_length=config.dataset_config.time_series_length,
            use_window_counts=self.use_window_counts,
            relational_sampling_strategy=config.dataset_config.relational_sampling_strategy,
            relational_sampling_data_location=config.dataset_config.relational_sampling_data_location,
            num_correlates=config.dataset_config.num_correlates,
            use_on_disk_cache=not self.dataset_config.write_on_disk_cache,
        )

        logger.info("Loading val dataset")
        self.val_dataset = V3ShardedDataset(
            data_dirs=data_dirs,
            split="val",
            length=config.foundation_model_config.val_epoch_length
            * self.dataset_length_multiplier,
            context_length=config.dataset_config.time_series_length,
            use_window_counts=self.use_window_counts,
            relational_sampling_strategy=config.dataset_config.relational_sampling_strategy,
            relational_sampling_data_location=config.dataset_config.relational_sampling_data_location,
            num_correlates=config.dataset_config.num_correlates,
            use_on_disk_cache=not self.dataset_config.write_on_disk_cache,
        )

        logger.info("Loading test dataset")
        self.test_dataset = V3ShardedDataset(
            data_dirs=data_dirs,
            split="test",
            length=config.foundation_model_config.test_epoch_length
            * self.dataset_length_multiplier,
            context_length=config.dataset_config.time_series_length,
            use_window_counts=self.use_window_counts,
            relational_sampling_strategy=config.dataset_config.relational_sampling_strategy,
            relational_sampling_data_location=config.dataset_config.relational_sampling_data_location,
            num_correlates=config.dataset_config.num_correlates,
            use_on_disk_cache=not self.dataset_config.write_on_disk_cache,
        )

        super().__init__(
            config, self.train_dataset, self.val_dataset, self.test_dataset
        )

    def nonrelational_collate_fn(self, batch):
        """
        Reshapes a flat list of examples to
        [batch, num_correlates, window_size] - works even for the very last
        (potentially incomplete) batch on each GPU.
        """
        # Unpack the batch into tensors and indices
        tensors, dataset_indices, start_indices = zip(*batch)
        stacked = torch.stack(tensors)  # (B', window)

        # If the sampler delivered an incomplete batch, drop the trailing
        # samples so that we can still reshape safely
        remainder = stacked.shape[0] % self.num_correlates
        if remainder != 0:
            stacked = stacked[:-remainder]
            dataset_indices = dataset_indices[:-remainder]
            start_indices = start_indices[:-remainder]
        assert stacked.shape[0] % self.num_correlates == 0, (
            "Batch size is not divisible by num_correlates"
        )

        current_bs = stacked.shape[0] // self.num_correlates
        stacked = stacked.view(current_bs, self.num_correlates, -1)

        # Convert indices to numpy arrays and reshape to (batch_size, num_correlates)
        dataset_indices = np.array(dataset_indices, dtype=np.int64).reshape(
            current_bs, self.num_correlates
        )
        start_indices = np.array(start_indices, dtype=np.int64).reshape(
            current_bs, self.num_correlates
        )

        # Stack the indices into a single array with shape (batch_size, num_correlates, 2)
        sample_ids = np.stack([dataset_indices, start_indices], axis=-1)

        # this adds backward compatibility for pfn style networks
        correlate_start = self.config.dataset_config.continuous_start_idx
        correlate_end = self.config.dataset_config.continuous_end_idx
        timestamps_start = self.config.dataset_config.timestamp_start_idx
        timestamps_end = self.config.dataset_config.timestamp_end_idx

        return {
            "timeseries": stacked,
            "values": stacked[:, :, correlate_start:correlate_end],
            "timestamps": stacked[:, :, timestamps_start:timestamps_end],
            "sample_ids": sample_ids,  # shape: (batch_size, num_correlates, 2)
        }

    def relational_collate_fn(self, batch):
        """
        Collates batch for relational sampling.
        Each item in batch is already [num_correlates, feature_dim],
        so we just stack them to get [batch, num_correlates, feature_dim].
        """
        # Unpack the batch into tensors and indices
        tensors, dataset_indices_list, start_indices_list = zip(*batch)
        stacked = torch.stack(tensors)  # (batch, num_correlates, feature_dim)

        # Convert indices to numpy arrays
        dataset_indices = np.array(
            dataset_indices_list, dtype=np.int64
        )  # shape: (batch_size, num_correlates)
        start_indices = np.array(
            start_indices_list, dtype=np.int64
        )  # shape: (batch_size, num_correlates)

        # Stack the indices into a single array with shape (batch_size, num_correlates, 2)
        sample_ids = np.stack([dataset_indices, start_indices], axis=-1)

        return {
            "timeseries": stacked,
            "sample_ids": sample_ids,  # shape: (batch_size, num_correlates, 2)
        }

    def collate_fn(self, batch):
        if self.dataset_config.relational_sampling_strategy is None:
            return self.nonrelational_collate_fn(batch)
        else:
            return self.relational_collate_fn(batch)

    def worker_init_fn(self, worker_id):
        """Initialize each worker with a unique seed based on the global seed."""

        # If workers are not persistent, this will repeat data every epoch
        seed = self.config.seed + worker_id
        seed_everything(seed)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.effective_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            drop_last=True,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.effective_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.effective_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            worker_init_fn=self.worker_init_fn,
        )
