"""
Base class for relational classifiers.
"""

import io
import tarfile
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from synthefy_pkg.data.shard_utils import (
    ShardCacheOnDisk,
    ShardLookupTable,
    glob_and_sort_shards,
)


class DataLoaderWorker:
    """Worker class that maintains a shard cache across multiple samples."""

    def __init__(self):
        self.shard_cache = ShardCacheOnDisk()
        self.timestamps_shards: Optional[List[str]] = None
        self.values_shards: Optional[List[str]] = None
        self.text_embeddings_shards: Optional[List[str]] = None
        self.lookup_table: Optional[ShardLookupTable] = None

    def initialize(
        self,
        timestamps_shards: List[str],
        values_shards: List[str],
        text_embeddings_shards: List[str],
        lookup_table: ShardLookupTable,
    ):
        """Initialize the worker with shard paths and lookup table."""
        self.timestamps_shards = timestamps_shards
        self.values_shards = values_shards
        self.text_embeddings_shards = text_embeddings_shards
        self.lookup_table = lookup_table

    def load_sample(self, dataset_index: int, load_options: List[str]):
        """
        Load a single sample using the worker's cached resources.

        Args:
            dataset_index: Index of the dataset to load
            load_options: List of data types to load. Options: ["timestamps", "values", "text_embeddings"]

        Returns:
            Tuple of (dataset_index, loaded_data_dict) where loaded_data_dict contains only requested data types
        """
        if (
            self.lookup_table is None
            or self.timestamps_shards is None
            or self.values_shards is None
            or self.text_embeddings_shards is None
        ):
            raise RuntimeError("Worker not properly initialized")

        # Validate load options
        valid_options = {"timestamps", "values", "text_embeddings"}
        invalid_options = set(load_options) - valid_options
        if invalid_options:
            raise ValueError(
                f"Invalid load options: {invalid_options}. Valid options are: {valid_options}"
            )

        # Get shard and local index for this sample
        shard_index, local_index = self.lookup_table.get_shard_and_local_index(
            dataset_index
        )

        loaded_data = {}
        open_tarfiles = []

        try:
            # Load timestamps if requested
            if "timestamps" in load_options:
                timestamp_shard_path = self.timestamps_shards[shard_index]
                timestamp_tar, timestamp_members = self.shard_cache.get_tarfile(
                    timestamp_shard_path
                )
                open_tarfiles.append(timestamp_tar)

                timestamp_data = np.load(
                    io.BytesIO(
                        timestamp_tar.extractfile(
                            timestamp_members[local_index]
                        ).read()  # type: ignore
                    ),
                    allow_pickle=True,
                ).astype("datetime64[ns]")
                loaded_data["timestamps"] = timestamp_data

            # Load values if requested
            if "values" in load_options:
                value_shard_path = self.values_shards[shard_index]
                value_tar, value_members = self.shard_cache.get_tarfile(
                    value_shard_path
                )
                open_tarfiles.append(value_tar)

                value_data = np.load(
                    io.BytesIO(
                        value_tar.extractfile(value_members[local_index]).read()  # type: ignore
                    ),
                    allow_pickle=True,
                ).astype("float32")
                loaded_data["values"] = value_data

            # Load text embeddings if requested
            if "text_embeddings" in load_options:
                text_embedding_shard_path = self.text_embeddings_shards[
                    shard_index
                ]
                text_embedding_tar, text_embedding_members = (
                    self.shard_cache.get_tarfile(text_embedding_shard_path)
                )
                open_tarfiles.append(text_embedding_tar)

                text_embedding_data = np.load(
                    io.BytesIO(
                        text_embedding_tar.extractfile(
                            text_embedding_members[local_index]
                        ).read()  # type: ignore
                    )
                )
                loaded_data["text_embeddings"] = text_embedding_data

            return dataset_index, loaded_data
        finally:
            # Close all opened tarfiles to prevent memory leaks
            for tar in open_tarfiles:
                tar.close()


# Global worker instance for each process
_worker: Optional[DataLoaderWorker] = None


def _init_worker(
    timestamps_shards, values_shards, text_embeddings_shards, lookup_table
):
    """Initialize the global worker for this process."""
    global _worker
    _worker = DataLoaderWorker()
    _worker.initialize(
        timestamps_shards, values_shards, text_embeddings_shards, lookup_table
    )


def _load_single_sample(args):
    """Worker function to load a single sample using the global worker."""
    dataset_index, load_options = args
    global _worker
    if _worker is None:
        raise RuntimeError("Worker not initialized")
    return _worker.load_sample(dataset_index, load_options)


class V3ShardedDataIterator:
    """Iterator for traversing all data in V3 sharded datasets."""

    def __init__(self, data_dir: str, load_options: Optional[List[str]] = None):
        """
        Initialize the iterator.

        Args:
            data_dir: Directory containing the sharded data
            load_options: List of data types to load. Options: ["timestamps", "values", "text_embeddings"]
                         Defaults to all three if not specified.
        """
        self.data_dirs = [data_dir]
        self.load_options = load_options or [
            "timestamps",
            "values",
            "text_embeddings",
        ]

        # Validate load options
        valid_options = {"timestamps", "values", "text_embeddings"}
        invalid_options = set(self.load_options) - valid_options
        if invalid_options:
            raise ValueError(
                f"Invalid load options: {invalid_options}. Valid options are: {valid_options}"
            )

        # Get sorted shard paths
        (
            self.timestamps_shards,
            self.values_shards,
            self.text_embeddings_shards,
        ) = glob_and_sort_shards(self.data_dirs)

        # Verify all shard types have the same count
        assert (
            len(self.timestamps_shards)
            == len(self.values_shards)
            == len(self.text_embeddings_shards)
        ), (
            f"Number of shards do not match: {len(self.timestamps_shards)} != "
            f"{len(self.values_shards)} != {len(self.text_embeddings_shards)}"
        )

        # Create lookup table to get total count and shard mapping
        self.lookup_table = ShardLookupTable(shards=self.timestamps_shards)

        # Initialize shard cache
        self.shard_cache = ShardCacheOnDisk()

        # Iterator state
        self.current_index = 0
        self.total_samples = len(self.lookup_table)

        logger.info(
            f"Initialized iterator with {self.total_samples} total samples across {len(self.timestamps_shards)} shards"
        )
        logger.info(f"Will load: {', '.join(self.load_options)}")

    def __iter__(self) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        """Return the iterator object."""
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[int, Dict[str, np.ndarray]]:
        """Get the next data sample."""
        if self.current_index >= self.total_samples:
            raise StopIteration

        # Get shard and local index for current sample
        shard_index, local_index = self.lookup_table.get_shard_and_local_index(
            self.current_index
        )

        # Load data from shards
        loaded_data = self._get_npy_data(shard_index, local_index)

        dataset_index = self.current_index
        self.current_index += 1

        return dataset_index, loaded_data

    def _get_npy_data(
        self, shard_index: int, local_index: int
    ) -> Dict[str, np.ndarray]:
        """Load numpy data from the specified shard and local index."""
        loaded_data = {}
        open_tarfiles = []

        try:
            # Load timestamps if requested
            if "timestamps" in self.load_options:
                timestamp_shard_path = self.timestamps_shards[shard_index]
                timestamp_tar, timestamp_members = self.shard_cache.get_tarfile(
                    timestamp_shard_path
                )
                open_tarfiles.append(timestamp_tar)

                timestamp_data = np.load(
                    io.BytesIO(
                        timestamp_tar.extractfile(
                            timestamp_members[local_index]
                        ).read()  # type: ignore
                    ),
                    allow_pickle=True,
                ).astype("datetime64[ns]")
                loaded_data["timestamps"] = timestamp_data

            # Load values if requested
            if "values" in self.load_options:
                value_shard_path = self.values_shards[shard_index]
                value_tar, value_members = self.shard_cache.get_tarfile(
                    value_shard_path
                )
                open_tarfiles.append(value_tar)

                value_data = np.load(
                    io.BytesIO(
                        value_tar.extractfile(value_members[local_index]).read()  # type: ignore
                    ),
                    allow_pickle=True,
                ).astype("float32")
                loaded_data["values"] = value_data

            # Load text embeddings if requested
            if "text_embeddings" in self.load_options:
                text_embedding_shard_path = self.text_embeddings_shards[
                    shard_index
                ]
                text_embedding_tar, text_embedding_members = (
                    self.shard_cache.get_tarfile(text_embedding_shard_path)
                )
                open_tarfiles.append(text_embedding_tar)

                text_embedding_data = np.load(
                    io.BytesIO(
                        text_embedding_tar.extractfile(
                            text_embedding_members[local_index]
                        ).read()  # type: ignore
                    )
                )
                loaded_data["text_embeddings"] = text_embedding_data

            return loaded_data
        finally:
            # Close all opened tarfiles to prevent memory leaks
            for tar in open_tarfiles:
                tar.close()

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.total_samples

    def iter_multiprocessed(
        self, num_workers: Optional[int] = None, batch_size: int = 100
    ) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Multiprocessed version of the iterator for faster data loading.

        Args:
            num_workers: Number of worker processes. Defaults to cpu_count() // 2
            batch_size: Number of samples to process in each batch

        Yields:
            Tuple of (dataset_index, loaded_data_dict)
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() // 2)

        logger.info(
            f"Starting multiprocessed iteration with {num_workers} workers, batch size {batch_size}"
        )

        # Prepare list of (dataset_index, load_options) tuples
        worker_args = [
            (i, self.load_options) for i in range(self.total_samples)
        ]

        # Process in batches to avoid memory issues
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                self.timestamps_shards,
                self.values_shards,
                self.text_embeddings_shards,
                self.lookup_table,
            ),
        ) as pool:
            for i in range(0, len(worker_args), batch_size):
                batch_args = worker_args[i : i + batch_size]

                # Process batch in parallel
                results = pool.map(_load_single_sample, batch_args)

                # Yield results in order
                for result in results:
                    yield result


class BaseRelationConstructor(ABC):
    """
    Base class for relation constructors.
    """

    def __init__(self, data_dir: str, constructor_name: str):
        self.data_dir = Path(data_dir)
        self.constructor_name = constructor_name
        # Ensure the relational_sampling directory exists
        self.relational_sampling_dir = (
            self.data_dir / "relational_sampling" / self.constructor_name
        )
        self.relational_sampling_dir.mkdir(parents=True, exist_ok=True)

    def iterate_over_data(
        self,
        load_options: Optional[List[str]] = None,
        multiprocess: bool = False,
        num_workers: Optional[int] = None,
        batch_size: int = 100,
    ) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Iterator over selected data types from V3 sharded data.

        Args:
            load_options: List of data types to load. Options: ["timestamps", "values", "text_embeddings"]
                         Defaults to all three if not specified.
            multiprocess: Whether to use multiprocessing for faster iteration
            num_workers: Number of worker processes (only used if multiprocess=True)
            batch_size: Batch size for multiprocessing (only used if multiprocess=True)

        Yields:
            Tuple of (dataset_index, loaded_data_dict)
            - dataset_index: Global index of the dataset
            - loaded_data_dict: Dictionary with keys from load_options containing numpy arrays

        Example:
            >>> constructor = ConcreteRelationConstructor("/path/to/data")
            >>> # Load only timestamps
            >>> for idx, data in constructor.iterate_over_data(load_options=["timestamps"]):
            ...     timestamps = data["timestamps"]
            ...     print(f"Dataset {idx} has {len(timestamps)} timestamps")
            >>>
            >>> # Load timestamps and values
            >>> for idx, data in constructor.iterate_over_data(load_options=["timestamps", "values"]):
            ...     timestamps = data["timestamps"]
            ...     values = data["values"]
            ...     print(f"Dataset {idx}: {len(timestamps)} timestamps, {len(values)} values")
        """
        iterator = V3ShardedDataIterator(str(self.data_dir), load_options)

        if multiprocess:
            for dataset_index, loaded_data in iterator.iter_multiprocessed(
                num_workers, batch_size
            ):
                yield dataset_index, loaded_data
        else:
            for dataset_index, loaded_data in iterator:
                yield dataset_index, loaded_data

    def get_total_samples(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples across all shards
        """
        iterator = V3ShardedDataIterator(
            str(self.data_dir), ["timestamps"]
        )  # Only load timestamps for counting
        return len(iterator)

    @abstractmethod
    def construct_relations(self, *args, **kwargs) -> None:
        """
        Construct relations from the given data.

        This method should be implemented by the subclasses.

        REQUIREMENTS:
        - Must save exactly 2 numpy files to {data_dir}/relational_sampling/:
          * classes.npy: 1D array containing integer class labels/indices (each element between 0 and N-1)
          * relational_matrix.npy: 2D square array with shape (N, N) containing relational data
        - Files should be saved using numpy.save()

        The implementation should:
        1. Process the input data to construct relations
        2. Determine the number of classes N
        3. Create a 1D classes array where each element is an integer between 0 and N-1
        4. Create a 2D square relational matrix with shape (N, N)
        5. Save both arrays to self.relational_sampling_dir with the specified names

        Args:
            *args: Variable positional arguments for relation construction
            **kwargs: Variable keyword arguments for relation construction

        Returns:
            None

        Raises:
            Should raise appropriate exceptions if relation construction fails
            or if the required files cannot be saved.
        """
        pass

    def validate_output_files(self) -> bool:
        """
        Validate that classes.npy and relational_matrix.npy exist with correct shapes.

        Returns:
            bool: True if validation passes

        Raises:
            FileNotFoundError: If the required files are not found
            ValueError: If the files have incorrect shapes or invalid values
        """
        classes_file = self.relational_sampling_dir / "classes.npy"
        relational_matrix_file = (
            self.relational_sampling_dir / "relational_matrix.npy"
        )

        # Check if both files exist
        if not classes_file.exists():
            raise FileNotFoundError(
                f"Required file 'classes.npy' not found in {self.relational_sampling_dir}. "
                "construct_relations() must save classes.npy and relational_matrix.npy."
            )

        if not relational_matrix_file.exists():
            raise FileNotFoundError(
                f"Required file 'relational_matrix.npy' not found in {self.relational_sampling_dir}. "
                "construct_relations() must save classes.npy and relational_matrix.npy."
            )

        # Load and validate shapes
        try:
            classes = np.load(classes_file)
            relational_matrix = np.load(relational_matrix_file)
        except Exception as e:
            raise ValueError(f"Error loading numpy files: {e}")

        # Validate classes shape (should be 1D)
        if classes.ndim != 1:
            raise ValueError(
                f"classes.npy must be 1-dimensional, "
                f"but got shape {classes.shape}"
            )

        # Validate relational_matrix shape (should be 2D and square)
        if relational_matrix.ndim != 2:
            raise ValueError(
                f"relational_matrix.npy must be 2-dimensional, "
                f"but got shape {relational_matrix.shape}"
            )

        if relational_matrix.shape[0] != relational_matrix.shape[1]:
            raise ValueError(
                f"relational_matrix.npy must be square (N, N), "
                f"but got shape {relational_matrix.shape}"
            )

        N = relational_matrix.shape[0]

        # Validate that classes contains integers between 0 and N-1
        if not np.issubdtype(classes.dtype, np.integer):
            raise ValueError(
                f"classes.npy must contain integers, but got dtype {classes.dtype}"
            )

        if len(classes) > 0:  # Only check if classes is not empty
            min_class = np.min(classes)
            max_class = np.max(classes)

            if min_class < 0:
                raise ValueError(
                    f"classes.npy must contain integers >= 0, but found minimum value {min_class}"
                )

            if max_class >= N:
                raise ValueError(
                    f"classes.npy must contain integers < {N} (relational_matrix size), "
                    f"but found maximum value {max_class}"
                )

        return True

    def save_numpy_files(
        self,
        classes: np.ndarray,
        relational_matrix: np.ndarray,
        subdirectory: Optional[str] = None,
    ) -> None:
        """
        Save classes and relational matrix to numpy files.
        """
        if subdirectory is not None:
            subdirectory_path = self.relational_sampling_dir / subdirectory
            subdirectory_path.mkdir(parents=True, exist_ok=True)

            self.relational_sampling_dir = subdirectory_path
        else:
            subdirectory_path = self.relational_sampling_dir

        np.save(subdirectory_path / "classes.npy", classes)
        np.save(
            subdirectory_path / "relational_matrix.npy",
            relational_matrix,
        )

        logger.info(
            f"Saved classes and relational matrix to {subdirectory_path}"
        )

    def run_with_validation(self, *args, **kwargs) -> None:
        """
        Run construct_relations and validate the output files.

        Args:
            *args: Arguments to pass to construct_relations
            **kwargs: Keyword arguments to pass to construct_relations
        """
        self.construct_relations(*args, **kwargs)
        self.validate_output_files()
