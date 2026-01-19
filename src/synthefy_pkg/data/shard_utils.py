import glob
import os
import pickle
import re
import tarfile
from multiprocessing import Pool, cpu_count
from typing import List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm


def alphanum_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def glob_and_sort_shards(data_dirs: List[str]):
    # Sort shards lexicographically by extracting and comparing numbers

    timestamps_shards = []
    values_shards = []
    text_embeddings_shards = []

    for data_dir in data_dirs:
        timestamps_shards.extend(
            glob.glob(os.path.join(data_dir, "timestamps", "*.tar"))
        )
        values_shards.extend(
            glob.glob(os.path.join(data_dir, "values", "*.tar"))
        )
        text_embeddings_shards.extend(
            glob.glob(os.path.join(data_dir, "text_embeddings", "*.tar"))
        )

    return (
        sorted(timestamps_shards, key=alphanum_key),
        sorted(values_shards, key=alphanum_key),
        sorted(text_embeddings_shards, key=alphanum_key),
    )


def compute_single_shard_header(shard: str):
    """Process a single shard to compute and save its header."""
    try:
        with tarfile.open(shard, "r") as tar:
            members = tar.getmembers()
            with open(shard.replace(".tar", ".pkl"), "wb") as f:
                pickle.dump(members, f)
        return f"Processed {shard}"
    except Exception as e:
        return f"Error processing {shard}: {str(e)}"


def compute_on_disk_headers(
    shards: List[str],
    use_multiprocessing: bool = True,
    num_workers: Optional[int] = None,
):
    """Compute and save on-disk headers for shards, optionally using multiprocessing.

    Args:
        shards: List of shard file paths
        use_multiprocessing: Whether to use multiprocessing for parallel processing
        num_workers: Number of worker processes (defaults to cpu_count() / 4)
    """
    if not shards:
        logger.info("No shards to process")
        return

    if not use_multiprocessing or len(shards) == 1:
        # Sequential processing with progress bar
        logger.info(f"Processing {len(shards)} shards sequentially")
        for shard in tqdm(shards, desc="Processing shards"):
            result = compute_single_shard_header(shard)
            if "Error" in result:
                logger.error(result)
    else:
        # Parallel processing with progress bar
        if num_workers is None:
            num_workers = max(1, cpu_count() // 4)
        num_workers = min(num_workers, len(shards))

        logger.info(
            f"Processing {len(shards)} shards for on-disk headers using {num_workers} workers"
        )

        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(compute_single_shard_header, shards),
                    total=len(shards),
                    desc=f"Processing shards with {num_workers} workers",
                )
            )

        # Log only errors
        for result in results:
            if "Error" in result:
                logger.error(result)


class ShardCacheOnDisk:
    def __init__(self, *args, **kwargs):
        pass  # No need to store anything

    def get_tarfile(self, path: str) -> tuple[tarfile.TarFile, list]:
        """Open tarfile and return it with members. Caller must close the tarfile."""
        with open(path.replace(".tar", ".pkl"), "rb") as f:
            members = pickle.load(f)

        tarfile_obj = tarfile.open(path, "r")
        return tarfile_obj, members

    def clear(self):
        pass  # Nothing to clear

    def __del__(self):
        pass  # Nothing to clean up


class ShardLookupTable:
    def __init__(
        self,
        shards: List[str],
        shard_counts_path: Optional[str] = None,
        use_on_disk_cache: bool = True,
    ):
        self.shards = shards
        self.use_on_disk_cache = use_on_disk_cache
        self.shard_counts = (
            self._compute_shard_counts()
            if shard_counts_path is None
            else self._load_shard_counts(shard_counts_path)
        )
        self.cumulative_counts = np.cumsum([0] + self.shard_counts)
        logger.info(f"cumulative counts: {self.cumulative_counts}")

    def _load_shard_counts(self, shard_counts_path: str):
        shard_counts = pickle.load(open(shard_counts_path, "rb"))
        assert len(shard_counts) == len(self.shards)
        return shard_counts

    @staticmethod
    def _count_shard(args):
        shard_path, use_on_disk_cache = args
        pkl_path = shard_path.replace(".tar", ".pkl")
        if use_on_disk_cache and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                members = pickle.load(f)
        else:
            with tarfile.open(shard_path, "r") as tar:
                members = tar.getmembers()

        return len([m for m in members if m.name.endswith(".npy")])

    def _compute_shard_counts(self):
        n_cores = max(1, cpu_count() // 2)

        with Pool(processes=n_cores) as pool:
            shard_counts = list(
                tqdm(
                    pool.imap(
                        self._count_shard,
                        zip(
                            self.shards,
                            [self.use_on_disk_cache] * len(self.shards),
                        ),
                    ),
                    total=len(self.shards),
                    desc=f"Computing shard counts using {n_cores} cores",
                )
            )

        return shard_counts

    def get_shard_and_local_index(self, index: int) -> tuple[int, int]:
        """Get both the shard index and the local index within that shard.

        Args:
            index: Global index of the sample

        Returns:
            tuple: (shard_index, local_index)
            - shard_index: Which shard contains this sample
            - local_index: Index of the sample within that shard
        """
        shard_index = int(
            np.searchsorted(self.cumulative_counts, index, side="right") - 1
        )
        local_index = int(index - self.cumulative_counts[shard_index])
        return shard_index, local_index

    def get_shard_index(self, index: int) -> int:
        """Get just the shard index for backward compatibility.

        Args:
            index: Global index of the sample

        Returns:
            int: Index of the shard containing this sample
        """
        return self.get_shard_and_local_index(index)[0]

    def __len__(self):
        return self.cumulative_counts[-1]


class ShardCache:
    def __init__(self, max_cache_size: int):
        self.max_cache_size = max_cache_size
        self.cache = {}  # Maps path -> (tarfile object, members list)
        self.cache_order = []  # LRU order of paths

    def __len__(self):
        return len(self.cache)

    def get_tarfile(self, path: str) -> tuple[tarfile.TarFile, list]:
        """Get a tarfile and its members from the cache or load if not present.

        Args:
            path: Path to the tarfile

        Returns:
            tuple: (TarFile, list of TarInfo members)
        """
        # If in cache, move to most recently used and return
        if path in self.cache:
            # Update LRU order
            self.cache_order.remove(path)
            self.cache_order.append(path)
            return self.cache[path]

        # Load new tarfile
        tar = tarfile.open(path, "r")
        members = tar.getmembers()  # Cache the members list

        # If adding this would exceed max size, remove least recently used first
        if len(self.cache) >= self.max_cache_size:
            lru_path = self.cache_order[0]
            self.cache[lru_path][0].close()  # Close the tarfile
            del self.cache[lru_path]
            self.cache_order.pop(0)

        # Add to cache
        self.cache[path] = (tar, members)
        self.cache_order.append(path)

        return tar, members

    def clear(self):
        """Clear the cache and close all open tarfiles."""
        for tar, _ in self.cache.values():
            tar.close()
        self.cache.clear()
        self.cache_order.clear()

    def __del__(self):
        """Ensure all tarfiles are closed when cache is destroyed."""
        self.clear()
