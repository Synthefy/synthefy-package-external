import io
import json
import os
import sqlite3
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class ShardedDatasetWriter:
    """
    Columnar tar-sharded dataset writer.

    Each field (e.g. history, target, forecast_modelA) has its own shard directory.
    A central SQLite index stores per-sample offsets per field for O(1) random access.

    Supports:
    - add_fields(): register multiple fields at once
    - register_samples(): register new sample IDs and metadata (optional, auto-generated if not provided)
    - write_shard(): batch write arrays for one field
    - open_shard(): streaming write context for a field
    - add_sample(): add multiple fields for a single sample in one call (auto-generates ID if not provided)
    - update_sample(): update fields for an existing sample ID
    """

    # Batch size for commits - only commit every N samples for performance
    # Aligned with default samples_per_shard for optimal I/O (one commit per shard)
    COMMIT_BATCH_SIZE = 10000

    def __init__(self, root: str, samples_per_shard: int = 1000):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True, parents=True)
        self.db_path = self.root / "index.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()
        self._tar_cache = {}
        self.samples_per_shard = samples_per_shard
        # Track current shard per field
        self._field_current_shard: Dict[str, int] = {}
        self._field_current_count: Dict[str, int] = {}
        # Track samples since last commit for batched commits
        self._samples_since_commit = 0

    # ------------------------------------------------------------
    # Database schema
    # ------------------------------------------------------------

    def _init_schema(self):
        cur = self.conn.cursor()
        # Enable WAL mode for much faster writes (allows concurrent reads during writes)
        cur.execute("PRAGMA journal_mode=WAL")
        # Reduce synchronous level for better write performance
        # NORMAL is safe for WAL mode and much faster than FULL
        cur.execute("PRAGMA synchronous=NORMAL")
        # Increase cache size for better performance (negative = KB, so -64000 = 64MB)
        cur.execute("PRAGMA cache_size=-64000")
        # Store temp tables in memory
        cur.execute("PRAGMA temp_store=MEMORY")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id TEXT PRIMARY KEY,
            meta_json TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fields (
            name TEXT PRIMARY KEY,
            num_shards INTEGER,
            base_dir TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS counters (
            name TEXT PRIMARY KEY,
            value INTEGER
        )
        """)
        # Initialize sample counter if not exists
        cur.execute(
            "INSERT OR IGNORE INTO counters (name, value) VALUES ('sample_id', 0)"
        )
        self.conn.commit()

    def _generate_sample_id(self) -> str:
        """Generate a unique sample ID using an auto-incrementing counter."""
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE counters SET value = value + 1 WHERE name = 'sample_id'"
        )
        cur.execute("SELECT value FROM counters WHERE name = 'sample_id'")
        counter = cur.fetchone()[0]
        self.conn.commit()
        return f"sample_{counter:012d}"

    # ------------------------------------------------------------
    # Schema registration
    # ------------------------------------------------------------

    def add_field(self, name: str):
        """Add a single new field."""
        base_dir = self.root / f"{name}_shards"
        base_dir.mkdir(exist_ok=True)
        cur = self.conn.cursor()
        # Check if columns exist before adding them (SQLite doesn't support IF NOT EXISTS)
        columns = [
            row[1]
            for row in cur.execute("PRAGMA table_info(samples)").fetchall()
        ]
        if f"{name}_offset" not in columns:
            cur.execute(f"ALTER TABLE samples ADD COLUMN {name}_offset INTEGER")
        if f"{name}_shard" not in columns:
            cur.execute(f"ALTER TABLE samples ADD COLUMN {name}_shard INTEGER")
        cur.execute(
            "INSERT OR IGNORE INTO fields (name, num_shards, base_dir) VALUES (?, ?, ?)",
            (name, 0, str(base_dir)),
        )
        self.conn.commit()

    def add_fields(self, names: List[str]):
        """Add multiple new fields at once."""
        cur = self.conn.cursor()
        # Get existing columns once to check against all fields
        columns = [
            row[1]
            for row in cur.execute("PRAGMA table_info(samples)").fetchall()
        ]
        for name in names:
            base_dir = self.root / f"{name}_shards"
            base_dir.mkdir(exist_ok=True)
            # Check if columns exist before adding them (SQLite doesn't support IF NOT EXISTS)
            if f"{name}_offset" not in columns:
                cur.execute(
                    f"ALTER TABLE samples ADD COLUMN {name}_offset INTEGER"
                )
                columns.append(
                    f"{name}_offset"
                )  # Update local list to avoid re-checking
            if f"{name}_shard" not in columns:
                cur.execute(
                    f"ALTER TABLE samples ADD COLUMN {name}_shard INTEGER"
                )
                columns.append(f"{name}_shard")
            cur.execute(
                "INSERT OR IGNORE INTO fields (name, num_shards, base_dir) VALUES (?, ?, ?)",
                (name, 0, str(base_dir)),
            )
        self.conn.commit()

    # ------------------------------------------------------------
    # Sample registration
    # ------------------------------------------------------------

    def register_samples(
        self, sample_ids: List[str], metadata: Optional[List[dict]] = None
    ):
        """Register sample IDs and optional metadata."""
        cur = self.conn.cursor()
        for i, sid in enumerate(sample_ids):
            meta = json.dumps(metadata[i]) if metadata else "{}"
            cur.execute(
                "INSERT OR IGNORE INTO samples (id, meta_json) VALUES (?, ?)",
                (sid, meta),
            )
        self.conn.commit()

    # ------------------------------------------------------------
    # Batch write (old API)
    # ------------------------------------------------------------

    def write_shard(
        self,
        field_name: str,
        shard_id: int,
        samples: List[str],
        arrays: List[np.ndarray],
    ):
        """Write one tar shard for a single field."""
        base_dir = Path(
            self.conn.execute(
                "SELECT base_dir FROM fields WHERE name=?", (field_name,)
            ).fetchone()[0]
        )
        tar_path = base_dir / f"shard_{shard_id:04d}.tar"

        with tarfile.open(tar_path, "w") as tar:
            offset = 0
            for sample_id, arr in zip(samples, arrays):
                buf = io.BytesIO()
                np.save(buf, arr)
                size = buf.tell()
                buf.seek(0)
                info = tarfile.TarInfo(name=f"{sample_id}.npy")
                info.size = size
                tar.addfile(info, buf)
                self.conn.execute(
                    f"UPDATE samples SET {field_name}_offset=?, {field_name}_shard=? WHERE id=?",
                    (offset, shard_id, sample_id),
                )
                offset += size
        self.conn.execute(
            "UPDATE fields SET num_shards=? WHERE name=?",
            (shard_id + 1, field_name),
        )
        self.conn.commit()

    # ------------------------------------------------------------
    # Streaming write (context manager)
    # ------------------------------------------------------------

    class _ShardHandle:
        def __init__(
            self, outer, field_name: str, shard_id: int, tar_path: Path
        ):
            self.outer = outer
            self.field_name = field_name
            self.shard_id = shard_id
            self.tar = tarfile.open(tar_path, "w")
            self.offset = 0

        def add_sample(self, sample_id: str, arr: np.ndarray):
            buf = io.BytesIO()
            np.save(buf, arr)
            size = buf.tell()
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{sample_id}.npy")
            info.size = size
            self.tar.addfile(info, buf)
            self.outer.conn.execute(
                f"UPDATE samples SET {self.field_name}_offset=?, {self.field_name}_shard=? WHERE id=?",
                (self.offset, self.shard_id, sample_id),
            )
            self.offset += size
            # periodic flush
            if self.offset % (100 * 1024 * 1024) < size:
                self.outer.conn.commit()

        def close(self):
            self.tar.close()
            self.outer.conn.commit()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    def open_shard(self, field_name: str, shard_id: int):
        """Context manager for streaming writes into a single tar shard."""
        base_dir = Path(
            self.conn.execute(
                "SELECT base_dir FROM fields WHERE name=?", (field_name,)
            ).fetchone()[0]
        )
        tar_path = base_dir / f"shard_{shard_id:04d}.tar"
        return self._ShardHandle(self, field_name, shard_id, tar_path)

    # ------------------------------------------------------------
    # Multi-field per-sample write (new API)
    # ------------------------------------------------------------

    def add_sample(
        self,
        arrays: Dict[str, np.ndarray],
        sample_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Add all fields for a single sample in one call.
        Each field is written to its own tar file and its offset recorded.
        The writer automatically manages per-field shard allocation.

        Args:
            arrays: Dictionary mapping field names to numpy arrays
            sample_id: Optional sample ID. If None, a unique ID is auto-generated.
            metadata: Optional metadata dictionary to store with the sample

        Returns:
            The sample ID (either provided or auto-generated)
        """
        # Auto-generate sample ID if not provided
        if sample_id is None:
            sample_id = self._generate_sample_id()

        # Register the sample in the database
        meta_json = json.dumps(metadata) if metadata else "{}"
        self.conn.execute(
            "INSERT OR IGNORE INTO samples (id, meta_json) VALUES (?, ?)",
            (sample_id, meta_json),
        )

        # Cache field base directories to avoid repeated SQL lookups
        if not hasattr(self, "_field_base_dirs"):
            self._field_base_dirs: Dict[str, Path] = {}

        for field_name, arr in arrays.items():
            # Initialize field tracking if first time seeing this field
            if field_name not in self._field_current_shard:
                self._field_current_shard[field_name] = 0
                self._field_current_count[field_name] = 0

            # Check if this field needs to rotate to a new shard
            if self._field_current_count[field_name] >= self.samples_per_shard:
                self._rotate_field_shard(field_name)

            shard_id = self._field_current_shard[field_name]

            # lazy open per-field tar (cache key includes both field and shard)
            cache_key = (field_name, shard_id)
            if cache_key not in self._tar_cache:
                # Cache the base_dir lookup to avoid repeated SQL queries
                if field_name not in self._field_base_dirs:
                    base_dir_result = self.conn.execute(
                        "SELECT base_dir FROM fields WHERE name=?",
                        (field_name,),
                    ).fetchone()
                    if not base_dir_result:
                        raise ValueError(
                            f"Field '{field_name}' not registered."
                        )
                    self._field_base_dirs[field_name] = Path(base_dir_result[0])

                base_dir = self._field_base_dirs[field_name]
                base_dir.mkdir(exist_ok=True)
                tar_path = base_dir / f"shard_{shard_id:04d}.tar"
                self._tar_cache[cache_key] = {
                    "tar": tarfile.open(tar_path, "a"),
                    "offset": 0,
                }

            info = self._tar_cache[cache_key]
            tar, offset = info["tar"], info["offset"]

            buf = io.BytesIO()
            np.save(buf, arr)
            size = buf.tell()
            buf.seek(0)

            tinfo = tarfile.TarInfo(name=f"{sample_id}.npy")
            tinfo.size = size
            tar.addfile(tinfo, buf)

            self.conn.execute(
                f"UPDATE samples SET {field_name}_offset=?, {field_name}_shard=? WHERE id=?",
                (offset, shard_id, sample_id),
            )
            info["offset"] += size
            self._field_current_count[field_name] += 1

        # Batched commits: only commit every COMMIT_BATCH_SIZE samples
        self._samples_since_commit += 1
        if self._samples_since_commit >= self.COMMIT_BATCH_SIZE:
            self.conn.commit()
            self._samples_since_commit = 0

        return sample_id

    def update_sample(
        self,
        sample_id: str,
        arrays: Dict[str, np.ndarray],
        metadata: Optional[dict] = None,
    ):
        """
        Update fields for an existing sample.

        This appends new data to the tar files and updates the offset pointers.
        The old data becomes unused space in the tar file until a compaction is performed.

        Args:
            sample_id: The sample ID to update
            arrays: Dictionary mapping field names to new numpy arrays
            metadata: Optional new metadata dictionary (if provided, replaces existing metadata)
        """
        # Verify sample exists
        cur = self.conn.cursor()
        result = cur.execute(
            "SELECT id FROM samples WHERE id=?", (sample_id,)
        ).fetchone()
        if not result:
            raise ValueError(
                f"Sample ID '{sample_id}' does not exist. Use add_sample() to create new samples."
            )

        # Update metadata if provided
        if metadata is not None:
            meta_json = json.dumps(metadata)
            self.conn.execute(
                "UPDATE samples SET meta_json=? WHERE id=?",
                (meta_json, sample_id),
            )

        # Cache field base directories to avoid repeated SQL lookups
        if not hasattr(self, "_field_base_dirs"):
            self._field_base_dirs: Dict[str, Path] = {}

        # Update each field
        for field_name, arr in arrays.items():
            # Get current shard for this field (use existing or initialize)
            if field_name not in self._field_current_shard:
                self._field_current_shard[field_name] = 0
                self._field_current_count[field_name] = 0

            # Check if this field needs to rotate to a new shard
            if self._field_current_count[field_name] >= self.samples_per_shard:
                self._rotate_field_shard(field_name)

            shard_id = self._field_current_shard[field_name]

            # lazy open per-field tar (cache key includes both field and shard)
            cache_key = (field_name, shard_id)
            if cache_key not in self._tar_cache:
                # Cache the base_dir lookup to avoid repeated SQL queries
                if field_name not in self._field_base_dirs:
                    base_dir_result = self.conn.execute(
                        "SELECT base_dir FROM fields WHERE name=?",
                        (field_name,),
                    ).fetchone()
                    if not base_dir_result:
                        raise ValueError(
                            f"Field '{field_name}' not registered."
                        )
                    self._field_base_dirs[field_name] = Path(base_dir_result[0])

                base_dir = self._field_base_dirs[field_name]
                base_dir.mkdir(exist_ok=True)
                tar_path = base_dir / f"shard_{shard_id:04d}.tar"
                self._tar_cache[cache_key] = {
                    "tar": tarfile.open(tar_path, "a"),
                    "offset": 0,
                }

            info = self._tar_cache[cache_key]
            tar, offset = info["tar"], info["offset"]

            buf = io.BytesIO()
            np.save(buf, arr)
            size = buf.tell()
            buf.seek(0)

            tinfo = tarfile.TarInfo(name=f"{sample_id}.npy")
            tinfo.size = size
            tar.addfile(tinfo, buf)

            # Update the offset pointer to the new location
            self.conn.execute(
                f"UPDATE samples SET {field_name}_offset=?, {field_name}_shard=? WHERE id=?",
                (offset, shard_id, sample_id),
            )
            info["offset"] += size
            self._field_current_count[field_name] += 1

        # Batched commits: only commit every COMMIT_BATCH_SIZE samples
        self._samples_since_commit += 1
        if self._samples_since_commit >= self.COMMIT_BATCH_SIZE:
            self.conn.commit()
            self._samples_since_commit = 0

    def _rotate_field_shard(self, field_name: str):
        """Close current shard for a specific field and start a new one."""
        # Close only the tar files for this field
        old_shard = self._field_current_shard[field_name]
        cache_key = (field_name, old_shard)
        if cache_key in self._tar_cache:
            self._tar_cache[cache_key]["tar"].close()
            del self._tar_cache[cache_key]

        self._field_current_shard[field_name] += 1
        self._field_current_count[field_name] = 0

    def close_tars(self):
        """Close all open tar files."""
        for info in self._tar_cache.values():
            info["tar"].close()
        self._tar_cache.clear()
        self.conn.commit()

    # ------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------

    def close(self):
        """Close database connection and any open tars."""
        self.close_tars()
        self.conn.commit()
        self.conn.close()
