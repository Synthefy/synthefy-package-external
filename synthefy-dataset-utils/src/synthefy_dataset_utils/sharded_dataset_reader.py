import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class ShardedDatasetReader:
    """
    Columnar tar-sharded dataset reader.

    Each field (e.g., 'history', 'target', 'forecast_modelA') is stored in its own
    directory of tar shards.  The index.sqlite database tracks offsets per sample.

    This reader provides:
      - Random access to any sample by ID
      - Optional selective field reads
      - Metadata queries via SQL
      - Lazy caching of open tar handles
    """

    def __init__(self, root: str, cache_tars: bool = True):
        self.root = Path(root)
        self.db_path = self.root / "index.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cache_tars = cache_tars
        self._tar_cache: Dict[tuple, tarfile.TarFile] = {}

    # ------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------

    def list_fields(self) -> List[str]:
        """Return all field names available in this dataset."""
        rows = self.conn.execute("SELECT name FROM fields").fetchall()
        return [r["name"] for r in rows]

    def __len__(self) -> int:
        """Total number of registered samples."""
        (n,) = self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()
        return n

    def shard_counts(self, field_name: str) -> Dict[int, int]:
        """Return a dict of shard_id → number of samples for a specific field."""
        shard_col = f"{field_name}_shard"
        rows = self.conn.execute(
            f"SELECT {shard_col}, COUNT(*) FROM samples WHERE {shard_col} IS NOT NULL GROUP BY {shard_col}"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------

    def query_ids(
        self,
        where_clause: Optional[str] = None,
        params: tuple = (),
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Run arbitrary SQL-style queries against the samples table.
        Example:
            reader.query_ids("forecast_modelB_offset IS NOT NULL")
            reader.query_ids("json_extract(meta_json, '$.length') = ?", (256,))
        """
        sql = "SELECT id FROM samples"
        if where_clause:
            sql += f" WHERE {where_clause}"
        if limit:
            sql += f" LIMIT {limit}"
        rows = self.conn.execute(sql, params).fetchall()
        return [r["id"] for r in rows]

    # ------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------

    def get_sample(
        self, sample_id: str, fields: Optional[List[str]] = None
    ) -> tuple[Dict[str, np.ndarray], dict]:
        """
        Load one sample's arrays (for given fields) and its metadata.
        Returns: (dict of field→np.ndarray, metadata dict)
        """
        row = self.conn.execute(
            "SELECT * FROM samples WHERE id=?", (sample_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Sample {sample_id} not found in index")

        meta = json.loads(row["meta_json"]) if row["meta_json"] else {}

        # Default to all fields known (look for _offset columns)
        field_list = fields or [
            f[:-7] for f in row.keys() if f.endswith("_offset")
        ]
        arrays = {}

        for field_name in field_list:
            offset_col = f"{field_name}_offset"
            shard_col = f"{field_name}_shard"

            if offset_col not in row.keys() or row[offset_col] is None:
                continue  # field missing for this sample

            # offset = row[offset_col]  # type: ignore
            shard_id = row[shard_col]

            tar = self._get_tar(field_name, shard_id)
            arrays[field_name] = self._read_npy(tar, sample_id)

        return arrays, meta

    def get_many(
        self, sample_ids: List[str], fields: Optional[List[str]] = None
    ) -> List[tuple[Dict[str, np.ndarray], dict]]:
        """Batch load multiple samples."""
        return [self.get_sample(sid, fields) for sid in sample_ids]

    # ------------------------------------------------------------
    # Tar handling
    # ------------------------------------------------------------

    def _get_tar(self, field: str, shard_id: int) -> tarfile.TarFile:
        """
        Retrieve (and optionally cache) a tarfile handle for a given field and shard.
        """
        key = (field, shard_id)
        if self.cache_tars and key in self._tar_cache:
            return self._tar_cache[key]

        base_dir = self.conn.execute(
            "SELECT base_dir FROM fields WHERE name=?", (field,)
        ).fetchone()
        if not base_dir:
            raise KeyError(f"Field '{field}' not found in fields table")

        # Use self.root instead of the absolute path from the database
        # This allows datasets to be moved to different locations
        relative_dir = Path(base_dir[0]).name
        tar_path = self.root / relative_dir / f"shard_{shard_id:04d}.tar"
        if not tar_path.exists():
            raise FileNotFoundError(
                f"Missing tar file for field '{field}', shard {shard_id}"
            )

        tar = tarfile.open(tar_path, "r")
        if self.cache_tars:
            self._tar_cache[key] = tar
        return tar

    def _read_npy(self, tar: tarfile.TarFile, sample_id: str) -> np.ndarray:
        """Extract and load a NumPy array by sample ID."""
        member_name = f"{sample_id}.npy"
        try:
            member = tar.getmember(member_name)
        except KeyError:
            raise KeyError(f"Sample {sample_id} not found in tar {tar.name}")
        with tar.extractfile(member) as f:  # type: ignore
            # Read into BytesIO to avoid fileno() issues with tar file objects
            buf = io.BytesIO(f.read())
            return np.load(buf, allow_pickle=True)

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    def close(self):
        """Close open tar files and database connection."""
        for tar in self._tar_cache.values():
            tar.close()
        self._tar_cache.clear()
        self.conn.close()
