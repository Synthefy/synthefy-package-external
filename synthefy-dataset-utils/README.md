# synthefy-dataset-utils

A columnar, sharded dataset storage system optimized for large-scale machine learning workflows. Store and access multi-field datasets with O(1) random access and efficient iteration.

## Overview

This package provides a lightweight, efficient way to store and retrieve large datasets of numpy arrays with the following features:

- **Columnar storage**: Each field (e.g., `history`, `target`, `forecast`) is stored separately in its own tar shards
- **Random access**: O(1) sample lookup via SQLite index with byte offsets
- **Scalable**: Automatic sharding keeps individual tar files manageable
- **Flexible**: Add/update samples and fields incrementally
- **SQL queryable**: Filter samples by metadata or field presence using SQL
- **Zero dependencies**: Only requires `numpy` and Python stdlib

### Architecture

```
dataset_root/
├── index.sqlite              # SQLite index with offsets and metadata
├── history_shards/           # One directory per field
│   ├── shard_0000.tar
│   ├── shard_0001.tar
│   └── ...
├── target_shards/
│   ├── shard_0000.tar
│   └── ...
└── forecast_shards/
    └── ...
```

Each tar file contains numpy arrays saved as `.npy` files, and the SQLite index tracks the exact byte offset of each sample in each field's tar files.

## Installation

From the synthefy package workspace:

```bash
cd synthefy-dataset-utils
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

## Quick Start

### Creating a Dataset

```python
from synthefy_dataset_utils.sharded_dataset_writer import ShardedDatasetWriter
import numpy as np

# Initialize writer (creates directory and index)
writer = ShardedDatasetWriter("my_dataset", samples_per_shard=1000)

# Register fields (columns) you'll be storing
writer.add_fields(["history", "target", "forecast"])

# Add samples - automatically generates unique IDs
for i in range(10000):
    arrays = {
        "history": np.random.randn(100),
        "target": np.random.randn(50),
        "forecast": np.random.randn(50),
    }
    metadata = {"length": 100, "source": "synthetic"}
    sample_id = writer.add_sample(arrays, metadata=metadata)

# Close when done
writer.close()
```

### Reading a Dataset

```python
from synthefy_dataset_utils.sharded_dataset_reader import ShardedDatasetReader

# Open dataset for reading
reader = ShardedDatasetReader("my_dataset")

# Get dataset info
print(f"Total samples: {len(reader)}")
print(f"Available fields: {reader.list_fields()}")

# Load a specific sample (returns arrays dict and metadata dict)
arrays, metadata = reader.get_sample("sample_000000000001")
print(f"History shape: {arrays['history'].shape}")
print(f"Metadata: {metadata}")

# Load only specific fields
arrays, metadata = reader.get_sample("sample_000000000001", fields=["history"])

# Query samples by metadata
sample_ids = reader.query_ids("json_extract(meta_json, '$.length') = 100")

# Batch load multiple samples
results = reader.get_many(sample_ids[:100], fields=["history", "target"])

reader.close()
```

## API Reference

### ShardedDatasetWriter

#### Initialization

```python
writer = ShardedDatasetWriter(root: str, samples_per_shard: int = 1000)
```

- `root`: Directory path for the dataset
- `samples_per_shard`: Number of samples per tar file before creating a new shard

#### Field Management

```python
# Add a single field
writer.add_field("my_field")

# Add multiple fields at once (more efficient)
writer.add_fields(["field1", "field2", "field3"])
```

#### Writing Data

**Simple API (recommended)**: Write all fields for a sample at once

```python
# Auto-generated sample ID
sample_id = writer.add_sample(
    arrays={"field1": np.array(...), "field2": np.array(...)},
    metadata={"key": "value"}  # optional
)

# Custom sample ID
sample_id = writer.add_sample(
    arrays={"field1": np.array(...)},
    sample_id="my_custom_id",
    metadata={"key": "value"}
)

# Update existing sample (overwrites fields, creates new tar entries)
writer.update_sample(
    sample_id="my_custom_id",
    arrays={"field1": np.array(...)},  # only updates specified fields
    metadata={"key": "new_value"}  # optional, replaces existing metadata
)
```

**Advanced APIs**: Lower-level control

```python
# Register sample IDs upfront (optional)
writer.register_samples(
    sample_ids=["id1", "id2", "id3"],
    metadata=[{"key": "val1"}, {"key": "val2"}, {"key": "val3"}]
)

# Batch write a complete shard for one field
writer.write_shard(
    field_name="history",
    shard_id=0,
    samples=["id1", "id2", "id3"],
    arrays=[np.array(...), np.array(...), np.array(...)]
)

# Streaming writes with context manager
with writer.open_shard("history", shard_id=0) as shard:
    for sample_id, array in your_data_generator():
        shard.add_sample(sample_id, array)
```

#### Cleanup

```python
writer.close()  # Close all open tar files and database connection
```

### ShardedDatasetReader

#### Initialization

```python
reader = ShardedDatasetReader(root: str, cache_tars: bool = True)
```

- `root`: Dataset directory path
- `cache_tars`: Keep tar files open for faster repeated access

#### Introspection

```python
# Number of samples
total = len(reader)

# Available fields
fields = reader.list_fields()  # Returns list of field names

# Per-field shard distribution
distribution = reader.shard_counts("history")  # {shard_id: count}
```

#### Querying

```python
# Get all sample IDs
all_ids = reader.query_ids()

# Filter by field presence
ids_with_forecast = reader.query_ids("forecast_offset IS NOT NULL")

# Filter by metadata (use SQLite json_extract)
ids = reader.query_ids(
    "json_extract(meta_json, '$.length') > ?",
    params=(128,)
)

# Limit results
first_100 = reader.query_ids(limit=100)
```

#### Loading Data

```python
# Load single sample (all fields)
arrays, metadata = reader.get_sample("sample_000000000001")
# arrays is dict: {"history": np.array(...), "target": np.array(...), ...}
# metadata is dict from the stored JSON

# Load specific fields only
arrays, metadata = reader.get_sample(
    "sample_000000000001",
    fields=["history", "target"]
)

# Batch load
results = reader.get_many(
    sample_ids=["id1", "id2", "id3"],
    fields=["history"]  # optional
)
# Returns list of (arrays_dict, metadata_dict) tuples
```

#### Cleanup

```python
reader.close()  # Close all open tar files and database connection
```

## Common Patterns

### Iterating Through a Dataset

```python
reader = ShardedDatasetReader("my_dataset")

# Get all sample IDs
sample_ids = reader.query_ids()

# Process in batches
batch_size = 128
for i in range(0, len(sample_ids), batch_size):
    batch_ids = sample_ids[i:i+batch_size]
    batch_data = reader.get_many(batch_ids, fields=["history", "target"])
    
    for arrays, metadata in batch_data:
        # Your processing logic here
        history = arrays["history"]
        target = arrays["target"]
        # ...

reader.close()
```

### Adding a New Field to an Existing Dataset

```python
# Open existing dataset for writing
writer = ShardedDatasetWriter("my_dataset", samples_per_shard=1000)

# Add the new field
writer.add_field("new_forecast")

# Get existing sample IDs
reader = ShardedDatasetReader("my_dataset")
sample_ids = reader.query_ids()

# Populate the new field
for sample_id in sample_ids:
    # Generate or load your new data
    new_array = generate_forecast(sample_id)
    
    # Update the sample with the new field
    writer.update_sample(sample_id, {"new_forecast": new_array})

reader.close()
writer.close()
```

### Filtering and Subsampling

```python
reader = ShardedDatasetReader("my_dataset")

# Get samples with specific characteristics
long_sequences = reader.query_ids(
    "json_extract(meta_json, '$.length') > 256 AND forecast_offset IS NOT NULL"
)

# Random subsample
import random
random.shuffle(long_sequences)
train_ids = long_sequences[:8000]
val_ids = long_sequences[8000:]

# Load subsets
train_data = reader.get_many(train_ids)
val_data = reader.get_many(val_ids)
```

### Incremental Dataset Construction

```python
writer = ShardedDatasetWriter("growing_dataset", samples_per_shard=1000)
writer.add_fields(["input", "output"])

# Add data in multiple stages/sessions
def add_batch(data_source):
    for item in data_source:
        writer.add_sample({
            "input": item.x,
            "output": item.y
        }, metadata={"source": data_source.name})
    writer.close_tars()  # Optional: flush between batches

# Day 1
add_batch(source_a)

# Day 2 - reopen same dataset
writer = ShardedDatasetWriter("growing_dataset", samples_per_shard=1000)
add_batch(source_b)

# Day 3
writer = ShardedDatasetWriter("growing_dataset", samples_per_shard=1000)
add_batch(source_c)
```

## Performance Considerations

### Shard Size

- **Default**: 1000 samples per shard is a good balance for most use cases
- **Large arrays**: Use fewer samples per shard (e.g., 100-500)
- **Small arrays**: Can use more samples per shard (e.g., 5000-10000)
- **Goal**: Keep individual tar files between 100MB - 2GB for optimal I/O

### Tar Caching

- **Reader**: Enable `cache_tars=True` (default) for random access patterns
- **Reader**: Disable `cache_tars=False` for sequential one-pass reads to save memory
- **Writer**: Automatically manages open tar handles, calls `close_tars()` to force flush

### Random Access vs Sequential

- **Random access**: Efficient with SQLite index, ~microseconds per lookup
- **Sequential**: Iterate through query results rather than random sampling
- **Batch loading**: Use `get_many()` for better throughput

### Metadata Queries

- SQLite's `json_extract()` is powerful but can be slow on large datasets
- Consider adding indexed columns for frequently queried metadata:
  ```python
  # After initialization
  writer.conn.execute("ALTER TABLE samples ADD COLUMN length INTEGER")
  writer.conn.execute("CREATE INDEX idx_length ON samples(length)")
  # Then populate when adding samples
  ```

## License

MIT License - see repository for details

