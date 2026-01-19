import glob
import io
import os
import tarfile

import numpy as np
from loguru import logger


def detect_shard_change(batch_shard_indices):
    # detect if a shard index has changed anywhere in the batch
    unique_indices = np.unique(batch_shard_indices)
    return len(unique_indices) > 1


def get_shard_path(output_subdir, split, name, shard_idx):
    # get the shard index for a given file with the format {split}_{name}_shard_{shard_idx}.tar
    shard_suffix = (
        f"_shard_{shard_idx}"
        if isinstance(shard_idx, str) or shard_idx > 0
        else ""
    )
    return os.path.join(output_subdir, f"{split}_{name}{shard_suffix}.tar")


def convert_batch_idx_to_data(batch_idx, batch_size):
    # convert a batch index to corresponding window indexes for that batch
    return np.arange(batch_idx * batch_size, batch_idx * (1 + batch_size))


def handle_existing_shards(output_subdir, split, existing_handling, name):
    """
    Handle shards in a given folder:
    Either @param existing_handling == "continue", which will give back the set of existing shards without affecting them
    Or @param existing_handling == "clean", which will delete all shards and return an empty set
    """
    shard_path = get_shard_path(output_subdir, split, name, "*")
    # get all files corresponding to the shard path with '*' replaced with a number
    shard_files = glob.glob(shard_path)
    if existing_handling == "continue":
        shard_indices = [
            int(
                os.path.basename(shard_file)
                .split("_shard_")[1]
                .split(".tar")[0]
            )
            for shard_file in shard_files
        ]
        return set(shard_indices)
    elif existing_handling == "clean":
        for shard_file in shard_files:
            os.remove(shard_file)
        return set()
    else:
        return set()


def save_idx_matrix(
    trailing_values,
    last_shard_index,
    current_shard_idx,
    batch_shard_indices,
    batch_rows,
    name,
    output_subdir,
    split_name,
    force=False,
) -> tuple[np.ndarray, int, int]:
    """
    Save the window or dataset idx matrix in a format that can be dataloaded given i,j
    @param trailing_values: the values from the last save that need to be added
    @param last_shard_index: the index of the last shard added (thus, the shard index of the trailing values will be last_shard_index + 1)
    @param current_shard_idx: the index of the current shard to be added (should be last_shard_index + 1)
    @param batch_shard_indices: the shard index for each value in the batch
    @param batch_rows: the rows of the current batch to be saved (and trailing values for the next batch)
    @param name: the name of the shard (for saving, ex: dataset_relation_matrix, window_relation_matrix, etc.)
    @param output_subdir: the directory to save the shard
    @param split_name: the name of the split (for saving, ex: train, val, test)
    @param force: whether to force the saving of the shard even if it already exists
    """
    # get unique indices and sort them
    unique_indices = np.unique(batch_shard_indices)
    sorted_indices = np.sort(unique_indices)
    if force:
        sorted_indices = [int(current_shard_idx), int(current_shard_idx + 1)]

    if len(trailing_values.shape) > 0:
        batch_shard_indices = np.concatenate(
            [
                np.repeat(last_shard_index + 1, len(trailing_values)),
                batch_shard_indices,
            ]
        )
        print("trailing values", last_shard_index, trailing_values.shape)
        batch_rows = np.concatenate([trailing_values, batch_rows], axis=0)

    saved_num = 0
    for shard_idx in sorted_indices[:-1]:
        # TODO: Not sure about this implementation
        shard_idx_batch_rows = batch_rows[batch_shard_indices == shard_idx]
        print("shard idx batch rows", shard_idx, shard_idx_batch_rows.shape)
        # save the batch_chunk as a numpy array
        shard_suffix = f"_shard_{shard_idx}"  # if shard_num > 0 else ""
        tar_file_path = os.path.join(
            output_subdir, f"{split_name}_{name}{shard_suffix}.tar"
        )
        os.makedirs(output_subdir, exist_ok=True)
        with tarfile.open(
            tar_file_path,
            "w",
        ) as tar:
            for i in range(shard_idx_batch_rows.shape[0]):
                buf = io.BytesIO()
                # Save array to buffer
                np.save(buf, shard_idx_batch_rows[i : (i + 1)])
                buf.seek(0)
                saved_num += 1

                # Create tarinfo
                tarinfo = tarfile.TarInfo(
                    f"{split_name}_{name}{shard_suffix}_{i}.npy"
                )
                tarinfo.size = buf.getbuffer().nbytes

                # Add buffer to tar
                tar.addfile(tarinfo, buf)
            logger.info(
                f"Saved {tar_file_path} to tar file {shard_idx_batch_rows.shape}"
            )
    # all trailing values should be flushed, so return empty array
    if force and len(sorted_indices) == 0:
        return np.ndarray([]), current_shard_idx, saved_num
    trailing_values = batch_rows[batch_shard_indices == sorted_indices[-1]]
    return (
        (trailing_values, current_shard_idx, saved_num)
        if len(trailing_values) > 0
        else (np.ndarray([]), current_shard_idx, saved_num)
    )
