import os
import tarfile
import tempfile

import numpy as np


class StreamToShards:
    def __init__(self, filename_prefix: str, shard_size: int):
        self.filename_prefix = filename_prefix
        self.shard_size = shard_size
        self.current_shard_idx = 0
        self.current_file_count = 0
        self.current_tar = None

        # Validate that the parent directory of filename_prefix exists
        parent_dir = os.path.dirname(self.filename_prefix)
        if not os.path.exists(parent_dir):
            raise ValueError(f"Parent directory does not exist: {parent_dir}")

    def add_data(self, npy_array: np.ndarray, name_in_tar: str):
        if (
            self.current_file_count == 0
            or self.current_file_count >= self.shard_size
        ):
            if self.current_tar is not None:
                self.current_tar.close()
            self.current_tar = tarfile.open(
                f"{self.filename_prefix}_{self.current_shard_idx}.tar", "w"
            )
            self.current_shard_idx += 1
            self.current_file_count = 0

        # Ensure current_tar is a tarfile object
        if self.current_tar is None:
            raise RuntimeError("Tar file is not initialized.")

        # Create a temporary .npy file from the ndarray in a standard temp directory
        with tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False
        ) as temp_file:
            np.save(temp_file, npy_array)
            temp_file_path = temp_file.name

        # Add the .npy file to the current tar file
        self.current_tar.add(temp_file_path, arcname=name_in_tar)
        self.current_file_count += 1

        # Remove the temporary .npy file
        os.remove(temp_file_path)

    def __del__(self):
        if self.current_tar is not None:
            self.current_tar.close()
