"""
1/2: Share a single config and class implementation, since they share
dataloading, sorting, and splitting. It's essential that we create identical
mappings between dataset_idx -> timeseries and dataset_idx -> text embedding.

Inputs:
- List of directories where univariate data is located
- Output directory for shards
- Number of datasets per shard.
- Blind percentage
- Train/Val/Test splits for blind and pretrain

Steps:
- Validate config
- For each input directory, glob the directories within it.
- Save the globbed directories, and sort them somehow (250k sort is not fast)
- Perform blind / pretrain split, create directories for blind and pretrain

- For timeseries shards only:
    - For each globbed directory, load the data from parquet, convert to numpy format
    - Split based on timestamps into train / val / test splits
    - Standard scale the data
    - Save the scaler and data into a tar shard

- For text embedding shards:
    - For each globbed directory, load the data from json metadata file
    - Save the data into a tar shard
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import re
import time
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from synthefy_pkg.preprocessing.fmv3.stream_to_shards import StreamToShards


class TimeseriesShardConstructor:
    def __init__(
        self,
        input_data: list[str],
        output_location: str,
        blind_percentage: float,
        shard_size: int,
        device: str,
        use_number: int = -1,  # if we want to only use a subset of the data
    ):
        self.input_data = input_data
        self.output_location = output_location
        self.blind_percentage = blind_percentage
        self.shard_size = shard_size
        self.device = device if torch.cuda.is_available() else "cpu"

        self._validate_config()
        self._create_output_directories()
        data_dirs = self._glob_input_data()
        if use_number > 0:
            data_dirs = data_dirs[:use_number]
        self.blind_data_dirs, self.pretrain_data_dirs = (
            self._blind_pretrain_split(data_dirs)
        )

    def _validate_config(self):
        # Validate splits and percentages
        if self.blind_percentage < 0 or self.blind_percentage > 1:
            raise ValueError("blind_percentage must be between 0 and 1")

        # Validate input data directories exist
        for data_dir in self.input_data:
            if not os.path.exists(data_dir):
                raise ValueError(
                    f"Input data directory does not exist: {data_dir}"
                )

        # Validate output directory exists, create if not
        if not os.path.exists(self.output_location):
            logger.info(
                f"Output directory does not exist, creating: {self.output_location}"
            )
            os.makedirs(self.output_location)

    def _create_output_directories(self):
        """Create the required directory structure under output_location."""
        # Create blind and pretrain directories
        # Create main split directories
        blind_dir = os.path.join(self.output_location, "blind")
        pretrain_dir = os.path.join(self.output_location, "pretrain")

        # Create subdirectories for each split
        for split_dir in [blind_dir, pretrain_dir]:
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(os.path.join(split_dir, "timestamps"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "values"), exist_ok=True)
            os.makedirs(
                os.path.join(split_dir, "text_embeddings"), exist_ok=True
            )

    def _process_directory(self, data_dir):
        """Process a single input directory to find valid data directories."""
        valid_dirs = []
        for root, dirs, files in os.walk(data_dir):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    parquet_files = [
                        f
                        for f in os.listdir(dir_path)
                        if f.endswith(".parquet")
                    ]
                    json_files = [
                        f
                        for f in os.listdir(dir_path)
                        if f.endswith("metadata.json")
                    ]

                    if parquet_files and json_files:
                        valid_dirs.append(dir_path)
                    else:
                        logger.error(
                            f"Directory {dir_path} does not contain both .parquet and metadata.json files. Ignoring."
                        )
                except Exception as e:
                    logger.error(
                        f"Error processing directory {dir_path}: {str(e)}"
                    )

        return valid_dirs

    def _glob_input_data(self):
        """Find all valid data directories containing both parquet and metadata files in parallel."""
        import concurrent.futures

        start_time = time.time()

        # Use ProcessPoolExecutor to parallelize directory processing
        data_dirs = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each input directory for processing
            future_to_dir = {
                executor.submit(self._process_directory, d): d
                for d in self.input_data
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_dir):
                dir_path = future_to_dir[future]
                try:
                    valid_dirs = future.result()
                    data_dirs.extend(valid_dirs)
                    logger.debug(
                        f"Found {len(valid_dirs)} valid data directories in {dir_path}"
                    )
                except Exception as e:
                    logger.error(f"Exception processing {dir_path}: {str(e)}")

        logger.debug(
            f"Found {len(data_dirs)} total data directories in {time.time() - start_time:.2f} seconds."
        )

        # Sort the data files alphabetically, treating numbers lexicographically
        start_time = time.time()

        def alphanum_key(s):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("([0-9]+)", s)
            ]

        data_dirs.sort(key=alphanum_key)

        logger.debug(
            f"Sorted {len(data_dirs)} data directories in {time.time() - start_time:.2f} seconds."
        )

        return data_dirs

    def _blind_pretrain_split(self, data_dirs: list[str]):
        """Split data directories into blind and pretrain sets based on blind_percentage.

        Args:
            data_dirs: List of data directory paths

        Returns:
            Tuple[List[str], List[str]]: (blind_dirs, pretrain_dirs)
        """
        # Calculate number of blind directories
        logger.info(
            f"Splitting {len(data_dirs)} directories into blind and pretrain sets"
        )
        num_blind = int(len(data_dirs) * self.blind_percentage)

        # Randomly sample indices without replacement
        blind_indices = np.random.choice(
            len(data_dirs), size=num_blind, replace=False
        )

        # Use numpy indexing for blind dirs
        blind_dirs = np.array(data_dirs)[blind_indices].tolist()

        # Create boolean mask for pretrain dirs
        pretrain_mask = ~np.isin(np.arange(len(data_dirs)), blind_indices)
        pretrain_dirs = np.array(data_dirs)[pretrain_mask].tolist()

        return blind_dirs, pretrain_dirs

    def _get_timestamp_column(self, data_dir: str):
        metadata_file = next(
            f for f in os.listdir(data_dir) if f.endswith("metadata.json")
        )
        with open(os.path.join(data_dir, metadata_file), "r") as f:
            metadata = json.load(f)

        return metadata["timestamp_columns"][0]

    def _construct_timeseries_shards_by_worker(
        self,
        data_dirs: list[str],
        worker_id: int,
        blind_or_pretrain: Literal["blind", "pretrain"],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        window_counts = np.zeros(len(data_dirs))
        start_dates = np.zeros(len(data_dirs), dtype="datetime64[ns]")
        end_dates = np.zeros(len(data_dirs), dtype="datetime64[ns]")

        shard_prefix = os.path.join(self.output_location, blind_or_pretrain)
        values_shards = StreamToShards(
            os.path.join(shard_prefix, "values", f"{worker_id}"),
            self.shard_size,
        )

        timestamps_shards = StreamToShards(
            os.path.join(shard_prefix, "timestamps", f"{worker_id}"),
            self.shard_size,
        )

        # Add description to tqdm to identify the worker
        pbar = tqdm(
            data_dirs,
            desc=f"Worker {worker_id}",
            position=worker_id,
            leave=False,
        )
        for dataset_idx, data_dir in enumerate(pbar):
            # Load the data from the parquet file
            parquet_file = next(
                f for f in os.listdir(data_dir) if f.endswith(".parquet")
            )
            df = pd.read_parquet(os.path.join(data_dir, parquet_file))

            # Grab the column name of the timestamp column
            timestamp_col = self._get_timestamp_column(data_dir)

            timestamps = np.array(
                df[timestamp_col].values, dtype="datetime64[ns]"
            )
            timeseries = np.array(
                df.drop(columns=[timestamp_col]).values,
                dtype="float32",
            ).flatten()  # TODO: use einops to make this readable
            values_shards.add_data(timeseries, f"{dataset_idx}_values.npy")
            timestamps_shards.add_data(
                timestamps, f"{dataset_idx}_timestamps.npy"
            )

            window_counts[dataset_idx] = len(timeseries)
            start_dates[dataset_idx] = timestamps[0]
            end_dates[dataset_idx] = timestamps[-1]

            # Update progress bar description to show current file
            pbar.set_postfix({"file": os.path.basename(data_dir)})

        return window_counts, start_dates, end_dates

    def _partition(self, seq, n):
        """
        Split *seq* into *n* chunks whose sizes differ by at most 1.
        The first *len(seq) % n* chunks get one extra element.
        """
        k, extra = divmod(len(seq), n)  # base size k, and how many get +1
        chunks = [
            seq[
                i * k + min(i, extra) :   # start index
                (i + 1) * k
                + min(
                    i + 1, extra
                )  # end index
            ]
            for i in range(n)
        ]
        return chunks

    def construct_timeseries_shards_mp(
        self,
        blind_or_pretrain: Literal["blind", "pretrain"],
        num_workers: int = 32,
    ):
        """Process data with a cleaner progress display"""
        if blind_or_pretrain == "blind":
            data_dirs = self.blind_data_dirs
        else:
            data_dirs = self.pretrain_data_dirs

        # Print information about the processing task
        logger.info(
            f"Processing {len(data_dirs)} directories for {blind_or_pretrain} split"
        )

        # Instead of showing progress bars for each worker, just use a single master progress bar
        # This avoids the overlapping issue completely
        master_pbar = tqdm(
            total=len(data_dirs),
            desc=f"Processing {blind_or_pretrain} data",
            position=0,
            leave=True,
        )

        # Create a callback function to update the progress bar
        processed_items = 0

        def update_pbar(result):
            nonlocal processed_items
            # Each result represents one chunk of processed data
            chunk_size = len(result)
            processed_items += chunk_size
            master_pbar.update(chunk_size)
            master_pbar.set_postfix(
                {"processed": processed_items, "total": len(data_dirs)}
            )
            return result

        # Process data in parallel using ProcessPoolExecutor
        chunks = self._partition(data_dirs, num_workers)
        results = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Submit all tasks
            futures = []
            future_to_worker = {}  # Map futures to worker indices
            for i, chunk in enumerate(chunks):
                # Process each chunk without its own progress bar
                future = executor.submit(
                    self._construct_timeseries_shards_by_worker,
                    chunk,
                    i,
                    blind_or_pretrain,
                )
                future.add_done_callback(lambda f: update_pbar(f.result()))
                futures.append(future)
                future_to_worker[future] = i

            # Wait for all futures to complete and collect results in worker order
            worker_results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    worker_id = future_to_worker[future]
                    worker_results[worker_id] = result
                except Exception as e:
                    logger.error(f"Error in worker process: {str(e)}")

            # Sort results by worker ID to maintain order
            results = [
                worker_results[i]
                for i in range(len(chunks))
                if i in worker_results
            ]

        # Close the progress bar
        master_pbar.close()

        # Combine all window counts into a single array
        window_count_results = [result[0] for result in results]
        start_date_results = [result[1] for result in results]
        end_date_results = [result[2] for result in results]
        total_window_counts = np.concatenate(window_count_results)
        total_start_dates = np.concatenate(start_date_results)
        total_end_dates = np.concatenate(end_date_results)
        np.save(
            os.path.join(
                self.output_location, blind_or_pretrain, "window_counts.npy"
            ),
            total_window_counts,
        )
        np.save(
            os.path.join(
                self.output_location, blind_or_pretrain, "start_dates.npy"
            ),
            total_start_dates,
        )
        np.save(
            os.path.join(
                self.output_location, blind_or_pretrain, "end_dates.npy"
            ),
            total_end_dates,
        )

        logger.info(f"Finished processing {len(data_dirs)} directories")
        return total_window_counts

    def construct_timeseries_shards(
        self, blind_or_pretrain: Literal["blind", "pretrain"]
    ):
        if blind_or_pretrain == "blind":
            data_dirs = self.blind_data_dirs
            shard_prefix = os.path.join(self.output_location, "blind")
        else:
            data_dirs = self.pretrain_data_dirs
            shard_prefix = os.path.join(self.output_location, "pretrain")

        window_counts = np.zeros(len(data_dirs))

        values_shards = StreamToShards(
            os.path.join(shard_prefix, "values", "shard"), self.shard_size
        )

        timestamps_shards = StreamToShards(
            os.path.join(shard_prefix, "timestamps", "shard"), self.shard_size
        )

        # Improve tqdm display with more information
        pbar = tqdm(
            data_dirs, desc=f"Processing {blind_or_pretrain} timeseries"
        )
        for dataset_idx, data_dir in enumerate(pbar):
            # Update progress bar with current directory
            pbar.set_postfix({"directory": os.path.basename(data_dir)})

            # Load the data from the parquet file
            parquet_file = next(
                f for f in os.listdir(data_dir) if f.endswith(".parquet")
            )
            df = pd.read_parquet(os.path.join(data_dir, parquet_file))

            # Grab the column name of the timestamp column
            timestamp_col = self._get_timestamp_column(data_dir)

            timestamps = np.array(df[timestamp_col].values)
            timeseries = np.array(
                df.drop(columns=[timestamp_col]).values
            ).flatten()  # TODO: use einops to make this readable
            values_shards.add_data(timeseries, f"{dataset_idx}_values.npy")
            timestamps_shards.add_data(
                timestamps, f"{dataset_idx}_timestamps.npy"
            )
            window_counts[dataset_idx] = len(timeseries)

        np.save(os.path.join(shard_prefix, "window_counts.npy"), window_counts)

    def _construct_text_embedding_shards_by_worker(
        self,
        data_dirs: list[str],
        worker_id: int,
        blind_or_pretrain: Literal["blind", "pretrain"],
    ):
        """Process text embeddings with position-fixed progress bar."""
        try:
            shard_prefix = os.path.join(self.output_location, blind_or_pretrain)
            text_embedding_shards = StreamToShards(
                os.path.join(shard_prefix, "text_embeddings", f"{worker_id}"),
                self.shard_size,
            )

            # Initialize the sentence transformer model
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            model = model.to(self.device)
            model = model.eval()

            # Use tqdm with fixed position for this worker
            # This is critical - setting fixed position and consistent formatting
            with tqdm(
                total=len(data_dirs),
                desc=f"Worker {worker_id:<2}",
                position=worker_id,
                leave=True,
                ncols=100,  # Fixed width prevents jitter
            ) as pbar:
                for dataset_idx, data_dir in enumerate(data_dirs):
                    try:
                        # Load the data from the json file
                        json_file = next(
                            f
                            for f in os.listdir(data_dir)
                            if f.endswith("metadata.json")
                        )
                        with open(os.path.join(data_dir, json_file), "r") as f:
                            metadata = json.load(f)

                        # Get the column description
                        column_description = str(
                            metadata["columns"][0]["description"]
                        )

                        # If description is empty, use title instead
                        if (
                            not column_description
                            or len(column_description.strip()) == 0
                        ):
                            column_description = str(
                                metadata["columns"][0]["title"]
                            )

                        # Generate embedding
                        with torch.no_grad():
                            embedding = model.encode(
                                column_description,
                                convert_to_tensor=False,
                                show_progress_bar=False,
                            )

                        embedding = cast(np.ndarray, embedding)

                        text_embedding_shards.add_data(
                            embedding, f"{dataset_idx}_text_embedding.npy"
                        )

                        # Update the progress bar
                        filename = os.path.basename(data_dir)
                        pbar.update(1)
                        pbar.set_postfix(
                            {"file": filename[:20]}
                        )  # truncate long names

                    except Exception as e:
                        logger.error(
                            f"Worker {worker_id} error processing {data_dir}: {str(e)}"
                        )
                        pbar.update(1)  # still update the bar

            return True
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {str(e)}")
            return False

    def construct_text_embedding_shards_mp(
        self,
        blind_or_pretrain: Literal["blind", "pretrain"],
        num_workers: int = 32,
    ):
        """Process text embeddings with multiple worker progress bars using spawn method."""
        # Set spawn method - use try/except in case it's already been set
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        if blind_or_pretrain == "blind":
            data_dirs = self.blind_data_dirs
        else:
            data_dirs = self.pretrain_data_dirs

        # Print information about the processing task
        total_dirs = len(data_dirs)
        logger.info(
            f"Processing {total_dirs} directories for {blind_or_pretrain} text embeddings"
        )

        # Create chunks for processing
        chunks = self._partition(data_dirs, num_workers)

        # Print a blank line first to make room for progress bars
        for _ in range(num_workers + 1):
            print("")

        # Create a context manager to handle cursor positioning
        class ProgressManager:
            def __enter__(self):
                # Move cursor up num_workers+1 lines to leave space for all progress bars
                print(f"\033[{num_workers + 1}A", end="")

            def __exit__(self, *args):
                # Move cursor down past all progress bars when done
                print(f"\033[{num_workers + 1}B", end="")

        # Process in parallel with spawn context
        mp_ctx = multiprocessing.get_context("spawn")
        with (
            ProgressManager(),
            concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, mp_context=mp_ctx
            ) as executor,
        ):
            # Submit jobs and collect futures
            futures = [
                executor.submit(
                    self._construct_text_embedding_shards_by_worker,
                    chunk,
                    i,
                    blind_or_pretrain,
                )
                for i, chunk in enumerate(chunks)
            ]

            # Wait for completion
            concurrent.futures.wait(futures)

        # Log completion
        logger.info(
            f"Finished processing text embeddings for {total_dirs} directories"
        )

    def construct_text_embedding_shards(
        self, blind_or_pretrain: Literal["blind", "pretrain"]
    ):
        if blind_or_pretrain == "blind":
            data_dirs = self.blind_data_dirs
            shard_prefix = os.path.join(self.output_location, "blind")
        else:
            data_dirs = self.pretrain_data_dirs
            shard_prefix = os.path.join(self.output_location, "pretrain")

        text_embedding_shards = StreamToShards(
            os.path.join(shard_prefix, "text_embeddings", "shard"),
            self.shard_size,
        )

        # Initialize the sentence transformer model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model = model.to(self.device)
        model = model.eval()

        # Improved tqdm bar with more information
        pbar = tqdm(
            data_dirs, desc=f"Creating {blind_or_pretrain} text embeddings"
        )
        for dataset_idx, data_dir in enumerate(pbar):
            # Update progress bar with current directory
            pbar.set_postfix({"directory": os.path.basename(data_dir)})

            # Load the data from the json file
            json_file = next(
                f for f in os.listdir(data_dir) if f.endswith("metadata.json")
            )
            with open(os.path.join(data_dir, json_file), "r") as f:
                metadata = json.load(f)

            # Get the column description
            column_description = str(metadata["columns"][0]["description"])

            # If description is empty, use title instead
            if not column_description or len(column_description.strip()) == 0:
                column_description = str(metadata["columns"][0]["title"])

            # Generate embedding
            with torch.no_grad():
                embedding = model.encode(
                    column_description,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                )

                embedding = cast(np.ndarray, embedding)

            text_embedding_shards.add_data(
                embedding, f"{dataset_idx}_text_embedding.npy"
            )

    def _construct_cutoff_date_mapping_by_worker(
        self,
        data_dirs: list[str],
        worker_id: int,
        blind_or_pretrain: Literal["blind", "pretrain"],
        cutoff_date: str,
    ) -> np.ndarray:
        """Process a subset of directories to create cutoff date mapping."""
        # Create boolean array for this worker's chunk
        cutoff_mask = np.zeros((len(data_dirs), 2), dtype=bool)

        # Convert the cutoff date to a tz-naive timestamp
        cutoff_date_ts = pd.to_datetime(cutoff_date).tz_localize(None)  # type: ignore

        # Add description to tqdm to identify the worker
        pbar = tqdm(
            enumerate(data_dirs),
            total=len(data_dirs),
            desc=f"Worker {worker_id}",
            position=worker_id,
            leave=False,
        )
        for i, data_dir in pbar:
            try:
                # Load the data from the parquet file
                parquet_file = next(
                    f for f in os.listdir(data_dir) if f.endswith(".parquet")
                )
                df = pd.read_parquet(os.path.join(data_dir, parquet_file))

                # Grab the column name of the timestamp column
                timestamp_col = str(self._get_timestamp_column(data_dir))

                # Ensure the timestamps in the DataFrame are tz-naive
                df[timestamp_col] = pd.to_datetime(
                    df[timestamp_col]
                ).dt.tz_localize(None)

                if pd.to_datetime(df.iloc[0][timestamp_col]) < cutoff_date_ts:
                    cutoff_mask[i, 0] = True  # Can be used for training
                if pd.to_datetime(df.iloc[-1][timestamp_col]) > cutoff_date_ts:
                    cutoff_mask[i, 1] = (
                        True  # Can be used for validation / test
                    )

                # Update progress bar description to show current file
                pbar.set_postfix({"file": os.path.basename(data_dir)})
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} error processing {data_dir}: {str(e)}"
                )

        return cutoff_mask

    def construct_cutoff_date_mapping_mp(
        self,
        blind_or_pretrain: Literal["blind", "pretrain"],
        cutoff_date: str,
        num_workers: int = 32,
    ):
        """Process cutoff date mapping with multiprocessing."""
        if blind_or_pretrain == "blind":
            data_dirs = self.blind_data_dirs
            save_path = os.path.join(
                self.output_location, "blind", "cutoff_date_mapping.npy"
            )
        else:
            data_dirs = self.pretrain_data_dirs
            save_path = os.path.join(
                self.output_location, "pretrain", "cutoff_date_mapping.npy"
            )

        # Print information about the processing task
        logger.info(
            f"Processing {len(data_dirs)} directories for {blind_or_pretrain} cutoff date mapping"
        )

        # Create a single master progress bar
        master_pbar = tqdm(
            total=len(data_dirs),
            desc=f"Processing {blind_or_pretrain} cutoff mapping",
            position=0,
            leave=True,
        )

        # Create a callback function to update the progress bar
        processed_items = 0

        def update_pbar(result):
            nonlocal processed_items
            # Each result represents one chunk of processed data
            chunk_size = len(result)
            processed_items += chunk_size
            master_pbar.update(chunk_size)
            master_pbar.set_postfix(
                {"processed": processed_items, "total": len(data_dirs)}
            )
            return result

        # Process data in parallel using ProcessPoolExecutor
        chunks = self._partition(data_dirs, num_workers)
        results = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Submit all tasks
            futures = []
            future_to_worker = {}  # Map futures to worker indices
            for i, chunk in enumerate(chunks):
                # Process each chunk
                future = executor.submit(
                    self._construct_cutoff_date_mapping_by_worker,
                    chunk,
                    i,
                    blind_or_pretrain,
                    cutoff_date,
                )
                future.add_done_callback(lambda f: update_pbar(f.result()))
                futures.append(future)
                future_to_worker[future] = i

            # Wait for all futures to complete and collect results in worker order
            worker_results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    worker_id = future_to_worker[future]
                    worker_results[worker_id] = result
                except Exception as e:
                    logger.error(
                        f"Error in cutoff mapping worker process: {str(e)}"
                    )

            # Sort results by worker ID to maintain order
            results = [
                worker_results[i]
                for i in range(len(chunks))
                if i in worker_results
            ]

        # Close the progress bar
        master_pbar.close()

        # Combine all worker results into a single array
        # We need to create a new array of the right size and copy the results
        total_cutoff_mask = np.concatenate(results, axis=0)

        assert len(total_cutoff_mask) == len(data_dirs)
        np.save(save_path, total_cutoff_mask)

        logger.info(
            f"Finished processing cutoff date mapping for {len(data_dirs)} directories"
        )

    def construct_cutoff_date_mapping(
        self, blind_or_pretrain: Literal["blind", "pretrain"], cutoff_date: str
    ):
        """Process cutoff date mapping - either single-threaded or with multiprocessing."""
        # If multiple workers requested, use the multiprocessing version

        # Otherwise use the original single-threaded implementation
        if blind_or_pretrain == "blind":
            data_dirs = self.blind_data_dirs
            save_path = os.path.join(
                self.output_location, "blind", "cutoff_date_mapping.npy"
            )
        else:
            data_dirs = self.pretrain_data_dirs
            save_path = os.path.join(
                self.output_location, "pretrain", "cutoff_date_mapping.npy"
            )

        # Convert the cutoff date to a tz-naive timestamp
        cutoff_date = pd.to_datetime(cutoff_date).tz_localize(None)  # type: ignore

        # Create boolean array with length matching number of data directories
        cutoff_mask = np.zeros((len(data_dirs), 2), dtype=bool)

        # Improved tqdm bar
        pbar = tqdm(
            enumerate(data_dirs),
            total=len(data_dirs),
            desc=f"Creating {blind_or_pretrain} cutoff mapping",
        )
        for dataset_idx, data_dir in pbar:
            # Update progress bar with current directory
            pbar.set_postfix({"directory": os.path.basename(data_dir)})

            # Load the data from the parquet file
            parquet_file = next(
                f for f in os.listdir(data_dir) if f.endswith(".parquet")
            )
            df = pd.read_parquet(os.path.join(data_dir, parquet_file))

            # Grab the column name of the timestamp column
            timestamp_col = str(self._get_timestamp_column(data_dir))

            # Ensure the timestamps in the DataFrame are tz-naive
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col]
            ).dt.tz_localize(None)

            if pd.to_datetime(df.iloc[0][timestamp_col]) < cutoff_date:
                cutoff_mask[dataset_idx, 0] = True  # Can be used for training
            if pd.to_datetime(df.iloc[-1][timestamp_col]) > cutoff_date:
                cutoff_mask[dataset_idx, 1] = (
                    True  # Can be used for validation / test
                )
        np.save(save_path, cutoff_mask)


def main():
    parser = argparse.ArgumentParser(
        description="Construct timeseries shards from input data."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Set seed
    np.random.seed(config["seed"])

    # Instantiate TimeseriesShardConstructor with parameters from config
    constructor = TimeseriesShardConstructor(
        input_data=config["input_data"],
        output_location=config["output_location"],
        blind_percentage=config["blind_percentage"],
        shard_size=config["shard_size"],
        device=config["device"],
        use_number=config["use_number"],
    )

    if config["compute_timeseries_shards"]:
        if config["num_workers"] > 1:
            constructor.construct_timeseries_shards_mp(
                blind_or_pretrain=config["split"],
                num_workers=config["num_workers"],
            )
        else:
            constructor.construct_timeseries_shards(
                blind_or_pretrain=config["split"]
            )

    if config["text_embed"]:
        if config["num_workers"] > 1:
            constructor.construct_text_embedding_shards_mp(
                blind_or_pretrain=config["split"],
                num_workers=config["num_workers"],
            )
        else:
            constructor.construct_text_embedding_shards(
                blind_or_pretrain=config["split"]
            )

    if config["compute_cutoff_date_mapping"]:
        if config["num_workers"] > 1:
            constructor.construct_cutoff_date_mapping_mp(
                blind_or_pretrain=config["split"],
                cutoff_date=config["cutoff_date"],
                num_workers=config["num_workers"],
            )
        else:
            constructor.construct_cutoff_date_mapping(
                blind_or_pretrain=config["split"],
                cutoff_date=config["cutoff_date"],
            )

    # Save config file to output location
    config_save_path = os.path.join(
        config["output_location"], config["split"], "config.yaml"
    )
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main()
