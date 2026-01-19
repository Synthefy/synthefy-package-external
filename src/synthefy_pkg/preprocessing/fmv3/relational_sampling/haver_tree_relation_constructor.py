import concurrent.futures
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.preprocessing.fmv3.construct_shards import (
    TimeseriesShardConstructor,
)
from synthefy_pkg.preprocessing.fmv3.relational_sampling.base_relation_constructor import (
    BaseRelationConstructor,
)


class HaverTreeRelationConstructor(BaseRelationConstructor):
    """
    Haver tree relation constructor.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_data_dir = os.path.join(
            config["output_location"], config["split"]
        )
        self.input_data_dirs = config["input_data"]

        self.self_sampling_probability = config["self_sampling_probability"]

        self.relation_level = config["relation_level"]

        if self.relation_level == 0:
            super().__init__(self.output_data_dir, "haver_tree_0")
        elif self.relation_level == 1:
            super().__init__(self.output_data_dir, "haver_tree_1")
        else:
            raise ValueError(f"Invalid relation level: {self.relation_level}")

    @staticmethod
    def hash_text_description(
        chunk: list[str], worker_id: int
    ) -> list[Tuple[int, int]]:
        """
        Get 2 hashes from the text description.

        The first hash is the has of the first level.
        The second hash is the hash of the second level.
        """
        # Load the data from the json file
        hash_vals = []

        # Add progress bar for processing the chunk
        pbar = tqdm(
            chunk,
            desc=f"Worker {worker_id}",
            position=worker_id,
            leave=False,
        )

        for idx, data_dir in enumerate(pbar):
            json_file = next(
                f for f in os.listdir(data_dir) if f.endswith("metadata.json")
            )
            with open(os.path.join(data_dir, json_file), "r") as f:
                metadata = json.load(f)

            # Get the column description
            column_description = str(metadata["columns"][0]["description"])

            levels = column_description.split(":")

            if len(levels) < 1:
                hash_val = 0, 0
            else:
                hash_val = (
                    os.path.basename(os.path.dirname(data_dir)),
                    levels[0].lower().strip(),
                )
            hash_vals.append(hash_val)

        return hash_vals

    def _construct_relational_matrix(self, num_major_categories: int):
        # Create matrix with self_sampling_probability on diagonal
        relational_matrix = np.zeros(
            (num_major_categories, num_major_categories)
        )
        np.fill_diagonal(relational_matrix, self.self_sampling_probability)

        # Fill non-self elements such that rows sum to 1
        other_category_prob = (1 - self.self_sampling_probability) / (
            num_major_categories - 1
        )
        relational_matrix[~np.eye(num_major_categories, dtype=bool)] = (
            other_category_prob
        )

        return relational_matrix

    def _get_and_hash_text_descriptions(self) -> np.ndarray:
        """
        Get and hash the text descriptions from the data directory.
        """
        shard_constructor = TimeseriesShardConstructor(
            input_data=self.input_data_dirs,
            output_location=self.output_data_dir,
            blind_percentage=self.config["blind_percentage"],
            shard_size=self.config["shard_size"],
            device=self.config["device"],
        )

        """Process data with a cleaner progress display"""
        blind_or_pretrain = self.config["split"]
        if blind_or_pretrain == "blind":
            data_dirs = shard_constructor.blind_data_dirs
        else:
            data_dirs = shard_constructor.pretrain_data_dirs

        # Print information about the processing task
        logger.info(
            f"Processing {len(data_dirs)} directories for {self.config['split']} split"
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

        chunks = shard_constructor._partition(
            data_dirs, self.config["num_workers"]
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config["num_workers"]
        ) as executor:
            # Submit all tasks
            futures = []
            future_to_worker = {}  # Map futures to worker indices
            for i, chunk in enumerate(chunks):
                # Process each chunk without its own progress bar
                future = executor.submit(
                    HaverTreeRelationConstructor.hash_text_description,
                    chunk,
                    i,  # Pass worker_id
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
                for i in range(len(data_dirs))
                if i in worker_results
            ]

        # Flatten list of lists using numpy concatenate
        results = np.concatenate(results)

        return results

    def construct_relations(self) -> None:
        """
        Construct relations from the data.
        """

        """
        Instantiate a timeseries shard constructor.

        We only do this because we want to do data_dir preprocessing identically to how
        we do during the actual haver preprocessing steps.
        """
        results = self._get_and_hash_text_descriptions()

        if self.relation_level == 0:
            categories = np.unique(results[:, 0])
        elif self.relation_level == 1:
            categories = np.unique(results[:, 1])
        else:
            raise ValueError(f"Invalid relation level: {self.relation_level}")

        num_categories = len(categories)

        # Create mapping from major category to index
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        # Map each result's major category to its index
        classification_array = np.array(
            [category_to_idx[cat] for cat in results[:, self.relation_level]]
        )

        # Construct relational matrix
        relational_matrix = self._construct_relational_matrix(num_categories)

        # Convert float to string with underscores instead of decimal point
        prob_str = str(self.self_sampling_probability).replace(".", "_")
        self.save_numpy_files(classification_array, relational_matrix, prob_str)
