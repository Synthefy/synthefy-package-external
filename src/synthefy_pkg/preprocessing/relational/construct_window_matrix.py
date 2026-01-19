import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.configs.relational_config import RelationConfig
from synthefy_pkg.data.sharded_dataloader import ShardedDataloaderV1
from synthefy_pkg.preprocessing.relational.construct_dataset_matrix import (
    load_full_dataset_relation_matrix,
)
from synthefy_pkg.preprocessing.relational.shard_processing_utils import (
    detect_shard_change,
    handle_existing_shards,
    save_idx_matrix,
)
from synthefy_pkg.preprocessing.relational.window_processing_utils import (
    extract_id_from_window_batch,
    extract_times_from_window_batch,
    identify_overlapping_windows,
    identify_window_distance,
    identify_window_overlap_percentage,
)


def merge_relation_matrices(
    dataset_relation_matrix,
    window_relation_matrix,
    window_dataset_matrix,
    lambdas=[1, 1],
    merge_operator="prod",
):
    """
    Merge the dataset relation matrix with the window relation matrix
    Note that the window relation matrix may be a subset of the full relation matrix
    """
    # Extract indices for rows and columns
    row_indices = window_dataset_matrix[
        :, :, 0
    ]  # Shape: (batch_size_1, batch_size_2)
    col_indices = window_dataset_matrix[
        :, :, 1
    ]  # Shape: (batch_size_1, batch_size_2)

    # Get values from dataset relation matrix using these indices
    subset_dataset_relation_matrix = torch.tensor(
        dataset_relation_matrix[row_indices, col_indices],
        device=window_relation_matrix.device,
    )

    # Merge matrices
    merged_relation_matrix = (
        subset_dataset_relation_matrix * window_relation_matrix
        if merge_operator == "prod"
        else subset_dataset_relation_matrix * lambdas[0]
        + window_relation_matrix * lambdas[1]
    )
    return merged_relation_matrix


def compute_window_relation_matrix(
    relation_config: RelationConfig,
    loader_config,
    inner_loop_loader_config,
    dataset_relation_matrix_loader,
    relation_types=["time_overlap"],
    scaling_lambdas=[1],
    combine_operation="prod",
    split="all",
    window_size=None,
    top_k=3000,
    max_batches=20,
    existing_handling="continue",
):
    """
    Compute the relation matrix for window-specific operations
    This runs after the dataset relation matrix is computed and applies a mixing value
    to combine the relation matrix with the window-specific relations

    Args:
        window_path: the path to the window to compute the relation matrix for
        relation_type: the type of relation to compute
    """
    # Compute the pairwise distances between the dataset embeddings

    base_dataloader = ShardedDataloaderV1(
        loader_config, data_dir=relation_config.dataset_dir
    )
    base_dataloader.shuffle = False
    # TODO: possibly make this load in larger chunks (compared to base_dataloader)
    inner_loop_dataloader = ShardedDataloaderV1(
        inner_loop_loader_config, data_dir=relation_config.dataset_dir
    )
    inner_loop_dataloader.shuffle = True
    output_subdir = relation_config.output_subdir

    # load the dataset relation matrix, this might not be feasible in the future
    dataset_relation_matrix = load_full_dataset_relation_matrix(
        dataset_relation_matrix_loader, split, zero_min=True
    )

    if existing_handling == "continue":
        # get the shard indices from the existing files
        existing_shard_indices = handle_existing_shards(
            output_subdir, split, "continue", name="window_relation_matrix"
        )
    elif existing_handling == "clean":
        # delete the existing files
        existing_shard_indices = handle_existing_shards(
            output_subdir, split, "clean", name="window_relation_matrix"
        )
    else:
        existing_shard_indices = set()

    def get_dataloader(dataloader, split):
        if split == "train":
            return dataloader.train_dataloader()
        elif split == "val":
            return dataloader.val_dataloader()
        elif split == "test":
            return dataloader.test_dataloader()
        elif split == "all":
            return dataloader.get_all_dataloader()
        else:
            raise ValueError(f"Invalid split: {split}")

    # iterate through every batch
    current_shard_idx = -1
    outer_dataloader = get_dataloader(base_dataloader, split)
    top_k_stack = list()
    trailing_values = np.ndarray([])
    last_index = -1
    shard_index_stack = list()
    total_saved = 0
    total_saved_num = 0
    for i_batch_idx, batch in tqdm(
        enumerate(outer_dataloader), total=len(outer_dataloader)
    ):
        batch["timeseries"] = batch["timeseries"].reshape(
            -1, batch["timeseries"].shape[-1]
        )[batch["valid_indices"]]
        batch["window_indices"] = torch.tensor(batch["window_indices"])
        batch["shard_indices"] = torch.tensor(batch["shard_indices"])
        current_shard_indices = batch["shard_indices"].unique()
        # if all of the current shard indices are in the existing shard indices, skip this batch
        if all(
            shard_idx in existing_shard_indices
            for shard_idx in current_shard_indices
        ):
            # perform necessary operations to step iterators:
            current_shard_idx = batch["shard_indices"][-1]
            last_index = batch["window_indices"][-1]
            logger.info(
                f"skipping batch {i_batch_idx} from shard {current_shard_idx} window {last_index} because it already exists"
            )
            continue

        shard_index_stack.append(batch["shard_indices"])
        start_times_1, end_times_1 = extract_times_from_window_batch(
            batch,
            window_size,
        )
        start_ids = extract_id_from_window_batch(batch)
        start_ids = start_ids.reshape(-1, 1)  # Shape: (batch_size_1, 1)

        inner_dataloader = get_dataloader(
            inner_loop_dataloader, split
        )  # TODO: possibly make this load in larger chunks (compared to base_dataloader)
        batch_relation_rows = np.array([])
        batch_index_rows = list()
        batch_dataset_rows = list()

        total_batches = min(max_batches, len(inner_dataloader))
        if current_shard_idx == -1:
            current_shard_idx = batch["shard_indices"][0]

        # compare every batch to every other batch
        for j_batch_idx, other_batch in tqdm(
            enumerate(inner_dataloader), total=total_batches
        ):
            if j_batch_idx >= total_batches:
                break

            # get the valid indices
            other_batch["timeseries"] = other_batch["timeseries"].reshape(
                -1, other_batch["timeseries"].shape[-1]
            )[other_batch["valid_indices"]]
            other_batch["window_indices"] = torch.tensor(
                other_batch["window_indices"]
            )
            other_batch["shard_indices"] = torch.tensor(
                other_batch["shard_indices"]
            )
            batch_index_rows.append(other_batch["window_indices"])

            # initialize the batch chunk
            batch_chunk = torch.zeros(
                (len(batch["timeseries"]), len(other_batch["timeseries"])),
                device=relation_config.device,
            )

            # get the start and end times for the other batch
            start_times_2, end_times_2 = extract_times_from_window_batch(
                other_batch, window_size
            )
            end_ids = extract_id_from_window_batch(other_batch)
            # Reshape the arrays to match dimensions before concatenation
            end_ids = end_ids.reshape(-1, 1)  # Shape: (batch_size_2, 1)

            # Create matrices of repeated IDs to match dimensions
            end_ids_matrix = torch.tile(
                end_ids.T, (len(start_ids), 1)
            )  # Shape: (batch_size_1, batch_size_2)
            start_ids_matrix = torch.tile(
                start_ids, (1, len(end_ids))
            )  # Shape: (batch_size_1, batch_size_2)

            # Stack the matrices along a new axis
            window_dataset_matrix = torch.stack(
                [start_ids_matrix, end_ids_matrix], dim=-1
            )

            for scaling_lambda, relation_type in zip(
                scaling_lambdas, relation_types
            ):
                if relation_type == "time_overlap":
                    # gives 1 for windows that overlap in time, 0 otherwise
                    window_idx_matrix = identify_overlapping_windows(
                        start_times_1,
                        end_times_1,
                        start_times_2,
                        end_times_2,
                        device=relation_config.device,
                    )
                    batch_chunk = (
                        batch_chunk + window_idx_matrix * scaling_lambda
                    )

                elif relation_type == "time_proximity":
                    # gives 1 for windows that are close in time, 0 otherwise
                    window_idx_matrix = identify_window_distance(
                        start_times_1,
                        end_times_1,
                        start_times_2,
                        end_times_2,
                        device=relation_config.device,
                    )
                    batch_chunk = (
                        batch_chunk + window_idx_matrix * scaling_lambda
                    )

                elif relation_type == "overlap_percentage":
                    # gives the percentage of overlap between two windows
                    window_idx_matrix = identify_window_overlap_percentage(
                        start_times_1,
                        end_times_1,
                        start_times_2,
                        end_times_2,
                        device=relation_config.device,
                    )
                    batch_chunk = (
                        batch_chunk + window_idx_matrix * scaling_lambda
                    )
            # merge the window relation matrix with the dataset relation matrix, loading in the appropraite rows corresponding to datasets
            relation_matrix = (
                merge_relation_matrices(
                    dataset_relation_matrix,
                    batch_chunk,
                    window_dataset_matrix,
                    lambdas=[1, 1],
                    merge_operator=combine_operation,
                )
                .cpu()
                .numpy()
            )
            batch_dataset_rows.append(end_ids)
            if batch_relation_rows.shape[0] == 0:
                batch_relation_rows = relation_matrix
            else:
                batch_relation_rows = np.concatenate(
                    [batch_relation_rows, relation_matrix], axis=-1
                )

        # Next, save the batch relation rows
        batch_index_rows = np.concatenate(batch_index_rows, axis=-1)
        batch_dataset_rows = np.concatenate(batch_dataset_rows, axis=0)[:, 0]

        # to reduce the amount of sorting, sort each row individually and remove zero values
        # import time
        # start_time = time.time()
        # top_k_indices = np.array([])
        # removed = 0
        # for i in range(batch_relation_rows.shape[0]):
        #     removed += np.sum(batch_relation_rows[i] == 0)
        #     single_batch_index_row = batch_index_rows[batch_relation_rows[i] != 0]
        #     single_batch_relation_row = batch_relation_rows[i][batch_relation_rows[i] != 0]
        #     top_k_indices = np.argpartition(single_batch_relation_row, -top_k)[-top_k:]
        #     top_k_indices = np.concatenate([top_k_indices, single_batch_index_row[top_k_indices]])
        # logger.info(f"Processing removed {removed} zero values with min {np.min(batch_relation_rows)} and max {np.max(batch_relation_rows)}")
        # looped_end_time = time.time()

        # get the top k indices for each index, this operation is expensive
        top_k_indices = np.argpartition(batch_relation_rows, -top_k, axis=-1)[
            :, -top_k:
        ]

        row_indices = np.arange(batch_relation_rows.shape[0])[:, np.newaxis]
        logger.info(
            np.mean(batch_relation_rows[row_indices, top_k_indices], axis=-1)
        )

        top_k_indices = batch_index_rows[top_k_indices]
        logger.info(
            top_k_indices[np.random.randint(0, top_k_indices.shape[0]), :10]
        )
        top_k_stack.append(top_k_indices)
        total_stack = np.concatenate(top_k_stack, axis=0)

        if detect_shard_change(batch["shard_indices"]):
            # save the top k indices
            new_trailing_values, last_index, saved_num = save_idx_matrix(
                trailing_values,
                last_index,
                current_shard_idx,
                np.concatenate(shard_index_stack, axis=0),
                total_stack,
                "window_relation_matrix",
                output_subdir,
                split,
            )
            total_saved += (
                total_stack.shape[0]
                - new_trailing_values.shape[0]
                + (
                    trailing_values.shape[0]
                    if len(trailing_values.shape) > 0
                    else 0
                )
            )
            trailing_values = new_trailing_values
            total_saved_num += saved_num
            logger.info(
                f"total saved {total_saved}, {total_saved_num} with trailing {trailing_values.shape} and saved num {saved_num}"
            )
            current_shard_idx = batch["shard_indices"][-1]
            top_k_stack = list()
            shard_index_stack = list()
    if len(top_k_stack) > 0:
        # save the last top k indices
        trailing_values, last_index, saved_num = save_idx_matrix(
            trailing_values,
            last_index,
            current_shard_idx,
            np.concatenate(shard_index_stack, axis=0),
            np.concatenate(top_k_stack, axis=0),
            "window_relation_matrix",
            output_subdir,
            split,
            force=True,
        )
        total_saved_num += saved_num
        logger.info(
            f"total saved {total_saved}, {total_saved_num} with trailing {trailing_values.shape} and saved num {saved_num}"
        )
