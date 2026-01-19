import glob
import io
import os
import pickle
import tarfile

import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.configs.relational_config import RelationConfig
from synthefy_pkg.data.sharded_dataloader import ShardedDataloaderV1
from synthefy_pkg.preprocessing.relational.construct_dataset_matrix import (
    compute_dataset_relation_matrix,
    construct_dataset_embeddings_from_lookup_table,
    embed_time_series_from_enriched,
    handle_dataset_ids,
    load_full_dataset_relation_matrix,
    load_relational_dataset_matrix,
    relative_relation_matrix,
)
from synthefy_pkg.preprocessing.relational.construct_window_matrix import (
    compute_window_relation_matrix,
)
from synthefy_pkg.utils.retrieve_dataset_descriptions import (
    load_dataset_description,
)


def load_relational_config(
    relation_config_path: str, dataset_dir: str, output_dir: str
):
    with open(relation_config_path, "r") as f:
        relation_config = yaml.safe_load(f)
    relation_config["dataset_loader_config"]["dataset_dir"] = dataset_dir
    relation_config["dataset_loader_config"]["output_subdir"] = (
        output_dir if output_dir == "" else output_dir
    )
    relation_config["dataset_loader_config"]["dataset_name"] = dataset_dir
    relation_config["window_loader_config"]["dataset_name"] = dataset_dir
    relation_config["inner_loop_window_loader_config"]["dataset_name"] = (
        dataset_dir
    )
    relation_config["output_subdir"] = (
        dataset_dir if output_dir == "" else output_dir
    )

    relation_config = RelationConfig(relation_config)
    relation_config.dataset_loader_config.dataset_config.device = (
        relation_config.device
    )
    relation_config.dataset_loader_config.dataset_config.num_workers = (
        relation_config.num_workers
    )
    relation_config.window_loader_config.dataset_config.device = (
        relation_config.device
    )
    relation_config.window_loader_config.dataset_config.num_workers = (
        relation_config.num_workers
    )
    relation_config.inner_loop_window_loader_config.dataset_config.device = (
        relation_config.device
    )
    relation_config.inner_loop_window_loader_config.dataset_config.num_workers = relation_config.num_workers
    return relation_config


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory of the dataset to load from (after mix and split)",
    )
    parser.add_argument(
        "--enriched_dataset_dir",
        type=str,
        required=True,
        help="The directory of the dataset to load from (after preprocessing)",
    )
    parser.add_argument(
        "--relation_config",
        type=str,
        required=True,
        help="The path to the relation config file",
    )
    parser.add_argument(
        "--lookup_table_dir",
        type=str,
        default="",
        help="The path to the lookup table",
    )
    parser.add_argument(
        "--value_to_compute",
        type=str,
        default="compute_window",
        help="Whether to load the full dataset embeddings",
    )
    parser.add_argument(
        "--load_relational_matrix",
        type=int,
        default=0,
        help="The number of rows to load from the dataset relation matrix",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="",
        help="The directory to save the output to",
    )
    parser.add_argument(
        "--dataset_ids",
        type=str,
        default="",
        help="The ids of the datasets to load from the dataset relation matrix (default selects random ids)",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default="0",
        help="The uid of the dataset relation matrix",
    )
    parser.add_argument(
        "--existing_handling",
        type=str,
        default="continue",
        help="The existing handling strategy (continue or clean)",
    )
    parser.add_argument(
        "--slice_dataset_matrix",
        type=int,
        default=-1,
        help="The number of rows to slice from the dataset relation matrix",
    )
    parser.add_argument(
        "--load_dataset_dict",
        type=str,
        default="",
        help="where to load the dataset dict from",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    NUM_DATASETS = 226355

    relation_config = load_relational_config(
        args.relation_config, args.dataset_dir, args.output_dir
    )
    splits = relation_config.splits
    window_size = relation_config.window_size
    dataset_relation_types = relation_config.dataset_relation_types
    dataset_scaling_lambdas = relation_config.dataset_scaling_lambdas
    dataset_combine_operation = relation_config.dataset_combine_operation

    if args.value_to_compute == "load_relational_matrix":
        # TODO: I don't think this actually loads for anything at the moment
        for split in splits:
            # this option means that we are loading a subset of the dataset relation matrix
            dataset_id_vals = handle_dataset_ids(
                args.dataset_ids, NUM_DATASETS, args.load_relational_matrix
            )
            load_relational_dataset_matrix(
                dataset_id_vals,
                relation_config,
                args.output_dir,
                split=split,
                uid=args.uid,
            )
        exit()

    if args.value_to_compute == "compute_full_dataset_embeddings":
        # This option means to load in all of the data
        if args.slice_dataset_matrix > 0:
            indices = np.arange(args.slice_dataset_matrix)
        else:
            indices = np.array([])
        dataset_dict = embed_time_series_from_enriched(
            args.enriched_dataset_dir,
            relation_config,
            keep_time_series=True,
            indices=indices,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        pickle.dump(
            dataset_dict,
            open(
                os.path.join(
                    args.output_dir, "full_dataset_embeddings_lookup.pkl"
                ),
                "wb",
            ),
        )

    if args.value_to_compute == "compute_dataset_relation_matrix":
        if len(args.load_dataset_dict) != 0:
            if args.load_dataset_dict == "OUTPUT":
                dataset_id_vals = np.load(
                    os.path.join(args.output_dir, "partial_dataset_ids.npy")
                )
                dataset_dict = pickle.load(
                    open(
                        os.path.join(args.output_dir, "dataset_dict.pkl"), "rb"
                    )
                )
            else:
                dataset_id_vals = np.load(
                    os.path.join(
                        args.load_dataset_dict,
                        "partial_dataset_ids.npy",
                    )
                )
                dataset_dict = pickle.load(
                    open(
                        os.path.join(
                            args.load_dataset_dict, "dataset_dict.pkl"
                        ),
                        "rb",
                    )
                )
        else:
            dataset_id_vals = handle_dataset_ids(
                args.dataset_ids, NUM_DATASETS, args.load_relational_matrix
            )
            dataset_dict = embed_time_series_from_enriched(
                args.enriched_dataset_dir,
                relation_config,
                keep_time_series=True,
                indices=dataset_id_vals,
            )
            os.makedirs(args.output_dir, exist_ok=True)
            pickle.dump(
                dataset_dict,
                open(
                    os.path.join(args.output_dir, "dataset_dict.pkl"),
                    "wb",
                ),
            )
            np.save(
                os.path.join(args.output_dir, "partial_dataset_ids.npy"),
                dataset_dict["dataset_ids"],
            )
        for split in splits:
            compute_dataset_relation_matrix(
                dataset_dict,
                dataset_relation_types,
                dataset_scaling_lambdas,
                dataset_combine_operation,
                output_tar_path=relation_config.output_subdir,
                chunk_size=relation_config.dataset_chunk_size,
                split=split,
                device=relation_config.device,
                existing_handling=args.existing_handling,
            )
            dataset_relation_matrix_loader = ShardedDataloaderV1(
                relation_config.dataset_loader_config,
                name_specification="_dataset_relation_matrix",
                data_dir=relation_config.output_subdir,
            )
            dataset_relation_matrix = load_full_dataset_relation_matrix(
                dataset_relation_matrix_loader, split
            )
            matrix_name = f"{split}_relation_matrix_{args.uid}.npy"
            np.save(
                os.path.join(
                    args.output_dir,
                    matrix_name,
                ),
                dataset_relation_matrix,
            )
            logger.info(
                f"Saved partial relation matrix to {args.output_dir}/{matrix_name}"
            )
        # Partial relation matrices:
        # uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v3_series_cross_corr.yaml --dataset_dir /home/data/enriched_datasets/all_univariate_filtered/ --output_dir /home/data/relation_10k_pairs --load_relational_matrix 1000 --value_to_compute compute_dataset_relation_matrix --existing_handling clean --uid cross_correlation
        # uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v3_series_text_cosine.yaml --dataset_dir /home/data/enriched_datasets/all_univariate_filtered/ --output_dir /home/data/relation_10k_pairs_2 --load_relational_matrix 1000 --value_to_compute compute_dataset_relation_matrix --existing_handling clean --uid cosine_dist --load_dataset_dict /home/data/relation_10k_pairs/

    if args.value_to_compute == "compute_window_relation_matrix":
        for split in splits:
            if len(args.lookup_table_dir) != 0:
                lookup_table_dir = args.lookup_table_dir
            else:
                lookup_table_dir = args.dataset_dir
            dataset_list = load_dataset_description(
                None, lookup_table_dir, None, return_idxes=[0, 1, 2]
            )
            dataset_dict = embed_time_series_from_enriched(
                args.enriched_dataset_dir,
                relation_config,
                keep_time_series=True,
                lookup_table=dataset_list,
            )
            compute_dataset_relation_matrix(
                dataset_dict,
                dataset_relation_types,
                dataset_scaling_lambdas,
                dataset_combine_operation,
                output_tar_path=relation_config.output_subdir,
                chunk_size=relation_config.dataset_chunk_size,
                split=split,
                device=relation_config.device,
                existing_handling=args.existing_handling,
            )

            if args.slice_dataset_matrix > 0:
                dataset_relation_matrix_loader = ShardedDataloaderV1(
                    relation_config.dataset_loader_config,
                    name_specification="_dataset_relation_matrix",
                    data_dir=relation_config.output_subdir,
                )
                dataset_relation_matrix = load_full_dataset_relation_matrix(
                    dataset_relation_matrix_loader, split
                )

            dataset_relation_matrix_loader = ShardedDataloaderV1(
                relation_config.dataset_loader_config,
                name_specification="_dataset_relation_matrix",
                data_dir=relation_config.output_subdir,
            )

            loader_config = relation_config.window_loader_config
            inner_loop_loader_config = (
                relation_config.inner_loop_window_loader_config
            )
            window_relation_types = relation_config.window_relation_types
            window_scaling_lambdas = relation_config.window_scaling_lambdas
            window_combine_operation = relation_config.window_combine_operation

            compute_window_relation_matrix(
                relation_config,
                loader_config,
                inner_loop_loader_config,
                dataset_relation_matrix_loader,
                relation_types=window_relation_types,
                scaling_lambdas=window_scaling_lambdas,
                combine_operation=window_combine_operation,
                split=split,
                window_size=window_size,
                max_batches=relation_config.window_max_batches,
                existing_handling=args.existing_handling,
            )

# run commands:
# Constructing and saving the dataset relational matrix
# uv run src/synthefy_pkg/preprocessing/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation.yaml --dataset_dir /home/data/enriched_datasets/all_univariate --output_dir /home/data/relation_all --full-dataset-embeddings

# Loading the relational matrix (10k by 269115):
# uv run src/synthefy_pkg/preprocessing/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation.yaml --dataset_dir /home/data/enriched_datasets/all_univariate --output_dir /home/data/relation_all --load_relational_matrix 10000

# can convert to 10k by 10k
# uv run src/synthefy_pkg/preprocessing/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation.yaml --dataset_dir /home/data/enriched_datasets/all_univariate --output_dir /home/data/relation_all --slice_dataset_matrix 10000

# Constructing the k nearest windows matrix for a smaller dataset

# Testing with windowing with a 1k dataset:
# uv run src/synthefy_pkg/preprocessing/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation.yaml --dataset_dir /home/data/foundation_model_data_all_univariate_1000_0_1000/pretrain --output_dir /home/data/relation_test_1k

# uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation_cross_corr.yaml --dataset_dir /home/data/foundation_model_data_all_univariate_filtered_10000_0_10000_ts_2021-06-01/pretrain --output_dir /home/data/relation_10k_filtered_cross_corr_overlap --value_to_compute "compute_dataset_relation_matrix" --enriched_dataset_dir /home/data/enriched_datasets/all_univariate_filtered/

# Running on a 10k dataset
# uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation.yaml --dataset_dir /home/data/foundation_model_data_all_univariate_filtered_10000_0_10000_ts_2021-06-01/pretrain --output_dir /home/data/relation_10k_filtered_cos_overlap --value_to_compute "compute_window_relation_matrix" --enriched_dataset_dir /home/data/enriched_datasets/all_univariate_filtered/
# uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation_granger.yaml --dataset_dir /home/data/foundation_model_data_all_univariate_filtered_10000_0_10000_ts_2021-06-01/pretrain --output_dir /home/data/relation_10k_filtered_granger_overlap --value_to_compute "compute_window_relation_matrix" --enriched_dataset_dir /home/data/enriched_datasets/all_univariate_filtered/
# uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation_cross_corr.yaml --dataset_dir /home/data/foundation_model_data_all_univariate_filtered_10000_0_10000_ts_2021-06-01/pretrain --output_dir /home/data/relation_10k_filtered_cross_corr_overlap --value_to_compute "compute_window_relation_matrix" --enriched_dataset_dir /home/data/enriched_datasets/all_univariate_filtered/
# uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation_umap_text.yaml --dataset_dir /home/data/foundation_model_data_all_univariate_filtered_10000_0_10000_ts_2021-06-01/pretrain --output_dir /home/data/relation_10k_filtered_umap_text_overlap --value_to_compute "compute_window_relation_matrix" --enriched_dataset_dir /home/data/enriched_datasets/all_univariate_filtered/
# uv run src/synthefy_pkg/preprocessing/relational/construct_relational_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation_fourier.yaml --dataset_dir /home/data/foundation_model_data_all_univariate_filtered_10000_0_10000_ts_2021-06-01/pretrain --output_dir /home/data/relation_10k_filtered_fourier_overlap --value_to_compute "compute_window_relation_matrix" --enriched_dataset_dir /home/data/enriched_datasets/all_univariate_filtered/

# saved a lookup pickle at

# /home/data/relation_all/full_dataset_embeddings_lookup.pkl
