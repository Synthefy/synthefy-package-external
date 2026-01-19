import argparse
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.preprocessing.relational.construct_dataset_matrix import (
    embed_time_series_from_enriched,
)
from synthefy_pkg.preprocessing.relational.construct_relational_matrix import (
    load_relational_config,
)


def create_unembed_dataset_dict(
    dataset_dir,
    relation_config,
):
    dataset_dict = embed_time_series_from_enriched(
        dataset_dir,
        relation_config,
        keep_time_series=True,
    )
    zero_variance_series = dataset_dict["zero_variance_series"]
    logger.info(
        f"zero_variance_series: {np.sum(zero_variance_series.astype(int))} percentage: {np.sum(zero_variance_series.astype(int)) / len(zero_variance_series)}"
    )
    return dataset_dict


def process_single_dataset(args):
    """Process a single dataset for parallelization"""
    i, index, dataset_dir, output_dir = args

    basename = output_dir.split("/")[-1]
    home_basename = dataset_dir.split("/")[-1]
    target_dir = os.path.join(output_dir, f"{basename}_{i}")
    home_dir = os.path.join(dataset_dir, f"{home_basename}_{index}")

    try:
        # clean dir before copying
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(home_dir, target_dir)

        # revalue the dataset id in the preprocessed_array to i from index
        processed_array_path = os.path.join(home_dir, "preprocessed_array.npy")
        if os.path.exists(processed_array_path):
            processed_array = np.load(processed_array_path)
            processed_array[:, -1] = i
            np.save(
                os.path.join(target_dir, "preprocessed_array.npy"),
                processed_array,
            )

        # rename metadata and parquet to i from index, and new basename
        old_metadata_name = os.path.join(
            target_dir, f"{home_basename}_{index}_metadata.json"
        )
        metadata_name = os.path.join(
            target_dir, f"{basename}_{i}_metadata.json"
        )
        if os.path.exists(old_metadata_name):
            os.rename(old_metadata_name, metadata_name)

        old_parquet_name = os.path.join(
            target_dir, f"{home_basename}_{index}.parquet"
        )
        parquet_name = os.path.join(target_dir, f"{basename}_{i}.parquet")
        if os.path.exists(old_parquet_name):
            os.rename(old_parquet_name, parquet_name)

        return i, index, "Success"
    except Exception as e:
        return i, index, f"Error: {str(e)}"


def create_filtered_enriched_data(
    dataset_dir, output_dir, non_zero_variance_indices, max_workers=16
):
    """
    Creates a new dataset with the zero variance series removed, using parallel processing

    Args:
        dataset_dir: Source directory containing original datasets
        output_dir: Target directory for filtered datasets
        non_zero_variance_indices: Indices of non-zero variance datasets to keep
        max_workers: Number of parallel workers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for parallel processing
    args_list = [
        (i, index, dataset_dir, output_dir)
        for i, index in enumerate(non_zero_variance_indices)
    ]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_dataset, args) for args in args_list
        ]

        # Track progress and collect results
        results = []
        for future in tqdm(
            futures, total=len(futures), desc="Processing datasets"
        ):
            i, index, status = future.result()
            results.append((i, index, status))

            # Log results
            if status == "Success":
                logger.info(f"Copied dataset {index} to {i}")
            else:
                logger.error(f"Failed to copy dataset {index} to {i}: {status}")

    # Summary
    success_count = sum(1 for _, _, status in results if status == "Success")
    logger.info(
        f"Successfully processed {success_count} out of {len(non_zero_variance_indices)} datasets"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--enriched_output_dir", type=str)
    parser.add_argument("--load_data", action="store_true")
    parser.add_argument("--relation_config", type=str, default="")
    args = parser.parse_args()

    if args.relation_config:
        relation_config = load_relational_config(
            args.relation_config, args.dataset_dir, args.output_dir
        )
    else:
        relation_config = None

    if args.load_data:
        dataset_dict = create_unembed_dataset_dict(
            args.dataset_dir, relation_config
        )
        os.makedirs(args.output_dir, exist_ok=True)
        pickle.dump(
            dataset_dict,
            open(os.path.join(args.output_dir, "dataset_dict.pkl"), "wb"),
        )
        print(
            dataset_dict["dataset_ids"].shape,
            dataset_dict["zero_variance_series"].shape,
        )
        pickle.dump(
            dataset_dict["dataset_ids"][
                dataset_dict["zero_variance_series"].astype(bool)
            ],
            open(
                os.path.join(args.output_dir, "zero_variance_indices.pkl"), "wb"
            ),
        )
        pickle.dump(
            dataset_dict["dataset_ids"][
                ~dataset_dict["zero_variance_series"].astype(bool)
            ],
            open(
                os.path.join(args.output_dir, "non_zero_variance_indices.pkl"),
                "wb",
            ),
        )
        logger.info(
            f"Saved zero variance indices to {args.output_dir}, dataset_dict.pkl, zero_variance_indices.pkl, non_zero_variance_indices.pkl"
        )
    else:
        dataset_dict = pickle.load(
            open(os.path.join(args.output_dir, "dataset_dict.pkl"), "rb")
        )
        zero_variance_indices = pickle.load(
            open(
                os.path.join(args.output_dir, "zero_variance_indices.pkl"), "rb"
            )
        )
        non_zero_variance_indices = pickle.load(
            open(
                os.path.join(args.output_dir, "non_zero_variance_indices.pkl"),
                "rb",
            )
        )
    print(
        "total_datasets",
        np.sum([len(ts) for ts in dataset_dict["time_series_full"]]),
    )
    if args.enriched_output_dir:
        create_filtered_enriched_data(
            args.dataset_dir,
            args.enriched_output_dir,
            non_zero_variance_indices,
        )

    # run to compute all dataset values
    # uv run src/synthefy_pkg/preprocessing/relational/create_zero_variance_list.py --dataset_dir /home/data/enriched_datasets/all_univariate --output_dir /home/data/all_univariate_dataset_dict/ --load_data
    # run with embeddings:
    # uv run src/synthefy_pkg/preprocessing/relational/create_zero_variance_list.py --dataset_dir /home/data/enriched_datasets/all_univariate_filtered --output_dir /home/data/all_univariate_filtered_dataset_dict/ --load_data --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v3_series_text_fourier_embed.yaml
    # run to create enriched data (data already generated):
    # uv run src/synthefy_pkg/preprocessing/relational/create_zero_variance_list.py --dataset_dir /home/data/enriched_datasets/all_univariate --output_dir /home/data/all_univariate_dataset_dict/ --enriched_output_dir /home/data/all_univariate_filtered
