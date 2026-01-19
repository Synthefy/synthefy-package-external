import os

import holidays
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

import synthefy_pkg.utils.fm_utils as fm_utils
from synthefy_pkg.configs.relational_config import RelationConfig
from synthefy_pkg.data.sharded_dataloader import ShardedDataloaderV1
from synthefy_pkg.preprocessing.fmv2_preprocess import (
    NORM_RANGES,
    TEXT_EMBEDDING_DIM,
    TIMESTAMPS_FEATURES,
    convert_time_to_vector,
)
from synthefy_pkg.preprocessing.relational.construct_series_matrix import (
    compute_series_similarity,
    reduce_dimensions,
    timeseries_to_fixed_vector,
)
from synthefy_pkg.preprocessing.relational.shard_processing_utils import (
    handle_existing_shards,
    save_idx_matrix,
)
from synthefy_pkg.preprocessing.relational.time_series_comparison_utils import (
    SERIES_COMPARISON_METHODS,
)


def relative_relation_matrix(relation_dataset_matrix, dataset_ids):
    """
    Compute the relation matrix for only the datasets in dataset_ids
    """
    relative_relation_matrix = relation_dataset_matrix[:, dataset_ids]
    return relative_relation_matrix


def load_full_dataset_relation_matrix(
    dataset_relation_matrix_loader, split, select_dataset_ids=[], zero_min=False
):
    dataset_relation_matrix_loader.shuffle = False
    if split == "train":
        dataset_relation_matrix_loader = (
            dataset_relation_matrix_loader.train_dataloader()
        )
    elif split == "val":
        dataset_relation_matrix_loader = (
            dataset_relation_matrix_loader.val_dataloader()
        )
    elif split == "test":
        dataset_relation_matrix_loader = (
            dataset_relation_matrix_loader.test_dataloader()
        )
    else:
        dataset_relation_matrix_loader = (
            dataset_relation_matrix_loader.get_all_dataloader()
        )

    loaded_values = np.ndarray([])
    counter = 0
    batch_counter = 0
    batch_size = dataset_relation_matrix_loader.batch_size
    batch_id_for_select_dataset_ids = [
        select_dataset_ids[i] // batch_size
        for i in range(len(select_dataset_ids))
    ]
    check_batch_ids = np.unique(batch_id_for_select_dataset_ids)
    check_batch_ids = np.sort(check_batch_ids)
    total_added = 0
    for batch in tqdm(
        dataset_relation_matrix_loader,
        total=len(dataset_relation_matrix_loader),
    ):
        batch = (
            batch["timeseries"]
            .reshape(-1, batch["timeseries"].shape[-1])[batch["valid_indices"]]
            .cpu()
            .numpy()
        )
        counter += len(batch)
        if len(select_dataset_ids) > 0:
            if batch_counter in check_batch_ids:
                dataset_ids_in_batch = [
                    id % batch_size
                    for id, batch_id in zip(
                        select_dataset_ids, batch_id_for_select_dataset_ids
                    )
                    if batch_id == batch_counter
                ]
                # print(list(zip(dataset_ids_in_batch, [
                #     id
                #     for id, batch_id in zip(
                #         select_dataset_ids, batch_id_for_select_dataset_ids
                #     )
                #     if batch_id == batch_counter
                # ])))
                total_added += len(dataset_ids_in_batch)
                if len(loaded_values.shape) == 0:
                    loaded_values = batch[dataset_ids_in_batch]
                else:
                    loaded_values = np.concatenate(
                        [loaded_values, batch[dataset_ids_in_batch]], axis=0
                    )
        else:
            if len(loaded_values.shape) == 0:
                loaded_values = batch
            else:
                loaded_values = np.concatenate([loaded_values, batch], axis=0)
        batch_counter += 1
    print("loaded in", loaded_values.shape)
    if zero_min:
        invalid_indices = loaded_values == -9999
        loaded_values = loaded_values - np.min(loaded_values[~invalid_indices])
        loaded_values[invalid_indices] = 0
    return loaded_values


def compute_dataset_relation_matrix(
    dataset_dict: dict[str, torch.Tensor],
    relation_types=["dataset_cosine"],
    scaling_lambdas=[1],
    combine_operation="sum",
    output_tar_path: str | None = None,
    chunk_size: int = 2000,
    shard_size: int = 200,
    split: str = "train",
    device="cpu",
    existing_handling="continue",
):
    """
    Compute the relation matrix of the dataset embeddings in a streaming fashion.
    If output_tar_path is provided, saves results directly to tar file instead of returning matrix.
    """
    n_datasets = len(dataset_dict["dataset_embeddings"])

    # If saving to tar, create the tar file

    # Pre-compute any reusable values
    dataset_embeddings = torch.zeros((n_datasets, 0)).to(device)
    series_embeddings = torch.zeros((n_datasets, 0)).to(device)
    reduced_embeddings = torch.zeros((n_datasets, 0)).to(device)
    reduced_series_embeddings = torch.zeros((n_datasets, 0)).to(device)
    embedding_dict = {}
    if "text_cosine" in relation_types or "text_euclidean" in relation_types:
        dataset_embeddings = dataset_dict["dataset_embeddings"]
        dataset_embeddings = torch.tensor(dataset_embeddings).to(device)
        embedding_dict["text_cosine"] = dataset_embeddings
        embedding_dict["text_euclidean"] = dataset_embeddings
    if (
        "series_cosine" in relation_types
        or "series_euclidean" in relation_types
    ):
        series_embeddings = dataset_dict["time_series_embeddings"]
        series_embeddings = torch.tensor(series_embeddings).to(device)
        embedding_dict["series_cosine"] = series_embeddings
        embedding_dict["series_euclidean"] = series_embeddings
    if (
        "reduced_text_cosine" in relation_types
        or "reduced_text_euclidean" in relation_types
    ):
        reduced_embeddings = dataset_dict["reduced_embeddings"]
        reduced_embeddings = torch.tensor(reduced_embeddings).to(device)
        embedding_dict["reduced_text_cosine"] = reduced_embeddings
        embedding_dict["reduced_text_euclidean"] = reduced_embeddings
    if (
        "reduced_series_cosine" in relation_types
        or "reduced_series_euclidean" in relation_types
    ):
        reduced_series_embeddings = dataset_dict["reduced_time_series"]
        reduced_series_embeddings = torch.tensor(reduced_series_embeddings).to(
            device
        )
        embedding_dict["reduced_series_cosine"] = reduced_series_embeddings
        embedding_dict["reduced_series_euclidean"] = reduced_series_embeddings
    logger.info(f"Loaded {dataset_embeddings.shape} dataset embeddings")

    if existing_handling == "continue":
        # get the shard indices from the existing files
        existing_shard_indices = handle_existing_shards(
            output_tar_path, split, "continue", name="dataset_relation_matrix"
        )
    elif existing_handling == "clean":
        # delete the existing files
        existing_shard_indices = handle_existing_shards(
            output_tar_path, split, "clean", name="dataset_relation_matrix"
        )
    else:
        existing_shard_indices = set()

    # Process in chunks of rows
    chunk_idx = 0
    shard = list()
    shard_idx = 0
    total_saved = 0
    for i in tqdm(
        range(0, int(np.ceil(n_datasets / chunk_size)) * chunk_size, chunk_size)
    ):
        # Initialize chunk result
        chunk_result = (
            torch.zeros((min(chunk_size, n_datasets - i), n_datasets))
            if combine_operation == "sum"
            else torch.ones((min(chunk_size, n_datasets - i), n_datasets))
        )

        if shard_idx in existing_shard_indices:
            # perform necessary operations to step iterators:
            logger.info(
                f"skipping chunk {chunk_idx} from shard {shard_idx} at index {i} because it already exists, next shard {(chunk_idx + 1) // shard_size}"
            )
            chunk_idx += 1
            shard_idx = chunk_idx // shard_size
            continue

        for scaling_lambda, relation_type in zip(
            scaling_lambdas, relation_types
        ):
            if relation_type in [
                "text_cosine",
                "series_cosine",
                "reduced_text_cosine",
                "reduced_series_cosine",
            ]:
                # Get current chunk
                chunk_i = embedding_dict[relation_type][i : i + chunk_size]
                chunk_j = embedding_dict[relation_type]

                # Compute cosine similarity for chunk against all
                chunk_similarity = torch.nn.functional.cosine_similarity(
                    chunk_i.unsqueeze(1),
                    chunk_j.unsqueeze(0),
                    dim=-1,
                )
                # chunk_similarity = np.dot(chunk_i, chunk_j.T) / (
                #     norms[i : i + chunk_size, None] * norms[None, :]
                # )
                chunk_result = (
                    chunk_result + scaling_lambda * chunk_similarity.to("cpu")
                    if combine_operation == "sum"
                    else chunk_result
                    * scaling_lambda
                    * chunk_similarity.to("cpu")
                )

            elif relation_type in [
                "text_euclidean",
                "series_euclidean",
                "reduced_text_euclidean",
                "reduced_series_euclidean",
            ]:
                chunk_i = embedding_dict[relation_type][i : i + chunk_size]
                chunk_j = embedding_dict[relation_type]
                # Compute euclidean distance for chunk against all
                chunk_dist = torch.nn.functional.pairwise_distance(
                    chunk_i[:, None, :], chunk_j[None, :, :], p=2
                )
                chunk_result = (
                    chunk_result + scaling_lambda * chunk_dist.to("cpu")
                    if combine_operation == "sum"
                    else chunk_result * scaling_lambda * chunk_dist.to("cpu")
                )

            elif relation_type == "match_frequency":
                frequencies = np.array(dataset_dict["frequencies"])
                chunk_i = np.array(frequencies[i : i + chunk_size])
                # Compute frequency matches for chunk against all
                chunk_matches = torch.tensor(
                    (chunk_i[:, None] == np.array(frequencies[None, :])).astype(
                        float
                    )
                ).to("cpu")
                chunk_result = (
                    chunk_result + scaling_lambda * chunk_matches
                    if combine_operation == "sum"
                    else chunk_result * scaling_lambda * chunk_matches
                )
            elif relation_type in SERIES_COMPARISON_METHODS:
                import time

                start_time = time.time()
                chunk_vals = torch.tensor(
                    compute_series_similarity(
                        dataset_dict["time_series_full"],
                        np.arange(
                            i,
                            min(
                                i + chunk_size,
                                len(dataset_dict["time_series_full"]),
                            ),
                        ),
                        relation_type,
                    )
                )
                logger.info(
                    f"Time taken for {relation_type}: {time.time() - start_time} seconds"
                )
                chunk_result = (
                    chunk_result + scaling_lambda * chunk_vals
                    if combine_operation == "sum"
                    else chunk_result * scaling_lambda * chunk_vals
                )

            # TODO: some other relations:
            # - overlap in time, relative to the overall time covered by the datasets
            # - use parameters to define series (dimensionality reduction, ARIMA, ODE fitting, etc.), then cluster
            # - use series similarity operations (Granger, DTW, cross-correlation, covariance, etc.)
            # - dimensionality reduction and clustering for text embeddings
            # - Use a TS-language "clip" shared embedding space to define relations

        # give invalid indicies low value relations
        chunk_result[dataset_dict["invalid_indices"]] = -9999
        chunk_result[:, dataset_dict["invalid_indices"]] = -9999
        num_nans = torch.sum(torch.isnan(chunk_result))
        assert num_nans == 0, f"Number of NaNs: {num_nans}"

        # Either save to tar or accumulate in memory
        if output_tar_path and len(shard) == shard_size:
            # Save chunk to tar file
            shard_result = np.concatenate(shard, axis=0)
            total_saved += shard_result.shape[0]
            save_idx_matrix(
                np.ndarray([]),
                shard_idx - 1,
                shard_idx,
                np.ones(shard_result.shape[0]) * shard_idx,
                shard_result,
                "dataset_relation_matrix",
                output_tar_path,
                split,
                force=True,
            )
            shard_idx += 1
            shard = [chunk_result.cpu().numpy()]
        elif output_tar_path and len(shard) < shard_size:
            shard.append(chunk_result.cpu().numpy())
        else:
            # If not saving to tar, we need to accumulate results
            if i == 0:
                full_result = (
                    np.zeros((n_datasets, n_datasets))
                    if combine_operation == "sum"
                    else np.ones((n_datasets, n_datasets))
                )
            full_result[i : i + chunk_size] = chunk_result
        chunk_idx += 1
    if len(shard) > 0:
        shard_result = np.concatenate(shard, axis=0)
        total_saved += shard_result.shape[0]
        logger.info(
            f"last shard shape {shard_result.shape} shard_idx {shard_idx} total_saved {total_saved}"
        )
        save_idx_matrix(
            np.ndarray([]),
            shard_idx - 1,
            shard_idx,
            np.ones(shard_result.shape[0]) * shard_idx,
            shard_result,
            "dataset_relation_matrix",
            output_tar_path,
            split,
            force=True,
        )
    # Close tar file if we're using it
    if not output_tar_path:
        return full_result


def construct_dataset_embeddings_from_lookup_table(
    dataset_list: list, dataset_dir: str
):
    """
    Construct the dataset embeddings from the dataset list and all dataset series.
    """
    # TODO: make this streaming
    # TODO: add embeddings of the series
    dataset_embeddings = torch.tensor(
        np.stack(
            [
                dl[2] if dl[2].shape[0] > 0 else np.ones(TEXT_EMBEDDING_DIM)
                for dl in dataset_list
            ],
            axis=0,
        )
    )
    invalid_indices = np.array(
        [0 if dl[2].shape[0] > 0 else 1 for dl in dataset_list]
    )
    dataset_frequencies = [dl[1] for dl in dataset_list]
    dataset_text = [dl[0] for dl in dataset_list]
    return {
        "dataset_embeddings": dataset_embeddings,
        "dataset_text": dataset_text,
        "frequencies": dataset_frequencies,
        "invalid_indices": invalid_indices,
    }


# Define worker function for parallel processing
def process_dataset(info_tuple):
    (
        emb_path,
        desc_path,
        set_path,
        id,
        us_holidays,
        relation_config,
        keep_time_series,
    ) = info_tuple
    try:
        result = {}

        # Load metadata (lightweight)
        metadata = fm_utils.load_metadata_from_directory(set_path)
        result["dataset_name"] = metadata["columns"][0]["title"]
        result["dataset_frequency"] = metadata["frequency"]

        # Load time series data efficiently
        df = pd.read_parquet(emb_path)

        # Process time series
        time_series_raw = df[df.columns[0]].to_numpy()
        valid_mask = ~np.isnan(time_series_raw)

        # Calculate statistics only once
        mean = np.mean(time_series_raw[valid_mask]) if np.any(valid_mask) else 0
        std = np.std(time_series_raw[valid_mask]) if np.any(valid_mask) else 0

        result["zero_variance"] = std < 1e-10
        if result["zero_variance"]:
            std = std + 0.001  # add a small epsilon to avoid division by zero
        result["time_series_mean_variance"] = np.array([mean, std])

        # Normalize time series
        time_series = (time_series_raw - mean) / std
        time_series = np.nan_to_num(time_series, nan=float(mean))

        # Process timestamps
        timestamp_series = df[df.columns[1]]
        if not pd.api.types.is_datetime64_any_dtype(timestamp_series):
            timestamp_series = pd.to_datetime(timestamp_series, errors="coerce")

        # Use the global constant
        features = TIMESTAMPS_FEATURES
        timestamps = np.zeros((len(timestamp_series), len(features)))
        nan_mask = timestamp_series.isna().values

        timestamps = convert_time_to_vector(
            timestamps,
            features,
            NORM_RANGES,
            timestamp_series,
            us_holidays,
            0,
            np.array(nan_mask),
        )

        start_times = timestamps[0]
        end_times = timestamps[-1]
        result["dataset_time_range"] = np.concatenate(
            [start_times, end_times], axis=0
        )

        if keep_time_series:
            result["time_series_full"] = time_series

        if relation_config is not None:
            # Compute embeddings
            methods = getattr(
                relation_config, "series_embedding_methods", ["fft"]
            )
            dims = getattr(relation_config, "series_embedding_dims", [32])
            result["time_series_embedding"] = timeseries_to_fixed_vector(
                time_series, methods=methods, embed_dims=dims
            )

            # Keep full time series if requested

        # Load description embedding
        result["dataset_embedding"] = np.load(desc_path)

        return id, result

    except Exception as e:
        logger.error(f"Error processing dataset {id}: {e}")
        return id, None


def embed_time_series_from_enriched(
    dataset_dir: str,
    relation_config: RelationConfig | None = None,
    keep_time_series: bool = False,
    lookup_table: list = [],
    indices: np.ndarray = np.array([]),
    num_workers: int = 16,
):
    """
    Embed the time series from the dataset directory to a fixed vector representation
    """
    # Determine which datasets to process
    if len(indices) > 0:
        dataset_ids = set(indices.tolist())
        dataset_paths = [
            os.path.join(dataset_dir, d)
            for d in os.listdir(dataset_dir)
            if int(d.split("_")[-1]) in dataset_ids
        ]
    elif len(lookup_table) > 0:
        dataset_ids = set(np.arange(len(lookup_table)))
        dataset_paths = [
            os.path.join(dataset_dir, d)
            for d in os.listdir(dataset_dir)
            if int(d.split("_")[-1]) in dataset_ids
        ]
    else:
        # otherwise find all the directories in the enriched dataset directory
        dataset_paths = [
            os.path.join(dataset_dir, d)
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
        dataset_ids = np.array(
            [
                int(d.split("_")[-1])
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ]
        ).tolist()

    dataset_ids = list(dataset_ids)
    dataset_ids.sort()
    dataset_paths.sort(key=lambda x: int(x.split("_")[-1]))

    # Initialize result containers
    results = {
        "dataset_names": [],
        "dataset_frequencies": [],
        "time_series_list": [] if keep_time_series else None,
        "embeddings": [],
        "dataset_time_ranges": [],
        "dataset_embeddings": [],
        "time_series_mean_variance": [],
        "zero_variance_series": [],
    }

    # Pre-load holidays to avoid reloading in each worker
    us_holidays = holidays.country_holidays("US")

    # Pre-compute all paths
    dataset_info = []
    for path, id in zip(dataset_paths, dataset_ids):
        emb_path = os.path.join(path, os.path.basename(path) + ".parquet")
        desc_path = os.path.join(path, "description_embedding.npy")
        dataset_info.append(
            (
                emb_path,
                desc_path,
                path,
                id,
                us_holidays,
                relation_config,
                keep_time_series,
            )
        )

    # Process datasets in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed

    processed_results = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_dataset, info) for info in dataset_info
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing datasets",
        ):
            id, result = future.result()
            if result is not None:
                processed_results[id] = result

    # Collect results in the correct order
    for id in dataset_ids:
        if id in processed_results:
            result = processed_results[id]
            results["dataset_names"].append(result.get("dataset_name", ""))
            results["dataset_frequencies"].append(
                result.get("dataset_frequency", "")
            )
            results["dataset_time_ranges"].append(
                result.get(
                    "dataset_time_range", np.zeros(len(TIMESTAMPS_FEATURES) * 2)
                )
            )
            results["time_series_mean_variance"].append(
                result.get("time_series_mean_variance", np.array([0, 1]))
            )
            results["zero_variance_series"].append(
                result.get("zero_variance", True)
            )

            if relation_config is not None:
                results["embeddings"].append(
                    result.get("time_series_embedding", np.zeros(1))
                )
                results["dataset_embeddings"].append(
                    result.get("dataset_embedding", np.zeros(1))
                )
            if keep_time_series:
                results["time_series_list"].append(
                    result.get("time_series_full", np.zeros(1))
                )

    # Handle dimension reduction
    if relation_config is not None and results["embeddings"]:
        reduced_time_series = reduce_dimensions(
            np.stack(results["embeddings"], axis=0),
            method=relation_config.series_reduce_embed_method,
            n_components=relation_config.reduced_series_embedding_dim,
        )
        reduced_embeddings = reduce_dimensions(
            np.stack(results["dataset_embeddings"], axis=0),
            method=relation_config.dataset_reduce_embed_method,
            n_components=relation_config.reduced_dataset_embedding_dim,
        )
    else:
        results["embeddings"] = [np.zeros((1,))]
        results["dataset_embeddings"] = [np.zeros((1,))]
        reduced_time_series = np.zeros((1, 1))
        reduced_embeddings = np.zeros((1, 1))

    # Prepare final output
    return {
        "time_series_embeddings": np.stack(results["embeddings"], axis=0)
        if results["embeddings"]
        else np.zeros((0, 1)),
        "time_series_mean_variance": np.stack(
            results["time_series_mean_variance"], axis=0
        ),
        "time_series_full": results["time_series_list"]
        if keep_time_series
        else [],
        "dataset_embeddings": np.stack(results["dataset_embeddings"], axis=0)
        if results["dataset_embeddings"]
        else np.zeros((0, 1)),
        "frequencies": results["dataset_frequencies"],
        "time_ranges": np.stack(results["dataset_time_ranges"], axis=0),
        "invalid_indices": np.ones(len(results["dataset_frequencies"])),
        "reduced_time_series": reduced_time_series,
        "reduced_embeddings": reduced_embeddings,
        "dataset_ids": np.array(dataset_ids),
        "zero_variance_series": np.array(results["zero_variance_series"]),
    }


def load_relational_dataset_matrix(
    dataset_ids: np.ndarray,
    relation_config: RelationConfig,
    output_dir: str,
    split: str = "train",
    uid: str = "0",
    slice_dataset_matrix_num: int = -1,
    save=False,
    load_from_dir="",
):
    # create a dataloader for the relational matrix, shuffle=false to load in order
    dataset_relation_matrix_loader = ShardedDataloaderV1(
        relation_config.dataset_loader_config,
        name_specification="_dataset_relation_matrix",
        data_dir=relation_config.output_subdir
        if len(load_from_dir) == 0
        else load_from_dir,
    )
    dataset_relation_matrix_loader.shuffle = False
    dataset_relation_matrix_loader.batch_size = len(dataset_ids)
    loaded_rows = load_full_dataset_relation_matrix(
        dataset_relation_matrix_loader, split, select_dataset_ids=dataset_ids
    )
    np.save(
        os.path.join(output_dir, f"loaded_rows_sample_{uid}.npy"), loaded_rows
    )
    if save:
        if slice_dataset_matrix_num > 0:
            # save the dataset ids
            dataset_ids = np.arange(
                slice_dataset_matrix_num
            )  # TODO: generate a random slice
            np.save(
                os.path.join(output_dir, f"dataset_idxes_{uid}.npy"),
                dataset_ids,
            )
            sliced_matrix = relative_relation_matrix(
                loaded_rows,
                dataset_ids,
            )
            np.save(
                os.path.join(
                    relation_config.output_subdir,
                    f"sliced_dataset_relation_matrix_{uid}.npy",
                ),
                sliced_matrix,
            )
        else:
            # save everything
            np.save(
                os.path.join(
                    relation_config.output_subdir,
                    f"relation_matrix_{uid}.npy",
                ),
                loaded_rows,
            )

    return loaded_rows


def handle_dataset_ids(
    dataset_ids: str, NUM_DATASETS: int, load_relational_matrix: int
) -> np.ndarray:
    """
    Handle the dataset ids for the relational matrix
    If dataset_ids is a comma separated list, return the array of dataset ids
    If dataset_ids is a range separated by a dash, return the arange of dataset ids
    If dataset_ids is empty, check load_relational_matrix
    If load_relational_matrix is greater than 0, return a random sample of dataset ids of size load_relational_matrix
    If load_relational_matrix is 0, return all dataset ids
    """
    if len(dataset_ids) > 0:
        if "," in dataset_ids:
            dataset_id_vals = np.array(dataset_ids.split(","))
        elif "-" in dataset_ids:
            dataset_id_vals = np.arange(
                int(dataset_ids.split("-")[0]),
                int(dataset_ids.split("-")[1]),
            )
        else:
            dataset_id_vals = np.array([int(dataset_ids)])
    else:
        if load_relational_matrix > 0:
            dataset_id_vals = np.random.choice(
                np.arange(NUM_DATASETS),
                size=load_relational_matrix,
                replace=False,
            ).tolist()
        else:
            dataset_id_vals = np.arange(NUM_DATASETS)
    return np.array(dataset_id_vals)
