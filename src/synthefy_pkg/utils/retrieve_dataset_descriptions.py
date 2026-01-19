import glob
import os
import pickle

import numpy as np


def load_dataset_description(
    dataset_ids: np.ndarray | None,
    base_dir: str,
    dataset_lookup: list | None = None,
    return_idxes: list[int] = [0],
) -> list:
    """
    Loads the dataset description from the given base directory (path to the dataset).

    Args:
        dataset_ids (np.ndarray): The dataset IDs to load.
        base_dir (str): The base directory to load the dataset description from.
        return_idx (int): The index of the dataset description to return.

    Returns:
        list of strings: The dataset descriptions in the order of the dataset IDs returned.
    """
    if dataset_lookup is None:
        dataset_description_path = os.path.join(base_dir, "dataset_lookup.pkl")
        with open(dataset_description_path, "rb") as f:
            dataset_lookup = pickle.load(f)
    else:
        dataset_lookup = dataset_lookup
    assert isinstance(dataset_lookup, list), "dataset_lookup must be a list"
    if dataset_ids is None:
        dataset_ids = np.arange(len(dataset_lookup))
    descriptions = [
        [dataset_lookup[int(did)][return_idx] for return_idx in return_idxes]
        for did in dataset_ids
    ]
    return descriptions


def load_dataset_description_from_enriched(
    base_dir: str,
) -> np.ndarray:
    """
    Loads the dataset description from the given base directory (path to the dataset).

    Args:
        dataset_ids (np.ndarray): The dataset IDs to load.
        base_dir (str): The base directory to load the dataset description from.
        return_idx (int): The index of the dataset description to return.

    Returns:
        list of strings: The dataset descriptions in the order of the dataset IDs returned.
    """
    dataset_paths = glob.glob(
        os.path.join(base_dir, "*description_embedding.npy")
    )
    dataset_paths = list(dataset_paths)
    dataset_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
    embeddings = np.array([])
    for dataset_path in dataset_paths:
        embedding = np.load(dataset_path)
        embeddings = np.append(embeddings, embedding)
    return embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", nargs="+", type=int, required=True)
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/local/synthefy_data/fmv2_10k/pretrain",
    )
    args = parser.parse_args()

    dataset_ids = np.array(args.dataset_ids)
    dataset_lookup = None
    descriptions = load_dataset_description(
        dataset_ids, args.base_dir, dataset_lookup
    )
    for desc in descriptions:
        print(desc)
