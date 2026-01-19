import os
import pickle

import numpy as np
from loguru import logger
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from synthefy_pkg.utils.retrieve_dataset_descriptions import (
    load_dataset_description_from_enriched,
)


def compute_clustering(similarity_or_data_matrix, method="kmeans", **kwargs):
    if method == "hierarchical":
        similarity_matrix = similarity_or_data_matrix[1]
        model = AgglomerativeClustering(
            metric="precomputed",
            n_clusters=kwargs["num_clusters"],
            linkage="complete",
        ).fit(similarity_matrix)

        labels = model.labels_

    elif method == "kmeans":
        data_matrix = similarity_or_data_matrix[0]

        # Apply K-means clustering
        num_clusters = kwargs["num_clusters"]  # Specify the number of clusters
        model = KMeans(
            n_clusters=num_clusters, random_state=kwargs["random_state"]
        )
        sampled_indices = np.random.choice(
            data_matrix.shape[0], 30000, replace=False
        )
        model.fit(data_matrix[sampled_indices])

        # Get cluster labels
        labels = model.labels_

        # label the whole dataset:
        labels = model.predict(data_matrix)

    return model, labels


def compute_k_closest_datasets(
    similarity_matrix: np.ndarray, k=10, dataset_ids=None
):
    # Get the k-closest datasets for each dataset
    if dataset_ids is None:
        smallest_indices = np.argpartition(similarity_matrix, k)[:, :k]
        largest_indices = np.argpartition(similarity_matrix, -k)[:, -k:]
    else:
        distances = np.mean(similarity_matrix, axis=0)
        distances = np.delete(distances, dataset_ids)
        smallest_indices = np.argpartition(distances, k)[:k]
        largest_indices = np.argpartition(distances, -k)[-k:]
    # Calculate feature dimension
    return smallest_indices, largest_indices


def compute_cluster_distance(relative_matrix, labels):
    """
    Compute distances between cluster centers from the relative distance matrix.

    Args:
        relative_matrix: Matrix of distances between data points
        labels: Cluster labels for each data point

    Returns:
        cluster_distance_matrix: Matrix of distances between cluster centers
    """
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Create mapping from cluster labels to indices
    # label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Initialize distance matrix between clusters
    cluster_distance_matrix = np.zeros((n_clusters, n_clusters))

    # For each pair of clusters
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i == j:
                continue

            # Get indices of points in each cluster
            cluster_i_indices = np.where(labels == label_i)[0]
            cluster_j_indices = np.where(labels == label_j)[0]

            # Check if indices are within bounds of relative_matrix
            valid_i = cluster_i_indices[
                cluster_i_indices < relative_matrix.shape[0]
            ]
            valid_j = cluster_j_indices[
                cluster_j_indices < relative_matrix.shape[1]
            ]

            if len(valid_i) == 0 or len(valid_j) == 0:
                # Skip if no valid indices
                cluster_distance_matrix[i, j] = np.inf
                continue

            # Compute average distance between points in the two clusters
            distances = []
            for idx_i in valid_i:
                for idx_j in valid_j:
                    if (
                        idx_j < relative_matrix.shape[1]
                        and idx_i < relative_matrix.shape[0]
                    ):
                        distances.append(relative_matrix[idx_i, idx_j])

            if distances:
                cluster_distance_matrix[i, j] = np.mean(distances)
            else:
                cluster_distance_matrix[i, j] = np.inf

    return cluster_distance_matrix


def relative_relation_matrix(relation_dataset_matrix, dataset_ids):
    """
    Compute the relation matrix for only the datasets in dataset_ids
    """
    relative_relation_matrix = relation_dataset_matrix[:, dataset_ids]
    return relative_relation_matrix


def pretrain_blind_by_cluster(
    output_dir,
    dataset_dict,
    relation_dataset_matrix,
    uid=0,
    method="kmeans",
    num_clusters=10,
    random_state=42,
    dataset_ids=np.array([]),
    embedding_choice="text",
):
    """
    Generate pretrain and blind splits by only putting clusters into the sets
    Can also compute the furthest clusters
    """
    # Load the data
    if embedding_choice == "text":
        embeddings_matrix = dataset_dict["dataset_embeddings"]
    elif embedding_choice == "time_series":
        embeddings_matrix = dataset_dict["time_series_embeddings"]
    elif embedding_choice == "reduced_text":
        embeddings_matrix = dataset_dict["reduced_embeddings"]
    elif embedding_choice == "reduced_time_series":
        embeddings_matrix = dataset_dict["reduced_time_series"]
    else:
        raise ValueError(f"Invalid embedding choice: {embedding_choice}")

    if method == "closest":
        dataset_ids = np.load(
            os.path.join(output_dir, f"dataset_idxes_{uid}.npy")
        )
        smallest_indices, largest_indices = compute_k_closest_datasets(
            relation_dataset_matrix, k=num_clusters, dataset_ids=dataset_ids
        )
        print(smallest_indices.shape, largest_indices.shape)
        return {
            "pretrain": largest_indices,
            "blind": smallest_indices,
            "cluster_model": None,
            "cluster_labels": None,
        }

    if len(relation_dataset_matrix) > 0:
        print(relation_dataset_matrix.shape)
        relative_matrix = relative_relation_matrix(
            relation_dataset_matrix, dataset_ids
        )
    else:
        relative_matrix = embeddings_matrix

    # get cluster centers
    model, labels = compute_clustering(
        (relative_matrix, embeddings_matrix),
        method=method,
        num_clusters=num_clusters,
        random_state=random_state,
    )

    assert isinstance(labels, np.ndarray)
    unique_labels = np.unique(labels)

    print(relative_matrix.shape, labels.shape)
    cluster_distance_matrix = compute_cluster_distance(relative_matrix, labels)
    # get the furthest clusters
    sorted_clusters = np.argsort(np.sum(cluster_distance_matrix, axis=1))

    # figure out how much data to put in pretrain and blind
    cluster_dataset_size = np.array(
        [np.sum(labels == label) for label in unique_labels]
    )
    total_size = np.sum(cluster_dataset_size)
    pretrain_size = int(total_size * 0.9)

    pretrain_blind_dict = {
        "pretrain": [],
        "blind": [],
        "cluster_model": model,
        "cluster_labels": labels,
    }

    current_size = 0
    for cluster in sorted_clusters:
        if current_size > pretrain_size:
            pretrain_blind_dict["blind"].append(cluster)
        else:
            pretrain_blind_dict["pretrain"].append(cluster)
        current_size += cluster_dataset_size[cluster]

    return pretrain_blind_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--uid", type=str, default="0")
    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--load_dataset_dict", type=str, default="")
    parser.add_argument("--embedding_choice", type=str, default="text")
    parser.add_argument("--relational_matrix", type=str, default="")
    args = parser.parse_args()

    if len(args.load_dataset_dict) != 0:
        if args.load_dataset_dict == "OUTPUT":
            dataset_id_vals = np.load(
                os.path.join(
                    args.output_dir, f"partial_dataset_ids_{args.uid}.npy"
                )
            )
            dataset_dict = pickle.load(
                open(os.path.join(args.output_dir, "dataset_dict.pkl"), "rb")
            )
        else:
            dataset_dict = pickle.load(
                open(
                    os.path.join(args.load_dataset_dict, "dataset_dict.pkl"),
                    "rb",
                )
            )

    if len(args.relational_matrix) > 0:
        relation_dataset_matrix = np.load(args.relational_matrix)
    else:
        relation_dataset_matrix = np.array([])

    pt_blind_dict = pretrain_blind_by_cluster(
        args.output_dir,
        dataset_dict,
        relation_dataset_matrix,
        args.uid,
        args.method,
        args.num_clusters,
        args.random_state,
        embedding_choice=args.embedding_choice,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump(
        pt_blind_dict,
        open(
            os.path.join(
                args.output_dir, f"pretrain_blind_dict_{args.uid}.pkl"
            ),
            "wb",
        ),
    )
    logger.info(
        f"Saved pretrain_blind_dict_{args.uid}.pkl to {args.output_dir}"
    )

    # uv run src/synthefy_pkg/preprocessing/relational/simple_cluster_relational_matrix.py --output_dir /home/data/all_univariate_filtered_clustering --uid 0 --method kmeans --num_clusters 50 --random_state 42 --load_dataset_dict /home/data/all_univariate_filtered_dataset_dict --embedding_choice reduced_text
