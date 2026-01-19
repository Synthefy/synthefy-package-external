import pdb
from typing import Any

import numpy as np
from loguru import logger
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from synthefy_pkg.preprocessing.fmv3.relational_sampling.base_relation_constructor import (
    BaseRelationConstructor,
)


class KMeansRelationConstructor(BaseRelationConstructor):
    """
    K-Means relation constructor.

    Assigns each sample to a class based on the k-means clustering of the text embeddings.
    Then, allow sampling proportional to the cluster cosine distance.
    """

    def __init__(self, data_dir: str, num_classes: int):
        super().__init__(data_dir, "k_means")
        self.num_classes = num_classes
        self.total_samples = self.get_total_samples()

    def _compute_relational_matrix_from_centroids(
        self, centroids: np.ndarray
    ) -> np.ndarray:
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        # We construct the similarity matrix simply based on the cosine similarity between each centroid
        similarity_matrix = centroids @ centroids.T

        # Convert to correlation strength by taking absolute value
        # This measures strength of correlation regardless of direction (positive or negative)
        correlation_strength = np.abs(similarity_matrix)

        # Normalize each row to sum to 1 (probabilities)
        relational_matrix = correlation_strength / correlation_strength.sum(
            axis=1, keepdims=True
        )

        return relational_matrix

    def _compute_centroids(self) -> tuple[Any, MiniBatchKMeans]:
        mbkm = MiniBatchKMeans(
            n_clusters=self.num_classes,
            batch_size=1000,
            max_iter=1,
            init="k-means++",
            random_state=42,
        )

        batch = []
        for dataset_index, data in tqdm(
            self.iterate_over_data(
                load_options=["text_embeddings"],
                multiprocess=True,
                num_workers=32,
                batch_size=1000,
            ),
            total=self.total_samples,
        ):
            batch.append(data["text_embeddings"].reshape(1, -1))
            if (dataset_index + 1) % 1000 == 0:
                batch = np.concatenate(batch, axis=0)
                mbkm.partial_fit(batch)
                batch = []

        # Process remaining samples in the last partial batch
        if batch:
            batch = np.concatenate(batch, axis=0)
            mbkm.partial_fit(batch)

        centroids = mbkm.cluster_centers_

        return centroids, mbkm

    def _classify_samples(
        self, centroids: np.ndarray, mbkm: MiniBatchKMeans
    ) -> np.ndarray:
        # Compute cosine distance between each sample and each centroid
        classes = []
        batch = []
        for dataset_index, data in tqdm(
            self.iterate_over_data(
                load_options=["text_embeddings"],
                multiprocess=True,
                num_workers=32,
                batch_size=1000,
            ),
            total=self.total_samples,
        ):
            batch.append(data["text_embeddings"].reshape(1, -1))
            if (dataset_index + 1) % 1000 == 0:
                batch = np.concatenate(batch, axis=0)

                predictions = mbkm.predict(batch)
                classes.append(predictions)

                batch = []

        # Process remaining samples in the last partial batch
        if batch:
            batch = np.concatenate(batch, axis=0)
            predictions = mbkm.predict(batch)
            classes.append(predictions)

        return np.concatenate(classes, axis=0)

    def construct_relations(self) -> None:
        centroids, mbkm = self._compute_centroids()
        relational_matrix = self._compute_relational_matrix_from_centroids(
            centroids
        )
        classes = self._classify_samples(centroids, mbkm)
        self.save_numpy_files(classes, relational_matrix)
        logger.info(
            f"Saved {self.num_classes} classes and relational matrix to {self.relational_sampling_dir}"
        )
