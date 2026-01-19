import argparse
import itertools
import os
import pickle
import random
from typing import List, Tuple

import matplotlib.cm as cm  # Add this import
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import seaborn as sns
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_dataset_lookup(
    data_dir: str, random_sample_max: int = 10000
) -> Tuple[List[str], List[List[Tuple]]]:
    """
    Load the dataset_lookup pickle file.

    Args:
        data_dir: Directory containing dataset_lookup.pkl

    Returns:
        The dataset_lookup list containing (title, frequency, embedding, scalars, num_rows, num_windows)
    """
    data_dirs = data_dir.split(",")
    lookup_paths = [
        os.path.join(data_dir, "dataset_lookup.pkl") for data_dir in data_dirs
    ]
    full_dataset_lookup = []
    for lookup_path in lookup_paths:
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(
                f"Dataset lookup file not found at {lookup_path}"
            )

        with open(lookup_path, "rb") as f:
            dataset_lookup = pickle.load(f)

        logger.info(f"Loaded dataset lookup with {len(dataset_lookup)} entries")
        # if there are probably going to be too many datasets, randomly sample a subset
        if len(dataset_lookup) > random_sample_max:
            dataset_lookup = random.sample(dataset_lookup, random_sample_max)
        full_dataset_lookup.append(dataset_lookup)
    return data_dirs, full_dataset_lookup


def create_embedding_matrix(
    dataset_lookup: List[Tuple],
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract embeddings and titles from the dataset_lookup.

    Args:
        dataset_lookup: List of tuples from the pickle file

    Returns:
        Tuple of (embedding_matrix, titles)
    """
    # Filter out entries with empty embeddings
    valid_entries = [
        (title, embedding)
        for title, _, embedding, _, _, _ in dataset_lookup
        if isinstance(embedding, np.ndarray) and embedding.size > 0
    ]

    if not valid_entries:
        raise ValueError("No valid embeddings found in the dataset lookup")

    titles = [entry[0] for entry in valid_entries]
    embeddings = [entry[1] for entry in valid_entries]

    # Stack embeddings into a single matrix
    embedding_matrix = np.vstack(embeddings)

    logger.info(f"Created embedding matrix with shape {embedding_matrix.shape}")
    return embedding_matrix, titles


def cosine_similarity(
    embedding_matrix: np.ndarray, other_embedding_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between two matrices of embeddings.

    Args:
        embedding_matrix: First matrix of embeddings
        other_embedding_matrix: Second matrix of embeddings

    Returns:
        Matrix of cosine similarities between all pairs of embeddings
    """
    # Normalize the embeddings
    norm1 = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norm2 = np.linalg.norm(other_embedding_matrix, axis=1, keepdims=True)

    # Compute dot product and divide by norms
    return np.dot(embedding_matrix, other_embedding_matrix.T) / (
        norm1 * norm2.T
    )


def format_multiline_text(text, chars_per_line=80):
    """Split text into multiple lines for better readability"""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= chars_per_line:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def compute_embeddings(
    embedding_matrix: np.ndarray, operations_to_run: list[str] = ["tsne", "pca"]
):
    embeddings, compute_info = dict(), dict()
    if "umap" in operations_to_run:
        import umap
        # UMAP = None
        # raise NotImplementedError("UMAP is not installed")

        logger.info("Running UMAP dimensionality reduction...")
        umap_reducer = umap.UMAP(random_state=42)
        umap_embedding = umap_reducer.fit_transform(embedding_matrix)

        # umap without labels
        plt.figure(figsize=(12, 10))
        if isinstance(umap_embedding, tuple):
            # Extract first element if it's a tuple
            umap_data = umap_embedding[0]
        else:
            umap_data = umap_embedding

        # type checker is crashing out here
        if sp.issparse(umap_data):
            assert hasattr(umap_data, "toarray"), (
                "UMAP embedding is not a valid sparse matrix"
            )
            if isinstance(umap_data, sp.csr_matrix):
                umap_data = umap_data.toarray()
            else:
                umap_data = umap_data
        assert isinstance(umap_data, np.ndarray), (
            "UMAP embedding is not a valid array"
        )
        embeddings["umap"] = umap_data
    if "tsne" in operations_to_run:
        logger.info("Running t-SNE dimensionality reduction...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(embedding_matrix) - 1),
        )
        tsne_embedding = tsne.fit_transform(embedding_matrix)
        embeddings["tsne"] = tsne_embedding
    if "pca" in operations_to_run:
        logger.info("Running PCA dimensionality reduction...")
        pca = PCA(n_components=2, random_state=42)
        pca_embedding = pca.fit_transform(embedding_matrix)
        logger.info(
            f"PCA explained variance ratio: {pca.explained_variance_ratio_}"
        )
        embeddings["pca"] = pca_embedding
        compute_info["pca"] = pca.explained_variance_ratio_
    if "pacmap" in operations_to_run:
        import pacmap
        # PaCMAP = None
        # raise NotImplementedError("PaCMAP is not installed")

        logger.info("Running PaCMAP dimensionality reduction...")
        pm = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=42)
        pacmap_embedding = pm.fit_transform(embedding_matrix)
        embeddings["pacmap"] = pacmap_embedding
    return embeddings, compute_info


def visualize_dimension_reduction(
    embedding_data: np.ndarray,
    colors: np.ndarray,
    subset: np.ndarray,
    output_dir: str,
    method_name: str,
    compute_info: dict = {},
):
    """
    Visualize dimensionality reduction results with and without labels.

    Args:
        embedding_data: 2D array of embedding coordinates
        colors: Array of colors for each point
        subset: Indices of points to label
        output_dir: Directory to save plots
        method_name: Name of the dimensionality reduction method (e.g., 'umap', 'tsne', 'pca')
        compute_info: Optional dictionary containing additional info (e.g., explained variance for PCA)
    """
    # Plot without labels
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_data[:, 0], embedding_data[:, 1], c=colors)

    title = f"{method_name.upper()} Projection of Dataset Embeddings"
    if compute_info and method_name.lower() == "pca":
        var_explained = compute_info["pca"]
        title += f"\nExplained variance: {var_explained[0]:.2f}, {var_explained[1]:.2f}"

    plt.title(title)
    plt.tight_layout()

    # Save axis limits to reuse
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.savefig(
        os.path.join(output_dir, f"{method_name}_embeddings.png"), dpi=300
    )
    plt.close()

    # Plot with labels
    plt.figure(figsize=(12, 10))
    plt.scatter(
        embedding_data[subset, 0],
        embedding_data[subset, 1],
        c=colors[subset],
    )

    # Add labels for subset
    for idx in subset:
        plt.annotate(
            str(idx),
            (embedding_data[idx, 0], embedding_data[idx, 1]),
            fontsize=8,
            alpha=0.8,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7
            ),
        )

    # Set the same axis limits as the main plot
    plt.xlim(xlim)
    plt.ylim(ylim)

    title = f"{method_name.upper()} Projection with Selected Labels"
    if compute_info and method_name.lower() == "pca":
        var_explained = compute_info["pca"]
        title += f"\nExplained variance: {var_explained[0]:.2f}, {var_explained[1]:.2f}"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{method_name}_embeddings_labeled.png"),
        dpi=300,
    )
    plt.close()


def visualize_embeddings_color_by_dataset(
    embeddings_dataset_dict: dict,
    output_dir: str,
    operations_to_run: list[str] = ["tsne", "pca"],  # umap, pacmap
):
    """
    Visualize embeddings color by dataset.

    Args:
        embeddings_dataset_dict: Dictionary of dataset_path to embeddings, titles and compute_info
        output_dir: Directory to save visualization plots
        operations_to_run: List of operations to run
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a colormap for datasets
    n_datasets = len(embeddings_dataset_dict)
    dataset_colors = np.array(
        cm.get_cmap("tab20")(np.linspace(0, 1, n_datasets))
    )

    # Create dataset to color mapping
    dataset_color_map = {
        dataset: color
        for dataset, color in zip(
            embeddings_dataset_dict.keys(), dataset_colors
        )
    }

    # Create legend
    plt.figure(figsize=(15, max(10, n_datasets * 0.4)))
    for dataset, color in dataset_color_map.items():
        plt.plot(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=10,
            linestyle="",
            label=f"{dataset}",
        )

    plt.legend(
        loc="center",
        fontsize=10,
        frameon=True,
        title="Dataset Mapping",
        bbox_to_anchor=(0.5, 0.5),
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "dataset_legend.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # For each dimensionality reduction method
    for method in operations_to_run:
        # Combine all embeddings for this method
        all_embeddings = []
        all_colors = []
        all_titles = []
        all_compute_info = {}

        for dataset, (
            embeddings,
            titles,
            compute_info,
        ) in embeddings_dataset_dict.items():
            if method in embeddings and embeddings[method] is not None:
                all_embeddings.append(embeddings[method])
                all_colors.extend(
                    [dataset_color_map[dataset]] * len(embeddings[method])
                )
                all_titles.extend(titles)
                if method == "pca":
                    all_compute_info["pca"] = compute_info["pca"]

        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            combined_colors = np.array(all_colors)

            # Visualize combined embeddings
            visualize_dimension_reduction(
                embedding_data=combined_embeddings,
                colors=combined_colors,
                subset=np.arange(
                    len(combined_embeddings)
                ),  # No subset for dataset coloring
                output_dir=output_dir,
                method_name=f"{method}_by_dataset",
                compute_info=all_compute_info if method == "pca" else {},
            )

    logger.success(
        f"Visualization plots for {' '.join(operations_to_run)} saved to {output_dir}"
    )


def visualize_embeddings_color_by_point(
    titles: List[str],
    output_dir: str,
    operations_to_run: list[str] = ["tsne", "pca"],  # umap, pacmap
    embeddings: dict = {},
    compute_info: dict = {},
):
    """
    Run dimensionality reduction methods and create visualizations.

    Args:
        titles: List of titles corresponding to the embeddings
        output_dir: Directory to save visualization plots
        operations_to_run: List of operations to run
        embeddings: Dictionary of embeddings
        compute_info: Dictionary of compute info
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a colormap
    n_points = len(titles)
    colors = np.array(cm.get_cmap("viridis")(np.linspace(0, 1, n_points)))

    # Select random subset for labeling
    NUM_SUBSET = 10
    subset = np.random.choice(
        list(range(len(titles))), size=NUM_SUBSET, replace=False
    ).astype(int)
    subset_titles = [titles[i] for i in subset]
    subset_titles = [format_multiline_text(title) for title in subset_titles]

    # Create a figure for the legend
    plt.figure(figsize=(15, max(10, NUM_SUBSET * 0.4)))

    # Create a legend mapping
    for i, title in enumerate(subset_titles):
        idx = subset[i]
        plt.plot(
            [0],
            [0],
            color=colors[idx],
            marker="o",
            markersize=10,
            linestyle="",
            label=f"{idx}: {title}",
        )

    # Create the legend
    plt.legend(
        loc="center",
        fontsize=10,
        frameon=True,
        title="Dataset Index and Title Mapping",
        bbox_to_anchor=(0.5, 0.5),
    )

    # Remove axes and adjust layout
    plt.axis("off")
    plt.tight_layout()

    # Save the legend as a separate file
    plt.savefig(
        os.path.join(output_dir, "dataset_legend.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Visualize each dimensionality reduction method
    for method in operations_to_run:
        if method in embeddings:
            embedding_data = embeddings[method]
            if embedding_data is not None:
                visualize_dimension_reduction(
                    embedding_data=embedding_data,
                    colors=colors,
                    subset=subset,
                    output_dir=output_dir,
                    method_name=method,
                    compute_info=compute_info if method == "pca" else dict(),
                )

    logger.success(
        f"Visualization plots for {' '.join(operations_to_run)} saved to {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize dataset embeddings using UMAP and t-SNE"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing dataset_lookup.pkl (e.g., /path/to/output_dir/pretrain)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embedding_visualizations",
        help="Directory to save visualization plots",
    )
    parser.add_argument(
        "--operations_to_run",
        type=str,
        default="umap tsne pca pacmap",
    )
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=10000,
        help="Maximum number of datasets to visualize",
    )
    args = parser.parse_args()

    # Load dataset lookup
    dataset_titles, dataset_lookups = load_dataset_lookup(
        args.data_dir, args.max_datasets
    )

    # Create embedding matrix
    embedding_matrix_list = []
    titles_list = []
    for dataset_lookup in dataset_lookups:
        embedding_matrix, titles = create_embedding_matrix(dataset_lookup)
        embedding_matrix_list.append(embedding_matrix)
        titles_list.append(titles)

    # create embeddings
    embeddings_dataset_dict = dict()
    for embedding_matrix, titles, data_dir in zip(
        embedding_matrix_list, titles_list, dataset_titles
    ):
        embeddings, compute_info = compute_embeddings(
            embedding_matrix, args.operations_to_run.split(" ")
        )
        embeddings_dataset_dict[data_dir] = (embeddings, titles, compute_info)

    if len(embeddings_dataset_dict) > 1:
        visualize_embeddings_color_by_dataset(
            embeddings_dataset_dict,
            args.output_dir,
            args.operations_to_run.split(" "),
        )

        # for every pair of datasets, compute the cosine similarity matrix
        embeddings_matrix_dict = {
            dataset: embedding_matrix
            for (dataset, embedding_matrix) in zip(
                dataset_titles, embedding_matrix_list
            )
        }
        # save mean distances to a file
        with open(
            os.path.join(args.output_dir, "mean_distances.txt"), "w"
        ) as f:
            for dataset1, dataset2 in itertools.combinations_with_replacement(
                embeddings_matrix_dict.keys(), 2
            ):
                embedding_matrix1 = embeddings_matrix_dict[dataset1]
                embedding_matrix2 = embeddings_matrix_dict[dataset2]
                cosine_similarity_matrix = cosine_similarity(
                    embedding_matrix1, embedding_matrix2
                )

                # log the mean cosine similarity for each pair of datasets:
                logger.info(
                    f"Mean cosine similarity between {dataset1} and {dataset2}: {np.mean(cosine_similarity_matrix)}"
                )
                f.write(
                    f"Mean cosine similarity between {dataset1} and {dataset2}: {np.mean(cosine_similarity_matrix)}\n"
                )
    else:
        # Visualize embeddings
        visualize_embeddings_color_by_point(
            titles,
            args.output_dir,
            args.operations_to_run.split(" "),
            embeddings,
            compute_info,
        )


if __name__ == "__main__":
    main()

    # visualize embeddings
    # uv run src/synthefy_pkg/scripts/text_visualization.py --data_dir /home/data/foundation_model_data_all_univariate_filtered_226355_0_226355_ts_2021-06-01/pretrain --output_dir /home/data/visualize/embedding_visualizations
    # uv run src/synthefy_pkg/scripts/text_visualization.py --data_dir /home/data/foundation_model_data_all_univariate_filtered_226355_0_226355_ts_2021-06-01/pretrain,/home/data/algo8,/home/data/algo8_metadata/pretrain --output_dir /home/data/visualize/embedding_visualizations
