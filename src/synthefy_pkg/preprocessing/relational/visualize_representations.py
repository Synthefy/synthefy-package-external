import argparse
import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def visualize_text_series_scatterplot(
    text_matrix: np.ndarray,
    series_matrix: np.ndarray,
):
    """
    Randomly sample 2D indices from the series matrix, then plot the scatterplot of the value of the text against the value of the series
    """
    # Randomly sample 2D indices from the series matrix
    indices = np.random.randint(0, series_matrix.shape[0], size=(1000, 2))

    sampled_text = text_matrix[indices[:, 0], indices[:, 1]]
    sampled_series = series_matrix[indices[:, 0], indices[:, 1]]
    sampled_text[sampled_text == -9999] = 0
    sampled_series[sampled_series == -9999] = 0

    # Calculate regression fit
    slope, intercept = np.polyfit(sampled_text, sampled_series, 1)

    # Calculate r-value (correlation coefficient)
    r_value = np.corrcoef(sampled_text, sampled_series)[0, 1]

    # Plot the scatterplot
    plt.scatter(sampled_text, sampled_series, alpha=0.7)

    # Add regression line
    x_range = np.linspace(np.min(sampled_text), np.max(sampled_text), 100)
    plt.plot(
        x_range,
        slope * x_range + intercept,
        "r-",
        label=f"y = {slope:.4f}x + {intercept:.4f}\nr = {r_value:.4f}",
    )

    # Add labels and legend
    plt.xlabel("Text Similarity")
    plt.ylabel("Series Similarity")
    plt.title("Text vs. Series Similarity")
    plt.legend()

    plt.show()
    plt.savefig(os.path.join(args.output_dir, "text_series_scatterplot.png"))
    plt.close()


def visualize_series_embeddings_scatterplot(
    dataset_dict: dict,
):
    """
    Visualize the low dimensional embeddings of the series
    """
    # Get the series embeddings
    series_embeddings = dataset_dict["reduced_time_series"]
    print(np.min(series_embeddings[:, 0]), np.max(series_embeddings[:, 0]))
    print(np.min(series_embeddings[:, 1]), np.max(series_embeddings[:, 1]))
    print(len(series_embeddings))

    # Plot the scatterplot of the value of the text against the value of the series
    plt.scatter(series_embeddings[:, 0], series_embeddings[:, 1], s=10)
    plt.show()
    plt.savefig(
        os.path.join(args.output_dir, "series_embeddings_scatterplot.png")
    )
    plt.close()


def visualize_text_embeddings_scatterplot(
    dataset_dict: dict,
):
    """
    Visualize the low dimensional embeddings of the series
    """
    # Get the series embeddings
    text_embeddings = dataset_dict["reduced_embeddings"]
    print(np.min(text_embeddings[:, 0]), np.max(text_embeddings[:, 0]))
    print(np.min(text_embeddings[:, 1]), np.max(text_embeddings[:, 1]))
    print(len(text_embeddings))

    # Plot the scatterplot of the value of the text against the value of the series
    plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], s=10)
    plt.show()
    plt.savefig(
        os.path.join(args.output_dir, "text_embeddings_scatterplot.png")
    )
    plt.close()


def visualize_dataset_series_embeddings_color_scatterplot(
    dataset_dict: dict,
):
    """
    Visualize the low dimensional embeddings of the text
    color the points based on both dimensions of the series embeddings
    """
    # Get the text embeddings
    text_embeddings = dataset_dict["reduced_embeddings"]

    # Get the series embeddings
    series_embeddings = dataset_dict["reduced_time_series"]

    # Create a colormap using both dimensions of the series embeddings
    # Normalize the values to [0,1] range for each dimension
    x_norm = (series_embeddings[:, 0] - np.min(series_embeddings[:, 0])) / (
        np.max(series_embeddings[:, 0])
        - np.min(series_embeddings[:, 0])
        + 1e-10
    )
    y_norm = (series_embeddings[:, 1] - np.min(series_embeddings[:, 1])) / (
        np.max(series_embeddings[:, 1])
        - np.min(series_embeddings[:, 1])
        + 1e-10
    )

    # Create a 2D colormap by mapping the two dimensions to RGB values
    # Use the first dimension for red-blue and second for green-yellow
    colors = np.zeros((len(x_norm), 3))
    colors[:, 0] = x_norm  # Red channel from first dimension
    colors[:, 1] = y_norm  # Green channel from second dimension
    colors[:, 2] = 1 - (x_norm + y_norm) / 2  # Blue channel as complement

    # Plot the scatterplot with the 2D color mapping
    plt.figure(figsize=(10, 8))
    plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], c=colors)

    # Add a colorbar legend to explain the mapping
    plt.colorbar(
        cm.ScalarMappable(cmap="rainbow"),
        label="Series embedding dimensions (dim0=R, dim1=G)",
    )
    # Add labels and title
    plt.xlabel("Text Embedding Dimension 1")
    plt.ylabel("Text Embedding Dimension 2")
    plt.title("Text Embeddings Colored by Series Embeddings")

    # Add a small legend explaining the color mapping
    plt.figtext(
        0.15,
        0.02,
        "Color mapping: Red = high dim0, Green = high dim1, Blue = low in both",
        ha="left",
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(
            args.output_dir, "text_series_embeddings_color_scatterplot.png"
        )
    )
    plt.close()


def visualize_clusters_scatterplot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str | None = None,
    title: str = "Cluster Visualization",
    s: int = 15,
    alpha: float = 0.7,
    embedding_type: str = "text",
):
    """
    Visualize clusters by coloring points according to cluster labels.

    Args:
        embeddings: 2D embeddings array of shape (n_samples, 2)
        labels: Cluster labels for each point
        output_path: Path to save the figure
        title: Plot title
        s: Marker size
        alpha: Marker transparency
        embedding_type: 'text' or 'series' to label axes appropriately
    """
    # Create figure
    plt.figure(figsize=(12, 10))

    # Get unique labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Use a colormap that works well for categorical data
    cmap = cm.get_cmap("tab10" if n_clusters <= 10 else "tab20")

    # Plot each cluster
    for i, label in enumerate(unique_labels):
        color = cmap(i / max(1, len(unique_labels) - 1))
        # Get mask for points in this cluster
        mask = labels == label

        # Plot points
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            s=s,
            alpha=alpha,
            color=color,
            label=f"Cluster {label}" if label != -1 else "Noise",
        )

    # Add title and labels
    plt.title(title)
    plt.xlabel(f"{embedding_type.capitalize()} Embedding Dimension 1")
    plt.ylabel(f"{embedding_type.capitalize()} Embedding Dimension 2")

    # Add legend (only if not too many clusters)
    if n_clusters <= 20:  # For readability
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), ncol=1)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def visualize_clusters_series_embeddings(
    dataset_dict: dict, clustering: dict, output_dir: str
):
    """
    Visualize the clusters of the series embeddings

    Args:
        dataset_dict: Dictionary containing the embeddings
        clustering: Dictionary containing cluster labels
        output_dir: Directory to save output
    """
    # Get series embeddings
    series_embeddings = dataset_dict["reduced_time_series"]

    # Get cluster labels
    cluster_labels = clustering

    # Ensure we have valid embeddings
    valid_mask = np.all(np.isfinite(series_embeddings), axis=1)
    if not np.any(valid_mask):
        print("No valid embeddings to visualize")
        return

    # Filter to valid embeddings only
    valid_embeddings = series_embeddings[valid_mask]
    valid_labels = cluster_labels[valid_mask]

    # Set up output path if directory provided
    output_path = os.path.join(output_dir, "series_clusters_scatterplot.png")

    # Visualize clusters
    visualize_clusters_scatterplot(
        valid_embeddings,
        valid_labels,
        output_path=output_path,
        title="Time Series Embeddings Colored by Cluster",
        embedding_type="series",
    )


def visualize_clusters_text_embeddings(
    dataset_dict: dict, clustering: dict, output_dir: str
):
    """
    Visualize the clusters of the text embeddings

    Args:
        dataset_dict: Dictionary containing the embeddings
        clustering: Dictionary containing cluster labels
        output_dir: Directory to save output
    """
    # Get text embeddings
    text_embeddings = dataset_dict["reduced_embeddings"]

    # Get cluster labels
    cluster_labels = clustering

    # Ensure we have valid embeddings
    valid_mask = np.all(np.isfinite(text_embeddings), axis=1)
    if not np.any(valid_mask):
        print("No valid embeddings to visualize")
        return

    # Filter to valid embeddings only
    valid_embeddings = text_embeddings[valid_mask]
    valid_labels = cluster_labels[valid_mask]

    # Set up output path if directory provided
    output_path = None
    output_path = os.path.join(output_dir, "text_clusters_scatterplot.png")

    # Visualize clusters
    visualize_clusters_scatterplot(
        valid_embeddings,
        valid_labels,
        output_path=output_path,
        title="Text Embeddings Colored by Cluster",
        embedding_type="text",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dict", type=str, required=True)
    parser.add_argument("--text_matrix", type=str, required=True)
    parser.add_argument("--series_matrix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clustering", type=str, required=True)
    args = parser.parse_args()

    # Load the dataset
    dataset_dict = pickle.load(open(args.dataset_dict, "rb"))
    clustering = pickle.load(open(args.clustering, "rb"))

    text_matrix = np.load(args.text_matrix)
    series_matrix = np.load(args.series_matrix)

    # Visualize the text series scatterplot
    visualize_text_series_scatterplot(text_matrix, series_matrix)

    # Visualize the series embeddings scatterplot
    visualize_series_embeddings_scatterplot(dataset_dict)

    # Visualize the dataset series embeddings color scatterplot
    visualize_dataset_series_embeddings_color_scatterplot(dataset_dict)

    # Visualize the text embeddings scatterplot
    visualize_text_embeddings_scatterplot(dataset_dict)

    # Visualize the clusters of the series embeddings
    print(list(clustering.keys()))
    visualize_clusters_series_embeddings(
        dataset_dict, clustering["cluster_labels"], args.output_dir
    )

    # Visualize the clusters of the text embeddings
    visualize_clusters_text_embeddings(
        dataset_dict, clustering["cluster_labels"], args.output_dir
    )

    # uv run src/synthefy_pkg/preprocessing/relational/visualize_representations.py --dataset_dict /home/data/all_univariate_filtered_dataset_dict/dataset_dict.pkl --text_matrix /home/data/visualize/dataset_references/relation_matrix_cosine_dist.npy --series_matrix /home/data/visualize/dataset_references/relation_matrix_cross_correlation.npy --output_dir /home/data/visualize/dataset_references/ --clustering /home/data/all_univariate_filtered_clustering/pretrain_blind_dict_0.pkl
