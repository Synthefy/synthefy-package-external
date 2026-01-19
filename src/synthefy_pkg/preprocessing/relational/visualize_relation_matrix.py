import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from synthefy_pkg.preprocessing.relational.construct_dataset_matrix import (
    RelationConfig,
    load_relational_dataset_matrix,
)
from synthefy_pkg.preprocessing.relational.construct_relational_matrix import (
    load_relational_config,
)


def visualize_relation_matrix(
    matrix: np.ndarray,
    output_dir: str,
    title: str = "Relation Matrix",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Visualize a full relation matrix as a heatmap.

    Args:
        matrix: 2D numpy array of relations
        output_dir: Directory to save the figure
        title: Plot title
        cmap: Colormap to use
        vmin, vmax: Value range for colormap
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Similarity")
    plt.title(title)
    plt.tight_layout()
    logger.info(
        f"Saving relation matrix heatmap to {output_dir}/relation_matrix_heatmap.png"
    )
    plt.savefig(
        os.path.join(output_dir, "relation_matrix_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualize_relation_histogram(
    matrix: np.ndarray,
    output_dir: str,
    title: str = "Relation Value Distribution",
    bins: int = 100,
):
    """
    Create a histogram of all relation values in the matrix.

    Args:
        matrix: 2D numpy array of relations
        output_dir: Directory to save the figure
        title: Plot title
        bins: Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    plt.hist(matrix.flatten(), bins=bins, alpha=0.7)
    plt.title(title)
    plt.xlabel("Relation Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Add statistics
    values = matrix.flatten()
    stats_text = f"Mean: {np.mean(values):.3f}\nStd: {np.std(values):.3f}\nMax: {np.max(values):.3f}\nMin: {np.min(values):.3f}"
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    logger.info(
        f"Saving relation histogram to {output_dir}/relation_histogram.png"
    )
    plt.savefig(os.path.join(output_dir, "relation_histogram.png"), dpi=300)
    plt.close()


def visualize_top_relationships(
    matrix: np.ndarray,
    output_dir: str,
    n_top: int = 500,
    n_random_rows: int = 20,
    title_prefix: str = "Top Relationships",
):
    """
    Generate a histogram of all top relationships and histograms for randomly selected rows.

    Args:
        matrix: 2D numpy array of relations
        output_dir: Directory to save histograms
        n_top: Number of top relationships to consider per row
        n_random_rows: Number of random rows to visualize
        title_prefix: Prefix for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all top k values across all rows
    all_top_values = []
    for i in range(len(matrix)):
        row = matrix[i].copy()
        row[i] = -np.inf  # Exclude self-similarity
        top_indices = np.argsort(row)[-n_top:]
        all_top_values.extend(row[top_indices])

    # Create histogram of all top values
    plt.figure(figsize=(12, 6))
    plt.hist(all_top_values, bins=50, alpha=0.7)
    plt.title(f"{title_prefix} - All Top {n_top} Values")
    plt.xlabel("Similarity Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Mean: {np.mean(all_top_values):.3f}\nStd: {np.std(all_top_values):.3f}\nMax: {np.max(all_top_values):.3f}"
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    logger.info(
        f"Saving combined top relationships histogram to {output_dir}/all_top_relationships.png"
    )
    plt.savefig(os.path.join(output_dir, "all_top_relationships.png"), dpi=300)
    plt.close()

    # Visualize n_random_rows randomly selected rows
    random_rows = np.random.choice(len(matrix), n_random_rows, replace=False)
    # clean the directory of random row values before saving
    for file in os.listdir(output_dir):
        if file.startswith("top_relationships_row_"):
            os.remove(os.path.join(output_dir, file))
    for i in random_rows:
        row = matrix[i].copy()
        row[i] = -np.inf  # Exclude self-similarity
        top_indices = np.argsort(row)[-n_top:]
        top_values = row[top_indices]

        plt.figure(figsize=(10, 6))
        plt.hist(top_values, bins=50, alpha=0.7)
        plt.title(f"{title_prefix} for Random Row {i}")
        plt.xlabel("Similarity Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Mean: {np.mean(top_values):.3f}\nStd: {np.std(top_values):.3f}\nMax: {np.max(top_values):.3f}"
        plt.text(
            0.95,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        logger.info(
            f"Saving top relationships histogram for random row {i} to {output_dir}/top_relationships_row_{i}.png"
        )
        plt.savefig(
            os.path.join(output_dir, f"top_relationships_row_{i}.png"), dpi=300
        )
        plt.close()


def visualize_relation_matrix_analysis(
    relation_config: RelationConfig,
    relation_dataset_dir: str,
    output_dir: str,
    title_prefix: str = "Relation Matrix Analysis",
    split: str = "train",
    uid: str = "0",
):
    """
    Load a relation matrix and create comprehensive visualizations.

    Args:
        relation_config: Configuration for loading the relation matrix
        output_dir: Directory to save visualizations
        title_prefix: Prefix for plot titles
        split: Dataset split to use
        uid: Unique identifier for the analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load matrix
    logger.info("Loading relation matrix")

    # Find out the number of rows in the dataset

    matrix = load_relational_dataset_matrix(
        dataset_ids=np.arange(5000),
        relation_config=relation_config,
        output_dir=output_dir,
        split=split,
        uid=uid,
        slice_dataset_matrix_num=-1,  # Load full matrix
        save=False,
        load_from_dir=relation_dataset_dir,
    )
    matrix[matrix == -9999] = 0
    print(np.min(matrix), np.max(matrix))

    # Visualize full matrix
    logger.info("Creating heatmap visualization")
    visualize_relation_matrix(
        matrix, output_dir, title=f"{title_prefix} - Full Matrix"
    )

    # Visualize overall distribution
    logger.info("Creating overall distribution histogram")
    visualize_relation_histogram(
        matrix, output_dir, title=f"{title_prefix} - Value Distribution"
    )

    # Visualize top relationships for each row
    logger.info("Creating top relationships histograms")
    visualize_top_relationships(
        matrix,
        os.path.join(output_dir, "top_relationships"),
        title_prefix=f"{title_prefix} - Top Relationships",
    )

    logger.success(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize relation matrix analysis"
    )
    parser.add_argument(
        "--relation_config",
        type=str,
        required=True,
        help="Path to relation config file",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--relation_dataset_dir",
        type=str,
        required=True,
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="Relation Matrix Analysis",
        help="Prefix for plot titles",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--uid",
        type=str,
        default="0",
        help="Unique identifier for the analysis",
    )
    args = parser.parse_args()

    # Load relation config
    relation_config = load_relational_config(
        args.relation_config, args.dataset_dir, args.output_dir
    )

    visualize_relation_matrix_analysis(
        relation_config=relation_config,
        relation_dataset_dir=args.relation_dataset_dir,
        output_dir=args.output_dir,
        title_prefix=args.title_prefix,
        split=args.split,
        uid=args.uid,
    )
    # uv run src/synthefy_pkg/preprocessing/relational/visualize_relation_matrix.py --relation_config src/synthefy_pkg/foundation_model/configs/relation_configs/config_foundation_model_v2_relation_granger.yaml --relation_dataset_dir /home/data/relation_5k_filtered_granger_overlap/ --dataset_dir /home/data/foundation_model_data_all_univariate_filtered_5000_0_5000_ts_2021-06-01/ --output_dir /home/data/visualize/dataset_histograms/
