"""
Visualization utilities for SCM priors and generated DAGs.
"""

from typing import Dict, List, Optional, Tuple, Union, cast

import matplotlib.axes as mpl_axes  # type: ignore
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from synthefy_pkg.prior.mlp_scm import MLPSCM
from synthefy_pkg.prior.tree_scm import TreeSCM


def get_input_distribution(
    dag: Union[MLPSCM, TreeSCM], feature_idx: int
) -> str:
    """
    Get the distribution type for a given input feature.

    Parameters
    ----------
    dag : Union[MLPSCM, TreeSCM]
        The DAG containing the feature
    feature_idx : int
        Index of the feature to check

    Returns
    -------
    str
        Distribution type of the feature
    """
    if not hasattr(dag, "xsampler"):
        return "unknown"

    if dag.xsampler.sampling == "normal":
        return "normal"
    elif dag.xsampler.sampling == "uniform":
        return "uniform"
    elif dag.xsampler.sampling == "mixed":
        # For mixed sampling, we can't know exactly which distribution was used
        # for each feature, so we'll just indicate it's from a mixed distribution
        return "mixed"
    return "unknown"


def generate_summary_stats(X: torch.Tensor, y: torch.Tensor):
    """
    Generate summary statistics for features and target values.

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix of shape (n_samples, n_features)
    y : torch.Tensor
        Target values of shape (n_samples,)

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary containing statistics for features and target
    """
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    # Calculate statistics for each feature
    feature_stats = {}
    for i in range(X_np.shape[1]):
        feature_stats[f"X{i}"] = {
            "mean": float(np.mean(X_np[:, i])),
            "std": float(np.std(X_np[:, i])),
            "min": float(np.min(X_np[:, i])),
            "max": float(np.max(X_np[:, i])),
            "skew": float(stats.skew(X_np[:, i])),
            "kurtosis": float(stats.kurtosis(X_np[:, i])),
        }

    # Calculate statistics for target
    target_stats = {
        "mean": float(np.mean(y_np)),
        "std": float(np.std(y_np)),
        "min": float(np.min(y_np)),
        "max": float(np.max(y_np)),
        "skew": float(stats.skew(y_np)),
        "kurtosis": float(stats.kurtosis(y_np)),
    }

    # Add correlation with target
    for i in range(X_np.shape[1]):
        feature_stats[f"X{i}"]["corr_with_target"] = float(
            np.corrcoef(X_np[:, i], y_np)[0, 1]
        )

    return {"features": feature_stats, "target": target_stats}


def visualize_stats(
    X: torch.Tensor,
    y: torch.Tensor,
    dag: Optional[Union[MLPSCM, TreeSCM]] = None,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize statistics and distributions of features and target.

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix
    y : torch.Tensor
        Target values
    dag : Optional[Union[MLPSCM, TreeSCM]], default=None
        The DAG model used to generate the data
    feature_names : Optional[List[str]], default=None
        Names of the features
    figsize : Tuple[int, int], default=(15, 10)
        Figure size in inches
    save_path : Optional[str], default=None
        If provided, save the figure to this path
    show : bool, default=True
        Whether to display the figure
    """
    # Create figure with three subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # Feature distributions
    ax2 = fig.add_subplot(gs[0, 1])  # Target distribution
    ax3 = fig.add_subplot(gs[1, :])  # Summary statistics and DAG parameters

    # Get number of features
    n_features = X.shape[1]

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Convert to numpy
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    # Plot feature distributions
    for i in range(n_features):
        ax1.hist(X_np[:, i], bins=30, alpha=0.3, label=feature_names[i])

    ax1.set_title("Feature Distributions")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot target distribution
    ax2.hist(y_np, bins=30, color="purple", alpha=0.7)
    ax2.set_title("Target Distribution")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")

    # Generate statistics
    stats_dict = generate_summary_stats(X, y)

    # Add target statistics as text
    target_stats = stats_dict["target"]
    target_text = "Target Statistics:\n"
    for stat, value in target_stats.items():
        target_text += f"{stat}: {value:.3f}\n"

    # Add target text to the plot
    ax2.text(
        0.95,
        0.95,
        target_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontfamily="monospace",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Generate and display summary statistics for features
    # stats_text = "Feature Statistics (Top 5 by correlation with target):\n"
    # Sort features by correlation with target
    sorted_features = sorted(
        stats_dict["features"].items(),
        key=lambda x: abs(x[1]["corr_with_target"]),
        reverse=True,
    )[:5]

    # Prepare all text content
    # all_text = []

    # Add feature statistics as blocks
    feature_blocks = []
    for feature, stat_values in sorted_features:
        feature_block = [f"\n{feature}:"]
        for stat, value in stat_values.items():
            feature_block.append(f"  {stat}: {value:.3f}")
        feature_blocks.append(feature_block)

    # Add DAG parameters if provided
    dag_blocks = []
    if dag is not None:
        dag_blocks.append(["\nDAG Parameters:"])
        if isinstance(dag, MLPSCM):
            left_col = [
                ("Model Type:", "MLPSCM"),
                ("Number of Layers:", str(dag.num_layers)),
                ("Hidden Dimension:", str(dag.hidden_dim)),
                ("Block-wise Dropout:", str(dag.block_wise_dropout)),
                ("Initialization Std:", f"{dag.init_std:.3f}"),
                ("Number of Causes:", str(dag.num_causes)),
            ]
            right_col = [
                ("Is Causal:", str(dag.is_causal)),
                ("Y is Effect:", str(dag.y_is_effect)),
                ("In Clique:", str(dag.in_clique)),
            ]
            if not dag.block_wise_dropout:
                right_col.insert(
                    0, ("MLP Dropout Prob:", f"{dag.mlp_dropout_prob:.3f}")
                )
        elif isinstance(dag, TreeSCM):
            left_col = [
                ("Model Type:", "TreeSCM"),
                ("Tree Model:", str(dag.tree_model)),
                ("Tree Depth Lambda:", f"{dag.tree_depth_lambda:.3f}"),
                (
                    "Tree N Estimators Lambda:",
                    f"{dag.tree_n_estimators_lambda:.3f}",
                ),
                ("Number of Causes:", str(dag.num_causes)),
            ]
            right_col = [
                ("Is Causal:", str(dag.is_causal)),
                ("Y is Effect:", str(dag.y_is_effect)),
                ("In Clique:", str(dag.in_clique)),
            ]

        # Find the maximum width for labels and values
        left_label_width = max(len(label) for label, _ in left_col)
        right_label_width = max(len(label) for label, _ in right_col)

        # Add DAG parameters as a block
        dag_block = []
        for i in range(max(len(left_col), len(right_col))):
            if i < len(left_col):
                left_label, left_value = left_col[i]
                left = f"{left_label:<{left_label_width}} {left_value}"
            else:
                left = ""

            if i < len(right_col):
                right_label, right_value = right_col[i]
                right = f"{right_label:<{right_label_width}} {right_value}"
            else:
                right = ""

            dag_block.append(f"{left:<50} {right}")
        dag_blocks.append(dag_block)

    # Combine all blocks
    all_blocks = feature_blocks + dag_blocks

    # first determine where to split the blocks into two columns
    mid_point = len(all_blocks) // 2
    left_blocks = all_blocks[:mid_point]
    right_blocks = all_blocks[mid_point:]

    # Create the final text with two columns
    final_text = ""
    for block in left_blocks:
        final_text += "\n".join(block)
        final_text += "\n"

    # write out the blocks to the left column
    ax3.text(
        0.05,
        0.95,
        final_text,
        transform=ax3.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        fontsize=10,
    )

    # write out the blocks to the right column
    final_text = ""
    for block in right_blocks:
        final_text += "\n".join(block)
        final_text += "\n"

    ax3.text(
        0.55,
        0.95,
        final_text,
        transform=ax3.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        fontsize=10,
    )
    ax3.axis("off")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch_signals(
    X: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    seq_lens: torch.Tensor,
    train_sizes: torch.Tensor,
    save_dir: Optional[str] = None,
    batch_idx: int = 0,
    max_features_per_plot: int = 10,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
    node_info_array: Optional[np.ndarray] = None,
    target_node_info_array: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize each signal in the batch when running single_dataset.get_batch.

    This function creates time series plots for each dataset in the batch, showing
    both features and target values over time. It handles variable sequence lengths
    and different numbers of features per dataset.

    Parameters
    ----------
    X : torch.Tensor
        Features tensor of shape (batch_size, seq_len, max_features)
    y : torch.Tensor
        Labels tensor of shape (batch_size, seq_len)
    d : torch.Tensor
        Number of active features per dataset, shape (batch_size,)
    seq_lens : torch.Tensor
        Sequence length for each dataset, shape (batch_size,)
    train_sizes : torch.Tensor
        Train/test split position for each dataset, shape (batch_size,)
    save_dir : Optional[str], default=None
        Directory to save the visualization plots. If None, plots are not saved.
    batch_idx : int, default=0
        Batch index for file naming
    max_features_per_plot : int, default=10
        Maximum number of features to show in each subplot to avoid overcrowding
    figsize : Tuple[int, int], default=(15, 10)
        Figure size in inches
    show : bool, default=True
        Whether to display the figure
    node_info_array : Optional[np.ndarray], default=None
        Node information array of shape (batch_size, max_features, 2) where the last dimension
        contains [layer_number, index_within_layer] for each feature
    """
    batch_size = X.shape[0]

    # Convert tensors to numpy for plotting
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    d_np = d.cpu().numpy()
    seq_lens_np = seq_lens.cpu().numpy()
    train_sizes_np = train_sizes.cpu().numpy()
    print(d_np)

    for dataset_idx in range(batch_size):
        # Get dataset-specific parameters
        seq_len = int(seq_lens_np[dataset_idx])
        num_features = int(d_np[dataset_idx])
        train_size = int(train_sizes_np[dataset_idx])

        # Extract data for this dataset
        print(X_np.shape)
        dataset_X = X_np[dataset_idx, :seq_len, :num_features]
        dataset_y = y_np[dataset_idx, :seq_len]

        # Create time indices
        time_indices = np.arange(seq_len)

        # Calculate number of plots needed
        total_features = num_features + 1  # +1 for target

        # Adjust total_features to match the actual number of features with node info
        if (
            node_info_array is not None
            and dataset_idx < node_info_array.shape[0]
        ):
            # Count features with valid node info (including time features with layer = -1)
            valid_features = np.any(
                node_info_array[dataset_idx] != [0, 0], axis=1
            )
            actual_features_with_info = np.sum(valid_features)
            total_features = actual_features_with_info + 1  # +1 for target

        num_plots = (
            total_features + max_features_per_plot - 1
        ) // max_features_per_plot

        for plot_idx in range(num_plots):
            # Calculate features for this plot
            start_feature = plot_idx * max_features_per_plot
            end_feature = min(
                start_feature + max_features_per_plot, total_features
            )
            features_in_plot = end_feature - start_feature

            # Determine if this plot contains the target
            target_in_plot = (
                start_feature <= actual_features_with_info < end_feature
            )
            features_before_target = max(
                0, actual_features_with_info - start_feature
            )

            # Create figure for this plot
            fig, axes = plt.subplots(
                features_in_plot, 1, figsize=figsize, sharex=True
            )

            # Handle different axes structures
            if features_in_plot == 1:
                axes_list = [axes]
            elif isinstance(axes, np.ndarray):
                axes_list = axes.flatten().tolist()
            else:
                axes_list = axes if isinstance(axes, list) else [axes]

            # Plot features in this range
            for i, feature_idx in enumerate(
                range(
                    start_feature, min(end_feature, actual_features_with_info)
                )
            ):
                if i < len(axes_list):
                    ax = axes_list[i]
                    if hasattr(ax, "plot"):
                        # Get node information if available
                        if (
                            node_info_array is not None
                            and dataset_idx < node_info_array.shape[0]
                            and feature_idx < node_info_array.shape[1]
                        ):
                            layer, local_idx = node_info_array[
                                dataset_idx, feature_idx
                            ]
                            # Check if this is a time feature (layer = -1)
                            if layer == -1:
                                feature_label = f"TimeF{local_idx}"
                                ylabel = f"TimeF{local_idx}"
                            else:
                                feature_label = (
                                    f"F{feature_idx} (L{layer}N{local_idx})"
                                )
                                ylabel = f"F{feature_idx}\nL{layer}N{local_idx}"
                        else:
                            feature_label = f"Feature {feature_idx}"
                            ylabel = f"Feature {feature_idx}"

                        ax.plot(  # type: ignore
                            time_indices,
                            dataset_X[:, feature_idx],  # type: ignore
                            label=feature_label,
                            linewidth=2,
                        )
                        ax.set_ylabel(ylabel)  # type: ignore
                        ax.grid(True, alpha=0.3)  # type: ignore

                        # Add train/test split line
                        if train_size < seq_len:
                            ax.axvline(  # type: ignore
                                x=train_size,
                                color="red",
                                linestyle="--",  # type: ignore
                                alpha=0.7,
                                label="Train/Test Split",
                            )

                        # Add legend for the first few features
                        if i < 3:
                            ax.legend(loc="upper right")  # type: ignore
                    else:
                        print(
                            f"Warning: axes_list[{i}] is not a matplotlib axes object"
                        )
                else:
                    print(
                        f"Warning: No axes available for feature {feature_idx}"
                    )

            # Plot target if it belongs in this plot
            if target_in_plot:
                target_ax_idx = features_before_target
                if target_ax_idx < len(axes_list):
                    ax_target = axes_list[target_ax_idx]
                    if hasattr(ax_target, "plot"):
                        # Get target node information if available
                        if (
                            target_node_info_array is not None
                            and dataset_idx < target_node_info_array.shape[0]
                        ):
                            layer, local_idx = target_node_info_array[
                                dataset_idx
                            ]
                            target_label = f"Target (L{layer}N{local_idx})"
                            ylabel = f"Target\nL{layer}N{local_idx}"
                        else:
                            target_label = "Target"
                            ylabel = "Target"

                        ax_target.plot(  # type: ignore
                            time_indices,
                            dataset_y,  # type: ignore
                            label=target_label,
                            color="purple",
                            linewidth=2,
                        )
                        ax_target.set_ylabel(ylabel)  # type: ignore
                        ax_target.grid(True, alpha=0.3)  # type: ignore

                        # Add train/test split line for target
                        if train_size < seq_len:
                            ax_target.axvline(  # type: ignore
                                x=train_size,
                                color="red",
                                linestyle="--",  # type: ignore
                                alpha=0.7,
                                label="Train/Test Split",
                            )

                        ax_target.legend(loc="upper right")  # type: ignore
                    else:
                        print(
                            "Warning: ax_target is not a matplotlib axes object"
                        )
                else:
                    print("Warning: No axes available for target plot")

            # Set x-label for the last subplot
            if len(axes_list) > 0:
                last_ax = axes_list[-1]
                if hasattr(last_ax, "set_xlabel"):
                    last_ax.set_xlabel("Time Step")  # type: ignore
            else:
                print("Warning: No axes available for x-label")

            # Add title with dataset information
            if num_plots == 1:
                title = f"Dataset {dataset_idx} (Batch {batch_idx})\nSequence Length: {seq_len}, Features: {num_features}, Train Size: {train_size}"
            else:
                title = f"Dataset {dataset_idx} (Batch {batch_idx}) - Plot {plot_idx + 1}/{num_plots}\nFeatures {start_feature} to {end_feature - 1}"

            fig.suptitle(title, fontsize=14, fontweight="bold")

            plt.tight_layout()

            # Save if directory provided
            if save_dir:
                import os

                os.makedirs(save_dir, exist_ok=True)
                if num_plots == 1:
                    save_path = os.path.join(
                        save_dir,
                        f"batch_{batch_idx:03d}_dataset_{dataset_idx:03d}_signals.png",
                    )
                else:
                    save_path = os.path.join(
                        save_dir,
                        f"batch_{batch_idx:03d}_dataset_{dataset_idx:03d}_signals_part{plot_idx + 1}.png",
                    )
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                print(f"Saved signal visualization to {save_path}")

            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()


def visualize_batch_summary(
    X: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    seq_lens: torch.Tensor,
    train_sizes: torch.Tensor,
    save_dir: Optional[str] = None,
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
) -> None:
    """
    Create a summary visualization of the entire batch.

    This function creates a comprehensive overview of all datasets in the batch,
    including statistics, distributions, and sample time series.

    Parameters
    ----------
    X : torch.Tensor
        Features tensor of shape (batch_size, seq_len, max_features)
    y : torch.Tensor
        Labels tensor of shape (batch_size, seq_len)
    d : torch.Tensor
        Number of active features per dataset, shape (batch_size,)
    seq_lens : torch.Tensor
        Sequence length for each dataset, shape (batch_size,)
    train_sizes : torch.Tensor
        Train/test split position for each dataset, shape (batch_size,)
    save_dir : Optional[str], default=None
        Directory to save the visualization plots. If None, plots are not saved.
    batch_idx : int, default=0
        Batch index for file naming
    figsize : Tuple[int, int], default=(15, 10)
        Figure size in inches
    show : bool, default=True
        Whether to display the figure
    """
    batch_size = X.shape[0]

    # Convert tensors to numpy for plotting
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    d_np = d.cpu().numpy()
    seq_lens_np = seq_lens.cpu().numpy()
    train_sizes_np = train_sizes.cpu().numpy()

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3)

    # Plot 1: Sequence length distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(seq_lens_np, bins=min(10, batch_size), alpha=0.7, color="blue")
    ax1.set_title("Sequence Length Distribution")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Count")

    # Plot 2: Number of features distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(d_np, bins=min(10, batch_size), alpha=0.7, color="green")
    ax2.set_title("Number of Features Distribution")
    ax2.set_xlabel("Number of Features")
    ax2.set_ylabel("Count")

    # Plot 3: Train size distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(
        train_sizes_np, bins=min(10, batch_size), alpha=0.7, color="orange"
    )
    ax3.set_title("Train Size Distribution")
    ax3.set_xlabel("Train Size")
    ax3.set_ylabel("Count")

    # Plot 4: Sample time series (first dataset)
    ax4 = fig.add_subplot(gs[1, :])
    dataset_idx = 0
    seq_len = int(seq_lens_np[dataset_idx])
    num_features = min(3, int(d_np[dataset_idx]))  # Show first 3 features
    time_indices = np.arange(seq_len)

    for feature_idx in range(num_features):
        ax4.plot(
            time_indices,
            X_np[dataset_idx, :seq_len, feature_idx],
            label=f"Feature {feature_idx}",
            alpha=0.8,
        )

    # Add target
    ax4.plot(
        time_indices,
        y_np[dataset_idx, :seq_len],
        label="Target",
        color="purple",
        linewidth=2,
    )

    # Add train/test split
    train_size = int(train_sizes_np[dataset_idx])
    if train_size < seq_len:
        ax4.axvline(
            x=train_size,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Train/Test Split",
        )

    ax4.set_title("Sample Time Series (Dataset 0)")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Value")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Batch statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    # Calculate and display statistics
    stats_text = f"Batch {batch_idx} Summary Statistics:\n\n"
    stats_text += f"Batch Size: {batch_size}\n"
    stats_text += f"Average Sequence Length: {np.mean(seq_lens_np):.1f} ± {np.std(seq_lens_np):.1f}\n"
    stats_text += f"Average Number of Features: {np.mean(d_np):.1f} ± {np.std(d_np):.1f}\n"
    stats_text += f"Average Train Size: {np.mean(train_sizes_np):.1f} ± {np.std(train_sizes_np):.1f}\n"
    stats_text += f"Min/Max Sequence Length: {np.min(seq_lens_np)} / {np.max(seq_lens_np)}\n"
    stats_text += f"Min/Max Features: {np.min(d_np)} / {np.max(d_np)}\n"
    stats_text += f"Min/Max Train Size: {np.min(train_sizes_np)} / {np.max(train_sizes_np)}\n\n"

    # Add target statistics
    all_targets = np.concatenate(
        [y_np[i, : seq_lens_np[i]] for i in range(batch_size)]
    )
    stats_text += "Target Statistics:\n"
    stats_text += f"  Mean: {np.mean(all_targets):.3f}\n"
    stats_text += f"  Std: {np.std(all_targets):.3f}\n"
    stats_text += f"  Min: {np.min(all_targets):.3f}\n"
    stats_text += f"  Max: {np.max(all_targets):.3f}\n"

    ax5.text(
        0.05,
        0.95,
        stats_text,
        transform=ax5.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    # Save if directory provided
    if save_dir:
        import os

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"batch_{batch_idx:03d}_summary.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved batch summary to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
