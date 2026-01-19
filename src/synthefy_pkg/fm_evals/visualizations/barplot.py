"""
Barplot visualization functions for dataset results.
"""

import os
from typing import List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from synthefy_pkg.fm_evals.formats.dataset_result_format import (
    DatasetResultFormat,
)


def plot_metrics_comparisons(
    dataset_results: List[DatasetResultFormat],
    model_names: List[str],
    output_path: str,
    figsize: tuple = (15, 10),
    dpi: int = 300,
    style: str = "whitegrid",
    palette: str = "husl",
    save_format: str = "png",
) -> None:
    """
    Create barplots for each metric comparing models across dataset results.

    Parameters
    ----------
    dataset_results : List[DatasetResultFormat]
        List of dataset result objects, one per model
    model_names : List[str]
        List of model names corresponding to the dataset results
    output_path : str
        Directory path where to save the plots
    figsize : tuple, optional
        Figure size (width, height) in inches, default (15, 10)
    dpi : int, optional
        DPI for saved plots, default 300
    style : str, optional
        Seaborn style to use, default "whitegrid"
    palette : str, optional
        Color palette for the plots, default "husl"
    save_format : str, optional
        Format to save plots (png, pdf, svg, etc.), default "png"

    Raises
    ------
    ValueError
        If the number of dataset results doesn't match the number of model names
        If any dataset result has no metrics
    """
    # Validate inputs
    if len(dataset_results) != len(model_names):
        raise ValueError(
            f"Number of dataset results ({len(dataset_results)}) must match "
            f"number of model names ({len(model_names)})"
        )

    # Check that all dataset results have metrics
    for i, result in enumerate(dataset_results):
        if result.metrics is None:
            raise ValueError(
                f"Dataset result {i} (model: {model_names[i]}) has no metrics"
            )

    # Set up the plotting style
    sns.set_style(style)  # type: ignore
    plt.rcParams["figure.dpi"] = dpi

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Define the metrics to plot
    metrics = {
        "mae": "Mean Absolute Error (MAE)",
        "mape": "Mean Absolute Percentage Error (MAPE)",
        "median_mae": "Median Absolute Error (Median MAE)",
        "median_mape": "Median Absolute Percentage Error (Median MAPE)",
    }

    # Create a single figure with subplots for all metrics
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Set color palette
    colors = sns.color_palette(palette, len(model_names))

    # Create barplots for each metric
    for idx, (metric_key, metric_title) in enumerate(metrics.items()):
        ax = cast(Axes, axes[idx])

        # Extract metric values for each model
        metric_values = []
        for result in dataset_results:
            metric_value = getattr(result.metrics, metric_key)
            metric_values.append(metric_value)

        # Create bar plot
        bars = ax.bar(
            model_names,
            metric_values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on top of bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Customize the plot
        ax.set_title(metric_title, fontsize=14, fontweight="bold", pad=20)
        ax.set_ylabel("Error Value", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Adjust layout and save
    plt.tight_layout()

    # Save the combined plot
    combined_filename = f"all_metrics_comparison.{save_format}"
    combined_filepath = os.path.join(output_path, combined_filename)
    plt.savefig(combined_filepath, bbox_inches="tight", dpi=dpi)
    plt.close()

    # Create individual plots for each metric
    for metric_key, metric_title in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = cast(Axes, ax)

        # Extract metric values for each model
        metric_values = []
        for result in dataset_results:
            metric_value = getattr(result.metrics, metric_key)
            metric_values.append(metric_value)

        # Create bar plot
        bars = ax.bar(
            model_names,
            metric_values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on top of bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Customize the plot
        ax.set_title(metric_title, fontsize=16, fontweight="bold", pad=20)
        ax.set_ylabel("Error Value", fontsize=14)
        ax.set_xlabel("Model", fontsize=14)
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Adjust layout and save
        plt.tight_layout()

        # Save individual plot
        individual_filename = f"{metric_key}_comparison.{save_format}"
        individual_filepath = os.path.join(output_path, individual_filename)
        plt.savefig(individual_filepath, bbox_inches="tight", dpi=dpi)
        plt.close()
