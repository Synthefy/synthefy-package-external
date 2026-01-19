import base64
import glob
import json
import os
from ast import Tuple
from typing import Any, List, cast

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

COMPILE = True

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


def check_plot_exists(plot_path, error_msg):
    """
    Check if the plot file exists at the given path. If not, log an error and raise a FileNotFoundError.

    Args:
        plot_path (str): Path to the plot file.
        error_msg (str): Error message to log and raise if the file is not found.

    Raises:
        FileNotFoundError: If the plot file is not found at the given path.
    """
    if not os.path.isfile(plot_path):
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def get_image_base64(image_path: str) -> str:
    """
    Read an image file and convert it to base64 encoding.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.

    Raises:
        FileNotFoundError: If the image file is not found at the given path.
    """

    if not os.path.isfile(image_path):
        error_msg = f"Image file not found: {image_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded_string


def _read_and_concat_logs(
    input_logs_dir,
):
    """
    Read all metrics.csv files in the input logs directory and concatenate them into a single DataFrame.

    Args:
        input_logs_dir (str): Path to the directory containing Lightning log version subdirectories.

    Returns:
        tuple: A tuple containing:
            - DataFrame with concatenated metrics from all valid version directories
            - List of epoch numbers where checkpoints were restarted

    Raises:
        FileNotFoundError: If input directory doesn't exist or no version directories found
        ValueError: If no valid metrics data found in any version directory
    """
    if not os.path.isdir(input_logs_dir):
        raise FileNotFoundError(
            f"Input logs directory not found: {input_logs_dir}"
        )

    # Locate version directories (e.g., version_0, version_1, …)
    version_dirs = sorted(
        glob.glob(os.path.join(input_logs_dir, "version_*")),
        key=lambda x: (
            int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
        ),
    )

    if not version_dirs:
        raise FileNotFoundError(
            f"No version directories found in: {input_logs_dir}"
        )

    epoch_offset = 0  # Cumulative epoch offset
    checkpoint_epochs = []  # List of epochs where a new CSV (checkpoint restart) begins
    dfs = []

    for i, vdir in enumerate(version_dirs):
        metrics_csv_path = os.path.join(vdir, "metrics.csv")
        if not os.path.exists(metrics_csv_path):
            logger.warning(
                f"metrics.csv not found in {vdir}. Skipping this version..."
            )
            continue

        df = pd.read_csv(metrics_csv_path)

        if "epoch" not in df.columns:
            logger.warning(
                f"'epoch' column not found in {metrics_csv_path}. Skipping this version..."
            )
            continue

        if "train_loss_epoch" not in df.columns:
            logger.warning(
                f"'train_loss_epoch' column not found in {metrics_csv_path}. Skipping this version..."
            )
            continue

        # Record the starting epoch for this CSV (if not the first run).
        local_start = epoch_offset
        if i > 0:
            checkpoint_epochs.append(local_start)

        # Determine the highest epoch with complete training loss.
        complete_epochs = df.loc[df["train_loss_epoch"].notna(), "epoch"]
        if complete_epochs.empty:
            logger.warning(
                f"No complete training epoch found in {metrics_csv_path}. Skipping this version..."
            )
            continue

        last_complete_epoch = complete_epochs.max()
        # Discard any rows beyond the last complete epoch.
        df = df[df["epoch"] <= last_complete_epoch].copy()

        # Offset the local epoch values so the combined curve is continuous.
        df["epoch"] = df["epoch"] + epoch_offset

        # Update epoch_offset to the maximum epoch from this CSV.
        epoch_offset = int(df["epoch"].max())
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid metrics data found in any version directory")

    return pd.concat(dfs, ignore_index=True), checkpoint_epochs


def plot_learning_curve(
    input_logs_dir,
    output_fig_path,
    run_name="",
    dataset_name="",
    log_to_mlflow=False,
    experiment_name: str = "",
    tracking_uri: str = "",
    run_id: str | None = None,
):
    """
    Plots the learning curve for training, validation, and test losses by combining the
    metrics.csv files from all Lightning log versions in the specified logs directory.

    For each CSV file, the function:
      - Finds the maximum epoch for which a complete training loss (train_loss_epoch) is recorded.
      - Discards any rows corresponding to epochs beyond that maximum (i.e. incomplete epochs).
      - Offsets the epoch numbers so that the combined learning curve is continuous.
      - Records the starting epoch of each CSV (except the first) and draws a vertical dashed line
        on the plot to mark the restart from a checkpoint.

    Args:
        input_logs_dir: Path to the Lightning logs directory containing subdirectories
                        like 'version_0', 'version_1', etc. Each subdirectory should have a
                        metrics.csv file.
        output_fig_path: Path to save the output figure of the learning curve.
        run_name: Name of the run to display in the plot title.
        dataset_name: Name of the dataset to display in the plot title.
        log_to_mlflow: Whether to log the plot to MLflow as an artifact (default: False).
        experiment_name: Name of the MLflow experiment.
        tracking_uri: MLflow tracking URI.
        run_id: Run ID for logging to MLflow.

    Raises:
        FileNotFoundError: If the input logs directory or metrics.csv files are not found.
        ValueError: If no valid metrics data is found.
    """

    combined_df, checkpoint_epochs = _read_and_concat_logs(input_logs_dir)
    if combined_df.empty:
        logger.error(
            "No valid metrics found in the provided logs directory. Skipping plot..."
        )
        raise ValueError(
            "No valid metrics found in the provided logs directory."
        )

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_fig_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Extract losses
    try:
        train_epochs = combined_df.dropna(subset=["train_loss_epoch"]).copy()
    except KeyError:
        logger.error("No training loss found in CSV logs. Skipping plot...")
        raise ValueError("No training loss found in CSV logs.")

    val_points = (
        combined_df.dropna(subset=["val_loss"]).copy()
        if "val_loss" in combined_df.columns
        else pd.DataFrame()
    )
    test_points = (
        combined_df.dropna(subset=["test_loss"]).copy()
        if "test_loss" in combined_df.columns
        else pd.DataFrame()
    )

    # Create the plot
    fig: Figure | Any
    ax: Axes | Any
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot training loss
    ax.plot(
        train_epochs["epoch"],
        train_epochs["train_loss_epoch"],
        color="#D62728",
        marker="o",
        markerfacecolor="#FFCDD2",
        markersize=10,
        linewidth=2,
        label="Training Loss",
    )

    # Plot validation loss if available
    if not val_points.empty:
        ax.plot(
            val_points["epoch"],
            val_points["val_loss"],
            color="#2CA02C",
            marker="*",
            linewidth=2,
            markersize=10,
            markerfacecolor="#C8E6C9",
            label="Validation Loss",
        )
        # Mark the minimum validation loss with a star
        min_val_idx = val_points["val_loss"].idxmin()
        min_epoch = val_points.loc[min_val_idx, "epoch"]
        min_val = val_points.loc[min_val_idx, "val_loss"]
        ax.plot(
            min_epoch,
            min_val,
            marker="*",
            markersize=10,
            color="gold",
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=100,
            label="Min Validation Loss",
        )

    # Plot test loss if available
    if not test_points.empty:
        ax.plot(
            test_points["epoch"],
            test_points["test_loss"],
            color="black",
            marker="s",
            linewidth=2,
            markersize=10,
            markerfacecolor="#BBDEFB",
            label="Test Loss",
        )

    # Add vertical dashed lines for checkpoint restarts
    for j, cp_epoch in enumerate(checkpoint_epochs):
        # Add a label only once for clarity
        if j == 0:
            ax.axvline(
                x=cp_epoch,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="Checkpoint Restart",
            )
        else:
            ax.axvline(
                x=cp_epoch,
                color="gray",
                linestyle="--",
                linewidth=1.5,
            )

    # Configure plot styling
    ax.set_title("Training Progress Over Epochs", fontsize=24, pad=15)
    ax.set_xlabel("Epoch", fontsize=20, labelpad=12)
    ax.set_ylabel("Loss", fontsize=20, labelpad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=14)

    # Optionally add run and dataset info text
    info_text = []
    if run_name:
        info_text.append(f"Run Name: {run_name}")
    if dataset_name:
        info_text.append(f"Dataset Name: {dataset_name}")

    if info_text:
        ax.text(
            0.01,
            0.98,
            "\n".join(info_text),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor="white",
                edgecolor="lightgray",
                boxstyle="square,pad=0.7",
            ),
        )

    # Add legend and save the figure
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    plt.tight_layout()

    success = False
    try:
        fig.savefig(output_fig_path)
        success = True
    except Exception as e:
        logger.error(f"Failed to save plot to {output_fig_path}: {str(e)}")
        raise
    finally:
        plt.close(fig)

    # Log to MLflow only if the plot was successfully created and saved
    if success and log_to_mlflow:
        try:
            if experiment_name and tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
            mlflow.log_artifact(output_fig_path, "plots", run_id=run_id)
        except Exception as e:
            logger.error(f"Failed to log plot to MLflow: {str(e)}")


def downsample_by_random_sampling(true_values, pred_values, factor, seed=42):
    """
    Downsample two 1D arrays by randomly selecting one point from each group of consecutive points,
    ensuring both arrays are downsampled based on the same indices.

    Args:
        true_values (numpy.ndarray): The first input 1D array to be downsampled.
        pred_values (numpy.ndarray): The second input 1D array to be downsampled.
        factor (int): The downsampling factor. One random point will be selected from each
                      'factor' consecutive points. Must be greater than 0.
        seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        tuple: Two downsampled arrays with length approximately len(true_values)/factor and len(pred_values)/factor.

    Raises:
        ValueError: If factor is less than 1 or if the inputs are not 1D numpy arrays.
    """
    # Check if true_values and pred_values are 1D numpy arrays
    if not isinstance(true_values, np.ndarray) or len(true_values.shape) != 1:
        raise ValueError("Input true_values must be a 1D numpy array")
    if not isinstance(pred_values, np.ndarray) or len(pred_values.shape) != 1:
        raise ValueError("Input pred_values must be a 1D numpy array")

    if factor < 1:
        raise ValueError("Factor must be greater than 0")

    if factor == 1:
        return true_values, pred_values

    np.random.seed(seed)

    n = len(true_values)
    # Ensure true_values and pred_values have the same length
    if len(pred_values) != n:
        raise ValueError(
            "Input arrays true_values and pred_values must have the same length"
        )

    # Trim true_values and pred_values to ensure length is a multiple of factor
    trimmed_length = n - (n % factor)
    true_values = true_values[:trimmed_length]
    pred_values = pred_values[:trimmed_length]

    # Reshape into groups of 'factor' consecutive points
    reshaped_true_values = true_values.reshape(-1, factor)
    reshaped_pred_values = pred_values.reshape(-1, factor)

    # For each group, randomly choose one index
    random_indices = np.random.randint(
        0, factor, size=reshaped_true_values.shape[0]
    )

    # Extract the randomly selected points for both true_values and pred_values
    downsampled_true_values = reshaped_true_values[
        np.arange(reshaped_true_values.shape[0]), random_indices
    ]
    downsampled_pred_values = reshaped_pred_values[
        np.arange(reshaped_pred_values.shape[0]), random_indices
    ]
    return downsampled_true_values, downsampled_pred_values


# Maximum number of data points before downsampling
MAX_DATA_POINTS = 2000000


def determine_dynamic_downsample_factor(num_steps, num_channels):
    """
    Determine the dynamic downsample factor based on the data size.

    Calculate downsample factor considering both data size and channels together.
    For larger datasets or more channels, we need more aggressive downsampling.

    Args:
        num_steps (int): Number of time steps in the data
        num_channels (int): Number of channels/variables in the data

    Returns:
        int: The calculated downsample factor (minimum 1)
    """

    # The total data points to consider is num_steps * num_channels
    total_data_points = num_steps * num_channels

    # Determine downsample factor based on total data points
    downsample_factor = max(1, int(total_data_points / MAX_DATA_POINTS))

    logger.info(
        f"Total data points: {total_data_points}, downsample factor: {downsample_factor}"
    )

    # Ensure we have a minimum factor of 1
    downsample_factor = max(1, downsample_factor)

    return downsample_factor


def dynamic_downsample_true_and_predicted_values(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    num_channels: int,
    downsample_factor: int | None = None,
):
    """
    Downsample two lists of values based on the data size and number of channels.

    Args:
        true_values: The true values to downsample.
        pred_values: The predicted values to downsample.
        num_channels: The number of channels/variables in the data.
        downsample_factor (optional): The downsample factor to use. If not provided, the function will calculate an appropriate factor.

    Returns:
        tuple: A tuple containing the downsampled true and predicted values.
    """
    if not downsample_factor:
        downsample_factor = determine_dynamic_downsample_factor(
            len(true_values), num_channels
        )

    downsampled_true_values, downsampled_pred_values = (
        downsample_by_random_sampling(
            true_values, pred_values, downsample_factor
        )
    )

    return downsampled_true_values, downsampled_pred_values


def _load_metadata_col_names(dataset_name: str):
    """Load discrete and continuous metadata column names from files."""
    discrete_filepath = os.path.join(
        str(SYNTHEFY_DATASETS_BASE),
        dataset_name,
        "discrete_windows_columns.json",
    )
    continuous_filepath = os.path.join(
        str(SYNTHEFY_DATASETS_BASE),
        dataset_name,
        "continuous_windows_columns.json",
    )

    discrete_col_names = []
    continuous_col_names = []

    if os.path.exists(discrete_filepath):
        with open(discrete_filepath, "r") as f:
            discrete_col_names = json.load(f)

    if os.path.exists(continuous_filepath):
        with open(continuous_filepath, "r") as f:
            continuous_col_names = json.load(f)

    return discrete_col_names, continuous_col_names


def plot_autoregressive_loss(
    input_logs_dir,
    output_fig_path,
    run_name="",
    dataset_name="",
    log_to_mlflow=False,
    experiment_name: str = "",
    tracking_uri: str = "",
    run_id: str | None = None,
) -> None:
    """
    Plot autoregressive MSE over epochs and the correlation between validation loss
    and autoregressive MSE.

    Args:
        input_logs_dir: Path to the Lightning logs directory containing subdirectories
                        like 'version_0', 'version_1', etc. Each subdirectory should have a
                        metrics.csv file.
        output_fig_path: Path to save the output figure.
        run_name: Name of the run to display in the plot title.
        dataset_name: Name of the dataset to display in the plot title.
        log_to_mlflow: Whether to log the plot to MLflow as an artifact (default: False).
        experiment_name: Name of the MLflow experiment.
        tracking_uri: MLflow tracking URI.
        run_id: Run ID for logging to MLflow.
    """
    # Read and concatenate logs
    combined_df, _ = _read_and_concat_logs(input_logs_dir)
    # Prepare output directory
    output_dir = os.path.dirname(output_fig_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Only proceed if autoregressive MSE data exists
    if (
        "val_autoregressive_mse" not in combined_df.columns
        or combined_df["val_autoregressive_mse"].isna().all()
    ):
        logger.warning(
            "No autoregressive MSE data found in logs; skipping autoregressive plots."
        )
        return

    # 1) Plot autoregressive MSE vs epoch
    ar_points = combined_df.dropna(subset=["val_autoregressive_mse"]).copy()
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    fig2 = cast(Figure, fig2)
    ax2 = cast(Axes, ax2)
    ax2.plot(
        ar_points["epoch"],
        ar_points["val_autoregressive_mse"],
        color="#9467BD",
        marker="^",
        linewidth=2,
        markersize=10,
        markerfacecolor="#E6E6FA",
        label="Val Autoregressive MSE",
    )
    ax2.set_title(
        "Validation Autoregressive MSE Over Epochs", fontsize=24, pad=15
    )
    ax2.set_xlabel("Epoch", fontsize=20, labelpad=12)
    ax2.set_ylabel("Autoregressive MSE", fontsize=20, labelpad=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.tick_params(axis="both", labelsize=14)
    # Annotate run/dataset
    info_text = []
    if run_name:
        info_text.append(f"Run Name: {run_name}")
    if dataset_name:
        info_text.append(f"Dataset Name: {dataset_name}")
    if info_text:
        ax2.text(
            0.01,
            0.98,
            "\n".join(info_text),
            transform=ax2.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor="white",
                edgecolor="lightgray",
                boxstyle="square,pad=0.7",
            ),
        )
    ax2.legend(loc="upper right", frameon=True, fontsize=12)
    plt.tight_layout()
    try:
        # Save to the main output path
        fig2.savefig(output_fig_path)

        # Log to MLflow if enabled
        if log_to_mlflow:
            try:
                if experiment_name and tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_experiment(experiment_name)
                mlflow.log_artifact(output_fig_path, "plots", run_id=run_id)
            except Exception as e:
                logger.error(
                    f"Failed to log autoregressive plot to MLflow: {str(e)}"
                )
    except Exception as e:
        logger.error(f"Failed to save autoregressive plot: {str(e)}")
    finally:
        plt.close(fig2)

    # 2) Scatter plot: val_loss vs val_autoregressive_mse for correlation
    data = combined_df.dropna(
        subset=["val_loss", "val_autoregressive_mse"]
    ).copy()
    fig3, ax3 = plt.subplots(figsize=(16, 10))
    fig3 = cast(Figure, fig3)
    ax3 = cast(Axes, ax3)
    ax3.scatter(
        data["val_loss"],
        data["val_autoregressive_mse"],
        color="#9467BD",
        alpha=0.7,
        edgecolors="k",
        label="Data Points",
    )

    ax3.set_title("Validation Loss and Autoregressive MSE", fontsize=24, pad=15)
    ax3.set_xlabel("Validation Loss", fontsize=20, labelpad=12)
    ax3.set_ylabel("Autoregressive MSE", fontsize=20, labelpad=12)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.tick_params(axis="both", labelsize=14)
    if info_text:
        ax3.text(
            0.01,
            0.98,
            "\n".join(info_text),
            transform=ax3.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor="white",
                edgecolor="lightgray",
                boxstyle="square,pad=0.7",
            ),
        )
    ax3.legend(loc="upper right", frameon=True, fontsize=12)
    plt.tight_layout()
    corr_fig_path = os.path.join(
        output_dir, "val_loss_vs_autoregressive_mse.png"
    )
    try:
        fig3.savefig(corr_fig_path)
        # Log to MLflow if enabled
        if log_to_mlflow:
            try:
                if experiment_name and tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_experiment(experiment_name)
                mlflow.log_artifact(corr_fig_path, "plots", run_id=run_id)
            except Exception as e:
                logger.error(
                    f"Failed to log correlation plot to MLflow: {str(e)}"
                )
    except Exception as e:
        logger.error(f"Failed to save correlation plot: {str(e)}")
    finally:
        plt.close(fig3)
