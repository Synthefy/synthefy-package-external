import argparse
import json
import os
import re
import time
from collections import OrderedDict
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.scripts.generate_synthetic_dataset_with_baseline import (
    load_config,
)
from synthefy_pkg.utils.basic_utils import ENDC, OKBLUE, OKYELLOW
from synthefy_pkg.utils.scaling_utils import (
    load_timeseries_col_names,
    transform_using_scaler,
)

# Set flag to true to precompile this file
COMPILE = False

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


def plot_histograms(
    all_metrics: dict,
    plot_path: str = "histogram.png",
    figsize: tuple = (28, 12),
):
    """
    Plots histograms of model performance metrics across different data splits.

    Parameters:
    all_metrics (dict): A dictionary where keys are split names (e.g., 'train', 'val', 'test') and values are dictionaries
                        of metrics for different models. Each inner dictionary should have metric names as keys and
                        corresponding values as lists of metric values.
    plot_path (str): The file path where the histogram plot will be saved. Default is 'histogram.png'.

    Returns:
    None
    """

    # Convert the data into a DataFrame for easy manipulation
    dfs = {
        split: pd.DataFrame(metrics).T for split, metrics in all_metrics.items()
    }

    # Mapping of original column names to user-friendly names
    column_mapping = {
        "mse_all": "Mean MSE",
        "mae_all": "Mean MAE",
        "rmse_all": "Mean RMSE",
        "mse_median_all": "Median MSE",
        "mae_median_all": "Median MAE",
        "rmse_median_all": "Median RMSE",
    }

    # Rename columns for all DataFrames
    dfs_renamed = {
        split: df.rename(columns=column_mapping, inplace=False)
        for split, df in dfs.items()
    }

    # Initialize the matplotlib figure
    n_splits = len(dfs_renamed)
    fig, axes = plt.subplots(1, n_splits, figsize=figsize, sharey=True)

    # Handle single split case by wrapping axes in list
    if n_splits == 1:
        axes = [axes]

    for i, (ax, (split, df)) in enumerate(zip(axes, dfs_renamed.items())):
        # Transpose the DataFrame to plot metrics as grouped bars for each model
        df.T.plot(
            kind="bar", ax=ax, alpha=0.75, width=0.7, legend=False
        )  # Disable legend for individual plots

        # Set labels and title for the subplot
        ax.set_xlabel(
            f"{split.capitalize()}", fontsize=14
        )  # Add split info as x-label
        ax.set_title("", fontsize=0)  # Remove the subplot title
        ax.set_ylabel(
            "Value" if i == 0 else ""
        )  # Add y-label only for the first subplot

        # Adjust the x-axis tick labels
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=0, fontsize=14)

        # Add legend inside the first subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper left", fontsize=12)

    # Add a single title to the whole figure
    fig.suptitle(
        "Comparison of Model Performance Across Different Splits",
        fontsize=20,
        y=0.98,
    )

    # Adjust the layout to include room for x-labels and the title
    plt.tight_layout(
        rect=[0, 0, 1, 0.9]
    )  # Adjust rect to make room for the title

    # Save and display the plot
    plt.savefig(plot_path)


def plot_box_plot(
    all_metrics,
    save_dir: str,
    model_names: list,
    figsize: tuple = (18, 12),
    remove_outliers_from_plot: bool = False,
) -> None:
    """
    Plot the box plot for the metrics across multiple models.

    Args:
        all_metrics (dict): A dictionary containing the metrics for each split and model.
        save_dir (str): Directory to save the plot.
        model_names (list): List of model names to compare.
    """
    # Prepare data for box plot
    data = []
    for split, metrics in all_metrics.items():
        for model_name in model_names:
            mse_values = metrics[model_name]["mse"]

            # Append MSE values for each model and split
            for mse in mse_values:
                data.append(
                    {
                        "Split": split.capitalize(),
                        "Model": model_name,
                        "MSE": mse,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    if remove_outliers_from_plot:
        print(OKYELLOW + "Removing outliers from plot" + ENDC)
        # remove outliers for better visualization
        df = df[df["MSE"] < df["MSE"].quantile(0.9)]

    # Set the style
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    plt.figure(figsize=figsize)

    # Create boxplot
    sns.boxplot(x="Split", y="MSE", hue="Model", data=df)

    # Set the style
    sns.set_palette("Set2")

    # Add grid
    plt.grid(axis="y", linestyle="--")

    # Add legend outside the plot
    plt.legend(loc="upper left")

    # Set title and labels
    plt.title("MSE Comparison across Models")
    plt.xlabel("")
    plt.ylabel("MSE")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(save_dir, "mse_box_plot.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Box plot saved to {plot_path}")


def normalize_timeseries_for_comparison(
    original_timeseries_i: np.ndarray, model_forecasts_i: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize time series data for comparison using MinMaxScaler.
    Each channel is normalized independently based on the original time series range.

    Args:
        original_timeseries_i (np.ndarray): Original time series data of shape (num_channels, time_steps)
        model_forecasts_i (np.ndarray): Model forecast data of shape (num_channels, time_steps)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized original and forecast time series

    Raises:
        ValueError: If inputs have incorrect shapes or dimensions
    """
    # Validate input shapes
    if original_timeseries_i.ndim != 2 or model_forecasts_i.ndim != 2:
        raise ValueError(
            "Inputs must be 2D arrays with shape (num_channels, time_steps)"
        )

    if original_timeseries_i.shape != model_forecasts_i.shape:
        raise ValueError(
            f"Shape mismatch: original shape {original_timeseries_i.shape} != forecast shape {model_forecasts_i.shape}"
        )

    num_channels = original_timeseries_i.shape[0]
    normalized_original = np.zeros_like(original_timeseries_i)
    normalized_forecast = np.zeros_like(model_forecasts_i)

    for channel in range(num_channels):
        # Initialize and fit scaler on original data
        scaler = MinMaxScaler()

        # Reshape from 1D (time_steps,) to (time_steps, 1) as MinMaxScaler requires 2D input
        # MinMaxScaler expects 2D arrays where:
        # - Each row is a sample (in our case, each time step)
        # - Each column is a feature (in our case, just one feature - the channel value)
        original_channel = original_timeseries_i[channel].reshape(-1, 1)
        scaler.fit(original_channel)

        # Transform data and reshape back to 1D arrays
        # 1. Reshape to (time_steps, 1) for scaler.transform()
        # 2. Use ravel() to convert back to 1D array of shape (time_steps,)
        normalized_original[channel] = scaler.transform(
            original_channel
        ).ravel()
        normalized_forecast[channel] = scaler.transform(
            model_forecasts_i[channel].reshape(-1, 1)
        ).ravel()

    return normalized_original, normalized_forecast


def compare_timeseries_data(
    aligned_data_dict,
    model_names: list,
    save_dir: str,
    column_names: list,
    forecast_length: int,
    top_k: int = 20,
    plot_rows: int = 3,
    plot_cols: int = 2,
    figsize: tuple = (18, 12),
    remove_outliers_from_plot: bool = False,
    splits: Tuple[str, str] = ("test",),
) -> None:
    """
    Compare the time series data across multiple models.

    Args:
        aligned_data_dict (dict): Dictionary containing aligned time series data.
        save_dir (str): Directory to save the results.
        top_k (int): Number of top samples to visualize.
        model_names (list): List of model names to compare.
        plot_rows (int): Number of rows to plot.
        plot_cols (int): Number of columns to plot.
        figsize (tuple): Size of the figure.
        scaled (bool): Whether the data is scaled or not.
    """
    all_metrics = {
        split: {model: None for model in model_names} for split in splits
    }

    for split in splits:
        metrics_by_model = {
            model: {"mse": [], "mae": [], "rmse": [], "transformed_mse": []}
            for model in model_names
        }

        print(OKBLUE + f"Comparing the {split} forecast data" + ENDC)
        # Get the original timeseries data
        original_timeseries = aligned_data_dict[split]["original_timeseries"]

        # Calculate per-sample metrics for each model
        for model_name in model_names:
            model_forecasts = aligned_data_dict[split][
                f"{model_name}_synthetic_timeseries"
            ]

            for i in range(len(original_timeseries)):
                original_timeseries_i = original_timeseries[i]
                model_forecasts_i = model_forecasts[i]

                # Calculate metrics
                mse = np.mean((original_timeseries_i - model_forecasts_i) ** 2)
                mae = np.mean(np.abs(original_timeseries_i - model_forecasts_i))
                rmse = np.sqrt(mse)

                # Normalize data for transformed MSE calculation
                transformed_original, transformed_forecast = (
                    normalize_timeseries_for_comparison(
                        original_timeseries_i, model_forecasts_i
                    )
                )
                transformed_mse = np.mean(
                    (transformed_original - transformed_forecast) ** 2
                )

                metrics_by_model[model_name]["mse"].append(mse)
                metrics_by_model[model_name]["mae"].append(mae)
                metrics_by_model[model_name]["rmse"].append(rmse)
                metrics_by_model[model_name]["transformed_mse"].append(
                    transformed_mse
                )

            # Calculate mean metrics
            metrics_by_model[model_name]["mse_all"] = np.mean(
                metrics_by_model[model_name]["mse"]
            )
            metrics_by_model[model_name]["mae_all"] = np.mean(
                metrics_by_model[model_name]["mae"]
            )
            metrics_by_model[model_name]["rmse_all"] = np.mean(
                metrics_by_model[model_name]["rmse"]
            )
            metrics_by_model[model_name]["mse_median_all"] = np.median(
                metrics_by_model[model_name]["mse"]
            )
            metrics_by_model[model_name]["mae_median_all"] = np.median(
                metrics_by_model[model_name]["mae"]
            )

            print(
                OKBLUE
                + f"{split} {model_name} - MSE: {metrics_by_model[model_name]['mse_all']}, "
                + f"MAE: {metrics_by_model[model_name]['mae_all']}, "
                + f"RMSE: {metrics_by_model[model_name]['rmse_all']}, "
                + f"Median MSE: {metrics_by_model[model_name]['mse_median_all']}, "
                + f"Median MAE: {metrics_by_model[model_name]['mae_median_all']}"
                + ENDC
            )
        # Save all metrics for the split
        all_metrics[split] = metrics_by_model

        # Sort by MSE for the first model to select top_k samples
        sorted_indices = np.argsort(
            metrics_by_model[model_names[0]]["transformed_mse"]
        )

        # Get the top_k samples
        top_k_ground_truth = original_timeseries[sorted_indices[:top_k]]
        top_k_forecasts = {
            model_name: aligned_data_dict[split][
                f"{model_name}_synthetic_timeseries"
            ][sorted_indices[:top_k]]
            for model_name in model_names
        }

        # Create a directory for split-specific visualizations
        save_path = os.path.join(save_dir, f"{split}_timeseries_comparison")
        os.makedirs(save_path, exist_ok=True)

        # Plot the timeseries data of the top_k samples
        top_k = min(top_k, len(original_timeseries))
        for i in range(top_k):
            # To have a clear visualization, we plot the first plot_rows*plot_cols channels.
            # Note: If either of plot_rows or plot_cols is large, the subplots may overlap and become unclear.
            # Consider reducing these values or increasing the figure size (figsize parameter) if plots appear crowded.
            fig, axes = plt.subplots(
                plot_rows, plot_cols, figsize=figsize, sharex=True, sharey=True
            )
            axes = axes.flatten()

            colors = sns.color_palette("deep", len(model_names) + 1)
            line_styles = ["-", "--", "-.", ":"]

            for channel_i in range(
                min(plot_rows * plot_cols, len(top_k_ground_truth[i]))
            ):
                ax = axes[channel_i]
                ax.plot(
                    top_k_ground_truth[i][channel_i],
                    label="Ground Truth",
                    color="black",
                    linewidth=2,
                )

                for idx, model_name in enumerate(model_names):
                    ax.plot(
                        top_k_forecasts[model_name][i][channel_i],
                        label=model_name,
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=2,
                        color=colors[idx],
                    )
                ax.set_title(column_names[channel_i], weight="bold")
                ax.legend(
                    loc="upper left",
                    fontsize="small",
                    frameon=True,
                    facecolor="white",
                )

            # Add vertical line at forecast point
            for ax in axes:
                # Get the x-axis data points
                x_data = np.arange(len(top_k_ground_truth[i][0]))
                # Place vertical line at the forecast start point
                ax.axvline(
                    x=x_data[-forecast_length - 1],
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                    label="Forecast Start",
                )

            fig.suptitle(
                f"Sample {i + 1} of {top_k} Comparing Models for '{split}' Split",
                fontsize=16,
                weight="bold",
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(
                os.path.join(save_path, f"comparison_samples_{i}.png"),
                bbox_inches="tight",
            )
            plt.close()

        print(
            OKBLUE
            + f"Saved {split} timeseries comparison plots to {save_path}"
            + ENDC
        )

    # Convert all_metrics to JSON-serializable format
    all_metrics_serializable = {}
    for split, metrics in all_metrics.items():
        all_metrics_serializable[split] = {}
        for model, model_metrics in metrics.items():
            all_metrics_serializable[split][model] = {}
            for metric, value in model_metrics.items():
                if type(value) is np.float64 or type(value) is np.float32:
                    all_metrics_serializable[split][model][metric] = float(
                        value
                    )

    # Save metrics to a JSON file
    json_save_path = os.path.join(save_dir, "all_metrics.json")
    with open(json_save_path, "w") as json_file:
        json.dump(all_metrics_serializable, json_file, indent=4)

    print(OKBLUE + f"Saved all metrics to {json_save_path}" + ENDC)

    # Plot the histograms of all metrics
    plot_histograms(
        all_metrics_serializable,
        os.path.join(save_dir, "all_metrics_histogram.png"),
        figsize=figsize,
    )

    plot_box_plot(
        all_metrics,
        save_dir,
        model_names,
        figsize=figsize,
        remove_outliers_from_plot=remove_outliers_from_plot,
    )


def load_timeseries(
    saved_data_path,
    try_use_probabilistic_forecast: bool = True,
    splits: Tuple[str, str] = ("test",),
) -> Tuple[dict, bool]:
    """
    Determine if the data is saved as a .h5 or .pkl file and load the timeseries data.
    """
    # Check any existing splits for h5 vs pkl
    file_types = {split: None for split in splits}

    use_probabilistic_forecast: bool = False

    for split in splits:
        # Make sure the dataset folder exists
        data_save_path = os.path.join(saved_data_path, f"{split}_dataset")
        if try_use_probabilistic_forecast:
            probabilistic_path = os.path.join(
                saved_data_path, f"probabilistic_{split}_dataset"
            )
            if os.path.exists(probabilistic_path):
                data_save_path = probabilistic_path
                use_probabilistic_forecast = True
            else:
                data_save_path = os.path.join(
                    saved_data_path, f"{split}_dataset"
                )
                print(
                    OKYELLOW
                    + f"Probabilistic data {probabilistic_path} does not exist. Using non-probabilistic save directory instead: {data_save_path}"
                    + ENDC
                )
                use_probabilistic_forecast = False

        if os.path.exists(data_save_path):
            files = os.listdir(data_save_path)

            # Check if the split has h5 or pkl files
            if any(file.endswith(".h5") for file in files):
                file_types[split] = "h5"
            elif any(file.endswith(".pkl") for file in files):
                file_types[split] = "pkl"
            else:
                raise FileNotFoundError(
                    f"No .h5 or .pkl files found in {split}_dataset folder."
                )
        else:
            raise FileNotFoundError(
                f"Missing {split}_dataset folder in {saved_data_path}."
            )

    # All splits should have the same file type
    if all(file_type == "h5" for file_type in file_types.values()):
        return _load_timeseries_h5(
            saved_data_path, use_probabilistic_forecast, splits
        )
    elif all(file_type == "pkl" for file_type in file_types.values()):
        return _load_timeseries_pkl(
            saved_data_path, use_probabilistic_forecast, splits
        )
    else:
        raise ValueError(
            "Data is not consistent with either .h5 or .pkl format."
        )


def _load_timeseries_h5(
    data_save_path,
    use_probabilistic_forecast: bool = False,
    splits: Tuple[str, str] = ("test",),
) -> Tuple[dict, bool]:
    """
    Load the synthetic and original timeseries from h5 file (generated by Chronos script).
    """
    loaded_data_dict = {
        split: {
            "original_timeseries": [],
            "synthetic_timeseries": [],
        }
        for split in splits
    }

    for type_i in splits:
        if use_probabilistic_forecast:
            h5_file = os.path.join(
                data_save_path,
                f"probabilistic_{type_i}_dataset",
                f"probabilistic_{type_i}_combined_data.h5",
            )
        else:
            h5_file = os.path.join(
                data_save_path,
                f"{type_i}_dataset",
                f"{type_i}_combined_data.h5",
            )

        data_dict = {}
        with h5py.File(h5_file, "r") as f:
            for key in f.keys():
                data_dict[key] = f[key][:]

        loaded_data_dict[type_i]["synthetic_timeseries"] = data_dict[
            "synthetic_timeseries"
        ]
        loaded_data_dict[type_i]["original_timeseries"] = data_dict[
            "original_timeseries"
        ]

        print(OKBLUE + f"Loaded data from {h5_file}" + ENDC)

    return loaded_data_dict, use_probabilistic_forecast


def _load_timeseries_pkl(
    data_save_path,
    use_probabilistic_forecast: bool = False,
    splits: Tuple[str, str] = ("test",),
) -> Tuple[dict, bool]:
    """
    Load the true and synthesized timeseries data into the
    `all_window_info` dict
    """
    loaded_data_dict = {
        split: {
            "original_timeseries": [],
            "synthetic_timeseries": [],
        }
        for split in splits
    }
    for type_i in splits:
        if use_probabilistic_forecast:
            assert type_i == "test", (
                "Probabilistic forecast is only supported for test data"
            )
            type_save_path = os.path.join(
                data_save_path, f"probabilistic_{type_i}_dataset"
            )
        else:
            type_save_path = os.path.join(data_save_path, f"{type_i}_dataset")

        # Note: We load the .pkl files in ascending order to preserve ordering.
        # This is critical to make sure the data is aligned correctly.
        def natural_sort_key(filename):
            return [
                int(part) if part.isdigit() else part
                for part in re.split(r"(\d+)", filename)
            ]

        pkl_files = sorted(
            [
                f
                for f in os.listdir(type_save_path)
                if f.endswith(".pkl") and f.startswith("combined_data")
            ],
            key=natural_sort_key,  # Use natural sorting key
        )

        data_dict = {
            "original_timeseries": [],
            "synthetic_timeseries": [],
            "discrete_conditions": [],
            "continuous_conditions": [],
        }

        for pkl_file in pkl_files:
            pkl_file_path = os.path.join(type_save_path, pkl_file)
            print(OKBLUE + f"Loading data from {pkl_file_path}" + ENDC)
            unpickle_data = torch.load(pkl_file_path)
            if isinstance(data_dict["original_timeseries"], list):
                data_dict = unpickle_data
            else:
                # np concatenate the data
                data_dict["original_timeseries"] = np.concatenate(
                    (
                        data_dict["original_timeseries"],
                        unpickle_data["original_timeseries"],
                    ),
                    axis=0,
                )
                data_dict["synthetic_timeseries"] = np.concatenate(
                    (
                        data_dict["synthetic_timeseries"],
                        unpickle_data["synthetic_timeseries"],
                    ),
                    axis=0,
                )
        loaded_data_dict[type_i]["synthetic_timeseries"] = data_dict[
            "synthetic_timeseries"
        ]
        loaded_data_dict[type_i]["original_timeseries"] = data_dict[
            "original_timeseries"
        ]

    return loaded_data_dict, use_probabilistic_forecast


def align_timeseries_data(
    loaded_data_dicts, model_names, splits: Tuple[str, str] = ("test",)
):
    """
    Align the timeseries data across multiple models.

    Args:
        loaded_data_dicts (dict): Dictionary where keys are model names and values are loaded data dictionaries.
        model_names (list): List of model names to align.

    Returns:
        dict: Aligned data dictionary with original and synthetic time series for each model.
    """
    all_aligned_data_dict = {split: None for split in splits}

    for split in splits:
        original_timeseries = loaded_data_dicts[model_names[0]][split][
            "original_timeseries"
        ]  # Use the first model as the reference
        aligned_data_dict = {"original_timeseries": original_timeseries}

        for model_name in model_names:
            synthetic_timeseries = loaded_data_dicts[model_name][split][
                "synthetic_timeseries"
            ]
            aligned_model_synthetic = []

            for i, query_time_series in enumerate(original_timeseries):
                # Compute MSE between the reference time series and all time series in the current model
                mse_array = np.mean(
                    (query_time_series - synthetic_timeseries) ** 2, axis=(1, 2)
                )

                # Find the index of the minimum MSE
                best_index = np.argmin(mse_array)

                # Append the best matching synthetic time series
                aligned_model_synthetic.append(synthetic_timeseries[best_index])

            # Convert to numpy and store in the aligned data dict
            aligned_data_dict[f"{model_name}_synthetic_timeseries"] = np.array(
                aligned_model_synthetic
            )

        # Add the aligned data for this split
        all_aligned_data_dict[split] = aligned_data_dict

    return all_aligned_data_dict


def compare_models(
    configs: List[DictConfig],
    top_k: int = 20,
    plot_rows: int = 3,
    plot_cols: int = 2,
    figsize: tuple = (18, 12),
    splits: Tuple[str, str] = ("test",),
) -> None:
    configurations = [Configuration(config) for config in configs]
    remove_outliers_from_plot = any(
        config.remove_outliers_from_plot for config in configurations
    )
    print(
        OKYELLOW
        + f"Removing outliers from plot: {remove_outliers_from_plot}"
        + ENDC
    )
    loaded_data_dicts = {}
    model_names = []
    model_names_to_use_probabilistic_forecast = OrderedDict()

    for configuration in configurations:
        save_dir = configuration.get_save_dir(SYNTHEFY_DATASETS_BASE)
        model_name = configuration.denoiser_config.denoiser_name
        print(OKBLUE + f"Loading {model_name} data from {save_dir}" + ENDC)

        # prefer to use probabilistic forecast if available
        loaded_data_dict, use_probabilistic_forecast = load_timeseries(
            save_dir, try_use_probabilistic_forecast=True, splits=splits
        )

        print(
            OKBLUE
            + f"Loaded the timeseries data of size for {model_name} {len(loaded_data_dict['test']['original_timeseries'])}; {use_probabilistic_forecast=}"
            + ENDC
        )

        loaded_data_dicts[model_name] = loaded_data_dict
        model_names.append(model_name)
        model_names_to_use_probabilistic_forecast[model_name] = (
            use_probabilistic_forecast
        )

    start_time = time.time()

    aligned_data_dict = {split: None for split in splits}
    unscaled_aligned_data_dict = {split: None for split in splits}
    for split in splits:
        per_split_original_timeseries = loaded_data_dicts[model_names[0]][
            split
        ]["original_timeseries"].copy()  # loading the original timeseries data
        split_aligned_data_dict = {
            "original_timeseries": per_split_original_timeseries
        }
        per_split_unscaled_original_timeseries = transform_using_scaler(
            windows=per_split_original_timeseries.copy(),
            timeseries_or_continuous="timeseries",
            dataset_name=configurations[0].dataset_config.dataset_name,
            inverse_transform=True,
            transpose_timeseries=True,
        )

        # adding the original timeseries data to the split aligned data dict
        unscaled_split_aligned_data_dict = {
            "original_timeseries": per_split_unscaled_original_timeseries
        }
        for (
            model_name,
            use_probabilistic_forecast,
        ) in model_names_to_use_probabilistic_forecast.items():
            per_split_per_model_synthetic_timeseries = loaded_data_dicts[
                model_name
            ][split]["synthetic_timeseries"].copy()
            if use_probabilistic_forecast:
                print(
                    OKBLUE
                    + f"Using probabilistic forecast for {model_name}; downsampling across the probabilistic dimension"
                    + ENDC
                )
                assert (
                    len(per_split_per_model_synthetic_timeseries.shape) == 4
                ), "Probabilistic forecast should have 4 dimensions"
                per_split_per_model_synthetic_timeseries = (
                    per_split_per_model_synthetic_timeseries[:, 0, :, :]
                )
            per_split_per_model_unscaled_synthetic_timeseries = (
                transform_using_scaler(
                    windows=per_split_per_model_synthetic_timeseries.copy(),
                    timeseries_or_continuous="timeseries",
                    dataset_name=configurations[0].dataset_config.dataset_name,
                    inverse_transform=True,
                    transpose_timeseries=True,
                )
            )

            split_aligned_data_dict[f"{model_name}_synthetic_timeseries"] = (
                per_split_per_model_synthetic_timeseries  # adding the synthetic timeseries data to the split aligned data dict
            )
            unscaled_split_aligned_data_dict[
                f"{model_name}_synthetic_timeseries"
            ] = per_split_per_model_unscaled_synthetic_timeseries  # adding the unscaled synthetic time series data to the split aligned data dict
        aligned_data_dict[split] = (
            split_aligned_data_dict  # adding the split aligned data dict to the aligned data dict
        )
        unscaled_aligned_data_dict[split] = (
            unscaled_split_aligned_data_dict  # adding the split aligned data dict to the unscaled aligned data dict
        )

    end_time = time.time()
    print(
        OKBLUE
        + f"Time taken to align the timeseries data: {end_time - start_time} seconds"
        + ENDC
    )

    # more sanity checks
    model0_original_test_timeseries = loaded_data_dicts[model_names[0]]["test"][
        "original_timeseries"
    ]
    model1_original_test_timeseries = loaded_data_dicts[model_names[1]]["test"][
        "original_timeseries"
    ]
    original_timeseries_data = {}
    for model_name in model_names:
        original_timeseries_data[model_name] = loaded_data_dicts[model_name][
            "test"
        ]["original_timeseries"]

    model0_synthetic_test_timeseries = loaded_data_dicts[model_names[0]][
        "test"
    ]["synthetic_timeseries"]
    # TODO: in the future, compute the metrics (including MSE box plots) across all prob samples
    if model_names_to_use_probabilistic_forecast[model_names[0]]:
        model0_synthetic_test_timeseries = model0_synthetic_test_timeseries[
            :, 0, :, :
        ]

    main_configuration = configurations[0]
    column_names = load_timeseries_col_names(
        main_configuration.dataset_config.dataset_name,
        main_configuration.dataset_config.num_channels,
    )

    assert np.all(
        model0_original_test_timeseries == model1_original_test_timeseries
    ), "Original test timeseries data is not the same for all models"
    history_length = (
        configurations[0].dataset_config.time_series_length
        - configurations[0].dataset_config.forecast_length
    )
    assert np.all(
        model0_original_test_timeseries[:, :, :history_length]
        == model0_synthetic_test_timeseries[:, :, :history_length]
    ), "History should be the same for original and synthetic timeseries"
    print(OKBLUE + "Sanity checks passed" + ENDC)

    print(OKYELLOW + "Comparing the timeseries data" + ENDC)

    save_dir = os.path.join(
        main_configuration.get_save_dir(SYNTHEFY_DATASETS_BASE),
        "comparison_eval",
    )
    scaled_output_path = os.path.join(save_dir, "scaled")
    compare_timeseries_data(
        aligned_data_dict,
        model_names,
        save_dir=scaled_output_path,
        column_names=column_names,
        forecast_length=main_configuration.dataset_config.forecast_length,
        plot_rows=plot_rows,
        plot_cols=plot_cols,
        figsize=figsize,
        top_k=top_k,
        remove_outliers_from_plot=remove_outliers_from_plot,
        splits=splits,
    )
    print(OKYELLOW + "Comparing the unscaled timeseries data" + ENDC)
    unscaled_output_path = os.path.join(save_dir, "unscaled")
    compare_timeseries_data(
        unscaled_aligned_data_dict,
        model_names,
        save_dir=unscaled_output_path,
        column_names=column_names,
        forecast_length=main_configuration.dataset_config.forecast_length,
        plot_rows=plot_rows,
        plot_cols=plot_cols,
        figsize=figsize,
        top_k=top_k,
        remove_outliers_from_plot=remove_outliers_from_plot,
        splits=splits,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_paths", type=str, nargs="+", required=True)
    parser.add_argument("--top_k", type=int, default=20)

    parser.add_argument(
        "--plot_rows",
        type=int,
        default=3,
        help="Number of rows in the subplot grid",
    )
    parser.add_argument(
        "--plot_cols",
        type=int,
        default=2,
        help="Number of columns in the subplot grid",
    )
    parser.add_argument(
        "--height", type=int, default=18, help="Height of the output plots"
    )
    parser.add_argument(
        "--width", type=int, default=12, help="Width of the output plots"
    )

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--plot_rows",
        type=int,
        default=3,
        help="Number of rows in the subplot grid",
    )
    parser.add_argument(
        "--plot_cols",
        type=int,
        default=2,
        help="Number of columns in the subplot grid",
    )
    parser.add_argument(
        "--height", type=int, default=18, help="Height of the output plots"
    )
    parser.add_argument(
        "--width", type=int, default=12, help="Width of the output plots"
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=("test",),
        help="Splits to compare",
    )

    args = parser.parse_args()
    configs = [load_config(config_path) for config_path in args.config_paths]
    compare_models(
        configs,
        args.top_k,
        args.plot_rows,
        args.plot_cols,
        (args.height, args.width),
        args.splits,
    )

"""
To use this script, use `examples/compare_forecasting_models.py`
python examples/compare_forecasting_models.py \
    --config_path examples/configs/config_air_quality_forecasting_sfm2.yaml \
    --others chronos prophet STL \
"""
