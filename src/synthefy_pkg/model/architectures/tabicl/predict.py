"""
General prediction and evaluation utilities for TabICL models.

This script loads all datasets from talent benchmark, combines numerical and categorical features, and benchmarks TabICLClassifier on classification tasks.


Example usage: (predict)

export EXPERIMENT=/workspace/raghav/data/training_logs/synthetic_tabular/icl_single_tabular/match && \
uv run src/synthefy_pkg/model/architectures/tabicl/predict.py predict \
--checkpoint_path $EXPERIMENT/checkpoints/checkpoint_step_282000_val_loss_0.4759.ckpt \
--base_path /workspace/data/synthetic_data/icl_match_synthetic_series_csv \
--results_dir_path $EXPERIMENT/results/match \
--max_eval 20 \
--plot

Single line:
export EXPERIMENT=/workspace/raghav/data/training_logs/synthetic_tabular/icl_single_tabular/match && uv run src/synthefy_pkg/model/architectures/tabicl/predict.py predict --checkpoint_path $EXPERIMENT/checkpoints/checkpoint_step_282000_val_loss_0.4759.ckpt --base_path /workspace/data/synthetic_data/icl_match_synthetic_series_csv --results_dir_path $EXPERIMENT/results/match --plot

Example usage: (compile)
uv run src/synthefy_pkg/model/architectures/tabicl/predict.py compile --results_dir /workspace/raghav/data/training_logs/synthetic_tabular/curriculum_series/tiny_lag

Old:
# python src/synthefy_pkg/model/architectures/tabicl/predict.py predict --results_dir_path /mnt/workspace1/data/eval_results/icl_synthetic_series_csv/icl_tiny/results --checkpoint_path /workspace/data/synthefy_data/training_logs/synthetic_tabular6/tabicl_series_regression_check_training/synthefy_foundation_model_v3_forecasting_prior_config_path-src_synthefy_pkg_prior_config_synthetic_configs_config_small_tabular.yaml_is_regression-True_is_tabular-True_trial1/checkpoints/checkpoint_step_10000_val_loss_-0.4320.ckpt --base_path /workspace/data/synthetic_data/icl_synthetic_series_csv --plot --max_eval 100 --compare_to_univariate
# python src/synthefy_pkg/model/architectures/tabicl/predict.py predict --results_dir_path /mnt/workspace1/data/eval_results/weather_mpi_beutenberg/icl_tiny/results --checkpoint_path /workspace/data/synthefy_data/training_logs/synthetic_tabular6/tabicl_series_regression_check_training/synthefy_foundation_model_v3_forecasting_prior_config_path-src_synthefy_pkg_prior_config_synthetic_configs_config_small_tabular.yaml_is_regression-True_is_tabular-True_trial1/checkpoints/checkpoint_step_10000_val_loss_-0.4320.ckpt --base_path /mnt/workspace1/data/sampled_datasets/weather_mpi_beutenberg --plot --max_eval 100 --compare_to_univariate
# python src/synthefy_pkg/model/architectures/tabicl/predict.py predict --results_dir_path /mnt/workspace1/data/eval_results/traffic_PeMS/icl_tiny/results --checkpoint_path /workspace/data/synthefy_data/training_logs/synthetic_tabular6/tabicl_series_regression_check_training/synthefy_foundation_model_v3_forecasting_prior_config_path-src_synthefy_pkg_prior_config_synthetic_configs_config_small_tabular.yaml_is_regression-True_is_tabular-True_trial1/checkpoints/checkpoint_step_10000_val_loss_-0.4320.ckpt --base_path /mnt/workspace1/data/sampled_datasets/traffic_PeMS --plot --max_eval 100 --compare_to_univariate
# python src/synthefy_pkg/model/architectures/tabicl/predict.py predict --results_dir_path /mnt/workspace1/data/eval_results/solar_Alabama/icl_tiny/results --checkpoint_path /workspace/data/synthefy_data/training_logs/synthetic_tabular6/tabicl_series_regression_check_training/synthefy_foundation_model_v3_forecasting_prior_config_path-src_synthefy_pkg_prior_config_synthetic_configs_config_small_tabular.yaml_is_regression-True_is_tabular-True_trial1/checkpoints/checkpoint_step_10000_val_loss_-0.4320.ckpt --base_path //mnt/workspace1/data/sampled_datasets/solar_Alabama --plot --max_eval 100 --compare_to_univariate

"""

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.data.load_benchmark_datasets import (
    check_json,
    load_csv_dataset,
    load_talent_dataset,
)
from synthefy_pkg.model.architectures.tabicl.icl_inference.classifier import (
    TabICLClassifier,
)
from synthefy_pkg.model.architectures.tabicl.icl_inference.regressor import (
    TabICLGridRegressor,
    TabICLRegressor,
)
from synthefy_pkg.model.architectures.tabicl_wrapper import (
    retrieve_tabicl_args,
)


def reset_model_state(model):
    """Reset model state to clear any cached data or internal state"""
    try:
        # Try to call reset method if it exists
        if hasattr(model, "reset"):
            model.reset()
        # Clear any cached data
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        # Reset any internal state
        if hasattr(model, "_reset_state"):
            model._reset_state()
    except Exception as e:
        logger.warning(f"Could not reset model state: {e}")


def fit_y_scaler(y_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on y_train data.

    Args:
        y_train: Training target values as numpy array

    Returns:
        StandardScaler: Fitted scaler that can be used to transform y values
    """

    # Simple imputer for missing values
    if np.isnan(y_train).any():
        # Use mean imputation for missing values
        y_train_mean = np.nanmean(y_train)
        y_train = np.where(np.isnan(y_train), y_train_mean, y_train)
        logger.info(
            f"Imputed {np.isnan(y_train).sum()} missing values in y_train with mean: {y_train_mean:.4f}"
        )

    scaler = StandardScaler()
    # Reshape y_train to 2D if it's 1D (required by StandardScaler)
    y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    scaler.fit(y_train_2d)
    return scaler


def calculate_regression_metrics(
    predictions: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[float, float]:
    """
    Calculate MSE and MAE for regression predictions with optional scaling.

    Args:
        predictions: Model predictions as numpy array
        y_test: Ground truth values as numpy array
        scaler: Optional StandardScaler to scale predictions and ground truth

    Returns:
        Tuple[float, float]: (mse, mae) values
    """
    if scaler is not None:
        # Simple imputer for missing values
        if np.isnan(y_test).any():
            # Use mean imputation for missing values
            y_test_mean = np.nanmean(y_test)
            y_test = np.where(np.isnan(y_test), y_test_mean, y_test)
            logger.info(
                f"Imputed {np.isnan(y_test).sum()} missing values in y_test with mean: {y_test_mean:.4f}"
            )

        predictions_scaled = np.asarray(
            scaler.transform(predictions.reshape(-1, 1))
        ).ravel()
        y_test_scaled = np.asarray(
            scaler.transform(y_test.reshape(-1, 1))
        ).ravel()
        mse = np.mean((predictions_scaled - y_test_scaled) ** 2)
        mae = np.mean(np.abs(predictions_scaled - y_test_scaled))
    else:
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))

    return float(mse), float(mae)


def convert_checkpoint(checkpoint_path, config):
    # Helper to convert checkpoint to TabICL classifier compatible format to allow ensembling
    # Expected fields:
    # assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
    # assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."
    # self.model_path_ = model_path_
    # self.model_ = TabICL(**checkpoint["config"])
    # self.model_.load_state_dict(checkpoint["state_dict"])

    final_path = checkpoint_path.replace(".ckpt", ".compat.ckpt")
    checkpoint = torch.load(checkpoint_path)

    # Load config from checkpoint path

    logger.info(f"Converting checkpoint {checkpoint_path} to compatible format")

    model_config = retrieve_tabicl_args(
        config, config.foundation_model_config.model_name
    )

    # Transform state dict keys by removing the prefix
    state_dict = checkpoint["state_dict"]
    new_state_dict = {
        k.replace("decoder_model.model.", ""): v for k, v in state_dict.items()
    }
    # Filter out keys containing 'distribution'
    new_state_dict = {
        k: v for k, v in new_state_dict.items() if "distribution" not in k
    }

    checkpoint["state_dict"] = new_state_dict

    checkpoint["config"] = model_config

    torch.save(checkpoint, final_path)
    logger.info(f"Saved checkpoint to {final_path}")
    return final_path


def create_tabicl_model(config, checkpoint_path):
    """Create and return appropriate TabICL model based on configuration."""
    if config.tabicl_config.is_regression:
        if config.tabicl_config.use_full_reg:
            return TabICLGridRegressor(
                config=config,
                model_path=checkpoint_path,
            )
        else:
            return TabICLRegressor(
                config=config,
                model_path=checkpoint_path,
            )
    else:
        converted_checkpoint_path = convert_checkpoint(checkpoint_path, config)
        return TabICLClassifier(
            model_path=converted_checkpoint_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )


def evaluate_classification_future_leaked(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier: TabICLClassifier,
    num_timestamp_features: int = 0,
) -> Dict[str, Any]:
    """Evaluate classification task with future leakage"""
    results = {}

    # Train and evaluate multivariate classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy_preds = predictions == y_test
    if isinstance(accuracy_preds, torch.Tensor):
        accuracy = accuracy_preds.mean()
    else:
        accuracy = np.mean(accuracy_preds)

    results["accuracy_fl"] = float(accuracy)
    results["predictions_fl"] = predictions
    return results


def evaluate_classification_univariate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier: TabICLClassifier,
    num_timestamp_features: int = 0,
) -> Dict[str, Any]:
    """Evaluate univariate classification task using first num_timestamp_features columns"""
    results = {}

    X_train_univariate = X_train.iloc[:, :num_timestamp_features]
    X_test_univariate = X_test.iloc[:, :num_timestamp_features]

    classifier.fit(X_train_univariate, y_train)
    predictions_univariate = classifier.predict(X_test_univariate)
    accuracy_preds_uni = predictions_univariate == y_test
    if isinstance(accuracy_preds_uni, torch.Tensor):
        accuracy_univariate = accuracy_preds_uni.mean()
    else:
        accuracy_univariate = np.mean(accuracy_preds_uni)

    results["accuracy_uni"] = float(accuracy_univariate)
    results["predictions_uni"] = predictions_univariate
    return results


def evaluate_regression_future_leaked(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLRegressor,
    num_timestamp_features: int = 0,
    is_auto_regressive: bool = False,
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """
    Evaluate standard regression task with future leakage.
    No concept of auto-regressive prediction here.
    """
    results = {}

    # Train and evaluate multivariate regressor
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test, y_test)

    if not is_auto_regressive:
        # Calculate MSE and MAE using the utility function
        mse, mae = calculate_regression_metrics(predictions, y_test, scaler)
        results["mse_fl"] = mse
        results["mae_fl"] = mae

    results["predictions_fl"] = predictions

    return results


def evaluate_regression_univariate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLRegressor,
    num_timestamp_features: int = 0,
    is_auto_regressive: bool = False,
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """Evaluate univariate regression task using first num_timestamp_features columns"""
    results = {}

    X_train_univariate = X_train.iloc[:, :num_timestamp_features]
    X_test_univariate = X_test.iloc[:, :num_timestamp_features]

    regressor.fit(X_train_univariate, y_train)
    predictions_univariate = regressor.predict(X_test_univariate, y_test)

    if not is_auto_regressive:
        # Calculate MSE and MAE using the utility function
        mse_univariate, mae_univariate = calculate_regression_metrics(
            predictions_univariate, y_test, scaler
        )
        results["mse_uni"] = mse_univariate
        results["mae_uni"] = mae_univariate

    results["predictions_uni"] = predictions_univariate

    return results


def evaluate_regression_multivariate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLRegressor,
    num_timestamp_features: int,
    is_auto_regressive: bool = False,
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """Evaluate true multivariate regression without future leakage.

    We don't want to pass any features of the test set to the model except timestamp features at prediction time.
    Every feature must be predicted on the basis of just the fit set and the timestamp features.
    This function evaluates the multivariate regression model by training on progressively more features
    and predicting on the test set with each additional feature. It returns the predictions and ground truth
    for the final prediction step.

    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        y_train: Training target values
        y_test: Test target values
        regressor: The TabICL regressor model
        num_timestamp_features: Number of timestamp features to preserve
    """

    results = {}

    # Preserve original column names
    X_train_cols = X_train.columns.tolist()

    # Start with only timestamp features for test set
    timestamp_cols = X_train_cols[:num_timestamp_features]
    X_test_i_df = X_test[timestamp_cols].copy()

    # Loop through non-timestamp features to predict them sequentially
    feature_cols_to_predict = X_train_cols[num_timestamp_features:]

    for i, col_to_predict in enumerate(feature_cols_to_predict):
        # Features to use for this prediction step
        current_feature_cols = X_train_cols[: num_timestamp_features + i]

        # Prepare training data for this step
        X_train_i = X_train[current_feature_cols]
        y_train_i = X_train[col_to_predict].values

        regressor.fit(X_train_i, y_train_i)

        # Predict the next feature for the test set
        # The y_test equivalent for this intermediate prediction is not available, so we pass a dummy array.
        y_test_pred_i = regressor.predict(
            X_test_i_df, np.zeros_like(y_train_i[: len(X_test_i_df)])
        )

        # Add the predicted feature to the test features for the next iteration
        X_test_i_df[col_to_predict] = y_test_pred_i

    # Final prediction of the actual target y_test
    regressor.fit(X_train, y_train)

    # Use the generated test features to predict y_test
    predictions = regressor.predict(X_test_i_df, y_test)

    assert predictions is not None, "Predictions should not be None"

    if not is_auto_regressive:
        # Calculate MSE and MAE using the utility function
        mse, mae = calculate_regression_metrics(predictions, y_test, scaler)
        results["mse_multi"] = mse
        results["mae_multi"] = mae

    results["predictions_multi"] = predictions
    return results


def evaluate_regression_autoregressive(
    evaluation_function,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLRegressor,
    scaler: Optional[StandardScaler] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Autoregressive wrapper for regression evaluation functions.
    Predicts one row at a time, using previous predictions as part of training data.

    Args:
        evaluation_function: The evaluation function to wrap (e.g., evaluate_regression_univariate)
        X_train, X_test, y_train, y_test: Data arrays
        regressor: The TabICL regressor
        scaler: Optional StandardScaler to scale predictions before calculating metrics
        **kwargs: Additional arguments to pass to the evaluation function

    Returns:
        Dictionary with autoregressive results (mse_auto, mae_auto, predictions_auto)
    """
    results = {}

    # Initialize iterative training data
    X_train_iter = X_train.copy()
    y_train_iter = y_train.copy()
    predictions_list = []

    for i in range(len(X_test)):
        # Get single test row
        X_test_row = X_test.iloc[i : i + 1]  # Keep as DataFrame with single row
        y_test_row = y_test[i : i + 1]  # Single target value

        # Call evaluation function with current training data and single test row
        step_results = evaluation_function(
            X_train_iter,
            X_test_row,
            y_train_iter,
            y_test_row,
            regressor,
            is_auto_regressive=True,
            scaler=scaler,
            **kwargs,
        )

        # Extract prediction from the evaluation function result
        # Look for predictions in the results (could be predictions_uni, predictions_multi, etc.)
        prediction_key = None
        for key in step_results.keys():
            if key.startswith("predictions_"):
                prediction_key = key
                break

        if prediction_key is None:
            raise ValueError(
                "No predictions found in evaluation function results"
            )

        pred_value = step_results[prediction_key][
            0
        ]  # Extract scalar prediction

        predictions_list.append(pred_value)

        # Append prediction to training targets and test row to training features
        y_train_iter = np.append(y_train_iter, pred_value)
        X_train_iter = pd.concat([X_train_iter, X_test_row], ignore_index=True)

    # Extract suffix from prediction key (e.g., "uni" from "predictions_uni")
    suffix = prediction_key.replace("predictions_", "")

    # Concatenate all predictions and ground truth
    predictions_autoregressive = np.array(predictions_list)

    # Calculate MSE and MAE using the utility function
    mse_autoregressive, mae_autoregressive = calculate_regression_metrics(
        predictions_autoregressive, y_test, scaler
    )

    results[f"mse_{suffix}_auto"] = mse_autoregressive
    results[f"mae_{suffix}_auto"] = mae_autoregressive
    results[f"predictions_{suffix}_auto"] = predictions_autoregressive

    return results


def save_predictions_dict(
    predictions_dict: Dict[str, np.ndarray],
    dataset_name: str,
    results_dir_path: str,
):
    """Save predictions from a dictionary with human-readable keys"""
    os.makedirs(results_dir_path, exist_ok=True)

    for key, predictions in predictions_dict.items():
        # Convert human readable key to filename format (lowercase, replace spaces with _)
        filename_key = key.lower().replace(" ", "_")
        pred_path = os.path.join(
            results_dir_path, f"{dataset_name}_{filename_key}_unscaled.npy"
        )
        np.save(pred_path, predictions)
        logger.trace(f"Saved {key} predictions to {pred_path}")


def generate_comparison_plot_dict(
    predictions_dict: Dict[str, np.ndarray],
    dataset_name: str,
    results_dir_path: str,
    timestamp_train: Optional[np.ndarray] = None,
    timestamp_test: Optional[np.ndarray] = None,
    mae_values: Optional[Dict[str, float]] = None,
    suffix: Optional[str] = "",
):
    """Generate comparison plot for all available prediction arrays using timestamps as X-axis"""
    plt.figure(figsize=(12, 8))

    # Set larger font sizes
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
        }
    )

    # Plot all predictions with different colors and labels
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
    ]

    # Use timestamps for X-axis if available, otherwise use indices
    use_timestamps = timestamp_train is not None and timestamp_test is not None

    # Convert timestamps to datetime if needed
    ts_train = None
    ts_test = None
    if use_timestamps:
        if timestamp_train is not None and len(timestamp_train) > 0:
            if not isinstance(timestamp_train[0], pd.Timestamp):
                ts_train = pd.to_datetime(timestamp_train)
            else:
                ts_train = timestamp_train
        if timestamp_test is not None and len(timestamp_test) > 0:
            if not isinstance(timestamp_test[0], pd.Timestamp):
                ts_test = pd.to_datetime(timestamp_test)
            else:
                ts_test = timestamp_test

    # First, plot ground truth history if it exists (to appear earlier on x-axis)
    if "Ground Truth History" in predictions_dict:
        history = predictions_dict["Ground Truth History"]
        if use_timestamps and ts_train is not None:
            plt.plot(
                ts_train,
                history,
                label="Ground Truth History",
                alpha=0.7,
                color="black",
                linestyle="--",
            )
        else:
            plt.plot(
                range(len(history)),
                history,
                label="Ground Truth History",
                alpha=0.7,
                color="black",
                linestyle="--",
            )

    # Then plot all other predictions
    for i, (key, predictions) in enumerate(predictions_dict.items()):
        if predictions.ndim == 2:
            # Assume last column is the target
            predictions = predictions[:, -1]

        # Check if predictions is a 1D array, if not raise ValueError
        if predictions.ndim != 1:
            raise ValueError(f"Predictions for {key} must be 1D array, got shape {predictions.shape}")

        if key == "Ground Truth History":
            continue  # Skip as we already plotted it

        if "ground" in key.lower() or "gt" in key.lower():
            color = "black"
        else:
            color = colors[i % len(colors)]

        # Create label with MAE value if available
        label = key
        if mae_values is not None and key in mae_values:
            mae_val = mae_values[key]
            label = f"{key} (MAE: {mae_val:.4f})"

        # For predictions other than history, use test timestamps if available
        if (
            use_timestamps
            and key != "Ground Truth History"
            and ts_test is not None
        ):
            plt.plot(ts_test, predictions, label=label, alpha=0.7, color=color)
        else:
            # For predictions other than history, offset the x-axis to start after history
            if (
                "Ground Truth History" in predictions_dict
                and not use_timestamps
            ):
                x_offset = len(predictions_dict["Ground Truth History"])
                plt.plot(
                    range(x_offset, x_offset + len(predictions)),
                    predictions,
                    label=label,
                    alpha=0.7,
                    color=color,
                )
            else:
                plt.plot(predictions, label=label, alpha=0.7, color=color)

    plt.title(f"Predictions Comparison for {dataset_name}", fontsize=20)
    if use_timestamps:
        plt.xlabel("Timestamp", fontsize=18)
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
    else:
        plt.xlabel("Running index", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    # Remove tight_layout to avoid warning
    # plt.tight_layout()

    plot_path = os.path.join(
        results_dir_path, f"{dataset_name}_comparison_plot{suffix}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.trace(f"Saved comparison plot to {plot_path}")


def load_predictions_dict(
    dataset_name: str,
    results_dir_path: str,
    expected_keys: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load saved prediction arrays from disk and create a predictions dict
    that can be passed to generate_comparison_plot_dict.

    Args:
        dataset_name: Name of the dataset (e.g., "dataset_001")
        results_dir_path: Path to the results directory where .npy files are saved
        expected_keys: Optional list of expected keys to load. If None, loads all available.
                      Common keys: ["Future Leaked", "Univariate", "True Multivariate",
                                   "Ground Truth", "Ground Truth History"]

    Returns:
        Dictionary mapping human-readable keys to numpy arrays
    """
    predictions_dict = {}

    # Define the mapping from human-readable keys to filename patterns
    key_to_filename = {
        "Future Leaked": f"{dataset_name}_future_leaked_unscaled.npy",
        "Univariate": f"{dataset_name}_univariate_unscaled.npy",
        "True Multivariate": f"{dataset_name}_true_multivariate_unscaled.npy",
        "Univariate Autoregressive": f"{dataset_name}_univariate_autoregressive_unscaled.npy",
        "True Multivariate Autoregressive": f"{dataset_name}_true_multivariate_autoregressive_unscaled.npy",
        "Ground Truth": f"{dataset_name}_ground_truth_unscaled.npy",
        "Ground Truth History": f"{dataset_name}_ground_truth_history_unscaled.npy"
    }

    # If expected_keys is provided, only load those keys
    keys_to_load = expected_keys if expected_keys is not None else key_to_filename.keys()

    for key in keys_to_load:
        if key not in key_to_filename:
            logger.error(f"Unknown key: {key}")
            continue

        file_path = os.path.join(results_dir_path, key_to_filename[key])
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            continue

        try:
            predictions_dict[key] = np.load(file_path)
        except Exception as e:
            logger.exception(f"Error loading {key} from {file_path}: {e}")


    return predictions_dict

def load_mae_values(dataset_name: str, results_dir_path: str) -> Optional[Dict[str, float]]:
    """
    Load MAE values from the results.json file for the given dataset.

    Args:
        dataset_name: Name of the dataset
        results_dir_path: Path to the results directory

    Returns:
        Dictionary mapping prediction keys to MAE values, or None if not found
    """
    results_file = os.path.join(results_dir_path, "results.json")
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return None

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        if dataset_name not in results:
            logger.error(f"Dataset {dataset_name} not found in results.json")
            return None

        dataset_results = results[dataset_name]

        # Create mapping from prediction keys to MAE values
        mae_mapping = {}
        key_mapping = {
            "Future Leaked": "mae_fl",
            "Univariate": "mae_uni",
            "True Multivariate": "mae_multi",
            "Univariate Autoregressive": "mae_uni_auto",
            "True Multivariate Autoregressive": "mae_multi_auto"
        }

        for pred_key, mae_key in key_mapping.items():
            if mae_key in dataset_results:
                try:
                    mae_mapping[pred_key] = float(dataset_results[mae_key])
                except (ValueError, TypeError):
                    logger.error(f"Could not convert {mae_key} to float: {dataset_results[mae_key]}")

        return mae_mapping if mae_mapping else None

    except Exception as e:
        logger.exception(f"Error loading MAE values: {e}")
        return None

def create_comparison_plot_from_saved_data(
    dataset_name: str,
    results_dir_path: str,
    timestamp_train: Optional[np.ndarray] = None,
    timestamp_test: Optional[np.ndarray] = None,
    expected_keys: Optional[List[str]] = None,
    suffix: Optional[str] = "",
):
    """
    Load saved prediction arrays and create a comparison plot.

    Args:
        dataset_name: Name of the dataset
        results_dir_path: Path to the results directory
        timestamp_train: Optional training timestamps
        timestamp_test: Optional test timestamps
        expected_keys: Optional list of keys to load (if None, loads all available)
    """
    # Load predictions
    predictions_dict = load_predictions_dict(dataset_name, results_dir_path, expected_keys)

    if not predictions_dict:
        logger.error("No predictions found to plot!")
        return

    # Load MAE values
    mae_values = load_mae_values(dataset_name, results_dir_path)

    # Generate the plot
    generate_comparison_plot_dict(
        predictions_dict,
        dataset_name,
        results_dir_path,
        timestamp_train,
        timestamp_test,
        mae_values,
        suffix=suffix,
    )

    logger.trace(f"Comparison plot generated for {dataset_name}")


def compile_all_results_to_markdown(results_dir):
    """
    Find all results.json files and compile them into markdown tables for MSE and MAE.
    Aggregates results at the folder level (each folder containing results.json is one row).

    Args:
        results_dir (str): Path to the results directory to search

    Returns:
        str: Markdown formatted tables
    """
    if not os.path.exists(results_dir):
        return f"Error: Results directory '{results_dir}' does not exist."

    # Find all results.json files
    results_files = find_results_json_files(results_dir)

    if not results_files:
        return f"No results.json files found in '{results_dir}'"

    # Collect all results from each folder
    folder_results = {}
    folder_names = []

    for results_file in results_files:
        # Get the folder name (parent directory of results.json)
        folder_name = os.path.basename(os.path.dirname(results_file))
        folder_names.append(folder_name)

        try:
            with open(results_file, "r") as f:
                results_data = json.load(f)
            folder_results[folder_name] = results_data
        except Exception as e:
            logger.error(f"Error processing {results_file}: {e}")

    if not folder_results:
        return "No valid results found."

    # Get all unique model types (FL, Uni, Multi, etc.)
    all_model_types = set()
    for folder_data in folder_results.values():
        for dataset_data in folder_data.values():
            for metric_name in dataset_data.keys():
                if metric_name.startswith("mse_") or metric_name.startswith("mae_"):
                    model_type = metric_name[4:]  # Remove prefix
                    all_model_types.add(model_type)
    all_model_types = sorted(list(all_model_types))

    # Create DataFrames for MSE and MAE
    mse_data = []
    mae_data = []

    for folder_name in folder_names:
        mse_row = {"Folder": folder_name}
        mae_row = {"Folder": folder_name}

        if folder_name in folder_results:
            folder_data = folder_results[folder_name]

            # Aggregate metrics across all datasets in this folder
            for model_type in all_model_types:
                mse_key = f"mse_{model_type}"
                mae_key = f"mae_{model_type}"

                # Collect all valid MSE values for this model type
                mse_values = []
                mae_values = []

                for dataset_data in folder_data.values():
                    if mse_key in dataset_data:
                        try:
                            mse_val = float(dataset_data[mse_key])
                            if mse_val != -1.0:  # Skip invalid values
                                mse_values.append(mse_val)
                        except (ValueError, TypeError):
                            pass

                    if mae_key in dataset_data:
                        try:
                            mae_val = float(dataset_data[mae_key])
                            if mae_val != -1.0:  # Skip invalid values
                                mae_values.append(mae_val)
                        except (ValueError, TypeError):
                            pass

                # Calculate average if we have valid values
                if mse_values:
                    mse_row[model_type] = np.mean(mse_values)
                else:
                    mse_row[model_type] = -1.0

                if mae_values:
                    mae_row[model_type] = np.mean(mae_values)
                else:
                    mae_row[model_type] = -1.0
        else:
            # Folder not found (shouldn't happen, but just in case)
            for model_type in all_model_types:
                mse_row[model_type] = -1.0
                mae_row[model_type] = -1.0

        mse_data.append(mse_row)
        mae_data.append(mae_row)

    # Create DataFrames
    mse_df = pd.DataFrame(mse_data)
    mae_df = pd.DataFrame(mae_data)

    # Drop columns where all values are NaN (-1.0 in this case)
    mse_df_cleaned = mse_df.dropna(axis=1, how='all')
    mae_df_cleaned = mae_df.dropna(axis=1, how='all')

    # Also drop columns where all non-Folder values are -1.0 (invalid values)
    for df in [mse_df_cleaned, mae_df_cleaned]:
        for col in df.columns:
            if col != 'Folder':  # Skip the Folder column
                if df[col].isin([-1.0]).all():
                    df.drop(columns=[col], inplace=True)

    # Generate markdown tables
    markdown_output = []

    # MSE Table
    markdown_output.append("## Mean Squared Error (MSE) - Aggregated by Folder")
    markdown_output.append("")

    # Use pandas to_markdown directly
    mse_md = mse_df_cleaned.to_markdown(index=False, floatfmt=".4f")
    markdown_output.append(mse_md)

    markdown_output.append("")

    # MAE Table
    markdown_output.append("## Mean Absolute Error (MAE) - Aggregated by Folder")
    markdown_output.append("")

    # Use pandas to_markdown directly
    mae_md = mae_df_cleaned.to_markdown(index=False, floatfmt=".4f")
    markdown_output.append(mae_md)

    return "\n".join(markdown_output)


def print_results(folder_name, averaged_results):
    """Print the averaged results grouped by MSE and MAE for a specific folder."""
    print(f"\n{'=' * 60}")
    print(f"Results for: {folder_name}")
    print(f"{'=' * 60}")

    print("\nMean Squared Error (MSE):")
    print("-" * 30)
    for model_type, avg_value in averaged_results["MSE"].items():
        print(f"  {model_type:8}: {avg_value:.4f}")

    print("\nMean Absolute Error (MAE):")
    print("-" * 30)
    for model_type, avg_value in averaged_results["MAE"].items():
        print(f"  {model_type:8}: {avg_value:.4f}")


def find_results_json_files(results_dir):
    """
    Recursively find all results.json files in the given directory.

    Args:
        results_dir (str): Path to the results directory to search

    Returns:
        list: List of paths to results.json files
    """
    results_files = []

    for root, dirs, files in os.walk(results_dir):
        if "results.json" in files:
            results_files.append(os.path.join(root, "results.json"))

    return sorted(results_files)


def find_and_compile_all_results(results_dir, output_file=None):
    """
    Find all folders containing results.json files and compile results into markdown tables.

    Args:
        results_dir (str): Path to the results directory to search
        output_file (str, optional): Path to save the markdown output. If None, saves to agg.md in results_dir.
    """
    if not os.path.exists(results_dir):
        error_msg = f"Error: Results directory '{results_dir}' does not exist."
        if output_file:
            with open(output_file, 'w') as f:
                f.write(error_msg)
        else:
            logger.error(error_msg)
        return

    # Find all results.json files
    results_files = find_results_json_files(results_dir)

    if not results_files:
        error_msg = f"No results.json files found in '{results_dir}'"
        if output_file:
            with open(output_file, 'w') as f:
                f.write(error_msg)
        else:
            logger.error(error_msg)
        return

    logger.info(f"Found {len(results_files)} results.json files in '{results_dir}'")

    # Generate markdown tables
    markdown_output = compile_all_results_to_markdown(results_dir)

    if output_file:
        # Save to file
        with open(output_file, 'w') as f:
            f.write(markdown_output)
        logger.info(f"Results saved to {output_file}")
    else:
        # Save to default location
        default_output_file = os.path.join(results_dir, "agg.md")
        with open(default_output_file, 'w') as f:
            f.write(markdown_output)
        logger.info(f"Results saved to {default_output_file}")


def run_prediction(args):
    """Run TabICL prediction on datasets based on provided arguments."""
    logger.add(
        os.path.join(args.results_dir_path, "logs/predict.log"), level="ERROR"
    )

    if args.dataset_type not in ["talent", "csv"]:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    # Load configuration from checkpoint
    ckpt = torch.load(args.checkpoint_path)
    config = Configuration(config=ckpt["hyper_parameters"])
    config.dataset_config.num_correlates = 49

    # alternative_config = Configuration(config_filepath="examples/configs/foundation_model_configs/config_icl_synthetic_train.yaml")
    # if not hasattr(config, 'tabicl_config'):
    #     config.tabicl_config = alternative_config.tabicl_config
    # if not hasattr(config, 'prior_config'):
    #     config.prior_config = alternative_config.prior_config

    # Get list of datasets based on dataset type
    if args.dataset_type == "talent":
        datasets = os.listdir(args.base_path)
        datasets = [
            d
            for d in datasets
            if os.path.isdir(os.path.join(args.base_path, d))
        ]
    elif args.dataset_type == "csv":
        datasets = os.listdir(args.base_path)
        datasets = [
            d
            for d in datasets
            if os.path.isfile(os.path.join(args.base_path, d))
            and d.endswith(".csv")
        ]
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    datasets.sort()
    logger.info(f"Found {len(datasets)} datasets")

    # Limit number of datasets if max_eval is specified
    if args.max_eval >= 0:
        datasets = datasets[: args.max_eval]

    results = {}

    pbar = tqdm(datasets, desc="Evaluating datasets")
    for dataset_name in pbar:
        try:
            # Create fresh model instance for each dataset to avoid state contamination
            model = create_tabicl_model(config, args.checkpoint_path)

            # Load dataset based on type
            if args.dataset_type == "talent":
                # Check if dataset is valid for the given task type
                json_path = os.path.join(
                    args.base_path, dataset_name, "info.json"
                )
                if not check_json(
                    json_path, config.tabicl_config.is_regression
                ):
                    continue

                # Load talent dataset
                dataset_dict = load_talent_dataset(dataset_name, args.base_path)
            elif args.dataset_type == "csv":
                # Load CSV dataset
                dataset_dict = load_csv_dataset(dataset_name, args.base_path)
                dataset_dict["dataset_name"] = dataset_name

            # Extract dataset components
            X_train = dataset_dict["X_train"]
            X_test = dataset_dict["X_test"]
            y_train = dataset_dict["y_train"]
            y_test = dataset_dict["y_test"]
            if "num_timestamp_features" in dataset_dict:
                num_timestamp_features = dataset_dict["num_timestamp_features"]
            else:
                num_timestamp_features = 0

            # Extract timestamps if available
            timestamp_train = dataset_dict.get("timestamp_train")
            timestamp_test = dataset_dict.get("timestamp_test")

            # Update progress bar description with dataset info
            pbar.set_description(
                f"{dataset_name}: X_train{X_train.shape}, X_test{X_test.shape}"
            )

            # Evaluate based on task type
            if config.tabicl_config.is_regression:
                regressor = model
                assert isinstance(regressor, TabICLRegressor)

                # Create scaler for y_train
                scaler = fit_y_scaler(y_train)

                # Always run future leaked evaluation
                future_leaked_results = evaluate_regression_future_leaked(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    regressor,
                    num_timestamp_features,
                    scaler=scaler,
                )

                if num_timestamp_features > 0:
                    # Create fresh model for univariate evaluation
                    univariate_model = create_tabicl_model(
                        config, args.checkpoint_path
                    )
                    univariate_regressor = univariate_model
                    assert isinstance(univariate_regressor, TabICLRegressor)

                    if args.autoregressive:
                        univariate_results = evaluate_regression_autoregressive(
                            evaluate_regression_univariate,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            univariate_regressor,
                            scaler=scaler,
                            num_timestamp_features=num_timestamp_features,
                        )
                    else:
                        univariate_results = evaluate_regression_univariate(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            univariate_regressor,
                            num_timestamp_features,
                            scaler=scaler,
                        )

                    # Create fresh model for multivariate evaluation
                    multivariate_model = create_tabicl_model(
                        config, args.checkpoint_path
                    )
                    multivariate_regressor = multivariate_model
                    assert isinstance(multivariate_regressor, TabICLRegressor)

                    if args.autoregressive:
                        multivariate_results = (
                            evaluate_regression_autoregressive(
                                evaluate_regression_multivariate,
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                multivariate_regressor,
                                scaler=scaler,
                                num_timestamp_features=num_timestamp_features,
                            )
                        )
                    else:
                        multivariate_results = evaluate_regression_multivariate(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            multivariate_regressor,
                            num_timestamp_features,
                            scaler=scaler,
                        )

                else:
                    univariate_results = {}
                    multivariate_results = {}

                # Combine all results
                dataset_results = {
                    **future_leaked_results,
                    **univariate_results,
                    **multivariate_results,
                }

                # Collect all predictions and ground truth for saving and plotting
                predictions_dict = {}
                if "predictions_fl" in dataset_results:
                    predictions_dict["Future Leaked"] = dataset_results[
                        "predictions_fl"
                    ]
                if "predictions_uni" in dataset_results:
                    predictions_dict["Univariate"] = dataset_results[
                        "predictions_uni"
                    ]
                if "predictions_multi" in dataset_results:
                    predictions_dict["True Multivariate"] = dataset_results[
                        "predictions_multi"
                    ]
                if "predictions_uni_auto" in dataset_results:
                    predictions_dict["Univariate Autoregressive"] = (
                        dataset_results["predictions_uni_auto"]
                    )
                if "predictions_multi_auto" in dataset_results:
                    predictions_dict["True Multivariate Autoregressive"] = (
                        dataset_results["predictions_multi_auto"]
                    )
                # Add original ground truth data directly
                predictions_dict["Ground Truth"] = y_test
                predictions_dict["Ground Truth History"] = y_train

                # Save predictions if any are available
                if predictions_dict:
                    save_predictions_dict(
                        predictions_dict, dataset_name, args.results_dir_path
                    )

                # Generate comparison plot if requested
                if args.plot and predictions_dict:
                    # Create mapping from prediction keys to MAE values
                    mae_mapping = {}
                    if (
                        "Future Leaked" in predictions_dict
                        and "mae_fl" in dataset_results
                    ):
                        mae_mapping["Future Leaked"] = float(
                            dataset_results["mae_fl"]
                        )
                    if (
                        "Univariate" in predictions_dict
                        and "mae_uni" in dataset_results
                    ):
                        mae_mapping["Univariate"] = float(
                            dataset_results["mae_uni"]
                        )
                    if (
                        "True Multivariate" in predictions_dict
                        and "mae_multi" in dataset_results
                    ):
                        mae_mapping["True Multivariate"] = float(
                            dataset_results["mae_multi"]
                        )
                    if (
                        "Univariate Autoregressive" in predictions_dict
                        and "mae_uni_auto" in dataset_results
                    ):
                        mae_mapping["Univariate Autoregressive"] = float(
                            dataset_results["mae_uni_auto"]
                        )
                    if (
                        "True Multivariate Autoregressive" in predictions_dict
                        and "mae_multi_auto" in dataset_results
                    ):
                        mae_mapping["True Multivariate Autoregressive"] = float(
                            dataset_results["mae_multi_auto"]
                        )

                    generate_comparison_plot_dict(
                        predictions_dict,
                        dataset_name,
                        args.results_dir_path,
                        timestamp_train,
                        timestamp_test,
                        mae_mapping,
                    )

                # Store regression results
                results[dataset_name] = {
                    "mode": config.tabicl_config.is_regression,
                    "mse_fl": f"{dataset_results.get('mse_fl', -1.0):.4f}",
                    "mae_fl": f"{dataset_results.get('mae_fl', -1.0):.4f}",
                    "mse_uni": f"{dataset_results.get('mse_uni', -1.0):.4f}",
                    "mae_uni": f"{dataset_results.get('mae_uni', -1.0):.4f}",
                    "mse_multi": f"{dataset_results.get('mse_multi', -1.0):.4f}",
                    "mae_multi": f"{dataset_results.get('mae_multi', -1.0):.4f}",
                    "mse_uni_auto": f"{dataset_results.get('mse_uni_auto', -1.0):.4f}",
                    "mae_uni_auto": f"{dataset_results.get('mae_uni_auto', -1.0):.4f}",
                    "mse_multi_auto": f"{dataset_results.get('mse_multi_auto', -1.0):.4f}",
                    "mae_multi_auto": f"{dataset_results.get('mae_multi_auto', -1.0):.4f}",
                    "has_timestamps": timestamp_train is not None
                    and timestamp_test is not None,
                }
            else:
                classifier = model
                assert isinstance(classifier, TabICLClassifier)

                # Always run future leaked evaluation
                future_leaked_results = evaluate_classification_future_leaked(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    classifier,
                    num_timestamp_features,
                )

                # Create fresh model for univariate evaluation
                univariate_model = create_tabicl_model(
                    config, args.checkpoint_path
                )
                univariate_classifier = univariate_model
                assert isinstance(univariate_classifier, TabICLClassifier)

                # Always run univariate evaluation
                univariate_results = evaluate_classification_univariate(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    univariate_classifier,
                    num_timestamp_features,
                )

                # Combine all results
                dataset_results = {
                    **future_leaked_results,
                    **univariate_results,
                }

                # Collect all predictions for saving (classification predictions are class labels)
                predictions_dict = {}
                if "predictions_fl" in dataset_results:
                    predictions_dict["Future Leaked"] = dataset_results[
                        "predictions_fl"
                    ]
                if "predictions_uni" in dataset_results:
                    predictions_dict["Univariate"] = dataset_results[
                        "predictions_uni"
                    ]
                # Add original ground truth data directly
                predictions_dict["Ground Truth"] = y_test

                # Save predictions if any are available
                if predictions_dict:
                    save_predictions_dict(
                        predictions_dict, dataset_name, args.results_dir_path
                    )

                # Log all results
                logger.info(
                    f"{dataset_name}: Accuracy (FL): {dataset_results['accuracy_fl']:.4f}, "
                    f"Accuracy (Uni): {dataset_results['accuracy_uni']:.4f}"
                )

                # Store classification results
                results[dataset_name] = {
                    "accuracy_fl": f"{dataset_results['accuracy_fl']:.4f}",
                    "accuracy_uni": f"{dataset_results['accuracy_uni']:.4f}",
                    "has_timestamps": timestamp_train is not None
                    and timestamp_test is not None,
                }

            # Cleanup after each dataset evaluation
            del model
            if "univariate_model" in locals():
                del univariate_model
            if "multivariate_model" in locals():
                del multivariate_model
            del X_train, X_test, y_train, y_test, dataset_dict

            # Clear PyTorch cache and run garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception:
            logger.exception(f"Error evaluating {dataset_name}")

    # Save final results
    os.makedirs(args.results_dir_path, exist_ok=True)

    results_path = os.path.join(args.results_dir_path, "results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved results to {results_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="TabICL prediction and results compilation utilities."
    )

    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Available modes")

    # Predict mode (existing functionality)
    predict_parser = subparsers.add_parser(
        "predict", help="Run TabICL prediction on datasets"
    )
    predict_parser.add_argument(
        "--base_path",
        type=str,
        default="/workspace/data/talent_benchmark/data",
        help="Base path to the benchmark data directory",
    )
    predict_parser.add_argument(
        "--dataset_type",
        type=str,
        default="csv",  # talent
        help="Type of dataset to evaluate",
    )
    predict_parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    predict_parser.add_argument(
        "--results_dir_path",
        type=str,
        default=".",
        help="Directory to save results and error files",
    )
    predict_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save plots for predictions vs ground truth (regression only)",
    )
    predict_parser.add_argument(
        "--max_eval",
        type=int,
        default=-1,
        help="Maximum number of datasets to evaluate",
    )
    predict_parser.add_argument(
        "--autoregressive",
        action="store_true",
        help="Use autoregressive prediction for univariate and multivariate regression (predict one row at a time)",
    )

    # Compile mode (new functionality)
    compile_parser = subparsers.add_parser(
        "compile", help="Compile results from JSON files into markdown tables"
    )
    compile_parser.add_argument(
        "--results_dir",
        type=str,
        default="/workspace/raghav/data/training_logs/synthetic_tabular/icl_single_tabular/match/results",
        help="Path to the results directory to search for results.json files (default: %(default)s)",
    )
    compile_parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the markdown output. If not specified, saves to agg.md in results_dir.",
    )

    args = parser.parse_args()

    # Check if mode is specified
    if not args.mode:
        parser.print_help()
        return

    if args.mode == "compile":
        # Find and compile all results
        find_and_compile_all_results(args.results_dir, args.output_file)
        return

    elif args.mode == "predict":
        # Run prediction functionality
        run_prediction(args)


if __name__ == "__main__":
    main()
