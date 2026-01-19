#!/usr/bin/env python3
"""
LightGBM-based TSTR/TRTR/TRSTR implementation with Celery integration.

This script loads time-series datasets (stored as .npy files),
and trains a LightGBM model.
It supports both classification and regression tasks (as defined in the config),
and can run either synchronously or as a long-running asynchronous task via Celery.
The evaluation results, configuration, and trained model are saved to disk.

Usage:
    Synchronous mode:
        python3 tstr_lightgbm.py --config path/to/config.yaml --synthetic_percentage 50 [--learning_rate 0.001] [--max_epoch 100] [--task_type classification] [--target_column 0]

    Asynchronous mode using Celery:
        python3 tstr_lightgbm.py --config path/to/config.yaml --synthetic_percentage 50 [--learning_rate 0.001] [--max_epoch 100] [--task_type regression] [--target_column 1] --use_celery

Before using asynchronous mode, ensure you have a Redis server running (default: redis://localhost:6379/0)
and launch a Celery worker:
       celery -A src.synthefy_pkg.app.celery_app worker --loglevel=info --pool=solo
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import mode
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    multilabel_confusion_matrix,
    r2_score,
)

from synthefy_pkg.app.celery_app import celery_app
from synthefy_pkg.utils.tstr_utils import (
    construct_dataset_paths_by_config,
    convert_h5_to_npy,
    convert_pkls_to_npy,
    get_classification_indices,
    get_regression_index,
    sample_synthetic_data,
)

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE", "")
DEFAULT_MAX_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SYNTHETIC_PERCENTAGE = 100


#############################################
# Feature Extraction functionality  #
#############################################


def extract_features_simple(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Extract simple statistical features from raw time-series data.

    Args:
        X (np.ndarray): Time-series array of shape (num_samples, channels, num_timesteps).

    Returns:
        Tuple[np.ndarray, List[str]]: Extracted features and the list of feature names.
    """
    N, channels, window_size = X.shape
    features_list = []
    feature_names = []

    for i in range(channels):
        channel_data = X[:, i, :]  # Shape: (N, window_size)
        # Extract simple statistics
        features_list.append(np.mean(channel_data, axis=1))
        feature_names.append(f"channel_{i}_mean")
        features_list.append(np.std(channel_data, axis=1))
        feature_names.append(f"channel_{i}_std")
        features_list.append(np.min(channel_data, axis=1))
        feature_names.append(f"channel_{i}_min")
        features_list.append(np.max(channel_data, axis=1))
        feature_names.append(f"channel_{i}_max")

    features = np.column_stack(features_list)
    return features, feature_names


def load_timeseries_data(
    ts_paths: List[str],
    split: str,
    path_types: Dict[str, Any],
    synthetic_percentage: int = 100,
) -> np.ndarray:
    """
    Loads the time series data for the given split and task mode.

    Args:
        ts_paths (List[str]): List of file paths for time series data.
        split (str): One of "train", "val", or "test".
        path_types (Dict[str, Any]): Dictionary mapping path strings to "synthetic" or "original".
        synthetic_percentage (int): Percentage of synthetic data to use (0-100).

    Returns:
        np.ndarray: Loaded time series data.
    """
    ts_list: List[np.ndarray] = []
    for path in ts_paths:
        if not os.path.exists(path):
            directory = os.path.dirname(path)
            logger.warning(
                f"Timeseries file not found: {path}. Attempting conversion."
            )
            try:
                convert_h5_to_npy(directory, split)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(
                    f"H5 conversion failed: {e}. Trying PKL conversion."
                )
                try:
                    convert_pkls_to_npy(directory, split)
                except Exception as e_pkl:
                    logger.error(f"PKL conversion failed: {e_pkl}")
                    raise FileNotFoundError(
                        f"Timeseries file not found after conversion attempts: {path}"
                    )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Timeseries file still not found: {path}"
                )
        ts = np.load(path, allow_pickle=True)
        if path_types[path] == "synthetic":
            ts = sample_synthetic_data(ts, synthetic_percentage)
        ts_list.append(ts)

    X = np.concatenate(ts_list, axis=0)
    return X


def load_discrete_conditions(
    disc_paths: List[str],
    split: str,
    path_types: Dict[str, Any],
    synthetic_percentage: int = 100,
) -> np.ndarray:
    """
    Loads discrete condition data for classification tasks.

    Args:
        disc_paths (List[str]): List of file paths for discrete conditions.
        split (str): One of "train", "val", or "test".
        path_types (Dict[str, Any]): Dictionary mapping path strings to "synthetic" or "original".
        synthetic_percentage (int): Percentage of synthetic data to use (0-100).

    Returns:
        np.ndarray: Processed discrete conditions.
    """
    disc_list: List[np.ndarray] = []
    for i, path in enumerate(disc_paths):
        if not os.path.exists(path):
            directory = os.path.dirname(path)
            logger.warning(
                f"Discrete conditions file not found: {path}. Attempting conversion."
            )
            try:
                convert_h5_to_npy(directory, split)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(
                    f"H5 conversion failed for discrete conditions: {e}. Trying PKL conversion."
                )
                try:
                    convert_pkls_to_npy(directory, split)
                except Exception as e_pkl:
                    logger.error(f"PKL conversion failed: {e_pkl}")
                    raise FileNotFoundError(
                        f"Discrete conditions file not found after conversion attempts: {path}"
                    )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Discrete conditions file still not found: {path}"
                )
        disc = np.load(path, allow_pickle=True)
        if path_types[path + f"_{i}"] == "synthetic":
            disc = sample_synthetic_data(disc, synthetic_percentage)
        disc_list.append(disc)

    y_disc_full = np.concatenate(disc_list, axis=0)
    return y_disc_full


def load_continuous_conditions(
    cont_paths: List[str],
    split: str,
    path_types: Dict[str, Any],
    synthetic_percentage: int = 100,
) -> np.ndarray:
    """
    Loads continuous condition data for regression tasks.

    Args:
        cont_paths (List[str]): List of file paths for continuous conditions.
        split (str): One of "train", "val", or "test".
        path_types (Dict[str, Any]): Dictionary mapping path strings to "synthetic" or "original".
        synthetic_percentage (int): Percentage of synthetic data to use (0-100).
    Returns:
        np.ndarray: Processed continuous conditions.
    """
    cont_list: List[np.ndarray] = []
    for i, path in enumerate(cont_paths):
        if not os.path.exists(path):
            directory = os.path.dirname(path)
            logger.warning(
                f"Continuous conditions file not found: {path}. Attempting conversion."
            )
            try:
                convert_h5_to_npy(directory, split)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(
                    f"H5 conversion failed for continuous conditions: {e}. Trying PKL conversion."
                )
                try:
                    convert_pkls_to_npy(directory, split)
                except Exception as e_pkl:
                    logger.error(f"PKL conversion failed: {e_pkl}")
                    raise FileNotFoundError(
                        f"Continuous conditions file not found after conversion attempts: {path}"
                    )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Continuous conditions file still not found: {path}"
                )
        cont = np.load(path, allow_pickle=True)
        if path_types[path + f"_{i}"] == "synthetic":
            cont = sample_synthetic_data(cont, synthetic_percentage)
        cont_list.append(cont)

    y_cont_full = np.concatenate(cont_list, axis=0)
    y_cont_full = np.transpose(y_cont_full, (0, 2, 1))
    return y_cont_full


def load_data(
    config: Dict[str, Any],
    split: str,
    synthetic_percentage: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the dataset for the given split, applies feature extraction, and processes labels.

    Args:
        config (dict): Configuration dictionary.
        split (str): One of "train", "val", or "test".
        synthetic_percentage (int): Percentage of synthetic data to use (0-100).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X_features, y)
    """

    synthetic_or_original_or_custom = config["tstr_config"][
        "synthetic_or_original_or_custom"
    ].lower()

    ts_paths, disc_paths, cont_paths, path_types = (
        construct_dataset_paths_by_config(
            config,
            split,
            synthetic_or_original_or_custom=synthetic_or_original_or_custom,
        )
    )

    # Load time series data and apply sampling if needed.
    X_raw = load_timeseries_data(
        ts_paths, split, path_types, synthetic_percentage
    )
    logger.info(f"Raw time-series shape for '{split}' split: {X_raw.shape}")

    logger.info("Extracting features from time series...")
    X_features, feature_names = extract_features_simple(X_raw)
    logger.info(
        f"Extracted features shape: {X_features.shape} with {len(feature_names)} features."
    )

    task = config["tstr_config"]["classification_or_regression"].lower()
    if task == "classification":
        y_disc = load_discrete_conditions(
            disc_paths, split, path_types, synthetic_percentage
        )
        logger.info(f"Discrete conditions shape: {y_disc.shape}")

        target_column = get_classification_indices(
            config["tstr_config"]["dataset"]
        )[0]

        # If the discrete conditions are 2D, expand dimensions and repeat along the window axis.
        if len(y_disc.shape) == 2:
            y_disc = np.expand_dims(y_disc, axis=1)
            y_disc = np.repeat(y_disc, X_raw.shape[2], axis=1)

        y_disc = y_disc[:, :, target_column]

        unique_values = np.unique(y_disc)
        if len(unique_values) > 1:
            logger.warning(
                "Multiple unique values found in the target column. Using mode for each window."
            )

        y = mode(y_disc, axis=1)[0].flatten()

    elif task == "regression":
        target_column = get_regression_index(config["tstr_config"]["dataset"])
        y_cont = load_continuous_conditions(
            cont_paths, split, path_types, synthetic_percentage
        )
        logger.info(f"Continuous conditions shape: {y_cont.shape}")
        y_cont = y_cont[:, :, target_column]
        y = np.mean(y_cont, axis=1)
    else:
        raise ValueError(
            "Invalid task specified in config. Must be 'classification' or 'regression'."
        )

    logger.info(
        f"Loaded '{split}' data: Features shape: {X_features.shape}, Labels shape: {y.shape}"
    )
    return X_features, y


#############################################
# Evaluation Functions                      #
#############################################


def eval_multilabel_classification_results(
    predictions: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate model results for multi-label classification.

    Args:
        predictions: Predicted labels.
        ground_truth: Ground truth labels.

    Returns:
        Dictionary containing the results of the evaluation.
    """
    try:
        pred_classes = (predictions > 0.5).astype(int)
        exact_match_accuracy = float(
            np.mean(np.all(pred_classes == ground_truth, axis=1))
        )
        per_label_accuracy = np.mean(pred_classes == ground_truth, axis=0)
        mcm = multilabel_confusion_matrix(ground_truth, pred_classes)
        mcm_list = mcm.tolist()
        per_class_metrics = []
        for cm in mcm:
            tn, fp, fn, tp = cm.ravel()
            per_class_metrics.append(
                {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                }
            )
        logger.info(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
        for i, acc in enumerate(per_label_accuracy):
            logger.info(f"Accuracy for label {i}: {acc:.4f}")
        return {
            "multilabel_confusion_matrices": mcm_list,
            "per_class_metrics": per_class_metrics,
            "exact_match_accuracy": exact_match_accuracy,
            "per_label_accuracy": per_label_accuracy.tolist(),
        }
    except Exception as e:
        logger.error(f"Error evaluating multilabel classification results: {e}")
        raise


def eval_classification_results(
    predictions: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate model results for classification.

    Args:
        predictions: Predicted labels.
        ground_truth: Ground truth labels.

    Returns:
        Dictionary containing the results of the evaluation.
    """
    try:
        accuracy = float(accuracy_score(ground_truth, predictions))
        cm = confusion_matrix(ground_truth, predictions).tolist()
        report_dict = classification_report(
            ground_truth, predictions, output_dict=True
        )
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Confusion Matrix: {cm}")
        logger.info(f"Classification Report: {report_dict}")
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": report_dict,
        }
    except Exception as e:
        logger.error(f"Error evaluating classification results: {e}")
        raise


def eval_regression_results(
    predictions: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, Any]:
    """Evaluate regression results using various metrics."""
    mse = round(float(mean_squared_error(ground_truth, predictions)), 3)
    rmse = round(float(np.sqrt(mse)), 3)
    mae = round(float(mean_absolute_error(ground_truth, predictions)), 3)
    r2 = round(float(r2_score(ground_truth, predictions)), 3)
    epsilon = 1e-10
    percentage_errors = (
        np.abs((ground_truth - predictions) / (ground_truth + epsilon)) * 100
    )
    mape = round(float(np.mean(percentage_errors)), 3)
    mdape = round(float(np.median(percentage_errors)), 3)
    numerator = np.abs(ground_truth - predictions)
    denominator = np.abs(ground_truth) + np.abs(predictions) + epsilon
    smape = round(float(200 * np.mean(numerator / denominator)), 3)
    logger.info(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    logger.info(f"MAPE: {mape:.3f}%, MDAPE: {mdape:.3f}%, SMAPE: {smape:.3f}%")
    logger.info(f"R2 Score: {r2:.3f}")
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "MDAPE": mdape,
        "SMAPE": smape,
        "R2": r2,
    }


#############################################
# Training and Evaluation for each Task     #
#############################################


def custom_callback(env, celery_self=None):
    """Custom LightGBM callback to log progress during training (every 10 iterations)."""
    if env.iteration % 10 == 0:
        eval_info = [
            f"{data_name}'s {eval_name}: {result_val:.5f}"
            for data_name, eval_name, result_val, _ in env.evaluation_result_list
        ]
        logger.info(f"Iteration {env.iteration}: " + ", ".join(eval_info))

        if celery_self is not None:
            celery_self.update_state(
                state="PROGRESS",
                meta={
                    "epoch": env.iteration,
                    "evaluation": ", ".join(eval_info),
                },
            )


def load_train_val_test_data(
    config: Dict[str, Any], synthetic_percentage: int = 100
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load the train, val, and test data.

    Args:
        config: Configuration dictionary.
        synthetic_percentage: Percentage of synthetic data to use (0-100).

    Returns:
        Tuple containing: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    X_train, y_train = load_data(config, "train", synthetic_percentage)
    logger.info("Loading validation dataset...")
    X_val, y_val = load_data(config, "val", synthetic_percentage)
    logger.info("Loading test dataset...")
    X_test, y_test = load_data(config, "test", synthetic_percentage)
    logger.info(
        f"Data Shapes: Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(
    max_epochs: int,
    learning_rate: float,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    celery_self=None,
) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
    """
    Train a LightGBM model.

    Args:
        max_epochs: Maximum number of boosting rounds.
        learning_rate: Learning rate for the model.
        task_type: Type of task to train (classification or regression).
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        celery_self: Celery task instance.
    Returns:
        Trained LightGBM model.
    """

    if task_type == "classification":
        model = lgb.LGBMClassifier(
            learning_rate=learning_rate,
            n_estimators=max_epochs,
            verbose=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            n_estimators=max_epochs,
            verbose=-1,
        )

    logger.info("Training LightGBM model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lambda env: custom_callback(env, celery_self=celery_self),
            lgb.early_stopping(stopping_rounds=10),
        ],
    )
    return model


def evaluate_model(
    model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    """
    Evaluate the model on the test dataset.

    Args:
        model: Trained LightGBM model.
        X_test: Test features.
        y_test: Test labels.
        task_type: Type of task to evaluate (classification or regression).

    Returns:
        Dictionary containing the results of the evaluation.
    """
    logger.info("Evaluating model on test dataset...")

    if task_type == "classification":
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            if isinstance(model, lgb.LGBMClassifier) and hasattr(
                model, "predict_proba"
            ):
                y_pred = np.array(model.predict_proba(X_test))
            else:
                y_pred = np.array(model.predict(X_test))
            results = eval_multilabel_classification_results(y_pred, y_test)
        else:
            y_pred = np.array(model.predict(X_test))
            results = eval_classification_results(y_pred, y_test)
    else:
        y_pred = np.array(model.predict(X_test))
        results = eval_regression_results(y_pred, y_test)

    return results


def save_results(
    config: Dict[str, Any],
    results: Dict[str, Any],
    model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    synthetic_or_original_or_custom: str,
    synthetic_percentage: int = 100,
):
    """
    Save the results of the training and evaluation.

    Args:
        config: Configuration dictionary.
        results: Results of the training and evaluation.
        model: Trained LightGBM model.
        synthetic_or_original_or_custom: Whether to use synthetic, original, or custom data.
        synthetic_percentage: Percentage of synthetic data to use (0-100).
    """
    dataset_config = config.get("dataset_config", {})
    tstr_config = config.get("tstr_config", {})
    training_config = tstr_config.get("training", {})

    base_log_dir = training_config.get("log_dir", "/tmp/tstr_lightgbm_training")
    dataset_name = dataset_config.get("dataset_name", "unknown_dataset")

    dir_name = f"{dataset_name}-{synthetic_or_original_or_custom}"
    if synthetic_or_original_or_custom == "custom":
        dir_name += f"-synt{synthetic_percentage}pct"
    dir_name += f"-{training_config.get('max_epochs', DEFAULT_MAX_EPOCHS)}-{training_config.get('learning_rate', DEFAULT_LEARNING_RATE)}"

    mode_output_dir = os.path.join(base_log_dir, dir_name)
    os.makedirs(mode_output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {mode_output_dir}")

    results_path = os.path.join(mode_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved evaluation results to: {results_path}")

    config_out_path = os.path.join(mode_output_dir, "tstr_config.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(tstr_config, f)
    logger.info(f"Saved tstr_config to: {config_out_path}")

    model_path = os.path.join(mode_output_dir, "lightgbm_model.txt")
    booster = model.booster_
    booster.save_model(model_path)
    logger.info(f"Saved LightGBM model to: {model_path}")


def train_and_evaluate_task(
    config: Dict[str, Any],
    synthetic_percentage: int = 100,
    celery_self=None,
) -> Dict[str, Any]:
    """Train and evaluate a LightGBM model.

    Args:
        config: Configuration dictionary.
        synthetic_percentage: Percentage of synthetic data to use (0-100).
        celery_self: Celery task instance.
    Returns:
        Dictionary containing the results of the training and evaluation.
    """

    tstr_config = config.get("tstr_config", {})
    training_config = tstr_config.get("training", {})

    task_type = tstr_config.get("classification_or_regression", "").lower()
    if task_type not in ["classification", "regression"]:
        logger.error("Task must be either 'classification' or 'regression'.")
        sys.exit("Invalid task specified in config.")

    X_train, y_train, X_val, y_val, X_test, y_test = load_train_val_test_data(
        config, synthetic_percentage
    )

    model = train_model(
        training_config.get("max_epochs", DEFAULT_MAX_EPOCHS),
        training_config.get("learning_rate", DEFAULT_LEARNING_RATE),
        task_type,
        X_train,
        y_train,
        X_val,
        y_val,
        celery_self=celery_self,
    )

    synthetic_or_original_or_custom = config["tstr_config"][
        "synthetic_or_original_or_custom"
    ]
    results = evaluate_model(model, X_test, y_test, task_type)
    save_results(
        config,
        results,
        model,
        synthetic_or_original_or_custom,
        synthetic_percentage,
    )
    return {task_type: results}


#############################################
# Celery Task Definition                    #
#############################################


@celery_app.task(name="tstr_lightgbm.train_model_task", bind=True)
def train_model_task(
    self,
    config_path: str,
    synthetic_percentage: int = 100,
    learning_rate: Optional[float] = None,
    max_epoch: Optional[int] = None,
    task_type: Optional[str] = None,
    target_column: Optional[int] = None,
    synthetic_or_original_or_custom: Optional[str] = None,
) -> str:
    """
    Celery task that loads the config file and runs the training process for the specified task.

    Args:
        config_path (str): Path to the YAML configuration file.
        synthetic_percentage (int): Percentage of synthetic data to use (0-100).
        learning_rate (float, optional): Override learning rate from config.
        max_epoch (int, optional): Override maximum epochs from config.
        task_type (str, optional): Override task type ("classification" or "regression").
        target_column (int, optional): Override target column index.
        synthetic_or_original_or_custom (str, optional): Override synthetic_or_original_or_custom from config.
    Returns:
        str: Completion message.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config parameters if provided
    if learning_rate is not None:
        config["tstr_config"]["training"]["learning_rate"] = learning_rate
    if max_epoch is not None:
        config["tstr_config"]["training"]["max_epochs"] = max_epoch
    if task_type is not None:
        config["tstr_config"]["classification_or_regression"] = task_type
    if synthetic_or_original_or_custom is not None:
        config["tstr_config"]["synthetic_or_original_or_custom"] = (
            synthetic_or_original_or_custom
        )
    if target_column is not None:
        if (
            config["tstr_config"]["classification_or_regression"].lower()
            == "classification"
        ):
            config["tstr_config"]["dataset"]["classification_start_index"] = (
                target_column
            )
            config["tstr_config"]["dataset"][
                "classification_inclusive_end_index"
            ] = target_column
        else:
            config["tstr_config"]["dataset"]["index_of_interest"] = (
                target_column
            )

    results_summary = {}
    results = train_and_evaluate_task(
        config,
        synthetic_percentage,
        celery_self=self,
    )
    results_summary.update(results)
    return json.dumps(results_summary, indent=4)


#############################################
# Main Function                             #
#############################################


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LightGBM TSTR/TRTR/TRSTR Training Script with Celery integration"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--synthetic_percentage",
        type=int,
        default=DEFAULT_SYNTHETIC_PERCENTAGE,
        help="Percentage of synthetic data to use in TRSTR mode (0-100)",
    )
    parser.add_argument(
        "--use_celery",
        action="store_true",
        help="Run training asynchronously via Celery",
    )
    # New command line arguments for overriding configuration values
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Override max epochs from config",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["classification", "regression"],
        default=None,
        help="Override task type (classification or regression)",
    )
    parser.add_argument(
        "--target_column",
        type=int,
        default=None,
        help="Override target column index",
    )
    parser.add_argument(
        "--synthetic_or_original_or_custom",
        type=str,
        choices=["synthetic", "original", "custom"],
        default=None,
        help="Override synthetic_or_original_or_custom from config",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(f"Config file not found: {args.config}")

    if args.synthetic_percentage < 0 or args.synthetic_percentage > 100:
        sys.exit(
            f"Invalid synthetic_percentage: {args.synthetic_percentage}. Must be between 0-100."
        )

    # Load the configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply command line overrides to the config
    if args.learning_rate is not None:
        config["tstr_config"]["training"]["learning_rate"] = args.learning_rate
    if args.max_epoch is not None:
        config["tstr_config"]["training"]["max_epochs"] = args.max_epoch
    if args.task_type is not None:
        config["tstr_config"]["classification_or_regression"] = args.task_type
    if args.synthetic_or_original_or_custom is not None:
        config["tstr_config"]["synthetic_or_original_or_custom"] = (
            args.synthetic_or_original_or_custom
        )
    if args.target_column is not None:
        if (
            config["tstr_config"]["classification_or_regression"].lower()
            == "classification"
        ):
            config["tstr_config"]["dataset"]["classification_start_index"] = (
                args.target_column
            )
            config["tstr_config"]["dataset"][
                "classification_inclusive_end_index"
            ] = args.target_column
        else:
            config["tstr_config"]["dataset"]["index_of_interest"] = (
                args.target_column
            )

    if args.use_celery:
        # In asynchronous mode, pass the arguments to the Celery task.
        task = train_model_task.delay(
            args.config,
            args.synthetic_percentage,
            args.learning_rate,
            args.max_epoch,
            args.task_type,
            args.target_column,
            args.synthetic_or_original_or_custom,
        )
        logger.info(
            f"Training task scheduled asynchronously with task ID: {task.id}"
        )
        sys.exit(0)
    else:
        results_summary = {}
        results = train_and_evaluate_task(config, args.synthetic_percentage)
        results_summary.update(results)
        logger.info("Training task completed. Summary of evaluation metrics:")
        logger.info(json.dumps(results_summary, indent=4))


if __name__ == "__main__":
    main()
