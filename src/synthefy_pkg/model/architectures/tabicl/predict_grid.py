import argparse
import copy
import gc
import json
import os
from typing import Any, Dict, Optional

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
from synthefy_pkg.model.architectures.tabicl.predict import (
    calculate_regression_metrics,
    create_tabicl_model,
    find_and_compile_all_results,
    fit_y_scaler,
    generate_comparison_plot_dict,
    save_predictions_dict,
)
from synthefy_pkg.model.foundation_model.utils import generate_target_mask
 
DEFAULT_TARGET_MASK_KWARGS = {
    "mask_mixing_rates": [1.0],
    "target_masking_schemes": ["train_test"],
    "target_filtering_schemes": ["none"],
    "block_target_mask_mean": 0,
    "block_target_mask_range": 0,
    "block_mask_num": 1,
    "block_mask_every": True,
    "row_mask_ratio": 0.0,
    "row_mask_min": 0,
    "row_use_train_test": True,
    "target_mask_ratio": 0.0,
    "batch_size": 1,
    "device": "cpu",
}


def evaluate_grid_regression_future_leaked(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLGridRegressor,
    num_timestamp_features: int = 0,
    is_auto_regressive: bool = False,
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """
    Evaluate standard regression task with future leakage.
    No concept of auto-regressive prediction here.
    """

    results = {}
    target_mask_kwargs = copy.deepcopy(DEFAULT_TARGET_MASK_KWARGS)
    target_mask_kwargs.update(
        {
            "time_series_length": X_train.shape[0] + X_test.shape[0],
            "num_correlates": X_train.shape[1] + 1,
            "train_sizes": torch.tensor([X_train.shape[0]]),
            "seq_lens": torch.tensor([X_train.shape[0] + X_test.shape[0]]),
            "time_columns": num_timestamp_features,
        }
    )

    # Train and evaluate multivariate regressor
    regressor.fit(X_train, y_train)
    target_mask, _ = generate_target_mask(**target_mask_kwargs)
    target_mask = target_mask.reshape(1, -1, X_train.shape[0] + X_test.shape[0])
    predict_mask = target_mask[:, :, -X_test.shape[0] :]
    predict_mask = predict_mask.transpose(2, 1).cpu().numpy()[0]
    predictions = regressor.predict(X_test, y_test, predict_mask=predict_mask)

    if not is_auto_regressive:
        # Calculate MSE and MAE using the utility function
        mse, mae = calculate_regression_metrics(
            predictions[:, -1], y_test, scaler
        )
        results["mse_fl"] = mse
        results["mae_fl"] = mae

    results["predictions_fl"] = predictions

    return results
 

def evaluate_grid_regression_univariate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLGridRegressor,
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
    target_mask_kwargs = copy.deepcopy(DEFAULT_TARGET_MASK_KWARGS)
    # Create copies to avoid modifying the original data
    X_train_uni = X_train.iloc[:, :num_timestamp_features].copy()
    X_test_uni = X_test.iloc[:, :num_timestamp_features].copy()
    target_mask_kwargs.update(
        {
            "time_series_length": X_train_uni.shape[0] + X_test_uni.shape[0],
            "num_correlates": num_timestamp_features + 1,
            "train_sizes": torch.tensor([X_train_uni.shape[0]]),
            "seq_lens": torch.tensor(
                [X_train_uni.shape[0] + X_test_uni.shape[0]]
            ),
            "time_columns": num_timestamp_features,
        }
    )
    regressor.fit(X_train_uni, y_train)
    target_mask, _ = generate_target_mask(**target_mask_kwargs)
    target_mask = target_mask.reshape(
        1, -1, X_train_uni.shape[0] + X_test_uni.shape[0]
    )
    predict_mask = target_mask[:, :, -X_test_uni.shape[0] :]
    predict_mask = predict_mask.transpose(2, 1).cpu().numpy()[0]
    predictions = regressor.predict(
        X_test_uni, y_test, predict_mask=predict_mask
    )

    if not is_auto_regressive:
        # Calculate MSE and MAE using the utility function
        mse, mae = calculate_regression_metrics(
            predictions[:, -1], y_test, scaler
        )
        results["mse_uni"] = mse
        results["mae_uni"] = mae

    results["predictions_uni"] = predictions

    return results


def evaluate_grid_regression_multivariate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLGridRegressor,
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
    target_mask_kwargs = copy.deepcopy(DEFAULT_TARGET_MASK_KWARGS)
    target_mask_kwargs.update(
        {
            "target_masking_schemes": ["train_test_block"],
            "time_series_length": X_train.shape[0]
            + X_test.shape[
                0
            ],  # TODO: if autorgressive, just make X_test = 1 row and then reappend externally
            "num_correlates": X_train.shape[1] + 1,
            "train_sizes": torch.tensor([X_train.shape[0]]),
            "seq_lens": torch.tensor([X_train.shape[0] + X_test.shape[0]]),
            "time_columns": num_timestamp_features,
        }
    )

    # Train and evaluate multivariate regressor
    regressor.fit(X_train, y_train)
    target_mask, _ = generate_target_mask(**target_mask_kwargs)
    target_mask = target_mask.reshape(1, -1, X_train.shape[0] + X_test.shape[0])
    predict_mask = target_mask[:, :, -X_test.shape[0] :]
    predict_mask = predict_mask.transpose(2, 1).cpu().numpy()[0]
    predictions = regressor.predict(X_test, y_test, predict_mask=predict_mask)

    if not is_auto_regressive:
        # Calculate MSE and MAE using the utility function
        mse, mae = calculate_regression_metrics(
            predictions[:, -1], y_test, scaler
        )
        results["mse_multi"] = mse
        results["mae_multi"] = mae

    results["predictions_multi"] = predictions

    return results


def evaluate_grid_regression_autoregressive(
    evaluation_function,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    regressor: TabICLGridRegressor,
    scaler: Optional[StandardScaler] = None,
    num_timestamp_features: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Autoregressive wrapper for grid regression evaluation functions.
    Predicts one row at a time, using previous predictions as part of training data.

    Note: The grid regressor's predict method returns a concatenation of X_test and y_test
    predictions, so we need to extract only the target prediction (last column).

    Args:
        evaluation_function: The evaluation function to wrap (e.g., evaluate_grid_regression_univariate)
        X_train, X_test, y_train, y_test: Data arrays
        regressor: The TabICL grid regressor
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
            num_timestamp_features=num_timestamp_features,
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

        # For grid regressor, predictions are concatenated (X_test + y_test)
        # We need to extract only the target prediction (last column)
        pred_array = step_results[prediction_key]
        assert pred_array.ndim == 2, (
            f"Predictions are not 2D: {pred_array.shape} (expected shape: (1, n_features + 1))"
        )
        assert len(pred_array) == 1, (
            f"More than one row should not be predicted: {pred_array.shape} (expected shape: (1, n_features + 1))"
        )

        # Grid regressor returns (n_samples, n_features + 1), last column is target
        pred_value = pred_array[
            0, -1
        ]  # Extract target prediction from first (and only) row

        if evaluation_function == evaluate_grid_regression_univariate:
            # For univariate, we only need to append the timestamp features from the test row
            # Extract timestamp features from the test row
            X_test_row_timestamp = X_test_row.iloc[:, :num_timestamp_features]
            X_train_iter = pd.concat(
                [X_train_iter, X_test_row_timestamp], ignore_index=True
            )

        elif evaluation_function == evaluate_grid_regression_multivariate:
            # Unpack and append correlate preds for multivariate
            # Convert numpy array to DataFrame with same columns as X_train_iter
            pred_features = pred_array[:, :-1]  # Shape: (1, n_features)
            assert pred_features.shape[1] == X_train_iter.shape[1], (
                f"Predicted features shape {pred_features.shape} doesn't match "
                f"X_train_iter columns {X_train_iter.shape[1]}"
            )
            pred_features_df = pd.DataFrame(
                pred_features, columns=X_train_iter.columns
            )
            X_train_iter = pd.concat(
                [X_train_iter, pred_features_df], ignore_index=True
            )

        else:
            raise ValueError(
                f"Unknown evaluation function: {evaluation_function}"
            )

        predictions_list.append(pred_value)

        # Append prediction to training targets and test row to training features
        y_train_iter = np.append(y_train_iter, pred_value)

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

    # Create results directory and file path at the beginning
    os.makedirs(args.results_dir_path, exist_ok=True)
    results_path = os.path.join(args.results_dir_path, "results.json")

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
                assert isinstance(regressor, TabICLGridRegressor)

                # Create scaler for y_train
                scaler = fit_y_scaler(y_train)

                # Always run future leaked evaluation
                future_leaked_results = evaluate_grid_regression_future_leaked(
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
                    assert isinstance(univariate_regressor, TabICLGridRegressor)

                    # Always run non-autoregressive univariate evaluation
                    univariate_results = evaluate_grid_regression_univariate(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        univariate_regressor,
                        num_timestamp_features,
                        scaler=scaler,
                    )

                    # If autoregressive is enabled, also run autoregressive evaluation
                    if args.autoregressive:
                        # Create fresh model for autoregressive evaluation to avoid state contamination
                        univariate_auto_model = create_tabicl_model(
                            config, args.checkpoint_path
                        )
                        univariate_auto_regressor = univariate_auto_model
                        assert isinstance(
                            univariate_auto_regressor, TabICLGridRegressor
                        )

                        univariate_auto_results = (
                            evaluate_grid_regression_autoregressive(
                                evaluate_grid_regression_univariate,
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                univariate_auto_regressor,
                                scaler,
                                num_timestamp_features,
                            )
                        )
                        # Merge autoregressive results with non-autoregressive results
                        univariate_results.update(univariate_auto_results)

                    # Create fresh model for multivariate evaluation
                    multivariate_model = create_tabicl_model(
                        config, args.checkpoint_path
                    )
                    multivariate_regressor = multivariate_model
                    assert isinstance(
                        multivariate_regressor, TabICLGridRegressor
                    )

                    # Always run non-autoregressive multivariate evaluation
                    multivariate_results = (
                        evaluate_grid_regression_multivariate(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            multivariate_regressor,
                            num_timestamp_features,
                            scaler=scaler,
                        )
                    )

                    # If autoregressive is enabled, also run autoregressive evaluation
                    if args.autoregressive:
                        # Create fresh model for autoregressive evaluation to avoid state contamination
                        multivariate_auto_model = create_tabicl_model(
                            config, args.checkpoint_path
                        )
                        multivariate_auto_regressor = multivariate_auto_model
                        assert isinstance(
                            multivariate_auto_regressor, TabICLGridRegressor
                        )

                        multivariate_auto_results = (
                            evaluate_grid_regression_autoregressive(
                                evaluate_grid_regression_multivariate,
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                multivariate_auto_regressor,
                                scaler,
                                num_timestamp_features,
                            )
                        )
                        # Merge autoregressive results with non-autoregressive results
                        multivariate_results.update(multivariate_auto_results)

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

                # Save results after each dataset evaluation
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

            else:
                classifier = model
                assert isinstance(classifier, TabICLClassifier)

                # Classification not supported in grid version yet
                logger.error(
                    f"Classification not supported for {dataset_name} in grid version"
                )
                raise ValueError

            # Cleanup after each dataset evaluation
            del model
            if "univariate_model" in locals():
                del univariate_model
            if "multivariate_model" in locals():
                del multivariate_model
            if "univariate_auto_model" in locals():
                del univariate_auto_model
            if "multivariate_auto_model" in locals():
                del multivariate_auto_model
            del X_train, X_test, y_train, y_test, dataset_dict

            # Clear PyTorch cache and run garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception:
            logger.exception(f"Error evaluating {dataset_name}")

    # Save final results
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
        help="Enable autoregressive prediction in addition to standard prediction for univariate and multivariate regression (performs both evaluations)",
    )

    # Compile mode (new functionality)
    compile_parser = subparsers.add_parser(
        "compile", help="Compile results from JSON files"
    )
    compile_parser.add_argument(
        "--results_dir",
        type=str,
        default="/workspace/raghav/data/training_logs/synthetic_tabular/icl_single_tabular/match/results",
        help="Path to the results directory to search for results.json files (default: %(default)s)",
    )

    args = parser.parse_args()

    # Check if mode is specified
    if not args.mode:
        parser.print_help()
        return

    if args.mode == "compile":
        # Find and compile all results
        find_and_compile_all_results(args.results_dir)
        return

    elif args.mode == "predict":
        # Run prediction functionality
        run_prediction(args)


if __name__ == "__main__":
    main()

# /mnt/workspace1/data/synthefy_data/training_logs/synthetic_tabular_small/grid_tabicl/synthetic_icl_grid/checkpoints/checkpoint_step_200_val_loss_1.1128.ckpt
# uv run src/synthefy_pkg/model/architectures/tabicl/predict_grid.py predict --checkpoint_path /mnt/workspace1/data/synthefy_data/training_logs/synthetic_tabular_small/grid_tabicl/synthetic_icl_grid/checkpoints/checkpoint_step_300_val_loss_0.9683.ckpt --base_path /workspace/data/synthetic_data/icl_match_series_synthetic_csv/ --results_dir_path /workspace/data/eval_results/icl_synthetic_series_csv/grid_testing/ --plot
