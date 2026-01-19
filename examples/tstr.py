import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict

import h5py
import lightning as L
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    multilabel_confusion_matrix,
    r2_score,
)
from torch.utils.data import DataLoader

from synthefy_pkg.model.trainers.classifier_trainer import ClassifierTrainer
from synthefy_pkg.model.trainers.regressor_trainer import RegressorTrainer
from synthefy_pkg.utils.basic_utils import (
    ENDC,
    OKBLUE,
    get_num_devices,
    seed_everything,
)
from synthefy_pkg.utils.config_utils import load_yaml_config
from synthefy_pkg.utils.tstr_utils import (
    construct_dataset_paths_by_config,
    convert_h5_to_npy,
    get_classification_indices,
    get_regression_index,
    has_multiple_labels,
)

SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")
assert SYNTHEFY_PACKAGE_BASE is not None
load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))
SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE", "")

COMPILE = False
DEFAULT_MAX_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4


class TSTRDataset(torch.utils.data.Dataset):  # type: ignore[attr-defined]
    def __init__(
        self,
        config: DictConfig,
        split: str = "train",
    ) -> None:
        seed_everything(config["tstr_config"].get("seed", 42))
        self.horizon = config["dataset_config"]["time_series_length"]
        self.num_channels = config["dataset_config"]["num_channels"]

        self.split = split

        synthetic_or_original = config["tstr_config"][
            "synthetic_or_original_or_custom"
        ].lower()

        (
            self.timeseries_dataset_loc,
            self.discrete_conditions_loc,
            self.continuous_conditions_loc,
            _,
        ) = construct_dataset_paths_by_config(
            config,
            split=self.split,
            synthetic_or_original_or_custom=synthetic_or_original,
        )

        logger.info(
            OKBLUE
            + f"The dataset(s) location: {self.timeseries_dataset_loc}"
            + ENDC
        )
        logger.info(
            OKBLUE
            + f"The discrete labels(s) location: {self.discrete_conditions_loc}"
            + ENDC
        )
        logger.info(
            OKBLUE
            + f"The continuous labels(s) location: {self.continuous_conditions_loc}"
            + ENDC
        )

        self._load_datasets()

        if self.discrete_conditions.shape[0] == 0:
            self.discrete_conditions_exist = False
        else:
            self.discrete_conditions_exist = True
            if len(self.discrete_conditions.shape) == 3:
                assert (
                    self.timeseries_dataset.shape[-1]
                    == self.discrete_conditions.shape[1]
                ), (
                    f"The dimensions of {self.discrete_conditions = } are not correct"
                )
                if (
                    config.tstr_config.classification_or_regression
                    == "classification"
                ):
                    self.multi_label = has_multiple_labels(
                        self.discrete_conditions,
                        config.tstr_config.dataset.classification_indices,
                    )
            else:
                if (
                    config.tstr_config.classification_or_regression
                    == "classification"
                ):
                    self.multi_label = False
        assert self.timeseries_dataset.shape[-1] == self.horizon, (
            "The horizon is not correct"
        )
        assert self.timeseries_dataset.shape[-2] == self.num_channels, (
            "The number of channels is not correct"
        )

        # add continuous label embedding
        # if self.continuous_conditions.shape

    def _load_from_paths(
        self, ts_paths, discrete_cond_paths, continuous_cond_paths
    ):
        """Helper function to load and concatenate datasets."""
        timeseries_list = []
        discrete_conditions_list = []
        continuous_conditions_list = []

        # Convert single paths to lists
        ts_paths = [ts_paths] if isinstance(ts_paths, str) else ts_paths
        discrete_cond_paths = (
            [discrete_cond_paths]
            if isinstance(discrete_cond_paths, str)
            else discrete_cond_paths
        )
        continuous_cond_paths = (
            [continuous_cond_paths]
            if isinstance(continuous_cond_paths, str)
            else continuous_cond_paths
        )
        for ts_path, discrete_cond_path, continuous_cond_path in zip(
            ts_paths, discrete_cond_paths, continuous_cond_paths
        ):
            timeseries_list.append(np.load(ts_path, allow_pickle=True))
            discrete_conditions_list.append(
                np.load(discrete_cond_path, allow_pickle=True)
            )
            continuous_conditions_list.append(
                np.load(continuous_cond_path, allow_pickle=True)
            )

        logger.info(f"Loaded {len(timeseries_list)} datasets.")
        return (
            np.concatenate(timeseries_list, axis=0),
            np.concatenate(discrete_conditions_list, axis=0),
            np.concatenate(continuous_conditions_list, axis=0),
        )

    def _load_datasets(self):
        """Main function to load datasets, handling paths as lists."""
        try:
            (
                self.timeseries_dataset,
                self.discrete_conditions,
                self.continuous_conditions,
            ) = self._load_from_paths(
                self.timeseries_dataset_loc,
                self.discrete_conditions_loc,
                self.continuous_conditions_loc,
            )

            # unconditionally load - used for debugging.
            # for path in self.timeseries_dataset_loc:
            #     convert_pkls_to_npy(os.path.dirname(path), self.split)

        except Exception:
            logger.error(
                ".npy files not available, trying to convert .pkl to .npy"
            )

            # Convert PKL to NPY
            for path in self.timeseries_dataset_loc:
                convert_h5_to_npy(os.path.dirname(path), self.split)

            # Try loading again
            try:
                (
                    self.timeseries_dataset,
                    self.discrete_conditions,
                    self.continuous_conditions,
                ) = self._load_from_paths(
                    self.timeseries_dataset_loc,
                    self.discrete_conditions_loc,
                    self.continuous_conditions_loc,
                )
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                raise e

    def __len__(self):
        return self.timeseries_dataset.shape[0]

    def __getitem__(self, index):
        timeseries_full = self.timeseries_dataset[index]

        discrete_label_embedding = self.discrete_conditions[index]
        continuous_label_embedding = self.continuous_conditions[index]
        return {
            "timeseries_full": timeseries_full,
            "discrete_label_embedding": discrete_label_embedding,
            "continuous_label_embedding": continuous_label_embedding,
        }


class TSTRDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: DictConfig,
        train_dataset: TSTRDataset,
        val_dataset: TSTRDataset,
        test_dataset: TSTRDataset,
        forecasting: bool = False,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.forecasting = forecasting

        self.kwargs = (
            {
                "num_workers": config.tstr_config.num_workers,
                "pin_memory": True,
                "multiprocessing_context": "fork",
                "shuffle": True,  # this is critical to be true since the dataset labels will mostly all be in sequence
            }
            if config.device == "cuda"
            else {"num_workers": config.tstr_config.num_workers}
        )
        self.config = config

    def train_dataloader(self):
        if self.forecasting:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.tstr_config.dataset.batch_size,
                **self.kwargs,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.tstr_config.dataset.batch_size,
                **self.kwargs,
            )

    def val_dataloader(self):
        if self.forecasting:
            return DataLoader(
                self.val_dataset,
                batch_size=self.config.tstr_config.dataset.batch_size,
                **self.kwargs,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.config.tstr_config.dataset.batch_size,
                **self.kwargs,
            )

    def test_dataloader(self):
        if self.forecasting:
            return DataLoader(
                self.test_dataset,
                batch_size=self.config.tstr_config.dataset.batch_size,
                **self.kwargs,
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.config.tstr_config.dataset.batch_size,
                **self.kwargs,
            )


def analyze_dataset_statistics(
    dataset: TSTRDataset,
    config: DictConfig,
    split: str,
) -> None:
    classification_indices = config["dataset"]["classification_indices"]
    if len(dataset.discrete_conditions.shape) == 3:
        # Only use first timestep (multi_label = False by default)
        discrete_conditions = dataset.discrete_conditions[
            :, 0, classification_indices
        ]
    else:
        discrete_conditions = dataset.discrete_conditions[
            :, classification_indices
        ]

    label_to_count = defaultdict(int)
    for num_samples in range(discrete_conditions.shape[0]):
        for label_idx in range(discrete_conditions.shape[-1]):
            is_one = discrete_conditions[num_samples, label_idx]
            if is_one:
                label_to_count[label_idx] += 1

    label_to_count = dict(sorted(label_to_count.items()))
    logger.info(
        OKBLUE + f"Label distribution for {split}: {label_to_count}" + ENDC
    )


class TSTRDataLoader(TSTRDataModule):
    def __init__(self, config: DictConfig):
        logger.info("Loading train dataset")
        train_dataset = TSTRDataset(config, split="train")

        # Try to load validation dataset, fall back to test dataset if not available
        try:
            logger.info("Loading val dataset")
            val_dataset = TSTRDataset(config, split="val")
        except ValueError as e:
            if "No dataset paths found for custom dataset" in str(e):
                logger.warning(
                    "No validation dataset paths found, using test dataset for validation"
                )
                val_dataset = None
            else:
                raise e
        except FileNotFoundError:
            logger.warning(
                "No validation dataset found, using test dataset for validation"
            )
            val_dataset = None

        logger.info("Loading test dataset")
        test_dataset = TSTRDataset(config, split="test")

        # If val_dataset wasn't loaded, use test_dataset
        if val_dataset is None:
            val_dataset = test_dataset

        super().__init__(config, train_dataset, val_dataset, test_dataset)

        # analyze the dataset statistics
        if (
            config["tstr_config"]["classification_or_regression"]
            == "classification"
        ):
            analyze_dataset_statistics(
                train_dataset,
                config["tstr_config"],
                "train",
            )
            analyze_dataset_statistics(
                val_dataset,
                config["tstr_config"],
                "val",
            )
            analyze_dataset_statistics(
                test_dataset,
                config["tstr_config"],
                "test",
            )


def eval_multilabel_classification_results(
    predictions: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate the results of the model for multi-label classification
    predictions: np.ndarray of shape (num_samples, num_classes) with probabilities
    ground_truth: np.ndarray of shape (num_samples, num_classes) with binary values
    returns dict with per-label confusion matrices and basic metrics
    """
    # Convert tensor to numpy if needed and then convert probabilities to binary predictions
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Assert that predictions is now a numpy array
    assert isinstance(predictions, np.ndarray), (
        "Predictions should be a numpy array by now."
    )

    # Verify predictions are probabilities (between 0 and 1)
    if not np.all((predictions >= 0) & (predictions <= 1)):
        logger.warning(
            "Predictions don't appear to be probabilities (not all values between 0-1). "
            "Make sure sigmoid activation is applied before evaluation."
        )

    pred_classes = (predictions > 0.5).astype(int)
    true_classes = ground_truth

    try:
        # Calculate exact match accuracy (all labels must match)
        exact_match_accuracy = np.mean(
            np.all(pred_classes == true_classes, axis=1)
        )
        logger.info(f"\nExact Match Accuracy: {exact_match_accuracy:.4f}")

        # Calculate per-label accuracy
        per_label_accuracy = np.mean(pred_classes == true_classes, axis=0)
        logger.info("\nPer-label Accuracies:")
        for i, acc in enumerate(per_label_accuracy):
            logger.info(f"Class {i}: {acc:.4f}")

        # Calculate multilabel confusion matrix
        # Returns array of shape (n_classes, 2, 2)
        mcm = multilabel_confusion_matrix(true_classes, pred_classes)

        # Convert to list for JSON serialization
        mcm_list = mcm.tolist()

        logger.info("\nMultilabel Confusion Matrices (per class):")
        for i, cm in enumerate(mcm):
            logger.info(
                f"Class {i}: TN: {cm[0, 0]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TP: {cm[1, 1]}"
            )

        # Calculate per-class metrics from confusion matrices
        per_class_metrics = []
        for cm in mcm:
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
            per_class_metrics.append(metrics)

        logger.info(f"Per-class metrics: {per_class_metrics}")
        logger.info(f"Multilabel confusion matrices: {mcm_list}")

        return {
            "multilabel_confusion_matrices": mcm_list,
            "per_class_metrics": per_class_metrics,
            "exact_match_accuracy": float(exact_match_accuracy),
            "per_label_accuracy": per_label_accuracy.tolist(),
        }

    except Exception as e:
        logger.error(f"Error calculating multilabel confusion matrix: {e}")
        return {}


def eval_classification_results(
    predictions: torch.Tensor, ground_truth: torch.Tensor
) -> Dict[str, Any]:
    """
    Evaluate the results of the model
    predictions: 0/1 np.ndarray of shape (num_samples, num_classes)
    ground_truth: 0/1 np.ndarray of shape (num_samples, num_classes)

    returns dict with accuracy, confusion matrix, classification report
    """
    # Convert one-hot encoded arrays to class indices
    pred_classes = np.argmax(predictions, axis=1)
    # true_classes = np.argmax(ground_truth, axis=1)
    true_classes = ground_truth  # already in class indices

    accuracy = 0.0
    try:
        # Calculate accuracy
        accuracy = float(accuracy_score(true_classes, pred_classes))
        logger.info(f"\nAccuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")

    cm = None
    try:
        # Calculate and display confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        cm_list = cm.tolist()  # Convert numpy array to nested list
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
    except Exception as e:
        logger.error(f"Error calculating confusion matrix: {e}")

    report_dict = None
    try:
        # Generate classification report
        report = classification_report(true_classes, pred_classes)
        report_dict = classification_report(
            true_classes, pred_classes, output_dict=True
        )
        logger.info("\nClassification Report:")
        logger.info(f"\n{report}")
    except Exception as e:
        logger.error(f"Error calculating classification report: {e}")

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm_list,
        "classification_report": report_dict,
    }


def eval_regression_results(
    predictions: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate regression results using multiple metrics

    Args:
        predictions: np.ndarray of predicted values of (n_samples, 1) -> from index_of_interest
        ground_truth: np.ndarray of actual values of (n_samples, 1) -> from index_of_interest

    Returns:
        dict containing regression metrics
    """

    # Convert inputs to numpy arrays if they're torch Tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Basic metrics
    mse = round(float(mean_squared_error(ground_truth, predictions)), 3)
    rmse = round(float(np.sqrt(mse)), 3)
    mae = round(float(mean_absolute_error(ground_truth, predictions)), 3)
    r2 = round(float(r2_score(ground_truth, predictions)), 3)

    # Percentage errors
    epsilon = 1e-10  # Small constant to avoid division by zero
    percentage_errors = (
        np.abs((ground_truth - predictions) / (ground_truth + epsilon)) * 100
    )
    mape = round(float(percentage_errors.mean()), 3)
    mdape = round(float(np.median(percentage_errors)), 3)

    # SMAPE
    numerator = np.abs(ground_truth - predictions)
    denominator = np.abs(ground_truth) + np.abs(predictions) + epsilon
    smape_errors = numerator / denominator
    smape = round(float(200 * np.mean(smape_errors)), 3)

    # Log results
    logger.info(f"\nMSE: {mse:.3f}")
    logger.info(f"RMSE: {rmse:.3f}")
    logger.info(f"MAE: {mae:.3f}")
    logger.info(f"MAPE: {mape:.3f}%")
    logger.info(f"MDAPE: {mdape:.3f}%")
    logger.info(f"SMAPE: {smape:.3f}%")
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


class LossLoggerCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        train_acc = trainer.callback_metrics.get("train_accuracy")  # If logged
        if train_loss is not None:
            log_message = (
                f"Epoch {trainer.current_epoch}: Train Loss = {train_loss:.4f}"
            )
            if train_acc is not None:
                log_message += f", Train Accuracy = {train_acc:.4f}"
            logger.info(log_message)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_accuracy")  # If logged
        if val_loss is not None:
            log_message = (
                f"Epoch {trainer.current_epoch}: Val Loss = {val_loss:.4f}"
            )
            if val_acc is not None:
                log_message += f", Val Accuracy = {val_acc:.4f}"
            logger.info(log_message)


def setup_early_stopping(config: DictConfig) -> EarlyStopping:
    """
    Set up early stopping callback based on configuration.

    Args:
        config: Configuration dictionary with training parameters

    Returns:
        EarlyStopping callback
    """

    patience = config.tstr_config.training.get("early_stopping_patience", 10)
    monitor = config.tstr_config.training.get(
        "early_stopping_monitor", "val_loss"
    )
    mode = config.tstr_config.training.get("early_stopping_mode", "min")

    early_stopping = EarlyStopping(
        monitor=monitor, patience=patience, verbose=True, mode=mode
    )
    logger.info(
        f"Added early stopping with patience={patience}, monitor={monitor}, mode={mode}"
    )
    return early_stopping


def setup_model_checkpoint(
    config: DictConfig, checkpoint_dir: str
) -> ModelCheckpoint:
    """
    Set up model checkpoint callback based on configuration.

    Args:
        config: Configuration dictionary with training parameters
        checkpoint_dir: Directory to save checkpoints

    Returns:
        ModelCheckpoint callback
    """
    monitor = config.tstr_config.training.get("checkpoint_monitor", "val_loss")
    mode = config.tstr_config.training.get("checkpoint_mode", "min")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_model",
        save_top_k=config.tstr_config.training.get("save_top_k", 1),
        verbose=True,
        monitor=monitor,
        mode=mode,
    )
    logger.info(
        f"Added model checkpoint saving with monitor={monitor}, mode={mode}"
    )
    return checkpoint_callback


def main(config: DictConfig):
    config["tstr_config"]["dataset"]["dataset_name"] = config["dataset_config"][
        "dataset_name"
    ]

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.tstr_config.classification_or_regression == "classification":
        config["tstr_config"]["dataset"]["classification_indices"] = (
            get_classification_indices(config["tstr_config"]["dataset"])
        )
    elif config.tstr_config.classification_or_regression == "regression":
        config["tstr_config"]["dataset"]["regression_index"] = (
            get_regression_index(config["tstr_config"]["dataset"])
        )
    else:
        raise ValueError(
            f"Invalid value for classification_or_regression: {config.tstr_config.classification_or_regression}"
        )

    logger.info(f"Loaded config from: {args.config}")

    dataloader = TSTRDataLoader(config)

    multi_label = config.tstr_config.dataset.get("multi_label", None)
    if multi_label is not None:
        logger.warning(
            f"Multi-label classification was manually set in the config: {multi_label=}"
        )
    else:
        multi_label = dataloader.train_dataset.multi_label
        logger.warning(
            f"Multi-label classification is defined from dataset: {multi_label=}"
        )
    config.tstr_config.dataset.multi_label = multi_label

    if config.tstr_config.classification_or_regression == "classification":
        model = ClassifierTrainer(config["tstr_config"])
    elif config.tstr_config.classification_or_regression == "regression":
        model = RegressorTrainer(config["tstr_config"])
    else:
        raise ValueError(
            f"Invalid value for classification_or_regression: {config.tstr_config.classification_or_regression}"
        )

    torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(torch.float32)

    # Initialize callbacks + logging:

    if config.tstr_config.get("output_dir", None) is None:
        output_dir = os.path.join(
            config["tstr_config"]["training"].get(
                "log_dir", "/tmp/tstr_training"
            ),
            f"{config['dataset_config']['dataset_name']}-"
            f"{config['tstr_config']['synthetic_or_original_or_custom']}-"
            f"{config['tstr_config']['training'].get('max_epochs', DEFAULT_MAX_EPOCHS)}-"
            f"{config['tstr_config']['training'].get('learning_rate', DEFAULT_LEARNING_RATE)}",
        )
    else:
        output_dir = config.tstr_config.output_dir

    os.makedirs(output_dir, exist_ok=True)

    callbacks: list[Callback] = [LossLoggerCallback()]

    logger.info(f"Output directory: {output_dir}")

    # Set up checkpoint directory as a subdirectory of
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Add early stopping if configured
    early_stopping = setup_early_stopping(config)
    if early_stopping:
        callbacks.append(early_stopping)

    # Add model checkpoint callback if save_model is enabled
    save_model = config.tstr_config.training.get("save_model", False)
    if save_model:
        checkpoint_callback = setup_model_checkpoint(config, checkpoint_dir)
        callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=get_num_devices(config.tstr_config.training.num_devices),
        max_epochs=config.tstr_config.training.max_epochs,
        check_val_every_n_epoch=config.tstr_config.training.check_val_every_n_epoch,
        log_every_n_steps=1,
        default_root_dir=output_dir,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=output_dir),
    )

    # Start training
    trainer.fit(
        model,
        dataloader.train_dataloader(),
        dataloader.val_dataloader(),
    )

    # Save the best model path in results if available
    best_model_path = None
    if save_model and checkpoint_callback.best_model_path:
        best_model_path = checkpoint_callback.best_model_path
        logger.info(f"Best model saved at: {best_model_path}")

        # Load the best model for evaluation
        model_class = (
            ClassifierTrainer
            if config.tstr_config.classification_or_regression
            == "classification"
            else RegressorTrainer
        )
        model = model_class.load_from_checkpoint(
            best_model_path, config=config["tstr_config"]
        )
        logger.info("Loaded best model for evaluation")

    # Run test and get results
    trainer.test(model, dataloader.test_dataloader())
    predictions = model.test_predictions
    ground_truth = model.test_labels

    if config.tstr_config.classification_or_regression == "classification":
        if config.tstr_config.dataset.multi_label:
            d = eval_multilabel_classification_results(
                predictions, ground_truth
            )
        else:
            d = eval_classification_results(predictions, ground_truth)

    elif config.tstr_config.classification_or_regression == "regression":
        d = eval_regression_results(predictions, ground_truth)

    # Add best model path to results if available
    if best_model_path is not None:
        d["best_model_path"] = best_model_path

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(d, f)
    # save the tstr config too
    with open(os.path.join(output_dir, "tstr_config.yaml"), "w") as f:
        yaml.dump(config["tstr_config"], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSTR Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config = DictConfig(config)
    main(config)
