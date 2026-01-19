from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd
import sklearn
import torch
from einops import rearrange, repeat
from packaging import version
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

# Lazy import to avoid circular import with gridicl_forecaster
if TYPE_CHECKING:
    import synthefy_pkg.model.trainers.timeseries_decoder_forecasting_foundation_trainer as tdf
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.architectures.tabicl.icl_inference.process_values import (
    PreprocessingPipeline,
    TransformToNumerical,
)
from synthefy_pkg.model.architectures.tabicl.inference_config import (
    InferenceConfig,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")


class TabICLRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        config: Configuration,
        model_path: str | Path,
        verbose: bool = False,
        use_amp: bool = True,
        inference_config: Optional[InferenceConfig | Dict] = None,
        trainer: Optional[
            tdf.TimeSeriesDecoderForecastingFoundationTrainer
        ] = None,  # used at train time
    ):
        self.config = config
        self.config.dataset_config.using_synthetic_data = False
        self.device = config.device
        # self.model_ = TimeSeriesDecoderForecastingTrainer(config)
        self.model_ = trainer

        self.model_path = model_path
        self._is_fitted = False
        self.verbose = verbose
        self.use_amp = use_amp
        self.inference_config = inference_config

        self._load_model()

    def _load_model(self):
        # Remove decoder from state dict?
        if self.model_ is None:
            self.model_ = tdf.TimeSeriesDecoderForecastingFoundationTrainer.load_from_checkpoint(
                self.model_path, config=self.config
            )
        # Discard logs
        self.model_.log_dir = "/tmp"  # type: ignore
        self.model_.eval()
        self.model_.to(self.device)

    def fit(self, X, y):
        """
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        """
        # Transform input features
        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        self.X_preprocessor_ = PreprocessingPipeline()
        X = self.X_encoder_.fit_transform(pd.DataFrame(X))
        X = self.X_preprocessor_.fit_transform(X)

        self.y_preprocessor_ = PreprocessingPipeline()
        # Expand and contract to be compatible with PreprocessingPipeline
        y = self.y_preprocessor_.fit_transform(np.expand_dims(y, axis=1))
        y = np.squeeze(y)

        if OLD_SKLEARN:
            # Workaround for compatibility with scikit-learn prior to v1.6
            X = check_array(X, dtype=None, force_all_finite=False)  # type: ignore
            y = check_array(
                y,
                dtype=None,  # type: ignore
                force_all_finite=False,
                ensure_2d=False,
            )
        else:
            X = check_array(X, dtype=None, force_all_finite=False)  # type: ignore
            y = check_array(
                y,
                dtype=None,  # type: ignore
                force_all_finite=False,
                ensure_2d=False,
            )

        self._load_model()

        # Inference configuration
        init_config = {
            "COL_CONFIG": {
                "device": self.device,
                "use_amp": self.use_amp,
                "verbose": self.verbose,
            },
            "ROW_CONFIG": {
                "device": self.device,
                "use_amp": self.use_amp,
                "verbose": self.verbose,
            },
            "ICL_CONFIG": {
                "device": self.device,
                "use_amp": self.use_amp,
                "verbose": self.verbose,
            },
        }
        # If None, default settings in InferenceConfig
        if self.inference_config is None:
            self.inference_config_ = InferenceConfig()
            self.inference_config_.update_from_dict(init_config)
        # If dict, update default settings
        elif isinstance(self.inference_config, dict):
            self.inference_config_ = InferenceConfig()
            for key, value in self.inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            self.inference_config_.update_from_dict(init_config)
        # If InferenceConfig, use as is
        else:
            self.inference_config_ = self.inference_config
        self._is_fitted = True

        # Assumes y is already a numerical features
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        # Check if y is numerical
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(
                "Target values must be numerical. Got dtype: %s" % y.dtype
            )
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X, y) -> np.ndarray:
        """
        X: np.ndarray, shape (n_test_samples, n_features)
        y: np.ndarray, shape (n_test_samples,)
        Returns:
            np.ndarray, shape (n_test_samples,)
        """
        # Transform input features
        assert isinstance(
            self.model_, tdf.TimeSeriesDecoderForecastingFoundationTrainer
        ), "Model not loaded"
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        X = self.X_encoder_.transform(pd.DataFrame(X))
        X = self.X_preprocessor_.transform(X)
        y = self.y_preprocessor_.transform(np.expand_dims(y, axis=1))
        y = np.squeeze(y)

        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[1] == self.X_.shape[1], (
            "X for fitting and prediction must have the same number of features"
        )

        # Prepare input for inference - create full sequence with features + targets
        # Combine: train_X + train_y, test_X + test_y(zeros) -> (seq_len, n_features + 1)
        train_data = np.column_stack([self.X_, self.y_])

        # No peeking target
        test_data = np.column_stack([X, np.zeros(X.shape[0])])
        values = np.concatenate([train_data, test_data], axis=0)

        values = rearrange(values, "seq_len n_features -> 1 n_features seq_len")

        input = {
            "train_sizes": torch.tensor(
                [self.X_.shape[0]], dtype=torch.int32
            ).to(self.device),
            "values": torch.tensor(values, dtype=torch.float32).to(self.device),
        }

        _, res = self.model_(input)

        # Extract only the test predictions (last n_test samples)
        if len(res["prediction"].shape) == 3:
            test_logits = res["prediction"][
                0, -len(y) :
            ]  # Remove batch dim and get test portion
        else:
            test_logits = res[
                "prediction"
            ]  # Remove batch dim and get test portion

        # Converts into a regression using bucket means
        test_predictions = (
            self.model_.decoder_model.distribution.mean(test_logits)
            .to("cpu")
            .numpy()
        )

        test_predictions = self.y_preprocessor_.inverse_transform(
            np.expand_dims(test_predictions, axis=1)
        )
        test_predictions = np.squeeze(test_predictions)

        return test_predictions


class TabICLGridRegressor(TabICLRegressor):
    def __init__(
        self,
        config: Configuration,
        model_path: str | Path,
        verbose: bool = False,
        use_amp: bool = True,
        inference_config: Optional[InferenceConfig | Dict] = None,
        trainer: Optional[
            tdf.TimeSeriesDecoderForecastingFoundationTrainer
        ] = None,  # used at train time
    ):
        super().__init__(
            config, model_path, verbose, use_amp, inference_config, trainer
        )

    def predict(self, X, y, predict_mask: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, shape (n_test_samples, n_features)
        y: np.ndarray, shape (n_test_samples,)
        predict_mask: np.ndarray, shape (n_test_samples, n_features + 1)
        Returns:
            np.ndarray, shape (n_test_samples,)
        """
        # Transform input features
        assert isinstance(
            self.model_, tdf.TimeSeriesDecoderForecastingFoundationTrainer
        ), "Model not loaded"
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        X = self.X_encoder_.transform(pd.DataFrame(X))
        X = self.X_preprocessor_.transform(X)
        y = self.y_preprocessor_.transform(np.expand_dims(y, axis=1))

        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[1] == self.X_.shape[1], (
            "X for fitting and prediction must have the same number of features"
        )

        # Prepare input for inference - create full sequence with features + targets
        # Combine: train_X + train_y, test_X + test_y(zeros) -> (seq_len, n_features + 1)
        train_data = np.column_stack([self.X_, self.y_])

        # No peeking target
        test_data = np.column_stack([X, y])
        test_data[predict_mask] = -100
        values = np.concatenate([train_data, test_data], axis=0)
        target_mask = np.concatenate(
            [
                np.zeros((self.X_.shape[0], self.X_.shape[1] + 1), dtype=bool),
                predict_mask,
            ],
            axis=0,
        )

        values = rearrange(values, "seq_len n_features -> 1 n_features seq_len")

        input = {
            "train_sizes": torch.tensor(
                [self.X_.shape[0]], dtype=torch.int32
            ).to(self.device),
            "values": torch.tensor(values, dtype=torch.float32).to(self.device),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool)
            .unsqueeze(0)
            .transpose(2, 1)
            .to(self.device),
        }

        _, res = self.model_(input)
        # Converts into a regression using bucket means
        test_predictions = (
            self.model_.decoder_model.distribution.mean(
                res["prediction_multivariate"]
            )
            .to("cpu")
            .numpy()
        )

        test_predictions_expanded = self.recover_predictions(
            test_data, test_predictions, predict_mask
        )

        # Apply inverse transform to all columns
        # Features (all columns except the last one)
        feature_columns = test_predictions_expanded[:, :-1]
        feature_columns_inverse = self.X_preprocessor_.inverse_transform(
            feature_columns
        )

        # Target column (last column)
        target_column = test_predictions_expanded[:, -1:]
        target_column_inverse = self.y_preprocessor_.inverse_transform(
            target_column
        )

        # Combine the inverse transformed features and target
        test_predictions_expanded = np.column_stack(
            [feature_columns_inverse, target_column_inverse]
        )

        return test_predictions_expanded

    def recover_predictions(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        predict_mask: np.ndarray,
    ) -> np.ndarray:
        """
        targets: np.ndarray, shape (n_test_rows, n_features + 1)
        predictions: np.ndarray, shape (number of masked indices,)
        predict_mask: np.ndarray, shape (n_test_rows, n_features + 1)

        Determines which indices in the targets array are masked and replaces them with the predictions
        Returns:
            np.ndarray, shape (n_test_rows, n_features + 1)
        """
        # Create a copy of targets to avoid modifying the original
        result = targets.copy()

        # Use advanced indexing to replace masked values with predictions
        result[predict_mask] = predictions[: np.sum(predict_mask)]

        return result
