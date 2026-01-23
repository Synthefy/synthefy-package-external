"""
Demo Synthesis Service - Lightweight wrapper around SynthesisExperiment.

Handles synthesis model inference for the demo application.
Uses DataPreprocessor for proper scaling and encoding like the production service.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from models import (
    DataFrameModel,
    DatasetName,
    ModelType,
    OneTimeSeries,
    RequiredColumns,
    TaskType,
)
from services.config_loader import ConfigLoader, get_config_loader

# Denoiser names for different model types
DENOISER_NAMES: Dict[str, str] = {
    "flexible": "flexible_patched_diffusion_transformer",
    "standard": "patched_diffusion_transformer",
}

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE",None)
if not SYNTHEFY_DATASETS_BASE:
    raise ValueError("SYNTHEFY_DATASETS_BASE environment variable must be set.")

# Model paths organized by dataset and model type
# Structure: DATASET_MODEL_PATHS[dataset_name][model_type] = path
DATASET_MODEL_PATHS: Dict[str, Dict[str, str]] = {
    "oura": {
        "standard": os.path.expanduser(
            f"{SYNTHEFY_DATASETS_BASE}/training_logs/oura/Time_Series_Diffusion_Training/"
            "synthesis_ppg/checkpoints/best_model.ckpt"
        ),
        "flexible": os.path.expanduser(
            f"{SYNTHEFY_DATASETS_BASE}/training_logs/oura/Time_Series_Diffusion_Training/"
            "synthesis_oura_flexible/checkpoints/best_model.ckpt"
        ),
    },
    "oura_subset": {
        "standard": os.path.expanduser(
            f"{SYNTHEFY_DATASETS_BASE}/training_logs/oura_subset/Time_Series_Diffusion_Training/"
            "synthesis_ppg/checkpoints/best_model.ckpt"
        ),
        "flexible": os.path.expanduser(
            f"{SYNTHEFY_DATASETS_BASE}/training_logs/oura_subset/Time_Series_Diffusion_Training/"
            "synthesis_oura_subset_flexible/checkpoints/best_model.ckpt"
        ),
    },
    "ppg": {
        "standard": os.path.expanduser(
            f"{SYNTHEFY_DATASETS_BASE}/training_logs/ppg/Time_Series_Diffusion_Training/"
            "synthesis_ppg/checkpoints/best_model.ckpt"
        ),
        "flexible": os.path.expanduser(
            f"{SYNTHEFY_DATASETS_BASE}/training_logs/ppg/Time_Series_Diffusion_Training/"
            "synthesis_ppg/checkpoints/best_model.ckpt"  # Same for ppg
        ),
    },
}


def get_model_path(dataset_name: str, model_type: str) -> str:
    """Get model checkpoint path for a dataset and model type.

    Args:
        dataset_name: Name of the dataset (oura, oura_subset, ppg)
        model_type: Model type (standard or flexible)

    Returns:
        Path to model checkpoint
    """
    if dataset_name in DATASET_MODEL_PATHS:
        return DATASET_MODEL_PATHS[dataset_name].get(
            model_type,
            DATASET_MODEL_PATHS[dataset_name][
                "flexible"
            ],  # Default to flexible
        )
    # Fallback
    return os.path.expanduser(
        f"{SYNTHEFY_DATASETS_BASE}/training_logs/synthesis/checkpoints/best_model.ckpt"
    )


class DemoSynthesisService:
    """Lightweight synthesis service for demo purposes.

    Wraps SynthesisExperiment for simple inference.
    Uses DataPreprocessor for proper scaling and encoding.
    """

    def __init__(
        self,
        dataset_name: str,
        model_type: str = "flexible",
        task_type: str = "synthesis",
        model_path: Optional[str] = None,
    ):
        """Initialize the synthesis service.

        Args:
            dataset_name: Name of the dataset for config loading
            model_type: Model type ('standard' or 'flexible')
            task_type: Task type ('synthesis' or 'forecast')
            model_path: Path to model checkpoint (optional, overrides default)
        """
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.task_type = task_type
        self.config_loader = get_config_loader(dataset_name)

        # Model path priority: explicit arg > env var > dataset+type default
        self.model_path = (
            model_path
            or os.getenv("SYNTHESIS_MODEL_PATH")
            or get_model_path(dataset_name, model_type)
        )

        logger.info(
            f"Initializing DemoSynthesisService for dataset: {dataset_name}, "
            f"model_type: {model_type}, task_type: {task_type}"
        )
        logger.info(f"Model path: {self.model_path}")

        # Lazy load the experiment to avoid import issues if synthefy_pkg not available
        self._experiment = None

        # Load scalers and encoders
        self._saved_scalers: Optional[Dict[str, Any]] = None
        self._encoders: Optional[Dict[str, Any]] = None

    def _load_scalers_and_encoders(self) -> None:
        """Load saved scalers and encoders for preprocessing."""
        if self._saved_scalers is not None:
            return  # Already loaded

        from synthefy_pkg.utils.scaling_utils import (
            load_continuous_scalers,
            load_discrete_encoders,
            load_timeseries_scalers,
        )

        logger.info(
            f"Loading scalers and encoders for dataset: {self.dataset_name}"
        )
        self._saved_scalers = {
            "timeseries": load_timeseries_scalers(self.dataset_name),
            "continuous": load_continuous_scalers(self.dataset_name),
        }
        self._encoders = load_discrete_encoders(self.dataset_name)
        logger.info(
            f"Loaded encoders: {list(self._encoders.keys()) if self._encoders else 'none'}"
        )

    @property
    def experiment(self):
        """Lazily load the SynthesisExperiment with model-type-specific config."""
        if self._experiment is None:
            from synthefy_pkg.experiments.synthesis_experiment import (
                SynthesisExperiment,
            )

            # Load the synthesis config
            config_path = self.config_loader.synthesis_config_path
            logger.info(f"Loading synthesis config from: {config_path}")

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Update denoiser_name based on model_type
            denoiser_name = DENOISER_NAMES.get(
                self.model_type, DENOISER_NAMES["flexible"]
            )
            if "denoiser_config" not in config:
                config["denoiser_config"] = {}
            config["denoiser_config"]["denoiser_name"] = denoiser_name

            # Randomize inference_seed to get different results on each run
            # Use a random seed instead of fixed seed for stochastic synthesis
            import random
            import time

            random_seed = int(time.time() * 1000) % 100000 + random.randint(
                0, 10000
            )
            if "execution_config" not in config:
                config["execution_config"] = {}
            config["execution_config"]["inference_seed"] = random_seed
            logger.info(f"Set random inference_seed: {random_seed}")

            logger.info(
                f"Using denoiser: {denoiser_name} for model_type: {self.model_type}"
            )
            logger.info(f"Using task_type: {self.task_type}")

            # Pass the modified config dict to SynthesisExperiment with task type
            # forecast_length will be set when generate() is called
            self._experiment = SynthesisExperiment(
                config, synthesis_task=self.task_type, forecast_length=96  # Default, will be overridden
            )
        return self._experiment

    def _prepare_conditions(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare continuous and discrete conditions from DataFrame using DataPreprocessor.

        Args:
            df: Input DataFrame with metadata columns

        Returns:
            Tuple of (continuous_conditions, discrete_conditions, original_discrete) as numpy arrays
        """
        from synthefy_pkg.preprocessing.preprocess import DataPreprocessor

        # Load scalers/encoders if not already loaded
        self._load_scalers_and_encoders()

        preprocess_config_path = self.config_loader.preprocess_config_path

        # Use DataPreprocessor like the production service
        preprocessor = DataPreprocessor(preprocess_config_path)

        # We don't need group labels for single window processing
        if not preprocessor.use_label_col_as_discrete_metadata:
            preprocessor.group_labels_cols = []

        # We don't need timeseries for metadata preprocessing
        preprocessor.timeseries_cols = []
        preprocessor.timeseries_scalers_info = {}

        # Process the data with saved scalers and encoders
        try:
            preprocessor.process_data(
                df,
                saved_scalers=self._saved_scalers or {},
                saved_encoders=self._encoders or {},
                save_files_on=False,
            )
        except Exception as e:
            logger.error(f"Preprocessing failed: {type(e).__name__}: {str(e)}")
            raise

        # Extract the processed windows
        continuous_conditions = preprocessor.windows_data_dict["continuous"][
            "windows"
        ]
        discrete_conditions = preprocessor.windows_data_dict["discrete"][
            "windows"
        ]
        original_discrete = preprocessor.windows_data_dict["original_discrete"][
            "windows"
        ]

        logger.debug(
            f"Continuous conditions shape: {continuous_conditions.shape}"
        )
        logger.debug(f"Discrete conditions shape: {discrete_conditions.shape}")
        logger.debug(f"Original discrete shape: {original_discrete.shape}")

        return (
            continuous_conditions.astype(np.float32),
            discrete_conditions.astype(np.float32),
            original_discrete,
        )

    def _generate_batch(
        self,
        continuous_conditions: np.ndarray,
        discrete_conditions: np.ndarray,
        timeseries: Optional[np.ndarray] = None,
        forecast_length: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic timeseries for a batch of conditions.

        This is more efficient than calling generate_one_synthetic_window N times
        because the model is loaded once and inference runs once with batch_size=N.

        Args:
            continuous_conditions: (batch_size, num_continuous, window_size)
            discrete_conditions: (batch_size, num_discrete, window_size)
            timeseries: Optional (batch_size, num_channels, window_size) for forecast

        Returns:
            Synthetic timeseries: (batch_size, num_channels, window_size)
        """
        from functools import partial

        import torch

        from synthefy_pkg.model.trainers.diffusion_model import (
            get_synthesis_via_diffusion,
        )
        from synthefy_pkg.utils.synthesis_utils import (
            forecast_via_diffusion,
        )

        # Setup inference once (loads model if not already loaded)
        experiment = self.experiment

        # Only setup if synthesizer not already loaded
        if experiment.synthesizer is None:
            logger.info("Loading model for batched inference...")
            experiment._setup_inference(self.model_path)
        else:
            logger.debug("Reusing already-loaded model for batched inference")

        if experiment.synthesizer is None:
            raise RuntimeError(
                "Synthesizer not loaded after _setup_inference()"
            )

        # Select synthesis function based on task type
        if self.task_type == "forecast":
            # Use forecast_length from parameter if provided, otherwise use experiment default
            effective_forecast_length = forecast_length if forecast_length is not None else experiment.forecast_length
            synthesis_function = partial(
                forecast_via_diffusion,
                forecast_length=effective_forecast_length,
            )
            logger.info(f"Using forecast_length={effective_forecast_length} for forecast task")
        else:
            synthesis_function = (
                get_synthesis_via_diffusion
            )

        # Convert to tensors
        continuous_tensor = torch.tensor(
            continuous_conditions, dtype=torch.float32
        )
        discrete_tensor = torch.tensor(discrete_conditions, dtype=torch.float32)

        device = experiment.synthesizer.config.device
        continuous_tensor = continuous_tensor.to(device)
        discrete_tensor = discrete_tensor.to(device)

        # Handle timeseries (for forecast) or create zeros
        batch_size = continuous_conditions.shape[0]
        if timeseries is not None:
            timeseries_tensor = torch.tensor(timeseries, dtype=torch.float32)
        else:
            timeseries_tensor = torch.zeros(
                batch_size,
                experiment.configuration.dataset_config.num_channels,
                experiment.configuration.dataset_config.time_series_length,
            )
        timeseries_tensor = timeseries_tensor.to(device)

        # Prepare kwargs for constraints if needed
        kwargs = {}
        if experiment.configuration.dataset_config.use_constraints:
            kwargs["dataset_config"] = experiment.configuration.dataset_config

        # Run batched synthesis
        logger.info(
            f"Running batched synthesis on device={device}, batch_size={batch_size}"
        )
        dataset_dict = synthesis_function(
            batch={
                "continuous_label_embedding": continuous_tensor,
                "discrete_label_embedding": discrete_tensor,
                "timeseries_full": timeseries_tensor,
            },
            synthesizer=experiment.synthesizer,
            **kwargs,
        )

        # Return numpy array
        result = dataset_dict["timeseries"]
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()

        return result

    def generate(
        self,
        data: DataFrameModel,
        num_samples: int = 2,
        ground_truth_prefix_length: int = 0,
        forecast_length: Optional[int] = None,
    ) -> List[OneTimeSeries]:
        """Generate synthetic time series from input data.

        Args:
            data: Input data as DataFrameModel (1 window)
            num_samples: Number of synthesis runs to average (default 2)
            ground_truth_prefix_length: For synthesis task, replace first N points
                with ground truth values (0 = disabled)
            forecast_length: For forecast task, number of time steps to forecast (default: 96)

        Returns:
            List of OneTimeSeries with synthetic values
        """
        from synthefy_pkg.utils.scaling_utils import transform_using_scaler

        df = data.to_dataframe()
        window_size = self.config_loader.get_window_size()

        # Handle NaN values in input: ffill -> bfill -> fillna(0)
        nan_count_before = df.isna().sum().sum()
        if nan_count_before > 0:
            nan_cols = df.columns[df.isna().any()].tolist()
            logger.warning(
                f"Input data has {nan_count_before} NaN values in columns: {nan_cols}. "
                "Applying ffill().bfill().fillna(0)"
            )
            df = df.ffill().bfill().fillna(0)

        # Validate data length
        if len(df) != window_size:
            raise ValueError(
                f"Input data has {len(df)} rows but expected {window_size} (window_size)"
            )

        # Prepare conditions using DataPreprocessor (proper scaling and encoding)
        continuous_conditions, discrete_conditions, original_discrete = (
            self._prepare_conditions(df)
        )

        logger.info("Running synthesis model inference...")
        logger.info(
            f"Continuous shape: {continuous_conditions.shape}, Discrete shape: {discrete_conditions.shape}"
        )

        # Prepare timeseries for forecast task
        timeseries_scaled = None
        if self.task_type == "forecast":
            # Get timeseries columns and scale them
            required_cols = self.config_loader.get_required_columns()
            timeseries_cols = required_cols.timeseries
            timeseries_data = df[
                timeseries_cols
            ].values.T  # (num_channels, window_size)
            timeseries_data = timeseries_data[
                np.newaxis, :, :
            ]  # (1, num_channels, window_size)

            # Scale the timeseries
            timeseries_scaled = transform_using_scaler(
                windows=timeseries_data,
                timeseries_or_continuous="timeseries",
                original_discrete_windows=original_discrete,
                dataset_name=self.dataset_name,
                inverse_transform=False,  # Forward transform (scaling)
            )
            logger.info(
                f"Scaled timeseries for forecast, shape: {timeseries_scaled.shape}"
            )

        # Run synthesis with batching for efficiency
        # Instead of running N times sequentially (loading model each time),
        # we batch the conditions and run once with batch_size=N
        logger.info(
            f"Running batched synthesis with {num_samples} sample(s)..."
        )

        # Tile conditions to create batch: (1, ...) -> (num_samples, ...)
        continuous_batch = np.tile(continuous_conditions, (num_samples, 1, 1))
        discrete_batch = np.tile(discrete_conditions, (num_samples, 1, 1))

        if timeseries_scaled is not None:
            timeseries_batch = np.tile(timeseries_scaled, (num_samples, 1, 1))
        else:
            timeseries_batch = None

        logger.info(
            f"Batched conditions: continuous={continuous_batch.shape}, discrete={discrete_batch.shape}"
        )

        # Run batched inference (model loaded once, inference runs once with batch)
        synthetic_batch = self._generate_batch(
            continuous_conditions=continuous_batch,
            discrete_conditions=discrete_batch,
            timeseries=timeseries_batch,
            forecast_length=forecast_length,
        )
        # synthetic_batch shape: (num_samples, num_channels, window_size)
        logger.info(f"Batched synthesis output shape: {synthetic_batch.shape}")

        # Compare samples to verify they're different (stochastic)
        if num_samples >= 2:
            output_1, output_2 = synthetic_batch[0:1], synthetic_batch[1:2]
            if np.array_equal(output_1, output_2):
                logger.warning(
                    "WARNING: First two samples in batch are identical! This is unexpected for stochastic synthesis."
                )
            else:
                diff = np.abs(output_1 - output_2)
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)
                logger.info(
                    f"Sample diversity check: mean_abs_diff={mean_diff:.6f}, max_abs_diff={max_diff:.6f}"
                )

        # Average across batch dimension: (num_samples, C, T) -> (1, C, T)
        synthetic_output = np.mean(synthetic_batch, axis=0, keepdims=True)
        logger.info(
            f"Averaged {num_samples} sample(s), output shape: {synthetic_output.shape}"
        )

        # Inverse transform (unscale) the synthetic output
        synthetic_unscaled = transform_using_scaler(
            windows=synthetic_output,
            timeseries_or_continuous="timeseries",
            original_discrete_windows=original_discrete,
            dataset_name=self.dataset_name,
            inverse_transform=True,
        )
        logger.info("Unscaled synthetic output")

        # Convert to list of OneTimeSeries
        required_cols = self.config_loader.get_required_columns()
        timeseries_cols = required_cols.timeseries

        # Apply ground truth prefix if enabled (synthesis task only)
        if ground_truth_prefix_length > 0 and self.task_type == "synthesis":
            prefix_len = min(ground_truth_prefix_length, window_size)
            logger.info(
                f"Applying ground truth prefix: first {prefix_len} points will match input data"
            )
            for i, col_name in enumerate(timeseries_cols):
                if col_name in df.columns:
                    ground_truth_values = df[col_name].values[:prefix_len]
                    # Log before/after for debugging
                    logger.debug(
                        f"{col_name}: synthetic[0:3] before={synthetic_unscaled[0, i, :3]}, "
                        f"ground_truth[0:3]={ground_truth_values[:3]}"
                    )
                    synthetic_unscaled[0, i, :prefix_len] = ground_truth_values
                    logger.debug(
                        f"{col_name}: synthetic[0:3] after={synthetic_unscaled[0, i, :3]}"
                    )
            logger.info(
                f"Ground truth prefix applied: first {prefix_len} points now match input"
            )

        result: List[OneTimeSeries] = []
        for i, col_name in enumerate(timeseries_cols):
            raw_values = synthetic_unscaled[0, i, :].tolist()
            # Find and log NaN/Inf indices
            nan_indices = [
                idx
                for idx, v in enumerate(raw_values)
                if v is None or np.isnan(v) or np.isinf(v)
            ]
            if nan_indices:
                logger.warning(
                    f"Synthetic {col_name}: {len(nan_indices)} NaN/Inf values at indices {nan_indices[:10]}"
                    + (
                        f"... (+{len(nan_indices) - 10} more)"
                        if len(nan_indices) > 10
                        else ""
                    )
                )
            # Convert NaN/Inf to None for JSON serialization
            values = [
                None if (v is None or np.isnan(v) or np.isinf(v)) else float(v)
                for v in raw_values
            ]
            result.append(
                OneTimeSeries(
                    name=f"{col_name}_synthetic",
                    values=values,
                )
            )

        return result

    def get_original_timeseries(
        self, data: DataFrameModel
    ) -> List[OneTimeSeries]:
        """Extract original timeseries from input data.

        Args:
            data: Input data as DataFrameModel

        Returns:
            List of OneTimeSeries with original values
        """
        df = data.to_dataframe()
        required_cols = self.config_loader.get_required_columns()
        timeseries_cols = required_cols.timeseries

        result: List[OneTimeSeries] = []
        for col_name in timeseries_cols:
            if col_name in df.columns:
                raw_values = df[col_name].tolist()
                # Find and log NaN/Inf indices
                nan_indices = [
                    idx
                    for idx, v in enumerate(raw_values)
                    if v is None
                    or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                ]
                if nan_indices:
                    logger.warning(
                        f"Original {col_name}: {len(nan_indices)} NaN/Inf values at indices {nan_indices[:10]}"
                        + (
                            f"... (+{len(nan_indices) - 10} more)"
                            if len(nan_indices) > 10
                            else ""
                        )
                    )
                # Convert NaN/Inf to None for JSON serialization
                values = [
                    None
                    if (
                        v is None
                        or (
                            isinstance(v, float)
                            and (np.isnan(v) or np.isinf(v))
                        )
                    )
                    else v
                    for v in raw_values
                ]
                result.append(
                    OneTimeSeries(
                        name=col_name,
                        values=values,
                    )
                )

        return result

    def cleanup(self):
        """Clean up model resources."""
        if self._experiment is not None:
            self._experiment.cleanup()
            self._experiment = None


# Cache for service instances - keyed by (dataset_name, model_type, task_type)
_synthesis_services: Dict[Tuple[str, str, str], DemoSynthesisService] = {}


def get_synthesis_service(
    dataset_name: str,
    model_type: str = "flexible",
    task_type: str = "synthesis",
) -> DemoSynthesisService:
    """Get a DemoSynthesisService instance (cached by dataset + model type + task type).

    Args:
        dataset_name: Name of the dataset
        model_type: Model type ('standard' or 'flexible')
        task_type: Task type ('synthesis' or 'forecast')

    Returns:
        DemoSynthesisService instance
    """
    cache_key = (dataset_name, model_type, task_type)
    if cache_key not in _synthesis_services:
        _synthesis_services[cache_key] = DemoSynthesisService(
            dataset_name, model_type=model_type, task_type=task_type
        )
    return _synthesis_services[cache_key]


class MockSynthesisService:
    """Mock synthesis service that generates realistic-looking synthetic data.

    Used for UI development when the real model is not available.
    """

    def __init__(self, dataset_name: str):
        """Initialize the mock synthesis service.

        Args:
            dataset_name: Name of the dataset for config loading
        """
        self.dataset_name = dataset_name
        self.config_loader = get_config_loader(dataset_name)
        logger.info(
            f"Initialized MockSynthesisService for dataset: {dataset_name}"
        )

    def generate(self, data: DataFrameModel) -> List[OneTimeSeries]:
        """Generate mock synthetic time series from input data.

        Creates synthetic data that:
        - Follows similar patterns to the original
        - Has slight variations (smoothing, noise, trend shifts)
        - Looks realistic for demo purposes
        - Runs synthesis twice and takes the mean

        Args:
            data: Input data as DataFrameModel (1 window)

        Returns:
            List of OneTimeSeries with mock synthetic values (averaged from 2 runs)
        """
        df = data.to_dataframe()
        required_cols = self.config_loader.get_required_columns()
        timeseries_cols = required_cols.timeseries

        result: List[OneTimeSeries] = []

        for col_name in timeseries_cols:
            if col_name not in df.columns:
                continue

            original_values = np.array(df[col_name].tolist(), dtype=np.float64)

            # Handle NaN values
            nan_mask = np.isnan(original_values)
            valid_values = original_values[~nan_mask]

            if len(valid_values) == 0:
                # All NaN - just return original with synthetic suffix
                result.append(
                    OneTimeSeries(
                        name=f"{col_name}_synthetic",
                        values=original_values.tolist(),
                    )
                )
                continue

            # Generate synthetic values twice and take the mean
            logger.info(f"Running mock synthesis twice for {col_name}...")
            synthetic_values_1 = self._generate_synthetic_channel(
                original_values, col_name
            )
            synthetic_values_2 = self._generate_synthetic_channel(
                original_values, col_name
            )

            # Compare the two runs (for mock, they should be identical since it's deterministic)
            diff = np.abs(synthetic_values_1 - synthetic_values_2)
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            if mean_diff < 1e-10:
                logger.info(
                    f"Mock synthesis runs for {col_name} are identical (expected for deterministic mock)"
                )
            else:
                logger.warning(
                    f"Mock synthesis runs for {col_name} differ: "
                    f"mean_abs_diff={mean_diff:.6f}, max_abs_diff={max_diff:.6f}"
                )

            synthetic_values = np.mean(
                [synthetic_values_1, synthetic_values_2], axis=0
            )

            result.append(
                OneTimeSeries(
                    name=f"{col_name}_synthetic",
                    values=synthetic_values.tolist(),
                )
            )

        logger.info(
            f"Generated {len(result)} mock synthetic timeseries (averaged from 2 runs)"
        )
        return result

    def _generate_synthetic_channel(
        self, original: np.ndarray, col_name: str
    ) -> np.ndarray:
        """Generate a single synthetic channel by adding 10% to original values.

        Simple transformation for easy visual comparison during UI development.

        Args:
            original: Original time series values
            col_name: Column name (for logging)

        Returns:
            Synthetic time series as numpy array (original + 10%)
        """
        # Simply add 10% to make difference clearly visible
        synthetic = original * 1.10

        logger.debug(
            f"Generated synthetic for {col_name}: +10% of original values"
        )

        return synthetic

    def get_original_timeseries(
        self, data: DataFrameModel
    ) -> List[OneTimeSeries]:
        """Extract original timeseries from input data.

        Args:
            data: Input data as DataFrameModel

        Returns:
            List of OneTimeSeries with original values
        """
        df = data.to_dataframe()
        required_cols = self.config_loader.get_required_columns()
        timeseries_cols = required_cols.timeseries

        result: List[OneTimeSeries] = []
        for col_name in timeseries_cols:
            if col_name in df.columns:
                values = df[col_name].tolist()
                result.append(
                    OneTimeSeries(
                        name=col_name,
                        values=values,
                    )
                )

        return result


# Cache for mock service instances
_mock_synthesis_services: Dict[str, MockSynthesisService] = {}


def get_mock_synthesis_service(
    dataset_name: str, task_type: str = "synthesis"
) -> "MockSynthesisService | MockForecastService":
    """Get a mock service instance (cached).

    Args:
        dataset_name: Name of the dataset
        task_type: Task type ('synthesis' or 'forecast')

    Returns:
        MockSynthesisService or MockForecastService instance
    """
    if task_type == "forecast":
        return get_mock_forecast_service(dataset_name)

    if dataset_name not in _mock_synthesis_services:
        _mock_synthesis_services[dataset_name] = MockSynthesisService(
            dataset_name
        )
    return _mock_synthesis_services[dataset_name]


class MockForecastService:
    """Mock forecast service that generates realistic-looking forecast data.

    Used for UI development when the real model is not available.
    Generates forecast continuation by extrapolating trends with noise.
    """

    # Default forecast length (should match model config)
    DEFAULT_FORECAST_LENGTH = 96

    def __init__(
        self, dataset_name: str, forecast_length: int = DEFAULT_FORECAST_LENGTH
    ):
        """Initialize the mock forecast service.

        Args:
            dataset_name: Name of the dataset for config loading
            forecast_length: Number of points to forecast (default: 96)
        """
        self.dataset_name = dataset_name
        self.forecast_length = forecast_length
        self.config_loader = get_config_loader(dataset_name)
        logger.info(
            f"Initialized MockForecastService for dataset: {dataset_name}, "
            f"forecast_length: {forecast_length}"
        )

    def generate(
        self,
        data: DataFrameModel,
        num_samples: int = 2,
        ground_truth_prefix_length: int = 0,
        forecast_length: Optional[int] = None,
    ) -> List[OneTimeSeries]:
        """Generate mock forecast time series from input data.

        Creates forecast data that:
        - Keeps the historical portion unchanged (first window_size - forecast_length points)
        - Generates a forecast for the last forecast_length points
        - Uses linear extrapolation with added noise for realistic appearance
        - Different metadata = different forecast (worse when metadata is modified)

        Args:
            data: Input data as DataFrameModel (1 window)
            num_samples: Number of synthesis runs to average (ignored in mock, for API compat)
            ground_truth_prefix_length: Ignored in forecast mode (for API compatibility)

        Returns:
            List of OneTimeSeries with forecast values
        """
        df = data.to_dataframe()
        required_cols = self.config_loader.get_required_columns()
        timeseries_cols = required_cols.timeseries
        window_size = len(df)

        result: List[OneTimeSeries] = []

        # Use forecast_length from parameter if provided, otherwise use instance default
        effective_forecast_length = forecast_length if forecast_length is not None else self.forecast_length

        # Calculate the context length (historical portion)
        context_length = window_size - effective_forecast_length

        # Compute a seed and bias from metadata columns
        # This ensures different metadata = different forecast
        metadata_seed, metadata_bias = self._compute_metadata_hash(
            df, required_cols
        )
        logger.info(
            f"Mock forecast using metadata_seed={metadata_seed}, metadata_bias={metadata_bias:.4f}"
        )

        for col_name in timeseries_cols:
            if col_name not in df.columns:
                continue

            original_values = np.array(df[col_name].tolist(), dtype=np.float64)

            # Handle NaN values
            nan_mask = np.isnan(original_values)
            if np.all(nan_mask):
                # All NaN - return as-is
                result.append(
                    OneTimeSeries(
                        name=f"{col_name}_synthetic",
                        values=original_values.tolist(),
                    )
                )
                continue

            # Generate forecast
            logger.info(
                f"Generating mock forecast for {col_name}: "
                f"context={context_length}, forecast={effective_forecast_length}"
            )

            synthetic_values = self._generate_forecast(
                original_values,
                context_length,
                effective_forecast_length,
                col_name,
                metadata_seed,
                metadata_bias,
            )

            result.append(
                OneTimeSeries(
                    name=f"{col_name}_synthetic",
                    values=synthetic_values.tolist(),
                )
            )

        logger.info(f"Generated {len(result)} mock forecast timeseries")
        return result

    def _compute_metadata_hash(
        self, df: pd.DataFrame, required_cols: RequiredColumns
    ) -> Tuple[int, float]:
        """Compute a hash and bias from metadata columns.

        The hash is used as a random seed so different metadata produces different forecasts.
        The bias represents how "modified" the metadata is - higher bias = worse forecast.

        Args:
            df: Input DataFrame
            required_cols: Required columns info

        Returns:
            Tuple of (seed, bias) where:
            - seed: int to use as random seed
            - bias: float [0, 1] representing deviation, higher = more error added
        """
        import hashlib

        # Get metadata columns (continuous + discrete, excluding timeseries)
        metadata_cols = required_cols.continuous + required_cols.discrete

        # Compute hash from metadata values
        hash_input = ""
        total_deviation = 0.0
        num_cols = 0

        for col in metadata_cols:
            if col in df.columns:
                values = np.array(df[col].values, dtype=np.float64)
                # Use mean of column for hashing (handles varying length)
                col_mean = float(np.nanmean(values)) if len(values) > 0 else 0.0
                hash_input += f"{col}:{col_mean:.6f};"

                # Track deviation from "typical" values for bias calculation
                # Assume typical values are around 0.5 for normalized data
                # or use the actual value as an indicator of modification
                if not np.isnan(col_mean):
                    total_deviation += abs(col_mean)
                    num_cols += 1

        # Generate seed from hash
        hash_bytes = hashlib.md5(hash_input.encode()).hexdigest()
        seed = int(hash_bytes[:8], 16) % (2**31)

        # Calculate bias: normalized deviation [0, 1]
        # Higher deviation from baseline = more error in forecast
        avg_deviation = total_deviation / num_cols if num_cols > 0 else 0
        # Scale so small changes produce noticeable but small bias
        bias = min(1.0, avg_deviation * 0.1)

        return seed, bias

    def _generate_forecast(
        self,
        original: np.ndarray,
        context_length: int,
        forecast_length: int,
        col_name: str,
        metadata_seed: int = 42,
        metadata_bias: float = 0.0,
    ) -> np.ndarray:
        """Generate a forecast continuation of the time series.

        The context (historical) portion remains IDENTICAL to the input.
        Only the forecast portion (last forecast_length points) is generated.

        The forecast is based on the ground truth with added noise, making it
        realistic and close to actual values. Different metadata_seed produces
        different forecasts. Higher metadata_bias adds more error.

        Args:
            original: Original time series values (full window)
            context_length: Number of points to use as context (history)
            col_name: Column name (for logging)
            metadata_seed: Random seed derived from metadata (different metadata = different forecast)
            metadata_bias: Bias [0, 1] to add extra error for modified metadata scenarios

        Returns:
            Full window where:
            - First context_length points are IDENTICAL to input
            - Last forecast_length points are generated forecast (close to ground truth)
        """
        synthetic = original.copy()

        # Context portion (first context_length points) stays EXACTLY the same
        # We only generate the forecast portion (last forecast_length points)

        # Extract the ground truth forecast portion (what we're trying to predict)
        ground_truth_forecast = original[context_length:context_length + forecast_length]

        # Extract context for statistics
        context = original[:context_length]
        valid_mask = ~np.isnan(context)
        valid_context = context[valid_mask]

        if len(valid_context) < 2:
            # Not enough data - return ground truth with small noise
            fill_value = valid_context[-1] if len(valid_context) > 0 else 0.0
            synthetic[context_length:context_length + forecast_length] = np.full(forecast_length, fill_value)
            logger.warning(
                f"Not enough valid context for {col_name}, using constant: {fill_value}"
            )
            return synthetic

        # Use metadata seed for reproducible but metadata-dependent randomness
        rng = np.random.RandomState(metadata_seed)

        # Calculate noise scale based on signal variance
        signal_std = np.std(valid_context)

        # Apply time shift to ground truth - makes forecast lag or lead the actual
        # Base shift: 1-2 time steps, modified: up to 3-4 time steps
        base_time_shift = rng.choice([-2, -1, 1, 2])
        extra_shift = int(rng.choice([-2, -1, 1, 2]) * metadata_bias)
        total_time_shift = base_time_shift + extra_shift

        # Shift the ground truth pattern
        if total_time_shift > 0:
            # Shift right (lag) - pad start with first value
            shifted_gt = np.concatenate(
                [
                    np.full(total_time_shift, ground_truth_forecast[0]),
                    ground_truth_forecast[:-total_time_shift],
                ]
            )
        elif total_time_shift < 0:
            # Shift left (lead) - pad end with last value
            shifted_gt = np.concatenate(
                [
                    ground_truth_forecast[-total_time_shift:],
                    np.full(-total_time_shift, ground_truth_forecast[-1]),
                ]
            )
        else:
            shifted_gt = ground_truth_forecast.copy()

        # Base forecast: add substantial noise so it's visibly different
        base_noise_scale = signal_std * 0.20  # 20% base noise
        base_noise = rng.randn(forecast_length) * base_noise_scale

        # Add extra error based on metadata_bias (makes "modified" forecasts worse)
        bias_noise_scale = (
            signal_std * 0.25 * metadata_bias
        )  # Up to 25% extra noise when modified
        bias_noise = rng.randn(forecast_length) * bias_noise_scale

        # Also add a systematic bias (shift) when metadata is modified
        # This makes the forecast consistently off in one direction
        systematic_bias = (
            signal_std * 0.15 * metadata_bias * rng.choice([-1, 1])
        )

        total_noise = base_noise + bias_noise + systematic_bias

        # Generate forecast from time-shifted ground truth plus noise
        forecast_values = shifted_gt + total_noise

        # Ensure forecast_values matches forecast_length
        if len(forecast_values) != forecast_length:
            # Truncate or pad if needed
            if len(forecast_values) > forecast_length:
                forecast_values = forecast_values[:forecast_length]
            else:
                forecast_values = np.pad(forecast_values, (0, forecast_length - len(forecast_values)), mode='edge')

        # Insert forecast into synthetic array (only replacing the forecast portion)
        # Context portion (0:context_length) remains unchanged from original
        synthetic[context_length:context_length + forecast_length] = forecast_values

        # Calculate error for logging
        mae = np.mean(np.abs(forecast_values - ground_truth_forecast[:len(forecast_values)]))
        logger.debug(
            f"Generated forecast for {col_name}: MAE={mae:.4f}, "
            f"bias={metadata_bias:.4f}, noise_std={np.std(total_noise):.4f}"
        )

        return synthetic

    def get_original_timeseries(
        self, data: DataFrameModel
    ) -> List[OneTimeSeries]:
        """Extract original timeseries from input data.

        Args:
            data: Input data as DataFrameModel

        Returns:
            List of OneTimeSeries with original values
        """
        df = data.to_dataframe()
        required_cols = self.config_loader.get_required_columns()
        timeseries_cols = required_cols.timeseries

        result: List[OneTimeSeries] = []
        for col_name in timeseries_cols:
            if col_name in df.columns:
                raw_values = df[col_name].tolist()
                # Convert NaN/Inf to None for JSON serialization
                values = [
                    None
                    if (
                        v is None
                        or (
                            isinstance(v, float)
                            and (np.isnan(v) or np.isinf(v))
                        )
                    )
                    else v
                    for v in raw_values
                ]
                result.append(
                    OneTimeSeries(
                        name=col_name,
                        values=values,
                    )
                )

        return result


# Cache for mock forecast service instances
_mock_forecast_services: Dict[str, MockForecastService] = {}


def get_mock_forecast_service(dataset_name: str) -> MockForecastService:
    """Get a MockForecastService instance (cached).

    Args:
        dataset_name: Name of the dataset

    Returns:
        MockForecastService instance
    """
    if dataset_name not in _mock_forecast_services:
        _mock_forecast_services[dataset_name] = MockForecastService(
            dataset_name
        )
    return _mock_forecast_services[dataset_name]
