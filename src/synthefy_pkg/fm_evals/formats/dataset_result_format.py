"""
A class that represents dataset results combining eval batches and forecast outputs.
"""

import os
import pickle
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)
from synthefy_pkg.fm_evals.formats.metrics import (
    SUPPORTED_METRICS,
    ForecastMetrics,
)
from synthefy_pkg.fm_evals.metrics.compute_metrics import compute_sample_metrics


class DatasetResultFormat:
    """
    A class that represents dataset results combining eval batches and forecast outputs.

    This format maintains lists of eval samples and forecast outputs, and automatically
    aggregates metrics across all batches.
    """

    def __init__(self):
        """Initialize an empty dataset result format."""
        self.eval_samples: List[List[SingleEvalSample]] = []  # [B][NC]
        self.forecast_outputs: List[List[SingleSampleForecast]] = []  # [B][NC]
        self.metrics: Optional[ForecastMetrics] = (
            None  # Aggregated metrics across all batches
        )

    def add_batch(
        self, eval_batch: EvalBatchFormat, forecast_output: ForecastOutputFormat
    ):
        """
        Add a batch of eval samples and forecast outputs.

        Parameters
        ----------
        eval_batch : EvalBatchFormat
            The eval batch to add
        forecast_output : ForecastOutputFormat
            The forecast output to add
        """
        # Filter eval_batch to only include forecast=True correlates (targets, not covariates)
        filtered_samples = []
        for row in eval_batch.samples:
            filtered_row = [sample for sample in row if sample.forecast]
            if filtered_row:
                filtered_samples.append(filtered_row)

        # Create filtered batch for validation and storage
        if filtered_samples:
            filtered_batch = EvalBatchFormat(samples=filtered_samples)
        else:
            filtered_batch = eval_batch

        # Validate that the batch sizes match
        assert filtered_batch.batch_size == forecast_output.batch_size, (
            f"Eval batch size ({filtered_batch.batch_size}) must match forecast output batch size ({forecast_output.batch_size})"
        )

        # Validate that the number of correlates match
        assert filtered_batch.num_correlates == forecast_output.num_correlates, (
            f"Eval batch num_correlates ({filtered_batch.num_correlates}) must match forecast output num_correlates ({forecast_output.num_correlates})"
        )

        # Add the filtered batches (only targets, not covariates)
        self.eval_samples.extend(filtered_batch.samples)
        self.forecast_outputs.extend(forecast_output.forecasts)

        # Update aggregated metrics
        self._update_metrics()

    def _update_metrics(self):
        """Update the aggregated metrics across all batches."""
        if len(self.eval_samples) == 0:
            self.metrics = None
            return

        # Create EvalBatchFormat and ForecastOutputFormat from our data
        eval_batch = EvalBatchFormat(samples=self.eval_samples)
        forecast_output = ForecastOutputFormat(forecasts=self.forecast_outputs)

        # Use the static method from BaseForecaster to compute metrics
        forecast_output_with_metrics = BaseForecaster._compute_metrics(
            eval_batch, forecast_output
        )

        # Extract the batch-level metrics
        self.metrics = forecast_output_with_metrics.metrics

    def __str__(self):
        """String representation of the dataset result format."""
        lines = []
        lines.append("DatasetResultFormat:")
        lines.append(f"  Number of batches: {len(self.eval_samples)}")
        lines.append(
            f"  Total samples: {len(self.eval_samples) * len(self.eval_samples[0]) if len(self.eval_samples) > 0 else 0}"
        )

        if self.metrics is not None:
            lines.append("  Aggregated metrics:")
            # Dynamically print all available metrics
            for field_name, field_value in self.metrics.__dict__.items():
                if field_name != "sample_id" and field_value is not None:
                    lines.append(f"    {field_name.upper()}: {field_value:.4f}")
        else:
            lines.append("  Aggregated metrics: None")

        return "\n".join(lines)

    @staticmethod
    def find_best_models(model_results: dict) -> dict:
        """
        Find the best performing models across multiple dataset results.

        Parameters
        ----------
        model_results : dict
            Dictionary mapping model names to their dataset results

        Returns
        -------
        dict
            Dictionary with best models for each metric
        """
        # Collect metrics for all models
        metrics_data = []

        for model_name, result_info in model_results.items():
            dataset_result = result_info["dataset_result"]

            if dataset_result.metrics is not None:
                # Dynamically collect all available metrics
                model_metrics = {"model_name": model_name}
                for (
                    field_name,
                    field_value,
                ) in dataset_result.metrics.__dict__.items():
                    if field_name != "sample_id" and field_value is not None:
                        model_metrics[field_name] = field_value
                metrics_data.append(model_metrics)

        if not metrics_data:
            return {}

        # Find best performing models for each metric
        best_models = {}

        # Get all available metric names (excluding sample_id)
        if metrics_data:
            available_metrics = [
                key for key in metrics_data[0].keys() if key != "model_name"
            ]

            # For each metric, find the model with the lowest value
            for metric_name in available_metrics:
                if any(metric_name in data for data in metrics_data):
                    best_model = min(metrics_data, key=lambda x: x[metric_name])
                    best_models[metric_name] = {
                        "model_name": best_model["model_name"],
                        "value": best_model[metric_name],
                    }

        return best_models

    def save_pkl(self, filepath: str):
        """
        Save the dataset result format to a pickle file.

        Parameters
        ----------
        filepath : str
            Path to save the pickle file
        """

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def save_h5(self, filepath: str):
        """
        Save the dataset result format to an HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to save the HDF5 file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Collect all samples
        all_history = []
        all_history_timestamps = []
        all_true_forecast = []
        all_true_forecast_timestamps = []
        all_predicted_forecast = []
        all_predicted_forecast_timestamps = []
        all_forecast_mask = []
        all_sample_ids = []
        all_model_names = []
        # Initialize metric collections using SUPPORTED_METRICS
        metric_collections = {
            metric_name: [] for metric_name in SUPPORTED_METRICS
        }

        for batch_idx, (eval_batch, forecast_batch) in enumerate(
            zip(self.eval_samples, self.forecast_outputs)
        ):
            for sample_idx, (eval_sample, forecast_sample) in enumerate(
                zip(eval_batch, forecast_batch)
            ):
                # Skip samples that shouldn't be forecasted
                if not eval_sample.forecast:
                    continue

                # History values and timestamps
                all_history.append(eval_sample.history_values)
                all_history_timestamps.append(eval_sample.history_timestamps)

                # True forecast values (target values) and timestamps
                all_true_forecast.append(eval_sample.target_values)
                all_true_forecast_timestamps.append(
                    eval_sample.target_timestamps
                )

                # Predicted forecast values and timestamps
                all_predicted_forecast.append(forecast_sample.values)
                all_predicted_forecast_timestamps.append(
                    forecast_sample.timestamps
                )

                # Create forecast mask (0 = valid, 1 = masked)
                # For now, assume all forecast points are valid (no masking)
                forecast_mask = np.zeros_like(
                    eval_sample.target_values, dtype=np.int32
                )
                all_forecast_mask.append(forecast_mask)

                # Dataset ID (use sample_id as string)
                sample_id = str(eval_sample.sample_id)
                all_sample_ids.append(sample_id)

                # Model name
                model_name = forecast_sample.model_name
                all_model_names.append(model_name)

                # Metrics
                if (
                    hasattr(forecast_sample, "metrics")
                    and forecast_sample.metrics is not None
                ):
                    for metric_name in SUPPORTED_METRICS:
                        metric_collections[metric_name].append(
                            getattr(forecast_sample.metrics, metric_name)
                        )
                else:
                    for metric_name in SUPPORTED_METRICS:
                        metric_collections[metric_name].append(np.nan)

        # Convert to numpy arrays
        # Use object arrays to handle both homogeneous and inhomogeneous shapes
        history = np.array(all_history, dtype=object)
        true_forecast = np.array(all_true_forecast, dtype=object)
        predicted_forecast = np.array(all_predicted_forecast, dtype=object)
        forecast_mask = np.array(all_forecast_mask, dtype=object)
        sample_ids = np.array(all_sample_ids, dtype="S")
        model_names = np.array(all_model_names, dtype="S")
        # Convert metric collections to numpy arrays
        metrics_arrays = {
            metric_name: np.array(values)
            for metric_name, values in metric_collections.items()
        }

        # Convert timestamps to a format H5Py can handle
        def convert_timestamps_to_int64(timestamps_list):
            converted = []
            for timestamps in timestamps_list:
                if len(timestamps) > 0:
                    # Convert to int64 nanoseconds
                    converted.append(timestamps.astype(np.int64))
                else:
                    converted.append(np.array([], dtype=np.int64))
            return np.array(converted, dtype=object)

        history_timestamps = convert_timestamps_to_int64(all_history_timestamps)
        true_forecast_timestamps = convert_timestamps_to_int64(
            all_true_forecast_timestamps
        )
        predicted_forecast_timestamps = convert_timestamps_to_int64(
            all_predicted_forecast_timestamps
        )

        # Save to HDF5 file
        with h5py.File(filepath, "w") as f:
            # Always use group format for object arrays
            history_group = f.create_group("history")
            for i, hist_data in enumerate(history):
                # Convert to regular numpy array if it's an object array (which it will be)
                if hist_data.dtype == object:
                    hist_data = np.array(hist_data, dtype=np.float64)
                history_group.create_dataset(f"sample_{i}", data=hist_data)

            # Save timestamp arrays as groups
            history_timestamps_group = f.create_group("history_timestamps")
            for i, ts_data in enumerate(history_timestamps):
                # Convert to regular numpy array if it's an object array
                if ts_data.dtype == object:
                    ts_data = np.array(ts_data, dtype=np.int64)
                history_timestamps_group.create_dataset(
                    f"sample_{i}", data=ts_data
                )

            true_forecast_group = f.create_group("true_forecast")
            for i, forecast_data in enumerate(true_forecast):
                # Convert to regular numpy array if it's an object array
                if forecast_data.dtype == object:
                    forecast_data = np.array(forecast_data, dtype=np.float64)
                true_forecast_group.create_dataset(
                    f"sample_{i}", data=forecast_data
                )

            true_forecast_timestamps_group = f.create_group(
                "true_forecast_timestamps"
            )
            for i, ts_data in enumerate(true_forecast_timestamps):
                # Convert to regular numpy array if it's an object array
                if ts_data.dtype == object:
                    ts_data = np.array(ts_data, dtype=np.int64)
                true_forecast_timestamps_group.create_dataset(
                    f"sample_{i}", data=ts_data
                )

            predicted_forecast_group = f.create_group("predicted_forecast")
            for i, forecast_data in enumerate(predicted_forecast):
                # Convert to regular numpy array if it's an object array
                if forecast_data.dtype == object:
                    forecast_data = np.array(forecast_data, dtype=np.float64)
                predicted_forecast_group.create_dataset(
                    f"sample_{i}", data=forecast_data
                )

            predicted_forecast_timestamps_group = f.create_group(
                "predicted_forecast_timestamps"
            )
            for i, ts_data in enumerate(predicted_forecast_timestamps):
                # Convert to regular numpy array if it's an object array
                if ts_data.dtype == object:
                    ts_data = np.array(ts_data, dtype=np.int64)
                predicted_forecast_timestamps_group.create_dataset(
                    f"sample_{i}", data=ts_data
                )

            forecast_mask_group = f.create_group("forecast_mask")
            for i, mask_data in enumerate(forecast_mask):
                # Convert to regular numpy array if it's an object array
                if mask_data.dtype == object:
                    mask_data = np.array(mask_data, dtype=np.int32)
                forecast_mask_group.create_dataset(
                    f"sample_{i}", data=mask_data
                )

            f.create_dataset("dataset_ids", data=sample_ids)
            f.create_dataset("model_names", data=model_names)
            # Create datasets for all supported metrics
            for metric_name in SUPPORTED_METRICS:
                f.create_dataset(metric_name, data=metrics_arrays[metric_name])

    def save_csv(self, filepath: str):
        """
        Save the dataset result format to a CSV file.

        Parameters
        ----------
        filepath : str
            Path to save the CSV file
        """
        # Ensure directory exists (only if filepath contains a directory)
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Collect all data for CSV export
        csv_data = []

        for batch_idx, (eval_batch, forecast_batch) in enumerate(
            zip(self.eval_samples, self.forecast_outputs)
        ):
            for sample_idx, (eval_sample, forecast_sample) in enumerate(
                zip(eval_batch, forecast_batch)
            ):
                # Skip samples that shouldn't be forecasted
                if not eval_sample.forecast:
                    continue

                # Get sample_id as string
                sample_id = str(eval_sample.sample_id)

                # Add history data (if available)
                if len(eval_sample.history_timestamps) > 0:
                    for h_idx, (timestamp, true_val) in enumerate(
                        zip(
                            eval_sample.history_timestamps,
                            eval_sample.history_values,
                        )
                    ):
                        row = {
                            "sample_id": sample_id,
                            "batch_idx": batch_idx,
                            "sample_idx": sample_idx,
                            "timestamp": timestamp,
                            "split": "history",
                            "history_length": len(eval_sample.history_values),
                            "forecast_step": -1,  # -1 indicates history
                            "true_value": true_val,
                            "predicted_value": np.nan,  # No predictions for history
                            "model_name": forecast_sample.model_name,
                        }

                        # Add column_name if available
                        if eval_sample.column_name is not None:
                            row["column_name"] = eval_sample.column_name

                        csv_data.append(row)

                # Add target/forecast data
                for t_idx, (timestamp, true_val, pred_val) in enumerate(
                    zip(
                        eval_sample.target_timestamps,
                        eval_sample.target_values,
                        forecast_sample.values,
                    )
                ):
                    row = {
                        "sample_id": sample_id,
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "timestamp": timestamp,
                        "split": "target",
                        "history_length": len(eval_sample.history_values),
                        "forecast_step": t_idx,
                        "true_value": true_val,
                        "predicted_value": pred_val,
                        "model_name": forecast_sample.model_name,
                    }

                    # Add column_name if available
                    if eval_sample.column_name is not None:
                        row["column_name"] = eval_sample.column_name

                    # Add metrics if available (only for target rows)
                    if (
                        hasattr(forecast_sample, "metrics")
                        and forecast_sample.metrics is not None
                    ):
                        for metric_name in SUPPORTED_METRICS:
                            row[metric_name] = getattr(
                                forecast_sample.metrics, metric_name
                            )
                    else:
                        for metric_name in SUPPORTED_METRICS:
                            row[metric_name] = np.nan

                    csv_data.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)

    @classmethod
    def load_pkl(cls, filepath: str):
        """
        Load a dataset result format from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the pickle file to load

        Returns
        -------
        DatasetResultFormat
            The loaded dataset result format
        """

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_csv(cls, filepath: str):
        """
        Load a dataset result format from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file to load

        Returns
        -------
        DatasetResultFormat
            The loaded dataset result format
        """
        df = pd.read_csv(filepath)

        # Create empty dataset result format
        dataset_result = cls()

        # Group by sample_id to reconstruct batches
        sample_groups = df.groupby("sample_id")

        # Track unique batch and sample indices to maintain structure
        batch_sample_map = {}

        for sample_id, group in sample_groups:
            # Get batch and sample indices from the first row
            batch_idx = group["batch_idx"].iloc[0]
            sample_idx = group["sample_idx"].iloc[0]

            # Create key for tracking batch structure
            batch_key = (batch_idx, sample_idx)

            if batch_key not in batch_sample_map:
                # Check if split column exists to determine if we have history data
                has_split = "split" in group.columns

                if has_split:
                    # Split data into history and target
                    history_mask = group["split"] == "history"
                    target_mask = group["split"] == "target"

                    history_data = group[history_mask]
                    target_data = group[target_mask]

                    # Extract history data
                    history_timestamps = (
                        history_data["timestamp"].values
                        if len(history_data) > 0
                        else np.array([])
                    )
                    history_values = (
                        history_data["true_value"].values
                        if len(history_data) > 0
                        else np.array([])
                    )

                    # Extract target data
                    target_timestamps = (
                        target_data["timestamp"].values
                        if len(target_data) > 0
                        else np.array([])
                    )
                    target_values = (
                        target_data["true_value"].values
                        if len(target_data) > 0
                        else np.array([])
                    )
                    predicted_values = (
                        target_data["predicted_value"].values
                        if len(target_data) > 0
                        else np.array([])
                    )
                else:
                    # Legacy format: all data is target (no history)
                    history_timestamps = np.array([])
                    history_values = np.array([])
                    target_timestamps = group["timestamp"].values
                    target_values = group["true_value"].values
                    predicted_values = group["predicted_value"].values

                # Extract metrics if available
                metrics = {}
                if any(col in group.columns for col in SUPPORTED_METRICS):
                    # Get metrics from target rows (not history)
                    target_metrics = group[group["split"] == "target"]
                    if len(target_metrics) > 0:
                        for metric_name in SUPPORTED_METRICS:
                            if metric_name in target_metrics.columns:
                                metrics[metric_name] = target_metrics[
                                    metric_name
                                ].iloc[0]
                            else:
                                metrics[metric_name] = None

                batch_sample_map[batch_key] = {
                    "sample_id": sample_id,
                    "history_timestamps": history_timestamps,
                    "history_values": history_values,
                    "target_timestamps": target_timestamps,
                    "target_values": target_values,
                    "predicted_values": predicted_values,
                    "history_length": group["history_length"].iloc[0],
                    "model_name": group["model_name"].iloc[0],
                    "column_name": group["column_name"].iloc[0]
                    if "column_name" in group.columns
                    else None,
                    "metrics": metrics,
                }

        # Reconstruct the data structure
        # We need to organize by batch_idx first, then by sample_idx
        batch_indices = sorted(set(key[0] for key in batch_sample_map.keys()))
        sample_indices = sorted(set(key[1] for key in batch_sample_map.keys()))

        # Initialize batches
        eval_samples = []
        forecast_outputs = []

        for batch_idx in batch_indices:
            batch_eval_samples = []
            batch_forecast_outputs = []

            for sample_idx in sample_indices:
                batch_key = (batch_idx, sample_idx)
                if batch_key in batch_sample_map:
                    data = batch_sample_map[batch_key]

                    # Convert timestamps to datetime64
                    history_timestamps = (
                        pd.to_datetime(
                            data["history_timestamps"]
                        ).values.astype("datetime64[ns]")
                        if len(data["history_timestamps"]) > 0
                        else np.array([], dtype="datetime64[ns]")
                    )
                    target_timestamps = (
                        pd.to_datetime(data["target_timestamps"]).values.astype(
                            "datetime64[ns]"
                        )
                        if len(data["target_timestamps"]) > 0
                        else np.array([], dtype="datetime64[ns]")
                    )

                    # Convert values to float64
                    history_values = (
                        data["history_values"].astype(np.float64)
                        if len(data["history_values"]) > 0
                        else np.array([], dtype=np.float64)
                    )
                    target_values = (
                        data["target_values"].astype(np.float64)
                        if len(data["target_values"]) > 0
                        else np.array([], dtype=np.float64)
                    )
                    predicted_values = (
                        data["predicted_values"].astype(np.float64)
                        if len(data["predicted_values"]) > 0
                        else np.array([], dtype=np.float64)
                    )

                    # Create SingleEvalSample
                    eval_sample = SingleEvalSample(
                        sample_id=data["sample_id"],
                        history_timestamps=history_timestamps,
                        history_values=history_values,
                        target_timestamps=target_timestamps,
                        target_values=target_values,
                        forecast=True,
                        metadata=False,
                        leak_target=False,
                        column_name=data["column_name"],
                    )
                    batch_eval_samples.append(eval_sample)

                    # Create SingleSampleForecast
                    forecast_sample = SingleSampleForecast(
                        sample_id=data["sample_id"],
                        timestamps=target_timestamps,
                        values=predicted_values,
                        model_name=data["model_name"],
                    )

                    # Use extracted metrics if available, otherwise compute them using compute_sample_metrics
                    if data["metrics"] and any(
                        v is not None for v in data["metrics"].values()
                    ):
                        # Create ForecastMetrics from extracted values
                        metrics_dict = {"sample_id": data["sample_id"]}
                        for metric_name in SUPPORTED_METRICS:
                            metrics_dict[metric_name] = data["metrics"].get(
                                metric_name, float("nan")
                            )
                        forecast_sample.metrics = ForecastMetrics(
                            **metrics_dict
                        )
                    else:
                        # Compute metrics using the new system
                        forecast_sample.metrics = compute_sample_metrics(
                            eval_sample, forecast_sample
                        )
                    batch_forecast_outputs.append(forecast_sample)

            if batch_eval_samples:  # Only add non-empty batches
                eval_samples.append(batch_eval_samples)
                forecast_outputs.append(batch_forecast_outputs)

        # Set the reconstructed data
        dataset_result.eval_samples = eval_samples
        dataset_result.forecast_outputs = forecast_outputs

        # Update aggregated metrics
        dataset_result._update_metrics()

        return dataset_result

    @classmethod
    def load_from_file(cls, filepath: str):
        if not os.path.exists(filepath):
            raise ValueError(f"File does not exist: {filepath}")

        file_ext = os.path.splitext(filepath)[1].lower()

        try:
            if file_ext == ".h5":
                return cls.load_h5(filepath)
            elif file_ext == ".csv":
                return cls.load_csv(filepath)
            elif file_ext == ".pkl" or file_ext == ".pickle":
                return cls.load_pkl(filepath)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}. Supported formats: .h5, .csv, .pkl"
                )
        except Exception as e:
            raise ValueError(f"Failed to load file {filepath}: {str(e)}")

    @classmethod
    def load_h5(cls, filepath: str):
        """
        Load a dataset result format from an HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to the HDF5 file to load

        Returns
        -------
        DatasetResultFormat
            The loaded dataset result format
        """
        # Create empty dataset result format
        dataset_result = cls()

        with h5py.File(filepath, "r") as f:
            # Load the datasets - always use group format
            history_group = f["history"]
            history = []
            for key in sorted(history_group.keys()):  # type: ignore
                history.append(np.array(history_group[key]))  # type: ignore
            history = np.array(history, dtype=object)

            true_forecast_group = f["true_forecast"]
            true_forecast = []
            for key in sorted(true_forecast_group.keys()):  # type: ignore
                true_forecast.append(np.array(true_forecast_group[key]))  # type: ignore
            true_forecast = np.array(true_forecast, dtype=object)

            predicted_forecast_group = f["predicted_forecast"]
            predicted_forecast = []
            for key in sorted(predicted_forecast_group.keys()):  # type: ignore
                predicted_forecast.append(
                    np.array(predicted_forecast_group[key])  # type: ignore
                )  # type: ignore
            predicted_forecast = np.array(predicted_forecast, dtype=object)

            sample_ids_dataset = f["dataset_ids"]

            # Load timestamp arrays - always use group format
            history_timestamps_group = f["history_timestamps"]
            history_timestamps = []
            for key in sorted(history_timestamps_group.keys()):  # type: ignore
                history_timestamps.append(
                    np.array(history_timestamps_group[key])  # type: ignore
                )  # type: ignore
            history_timestamps = np.array(history_timestamps, dtype=object)

            true_forecast_timestamps_group = f["true_forecast_timestamps"]
            true_forecast_timestamps = []
            for key in sorted(true_forecast_timestamps_group.keys()):  # type: ignore
                true_forecast_timestamps.append(
                    np.array(true_forecast_timestamps_group[key])  # type: ignore
                )  # type: ignore
            true_forecast_timestamps = np.array(
                true_forecast_timestamps, dtype=object
            )

            # Check if model_names dataset exists
            if "model_names" in f:
                model_names_dataset = f["model_names"]
                model_names = np.array(model_names_dataset)
                # Convert model_names from bytes to strings
                model_names = [
                    name.decode("utf-8")
                    if isinstance(name, bytes)
                    else str(name)
                    for name in model_names
                ]
            else:
                # Use default model name
                model_names = ["unknown"] * len(np.array(sample_ids_dataset))

            # Load metrics if available
            metrics = {}
            for metric_name in SUPPORTED_METRICS:
                if metric_name in f:
                    metrics[metric_name] = np.array(f[metric_name])
                else:
                    metrics[metric_name] = None

            # Convert to numpy arrays
            sample_ids = np.array(sample_ids_dataset)

            # Convert sample_ids from bytes to strings
            sample_ids = [
                sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)
                for sid in sample_ids
            ]

        # Reconstruct the data structure
        eval_samples = []
        forecast_outputs = []

        # Each row in the arrays represents one sample
        num_samples = len(history)
        for i in range(num_samples):
            # Get history timestamps and values
            history_length = len(history[i])
            if history_length > 0:
                # Convert from int64 nanoseconds back to datetime64[ns]
                history_timestamps_sample = history_timestamps[i].astype(
                    "datetime64[ns]"
                )
                history_values = history[i].astype(np.float64)
            else:
                history_timestamps_sample = np.array([], dtype="datetime64[ns]")
                history_values = np.array([], dtype=np.float64)

            # Get target timestamps and values
            target_length = len(true_forecast[i])
            if target_length > 0:
                # Convert from int64 nanoseconds back to datetime64[ns]
                target_timestamps = true_forecast_timestamps[i].astype(
                    "datetime64[ns]"
                )
                target_values = true_forecast[i].astype(np.float64)
            else:
                target_timestamps = np.array([], dtype="datetime64[ns]")
                target_values = np.array([], dtype=np.float64)

            # Create SingleEvalSample
            eval_sample = SingleEvalSample(
                sample_id=sample_ids[i],
                history_timestamps=history_timestamps_sample,
                history_values=history_values,
                target_timestamps=target_timestamps,
                target_values=target_values,
                forecast=True,
                metadata=False,
                leak_target=False,
                column_name=None,
            )

            # Create SingleSampleForecast
            forecast_sample = SingleSampleForecast(
                sample_id=sample_ids[i],
                timestamps=target_timestamps,
                values=predicted_forecast[i].astype(np.float64),
                model_name=model_names[i],  # Use the loaded model name
            )

            # Use loaded metrics if available, otherwise compute them
            if any(metrics[name] is not None for name in metrics.keys()):
                # Create ForecastMetrics from loaded values
                metrics_dict: dict = {"sample_id": sample_ids[i]}
                for metric_name in SUPPORTED_METRICS:
                    if metrics[metric_name] is not None:
                        metrics_dict[metric_name] = float(
                            metrics[metric_name][i]
                        )
                    else:
                        metrics_dict[metric_name] = float("nan")
                forecast_sample.metrics = ForecastMetrics(**metrics_dict)
            else:
                # Compute metrics using the new system
                forecast_sample.metrics = compute_sample_metrics(
                    eval_sample, forecast_sample
                )

            # Since H5 stores data as flat arrays, we'll organize them into batches
            # For simplicity, we'll put each sample in its own batch
            eval_samples.append([eval_sample])
            forecast_outputs.append([forecast_sample])

        # Set the reconstructed data
        dataset_result.eval_samples = eval_samples
        dataset_result.forecast_outputs = forecast_outputs

        # Update aggregated metrics
        dataset_result._update_metrics()

        return dataset_result

    def to_dfs(self) -> List[pd.DataFrame]:
        """
        Convert the dataset result format to a list of DataFrames for evaluation.

        Each batch of NC correlates will translate to one DataFrame. The DataFrames include:
        - Timestamps, history values, target values, and forecast values
        - Model information (model_name from forecast samples)
        - Individual sample metrics (MAE, MAPE) for each correlate
        - Sample metadata (sample_id, column_name, etc.)
        - Split information (history vs target)

        Returns
        -------
        List[pd.DataFrame]
            List of DataFrames, one for each batch of correlates
        """
        if len(self.eval_samples) == 0:
            return []

        result_dfs: List[pd.DataFrame] = []

        for batch_idx in range(len(self.eval_samples)):
            eval_batch = self.eval_samples[batch_idx]
            forecast_batch = self.forecast_outputs[batch_idx]

            # Validate batch sizes match
            assert len(eval_batch) == len(forecast_batch), (
                f"Eval batch size ({len(eval_batch)}) must match forecast batch size ({len(forecast_batch)})"
            )

            columns = {}
            split_col = None
            timestamps_ref = None
            metrics_data = {}
            metadata_data = {}

            for nc in range(len(eval_batch)):
                eval_sample = eval_batch[nc]
                forecast_sample = forecast_batch[nc]

                # Get eval sample as DataFrame
                eval_df = eval_sample.to_df()

                # Split into history and target
                if "split" in eval_df.columns:
                    history_mask = eval_df["split"] == "history"
                    target_mask = eval_df["split"] == "target"
                    history = eval_df[history_mask]
                    target = eval_df[target_mask]
                else:
                    # If no split, treat all as target
                    history = eval_df.iloc[0:0]
                    target = eval_df

                # Get timestamps and assert all correlates match
                timestamps = pd.concat(
                    [history["timestamps"], target["timestamps"]]
                ).to_numpy()
                if timestamps_ref is None:
                    timestamps_ref = timestamps
                else:
                    assert np.array_equal(timestamps, timestamps_ref), (
                        "Timestamps do not match across correlates!"
                    )

                # Get actuals
                actuals = pd.concat(
                    [history["values"], target["values"]]
                ).to_numpy()
                col_name = (
                    getattr(eval_sample, "column_name", None)
                    or f"correlate_{nc}"
                )
                columns[col_name] = actuals

                # Check if this correlate was forecasted
                is_forecasted = getattr(eval_sample, "forecast", True)
                if is_forecasted:
                    # Get forecast: pad with NaN for history, then forecast values for target
                    fcast_values = forecast_sample.to_df()["values"].to_numpy()
                    forecasts = np.concatenate(
                        [np.full(len(history), np.nan), fcast_values]
                    )
                    forecast_col = f"{col_name}_forecast"
                    columns[forecast_col] = forecasts

                    # Add sample-level metrics if available
                    if (
                        hasattr(forecast_sample, "metrics")
                        and forecast_sample.metrics is not None
                    ):
                        metrics_data[f"{col_name}_mae"] = (
                            forecast_sample.metrics.mae
                        )
                        metrics_data[f"{col_name}_mape"] = (
                            forecast_sample.metrics.mape
                        )

                    # Add model information
                    metadata_data[f"{col_name}_model_name"] = (
                        forecast_sample.model_name
                    )

                # Add sample metadata
                metadata_data[f"{col_name}_sample_id"] = str(
                    eval_sample.sample_id
                )
                metadata_data[f"{col_name}_column_name"] = (
                    eval_sample.column_name or f"correlate_{nc}"
                )
                # metadata_data[f"{col_name}_forecast"] = eval_sample.forecast
                metadata_data[f"{col_name}_metadata"] = eval_sample.metadata
                metadata_data[f"{col_name}_leak_target"] = (
                    eval_sample.leak_target
                )
                metadata_data[f"{col_name}_history_length"] = len(
                    eval_sample.history_values
                )
                metadata_data[f"{col_name}_target_length"] = len(
                    eval_sample.target_values
                )

                # Save split column if present (from eval_df)
                if split_col is None and "split" in eval_df.columns:
                    split_col = pd.concat(
                        [history["split"], target["split"]]
                    ).to_numpy()

            # Build DataFrame
            df = pd.DataFrame({"timestamps": timestamps_ref, **columns})
            if split_col is not None:
                df["split"] = split_col

            # Add batch-level metrics to every DataFrame for consistency
            if self.metrics is not None:
                df["batch_mae"] = self.metrics.mae
                df["batch_mape"] = self.metrics.mape

            # Add sample-level metrics as additional columns
            if metrics_data:
                for key, value in metrics_data.items():
                    df[key] = value

            # Add metadata as additional columns
            if metadata_data:
                for key, value in metadata_data.items():
                    df[key] = value

            result_dfs.append(df)

        return result_dfs
