import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.utils import fill_nan_values
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class ChronosForecaster(BaseForecaster):
    """
    Univariate forecaster using Amazon's Chronos pretrained time series models.

    Chronos is a family of pretrained time series forecasting models based on
    language model architectures that can perform zero-shot forecasting on new
    time series data.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device_map: str = "auto",
        torch_dtype=None,
        num_samples: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ):
        """
        Initialize the ChronosForecaster.

        Args:
            model_name: Chronos model to use. Options include:
                - amazon/chronos-t5-tiny (8M params)
                - amazon/chronos-t5-mini (20M params)
                - amazon/chronos-t5-small (46M params)
                - amazon/chronos-t5-base (200M params)
                - amazon/chronos-t5-large (710M params)
                - amazon/chronos-bolt-tiny (9M params, faster)
                - amazon/chronos-bolt-mini (21M params, faster)
                - amazon/chronos-bolt-small (48M params, faster)
                - amazon/chronos-bolt-base (205M params, faster)
            device_map: Device to run on ("auto", "cpu", "cuda", "mps")
            torch_dtype: Torch data type (None for auto, torch.bfloat16
                recommended)
            num_samples: Number of forecast samples to generate
            temperature: Sampling temperature for generation
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        """
        super().__init__("ChronosForecaster")
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.pipeline = None
        self.fitted_sample_ids = set()
        self.B = 0
        self.NC = 0

        # Try to import chronos - will fail gracefully if not installed
        self._import_chronos()

        # Load the model during initialization since it's pretrained
        self._load_model()

    def _import_chronos(self):
        """Import chronos with helpful error message if not available."""
        try:
            from chronos import ChronosPipeline

            self.ChronosPipeline = ChronosPipeline
        except ImportError:
            raise ImportError(
                "Chronos is not installed. Please install it with:\n"
                "pip install git+https://github.com/amazon-science/"
                "chronos-forecasting.git\n"
                "or for Apple Silicon Macs:\n"
                "pip install git+https://github.com/amazon-science/"
                "chronos-forecasting.git@mlx"
            )

    def _load_model(self):
        """Load the Chronos model during initialization."""
        try:
            logger.info("Loading ch model")
            self.pipeline = self.ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
            )
            logger.info("ch model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ch model: {e}")
            raise RuntimeError(
                "Failed to initialize chf with model "
                "Please check if the model name is correct "
                "and you have sufficient memory/disk space."
            ) from e

    def fit(self, batch: EvalBatchFormat) -> bool:
        """
        Fit the Chronos model. Since Chronos is pretrained, this mainly
        involves loading the model and preparing the sample tracking.

        Args:
            batch: Batch of evaluation samples

        Returns:
            True if fitting succeeded
        """
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.fitted_sample_ids = set()

        # Track which samples we can forecast
        for i in tqdm(range(self.B), desc="Preparing ch"):
            for j in range(self.NC):
                sample = batch[i, j]
                if not sample.forecast:
                    continue

                item_id = str(sample.sample_id)

                # Check if we have sufficient data
                if len(sample.history_values) < 1:
                    logger.warning(
                        f"cf: No historical data for sample "
                        f"{i}, correlate {j} (sample_id={item_id})"
                    )
                    continue

                # Check for non-finite values
                if not np.all(np.isfinite(sample.history_values)):
                    logger.warning(
                        f"cf: Non-finite values in history for "
                        f"sample {i}, correlate {j} (sample_id={item_id})"
                    )

                self.fitted_sample_ids.add(item_id)

        logger.info(f"cf fitted for {len(self.fitted_sample_ids)} samples")
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """
        Generate forecasts using the Chronos model.

        Args:
            batch: Batch of evaluation samples

        Returns:
            Forecast output with predictions for each sample
        """
        # Model is loaded during __init__, so pipeline should never be None
        assert self.pipeline is not None, (
            "Pipeline should be loaded during initialization"
        )

        B = batch.batch_size
        NC = batch.num_correlates
        T = batch.target_length

        forecasts = []

        for i in tqdm(range(B), desc="Predicting ch"):
            row = []

            for j in range(NC):
                sample = batch[i, j]

                # Handle non-forecast samples
                if not sample.forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    continue

                item_id = str(sample.sample_id)

                # Check if this sample was seen during fit
                if item_id not in self.fitted_sample_ids:
                    logger.warning(
                        f"ChronosForecaster: Sample {i}, correlate {j} "
                        f"(sample_id={item_id}) was not seen during fit"
                    )
                    pred = np.full((T,), np.nan, dtype=np.float32)
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=sample.target_timestamps,
                            values=pred,
                            model_name=self.name,
                        )
                    )
                    continue

                try:
                    sample.history_values = fill_nan_values(
                        sample.history_values
                    )
                    # Prepare context data for Chronos
                    # Chronos expects 1D tensor or list of values
                    context = torch.tensor(
                        sample.history_values, dtype=torch.float32
                    )

                    # Get prediction length
                    prediction_length = len(sample.target_timestamps)

                    # Generate forecast using Chronos
                    # forecast shape: [num_series, num_samples,
                    # prediction_length]
                    forecast = self.pipeline.predict(
                        context=context,
                        prediction_length=prediction_length,
                        num_samples=self.num_samples,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                    )

                    # Extract median forecast from samples
                    # forecast[0] has shape [num_samples, prediction_length]
                    forecast_samples = forecast[0].numpy()
                    pred = np.median(forecast_samples, axis=0).astype(
                        np.float32
                    )

                except Exception as e:
                    logger.warning(
                        f"ChronosForecaster: Error predicting for sample {i}, "
                        f"correlate {j} (sample_id={item_id}): {e}"
                    )
                    pred = np.full((T,), np.nan, dtype=np.float32)

                row.append(
                    SingleSampleForecast(
                        sample_id=sample.sample_id,
                        timestamps=sample.target_timestamps,
                        values=pred,
                        model_name=self.name,
                    )
                )

            forecasts.append(row)

        return ForecastOutputFormat(forecasts)


class ChronosBoltForecaster(BaseForecaster):
    """
    Univariate forecaster using Amazon's Chronos-Bolt pretrained time series models.

    Chronos-Bolt models are faster and more memory-efficient versions of the
    original Chronos models, offering up to 250x speedup with improved accuracy.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-small",
        device_map: str = "auto",
        torch_dtype=None,
    ):
        """
        Initialize the ChronosBoltForecaster.

        Args:
            model_name: Chronos-Bolt model to use. Options include:
                - amazon/chronos-bolt-tiny (9M params)
                - amazon/chronos-bolt-mini (21M params)
                - amazon/chronos-bolt-small (48M params)
                - amazon/chronos-bolt-base (205M params)
            device_map: Device to run on ("auto", "cpu", "cuda", "mps")
            torch_dtype: Torch data type (None for auto, torch.bfloat16
                recommended)
        """
        super().__init__("ChronosBoltForecaster")
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype or torch.bfloat16

        self.pipeline = None
        self.fitted_sample_ids = set()
        self.B = 0
        self.NC = 0

        # Try to import chronos - will fail gracefully if not installed
        self._import_chronos()

        # Load the model during initialization since it's pretrained
        self._load_model()

    def _import_chronos(self):
        """Import chronos with helpful error message if not available."""
        try:
            from chronos import BaseChronosPipeline

            self.BaseChronosPipeline = BaseChronosPipeline
        except ImportError:
            raise ImportError(
                "ch is not installed. Please install it with:\n"
                "pip install ch-forecasting\n"
                "Note: chb requires the newer ch-forecasting "
                "package"
            )

    def _load_model(self):
        """Load the Chronos-Bolt model during initialization."""
        try:
            logger.info("Loading chb model")
            self.pipeline = self.BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
            )
            logger.info("chb model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load chb model: {e}")
            raise RuntimeError(
                "Failed to initialize chbf with model "
                "Please check if the model name is correct "
                "and you have sufficient memory/disk space."
            ) from e

    def fit(self, batch: EvalBatchFormat) -> bool:
        """
        Fit the Chronos-Bolt model. Since Chronos is pretrained, this mainly
        involves loading the model and preparing the sample tracking.

        Args:
            batch: Batch of evaluation samples

        Returns:
            True if fitting succeeded
        """
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.fitted_sample_ids = set()

        # Track which samples we can forecast
        for i in tqdm(range(self.B), desc="Preparing ch"):
            for j in range(self.NC):
                sample = batch[i, j]
                if not sample.forecast:
                    continue

                item_id = str(sample.sample_id)

                # Check if we have sufficient data
                if len(sample.history_values) < 1:
                    logger.warning(
                        f"chbf: No historical data for "
                        f"sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    continue

                # Check for non-finite values
                if not np.all(np.isfinite(sample.history_values)):
                    logger.warning(
                        f"chbf: Non-finite values in history "
                        f"for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    continue

                self.fitted_sample_ids.add(item_id)

        logger.info(f"chbf fitted for {len(self.fitted_sample_ids)} ch samples")
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """
        Generate forecasts using the Chronos-Bolt model.

        Args:
            batch: Batch of evaluation samples

        Returns:
            Forecast output with predictions for each sample
        """
        # Model is loaded during __init__, so pipeline should never be None
        assert self.pipeline is not None, (
            "Pipeline should be loaded during initialization"
        )

        B = batch.batch_size
        NC = batch.num_correlates
        T = batch.target_length

        forecasts = []

        for i in tqdm(range(B), desc="Predicting ch"):
            row = []

            for j in range(NC):
                sample = batch[i, j]

                # Handle non-forecast samples
                if not sample.forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    continue

                item_id = str(sample.sample_id)

                # Check if this sample was seen during fit
                if item_id not in self.fitted_sample_ids:
                    logger.warning(
                        f"chbf: Sample {i}, correlate {j} "
                        f"(sample_id={item_id}) was not seen during fit"
                    )
                    pred = np.full((T,), np.nan, dtype=np.float32)
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=sample.target_timestamps,
                            values=pred,
                            model_name=self.name,
                        )
                    )
                    continue

                try:
                    # Prepare context data for Chronos-Bolt
                    context = torch.tensor(
                        sample.history_values, dtype=torch.float32
                    )

                    # Get prediction length
                    prediction_length = len(sample.target_timestamps)

                    # Generate forecast using Chronos-Bolt
                    # Chronos-Bolt generates quantile forecasts
                    # forecast shape: [num_series, num_quantiles,
                    # prediction_length]
                    forecast = self.pipeline.predict(
                        context=context,
                        prediction_length=prediction_length,
                    )

                    # Extract median forecast (quantile 0.5)
                    # Assuming the median is in the middle of the quantile
                    # dimension
                    forecast_quantiles = forecast[0].numpy()
                    median_idx = forecast_quantiles.shape[0] // 2
                    pred = forecast_quantiles[median_idx].astype(np.float32)

                except Exception as e:
                    logger.warning(
                        f"chbf: Error predicting for sample {i}, "
                        f"correlate {j} (sample_id={item_id}): {e}"
                    )
                    pred = np.full((T,), np.nan, dtype=np.float32)

                row.append(
                    SingleSampleForecast(
                        sample_id=sample.sample_id,
                        timestamps=sample.target_timestamps,
                        values=pred,
                        model_name=self.name,
                    )
                )

            forecasts.append(row)

        return ForecastOutputFormat(forecasts)
