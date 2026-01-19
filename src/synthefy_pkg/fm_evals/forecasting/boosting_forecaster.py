import itertools
import sys
from typing import Literal, Optional

import numpy as np
from loguru import logger

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.causal_impact_forecaster import (
    CausalImpactForecaster,
)
from synthefy_pkg.fm_evals.forecasting.chronos_forecaster import (
    ChronosForecaster,
)
from synthefy_pkg.fm_evals.forecasting.mitra_forecaster import MitraForecaster
from synthefy_pkg.fm_evals.forecasting.prophet_forecaster import (
    ProphetForecaster,
)
from synthefy_pkg.fm_evals.forecasting.sarimax_forecaster import (
    SARIMAXForecaster,
)
from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
    TabPFNMultivariateForecaster,
    TabPFNUnivariateForecaster,
)
from synthefy_pkg.fm_evals.forecasting.toto_forecaster import TotoForecaster
from synthefy_pkg.fm_evals.forecasting.utils import fill_nan_values
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)

COMPILE = True


class BaseBoostingForecaster(BaseForecaster):
    def __init__(
        self,
        ensembling_model: Literal["TabPFN", "Mitra"],
        boosting_models: list[str] = ["prophet"],
        **kwargs,
    ):
        """
        Base class for boosting forecasters.
        Propogate params needed for boosting models in through kwargs. (eg. see SARIMAX)

        Args:
            ensembling_model: The model to use for ensembling.
            boosting_models: The models to use for boosting.

        """
        self.name = f"{ensembling_model}BoostingForecaster"
        super().__init__(self.name)
        self.boosting_forecasters = self._instantiate_boosting_models(
            boosting_models,
            **kwargs,
        )
        # Set by subclasses. Kept Optional for type-checkers; guarded at use sites.
        self.ensembling_forecaster: Optional[BaseForecaster] = None

    def _instantiate_boosting_models(
        self, boosting_models: list[str], **kwargs
    ) -> list[BaseForecaster]:
        boosting_forecasters = []
        for model in boosting_models:
            if model == "tabpfn_multivariate":
                boosting_forecasters.append(
                    TabPFNMultivariateForecaster(
                        future_leak=True,
                        individual_correlate_timestamps=False,
                        add_running_index=False,
                        add_calendar_features=True,
                    )
                )
            elif model == "tabpfn_univariate":
                boosting_forecasters.append(TabPFNUnivariateForecaster())
            elif model == "causal_impact":
                boosting_forecasters.append(CausalImpactForecaster())
            elif model == "prophet":
                boosting_forecasters.append(ProphetForecaster())
            elif model == "chronos":
                boosting_forecasters.append(ChronosForecaster())
            elif model == "toto":
                boosting_forecasters.append(TotoForecaster())
            elif model == "sarimax":
                assert "seasonal_period" in kwargs, (
                    "seasonal_period must be provided for SARIMAX"
                )
                seasonal_period = kwargs["seasonal_period"]
                boosting_forecasters.append(
                    SARIMAXForecaster(
                        seasonal_period=seasonal_period,
                        future_leaked=True,
                    )
                )

        return boosting_forecasters

    def fit(self, batch: EvalBatchFormat) -> bool:
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """
        1. Chop the batch into prehistory and history depending on the cutoff date or rows
        2. So now, we have prehistory, history, and target.
        3. for each of the boosting models,
            3.1 Use prehistory to forecast history
            3.2 Use (history + prehistory) to forecast target
        4. Add history forecasts as a correlate for history
        5. add target forecasts as a correlate for target
        6. Use TabPFN to fit on the history and predict forecasts.
        """
        all_output_rows = []
        for batch_idx in range(batch.batch_size):
            single_sample_batch = EvalBatchFormat([batch.samples[batch_idx]])

            num_original_correlates = single_sample_batch.num_correlates

            (
                sample_ids,
                history_timestamps,
                history_values,
                target_timestamps,
                target_values,
            ) = single_sample_batch.to_arrays(targets_only=False)

            # Chop arrays in half along the second axis (axis=1)
            # Get the midpoint for the second axis
            mid = history_timestamps.shape[2] // 2

            prehistory_timestamps, history_timestamps = (
                history_timestamps[:, :, :mid],
                history_timestamps[:, :, mid:],
            )
            prehistory_values, history_values = (
                history_values[:, :, :mid],
                history_values[:, :, mid:],
            )

            forecast_samples = np.full(
                (history_timestamps.shape[0], history_timestamps.shape[1]),
                False,
            )
            for b in range(single_sample_batch.batch_size):
                for nc in range(single_sample_batch.num_correlates):
                    if single_sample_batch.samples[b][nc].forecast:
                        forecast_samples[b, nc] = True

            # Check for inf or nan in prehistory and history values
            if np.any(np.isnan(prehistory_values)):
                logger.info("Prehistory values contain nan, filling with mean")
                prehistory_values = fill_nan_values(prehistory_values)
            if np.any(np.isnan(history_values)):
                logger.info("History values contain nan, filling with mean")
                history_values = fill_nan_values(history_values)
            if np.any(np.isinf(prehistory_values)):
                logger.info(
                    "Prehistory values contain inf, not doing anything for now"
                )
            if np.any(np.isinf(history_values)):
                logger.info(
                    "History values contain inf, not doing anything for now"
                )

            for idx in range(prehistory_values.shape[1]):
                # Check if prehistory values for this correlate are constant
                if np.allclose(
                    prehistory_values[0, idx], prehistory_values[0, idx][0]
                ):
                    logger.warning(
                        f"Prehistory values for correlate {idx} are constant"
                    )
                    # Add small amount of noise to prevent constant values
                    noise_scale = 1e-3
                    prehistory_values = prehistory_values + np.random.normal(
                        0, noise_scale, prehistory_values.shape
                    )
                    history_values = history_values + np.random.normal(
                        0, noise_scale, history_values.shape
                    )

            prehistory_history_eval_batch = EvalBatchFormat.from_arrays(
                sample_ids=sample_ids,
                history_timestamps=prehistory_timestamps,
                history_values=prehistory_values,
                target_timestamps=history_timestamps,
                target_values=history_values,
                forecast=forecast_samples,
                metadata=True,  # Since this is technically still part of the history, we can metadata / leak whatever we want
                leak_target=True,  # Since this is technically still part of the history, we can metadata / leak whatever we want
            )

            # Use prehistory to forecast the history
            logger.info("Using prehistory to forecast history")
            history_forecast_arrays = []
            history_timestamp_arrays = []

            for model in self.boosting_forecasters:
                status = model.fit(prehistory_history_eval_batch)

                if status:
                    forecast = model._predict(prehistory_history_eval_batch)
                    _, timestamps, forecast_values = forecast.to_arrays()
                    history_forecast_arrays.append(forecast_values)
                    history_timestamp_arrays.append(timestamps)
                else:
                    history_forecast_arrays.append(
                        np.zeros_like(history_values)
                    )
                    history_timestamp_arrays.append(history_timestamps)

            # Use (history + prehistory) to forecast the target (normal forecasting with boosting models)
            logger.info("Using (history + prehistory) to forecast target")
            target_forecast_arrays = []
            target_timestamp_arrays = []
            boosting_forecast_ids = []
            for model in self.boosting_forecasters:
                model.fit(single_sample_batch)
                forecast = model.predict(single_sample_batch)
                this_sample_ids, timestamps, forecast_values = (
                    forecast.to_arrays()
                )
                target_forecast_arrays.append(forecast_values)
                target_timestamp_arrays.append(timestamps)
                boosting_forecast_ids.append(this_sample_ids)

            num_targets = history_values.shape[1]
            history_timestamps = np.concatenate(
                [history_timestamps] + history_timestamp_arrays, axis=1
            )
            history_values = np.concatenate(
                [history_values] + history_forecast_arrays, axis=1
            )
            target_timestamps = np.concatenate(
                [target_timestamps] + target_timestamp_arrays, axis=1
            )
            target_values = np.concatenate(
                [target_values] + target_forecast_arrays, axis=1
            )
            augmented_sample_ids = np.concatenate(
                [sample_ids] + boosting_forecast_ids, axis=1
            )

            forecast_mask = np.full(
                (history_timestamps.shape[0], history_timestamps.shape[1]),
                False,
            )
            metadata_mask = np.full(
                (history_timestamps.shape[0], history_timestamps.shape[1]),
                True,
            )
            leak_target_mask = np.full(
                (history_timestamps.shape[0], history_timestamps.shape[1]),
                True,
            )
            for b in range(augmented_sample_ids.shape[0]):
                for nc in range(augmented_sample_ids.shape[1]):
                    if nc < num_targets:
                        forecast_mask[b, nc] = single_sample_batch.samples[b][
                            nc
                        ].forecast

                        metadata_mask[b, nc] = single_sample_batch.samples[b][
                            nc
                        ].metadata

                        leak_target_mask[b, nc] = single_sample_batch.samples[
                            b
                        ][nc].leak_target

            # Check for inf or larger than float64 in history values
            if np.any(np.isinf(history_values)) or np.any(
                np.abs(history_values) > np.finfo(np.float64).max
            ):
                logger.warning(
                    "History values contain inf or are larger than float64"
                )
                history_values = np.clip(
                    history_values,
                    -np.finfo(np.float64).max,
                    np.finfo(np.float64).max,
                )

            if np.any(np.isnan(target_values)):
                logger.warning("Target values contain nan, filling with mean")
                target_values = fill_nan_values(target_values)
            if np.any(np.isinf(target_values)):
                logger.warning(
                    "Target values contain inf, not doing anything for now"
                )
            if np.any(np.isnan(history_values)):
                logger.warning("History values contain nan, filling with mean")
                history_values = fill_nan_values(history_values)
            if np.any(np.isinf(history_values)):
                logger.warning(
                    "History values contain inf, not doing anything for now"
                )

            augmented_eval_batch = EvalBatchFormat.from_arrays(
                sample_ids=augmented_sample_ids,
                history_timestamps=history_timestamps,
                history_values=history_values,
                target_timestamps=target_timestamps,
                target_values=target_values,
                forecast=forecast_mask,
                metadata=leak_target_mask,  # This is not a bug. For boosting, we only use historical metadata if it's allowed to leak
                leak_target=leak_target_mask,
            )

            ensembler = self.ensembling_forecaster
            if ensembler is None:
                raise RuntimeError(
                    "ensembling_forecaster must be set by subclasses of BaseBoostingForecaster"
                )

            try:
                ensembler.fit(augmented_eval_batch)
                forecast = ensembler._predict(augmented_eval_batch)
            except Exception as e:
                logger.error(
                    f"Error fitting or predicting with {ensembler.name}: {e}"
                )
                import traceback

                traceback.print_exc()
                sys.exit(1)

            x = [row[0:num_original_correlates] for row in forecast.forecasts]

            for row in x:
                for sample in row:
                    sample.model_name = self.name

            all_output_rows.append(x)

        return ForecastOutputFormat(list(itertools.chain(*all_output_rows)))


class TabPFNBoostingForecaster(BaseBoostingForecaster):
    def __init__(
        self,
        boosting_models: list[str] = ["prophet"],
        future_leak: bool = False,
        **kwargs,
    ):
        super().__init__(
            ensembling_model="TabPFN",
            boosting_models=boosting_models,
            **kwargs,
        )
        self.ensembling_forecaster = TabPFNMultivariateForecaster(
            future_leak=future_leak,
            individual_correlate_timestamps=False,
            add_running_index=True,
            add_calendar_features=True,
        )


class MitraBoostingForecaster(BaseBoostingForecaster):
    def __init__(
        self,
        boosting_models: list[str] = ["prophet"],
        future_leak: bool = False,
        **kwargs,
    ):
        super().__init__(
            ensembling_model="Mitra",
            boosting_models=boosting_models,
            **kwargs,
        )
        self.ensembling_forecaster = MitraForecaster(
            multivariate=True,
            future_leak=future_leak,
        )
