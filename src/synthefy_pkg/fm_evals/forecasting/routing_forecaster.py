from functools import partial
from typing import Union

from loguru import logger

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.boosting_forecaster import (
    MitraBoostingForecaster,
    TabPFNBoostingForecaster,
)
from synthefy_pkg.fm_evals.forecasting.chronos_forecaster import (
    ChronosForecaster,
)
from synthefy_pkg.fm_evals.forecasting.mitra_forecaster import (
    MitraForecaster,
)
from synthefy_pkg.fm_evals.forecasting.prophet_forecaster import (
    ProphetForecaster,
)
from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
    TabPFNMultivariateForecaster,
    TabPFNUnivariateForecaster,
)
from synthefy_pkg.fm_evals.forecasting.toto_forecaster import (
    TotoForecaster,
    TotoUnivariateForecaster,
)
from synthefy_pkg.fm_evals.forecasting.utils import split_eval_batch
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)
from synthefy_pkg.fm_evals.formats.metrics import SUPPORTED_METRICS

COMPILE = True

SUPPORTED_FORECASTERS = {
    "tabpfn_univariate": partial(TabPFNUnivariateForecaster),
    "tabpfn_multivariate": partial(
        TabPFNMultivariateForecaster,
        individual_correlate_timestamps=False,
        num_covariate_lags=0,
    ),
    "tabpfn_multivariate_with_lags": partial(
        TabPFNMultivariateForecaster,
        individual_correlate_timestamps=False,
    ),
    "tabpfn_boosting_on_toto": partial(
        TabPFNBoostingForecaster, boosting_models=["toto"]
    ),
    "tabpfn_boosting_on_prophet": partial(
        TabPFNBoostingForecaster, boosting_models=["prophet"]
    ),
    "tabpfn_boosting_on_tabpfn_univariate": partial(
        TabPFNBoostingForecaster, boosting_models=["tabpfn_univariate"]
    ),
    "tabpfn_boosting_on_chronos": partial(
        TabPFNBoostingForecaster, boosting_models=["chronos"]
    ),
    "chronos": partial(ChronosForecaster),
    "prophet": partial(ProphetForecaster),
    "toto_univariate": partial(TotoUnivariateForecaster),
    "toto_multivariate": partial(TotoForecaster),
    "mitra_univariate": partial(MitraForecaster, multivariate=False),
    "mitra_multivariate": partial(MitraForecaster, multivariate=True),
    "mitra_boosting_on_toto": partial(
        MitraBoostingForecaster, boosting_models=["toto"]
    ),
    "mitra_boosting_on_tabpfn_univariate": partial(
        MitraBoostingForecaster, boosting_models=["tabpfn_univariate"]
    ),
    "mitra_boosting_on_tabpfn_multivariate": partial(
        MitraBoostingForecaster, boosting_models=["tabpfn_multivariate"]
    ),
    "mitra_boosting_on_prophet": partial(
        MitraBoostingForecaster, boosting_models=["prophet"]
    ),
    "mitra_boosting_on_chronos": partial(
        MitraBoostingForecaster, boosting_models=["chronos"]
    ),
}


class RoutingForecaster(BaseForecaster):
    def __init__(
        self,
        model_options: list[str] = list(SUPPORTED_FORECASTERS.keys()),
        criterion: str = "mape",
    ):
        self.name = "RoutingForecaster"
        super().__init__(self.name)

        self.model_options = model_options
        self.criterion = criterion

        self._validate_model_options(model_options)
        self._validate_criterion(criterion)

        self.candidate_models = self._get_models(model_options)

    def _validate_model_options(self, model_options: list[str]) -> None:
        for model in model_options:
            if model not in SUPPORTED_FORECASTERS:
                raise ValueError(f"Model {model} not supported")

    def _validate_criterion(self, criterion: str) -> None:
        if criterion not in SUPPORTED_METRICS:
            raise ValueError(f"Criterion {criterion} not supported")

    def _get_model_by_name(
        self, model_name: str
    ) -> Union[list[BaseForecaster], BaseForecaster]:
        return SUPPORTED_FORECASTERS[model_name]()

    def _get_models(self, model_options: list[str]) -> list[BaseForecaster]:
        models: list[BaseForecaster] = []
        for model_name in model_options:
            if isinstance(model_name, list):
                models.extend(self._get_models(model_name))
            else:
                resolved = self._get_model_by_name(model_name)
                if isinstance(resolved, list):
                    models.extend(resolved)
                else:
                    models.append(resolved)
        return models

    @staticmethod
    def _obfuscate_name(name: str) -> str:
        """Obfuscate forecaster name by taking first 2 letters (lowercase)."""
        # Remove any suffix like " (future leak)"
        if " (" in name:
            name = name.split(" (")[0]
        # Remove "Forecaster" suffix
        if name.endswith("Forecaster"):
            name = name[:-10]
        # Handle special cases
        if "TabPFN" in name:
            # Extract "TabPFN" part and take first letter of "Tab" + first letter of "PFN"
            return "tp"
        elif name.startswith("ToTo") or name.startswith("Toto"):
            return "to"
        elif name.startswith("Chronos"):
            return "ch"
        elif name.startswith("Prophet"):
            return "pr"
        elif name.startswith("AutoARIMA"):
            return "au"
        elif name.startswith("LLM"):
            return "ll"
        elif name.startswith("GridICL"):
            return "gr"
        elif name.startswith("CausalImpact"):
            return "ca"
        elif name.startswith("Routing"):
            return "rf"
        # Take first 2 letters and lowercase
        return name[:2].lower() if len(name) >= 2 else name.lower()

    def fit(self, batch: EvalBatchFormat) -> bool:
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        """
        Route the forecast to the best-performing candidate model based on a
        small validation split carved from the end of the history.
        """
        logger.info(
            f"rf: evaluating {len(self.candidate_models)} candidates: "
            + ", ".join([self._obfuscate_name(m.get_name()) for m in self.candidate_models])
        )

        # Get the longest forecast
        max_forecast_length = 0
        for sample in batch.samples:
            for correlate in sample:
                max_forecast_length = max(
                    max_forecast_length,
                    len(correlate.target_timestamps),
                )

        val_rows = max_forecast_length
        try:
            val_batch = split_eval_batch(batch, num_target_rows=val_rows)
        except Exception as e:
            logger.warning(
                f"rf: fallback to first model due to split error: {e}"
            )
            best_model = self.candidate_models[0]
            best_model.fit(batch)
            return best_model._predict(batch)

        # Evaluate each model on the validation batch; lower metric is better
        best_model: BaseForecaster | None = None
        best_score: float = float("inf")

        for model in self.candidate_models:
            try:
                model.fit(val_batch)
                val_output = model.predict(val_batch)
                # Batch-level metric is populated by BaseForecaster.predict
                score = getattr(
                    val_output.metrics, self.criterion, float("nan")
                )
                if score is None or (score != score):  # NaN check
                    score = float("inf")
                logger.info(
                    f"rf: {self._obfuscate_name(model.get_name())} {self.criterion}={score}"
                )
                if score < best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.warning(
                    f"rf: candidate {self._obfuscate_name(model.get_name())} failed on validation: {e}"
                )

        if best_model is None:
            logger.warning(
                "rf: no candidate succeeded on validation; using first model"
            )
            best_model = self.candidate_models[0]

        # Fit the chosen model on the full batch and return its raw predictions
        logger.info(
            f"rf: selected model {self._obfuscate_name(best_model.get_name())} by {self.criterion}={best_score}"
        )
        best_model.fit(batch)
        return best_model._predict(batch)
