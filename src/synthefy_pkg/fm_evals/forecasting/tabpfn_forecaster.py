import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from loguru import logger
from tabpfn_time_series import (
    DefaultFeatures,
    FeatureTransformer,
    TabPFNMode,
    TabPFNTimeSeriesPredictor,
)
from tqdm import tqdm

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.features import (
    make_covariate_lag_feature,
    make_ema_features,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class TabPFNUnivariateForecaster(BaseForecaster):
    def __init__(
        self,
        add_running_index: bool = True,
        add_calendar_features: bool = True,
    ):
        super().__init__("TabPFNUnivariateForecaster")
        self.predictors = []  # One predictor per (batch, correlate) pair
        self.ts_train_data_for_fit = []  # Store train data for each (batch, correlate) pair
        self.selected_features = []
        if add_running_index:
            self.selected_features.append(DefaultFeatures.add_running_index)
        if add_calendar_features:
            self.selected_features.append(DefaultFeatures.add_calendar_features)
        self.B = 0
        self.NC = 0
        self.fitted_sample_ids = set()

    def fit(self, batch: EvalBatchFormat):
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.predictors = []
        self.ts_train_data_for_fit = []
        self.fitted_sample_ids = set()
        for i in tqdm(range(self.B), desc="Fitting tp"):
            for j in tqdm(
                range(self.NC), desc="Fitting tp"
            ):
                sample = batch[i, j]
                if not sample.forecast:
                    self.predictors.append(None)
                    self.ts_train_data_for_fit.append(None)
                    continue
                history_timestamps = sample.history_timestamps
                history_values = sample.history_values
                item_id = str(sample.sample_id)
                self.fitted_sample_ids.add(item_id)
                train_df = pd.DataFrame(
                    {
                        "target": history_values,
                        "item_id": [item_id] * len(history_timestamps),
                        "timestamp": pd.to_datetime(history_timestamps),
                    }
                )
                train_df = train_df.set_index(["item_id", "timestamp"])
                train_df = train_df.dropna(subset=["target"])
                if len(train_df) > 10000:
                    logger.warning(
                        f"tp: train_df for sample {i}, correlate {j} (sample_id={item_id}) has {len(train_df)} rows, truncating to last 10000 rows."
                    )
                    train_df = train_df.iloc[-10000:]

                if train_df.empty:
                    self.predictors.append(None)
                    self.ts_train_data_for_fit.append(None)
                    logger.warning(
                        f"tp: No data to fit for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    continue
                elif len(train_df) == 1:
                    self.predictors.append(None)
                    self.ts_train_data_for_fit.append(None)
                    logger.warning(
                        f"tp: Only one data point to fit for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    continue

                ts_train_data = TimeSeriesDataFrame(train_df).sort_index()
                predictor = TabPFNTimeSeriesPredictor(
                    tabpfn_mode=TabPFNMode.LOCAL
                )
                self.predictors.append(predictor)
                self.ts_train_data_for_fit.append(ts_train_data)

        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        if not self.predictors or not self.ts_train_data_for_fit:
            raise ValueError("tp model not fitted yet")
        B = batch.batch_size
        NC = batch.num_correlates
        T = batch.target_length
        idx = 0

        # Check that all sample_ids in predict are in fitted_sample_ids
        for i in range(B):
            for j in range(NC):
                sample = batch[i, j]
                if not sample.forecast:
                    continue
                pred_id = str(sample.sample_id)
                if pred_id not in self.fitted_sample_ids:
                    raise ValueError(
                        f"tp: Attempting to predict for sample_id '{pred_id}' which was not seen during fit."
                    )

        forecasts = []
        for i in tqdm(range(B), desc="Predicting tp"):
            row = []
            for j in tqdm(
                range(NC), desc="Predicting tp"
            ):
                sample = batch[i, j]
                if not sample.forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    idx += 1
                    continue
                if self.predictors[idx] is None:
                    # If we don't have a predictor for this sample, set the prediction to nan
                    pred = np.full((T,), np.nan, dtype=np.float32)
                else:
                    item_id = str(batch[i, j].sample_id)
                    target_timestamps = pd.to_datetime(
                        batch[i, j].target_timestamps
                    )
                    test_df = pd.DataFrame(
                        {
                            "target": [np.nan] * len(target_timestamps),
                            "item_id": [item_id] * len(target_timestamps),
                            "timestamp": target_timestamps,
                        }
                    )
                    test_df = test_df.set_index(["item_id", "timestamp"])
                    if test_df.empty:
                        row.append(
                            SingleSampleForecast(
                                sample_id=batch[i, j].sample_id,
                                timestamps=np.array([]),
                                values=np.array([], dtype=np.float32),
                                model_name=self.name,
                            )
                        )
                        idx += 1
                        continue
                    ts_test_data = TimeSeriesDataFrame(test_df).sort_index()
                    train_with_features, test_with_features = (
                        FeatureTransformer.add_features(
                            self.ts_train_data_for_fit[idx],
                            ts_test_data,
                            self.selected_features,
                        )
                    )
                    pred_df = self.predictors[idx].predict(
                        train_with_features, test_with_features
                    )
                    pred = pred_df["target"].values
                row.append(
                    SingleSampleForecast(
                        sample_id=batch[i, j].sample_id,
                        timestamps=batch[i, j].target_timestamps,
                        values=pred,
                        model_name=self.name,
                    )
                )
                idx += 1

            forecasts.append(row)
        return ForecastOutputFormat(forecasts)


class TabPFNMultivariateForecaster(BaseForecaster):
    def __init__(
        self,
        future_leak: bool = False,
        individual_correlate_timestamps: bool = True,
        num_covariate_lags: int = 5,
        add_running_index: bool = True,
        add_calendar_features: bool = True,
        add_ema_features: bool = False,
    ):
        if future_leak:
            self.name = "tp (future leak)"
        else:
            self.name = "tp"
        super().__init__(self.name)

        self.predictors = []  # One predictor per (batch, correlate) pair
        self.ts_train_data_for_fit = []  # Store train data for each (batch, correlate) pair

        self.selected_features = []
        if add_ema_features:
            self.selected_features.append(make_ema_features())
        if num_covariate_lags > 0:
            self.selected_features.append(
                make_covariate_lag_feature(
                    lags=list(range(1, num_covariate_lags + 1))
                )
            )
        if add_running_index:
            self.selected_features.append(DefaultFeatures.add_running_index)
        if add_calendar_features:
            self.selected_features.append(DefaultFeatures.add_calendar_features)

        self.B = 0
        self.NC = 0
        self.fitted_sample_ids = set()
        self.future_leak = future_leak
        self.individual_correlate_timestamps = individual_correlate_timestamps

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
        elif name.startswith("ToTo"):
            return "to"
        # Take first 2 letters and lowercase
        return name[:2].lower() if len(name) >= 2 else name.lower()

    def fit(self, batch: EvalBatchFormat):
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.predictors = []
        self.ts_train_data_for_fit = []
        self.fitted_sample_ids = set()
        for i in tqdm(range(self.B), desc=f"Fitting {self._obfuscate_name(self.name)}"):
            for j in tqdm(range(self.NC), desc=f"Fitting {self._obfuscate_name(self.name)}"):
                sample = batch[
                    i, j
                ]  # The batch, correlate pair we are constructing the data for

                # If the sample is not to be forecasted, we don't do anything
                if not sample.forecast:
                    self.predictors.append(None)
                    self.ts_train_data_for_fit.append(None)
                    continue

                # Add the sample_id to the set of seen ids
                item_id = str(sample.sample_id)
                self.fitted_sample_ids.add(item_id)

                # Build multivariate train DataFrame
                data = {}  # The data we are going to use to fit the model

                # Target correlate (j)
                data["target"] = batch[i, j].history_values.tolist()
                data["timestamp"] = pd.to_datetime(
                    batch[i, j].history_timestamps
                ).tolist()

                # Other correlates (k ≠ j)
                for k in range(self.NC):
                    if k == j:
                        continue

                    # If the correlate is only a target (a target other than j)
                    # We should not use it as metadata for target j
                    if not batch[i, k].metadata:
                        continue

                    # if the correlate has no useful data, skip it
                    if pd.isna(batch[i, k].history_values).all():
                        continue

                    # Add the correlate to the data
                    # With its own timestamp column if we are using individual timestamps
                    data[f"correlate_{k}"] = batch[i, k].history_values.tolist()
                    if self.individual_correlate_timestamps:
                        data[f"timestamp_{k}"] = (
                            pd.to_datetime(
                                batch[i, k].history_timestamps
                            ).astype(np.int64)
                            / 1e9
                        ).tolist()

                # TabPFN requires an item_id column
                data["item_id"] = [item_id] * len(
                    batch[i, j].history_timestamps
                )

                # Convert the data to a pandas DataFrame
                train_df = pd.DataFrame(data)
                train_df = train_df.set_index(["item_id", "timestamp"])
                train_df = train_df.dropna(subset=["target"])

                # Truncate the data to the last 10000 rows if it's too big
                if len(train_df) > 10000:
                    logger.warning(
                        f"{self._obfuscate_name(self.name)}: train_df for sample {i}, correlate {j} (sample_id={item_id}) has {len(train_df)} rows, truncating to last 10000 rows."
                    )
                    train_df = train_df.iloc[-10000:]

                # If the data is empty, we can't fit the model
                if train_df.empty:
                    self.predictors.append(None)
                    self.ts_train_data_for_fit.append(None)
                    logger.warning(
                        f"{self._obfuscate_name(self.name)}: No data to fit for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    continue

                # If the data has only one row, we can't fit the model
                elif len(train_df) == 1:
                    self.predictors.append(None)
                    self.ts_train_data_for_fit.append(None)
                    logger.warning(
                        f"{self._obfuscate_name(self.name)}: Only one data point to fit for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    continue

                ts_train_data = TimeSeriesDataFrame(train_df).sort_index()
                predictor = TabPFNTimeSeriesPredictor(
                    tabpfn_mode=TabPFNMode.LOCAL,
                )
                self.predictors.append(predictor)
                self.ts_train_data_for_fit.append(ts_train_data)

        return True

    def _validate_seen_sample_ids(self, batch: EvalBatchFormat):
        B = batch.batch_size
        NC = batch.num_correlates

        # Check that all sample_ids in predict are in fitted_sample_ids
        for i in range(B):
            for j in range(NC):
                sample = batch[i, j]
                if not sample.forecast:
                    continue
                pred_id = str(sample.sample_id)
                if pred_id not in self.fitted_sample_ids:
                    raise ValueError(
                        f"{self.name}: Attempting to predict for sample_id '{pred_id}' which was not seen during fit."
                    )

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        if not self.predictors or not self.ts_train_data_for_fit:
            raise ValueError(f"{self.name} model not fitted yet")

        self._validate_seen_sample_ids(batch)

        B = batch.batch_size
        NC = batch.num_correlates
        T = batch.target_length
        idx = 0

        forecasts = []
        for i in tqdm(range(B), desc=f"Predicting {self._obfuscate_name(self.name)} (batches)"):
            row = []
            for j in tqdm(
                range(NC), desc=f"Predicting {self._obfuscate_name(self.name)} (correlates)"
            ):
                sample = batch[i, j]

                # If the sample is not to be forecasted, we don't do anything
                if not sample.forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    idx += 1
                    continue

                # If we don't have a predictor for this sample, set the prediction to nan
                if self.predictors[idx] is None:
                    pred = np.full((T,), np.nan, dtype=np.float32)
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=batch[i, j].target_timestamps,
                            values=pred,
                            model_name=self.name,
                        )
                    )
                    idx += 1
                    continue

                # Get the item_id and target_timestamps
                item_id = str(batch[i, j].sample_id)
                target_timestamps = pd.to_datetime(
                    batch[i, j].target_timestamps
                ).tolist()

                # Build multivariate test DataFrame
                float_data = {}

                # The target is always nan for the target correlate in TabPFN
                float_data["target"] = [np.nan] * len(target_timestamps)

                # Other correlates (k ≠ j)
                for k in range(self.NC):
                    if k == j:
                        continue

                    # If the correlate is not to be used as metadata, skip it
                    if not batch[i, k].metadata:
                        continue

                    leak = batch[i, k].leak_target

                    # If self.future_leak is True, then we IGNORE allowable vs not allowable leakage
                    # If leak := batch[i, k].leak_target, then that construes allowable leakage
                    # even for the multivariate (non-future-leaked) case.
                    if self.future_leak or leak:
                        float_data[f"correlate_{k}"] = batch[
                            i, k
                        ].target_values.tolist()
                    else:
                        float_data[f"correlate_{k}"] = [np.nan] * len(
                            target_timestamps
                        )

                    # If we are using individual timestamps for each correlate, add the timestamp column
                    if self.individual_correlate_timestamps:
                        float_data[f"timestamp_{k}"] = (
                            pd.to_datetime(
                                batch[i, k].target_timestamps
                            ).astype(np.int64)
                            / 1e9
                        ).tolist()

                # Convert the data to a pandas DataFrame
                test_df = pd.DataFrame(float_data)

                # Add the item_id and timestamp columns
                test_df["item_id"] = [str(item_id)] * len(target_timestamps)
                test_df["timestamp"] = target_timestamps
                test_df = test_df.set_index(["item_id", "timestamp"])

                # If the data is empty, we can't predict
                if test_df.empty:
                    row.append(
                        SingleSampleForecast(
                            sample_id=batch[i, j].sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    idx += 1
                    continue

                # Convert the data to a TimeSeriesDataFrame
                ts_test_data = TimeSeriesDataFrame(test_df).sort_index()

                # Add the features to the data
                train_with_features, test_with_features = (
                    FeatureTransformer.add_features(
                        self.ts_train_data_for_fit[idx],
                        ts_test_data,
                        self.selected_features,
                    )
                )

                # Predict the target
                pred_df = self.predictors[idx].predict(
                    train_with_features, test_with_features
                )
                pred = pred_df["target"].values

                # Add the prediction to the row
                row.append(
                    SingleSampleForecast(
                        sample_id=batch[i, j].sample_id,
                        timestamps=batch[i, j].target_timestamps,
                        values=pred,
                        model_name=self.name,
                    )
                )
                idx += 1
            forecasts.append(row)

        return ForecastOutputFormat(forecasts)
