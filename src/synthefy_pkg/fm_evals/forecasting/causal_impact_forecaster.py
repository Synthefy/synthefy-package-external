import warnings

import numpy as np
import pandas as pd
from causalimpact import CausalImpact
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class CausalImpactForecaster(BaseForecaster):
    def __init__(self, nseasons=7, verbose=False):
        super().__init__("CausalImpactForecaster")
        self.nseasons = nseasons
        self.verbose = verbose
        self.fitted_sample_ids = set()
        self.train_data = []  # Store train data for each (batch, correlate) pair
        self.B = 0
        self.NC = 0

    def fit(self, batch: EvalBatchFormat):
        self.B = batch.batch_size
        self.NC = batch.num_correlates
        self.fitted_sample_ids = set()
        self.train_data = []
        for i in range(self.B):
            for j in range(self.NC):
                sample = batch[i, j]
                if not sample.forecast:
                    self.train_data.append(None)
                    continue
                item_id = str(sample.sample_id)
                self.fitted_sample_ids.add(item_id)
                # Build DataFrame for CausalImpact: target and covariates
                data = {}
                data["target"] = sample.history_values.tolist()
                data["timestamp"] = pd.to_datetime(
                    sample.history_timestamps
                ).tolist()
                covariate_names = []
                covariate_values = {}
                for k in range(self.NC):
                    if k == j:
                        continue
                    if not batch[i, k].metadata:
                        continue
                    cov_name = f"correlate_{k}"
                    hist_vals = batch[i, k].history_values.tolist()
                    fut_vals = batch[i, k].target_values.tolist()
                    # Only include if no NaN in both history and future
                    if np.isnan(hist_vals).any() or np.isnan(fut_vals).any():
                        continue
                    data[cov_name] = hist_vals
                    covariate_values[cov_name] = fut_vals
                    covariate_names.append(cov_name)
                columns = ["target"] + covariate_names
                df = pd.DataFrame(data)
                df = df.set_index("timestamp")
                df = df[columns]  # Ensure target is first, covariates in order
                df = df.dropna(
                    subset=["target"]
                )  # CausalImpact requires no NaN in target
                if isinstance(df.index, pd.DatetimeIndex):
                    freq = pd.infer_freq(df.index)
                    if freq:
                        df = df.asfreq(freq)
                if df.empty or len(df) < 2:
                    logger.warning(
                        f"CausalImpactForecaster: Not enough data to fit for sample {i}, correlate {j} (sample_id={item_id})"
                    )
                    self.train_data.append(None)
                    continue
                self.train_data.append((df, covariate_names, covariate_values))
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        if not self.train_data:
            raise ValueError("CausalImpactForecaster model not fitted yet")
        B = batch.batch_size
        NC = batch.num_correlates
        T = batch.target_length
        idx = 0
        forecasts = []
        for i in tqdm(range(B), desc="Predicting ca"):
            row = []
            for j in range(NC):
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
                item_id = str(sample.sample_id)
                if (
                    item_id not in self.fitted_sample_ids
                    or self.train_data[idx] is None
                ):
                    logger.warning(
                        f"CausalImpactForecaster: No fitted data for sample {i}, correlate {j} (sample_id={item_id})"
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
                    idx += 1
                    continue
                history_df, covariate_names, covariate_values = self.train_data[
                    idx
                ]
                # Build prediction (future) DataFrame: target is actual observed, covariates are their values if available
                pred_index = pd.to_datetime(sample.target_timestamps)
                pred_data = {"target": sample.target_values.tolist()}
                for cov_name in covariate_names:
                    pred_data[cov_name] = covariate_values[cov_name]
                pred_df = pd.DataFrame(pred_data, index=pred_index)
                columns = ["target"] + covariate_names
                pred_df = pred_df[columns]
                combined_df = pd.concat([history_df, pred_df])
                combined_df = combined_df[
                    ~combined_df.index.duplicated(keep="first")
                ]
                combined_df = combined_df[
                    columns
                ]  # Ensure target is first, covariates in order
                # Pre/post periods
                pre_period = [history_df.index.min(), history_df.index.max()]
                post_period = [pred_index.min(), pred_index.max()]
                # Fit CausalImpact
                try:
                    if self.verbose:
                        model = CausalImpact(
                            combined_df,
                            pre_period,
                            post_period,
                            model_args={"nseasons": self.nseasons},
                        )
                    else:
                        # Suppress warnings when verbose=False
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", category=FutureWarning
                            )
                            warnings.filterwarnings(
                                "ignore", category=UserWarning
                            )
                            warnings.filterwarnings(
                                "ignore", message=".*ValueWarning.*"
                            )
                            warnings.filterwarnings(
                                "ignore", message=".*DataFrame.applymap.*"
                            )
                            warnings.filterwarnings(
                                "ignore", message=".*Series.__getitem__.*"
                            )
                            warnings.filterwarnings(
                                "ignore",
                                message=".*date index has been provided.*",
                            )
                            warnings.filterwarnings(
                                "ignore",
                                message=".*Keyword arguments have been passed.*",
                            )
                            warnings.filterwarnings(
                                "ignore",
                                message=".*No supported index is available.*",
                            )
                            warnings.filterwarnings(
                                "ignore",
                                message=".*Unknown keyword arguments.*",
                            )
                            model = CausalImpact(
                                combined_df,
                                pre_period,
                                post_period,
                                model_args={"nseasons": self.nseasons},
                            )
                    if model.inferences is not None:
                        preds = model.inferences.loc[pred_index, "preds"]
                        pred = preds.values.astype(np.float32)
                    else:
                        pred = np.full((T,), np.nan, dtype=np.float32)
                except Exception as e:
                    logger.warning(
                        f"CausalImpactForecaster: Exception for sample {i}, correlate {j} (sample_id={item_id}): {e}"
                    )
                    pred = np.full(
                        sample.target_timestamps.shape, np.nan, dtype=np.float32
                    )
                row.append(
                    SingleSampleForecast(
                        sample_id=sample.sample_id,
                        timestamps=sample.target_timestamps,
                        values=pred,
                        model_name=self.name,
                    )
                )
                idx += 1
            forecasts.append(row)
        return ForecastOutputFormat(forecasts)
