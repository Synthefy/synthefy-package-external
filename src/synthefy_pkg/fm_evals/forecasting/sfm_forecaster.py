import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)
from synthefy_pkg.model.foundation_model.base_foundation_forecasting_model import (
    BaseFoundationForecastingModel,
)

# NOTE: TimeSeriesDecoderForecastingFoundationTrainer import removed - unused, causes circular import
from synthefy_pkg.preprocessing.fm_text_embedder import TEXT_EMBEDDING_DIM
from synthefy_pkg.preprocessing.fmv2_preprocess import (
    NORM_RANGES,
    TIMESTAMPS_FEATURES,
    convert_time_to_vector,
)
from synthefy_pkg.utils.synthesis_utils import load_forecast_model

COMPILE = True


class SFMForecaster(BaseForecaster):
    def __init__(
        self,
        model_checkpoint_path: str,
        config_path: str,
        history_length: int,
        forecast_length: int,
    ):
        self.name = "Synthefy Foundation Model"
        super().__init__(self.name)

        self.model_checkpoint_path = model_checkpoint_path
        self.config_path = config_path

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # update dataset_config with passed in parameters
        config_dict = self._update_config_length_parameters(
            config_dict, history_length, forecast_length
        )

        config = Configuration(config=config_dict)

        _ = torch.load(
            self.model_checkpoint_path,
            map_location=torch.device(config.device),
        )

        # update config with training_config from checkpoint
        # config = self._extract_config(checkpoint, config)

        self.model_trainer, _, _ = load_forecast_model(
            config, self.model_checkpoint_path
        )
        self.model_trainer.eval()

        logger.info(
            f"SFMForecaster: loaded model from {self.model_checkpoint_path}"
        )

        self.fitted_sample_ids = set()

    def fit(self, batch: EvalBatchFormat) -> bool:
        B = batch.batch_size
        NC = batch.num_correlates
        self.fitted_sample_ids = set()

        for i in range(B):
            for j in range(NC):
                try:
                    sample_id = str(batch[i, j].sample_id)
                    self.fitted_sample_ids.add(sample_id)
                except (IndexError, AttributeError, TypeError) as e:
                    logger.error(
                        f"Error accessing sample_id for batch[{i}, {j}]: {e}"
                    )
                    return False
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        B = batch.batch_size
        NC = batch.num_correlates
        T = batch.target_length

        forecasts = []
        for i in range(B):
            row = []
            for j in range(NC):
                # create input dataframe for model.predict()
                timestamps = pd.to_datetime(batch[i, j].history_timestamps)
                target_df = pd.DataFrame({"timestamp": timestamps})

                target_columns = [f"target_{j}"]
                target_df[target_columns[0]] = batch[
                    i, j
                ].history_values.tolist()

                covariate_columns = []
                for k in range(NC):
                    if k != j:
                        col_name = f"covariate_{k}"
                        target_df[col_name] = batch[
                            i, k
                        ].history_values.tolist()
                        covariate_columns.append(col_name)

                future_timestamps = [
                    pd.to_datetime(ts) for ts in batch[i, j].target_timestamps
                ]

                forecast_dataset = self.model_trainer.decoder_model.predict(
                    target_df=target_df,
                    covariate_columns=covariate_columns,
                    metadata_dataframes=[],
                    target_columns=target_columns,
                    forecasting_timestamp=datetime.min,  # not used in this implementation
                    future_time_stamps=future_timestamps,
                    timestamp_column="timestamp",
                )

                # Extract forecast values for this correlate
                forecast_values = np.full(T, np.nan, dtype=np.float32)
                col_name = f"target_{j}"

                forecast_group = forecast_dataset.values[0]
                if forecast_group.target_column == col_name:
                    forecast_values = np.array(
                        forecast_group.forecasts, dtype=np.float32
                    )
                else:
                    logger.warning(
                        f"Forecast group target column '{forecast_group.target_column}' "
                        f"doesn't match expected '{col_name}'"
                    )

                row.append(
                    SingleSampleForecast(
                        sample_id=batch[i, j].sample_id,
                        timestamps=batch[i, j].target_timestamps,
                        values=forecast_values,
                        model_name=self.name,
                    )
                )
            forecasts.append(row)
        return ForecastOutputFormat(forecasts)

    def _extract_config(self, checkpoint, config):
        if "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]
            if "training_config" in hyper_params:
                training_config_dict = hyper_params["training_config"]
                for key, value in training_config_dict.items():
                    setattr(config.training_config, key, value)
            else:
                logger.warning(
                    "training_config not found in hyper_parameters, using config file"
                )
        else:
            logger.warning(
                "hyper_parameters not found in checkpoint, using config file"
            )

        return config

    def _update_config_length_parameters(
        self, config_dict, history_length, forecast_length
    ):
        time_series_length = history_length + forecast_length

        config_dict["dataset_config"]["forecast_length"] = forecast_length
        config_dict["dataset_config"]["time_series_length"] = time_series_length
        config_dict["dataset_config"]["timestamp_start_idx"] = 3
        config_dict["dataset_config"]["timestamp_end_idx"] = (
            config_dict["dataset_config"]["timestamp_start_idx"]
            + len(TIMESTAMPS_FEATURES)
            * config_dict["dataset_config"]["time_series_length"]
        )
        config_dict["dataset_config"]["dataset_description_start_idx"] = (
            config_dict["dataset_config"]["timestamp_end_idx"]
        )
        config_dict["dataset_config"]["dataset_description_end_idx"] = (
            config_dict["dataset_config"]["dataset_description_start_idx"] + 384
        )
        config_dict["dataset_config"]["continuous_start_idx"] = config_dict[
            "dataset_config"
        ]["dataset_description_end_idx"]
        config_dict["dataset_config"]["continuous_end_idx"] = (
            config_dict["dataset_config"]["continuous_start_idx"]
            + time_series_length
        )
        config_dict["dataset_config"]["dataset_idx_start_idx"] = config_dict[
            "dataset_config"
        ]["continuous_end_idx"]
        config_dict["dataset_config"]["dataset_idx_end_idx"] = (
            config_dict["dataset_config"]["dataset_idx_start_idx"] + 1
        )
        config_dict["dataset_config"]["metadata_length"] = config_dict[
            "dataset_config"
        ]["dataset_idx_end_idx"]

        return config_dict
