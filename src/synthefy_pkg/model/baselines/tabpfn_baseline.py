import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from loguru import logger
from tabpfn_time_series import TabPFNMode, TabPFNTimeSeriesPredictor

from synthefy_pkg.model.baselines.forecast_via_diffusion_baseline import (
    MetadataStrategy,
)
from synthefy_pkg.model.forecasting.synthefy_forecasting_model_v1 import (
    SynthefyForecastingModelV1,
)

COMPILE = True


class TabPFNBaseline:
    def __init__(
        self,
        device,
        batch_size,
        pred_len,
        prepare_fn,
        use_metadata=False,
        metadata_strategy=MetadataStrategy.NAIVE,
    ):
        # Set the device to either cpu or gpu
        # set per core batch size
        # horizon_len = pred_lens
        self.device = device
        self.prepare_fn = prepare_fn
        self.batch_size = batch_size
        self.pred_len = pred_len
        self.use_metadata = use_metadata
        self.metadata_strategy = metadata_strategy
        self.decoder_model: Optional[SynthefyForecastingModelV1] = (
            None  # typically not set; only used within TabPFNForecastExperiment
        )

    def batch_to_train_data_multivariate(
        self,
        batch: Dict,
        timestamps: List[datetime],
        add_sequential_feature: bool,
    ):
        effective_batch_size, hist_len, num_channels = batch["history"].shape
        max_sequential_index = 0
        train_data = []
        y_train = batch["history"].cpu().numpy()
        for i in range(effective_batch_size):
            for j in range(hist_len):
                # Create a dictionary for each data point
                data_point = {
                    "id": f"series_{i}",
                    "timestamp": timestamps[j],
                }

                # Add the timeseries columns
                for c in range(num_channels):
                    data_point[f"target_{c}"] = float(y_train[i, j, c])
                if add_sequential_feature:
                    data_point["sequential_feature"] = float(j)
                    max_sequential_index = max(max_sequential_index, j)

                if self.use_metadata:
                    # Add continuous conditions
                    for c in range(
                        batch["history_continuous_cond_input"].shape[2]
                    ):
                        data_point[f"feature_continuous_{c}"] = float(
                            batch["history_continuous_cond_input"][i, j, c]
                            .cpu()
                            .numpy()
                        )

                    for c in range(
                        batch["history_discrete_cond_input"].shape[2]
                    ):
                        data_point[f"feature_discrete_{c}"] = float(
                            batch["history_discrete_cond_input"][i, j, c]
                            .cpu()
                            .numpy()
                        )

                train_data.append(data_point)
        return train_data, max_sequential_index

    def batch_to_test_data_multivariate(
        self,
        batch: Dict,
        future_timestamps: List[datetime],
        add_sequential_feature: bool,
        max_sequential_index: int,
    ):
        effective_batch_size, hist_len, num_channels = batch["history"].shape
        test_data = []
        for i in range(effective_batch_size):
            for j in range(self.pred_len):
                # Create a dictionary for each data point
                data_point = {
                    "id": f"series_{i}",
                    "timestamp": future_timestamps[j],
                }

                # Add the timeseries columns
                for c in range(num_channels):
                    data_point[f"target_{c}"] = float("nan")

                if add_sequential_feature:
                    data_point["sequential_feature"] = float(
                        max_sequential_index + j
                    )

                if self.use_metadata:
                    # Add continuous conditions
                    for c in range(
                        batch["forecast_continuous_cond_input"].shape[2]
                    ):
                        if self.metadata_strategy == MetadataStrategy.NAIVE:
                            data_point[f"feature_continuous_{c}"] = float(
                                batch["forecast_continuous_cond_input"][i, j, c]
                                .cpu()
                                .numpy()
                            )
                        elif self.metadata_strategy == MetadataStrategy.DELETE:
                            data_point[f"feature_continuous_{c}"] = float("nan")
                        elif self.metadata_strategy == MetadataStrategy.REPEAT:
                            raise NotImplementedError(
                                "Repeat metadata strategy not implemented"
                            )
                        elif (
                            self.metadata_strategy
                            == MetadataStrategy.REPEAT_WINDOW
                        ):
                            raise NotImplementedError(
                                "Repeat window metadata strategy not implemented"
                            )
                        else:
                            raise ValueError(
                                f"Invalid metadata strategy: {self.metadata_strategy}"
                            )

                    # Add discrete conditions
                    for c in range(
                        batch["forecast_discrete_cond_input"].shape[2]
                    ):
                        if self.metadata_strategy == MetadataStrategy.NAIVE:
                            data_point[f"feature_discrete_{c}"] = float(
                                batch["forecast_discrete_cond_input"][i, j, c]
                                .cpu()
                                .numpy()
                            )
                        elif self.metadata_strategy == MetadataStrategy.DELETE:
                            data_point[f"feature_discrete_{c}"] = float("nan")
                        elif self.metadata_strategy == MetadataStrategy.REPEAT:
                            raise NotImplementedError(
                                "Repeat metadata strategy not implemented"
                            )
                        elif (
                            self.metadata_strategy
                            == MetadataStrategy.REPEAT_WINDOW
                        ):
                            raise NotImplementedError(
                                "Repeat window metadata strategy not implemented"
                            )
                        else:
                            raise ValueError(
                                f"Invalid metadata strategy: {self.metadata_strategy}"
                            )

                test_data.append(data_point)
        return test_data

    def multivariate_to_univariate_tsdf(
        self, data: List[Dict], num_channels: int
    ):
        data_univariate = {}
        for c in range(num_channels):
            # Create a new list with the target column renamed for this channel
            data_for_channel = []
            for item in data:
                item_copy = item.copy()
                item_copy["target"] = item_copy.pop(f"target_{c}")
                data_for_channel.append(item_copy)

            df = pd.DataFrame(data_for_channel)
            tsdf = TimeSeriesDataFrame.from_data_frame(
                df, id_column="id", timestamp_column="timestamp"
            )

            data_univariate[c] = tsdf
        return data_univariate

    def all_predictions_to_y_pred(
        self,
        all_predictions: Dict[int, pd.DataFrame],
        effective_batch_size: int,
        num_channels: int,
    ):
        # Initialize predictions array with the correct shape
        y_pred = np.zeros((effective_batch_size, num_channels, self.pred_len))

        # Process predictions from each channel
        for channel, predictions_df in all_predictions.items():
            # Iterate through each series
            for series_idx in range(effective_batch_size):
                series_id = f"series_{series_idx}"
                if series_id in predictions_df.index.unique(level=0):
                    # Extract predictions for this series and channel
                    series_preds = predictions_df.loc[series_id].values
                    if len(series_preds.shape) > 1:
                        series_preds = predictions_df.loc[series_id][
                            "target"
                        ].values
                    # Fill in the predictions (up to available length)
                    pred_len = min(len(series_preds), self.pred_len)
                    y_pred[series_idx, channel, :pred_len] = series_preds[
                        :pred_len
                    ]

        return y_pred

    def synthesis_function(self, batch, synthesizer):
        batch = self.prepare_fn(
            train_batch=batch
        )  # SynthefyForecastingModelV1.prepare_training_input()

        effective_batch_size, hist_len, num_channels = batch["history"].shape

        # TODO: Future: Get the actual timestamps from the data.
        base_date = datetime(2023, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(hist_len)]
        future_timestamps = [
            base_date + timedelta(hours=hist_len + i)
            for i in range(self.pred_len)
        ]

        add_sequential_feature = num_channels == 1 and (
            not self.use_metadata
            or (
                self.use_metadata
                and batch["history_continuous_cond_input"].shape[2] == 0
                and batch["history_discrete_cond_input"].shape[2] == 0
            )
        )
        add_sequential_feature = True

        train_data, max_sequential_index = (
            self.batch_to_train_data_multivariate(
                batch, timestamps, add_sequential_feature
            )
        )
        test_data = self.batch_to_test_data_multivariate(
            batch,
            future_timestamps,
            add_sequential_feature,
            max_sequential_index,
        )

        # TabPFN can only generate 1 column at a time; that column is called "target"
        # So we need to generate a new dataframe for each timeseries channel.
        # We retain the other non-target timeseries columns as features.
        train_data_univariate = self.multivariate_to_univariate_tsdf(
            train_data, num_channels
        )
        test_data_univariate = self.multivariate_to_univariate_tsdf(
            test_data, num_channels
        )

        assert len(test_data_univariate) == num_channels
        assert len(train_data_univariate) == num_channels
        assert train_data_univariate.keys() == test_data_univariate.keys()

        all_predictions = {}
        for c in range(num_channels):
            logger.info(f"Predicting channel {c}")
            train_tsdf = train_data_univariate[c]
            test_tsdf = test_data_univariate[c]

            assert "target" in train_tsdf.columns
            assert "target" in test_tsdf.columns

            # Initialize the predictor
            predictor = TabPFNTimeSeriesPredictor(
                tabpfn_mode=TabPFNMode.LOCAL
                if self.device in ("cuda", "cpu", "gpu")
                else TabPFNMode.CLIENT,
            )

            # Time the prediction process
            start_time = time.time()

            # Make predictions
            try:
                predictions = predictor.predict(train_tsdf, test_tsdf)
            except ValueError as e:
                logger.error(f"Error predicting channel {c}: {e}")
                # Typical problem: All features are constant and would have been removed! Unable to predict using TabPFN.
                raise e
            all_predictions[c] = predictions
            end_time = time.time()
            prediction_time = end_time - start_time
            print(f"TabPFN prediction time: {prediction_time:.4f} seconds")

        y_pred = self.all_predictions_to_y_pred(
            all_predictions, effective_batch_size, num_channels
        )

        # concat history and forecast
        concatenated = np.concatenate(
            [batch["history"].cpu().numpy().transpose(0, 2, 1), y_pred],
            axis=2,
        )

        # Output needs to be in the shape (batch, channel_dim, seq_len)
        return {
            "timeseries": concatenated,
            "discrete_conditions": batch["full_discrete_conditions"]
            .cpu()
            .numpy(),
            "continuous_conditions": batch["full_continuous_conditions"]
            .cpu()
            .numpy(),
        }
