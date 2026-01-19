from datetime import datetime
from typing import Dict, List, Optional

import holidays
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange, repeat
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from synthefy_pkg.app.data_models import (
    ConfidenceInterval,
    ForecastDataset,
    ForecastGroup,
)
from synthefy_pkg.app.utils.api_utils import (
    format_timestamp_with_optional_fractional_seconds,
)
from synthefy_pkg.app.utils.fm_model_utils import post_process_forecast_values
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.architectures.decoder_utils import PositionalEmbedding
from synthefy_pkg.model.foundation_model.utils import (
    NULL_TOKEN,
    continuous_to_token,
    generate_target_mask,
    get_large_negative_number,
    obtain_mask,
)
from synthefy_pkg.preprocessing.fmv2_preprocess import (
    NORM_RANGES,
    TIMESTAMPS_FEATURES,
    convert_time_to_vector,
)

batch_idx = 0


class BaseFoundationForecastingModel(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config

        self.target_masking_schemes = (
            config.foundation_model_config.masking_schemes
        )  # block, train_test, target_row, random
        self.attention_masking_scheme = (
            config.foundation_model_config.attention_masking_scheme
        )  # causal, all_to_all, train_test_block
        self.time_columns = (
            config.dataset_config.time_columns
        )  # the number of correlates that encode time
        self.row_use_train_test = config.foundation_model_config.row_use_lengths

        # toggle ablations
        self.random_vector_instead_text = (
            config.foundation_model_config.random_vector_instead_text
        )
        self.mask_timestamp = config.foundation_model_config.mask_timestamp
        self.position_embedding = (
            config.foundation_model_config.position_embedding
        )

        # constrct the text description random vector
        if self.random_vector_instead_text:
            self.text_description_random_vector = torch.randn(
                config.dataset_config.num_datasets,
                config.dataset_config.text_embedding_dim,
            )

        # TimesFM Decoder model parameters
        self.decoder_input_patch_len = (
            config.foundation_model_config.decoder_input_patch_len
        )
        self.decoder_output_patch_len = (
            config.foundation_model_config.decoder_output_patch_len
        )
        self.decoder_num_layers = (
            config.foundation_model_config.decoder_num_layers
        )
        self.decoder_model_dims = (
            config.foundation_model_config.decoder_model_dims
        )
        self.decoder_num_heads = (
            config.foundation_model_config.decoder_num_heads
        )

        # Forecasting configuration
        self.horizon_len = config.dataset_config.forecast_length
        self.context_len = (
            config.dataset_config.time_series_length - self.horizon_len
        )
        self.time_series_length = config.dataset_config.time_series_length

        # Dataset and device specific configuration
        self.device = config.device
        self.batch_size = config.dataset_config.batch_size
        self.window_size = config.dataset_config.time_series_length
        self.dataset_config = config.dataset_config
        self.use_metadata = config.foundation_model_config.use_metadata
        self.num_correlates = config.dataset_config.num_correlates

        self.target_mask_ratio = config.training_config.target_mask_ratio
        self.description_mask_ratio = (
            config.training_config.description_mask_ratio
        )
        self.block_target_mask_range = (
            config.foundation_model_config.block_target_mask_range
        )  # default is 0
        self.block_target_mask_mean = (
            config.foundation_model_config.block_target_mask_mean
        )  # default is 0
        self.block_mask_num = (
            config.foundation_model_config.block_mask_num
        )  # default is 0
        self.block_mask_every = config.foundation_model_config.block_mask_every
        self.row_mask_ratio = config.foundation_model_config.row_mask_ratio
        self.row_mask_min = config.foundation_model_config.row_mask_min

        if self.position_embedding != "none":
            if self.position_embedding == "all":
                self.position_embedder = PositionalEmbedding(
                    d_model=self.decoder_model_dims,
                    max_len=self.window_size * self.num_correlates,
                )
            elif self.position_embedding == "correlate":
                self.position_embedder = PositionalEmbedding(
                    d_model=self.decoder_model_dims,
                    max_len=self.window_size,
                )

        self.using_synthetic_data = config.dataset_config.using_synthetic_data
        # if using synthetic data, we need to set the prepare_training_input function
        if self.using_synthetic_data:
            self.prepare_training_input = (
                self.prepare_training_input_for_synthetic_data
            )
        else:
            self.prepare_training_input = (
                self.prepare_training_input_for_real_data
            )

    def _prepare_column_identifiers(
        self, batch_size, num_correlates, time_series_length
    ):
        column_identifiers = torch.eye(num_correlates).to(self.device).float()
        column_identifiers = repeat(
            column_identifiers,
            "nc n -> (nc t) n",
            t=time_series_length,
        )
        column_identifiers = repeat(
            column_identifiers,
            "(nc t) n -> b (nc t) n",
            b=batch_size,
            nc=num_correlates,
            t=time_series_length,
        )
        return column_identifiers

    def prepare_training_input_for_synthetic_data(
        self, train_batch, *args, **kwargs
    ):
        X = train_batch["X"]  # B, T, C
        y = train_batch["y"]  # B, T
        d = train_batch["d"].clone()  # B

        invalid_mask = torch.zeros(X.shape[0], X.shape[1], X.shape[2] + 1).to(
            self.device
        )
        all_continuous = torch.zeros(X.shape[0], X.shape[1], X.shape[2] + 1).to(
            self.device
        )
        for bidx in range(X.shape[0]):
            valid_inputs = d[bidx].clone().item()
            d[bidx] = d[bidx] + 1
            invalid_mask[bidx, :, valid_inputs + 1 :] = 1
            all_continuous[bidx, :, :valid_inputs] = X[bidx, :, :valid_inputs]
            all_continuous[bidx, :, valid_inputs] = y[bidx, :]
        invalid_mask = rearrange(invalid_mask, "b t nc -> b nc t")
        invalid_mask = rearrange(invalid_mask, "b nc t -> b (nc t)")
        invalid_mask = invalid_mask.bool()
        all_continuous = rearrange(all_continuous, "b t nc -> b nc t")
        all_continuous = rearrange(
            all_continuous,
            "b nc t -> b nc t 1",
        )

        batch_size = all_continuous.shape[0]
        num_correlates = all_continuous.shape[1]
        time_series_length = all_continuous.shape[2]

        all_descriptions = torch.ones(
            batch_size,
            num_correlates,
            self.dataset_config.dataset_description_end_idx
            - self.dataset_config.dataset_description_start_idx,
        )
        all_descriptions = repeat(
            all_descriptions,
            "b nc textf -> b nc t textf",
            t=time_series_length,
        )

        # create a fake set of timestamps using pandas datetime with hours minutes seconds
        timestamps = pd.date_range(
            start="2010-01-01", periods=time_series_length, freq="W", tz="UTC"
        )
        # convert to np.datetime64
        timestamps = np.array(timestamps, dtype=np.datetime64)

        timestamp_embeddings = np.zeros(
            (len(timestamps), len(TIMESTAMPS_FEATURES))
        )
        pd_timestamps = pd.Series(timestamps)
        us_holidays = holidays.country_holidays("US")
        timestamp_embeddings = convert_time_to_vector(
            timestamp_embeddings,
            TIMESTAMPS_FEATURES,
            NORM_RANGES,
            pd_timestamps,
            us_holidays,
            0,
            np.array(pd_timestamps.isna().values),
        )
        all_timestamps = (
            torch.tensor(timestamp_embeddings)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, num_correlates, 1, 1)
            .to(self.device)
        )

        all_dataset_ids = torch.zeros(batch_size, num_correlates).to(
            self.device
        )

        all_timestamps_mask = rearrange(
            obtain_mask(all_timestamps, category="continuous"),
            "b nc t -> b (nc t)",
        ).to(self.device)  # B X NUM_CORRELATES X TIME_SERIES_LENGTH
        all_descriptions_mask = rearrange(
            obtain_mask(all_descriptions, category="textual"),
            "b nc t -> b (nc t)",
        ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        all_continuous_mask = rearrange(
            obtain_mask(all_continuous, category="continuous"),
            "b nc t -> b (nc t)",
        ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        mask = torch.logical_or(
            torch.logical_or(all_continuous_mask, all_timestamps_mask),
            all_descriptions_mask,
        ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        mask = torch.logical_or(mask, invalid_mask).to(
            self.device
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        # flatten individual token elements
        all_timestamps = rearrange(
            all_timestamps, "b nc t tf -> b (nc t) tf"
        ).to(self.device)
        all_descriptions = rearrange(
            all_descriptions, "b nc t textf -> b (nc t) textf"
        ).to(self.device)
        all_continuous = rearrange(
            all_continuous, "b nc t cf -> b (nc t) cf"
        ).to(self.device)

        # now we make the elements have null token values for the mask locations
        # a timestamp null token is -1 for all timestamp features
        all_timestamps[mask] = -1
        # a textual null token is 0 for all textual features
        all_descriptions[mask] = 0
        # a continuous null token is -1 for all continuous features
        all_continuous[mask] = -1

        # convert the continuous values to tokens and remove the extra dimension
        all_continuous_tokens = continuous_to_token(all_continuous, mask).to(
            self.device
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH) X 9

        target_mask = (
            torch.rand(batch_size, num_correlates, time_series_length)
            < self.target_mask_ratio
        )
        target_mask = rearrange(target_mask, "b nc t -> b (nc t)")
        target_mask = target_mask.bool().to(self.device)
        target_mask = torch.logical_and(target_mask, ~mask).to(
            self.device
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        # create the column indicators
        column_identifiers = self._prepare_column_identifiers(
            all_timestamps.shape[0],
            num_correlates,
            time_series_length,
        ).to(self.device)

        description_mask_flag = torch.rand(1)[0] < self.description_mask_ratio

        if self.dataset_config.train_as_univariate_model:
            all_timestamps = rearrange(
                all_timestamps, "b (nc t) tf -> (b nc) t tf", nc=num_correlates
            )
            all_descriptions = rearrange(
                all_descriptions,
                "b (nc t) textf -> (b nc) t textf",
                nc=num_correlates,
            )
            all_continuous = rearrange(
                all_continuous, "b (nc t) cf -> (b nc) t cf", nc=num_correlates
            )
            all_continuous_tokens = rearrange(
                all_continuous_tokens,
                "b (nc t) cf -> (b nc) t cf",
                nc=num_correlates,
            )
            mask = rearrange(mask, "b (nc t) -> (b nc) t", nc=num_correlates)
            target_mask = rearrange(
                target_mask, "b (nc t) -> (b nc) t", nc=num_correlates
            )
            column_identifiers = rearrange(
                column_identifiers,
                "b (nc t) ncmax -> (b nc) t ncmax",
                nc=num_correlates,
            )
            all_dataset_ids = rearrange(
                all_dataset_ids, "b nc -> (b nc)", nc=num_correlates
            )
            all_dataset_ids = all_dataset_ids.unsqueeze(-1)

            all_invalid_mask_sum = torch.sum(mask, dim=-1)
            all_valid_rows = torch.where(
                all_invalid_mask_sum != mask.shape[-1]
            )[0].to(self.device)
            all_timestamps = all_timestamps[all_valid_rows]
            all_descriptions = all_descriptions[all_valid_rows]
            all_continuous = all_continuous[all_valid_rows]
            all_continuous_tokens = all_continuous_tokens[all_valid_rows]
            mask = mask[all_valid_rows]
            target_mask = target_mask[all_valid_rows]
            column_identifiers = column_identifiers[all_valid_rows]
            all_dataset_ids = all_dataset_ids[all_valid_rows]

        print("target_mask.shape", target_mask.shape)

        decoder_input = {
            "timestamps": all_timestamps,
            "descriptions": all_descriptions,
            "continuous": all_continuous,
            "continuous_tokens": all_continuous_tokens,
            "mask": mask,
            "target_mask": target_mask,
            "dataset_ids": all_dataset_ids,
            "target": all_continuous,
            "column_identifiers": column_identifiers,
            "description_mask_flag": description_mask_flag,
        }
        for key in decoder_input:
            if key == "dataset_ids":
                continue
            if decoder_input[key] is not None:
                assert torch.all(
                    torch.logical_not(torch.isnan(decoder_input[key]))
                ), f"NaN values found in {key}"
                assert torch.all(decoder_input[key] != NULL_TOKEN), (
                    f"NULL_TOKEN values found in {key}"
                )

        return decoder_input

    def prepare_training_input_for_tabicl_embedder(self, src, *args, **kwargs):
        self.device = next(self.parameters()).device

        batch_size = src.shape[0]
        time_series_length = src.shape[1]
        num_correlates = src.shape[2]

        invalid_mask = torch.zeros(
            batch_size, time_series_length, num_correlates
        ).to(self.device)
        invalid_mask = rearrange(invalid_mask, "b t nc -> b nc t")
        invalid_mask = rearrange(invalid_mask, "b nc t -> b (nc t)")
        invalid_mask = invalid_mask.bool()
        all_continuous = rearrange(src, "b t nc -> b nc t")
        all_continuous = rearrange(
            all_continuous,
            "b nc t -> b nc t 1",
        )

        all_descriptions = torch.ones(
            batch_size,
            num_correlates,
            self.dataset_config.dataset_description_end_idx
            - self.dataset_config.dataset_description_start_idx,
        )
        all_descriptions = repeat(
            all_descriptions,
            "b nc textf -> b nc t textf",
            t=time_series_length,
        )

        # create a fake set of timestamps using pandas datetime with hours minutes seconds
        timestamps = pd.date_range(
            start="2010-01-01", periods=time_series_length, freq="W", tz="UTC"
        )
        # convert to np.datetime64
        timestamps = np.array(timestamps, dtype=np.datetime64)

        timestamp_embeddings = np.zeros(
            (len(timestamps), len(TIMESTAMPS_FEATURES))
        )
        pd_timestamps = pd.Series(timestamps)
        us_holidays = holidays.country_holidays("US")
        timestamp_embeddings = convert_time_to_vector(
            timestamp_embeddings,
            TIMESTAMPS_FEATURES,
            NORM_RANGES,
            pd_timestamps,
            us_holidays,
            0,
            np.array(pd_timestamps.isna().values),
        )
        all_timestamps = (
            torch.tensor(timestamp_embeddings)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, num_correlates, 1, 1)
            .to(self.device)
        )

        all_dataset_ids = torch.zeros(batch_size, num_correlates).to(
            self.device
        )

        all_timestamps_mask = rearrange(
            obtain_mask(all_timestamps, category="continuous"),
            "b nc t -> b (nc t)",
        ).to(self.device)  # B X NUM_CORRELATES X TIME_SERIES_LENGTH
        all_descriptions_mask = rearrange(
            obtain_mask(all_descriptions, category="textual"),
            "b nc t -> b (nc t)",
        ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        all_continuous_mask = rearrange(
            obtain_mask(all_continuous, category="continuous"),
            "b nc t -> b (nc t)",
        ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        mask = torch.logical_or(
            torch.logical_or(all_continuous_mask, all_timestamps_mask),
            all_descriptions_mask,
        ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        mask = torch.logical_or(mask, invalid_mask).to(
            self.device
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        # flatten individual token elements
        all_timestamps = rearrange(
            all_timestamps, "b nc t tf -> b (nc t) tf"
        ).to(self.device)
        all_descriptions = rearrange(
            all_descriptions, "b nc t textf -> b (nc t) textf"
        ).to(self.device)
        all_continuous = rearrange(
            all_continuous, "b nc t cf -> b (nc t) cf"
        ).to(self.device)

        # now we make the elements have null token values for the mask locations
        # a timestamp null token is -1 for all timestamp features
        all_timestamps[mask] = -1
        # a textual null token is 0 for all textual features
        all_descriptions[mask] = 0
        # a continuous null token is -1 for all continuous features
        all_continuous[mask] = -1

        # convert the continuous values to tokens and remove the extra dimension
        all_continuous_tokens = continuous_to_token(all_continuous, mask).to(
            self.device
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH) X 9

        target_mask = kwargs.get("target_mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
        target_mask = rearrange(target_mask, "b nc t -> b (nc t)")

        assert target_mask is not None, "target_mask is None"
        target_mask = torch.logical_and(target_mask, ~mask).to(
            self.device
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        # create the column indicators
        column_identifiers = self._prepare_column_identifiers(
            all_timestamps.shape[0],
            num_correlates,
            time_series_length,
        ).to(self.device)

        description_mask_flag = torch.rand(1)[0] < self.description_mask_ratio

        if self.dataset_config.train_as_univariate_model:
            all_timestamps = rearrange(
                all_timestamps, "b (nc t) tf -> (b nc) t tf", nc=num_correlates
            )
            all_descriptions = rearrange(
                all_descriptions,
                "b (nc t) textf -> (b nc) t textf",
                nc=num_correlates,
            )
            all_continuous = rearrange(
                all_continuous, "b (nc t) cf -> (b nc) t cf", nc=num_correlates
            )
            all_continuous_tokens = rearrange(
                all_continuous_tokens,
                "b (nc t) cf -> (b nc) t cf",
                nc=num_correlates,
            )
            mask = rearrange(mask, "b (nc t) -> (b nc) t", nc=num_correlates)
            target_mask = rearrange(
                target_mask, "b (nc t) -> (b nc) t", nc=num_correlates
            )
            column_identifiers = rearrange(
                column_identifiers,
                "b (nc t) ncmax -> (b nc) t ncmax",
                nc=num_correlates,
            )
            all_dataset_ids = rearrange(
                all_dataset_ids, "b nc -> (b nc)", nc=num_correlates
            )
            all_dataset_ids = all_dataset_ids.unsqueeze(-1)

            all_invalid_mask_sum = torch.sum(mask, dim=-1)
            all_valid_rows = torch.where(
                all_invalid_mask_sum != mask.shape[-1]
            )[0].to(self.device)
            all_timestamps = all_timestamps[all_valid_rows]
            all_descriptions = all_descriptions[all_valid_rows]
            all_continuous = all_continuous[all_valid_rows]
            all_continuous_tokens = all_continuous_tokens[all_valid_rows]
            mask = mask[all_valid_rows]
            target_mask = target_mask[all_valid_rows]
            column_identifiers = column_identifiers[all_valid_rows]
            all_dataset_ids = all_dataset_ids[all_valid_rows]

        decoder_input = {
            "timestamps": all_timestamps,
            "descriptions": all_descriptions,
            "continuous": all_continuous,
            "continuous_tokens": all_continuous_tokens,
            "mask": mask,
            "target_mask": target_mask,
            "dataset_ids": all_dataset_ids,
            "target": all_continuous,
            "column_identifiers": column_identifiers,
            "description_mask_flag": description_mask_flag,
        }
        for key in decoder_input:
            if key == "dataset_ids":
                continue
            if decoder_input[key] is not None:
                assert torch.all(
                    torch.logical_not(torch.isnan(decoder_input[key]))
                ), f"NaN values found in {key}"
                assert torch.all(decoder_input[key] != NULL_TOKEN), (
                    f"NULL_TOKEN values found in {key}"
                )

        return decoder_input

    def prepare_training_input_for_real_data(
        self, train_batch, *args, **kwargs
    ):
        window = train_batch["timeseries"].to(self.device).float()
        all_timestamps = window[
            ...,
            self.dataset_config.timestamp_start_idx : self.dataset_config.timestamp_end_idx,
        ]  # B X NUM_CORRELATES X (TIME_SERIES_LENGTH X NUM_TIMESTAMP_FEATURES)
        all_descriptions = window[
            ...,
            self.dataset_config.dataset_description_start_idx : self.dataset_config.dataset_description_end_idx,
        ]  # B X NUM_CORRELATES X NUM_TEXTUAL_FEATURES
        all_continuous = window[
            ...,
            self.dataset_config.continuous_start_idx : self.dataset_config.continuous_end_idx,
        ]  # B X NUM_CORRELATES X TIME_SERIES_LENGTH
        all_dataset_ids = window[..., -1]  # B X NUM_CORRELATES
        # assert torch.all(
        #     all_dataset_ids <= self.dataset_config.num_datasets
        # ), "Dataset ids are out of bounds"
        num_correlates = all_dataset_ids.shape[1]
        time_series_length = all_continuous.shape[2]

        # reshape the multi modal information
        all_timestamps = rearrange(
            all_timestamps,
            "b nc (t tf) -> b nc t tf",
            t=self.dataset_config.time_series_length,
            tf=self.dataset_config.num_timestamp_features,
        )
        all_descriptions = repeat(
            all_descriptions,
            "b nc textf -> b nc t textf",
            t=self.dataset_config.time_series_length,
        )
        all_continuous = rearrange(
            all_continuous,
            "b nc t -> b nc t 1",
        )

        # obtain mask
        all_timestamps_mask = rearrange(
            obtain_mask(all_timestamps, category="continuous"),
            "b nc t -> b (nc t)",
        )  # B X NUM_CORRELATES X TIME_SERIES_LENGTH
        all_descriptions_mask = rearrange(
            obtain_mask(all_descriptions, category="textual"),
            "b nc t -> b (nc t)",
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        all_continuous_mask = rearrange(
            obtain_mask(all_continuous, category="continuous"),
            "b nc t -> b (nc t)",
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        # combine masks
        # we ignore a token even if any one of the 3 elements (timestamps, text, value) is invalid
        mask = torch.logical_or(
            torch.logical_or(all_continuous_mask, all_timestamps_mask),
            all_descriptions_mask,
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        # flatten individual token elements
        all_timestamps = rearrange(all_timestamps, "b nc t tf -> b (nc t) tf")
        all_descriptions = rearrange(
            all_descriptions, "b nc t textf -> b (nc t) textf"
        )
        all_continuous = rearrange(all_continuous, "b nc t cf -> b (nc t) cf")

        # now we make the elements have null token values for the mask locations
        # a timestamp null token is -1 for all timestamp features
        all_timestamps[mask] = -1
        # a textual null token is 0 for all textual features
        all_descriptions[mask] = 0
        # a continuous null token is -1 for all continuous features
        all_continuous[mask] = -1

        # convert the continuous values to tokens and remove the extra dimension
        all_continuous_tokens = continuous_to_token(
            all_continuous, mask
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH) X 9

        batch_size = all_continuous.shape[0]
        num_tokens = all_continuous.shape[1]
        target_mask = (
            torch.rand(
                batch_size,
                num_tokens,
            )
            < self.target_mask_ratio
        ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        target_mask = torch.logical_and(target_mask, ~mask).to(
            self.device
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        # create the column indicators
        column_identifiers = self._prepare_column_identifiers(
            all_timestamps.shape[0],
            num_correlates,
            time_series_length,
        )

        description_mask_flag = torch.rand(1)[0] < self.description_mask_ratio

        decoder_input = {
            "timestamps": all_timestamps,
            "descriptions": all_descriptions,
            "continuous": all_continuous,
            "continuous_tokens": all_continuous_tokens,
            "mask": mask,
            "target_mask": target_mask,
            "dataset_ids": all_dataset_ids,
            "target": all_continuous,
            "column_identifiers": column_identifiers,
            "description_mask_flag": description_mask_flag,
        }
        for key in decoder_input:
            if key == "dataset_ids":
                continue
            if decoder_input[key] is not None:
                assert torch.all(
                    torch.logical_not(torch.isnan(decoder_input[key]))
                ), f"NaN values found in {key}"
                assert torch.all(decoder_input[key] != NULL_TOKEN), (
                    f"NULL_TOKEN values found in {key}"
                )

        return decoder_input

    def autoregressive_forecast(
        self,
        batch,
        synthesizer,
        history_length=None,
        forecast_length=None,
        use_ground_truth_for_next_step=False,
        use_text_description_mask=True,
        *args,
        **kwargs,
    ):
        # split the input into history and forecast
        if history_length is None:
            history_length = (
                synthesizer.dataset_config.time_series_length
                - synthesizer.dataset_config.forecast_length
            )
        if forecast_length is None:
            forecast_length = synthesizer.dataset_config.forecast_length

        timeseries_length = history_length + forecast_length

        decoder_input = synthesizer.prepare_training_input(
            batch, log_dir=None
        )  # SynthefyForecastingModelV1.prepare_training_input()

        batch_size = decoder_input["timestamps"].shape[0]
        num_timestamps = decoder_input["timestamps"].shape[1]
        num_correlates = int(num_timestamps / timeseries_length)

        # reshape the input to explicitly deal with correlates
        # we first separate nc and t
        timestamps = rearrange(
            decoder_input["timestamps"],
            "b (nc t) tf -> b nc t tf",
            nc=num_correlates,
            t=timeseries_length,
        )  # B X NUM_CORRELATES X TIME_SERIES_LENGTH X NUM_TIMESTAMP_FEATURES
        descriptions = rearrange(
            decoder_input["descriptions"],
            "b (nc t) textf -> b nc t textf",
            nc=num_correlates,
            t=timeseries_length,
        )  # B X NUM_CORRELATES X TIME_SERIES_LENGTH X TEXT_EMBEDDING_DIM
        continuous_tokens = rearrange(
            decoder_input["continuous_tokens"],
            "b (nc t) tf -> b nc t tf",
            nc=num_correlates,
            t=timeseries_length,
        )  # B X NUM_CORRELATES X TIME_SERIES_LENGTH X TIMESERIES_TOKEN_DIMS
        invalid_mask = rearrange(
            decoder_input["mask"],
            "b (nc t) -> b nc t",
            nc=num_correlates,
            t=timeseries_length,
        )
        column_identifiers = rearrange(
            decoder_input["column_identifiers"],
            "b (nc t) ncmax -> b nc t ncmax",
            nc=num_correlates,
            t=timeseries_length,
        )

        timestamps_history = timestamps[..., :history_length, :]
        timestamps_forecast = timestamps[..., history_length:, :]

        descriptions_history = descriptions[..., :history_length, :]
        descriptions_forecast = descriptions[..., history_length:, :]

        column_identifiers_history = column_identifiers[..., :history_length, :]
        column_identifiers_forecast = column_identifiers[
            ..., history_length:, :
        ]

        continuous_tokens_history = continuous_tokens[..., :history_length, :]
        continuous_tokens_forecast = continuous_tokens[..., history_length:, :]

        invalid_mask_history = invalid_mask[..., :history_length]
        invalid_mask_forecast = invalid_mask[..., history_length:]

        target_mask_history = (
            torch.zeros_like(invalid_mask_history).bool().to(self.device)
        )

        # initialize the forecast input
        invalid_mask_forecast_input = invalid_mask_history.clone()
        target_mask_forecast_input = target_mask_history.clone()
        continuous_tokens_forecast_input = continuous_tokens_history.clone()
        timestamps_forecast_input = timestamps_history.clone()
        descriptions_forecast_input = descriptions_history.clone()
        column_identifiers_forecast_input = column_identifiers_history.clone()
        forecast_output_list = []
        for fidx in tqdm(
            range(forecast_length), total=forecast_length, desc="Forecasting"
        ):
            # for each input, we combine nc and t
            # create a invalid indicator flag for the timestep we are going to forecast
            invalid_indicator_flag = invalid_mask_forecast[
                ..., fidx : fidx + 1
            ].to(self.device)

            # update the invalid mask for the forecast
            invalid_mask_forecast_input = torch.cat(
                [invalid_mask_forecast_input, invalid_indicator_flag],
                dim=-1,
            )
            invalid_mask_forecast_input = rearrange(
                invalid_mask_forecast_input, "b nc t -> b (nc t)"
            )

            # create a target indicator flag for the forecast
            target_indicator_flag = (
                torch.ones((batch_size, num_correlates, 1))
                .bool()
                .to(self.device)
            )
            # update the target mask for the forecast
            target_mask_forecast_input = torch.cat(
                [target_mask_forecast_input, target_indicator_flag], dim=-1
            )
            target_mask_forecast_input = rearrange(
                target_mask_forecast_input, "b nc t -> b (nc t)"
            )

            # create a continuous null token for the forecast
            continuous_null_token = (
                torch.ones(
                    (
                        batch_size,
                        num_correlates,
                        1,
                        self.config.foundation_model_config.timeseries_token_dims,
                    )
                )
                * -1
            ).to(self.device)
            # update the continuous tokens for the forecast
            continuous_tokens_forecast_input = torch.cat(
                [continuous_tokens_forecast_input, continuous_null_token],
                dim=-2,
            )
            continuous_tokens_forecast_input = rearrange(
                continuous_tokens_forecast_input,
                "b nc t cf -> b (nc t) cf",
            )

            # obtain the timestamp to forecast
            timestamp_to_forecast = rearrange(
                timestamps_forecast[..., fidx, :],
                "b nc tf -> b nc 1 tf",
            )
            # update the timestamps for the forecast
            timestamps_forecast_input = torch.cat(
                [timestamps_forecast_input, timestamp_to_forecast], dim=-2
            )
            timestamps_forecast_input = rearrange(
                timestamps_forecast_input, "b nc t tf -> b (nc t) tf"
            )

            # obtain the description of the forecast
            description_of_forecast = rearrange(
                descriptions_forecast[..., fidx, :],
                "b nc textf -> b nc 1 textf",
            )
            # update the descriptions for the forecast
            descriptions_forecast_input = torch.cat(
                [descriptions_forecast_input, description_of_forecast],
                dim=-2,
            )
            descriptions_forecast_input = rearrange(
                descriptions_forecast_input, "b nc t textf -> b (nc t) textf"
            )
            # obtain the column identifiers for the forecast
            column_identifiers_of_forecast = rearrange(
                column_identifiers_forecast[..., fidx, :],
                "b nc ncmax -> b nc 1 ncmax",
            )
            # update the column identifiers for the forecast
            column_identifiers_forecast_input = torch.cat(
                [
                    column_identifiers_forecast_input,
                    column_identifiers_of_forecast,
                ],
                dim=-2,
            )
            column_identifiers_forecast_input = rearrange(
                column_identifiers_forecast_input,
                "b nc t ncmax -> b (nc t) ncmax",
            )

            input_dict = {
                "timestamps": timestamps_forecast_input,
                "descriptions": descriptions_forecast_input,
                "continuous_tokens": continuous_tokens_forecast_input,
                "mask": invalid_mask_forecast_input,
                "target_mask": target_mask_forecast_input,
                "window_size": history_length + fidx + 1,
                "column_identifiers": column_identifiers_forecast_input,
                "description_mask_flag": use_text_description_mask,
            }

            forecast_output = rearrange(
                self.forward(input_dict)["prediction"],
                "b (nc t) 1 -> b nc t 1",
                nc=num_correlates,
                t=history_length + fidx + 1,
            )
            forecast_output = forecast_output[..., -1, :]
            forecast_output_list.append(forecast_output[..., 0])
            forecast_output_token = continuous_to_token(
                forecast_output,
                torch.zeros((batch_size, num_correlates)).to(self.device),
            )
            # we then separate nc and t after the forward pass
            continuous_tokens_forecast_input = rearrange(
                continuous_tokens_forecast_input,
                "b (nc t) cf -> b nc t cf",
                nc=num_correlates,
                t=history_length + fidx + 1,
            )
            if use_ground_truth_for_next_step:
                # TODO: What if the ground truth is NaN or invalid?
                continuous_tokens_forecast_input[..., -1, :] = (
                    continuous_tokens_forecast[..., fidx, :]
                )
                assert (
                    history_length + fidx + 1
                    == continuous_tokens_forecast_input.shape[-2]
                ), "Shape mismatch in continuous tokens forecast input"
            else:
                continuous_tokens_forecast_input[..., -1, :] = (
                    forecast_output_token
                )

            invalid_mask_forecast_input = rearrange(
                invalid_mask_forecast_input,
                "b (nc t) -> b nc t",
                nc=num_correlates,
                t=history_length + fidx + 1,
            )
            # update the target mask
            target_mask_forecast_input = rearrange(
                target_mask_forecast_input,
                "b (nc t) -> b nc t",
                nc=num_correlates,
                t=history_length + fidx + 1,
            )
            target_mask_forecast_input[..., -1] = False
            assert not torch.any(target_mask_forecast_input), (
                "Target mask forecast input is not set to False"
            )
            timestamps_forecast_input = rearrange(
                timestamps_forecast_input,
                "b (nc t) tf -> b nc t tf",
                nc=num_correlates,
                t=history_length + fidx + 1,
            )
            descriptions_forecast_input = rearrange(
                descriptions_forecast_input,
                "b (nc t) textf -> b nc t textf",
                nc=num_correlates,
                t=history_length + fidx + 1,
            )
            column_identifiers_forecast_input = rearrange(
                column_identifiers_forecast_input,
                "b (nc t) ncmax -> b nc t ncmax",
                nc=num_correlates,
                t=history_length + fidx + 1,
            )

        forecast_output = torch.stack(forecast_output_list, dim=-1)

        return forecast_output

    def synthesis_function(
        self,
        batch,
        synthesizer,
        history_length=None,
        forecast_length=None,
        use_ground_truth_for_next_step=False,
    ):
        predicted_forecast = self.autoregressive_forecast(
            batch,
            synthesizer,
            history_length,
            forecast_length,
            use_ground_truth_for_next_step,
        )
        if history_length is None:
            history_length = (
                synthesizer.dataset_config.time_series_length
                - synthesizer.dataset_config.forecast_length
            )
        if forecast_length is None:
            forecast_length = synthesizer.dataset_config.forecast_length
        timeseries_length = history_length + forecast_length

        decoder_input = synthesizer.prepare_training_input(
            batch, log_dir=None
        )  # SynthefyForecastingModelV1.prepare_training_input()

        ts = (
            decoder_input["continuous"]
            .squeeze(-1)
            .reshape(
                -1,
                self.dataset_config.num_correlates,
                timeseries_length,
            )
        )
        true_history = ts[
            :,
            :,
            : timeseries_length - forecast_length,
        ]
        true_forecast = ts[:, :, -forecast_length:]
        mask = decoder_input["mask"].reshape(
            -1,
            self.dataset_config.num_correlates,
            timeseries_length,
        )
        forecast_mask = mask[:, :, -forecast_length:]

        return {
            "predicted_forecast": predicted_forecast,
            "true_forecast": true_forecast,
            "forecast_mask": forecast_mask,
            "history": true_history,
            "dataset_ids": decoder_input["dataset_ids"],
        }

    def process_single_parquet_file(
        self,
        df: pd.DataFrame,
        target_column: str,
        timestamp_column: str,
        max_context_length: int = 256,
    ):
        target_arr = np.expand_dims(df[target_column].to_numpy(), axis=-1)
        scaler = StandardScaler()
        scaled_target_arr = scaler.fit_transform(target_arr)
        scaled_target_arr = scaled_target_arr[:, 0]
        scaled_target_arr = scaled_target_arr[-max_context_length:]

        assert scaler.var_ is not None, "Variance is not None"
        assert scaler.mean_ is not None, "Mean is not None"
        assert scaler.scale_ is not None, "Scale is not None"
        scaler_arr = np.array(
            [scaler.mean_[0], scaler.scale_[0], scaler.var_[0]]  # type: ignore
        )
        scaler_tensor = torch.tensor(scaler_arr, dtype=torch.float32)

        timestamp_data = df[timestamp_column].to_numpy()[-max_context_length:]

        pad_length = max_context_length - len(scaled_target_arr)

        timestamp_pad = np.full(pad_length, np.nan, dtype=np.datetime64)
        value_pad = np.full(pad_length, np.nan)

        # Concatenate padding with data
        timestamp_slice = np.concatenate([timestamp_pad, timestamp_data])
        value_slice = np.concatenate([value_pad, scaled_target_arr])
        value_slice = torch.tensor(value_slice, dtype=torch.float32)
        timestamp_embeddings = np.zeros(
            (len(timestamp_slice), len(TIMESTAMPS_FEATURES))
        )
        pd_timestamp_slice = pd.to_datetime(
            pd.Series(timestamp_slice), format="%Y-%m-%d", errors="coerce"
        )
        us_holidays = holidays.country_holidays("US")
        timestamp_embeddings = convert_time_to_vector(
            timestamp_embeddings,
            TIMESTAMPS_FEATURES,
            NORM_RANGES,
            pd_timestamp_slice,
            us_holidays,
            0,
            np.array(pd_timestamp_slice.isna().values),
        )

        timestamp_embeddings = torch.tensor(timestamp_embeddings)

        output = torch.cat(
            [
                scaler_tensor,  # scalars
                timestamp_embeddings.flatten(),  # flattened timestamp embeddings
                torch.ones(
                    self.config.dataset_config.text_embedding_dim,
                ),  # text embedding
                value_slice,  # value
                torch.tensor([0], dtype=torch.float32),  # dataset index
            ]
        )
        return output

    def process_single_parquet_file_for_synthetic_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        timestamp_column: str,
        max_context_length: int = 256,
    ):
        target_arr = np.expand_dims(df[target_column].to_numpy(), axis=-1)
        scaler = StandardScaler()
        scaled_target_arr = scaler.fit_transform(target_arr)
        scaled_target_arr = scaled_target_arr[:, 0]
        scaled_target_arr = scaled_target_arr[-max_context_length:]

        assert scaler.var_ is not None, "Variance is not None"
        assert scaler.mean_ is not None, "Mean is not None"
        assert scaler.scale_ is not None, "Scale is not None"
        scaler_arr = np.array(
            [scaler.mean_[0], scaler.scale_[0], scaler.var_[0]]  # type: ignore
        )
        scaler_tensor = torch.tensor(scaler_arr, dtype=torch.float32)

        timestamp_data = df[timestamp_column].to_numpy()[-max_context_length:]

        pad_length = max_context_length - len(scaled_target_arr)

        timestamp_pad = np.full(pad_length, np.nan, dtype=np.datetime64)
        value_pad = np.full(pad_length, np.nan)

        # Concatenate padding with data
        timestamp_slice = np.concatenate([timestamp_pad, timestamp_data])
        value_slice = np.concatenate([value_pad, scaled_target_arr])
        value_slice = torch.tensor(value_slice, dtype=torch.float32)
        timestamp_embeddings = np.zeros(
            (len(timestamp_slice), len(TIMESTAMPS_FEATURES))
        )
        pd_timestamp_slice = pd.to_datetime(
            pd.Series(timestamp_slice), format="%Y-%m-%d", errors="coerce"
        )
        us_holidays = holidays.country_holidays("US")
        timestamp_embeddings = convert_time_to_vector(
            timestamp_embeddings,
            TIMESTAMPS_FEATURES,
            NORM_RANGES,
            pd_timestamp_slice,
            us_holidays,
            0,
            np.array(pd_timestamp_slice.isna().values),
        )

        timestamp_embeddings = torch.tensor(timestamp_embeddings)

        output = torch.cat(
            [
                scaler_tensor,  # scalars
                timestamp_embeddings.flatten(),  # flattened timestamp embeddings
                torch.ones(
                    self.config.dataset_config.text_embedding_dim,
                ),  # text embedding
                value_slice,  # value
                torch.tensor([0], dtype=torch.float32),  # dataset index
            ]
        )
        return output

    def predict(
        self,
        target_df: pd.DataFrame,
        covariate_columns: List[str],
        metadata_dataframes: List[pd.DataFrame],
        target_columns: List[str],
        forecasting_timestamp: datetime,
        future_time_stamps: List[datetime],
        ground_truth_df: pd.DataFrame | None = None,
        remove_all_metadata: bool = False,
        covariate_columns_to_leak: List[str] | None = None,
        metadata_dataframes_leak_idxs: List[int] | None = None,
        quantiles: List[float] = [0.1, 0.9],
        timestamp_column: str = "Date",
        use_ground_truth_for_next_step: bool = False,
    ) -> ForecastDataset:
        """
        This function is used to predict the future values of the target columns.
        Args:
            target_df: pd.DataFrame, the dataframe containing the target columns
            covariate_columns: List[str], the columns to use as covariates
            metadata_dataframes: List[pd.DataFrame], the metadata dataframes
            target_columns: List[str], the columns to predict
            forecasting_timestamp: datetime, the timestamp to forecast
            future_time_stamps: List[datetime], the future timestamps to forecast
            ground_truth_df: pd.DataFrame, the dataframe containing the ground truth values
            remove_all_metadata: bool, whether to remove all metadata
            covariate_columns_to_leak: List[str], the columns to leak
            metadata_dataframes_leak_idxs: List[int], the indices of the metadata dataframes to leak
            quantiles: List[float], the quantiles to predict
            timestamp_column: str, the column containing the timestamps
        Returns:
            ForecastDataset: a dataset containing the ground truth and predicted values
        """
        covariates = []
        output_dim = (
            (len(TIMESTAMPS_FEATURES) + 1)
            * self.dataset_config.time_series_length
            + self.config.dataset_config.text_embedding_dim
            + 3
            + 1
        )
        for covariate_column in covariate_columns:
            if self.config.dataset_config.using_synthetic_data:
                processed_arr = self.process_single_parquet_file(
                    target_df, covariate_column, timestamp_column
                )
            else:
                processed_arr = self.process_single_parquet_file(
                    target_df, covariate_column, timestamp_column
                )

            covariates.append(processed_arr)

        # convert to tensors
        covariates = (
            torch.stack(covariates, dim=0)
            if len(covariates) > 0
            else torch.zeros((0, output_dim))
        )

        forecast_groups = []
        for target_column in target_columns:
            processed_arr = self.process_single_parquet_file(
                target_df,
                target_column,
                timestamp_column,
                max_context_length=self.dataset_config.time_series_length,
            )
            processed_arr = processed_arr.unsqueeze(0)

            # if number of covariates is less than num_correlates, pad with all nans
            if len(covariates) < self.dataset_config.num_correlates - 1:
                num_correlates_to_pad = (
                    self.dataset_config.num_correlates - 1 - len(covariates)
                )
                dummy_correlates = torch.full(
                    (num_correlates_to_pad, output_dim), np.nan
                )
                covariates = torch.cat([dummy_correlates, covariates], dim=0)

            if len(covariates) > self.dataset_config.num_correlates - 1:
                covariates = covariates[
                    : self.dataset_config.num_correlates - 1
                ]

            # concatenate covariates and targets
            covariates_and_target = torch.cat(
                [covariates, processed_arr], dim=0
            )
            # last row is the target
            model_input = covariates_and_target.unsqueeze(0)
            batch_dict = {"timeseries": model_input}

            with torch.no_grad():  # Disable gradient computation
                output_dict = self.synthesis_function(
                    batch=batch_dict,
                    synthesizer=self,
                    forecast_length=None,  # TODO: we can specify forcast lenght from user's data: len(future_time_stamps)
                )  # Pass self as synthesizer
            true_forecast = output_dict["true_forecast"]
            predicted_forecast = output_dict["predicted_forecast"]

            true_forecast_np = true_forecast[0, -1].detach().cpu().numpy()
            predicted_forecast_np = (
                predicted_forecast[0, -1].detach().cpu().numpy()
            )
            batch_scalers = model_input[0, -1, :3].detach().cpu().numpy()

            # unscale the forecast

            true_forecast_np_unscaled = (
                true_forecast_np * batch_scalers[1] + batch_scalers[0]
            )
            predicted_forecast_np_unscaled = (
                predicted_forecast_np * batch_scalers[1] + batch_scalers[0]
            )

            forecast_values = predicted_forecast_np_unscaled.tolist()
            ground_truth_values = true_forecast_np_unscaled.tolist()

            # Dummy confidence intervals (empty or zeros)
            confidence_intervals = [
                ConfidenceInterval(lower=0.0, upper=0.0)
                for _ in forecast_values
            ]
            univariate_confidence_intervals = [
                ConfidenceInterval(lower=0.0, upper=0.0)
                for _ in forecast_values
            ]

            # post process the forecast values given the future_time_stamps
            (
                forecast_values,
                ground_truth_values,
                confidence_intervals,
                univariate_confidence_intervals,
            ) = post_process_forecast_values(
                forecast_values=forecast_values,
                confidence_intervals=confidence_intervals,
                univariate_confidence_intervals=univariate_confidence_intervals,
                ground_truth_values=ground_truth_values,
                future_time_stamps=future_time_stamps,
            )

            forecast_group = ForecastGroup(
                target_column=target_column,
                forecasts=forecast_values,
                confidence_intervals=confidence_intervals,
                univariate_forecasts=forecast_values,  # or use a different value if available
                univariate_confidence_intervals=univariate_confidence_intervals,
                ground_truth=ground_truth_values,
            )
            forecast_groups.append(forecast_group)
            torch.cuda.empty_cache()

        # Use stringified timestamps for the forecast period with proper fractional second formatting
        forecast_timestamps = [
            format_timestamp_with_optional_fractional_seconds(pd.Timestamp(ts))
            for ts in future_time_stamps
        ]

        return ForecastDataset(
            timestamps=forecast_timestamps, values=forecast_groups
        )

    def predict_synthetic_data(
        self,
        target_df: pd.DataFrame,
        covariate_columns: List[str],
        metadata_dataframes: List[pd.DataFrame],
        target_columns: List[str],
        forecasting_timestamp: datetime,
        future_time_stamps: List[datetime],
        ground_truth_df: pd.DataFrame | None = None,
        remove_all_metadata: bool = False,
        covariate_columns_to_leak: List[str] | None = None,
        metadata_dataframes_leak_idxs: List[int] | None = None,
        quantiles: List[float] = [0.1, 0.9],
        timestamp_column: str = "Date",
        use_ground_truth_for_next_step: bool = False,
    ):
        """
        This function is used to predict the future values of the target columns.
        Args:
            target_df: pd.DataFrame, the dataframe containing the target columns
            covariate_columns: List[str], the columns to use as covariates
            metadata_dataframes: List[pd.DataFrame], the metadata dataframes
            target_columns: List[str], the columns to predict
            forecasting_timestamp: datetime, the timestamp to forecast
            future_time_stamps: List[datetime], the future timestamps to forecast
            ground_truth_df: pd.DataFrame, the dataframe containing the ground truth values
            remove_all_metadata: bool, whether to remove all metadata
            covariate_columns_to_leak: List[str], the columns to leak
            metadata_dataframes_leak_idxs: List[int], the indices of the metadata dataframes to leak
            quantiles: List[float], the quantiles to predict
            timestamp_column: str, the column containing the timestamps
        Returns:
            ForecastDataset: a dataset containing the ground truth and predicted values
        """
        # extract every coloumn except timestamp
        target_df = target_df.drop(
            columns=["minute", "hour", "day", "month", "year"]
        )
        targets = target_df.to_numpy()

        # convert to tensors
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        targets = scaler.fit_transform(targets)

        targets = torch.tensor(targets, dtype=torch.float32)
        targets = targets.unsqueeze(0)

        X = targets[..., :-1]
        y = targets[..., -1]
        d = torch.tensor([targets.shape[2] - 1], dtype=torch.long)

        num_max_correlates = self.dataset_config.num_correlates
        num_available_correlates = targets.shape[2]
        num_dummy_correlates = num_max_correlates - num_available_correlates
        dummy_correlates = torch.zeros(
            (1, self.dataset_config.time_series_length, num_dummy_correlates),
            dtype=torch.float32,
        )

        train_batch = {
            "X": torch.cat([X, dummy_correlates], dim=-1),
            "y": y,
            "d": d,
        }

        with torch.no_grad():  # Disable gradient computation
            forecast_output = self.autoregressive_forecast(
                train_batch, self
            )  # Pass self as synthesizer

            forecast_output = forecast_output[:, 0]
            forecast_output = forecast_output.detach().cpu()
            gt = rearrange(targets[0], "t c -> c t")
            gt_history = gt[:, : -forecast_output.shape[1]]
            forecast = torch.cat([gt_history, forecast_output], dim=1)

        return gt, forecast
