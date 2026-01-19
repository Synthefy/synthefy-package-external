import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from timesfm.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    ResidualBlock,
    causal_mask,
    convert_paddings_to_mask,
    get_large_negative_number,
    merge_masks,
)
from timesfm.timesfm_base import TimesFmCheckpoint
from torch import nn

NULL_TOKEN = -999999


def construct_timestamp_mask(
    timestamps,
    invalid_mask,
    timestamp_based_mask=True,
    timestamp_distance_mask=-1,
):
    """
    Construct a mask from the timestamps

    timestamps: Batch x num_correlates x window_size x num_timestamps
    timestamp_mask: Batch x (num_correlates x window_size)
    """
    # TODO: get from config later, but making this function self contained
    relevant_timestamp_indices = [
        0,
        2,
        5,
        6,
        7,
        8,
    ]  # year, month, day, hour, minute, second
    offsets = [12, 31, 24, 60, 60, 1]
    prod = []
    for i in range(0, len(offsets)):
        prod.append(np.prod(offsets[i:]))
    multiplies = torch.tensor(prod).to(timestamps.device).float()
    multiplies = (
        multiplies.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    )  # 1 X 1 X 1 X NUM_TIMESTAMP_FEATURES

    # produce 1 between any two values where the second value is larger than the first
    # 0 if they are equal, -1 if the second value is smaller than the first
    if timestamp_based_mask:
        relevant_timestamps = timestamps[..., relevant_timestamp_indices]
        relevant_timestamps = torch.sum(
            relevant_timestamps * multiplies, dim=-1
        ).view(
            relevant_timestamps.shape[0], -1
        )  # B X NUM_CORRELATES X TIME_SERIES_LENGTH

        ts_i = relevant_timestamps.unsqueeze(
            -1
        )  # B X NUM_CORRELATES * TIME_SERIES_LENGTH X 1
        ts_j = relevant_timestamps.unsqueeze(
            -2
        )  # B X 1 X NUM_CORRELATES * TIME_SERIES_LENGTH

        # produce 1 between any two values where the second value is larger than the first
        # 0 if they are equal, -1 if the second value is smaller than the first
        mask = (
            ts_i < ts_j
        )  # Batch x (window_size * num_correlates) x (window_size * num_correlates)
        if timestamp_distance_mask > 0:
            # mask = 1 for edges that are greater than the timestamp distance mask
            mask = mask | (np.abs(ts_i - ts_j) > timestamp_distance_mask)
    else:
        # allow all edges
        batch_size = timestamps.shape[0]
        num_correlates = timestamps.shape[1]
        window_size = timestamps.shape[2]
        mask = torch.zeros(
            batch_size,
            num_correlates * window_size,
            num_correlates * window_size,
        ).to(timestamps.device)

    mask = torch.logical_or(mask, invalid_mask.unsqueeze(-1))
    mask = torch.logical_or(mask, invalid_mask.unsqueeze(-2))
    mask = mask.float()

    mask = mask.unsqueeze(
        1
    )  # Batch x 1 x (window_size * num_correlates) x (window_size * num_correlates)

    large_negative_number = get_large_negative_number(timestamps.dtype)
    mask = mask * large_negative_number

    return mask


class SynthefyFoundationModelContinuousEmbedder(nn.Module):
    """
    Model with 3 residual blocks to process time series tokens, textual tokens, and timestamp tokens.
    Using TimesFM's ResidualBlock implementation for consistency.
    """

    def __init__(
        self,
        model_dims,
        timeseries_token_dims,
        textual_token_dims,
        timestamp_token_dims,
        device=None,
    ):
        super(SynthefyFoundationModelContinuousEmbedder, self).__init__()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model dimensions
        self.model_dims = model_dims
        self.hidden_dims = self.model_dims

        # Define input dimensions from config
        self.timeseries_token_dims = timeseries_token_dims
        self.textual_token_dims = textual_token_dims
        self.timestamp_token_dims = timestamp_token_dims

        # Residual blocks for each modality
        self.timeseries_processor = ResidualBlock(
            input_dims=self.timeseries_token_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

        self.textual_processor = ResidualBlock(
            input_dims=self.textual_token_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

        self.timestamp_processor = ResidualBlock(
            input_dims=self.timestamp_token_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

        # Optional: add a fusion layer to combine the processed tokens
        self.fusion_layer = ResidualBlock(
            input_dims=3
            * self.model_dims,  # Concatenated output from all three processors
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

    def forward(self, sfm_input):
        """
        Process each type of token through its respective residual block.

        Args:
            sfm_input: Tokens representing time series values [batch, seq_len, timeseries_token_dims]

        Returns:
            Processed sequence representation [batch, seq_len, model_dims]
        """
        sfm_continuous_input = sfm_input[..., : self.timeseries_token_dims]
        sfm_textual_input = sfm_input[
            ...,
            self.timeseries_token_dims : self.timeseries_token_dims
            + self.textual_token_dims,
        ]
        sfm_timestamp_input = sfm_input[
            ..., self.timeseries_token_dims + self.textual_token_dims :
        ]
        # Process each modality
        processed_timeseries = self.timeseries_processor(sfm_continuous_input)
        processed_textual = self.textual_processor(sfm_textual_input)
        processed_timestamps = self.timestamp_processor(sfm_timestamp_input)

        # Combine the processed tokens
        combined = torch.cat(
            [processed_timeseries, processed_textual, processed_timestamps],
            dim=-1,
        )
        fused = self.fusion_layer(combined)

        return fused


class SynthefyFoundationForecastingModelV2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # toggle ablations
        self.random_vector_instead_text = (
            config.foundation_model_config.random_vector_instead_text
        )
        self.mask_timestamp = config.foundation_model_config.mask_timestamp
        # TODO: not added to fm config yet
        self.timestamp_distance_mask = (
            -1
        )  # config.foundation_model_config.timestamp_distance_mask
        self.timestamp_based_mask = True

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
        self.dataset_config = config.dataset_config
        self.use_metadata = config.foundation_model_config.use_metadata

        # Directly instantiate a PatchedTimeSeriesDecoder from TimesFM
        # self.model_dims is also the hidden size of the attention layers
        # so, each head dimension should be self.model_dims // self.num_heads

        self.decoder_model = PatchedTimeSeriesDecoder(
            self.decoder_tfm_config
        ).to(self.device)

        # Put the model into training mode
        self.decoder_model.train()

        self.continuous_embedder = SynthefyFoundationModelContinuousEmbedder(
            model_dims=self.decoder_model_dims,
            timeseries_token_dims=config.foundation_model_config.timeseries_token_dims,
            textual_token_dims=self.dataset_config.text_embedding_dim,
            timestamp_token_dims=self.dataset_config.num_timestamp_features,
        ).to(self.device)

        # num_patches = self.context_len // self.decoder_input_patch_len
        self.postprocessor = ResidualBlock(
            input_dims=self.decoder_model_dims * 2,
            hidden_dims=self.decoder_model_dims,
            output_dims=1,
        ).to(self.device)

    def continuous_to_token(
        self, continuous_input: torch.Tensor, continuous_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert continuous values to tokenized representation.
        Each continuous value is represented by 9 elements:
        - 1 element for sign (0 = negative, 1 = positive)
        - 4 elements for whole number digits
        - 4 elements for decimal digits

        If a value is equal to the NULL_TOKEN, all 9 elements will be set to -1.

        Args:
            continuous_input: Input tensor of continuous values
            continuous_mask: Mask tensor of shape [..., 9] with 1s for non-null tokens and 0s for null tokens

        Returns:
            Tokenized representation with shape [..., 9] for each input value
        """
        # Create output tensor with extra dimension
        orig_shape = continuous_input.shape
        output_shape = orig_shape + (9,)
        tokens = torch.zeros(output_shape, device=continuous_input.device)

        # Create mask for null tokens
        null_mask = continuous_mask

        # Process non-null values
        valid_values = continuous_input.clone()

        # Get sign (1 for positive, 0 for negative)
        sign = (valid_values >= 0).float()

        # Get absolute values
        abs_values = torch.abs(valid_values)

        # Extract whole part and decimal part
        whole_part = torch.floor(abs_values)
        decimal_part = abs_values - whole_part

        # Convert whole part to 4 digits
        thousands = torch.fmod(torch.floor(whole_part / 1000), 10)
        hundreds = torch.fmod(torch.floor(whole_part / 100), 10)
        tens = torch.fmod(torch.floor(whole_part / 10), 10)
        ones = torch.fmod(whole_part, 10)

        # Convert decimal part to 4 digits
        tenths = torch.fmod(torch.floor(decimal_part * 10), 10)
        hundredths = torch.fmod(torch.floor(decimal_part * 100), 10)
        thousandths = torch.fmod(torch.floor(decimal_part * 1000), 10)
        ten_thousandths = torch.fmod(torch.floor(decimal_part * 10000), 10)

        # Fill in the tokens tensor
        tokens[..., 0] = sign
        tokens[..., 1] = thousands
        tokens[..., 2] = hundreds
        tokens[..., 3] = tens
        tokens[..., 4] = ones
        tokens[..., 5] = tenths
        tokens[..., 6] = hundredths
        tokens[..., 7] = thousandths
        tokens[..., 8] = ten_thousandths

        # Set all token elements to -1 for null tokens
        if null_mask.any():
            # Expand null_mask to match token dimensions
            expanded_null_mask = null_mask.unsqueeze(-1).expand_as(tokens)
            tokens = torch.where(
                expanded_null_mask,
                torch.tensor(-1.0, device=tokens.device),
                tokens,
            )

        return tokens

    def patchify_masks(self, mask):
        patched_padding = torch.min(mask, dim=-1)[
            0
        ]  # Get the values from the min result
        return patched_padding

    def split_into_history_and_forecast(self, inp_tensor):
        """
        Split the input into history and forecast along the time dimension.
        The time dimension is the last dimension of the input.
        """
        inp_tensor_history = inp_tensor[..., : -self.horizon_len]
        inp_tensor_forecast = inp_tensor[..., -self.horizon_len :]
        return inp_tensor_history, inp_tensor_forecast

    def clear_nan_null_and_obtain_mask(self, elem, category=None):
        nan_mask = torch.isnan(elem)
        null_mask = elem == NULL_TOKEN
        if category == "textual":
            elem[nan_mask] = 0
            elem[null_mask] = 0
        else:
            elem[nan_mask] = -1
            elem[null_mask] = -1
        mask = torch.logical_or(nan_mask, null_mask)
        mask = mask.sum(dim=-2).bool()
        if category == "textual":
            invalid_mask = (elem == 0).all(dim=-2)
            mask = torch.logical_or(mask, invalid_mask)

        return elem, mask

    def prepare_training_input(self, train_batch, *args, **kwargs):
        train_batch = train_batch["timeseries"].to(self.device).float()
        batch_size = train_batch.shape[0]
        num_correlates = train_batch.shape[
            1
        ]  # univariate forecaster should be 1

        # extracting multi modal information
        all_timestamps = train_batch[
            ...,
            self.dataset_config.timestamp_start_idx : self.dataset_config.timestamp_end_idx,
        ]
        all_descriptions = train_batch[
            ...,
            self.dataset_config.dataset_description_start_idx : self.dataset_config.dataset_description_end_idx,
        ]
        all_continuous = train_batch[
            ...,
            self.dataset_config.continuous_start_idx : self.dataset_config.continuous_end_idx,
        ]
        all_dataset_ids = train_batch[..., -1]

        # reshape the multi modal information
        all_timestamps = all_timestamps.reshape(
            batch_size,
            num_correlates,
            self.dataset_config.time_series_length,
            self.dataset_config.num_timestamp_features,
        ).permute(
            0, 1, 3, 2
        )  # B X NUM_CORRELATES X NUM_TIMESTAMP_FEATURES X TIME_SERIES_LENGTH
        all_descriptions = all_descriptions.unsqueeze(
            -1
        )  # B X NUM_CORRELATES X NUM_TEXTUAL_FEATURES X 1
        all_continuous = all_continuous.unsqueeze(
            -2
        )  # B X NUM_CORRELATES X 1 X TIME_SERIES_LENGTH

        # clear nan, NULL_TOKEN and obtain mask
        all_timestamps, all_timestamps_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_timestamps, category="continuous"
            )
        )
        all_descriptions, all_descriptions_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_descriptions, category="textual"
            )
        )
        all_continuous, all_continuous_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_continuous, category="continuous"
            )
        )

        all_dataset_ids, all_dataset_ids_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_dataset_ids, category="textual"
            )
        )

        decoder_input = {
            "timestamps": all_timestamps,
            "descriptions": all_descriptions,
            "continuous": all_continuous,  # roughly concat(history, forecast)
            "timestamps_mask": all_timestamps_mask,
            "descriptions_mask": all_descriptions_mask,
            "continuous_mask": all_continuous_mask,
            "dataset_ids": all_dataset_ids,
            "dataset_ids_mask": all_dataset_ids_mask,
        }

        # These extra fields are added to enable compatibility with baselines.
        decoder_input["forecast"] = (
            decoder_input["continuous"]
            .squeeze(dim=2)[..., -self.horizon_len :]
            .permute(0, 2, 1)
        )  # torch.Size([1024, 20, 128])
        # B X NUM_CORRELATES X FORECAST_LEN
        decoder_input["history"] = (
            decoder_input["continuous"]
            .squeeze(dim=2)[..., : -self.horizon_len]
            .permute(0, 2, 1)
        )  # torch.Size([1024, 20, 128])
        decoder_input["timeseries_full"] = (
            decoder_input["continuous"].squeeze(dim=2).permute(0, 2, 1)
        )  # torch.Size([1024, 20, 256]

        decoder_input["full_discrete_conditions"] = torch.zeros(
            batch_size,
            self.dataset_config.time_series_length,
            0,  # No concept of num_discrete_conditions in FMV2
        )  # (batch, time_series_len, num_discrete_conditions)

        decoder_input["full_continuous_conditions"] = torch.zeros(
            batch_size,
            self.dataset_config.time_series_length,
            0,  # No concept of num_continuous_conditions in FMV2
        )  # (batch, time_series_len, num_continuous)

        # TODO:
        # Scalars are the first 3 dims in the window. mean, variance, etc.
        # Scalar handling should be in the baseline or somewhere related to it.

        for key in decoder_input:
            if decoder_input[key] is not None:
                assert torch.all(
                    torch.logical_not(torch.isnan(decoder_input[key]))
                ), f"NaN values found in {key}"
                assert torch.all(decoder_input[key] != NULL_TOKEN), (
                    f"NULL_TOKEN values found in {key}"
                )

        return decoder_input

    def forward(self, decoder_input):
        batch_size = decoder_input["timestamps"].shape[0]
        num_correlates = decoder_input["timestamps"].shape[1]

        # encode timestamps
        if self.mask_timestamp:
            timestamps = torch.zeros(
                batch_size * num_correlates,
                self.dataset_config.time_series_length,
                self.dataset_config.num_timestamp_features,
            ).to(self.device)
            # timestamps_mask = torch.ones_like(timestamps)  # should it be this?
            timestamps_mask = decoder_input["timestamps_mask"].reshape(
                -1,
                self.dataset_config.time_series_length,
            )
        else:
            timestamps = (
                decoder_input["timestamps"]
                .reshape(
                    -1,
                    self.dataset_config.num_timestamp_features,
                    self.dataset_config.time_series_length,
                )
                .permute(0, 2, 1)
            )
            timestamps_mask = decoder_input["timestamps_mask"].reshape(
                -1,
                self.dataset_config.time_series_length,
            )

        if self.random_vector_instead_text:
            replacement_descriptions = (
                self.text_description_random_vector[
                    decoder_input["dataset_ids"].long().to("cpu")
                ]  # Batch size * num_correlates * 1 * text_embedding_dim
                .reshape(
                    batch_size * num_correlates,
                    1,
                    self.dataset_config.text_embedding_dim,
                )
                .repeat(1, self.dataset_config.time_series_length, 1)
                .to(self.device)
            )
            descriptions_mask = (
                decoder_input["descriptions_mask"]
                .reshape(-1, 1)
                .repeat(1, self.dataset_config.time_series_length)
            )
            # assign the replacement descriptions to the descriptions where the descriptions mask is 1
            descriptions = (
                replacement_descriptions * descriptions_mask.unsqueeze(-1)
            )
        else:
            descriptions = (
                decoder_input["descriptions"]
                .reshape(-1, self.dataset_config.text_embedding_dim, 1)
                .permute(0, 2, 1)
                .repeat(1, self.dataset_config.time_series_length, 1)
            )
            descriptions_mask = (
                decoder_input["descriptions_mask"]
                .reshape(-1, 1)
                .repeat(1, self.dataset_config.time_series_length)
            )

        continuous = (
            decoder_input["continuous"]
            .reshape(-1, 1, self.dataset_config.time_series_length)
            .squeeze(-2)
        )
        continuous_mask = decoder_input["continuous_mask"].reshape(
            -1, self.dataset_config.time_series_length
        )
        # size of continuous = (B X NC) X H
        # size of descriptions = (B X NC) X TEXTDIM
        continuous = self.continuous_to_token(
            continuous,
            continuous_mask,
        )
        sfm_input = torch.cat(
            [continuous, descriptions, timestamps],
            dim=-1,
        )
        sfm_input_mask = torch.logical_or(
            continuous_mask,
            torch.logical_or(descriptions_mask, timestamps_mask),
        )
        sfm_embeddings = self.continuous_embedder(
            sfm_input,
        )

        # NOTE: not needed
        # sfm_embeddings are the input to the stacked_transformer.
        # The size is B X (NC X H) X MODELDIM
        # all zeros indicate null token here
        sfm_embeddings = sfm_embeddings * (
            1 - sfm_input_mask.float().unsqueeze(-1)
        ) + torch.zeros_like(sfm_embeddings) * sfm_input_mask.float().unsqueeze(
            -1
        )
        sfm_embeddings = sfm_embeddings.reshape(
            batch_size,
            num_correlates,
            self.dataset_config.time_series_length,
            self.decoder_model_dims,
        )
        sfm_input_mask = sfm_input_mask.reshape(
            batch_size,
            num_correlates,
            self.dataset_config.time_series_length,
        )
        sfm_embeddings = sfm_embeddings.reshape(
            batch_size, -1, self.decoder_model_dims
        )
        sfm_input_mask = sfm_input_mask.reshape(batch_size, -1)

        timestamp_based_mask = construct_timestamp_mask(
            decoder_input["timestamps"].permute(0, 1, 3, 2),
            sfm_input_mask,
            timestamp_based_mask=self.timestamp_based_mask,
            timestamp_distance_mask=self.timestamp_distance_mask,
        )

        hidden_states = sfm_embeddings
        for stacked_transformer_idx in range(
            len(self.decoder_model.stacked_transformer.layers)
        ):
            # we then apply the stacked transformer layer
            layer = self.decoder_model.stacked_transformer.layers[
                stacked_transformer_idx
            ]
            _, hidden_states = layer(
                hidden_states=hidden_states,
                mask=timestamp_based_mask,
                paddings=None,
                kv_write_indices=None,
                kv_cache=None,
            )  # B, NC X H, CHANNEL

        continuous_target_null_token = torch.ones_like(continuous) * -1
        # continuous null values are represented by -1s.
        continuous_target_null_token = continuous_target_null_token.to(
            self.device
        ).float()
        sfm_target = torch.cat(
            [continuous_target_null_token, descriptions, timestamps],
            dim=-1,
        )
        # sfm_target_embeddings are used to decoder the output value for a required timestamp
        # the target_embedding contains the required text descriptions, timestamps,
        # and null_token indicating the value to be predicted.
        # same size as sfm_embedding
        sfm_target_embeddings = self.continuous_embedder(sfm_target).reshape(
            batch_size, -1, self.decoder_model_dims
        )
        sfm_target_embeddings = sfm_target_embeddings[:, 1:]
        # shifting the embeddings by 1 column and appending with zeros for the last column
        sfm_target_embedding_null_token = (
            torch.zeros_like(sfm_target_embeddings[:, 0])
            .unsqueeze(1)
            .to(self.device)
            .float()
        )
        sfm_target_embeddings = torch.cat(
            [sfm_target_embeddings, sfm_target_embedding_null_token], dim=1
        )

        model_decoding_output = torch.cat(
            [hidden_states, sfm_target_embeddings], dim=-1
        )
        pred_forecast = self.postprocessor(model_decoding_output)

        target_mask = torch.cat(
            [
                sfm_input_mask[:, 1:],
                torch.ones(batch_size, 1).bool().to(self.device),
            ],
            dim=-1,
        )
        invalid_indices = (
            torch.arange(1, num_correlates + 1)
            * self.dataset_config.time_series_length
        ) - 1
        target_mask[:, invalid_indices] = True
        target_mask = torch.logical_or(target_mask, sfm_input_mask)
        target_mask = target_mask.float()

        return {"prediction": pred_forecast, "target_mask": target_mask}

    def get_gradient_stats(self):
        """Get statistics about recorded gradient values."""
        if not self.max_gradient_values:
            return {}

        values = np.array(self.max_gradient_values).astype(float)
        return {
            "max_gradient_mean": float(np.mean(values)),
            "max_gradient_std": float(np.std(values)),
            "max_gradient_min": float(np.min(values)),
            "max_gradient_max": float(np.max(values)),
        }


def get_target_embeddings(synthesizer, decoder_target_subset):
    batch_size = decoder_target_subset["timestamps"].shape[0]
    num_correlates = decoder_target_subset["timestamps"].shape[1]

    context_length = decoder_target_subset["timestamps"].shape[-1]

    # encode timestamps
    if synthesizer.mask_timestamp:
        timestamps = torch.zeros(
            batch_size * num_correlates,
            context_length,
            synthesizer.dataset_config.num_timestamp_features,
        ).to(synthesizer.device)
        timestamps_mask = torch.ones_like(timestamps)
    else:
        timestamps = (
            decoder_target_subset["timestamps"]
            .reshape(
                -1,
                synthesizer.dataset_config.num_timestamp_features,
                context_length,
            )
            .permute(0, 2, 1)
        )
        timestamps_mask = decoder_target_subset["timestamps_mask"].reshape(
            -1,
            context_length,
        )
        del timestamps_mask

    if synthesizer.random_vector_instead_text:
        replacement_descriptions = (
            synthesizer.text_description_random_vector[
                decoder_target_subset["dataset_ids"]
            ]
            .reshape(
                batch_size,
                num_correlates,
                1,
                synthesizer.dataset_config.text_embedding_dim,
            )
            .repeat(1, 1, context_length, 1)
        )
        descriptions_mask = (
            decoder_target_subset["descriptions_mask"]
            .reshape(-1, 1)
            .repeat(1, context_length)
        )
        del descriptions_mask
        # assign the replacement descriptions to the descriptions where the descriptions mask is 1
        descriptions = replacement_descriptions * decoder_target_subset[
            "descriptions_mask"
        ].unsqueeze(-1)
    else:
        descriptions = (
            decoder_target_subset["descriptions"]
            .reshape(-1, synthesizer.dataset_config.text_embedding_dim, 1)
            .permute(0, 2, 1)
            .repeat(1, context_length, 1)
        )
        descriptions_mask = (
            decoder_target_subset["descriptions_mask"]
            .reshape(-1, 1)
            .repeat(1, context_length)
        )
        del descriptions_mask

    continuous = (
        decoder_target_subset["continuous"]
        .reshape(-1, 1, context_length)
        .squeeze(-2)
    )
    continuous_mask = decoder_target_subset["continuous_mask"].reshape(
        -1, context_length
    )
    continuous = synthesizer.continuous_to_token(
        continuous,
        continuous_mask,
    )
    continuous_target_null_token = torch.ones_like(continuous) * -1
    continuous_target_null_token = continuous_target_null_token.to(
        synthesizer.device
    ).float()
    sfm_target = torch.cat(
        [continuous_target_null_token, descriptions, timestamps],
        dim=-1,
    )
    sfm_target_embeddings = synthesizer.continuous_embedder(
        sfm_target,
    )
    sfm_target_embeddings = sfm_target_embeddings.reshape(
        batch_size,
        num_correlates,
        context_length,
        synthesizer.decoder_model_dims,
    )
    return sfm_target_embeddings


# Same as SynthefyFoundationForecastingModelV2.forward(), except that it just returns the hidden states.
def get_hidden_states(synthesizer, decoder_input_subset):
    batch_size = decoder_input_subset["timestamps"].shape[0]
    num_correlates = decoder_input_subset["timestamps"].shape[1]

    context_length = decoder_input_subset["timestamps"].shape[-1]

    # encode timestamps
    if synthesizer.mask_timestamp:
        timestamps = torch.zeros(
            batch_size * num_correlates,
            context_length,
            synthesizer.dataset_config.num_timestamp_features,
        ).to(synthesizer.device)
        timestamps_mask = torch.ones_like(timestamps)
    else:
        timestamps = (
            decoder_input_subset["timestamps"]
            .reshape(
                -1,
                synthesizer.dataset_config.num_timestamp_features,
                context_length,
            )
            .permute(0, 2, 1)
        )
        timestamps_mask = decoder_input_subset["timestamps_mask"].reshape(
            -1,
            context_length,
        )

    if synthesizer.random_vector_instead_text:
        replacement_descriptions = (
            synthesizer.text_description_random_vector[
                decoder_input_subset["dataset_ids"]
            ]
            .reshape(
                batch_size,
                num_correlates,
                1,
                synthesizer.dataset_config.text_embedding_dim,
            )
            .repeat(1, 1, context_length, 1)
        )
        descriptions_mask = (
            decoder_input_subset["descriptions_mask"]
            .reshape(-1, 1)
            .repeat(1, context_length)
        )
        # assign the replacement descriptions to the descriptions where the descriptions mask is 1
        descriptions = replacement_descriptions * decoder_input_subset[
            "descriptions_mask"
        ].unsqueeze(-1)
    else:
        descriptions = (
            decoder_input_subset["descriptions"]
            .reshape(-1, synthesizer.dataset_config.text_embedding_dim, 1)
            .permute(0, 2, 1)
            .repeat(1, context_length, 1)
        )
        descriptions_mask = (
            decoder_input_subset["descriptions_mask"]
            .reshape(-1, 1)
            .repeat(1, context_length)
        )

    continuous = (
        decoder_input_subset["continuous"]
        .reshape(-1, 1, context_length)
        .squeeze(-2)
    )
    continuous_mask = decoder_input_subset["continuous_mask"].reshape(
        -1, context_length
    )
    continuous = synthesizer.continuous_to_token(
        continuous,
        continuous_mask,
    )
    sfm_input = torch.cat(
        [continuous, descriptions, timestamps],
        dim=-1,
    )
    sfm_input_mask = torch.logical_or(
        continuous_mask,
        torch.logical_or(descriptions_mask, timestamps_mask),
    )
    sfm_embeddings = synthesizer.continuous_embedder(
        sfm_input,
    )

    # NOTE: not needed
    sfm_embeddings = sfm_embeddings * (
        1 - sfm_input_mask.float().unsqueeze(-1)
    ) + torch.zeros_like(sfm_embeddings) * sfm_input_mask.float().unsqueeze(-1)
    sfm_embeddings = sfm_embeddings.reshape(
        batch_size,
        num_correlates,
        context_length,
        synthesizer.decoder_model_dims,
    )
    sfm_input_mask = sfm_input_mask.reshape(
        batch_size,
        num_correlates,
        context_length,
    )
    sfm_embeddings = sfm_embeddings.reshape(
        batch_size, -1, synthesizer.decoder_model_dims
    )
    sfm_input_mask = sfm_input_mask.reshape(batch_size, -1)

    timestamp_based_mask = construct_timestamp_mask(
        decoder_input_subset["timestamps"].permute(0, 1, 3, 2), sfm_input_mask
    )

    hidden_states = sfm_embeddings
    for stacked_transformer_idx in range(
        len(synthesizer.decoder_model.stacked_transformer.layers)
    ):
        # we then apply the stacked transformer layer
        layer = synthesizer.decoder_model.stacked_transformer.layers[
            stacked_transformer_idx
        ]
        _, hidden_states = layer(
            hidden_states=hidden_states,
            mask=timestamp_based_mask,
            paddings=None,
            kv_write_indices=None,
            kv_cache=None,
        )  # B, NC X H, CHANNEL

    return hidden_states


def forecast_next_step(
    synthesizer, required_hidden_states, required_target_embeddings
):
    emb = torch.cat(
        [required_hidden_states, required_target_embeddings], dim=-1
    )
    return synthesizer.postprocessor(emb)
