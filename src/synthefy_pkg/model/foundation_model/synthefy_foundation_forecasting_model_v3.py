import numpy as np
import torch
from einops import rearrange, reduce, repeat
from timesfm.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    ResidualBlock,
    get_large_negative_number,
)
from torch import nn
from tqdm import tqdm

from synthefy_pkg.model.architectures.decoder_utils import PositionalEmbedding

NULL_TOKEN = -999999


def continuous_to_token(
    continuous_input: torch.Tensor, continuous_mask: torch.Tensor
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

    # remove the extra dimension that comes from the continuous values
    tokens = tokens.squeeze(-2)

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


def obtain_timestamps_in_seconds(timestamps):
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
    multiplies = multiplies.unsqueeze(0).unsqueeze(
        0
    )  # 1 X 1 X NUM_TIMESTAMP_FEATURES

    relevant_timestamps = timestamps[..., relevant_timestamp_indices]
    timestamps_in_seconds = torch.sum(relevant_timestamps * multiplies, dim=-1)
    return timestamps_in_seconds


def obtain_causal_attn_mask(timestamps, invalid_mask, target_mask):
    """
    Construct a mask from the timestamps

    timestamps: Batch x num_correlates x window_size x num_timestamps
    timestamp_mask: Batch x (num_correlates x window_size)
    """
    # TODO: get from config later, but making this function self contained
    timestamps_in_seconds = obtain_timestamps_in_seconds(timestamps)
    ts_i = timestamps_in_seconds.unsqueeze(
        -1
    )  # B X NUM_CORRELATES * TIME_SERIES_LENGTH X 1
    ts_j = timestamps_in_seconds.unsqueeze(
        -2
    )  # B X 1 X NUM_CORRELATES * TIME_SERIES_LENGTH

    # produce 1 between any two values where the second value is larger than the first
    # 0 if they are equal, -1 if the second value is smaller than the first
    # NOTE: <= is used instead of < to ensure that the duplicate timestamps are masked
    causal_attn_mask = (
        ts_i <= ts_j
    )  # Batch x (window_size * num_correlates) x (window_size * num_correlates)

    # applying the invalid mask
    causal_attn_mask = torch.logical_or(
        causal_attn_mask, invalid_mask.unsqueeze(-1)
    )
    causal_attn_mask = torch.logical_or(
        causal_attn_mask, invalid_mask.unsqueeze(-2)
    )

    # applying the target mask (B, 1, NT)
    causal_attn_mask = torch.logical_or(
        causal_attn_mask, target_mask.unsqueeze(-2)
    )

    # applying self attention mask
    causal_attn_mask = torch.logical_and(
        causal_attn_mask,
        (1 - torch.eye(causal_attn_mask.shape[-1]))
        .unsqueeze(0)
        .bool()
        .repeat(causal_attn_mask.shape[0], 1, 1)
        .to(causal_attn_mask.device),
    )

    causal_attn_mask = causal_attn_mask.float()

    causal_attn_mask = causal_attn_mask.unsqueeze(
        1
    )  # Batch x 1 x (window_size * num_correlates) x (window_size * num_correlates)

    large_negative_number = get_large_negative_number(timestamps.dtype)
    causal_attn_mask = causal_attn_mask * large_negative_number

    return causal_attn_mask


def obtain_mask(elem, category=None):
    nan_mask = torch.isnan(elem)
    null_mask = elem == NULL_TOKEN
    mask = torch.logical_or(nan_mask, null_mask)
    mask_einops = reduce(mask, "b nc t f -> b nc t", "sum").bool()
    mask = mask.sum(dim=-1).bool()
    assert torch.equal(mask, mask_einops), "The mask is not equal"
    if category == "textual":
        invalid_mask = (elem == 0).all(dim=-1)
        mask = torch.logical_or(mask, invalid_mask)
    return mask


def get_torch_transformer(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=channels,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


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


class SynthefyFoundationForecastingModelV3(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

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

        # num_patches = self.context_len // self.decoder_input_patch_len
        self.postprocessor = ResidualBlock(
            input_dims=self.decoder_model_dims,
            hidden_dims=self.decoder_model_dims,
            output_dims=1,
        ).to(self.device)

        self.target_mask_ratio = config.training_config.target_mask_ratio
        self.block_target_mask_range = (
            config.foundation_model_config.block_target_mask_range
        )
        self.block_target_mask_mean = (
            config.foundation_model_config.block_target_mask_mean
        )
        self.block_mask_num = config.foundation_model_config.block_mask_num

        # Initialize scaling for all attention layers in the decoder
        self._initialize_attention_scaling()

    def _initialize_attention_scaling(self):
        # Iterate over all layers in the decoder
        for layer in self.decoder_model.stacked_transformer.layers:
            # Access the TimesFMAttention instance
            attention_layer = layer.self_attn
            # Initialize the scaling parameter
            nn.init.uniform_(attention_layer.scaling, -0.5, 0.5)

    def _generate_target_mask(self, batch_size, num_correlates):
        # generates a target mask, that is, which values to predict
        # if self.block_target_mask_mean > 0, then we mask blocks of a certain length
        #     If the range is 0 and the number of mask is length / 2 and the number of masks is 1, mask out the future
        #     otherwise, select block_mask_num random blocks of length self.block_target_mask_mean +- self.block_target_mask_range
        # if self.block_target_mask_mean <= 0, then we mask a random number of values between 0 and self.target_mask_ratio
        if self.block_target_mask_mean > 0:
            if (
                self.block_target_mask_mean
                == self.dataset_config.time_series_length / 2
                and self.block_target_mask_range == 0
                and self.block_mask_num == 1
            ):
                # this is the case where we mask the future half of the time series
                target_mask = torch.zeros(
                    batch_size,
                    num_correlates,
                    self.dataset_config.time_series_length,
                ).to(self.device)
                target_mask[
                    ..., self.dataset_config.time_series_length // 2 :
                ] = 1
                target_mask = target_mask.reshape(
                    batch_size,
                    num_correlates * self.dataset_config.time_series_length,
                )
            else:
                # randomly decide on lengths for the block masks
                target_mask_lengths = torch.randint(
                    max(
                        0,
                        self.block_target_mask_mean
                        - self.block_target_mask_range,
                    ),
                    min(
                        self.dataset_config.time_series_length,
                        self.block_target_mask_mean
                        + self.block_target_mask_range,
                    ),
                    (batch_size, num_correlates, self.block_mask_num),
                )
                # Initialize target_mask with correct shape
                target_mask = torch.zeros(
                    batch_size,
                    num_correlates,
                    self.dataset_config.time_series_length,
                ).to(self.device)

                # Generate random starting positions for each mask
                # Ensure start position + mask length doesn't exceed time_series_length
                max_starts = torch.clamp(
                    self.dataset_config.time_series_length
                    - target_mask_lengths,
                    min=0,
                )
                random_starts = torch.floor(
                    torch.rand(
                        batch_size, num_correlates, self.block_mask_num
                    ).to(self.device)
                    * (max_starts + 1)
                ).long()

                # Create position indices tensor
                positions = (
                    torch.arange(self.dataset_config.time_series_length)
                    .view(1, 1, 1, -1)
                    .to(self.device)
                )

                # Calculate mask for each random segment
                # For each (batch, correlate, block), a position is masked if:
                # start_pos <= position < start_pos + mask_length
                starts = random_starts.unsqueeze(-1)  # [B, C, N, 1]
                lengths = target_mask_lengths.unsqueeze(-1)  # [B, C, N, 1]
                block_masks = (
                    (positions >= starts) & (positions < starts + lengths)
                ).float()

                # Combine all block masks (logical OR)
                target_mask = torch.max(block_masks, dim=2)[0]

                # Reshape to match expected output shape
                target_mask = target_mask.reshape(
                    batch_size,
                    num_correlates * self.dataset_config.time_series_length,
                )
        else:
            # randomly mask
            target_mask = (
                torch.rand(
                    batch_size,
                    num_correlates * self.dataset_config.time_series_length,
                )
                < self.target_mask_ratio
            ).to(self.device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        return target_mask

    def prepare_training_input(self, train_batch, *args, **kwargs):
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

        target_mask = self._generate_target_mask(
            batch_size=self.batch_size,
            num_correlates=self.num_correlates,
        )

        target_mask = torch.logical_and(
            target_mask, ~mask
        )  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)

        decoder_input = {
            "timestamps": all_timestamps,
            "descriptions": all_descriptions,
            "continuous": all_continuous,
            "continuous_tokens": all_continuous_tokens,
            "mask": mask,
            "target_mask": target_mask,
            "dataset_ids": all_dataset_ids,
            "target": all_continuous,
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

    def forward(self, decoder_input, inference_mode=False):
        # invalid or nan timestamps are set to -1
        # invalid or nan descriptions are set to 0
        # invalid or nan continuous tokens are set to -1
        timestamps = decoder_input["timestamps"]
        descriptions = decoder_input["descriptions"]
        continuous_tokens = decoder_input["continuous_tokens"]

        invalid_mask = decoder_input["mask"]
        target_mask = decoder_input["target_mask"]

        # target continuous tokens are set to -1
        if inference_mode:
            # the target continuous tokens are already set to -1
            assert torch.all(continuous_tokens[target_mask] == -1), (
                "Target continuous tokens are not set to -1"
            )
        continuous_tokens[target_mask] = -1

        # masking out the timestamp ONLY affects the tokens
        sfm_input = torch.cat(
            [
                continuous_tokens,
                descriptions,
                timestamps * int(not self.mask_timestamp),
            ],
            dim=-1,
        )
        # invalid tokens represented by mask are set to (-1,0,-1)
        # target tokens represented by target_mask are set to (-1,text,timestamp)
        sfm_embeddings = self.continuous_embedder(
            sfm_input,
        )

        # apply position embeddings to all tokens before zeroing out invalids
        if self.position_embedding == "all":
            sfm_embeddings = self.position_embedder(sfm_embeddings)
        elif self.position_embedding == "correlate":
            # first break into correlates but keep the batches
            sfm_embeddings = rearrange(
                sfm_embeddings,
                "b (nc t) d -> (b nc) t d",
                nc=self.num_correlates,
            )
            # apply position embeddings to tokens by correlate TODO: not sure this works
            sfm_embeddings = self.position_embedder(sfm_embeddings)
            # then break back into single stream of tokens
            sfm_embeddings = rearrange(
                sfm_embeddings, "(b nc) t d -> b (nc t) d", b=self.batch_size
            )

        sfm_embeddings = sfm_embeddings * (
            1 - invalid_mask.float().unsqueeze(-1)
        ) + torch.zeros_like(sfm_embeddings) * invalid_mask.float().unsqueeze(
            -1
        )
        # invalid tokens are set to 0

        causal_attn_mask = obtain_causal_attn_mask(
            timestamps, invalid_mask, target_mask
        )

        hidden_states = sfm_embeddings
        for stacked_transformer_idx in range(
            len(self.decoder_model.stacked_transformer.layers)
        ):
            # we then apply the stacked transformer layer
            layer = self.decoder_model.stacked_transformer.layers[
                stacked_transformer_idx
            ]

            scores, hidden_states = layer(
                hidden_states=hidden_states,
                mask=causal_attn_mask,
                paddings=None,
                kv_write_indices=None,
                kv_cache=None,
            )  # B, NC X H, CHANNEL

        pred_forecast = self.postprocessor(hidden_states)

        return {"prediction": pred_forecast, "target_mask": target_mask.float()}

    def autoregressive_forecast(
        self,
        batch,
        synthesizer,
        history_length=None,
        forecast_length=None,
        use_ground_truth_for_next_step=False,
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

        timestamps_history = timestamps[..., :history_length, :]
        timestamps_forecast = timestamps[..., history_length:, :]

        descriptions_history = descriptions[..., :history_length, :]
        descriptions_forecast = descriptions[..., history_length:, :]

        continuous_tokens_history = continuous_tokens[..., :history_length, :]
        continuous_tokens_forecast = continuous_tokens[..., history_length:, :]

        invalid_mask_history = invalid_mask[..., :history_length]
        # Remove or comment out the next line to fix F841
        # invalid_mask_forecast = invalid_mask[..., history_length:]

        target_mask_history = (
            torch.zeros_like(invalid_mask_history).bool().to(self.device)
        )

        # initialize the forecast input
        invalid_mask_forecast_input = invalid_mask_history.clone()
        target_mask_forecast_input = target_mask_history.clone()
        continuous_tokens_forecast_input = continuous_tokens_history.clone()
        timestamps_forecast_input = timestamps_history.clone()
        descriptions_forecast_input = descriptions_history.clone()
        forecast_output_list = []
        for fidx in tqdm(
            range(forecast_length), total=forecast_length, desc="Forecasting"
        ):
            # for each input, we combine nc and t
            # create a invalid indicator flag for the forecast
            invalid_indicator_flag = (
                torch.zeros((batch_size, num_correlates, 1))
                .bool()
                .to(self.device)
            )
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

            input_dict = {
                "timestamps": timestamps_forecast_input,
                "descriptions": descriptions_forecast_input,
                "continuous_tokens": continuous_tokens_forecast_input,
                "mask": invalid_mask_forecast_input,
                "target_mask": target_mask_forecast_input,
                "window_size": history_length + fidx + 1,
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
        # print(
        #     "decoder_input['continuous'].shape",
        #     decoder_input["continuous"].shape,
        # )
        # print("ts.shape", ts.shape)
        true_history = ts[
            :,
            :,
            : timeseries_length - forecast_length,
        ]
        # print("true_history.shape", true_history.shape)
        true_forecast = ts[:, :, -forecast_length:]
        # print("true_forecast.shape", true_forecast.shape)
        mask = decoder_input["mask"].reshape(
            -1,
            self.dataset_config.num_correlates,
            timeseries_length,
        )
        # print("mask.shape", mask.shape)
        forecast_mask = mask[:, :, -forecast_length:]
        # print("forecast_mask.shape", forecast_mask.shape)

        return {
            "predicted_forecast": predicted_forecast,
            "true_forecast": true_forecast,
            "forecast_mask": forecast_mask,
            "history": true_history,
            "dataset_ids": decoder_input["dataset_ids"],
        }
