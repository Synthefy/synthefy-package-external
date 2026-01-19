import torch
from einops import rearrange, repeat
from loguru import logger
from torch import nn

from synthefy_pkg.model.architectures.decoder_utils import PositionalEmbedding
from synthefy_pkg.model.foundation_model.base_foundation_forecasting_model import (
    BaseFoundationForecastingModel,
)
from synthefy_pkg.model.foundation_model.continuous_embedder import (
    SynthefyFoundationModelContinuousEmbedder,
)
from synthefy_pkg.model.foundation_model.distributional_forecasting_utils import (
    FullSupportBarDistribution,
)
from synthefy_pkg.model.foundation_model.residual_block import ResidualBlock
from synthefy_pkg.model.foundation_model.stacked_decoder import (
    CorrelateAttention,
    StackedDecoder,
)
from synthefy_pkg.model.foundation_model.utils import (
    get_large_negative_number,
    obtain_causal_attn_mask,
    obtain_column_attn_mask,
    obtain_future_correlate_attn_mask,
    obtain_future_triangular_attn_mask,
    obtain_interpolation_attn_mask,
    obtain_row_attn_mask,
    obtain_train_test_block_attn_mask,
)


class SynthefyFoundationForecastingModelV3E(BaseFoundationForecastingModel):
    def __init__(self, config):
        super().__init__(config)

        self.correlate_attention_scaling = (
            config.foundation_model_config.correlate_attention_scaling
        )
        if self.correlate_attention_scaling > 0:
            self.correlate_attention = CorrelateAttention(
                hidden_size=int(self.decoder_model_dims * 2),
                num_heads=self.decoder_num_heads,
                num_k_heads=self.decoder_num_heads,
                head_dim=self.decoder_model_dims,
            )
        # self.correlate_attention_scaling = nn.Parameter(torch.zeros(1)) # could use learnable parameter possibly
        self.stacked_transformer = StackedDecoder(
            hidden_size=self.decoder_model_dims,
            intermediate_size=self.decoder_model_dims,
            num_heads=self.decoder_num_heads,
            num_kv_heads=self.decoder_num_heads,
            head_dim=self.decoder_model_dims // self.decoder_num_heads,
            num_layers=self.decoder_num_layers,
        )

        # Put the model into training mode
        # self.stacked_transformer.train()

        self.use_column_identifier = (
            config.foundation_model_config.use_column_identifier
        )
        if self.use_column_identifier:
            logger.info(
                "The training uses column identifiers: ",
                self.use_column_identifier,
            )
            logger.info(
                "The text info is used with probability: ",
                config.training_config.description_mask_ratio,
            )
        else:
            logger.info(
                "The training does not use column identifiers: ",
                self.use_column_identifier,
            )
            if config.training_config.description_mask_ratio > 0:
                raise ValueError(
                    "Description mask ratio is greater than 0 but column identifiers are not used"
                )
                # if the column identifiers are not used, then the description mask ratio should be 0
                # menaing, we never mask the text info as this is essential for the uniqueness of tokens across correlates.

        if self.config.foundation_model_config.replace_timestamp_with_pos_enc:
            self.timestamp_pos_embedder = PositionalEmbedding(
                d_model=self.decoder_model_dims,
                max_len=self.time_series_length,
            )
            timestamp_token_dims = self.decoder_model_dims
        else:
            timestamp_token_dims = self.dataset_config.num_timestamp_features
        self.continuous_embedder = SynthefyFoundationModelContinuousEmbedder(
            model_dims=self.decoder_model_dims,
            timeseries_token_dims=config.foundation_model_config.timeseries_token_dims,
            textual_token_dims=self.dataset_config.text_embedding_dim,
            timestamp_token_dims=timestamp_token_dims,
            max_columns=self.dataset_config.num_correlates,
            use_column_identifier=self.use_column_identifier,
            train_as_univariate_model=self.dataset_config.train_as_univariate_model,
        )

        # what kind of postprocessor to use?
        if self.config.foundation_model_config.generate_point_forecast:
            logger.info("Learning a model for point forecasts")
            self.postprocessor = ResidualBlock(
                input_dims=self.decoder_model_dims,
                hidden_dims=self.decoder_model_dims,
                output_dims=1,
            )
        elif self.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
            logger.info(
                "Learning a model for probabilistic forecasts using bins"
            )
            self.postprocessor = ResidualBlock(
                input_dims=self.decoder_model_dims,
                hidden_dims=self.decoder_model_dims,
                output_dims=self.config.foundation_model_config.num_bins,
            )
            num_bins = self.config.foundation_model_config.num_bins

            # Create num_bins+1 borders that are gaussian distributed from -100 to 100
            normal = torch.distributions.Normal(0, 1)
            # Get evenly spaced points in the CDF of a standard normal
            quantiles = torch.linspace(0.001, 0.999, num_bins + 1)
            # Transform to standard normal quantiles
            borders = normal.icdf(quantiles)
            # Scale to [-100, 100] range
            borders = (
                borders
                * self.config.foundation_model_config.absolute_max_bar_value
                / borders.abs().max()
            )
            print(borders)
            self.distribution = FullSupportBarDistribution(borders)
        else:
            raise ValueError("Invalid postprocessor configuration")

        # Initialize scaling for all attention layers in the decoder
        self._initialize_attention_scaling()

    def _initialize_attention_scaling(self):
        # Iterate over all layers in the decoder
        for layer in self.stacked_transformer.layers:
            # Access the TimesFMAttention instance
            attention_layer = layer.self_attn
            # Initialize the scaling parameter
            nn.init.uniform_(attention_layer.scaling, -0.5, 0.5)

    def forward(
        self, decoder_input, inference_mode=False, keep_attention_weights=False
    ):
        self.device = next(self.parameters()).device

        # invalid or nan timestamps are set to -1
        # invalid or nan descriptions are set to 0
        # invalid or nan continuous tokens are set to -1
        timestamps = decoder_input["timestamps"].to(self.device).float()
        descriptions = decoder_input["descriptions"].to(self.device).float()
        continuous_tokens = (
            decoder_input["continuous_tokens"].to(self.device).float()
        )
        column_identifiers = (
            decoder_input["column_identifiers"].to(self.device).float()
        )
        description_mask_flag = decoder_input["description_mask_flag"]

        if description_mask_flag:
            assert self.use_column_identifier, (
                "Description mask flag is true but column identifiers are not used"
            )
            descriptions = torch.zeros_like(descriptions)
            # note that description_mask_flag can be true only if the description mask ratio is greater than 0
            # and the description mask ratio is greater than 0 only when the column identifiers are used
            # logic being, we have column identifiers for unique identification of correlates
            # and so we mask the text info

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
        if self.config.foundation_model_config.replace_timestamp_with_pos_enc:
            timestamps_pos_embeddings = self.timestamp_pos_embedder(timestamps)
            timestamps = repeat(
                timestamps_pos_embeddings,
                "1 t d -> b t d",
                b=timestamps.shape[0],
            )
        sfm_input = torch.cat(
            [
                continuous_tokens,
                descriptions,
                timestamps * int(not self.mask_timestamp),
                column_identifiers,
            ],
            dim=-1,
        )
        batch_size = sfm_input.shape[0]
        # invalid tokens represented by mask are set to (-1,0,-1)
        # target tokens represented by target_mask are set to (-1,text,timestamp)
        sfm_embeddings = self.continuous_embedder(
            sfm_input,
        )

        # apply position embeddings to all tokens before zeroing out invalids
        if self.position_embedding == "all":
            sfm_embeddings = sfm_embeddings + self.position_embedder(
                sfm_embeddings
            )
        elif self.position_embedding == "correlate":
            # first break into correlates but keep the batches
            sfm_embeddings = rearrange(
                sfm_embeddings,
                "b (nc t) d -> (b nc) t d",
                nc=self.num_correlates,
            )
            # apply position embeddings to tokens by correlate TODO: not sure this works
            sfm_embeddings = sfm_embeddings + self.position_embedder(
                sfm_embeddings
            )
            # then break back into single stream of tokens
            sfm_embeddings = rearrange(
                sfm_embeddings, "(b nc) t d -> b (nc t) d", b=batch_size
            )

        sfm_embeddings = sfm_embeddings * (
            1 - invalid_mask.float().unsqueeze(-1)
        ) + torch.zeros_like(sfm_embeddings) * invalid_mask.float().unsqueeze(
            -1
        )
        # invalid tokens are set to 0

        if self.attention_masking_scheme == "train_test_block":
            causal_attn_mask = obtain_train_test_block_attn_mask(
                decoder_input["train_sizes"],
                decoder_input["mask"],
                self.time_series_length,
                self.num_correlates,
                batch_size,
                self.device,
                sfm_embeddings.dtype,
            )
        elif self.attention_masking_scheme == "future_triangular":
            causal_attn_mask = obtain_future_triangular_attn_mask(
                batch_size,
                self.num_correlates,
                self.time_series_length,
                invalid_mask,
                target_mask,
                self.device,
                sfm_embeddings.dtype,
            )
        elif self.attention_masking_scheme == "interpolation":
            causal_attn_mask = obtain_interpolation_attn_mask(
                batch_size,
                self.num_correlates,
                self.time_series_length,
                invalid_mask,
                target_mask,
                self.device,
                sfm_embeddings.dtype,
            )
        elif self.attention_masking_scheme == "future_correlate":
            causal_attn_mask = obtain_future_correlate_attn_mask(
                batch_size,
                self.num_correlates,
                self.time_series_length,
                invalid_mask,
                target_mask,
                self.device,
                sfm_embeddings.dtype,
            )
        else:
            causal_attn_mask = obtain_causal_attn_mask(
                timestamps, invalid_mask, target_mask
            )

        hidden_states = sfm_embeddings
        for stacked_transformer_idx in range(
            len(self.stacked_transformer.layers)
        ):
            # we then apply the stacked transformer layer
            layer = self.stacked_transformer.layers[stacked_transformer_idx]

            scores, hidden_states = layer(
                hidden_states=hidden_states,
                mask=causal_attn_mask,
                paddings=None,
                kv_write_indices=None,
                kv_cache=None,
                keep_attention_weights=keep_attention_weights,
            )  # B, NC X H, CHANNEL

        pred_forecast = self.postprocessor(hidden_states)

        if self.config.foundation_model_config.generate_probabilistic_forecast_using_bins:
            mode_forecast = self.distribution.mode(pred_forecast).unsqueeze(-1)
            mean_forecast = self.distribution.mean(pred_forecast).unsqueeze(-1)
            return {
                "prediction": mean_forecast,
                "target_mask": target_mask.float(),
                "mode_forecast": mode_forecast,
                "logits": pred_forecast,
                "tokens_index": (
                    None,
                    None,
                ),
                "embeddings": hidden_states,
            }
        else:
            return {
                "prediction": pred_forecast,
                "target_mask": target_mask.float(),
                "tokens_index": (
                    None,
                    None,
                ),
                "embeddings": hidden_states,
            }
