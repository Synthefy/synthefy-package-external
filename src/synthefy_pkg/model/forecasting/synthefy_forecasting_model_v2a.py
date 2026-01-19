import os
from pathlib import Path

import torch
from torch import nn

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.architectures.decoder_utils import (
    AttentionLayer,
    EncoderAvecCrossAttentionTriple,
    EncoderLayer,
    FullAttention,
    PatchEmbedding,
    get_torch_trans_decoder,
)
from synthefy_pkg.model.diffusion.diffusion_transformer import (
    MetaDataEncoder,
    TimestampEncoder,
)
from synthefy_pkg.model.forecasting.synthefy_forecasting_model_v1 import (
    FlattenHead,
)

COMPILE = True


class SynthefyForecastingModelV2a(nn.Module):
    def __init__(self, config: Configuration):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.config = config

        self.decoder_config = self.config.decoder_config
        self.dataset_config = self.config.dataset_config
        self.denoiser_config = self.config.denoiser_config
        self.device = self.config.device
        self.seq_len = self.dataset_config.time_series_length
        self.pred_len = self.dataset_config.forecast_length
        padding = self.decoder_config.stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model=self.decoder_config.d_model,
            patch_len=self.decoder_config.patch_len,
            stride=self.decoder_config.stride,
            padding=padding,
            dropout=self.decoder_config.dropout,
        )

        # metadata encoder
        if self.dataset_config.use_timestamp:
            self.timestamp_encoder = TimestampEncoder(self.config)

        if self.denoiser_config.use_metadata:
            self.metadata_encoder = MetaDataEncoder(
                self.dataset_config,
                self.denoiser_config,
                self.device,
            )

        # Encoder
        self.encoder = EncoderAvecCrossAttentionTriple(
            self_attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.decoder_config.factor,
                            attention_dropout=self.decoder_config.dropout,
                            output_attention=self.decoder_config.output_attention,
                        ),
                        self.decoder_config.d_model,
                        self.decoder_config.n_heads,
                    ),
                    self.decoder_config.d_model,
                    self.decoder_config.d_ff,
                    dropout=self.decoder_config.dropout,
                    activation=self.decoder_config.activation,
                )
                for _ in range(self.decoder_config.e_layers)
            ],
            cross_attn_layers1=[
                get_torch_trans_decoder(
                    heads=self.decoder_config.n_heads,
                    layers=1,
                    channels=self.decoder_config.d_model,
                )
                for _ in range(self.decoder_config.e_layers)
            ],
            cross_attn_layers2=[
                get_torch_trans_decoder(
                    heads=self.decoder_config.n_heads,
                    layers=1,
                    channels=self.decoder_config.d_model,
                )
                for _ in range(self.decoder_config.e_layers)
            ],
            conv_layers=None,
            norm_layer=torch.nn.LayerNorm(self.decoder_config.d_model),
        )

        # Prediction Head
        self.history_length = self.seq_len - self.pred_len
        self.num_patches = int(
            (self.history_length - self.decoder_config.patch_len)
            / self.decoder_config.stride
            + 2
        )
        self.head_nf = self.decoder_config.d_model * self.num_patches
        self.head = FlattenHead(
            self.dataset_config.num_channels,
            self.head_nf,
            self.pred_len,
            head_dropout=self.decoder_config.dropout,
        )

    def prepare_training_input(self, train_batch, *args, **kwargs):
        # sample: shape [batch, channels, time]
        sample = train_batch["timeseries_full"].float().to(self.device)

        # Ensure number of channels is correct
        assert sample.shape[1] == self.dataset_config.num_channels

        # discrete conditions: shape [batch, time, num_discrete_conditions]
        # or shape [batch, num_discrete_conditions] if conditions are to be repeated
        # across the timeseries
        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )

        # If discrete conditions should be repeated across the timeseries,
        # unsqueeze: [batch, num_discrete_conditions] -> [batch, 1, num_discrete_conditions]
        # repeat: [batch, 1, num_discrete_conditions] -> [batch, time, num_discrete_conditions]
        if len(discrete_label_embedding.shape) == 2:
            discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
            discrete_label_embedding = discrete_label_embedding.repeat(
                1, sample.shape[2], 1
            )

        # Ensure discrete conditions time dimension is correct
        assert discrete_label_embedding.shape[1] == sample.shape[2], (
            "Wrong shape for discrete_label_embedding"
        )

        # Ensure discrete conditions number of conditions is correct
        assert (
            discrete_label_embedding.shape[2]
            == self.dataset_config.num_discrete_conditions
        ), "Wrong shape for discrete_label_embedding"

        # continuous conditions: shape [batch, time, num_continuous_conditions]
        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )

        # If continuous conditions should be repeated across the timeseries,
        # unsqueeze: [batch, num_continuous_conditions] -> [batch, 1, num_continuous_conditions]
        # repeat: [batch, 1, num_continuous_conditions] -> [batch, time, num_continuous_conditions]
        if len(continuous_label_embedding.shape) == 2:
            continuous_label_embedding = continuous_label_embedding.unsqueeze(1)
            continuous_label_embedding = continuous_label_embedding.repeat(
                1, sample.shape[2], 1
            )

        # Ensure continuous conditions time dimension is correct
        assert continuous_label_embedding.shape[1] == sample.shape[2], (
            "Wrong shape for continuous_label_embedding"
        )

        # Ensure continuous conditions number of conditions is correct
        assert (
            continuous_label_embedding.shape[2]
            == self.dataset_config.num_continuous_labels
        ), "Wrong shape for continuous_label_embedding"

        # Split samples across time dimension to history and forecast
        history = sample[:, :, : -self.pred_len]
        forecast = sample[:, :, -self.pred_len :]

        # [batch, channels, time] -> [batch, time, channels]
        history = history.permute(0, 2, 1)
        forecast = forecast.permute(0, 2, 1)

        # Split discrete conditions across time dimension to history and forecast
        history_discrete_cond_input = discrete_label_embedding[
            :, : -self.pred_len, :
        ]
        forecast_discrete_cond_input = discrete_label_embedding[
            :, -self.pred_len :, :
        ]

        # Split continuous conditions across time dimension to history and forecast
        history_continuous_cond_input = continuous_label_embedding[
            :, : -self.pred_len, :
        ]
        forecast_continuous_cond_input = continuous_label_embedding[
            :, -self.pred_len :, :
        ]
        mask = None

        denoiser_input = {
            "history": history,
            "forecast": forecast,
            "history_discrete_cond_input": history_discrete_cond_input,
            "history_continuous_cond_input": history_continuous_cond_input,
            "forecast_discrete_cond_input": forecast_discrete_cond_input,
            "forecast_continuous_cond_input": forecast_continuous_cond_input,
            "mask": mask,
            "full_discrete_conditions": discrete_label_embedding,
            "full_continuous_conditions": continuous_label_embedding,
        }

        if self.dataset_config.use_timestamp:
            # timestamp conditions shape: [batch, time, num_timestamp_conditions]
            timestamp_label_embedding = (
                train_batch["timestamp_label_embedding"].float().to(self.device)
            )

            # Ensure timestamp conditions time dimension is correct
            assert timestamp_label_embedding.shape[1] == sample.shape[2], (
                "Wrong shape for timestamp_label_embedding"
            )

            # Ensure timestamp conditions number of conditions is correct
            assert (
                timestamp_label_embedding.shape[2]
                == self.dataset_config.num_timestamp_labels
            ), "Wrong shape for timestamp_label_embedding"

            # Split timestamp conditions across time dimension to history and forecast
            history_time_feat = timestamp_label_embedding[
                :, : -self.pred_len, :
            ]
            forecast_time_feat = timestamp_label_embedding[
                :, -self.pred_len :, :
            ]

            denoiser_input["history_time_feat"] = history_time_feat
            denoiser_input["forecast_time_feat"] = forecast_time_feat

        return denoiser_input

    def forward(self, forecast_input):
        # x_enc = timeseries history. Shape = [batch, time, channels]
        x_enc = forecast_input["history"]

        # If we're using metadata, call the metadata encoder,
        # else, create a tensor of zeros of shape [batch, time, d_model]
        if self.denoiser_config.use_metadata:
            cond_in_metadata = self.metadata_encoder(
                discrete_conditions=forecast_input[
                    "history_discrete_cond_input"
                ],
                continuous_conditions=forecast_input[
                    "history_continuous_cond_input"
                ],
            )
        else:
            cond_in_metadata = torch.zeros(
                x_enc.shape[0], x_enc.shape[1], self.decoder_config.d_model
            ).to(self.device)

        # If we're using timestamp, call the timestamp encoder,
        # else, create a tensor of zeros of shape [batch, time, d_model]
        if self.dataset_config.use_timestamp:
            history_time_feat = forecast_input["history_time_feat"]
            cond_in_timestamp = self.timestamp_encoder(history_time_feat)
        else:
            cond_in_timestamp = torch.zeros(
                x_enc.shape[0], x_enc.shape[1], self.decoder_config.d_model
            ).to(self.device)

        # [batch, time, d_model] -> [batch, channel, time, d_model]
        cond_in_metadata = cond_in_metadata.unsqueeze(1).repeat(
            1, self.dataset_config.num_channels, 1, 1
        )

        # [batch, channel, time, d_model] -> [batch * channel, time, d_model]
        cond_in_metadata = cond_in_metadata.reshape(
            -1, cond_in_metadata.shape[-2], cond_in_metadata.shape[-1]
        )

        # [batch, time, d_model] -> [batch, channel, time, d_model]
        cond_in_timestamp = cond_in_timestamp.unsqueeze(1).repeat(
            1, self.dataset_config.num_channels, 1, 1
        )

        # [batch, channel, time, d_model] -> [batch * channel, time, d_model]
        cond_in_timestamp = cond_in_timestamp.reshape(
            -1, cond_in_timestamp.shape[-2], cond_in_timestamp.shape[-1]
        )

        # Standard normalize across the time dimension
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        # do patching and embedding
        # [batch, time, channels] -> [batch, channels, time]
        x_enc = x_enc.permute(0, 2, 1)

        # [batch, channels, time] -> [batch, num_patches , d_model]?
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Set attention mode for encoder
        if (
            self.dataset_config.use_metadata
            and self.dataset_config.use_timestamp
        ):
            att_mode = "all"
        elif (
            self.dataset_config.use_metadata
            and not self.dataset_config.use_timestamp
        ):
            att_mode = "selfandmetadata"
        elif (
            not self.dataset_config.use_metadata
            and self.dataset_config.use_timestamp
        ):
            att_mode = "selfandtimestamp"
        else:
            att_mode = "self"

        # Encoder
        enc_out, attns = self.encoder(
            enc_out, cond_in_timestamp, cond_in_metadata, att_mode
        )

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )  # [batch, n_vars, num_patches, d_model]

        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        assert dec_out.shape[1] == self.pred_len

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]#
