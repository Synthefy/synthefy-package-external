import os
import pickle

import torch

from synthefy_pkg.model.architectures.informer.attention import (
    AttentionLayer,
    FullAttention,
)
from synthefy_pkg.model.architectures.informer.embedding import DataEmbedding
from synthefy_pkg.model.architectures.informer.encoder import (
    Encoder,
    EncoderLayer,
)
from synthefy_pkg.model.architectures.informer.informer_utils import (
    Conv1d_with_init,
    ConvLayer,
)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, cltsp_config, dataset_config, device, encode_ts=True):
        super(TransformerEncoder, self).__init__()
        self.model_config = cltsp_config
        self.dataset_config = dataset_config
        self.device = device
        self.num_input_features = (
            self.dataset_config.num_channels
            if encode_ts
            else self.model_config.d_model
        )
        self.horizon = self.dataset_config.required_time_series_length

        self.d_model = self.model_config.d_model
        self.d_keys = self.model_config.d_keys
        self.d_values = self.model_config.d_values
        self.n_heads = self.model_config.n_heads
        self.d_ff = self.model_config.d_ff

        self.dropout = self.model_config.dropout
        self.activation = self.model_config.activation

        # Position and token embedding
        self.enc_embedding = DataEmbedding(
            self.num_input_features, self.d_model, self.dropout
        )

        """
        First we build the Informer encoder stack
        """
        Attn = FullAttention
        # Encoder
        self.num_encoder_layers = self.model_config.num_encoder_layers
        conv_layers = []
        for layer_idx in range(self.num_encoder_layers - 1):
            if layer_idx % 3 == 0 and layer_idx != 0:
                conv_layers.append(ConvLayer(self.d_model, downsample=True))
            else:
                conv_layers.append(ConvLayer(self.d_model, downsample=False))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            mask_flag=False,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        d_model=self.d_model,
                        d_keys=self.d_keys,
                        d_values=self.d_values,
                        n_heads=self.n_heads,
                        mix=False,
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_encoder_layers)
            ],
            conv_layers,
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )

        # self.compress_encoded only changes the number of channels, not the length
        # the number of channels is reduced by a factor of 2 for each compression layer
        self.num_compression_layers = self.model_config.num_compression_layers
        encoder_compression_layers = []
        for power in range(self.num_compression_layers):
            encoder_compression_layers.append(
                Conv1d_with_init(
                    self.d_model // 2**power,
                    self.d_model // 2 ** (power + 1),
                    1,
                )
            )
            encoder_compression_layers.append(torch.nn.GELU())
        self.compress_encoded = torch.nn.Sequential(*encoder_compression_layers)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        enc_in = self.enc_embedding(x_enc)
        encoded, _ = self.encoder(enc_in, attn_mask=None)
        encoded = self.compress_encoded(encoded.transpose(1, 2)).transpose(1, 2)
        encoded = encoded.reshape(
            encoded.shape[0], encoded.shape[1] * encoded.shape[2]
        )
        return encoded


class DiscreteFCEncoder(torch.nn.Module):
    def __init__(
        self, embedding_dim, initial_projection_dim, projection_dim, dropout
    ):
        super(DiscreteFCEncoder, self).__init__()
        self.projection1 = torch.nn.Linear(
            embedding_dim, initial_projection_dim
        )
        self.projection2 = torch.nn.Linear(
            initial_projection_dim, projection_dim
        )
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection1(inp)
        projected = self.gelu(projected)
        projected = self.projection2(projected)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class FCEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(FCEncoder, self).__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection(inp)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection(inp)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ConditionEncoder(torch.nn.Module):
    def __init__(self, cltsp_config, dataset_config, device):
        super(ConditionEncoder, self).__init__()
        self.device = device
        self.cltsp_config = cltsp_config
        self.dataset_config = dataset_config

        num_discrete_conditions = self.dataset_config.num_discrete_conditions
        num_discrete_labels = self.dataset_config.num_discrete_labels
        num_continuous_labels = self.dataset_config.num_continuous_labels

        self.discrete_condition_encoder_exists = (
            True if num_discrete_labels > 0 else False
        )
        self.continuous_condition_encoder_exists = (
            True if num_continuous_labels > 0 else False
        )
        self.combined_condition_encoder_exists = (
            self.discrete_condition_encoder_exists
            and self.continuous_condition_encoder_exists
        )

        if self.discrete_condition_encoder_exists:
            self.discrete_condition_encoder = DiscreteFCEncoder(
                embedding_dim=num_discrete_conditions,
                initial_projection_dim=int(
                    num_discrete_labels
                    * self.cltsp_config.discrete_condition_embedding_dim
                ),
                projection_dim=self.dataset_config.latent_dim,
                dropout=0.1,
            )

        if self.continuous_condition_encoder_exists:
            self.continuous_condition_encoder = FCEncoder(
                embedding_dim=num_continuous_labels,
                projection_dim=self.dataset_config.latent_dim,
                dropout=0.1,
            )

        if self.combined_condition_encoder_exists:
            self.combined_condition_encoder = FCEncoder(
                embedding_dim=self.dataset_config.latent_dim * 2,
                projection_dim=self.dataset_config.latent_dim,
                dropout=0.1,
            )

        self.projection_head = ProjectionHead(
            embedding_dim=self.dataset_config.latent_dim,
            projection_dim=self.cltsp_config.d_model,
            dropout=0.1,
        )

        self.condition_transformer_encoder = TransformerEncoder(
            cltsp_config=cltsp_config,
            dataset_config=dataset_config,
            device=device,
            encode_ts=False,
        )

        factor = 3
        num_reductions = 0
        for eidx in range(
            self.condition_transformer_encoder.num_encoder_layers
        ):
            if eidx % factor == 0 and eidx != 0:
                num_reductions += 1

        encoded_size = int(
            (self.condition_transformer_encoder.horizon // 2**num_reductions)
            * (
                self.condition_transformer_encoder.d_model
                // 2**self.condition_transformer_encoder.num_compression_layers
            )
        )

        if encoded_size < self.dataset_config.latent_dim:
            raise ValueError(
                "The latent dimension is larger than the encoded size. Please reduce the latent dimension."
            )

        self.condition_embedding_projection = ProjectionHead(
            embedding_dim=encoded_size,
            projection_dim=self.dataset_config.latent_dim,
            dropout=0.1,
        )

        if (
            self.dataset_config.latent_dim
            > self.dataset_config.num_channels
            * self.dataset_config.required_time_series_length
        ):
            raise ValueError(
                "The latent dimension is larger than the time series. Please reduce the latent dimension."
            )

    def forward(self, discrete_conditions, continuous_conditions):
        if self.discrete_condition_encoder_exists:
            discrete_conditions = self.discrete_condition_encoder(
                discrete_conditions
            )
        if self.continuous_condition_encoder_exists:
            continuous_conditions = self.continuous_condition_encoder(
                continuous_conditions
            )
        if self.combined_condition_encoder_exists:
            combined_conditions = torch.cat(
                [discrete_conditions, continuous_conditions], dim=-1
            )
            combined_conditions = self.combined_condition_encoder(
                combined_conditions
            )
        else:
            combined_conditions = (
                discrete_conditions
                if self.discrete_condition_encoder_exists
                else continuous_conditions
            )
        combined_conditions = self.projection_head(combined_conditions)
        condition_enc = self.condition_transformer_encoder(combined_conditions)
        condition_enc = self.condition_embedding_projection(condition_enc)

        return condition_enc


class TimeSeriesEncoder(torch.nn.Module):
    def __init__(self, cltsp_config, dataset_config, device):
        super(TimeSeriesEncoder, self).__init__()
        self.cltsp_config = cltsp_config
        self.dataset_config = dataset_config
        self.device = device
        self.timeseries_transformer_encoder = TransformerEncoder(
            cltsp_config=self.cltsp_config,
            dataset_config=self.dataset_config,
            device=self.device,
            encode_ts=True,
        )

        factor = 3  # need to add this to the config - determines after how mamy encoder layers, we compress the input
        num_reductions = 0
        for eidx in range(
            self.timeseries_transformer_encoder.num_encoder_layers
        ):
            if eidx % factor == 0 and eidx != 0:
                num_reductions += 1

        encoded_size = int(
            (self.timeseries_transformer_encoder.horizon // 2**num_reductions)
            * (
                self.timeseries_transformer_encoder.d_model
                // 2**self.timeseries_transformer_encoder.num_compression_layers
            )
        )

        if encoded_size < self.dataset_config.latent_dim:
            raise ValueError(
                "The latent dimension is larger than the encoded size. Please reduce the latent dimension."
            )

        self.timeseries_embedding_projection = ProjectionHead(
            embedding_dim=encoded_size,
            projection_dim=self.dataset_config.latent_dim,
            dropout=0.1,
        )

        if (
            self.dataset_config.latent_dim
            > self.dataset_config.num_channels
            * self.dataset_config.required_time_series_length
        ):
            raise ValueError(
                "The latent dimension is larger than the time series. Please reduce the latent dimension."
            )

    def forward(self, x):
        x = self.timeseries_transformer_encoder(x)
        x = self.timeseries_embedding_projection(x)
        return x
