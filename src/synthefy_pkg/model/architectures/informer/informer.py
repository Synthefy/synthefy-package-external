"""
This is a slightly modified PyTorch implementation of the paper "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting".
The paper can be found at https://arxiv.org/abs/2012.07436.
We have included mean and variance projection layers in the encoder to train a variational autoencoder.
"""

from typing import Dict, List, Tuple

import torch

from synthefy_pkg.model.architectures.informer.attention import (
    AttentionLayer,
    FullAttention,
)
from synthefy_pkg.model.architectures.informer.decoder import (
    Decoder,
    DecoderLayer,
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


class InformerAutoEncoder(torch.nn.Module):
    """
    Informer model.
    """

    def __init__(
        self,
        autoencoder_config,
        dataset_config,
        device,
    ):
        super(InformerAutoEncoder, self).__init__()
        self.model_config = autoencoder_config
        self.dataset_config = dataset_config
        self.device = device

        self.num_input_features = self.dataset_config.num_input_features
        self.horizon = self.dataset_config.time_series_length

        self.d_model = self.model_config.d_model
        self.d_keys = self.model_config.d_keys
        self.d_values = self.model_config.d_values
        self.n_heads = self.model_config.n_heads
        self.dropout = self.model_config.dropout
        self.d_ff = self.model_config.d_ff
        self.activation = self.model_config.activation

        # Position and token embedding
        self.enc_embedding = DataEmbedding(
            self.num_input_features, self.d_model, self.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.num_input_features, self.d_model, self.dropout
        )

        # encoder
        self.num_encoder_layers = self.model_config.num_encoder_layers
        if self.model_config.use_conv:
            conv_layers = [
                ConvLayer(self.d_model, downsample=False)
                for _ in range(self.num_encoder_layers - 1)
            ]
        else:
            conv_layers = []
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
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

        self.num_compression_layers = self.model_config.num_compression_layers
        compression_layers = []
        for power in range(self.num_compression_layers):
            compression_layers.append(
                Conv1d_with_init(
                    self.d_model // 2**power,
                    self.d_model // 2 ** (power + 1),
                    1,
                )
            )
            compression_layers.append(torch.nn.GELU())
        self.compression_layers = torch.nn.Sequential(*compression_layers)

        decompression_layers = []
        for power in reversed(range(self.num_compression_layers)):
            decompression_layers.append(
                Conv1d_with_init(
                    self.d_model // 2 ** (power + 1),
                    self.d_model // 2**power,
                    1,
                )
            )
            decompression_layers.append(torch.nn.GELU())
        self.decompression_layers = torch.nn.Sequential(*decompression_layers)

        # encoder
        self.num_decoder_layers = self.model_config.num_decoder_layers
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        d_model=self.d_model,
                        d_keys=self.d_keys,
                        d_values=self.d_values,
                        n_heads=self.n_heads,
                        mix=False,
                    ),
                    AttentionLayer(
                        FullAttention(
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
                for _ in range(self.num_decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )

        self.projection = torch.nn.Linear(
            self.d_model, self.num_input_features, bias=True
        )

    def encode(self, x_enc: torch.Tensor):
        enc_in = self.enc_embedding(x_enc)
        encoded_out, attns = self.encoder(enc_in, attn_mask=None)
        encoded = self.compression_layers(
            encoded_out.transpose(1, 2)
        ).transpose(1, 2)
        return encoded

    def forward(self, x_enc: torch.Tensor):
        enc_in = self.enc_embedding(x_enc)
        encoded, attns = self.encoder(enc_in, attn_mask=None)
        encoded = self.compression_layers(encoded.transpose(1, 2)).transpose(
            1, 2
        )
        # print(encoded.shape)
        cross = self.decompression_layers(encoded.transpose(1, 2)).transpose(
            1, 2
        )
        x_dec = torch.zeros_like(x_enc).cuda()
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, cross, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)
        return dec_out

    def decode(self, encoded: torch.Tensor):
        cross = self.decompression_layers(encoded.transpose(1, 2)).transpose(
            1, 2
        )
        x_dec = torch.zeros(
            (cross.shape[0], self.horizon, self.num_input_features)
        )
        x_dec = x_dec.to(self.device)
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, cross, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)
        return dec_out

    def prepare_input(self, batch):
        timeseries = batch["timeseries"]
        timeseries = timeseries.float().cuda()
        if len(timeseries.shape) == 2:
            timeseries = timeseries.unsqueeze(-1)
        assert timeseries.shape[-1] == self.num_input_features, (
            "The number of input features is not correct"
        )
        return timeseries
