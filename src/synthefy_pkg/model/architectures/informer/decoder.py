from typing import Optional

import torch

from synthefy_pkg.model.architectures.informer.attention import (
    AttentionLayer,
    FullAttention,
)


class DecoderLayer(torch.nn.Module):
    """
    Decoder layer used in the decoder.
    """

    def __init__(
        self,
        self_attention: AttentionLayer,
        cross_attention: AttentionLayer,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            self_attention: Self attention layer, Object of AttentionLayer class.
            cross_attention: Cross attention layer, Object of AttentionLayer class.
            d_model: Dimensionality of the model.
            d_ff: Dimensionality of the feed forward layer.
            dropout: Dropout rate.
            activation: Activation function.
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = torch.nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1
        )  # acts only on the features and not the time dimension
        self.conv2 = torch.nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1
        )  # acts only on the features and not the time dimension
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = (
            torch.nn.functional.relu
            if activation == "relu"
            else torch.nn.functional.gelu
        )

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, inp_seq_len, d_model).
            cross: Cross tensor of shape (batch_size, enc_seq_len, d_model).
            x_mask: Attention mask of shape (batch_size, seq_len).
            cross_mask: Attention mask of shape (batch_size, seq_len).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        Note that the input sequence length and the encoder sequence length need not be the same.
        """
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask)[0]
        )  # self attention
        x = self.norm1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(torch.nn.Module):
    def __init__(self, layers: list, norm_layer: torch.nn.LayerNorm) -> None:
        """
        Args:
            layers: List of decoder layers.
            norm_layer: Layer normalization layer.
        """
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask
            )  # transformer decoder

        if self.norm is not None:
            x = self.norm(x)

        return x
