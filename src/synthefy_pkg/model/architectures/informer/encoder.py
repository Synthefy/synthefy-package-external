from typing import Optional, Tuple

import torch

from synthefy_pkg.model.architectures.informer.attention import (
    AttentionLayer,
    FullAttention,
)


class EncoderLayer(torch.nn.Module):
    """
    Encoder layer used in the encoder.
    """

    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            attention: Attention layer, Object of AttentionLayer class.
            d_model: Dimensionality of the model.
            d_ff: Dimensionality of the feed forward layer.
            dropout: Dropout rate.
            activation: Activation function.
        """
        super(EncoderLayer, self).__init__()
        assert d_ff is not None, "d_ff must be specified"
        self.attention = attention
        self.conv1 = torch.nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1
        )  # does not change length
        self.conv2 = torch.nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1
        )  # does not change length
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = (
            torch.nn.functional.relu
            if activation == "relu"
            else torch.nn.functional.gelu
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attn_mask: Attention mask of shape (batch_size, seq_len).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
            Attention tensor of shape (batch_size, n_heads, seq_len, seq_len).
        """
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        # print(x.shape)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # print(y.shape, "-----")

        return self.norm2(x + y), attn


class Encoder(torch.nn.Module):
    """
    Encoder.
    """

    def __init__(
        self,
        attn_layers: list,
        conv_layers: list,
        norm_layer: torch.nn.LayerNorm,
    ) -> None:
        """
        Args:
            attn_layers: List of attention layers.
            conv_layers: List of convolutional layers.
            norm_layer: Layer normalization layer.
        """
        super(Encoder, self).__init__()
        self.attn_layers = torch.nn.ModuleList(attn_layers)
        self.conv_layers = (
            torch.nn.ModuleList(conv_layers)
            if conv_layers is not None
            else None
        )
        self.norm = norm_layer

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attn_mask: Attention mask of shape (batch_size, seq_len).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
            list of attention tensors of shape (batch_size, n_heads, seq_len, seq_len).
        """
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(
                self.attn_layers, self.conv_layers
            ):
                x, attn = attn_layer(
                    x, attn_mask=attn_mask
                )  # encoder operation
                x = conv_layer(x)  # compress along the time dimension
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
