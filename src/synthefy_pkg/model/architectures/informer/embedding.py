import math
from typing import Dict, List, Tuple

import torch


class PositionalEmbedding(torch.nn.Module):
    """
    Positional embedding for time series data.
    The class creates positional embeddings for time series data.
    The size of the positional embedding is d_model.
    This is a static positional embedding, i.e. the embeddings are not learned.
    This implementation is based on the paper "Attention is all you need" (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Args:
            d_model: size of the positional embedding
            max_len: maximum length of the input sequence
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  # [max_len, d_model]
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(
            0
        )  # [1, max_len, d_model] to repeat over batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            positional encoding of shape [batch_size, seq_len, d_model]
        """
        return self.pe[:, : x.size(1)]  # applied to time


class TokenEmbedding(torch.nn.Module):
    """
    Token embedding for time series data.
    The class creates token embeddings for time series data.
    The size of the token embedding is d_model.
    This is not a static positional embedding, i.e. the embeddings are learned.
    This embedding can be better understood as capturing the local structure of the time series.
    """

    def __init__(self, c_in: int, d_model: int, kernel_size: int = 3) -> None:
        """
        Args:
            c_in: number of input channels
            d_model: size of the token embedding
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if kernel_size > 1 else 0
        self.tokenConv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape [batch_size, seq_len, c_in]
        Returns:
            token embedding of shape [batch_size, seq_len, d_model]
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(torch.nn.Module):
    """
    Data embedding for time series data.
    The class creates data embeddings for time series data.
    The size of the data embedding is d_model.
    The data embedding is the sum of the token embedding and the positional embedding.
    """

    def __init__(
        self, c_in: int, d_model: int, dropout: int, kernel_size=3
    ) -> None:
        """
        Args:
            c_in: number of input channels
            d_model: size of the data embedding
            dropout: dropout rate
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(
            c_in=c_in, d_model=d_model, kernel_size=kernel_size
        )  # token embedding
        self.position_embedding = PositionalEmbedding(
            d_model=d_model
        )  # positional embedding
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape [batch_size, seq_len, c_in]
        Returns:
            data embedding of shape [batch_size, seq_len, d_model]
        """
        # print(self.value_embedding(x).shape)
        # print(self.position_embedding(x).shape)
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
