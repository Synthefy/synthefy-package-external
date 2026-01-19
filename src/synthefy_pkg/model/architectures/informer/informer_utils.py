from typing import Dict, List, Tuple

import torch


class ConvLayer(torch.nn.Module):
    """
    Convolutional layer used in the encoder and decoder.
    """

    def __init__(self, c_in: int, downsample: bool = True) -> None:
        """
        Args:
            c_in: Number of input channels.
        The layer reduces the time dimension by a factor of 2.
        The number of channels is kept constant.
        """
        super(ConvLayer, self).__init__()
        self.downConv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = torch.nn.BatchNorm1d(c_in)
        self.activation = torch.nn.LeakyReLU(0.1)
        self.downsample = downsample
        if self.downsample:
            self.maxPool = torch.nn.MaxPool1d(
                kernel_size=3, stride=2, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Output tensor of shape (batch_size, seq_len / 2, d_model).
        """
        x = torch.einsum("bld->bdl", x)  # (batch_size, d_model, seq_len)
        x = self.downConv(x)  # (batch_size, d_model, seq_len)
        x = self.norm(x)
        x = self.activation(x)
        if self.downsample:
            x = self.maxPool(x)  # (batch_size, d_model, seq_len / 2)
        x = torch.einsum("bdl->bld", x)  # (batch_size, seq_len / 2, d_model)
        return x


def Conv1d_with_init(
    in_channels: int, out_channels: int, kernel_size: int
) -> torch.nn.Conv1d:
    """
    Convolutional layer with kaiming normal initialization.
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size.
    Returns:
        Convolutional layer.
    """
    layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
    torch.nn.init.kaiming_normal_(layer.weight)
    return layer
