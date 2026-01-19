from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from synthefy_pkg.model.architectures.tabicl.encoders import Encoder
from synthefy_pkg.model.architectures.tabicl.inference import InferenceManager
from synthefy_pkg.model.architectures.tabicl.inference_config import MgrConfig
from synthefy_pkg.model.architectures.tabicl.layers import (
    ClassNode,
    OneHotAndLinear,
)

COMPILE = False


class ICRLearning(nn.Module):
    """Dataset-wise in-context learning with automatic hierarchical classification support.

    This module implements in-context learning that:
    1. Takes row representations and training labels as input
    2. Conditions the model on training examples
    3. Makes predictions for test examples based on learned patterns
    4. Automatically handles both small and large label spaces

    Parameters
    ----------
    output_dim : int
        Number of output channels

    d_model : int
        Model dimension

    num_blocks : int
        Number of blocks used in the ICL encoder

    nhead : int
        Number of attention heads of the ICL encoder

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        output_dim: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        self.y_encoder = nn.Conv1d(
            in_channels=1, out_channels=d_model, kernel_size=1, bias=True
        )  # 1D conv to update the y_encoder to the dimension of the row representations
        # TODO: decoder here is deprecated, to upgrade regression will need to
        # update to new decoder signature
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.output_dim),
        )

        self.inference_mgr = InferenceManager(
            enc_name="tf_icl", out_dim=self.output_dim
        )

    def _icl_predictions(
        self, R: Tensor, y_train: Tensor, target_mask: Optional[Tensor] = None
    ) -> Tensor:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data
        """
        train_size = y_train.shape[1]
        y_enc = self.y_encoder(y_train.float().unsqueeze(1)).transpose(1, 2)
        R[:, :train_size] = R[:, :train_size] + y_enc
        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)  # (B, T, output_dim)

        return out

    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate predictions for standard classification with up to `max_classes` classes.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data

        return_logits : bool, default=False
            If True, return logits instead of probabilities (not used in regression)

        softmax_temperature : float, default=0.9
            Temperature for the softmax function (not used in regression)

        auto_batch : bool, default=True
            Whether to use InferenceManager to automatically split inputs into smaller batches
        """

        train_size = y_train.shape[1]
        out = self.inference_mgr(
            self._icl_predictions,
            inputs=OrderedDict(
                [("R", R), ("y_train", y_train), ("target_mask", target_mask)]
            ),
            auto_batch=auto_batch,
        )
        if target_mask is None:
            out = out[
                :,
                train_size:,
            ]

        return out

    def _inference_forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: Optional[MgrConfig] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """In-context learning based on learned row representations for inference.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data

        return_logits : bool, default=True
            If True, return logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager

        Returns
        -------
        Tensor
            Raw logits or probabilities for test samples of shape (B, T-train_size, num_classes)
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=R.device,
                use_amp=True,
                verbose=False,
            )
        assert mgr_config is not None
        self.inference_mgr.configure(**mgr_config)  # type: ignore

        # Standard regression
        out = self._predict_standard(
            R,
            y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            target_mask=target_mask,
        )
        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: Optional[MgrConfig] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data

        return_logits : bool, default=True
            If True, return logits instead of probabilities. Used only in inference mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in inference mode.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        Returns
        -------
        Tensor
            For training mode:
              Raw logits of shape (B, T-train_size, max_classes), which will be further handled by the training code.

            For inference mode:
              Raw logits or probabilities for test samples of shape (B, T-train_size, num_classes).
        """

        if self.training:
            train_size = y_train.shape[1]
            out = self._icl_predictions(R, y_train, target_mask)
            if target_mask is None:
                out = out[:, train_size:]
        else:
            out = self._inference_forward(
                R,
                y_train,
                return_logits,
                softmax_temperature,
                mgr_config,
                target_mask,
            )

        return out
