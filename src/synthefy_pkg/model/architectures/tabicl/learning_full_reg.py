from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from synthefy_pkg.configs.tabicl_config import TokenDecoderConfig
from synthefy_pkg.model.architectures.decoders.decoders import init_decoders
from synthefy_pkg.model.architectures.tabicl.inference import InferenceManager
from synthefy_pkg.model.architectures.tabicl.inference_config import MgrConfig

COMPILE = False


class ICRFullLearning(nn.Module):
    """Takes in a full table of embeddings and makes predictions for each value in the target mask

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
        decoder_config: TokenDecoderConfig,
        norm_first: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.norm_first = norm_first

        decoder_config.model_dim = d_model
        decoder_config.output_dim = output_dim

        self.decoder = init_decoders(decoder_config)

        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        self.inference_mgr = InferenceManager(
            enc_name="tf_icl", out_dim=self.output_dim
        )

    def _icl_predictions(
        self, X: Tensor, R: Tensor, target_mask: Tensor
    ) -> Tensor:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of correlates
             - D is the dimension of row representations

        target_mask : Tensor of shape (B, T, H)
            Mask of shape (B, T, H) where 1 indicates a target value and 0 indicates a non-target value
        """
        if self.norm_first:
            R = self.ln(R)
        R = self.decoder(R, X)

        return R[target_mask.transpose(1, 2)]

    def _predict_standard(
        self,
        X: Tensor,
        R: Tensor,
        y_train: Tensor,
        target_mask: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
    ) -> Tensor:
        """Generate predictions for standard classification with up to `max_classes` classes.

        Parameters
        ----------
        X : Tensor
            Input features of shape (..., T, 1) where:
             - ... represents arbitrary batch dimensions
             - T is the number of samples (rows)

        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of correlates
             - D is the dimension of row representations

        y_train : Tensor of shape (B, T, H)
            Full targets, where T is the number of samples (rows) and H is the number of correlates

        return_logits : bool, default=False
            If True, return logits instead of probabilities (not used in regression)

        softmax_temperature : float, default=0.9
            Temperature for the softmax function (not used in regression)

        auto_batch : bool, default=True
            Whether to use InferenceManager to automatically split inputs into smaller batches
        """
        with torch.no_grad():
            out = self._icl_predictions(X, R, target_mask)
        # out = self.inference_mgr(
        #     self._icl_predictions,
        #     inputs=OrderedDict(
        #         [("R", R), ("X", X), ("target_mask", target_mask)]
        #     ),
        #     auto_batch=auto_batch,
        # )

        return out

    def _inference_forward(
        self,
        X: Tensor,
        R: Tensor,
        y_train: Tensor,
        target_mask: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: Optional[MgrConfig] = None,
    ) -> Tensor:
        """In-context learning based on learned row representations for inference.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, H, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of correlates
             - D is the dimension of row representations

        y_train : Tensor of shape (B, T, H)
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
            X,
            R,
            y_train=y_train,
            target_mask=target_mask,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
        )
        return out

    def forward(
        self,
        X: Tensor,
        R: Tensor,
        y_train: Tensor,
        target_mask: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: Optional[MgrConfig] = None,
    ) -> Tensor:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, H, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of correlates
             - D is the dimension of row representations

        y_train : Tensor of shape (B, T, H)
            Full targets, where T is the number of samples (rows) and H is the number of correlates

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
            out = self._icl_predictions(X, R, target_mask)
        else:
            out = self._inference_forward(
                X,
                R,
                y_train,
                target_mask,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=mgr_config,
            )

        return out
