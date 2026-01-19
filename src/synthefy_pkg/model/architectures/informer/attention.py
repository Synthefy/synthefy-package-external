import math
from typing import Optional, Tuple

import numpy as np
import torch


class TriangularCausalMask:
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    """

    def __init__(self, B: int, L: int, device: str = "cpu") -> None:
        """
        Args:
            B: Batch size.
            L: Sequence length.
            device: Device to store the mask.
        """
        # the mask is the same for all heads, so we only need to compute it once.
        mask_shape = [B, 1, L, L]  # batch_size, n_heads, seq_len, seq_len
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(torch.nn.Module):
    """
    Multi-headed attention. Same as the attention described in the transformer paper.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        """
        Args:
            mask_flag: Flag to indicate whether to use the mask.
            attention_dropout: Dropout rate.
            output_attention: Flag to indicate whether to output attention weights.
        """
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = torch.nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            queries: Queries tensor of shape (batch_size, query_seq_len, n_heads, d_keys).
            keys: Keys tensor of shape (batch_size, key_seq_len, n_heads, d_keys).
            values: Values tensor of shape (batch_size, seq_len, d_model).
            attn_mask: Attention mask of shape (batch_size, 1, query_seq_len, query_seq_len).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, L, H, E = queries.shape  # batch_size, seq_len, n_heads, d_keys
        _, S, _, D = values.shape  # batch_size, seq_len, n_heads, d_values
        scale = 1.0 / math.sqrt(E)  # 1/sqrt(d_keys) to reduce the variance.

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # QK^T

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(
                    B, L, device=str(queries.device)
                )  # only for transformer decoder

            scores.masked_fill_(
                attn_mask.mask, -np.inf
            )  # fills the upper triangle with -inf

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # across the dimension s, the probabilities sum to 1.

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(torch.nn.Module):
    """
    Attention layer used in the encoder and decoder.
    """

    def __init__(
        self,
        attention: FullAttention,
        d_model: int,
        d_keys: int,
        d_values: int,
        n_heads: int,
        mix: bool = False,
    ) -> None:
        """
        Args:
            attention: Attention layer.
            d_model: Dimensionality of the model.
            d_keys: Dimensionality of the keys.
            d_values: Dimensionality of the values.
            n_heads: Number of attention heads.
            mix: Flag to indicate whether to mix the heads.
        """
        super(AttentionLayer, self).__init__()
        assert d_model % n_heads == 0, "d_model % n_heads == 0 is not true."

        self.n_heads = n_heads
        self.inner_attention = attention
        self.query_projection = torch.nn.Linear(d_model, d_keys * self.n_heads)
        self.key_projection = torch.nn.Linear(d_model, d_keys * self.n_heads)
        self.value_projection = torch.nn.Linear(
            d_model, d_values * self.n_heads
        )
        self.out_projection = torch.nn.Linear(d_values * n_heads, d_model)

        self.mix = mix
        # need to understand the need for mix and how it is used.

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            queries: Queries tensor of shape (batch_size, seq_len, d_model).
            keys: Keys tensor of shape (batch_size, seq_len, d_model).
            values: Values tensor of shape (batch_size, seq_len, d_model).
            attn_mask: Attention mask of shape (batch_size, seq_len).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        As a high level description, the attention layer takes in queries, keys and values which are the same tensor in Informer.
        It then projects them to d_keys, d_keys and d_values respectively, with multiple attention heads.
        Then the attention is computed and the output is projected back to d_model.
        """
        B, L, _ = queries.shape  # batch_size, seq_len, d_model
        (
            _,
            S,
            _,
        ) = keys.shape  # batch_size, seq_len, d_model, it is S and not L because it could be a cross attention.
        # for instance, our implementation has L = 96 and S = 12/24/48/96 depending on the number of encoders.

        assert keys.shape == values.shape, (
            "Keys and values must have the same shape."
        )

        queries = self.query_projection(queries).view(
            B, L, self.n_heads, -1
        )  # batch_size, seq_len, n_heads, d_keys
        keys = self.key_projection(keys).view(
            B, S, self.n_heads, -1
        )  # batch_size, seq_len, n_heads, d_keys
        values = self.value_projection(values).view(
            B, S, self.n_heads, -1
        )  # batch_size, seq_len, n_heads, d_values

        assert queries.shape[-1] == keys.shape[-1], (
            "The last dimension of Q, K must be equal. Only then QK^T is possible."
        )

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)  # batch_size, seq_len, d_model

        return self.out_projection(out), attn
