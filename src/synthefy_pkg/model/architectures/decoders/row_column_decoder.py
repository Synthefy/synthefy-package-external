import torch
import torch.nn as nn

from synthefy_pkg.model.architectures.tabicl.multilayer_tabicl import (
    RowInteraction,
)


class RowColumnDecoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        row_num_blocks: int,
        row_nhead: int,
        row_num_cls: int,
        col_num_blocks: int,
        col_nhead: int,
        col_num_cls: int,
        rope_base: float,
        ff_factor: float,
        dropout: float,
        activation: str,
        norm_first: bool,
        weight_range: tuple[float, float],
        final_model: str = "linear",
        output_dim: int = 1,
        hidden_dim: int = 128,
    ):
        super(RowColumnDecoder, self).__init__()
        self.row_interaction = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            num_cls=row_num_cls,
            rope_base=rope_base,
            dim_feedforward=int(embed_dim * ff_factor),
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            weight_range=weight_range,
            aggregate_operation="select_last",
        )
        self.column_interaction = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_cls=col_num_cls,
            rope_base=rope_base,
            dim_feedforward=int(embed_dim * ff_factor),
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            weight_range=weight_range,
            aggregate_operation="select_last",
        )
        self.reserve_row_tokens = row_num_cls
        self.reserve_col_tokens = col_num_cls
        self.final_model = final_model
        if self.final_model == "linear":
            self.final_layer = torch.nn.Linear(
                (
                    embed_dim * self.reserve_row_tokens
                    + embed_dim * self.reserve_col_tokens
                ),
                output_dim,
            )
        elif self.final_model == "mlp":
            self.final_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    (
                        embed_dim * self.reserve_row_tokens
                        + embed_dim * self.reserve_col_tokens
                    ),
                    hidden_dim,
                ),
                torch.nn.ReLU() if activation == "relu" else torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(
                f"Invalid final model: {self.final_model}, only linear and mlp are supported"
            )

    def forward(self, embeddings: torch.Tensor, x: torch.Tensor):
        assert self.reserve_row_tokens > 0 and self.reserve_col_tokens > 0, (
            "uses reserve tokens to collapse information"
        )
        # Pad with -100.0 to mark inputs that should be skipped in SkippableLinear and SetTransformer
        row_embeddings = nn.functional.pad(
            embeddings, (0, 0, self.reserve_row_tokens, 0), value=-100.0
        )
        col_embeddings = nn.functional.pad(
            embeddings, (0, self.reserve_col_tokens, 0, 0), value=-100.0
        ).transpose(2, 1)

        row_embeddings = self.row_interaction(row_embeddings)  # b T d
        col_embeddings = self.column_interaction(col_embeddings)  # b nc d
        # Reshape for broadcasting
        row_embeddings_expanded = row_embeddings.unsqueeze(2).expand(
            -1, -1, col_embeddings.size(1), -1
        )  # [b, T, nc, d]
        col_embeddings_expanded = col_embeddings.unsqueeze(1).expand(
            -1, row_embeddings.size(1), -1, -1
        )  # [b, T, nc, d]

        # Concatenate along the middle dimension
        result = torch.cat(
            [row_embeddings_expanded, col_embeddings_expanded], dim=-1
        )  # [b, T, nc, 2d]

        result = self.final_layer(result)  # project to b, T, nc, output_dim
        return result
