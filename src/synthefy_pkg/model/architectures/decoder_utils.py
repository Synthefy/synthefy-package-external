import math

import numpy as np
import torch

COMPILE = True


# Note: This is different than the ConvLayer in diffusion_transformer.py
class ConvLayer(torch.nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = torch.nn.BatchNorm1d(c_in)
        self.activation = torch.nn.ELU()
        self.maxPool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(torch.nn.Module):
    def __init__(
        self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = torch.nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = (
            torch.nn.functional.relu
            if activation == "relu"
            else torch.nn.functional.gelu
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x, attn_mask=attn_mask, tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(torch.nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = torch.nn.ModuleList(attn_layers)
        self.conv_layers = (
            torch.nn.ModuleList(conv_layers)
            if conv_layers is not None
            else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


def get_torch_trans_decoder(heads=8, layers=1, channels=64):
    decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=channels,
        activation="gelu",
        batch_first=True,
    )
    return torch.nn.TransformerDecoder(decoder_layer, num_layers=layers)


class EncoderAvecCrossAttention(torch.nn.Module):
    def __init__(
        self,
        self_attn_layers,
        cross_attn_layers,
        conv_layers=None,
        norm_layer=None,
    ):
        super(EncoderAvecCrossAttention, self).__init__()
        self.self_attn_layers = torch.nn.ModuleList(self_attn_layers)
        self.cross_attn_layers = torch.nn.ModuleList(cross_attn_layers)
        self.conv_layers = (
            torch.nn.ModuleList(conv_layers)
            if conv_layers is not None
            else None
        )
        self.norm = norm_layer

    def forward(self, x, y, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (self_attn_layer, cross_attn_layer, conv_layer) in enumerate(
                zip(
                    self.self_attn_layers,
                    self.cross_attn_layers,
                    self.conv_layers,
                )
            ):
                delta = delta if i == 0 else None
                x, attn = self_attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
                x = cross_attn_layer(tgt=x, memory=y)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for i, (self_attn_layer, cross_attn_layer) in enumerate(
                zip(self.self_attn_layers, self.cross_attn_layers)
            ):
                x, attn = self_attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
                x = cross_attn_layer(tgt=x, memory=y)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# TODO: Modernize this...
class EncoderAvecCrossAttentionTriple(torch.nn.Module):
    def __init__(
        self,
        self_attn_layers,
        cross_attn_layers1,
        cross_attn_layers2,
        conv_layers=None,
        norm_layer=None,
    ):
        super(EncoderAvecCrossAttentionTriple, self).__init__()
        self.self_attn_layers = torch.nn.ModuleList(self_attn_layers)
        self.cross_attn_layers1 = torch.nn.ModuleList(
            cross_attn_layers1
        )  # for timestamps
        self.cross_attn_layers2 = torch.nn.ModuleList(
            cross_attn_layers2
        )  # for metadata
        self.conv_layers = (
            torch.nn.ModuleList(conv_layers)
            if conv_layers is not None
            else None
        )
        self.norm = norm_layer

    def forward(
        self, x, y1, y2, att_mode, attn_mask=None, tau=None, delta=None
    ):
        # x [B, L, D]
        # y1 timestamps
        # y2 metadata
        attns = []
        if self.conv_layers is not None:
            for i, (
                self_attn_layer,
                cross_attn_layer1,
                cross_attn_layer2,
                conv_layer,
            ) in enumerate(
                zip(
                    self.self_attn_layers,
                    self.cross_attn_layers1,
                    self.cross_attn_layers2,
                    self.conv_layers,
                )
            ):
                delta = delta if i == 0 else None
                x1, attn = self_attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
                x2 = cross_attn_layer1(tgt=x, memory=y1)
                x3 = cross_attn_layer2(tgt=x, memory=y2)
                if att_mode == "selfandtimestamp":
                    x = x1 + x2
                elif att_mode == "self":
                    x = x1
                elif att_mode == "timestamps":
                    x = x2
                elif att_mode == "onlymetadata":
                    x = x3
                elif att_mode == "timestampsandmetadata":
                    x = x2 + x3
                elif att_mode == "selfandmetadata":
                    x = x1 + x3
                elif att_mode == "all":
                    x = x1 + x2 + x3
                else:
                    raise ValueError("att_mode not recognized")
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            # print(x.shape)
            # print(y1.shape)
            # print(y2.shape)
            # exit()
            for i, (
                self_attn_layer,
                cross_attn_layer1,
                cross_attn_layer2,
            ) in enumerate(
                zip(
                    self.self_attn_layers,
                    self.cross_attn_layers1,
                    self.cross_attn_layers2,
                )
            ):
                x1, attn = self_attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
                x2 = cross_attn_layer1(tgt=x, memory=y1)
                x3 = cross_attn_layer2(tgt=x, memory=y2)
                if att_mode == "selfandtimestamp":
                    x = x1 + x2
                elif att_mode == "self":
                    x = x1
                elif att_mode == "timestamps":
                    x = x2
                elif att_mode == "onlymetadata":
                    x = x3
                elif att_mode == "timestampsandmetadata":
                    x = x2 + x3
                elif att_mode == "selfandmetadata":
                    x = x1 + x3
                elif att_mode == "all":
                    x = x1 + x2 + x3
                else:
                    raise ValueError("att_mode not recognized")
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = torch.nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1
        )
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
        self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None
    ):
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[
                0
            ]
        )
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(torch.nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None
    ):
        for layer in self.layers:
            x = layer(
                x,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta,
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(torch.nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = torch.nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(torch.nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = torch.nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = torch.nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = torch.nn.Linear(d_model, d_values * n_heads)
        self.out_projection = torch.nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PatchEmbedding(torch.nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = torch.nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = torch.nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # TODO Document these shapes explicitly.
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_inverted(torch.nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = torch.nn.Linear(c_in, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)
