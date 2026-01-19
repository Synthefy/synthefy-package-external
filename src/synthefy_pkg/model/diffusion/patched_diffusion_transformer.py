"""
Patched Diffusion Transformer
1. Patched input to the denoiser
2. Patching the input of the metadata encoder
"""

import math
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from synthefy_pkg.configs.denoiser_configs import DenoiserConfig
from synthefy_pkg.configs.execution_configurations import Configuration

# TODO: Make model hyper parameters configurable.
# e.g. dropout=0.1 should be configurable.

COMPILE = True


class DiscreteFCEncoder(torch.nn.Module):
    def __init__(
        self, embedding_dim, initial_projection_dim, projection_dim, dropout
    ):
        super(DiscreteFCEncoder, self).__init__()
        self.projection1 = torch.nn.Linear(
            embedding_dim, initial_projection_dim
        )
        self.projection2 = torch.nn.Linear(
            initial_projection_dim, projection_dim
        )
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection1(inp)
        projected = self.gelu(projected)
        projected = self.projection2(projected)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class FCEncoder(torch.nn.Module):
    def __init__(
        self, embedding_dim, projection_dim, dropout, use_layer_norm=True
    ):
        super(FCEncoder, self).__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        if use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(projection_dim)
        else:
            self.layer_norm = None

        # initialize all the weights to zero in the output projection layer
        # self.fc.weight.data.zero_()
        # self.fc.bias.data.zero_()
        # self.projection.weight.data.zero_()
        # self.projection.bias.data.zero_()

        # initialize the weights to kaiming normal
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.projection.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.projection.bias)

    def forward(self, inp):
        projected = self.projection(inp)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class ConvLayer(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super(ConvLayer, self).__init__()
        self.downConv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = torch.nn.BatchNorm1d(c_out)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, c_in, horizon).
        Returns:
            Output tensor of shape (batch_size, c_out, horizon).
        """
        # (batch_size, d_model, seq_len)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MetaDataEncoder(torch.nn.Module):
    def __init__(self, dataset_config, denoiser_config, device):
        super(MetaDataEncoder, self).__init__()
        self.device = device
        self.dataset_config = dataset_config
        self.denoiser_config: Union[DenoiserConfig, DictConfig] = (
            denoiser_config
        )

        num_discrete_conditions = self.dataset_config.num_discrete_conditions
        # Note: this is only used for backward compatibility for timeseries synthesis repo.
        num_discrete_labels = self.dataset_config.num_discrete_labels
        num_continuous_labels = self.dataset_config.num_continuous_labels

        self.discrete_condition_encoder_exists = (
            True if num_discrete_conditions > 0 else False
        )
        self.continuous_condition_encoder_exists = (
            True if num_continuous_labels > 0 else False
        )
        self.combined_condition_encoder_exists = (
            self.discrete_condition_encoder_exists
            and self.continuous_condition_encoder_exists
        )

        # This is where the metadata_encoder_config is used.
        # We don't have a separate class for MetaDataEncoder().
        if self.discrete_condition_encoder_exists:
            self.discrete_condition_encoder = DiscreteFCEncoder(
                embedding_dim=num_discrete_conditions,
                initial_projection_dim=int(
                    num_discrete_labels
                    * self.dataset_config.discrete_condition_embedding_dim
                ),
                projection_dim=self.denoiser_config.metadata_encoder_config.channels,
                dropout=0.1,
            )

        if self.continuous_condition_encoder_exists:
            self.continuous_condition_encoder = FCEncoder(
                embedding_dim=num_continuous_labels,
                projection_dim=self.denoiser_config.metadata_encoder_config.channels,
                dropout=0.1,
            )

        if self.combined_condition_encoder_exists:
            self.combined_condition_encoder = FCEncoder(
                embedding_dim=self.denoiser_config.metadata_encoder_config.channels
                * 2,
                projection_dim=self.denoiser_config.metadata_encoder_config.channels,
                dropout=0.1,
            )

        self.condition_transformer_encoder = get_torch_transformer(
            heads=self.denoiser_config.metadata_encoder_config.n_heads,
            layers=self.denoiser_config.metadata_encoder_config.num_encoder_layers,
            channels=self.denoiser_config.metadata_encoder_config.channels,
        )

        self.patching_layer = FCEncoder(
            embedding_dim=self.denoiser_config.patch_len
            * self.denoiser_config.metadata_encoder_config.channels,
            projection_dim=self.denoiser_config.metadata_encoder_config.channels,
            dropout=0.1,
        )

    def position_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, discrete_conditions, continuous_conditions):
        # Note: This will break on data that doesn't have discrete_conditions.
        # Figure out a more robust way to get these values.
        # The inputs should have length (L) of time_series_length - forecast_length

        B = discrete_conditions.shape[0]
        # L = (
        #     self.dataset_config.time_series_length
        #     - self.dataset_config.forecast_length
        # )

        if self.discrete_condition_encoder_exists:
            discrete_conditions = self.discrete_condition_encoder(
                discrete_conditions
            )
        if self.continuous_condition_encoder_exists:
            continuous_conditions = self.continuous_condition_encoder(
                continuous_conditions
            )
        if self.combined_condition_encoder_exists:
            combined_conditions = torch.cat(
                [discrete_conditions, continuous_conditions], dim=-1
            )
            combined_conditions = self.combined_condition_encoder(
                combined_conditions
            )
        else:
            combined_conditions = (
                discrete_conditions
                if self.discrete_condition_encoder_exists
                else continuous_conditions
            )
        combined_conditions = combined_conditions.reshape(
            B,
            -1,
            self.denoiser_config.patch_len,  # type: ignore
            self.denoiser_config.metadata_encoder_config.channels,
        )
        combined_conditions = combined_conditions.flatten(start_dim=2)
        combined_conditions = self.patching_layer(combined_conditions)

        return combined_conditions  # B, L, C


def get_torch_transformer(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=channels,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_torch_trans_decoder(heads=8, layers=1, channels=64):
    decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=channels,
        activation="gelu",
        batch_first=True,
    )
    return torch.nn.TransformerDecoder(decoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_transformer(
            heads=nheads, layers=1, channels=channels
        )
        self.feature_layer = get_torch_transformer(
            heads=nheads, layers=1, channels=channels
        )

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = (
            y.reshape(B, channel, K, L)
            .permute(0, 2, 1, 3)
            .reshape(B * K, channel, L)
        )
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = (
            y.reshape(B, K, channel, L)
            .permute(0, 2, 1, 3)
            .reshape(B, channel, K * L)
        )
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = (
            y.reshape(B, channel, K, L)
            .permute(0, 3, 1, 2)
            .reshape(B * L, channel, K)
        )
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = (
            y.reshape(B, L, channel, K)
            .permute(0, 2, 3, 1)
            .reshape(B, channel, K * L)
        )
        return y

    def forward(self, x, side_info, diffusion_emb, cond_in=None):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        if cond_in is not None:
            cond_in = cond_in.reshape(B, channel, K * L)
            y = x + diffusion_emb + cond_in  # (B,channel,K*L)
        else:
            y = x + diffusion_emb  # (B,channel,K*L)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.side_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "diffusion_embedding",
            self._build_embedding(num_steps, int(embedding_dim / 2)),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.diffusion_embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1
        )  # (T,dim*2)
        return table


class PatchedDiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser_config = config.denoiser_config
        self.dataset_config = config.dataset_config
        self.device = self.config.device

        # for each timestep in the timeseries, we have a positional embedding
        self.pos_embedding_dim = (
            self.denoiser_config.positional_embedding_dim
        )  # 128
        self.channels = self.denoiser_config.channels  # 512

        self.num_input_channels = self.dataset_config.num_channels  # K
        self.channel_embedding = torch.nn.Embedding(
            num_embeddings=self.num_input_channels,
            embedding_dim=self.denoiser_config.channel_embedding_dim,
        )  # 16

        # metadata encoder

        if (
            self.dataset_config.num_discrete_conditions > 0
            or self.dataset_config.num_continuous_labels > 0
        ) and self.denoiser_config.use_metadata:
            self.metadata_encoder = MetaDataEncoder(
                dataset_config=self.dataset_config,
                denoiser_config=self.denoiser_config,
                device=self.device,
            )
        else:
            self.metadata_encoder = None

        self.input_projection_layer = FCEncoder(
            self.denoiser_config.patch_len + self.channels,
            self.channels,
            dropout=0.1,
        )
        self.output_projection_layer = FCEncoder(
            self.channels + self.channels,
            self.denoiser_config.patch_len,
            dropout=0.1,
            use_layer_norm=False,
        )

        self.num_patches = (
            self.dataset_config.time_series_length
            // self.denoiser_config.patch_len
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=self.pos_embedding_dim
                    + self.denoiser_config.channel_embedding_dim,  # 128
                    channels=self.channels,  # 256
                    diffusion_embedding_dim=self.channels,  # 256
                    nheads=self.denoiser_config.n_heads,  # 16
                )
                for _ in range(self.denoiser_config.n_layers)
            ]
        )

        T = self.denoiser_config.T
        beta_0 = self.denoiser_config.beta_0  # 0.0001
        beta_T = self.denoiser_config.beta_T  # 0.1
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=T,
            embedding_dim=self.channels,
        )
        self.diffusion_hyperparameters = self.calc_diffusion_hyperparams(
            T=T,
            beta_0=beta_0,
            beta_T=beta_T,
        )

    def calc_diffusion_hyperparams(self, T, beta_0, beta_T):
        Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
        Alpha = 1 - Beta
        Alpha_bar = Alpha + 0
        Beta_tilde = Beta + 0
        for t in range(1, T):
            Alpha_bar[t] *= Alpha_bar[t - 1]
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        Sigma = torch.sqrt(Beta_tilde)

        Beta = Beta.to(self.device)
        Alpha = Alpha.to(self.device)
        Alpha_bar = Alpha_bar.to(self.device)
        Sigma = Sigma.to(self.device)

        _dh = {}
        _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
            T,
            Beta,
            Alpha,
            Alpha_bar,
            Sigma,
        )
        diffusion_hyperparams = _dh
        return diffusion_hyperparams

    def position_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, time_points):
        B = time_points.shape[0]
        L = time_points.shape[1]
        time_embed = self.position_embedding(
            time_points, self.pos_embedding_dim
        )  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).repeat(
            1, 1, self.num_input_channels, 1
        )  # (B, L, K, emb)
        feature_embed = self.channel_embedding(
            torch.arange(self.num_input_channels).to(self.device)
        )  # (K,emb)
        feature_embed = (
            feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        )
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, emb, K, L  )
        return side_info.to(self.device)

    def prepare_training_input(self, train_batch):
        # sample
        sample = train_batch["timeseries_full"].float().to(self.device)
        assert sample.shape[1] == self.num_input_channels

        # discrete and continuous condition input
        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )
        if len(discrete_label_embedding.shape) == 2:
            discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
            discrete_label_embedding = discrete_label_embedding.repeat(
                1, sample.shape[2], 1
            )
        assert discrete_label_embedding.shape[1] == sample.shape[2], (
            f"Wrong shape for discrete_label_embedding: {discrete_label_embedding.shape[1]} != {sample.shape[2]}"
        )
        assert (
            discrete_label_embedding.shape[2]
            == self.dataset_config.num_discrete_conditions
        ), (
            f"Wrong shape for discrete_label_embedding: {discrete_label_embedding.shape[2]} != {self.dataset_config.num_discrete_conditions}"
        )

        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )
        if len(continuous_label_embedding.shape) == 2:
            continuous_label_embedding = continuous_label_embedding.unsqueeze(1)
            continuous_label_embedding = continuous_label_embedding.repeat(
                1, sample.shape[2], 1
            )
        # diffusion step
        _dh = self.diffusion_hyperparameters
        B = sample.shape[0]
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
        t = torch.randint(
            0,
            T,
            (B,),
        )

        # noise and noisy data
        current_alpha_bar = (
            Alpha_bar[t].unsqueeze(1).unsqueeze(1).to(self.device)
        )
        noise = torch.randn_like(sample).float().to(self.device)
        noisy_data = (
            torch.sqrt(current_alpha_bar) * sample
            + torch.sqrt(1.0 - current_alpha_bar) * noise
        )
        denoiser_input = {
            "sample": sample,
            "noisy_sample": noisy_data,
            "noise": noise,
            "discrete_cond_input": discrete_label_embedding,
            "continuous_cond_input": continuous_label_embedding,
            "diffusion_step": t,
        }

        return denoiser_input

    def forward(self, denoiser_input):
        noisy_input = denoiser_input["noisy_sample"]  # (B, K, L)

        B = noisy_input.shape[0]  # B
        K = noisy_input.shape[1]  # K
        L = noisy_input.shape[2]  # L

        tp = (
            torch.arange(self.num_patches)
            .unsqueeze(0)
            .repeat(B, 1)
            .float()
            .to(self.device)
        )
        side_info = self.get_side_info(tp)

        if self.metadata_encoder is not None:
            cond_in = self.metadata_encoder(
                discrete_conditions=denoiser_input["discrete_cond_input"],
                continuous_conditions=denoiser_input["continuous_cond_input"],
            )
            cond_in = torch.einsum("blc->bcl", cond_in)  # (B,channels,L)
            cond_in = cond_in.unsqueeze(2).repeat(
                1, 1, K, 1
            )  # (B,channels,K,L)
        else:
            cond_in = None

        diffusion_step = denoiser_input["diffusion_step"]
        diffusion_step = diffusion_step.long()
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        x = noisy_input.unfold(
            dimension=-1,
            size=self.denoiser_config.patch_len,
            step=self.denoiser_config.stride,
        )

        xin = torch.cat(
            [
                x,
                diffusion_emb.unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, x.shape[1], x.shape[2], 1),
            ],
            dim=-1,
        )
        xin = self.input_projection_layer(xin)
        xin = xin.permute(0, 3, 1, 2)

        skip = []
        for layer in self.residual_layers:
            xin, skip_connection = layer(xin, side_info, diffusion_emb, cond_in)
            skip.append(skip_connection)

        xout = torch.sum(torch.stack(skip), dim=0) / math.sqrt(
            len(self.residual_layers)
        )
        xout = xout.permute(0, 2, 3, 1)
        xout = torch.cat(
            [
                xout,
                diffusion_emb.unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, xout.shape[1], xout.shape[2], 1),
            ],
            dim=-1,
        )
        xout = self.output_projection_layer(xout)
        xout = xout.reshape(B, K, L)
        return xout

    def prepare_output(self, synthesized):
        return synthesized.detach().cpu().numpy()
