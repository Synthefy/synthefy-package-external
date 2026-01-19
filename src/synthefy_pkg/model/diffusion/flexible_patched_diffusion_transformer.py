"""
Patched Diffusion Transformer
1. Patched input to the denoiser
2. Patching the input of the metadata encoder
"""

import json
import math
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
from omegaconf import DictConfig
from synthefy_pkg.configs.denoiser_configs import DenoiserConfig
from synthefy_pkg.configs.execution_configurations import Configuration
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Make model hyper parameters configurable.
# e.g. dropout=0.1 should be configurable.

COMPILE = True


SYNTHEFY_DATASETS_BASE = str(os.getenv("SYNTHEFY_DATASETS_BASE"))


class DiscreteFCEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, initial_projection_dim, projection_dim, dropout):
        super(DiscreteFCEncoder, self).__init__()
        self.projection1 = torch.nn.Linear(embedding_dim, initial_projection_dim)
        self.projection2 = torch.nn.Linear(initial_projection_dim, projection_dim)
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
    def __init__(self, embedding_dim, projection_dim, dropout, use_layer_norm=True):
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
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_() 
        self.projection.weight.data.zero_()
        self.projection.bias.data.zero_() 

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
        self.denoiser_config = denoiser_config

        self.num_discrete_conditions = self.dataset_config.num_discrete_conditions
        # Note: this is only used for backward compatibility for timeseries synthesis repo.
        self.num_discrete_labels = self.dataset_config.num_discrete_labels
        self.num_continuous_labels = self.dataset_config.num_continuous_labels

        self.discrete_condition_encoder_exists = (
            True if self.num_discrete_conditions > 0 else False
        )
        self.continuous_condition_encoder_exists = (
            True if self.num_continuous_labels > 0 else False
        )
        self.combined_condition_encoder_exists = (
            self.discrete_condition_encoder_exists
            and self.continuous_condition_encoder_exists
        )

        # This is where the metadata_encoder_config is used.
        # We don't have a separate class for MetaDataEncoder().
        if self.discrete_condition_encoder_exists:
            self.discrete_condition_patched_encoder = FCEncoder(self.denoiser_config.patch_len * self.denoiser_config.metadata_embedding_dim, self.denoiser_config.channels, dropout=0.1, use_layer_norm=True)
        if self.continuous_condition_encoder_exists:
            self.continuous_condition_embedding_layer = torch.nn.Linear(self.num_continuous_labels, int(self.denoiser_config.channels / 2)).to(self.device)
            self.continuous_condition_tokenizer = FCEncoder(self.denoiser_config.patch_len, int(self.denoiser_config.channels / 2), dropout=0.1, use_layer_norm=True)

        self.num_patches = self.dataset_config.time_series_length // self.denoiser_config.patch_len

        self.condition_transformer_encoder = get_torch_transformer(
            heads=self.denoiser_config.n_heads,
            layers=self.denoiser_config.n_layers,
            channels=self.denoiser_config.channels,
        )
        
        labels_description_pkl_loc = os.path.join(
            SYNTHEFY_DATASETS_BASE,
            dataset_config.dataset_name,
            "labels_description.pkl",
        )
        with open(labels_description_pkl_loc, "rb") as f:
            labels_description = pickle.load(f)
            
        colnames_loc = os.path.join(
            SYNTHEFY_DATASETS_BASE, dataset_config.dataset_name, "colnames.json"
        )
        with open(colnames_loc, "r") as f:
            colnames = json.load(f)
        discrete_labels_used = colnames["original_discrete_colnames"]

        self.discrete_metadata_indices = {}
        self.discrete_metadata_encoders = nn.ModuleDict()
        discrete_start_index = 0
        self.discrete_keys = []
        for key in discrete_labels_used:
            mod_key = "discrete_" + key
            num_discrete_labels_per_key = len(labels_description["discrete_labels"][key])
            self.discrete_metadata_indices[mod_key] = (
                discrete_start_index,
                discrete_start_index + num_discrete_labels_per_key,
            )
            self.discrete_keys.append(mod_key)
            self.discrete_metadata_encoders[mod_key] = DiscreteFCEncoder(
                embedding_dim=num_discrete_labels_per_key,
                initial_projection_dim=self.denoiser_config.channels,
                projection_dim=self.denoiser_config.metadata_embedding_dim,
                dropout=0.1,
            )
            discrete_start_index += num_discrete_labels_per_key
            
        self.total_metadata = self.num_continuous_labels + self.num_discrete_labels
        self.total_patches = self.num_patches * self.total_metadata

        self.final_projection = torch.nn.Linear(self.total_metadata * self.denoiser_config.channels, self.denoiser_config.channels)


    def position_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def forward(self, discrete_conditions, continuous_conditions, unmask_indices=None, forecast_flag=True):
        B = discrete_conditions.shape[0]
                    
        if self.discrete_condition_encoder_exists:
            discrete_embeddings_patched_list = []
            for key in self.discrete_keys:
                start_idx, end_idx = self.discrete_metadata_indices[key]
                part_discrete_conditions = discrete_conditions[:, :, start_idx:end_idx] 
                part_discrete_embeddings = self.discrete_metadata_encoders[key](part_discrete_conditions)
                part_discrete_embeddings_patched = part_discrete_embeddings.reshape(B, self.num_patches, self.denoiser_config.patch_len, self.denoiser_config.metadata_embedding_dim)
                discrete_embeddings_patched_list.append(part_discrete_embeddings_patched)
            discrete_embeddings_patched = torch.cat(discrete_embeddings_patched_list, dim=1)
            discrete_embeddings_flattened = discrete_embeddings_patched.reshape(B, self.num_patches * len(self.discrete_keys), -1)
            discrete_conditions_with_tokens = self.discrete_condition_patched_encoder(discrete_embeddings_flattened)
            
        if self.continuous_condition_encoder_exists:
            continuous_conditions_reshaped = continuous_conditions.permute(0,2,1)
            continuous_conditions_patched = continuous_conditions_reshaped.unfold(dimension=-1, size=self.denoiser_config.patch_len, step=self.denoiser_config.stride)
            continuous_conditions_patched = self.continuous_condition_tokenizer(continuous_conditions_patched)
            continuous_conditions_patched = continuous_conditions_patched.permute(0,1,3,2)
            
            continuous_conditions_global = torch.eye(self.num_continuous_labels).to(self.device) 
            continuous_conditions_global = self.continuous_condition_embedding_layer(continuous_conditions_global) 
            continuous_conditions_global = continuous_conditions_global.unsqueeze(0).unsqueeze(-1).repeat(B,1,1,self.num_patches)

            continuous_conditions_with_tokens = torch.cat([continuous_conditions_global, continuous_conditions_patched], dim=2)
            continuous_conditions_with_tokens = continuous_conditions_with_tokens.permute(0,1,3,2)
            continuous_conditions_with_tokens = continuous_conditions_with_tokens.reshape(B, -1, self.denoiser_config.channels)
          

        if self.discrete_condition_encoder_exists and self.continuous_condition_encoder_exists:
            combined_conditions = torch.cat(
                [discrete_conditions_with_tokens, continuous_conditions_with_tokens], dim=1
            )
        else:
            combined_conditions = (
                discrete_conditions_with_tokens
                if self.discrete_condition_encoder_exists
                else continuous_conditions_with_tokens
            )

        tp = torch.arange(combined_conditions.shape[1]).unsqueeze(0).repeat(B, 1).float().to(self.device)
        pos_emb = self.position_embedding(
            tp, self.denoiser_config.channels
        )
        combined_conditions_pos_encoded = combined_conditions + pos_emb
        
        if unmask_indices is not None:
            combined_conditions_pos_encoded_reshaped = combined_conditions_pos_encoded.reshape(B, self.total_metadata, self.num_patches, self.denoiser_config.channels)
            masked_combined_conditions = torch.zeros_like(combined_conditions_pos_encoded_reshaped)
            masked_combined_conditions[:, unmask_indices] = combined_conditions_pos_encoded_reshaped[:, unmask_indices]
            masked_combined_conditions = masked_combined_conditions.reshape(B, -1, self.denoiser_config.channels)
        else:
            masked_combined_conditions = combined_conditions_pos_encoded

        if forecast_flag:
            masked_combined_conditions_reshaped = masked_combined_conditions.reshape(B, self.total_metadata, self.num_patches, self.denoiser_config.channels)
            masked_combined_conditions_reshaped[:, :, -4:] = 0
            masked_combined_conditions = masked_combined_conditions_reshaped.reshape(B, -1, self.denoiser_config.channels)

        metadata_enc = self.condition_transformer_encoder(
            masked_combined_conditions.permute(1, 0, 2)
        )  # L, B, C, for the transformer to act across the time dimension
        metadata_enc = metadata_enc.permute(1, 0, 2)
        metadata_enc = metadata_enc.reshape(B, self.total_metadata, self.num_patches, self.denoiser_config.channels)
        metadata_enc = metadata_enc.permute(0, 2, 1, 3)
        metadata_enc = metadata_enc.reshape(B, self.num_patches, -1)
        metadata_enc = self.final_projection(metadata_enc)

        return metadata_enc  # B, C



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


class FlexiblePatchedDiffusionTransformer(nn.Module):
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

        total_metadata = self.dataset_config.num_discrete_labels + self.dataset_config.num_continuous_labels
        
        # randomly mask out some metadata
        num_indices_to_unmask = np.random.randint(1, total_metadata + 1)
        unmask_indices = np.random.choice(range(total_metadata), num_indices_to_unmask, replace=False) 
        forecast_flag = random.choice([True, False])


        denoiser_input = {
            "sample": sample,
            "noisy_sample": noisy_data,
            "noise": noise,
            "discrete_cond_input": discrete_label_embedding,
            "continuous_cond_input": continuous_label_embedding,
            "diffusion_step": t,
            "unmask_indices": unmask_indices,
            "forecast_flag": forecast_flag,
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
                unmask_indices=denoiser_input["unmask_indices"] if "unmask_indices" in denoiser_input else None,
                forecast_flag=denoiser_input["forecast_flag"] if "forecast_flag" in denoiser_input else True,
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
