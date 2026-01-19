import torch
from torch import nn

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.architectures.decoder_utils import (
    AttentionLayer,
    EncoderAvecCrossAttention,
    EncoderLayer,
    FullAttention,
    PatchEmbedding,
    get_torch_trans_decoder,
)
from synthefy_pkg.model.diffusion.diffusion_transformer import MetaDataEncoder

COMPILE = True


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        """
        n_vars: typically the num_channels
        nf: number of features; typically patch_num * d_model
        target_window: typically the pred_len
        """
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(
            nf, target_window
        )  # output of the linear layer needs to be the forecast_length
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # [bs, channels=nvars, d_model, patch_num]
        # [bs, nvars, d_model, patch_num] > [bs, nvars, patch_num * d_model]
        x = self.flatten(x)

        # Linear projects from [bs, nvars, patch_num * d_model] to [bs, nvars, forecast_length]
        x = self.linear(x)

        x = self.dropout(x)
        # [bs, nvars, forecast_length]
        return x


class SynthefyForecastingModelV1(nn.Module):
    def __init__(self, config: Configuration):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        assert config is not None, "Config cannot be None"

        self.config = config
        self.decoder_config = self.config.decoder_config
        self.dataset_config = self.config.dataset_config
        self.denoiser_config = self.config.denoiser_config
        self.metadata_encoder_config = self.config.metadata_encoder_config
        self.device = self.config.device
        self.seq_len = self.dataset_config.time_series_length
        self.pred_len = self.dataset_config.forecast_length

        assert self.decoder_config is not None, "decoder_config cannot be None"
        assert self.denoiser_config is not None, (
            "denoiser_config cannot be None"
        )
        padding = self.denoiser_config.stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model=self.denoiser_config.d_model,
            patch_len=self.denoiser_config.patch_len,
            stride=self.denoiser_config.stride,
            padding=padding,
            dropout=self.denoiser_config.dropout,
        )

        # TODO: We have both DenoiserConfig and decoder_config, which have some similar but some different values.

        # metadata encoder
        if self.dataset_config.use_metadata:
            # This is getting a different decoder_config than synthesis.
            self.metadata_encoder = MetaDataEncoder(
                self.dataset_config,
                self.denoiser_config,
                self.device,
            )

        if self.dataset_config.use_timestamp:
            self.timestamp_encoder = nn.Linear(
                self.dataset_config.num_timestamp_labels,
                self.denoiser_config.d_model,
            )

        # Encoder
        self.encoder = EncoderAvecCrossAttention(
            self_attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.denoiser_config.factor,
                            attention_dropout=self.denoiser_config.dropout,
                            output_attention=self.denoiser_config.output_attention,
                        ),
                        self.denoiser_config.d_model,
                        self.denoiser_config.n_heads,
                    ),
                    self.denoiser_config.d_model,
                    self.denoiser_config.d_ff,
                    dropout=self.denoiser_config.dropout,
                    activation=self.denoiser_config.activation,
                )
                for _ in range(self.denoiser_config.e_layers)
            ],
            cross_attn_layers=[
                get_torch_trans_decoder(
                    heads=self.denoiser_config.n_heads,
                    layers=1,
                    channels=self.denoiser_config.d_model,
                )
                for _ in range(self.denoiser_config.e_layers)
            ],
            conv_layers=None,
            norm_layer=torch.nn.LayerNorm(self.denoiser_config.d_model),
        )

        # Prediction Head
        self.history_length = self.seq_len - self.pred_len
        self.num_patches = int(
            (self.history_length - self.denoiser_config.patch_len)
            / self.denoiser_config.stride
            + 2
        )
        self.head_nf = self.denoiser_config.d_model * self.num_patches
        self.head = FlattenHead(
            self.dataset_config.num_channels,
            self.head_nf,
            self.pred_len,
            head_dropout=self.denoiser_config.dropout,
        )

    def prepare_training_input(self, train_batch, *args, **kwargs):
        # sample
        sample = train_batch["timeseries_full"].float().to(self.device)
        assert sample.shape[1] == self.dataset_config.num_channels, (
            f"{sample.shape[1]} != {self.dataset_config.num_channels}"
        )

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
        assert continuous_label_embedding.shape[1] == sample.shape[2], (
            "Wrong shape for continuous_label_embedding"
        )
        assert (
            continuous_label_embedding.shape[2]
            == self.dataset_config.num_continuous_labels
        ), "Wrong shape for continuous_label_embedding"

        history = sample[
            :, :, : -self.pred_len
        ]  # (batch_size, num_channels, history_len)
        forecast = sample[:, :, -self.pred_len :]
        history = history.permute(0, 2, 1)
        forecast = forecast.permute(0, 2, 1)
        history_discrete_cond_input = discrete_label_embedding[
            :, : -self.pred_len, :
        ]
        forecast_discrete_cond_input = discrete_label_embedding[
            :, -self.pred_len :, :
        ]
        history_continuous_cond_input = continuous_label_embedding[
            :, : -self.pred_len, :
        ]
        forecast_continuous_cond_input = continuous_label_embedding[
            :, -self.pred_len :, :
        ]
        mask = None

        denoiser_input = {
            "history": history,
            "forecast": forecast,
            "history_discrete_cond_input": history_discrete_cond_input,
            "history_continuous_cond_input": history_continuous_cond_input,
            "forecast_discrete_cond_input": forecast_discrete_cond_input,
            "forecast_continuous_cond_input": forecast_continuous_cond_input,
            "mask": mask,
            "full_discrete_conditions": discrete_label_embedding,
            "full_continuous_conditions": continuous_label_embedding,
        }

        if self.dataset_config.use_timestamp:
            # timestamp condition input
            timestamp_label_embedding = (
                train_batch["timestamp_label_embedding"].float().to(self.device)
            )
            assert timestamp_label_embedding.shape[1] == sample.shape[2], (
                "Wrong shape for timestamp_label_embedding"
            )
            assert (
                timestamp_label_embedding.shape[2]
                == self.dataset_config.num_timestamp_labels
            ), "Wrong shape for timestamp_label_embedding"

            history_time_feat = timestamp_label_embedding[
                :, : -self.pred_len, :
            ]
            forecast_time_feat = timestamp_label_embedding[
                :, -self.pred_len :, :
            ]

            denoiser_input["history_time_feat"] = history_time_feat
            denoiser_input["forecast_time_feat"] = forecast_time_feat

        return denoiser_input

    def forward(self, forecast_input):
        x_enc = forecast_input["history"]
        # (bs=batch_size, length=history_len, nvars=num_channels)

        assert self.denoiser_config is not None, (
            "denoiser_config cannot be None"
        )
        if self.dataset_config.use_metadata:
            cond_in = self.metadata_encoder(
                discrete_conditions=forecast_input[
                    "history_discrete_cond_input"
                ],
                continuous_conditions=forecast_input[
                    "history_continuous_cond_input"
                ],
            )
        else:
            cond_in = torch.zeros(
                x_enc.shape[0], x_enc.shape[1], self.denoiser_config.d_model
            ).to(self.device)

        if self.dataset_config.use_timestamp:
            history_time_feat = forecast_input["history_time_feat"]
            history_time_proj = self.timestamp_encoder(history_time_feat)
        else:
            history_time_proj = torch.zeros(
                x_enc.shape[0], x_enc.shape[1], self.denoiser_config.d_model
            ).to(self.device)

        cond_in = cond_in + history_time_proj
        cond_in = cond_in.unsqueeze(1).repeat(
            1, self.dataset_config.num_channels, 1, 1
        )
        cond_in = cond_in.reshape(-1, cond_in.shape[-2], cond_in.shape[-1])

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev  # normalize across history_len

        # TODO Document these shapes explicitly.
        # do patching and embedding
        # (bs, nvars, patch_num)?
        x_enc = x_enc.permute(0, 2, 1)

        # u: [bs * nvars, patch_num, d_model]
        # (bs, nvars, patch_num) > (bs * nvars, patch_num, d_model)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars, patch_num, d_model] > [bs * nvars, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out, cond_in)

        # z: [bs * nvars, patch_num, d_model] > [bs, nvars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        # (bs, nvars, patch_num, d_model) > (bs, nvars, d_model, patch_num)
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        # [bs, nvars, d_model, patch_num] > [bs, nvars, pred_len]
        dec_out = self.head(enc_out)

        # [bs, nvars, pred_len] > [bs, pred_len, nvars]
        dec_out = dec_out.permute(0, 2, 1)
        assert dec_out.shape[1] == self.pred_len

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
