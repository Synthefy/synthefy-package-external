import os
from pathlib import Path

import torch
from torch import nn

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.architectures.decoder_utils import (
    AttentionLayer,
    EncoderAvecCrossAttentionTriple,
    EncoderLayer,
    FullAttention,
    PatchEmbedding,
    get_torch_trans_decoder,
)
from synthefy_pkg.model.diffusion.diffusion_transformer import (
    MetaDataEncoder,
    TimestampEncoder,
)
from synthefy_pkg.model.forecasting.synthefy_forecasting_model_v1 import (
    FlattenHead,
)

COMPILE = True


def freeze_model(model):
    for layer in model.modules():
        for param in layer.parameters():
            param.requires_grad = False


def count_parameters(model):
    # total and trainable param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Total Parameters: {total_params}, Trainable Parameters: {trainable_params}"
    )


def initialize_weights_to_zero_and_freeze(model):
    for layer in model.modules():
        if (
            isinstance(layer, nn.Linear)
            or isinstance(layer, nn.Conv1d)
            or isinstance(layer, nn.Conv2d)
        ):
            nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif (
            isinstance(layer, nn.LayerNorm)
            or isinstance(layer, nn.BatchNorm1d)
            or isinstance(layer, nn.BatchNorm2d)
        ):
            nn.init.constant_(
                layer.weight, 1
            )  # Usually, norm layers have weight initialized to 1
            nn.init.constant_(layer.bias, 0)
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze_model(model):
    for layer in model.modules():
        for param in layer.parameters():
            param.requires_grad = True


def initialize_cross_attn_weights_to_zero(model):
    for layer in model.cross_attn_layers1:
        if (
            isinstance(layer, nn.Linear)
            or isinstance(layer, nn.Conv1d)
            or isinstance(layer, nn.Conv2d)
        ):
            nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    for layer in model.cross_attn_layers2:
        if (
            isinstance(layer, nn.Linear)
            or isinstance(layer, nn.Conv1d)
            or isinstance(layer, nn.Conv2d)
        ):
            nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


class SynthefyForecastingModelV2(nn.Module):
    def __init__(self, config: Configuration):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.flip = False
        self.config = config

        self.decoder_config = self.config.decoder_config
        self.dataset_config = self.config.dataset_config
        self.denoiser_config = self.config.denoiser_config
        self.device = self.config.device
        self.seq_len = self.dataset_config.time_series_length
        self.pred_len = self.dataset_config.forecast_length
        padding = self.decoder_config.stride

        # this variable holds importance only during training
        # pre-training is false if full training with metadata is done, enabling both cross attentions
        # it is true otherwise and disables usage of metadata cross attention
        # if finetuning, it is shut down in between and metadata cross attention is used
        self.pre_training = True

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model=self.decoder_config.d_model,
            patch_len=self.decoder_config.patch_len,
            stride=self.decoder_config.stride,
            padding=padding,
            dropout=self.decoder_config.dropout,
        )

        # metadata encoder
        if self.dataset_config.use_timestamp:
            self.timestamp_encoder = TimestampEncoder(self.config)
            initialize_weights_to_zero_and_freeze(self.timestamp_encoder)
            print("Timestamp Encoder Initialized with Zero Weights and Frozen")
            # NOT FREEZING THIS SINCE WE DO NOT COMPARE NON-TIMESTAMP PERFORMANCE

        if self.denoiser_config.use_metadata:
            self.metadata_encoder = MetaDataEncoder(
                self.dataset_config,
                self.denoiser_config,
                self.device,
                # decoder_config vs denoiser_config???
            )
            initialize_weights_to_zero_and_freeze(self.metadata_encoder)
            print("Metadata Encoder Initialized with Zero Weights and Frozen")

        # Encoder
        self.encoder = EncoderAvecCrossAttentionTriple(
            self_attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.decoder_config.factor,
                            attention_dropout=self.decoder_config.dropout,
                            output_attention=self.decoder_config.output_attention,
                        ),
                        self.decoder_config.d_model,
                        self.decoder_config.n_heads,
                    ),
                    self.decoder_config.d_model,
                    self.decoder_config.d_ff,
                    dropout=self.decoder_config.dropout,
                    activation=self.decoder_config.activation,
                )
                for _ in range(self.decoder_config.e_layers)
            ],
            cross_attn_layers1=[
                get_torch_trans_decoder(
                    heads=self.decoder_config.n_heads,
                    layers=1,
                    channels=self.decoder_config.d_model,
                )
                for _ in range(self.decoder_config.e_layers)
            ],
            cross_attn_layers2=[
                get_torch_trans_decoder(
                    heads=self.decoder_config.n_heads,
                    layers=1,
                    channels=self.decoder_config.d_model,
                )
                for _ in range(self.decoder_config.e_layers)
            ],
            conv_layers=None,
            norm_layer=torch.nn.LayerNorm(self.decoder_config.d_model),
        )

        initialize_cross_attn_weights_to_zero(self.encoder)

        if not self.denoiser_config.use_metadata:
            print("Cross Attention Weights Initialized with Zero Weights")
            print("freezing metadata cross attn layers")
            for param in self.encoder.cross_attn_layers2.parameters():
                param.requires_grad = False

        if not self.dataset_config.use_timestamp:
            print("Cross Attention Weights Initialized with Zero Weights")
            print("freezing both cross attn layers")
            for param in self.encoder.cross_attn_layers1.parameters():
                param.requires_grad = False
            for param in self.encoder.cross_attn_layers2.parameters():
                param.requires_grad = False

        # Prediction Head
        self.history_length = self.seq_len - self.pred_len
        self.num_patches = int(
            (self.history_length - self.decoder_config.patch_len)
            / self.decoder_config.stride
            + 2
        )
        self.head_nf = self.decoder_config.d_model * self.num_patches
        self.head = FlattenHead(
            self.dataset_config.num_channels,
            self.head_nf,
            self.pred_len,
            head_dropout=self.decoder_config.dropout,
        )

    def prepare_training_input(self, train_batch, log_dir, *args, **kwargs):
        self.epoch = train_batch["epoch"]
        if (
            self.epoch
            == int(
                self.denoiser_config.ratio
                * self.config.training_config.max_epochs
            )
            and not self.flip
        ):
            # once the pre-training is over, save the best model
            model_path = Path(log_dir) / Path("checkpoints/best_model.ckpt")
            # copy this and save in the same path as best_model_{epoch}.ckpt
            os.system(
                f"cp {model_path} {Path(log_dir) / Path('checkpoints/best_model_' + str(self.epoch) + '.ckpt')}"
            )
            # load the best model
            loaded_model = torch.load(model_path)
            # convert state_dict of loaded_model in appropriate format for setting weifhts and biases aas that of current model
            loaded_model["state_dict"] = {
                k.replace("decoder_model.", ""): v
                for k, v in loaded_model["state_dict"].items()
            }
            # use this to load
            self.load_state_dict(loaded_model["state_dict"])
            # unfreeze metadata encoder
            # print("unfreezing metadata encoder")
            # unfreeze_model(self.metadata_encoder)
            # unfreeze timestamp encoder
            if self.dataset_config.use_timestamp:
                print("unfreezing timestamp encoder")
                unfreeze_model(self.timestamp_encoder)
            # freeze encoder
            print("freezing encoder")
            freeze_model(self.encoder)
            # unfreeze cross attn layers
            print("unfreezing timestamp cross attn layers")
            for param in self.encoder.cross_attn_layers1.parameters():
                param.requires_grad = True
            # freeze patch embedding
            freeze_model(self.patch_embedding)
            # end self.pre_training
            self.pre_training = False

            self.flip = True

        # sample
        sample = train_batch["timeseries_full"].float().to(self.device)
        assert sample.shape[1] == self.dataset_config.num_channels

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
            "Wrong shape for discrete_label_embedding"
        )
        assert (
            discrete_label_embedding.shape[2]
            == self.dataset_config.num_discrete_conditions
        ), "Wrong shape for discrete_label_embedding"

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

        history = sample[:, :, : -self.pred_len]
        forecast = sample[:, :, -self.pred_len :]
        history = history.permute(0, 2, 1)
        forecast = forecast.permute(0, 2, 1)
        # history_time_feat = timestamp_label_embedding[:, :-self.pred_len, :]
        # forecast_time_feat = timestamp_label_embedding[:, -self.pred_len:, :]
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
        cond_in_metadata = torch.zeros(
            x_enc.shape[0], x_enc.shape[1], self.decoder_config.d_model
        ).to(self.device)
        if not self.pre_training:
            if self.dataset_config.use_timestamp:
                history_time_feat = forecast_input["history_time_feat"]
                history_time_proj = self.timestamp_encoder(history_time_feat)
            else:
                history_time_proj = torch.zeros(
                    x_enc.shape[0], x_enc.shape[1], self.decoder_config.d_model
                ).to(self.device)
            cond_in_timestamp = history_time_proj
        else:
            cond_in_timestamp = torch.zeros(
                x_enc.shape[0], x_enc.shape[1], self.decoder_config.d_model
            ).to(self.device)

        # cond_in = cond_in.unsqueeze(1).repeat(1, self.dataset_config.num_channels, 1, 1)
        # cond_in = cond_in.reshape(-1, cond_in.shape[-2], cond_in.shape[-1])

        cond_in_metadata = cond_in_metadata.unsqueeze(1).repeat(
            1, self.dataset_config.num_channels, 1, 1
        )
        cond_in_metadata = cond_in_metadata.reshape(
            -1, cond_in_metadata.shape[-2], cond_in_metadata.shape[-1]
        )
        cond_in_timestamp = cond_in_timestamp.unsqueeze(1).repeat(
            1, self.dataset_config.num_channels, 1, 1
        )
        cond_in_timestamp = cond_in_timestamp.reshape(
            -1, cond_in_timestamp.shape[-2], cond_in_timestamp.shape[-1]
        )

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # print(enc_out.shape)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        if self.pre_training:
            att_mode = "self"
        else:
            att_mode = "selfandtimestamp"

        enc_out, attns = self.encoder(
            enc_out, cond_in_timestamp, cond_in_metadata, att_mode
        )
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
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
