from torch import nn
from transformers import PatchTSTConfig, PatchTSTForPrediction


class PatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.dataset_config = config.dataset_config

        self.patchtst_params = config.patchtst_config

        self.patchtst_config = PatchTSTConfig(
            num_input_channels=self.patchtst_params.num_input_channels,
            context_length=self.patchtst_params.context_length,
            prediction_length=self.patchtst_params.prediction_length,
            distribution_output=self.patchtst_params.distribution_output,
            loss=self.patchtst_params.loss,
            patch_length=self.patchtst_params.patch_length,
            patch_stride=self.patchtst_params.patch_stride,
            num_hidden_layers=self.patchtst_params.num_hidden_layers,
            d_model=self.patchtst_params.d_model,
            num_attention_heads=self.patchtst_params.num_attention_heads,
            share_embedding=self.patchtst_params.share_embedding,
            channel_attention=self.patchtst_params.channel_attention,
            ffn_dim=self.patchtst_params.ffn_dim,
            norm_type=self.patchtst_params.norm_type,
            norm_eps=self.patchtst_params.norm_eps,
            attention_dropout=self.patchtst_params.attention_dropout,
            positional_dropout=self.patchtst_params.positional_dropout,
            path_dropout=self.patchtst_params.path_dropout,
            ff_dropout=self.patchtst_params.ff_dropout,
            bias=self.patchtst_params.bias,
            activation_function=self.patchtst_params.activation_function,
            pre_norm=self.patchtst_params.pre_norm,
            positional_encoding_type=self.patchtst_params.positional_encoding_type,
            use_cls_token=self.patchtst_params.use_cls_token,
            init_std=self.patchtst_params.init_std,
            share_projection=self.patchtst_params.share_projection,
            scaling=self.patchtst_params.scaling,
            do_mask_input=self.patchtst_params.do_mask_input,
            mask_type=self.patchtst_params.mask_type,
            random_mask_ratio=self.patchtst_params.random_mask_ratio,
            num_forecast_mask_patches=self.patchtst_params.num_forecast_mask_patches,
            channel_consistent_masking=self.patchtst_params.channel_consistent_masking,
            unmasked_channel_indices=self.patchtst_params.unmasked_channel_indices,
            mask_value=self.patchtst_params.mask_value,
            pooling_type=self.patchtst_params.pooling_type,
            head_dropout=self.patchtst_params.head_dropout,
            output_range=self.patchtst_params.output_range,
            num_parallel_samples=self.patchtst_params.num_parallel_samples,
        )

        self.model = PatchTSTForPrediction(self.patchtst_config).to(self.device)
        self.model.train()

    def prepare_training_input(self, train_batch, *args, **kwargs):
        # sample - originally shape (batch, channel, time)
        sample = train_batch["timeseries_full"].float().to(self.device)
        assert sample.shape[1] == self.dataset_config.num_channels, (
            f"{sample.shape[1]} != {self.dataset_config.num_channels}"
        )

        # history and forecast still required for validation
        horizon_len = self.patchtst_params.prediction_length
        history = sample[
            :, :, :-horizon_len
        ]  # (batch_size, num_channels, history_len)
        forecast = sample[:, :, -horizon_len:]

        # Permute the history and forecast to expected shape
        # (batch, channel, time) > (batch, time, channel)
        history = history.permute(0, 2, 1)
        forecast = forecast.permute(0, 2, 1)

        decoder_input = {
            "history": history,
            "forecast": forecast,
        }

        return decoder_input

    def forward(self, forecast_input):
        history = forecast_input["history"]

        output = self.model(past_values=history)

        return output.prediction_outputs
