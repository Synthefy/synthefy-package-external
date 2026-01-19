import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from timesfm.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    ResidualBlock,
    causal_mask,
    convert_paddings_to_mask,
    merge_masks,
)
from timesfm.timesfm_base import TimesFmCheckpoint
from torch import nn

NULL_TOKEN = -999999


def get_torch_transformer(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=channels,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class MetaDataEncoder(torch.nn.Module):
    def __init__(self, config, device):
        super(MetaDataEncoder, self).__init__()
        self.device = device
        self.dataset_config = config.dataset_config
        self.metadata_encoder_config = config.metadata_encoder_config
        self.foundation_model_config = config.foundation_model_config

        self.timestamp_start_idx = config.dataset_config.timestamp_start_idx
        self.timestamp_end_idx = config.dataset_config.timestamp_end_idx
        self.dataset_description_start_idx = (
            config.dataset_config.dataset_description_start_idx
        )
        self.dataset_description_end_idx = (
            config.dataset_config.dataset_description_end_idx
        )
        self.text_description_start_idx = (
            config.dataset_config.text_description_start_idx
        )
        self.text_description_end_idx = (
            config.dataset_config.text_description_end_idx
        )
        self.continuous_start_idx = config.dataset_config.continuous_start_idx
        self.continuous_end_idx = config.dataset_config.continuous_end_idx
        self.retrieved_timeseries_start_idx = (
            config.dataset_config.retrieved_timeseries_start_idx
        )
        self.retrieved_timeseries_end_idx = (
            config.dataset_config.retrieved_timeseries_end_idx
        )
        self.time_varying_textual_metadata_start_idx = (
            config.dataset_config.time_varying_textual_metadata_start_idx
        )
        self.time_varying_textual_metadata_end_idx = (
            config.dataset_config.time_varying_textual_metadata_end_idx
        )

        self.condition_transformer_encoder = get_torch_transformer(
            heads=self.metadata_encoder_config.n_heads,
            layers=self.metadata_encoder_config.num_encoder_layers,
            channels=self.metadata_encoder_config.channels,
        )

        self.textual_encoder = ResidualBlock(
            input_dims=self.dataset_config.text_embedding_dim,
            hidden_dims=self.metadata_encoder_config.channels,
            output_dims=self.metadata_encoder_config.channels,
        ).to(self.device)

        self.mask_input_dim = self.metadata_encoder_config.patch_len * int(
            self.metadata_encoder_config.mask_as_input
        )
        self.text_input_dim = (
            self.metadata_encoder_config.channels
            if self.metadata_encoder_config.pre_encode_text
            else self.dataset_config.text_embedding_dim
        )

        self.continuous_encoder = ResidualBlock(
            input_dims=self.metadata_encoder_config.patch_len
            + self.mask_input_dim
            + self.text_input_dim,
            hidden_dims=self.metadata_encoder_config.channels,
            output_dims=self.metadata_encoder_config.channels,
        ).to(self.device)

        self.discrete_condition_encoder = ResidualBlock(
            input_dims=self.metadata_encoder_config.patch_len
            * self.text_input_dim,
            hidden_dims=self.metadata_encoder_config.channels,
            output_dims=self.metadata_encoder_config.channels,
        ).to(self.device)

        self.timestamp_encoder = ResidualBlock(
            input_dims=self.metadata_encoder_config.patch_len
            * self.dataset_config.num_timestamp_features,
            hidden_dims=self.metadata_encoder_config.channels,
            output_dims=self.metadata_encoder_config.channels,
        ).to(self.device)

        self.projection_layer = nn.Linear(
            self.metadata_encoder_config.channels,
            self.foundation_model_config.decoder_model_dims,
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

    def patchify_masks(self, mask):
        patched_padding = torch.min(mask, dim=-1)[
            0
        ]  # Get the values from the min result
        return patched_padding

    def forward(self, exogenous_input_dict):
        B = exogenous_input_dict["continuous_input_time_series_history"].shape[
            0
        ]

        # process the dataset description input first
        dataset_description_input = exogenous_input_dict[
            "dataset_description_input"
        ].squeeze(-1)
        dataset_description_token = self.textual_encoder(
            dataset_description_input
        )
        dataset_description_token = dataset_description_token.unsqueeze(1)
        dataset_description_token_mask = exogenous_input_dict[
            "dataset_description_input_mask"
        ]

        # process the text description input next
        text_description_input = exogenous_input_dict[
            "text_description_input"
        ].squeeze(-1)
        text_description_token = self.textual_encoder(text_description_input)
        text_description_token = text_description_token.unsqueeze(1)
        text_description_mask = exogenous_input_dict[
            "text_description_input_mask"
        ]

        # process the continuous input next
        # process the time series history
        continuous_input_time_series_history = exogenous_input_dict[
            "continuous_input_time_series_history"
        ].squeeze(-2)  # (B, NEXOG, THIST)
        continuous_input_time_series_history_patched = (
            continuous_input_time_series_history.unfold(
                dimension=-1,
                size=self.metadata_encoder_config.patch_len,
                step=self.metadata_encoder_config.stride,
            )
        )  # (B, NEXOG, THIST // PATCH_LEN, PATCH_LEN)
        num_history_patches = (
            continuous_input_time_series_history_patched.shape[-2]
        )

        # process the text description history
        continuous_input_text_description = exogenous_input_dict[
            "continuous_input_text_description"
        ].squeeze(-1)  # (B, NEXOG, TDESC)
        continuous_input_text_description_token = self.textual_encoder(
            continuous_input_text_description
        )  # (B, NEXOG, C)
        continuous_input_text_description_token_for_patches = (
            (
                continuous_input_text_description_token.unsqueeze(-2).repeat(
                    1, 1, num_history_patches, 1
                )  # (B, NEXOG, THIST // PATCH_LEN, C)
            )
            if self.metadata_encoder_config.pre_encode_text
            else (
                continuous_input_text_description.unsqueeze(-2).repeat(
                    1, 1, num_history_patches, 1
                )  # (B, NEXOG, THIST // PATCH_LEN, TDESC)
            )
        )

        # process the history mask
        continuous_input_history_mask = exogenous_input_dict[
            "continuous_input_history_mask"
        ]  # (B, NEXOG, THIST)
        continuous_input_history_mask_for_patches = (
            continuous_input_history_mask.unfold(
                dimension=-1,
                size=self.metadata_encoder_config.patch_len,
                step=self.metadata_encoder_config.stride,
            )
        )  # (B, NEXOG, THIST // PATCH_LEN, PATCH_LEN)
        continuous_input_history = torch.cat(
            [
                continuous_input_text_description_token_for_patches,
                continuous_input_time_series_history_patched,
                continuous_input_history_mask_for_patches,
            ]
            if self.metadata_encoder_config.mask_as_input
            else [
                continuous_input_time_series_history_patched,
                continuous_input_text_description_token_for_patches,
            ],
            dim=-1,
        )  # (B, NEXOG, THIST // PATCH_LEN, C + PATCH_LEN + PATCH_LEN)

        continuous_input_history_token = self.continuous_encoder(
            continuous_input_history
        )
        continuous_input_history_token = continuous_input_history_token.reshape(
            B, -1, self.metadata_encoder_config.channels
        )
        continuous_input_history_token_mask = self.patchify_masks(
            continuous_input_history_mask_for_patches,
        )
        continuous_input_history_token_mask = (
            continuous_input_history_token_mask.reshape(B, -1)
        )

        # process the time varying textual metadata input similarly to the continuous input
        time_varying_textual_metadata_history = exogenous_input_dict[
            "time_varying_textual_metadata_history"
        ].permute(0, 2, 1)  # (B, T, TDESC)
        time_varying_textual_metadata_history_token = self.textual_encoder(
            time_varying_textual_metadata_history
        )  # (B, T, C)
        # TODO: we could add pre-encode blocking here too
        time_varying_textual_metadata_history_token_patched = (
            (
                time_varying_textual_metadata_history_token.reshape(
                    B,
                    -1,
                    self.metadata_encoder_config.patch_len,
                    self.metadata_encoder_config.channels,
                )  # (B, T // PATCH_LEN, PATCH_LEN, C)
            )
            if self.metadata_encoder_config.pre_encode_text
            else (
                time_varying_textual_metadata_history.reshape(
                    B,
                    -1,
                    self.metadata_encoder_config.patch_len,
                    self.dataset_config.text_embedding_dim,
                )
            )
        )

        num_time_varying_history_patches = (
            time_varying_textual_metadata_history_token_patched.shape[1]
        )
        if num_time_varying_history_patches == 0:
            time_varying_textual_metadata_history_token_patched = (
                time_varying_textual_metadata_history_token_patched.reshape(
                    B,
                    0,
                    self.metadata_encoder_config.patch_len
                    * self.text_input_dim,
                )
            )  # (B, T // PATCH_LEN, PATCH_LEN * C)
        else:
            time_varying_textual_metadata_history_token_patched = (
                time_varying_textual_metadata_history_token_patched.reshape(
                    B,
                    num_time_varying_history_patches,
                    -1,
                )
            )  # (B, T // PATCH_LEN, PATCH_LEN * C)
        time_varying_textual_metadata_history_token = (
            self.discrete_condition_encoder(
                time_varying_textual_metadata_history_token_patched
            )
        )  # (B, T // PATCH_LEN, C)
        time_varying_textual_metadata_history_mask = exogenous_input_dict[
            "time_varying_textual_metadata_history_mask"
        ]  # (B, T)
        if num_time_varying_history_patches == 0:
            time_varying_textual_metadata_history_mask_patched = (
                time_varying_textual_metadata_history_mask.reshape(
                    B,
                    0,
                    self.metadata_encoder_config.patch_len,
                )
            )  # (B, T // PATCH_LEN, PATCH_LEN)
            time_varying_textual_metadata_history_token_mask = (
                self.patchify_masks(
                    time_varying_textual_metadata_history_mask_patched
                )
            )  # (B, T // PATCH_LEN)
        else:
            time_varying_textual_metadata_history_mask_patched = (
                time_varying_textual_metadata_history_mask.unfold(
                    dimension=-1,
                    size=self.metadata_encoder_config.patch_len,
                    step=self.metadata_encoder_config.stride,
                )
            )  # (B, T // PATCH_LEN, PATCH_LEN)
            time_varying_textual_metadata_history_token_mask = (
                self.patchify_masks(
                    time_varying_textual_metadata_history_mask_patched,
                )
            )  # (B, T // PATCH_LEN)

        # process the timestamp input next
        timestamp_input_history = exogenous_input_dict[
            "timestamp_input_history"
        ]
        timestamp_input_history_patched = timestamp_input_history.unfold(
            dimension=-1,
            size=self.metadata_encoder_config.patch_len,
            step=self.metadata_encoder_config.stride,
        )
        timestamp_input_history_patched = (
            timestamp_input_history_patched.permute(0, 2, 1, 3).reshape(
                B,
                -1,
                self.metadata_encoder_config.patch_len
                * self.dataset_config.num_timestamp_features,
            )
        )
        timestamp_input_history_token = self.timestamp_encoder(
            timestamp_input_history_patched
        )
        timestamp_input_history_mask = exogenous_input_dict[
            "timestamp_input_history_mask"
        ]
        timestamp_input_history_mask_patched = (
            timestamp_input_history_mask.unfold(
                dimension=-1,
                size=self.metadata_encoder_config.patch_len,
                step=self.metadata_encoder_config.stride,
            )
        )
        timestamp_input_history_token_mask = self.patchify_masks(
            timestamp_input_history_mask_patched
        )
        # NOTE: Ignoring the retrieved timeseries input for now

        # concatenate the metadata tokens
        metadata_tokens = torch.cat(
            [
                dataset_description_token,
                text_description_token,
                continuous_input_history_token,
                time_varying_textual_metadata_history_token,
                timestamp_input_history_token,
            ]
            if self.metadata_encoder_config.include_metadata_timestamp_tokens
            else [
                dataset_description_token,
                text_description_token,
                continuous_input_history_token,
                time_varying_textual_metadata_history_token,
            ],
            dim=1,
        )

        # concatenate the metadata tokens mask
        metadata_tokens_mask = torch.cat(
            [
                dataset_description_token_mask,
                text_description_mask,
                continuous_input_history_token_mask,
                time_varying_textual_metadata_history_token_mask,
                timestamp_input_history_token_mask,
            ]
            if self.metadata_encoder_config.include_metadata_timestamp_tokens
            else [
                dataset_description_token_mask,
                text_description_mask,
                continuous_input_history_token_mask,
                time_varying_textual_metadata_history_token_mask,
            ],
            dim=1,
        )

        # apply the mask to the metadata tokens
        metadata_tokens_mask_expanded = (
            metadata_tokens_mask.unsqueeze(-1)
            .repeat(1, 1, self.metadata_encoder_config.channels)
            .float()
        )
        metadata_tokens = metadata_tokens * (
            1 - metadata_tokens_mask_expanded
        ) + metadata_tokens_mask_expanded * torch.zeros_like(metadata_tokens)

        tp = (
            torch.arange(metadata_tokens.shape[1])
            .unsqueeze(0)
            .repeat(B, 1)
            .float()
            .to(self.device)
        )
        pos_emb = self.position_embedding(
            tp, self.metadata_encoder_config.channels
        )
        metadata_tokens_pos_encoded = metadata_tokens + pos_emb

        metadata_enc = self.condition_transformer_encoder(
            metadata_tokens_pos_encoded.permute(1, 0, 2)
        )  # L, B, C, for the transformer to act across the time dimension
        metadata_enc = metadata_enc.permute(1, 0, 2)
        metadata_enc = self.projection_layer(metadata_enc)
        return metadata_enc, metadata_tokens_mask  # B, L, C


class SynthefyFoundationForecastingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Assume pretrained if a checkpoint name is provided in config
        self.pretrained = False
        if (
            hasattr(config.foundation_model_config, "decoder_checkpoint_name")
            and config.foundation_model_config.decoder_checkpoint_name
            is not None
            and len(config.foundation_model_config.decoder_checkpoint_name) > 0
        ):
            self.decoder_checkpoint_name = (
                config.foundation_model_config.decoder_checkpoint_name
            )
            self.pretrained = True

        # TimesFM Decoder model parameters
        self.decoder_input_patch_len = (
            config.foundation_model_config.decoder_input_patch_len
        )
        self.decoder_output_patch_len = (
            config.foundation_model_config.decoder_output_patch_len
        )
        self.decoder_num_layers = (
            config.foundation_model_config.decoder_num_layers
        )
        self.decoder_model_dims = (
            config.foundation_model_config.decoder_model_dims
        )
        self.decoder_num_heads = (
            config.foundation_model_config.decoder_num_heads
        )

        # Forecasting configuration
        self.horizon_len = config.dataset_config.forecast_length
        self.context_len = (
            config.dataset_config.time_series_length - self.horizon_len
        )
        self.time_series_length = config.dataset_config.time_series_length

        # Dataset and device specific configuration
        self.device = config.device
        self.batch_size = config.dataset_config.batch_size
        self.dataset_config = config.dataset_config

        # Directly instantiate a PatchedTimeSeriesDecoder from TimesFM
        # self.model_dims is also the hidden size of the attention layers
        # so, each head dimension should be self.model_dims // self.num_heads
        self.decoder_model = PatchedTimeSeriesDecoder(
            self.decoder_tfm_config
        ).to(self.device)

        # Create the metadata encoder if use_metadata is True
        if self.config.foundation_model_config.use_metadata:
            self.use_metadata = True
            # initialize the metadata encoder
            self.metadata_encoder = MetaDataEncoder(self.config, self.device)
        else:
            self.use_metadata = False

        # Load the pretrained model state dictionary
        if self.pretrained:
            state_dict = self._get_hf_state_dict()
            self.decoder_model.load_state_dict(state_dict)

        # Put the model into training mode
        self.decoder_model.train()
        self.metadata_encoder.train()

    def _get_hf_state_dict(self):
        """
        Pull the pretrained model from huggingface and return the state dict.
        """
        checkpoint = TimesFmCheckpoint(
            huggingface_repo_id=self.decoder_checkpoint_name
        )

        repo_id = checkpoint.huggingface_repo_id
        if repo_id is None:
            raise ValueError("huggingface_repo_id cannot be None")
        download_path = snapshot_download(
            repo_id, local_dir=checkpoint.local_dir
        )
        checkpoint_path = os.path.join(
            download_path,
            "torch_model.ckpt",
        )

        loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
        return loaded_checkpoint

    def _generate_random_mask(self, effective_batch_size: int):
        """
        Algorithm for generating random mask as described in the timesfm paper:
            1. Generate a random integer for each sample in the batch in the range [0, p),
            where p is the input patch length.
            2. A 1 in the mask represents an invalid input, i.e. masked OUT.
            3. Only a subset of the first input patch should be masked out.
        """
        # Generate a random integer for each batch row in [0, p)
        rand_lengths = np.random.randint(
            0, self.decoder_input_patch_len, size=effective_batch_size
        )

        # Create the mask by broadcasting the comparison over each row
        mask = (
            np.arange(self.time_series_length) < rand_lengths[:, None]
        ).astype(int)

        return torch.tensor(mask, dtype=torch.float32).to(self.device)

    def split_into_history_and_forecast(self, inp_tensor):
        """
        Split the input into history and forecast along the time dimension.
        The time dimension is the last dimension of the input.
        """
        inp_tensor_history = inp_tensor[..., : -self.horizon_len]
        inp_tensor_forecast = inp_tensor[..., -self.horizon_len :]
        return inp_tensor_history, inp_tensor_forecast

    def clear_nan_null_and_obtain_mask(self, elem, category=None):
        nan_mask = torch.isnan(elem)
        null_mask = elem == NULL_TOKEN
        if category == "textual":
            elem[nan_mask] = 0
            elem[null_mask] = 0
        else:
            elem[nan_mask] = -1
            elem[null_mask] = -1
        mask = torch.logical_or(nan_mask, null_mask)
        mask = mask.sum(dim=-2).bool()
        if category == "textual":
            invalid_mask = (elem == 0).all(dim=-2)
            mask = torch.logical_or(mask, invalid_mask)

        return elem, mask

    def extract_input_and_mask(self, input, category=None):
        """
        Prepare the input for the decoder.
        1. replace nan with -1
        2. replace NULL_TOKEN with -1
        3. validate the input for consistency
        4. split the input into history and forecast
        5. split the input mask into history and forecast
        """
        input, input_mask = self.clear_nan_null_and_obtain_mask(
            input, category=category
        )
        if input.shape[-1] == 1:
            return input, None, input_mask, None
        else:
            input_history, input_forecast = (
                self.split_into_history_and_forecast(input)
            )
            input_history_mask, input_forecast_mask = (
                self.split_into_history_and_forecast(input_mask)
            )
            return (
                input_history,
                input_forecast,
                input_history_mask,
                input_forecast_mask,
            )

    def prepare_training_input(self, train_batch, *args, **kwargs):
        sample = train_batch["timeseries"].float().to(self.device)
        batch_size = sample.shape[0]
        history, forecast, history_mask, forecast_mask = (
            self.extract_input_and_mask(sample, category="continuous")
        )

        # NOTE: sai: The timestamps are provided as input, so I think we don't need to provide a frequency input
        frequency = torch.tensor(
            np.zeros((batch_size, 1)), dtype=torch.long
        ).to(self.device)

        metadata = (
            train_batch["metadata"].float().to(self.device).squeeze(1)
        )  # 170712 - 384

        # extract timestamp input
        # this can be NULL_TOKEN or nan
        timestamp_input = metadata[
            :,
            self.dataset_config.timestamp_start_idx : self.dataset_config.timestamp_end_idx,
        ]
        timestamp_input = timestamp_input.reshape(
            batch_size, self.dataset_config.num_timestamp_features, -1
        )
        (
            timestamp_input_history,
            timestamp_input_forecast,
            timestamp_input_history_mask,
            timestamp_input_forecast_mask,
        ) = self.extract_input_and_mask(timestamp_input, category="continuous")

        # extract dataset description input
        # this can be all 0's or nans or NULL_TOKEN
        dataset_description_input = metadata[
            :,
            self.dataset_config.dataset_description_start_idx : self.dataset_config.dataset_description_end_idx,
        ].unsqueeze(-1)

        dataset_description_input, _, dataset_description_input_mask, _ = (
            self.extract_input_and_mask(
                dataset_description_input, category="textual"
            )
        )

        # extract text description input
        # this can be all 0's
        text_description_input = metadata[
            :,
            self.dataset_config.text_description_start_idx : self.dataset_config.text_description_end_idx,
        ].unsqueeze(-1)
        text_description_input, _, text_description_input_mask, _ = (
            self.extract_input_and_mask(
                text_description_input, category="textual"
            )
        )

        # extract continuous input
        # the text description can never by invalid values, can be zeros
        # null values are only present in the numerical metadata
        continuous_input = metadata[
            :,
            self.dataset_config.continuous_start_idx : self.dataset_config.continuous_end_idx,
        ]
        num_elem_per_cont_metadata = (
            self.dataset_config.text_embedding_dim
            + self.dataset_config.time_series_length
        )
        continuous_input = continuous_input.reshape(
            batch_size, -1, num_elem_per_cont_metadata
        )
        continuous_input_text_description = continuous_input[
            :, :, : self.dataset_config.text_embedding_dim
        ].unsqueeze(-1)  # creating the time axis for the text description
        continuous_input_time_series = continuous_input[
            :, :, self.dataset_config.text_embedding_dim :
        ].unsqueeze(-2)  # creating the value axis for the time series
        (
            continuous_input_time_series_history,
            continuous_input_time_series_forecast,
            continuous_input_time_series_history_mask,
            continuous_input_time_series_forecast_mask,
        ) = self.extract_input_and_mask(
            continuous_input_time_series, category="continuous"
        )
        (
            continuous_input_text_description,
            _,
            continuous_input_text_description_mask,
            _,
        ) = self.extract_input_and_mask(
            continuous_input_text_description, category="textual"
        )
        continuous_input_history_mask = torch.logical_or(
            continuous_input_time_series_history_mask,
            continuous_input_text_description_mask.repeat(
                1,
                1,
                continuous_input_time_series_history_mask.shape[-1],
            ),
        )

        # extract time varying textual metadata input
        # this can be all 0's, can never be nan or NULL_TOKEN
        time_varying_textual_metadata_input = metadata[
            :,
            self.dataset_config.time_varying_textual_metadata_start_idx : self.dataset_config.time_varying_textual_metadata_end_idx,
        ]
        time_varying_textual_metadata_input = (
            time_varying_textual_metadata_input.reshape(
                batch_size, -1, self.dataset_config.text_embedding_dim
            )
        ).permute(0, 2, 1)

        (
            time_varying_textual_metadata_history,
            time_varying_textual_metadata_forecast,
            time_varying_textual_metadata_history_mask,
            time_varying_textual_metadata_forecast_mask,
        ) = self.extract_input_and_mask(
            time_varying_textual_metadata_input, category="textual"
        )

        exogenous_input_dict = {
            "timestamp_input_history": timestamp_input_history,
            "timestamp_input_history_mask": timestamp_input_history_mask,
            "timestamp_input_forecast": timestamp_input_forecast,
            "timestamp_input_forecast_mask": timestamp_input_forecast_mask,
            "dataset_description_input": dataset_description_input,
            "dataset_description_input_mask": dataset_description_input_mask,
            "text_description_input": text_description_input,
            "text_description_input_mask": text_description_input_mask,
            "continuous_input_time_series_history": continuous_input_time_series_history,
            "continuous_input_text_description": continuous_input_text_description,
            "continuous_input_history_mask": continuous_input_history_mask,
            "time_varying_textual_metadata_history": time_varying_textual_metadata_history,
            "time_varying_textual_metadata_history_mask": time_varying_textual_metadata_history_mask,
            "time_varying_textual_metadata_forecast": time_varying_textual_metadata_forecast,
            "time_varying_textual_metadata_forecast_mask": time_varying_textual_metadata_forecast_mask,
        }

        decoder_input = {
            "sample": sample.permute(0, 2, 1),
            "frequency": frequency,
            "history": history.permute(0, 2, 1),
            "forecast": forecast.permute(0, 2, 1)
            if forecast is not None
            else None,
            "history_mask": history_mask,
            "forecast_mask": forecast_mask,
            "exogenous_input_dict": exogenous_input_dict,
        }

        return decoder_input

    def forward(self, decoder_input):
        if self.use_metadata:
            exogenous_input_dict = decoder_input["exogenous_input_dict"]
            cond_in, metadata_tokens_mask = self.metadata_encoder(
                exogenous_input_dict
            )
        else:
            cond_in = None
            metadata_tokens_mask = None
        # making history univariate
        history_ts = decoder_input["history"].squeeze(-1)

        # mask is of shape (batch * channel, time_series_length)
        ts_mask = torch.cat(
            [decoder_input["history_mask"], decoder_input["forecast_mask"]],
            dim=1,
        )

        # frequency is of shape (batch * channel, 1)
        frequency = decoder_input["frequency"]

        prediction, _ = self.decode_with_metadata(
            input_ts=history_ts,
            condition_input=cond_in,
            metadata_tokens_mask=metadata_tokens_mask,
            paddings=ts_mask,
            freq=frequency,
            horizon_len=self.horizon_len,
        )  # Result from model.decode is (batch * channel, time)

        # Reshape the prediction to (batch, channel, time)
        prediction = prediction.unsqueeze(-1)

        return prediction

    def decode_with_metadata(
        self,
        input_ts: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.LongTensor,
        horizon_len: int,
        condition_input: torch.Tensor | None,
        metadata_tokens_mask: torch.Tensor | None,
        output_patch_len: int | None = None,
        max_len: int = 512,
        return_forecast_on_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Auto-regressive decoding without caching.
        This function is a copy of the decode function from the original TimesFM code.
        We modify it to include the metadata input.
        """
        final_out = input_ts  # B.C X History
        context_len = final_out.shape[1]  # History
        full_outputs = []
        if paddings.shape[1] != final_out.shape[1] + horizon_len:
            # for history length = 64 and forecast length = 32, paddings shape is (B.C, 96)
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}"
            )
        paddings = paddings.float()
        if output_patch_len is None:
            output_patch_len = self.decoder_model.config.horizon_len

        num_decode_patches = (
            horizon_len + output_patch_len - 1
        ) // output_patch_len

        for step_index in range(num_decode_patches):
            current_padding = paddings[
                :, 0 : final_out.shape[1]
            ]  # for the first loop, current_padding is of shape (B.C, History)
            input_ts = final_out[
                :, -max_len:
            ]  # no effect, context length is always less than max_len (512)
            input_padding = current_padding[
                :, -max_len:
            ]  # no effect, context length is always less than max_len (512)

            num_outputs = len(self.decoder_model.config.quantiles) + 1
            model_input, patched_padding, stats, _ = (
                self.decoder_model._preprocess_input(
                    input_ts=input_ts.to(self.device),
                    input_padding=input_padding.to(self.device),
                )
            )

            f_emb = self.decoder_model.freq_emb(freq)  # B x 1 x D
            model_input += f_emb
            history_atten_mask = causal_mask(model_input)  # B x 1 x H x H

            # modifying the patched padding as per metadata input
            if condition_input is not None:
                num_metadata_tokens = condition_input.shape[1]
                num_history_tokens = model_input.shape[1]

                # padding the metadata input to the model input
                if metadata_tokens_mask is not None:
                    patched_padding = torch.cat(
                        [metadata_tokens_mask, patched_padding], dim=1
                    )
                # padding mask is not a causal mask, it is a mask to ignore invalid tokens
                padding_mask = convert_paddings_to_mask(
                    patched_padding, model_input.dtype
                )  # B x 1 x 1 x (M + H)

                # metadata attention mask is a causal mask for the metadata tokens
                metadata_atten_mask = torch.zeros(
                    (1, 1, num_metadata_tokens, num_metadata_tokens)
                ).to(self.device)  # 1 x 1 x M x M
                # metadata and history attention mask is a causal mask for the metadata and history tokens
                # but, currently it is more like a history attention mask
                metadata_and_history_atten_mask = torch.zeros(
                    (
                        1,
                        1,
                        num_metadata_tokens + num_history_tokens,
                        num_metadata_tokens + num_history_tokens,
                    )
                ).to(self.device)  # 1 x 1 x (M + H) x (M + H)
                metadata_and_history_atten_mask[
                    :, :, :num_metadata_tokens, :num_metadata_tokens
                ] = metadata_atten_mask
                metadata_and_history_atten_mask[
                    :, :, num_metadata_tokens:, num_metadata_tokens:
                ] = history_atten_mask

                mask = merge_masks(
                    padding_mask, metadata_and_history_atten_mask
                )
                hidden_states = torch.cat([condition_input, model_input], dim=1)
            else:
                padding_mask = convert_paddings_to_mask(
                    patched_padding, model_input.dtype
                )  # B x 1 x 1 x (M + H)
                mask = merge_masks(padding_mask, history_atten_mask)
                hidden_states = model_input
                num_metadata_tokens = 0

            for stacked_transformer_idx in range(
                len(self.decoder_model.stacked_transformer.layers)
            ):
                # we then apply the stacked transformer layer
                layer = self.decoder_model.stacked_transformer.layers[
                    stacked_transformer_idx
                ]
                _, hidden_states = layer(
                    hidden_states=hidden_states,
                    mask=mask,
                    paddings=patched_padding,
                    kv_write_indices=None,
                    kv_cache=None,
                )

            model_output = hidden_states[:, num_metadata_tokens:, :]
            if stats is not None:
                fprop_outputs = self.decoder_model._postprocess_output(
                    model_output, num_outputs, stats
                )
            else:
                raise ValueError("Stats are not available")

            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[
                    :, :-1, : self.decoder_model.config.patch_len, :
                ]
                new_full_ts = fprop_outputs.view(
                    new_full_ts.size(0), -1, new_full_ts.size(3)
                )

                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[
                :, -1, :output_patch_len, 0
            ]  # only the mean forecast
            new_full_ts = fprop_outputs[
                :, -1, :output_patch_len, :
            ]  # all the forecasts
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.cat(
                [final_out, new_ts], dim=-1
            )  # append the mean forecast to the final output which is the input to the next step

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = torch.cat(full_outputs, dim=1)[
                :,
                : (
                    context_len
                    - self.decoder_model.config.patch_len
                    + horizon_len
                ),
                :,
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.cat(full_outputs, dim=1)[:, 0:horizon_len, :]

        return (full_outputs[:, :, 0], full_outputs)
