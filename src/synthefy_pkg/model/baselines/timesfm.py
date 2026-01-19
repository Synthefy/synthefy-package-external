import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from loguru import logger
from timesfm.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    ResidualBlock,
    causal_mask,
    convert_paddings_to_mask,
    merge_masks,
)
from timesfm.timesfm_base import TimesFmCheckpoint
from torch import nn

from synthefy_pkg.model.architectures.decoder_utils import (
    get_torch_trans_decoder,
)
from synthefy_pkg.model.diffusion.diffusion_transformer import MetaDataEncoder
from synthefy_pkg.model.foundation_model.synthefy_foundation_forecasting_model_v2 import (
    NULL_TOKEN,
)


class TimesFM(nn.Module):
    def __init__(self, config):
        pass
        super().__init__()

        self.config = config

        # Assume pretrained if a checkpoint name is provided in config
        self.pretrained = False
        if hasattr(config.timesfm_config, "checkpoint_name"):
            self.checkpoint_name = config.timesfm_config.checkpoint_name
            self.pretrained = True

        # TimesFM model parameters
        self.input_patch_len = config.timesfm_config.input_patch_len
        self.output_patch_len = config.timesfm_config.output_patch_len
        self.num_layers = config.timesfm_config.num_layers
        self.model_dims = config.timesfm_config.model_dims
        self.num_heads = config.timesfm_config.num_heads

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
        self.model = PatchedTimeSeriesDecoder(self.tfm_config).to(self.device)

        # Create the metadata encoder if use_metadata is True
        if self.config.dataset_config.use_metadata:
            self.use_metadata = True
            # initialize the metadata encoder
            self.metadata_encoder = MetaDataEncoder(
                self.dataset_config, self.config, self.device
            )

            # initialize the hidden states encoders
            self.hidden_states_encoders = [
                ResidualBlock(
                    input_dims=self.model_dims,
                    hidden_dims=self.model_dims,
                    output_dims=self.config.metadata_encoder_config.channels,
                ).to(self.device)
                for _ in range(self.num_layers)
            ]

            # initialize the cross attention layers
            self.cross_attn_layers = [
                get_torch_trans_decoder(
                    heads=self.num_heads,
                    layers=1,
                    channels=self.config.metadata_encoder_config.channels,
                ).to(self.device)
                for _ in range(self.num_layers)
            ]

            # initialize the hidden states decoders
            self.hidden_states_decoders = [
                ResidualBlock(
                    input_dims=self.config.metadata_encoder_config.channels,
                    hidden_dims=self.model_dims,
                    output_dims=self.model_dims,
                ).to(self.device)
                for _ in range(self.num_layers)
            ]

            if self.config.timesfm_config.zero_metadata:
                logger.warning(
                    "Zeroing out metadata encoder and decoder parameters. Please make sure this is intentional."
                )
                for idx in range(self.num_layers):
                    for param in self.hidden_states_encoders[idx].parameters():
                        param.data = torch.zeros_like(param.data)
                for param in self.hidden_states_decoders[idx].parameters():
                    param.data = torch.zeros_like(param.data)
                for param in self.cross_attn_layers[idx].parameters():
                    param.data = torch.zeros_like(param.data)
            else:
                logger.info(
                    "Not zeroing out metadata encoder and decoder parameters."
                )
        else:
            self.use_metadata = False

        # Load the pretrained model state dictionary
        if self.pretrained:
            state_dict = self._get_hf_state_dict()
            self.model.load_state_dict(state_dict)
        else:
            raise ValueError("TimesFM requires a pretrained model")

        # Put the model into training mode
        self.model.train()

        # Set the finetune_stacked_layers flag
        self.model.finetune_stacked_layers = (
            self.config.timesfm_config.finetune_stacked_layers
        )
        if not self.config.timesfm_config.finetune_stacked_layers:
            # set requires_grad to False for all parameters in the stacked transformer layers
            for layer in self.model.stacked_transformer.layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def _get_hf_state_dict(self):
        """
        Pull the pretrained model from huggingface and return the state dict.
        """
        checkpoint = TimesFmCheckpoint(huggingface_repo_id=self.checkpoint_name)

        checkpoint_path = checkpoint.path
        repo_id = checkpoint.huggingface_repo_id

        if repo_id is None:
            raise ValueError("Repo ID is None")

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                str(snapshot_download(repo_id, local_dir=checkpoint.local_dir)),
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
            0, self.input_patch_len, size=effective_batch_size
        )

        # Create the mask by broadcasting the comparison over each row
        mask = (
            np.arange(self.time_series_length) < rand_lengths[:, None]
        ).astype(int)

        return torch.tensor(mask, dtype=torch.float32).to(self.device)

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

    def _prepare_training_input_for_foundation_dataset(self, train_batch):
        train_batch = train_batch["timeseries"].to(self.device).float()
        batch_size = train_batch.shape[0]
        num_correlates = train_batch.shape[
            1
        ]  # univariate forecaster should be 1

        # extracting multi modal information
        all_timestamps = train_batch[
            ...,
            self.dataset_config.timestamp_start_idx : self.dataset_config.timestamp_end_idx,
        ]
        all_descriptions = train_batch[
            ...,
            self.dataset_config.dataset_description_start_idx : self.dataset_config.dataset_description_end_idx,
        ]
        all_continuous = train_batch[
            ...,
            self.dataset_config.continuous_start_idx : self.dataset_config.continuous_end_idx,
        ]
        all_dataset_ids = train_batch[..., -1]

        # reshape the multi modal information
        all_timestamps = all_timestamps.reshape(
            batch_size,
            num_correlates,
            self.dataset_config.time_series_length,
            self.dataset_config.num_timestamp_features,
        ).permute(
            0, 1, 3, 2
        )  # B X NUM_CORRELATES X NUM_TIMESTAMP_FEATURES X TIME_SERIES_LENGTH
        all_descriptions = all_descriptions.unsqueeze(
            -1
        )  # B X NUM_CORRELATES X NUM_TEXTUAL_FEATURES X 1
        all_continuous = all_continuous.unsqueeze(
            -2
        )  # B X NUM_CORRELATES X 1 X TIME_SERIES_LENGTH

        # clear nan, NULL_TOKEN and obtain mask
        all_timestamps, all_timestamps_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_timestamps, category="continuous"
            )
        )
        all_descriptions, all_descriptions_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_descriptions, category="textual"
            )
        )
        all_continuous, all_continuous_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_continuous, category="continuous"
            )
        )

        all_dataset_ids, all_dataset_ids_mask = (
            self.clear_nan_null_and_obtain_mask(
                all_dataset_ids, category="textual"
            )
        )

        decoder_input = {
            "timestamps": all_timestamps,
            "descriptions": all_descriptions,
            "continuous": all_continuous,  # roughly concat(history, forecast)
            "timestamps_mask": all_timestamps_mask,
            "descriptions_mask": all_descriptions_mask,
            "continuous_mask": all_continuous_mask,
            "dataset_ids": all_dataset_ids,
            "dataset_ids_mask": all_dataset_ids_mask,
        }

        # These extra fields are added to enable compatibility with baselines.
        decoder_input["forecast"] = (
            decoder_input["continuous"]
            .squeeze(dim=2)[..., -self.horizon_len :]
            .permute(0, 2, 1)
        )  # torch.Size([1024, 20, 128])
        # B X NUM_CORRELATES X FORECAST_LEN
        decoder_input["history"] = (
            decoder_input["continuous"]
            .squeeze(dim=2)[..., : -self.horizon_len]
            .permute(0, 2, 1)
        )  # torch.Size([1024, 20, 128])
        decoder_input["timeseries_full"] = (
            decoder_input["continuous"].squeeze(dim=2).permute(0, 2, 1)
        )  # torch.Size([1024, 20, 256]

        decoder_input["full_discrete_conditions"] = torch.zeros(
            batch_size,
            self.dataset_config.time_series_length,
            0,  # No concept of num_discrete_conditions in FMV2
        )  # (batch, time_series_len, num_discrete_conditions)

        decoder_input["full_continuous_conditions"] = torch.zeros(
            batch_size,
            self.dataset_config.time_series_length,
            0,  # No concept of num_continuous_conditions in FMV2
        )  # (batch, time_series_len, num_continuous)

        # Need to add a couple things for timesfm compatibility
        # sample key needs to exist and be of shape (effective_batch_size, time_series_length)
        decoder_input["sample"] = (
            decoder_input["timeseries_full"]
            .permute(0, 2, 1)
            .view(-1, self.dataset_config.time_series_length)
        )

        effective_batch_size = decoder_input["sample"].shape[0]
        decoder_input["mask"] = self._generate_random_mask(
            effective_batch_size=effective_batch_size
        )
        decoder_input["frequency"] = torch.tensor(
            np.zeros((effective_batch_size, 1)), dtype=torch.long
        ).to(self.device)
        decoder_input["effective_batch_size"] = effective_batch_size
        decoder_input["sampled_batch_size"] = batch_size

        # TODO:
        # Scalars are the first 3 dims in the window. mean, variance, etc.
        # Scalar handling should be in the baseline or somewhere related to it.

        for key in decoder_input:
            if decoder_input[key] is not None and isinstance(
                decoder_input[key], torch.Tensor
            ):
                assert torch.all(
                    torch.logical_not(torch.isnan(decoder_input[key]))
                ), f"NaN values found in {key}"
                assert torch.all(decoder_input[key] != NULL_TOKEN), (
                    f"NULL_TOKEN values found in {key}"
                )

        return decoder_input

    def _prepare_training_input_for_standard_dataset(self, train_batch):
        # sample - originally shape (batch, channel, time)
        sample = train_batch["timeseries_full"].float().to(self.device)
        assert sample.shape[1] == self.dataset_config.num_channels, (
            f"{sample.shape[1]} != {self.dataset_config.num_channels}"
        )

        # history and forecast still required for validation
        history = sample[
            :, :, : -self.horizon_len
        ]  # (batch_size, num_channels, history_len)
        forecast = sample[:, :, -self.horizon_len :]

        # Permute the history and forecast to expected shape
        # (batch, channel, time) > (batch, time, channel)
        history = history.permute(0, 2, 1)
        forecast = forecast.permute(0, 2, 1)

        # Generate a random mask as described in timesfm paper
        effective_batch_size = sample.shape[0] * sample.shape[1]
        mask = self._generate_random_mask(effective_batch_size)

        # Flatten the sample to (batch * channel, time)
        # Since TimesFM is univariate, there is no difference between the
        # channel and batch dimensions-- extra channels are treated as extra samples.
        sample = sample.view(-1, self.time_series_length)

        # Frequency input. We use 0 for frequency selection
        # when evaluating the pretrained model, so we use the same
        # for fine-tuning the model
        frequency = torch.tensor(
            np.zeros((effective_batch_size, 1)), dtype=torch.long
        ).to(self.device)

        # Add metadata to the input
        # Continuous label embedding shape: (batch, time_series_length, num_continuous_labels)
        continuous_conditions = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )

        # If the continuous conditions are of shape (batch, num_continuous_labels),
        # we need to repeat them to match the shape (batch, time_series_length, num_continuous_labels)
        if len(continuous_conditions.shape) == 2:
            continuous_conditions = continuous_conditions.unsqueeze(1)
            continuous_conditions = continuous_conditions.repeat(
                1, self.dataset_config.time_series_length, 1
            )

        # Slice out the history and forecast continuous conditions
        history_continuous_conditions = continuous_conditions[
            :, : -self.horizon_len, :
        ]
        forecast_continuous_conditions = continuous_conditions[
            :, -self.horizon_len :, :
        ]

        # Discrete label embedding shape: (batch, time_series_length, num_discrete_labels)
        discrete_conditions = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )

        # If the discrete conditions are of shape (batch, num_discrete_labels),
        # we need to repeat them to match the shape (batch, time_series_length, num_discrete_labels)
        if len(discrete_conditions.shape) == 2:
            discrete_conditions = discrete_conditions.unsqueeze(1)
            discrete_conditions = discrete_conditions.repeat(
                1, self.dataset_config.time_series_length, 1
            )

        # Slice out the history and forecast discrete conditions
        history_discrete_conditions = discrete_conditions[
            :, : -self.horizon_len, :
        ]
        forecast_discrete_conditions = discrete_conditions[
            :, -self.horizon_len :, :
        ]

        decoder_input = {
            "sample": sample,
            "mask": mask,
            "frequency": frequency,
            "history": history,
            "forecast": forecast,
            "history_continuous_conditions": history_continuous_conditions,
            "history_discrete_conditions": history_discrete_conditions,
            "forecast_continuous_conditions": forecast_continuous_conditions,
            "forecast_discrete_conditions": forecast_discrete_conditions,
            "effective_batch_size": effective_batch_size,
        }

        return decoder_input

    def prepare_training_input(
        self, train_batch, is_foundation_dataset=False, *args, **kwargs
    ):
        if is_foundation_dataset:
            return self._prepare_training_input_for_foundation_dataset(
                train_batch
            )
        else:
            return self._prepare_training_input_for_standard_dataset(
                train_batch
            )

    def forward(self, forecast_input):
        # input_ts is of shape (batch * channel, time_series_length)
        # We slice the time dimension to only keep the history
        if self.use_metadata:
            cond_in = self.metadata_encoder(
                discrete_conditions=forecast_input[
                    "history_discrete_conditions"
                ],
                continuous_conditions=forecast_input[
                    "history_continuous_conditions"
                ],
            )
            cond_in = cond_in.unsqueeze(1).repeat(
                1, self.dataset_config.num_channels, 1, 1
            )
            cond_in = cond_in.contiguous().view(
                -1, cond_in.shape[-2], cond_in.shape[-1]
            )
            cond_in = cond_in.to(self.device)

        input_ts = forecast_input["sample"][:, : -self.horizon_len]

        # mask is of shape (batch * channel, time_series_length)
        mask = forecast_input["mask"]

        # frequency is of shape (batch * channel, 1)
        frequency = forecast_input["frequency"]

        # timesfm only supports an integer number of input patches
        # We can pad the input to the next multiple of the input patch length
        if self.context_len % self.input_patch_len != 0:
            pad_len = self.input_patch_len - (
                self.context_len % self.input_patch_len
            )
            input_ts = torch.nn.functional.pad(
                input_ts, (pad_len, 0, 0, 0), mode="constant", value=0
            )
            mask = torch.nn.functional.pad(
                mask, (pad_len, 0, 0, 0), mode="constant", value=1
            )

        # timesfm implements model.decode which calls self.model.forward
        if self.use_metadata:
            prediction, _ = self.decode_with_metadata(
                input_ts=input_ts,
                paddings=mask,
                freq=frequency,
                horizon_len=self.horizon_len,
                condition_input=cond_in,
            )
        else:
            prediction, _ = self.model.decode(
                input_ts=input_ts,
                paddings=mask,
                freq=frequency,
                horizon_len=self.horizon_len,
            )  # Result from model.decode is (batch * channel, time)

        # Reshape the prediction to (batch, channel, time)
        sampled_batch_size = forecast_input["sampled_batch_size"]
        prediction = prediction.view(
            (
                sampled_batch_size,
                -1,  # for foundation datasets, this corresponds num_correlates as num_channels is not set
                self.horizon_len,
            )
        )

        # Permute the prediction to (batch, time, channel)
        prediction = prediction.permute(0, 2, 1)

        return prediction

    def decode_with_metadata(
        self,
        input_ts: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.LongTensor,
        horizon_len: int,
        condition_input: torch.Tensor,
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
        if output_patch_len is None:
            output_patch_len = (
                self.model.config.horizon_len
            )  # 128, set to the default value based on the pretrained model
        num_decode_patches = (
            horizon_len + output_patch_len - 1
        ) // output_patch_len  # the logic for this is not clear

        for step_index in range(num_decode_patches):
            current_padding = paddings[
                :, 0 : final_out.shape[1]
            ]  # for the first loop, current_padding is of shape (B.C, History)
            input_ts = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]

            # until here, the processing is similar to the original decode function

            # in the original decode function, fprop_outputs = self(input_ts, input_padding, freq)
            # we need to modify this to include the metadata input
            # we explicitly perform all the operations in the forward function

            num_outputs = len(self.model.config.quantiles) + 1
            model_input, patched_padding, stats, _ = (
                self.model._preprocess_input(
                    input_ts=input_ts,
                    input_padding=input_padding,
                )
            )

            if stats is None:
                raise ValueError("Output stats from _preprocess_input is None")

            f_emb = self.model.freq_emb(freq)  # B x 1 x D
            model_input += f_emb

            # instead of model_output = self.model.stacked_transformer(model_input, patched_padding),
            # we apply the stacked transformer layers one by one (check lines 505 to 525 of pytorch_patched_decoder.py)
            padding_mask = convert_paddings_to_mask(
                patched_padding, model_input.dtype
            )
            atten_mask = causal_mask(model_input)
            mask = merge_masks(padding_mask, atten_mask)
            hidden_states = model_input
            for stacked_transformer_idx in range(
                len(self.model.stacked_transformer.layers)
            ):
                # we first reduce the hidden states to the metadata dimension
                hidden_states_residual = self.hidden_states_encoders[
                    stacked_transformer_idx
                ](hidden_states)
                # we then apply the cross attention layer
                hidden_states_residual = self.cross_attn_layers[
                    stacked_transformer_idx
                ](
                    tgt=hidden_states_residual,
                    memory=condition_input,
                )
                # we then increase the hidden states residual back to the model dimension
                hidden_states_residual = self.hidden_states_decoders[
                    stacked_transformer_idx
                ](hidden_states_residual)

                # we then apply the stacked transformer layer
                layer = self.model.stacked_transformer.layers[
                    stacked_transformer_idx
                ]
                _, hidden_states = layer(
                    hidden_states=hidden_states,
                    mask=mask,
                    paddings=patched_padding,
                    kv_write_indices=None,
                    kv_cache=None,
                )

                # we add the residual to the hidden states
                hidden_states = hidden_states_residual + hidden_states

            model_output = hidden_states
            fprop_outputs = self.model._postprocess_output(
                model_output, num_outputs, stats
            )

            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[
                    :, :-1, : self.model.config.patch_len, :
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
                :, : (context_len - self.config.patch_len + horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.cat(full_outputs, dim=1)[:, 0:horizon_len, :]

        return (full_outputs[:, :, 0], full_outputs)
