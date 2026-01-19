import torch
import torch.nn as nn

from synthefy_pkg.model.foundation_model.residual_block import ResidualBlock


class SynthefyFoundationModelContinuousEmbedder(nn.Module):
    """
    Model with 3 residual blocks to process time series tokens, textual tokens, and timestamp tokens.
    Using TimesFM's ResidualBlock implementation for consistency.
    """

    def __init__(
        self,
        model_dims,
        timeseries_token_dims,
        textual_token_dims,
        timestamp_token_dims,
        max_columns=0,
        use_column_identifier=False,
        device=None,
        train_as_univariate_model=False,
    ):
        super(SynthefyFoundationModelContinuousEmbedder, self).__init__()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model dimensions
        self.model_dims = model_dims
        self.hidden_dims = self.model_dims

        # Define input dimensions from config
        self.timeseries_token_dims = timeseries_token_dims
        self.textual_token_dims = textual_token_dims
        self.timestamp_token_dims = timestamp_token_dims
        self.use_column_identifier = use_column_identifier
        self.max_columns = max_columns
        self.train_as_univariate_model = train_as_univariate_model

        # Residual blocks for each modality
        self.timeseries_processor = ResidualBlock(
            input_dims=self.timeseries_token_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

        self.textual_processor = ResidualBlock(
            input_dims=self.textual_token_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

        self.timestamp_processor = ResidualBlock(
            input_dims=self.timestamp_token_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

        if self.use_column_identifier and not self.train_as_univariate_model:
            self.column_identifier_processor = ResidualBlock(
                input_dims=self.max_columns,
                hidden_dims=self.hidden_dims,
                output_dims=self.model_dims,
            ).to(self.device)

        # Optional: add a fusion layer to combine the processed tokens
        input_dims = (
            4 * self.model_dims
            if self.use_column_identifier and not self.train_as_univariate_model
            else 3 * self.model_dims
        )
        self.fusion_layer = ResidualBlock(
            input_dims=input_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.model_dims,
        ).to(self.device)

    def forward(self, sfm_input):
        """
        Process each type of token through its respective residual block.

        Args:
            sfm_input: Tokens representing time series values [batch, seq_len, timeseries_token_dims]

        Returns:
            Processed sequence representation [batch, seq_len, model_dims]
        """
        sfm_continuous_input = sfm_input[..., : self.timeseries_token_dims]
        sfm_textual_input = sfm_input[
            ...,
            self.timeseries_token_dims : self.timeseries_token_dims
            + self.textual_token_dims,
        ]
        sfm_timestamp_input = sfm_input[
            ...,
            self.timeseries_token_dims
            + self.textual_token_dims : self.timeseries_token_dims
            + self.textual_token_dims
            + self.timestamp_token_dims,
        ]
        sfm_column_identifier_input = sfm_input[
            ...,
            self.timeseries_token_dims
            + self.textual_token_dims
            + self.timestamp_token_dims :,
        ]
        # Process each modality
        processed_timeseries = self.timeseries_processor(sfm_continuous_input)
        processed_textual = self.textual_processor(sfm_textual_input)
        processed_timestamps = self.timestamp_processor(sfm_timestamp_input)
        if self.use_column_identifier and not self.train_as_univariate_model:
            processed_column_identifier = self.column_identifier_processor(
                sfm_column_identifier_input
            )

        # Combine the processed tokens
        combined = torch.cat(
            [processed_timeseries, processed_textual, processed_timestamps],
            dim=-1,
        )
        if self.use_column_identifier and not self.train_as_univariate_model:
            combined = torch.cat(
                [combined, processed_column_identifier], dim=-1
            )
        fused = self.fusion_layer(combined)

        return fused