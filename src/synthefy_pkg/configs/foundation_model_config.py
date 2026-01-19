from dataclasses import dataclass, field

from synthefy_pkg.configs.base_config import BaseConfig

COMPILE = False


@dataclass
class FoundationModelConfig(BaseConfig):
    decoder_checkpoint_name: str | None = None
    decoder_input_patch_len: int = 32
    decoder_output_patch_len: int = 128
    decoder_num_layers: int = 20
    decoder_model_dims: int = 1280
    decoder_num_heads: int = 16
    use_metadata: bool = False
    finetune_stacked_layers: bool = False
    random_vector_instead_text: bool = False
    mask_timestamp: bool = False
    position_embedding: str = "none"
    model_name: str = "synthefy_foundation_forecasting_model_v3"
    context_length: int = 256
    train_epoch_length: int = 10000
    val_epoch_length: int = 1000
    test_epoch_length: int = 1000
    use_column_identifier: bool = False
    absolute_max_bar_value: float = 50.0
    generate_point_forecast: bool = True
    generate_probabilistic_forecast_using_bins: bool = False
    num_bins: int = 5000
    bound_output_scale: float = -1.0
    correlate_attention_scaling: float = 0
    masking_schemes: list[str] = field(default_factory=lambda: ["random"])
    mask_mixing_rates: list[float] = field(default_factory=lambda: [1.0])
    target_filtering_schemes: list[str] = field(
        default_factory=lambda: ["time_indices_filter"]
    )
    # per table. Future half blocking and block_mask_every = True requires block_mask_num == num_correlates
    block_mask_num: int = 0
    # mean of the mask size
    block_target_mask_mean: int = 0
    # spread of the mask size
    block_target_mask_range: int = 0
    # always mask portion of every column
    block_mask_every: bool = False
    row_mask_ratio: int = 0
    row_use_lengths: bool = False
    row_mask_min: int = 0
    attention_masking_scheme: str = "causal"
    timeseries_token_dims: int = 9
    replace_timestamp_with_pos_enc: bool = False
    external_forecasts_to_use: list[str] = field(default_factory=lambda: [])
    tasks: list[str] = field(default_factory=lambda: ["univariate", "multivariate", "future_leaked"])
 
    def __post_init__(
        self,
    ):
        self.load_config()
        self.load_values_from_config()
        self._validate_config()

    def _validate_config(self) -> None:
        # Some parameters must be set if using pretrained weights
        if self.decoder_checkpoint_name is None:
            pass
        elif "google/timesfm-1.0" in self.decoder_checkpoint_name:
            assert self.decoder_input_patch_len == 32
            assert self.decoder_output_patch_len == 128
            assert self.decoder_num_layers == 20
            assert self.decoder_model_dims == 1280
            assert self.decoder_num_heads == 16
        elif "google/timesfm-2.0" in self.decoder_checkpoint_name:
            assert self.decoder_input_patch_len == 32
            assert self.decoder_output_patch_len == 128
            assert self.decoder_num_layers == 50
            assert self.decoder_model_dims == 1280
            assert self.decoder_num_heads == 16
