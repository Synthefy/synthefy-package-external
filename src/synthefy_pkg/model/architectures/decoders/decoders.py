from synthefy_pkg.configs.tabicl_config import TokenDecoderConfig
from synthefy_pkg.model.architectures.decoders.linear_decoder import (
    LinearDecoder,
    MLPDecoder,
)
from synthefy_pkg.model.architectures.decoders.moe_ds_decoder import MixedMoE
from synthefy_pkg.model.architectures.decoders.row_column_decoder import (
    RowColumnDecoder,
)


def get_decoder_modules(
    input_expert_module_names: list[str],
    embed_expert_module_names: list[str],
    decoder_config: TokenDecoderConfig,
):
    input_expert_modules = []
    embed_expert_modules = []
    for module_name, module_config_path in zip(
        input_expert_module_names, decoder_config.input_expert_module_paths
    ):
        pass
        # TODO: implement initialization of the different models with their necessary inputs
        # if module_name == "chronos":
        #     input_expert_modules.append(ChronosDecoder(
        #         module_config_path,
        #     ))
        # elif module_name == "toto":
        #     input_expert_modules.append(TotoDecoder(
        #         module_config_path,
        #     ))
        # elif module_name == "TabPFN":
        #     input_expert_modules.append(TabPFN(
        #         module_config_path,
        #     ))
        # elif module_name == "sfm":
        #     input_expert_modules.append(SFMDecoder(
        #         module_config_path,
        #     ))
        # elif module_name == "icl":
        #     input_expert_modules.append(ICLDecoder(
        #         module_config_path,
        #     ))
        # else:
        #     raise ValueError(f"Invalid input expert module name: {module_name}")

    for module_name, module_config_path in zip(
        embed_expert_module_names, decoder_config.embed_expert_module_paths
    ):
        if module_name == "linear":
            embed_expert_modules.append(
                LinearDecoder(
                    decoder_config.model_dim,
                    decoder_config.output_dim,
                    decoder_config.dropout,
                )
            )
        elif module_name == "mlp":
            embed_expert_modules.append(
                MLPDecoder(
                    decoder_config.model_dim,
                    decoder_config.output_dim,
                    decoder_config.dropout,
                    decoder_config.num_layers,
                    decoder_config.hidden_dim,
                    decoder_config.activation,
                )
            )
        # TODO: we can add diffusion as a decoder here
        # elif module_name == "diffusion":
        # embed_expert_modules.append(DiffusionDecoder(
        #     decoder_config.model_dim,
        #     decoder_config.output_dim,
        #     decoder_config.dropout,
        # ))
        else:
            raise ValueError(f"Invalid embed expert module name: {module_name}")
    return input_expert_modules, embed_expert_modules


def init_decoders(decoder_config: TokenDecoderConfig):
    if decoder_config.decoder_type == "linear":
        return LinearDecoder(
            decoder_config.model_dim,
            decoder_config.output_dim,
            decoder_config.dropout,
        )
    elif decoder_config.decoder_type == "mlp":
        return MLPDecoder(
            decoder_config.model_dim,
            decoder_config.output_dim,
            decoder_config.dropout,
            decoder_config.num_layers,
            decoder_config.hidden_dim,
            decoder_config.activation,
        )
    elif decoder_config.decoder_type == "moe":
        input_expert_modules, embed_expert_modules = get_decoder_modules(
            decoder_config.input_expert_module_names,
            decoder_config.embed_expert_module_names,
            decoder_config,
        )
        # TODO: Does not handle routing to different GPUs yet
        return MixedMoE(
            input_expert_modules,
            embed_expert_modules,
        )
    elif decoder_config.decoder_type == "row_column":
        return RowColumnDecoder(
            decoder_config.model_dim,
            decoder_config.row_num_blocks,
            decoder_config.row_nhead,
            decoder_config.row_num_cls,
            decoder_config.col_num_blocks,
            decoder_config.col_nhead,
            decoder_config.col_num_cls,
            decoder_config.rope_base,
            decoder_config.ff_factor,
            decoder_config.dropout,
            decoder_config.activation,
            decoder_config.norm_first,
            decoder_config.weight_range,
        )
    else:
        raise ValueError(
            f"Decoder type {decoder_config.decoder_type} not supported"
        )
