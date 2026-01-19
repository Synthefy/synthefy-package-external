import os

import numpy as np
import torch
from loguru import logger

from synthefy_pkg.model.utils.attention_analysis import (
    AttentionAnalyzer,
)
from synthefy_pkg.model.utils.input_weight_analysis import (
    InputWeightAnalyzer,
)
from synthefy_pkg.model.utils.synthefy_foundation_model_analysis import (
    SynthefyFoundationModelAnalyzer,
)


def load_and_visualize_weights(
    config,
    forecast_experiment,
    attention_analyzer,
    input_weight_analyzer,
    args,
    split="test",
    num_batches=10,
    weight_type="attention",
    max_input_batch_size=2,  # TODO: modify this later
):
    dataloader = getattr(
        forecast_experiment.data_loader, f"{split}_dataloader"
    )()

    pattern_per_batch = []
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx} of {num_batches}")
        if batch_idx > num_batches:
            break

        # Get attention weights
        mask_indices = np.array([])
        # TODO: we should probably have a function for get_attention_weights that runs this directly, as well as other attentional operations
        if weight_type in ["attention", "input_gradient"]:
            input_batch = (
                attention_analyzer.model_analyzer.model.prepare_training_input(
                    batch
                )
            )
        else:
            mask_type = (
                "correlate"
                if weight_type == "counterfactual_correlate"
                else "input_value"
            )
            input_batch, mask_indices = (
                attention_analyzer.model_analyzer.create_counterfactual_batches(
                    batch, mask_type=mask_type
                )
            )
        if weight_type == "attention":
            attention_weights = (
                attention_analyzer.model_analyzer.get_attention_weights(
                    input_batch
                )
            )
        elif weight_type == "input_attention":
            attention_weights = input_weight_analyzer.model_analyzer.get_input_attention_weights(
                input_batch
            )
        elif weight_type == "input_gradient":
            attention_weights = (
                attention_analyzer.model_analyzer.compute_input_gradients(
                    input_batch
                )
            )
        elif weight_type in [
            "counterfactual_all_attention",
            "counterfactual_all_mask",
            "counterfactual_correlate",
        ]:
            if len(input_batch) > max_input_batch_size:
                attention_weights = []
                for i in range(
                    0,
                    input_batch["continuous_tokens"].shape[0],
                    max_input_batch_size,
                ):
                    input_batch_chunk = {
                        k: v[i : i + max_input_batch_size]
                        for k, v in input_batch.items()
                    }
                    mask_indices_chunk = mask_indices[
                        i : i + max_input_batch_size
                    ]
                    print(
                        mask_indices_chunk.shape,
                        mask_indices.shape,
                        input_batch_chunk["continuous_tokens"].shape,
                        input_batch["continuous_tokens"].shape,
                    )
                    if (
                        weight_type == "counterfactual_all_attention"
                    ):  # TODO: correlate_attention not supported yet
                        attention_weights.append(
                            attention_analyzer.model_analyzer.compute_attention_counterfactual(
                                input_batch_chunk, mask_indices_chunk
                            )["difference"]
                        )
                    elif (
                        weight_type == "counterfactual_all_mask"
                        or weight_type == "counterfactual_correlate"
                    ):
                        attention_weights.append(
                            attention_analyzer.model_analyzer.compute_counterfactual_masking(
                                input_batch_chunk,
                                mask_indices_chunk,
                                mask_type="input_value",
                            )["difference"]
                        )
                    else:
                        raise ValueError(
                            f"Weight type {weight_type} not supported"
                        )
                attention_weights = torch.cat(attention_weights, dim=0)
            else:
                attention_weights = attention_analyzer.model_analyzer.compute_counterfactual_all(
                    input_batch, mask_type="attention"
                )
        else:
            raise ValueError(f"Weight type {weight_type} not supported")
        # Analyze attention patterns
        os.makedirs(os.path.join(args.output_dir, "patterns"), exist_ok=True)

        if weight_type == "attention":
            print(attention_weights)
            patterns = attention_analyzer.analyze_attention_patterns(
                attention_weights,
                plot=True,
                save_path=f"{os.path.join(args.output_dir, 'patterns')}",
                num_heads=config.foundation_model_config.decoder_num_heads,
                batch_idx=batch_idx,
            )
        else:
            print([aw.shape for aw in attention_weights])
            patterns = input_weight_analyzer.analyze_input_patterns(
                attention_weights,
                plot=True,
                save_path=f"{os.path.join(args.output_dir, 'patterns')}",
                batch_idx=batch_idx,
            )

    if weight_type == "input_gradient":
        # Visualize input gradient for each layer and head
        input_weight_analyzer.visualize_inputs(
            attention_weights,
            visualize_correlates=False,
            save_path=f"{args.output_dir}/input_gradient.png",
        )
        input_weight_analyzer.visualize_inputs(
            attention_weights,
            visualize_correlates=True,
            save_path=f"{args.output_dir}/input_gradient_correlates.png",
        )
        for i in range(config.foundation_model_config.num_correlates):
            input_weight_analyzer.visualize_inputs(
                attention_weights,
                correlate_idx=i,
                visualize_correlates=True,
                save_path=f"{args.output_dir}/input_gradient_correlate_{i}.png",
            )
    else:
        # Visualize attention for each layer and head
        for layer_idx in range(len(attention_weights[0])):
            for head_idx in range(
                config.foundation_model_config.decoder_num_heads
            ):
                attention_analyzer.visualize_attention(
                    attention_weights,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    save_path=f"{args.output_dir}/layer_{layer_idx}_head_{head_idx}.png",
                )
    pattern_per_batch.append((batch, patterns))

    return pattern_per_batch


def main():
    """
    Example usage of the AttentionAnalyzer class.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze attention patterns in the foundation model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attention_analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="attention",
        help="Type of weight to analyze: attention, input_gradient, counterfactual_all_attention, counterfactual_all_mask, counterfactual_correlate",
    )
    # TODO: as the number of args increases, we should use a config file
    SPLIT = "test"
    NUM_BATCHES = 10

    args = parser.parse_args()

    # load config and model
    raise NotImplementedError(
        "FoundationForecastExperiment has been removed. This script needs to be updated to use an alternative experiment class."
    )
    # experiment = FoundationForecastExperiment(args.config_path)
    # experiment.configuration.dataset_config.batch_size = 1
    # experiment.configuration.training_config.num_ar_batches = 1
    # forecast_model = experiment._setup_inference(
    #     args.model_path
    # )  # TODO: fix this so that setup inference is not hidden?
    # model = forecast_model.decoder_model
    # config = experiment.configuration

    model_analyzer = SynthefyFoundationModelAnalyzer(model)
    attention_analyzer = AttentionAnalyzer(model_analyzer)
    input_weight_analyzer = InputWeightAnalyzer(model_analyzer)
    load_and_visualize_weights(
        config,
        experiment,
        attention_analyzer,
        input_weight_analyzer,
        args,
        split=SPLIT,
        num_batches=NUM_BATCHES,
        weight_type=args.weight_type,
    )


if __name__ == "__main__":
    main()

    # uv run src/synthefy_pkg/model/utils/run_weight_analysis.py --model_path /NEEDS V3E MODEL.ckpt --config_path /mnt/synthefy/synthefy_data/foundation_model_configs/sweep_configs/v3_relation_5k_types/config_dataset_name-fmv2_5k_202106_pretrain_use_relation_shards-False_trial0.yaml --output_dir /mnt/synthefy/sytnefy_data/attention_analysis_outputs/test_analysis_relation_5k --weight_type attention
    # uv run src/synthefy_pkg/model/utils/attention_analysis.py --config_path /workspace/data/synthefy_data/foundation_model_configs/sweep_configs/v3_relation_5k_types/config_dataset_name-fmv2_5k_202106_pretrain_use_relation_shards-False_trial0.yaml --output_dir /workspace/data/synthefy_data/attention_analysis_outputs/test_analysis_relation_5k --weight_type counterfactual_correlate
