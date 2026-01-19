import argparse
import json
import os
from typing import Optional

import numpy as np
import yaml
from loguru import logger

from synthefy_pkg.experiments.forecast_experiment import ForecastExperiment
from synthefy_pkg.experiments.synthesis_experiment import SynthesisExperiment
from synthefy_pkg.postprocessing.postprocess import Postprocessor


def main(
    config_filepath: str,
    model_checkpoint_path: str,
    metadata_for_synthesis_path: str,
    preprocess_config_path: str,
    num_windows_to_generate: int,
    save_dir: Optional[str] = None,
):
    with open(config_filepath, "r") as config_file:
        config = yaml.safe_load(config_file)

    if config.get("task", "").lower() == "forecast":
        raise ValueError("Forecast task is not supported for long term data generation")
        # experiment = ForecastExperiment(config_filepath)
    else:
        experiment = SynthesisExperiment(config_filepath)

    metadata_for_synthesis = json.load(open(metadata_for_synthesis_path))

    long_term_data = experiment.generate_long_term_synthetic_data(
        model_checkpoint_path=model_checkpoint_path,
        metadata_for_synthesis=metadata_for_synthesis,
        preprocess_config_path=preprocess_config_path,
        num_windows_to_generate=num_windows_to_generate,
    )
    logger.info(f"Finished generating long term data. Shape: {long_term_data.shape}")

    if save_dir is None:
        save_dir = "/tmp/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    logger.info(
        f"Saving long term data to {os.path.join(save_dir, 'long_term_data.npy')}"
    )
    # save the data
    np.save(os.path.join(save_dir, "long_term_data.npy"), long_term_data)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--config", type=str, default="config.yaml")
    argparse.add_argument("--model_checkpoint_path", type=str, required=True)
    argparse.add_argument("--metadata_for_synthesis_path", type=str, required=True)
    argparse.add_argument("--preprocess_config_path", type=str, required=True)
    argparse.add_argument(
        "--num_windows_to_generate", type=int, required=False, default=-1
    )
    argparse.add_argument("--save_dir", type=str, required=False)
    args = argparse.parse_args()

    main(
        config_filepath=args.config,
        model_checkpoint_path=args.model_checkpoint_path,
        metadata_for_synthesis_path=args.metadata_for_synthesis_path,
        preprocess_config_path=args.preprocess_config_path,
        num_windows_to_generate=args.num_windows_to_generate,
        save_dir=args.save_dir,
    )

"""
example command for synthesis (only works for synthesis):
python examples/generate_long_term_data.py \
    --config examples/configs/synthesis_configs/config_ppg_synthesis.yaml \
    --model_checkpoint_path /Users/raimi/synthefy_data/synthefy_package_datasets/training_logs/ppg/Time_Series_Diffusion_Training/long_term/checkpoints/best_model.ckpt \
    --metadata_for_synthesis_path /Users/raimi/synthefy_data/synthefy_package_datasets/ppg/subject1.json \
    --preprocess_config_path examples/configs/preprocessing_configs/config_ppg_preprocessing.json \
    --num_windows 2

"""
