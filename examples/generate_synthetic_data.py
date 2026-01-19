import argparse
import json
from typing import Optional

import pandas as pd
import yaml

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.experiments.forecast_experiment import ForecastExperiment
# from synthefy_pkg.experiments.foundation_forecast_experiment import (
#     FoundationForecastExperiment,
# )
from synthefy_pkg.experiments.synthesis_experiment import SynthesisExperiment
from synthefy_pkg.postprocessing.postprocess import Postprocessor


def main(
    config_filepath: str,
    model_checkpoint_path: str,
    metadata_path: Optional[str] = None,
    preprocess_config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    splits: list[str] = ["test"],
    plot_fourier: bool = False,
    downsample_factor: int | None = None,
    run_postprocessing: bool = True,
    output_filename_prefix: Optional[str] = None,
    synthesis_task: str = "synthesis",
    forecast_length: int = 32,
):
    with open(config_filepath, "r") as config_file:
        config = yaml.safe_load(config_file)

    if config.get("task", "").lower() == "forecast":
        # Cast to a Configuration object to get default values
        configuration = Configuration(config_filepath=config_filepath)

        if (
            configuration.dataset_config.dataloader_name
            == "FoundationModelDataLoader"
        ):
            raise NotImplementedError(
                "FoundationModelDataLoader is not supported for synthetic data generation"
            )
        elif (
            configuration.dataset_config.dataloader_name
            == "ShardedDataloaderV1"
            or configuration.dataset_config.dataloader_name
            == "V3ShardedDataloader"
            or configuration.dataset_config.dataloader_name
            == "OTFSyntheticDataloader"
        ):
            raise NotImplementedError(
                "FoundationForecastExperiment has been removed. Please use ForecastExperiment instead."
            )
            # experiment = FoundationForecastExperiment(config_filepath)
        else:
            experiment = ForecastExperiment(config_filepath)

    else:
        experiment = SynthesisExperiment(config_filepath, synthesis_task=synthesis_task, forecast_length=forecast_length)

    if metadata_path is None:
        # just do the train/test/val and postprocessing.
        is_main_process = experiment.generate_synthetic_data(
            model_checkpoint_path=model_checkpoint_path,
            splits=splits,
            output_dir=output_dir,
        )
        # Only run postprocessing on the main process (rank 0) - since multi gpu
        if is_main_process and run_postprocessing:
            postprocessor = Postprocessor(
                config_filepath=config_filepath,
                splits=splits,
                downsample_factor=downsample_factor,
            )
            postprocessor.postprocess(plot_fourier=plot_fourier)
    else:
        if not isinstance(experiment, SynthesisExperiment):
            raise ValueError(
                "Metadata json path is only supported for synthesis task"
            )
        # create synthetic data for all the windows in the metadata json
        if metadata_path.endswith(".json"):
            metadata_for_synthesis = pd.DataFrame(
                json.load(open(metadata_path))
            )
        elif metadata_path.endswith(".parquet"):
            metadata_for_synthesis = pd.read_parquet(metadata_path)
        else:
            raise ValueError(f"Unsupported metadata file type: {metadata_path}")

        experiment.generate_synthetic_data(
            model_checkpoint_path=model_checkpoint_path,
            metadata_for_synthesis=metadata_for_synthesis,
            preprocess_config_path=preprocess_config_path,
            output_dir=output_dir,
            output_filename_prefix=output_filename_prefix,
        )


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--config", type=str, default="config.yaml")
    argparse.add_argument("--model_checkpoint_path", type=str, required=True)
    argparse.add_argument("--metadata_path", type=str, default=None)
    argparse.add_argument("--preprocess_config_path", type=str, default=None)
    argparse.add_argument("--output_dir", type=str, default=None)
    argparse.add_argument("--output_filename_prefix", type=str, default=None)
    argparse.add_argument("--splits", type=str, nargs="+", default=["test"])
    argparse.add_argument(
        "--plot_fourier", type=bool, required=False, default=False
    )
    argparse.add_argument(
        "--no_run_postprocessing", action="store_true", default=False
    )
    argparse.add_argument(
        "--downsample_factor", type=int, required=False, default=None
    )
    argparse.add_argument(
        "--synthesis_task", type=str, required=False, default="synthesis", choices=["forecast", "synthesis"]
    )
    argparse.add_argument(
        "--forecast_length", type=int, required=False, default=0
    )

    args = argparse.parse_args()

    main(
        config_filepath=args.config,
        model_checkpoint_path=args.model_checkpoint_path,
        metadata_path=args.metadata_path,
        preprocess_config_path=args.preprocess_config_path,
        output_dir=args.output_dir,
        splits=args.splits,
        plot_fourier=args.plot_fourier,
        downsample_factor=args.downsample_factor,
        run_postprocessing=not args.no_run_postprocessing,
        output_filename_prefix=args.output_filename_prefix,
        synthesis_task=args.synthesis_task,
        forecast_length=args.forecast_length,
    )

    # uv run examples/generate_synthetic_data.py --config /mnt/workspace1/synthefy_data/foundation_model_configs/sweep_configs/synthetic_train_v3e/config_prior_config_path-src_synthefy_pkg_prior_config_synthetic_configs_config_medium_series.yaml_is_regression-True_trial0.yaml --model_checkpoint_path /mnt/workspace1/data/synthefy_data/training_logs/synthetic_tabular_v3e/MLFlow_Test/synthefy_foundation_model_v3_forecasting_prior_config_path-src_synthefy_pkg_prior_config_synthetic_configs_config_medium_series.yaml_is_regression-True_trial0/checkpoints/checkpoint_step_5000_val_loss_1.3436.ckpt
