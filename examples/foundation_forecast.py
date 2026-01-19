import argparse
from typing import Optional

# from synthefy_pkg.experiments.foundation_forecast_experiment import (
#     FoundationForecastExperiment,
# )
from synthefy_pkg.utils.basic_utils import clean_lightning_logs


def main(
    config_filepath: str = "config.yaml",
    model_checkpoint_path: Optional[str] = None,
):
    raise NotImplementedError(
        "FoundationForecastExperiment has been removed. This script needs to be updated to use an alternative experiment class."
    )
    # if not model_checkpoint_path:
    #     clean_lightning_logs(config_filepath)

    # experiment = FoundationForecastExperiment(config_filepath)

    # experiment.train(model_checkpoint_path=model_checkpoint_path)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--config",
        type=str,
        default="configs/foundation_model_configs/config_forecasting_foundation_model_v3e.yaml",
    )
    argparse.add_argument("--model_checkpoint_path", type=str, default=None)
    args = argparse.parse_args()

    main(args.config, args.model_checkpoint_path)
