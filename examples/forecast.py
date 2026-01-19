import argparse
from typing import Optional

from synthefy_pkg.experiments.forecast_experiment import ForecastExperiment
from synthefy_pkg.utils.basic_utils import clean_lightning_logs


def main(
    config_filepath: str = "config.yaml",
    model_checkpoint_path: Optional[str] = None,
):
    if not model_checkpoint_path:
        clean_lightning_logs(config_filepath)

    experiment = ForecastExperiment(config_filepath)
    experiment.train(model_checkpoint_path=model_checkpoint_path)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--config",
        type=str,
        default="configs/config_twamp_one_month_forecasting.yaml",
    )
    argparse.add_argument("--model_checkpoint_path", type=str, default=None)
    args = argparse.parse_args()

    main(args.config, args.model_checkpoint_path)
