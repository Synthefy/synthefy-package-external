import argparse
from typing import List, Tuple

from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.postprocessing.postprocess import Postprocessor
from synthefy_pkg.scripts.generate_synthetic_dataset_with_baseline import (
    load_config,
)
from synthefy_pkg.scripts.generate_synthetic_dataset_with_baseline import (
    run as sample_baseline,
)


def main(
    baseline: str,
    config_path: str,
    baseline_batch_size: int = -1,
    splits: Tuple[str] | List[str] = ("test",),
    plot_fourier: bool = False,
):
    config: DictConfig = load_config(config_path)

    config["execution_config"]["run_name"] = (
        config["execution_config"]["run_name"] + f"_{baseline}"
    )

    # Only chronos and prophet currently suppport probabilistic forecasts;
    # TimesFM and STL do not support probabilistic forecasting; run as if it's not enabled
    if baseline not in ("chronos", "prophet", "forecast_via_diffusion"):
        if hasattr(config, "denoiser_config"):
            config["denoiser_config"]["use_probabilistic_forecast"] = False
            logger.info(
                f"Setting denoiser_config.use_probabilistic_forecast to False for {baseline} baseline; does not support probabilistic forecasting"
            )

    if baseline_batch_size != -1:
        config["dataset_config"]["batch_size"] = baseline_batch_size
        logger.info(f"Overriding default batch size with {baseline_batch_size}")

    sample_baseline(config, baseline, splits=splits)
    postprocessor = Postprocessor(config=config, splits=list(splits))
    postprocessor.postprocess(plot_fourier=plot_fourier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=str, required=False, default="chronos"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=-1)
    parser.add_argument(
        "--plot_fourier", type=bool, required=False, default=False
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        help="List of split types to generate data for (e.g., train val test)",
        default=["test"],
    )
    args = parser.parse_args()
    main(
        baseline=args.baseline,
        config_path=args.config,
        baseline_batch_size=args.batch_size,
        splits=args.splits,
        plot_fourier=args.plot_fourier,
    )

"""
python3 examples/baseline_eval.py \
    --baseline chronos --config examples/configs/forecast_configs/config_twamp_one_month_forecasting.yaml \
    --batch_size 1024 \
    --splits test train val \
    --plot_fourier False

python3 examples/baseline_eval.py \
    --baseline chronos --config examples/configs/foundation_model_configs/config_forecasting_foundation_model_v2.yaml \
    --batch_size 1024 \
    --splits test train val \
    --plot_fourier False
"""
