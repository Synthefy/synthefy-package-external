import argparse
import os

from synthefy_pkg.scripts.compare_models import compare_models, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=False, default=None)
    parser.add_argument(
        "--others",
        type=str,
        nargs="+",
        required=True,
        help="Paths to other config files or just names of other models to compare against.",
    )
    parser.add_argument("--plot_rows", type=int, default=3)
    parser.add_argument("--plot_cols", type=int, default=2)
    parser.add_argument(
        "--height", type=int, default=18, help="Height of the output plots"
    )
    parser.add_argument(
        "--width", type=int, default=12, help="Width of the output plots"
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--splits", type=str, nargs="+", default=("test",), help="Splits to compare"
    )
    args = parser.parse_args()

    main_config = load_config(args.config_path)
    if args.model_name:
        main_config["denoiser_config"]["denoiser_name"] = args.model_name.capitalize()

    configs = [main_config]
    for other in args.others:
        if os.path.isfile(other):
            other_config = load_config(other)
            configs.append(other_config)
        else:
            other_config = main_config.copy()
            other_config["denoiser_config"]["denoiser_name"] = other.capitalize()
            other_config["execution_config"]["run_name"] += f"_{other.lower()}"
            configs.append(other_config)

    compare_models(
        configs=configs,
        top_k=args.top_k,
        plot_rows=args.plot_rows,
        plot_cols=args.plot_cols,
        figsize=(args.height, args.width),
        splits=args.splits,
    )
"""
python examples/compare_forecasting_models.py \
    --config_path examples/configs/forecast_configs/config_air_quality.yaml \
    --others chronos prophet STL \
"""
