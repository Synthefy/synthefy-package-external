import argparse
from typing import Optional

from synthefy_pkg.experiments.synthesis_experiment import SynthesisExperiment
from synthefy_pkg.utils.basic_utils import clean_lightning_logs


def main(
    config_filepath: str = "config.yaml",
    model_checkpoint_path: Optional[str] = None,
):
    if not model_checkpoint_path:
        clean_lightning_logs(config_filepath)

    experiment = SynthesisExperiment(config_filepath)
    experiment.train(model_checkpoint_path=model_checkpoint_path)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--config", type=str, default="config.yaml")
    argparse.add_argument("--model_checkpoint_path", type=str, default=None)
    args = argparse.parse_args()

    main(args.config, args.model_checkpoint_path)
