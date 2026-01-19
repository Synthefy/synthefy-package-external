import argparse
from typing import Optional

from synthefy_pkg.experiments.cltsp_experiment import CLTSPExperiment
from synthefy_pkg.utils.basic_utils import clean_lightning_logs


def main(
    config_filepath: str,
):
    experiment = CLTSPExperiment(config_filepath)
    experiment.train()


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--config",
        type=str,
        required=True,
    )
    args = argparse.parse_args()

    main(args.config)
