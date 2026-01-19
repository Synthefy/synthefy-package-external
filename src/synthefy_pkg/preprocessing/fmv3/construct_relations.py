import argparse
import os
from typing import Any, Dict

from loguru import logger

from synthefy_pkg.preprocessing.fmv3.relational_sampling.base_relation_constructor import (
    BaseRelationConstructor,
)
from synthefy_pkg.utils.config_utils import load_yaml_config


def get_relation_constructor(config: Dict[str, Any]) -> BaseRelationConstructor:
    """
    Get the relation constructor based on the configuration.
    """
    if config["relation_constructor"] == "random":
        from synthefy_pkg.preprocessing.fmv3.relational_sampling.random_relation_constructor import (
            RandomRelationConstructor,
        )

        return RandomRelationConstructor(
            os.path.join(config["output_location"], config["split"]),
            config["num_classes"],
        )
    elif config["relation_constructor"] == "kmeans":
        from synthefy_pkg.preprocessing.fmv3.relational_sampling.k_means_relation_constructor import (
            KMeansRelationConstructor,
        )

        return KMeansRelationConstructor(
            os.path.join(config["output_location"], config["split"]),
            config["num_classes"],
        )
    elif config["relation_constructor"] == "haver_tree":
        from synthefy_pkg.preprocessing.fmv3.relational_sampling.haver_tree_relation_constructor import (
            HaverTreeRelationConstructor,
        )

        return HaverTreeRelationConstructor(config)
    else:
        raise ValueError(
            f"Invalid relation constructor: {config['relation_constructor']}"
        )


def main(config: Dict[str, Any]) -> None:
    """
    Main function to construct relations from the provided configuration.

    Args:
        config: Configuration dictionary loaded from YAML file
    """
    relation_constructor = get_relation_constructor(config)
    relation_constructor.run_with_validation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct relations for foundation model v3 preprocessing"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to preprocessing configuration YAML file",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_yaml_config(args.config)

    # Run main function with loaded config
    main(config)
