#!/usr/bin/env python
"""
Example script to demonstrate how to use the dataset processing pipeline.
"""

import argparse
import logging
from pathlib import Path

from synthefy_pkg.preprocessing.fmv2_pre_preprocess import process_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the data processing pipeline with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the dataset processing pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Base output directory for processed data",
    )

    args = parser.parse_args()

    # Validate paths
    config_path = Path(args.config)
    output_path = Path(args.output)

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    if not output_path.exists():
        logger.info(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

    # Run the pipeline
    try:
        logger.info(f"Starting processing with config: {config_path}")
        process_dataset(config_path, output_path)
        logger.info("Processing completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
