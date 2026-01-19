import argparse
import asyncio
import json
import os
from typing import Any, Dict

import pandas as pd
from loguru import logger

from synthefy_pkg.app.data_models import MetaDataVariation, WindowFilters
from synthefy_pkg.app.utils.api_utils import (
    apply_metadata_variations,
    filter_window_dataframe_by_window_filters,
)
from synthefy_pkg.data.window_and_dataframe_utils import (
    convert_windows_to_dataframe,
)

SYNTHEFY_DATASETS_BASE = str(os.getenv("SYNTHEFY_DATASETS_BASE", ""))
SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE", ""))

assert SYNTHEFY_DATASETS_BASE != "" and SYNTHEFY_PACKAGE_BASE != "", (
    "SYNTHEFY_DATASETS_BASE and SYNTHEFY_PACKAGE_BASE must be set"
)


def main(config: Dict[str, Any]):
    """
    structure of the config:
    {
        "dataset_name": "ppg_hr_samsung",
        "split_type": "train",
        "window_filters": WindowFilters,
        "metadata_variations": List[List[MetaDataVariation]],
        "output_path": "<your_output_path>.parquet"
    }
    """
    output_path = config["output_path"]
    if os.path.exists(output_path):
        raise ValueError(f"Output path {output_path} already exists")

    dataset_name = config["dataset_name"]
    split_type = config["split_type"]
    window_filters = WindowFilters.model_validate(config["window_filters"])
    metadata_variations = [
        [
            MetaDataVariation.model_validate(variation)
            for variation in variations
        ]
        for variations in config["metadata_variations"]
    ]

    # 2. Convert it to a dataframe
    df = asyncio.run(
        convert_windows_to_dataframe(
            dataset_name,
            split_type,
        )
    )

    # 3. Filter is by the WindowFilters
    df = asyncio.run(
        filter_window_dataframe_by_window_filters(df, window_filters)
    )

    # 4. Apply the given metadata variations to the dataframe
    preprocess_config_path = os.path.join(
        SYNTHEFY_PACKAGE_BASE,
        f"examples/configs/preprocessing_configs/config_{dataset_name}_preprocessing.json",
    )
    preprocess_config = json.load(open(preprocess_config_path))
    window_size = preprocess_config["window_size"]

    df = asyncio.run(
        apply_metadata_variations(
            df,
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=len(df) // window_size,
            window_size=window_size,
        )
    )

    # save to output path from the config
    df.to_parquet(output_path)
    logger.success(f"Saved to {output_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the .json config file",
    )
    args = args.parse_args()
    config = json.load(open(args.config))
    main(config)
