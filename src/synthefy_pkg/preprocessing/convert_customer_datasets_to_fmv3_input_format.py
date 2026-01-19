import logging
import os

import numpy as np
from einops import rearrange
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def convert_customer_datasets_to_fmv3_input_format(
    target_timeseries_dir: str,
    metadata_timeseries_dir: str,
    dataset_name: str,
    output_dir: str,
):
    logger.info("Converting customer datasets to FTV3 input format")
    logger.info(f"Target timeseries directory: {target_timeseries_dir}")
    logger.info(f"Metadata timeseries directory: {metadata_timeseries_dir}")
    logger.info(f"Output directory: {output_dir}")

    num_target_timeseries = len(os.listdir(target_timeseries_dir))
    num_metadata_timeseries = len(os.listdir(metadata_timeseries_dir))
    num_groups = int(num_metadata_timeseries / num_target_timeseries)

    target_timeseries_list = []
    for target_ts_idx in tqdm(
        range(num_target_timeseries),
        total=num_target_timeseries,
        desc="Loading target timeseries",
    ):
        target_ts_npy_array = np.load(
            os.path.join(
                target_timeseries_dir,
                f"{dataset_name}_{target_ts_idx}",
                "preprocessed_array.npy",
            )
        )
        target_timeseries_list.append(target_ts_npy_array)

    target_timeseries = np.stack(target_timeseries_list, axis=0)

    # expand target_timeseries to add metadata columns
    target_timeseries = np.expand_dims(target_timeseries, axis=1)

    metadata_timeseries_list = []
    for metadata_ts_idx in tqdm(
        range(num_metadata_timeseries),
        total=num_metadata_timeseries,
        desc="Loading metadata timeseries",
    ):
        metadata_ts_npy_array = np.load(
            os.path.join(
                metadata_timeseries_dir,
                f"{dataset_name}_metadata_{metadata_ts_idx}",
                "preprocessed_array.npy",
            )
        )
        metadata_timeseries_list.append(metadata_ts_npy_array)

    metadata_timeseries = np.stack(metadata_timeseries_list, axis=0)
    metadata_timeseries = rearrange(
        metadata_timeseries,
        "(num_target_timeseries num_groups) num_windows num_features -> num_target_timeseries num_groups num_windows num_features",
        num_target_timeseries=num_target_timeseries,
        num_groups=num_groups,
    )

    # concatenate target_timeseries and metadata_timeseries
    fmv3_input = np.concatenate(
        [target_timeseries, metadata_timeseries], axis=1
    )
    fmv3_input = rearrange(fmv3_input, "n c w f -> n w c f")
    fmv3_input = rearrange(fmv3_input, "n w c f -> (n w) c f")

    logger.info(f"FMV3 input shape: {fmv3_input.shape}")

    os.makedirs(output_dir, exist_ok=True)
    # save the array
    np.save(os.path.join(output_dir, "timeseries.npy"), fmv3_input)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_timeseries_dir",
        type=str,
        help="target timeseries directory",
        required=True,
    )
    parser.add_argument(
        "--metadata_timeseries_dir",
        type=str,
        help="metadata timeseries directory",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset name",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="output directory", required=True
    )

    args = parser.parse_args()

    convert_customer_datasets_to_fmv3_input_format(
        target_timeseries_dir=args.target_timeseries_dir,
        metadata_timeseries_dir=args.metadata_timeseries_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
    )
