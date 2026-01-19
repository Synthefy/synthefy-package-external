import argparse
import os
import pickle
import shutil
from typing import Any, Dict, List

import yaml
from loguru import logger

SYNTHEFY_PACKAGE_BASE = os.environ.get(
    "SYNTHEFY_PACKAGE_BASE", "/home/synthefy/synthefy-package"
)


def generate_mix_and_split_config(
    input_base_dir: str,
    output_dir: str,
    dataset_paths: List[str] | dict[str, List[str]],
    train_ratio: float = 0.8,  # because there are many one window datasets, TODO: mix and split should prioritize train before test
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    window_size: int = 256,
    stride: int = 1,
    batch_size: int = 128,
    num_workers: int = 16,
    shuffle: bool = False,
    version: str = "v2.5",
    shard_size: int = 25000,
    num_scalars: int = 3,
    metadata_types_to_use: List[str] = [
        "timestamp",
        "dataset_description",
        "text_description",
        "continuous",
        "retrieved_timeseries",
        # "time_varying_textual_metadata"
    ],
    timestamp_split: str | None = None,
    skip_blind: bool = False,
) -> Dict[str, Any]:
    """
    Generate a mix and split configuration for multiple datasets.

    Args:
        input_base_dir: Base directory containing the datasets
        output_dir: Directory where mixed dataset will be saved
        dataset_paths: List of dataset paths relative to input_base_dir
        windows_per_million: Number of windows per million in final mixed dataset
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
    """
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # Create config structure
    config = {
        "input_base_dir": input_base_dir,
        "output_dir": output_dir,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "selected_datasets": [],
        "window_size": window_size,
        "stride": stride,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "metadata_types_to_use": metadata_types_to_use,
        "version": version,
        "shard_size": shard_size,
        "num_scalars": num_scalars,
        "timestamp_split": timestamp_split,
    }

    # Assign datasets alternating between pretrain and blind
    if not isinstance(dataset_paths, dict):
        for i, dataset_path in enumerate(dataset_paths):
            dataset_config = {
                "dataset_name": dataset_path,
                "usage": "pretrain" if skip_blind or i % 50 != 0 else "blind",
            }
            config["selected_datasets"].append(dataset_config)
    else:
        print(dataset_paths)
        for i, dataset_path in enumerate(dataset_paths["blind"]):
            dataset_config = {
                "dataset_name": dataset_path,
                "usage": "blind",
            }
            config["selected_datasets"].append(dataset_config)
        for i, dataset_path in enumerate(dataset_paths["pretrain"]):
            dataset_config = {
                "dataset_name": dataset_path,
                "usage": "pretrain",
            }
            config["selected_datasets"].append(dataset_config)

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save the configuration to a YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {output_path}")


if __name__ == "__main__":
    # generate using 5000 datasets
    parser = argparse.ArgumentParser(
        description="Generate mix and split configuration"
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=10000,
        help="Number of datasets to generate (default: 10000)",
    )
    parser.add_argument(
        "--dataset-ids-path",
        type=str,
        default="",
        help="Path to the dataset ids file, a pickle containing a dictionary of pretrain, blind to dataset ids",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=25000,
        help="Approximae max number of windows per shard (default: 25000)",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset value (default: 0)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Use version 1, 2, 2.5 if specified",
    )
    parser.add_argument(
        "--timestamp-split",
        type=str,
        default="",
        help="Timestamp to split the data at, e.g. 2020-01-01",
    )
    parser.add_argument(
        "--save-bash", action="store_true", help="Save bash file if specified"
    )
    parser.add_argument(
        "--clean-dir",
        action="store_true",
        help="cleans the directory if specified",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Dataset name to use (default: all_univariate)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/data/enriched_datasets",
        help="Base directory containing the datasets (default: /home/data/enriched_datasets)",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="/home/data/",
        help="Output base directory for the mixed dataset (default: /home/data/)",
    )
    parser.add_argument(
        "--skip-blind",
        action="store_true",
        help="Skip blind datasets (save all as pretrain)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio (default: 0.1)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio (default: 0.1)",
    )

    args = parser.parse_args()
    num_datasets = args.num_datasets
    offset = args.offset
    version = args.version

    if args.dataset_name == "":
        if version == "v2.5":
            dataset_name = "all_univariate"
        elif version == "v2":
            dataset_name = "all_rand_clean"
        else:
            dataset_name = "all_geq_128_monthly_cpi_filtered"
    else:
        dataset_name = args.dataset_name

    dataset_ids = (
        range(offset, offset + num_datasets)
        if len(args.dataset_ids_path) == 0
        else pickle.load(open(args.dataset_ids_path, "rb"))
    )
    if len(args.dataset_ids_path) > 0:
        datasets = pickle.load(open(args.dataset_ids_path, "rb"))
        datasets["blind"] = [
            f"{dataset_name}/{dataset_name}_{i}" for i in (datasets["blind"])
        ]
        datasets["pretrain"] = [
            f"{dataset_name}/{dataset_name}_{i}" for i in (datasets["pretrain"])
        ]
    else:
        datasets = [f"{dataset_name}/{dataset_name}_{i}" for i in (dataset_ids)]

    if len(args.dataset_ids_path) > 0:
        dataset_id_val = os.path.basename(args.dataset_ids_path).split(".")[0]
    else:
        dataset_id_val = f"_{num_datasets}_{offset}_{offset + num_datasets}"

    timestamp_split_str = (
        f"_ts_{args.timestamp_split}" if args.timestamp_split else ""
    )
    config = generate_mix_and_split_config(
        input_base_dir=args.base_dir,
        output_dir=os.path.join(
            args.output_base_dir,
            f"foundation_model_data_{dataset_name}{dataset_id_val}{timestamp_split_str}/",
        ),
        dataset_paths=datasets,
        shard_size=args.shard_size,
        version=version,
        timestamp_split=args.timestamp_split,
        skip_blind=args.skip_blind,
        train_ratio=1 - args.test_ratio - args.val_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
    )

    if args.clean_dir and os.path.exists(
        os.path.join(
            args.output_base_dir,
            f"foundation_model_data_{dataset_name}{dataset_id_val}/",
        )
    ):
        shutil.rmtree(
            os.path.join(
                args.output_base_dir,
                f"foundation_model_data_{dataset_name}{dataset_id_val}/",
            )
        )

    os.makedirs(os.path.join(args.output_base_dir, "data_prep"), exist_ok=True)

    # Save to file
    output_path = os.path.join(
        args.output_base_dir,
        "data_prep",
        f"generated_mix_and_split_config_{dataset_name}.yaml",
    )
    save_config(config, output_path)

    # create bash file to run the below preprocess script
    if args.save_bash:
        logger.info(
            f"Saving bash file to {os.path.join(args.output_base_dir, 'data_prep', f'run_fm_preprocess_{dataset_name}.sh')}"
        )
        with open(
            os.path.join(
                args.output_base_dir,
                "data_prep",
                f"run_fm_preprocess_{dataset_name}.sh",
            ),
            "w",
        ) as f:
            flag = False
            for i in range(offset, offset + num_datasets):
                simul = " &" if i % 8 != 0 else ""
                if version == "v2.5":
                    script_path = os.path.join(
                        SYNTHEFY_PACKAGE_BASE,
                        "src",
                        "synthefy_pkg",
                        "preprocessing",
                        "fmv2_preprocess.py",
                    )
                elif version == "v2":
                    script_path = os.path.join(
                        SYNTHEFY_PACKAGE_BASE,
                        "src",
                        "synthefy_pkg",
                        "preprocessing",
                        "fmv2_preprocess.py",
                    )
                else:
                    script_path = os.path.join(
                        SYNTHEFY_PACKAGE_BASE,
                        "src",
                        "synthefy_pkg",
                        "preprocessing",
                        "fm_preprocess.py",
                    )
                f.write(
                    f"python3 {script_path} --data_dir {args.base_dir}/{dataset_name}/{dataset_name}_{i}/{simul}\n"
                )
                flag = True
            if flag:
                # Ensures script will not exit before spawned commands finish
                f.write("\nwait\n")
    # To run preprocessing stack:
    # Perform mix and split to get sharded data after preprocessing, also generate bash files for running preprocessing.
    # python src/synthefy_pkg/preprocessing/generate_mix_and_split_config.py --version v2.5 --num-datasets 50 --clean-dir --dataset-name {DATASET_NAME}
    # if dataset name is not specificed will default to real data

    # If specified --save-bash (DON'T RUN THIS, it will overwrite existing data)
    # bash run_fm_preprocess_{DATASET_NAME}.sh
    # run mix and split
    # python src/synthefy_pkg/preprocessing/fmv2_sharded_mix_and_split.py --config src/synthefy_pkg/preprocessing/generated_mix_and_split_config_{DATASET_NAME}.yaml
