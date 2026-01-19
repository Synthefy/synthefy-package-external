#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path


def construct_output_dir(
    source_dataset: Path, relational_dir: Path, clean: bool
) -> Path:
    # Get the most relevant parts of each path
    source_name = "_".join(
        source_dataset.name.split("_")[-2:]
    )  # Take first part before underscore
    rel_name = "_".join(
        relational_dir.name.split("_")[3:]
    )  # Take first part before underscore

    # Create a simpler combined name
    combined_name = f"relational_{source_name}_{rel_name}"
    combined_dir = source_dataset.parent / "relational_datasets" / combined_name
    if clean:
        if combined_dir.exists():
            shutil.rmtree(combined_dir)

    # Create the output path in the same parent directory as source_dataset
    return combined_dir


def main():
    parser = argparse.ArgumentParser(
        description="Combine datasets and relational data into a new structure"
    )
    parser.add_argument("--source_dataset", help="Source dataset directory")
    parser.add_argument(
        "--relational_dir", help="Directory containing relational matrices"
    )
    parser.add_argument(
        "--clean", help="Clean the output directory", action="store_true"
    )
    args = parser.parse_args()

    # Convert paths to Path objects for easier handling
    source_dataset = Path(args.source_dataset)
    relational_dir = Path(args.relational_dir)
    output_dir = construct_output_dir(
        source_dataset, relational_dir, args.clean
    )

    print(f"Output directory will be: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create relation_shards directory
    relation_shards_dir = output_dir / "pretrain" / "relation_shards"
    relation_shards_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy source dataset to output directory
        print(f"Copying source dataset from {source_dataset} to {output_dir}")
        shutil.copytree(source_dataset, output_dir, dirs_exist_ok=True)

        # Copy relational matrices
        print(
            f"Copying relational matrices from {relational_dir} to {output_dir}"
        )
        for file in relational_dir.glob("*"):
            if file.is_file():
                print(
                    f"Copying {file} to {os.path.join(output_dir, 'pretrain', 'relation_shards')}"
                )
                shutil.copy2(
                    file,
                    os.path.join(output_dir, "pretrain", "relation_shards"),
                )

        print("Operation completed successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Clean up in case of error
        if output_dir.exists():
            shutil.rmtree(output_dir)
        exit(1)


if __name__ == "__main__":
    main()
    # usage:
    # python3 combine_relational_dataset.py --source_dataset /path/to/source_dataset --relational_dir /path/to/relational_dir

    # python3 src/synthefy_pkg/scripts/combine_relational_dataset.py --source_dataset /mnt/local/synthefy_data/fmv2_filtered_5k_202106 --relational_dir /mnt/local/synthefy_data/relational_matrices/relation_5k_filtered_new_cos_overlap
    # python3 src/synthefy_pkg/scripts/combine_relational_dataset.py --source_dataset /home/data/foundation_model_data_all_univariate_filtered_5000_0_5000_ts_2021-06-01/ --relational_dir /home/data/relational_matrices/relation_5k_filtered_granger_overlap/
    # python3 src/synthefy_pkg/scripts/combine_relational_dataset.py --source_dataset /mnt/local/synthefy_data/fmv2_filtered_5k_202106 --relational_dir /mnt/local/synthefy_data/relational_matrices/relation_5k_filtered_new_cross_corr_overlap
    # python3 src/synthefy_pkg/scripts/combine_relational_dataset.py --source_dataset /mnt/local/synthefy_data/fmv2_filtered_5k_202106 --relational_dir /mnt/local/synthefy_data/relational_matrices/relation_5k_filtered_new_fourier_overlap
    # python3 src/synthefy_pkg/scripts/combine_relational_dataset.py --source_dataset /mnt/local/synthefy_data/fmv2_filtered_5k_202106 --relational_dir /mnt/local/synthefy_data/relational_matrices/relation_5k_filtered_new_umap_text_overlap
