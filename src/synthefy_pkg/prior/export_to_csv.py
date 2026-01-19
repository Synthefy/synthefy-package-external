"""
Export prior datasets to CSV files for analysis and debugging.

This script generates batches of synthetic data from the prior distributions
and exports each dataset as a separate CSV file with proper column names.
This is useful for:
- Inspecting the generated data
- Debugging data generation issues
- Analyzing data characteristics
- Sharing data in a human-readable format

one line example usage:
python src/synthefy_pkg/prior/export_to_csv.py --config src/synthefy_pkg/prior/config/synthetic_configs/config_small_series.yaml --output_dir /tmp/exported_datasets --num_batches 2 --batch_size 4 --prefix synthetic_data

multi line example usage:
python src/synthefy_pkg/prior/export_to_csv.py \
  --config src/synthefy_pkg/prior/config/synthetic_configs/config_small_series.yaml \
  --output_dir /tmp/exported_datasets \
  --num_batches 32 \
  --batch_size 32 \
  --prefix synthetic_data
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.configs.tabicl_config import TabICLPriorConfig
from synthefy_pkg.prior.dataset import PriorDataset


def export_batch_to_csv(
    X: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    seq_lens: torch.Tensor,
    train_sizes: torch.Tensor,
    output_dir: Path,
    batch_idx: int = 0,
    prefix: str = "dataset",
    config: Optional[TabICLPriorConfig] = None,
) -> None:
    """
    Export a batch of datasets to individual CSV files.

    Parameters
    ----------
    X : torch.Tensor
        Features tensor of shape (batch_size, seq_len, max_features)
    y : torch.Tensor
        Labels tensor of shape (batch_size, seq_len)
    d : torch.Tensor
        Number of active features per dataset, shape (batch_size,)
    seq_lens : torch.Tensor
        Sequence length for each dataset, shape (batch_size,)
    train_sizes : torch.Tensor
        Train/test split position for each dataset, shape (batch_size,)
    output_dir : Path
        Directory to save CSV files
    batch_idx : int
        Batch index for file naming
    prefix : str
        Prefix for dataset file names
    """

    batch_size = X.shape[0]

    for i in range(batch_size):
        # Extract data for this dataset
        seq_len = int(seq_lens[i].item())
        num_features = int(d[i].item())
        train_size = int(train_sizes[i].item())

        # Get the actual data (only up to seq_len and num_features)
        dataset_X = X[i, :seq_len, :num_features].cpu().numpy()
        dataset_y = y[i, :seq_len].cpu().numpy()

        # Create column names
        # timestamp names for the first 5 columns if used in config:
        timestamp_cols = []
        if config is not None and config.add_synthetic_timestamps:
            if config.add_time_stamps_as_features:
                timestamp_cols = [
                    "year",
                    "hour_of_day_sin",
                    "hour_of_day_cos",
                    "day_of_week_sin",
                    "day_of_week_cos",
                    "day_of_month_sin",
                    "day_of_month_cos",
                    "day_of_year_sin",
                    "day_of_year_cos",
                    "week_of_year_sin",
                    "week_of_year_cos",
                    "month_of_year_sin",
                    "month_of_year_cos",
                    "running_index",
                ]
            else:
                timestamp_cols = ["minute", "hour", "day", "month", "year"]
        else:
            timestamp_cols = [f"feature_{j}" for j in range(5)]

        feature_cols = timestamp_cols + [
            f"feature_{j}" for j in range(len(timestamp_cols), num_features)
        ]

        # Create DataFrame
        df_data = {}

        # Add features
        for j, col_name in enumerate(feature_cols):
            df_data[col_name] = dataset_X[:, j]

        # Add target
        df_data["target"] = dataset_y

        # Add metadata columns

        # Create DataFrame
        df = pd.DataFrame(df_data)

        # Save to CSV
        filename = f"{prefix}_batch{batch_idx:03d}_dataset{i:03d}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)

        # Log info about this dataset
        logger.trace(
            f"Saved {filepath}: {seq_len} rows, {num_features} features, "
            f"train_size={train_size}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export prior datasets to CSV files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TabICL prior configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save CSV files",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="Number of batches to generate and export",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--prefix", type=str, default="dataset", help="Prefix for CSV filenames"
    )
    parser.add_argument(
        "--clean_dir",
        type=bool,
        default=False,
        help="Clean the output directory before exporting",
    )
    args = parser.parse_args()

    # Load configuration
    config = TabICLPriorConfig.from_yaml(args.config)

    # Override batch size if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_dir:
        # remove all .csv files in the output directory
        for file in output_dir.glob("*.csv"):
            file.unlink()
    # Create prior dataset
    prior_dataset = PriorDataset(config=config)

    logger.info(f"Using configuration: {config.prior_type}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generating {args.num_batches} batches")

    # Generate and export batches
    for batch_idx in tqdm(range(args.num_batches), desc="Exporting batches"):
        # Generate batch
        X, y, d, seq_lens, train_sizes = prior_dataset.get_batch()

        # Handle nested tensors (variable sequence lengths)
        if hasattr(X, "is_nested") and X.is_nested:
            logger.warning(
                "Nested tensors detected - extracting individual tensors"
            )
            # For nested tensors, we need to handle each dataset separately
            X_list = X.unbind()
            y_list = y.unbind()

            for i, (Xi, yi) in enumerate(zip(X_list, y_list)):
                # Create individual batch for each dataset
                X_single = Xi.unsqueeze(0)  # Add batch dimension
                y_single = yi.unsqueeze(0)
                d_single = d[i : i + 1]
                seq_lens_single = seq_lens[i : i + 1]
                train_sizes_single = train_sizes[i : i + 1]

                export_batch_to_csv(
                    X_single,
                    y_single,
                    d_single,
                    seq_lens_single,
                    train_sizes_single,
                    output_dir,
                    batch_idx * config.batch_size + i,
                    args.prefix,
                    config,
                )
        else:
            # Regular tensors
            export_batch_to_csv(
                X,
                y,
                d,
                seq_lens,
                train_sizes,
                output_dir,
                batch_idx,
                args.prefix,
                config
            )

    # Save configuration for reference
    config_path = output_dir / "config.yaml"
    config.save(str(config_path))
    logger.info(f"Saved configuration to {config_path}")

    logger.info(f"Export complete! CSV files saved to {output_dir}")


if __name__ == "__main__":
    main()
