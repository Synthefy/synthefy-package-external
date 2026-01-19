import argparse
import json
import os
from pprint import pprint
from typing import Optional

from loguru import logger

from synthefy_pkg.preprocessing.data_summarizer import DataSummarizer


def main(
    output_dir: str,
    dataset_path: str,
    config_path: Optional[str] = None,
    group_cols: Optional[str] = None,
):
    """
    Main function to run the DataSummarizer.

    :param dataset_path: Path to the dataset file.
    :param config_path: Path to the config file.
    :param group_cols: Comma-separated list of column names for grouping.
    :param output_dir: Directory to save output files
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load config if provided
    config = None
    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)

    # Process group_cols into a list if provided
    processed_group_cols = group_cols.split(",") if group_cols else None

    # Initialize DataSummarizer
    summarizer = DataSummarizer(
        data_input=dataset_path,
        save_path=output_dir,
        config=config,
        group_cols=processed_group_cols,
        compute_all=True,  # Default to True for examples
        analysis_functions=None,  # Run all analyses by default
    )

    # Generate summaries
    logger.info("Generating metadata summary...")
    summarizer.summarize_metadata()

    logger.info("Generating time series summary...")
    summarizer.summarize_time_series()

    # Generate HTML reports
    # Get basename from dataset path instead of config
    dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]
    html_path = os.path.join(
        output_dir, f"{dataset_basename}_data_summary.html"
    )

    logger.info(f"Generating HTML report at {html_path}...")
    summarizer.generate_html_report(output_html=html_path)

    # Output summary dictionary
    summary = summarizer.get_summary_dict()
    logger.info("Summary generated successfully.")
    pprint(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DataSummarizer to generate data summaries and reports."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset file (e.g., data/opanga_combined_1000.parquet)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config file (e.g., configs/preprocessing_configs/config_opanga_preprocessing.json)",
    )
    parser.add_argument(
        "--group_cols",
        type=str,
        default=None,
        help="Comma-separated list of column names for grouping (e.g., 'App Name,User Type')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output files",
    )

    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        config_path=args.config,
        group_cols=args.group_cols,
    )
