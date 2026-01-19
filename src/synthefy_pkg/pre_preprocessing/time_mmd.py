# Data from: https://github.com/AdityaLab/Time-MMD/

# Save the data to s3://synthefy-core/datasets/time_mmd/
# aws s3 cp ~/data/time_mmd/ s3://synthefy-core/datasets/time_mmd/ --recursive

import argparse
import os

import numpy as np
import pandas as pd
from loguru import logger

COMPILE = False

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE", "")


def preprocess_time_mmd_dataset(raw_dataset_name: str):
    # dataset_name = raw_dataset_name.capitalize()
    dataset_name = raw_dataset_name

    raw_dataset_path = os.path.join(SYNTHEFY_DATASETS_BASE, "Time-MMD")

    # Load the three CSV files
    data_path = os.path.join(
        raw_dataset_path, "numerical", dataset_name, f"{dataset_name}.csv"
    )
    report_data_path = os.path.join(
        raw_dataset_path, "textual", dataset_name, f"{dataset_name}_report.csv"
    )
    search_data_path = os.path.join(
        raw_dataset_path, "textual", dataset_name, f"{dataset_name}_search.csv"
    )

    # Read the CSV files into pandas DataFrames
    df = pd.read_csv(data_path)
    search_df = pd.read_csv(search_data_path)
    report_df = pd.read_csv(report_data_path)

    if dataset_name == "Health_AFR":
        # drop rows where the column OT is NaN
        df = df[df["OT"].notna()]

    # make a copy of df called original_df
    original_df = df.copy()

    if "Unnamed: 0" in search_df.columns:
        search_df.drop(columns=["Unnamed: 0"], inplace=True)
    if "Unnamed: 0" in report_df.columns:
        report_df.drop(columns=["Unnamed: 0"], inplace=True)

    logger.debug(f"{dataset_name} DataFrame shape: {df.shape}")
    logger.debug(f"{dataset_name} Search DataFrame shape: {search_df.shape}")
    logger.debug(f"{dataset_name} Report DataFrame shape: {report_df.shape}")

    def merge(val1, val2):
        return f"{val1} {val2}"

    # Updated strategy:
    # Add a new column to df called search_metadata and report_metadata
    # Iterate over the rows of df
    # Look at the date field in df
    # Then look at search_df for any rows that have a date that overlaps with the current row.
    # If there is a match, then merge the text columns from search_df and report_df into a new column.
    # Repeat this process for report_df, adding text into the report_metadata column.
    # Then merge the search_metadata and report_metadata columns into a new column called text_metadata in df by calling merge(search_metadata, report_metadata)
    # Drop the search_metadata and report_metadata columns.
    # Add new columns to df
    df["search_metadata"] = ""
    df["report_metadata"] = ""

    # Iterate over the rows of df
    for index, row in df.iterrows():
        # Get the date from the current row
        current_date = row["date"]

        # Find matching rows in search_df
        search_matches = search_df[
            (search_df["start_date"] <= current_date)
            & (search_df["end_date"] >= current_date)
        ]

        # Merge text columns from search_df into search_metadata
        search_metadata = " ".join(
            search_matches.apply(
                lambda x: " ".join([str(x[col]) for col in ["fact", "preds"]]),
                axis=1,
            )
        )

        # Find matching rows in report_df
        report_matches = report_df[
            (report_df["start_date"] <= current_date)
            & (report_df["end_date"] >= current_date)
        ]

        # Merge text columns from report_df into report_metadata
        report_metadata = " ".join(
            report_matches.apply(
                lambda x: " ".join([str(x[col]) for col in ["fact", "preds"]]),
                axis=1,
            )
        )

        # Assign the merged metadata to the respective columns in df
        df.at[index, "search_metadata"] = search_metadata
        df.at[index, "report_metadata"] = report_metadata

    # Merge search_metadata and report_metadata into text_metadata
    df["text_metadata"] = df.apply(
        lambda row: merge(row["search_metadata"], row["report_metadata"]),
        axis=1,
    )

    # Drop the search_metadata and report_metadata columns
    df.drop(columns=["search_metadata", "report_metadata"], inplace=True)

    # Check that the updated df is the same as the original, if text_metadata was dropped.
    # original_df = pd.read_csv(data_path)

    for col in df.drop(columns=["text_metadata"]).columns:
        if df[col].dtype == "float64":
            assert np.allclose(df[col], original_df[col], equal_nan=True)
        else:
            assert (df[col] == original_df[col]).all()

    logger.info(
        f"Number of rows with empty text_metadata: {df[df['text_metadata'].str.strip() == ''].shape[0]}"  # pyright: ignore
    )

    # Save the cleaned DataFrame to a new Parquet file
    output_path = os.path.join(
        SYNTHEFY_DATASETS_BASE,
        f"time_mmd_{dataset_name.lower()}",
        f"{dataset_name.lower()}_merged.parquet",
    )
    # mkdirs if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(f"Cleaned DataFrame saved to {output_path}")


if __name__ == "__main__":
    ALL_DATASETS = [
        "Agriculture",
        "Climate",
        "Economy",
        "Energy",
        "Environment",
        "Health_AFR",
        "Health_US",
        "Security",
        "SocialGood",
        "Traffic",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help=f"List of datasets to preprocess; options are {ALL_DATASETS}",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        if dataset.lower() not in [d.lower() for d in ALL_DATASETS]:
            raise ValueError(f"Dataset {dataset} not in {ALL_DATASETS}")
        logger.info(f"Preprocessing dataset {dataset}")
        preprocess_time_mmd_dataset(dataset)
