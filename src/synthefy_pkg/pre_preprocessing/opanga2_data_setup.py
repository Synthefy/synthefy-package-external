import glob
import os
from pathlib import Path

import pandas as pd

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
SYNTHEFY_DATASETS_BASE = Path(SYNTHEFY_DATASETS_BASE) / Path("opanga2")
NUM_LABELS_TO_KEEP = 1000  # Should move to the config...

COMPILE = False


# Read all the CSV files in the directory and combine them into a single pandas dataframe and save it as a parquet file
def combine_csv_files(base_path):
    # Get a list of all the CSV files in the directory
    csv_files = glob.glob(os.path.join(base_path, "*.csv"))
    # Read all the CSV files into a single pandas dataframe
    df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    # Save the dataframe to a parquet file
    df.to_parquet(os.path.join(base_path, "opanga_combined.parquet"))

    return df


def main():
    df = combine_csv_files(SYNTHEFY_DATASETS_BASE)
    df = pd.read_parquet(os.path.join(SYNTHEFY_DATASETS_BASE, "opanga_combined.parquet"))

    df["Cell Identifier"] = df["Cell Identifier"].astype(str)
    df["Packet Processor Name"] = df["Packet Processor Name"].astype(str)
    # Drop protocol column
    df = df.drop(columns=["Protocol"])

    tmp = (
        df.groupby(["Cell Identifier", "Packet Processor Name"])
        .count()
        .sort_values("Burst Width (ms)")
    )
    to_keep = []
    for xi in tmp.iloc[-NUM_LABELS_TO_KEEP:].index:
        to_keep.append(xi[0])

    # Filter the dataframe to only include the top 1000
    df = df[df["Cell Identifier"].isin(to_keep)]

    df["Read Time (Timestamp UTC+0)"] = pd.to_datetime(
        df["Read Time (Timestamp UTC+0)"], unit="ms", utc=True
    )
    # Sort the dataframe by "Read Time (Timestamp UTC+0)"
    df = df.sort_values("Read Time (Timestamp UTC+0)")
    # Drop duplicates time stamps
    df = df.drop_duplicates(
        subset=[
            "Read Time (Timestamp UTC+0)",
            "Cell Identifier",
            "Packet Processor Name",
        ]
    )

    df.to_parquet(os.path.join(SYNTHEFY_DATASETS_BASE, "opanga_combined_1000.parquet"))
    print("Processed opanga dataset")
    print("Number of labels: ", len(df["Cell Identifier"].unique()))
    print("Data Shape: ", df.shape)


if __name__ == "__main__":
    main()
