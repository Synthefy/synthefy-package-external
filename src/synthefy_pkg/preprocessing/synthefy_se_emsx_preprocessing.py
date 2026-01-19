import os

import numpy as np
import pandas as pd

np.random.seed(42)

SYNTHEFY_DATASETS_BASE = os.environ.get(
    "SYNTHEFY_DATASETS_BASE", "/nas/timeseries/datasets"
)
ROOT = os.path.join(SYNTHEFY_DATASETS_BASE, "se_EMSx")


def main():
    # total of 70 files
    files = []
    for i in range(1, 71):
        filepath = os.path.join(ROOT, f"{str(i)}.csv")
        files.append(filepath)

    # read all files
    dfs = []
    for file in files:
        print("Reading file: ", file)
        df = pd.read_csv(file, sep=";")
        dfs.append(df)

    # read the metadata file
    metadata = pd.read_csv(os.path.join(ROOT, "metadata.csv"), sep=",")

    modified_dfs = []
    for i, df in enumerate(dfs):
        print("Adding metadata to file: ", files[i])
        # required columns max_load, capacity, power, charge_efficiency, discharge_efficiency

        per_df_metadata = metadata.loc[metadata["site_id"] == i + 1]
        print(per_df_metadata)

        # add the per_df_metadata to the df
        df["max_load"] = per_df_metadata["max_load"].values[0]
        df["capacity"] = per_df_metadata["capacity"].values[0]
        df["power"] = per_df_metadata["power"].values[0]
        df["charge_efficiency"] = per_df_metadata["charge_efficiency"].values[0]
        df["discharge_efficiency"] = per_df_metadata["discharge_efficiency"].values[0]

        modified_dfs.append(df)

    reduced_dfs = []
    for df in modified_dfs:
        reduced_df = df[
            [
                "timestamp",
                "site_id",
                "actual_consumption",
                "actual_pv",
                "max_load",
                "capacity",
                "power",
                "charge_efficiency",
                "discharge_efficiency",
            ]
        ]
        reduced_dfs.append(reduced_df)

    for i in range(len(reduced_dfs)):
        reduced_dfs[i]["timestamp"] = pd.to_datetime(reduced_dfs[i]["timestamp"])

    # remove th +00:00 from the timestamp

    for i in range(len(reduced_dfs)):
        reduced_dfs[i]["timestamp"] = reduced_dfs[i]["timestamp"].dt.tz_localize(None)

    for i in range(len(reduced_dfs)):
        print(reduced_dfs[i].shape)

    # check if there are any missing values
    for i in range(len(reduced_dfs)):
        print(reduced_dfs[i].isnull().sum())

    # add train, test, val split as a column
    for i in range(len(reduced_dfs)):
        # make a new column named custom_split with the first 80 percent as train, next 10 percent as val and last 10 percent as test
        reduced_dfs[i]["custom_split"] = "train"
        reduced_dfs[i].loc[
            int(0.8 * reduced_dfs[i].shape[0]) : int(0.9 * reduced_dfs[i].shape[0]),
            "custom_split",
        ] = "val"
        reduced_dfs[i].loc[
            int(0.9 * reduced_dfs[i].shape[0]) :, "custom_split"
        ] = "test"

    # combine all the dataframes into one
    final_df = pd.concat(reduced_dfs)

    # save the final_df as a parquet file
    final_df.to_parquet(os.path.join(ROOT, "EMSx.parquet.test"))


if __name__ == "__main__":
    main()
