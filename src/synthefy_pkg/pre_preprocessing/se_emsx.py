import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(42)

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


def main():
    root = os.path.join(SYNTHEFY_DATASETS_BASE, "se_EMSx")

    # total of 70 files
    files = []
    for i in range(1, 71):
        filepath = os.path.join(root, str(i) + ".csv")
        files.append(filepath)

    # read all the data files (.csv)
    dfs = []
    for file in files:
        print("Reading file: ", file)
        df = pd.read_csv(file, sep=";")
        dfs.append(df)

    # read the metadata file
    metadata = pd.read_csv(
        os.path.join(SYNTHEFY_DATASETS_BASE, "se_EMSx", "metadata.csv"), sep=","
    )

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

    print("Metadata")
    print(metadata)
    print("===" * 3)

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

    print("Reduced DFs: reduced_dfs[2]")
    print(reduced_dfs[2])
    print("===" * 3)

    # Change the dtype of the timestamp column to datetime
    for i in range(len(reduced_dfs)):
        reduced_dfs[i]["timestamp"] = pd.to_datetime(reduced_dfs[i]["timestamp"])

    # remove the +00:00 from the timestamp
    for i in range(len(reduced_dfs)):
        reduced_dfs[i]["timestamp"] = reduced_dfs[i]["timestamp"].dt.tz_localize(None)

    print("Validating shapes")
    for i in range(len(reduced_dfs)):
        print(reduced_dfs[i].shape)
    print("===" * 3)

    print("Checking for missing values")
    for i in range(len(reduced_dfs)):
        print(reduced_dfs[i].isnull().sum())
    print("===" * 3)

    # add train, test, val split as a column
    sorted_reduced_dfs = []
    for reduced_df in reduced_dfs:
        # make a new column named custom_split with the first 80 percent as train, next 10 percent as val and last 10 percent as test
        # sort the dataframe by timestamp
        sorted_reduced_df = reduced_df.sort_values(by="timestamp")

        # sorted_reduced_df = reduced_df

        train_start = 0
        train_end = int(0.8 * sorted_reduced_df.shape[0])
        val_start = train_end
        val_end = int(0.9 * sorted_reduced_df.shape[0])
        test_start = val_end
        test_end = sorted_reduced_df.shape[0]

        # create a new column named custom_split
        custom_split = (
            ["train"] * (train_end - train_start)
            + ["val"] * (val_end - train_end)
            + ["test"] * (test_end - val_end)
        )

        sorted_reduced_df["custom_split"] = custom_split
        sorted_reduced_dfs.append(sorted_reduced_df)

    # combine all the dataframes into one
    final_df = pd.concat(sorted_reduced_dfs)

    print("Final DF")
    print(final_df)
    print("===" * 3)

    # save the final_df as a parquet file
    save_dir = os.path.join(SYNTHEFY_DATASETS_BASE, "se_EMSx_sorted")
    os.makedirs(save_dir, exist_ok=True)
    final_df.to_parquet(os.path.join(save_dir, "EMSx.parquet"))

    # Plot the timestamp column
    index = 3
    train_reduced_df = sorted_reduced_dfs[index][
        sorted_reduced_dfs[index]["custom_split"] == "train"
    ]
    val_reduced_df = sorted_reduced_dfs[index][
        sorted_reduced_dfs[index]["custom_split"] == "val"
    ]
    test_reduced_df = sorted_reduced_dfs[index][
        sorted_reduced_dfs[index]["custom_split"] == "test"
    ]
    print(train_reduced_df.shape)
    print(val_reduced_df.shape)
    print(test_reduced_df.shape)
    plt.plot(np.arange(train_reduced_df.shape[0]), train_reduced_df["timestamp"])
    plt.plot(
        np.arange(
            train_reduced_df.shape[0],
            train_reduced_df.shape[0] + val_reduced_df.shape[0],
        ),
        val_reduced_df["timestamp"],
    )
    plt.plot(
        np.arange(
            train_reduced_df.shape[0] + val_reduced_df.shape[0],
            train_reduced_df.shape[0]
            + val_reduced_df.shape[0]
            + test_reduced_df.shape[0],
        ),
        test_reduced_df["timestamp"],
    )
    plt.savefig(os.path.join(save_dir, "EMSx_split.png"))


if __name__ == "__main__":
    main()
