import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


def main(lag: int = 1024):
    """
    1024 is chosen because 1024 is the forecast length for the UC1 dataset.
    """

    loc: str = os.path.join(
        SYNTHEFY_DATASETS_BASE, "se_UC1device/D65AP7_F330_F500_rul.parquet"
    )

    df = pd.read_parquet(loc)

    unique_obsindices = df["ObsIndex"].unique()

    print(f"Original Dataset")
    print(df)
    print("===" * 10)

    ruls = []
    part_dfs = []
    for uidx in unique_obsindices:
        part_df = df[df["ObsIndex"] == uidx]
        ruls.append(part_df["rul"].max())

        # append phase1_energie_darc_total, phase1_temps_darc_de_rebond, etc as additional columns but with a lag of 1
        cols = part_df.columns[2:11]

        for col in cols:
            part_df[col + "_lag1"] = part_df[col].shift(lag)
            # fill the first 256 rows with zeros
            part_df[col + "_lag1"].iloc[:lag] = 0

        part_dfs.append(part_df)

    print(f"{max(ruls)=}")
    # 15_000 was selected since the max_rul was about 11_000.
    MAX_RUL: int = 15000

    # normalize the rul values
    for i, part_df in enumerate(part_dfs):
        part_df["rul"] = part_df["rul"] / MAX_RUL
        part_dfs[i] = part_df
        part_df["time_since_inception"] = part_df["time_since_inception"] / MAX_RUL

    # merge the dataframes to add the `_lag1` columns
    new_df = pd.concat(part_dfs)

    print(f"Dataset after adding lag columns")
    print(new_df)
    print("===" * 10)

    save_dir = os.path.join(SYNTHEFY_DATASETS_BASE, f"se_UC1device_lag_{lag}")
    os.makedirs(save_dir, exist_ok=True)
    new_df.to_parquet(os.path.join(save_dir, f"D65AP7_F330_F500_rul_lag{lag}.parquet"))

    ## Extra Analysis

    # for each of these indices, find the rul value
    ruls = []
    references = []
    obs_indices = []
    for idx in df["ObsIndex"].unique():
        part_df = df[df["ObsIndex"] == idx]
        ruls.append(part_df["rul"].max())
        references.append(part_df["reference"].unique()[0])
        obs_indices.append(idx)

    # store a dictionary with the obs_index as the key and the rul and reference as the value
    obs_index_rul_reference_dict = {
        obs_indices[i]: (ruls[i], references[i]) for i in range(len(obs_indices))
    }
    print(f"{obs_index_rul_reference_dict=}")

    # store the dictionary
    save_dir = os.path.join(SYNTHEFY_DATASETS_BASE, f"se_UC1device_lag_{lag}")
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        os.path.join(save_dir, "obs_index_rul_reference_dict.npy"),
        obs_index_rul_reference_dict,
    )

    print(f"Dataset after adding lag columns")
    print(df)
    print("===" * 10)

    part_df = df[df["ObsIndex"] == 1]

    print(f"Partial Dataset; only ObsIndex==1")
    print(part_df)
    print("===" * 10)

    plt.plot(part_df["cycle"])
    plt.savefig(os.path.join(save_dir, "cycle.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", type=int, default=1024, help="Lag window size")
    args = parser.parse_args()
    main(args.lag)
