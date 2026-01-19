import os

import pandas as pd


def main():
    base_dir = "/home/raimi2/kaggle_raw_data/motionsense"
    actions_dir = os.path.join(
        base_dir, "A_DeviceMotion_data/A_DeviceMotion_data/"
    )

    subject_df = pd.read_csv(os.path.join(base_dir, "data_subjects_info.csv"))
    subject_df = subject_df.rename(columns={"code": "subject_id"})
    subject_df["subject_id"] = subject_df["subject_id"].astype(str)

    df_list = []
    for action in os.listdir(actions_dir):
        action_dir = os.path.join(actions_dir, action)
        for file in os.listdir(action_dir):
            if file.endswith(".csv"):
                action = action_dir.split("/")[-1].split("_")[0]
                subject_id = file.split("_")[1].replace(".csv", "")
                print(f"Reading {file} for subject {subject_id}")
                df = pd.read_csv(os.path.join(action_dir, file))
                df["action"] = action
                df["subject_id"] = subject_id
                df_list.append(df)

    df = pd.concat(df_list)
    df = df.drop(columns=["Unnamed: 0"])

    df = df.merge(subject_df, on="subject_id", how="left")

    os.makedirs("/home/raimi2/data/motionsense", exist_ok=True)
    df.to_parquet(
        "/home/raimi2/data/motionsense/motionsense.parquet", index=False
    )


if __name__ == "__main__":
    main()
