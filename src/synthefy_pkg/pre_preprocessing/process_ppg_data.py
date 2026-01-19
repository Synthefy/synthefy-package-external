import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy import signal


def main():
    dfs = []
    d_labels = {}
    for user in [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
        "S11",
        "S12",
        "S13",
        "S14",
        "S15",
    ]:
        print(f"\nProcessing {user}...\n\n")
        dataset_path = "/Users/raimi/Downloads/ppg+dalia/PPG_FieldStudy"

        with open(os.path.join(dataset_path, user, f"{user}.pkl"), "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")

        # pickle file is all that's needed...

        timeseries_wrist = pickle_data["signal"]["wrist"]["BVP"]
        print(f"{timeseries_wrist.shape=}")

        label_timeseries = pickle_data["label"]
        print(f"{label_timeseries.shape=}")

        acc_data_wrist = pickle_data["signal"]["wrist"]["ACC"]
        print(f"{acc_data_wrist.shape=}")

        eda_data_wrist = pickle_data["signal"]["wrist"]["EDA"]
        print(f"{eda_data_wrist.shape=}")

        temp_data_wrist = pickle_data["signal"]["wrist"]["TEMP"]
        print(f"{temp_data_wrist.shape=}")

        acc_data_chest = pickle_data["signal"]["chest"]["ACC"]
        print(f"{acc_data_chest.shape=}")

        ecg_data = pickle_data["signal"]["chest"]["ECG"]
        print(f"{ecg_data.shape=}")

        emg_data = pickle_data["signal"]["chest"]["EMG"]
        print(f"{emg_data.shape=}")

        eda_data_chest = pickle_data["signal"]["chest"]["EDA"]
        print(f"{eda_data_chest.shape=}")

        temp_data = pickle_data["signal"]["chest"]["Temp"]
        print(f"{temp_data.shape=}")

        resp_data_chest = pickle_data["signal"]["chest"]["Resp"]
        print(f"{resp_data_chest.shape=}")

        rpeaks = pickle_data["rpeaks"]
        print(f"{rpeaks.shape=}")

        activity = pickle_data["activity"]
        print(f"{activity.shape=}")

        group_labels = pickle_data["questionnaire"]
        group_labels.update({"subject": user})
        print(group_labels)

        def downsample_signal(data, original_rate=700, target_rate=64):
            """Downsample a signal from original_rate to target_rate Hz."""

            # For multi-dimensional data, process each column separately
            if len(data.shape) > 1:
                return np.array(
                    [
                        downsample_signal(
                            data[:, i], original_rate, target_rate
                        )
                        for i in range(data.shape[1])
                    ]
                ).T

            # For very short signals, use simple resampling
            if len(data) < 30:  # arbitrary threshold
                expected_length = int(len(data) * (target_rate / original_rate))
                return signal.resample(data, expected_length)

            # For longer signals, use filtered decimation
            nyq = original_rate / 2
            cutoff = target_rate / 2
            b, a = signal.butter(4, cutoff / nyq, btype="low")  # pyright: ignore

            # Use smaller padlen for filtfilt
            padlen = min(len(data) - 1, 3 * max(len(a), len(b)))
            filtered = signal.filtfilt(b, a, data, padlen=padlen)

            # Ensure output length matches expected length
            expected_length = int(len(data) * (target_rate / original_rate))
            return signal.resample(filtered, expected_length)

        # upsample to match time series length
        acc_data_wrist_upsampled = acc_data_wrist.repeat(2, axis=0)
        activity_upsampled = activity.repeat(16, axis=0)
        eda_data_wrist_upsampled = eda_data_wrist.repeat(16, axis=0)
        temp_data_wrist_upsampled = temp_data_wrist.repeat(16, axis=0)

        downsampled_acc_data_chest = downsample_signal(acc_data_chest, 700, 64)
        downsampled_ecg_data = downsample_signal(ecg_data, 700, 64)
        downsampled_emg_data = downsample_signal(emg_data, 700, 64)
        downsampled_eda_data_chest = downsample_signal(eda_data_chest, 700, 64)
        downsampled_temp_data = downsample_signal(temp_data, 700, 64)
        downsampled_resp_data_chest = downsample_signal(
            resp_data_chest, 700, 64
        )

        d = {
            "BVP": timeseries_wrist,
            "activity": activity_upsampled,
            "ACC_wrist": acc_data_wrist_upsampled,
            "EDA_wrist": eda_data_wrist_upsampled,
            "TEMP_wrist": temp_data_wrist_upsampled,
            "ACC_chest": downsampled_acc_data_chest,
            "ECG": downsampled_ecg_data,
            "EMG": downsampled_emg_data,
            "EDA_chest": downsampled_eda_data_chest,
            "TEMP_chest": downsampled_temp_data,
            "RESP_chest": downsampled_resp_data_chest,
        }

        # Create a new dictionary for the flattened data
        flattened_dict = {}

        # Iterate over a copy of the original dictionary's items
        for key, value in d.copy().items():
            if len(value.shape) > 1 and value.shape[1] > 1:
                # Add each column as a separate entry
                for i in range(value.shape[1]):
                    flattened_dict[f"{key}_idx_{i}"] = value[:, i]
            else:
                # Keep single-column arrays as is
                flattened_dict[key] = (
                    value.reshape(-1) if len(value.shape) > 1 else value
                )

        # Replace the original dictionary
        d = flattened_dict

        # Print the results
        # assert the lenghts are all the same
        for key, value in d.items():
            assert len(value) == len(d["BVP"]), (
                f"{key} has a different length than BVP"
            )

        df_temp = pd.DataFrame(d)
        # add group labels to this
        for key, value in group_labels.items():
            df_temp[key] = value

        # save the labels for each subject
        d_labels[user] = list(label_timeseries)

        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(os.path.join(dataset_path, "processed_ppg_data.parquet"))

    # save the labels for each subject to json
    with open(
        os.path.join(dataset_path, "processed_ppg_labels.json"), "w"
    ) as f:
        json.dump(d_labels, f)

    # open the df and the labels and add the label to the df
    df = pd.read_parquet(
        os.path.join(dataset_path, "processed_ppg_data.parquet")
    )
    with open(
        os.path.join(dataset_path, "processed_ppg_labels.json"), "r"
    ) as f:
        d_labels = json.load(f)

    WINDOW_SIZE = 512
    STRIDE = 128

    dfs_with_heart_rate = []
    for subject in d_labels.keys():
        if subject not in df["subject"].unique():
            print(f"Subject {subject} not found in the dataframe")
            continue
        # Get the indices for this subject and create a copy
        df_subject = df[df["subject"] == subject].copy().reset_index(drop=True)
        df_subject["heart_rate"] = None
        num_windows = max(0, (len(df_subject) - WINDOW_SIZE) // STRIDE + 1)
        assert num_windows == len(d_labels[subject])
        # Apply labels using the original dataframe
        for window_idx in range(0, num_windows):
            start_idx = window_idx * STRIDE
            end_idx = start_idx + WINDOW_SIZE
            df_subject.loc[start_idx:end_idx, "heart_rate"] = d_labels[subject][
                window_idx
            ]
        dfs_with_heart_rate.append(df_subject)
    df_with_heart_rate = pd.concat(
        dfs_with_heart_rate, ignore_index=True
    ).ffill()
    df_with_heart_rate.to_parquet(
        os.path.join(dataset_path, "processed_ppg_data_with_heart_rate.parquet")
    )


if __name__ == "__main__":
    main()
