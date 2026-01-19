import os

import matplotlib.pyplot as plt
import pandas as pd


def main():

    try:
        df = pd.read_parquet("/Users/raimi/Downloads/vibration/vibration_data.parquet")
    except:
        df_list = []
        for i in range(5):
            dfd = pd.read_csv(f"/Users/raimi/Downloads/vibration/{i}D.csv")
            dfd["custom_split"] = "train"
            dfd["label"] = str(i)
            dfd["time_index"] = list(range(len(dfd)))
            dfe = pd.read_csv(f"/Users/raimi/Downloads/vibration/{i}E.csv")
            dfe["custom_split"] = "test"
            dfe["label"] = str(i)
            dfe["time_index"] = list(range(len(dfd), len(dfd) + len(dfe)))
            df_list.append(dfd)
            df_list.append(dfe)
            print(len(dfd), len(dfe))

        # Concatenate all DataFrames
        df = pd.concat(df_list)

    # Downsample by 16x
    df_downsampled = df.iloc[::16]

    # save downsampled data
    df_downsampled.to_parquet(
        "/Users/raimi/Downloads/vibration/vibration_data_downsampled.parquet"
    )

    # # Plotting Before Downsampling
    # def plot_sample_before_downsampling(data, label, save_path=None):
    #     plt.figure(figsize=(12, 6))
    #     sample = data[data["label"] == label].iloc[
    #         :1000
    #     ]  # Adjust the sample size as needed
    #     plt.plot(
    #         sample["time_index"],
    #         sample["Vibration_1"],
    #         label=f"Label {label} - Original",
    #     )
    #     plt.title(f"Before Downsampling - Label {label}")
    #     plt.xlabel("Time Index")
    #     plt.ylabel("Sensor Value")
    #     plt.legend()
    #     if save_path:
    #         plt.savefig(os.path.join(save_path, f"before_downsampling_label_{label}.png"))
    #     plt.show()

    # # Plotting After Downsampling
    # def plot_sample_after_downsampling(data, label, save_path=None):
    #     plt.figure(figsize=(12, 6))
    #     sample = data[data["label"] == label].iloc[
    #         :1000
    #     ]  # Adjust the sample size as needed
    #     plt.plot(
    #         sample["time_index"],
    #         sample["Vibration_1"],
    #         label=f"Label {label} - Downsampled",
    #         color="orange",
    #     )
    #     plt.title(f"After Downsampling - Label {label}")
    #     plt.xlabel("Time Index")
    #     plt.ylabel("Sensor Value")
    #     plt.legend()
    #     if save_path:
    #         plt.savefig(os.path.join(save_path, f"after_downsampling_label_{label}.png"))
    #     plt.show()

    # # Directory to save plots
    # plots_dir = "/Users/raimi/Downloads/vibration/plots"
    # os.makedirs(plots_dir, exist_ok=True)

    # # Specify which label to plot (e.g., label '0')
    # label_to_plot = "0"

    # # Plot Before and After Downsampling
    # plot_sample_before_downsampling(df, label_to_plot, save_path=plots_dir)
    # plot_sample_after_downsampling(df_downsampled, label_to_plot, save_path=plots_dir)


if __name__ == "__main__":
    main()
