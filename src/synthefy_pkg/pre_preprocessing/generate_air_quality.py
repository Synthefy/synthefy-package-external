import os

import pandas as pd

COMPILE = False


def main():
    # join the path to the parquet with the SYNTHEFY_DATASETS_BASE/air_quality
    SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
    parquet_path = os.path.join(
        SYNTHEFY_DATASETS_BASE, "air_quality", "air_quality_data.parquet"
    )
    SYNTHEFY_PACKAGE_BASE = os.getenv("SYNTHEFY_PACKAGE_BASE")

    data = pd.read_parquet(parquet_path)

    group_cols = ["station"]
    data_sorted = data.sort_values(group_cols + ["date_time"])

    for station, group in data_sorted.groupby("station"):
        window_size = 96
        partial_data = group.iloc[:window_size]

        time_series_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

        # Fill Null with 0 for time series cols
        partial_data[time_series_cols] = partial_data[time_series_cols].fillna(0)

        partial_data.to_json(
            os.path.join(
                SYNTHEFY_PACKAGE_BASE,
                "examples/air_quality_data_stream.json",
            )
        )
        break

    # with open("air_quality_data_window.json", "w") as f:
    #     f.write(partial_data.to_json())

    # time_series_cols = ["Unnamed: 0", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

    # air_quality_df = pd.read_csv("air_quality_data_window.csv")

    # Now drop the time-series cols
    # partial_data = partial_data.drop(columns=time_series_cols)
    # partial_data.to_csv("air_quality_data_window_metadata_only.csv")

    # partial_data.to_json("/Users/shubhankaragarwal/synthefy-package/src/synthefy_pkg/app/tests/test_jsons/air_quality_data_window_metadata_only.json")


if __name__ == "__main__":
    main()
