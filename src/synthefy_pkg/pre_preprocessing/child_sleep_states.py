import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.signal as signal

# Data from: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data


def main():
    df: pd.DataFrame = pd.read_parquet(
        "/home/raimi2/kaggle_raw_data/child-mind-institute-detect-sleep-states/train_series.parquet"
    )
    train_events: pd.DataFrame = pd.read_csv(
        "/home/raimi2/kaggle_raw_data/child-mind-institute-detect-sleep-states/train_events.csv"
    )

    # join based on series_id, timestamp, step?

    df_joined = df.merge(
        train_events, on=["series_id", "timestamp", "step"], how="left"
    )
    # Filter rows where event is not NA
    df_joined = df_joined[df_joined["event"].notna()]

    df_joined["event"] = pd.Series(df_joined["event"]).fillna("no_event")

    df_joined = pd.DataFrame(df_joined).sort_values(
        by=["series_id", "timestamp", "step"]
    )

    # bfill on "night" -> because we do not want to mix nights in each window
    if isinstance(df_joined["night"], pd.Series):
        df_joined["night"] = df_joined["night"].bfill()
    else:
        df_joined["night"] = pd.Series(df_joined["night"]).bfill()

    # should drop the data where nights are still na?? but just keep for now since it will be ffilled in preprocessing
    df_joined.to_parquet(
        "/home/raimi2/kaggle_raw_data/child-mind-institute-detect-sleep-states/event_data.parquet"
    )


if __name__ == "__main__":
    main()
