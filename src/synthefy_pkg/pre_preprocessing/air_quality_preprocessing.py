import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

COMPILE = False


def main():
    SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
    assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

    SYNTHEFY_DATASETS_BASE = Path(os.getenv("SYNTHEFY_DATASETS_BASE"))
    BASE_DATA_PATH = SYNTHEFY_DATASETS_BASE / "air_quality/PRSA_Data_20130301-20170228"
    OUTPUT_DATA_PATH = SYNTHEFY_DATASETS_BASE / "air_quality/"

    # Get all the csv files in the base data path
    csv_files = list(BASE_DATA_PATH.glob("*.csv"))

    data_to_combine = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        data_to_combine.append(df)

    df = pd.concat(data_to_combine)

    df["date_time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["year", "month", "day", "hour"])
    df = df.drop(columns=["No"])

    # convert object to str
    df["wd"] = df["wd"].astype(str)
    df["station"] = df["station"].astype(str)

    df.to_parquet(OUTPUT_DATA_PATH / "air_quality_data.parquet")

    print(df.head())


if __name__ == "__main__":
    main()
