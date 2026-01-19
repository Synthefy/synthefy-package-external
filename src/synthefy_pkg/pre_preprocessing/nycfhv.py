# Data from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

# Get ~5 years of data: Jan 2019 - Sept 2024
# 9 + (4 * 12) = 57 months
# High Volume For-Hire Vehicle Trip Records (PARQUET)

# curl cmd to download the data into ~/data/nyc_fhv/
# for i in {01..12}; do curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2019-$i.parquet; done
# for i in {01..12}; do curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2020-$i.parquet; done
# for i in {01..12}; do curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2021-$i.parquet; done
# for i in {01..12}; do curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-$i.parquet; done
# for i in {01..12}; do curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-$i.parquet; done
# for i in {01..12}; do curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-$i.parquet; done

# NYC Gas Prices data: https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_Y35NY_DPGw.xls
# curl -O https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_Y35NY_DPGw.xls

# Temperature Data: https://mesonet.agron.iastate.edu/request/download.phtml?network=NY_ASOS
# curl -o weather_nyc.csv https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=NYC&data=tmpf&year1=2019&month1=1&day1=1&year2=2024&month2=11&day2=21&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3&report_type=4

# Save the data to s3://synthefy-core/datasets/nyc_fhv/
# aws s3 cp ~/data/nycfhv/ s3://synthefy-core/datasets/nyc_fhv/ --recursive

# Note: p2.2xlarge tested with N_WORKERS=2. This process is memory-bound.

import os
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

DEBUG = True
N_WORKERS = 2
assert N_WORKERS > 0

COMPILE = False

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


"""
nycfhv data schema:

Data columns (total 24 columns):
#   Column                Dtype
---  ------                -----
0   hvfhs_license_num     object
1   dispatching_base_num  object
2   originating_base_num  object
3   request_datetime      datetime64[ns]
4   on_scene_datetime     datetime64[ns]
5   pickup_datetime       datetime64[ns]
6   dropoff_datetime      datetime64[ns]
7   PULocationID          int64
8   DOLocationID          int64
9   trip_miles            float64
10  trip_time             int64
11  base_passenger_fare   float64
12  tolls                 float64
13  bcf                   float64
14  sales_tax             float64
15  congestion_surcharge  float64
16  airport_fee           float64
17  tips                  float64
18  driver_pay            float64
19  shared_request_flag   object
20  shared_match_flag     object
21  access_a_ride_flag    object
22  wav_request_flag      object
23  wav_match_flag        object
"""

"""
Int64Index: 744 entries, 0 to 743
Data columns (total 4 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   date        744 non-null    datetime64[ns]
 1   hour        744 non-null    int64
 2   ride_count  744 non-null    int64
 3   gas_price   744 non-null    float64
"""

"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 744 entries, 0 to 743
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   date        744 non-null    datetime64[ns]
 1   hour        744 non-null    int64
 2   ride_count  744 non-null    int64
 3   date_hour   744 non-null    datetime64[ns]
 4   gas_price   744 non-null    float64
 5   tmpf        744 non-null    float64
dtypes: datetime64[ns](2), float64(2), int64(2)
memory usage: 40.7 KB
"""


def get_weather_data():
    # Read the data from CSV
    weather_file = (
        Path(SYNTHEFY_DATASETS_BASE) / Path("nycfhv") / Path("weather_nyc.csv")
    )
    assert weather_file.exists()
    df = pd.read_csv(weather_file)

    # 1. Convert 'valid' to datetime and round to the nearest hour
    df["valid"] = pd.to_datetime(df["valid"])
    df["date_hour"] = df["valid"].dt.round("H")

    # 2. Attempt to convert 'tmpf' to numeric, coercing errors to NaN (this will discard invalid values)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")

    # 3. Count the number of rows that could not be parsed (i.e., those that became NaN)
    discarded_rows_count = df["tmpf"].isna().sum()

    # 4. Discard rows where 'tmpf' is NaN (invalid entries)
    df = df.dropna(subset=["tmpf"])

    # 5. Group by 'date_hour' and take the mean (or another aggregation) to ensure one value per hour
    df_hourly = df.groupby("date_hour")["tmpf"].mean().reset_index()

    # 6. Generate a complete range of hours from 2019-01-01 to 2024-11-01
    start_date = pd.Timestamp("2019-01-01 00:00:00")
    end_date = pd.Timestamp("2024-11-01 23:00:00")
    all_hours = pd.date_range(start=start_date, end=end_date, freq="H")

    # 7. Create a DataFrame with the complete range of hours
    complete_hours_df = pd.DataFrame(all_hours, columns=["date_hour"])

    # 8. Merge the hourly data with the complete hours DataFrame
    final_df = pd.merge(complete_hours_df, df_hourly, on="date_hour", how="left")

    # 9. Fill missing temperature values with forward fill, then backward fill if needed
    final_df["tmpf"] = final_df["tmpf"].fillna(
        method="ffill"
    )  # forward fill missing values
    final_df["tmpf"] = final_df["tmpf"].fillna(
        method="bfill"
    )  # backward fill remaining missing values

    # Fallback: interpolate
    # final_df['tmpf'] = final_df['tmpf'].interpolate(method='linear')

    # Print the count of discarded rows and show the final DataFrame
    print(f"Number of discarded rows: {discarded_rows_count}")

    return final_df


def get_gas_prices():
    # load gas price data and join on date
    gas_prices_file = (
        Path(SYNTHEFY_DATASETS_BASE)
        / Path("nycfhv")
        / Path("EMM_EPMR_PTE_Y35NY_DPGw.xls")
    )

    gas_prices_data = pd.ExcelFile(gas_prices_file)

    # Sheet "Data 1" has the actual data
    gas_prices_df = gas_prices_data.parse("Data 1")

    # Drop the first 1 row
    gas_prices_df = gas_prices_df.drop(range(2))

    # Change the column names
    gas_prices_df.columns = ["date", "gas_price"]

    # Change the dtype of the date column
    gas_prices_df["date"] = pd.to_datetime(gas_prices_df["date"])

    # Change the dtype of the gas_price column
    gas_prices_df["gas_price"] = gas_prices_df["gas_price"].astype(float)

    # Create a complete date range from the earliest to the latest date
    full_date_range = pd.date_range(
        start=gas_prices_df["date"].min(), end=gas_prices_df["date"].max()
    )

    # Reindex the DataFrame to include the full date range
    gas_prices_filled = gas_prices_df.set_index("date").reindex(full_date_range)

    # Forward-fill the gas prices to fill in missing days
    gas_prices_filled["gas_price"] = gas_prices_filled["gas_price"].ffill()

    # Reset the index and rename columns for clarity
    gas_prices_filled.reset_index(inplace=True)
    gas_prices_filled.rename(columns={"index": "date"}, inplace=True)

    return gas_prices_filled


def make_hourly_rides(year, month):
    start_time = time.time()
    data_path = (
        Path(SYNTHEFY_DATASETS_BASE)
        / Path("nycfhv")
        / Path(f"fhvhv_tripdata_{year}-{month}.parquet")
    )

    try:
        assert data_path.exists()
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return pd.DataFrame()

    # Extract 'date' and 'hour' from 'pickup_datetime'
    df["date"] = df["pickup_datetime"].dt.date
    df["hour"] = df["pickup_datetime"].dt.hour

    # Combine 'date' and 'hour' without modifying the 'hour' column
    df["date_hour"] = pd.to_datetime(
        df["date"].astype(str) + "_" + df["hour"].astype(str).str.zfill(2),
        format="%Y-%m-%d_%H",
    )

    # For each month, we will create a new table with hourly ride data
    # Group by 'date' and 'hour', and count the number of rides for each hour-date combination (hourly rides)
    hourly_rides = df.groupby(["date", "hour"]).size().reset_index(name="ride_count")

    hourly_rides["date_hour"] = pd.to_datetime(
        hourly_rides["date"].astype(str)
        + "_"
        + hourly_rides["hour"].astype(str).str.zfill(2),
        format="%Y-%m-%d_%H",
    )

    hourly_rides["date"] = pd.to_datetime(hourly_rides["date"])

    # Sanity check for the number of rows
    # Note: this is only true when using pickup_datetime, not request_datetime
    assert len(hourly_rides) <= 31 * 24
    print(f"Completed {year}-{month} in {time.time() - start_time:.2f} seconds")

    return hourly_rides


def merge_hourly_rides_with_gas_prices_and_weather(
    hourly_rides, gas_prices_df, weather_df
):
    # Add gas_price col to hourly_rides
    hourly_rides = hourly_rides.merge(gas_prices_df, on="date", how="left")

    # Round both date_hour columns to the nearest hour
    hourly_rides["date_hour"] = hourly_rides["date_hour"].dt.round("H")
    weather_df["date_hour"] = weather_df["date_hour"].dt.round("H")

    # Add tmpf col to hourly_rides
    hourly_rides = hourly_rides.merge(weather_df, on="date_hour", how="left")

    return hourly_rides


def process_year_month(args):
    year, month, gas_prices_df, weather_df = args
    hourly_rides = make_hourly_rides(year, month)
    hourly_rides = merge_hourly_rides_with_gas_prices_and_weather(
        hourly_rides, gas_prices_df, weather_df
    )
    return hourly_rides


def main():
    gas_prices_df = get_gas_prices()
    weather_df = get_weather_data()

    if DEBUG:
        start_time = time.time()
        hourly_rides_list = []
        hourly_rides_list.append(
            process_year_month(("2019", "02", gas_prices_df, weather_df))
        )
        print(f"Completed in {time.time() - start_time:.2f} seconds")

    else:
        years = [str(y) for y in range(2019, 2024)]
        months = [str(m).zfill(2) for m in range(1, 13)]
        years_months = list(product(years, months)) + list(
            product(("2024",), [str(m).zfill(2) for m in range(1, 10)])
        )

        process_args = [
            (year, month, gas_prices_df, weather_df) for year, month in years_months
        ]

        if N_WORKERS > 1:
            print(f"Using {N_WORKERS} processes")

            start_time = time.time()
            with Pool(N_WORKERS) as pool:
                hourly_rides_list = pool.map(process_year_month, process_args)
            print(f"Completed in {time.time() - start_time:.2f} seconds")
        else:
            hourly_rides_list = []
            start_time = time.time()
            for args in process_args:
                hourly_rides_list.append(process_year_month(args))
            print(f"Completed in {time.time() - start_time:.2f} seconds")

    hourly_rides = pd.concat(hourly_rides_list)
    hourly_rides.to_parquet(
        Path(SYNTHEFY_DATASETS_BASE)
        / Path("nycfhv")
        / Path("hourly_rides_debug.parquet")
    )

    print(f"Hours (rows) in the dataset: {len(hourly_rides)=}")


if __name__ == "__main__":
    main()

"""
run interactively in ipython
ipython -i nyc_fhv.py

window calculation:
730 days = 730 * 24 = 17520 hours
120 window len (120 hours ~= 5 days)
60 stride (60 hours ~= 2.5 days)

17520 / 120 * 2 = 292 windows

onehot for gas_price?
"""
