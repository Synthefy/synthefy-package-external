import copy
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
from loguru import logger
from tqdm import tqdm


def csv_to_parquet_with_metadata(
    csv_path: str,
    output_dir: str,
    title: str,
    text_descriptor: str,
    frequency: str,
    domain: str,
    license: str,
    tags: Optional[List[str]] = None,
    url: Optional[List[str]] = None,
    timezone: str = "GMT+0",
    timezone_guessed: bool = False,
    time_column_name: str = "Date",
) -> Dict:
    """
    Convert a CSV file to separate Parquet files for each column and generate metadata JSON.

    Args:
        csv_path: Path to the input CSV file
        output_dir: Directory to save Parquet files and metadata
        title: Name of the dataset
        text_descriptor: Description of the dataset
        frequency: Frequency of the time series data (e.g., "Daily", "Hourly")
        domain: Domain of the dataset
        license: License name (e.g., "MIT", "Apache-2.0")
        tags: List of tags to categorize the dataset
        url: List of relevant URLs for the dataset
        timezone: Timezone offset from GMT (e.g., "GMT+0", "GMT+8")
        timezone_guessed: Whether the timezone was guessed

    Returns:
        Dict containing the metadata schema
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Initialize metadata
    metadata = {
        "title": title,
        "id": f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}",
        "text_descriptor": text_descriptor,
        "frequency": frequency,
        "tags": tags or [],
        "url": url or [],
        "license": license,
        "domain": domain,
        "timezone": timezone,
        "timezone_guessed": timezone_guessed,
        "size": os.path.getsize(csv_path),
        "length": len(df),
        "num_columns": len(df.columns),
        "start_date": df.index[0].isoformat()
        if isinstance(df.index, pd.DatetimeIndex)
        else None,
        "end_date": df.index[-1].isoformat()
        if isinstance(df.index, pd.DatetimeIndex)
        else None,
        "timestamp_column": time_column_name,
        "timestamps_columns": [time_column_name],
        "initial_columns": [],
        "final_columns": [],
    }

    print(df.columns)
    time_column = df[time_column_name]
    print(time_column.head())
    # remove time column
    df = df.drop(columns=[time_column_name])

    # assign start and end date based on time column
    metadata["start_date"] = (
        time_column[0].isoformat()
        if isinstance(time_column, pd.DatetimeIndex)
        else None
    )
    metadata["end_date"] = (
        time_column[-1].isoformat()
        if isinstance(time_column, pd.DatetimeIndex)
        else None
    )

    # Process each column
    for col in df.columns:
        new_metadata = copy.deepcopy(metadata)
        # Determine column type
        col_type = (
            "continuous" if pd.api.types.is_numeric_dtype(df[col]) else "text"
        )
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "text"
            new_metadata["timestamps_columns"].append(col)

        # Create column metadata
        col_metadata = {
            "title": f"{title} - {col}",
            "description": f"Column {col} from {title} dataset",
            "type": col_type,
            "is_metadata": "no",
            "units": None,
        }

        new_metadata["initial_columns"].append(col_metadata)

        # Create final column metadata with additional fields
        final_col_metadata = {
            **col_metadata,
            "id": f"{col.lower().replace(' ', '_')}",
            "column_id": col,
        }
        new_metadata["final_columns"].append(final_col_metadata)

        # Save column as parquet
        col_df = pd.DataFrame({col: df[col]})
        # index with time column
        col_df = col_df.set_index(time_column)

        # assign parameters
        new_metadata["length"] = len(col_df)
        new_metadata["num_columns"] = len(col_df.columns)
        new_metadata["size"] = (
            new_metadata["length"] * new_metadata["num_columns"]
        )

        print(
            output_path,
            f"{col.lower().replace(' ', '_').replace('/', '_div_')}.parquet",
        )
        parquet_path = (
            output_path
            / f"{col.lower().replace(' ', '_').replace('/', '_div_')}.parquet"
        )
        col_df.to_parquet(parquet_path)

        # Save metadata
        metadata_path = (
            output_path
            / f"{title.lower().replace(' ', '_').replace('/', '_div_')}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(new_metadata, f, indent=4)

    return metadata


def univariate_dataset_to_training_set(
    source_dir: str,
    target_dir: str,
    num_rows: int,
    num_cols: int,
    num_datasets_to_sample: int,
    name_prefix: str = "dataset",
    clean_dir: bool = False,
):
    """
    Convert a univariate dataset to a set of csvs with a fixed length of num_rows rows and num_cols columns by randomly sampling rows and columns from the set of univariate dataset parquets.
    """
    all_columns = list()
    for file in os.listdir(source_dir):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(source_dir, file))

            # since each df consists of a single column, we take just the column and add to all_columns
            all_columns.append(df)

    # TODO: assumes that all the columns are time synchronized
    # TODO: assumes that there is enough variation that random sampling produces sufficiently different datasets
    if clean_dir:
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)
    for i in tqdm(range(num_datasets_to_sample)):
        # sample num_rows rows and num_cols columns
        # Convert to indices and sample those instead
        sampled_indices = np.random.choice(
            len(all_columns), min(num_cols, len(all_columns)), replace=False
        )
        sampled_columns = [all_columns[idx] for idx in sampled_indices]
        sampled_df = pd.concat(sampled_columns, axis=1)

        # set the index based on the index of the first dataframe
        sampled_df.index = sampled_columns[0].index
        try:
            sampled_df.index = pd.to_datetime(sampled_df.index)
        except Exception as e:
            logger.info(f"Index is not a datetime index: {e}")

        # randomly select a starting point based on num_rows:
        start_index = np.random.randint(0, len(sampled_df) - num_rows)
        logger.info(f"start_index: {start_index}")
        sampled_df = sampled_df.iloc[start_index : start_index + num_rows]

        # convert timestamps into minute, hour, day, month, year
        if isinstance(sampled_df.index, pd.DatetimeIndex):
            sampled_df["minute"] = sampled_df.index.minute
            sampled_df["hour"] = sampled_df.index.hour
            sampled_df["day"] = sampled_df.index.day
            sampled_df["month"] = sampled_df.index.month
            sampled_df["year"] = sampled_df.index.year
        else:
            logger.info("Creating fake timestamps")
            # create fake timestamps, minutely
            sampled_df["minute"] = np.arange(0, num_rows) % 60
            sampled_df["hour"] = (np.arange(0, num_rows) // 60) % 24
            sampled_df["day"] = (np.arange(0, num_rows) // 60 // 24) % 30
            sampled_df["month"] = 0
            sampled_df["year"] = 10

        # reorder the columns so the first 5 are the timestamps
        sampled_df = sampled_df[
            ["minute", "hour", "day", "month", "year"]
            + list(sampled_df.columns[:-5])
        ]

        # save to csv
        logger.info(
            f"saved dataset {i} of shape {sampled_df.shape} to {os.path.join(target_dir, f'{name_prefix}_{i}.csv')}"
        )
        sampled_df.to_csv(
            os.path.join(target_dir, f"{name_prefix}_{i}.csv"), index=False
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--title", type=str, required=False)
    parser.add_argument("--text_descriptor", type=str, required=False)
    parser.add_argument("--frequency", type=str, required=False)
    parser.add_argument("--domain", type=str, required=False)
    parser.add_argument("--license", type=str, required=False)
    parser.add_argument("--tags", type=str, required=False)
    parser.add_argument("--url", type=str, required=False)
    parser.add_argument("--timezone", type=str, required=False)
    parser.add_argument("--timezone_guessed", type=bool, required=False)
    parser.add_argument("--time_column_name", type=str, required=False)
    parser.add_argument("--clean_dir", action="store_true", default=False)

    # params for saving to multivariate dataset
    parser.add_argument("--source_dir", type=str, required=False)
    parser.add_argument(
        "--save_to_multivariate", action="store_true", default=False
    )
    parser.add_argument("--num_rows", type=int, required=False)
    parser.add_argument("--num_cols", type=int, required=False)
    parser.add_argument("--num_datasets_to_sample", type=int, required=False)
    parser.add_argument("--name_prefix", type=str, required=False)
    args = parser.parse_args()

    if args.clean_dir:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(args.output_dir)

    if args.save_to_multivariate:
        univariate_dataset_to_training_set(
            source_dir=args.source_dir,
            target_dir=args.output_dir,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            num_datasets_to_sample=args.num_datasets_to_sample,
            name_prefix=args.name_prefix,
            clean_dir=args.clean_dir,
        )
    else:
        csv_to_parquet_with_metadata(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            title=args.title,
            text_descriptor=args.text_descriptor,
            frequency=args.frequency,
            domain=args.domain,
            license=args.license,
            tags=args.tags,
            url=args.url,
            timezone=args.timezone,
            timezone_guessed=args.timezone_guessed,
            time_column_name=args.time_column_name,
        )

    # uv run src/synthefy_pkg/scripts/csv_to_parquet.py --csv_path="/mnt/workspace1/data/multivariate_datasets/mpi_roof_weather.csv" --output_dir="/mnt/workspace1/data/enriched_datasets/weather_mpi_beutenberg" --title="Weather Data from sensors near WS Beutenberg collected at 10 min interval frequency" --text_descriptor="This dataset contains meteorological observations of WS Beutenberg in Germany from 2004 to 2023. Data points are recorded at 10-minute intervals, comprising a total of 20 variables such as temperature, air pressure, humidity etc." --frequency="10 minute" --domain="weather" --license="CC 1.0" --tags="weather,Germany" --url="https://www.bgc-jena.mpg.de/wetter/weather_data.html" --timezone="UTC+2" --timezone_guessed=True --time_column_name "Date Time" --clean_dir
    # uv run src/synthefy_pkg/scripts/csv_to_parquet.py --csv_path="/mnt/workspace1/data/multivariate_datasets/solar_Alabama.csv" --output_dir="/mnt/workspace1/data/enriched_datasets/solar_Alabama" --title="Alabama Solar Power data in Megawatt hours at 5 minute intervals" --text_descriptor="This dataset consists of 1 year (2006) of 5-minute solar power (mW) for 137 photovoltaic power plants in Alabama State." --frequency="5 minute" --domain="energy" --license="CC 1.0" --tags="energy,United States,solar,Alabama" --url="https://www.nrel.gov/grid/solar-power-data" --timezone="UTC-5" --timezone_guessed=True --time_column_name "Date Time" --clean_dir
    # uv run src/synthefy_pkg/scripts/csv_to_parquet.py --csv_path="/mnt/workspace1/data/multivariate_datasets/traffic_PeMS.csv" --output_dir="/mnt/workspace1/data/enriched_datasets/traffic_PeMS" --title="Traffic Load on San Francisco Bay Area freeways, collected hourly" --text_descriptor="This dataset describes the hourly road occupancy rates (ranges from 0 to 1) measured by 862 sensors on San Francisco Bay Area freeways from 2015 to 2016." --frequency="hourly" --domain="traffic" --license="CC 1.0" --tags="traffic,United States,San Francisco" --url="https://pems.dot.ca.gov/" --timezone="UTC-7" --timezone_guessed=True --time_column_name "Time" --clean_dir

    # uv run src/synthefy_pkg/scripts/csv_to_parquet.py --save_to_multivariate --source_dir="/mnt/workspace1/data/enriched_datasets/weather_mpi_beutenberg" --output_dir="/mnt/workspace1/data/sampled_datasets/weather_mpi_beutenberg" --num_rows=768 --num_cols=20 --num_datasets_to_sample=100 --name_prefix="weather_real" --clean_dir
    # uv run src/synthefy_pkg/scripts/csv_to_parquet.py --save_to_multivariate --source_dir="/mnt/workspace1/data/enriched_datasets/traffic_PeMS/" --output_dir="/mnt/workspace1/data/sampled_datasets/traffic_PeMS" --num_rows=768 --num_cols=35 --num_datasets_to_sample=100 --name_prefix="traffic_real" --clean_dir
    # uv run src/synthefy_pkg/scripts/csv_to_parquet.py --save_to_multivariate --source_dir="/mnt/workspace1/data/enriched_datasets/solar_Alabama/" --output_dir="/mnt/workspace1/data/sampled_datasets/solar_Alabama" --num_rows=768 --num_cols=35 --num_datasets_to_sample=100 --name_prefix="solar_real" --clean_dir
