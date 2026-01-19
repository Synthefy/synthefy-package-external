import json
import os
import sys
from multiprocessing import Manager, Pool, cpu_count

from haver import Haver
from loguru import logger
from tqdm import tqdm


def get_api_key():
    api_key = os.environ.get("API_HAVER_KEY")
    if not api_key:
        logger.error(
            "API_HAVER_KEY environment variable not set. Please set it before running this script."
        )
        sys.exit(1)
    return api_key


DATABASE_JSON_DIR = "/home/synthefy/data/haver_database_jsons"
OUTPUT_DIR = "/home/synthefy/data/fmv3_univariate/haver"


def extract_and_save_series_data(haver, series):
    database = series["databaseName"]
    series_name = series["name"]
    series_id = f"{series_name}@{database}"

    # Get the description
    text_description = series["description"]

    try:
        # Get the raw data
        series_df = haver.read_df(haver_codes=[series_id])

        # Keep only the date and value columns
        if not series_df.empty:
            series_df = series_df[["date", "value"]]

            # make the output directory for this series
            os.makedirs(
                os.path.join(OUTPUT_DIR, database, series_name), exist_ok=True
            )
            series_df.to_parquet(
                os.path.join(
                    OUTPUT_DIR, database, series_name, "timeseries.parquet"
                )
            )

            # Create metadata json with description
            metadata = {"columns": [{"description": text_description}]}
            with open(
                os.path.join(
                    OUTPUT_DIR, database, series_name, "metadata.json"
                ),
                "w",
            ) as f:
                json.dump(metadata, f, indent=4)
        else:
            # tqdm.write(f"No data found for {series_id}")
            pass
    except Exception:
        pass


def process_series(args):
    haver, series, progress_queue = args
    extract_and_save_series_data(haver, series)
    progress_queue.put(1)


def main():
    API_KEY = get_api_key()

    haver = Haver(api_key=API_KEY)

    # Get all the databases- should be 33 of them
    databases = haver.get_databases()

    all_series = []
    for database in tqdm(databases):
        with open(
            os.path.join(DATABASE_JSON_DIR, f"access_{database}.json"), "r"
        ) as f:
            database_series = json.load(f)

        logger.info(f"Number of series in {database}: {len(database_series)}")
        all_series.extend(database_series)

        # Make the output directory for this database
        os.makedirs(os.path.join(OUTPUT_DIR, database), exist_ok=True)

    logger.info(f"Total number of series: {len(all_series)}")

    # Use multiprocessing to process each series
    with Manager() as manager:
        progress_queue = manager.Queue()
        with Pool(cpu_count()) as pool:
            # Wrap the series with the haver object and progress queue
            args = [(haver, series, progress_queue) for series in all_series]
            # Start the pool
            pool.map_async(process_series, args)

            # Create a progress bar
            with tqdm(total=len(all_series)) as pbar:
                for _ in range(len(all_series)):
                    progress_queue.get()
                    pbar.update(1)


if __name__ == "__main__":
    main()
