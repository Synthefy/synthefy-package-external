import copy
import os
import sys
import time
from collections import Counter
from typing import Generator

import pandas as pd
import yaml
from loguru import logger

from synthefy_pkg.loaders.simple_loader import SimpleLoader

FREQUENCY_HIERARCHY = [
    "Hourly",
    "Daily",
    "Weekly",
    "Monthly",
    "Quarterly",
    "Yearly",
]

# Frequency to unit conversion
CONVERT_FREQUENCY_TO_UNIT = {
    "Yearly": "A",
    "Quarterly": "M",
    "Monthly": "M",
    "Weekly": "W",
    "Daily": "D",
    "Hourly": "h",
}

QUARTERS_PER_YEAR = 4
MONTHS_PER_QUARTER = 3
DAYS_PER_WEEK = 7
HOURS_PER_DAY = 24


class MergeLoader(SimpleLoader):
    """
    loads data from the standardized format into pandas dataframes.
    Handles mergeing of a separate set of filters to collect metadata series
    """

    def __init__(self, config_path):
        super().__init__(config_path)

    def _check_frequency_hierarchy(
        self, series_frequency: str, frequency: str
    ) -> bool:
        """
        checks if the frequency is higher than the series frequency
        """
        return FREQUENCY_HIERARCHY.index(frequency) < FREQUENCY_HIERARCHY.index(
            series_frequency
        )

    def _downsample_one_level(
        self,
        metaseries: pd.DataFrame,
        series_frequency: str,
        timestamp_col: str = "timestamp",
        value_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """
        Downsample time series data to a coarser granularity using means.

        Parameters:
        df (pd.DataFrame): Input dataframe with timestamp column
        from_unit (str): Current time unit ('D' for day, 'H' for hour)
        to_unit (str): Target time unit ('M' for month, 'D' for day)
        timestamp_col (str): Name of the timestamp column
        value_cols (list): List of columns to aggregate. If None, all numeric columns are aggregated

        Returns:
        pd.DataFrame: New dataframe with downsampled data
        """

        # Convert timestamps to datetime if they aren't already
        metaseries = metaseries.copy()
        metaseries[timestamp_col] = (
            pd.to_datetime(metaseries[timestamp_col])
            if timestamp_col in metaseries.columns
            else pd.to_datetime(metaseries.index)
        )

        # Set timestamp as index for resampling
        metaseries.set_index(timestamp_col, inplace=True)

        # Determine which columns to aggregate
        if value_cols is None:
            value_cols = [
                str(col)
                for col in metaseries.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()
            ]

        # Define resampling rule
        if series_frequency == "Hourly":
            rule = "D"
            series_frequency = "Daily"
        elif series_frequency == "Daily":
            rule = "W"
            series_frequency = "Weekly"
        elif series_frequency == "Weekly":
            rule = "M"
            series_frequency = "Monthly"
        elif series_frequency == "Monthly":
            rule = "Q"
            series_frequency = "Quarterly"
        elif series_frequency == "Quarterly":
            rule = "A"
            series_frequency = "Yearly"
        else:
            raise ValueError("Unsupported time unit conversion")

        # Perform resampling with mean aggregation
        resampled = metaseries[value_cols].resample(rule).mean()

        # Reset index to convert timestamp back to column, quarterly only handles end of quarter
        if rule == "Q":
            resampled.index = resampled.index.tz_localize(
                None
            )  # Remove timezone temporarily
            resampled.index = resampled.index.to_period(rule).to_timestamp(
                how="end"
            ) + pd.Timedelta(days=1)
            # cancel out hour/minute/second/microsecond
            resampled.index = resampled.index.map(
                lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
            )
        else:
            resampled.index = resampled.index.to_period(rule).to_timestamp(
                how="start"
            )
        return resampled.reset_index(), series_frequency

    def _upsample_one_level(
        self,
        metaseries: pd.DataFrame,
        series_frequency: str,
        timestamp_col: str = "timestamp",
    ) -> tuple[pd.DataFrame, str]:
        """
        Upsamples the metadata one level up in the frequency hierarchy.
        Creates a new dataframe where the values are duplicated for each timestamp at the next granularity level.
        """
        times = (
            pd.to_datetime(metaseries[timestamp_col])
            if timestamp_col in metaseries.columns
            else pd.to_datetime(metaseries.index)
        )
        to_unit = CONVERT_FREQUENCY_TO_UNIT[
            FREQUENCY_HIERARCHY[FREQUENCY_HIERARCHY.index(series_frequency) - 1]
        ]

        date_ranges = pd.Series()
        if series_frequency == "Yearly":
            # properly set the k_values for upsample to quarterly data

            date_ranges = pd.Series(
                [
                    (quarter).tz_localize(None).tz_localize("UTC")
                    for year in times
                    for quarter in pd.date_range(
                        start=f"{year}-01-01",
                        periods=QUARTERS_PER_YEAR,
                        freq="QS",
                    )
                ]
            )
            k_values = pd.Series(
                [QUARTERS_PER_YEAR] * len(metaseries)
            )  # should never be used

            series_frequency = "Quarterly"
        elif series_frequency == "Quarterly":
            k_values = pd.Series(
                [MONTHS_PER_QUARTER] * len(metaseries)
            )  # 3 months per quarter
            series_frequency = "Monthly"
        elif series_frequency == "Monthly":
            k_values = pd.Series(times).dt.days_in_month.map(
                lambda x: len(
                    pd.date_range(
                        start=pd.Timestamp(
                            "2024-01-01"
                        ),  # Use any month with same number of days
                        periods=x,
                        freq="D",
                    )
                    .to_period("W")
                    .unique()
                )
            )
            series_frequency = "Weekly"
        elif series_frequency == "Weekly":
            k_values = pd.Series(
                [DAYS_PER_WEEK] * len(metaseries)
            )  # 7 days per week
            series_frequency = "Daily"
        elif series_frequency == "Daily":
            k_values = pd.Series(
                [HOURS_PER_DAY] * len(metaseries)
            )  # 24 hours per day
            series_frequency = "Hourly"
        else:
            raise ValueError(f"Unsupported upsampling from {series_frequency}")

        # Create date ranges for each timestamp
        if series_frequency != "Quarterly":
            date_ranges = [
                pd.date_range(start=t, periods=k, freq=to_unit)
                for t, k in zip(times, k_values)
            ]

        # Flatten the date ranges
        new_timestamps = pd.concat([pd.Series(dr) for dr in date_ranges])

        # Repeat each row according to its k value
        new_df = metaseries.loc[metaseries.index.repeat(k_values)].copy()
        new_df[timestamp_col] = new_timestamps.values

        new_df = new_df.reset_index(drop=True)

        if new_df.index.duplicated().any():
            new_df = new_df[~new_df.index.duplicated(keep="last")]

        return new_df, series_frequency

    def _resample_metadata(
        self,
        metaseries: pd.DataFrame,
        metadata_frequency: str,
        main_series_frequency: str,
    ) -> pd.DataFrame:
        """
        resamples the metadata to the given frequency
        """
        if metadata_frequency == main_series_frequency:
            return metaseries
        else:
            if not self._check_frequency_hierarchy(
                metadata_frequency, main_series_frequency
            ):
                # upsampling if the series frequency is lower than frequency
                while metadata_frequency != main_series_frequency:
                    logger.info(
                        f"downsampling from {metadata_frequency} to {main_series_frequency}"
                    )
                    metaseries, metadata_frequency = self._downsample_one_level(
                        metaseries, metadata_frequency
                    )
            else:
                # downsampling if the series frequency is higher than frequency
                while metadata_frequency != main_series_frequency:
                    logger.info(
                        f"upsampling from {metadata_frequency} to {main_series_frequency}"
                    )
                    metaseries, metadata_frequency = self._upsample_one_level(
                        metaseries, metadata_frequency
                    )
            return metaseries

    def _generate_metadata_all_frequency_dict(
        self,
        metadata_generator: Generator[
            tuple[str, pd.DataFrame, dict[str, str | list[str]]], None, None
        ],
    ) -> list[tuple[dict[str, pd.DataFrame], dict[str, str | list[str]]]]:
        """
        generates a dictionary of frequency: metadata resampled to that frequency
        """
        metadatas = [md_tuple for md_tuple in metadata_generator]
        if self.config["metadata_n_longest"] > 0:
            metadatas = sorted(
                metadatas, key=lambda x: x[2]["length"], reverse=True
            )[: self.config["metadata_n_longest"]]
        metadata_dicts = list()
        for metadata_name, metaseries, metadata_meta in metadatas:
            metadata_dict: dict[str, pd.DataFrame] = dict()
            for frequency in FREQUENCY_HIERARCHY:
                # remove the timestamp column
                if "timestamp" in metaseries.columns:
                    timestamp_col = metaseries.pop("timestamp")
                    metaseries.index = timestamp_col
                # the frequency assignment is PURELY for the type checker, frequency can't actually be a list
                metadata_frequency: str = (
                    metadata_meta["frequency"]
                    if isinstance(metadata_meta["frequency"], str)
                    else metadata_meta["frequency"][0]
                )
                metadata_dict[frequency] = self._resample_metadata(
                    metaseries, metadata_frequency, frequency
                )
                if "timestamp" in metadata_dict[frequency].columns:
                    timestamp_col = metadata_dict[frequency].pop("timestamp")
                    metadata_dict[frequency].index = timestamp_col
            logger.info(
                f"resampled metadata {metaseries.shape}, with total nan ratio {metaseries.isna().sum() / len(metaseries)}"
            )
            metadata_dicts.append((metadata_dict, metadata_meta))
        return metadata_dicts

    def _merge_slice_metadata(
        self,
        metadata_dict: tuple[
            dict[str, pd.DataFrame], dict[str, str | list[str]]
        ],
        datas: pd.DataFrame,
        dataset_metadata: dict[str, str | list[str]],
        timestamp_col: pd.DatetimeIndex,
        metadata_seen_count: Counter,
    ) -> tuple[pd.DataFrame, dict[str, str | list[str]]]:
        """
        merges the metadata into the data
        """
        metadata_meta = copy.deepcopy(metadata_dict[1])
        metadata_series = metadata_dict[0][
            str(dataset_metadata["frequency"])
        ].copy()
        meta_timestamp_col = (
            pd.DatetimeIndex(metadata_series["timestamp"])
            if "timestamp" in metadata_series.columns
            else metadata_series.index
        )
        data_tz = pd.to_datetime(timestamp_col).tz
        meta_timestamps = pd.to_datetime(meta_timestamp_col)
        if meta_timestamps.tz is None:
            metadata_series["timestamp"] = meta_timestamps.tz_localize(
                data_tz, nonexistent="shift_forward"
            )
        else:
            metadata_series["timestamp"] = meta_timestamps.tz_convert(data_tz)
        metadata_series.set_index("timestamp", inplace=True)

        if FREQUENCY_HIERARCHY.index(
            str(dataset_metadata["frequency"])
        ) >= FREQUENCY_HIERARCHY.index("Daily"):
            # strip details
            metadata_series = self._apply_rounding_to_index(
                metadata_series, str(dataset_metadata["frequency"])
            )
        # clip metadata into the time range of the data
        metadata_series = metadata_series[
            (metadata_series.index >= datas.index.min())
            & (metadata_series.index <= datas.index.max())
        ]
        # interpolate metadata to the nearest data timestamp
        if metadata_series.index.duplicated().any():
            logger.info(
                "Warning: Duplicate timestamps in metadata_series. Taking last value."
            )
            metadata_series = (
                metadata_series.reset_index()
                .drop_duplicates(subset=["timestamp"], keep="last")
                .set_index("timestamp")
            )
        # once again, a line purely for the type checker
        metadata_series = (
            metadata_series
            if isinstance(metadata_series, pd.DataFrame)
            else pd.DataFrame(metadata_series)
        )

        nearest_timestamps = pd.Series(
            pd.merge_asof(
                pd.DataFrame({"timestamp": metadata_series.index}).sort_values(
                    "timestamp"
                ),
                pd.DataFrame(
                    {"timestamp": datas.index, "original": datas.index}
                ).sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )["original"]
        )

        # Map metadata to nearest timestamps
        metadata_series.index = nearest_timestamps

        # rename duplicate columns but otherwise keep them
        metadata_seen_count[metadata_series.columns[0]] += 1
        metadata_series.rename(
            columns={
                str(
                    metadata_series.columns[0]
                ): f"{metadata_series.columns[0]}_{metadata_seen_count[metadata_series.columns[0]]}"
            },
            inplace=True,
        )
        if isinstance(metadata_meta["columns"][0], dict):
            metadata_meta["columns"][0]["column_id"] = metadata_series.columns[
                0
            ]
        assert isinstance(dataset_metadata["columns"], list)
        dataset_metadata["columns"] += metadata_meta["columns"]
        return pd.merge(
            datas,
            metadata_series,
            left_index=True,
            right_index=True,
            how="left",
        ), dataset_metadata

    def load(
        self,
        standardized_dataset_category_names: list[str],
        filters: list[dict[str, str | list[str]]],
        metadata_filters: list[dict[str, str | list[str]]] | None = None,
        metadata_standardized_dataset_category_names: list[str] = [],
        start_end: dict[str, str] | None = None,
        complex_filter: str = "",
        metadata_complex_filter: str = "",
        post_filters: list[dict[str, str | list[str]]] | None = None,
        post_metadata_filters: list[dict[str, str | list[str]]] | None = None,
        post_complex_filter: str = "",
        post_metadata_complex_filter: str = "",
        existing_values_skip_dictionary: dict[str, int] = {},
    ) -> Generator[
        tuple[str, pd.DataFrame, dict[str, str | list[str]]], None, None
    ]:
        """
        returns a tuple of two sets of dataframes, one for metadata and one for the data
        the metadata will be organized as a dictionary of frequency: metadata at frequency
        identifies the frequencies based on the data files and matches metadata to data

        @param standardized_dataset_category_names: list of folders in /home/data/standardized/<DATANAME> to load data from
        @param filters: list of filters to apply to the data as list of dictionary with filter_name: filter_value
        @param metadata_filters: list of filters to apply to the metadata as list of dictionary with filter_name: filter_value
        @param metadata_dataname: name of the metadata dataname to load
        @param start_end: tuple of start and end dates to load from, slices dataframes by start and end dates
        @param complex_filter: complex filter that dictates merging or intersecting sets of datasets
        @param post_filters: list of filters to apply to the data after loading into dataframes, must be size or length of datanames
        @param post_metadata_filters: list of filters to apply to the metadata after loading, must be size or length of metadata_dataname
        @param post_complex_filter: complex filter to apply to the data after loading into dataframes
        @param post_metadata_complex_filter: complex filter to apply to the metadata after loading
        """
        # load metadata series

        data_generator = super().load(
            standardized_dataset_category_names,
            filters=filters,
            post_filters=post_filters,
            start_end_timestamps=start_end,
            complex_filter=complex_filter,
            post_complex_filter=post_complex_filter,
            existing_values_skip_dictionary=existing_values_skip_dictionary,
        )

        if metadata_filters is not None and len(metadata_filters) > 0:
            metadata_generator = super().load(
                metadata_standardized_dataset_category_names,
                filters=metadata_filters,
                post_filters=post_metadata_filters,
                start_end_timestamps=start_end,
                complex_filter=metadata_complex_filter,
                post_complex_filter=post_metadata_complex_filter,
            )
        else:  # no metadata filters, so we just return the data series
            yield from data_generator
            return

        # create a dictionary of frequency: metadata
        # metadata should handle all frequencies
        metadata_dicts = self._generate_metadata_all_frequency_dict(
            metadata_generator
        )

        # merge the metadata series into the data series
        for dataset_name, dataset, dataset_metadata in data_generator:
            dataset_metadata = copy.deepcopy(dataset_metadata)
            # handle timestamps as index column
            if "timestamp" in dataset.columns:
                timestamp_col = pd.DatetimeIndex(dataset["timestamp"])
                dataset.set_index("timestamp", inplace=True)
            else:
                timestamp_col = dataset.index

            # duplicate timestamps are not allowed (there also shouldn't be any in standardized data)
            if dataset.index.duplicated().any():
                dataset = dataset[~dataset.index.duplicated(keep="last")]

            dataset = (
                pd.DataFrame(dataset)
                if isinstance(dataset, pd.Series)
                else dataset
            )
            timestamp_col = (
                pd.DatetimeIndex(timestamp_col)
                if not isinstance(timestamp_col, pd.DatetimeIndex)
                else timestamp_col
            )

            # merge metadata into data
            metadata_seen_count = Counter()
            for metadata_dict in metadata_dicts:
                dataset, dataset_metadata = self._merge_slice_metadata(
                    metadata_dict,
                    dataset,
                    dataset_metadata,
                    timestamp_col,
                    metadata_seen_count,
                )
            if isinstance(dataset, pd.Series):
                dataset = dataset.to_frame()
                logger.info(
                    "had to convert series to dataframe, this is probably an error"
                )
            dataset.index.rename("index", inplace=True)
            dataset["timestamp"] = (
                dataset.index
            )  # ensure there is a timestamp column
            self.update_metadata_sizing(dataset_metadata, dataset)
            dataset_metadata["timestamp_columns"] = ["timestamp"]

            # overcomplicated code because the type checker is dumb
            dataset_metadata_columns = list()
            for col in dataset_metadata["columns"]:
                assert isinstance(col, dict)
                assert isinstance(col["column_id"], str)
                dataset_metadata_columns.append(col["column_id"])
            logger.info(
                f"saving with nulls {dataset[dataset.columns[0]].isna().sum()} in main"
            )
            yield (dataset_name, dataset, dataset_metadata)


if __name__ == "__main__":
    # loading options from a yaml file provided as a command line argument
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--outname", type=str, help="name of the save dataset", required=True
    )
    parser.add_argument(
        "--clean-dir", action="store_true", help="clean the output directory"
    )
    parser.add_argument(
        "--no-replace-existing",
        action="store_true",
        help="replace existing files",
    )
    parser.add_argument(
        "--logfile", type=str, help="file to save logs", default=""
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    loader = MergeLoader(args.config)
    if len(args.logfile) > 0:
        logger.add(args.logfile)
    start = time.time()

    # Note: parallelized loading not implemented since there appears to be no appreciable benefit
    if args.no_replace_existing:
        existing_values_skip_dictionary = (
            loader.generate_existing_values_skip_dict(args.outname)
        )
    else:
        existing_values_skip_dictionary: dict[str, int] = dict()
    data_generator = loader.load(
        config["standardized_dataset_category_names"],
        filters=config["time_series_filters"],
        metadata_filters=config["metadata_filters"],
        metadata_standardized_dataset_category_names=config[
            "meta_standardized_dataset_category_names"
        ],
        start_end=config["time_period"],
        complex_filter=config["complex_filter"]
        if "complex_filter" in config
        else "",
        metadata_complex_filter=config["metadata_complex_filter"]
        if "metadata_complex_filter" in config
        else "",
        post_filters=config["post_filters"]
        if "post_filters" in config
        else None,
        post_metadata_filters=config["post_metadata_filters"]
        if "post_metadata_filters" in config
        else None,
        post_complex_filter=config["post_complex_filter"]
        if "post_complex_filter" in config
        else "",
        post_metadata_complex_filter=config["post_metadata_complex_filter"]
        if "post_metadata_complex_filter" in config
        else "",
        existing_values_skip_dictionary=existing_values_skip_dictionary,
    )
    os.makedirs(os.path.join(config["output_dir"], args.outname), exist_ok=True)
    loader.save_parquet(
        data_generator,
        outname=args.outname,
        clean_dir=args.clean_dir,
        replace_existing=not args.no_replace_existing,
    )
    end = time.time()
    logger.info(f"time taken single {end - start}")
