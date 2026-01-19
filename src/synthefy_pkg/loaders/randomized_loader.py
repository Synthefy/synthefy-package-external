import copy
import os
import sys
from collections import Counter
from typing import Generator

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from loaders.simple_loader import SimpleLoader
from synthefy_pkg.loaders.simple_loader import SimpleLoader


class RandomLoader(SimpleLoader):
    """
    loads data from the standardized format into pandas dataframes.
    Handles mergeing of a random selection of the data into a dataframe
    """

    def __init__(self, config_path):
        super().__init__(config_path)
        self.num_datasets_to_save: int = (
            int(self.config["num_datasets_to_save"])
            if "num_datasets_to_save" in self.config
            else 100
        )
        self.sample_dataset_ratio: float = (
            float(self.config["sample_dataset_ratio"])
            if "sample_dataset_ratio" in self.config
            else 1.0
        )
        self.num_variates_per_enrich: int = (
            int(self.config["num_variates_per_enrich"])
            if "num_variates_per_enrich" in self.config
            else 5
        )
        self.num_files_per_subset: int = (
            int(self.config["num_files_per_subset"])
            if "num_files_per_subset" in self.config
            else 10
        )

    def _rename_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        rename the index to timestamp, then make the index just a range
        """
        if isinstance(df.index, pd.DatetimeIndex):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"index is not a datetime index: {df.index}")
            else:
                df["timestamp"] = df.index.copy()
            df.reset_index(drop=True, inplace=True)
        return df

    def _get_column_statistics(self, df: pd.DataFrame) -> dict[str, str]:
        """
        get the time and length statistics for the column to be added to the metadata
        """
        prefill_length = len(df)
        if len(df) == 0:
            return {
                "prefill_length": str(prefill_length),
                "start_time": str(pd.Timestamp.min),
                "end_time": str(pd.Timestamp.min),
            }
        start_time = df["timestamp"].iloc[0]
        end_time = df["timestamp"].iloc[-1]
        return {
            "prefill_length": str(prefill_length),
            "start_time": str(start_time),
            "end_time": str(end_time),
        }

    def _rename_timestamp_duplicate_column(
        self,
        df: pd.DataFrame,
        meta: dict[str, str | list[str | dict[str, str]]],
        seen_count: int,
    ) -> tuple[pd.DataFrame, dict[str, str | list[str | dict[str, str]]]]:
        """
        rename the column to avoid duplicate issues, and add the timestamp column to the metadata
        """
        df.rename(
            columns={
                str(df.columns[0]): f"{df.columns[0]}_{seen_count}",
                "timestamp": f"{df.columns[0]}_{seen_count}_timestamp",
            },
            inplace=True,
        )

        if isinstance(meta["columns"][0], dict):
            meta["columns"][0]["column_id"] = str(df.columns[0])
        assert isinstance(meta["columns"], list)

        # timestamp column has the same title, but must be run before reassigning the length of the dataframe
        meta["columns"].append(
            {
                "column_id": str(df.columns[1]),
                "title": "Timestamps",
                "description": "Timestamps for the dataset",
                "type": "timestamp",
                "is_metadata": "no",
                "units": "timestamp",
                "start_time": str(df[df.columns[1]].iloc[0]),
                "end_time": str(df[df.columns[1]].iloc[-1]),
                "prefill_length": str(len(df)),
            }
        )

        assert isinstance(meta["timestamps_columns"], list)
        meta["timestamps_columns"] = [str(df.columns[1])]

        return df, meta

    def _load_subset(
        self,
        data_generator: Generator[
            tuple[
                str, pd.DataFrame, dict[str, str | list[str | dict[str, str]]]
            ],
            None,
            None,
        ],
    ) -> list[
        tuple[str, pd.DataFrame, dict[str, str | list[str | dict[str, str]]]]
    ]:
        all_variate_tuples = []
        for dataset_name, dataset, dataset_metadata in data_generator:
            dataset = self._rename_index(dataset)
            all_variate_tuples.append((dataset_name, dataset, dataset_metadata))
        return all_variate_tuples

    def _merge_variate(
        self,
        result: pd.DataFrame,
        result_meta: dict[str, str | list[str | dict[str, str]]],
        new_name: str,
        new_data: pd.DataFrame,
        new_meta: dict[str, str | list[str | dict[str, str]]],
        series_seen_count: Counter,
    ) -> tuple[pd.DataFrame, dict[str, str | list[str | dict[str, str]]]]:
        """
        merges one variate into the result dataframe
        copies are needed to prevent reindexing from creating issues

        @param result: the result dataframe (dataframe for all variates)
        @param result_meta: the metadata for the result dataframe
        @param new_name: the name of the new variate to add
        @param new_data: the new variate data
        @param new_meta: the metadata for the new variate
        @param series_seen_count: the number of times we have seen the new variate's name (ideally, should be 1)
        """
        new_meta = copy.deepcopy(new_meta)
        new_data = new_data.copy()

        value_column = new_data.columns[0]

        # pad new data to the current length with nans
        current_length = len(result)
        if isinstance(new_meta["columns"][0], dict):
            column_stats: dict[str, str] = self._get_column_statistics(new_data)
            assert isinstance(new_meta["columns"], list)
            cur_col_values: dict[str, str] = new_meta["columns"][0]
            new_meta["columns"][0] = {**column_stats, **cur_col_values}

        new_data = new_data.reindex(range(current_length), method="ffill")

        # rename duplicate columns but otherwise keep them
        series_seen_count[value_column] += 1

        new_data, new_meta = self._rename_timestamp_duplicate_column(
            new_data, new_meta, series_seen_count[new_data.columns[0]]
        )

        assert isinstance(new_meta["columns"], list)
        assert isinstance(result_meta["columns"], list)
        result_meta["columns"] += new_meta["columns"]
        assert isinstance(result_meta["timestamps_columns"], list)
        result_meta["timestamps_columns"] += new_meta["timestamps_columns"]

        return pd.merge(
            result,
            new_data,
            left_index=True,
            right_index=True,
            how="left",
        ), result_meta

    def load(
        self,
        standardized_dataset_category_names: list[str],
        filters: list[dict[str, str | list[str]]],
        start_end: dict[str, str] | None = None,
        complex_filter: str = "",
        post_filters: list[dict[str, str | list[str]]] | None = None,
        post_complex_filter: str = "",
        existing_values_skip_dictionary: dict[str, int] = {},
    ) -> Generator[
        tuple[str, pd.DataFrame, dict[str, str | list[str | dict[str, str]]]],
        None,
        None,
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

        names_to_open = self.load_names_to_open(
            standardized_dataset_category_names, filters, complex_filter
        )
        name_column_dict, total_columns = self._compute_total_columns(
            names_to_open
        )

        # import pickle
        # # pickle.dump((names_to_open, name_column_dict, total_columns), open("names_to_open.pkl", "wb"))
        # names_to_open, name_column_dict, total_columns = pickle.load(open("names_to_open.pkl", "rb"))

        logger.info(f"total columns: {total_columns}")
        datasets_saved_num = 0

        seen_dataset_count = {ssn: 0 for ssn in names_to_open}

        while datasets_saved_num < self.num_datasets_to_save:
            # sample with probability proportional to the number of times we have seen the dataset, prioritizing less seen datasets at e^-v rate
            total_seen = sum([np.exp(-v) for v in seen_dataset_count.values()])
            subset_names = np.random.choice(
                names_to_open,
                min(len(names_to_open), self.num_files_per_subset),
                replace=False,
                p=[
                    np.exp(-v) / total_seen for v in seen_dataset_count.values()
                ],
            ).tolist()
            for ssn in subset_names:
                seen_dataset_count[ssn] += 1
            data_generator = super().load(
                standardized_dataset_category_names,
                filters=filters,
                post_filters=post_filters,
                prefiltered_names=subset_names,
                start_end_timestamps=start_end,
                complex_filter=complex_filter,
                post_complex_filter=post_complex_filter,
                existing_values_skip_dictionary=existing_values_skip_dictionary,
            )

            total_datasets_in_subset = sum(
                [name_column_dict[ssn] for ssn in subset_names]
            )
            all_variate_tuples = self._load_subset(data_generator)
            all_variate_tuples = np.array(all_variate_tuples, dtype=object)

            # import pickle
            # # pickle.dump(all_variate_tuples, open("all_variate_tuples.pkl", "wb"))
            # all_variate_tuples = pickle.load(open("all_variate_tuples.pkl", "rb"))

            seen_dataframe_count = {
                i: 0 for i in range(len(all_variate_tuples))
            }
            # sample more subsets from larger standardized datasets
            for i in range(
                int(self.sample_dataset_ratio * total_datasets_in_subset)
            ):
                # sample with probability proportional to the number of times we have seen the dataset, prioritizing less seen datasets at e^-v rate
                total_seen = sum(
                    [np.exp(-v) for v in seen_dataframe_count.values()]
                )
                subset_idxes = np.random.choice(
                    list(range(len(all_variate_tuples))),
                    self.num_variates_per_enrich,
                    replace=False,
                    p=[
                        np.exp(-v) / total_seen
                        for v in seen_dataframe_count.values()
                    ],
                )

                for ssn in subset_idxes:
                    seen_dataframe_count[ssn] += 1

                subset = all_variate_tuples[subset_idxes]

                # find the index with the dataframe that is longest
                longest_idx = np.argmax(
                    [len(df) for (name, df, meta) in subset]
                )
                name, loaded_df, meta = subset[longest_idx]
                series_seen_count = Counter()
                series_seen_count[loaded_df.columns[0]] += 1
                df, meta = loaded_df.copy(), copy.deepcopy(meta)
                meta["timestamps_columns"] = list()
                df, meta = self._rename_timestamp_duplicate_column(df, meta, 1)
                np.delete(subset, longest_idx)
                for var_name, variate_df, var_meta in subset:
                    # merge one vairiate at a time, fill back with NaN
                    df, meta = self._merge_variate(
                        df,
                        meta,
                        var_name,
                        variate_df,
                        var_meta,
                        series_seen_count,
                    )

                # merge metadata into data
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                    logger.info(
                        "had to convert series to dataframe, this is probably an error"
                    )
                for col in meta["columns"]:
                    assert isinstance(col, dict)
                    assert isinstance(col["column_id"], str)
                self.update_metadata_sizing(meta, df)
                logger.info(
                    f"saving with nulls {df[df.columns[0]].isna().sum()} in main"
                )

                yield (
                    name,
                    df,
                    meta,
                )  # note we just use the name of the longest dataframe
                datasets_saved_num += 1
                if datasets_saved_num >= self.num_datasets_to_save:
                    break


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

    loader = RandomLoader(args.config)
    if len(args.logfile) > 0:
        logger.add(args.logfile)

    import time

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
        start_end=config["time_period"],
        complex_filter=config["complex_filter"]
        if "complex_filter" in config
        else "",
        post_filters=config["post_filters"]
        if "post_filters" in config
        else None,
        post_complex_filter=config["post_complex_filter"]
        if "post_complex_filter" in config
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
