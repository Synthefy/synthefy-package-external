import copy
import json
import os
import shutil
import time
from typing import Generator

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# handles mapping from canonical frequency to frequency letter used in pandas
# TODO: when pre2processed (standardization) is merged, this should be imported
FREQUENCY_LETTER_MAP = {
    "Yearly": "Y",
    "Quarterly": "Q",
    "Monthly": "M",
    "Weekly": "W",
    "Daily": "D",
    "Hourly": "h",
}


def find_closing(s: str) -> int:
    """
    Helper function to process the complex filter string
    finds a closing parenthesis for a given opening parenthesis
    """
    count = 1
    for i, char in enumerate(s):
        if char == "(":
            count += 1
        elif char == ")":
            count -= 1
        if count == 0:
            return i
    return -1


class SimpleLoader:
    """
    loads data from the standardized format into pandas dataframes, where each dataframe has only a single data column and a timestamp column, and no other columns
    also saves the metadata corresponding to that dataframe, with the appropriate columns data and sizing in the metadata json
    Filtering is a logical operation on the sets datasets in standardized data, default AND
    """

    # Define ordered_list as a class variable
    ORDERED_FREQUENCIES = [
        "yearly",
        "quarterly",
        "monthly",
        "weekly",
        "daily",
        "hourly",
    ]

    def __init__(self, config_path: str):
        """
        config_path is a path to a descriptive yaml file to select which standardized data to load
        """
        if len(config_path) == 0:
            config = {}
            self.data_dir = ""
        else:
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.data_dir = config["input_data_dir"]
        self.config = config

    def _is_size_length_key(self, key: str) -> bool:
        """
        checks if the key is a size or length inequality key (inequality keys are special for numerical values)
        This helps check the key is correct before performing the operation
        """
        valid_patterns = ["size_leq", "size_geq", "length_leq", "length_geq"]
        return any(pattern in key for pattern in valid_patterns)

    def _check_size_length(
        self, size: int, length: int, key: str, value: str
    ) -> bool:
        """
        checks if the size and length of the dataframe satisfy the inequality key
        """
        if "size_leq" in key:
            return size <= int(value)
        elif "size_geq" in key:
            return size >= int(value)
        elif "length_leq" in key:
            return length <= int(value)
        elif "length_geq" in key:
            return length >= int(value)
        return False

    def _is_col_num_key(self, key: str) -> bool:
        """
        checks if the key is a number of columns inequality key
        """
        valid_patterns = ["num_columns_leq", "num_columns_geq"]
        return any(pattern in key for pattern in valid_patterns)

    def _check_col_num(self, num_columns: int, key: str, value: str) -> bool:
        """
        checks if the number of columns of the dataframe satisfies the inequality key
        """
        if "num_columns_leq" in key:
            return num_columns <= int(value)
        elif "num_columns_geq" in key:
            return num_columns >= int(value)
        return False

    def _is_frequency_key(self, key: str) -> bool:
        """
        checks if the key is a frequency inequality key
        """
        valid_patterns = ["frequency_leq", "frequency_geq", "frequency_eq"]
        return any(pattern in key for pattern in valid_patterns)

    def _check_frequency(self, frequency: str, key: str, value: str) -> bool:
        """
        checks if the frequency of the dataframe satisfies the inequality key
        """
        # just in case the capitalization is wrong
        frequency = frequency.lower()
        value = value.lower()

        # check if the frequency is in the accepted canonical frequency list: hourly, daily, weekly, monthly, quarterly, yearly
        if value not in self.ORDERED_FREQUENCIES:
            raise ValueError(
                f"Frequency {value} not in accepted canonical frequency list {self.ORDERED_FREQUENCIES}"
            )
        if frequency not in self.ORDERED_FREQUENCIES:
            return False

        # check if the frequency satisfies the inequality key
        if key.find("frequency_leq") != -1:
            if self.ORDERED_FREQUENCIES.index(
                frequency
            ) <= self.ORDERED_FREQUENCIES.index(value):
                return True
        elif key.find("frequency_geq") != -1:
            if self.ORDERED_FREQUENCIES.index(
                frequency
            ) >= self.ORDERED_FREQUENCIES.index(value):
                return True
        elif key.find("frequency_eq") != -1:
            if frequency == value:
                return True
        return False

    def _check_column_keywords(
        self, final_columns: list[dict], keywords: list[str]
    ) -> bool:
        """
        checks if any keywords are found in the title or description of any of the final_columns
        This requires iterating through each column in final_columns, then checking each keyword (checks all keywords)
        column keywords must be a list to work properly
        """
        for column in final_columns:
            title = str(column["title"]).lower()
            description = str(column["description"]).lower()
            for keyword in keywords:
                keyword = str(keyword).lower()
                if keyword in title or keyword in description:
                    return True
        return False

    def _stringify_filter(self, key: str, value: str | list[str]) -> str:
        """
        creates a string key for a filter
        """
        return (
            key + "_" + str(value)
            if not isinstance(value, list)
            else key + "_" + "_".join(value)
        )

    def _run_filters(
        self,
        filename: str,
        metadata: dict,
        filters: list[dict[str, str | list[str]]],
    ) -> dict[str, list[str]]:
        """
        runs all filters on the metadata for a single standardized dataset
        if a particular filter is met, will append to all_filter_filenames[filter_key_value]
        This is so that AND/OR operations can be applied on the filenames
        @param all_filter_filenames: a dictionary of filter key values to a list of filenames that match the filter (possible iterated over multiple datasets)
        @param filename: the filename of the standardized dataset
        @param metadata: the metadata of the standardized dataset
        @param filters: the filters to check the metadata against, key: value or list of values (ex. length_leq: 100)
        """
        filter_filename_map: dict[str, list[str]] = dict()
        for fvp in filters:
            # iterate over all filters, where each filter is a dictionary with a single key value pair (because we loaded a yaml)
            for key, value in fvp.items():
                # create a key for  the filter by combining with underscores
                filter_key = self._stringify_filter(key, value)
                # if the filter key is not already in the dictionary, add it
                if filter_key not in filter_filename_map:
                    filter_filename_map[filter_key] = list()
                # we need to check specialized keys that don't follow the normal format first:
                # check size and length
                if self._is_size_length_key(key):
                    if self._check_size_length(
                        int(metadata["size"]),
                        int(metadata["length"]),
                        key,
                        str(value),
                    ):
                        filter_filename_map[filter_key].append(filename)
                # check number of columns
                elif self._is_col_num_key(key):
                    if self._check_col_num(
                        int(metadata["num_columns"]), key, str(value)
                    ):
                        filter_filename_map[filter_key].append(filename)
                # check column keywords
                elif key == "column_keywords":
                    keywords = [
                        str(v)
                        for v in (value if isinstance(value, list) else [value])
                    ]
                    if self._check_column_keywords(
                        metadata["final_columns"], keywords
                    ):
                        filter_filename_map[filter_key].append(filename)
                # check final columns (TODO: I think this does the same thing as column_keywords)
                elif "final_columns" in key:
                    subcategory = key.split("_")[2]
                    for category in metadata["final_columns"]:
                        if category[subcategory].find(value) != -1:
                            filter_filename_map[filter_key].append(filename)
                # check frequency
                elif self._is_frequency_key(key):
                    if self._check_frequency(
                        metadata["frequency"], key, str(value)
                    ):
                        filter_filename_map[filter_key].append(filename)
                # check all other keys (canonical format)
                else:
                    if isinstance(metadata[key], str):
                        if metadata[key].lower().find(str(value).lower()) != -1:
                            filter_filename_map[filter_key].append(filename)
                    else:  # assume that non string is equivalent to float
                        if isinstance(value, list):
                            if metadata[key] in value:
                                filter_filename_map[filter_key].append(filename)
                        else:
                            if metadata[key] == float(value):
                                filter_filename_map[filter_key].append(filename)
        return filter_filename_map

    def _assign_filenames_to_filters(
        self,
        pth: str,
        filters: list[dict[str, str | list[str]]] | None,
    ) -> dict[str, list[str]] | list[str]:
        """
        computes the filenames that match the filters
        @param pth: the path to the directory to search for standardized datasets
        @param all_filter_filenames: a dictionary of filter key values to a list of filenames that match the filter
        @param filters: the filters to check the metadata against
        TODO: parallelize this, since there is an internal load operation needed to get the metadata
        """
        if filters is None:
            all_filter_filenames = list()
        else:
            all_filter_filenames = dict()
        # for every file in the directory, load the metadata and run the filters
        for root, dirs, filenames in os.walk(pth):
            for filename in filenames:
                # only load metadata files, not the parquet files
                if filename.endswith(".json"):
                    full_path = os.path.join(root, filename)
                    metadata = json.load(open(os.path.join(full_path), "r"))
                    filename = full_path.replace("_meta.json", "").replace(
                        "_metadata.json", ""
                    )
                    # if filters is None, all_filter_filenames is a list, not a dict, and we just append everything
                    if filters is None and isinstance(
                        all_filter_filenames, list
                    ):
                        all_filter_filenames.append(filename)
                    elif filters is None:
                        raise ValueError(
                            "Filters is None, but all_filter_filenames is not a list"
                        )
                    elif isinstance(all_filter_filenames, dict):
                        # this will add the filename to the filters that it passes
                        logger.info(
                            f"checking {filename} length {metadata['length']} size {metadata['size']} added? {sum([len(all_filter_filenames[key]) for key in all_filter_filenames])}"
                        )
                        filename_filter_mapping = self._run_filters(
                            filename, metadata, filters
                        )
                        for key in filename_filter_mapping.keys():
                            if key not in all_filter_filenames:
                                all_filter_filenames[key] = list()
                            all_filter_filenames[key] += (
                                filename_filter_mapping[key]
                            )
                    else:
                        raise ValueError(
                            "All_filter_filenames is not a list or dict or is a list and filters is not None"
                        )
        return all_filter_filenames

    def _apply_OR_filter(self, filter_names: list[set[str]]) -> set[str]:
        """
        helper function for complex filter logic, computes the union of the filenames for a set of filter names
        @param filter_names: a list of sets of filenames that the filter passes
        output: the set of all filenames that are in the given filters
        """
        # Use set union to efficiently combine all sets
        if not filter_names:
            return set()
        return set().union(*filter_names)

    def _apply_AND_filter(self, filter_names: list[set[str]]) -> set[str]:
        """
        helper function for complex filter logic, computes the intersection of the filenames for a set of filter names
        @param filter_names: a list of sets of filenames that match the filters
        output: the set of all filenames that are in all the given filters
        """
        all_filenames = set()
        for filter_set in filter_names:
            for value in filter_set:
                if np.all([value in fs for fs in filter_names]):
                    all_filenames.add(value)
        return all_filenames

    def _find_next_operator(self, remaining_filter: str) -> int:
        """
        Helper function for complex filter logic, finds the next operator (&,|) in the complex filter
        @param remaining_filter: the remaining filter to parse
        @return: the index of the next operator in the remaining filter, or the length of the string if no operator found
        """
        # Find positions of both operators
        and_pos = remaining_filter.find("&")
        or_pos = remaining_filter.find("|")

        # Handle cases where one or both operators are not found
        if and_pos == -1 and or_pos == -1:
            return len(remaining_filter)
        elif and_pos == -1:
            return or_pos
        elif or_pos == -1:
            return and_pos

        # Return the position of the first operator found
        return min(and_pos, or_pos)

    def _compute_filter_logic(
        self, complex_filter: str, all_filter_filenames: dict[str, list[str]]
    ) -> set[str]:
        """
        parses strings of the form (FILTER NAME A|FILTER NAME B)&(FILTER NAME C|FILTER NAME D), ignoring whitespace.
        the keys of all_filter_filenames are the filter names, and have the format: filter_key + "_" + str(filter_value)
        all operators are replaced with the set of filenames that match the operator
        the result is the set of filenames that match the complex filter
        @param complex_filter: the complex filter to parse
        @param all_filter_filenames: a dictionary of filter key values to a list of filenames that match the filter
        output: the set of all filenames that match the complex filter

        format: key_filter<operator>key_filter<operator>key_filter... Parentheses can replace key_filter with a complex filter
        Complex filter example: ~title_CPI&(length_geq_100|size_geq_10000)
        """
        complex_filter = complex_filter.strip(" ")  # strip whitespace
        # filter names is a list of the filters in order that they occur in the logical statement
        filter_names = list()
        operator_type = None
        remaining_filter = complex_filter

        # useful for computing negation (by subtracting from all_possible_filenames)
        all_possible_filenames = set(
            sum(
                [
                    list(all_filter_filenames[fn])
                    for fn in all_filter_filenames.keys()
                ],
                start=list(),
            )
        )

        # while there is a remaining filter, parse it
        while len(remaining_filter) > 0:
            # if the first character is a left parenthesis, compute the filter logic of the interior of the filter, then skip
            if remaining_filter[0] == "(":
                closing_bracket = find_closing(remaining_filter[1:])
                filter_names.append(
                    self._compute_filter_logic(
                        remaining_filter[1 : closing_bracket + 1],
                        all_filter_filenames,
                    )
                )
                remaining_filter = remaining_filter[closing_bracket + 2 :]
            # if the first character is a pipe, set the operator to OR
            elif remaining_filter[0] == "|":
                operator_type = "|"
                remaining_filter = remaining_filter[1:]
            # if the first character is an ampersand, set the operator to AND
            elif remaining_filter[0] == "&":
                operator_type = "&"
                remaining_filter = remaining_filter[1:]
            # if the first character is a tilde, compute the negation of the interior of the filter
            # negations can only handle parentheses or individual filters.
            elif remaining_filter[0] == "~":
                # if the next character is a left parenthesis, compute the negation of the interior of the filter
                # this operation is recursive
                if remaining_filter[1] == "(":
                    closing_bracket = find_closing(remaining_filter[2:])
                    filtered_filenames = self._compute_filter_logic(
                        remaining_filter[2 : closing_bracket + 2],
                        all_filter_filenames,
                    )
                    filter_names.append(
                        all_possible_filenames - filtered_filenames
                    )
                    remaining_filter = remaining_filter[closing_bracket + 3 :]
                else:
                    # if the next character is not a left parenthesis, compute the negation of the filter by finding the next operator
                    # since it cannot be a parenthesis, there can only be a filter value between now and the next operator.
                    next_operator = self._find_next_operator(remaining_filter)
                    filter_names.append(
                        all_possible_filenames
                        - self._compute_filter_logic(
                            remaining_filter[1 : next_operator + 1],
                            all_filter_filenames,
                        )
                    )
                    remaining_filter = remaining_filter[next_operator + 1 :]
            else:  # we found the first character of a filter name
                # find the next operator in the remaining filter
                next_operator = self._find_next_operator(remaining_filter)
                filter_names.append(
                    all_filter_filenames[remaining_filter[:next_operator]]
                )
                remaining_filter = remaining_filter[next_operator:]
        # perform the operator (and or or) over the filter names
        if operator_type == "&":
            filtered_filenames = self._apply_AND_filter(filter_names)
        elif operator_type == "|":
            filtered_filenames = self._apply_OR_filter(filter_names)
        else:
            filtered_filenames = (
                filter_names[0]
                if isinstance(filter_names[0], set)
                else set(filter_names[0])
            )
        return filtered_filenames

    def _apply_complex_filter(
        self, all_filter_filenames: dict[str, list[str]], complex_filter: str
    ) -> list[str]:
        """
        applies a complex filter to the filenames, or an AND filter if the complex filter is empty
        returns a list of unique filenames to load data from
        """
        # convert the list of filenames to a set of filenames
        for key, value in all_filter_filenames.items():
            all_filter_filenames[key] = list(
                set(value)
            )  # Convert to list after set operation

        if len(complex_filter) > 0:
            all_filenames = list(
                self._compute_filter_logic(complex_filter, all_filter_filenames)
            )
        else:  # assume AND filtering
            all_filenames = list()
            # We have to check against all keys because of AND, so it doesn't matter which key's values we iterate (uses the last key)
            for value in all_filter_filenames[
                list(all_filter_filenames.keys())[-1]
            ]:
                if np.all(
                    [
                        value in all_filter_filenames[key]
                        for key in all_filter_filenames.keys()
                    ]
                ):
                    all_filenames.append(value)
        return all_filenames

    def _get_filtered_filenames_to_read(
        self,
        standardized_dataset_category_names: list[str],
        filters: list[dict[str, str | list[str]]] | None,
        complex_filter: str,
    ) -> list[str]:
        """
        gets the filenames from all the standardized_dataset_category_name (top folders in the data_dir) that match the filters
        @param standardized_dataset_category_name: a list of standardized_dataset_category_names to filter
        @param filters: a list of filters to apply to the filenames
        @param complex_filter: a complex filter to apply to the filenames
        output: a list of filenames that match the filters
        """
        # retain only filenames that are in all filter filenames
        if filters is None:
            filename_list = self._assign_filenames_to_filters(
                self.data_dir, filters
            )  # this is just all filenames
            # this has to be checked AFTER return to resolve type checkers
            if filters is None and isinstance(filename_list, list):
                return filename_list
            else:
                raise ValueError("Output is not a list")

        # initialize the all_filter_filenames dictionary
        all_filter_filenames: dict[str, list[str]] = dict()
        for (
            standardized_dataset_category_name
        ) in standardized_dataset_category_names:
            if standardized_dataset_category_name == "ALL":
                # merge the two filename dictionaries
                new_filenames = self._assign_filenames_to_filters(
                    self.data_dir, filters
                )
                if isinstance(new_filenames, dict):
                    all_filter_filenames = {
                        **all_filter_filenames,
                        **new_filenames,
                    }
                else:
                    raise ValueError("new_filenames is not a dict")
            else:
                new_filenames = self._assign_filenames_to_filters(
                    os.path.join(
                        self.data_dir, standardized_dataset_category_name
                    ),
                    filters,
                )
                if isinstance(new_filenames, dict):
                    all_filter_filenames = {
                        **all_filter_filenames,
                        **new_filenames,
                    }
                else:
                    raise ValueError(
                        "New filenames is not a dict", type(new_filenames)
                    )
        if len(all_filter_filenames) == 0:
            return list()

        # filter the dict of filenames according to the complex filter
        all_filenames = self._apply_complex_filter(
            all_filter_filenames, complex_filter
        )
        all_filenames.sort()
        return all_filenames

    def _apply_post_filters(
        self,
        df: pd.DataFrame,
        name: str,
        meta: dict,
        post_filters: list[dict[str, str | list[str]]] | None = None,
        post_complex_filter: str = "",
    ) -> tuple[list[pd.DataFrame], list[str], list[dict]]:
        """
        applies the post filters to the dataframe
        TODO: clean up the logic, it was modified from treating the inputs as lists of dataframes, names and metadatas
        """
        name_df_meta = {name: (df, meta)}
        if post_filters is None or len(post_filters) == 0:
            return [df], [name], [meta]
        cropped_dfs = list()
        cropped_names = list()
        cropped_metas = list()

        all_filter_names = dict()
        if post_filters is None:
            return [df], [name], [meta]
        for fvp in post_filters:
            # TODO: right now only does AND filtering
            # after removing NAN values, filter again by size and length
            # if a category filter, split and use the category, otherwise just filter assuming the key is in the metadata
            for key, value in fvp.items():
                filter_key = self._stringify_filter(key, value)
                if filter_key not in all_filter_names:
                    all_filter_names[filter_key] = list()
                if self._is_size_length_key(key):
                    if self._check_size_length(
                        len(df.columns) * len(df), len(df), key, str(value)
                    ):
                        all_filter_names[filter_key].append(name)

        filenames_to_load = self._apply_complex_filter(
            all_filter_names, post_complex_filter
        )
        for name in filenames_to_load:
            cropped_dfs.append(name_df_meta[name][0])
            cropped_names.append(name)
            cropped_metas.append(name_df_meta[name][1])
        return cropped_dfs, cropped_names, cropped_metas

    def _compute_total_columns(
        self, names_to_open: list[str]
    ) -> tuple[dict[str, int], int]:
        total_columns = 0
        name_column_dict = dict()
        for name in names_to_open:
            try:
                meta = json.load(open(name + "_metadata.json", "r"))
            except FileNotFoundError:
                meta = json.load(open(name + "_meta.json", "r"))
                logger.warning(f"metadata file misnamed for {name}")
            name_column_dict[name] = meta["num_columns"]
            total_columns += meta["num_columns"]
        return name_column_dict, total_columns

    def update_metadata_sizing(
        self, dataset_metadata: dict, datas: pd.DataFrame
    ) -> dict:
        """
        assigns the size, length, and num_columns to the metadata
        typically run after slicing the dataframe columns into datas
        """
        dataset_metadata["size"] = datas.shape[0] * datas.shape[1]
        dataset_metadata["length"] = len(datas)
        dataset_metadata["num_columns"] = len(datas.columns)
        # rename final_columns to columns
        if "final_columns" in dataset_metadata:
            dataset_metadata["columns"] = dataset_metadata["final_columns"]
            del dataset_metadata["final_columns"]
        for i in range(1, len(dataset_metadata["columns"])):
            dataset_metadata["columns"][i]["is_metadata"] = "yes"
        dataset_metadata["timestamp_columns"] = ["timestamp"]

        # right now all columns are continuous, but updated loaders will need different logic
        dataset_metadata["num_continuous_columns"] = len(datas.columns)
        dataset_metadata["num_metadata_columns"] = len(datas.columns) - 1
        return dataset_metadata

    def load_names_to_open(
        self,
        standardized_dataset_category_names: list[str],
        filters: list[dict[str, str | list[str]]] | None = None,
        complex_filter: str = "",
    ) -> list[str]:
        """
        Loads the names of all files that match the filters, but NOT the prefilters
        Since all these files need to be opened, this allows for splitting of the filenames for parallel loading

        @param dataname: the name of the dataset to load
        @param filters: the filters to apply to the dataset
        @param complex_filter: the complex filter to apply to the dataset
        @return: the list of filenames that match the filters
        """
        filenames_to_open = self._get_filtered_filenames_to_read(
            standardized_dataset_category_names, filters, complex_filter
        )

        logger.info(f"all names {filenames_to_open} {len(filenames_to_open)}")
        return filenames_to_open

    def generate_existing_values_skip_dict(
        self, outname: str
    ) -> dict[str, int]:
        """
        generates a dictionary of the filenames already saved and their rows by loading all the existing metadata files and recording their name and skip index
        @param outname: the name of the output directory where we will save files to if they don't already exist
        """
        existing_values_skip_dict = dict()
        for name in os.listdir(
            os.path.join(self.config["output_dir"], outname)
        ):
            meta = json.load(
                open(
                    os.path.join(
                        self.config["output_dir"],
                        outname,
                        name,
                        name + "_metadata.json",
                    ),
                    "r",
                )
            )
            existing_values_skip_dict[meta["source_name"]] = max(
                existing_values_skip_dict.get(meta["source_name"], -1),
                meta["original_column_id"],
            )
        return existing_values_skip_dict

    def _apply_rounding_to_index(
        self, df: pd.DataFrame, frequency: str
    ) -> pd.DataFrame:
        """
        round the index to the nearest frequency
        """
        tz = (
            df.index.tz
            if isinstance(df.index, pd.DatetimeIndex)
            and df.index.tz is not None
            else "UTC"
        )
        df.index = df.index.map(
            lambda x: pd.Timestamp(
                year=(
                    x.year + (1 if x.month >= 7 else 0)  # Round to nearest year
                    if str(frequency) in ["Yearly"]
                    else x.year
                ),
                month=(
                    1
                    if str(frequency) in ["Yearly"]
                    else (
                        (x.month - 1) // 3 * 3 + 1
                        if str(frequency) == "Quarterly"
                        else (
                            x.month + (1 if x.day >= 16 else 0)
                        )  # Round to nearest month
                        if str(frequency) in ["Monthly"]
                        else x.month
                    )
                ),
                day=(
                    1
                    if str(frequency) in ["Monthly", "Quarterly", "Yearly"]
                    else x.day
                ),
                hour=(
                    0
                    if str(frequency) not in ["Hourly", "Minutely", "Secondly"]
                    else (x.hour + (1 if x.minute >= 30 else 0))
                ),  # Round to nearest hour
                minute=(
                    0
                    if str(frequency) not in ["Minutely", "Secondly"]
                    else x.minute
                ),
                second=(0 if str(frequency) not in ["Secondly"] else x.second),
                microsecond=0,
                tz=tz,
            )
        )
        return df

    def load(
        self,
        standardized_dataset_category_names: list[str] = [],
        filters: list[dict[str, str | list[str]]] | None = None,
        post_filters: list[dict[str, str | list[str]]] | None = None,
        start_end_timestamps: dict[str, str] | None = None,
        prefiltered_names: list[str] | None = None,
        complex_filter: str = "",
        post_complex_filter: str = "",
        include_metadata: bool = False,
        existing_values_skip_dictionary: dict[str, int] = {},
    ) -> Generator[tuple[str, pd.DataFrame, dict], None, None]:
        """
        return a generator of dataframes for each series (independent)
        @param filters is a list of dictionaries of metadata key and filter value.
        @param post_filters is a list of dictionaries of metadata key and filter value applied after slicing (only length and size operations allowed)
        @param start_end_timestamps is a dictionary of start and end dates to filter the data by
        @param prefiltered_names is a subset of filenames (skips filtering)
        @param complex_filter is a string of the form (FILTER NAME A|FILTER NAME B)&(FILTER NAME C|FILTER NAME D) to filter the data by
        """
        # get the filenames that match the filters
        if prefiltered_names is None:
            filenames_to_open = self._get_filtered_filenames_to_read(
                standardized_dataset_category_names, filters, complex_filter
            )
        else:
            filenames_to_open = prefiltered_names

        # if there are no filenames, return empty
        if len(filenames_to_open) == 0:
            return

        logger.info(f"all names {filenames_to_open} {len(filenames_to_open)}")
        # set return format
        # break apart all columns in all dataframes into separate dataframes, and cut off NAN values at the beginning and end
        # TODO: below should be with parallelism
        for name in filenames_to_open:
            try:
                meta = json.load(open(name + "_metadata.json", "r"))
            except FileNotFoundError:
                meta = json.load(open(name + "_meta.json", "r"))
                logger.warning(f"metadata file misnamed for {name}")
            df = pd.read_parquet(name + ".parquet")
            logger.info(
                f"starting load {name} {df.shape} column: count {meta['num_columns']} true {len(df.columns)} info {len(meta['final_columns'])}"
            )
            # check if the index is a subset of datetimes
            if not isinstance(
                df.index,
                (
                    pd.DatetimeIndex,
                    pd.PeriodIndex,
                    pd.TimedeltaIndex,
                    pd.Timestamp,
                ),
            ):
                # convert timestamp column to datetime index
                df = df.set_index(pd.to_datetime(df["timestamp"]))
            seen_timestamp = False
            for i, column in enumerate(df.columns):
                if (
                    name in existing_values_skip_dictionary
                    and i <= existing_values_skip_dictionary[name]
                ):
                    yield (name, pd.DataFrame(), dict())
                    continue
                # slice column to remove NAN values at the beginning and end
                first_valid = df[column].first_valid_index()
                last_valid = df[column].last_valid_index()
                # create a new dataframe with the indicies retaining the column name
                if first_valid is None and last_valid is None:
                    logger.info(f"skipping {name}, {column}, no valid values")
                    continue
                new_df = pd.DataFrame(df[column].loc[first_valid:last_valid])

                # skip any columns that don't contain numeric data (should only be skipping timestamp)
                if new_df[column].dtype not in [
                    "float64",
                    "int64",
                    "float32",
                    "int32",
                ] or (
                    (not include_metadata)
                    and (
                        meta["final_columns"][i - int(seen_timestamp)][
                            "is_metadata"
                        ]
                        != "no"
                    )
                ):
                    logger.info(
                        f"skipping {name}, {column}, {new_df[column].dtype}"
                    )
                    seen_timestamp = (
                        (column == "timestamp") or seen_timestamp
                    )  # either it is the timestamp or we have already seen it
                    continue
                idx = i - int(seen_timestamp)
                logger.info(f"loading {idx} seen timestamp {seen_timestamp}")
                metacol = meta["final_columns"][idx]
                frequency = meta["frequency"]
                # interpolate any missing values with NaN at the given frequency, then slice by start and end dates
                if self.config["nan_fill_frequency"]:
                    new_df = (
                        new_df.resample(FREQUENCY_LETTER_MAP[frequency])
                        .mean()
                        .fillna(np.nan)
                    )
                if start_end_timestamps is not None:
                    new_df = new_df[
                        pd.to_datetime(new_df.index, utc=True)
                        >= pd.to_datetime(
                            start_end_timestamps["start"], utc=True
                        )
                    ]
                    new_df = new_df[
                        pd.to_datetime(new_df.index, utc=True)
                        <= pd.to_datetime(start_end_timestamps["end"], utc=True)
                    ]
                new_df = pd.DataFrame(new_df)
                new_df.index.rename("index", inplace=True)
                new_df["timestamp"] = (
                    new_df.index
                )  # ensure there is a timestamp column

                nm = copy.deepcopy(meta)
                nm["final_columns"] = [metacol]
                nm = self.update_metadata_sizing(nm, new_df)
                # change the title and description to match that column information
                # (TODO: we might consider combining the title and description rather than replacing)
                nm["title"] = metacol["title"]
                nm["description"] = metacol["description"]
                nm["original_column_id"] = idx
                nm["source_name"] = name
                # applies the post filters (which return lists)
                new_df_lst, name_lst, nm_lst = self._apply_post_filters(
                    new_df, name, nm, post_filters, post_complex_filter
                )
                if len(new_df_lst) > 0:
                    yield (name_lst[0], new_df_lst[0], nm_lst[0])
                else:
                    logger.info(f"filtered out {name} {new_df.shape}")

    def _save_single_parquet(
        self,
        name: str,
        df: pd.DataFrame,
        meta: dict,
        outname: str,
        i: int,
        replace_existing: bool,
    ):
        """
        save a single parquet file

        @param name: the name of the file to save
        @param df: the dataframe to save
        @param meta: the metadata to save
        @param outname: the name of the output directory
        @param i: the index of the file to save
        @param replace_existing: whether to replace the existing file
        """
        os.makedirs(
            os.path.join(
                self.config["output_dir"],
                outname,
                f"{outname}_{i}",
            ),
            exist_ok=True,
        )
        if not replace_existing:
            if os.path.exists(
                os.path.join(
                    self.config["output_dir"],
                    outname,
                    outname + "_" + str(i),
                    outname + "_" + str(i) + ".parquet",
                )
            ):
                logger.info(
                    f"skipping because dataset already exists {name} {df.shape}"
                )
                return
        df.to_parquet(
            os.path.join(
                self.config["output_dir"],
                outname,
                outname + "_" + str(i),
                outname + "_" + str(i) + ".parquet",
            ),
        )
        with open(
            os.path.join(
                self.config["output_dir"],
                outname,
                outname + "_" + str(i),
                outname + "_" + str(i) + "_metadata.json",
            ),
            "w",
        ) as f:
            json.dump(meta, f, indent=4)
        logger.info(
            f"saved {name} {meta['title']} {df.shape} {os.path.join(self.config['output_dir'], outname, outname + '_' + str(i))}"
        )

    def save_parquet(
        self,
        load_generator: Generator[tuple[str, pd.DataFrame, dict], None, None],
        outname: str,
        clean_dir: bool = False,
        replace_existing: bool = True,
        start_index: int = 0,
    ):
        # randomly select train and test dataframes from the list of dataframes
        if clean_dir and os.path.exists(
            os.path.join(self.config["output_dir"], outname)
        ):
            shutil.rmtree(os.path.join(self.config["output_dir"], outname))
        os.makedirs(
            os.path.join(self.config["output_dir"], outname), exist_ok=True
        )
        # TODO: would be more efficient if it skipped existing files BEFORE loading them
        for i, (name, df, meta) in enumerate(load_generator):
            if df.empty:
                logger.info(f"skipping existing file {name}")
                continue
            self._save_single_parquet(
                name, df, meta, outname, i + start_index, replace_existing
            )


if __name__ == "__main__":
    # loading options from a yaml file provided as a command line argument
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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
    parser.add_argument(
        "--outname", type=str, help="name of the output directory", default=""
    )

    args = parser.parse_args()

    loader = SimpleLoader(args.config)
    config = loader.config
    if len(args.logfile) > 0:
        logger.add(args.logfile)

    start = time.time()
    # load and save dataframes
    if args.no_replace_existing:
        existing_values_skip_dictionary = (
            loader.generate_existing_values_skip_dict(args.outname)
        )
    else:
        existing_values_skip_dictionary: dict[str, int] = dict()
    load_generator = loader.load(
        config["standardized_dataset_category_names"],
        filters=config["time_series_filters"],
        post_filters=config["post_filters"],
        prefiltered_names=None,
        start_end_timestamps=config["time_period"],
        complex_filter=config["complex_filter"]
        if "complex_filter" in config
        else "",
        include_metadata=False,
        post_complex_filter=config["post_complex_filter"]
        if "post_complex_filter" in config
        else "",
        existing_values_skip_dictionary=existing_values_skip_dictionary,
    )
    # save dataframes
    loader.save_parquet(
        load_generator,
        outname=args.outname,
        clean_dir=args.clean_dir,
        replace_existing=not args.no_replace_existing,
    )
    end = time.time()
    logger.info(f"time taken single {end - start}")
