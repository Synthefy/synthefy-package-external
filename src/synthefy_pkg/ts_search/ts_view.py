import json
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from loguru import logger
from pydantic import BaseModel

from synthefy_pkg.data.synthefy_dataset import SynthefyDataset
from synthefy_pkg.utils.scaling_utils import (
    SCALER_FILENAMES,
    transform_using_scaler,
)

COMPILE = True

# GPT 4:
# llm = AzureChatOpenAI(
#     azure_endpoint="https://synthefy-synthesis.openai.azure.com/",
#     openai_api_key="98c35eba36fa48d2b61bbdf15b887311",
#     model_name="gpt-4",
#     azure_deployment="Synthefy_GPT4",
#     api_version="2024-02-15-preview",
# )
# GPT 4 turbo:
# llm = AzureChatOpenAI(
#     azure_endpoint="https://synthefy-synthesis.openai.azure.com/",
#     openai_api_key="98c35eba36fa48d2b61bbdf15b887311",
#     model_name="gpt-4",
#     azure_deployment="Synthefy_GPT_Turbo",
#     api_version="2024-02-15-preview",
# )
# GPT 4o:
llm = AzureChatOpenAI(
    azure_endpoint="https://synthefy-synthesis.openai.azure.com/",
    api_key="98c35eba36fa48d2b61bbdf15b887311", # type: ignore
    model="gpt-4o",
    azure_deployment="SynthefyGPT4o",
    api_version="2024-02-15-preview",
)
# GPT 3.5:
# llm = AzureChatOpenAI(
#     azure_endpoint="https://synthefy-synthesis.openai.azure.com/",
#     openai_api_key="98c35eba36fa48d2b61bbdf15b887311",
#     model_name="gpt-35-turbo-16k",
#     azure_deployment="SynthefyGPT3",
#     api_version="2024-02-15-preview",
# )


# TODO support strict LESS than/GREATER than instead of geq/leq
class Statistics(BaseModel):
    gt: Optional[float] = None
    lt: Optional[float] = None
    geq: Optional[float] = None
    leq: Optional[float] = None
    eq: Optional[float] = None

    ignore_zeroes: Optional[bool] = False


class ChannelStatistics(BaseModel):
    mean: Optional[Statistics] = None
    min: Optional[Statistics] = None
    max: Optional[Statistics] = None
    sd: Optional[Statistics] = None
    median: Optional[Statistics] = None

    percentile: Optional[Statistics] = None


class OrderBy(BaseModel):
    direction: Optional[Literal["asc", "desc"]] = None
    priority: Optional[int] = None


class ColumnInfo(BaseModel):
    statistics: Optional[ChannelStatistics] = None
    order_by: Optional[OrderBy] = None


class DiscreteColumnInfo(BaseModel):
    including: Optional[List[str]] = None
    excluding: Optional[List[str]] = None


class TimestampInfo(BaseModel):
    geq: Optional[str] = None
    leq: Optional[str] = None
    eq: Optional[str] = None
    order_by: Optional[OrderBy] = None
    relative_time: Optional[dict] = None


class TimeSeriesViewQuery(BaseModel):
    n_windows_to_return: Optional[int] = 5
    timeseries_columns: Optional[Dict[str, ColumnInfo]] = None
    continuous_columns: Optional[Dict[str, ColumnInfo]] = None
    discrete_columns: Optional[Dict[str, DiscreteColumnInfo]] = None
    timestamps: Optional[Dict[str, TimestampInfo]] = None


def unscale_data(data: np.ndarray, scaler_file_path: str) -> np.ndarray:
    """
    Scales timeseries/continuous conditions back to original.
    Parameters:
        - data: 3D np.array, dims for both types - (num_windows, windows_size, num_features)
        - scalers: dict containing scaler for each feature col of data.
    Returns:
        - scaled back data of the same size
    """
    scalers = pickle.load(
        open(
            scaler_file_path,
            "rb",
        )
    )
    if scalers is None or len(scalers) == 0:
        return data
    num_windows, _, num_features = data.shape
    for feature_idx in range(num_features):
        for window_idx in range(num_windows):
            # reshapes the data of shape (window_size,) to (window_size, 1) before inverse_transform,
            # then reshapes back to (window_size,) to put it back in the original window
            data[window_idx, :, feature_idx] = (
                scalers[feature_idx]
                .inverse_transform(data[window_idx, :, feature_idx].reshape(-1, 1))
                .reshape(-1)
            )
    return data


class TimeSeriesView:
    dataset: SynthefyDataset

    def __init__(
        self,
        config_path: str,
        is_unittest: bool = False,
        precompute_stats: bool = False,
    ) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        self.config_path = config_path
        self.precompute_stats = precompute_stats
        if not is_unittest:
            self.dataset = SynthefyDataset(config_source=config_path)
            self.dataset_name = self.dataset.config["filename"].split("/")[0]
            self.view_df_save_path = (
                f"/tmp/{self.dataset_name}_df_for_view_ui_saved.parquet"
            )

            self.dataset.load_windows(
                window_types=[
                    "timestamp",
                    "timeseries",
                    "continuous",
                    "original_discrete",
                ]
            )
            self.dataset.load_columns_and_timestamps()
            self.dataset.load_original_discrete_colnames()

            # unscale and decode all data
            logger.info("Scaling back continuous conditions")
            self.dataset.windows_data_dict["continuous"]["windows"] = transform_using_scaler(
                self.dataset.windows_data_dict["continuous"]["windows"],
                timeseries_or_continuous="continuous",
                dataset_name = self.dataset.output_path.split("/")[-1],
            )
            logger.info("Scaling back timeseries")

            self.dataset.windows_data_dict["timeseries"]["windows"] = transform_using_scaler(
                self.dataset.windows_data_dict["continuous"]["windows"],
                timeseries_or_continuous="timeseries",
                dataset_name = self.dataset.output_path.split("/")[-1],
            )

            logger.info("Loading original discrete data")
            self.original_discrete_cols = self.dataset.original_discrete_cols

            self.window_size = self.dataset.window_size

            self.timestamps_data_np = self.dataset.windows_data_dict["timestamp"][
                "windows"
            ]
            self.timeseries_data_np = self.dataset.windows_data_dict["timeseries"][
                "windows"
            ]
            self.continuous_data_np = self.dataset.windows_data_dict["continuous"][
                "windows"
            ]
            self.discrete_data_np = self.dataset.windows_data_dict["original_discrete"][
                "windows"
            ]

            self.timeseries_cols = self.dataset.timeseries_cols
            self.continuous_cols = self.dataset.continuous_cols
            self.timestamps_col = self.dataset.timestamps_col

            logger.info("Converting to df")
            self.df = self.npdata_to_df()
            if self.precompute_stats:
                self.precomputed_stats = self.compute_statistics(
                    self.df[
                        ["window_idx"]
                        + self.dataset.timeseries_cols
                        + self.dataset.continuous_cols
                    ]
                )

    def compute_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for the entire DataFrame.
        """
        logger.info("Computing stats")
        aggregated_stats_df = df.groupby("window_idx").agg(
            ["mean", "min", "max", "std", "median"]
        )
        assert isinstance(aggregated_stats_df, pd.DataFrame), "Expected a DataFrame"
        aggregated_stats: Dict[str, pd.DataFrame] = {
            col: aggregated_stats_df[col].reset_index().set_index("window_idx")
            for col in df.columns
            if col != "window_idx"
        }
        stats_df = pd.concat(
            [
                aggregated_stats[col].add_suffix(f"_{col}")
                for col in df.columns
                if col != "window_idx"
            ],
            axis=1,
        ).reset_index()

        stats_df.columns = stats_df.columns.str.replace(".", "_", regex=False)
        return stats_df

    def clean_query(self, query: TimeSeriesViewQuery) -> TimeSeriesViewQuery:
        """
        Cleans the query by removing columns with None values for both statistics and order_by.
        Removes columns that are not part of the original dataset (timeseries, continuous, discrete, timestamps).
        """

        def clean_columns(
            columns: Optional[Dict[str, Any]], data_type: str
        ) -> Optional[Dict[str, Any]]:
            """
            Cleans the columns dictionary by removing columns with None values.
            """

            if columns is None:
                return None

            pop_list = []
            for col, val in columns.items():
                if (
                    col not in self.timeseries_cols
                    and col not in self.continuous_cols
                    and col not in self.original_discrete_cols
                    and col not in self.timestamps_col
                ):
                    logger.warning(f"Column {col} not found in dataset")
                    pop_list.append(col)
                    continue

                if val is None:
                    pop_list.append(col)
                    continue

                if data_type == "discrete" and not (val.including or val.excluding):
                    pop_list.append(col)
                    continue

                if data_type != "discrete":
                    if val.statistics and all(
                        getattr(val.statistics, attr, None) is None
                        for attr in ["mean", "min", "max", "sd", "median", "percentile"]
                    ):
                        val.statistics = None

                    if val.order_by and all(
                        getattr(val.order_by, attr, None) is None
                        for attr in ["statistic", "direction", "priority"]
                    ):
                        val.order_by = None

                    if not (val.statistics or val.order_by):
                        pop_list.append(col)

            return {
                key: value for key, value in columns.items() if key not in pop_list
            } or None

        query.timeseries_columns = clean_columns(query.timeseries_columns, "timeseries")
        query.continuous_columns = clean_columns(query.continuous_columns, "continuous")
        query.discrete_columns = clean_columns(query.discrete_columns, "discrete")
        # TODO timestamps need cleaning in case name is wrong.

        logger.info(f"Cleaned query: {query}")

        return query

    def apply_timestamp_query(
        self, df: pd.DataFrame, query_timestamps: Optional[Dict[str, TimestampInfo]]
    ) -> pd.DataFrame:
        """
        Extract rows from the DataFrame based on timestamp conditions.
        Conditions are applied to the columns as specified in the query dictionary.

        Parameters:
            - df: DataFrame to apply query to
            - query_timestamps: query info dict for timestamp conditions

        Returns:
            - A DataFrame containing only the rows that meet the timestamp conditions.
        """
        if query_timestamps is None:
            logger.warning("No query info for timestamps")
            return df

        logger.info("Started timestamp query")
        filtered_df = df  # .copy()

        col_name, details = next(iter(query_timestamps.items()))
        logger.info(f"Processing: {col_name}")

        if details.relative_time:
            filtered_df = self.apply_relative_timestamp_query(
                filtered_df, col_name, details.relative_time
            )

        conditions = []
        if details.geq:
            conditions.append(f"{col_name} >= @pd.to_datetime(@details.geq)")
        if details.leq:
            conditions.append(f"{col_name} <= @pd.to_datetime(@details.leq)")
        if details.eq:
            conditions.append(f"{col_name} == @pd.to_datetime(@details.eq)")

        if conditions:
            filtered_df = filtered_df.query(
                " and ".join(conditions), local_dict={"details": details}
            )

        # Group by window_idx and filter out groups with size less than window_size
        valid_window_idxs = [
            window_idx
            for window_idx, group in filtered_df.groupby("window_idx")
            if len(group) == self.window_size
        ]

        filtered_df = filtered_df[filtered_df["window_idx"].isin(valid_window_idxs)]

        logger.info(f"Apply timestamp filtering: {df.shape} -> {filtered_df.shape}")
        return filtered_df

    def apply_relative_timestamp_query(
        self, df: pd.DataFrame, col_name: Union[str, List[str]], relative_time: dict
    ) -> pd.DataFrame:

        if isinstance(col_name, list):
            col_name = col_name[0]

        filtered_df = df.copy()

        if "hour" in relative_time:
            hour = relative_time["hour"]
            filtered_df = filtered_df[
                filtered_df[col_name].dt.hour.between(hour, hour + 1)
            ]
        if "minute" in relative_time:
            minute = relative_time["minute"]
            filtered_df = filtered_df[filtered_df[col_name].dt.minute == minute]
        if "day_of_week" in relative_time:
            day = relative_time["day_of_week"]
            filtered_df = filtered_df[
                filtered_df[col_name].dt.dayofweek == day - 1
            ]  # pandas uses 0-6
        if "day_of_month" in relative_time:
            day = relative_time["day_of_month"]
            filtered_df = filtered_df[filtered_df[col_name].dt.day == day]
        if "month" in relative_time:
            month = relative_time["month"]
            filtered_df = filtered_df[filtered_df[col_name].dt.month == month]

        # Group by window_idx and filter out groups with size less than window_size
        valid_window_idxs = [
            window_idx for window_idx, group in filtered_df.groupby("window_idx")
        ]

        filtered_df = df.copy()
        filtered_df = filtered_df[filtered_df["window_idx"].isin(valid_window_idxs)]

        logger.info(
            f"Apply relative timestamp filtering: {df.shape} -> {filtered_df.shape}"
        )
        return filtered_df

    def apply_continuous_query(
        self,
        df: pd.DataFrame,
        query_continuous: Optional[Dict[str, Dict[str, Any]]] = None,
        datatype: str = "",
    ) -> pd.DataFrame:
        """
        Extract rows from the DataFrame based on statistical conditions for continuous columns.
        Conditions are applied to the columns as specified in the query dictionary.

        Parameters:
            - df: DataFrame to apply query to
            - query_continuous: query info dict for continuous conditions
            - datatype: "timeseries" or "continuous"

        Returns:
            - A DataFrame containing only the rows that meet the statistical conditions.
        """
        if query_continuous is None:
            logger.warning(f"No query info for {datatype}")
            return df

        logger.info(f"Started {datatype} query")
        filtered_df = df  # .copy()
        if self.precompute_stats:
            stats_df = self.precomputed_stats.copy()
        else:
            stats_df = self.compute_statistics(
                df[["window_idx"] + list(query_continuous.keys())]
            )
        # add percentile to stats_df if its present in the query_continuous
        for col_name, details in query_continuous.items():
            if details.get("statistics") and details["statistics"].get("percentile"):
                percentile_info = details["statistics"]["percentile"]
                percentile_value = None
                comparison = None
                seen_percentiles = set()
                for comp, value in percentile_info.items():
                    if (
                        value is not None
                        and comp != "ignore_zeroes"
                        and value not in seen_percentiles
                    ):
                        percentile_value = value
                        comparison = comp
                        if percentile_info.get("ignore_zeroes"):
                            logger.info(f"Ignoring zeroes for {col_name}")
                            non_zero_values = df[df[col_name] != 0]
                            overall_percentile = (
                                non_zero_values.groupby("window_idx")
                                .agg({col_name: "median"})[col_name]
                                .quantile(percentile_value / 100)
                            )
                        else:
                            overall_percentile = (
                                df.groupby("window_idx")
                                .agg({col_name: "median"})[col_name]
                                .quantile(percentile_value / 100)
                            )
                        stats_df[
                            f"{col_name.replace('.', '_')}_{str(percentile_value).replace('.', '_')}_percentile"
                        ] = overall_percentile
                        seen_percentiles.add(percentile_value)

        ret_df = stats_df.copy()
        for col_name, details in query_continuous.items():
            if details["statistics"] is None:
                logger.warning(f"No stat info for {col_name}")
                continue

            logger.info(f"Processing: {col_name}")
            stats = details.get("statistics", {})

            mask = pd.Series(True, index=ret_df.index)

            for stat_type in ["mean", "min", "max", "sd", "median", "percentile"]:
                stat_condition = (
                    {} if stats.get(stat_type) is None else stats.get(stat_type)
                )
                for comparison, value in stat_condition.items():
                    if value is not None and comparison != "ignore_zeroes":
                        column_name = f"{stat_type}_{col_name.replace('.', '_')}"
                        if stat_type == "percentile":
                            column_name = f"median_{col_name.replace('.', '_')}"
                            percentile_column = f"{col_name.replace('.', '_')}_{str(value).replace('.', '_')}_percentile"
                            if comparison == "geq":
                                mask &= ret_df[column_name] >= ret_df[percentile_column]
                            elif comparison == "leq":
                                mask &= ret_df[column_name] <= ret_df[percentile_column]
                            elif comparison == "eq":
                                mask &= ret_df[column_name] == ret_df[percentile_column]
                            elif comparison == "gt":
                                mask &= ret_df[column_name] > ret_df[percentile_column]
                            elif comparison == "lt":
                                mask &= ret_df[column_name] < ret_df[percentile_column]
                        else:
                            if comparison == "geq":
                                mask &= ret_df[column_name] >= value
                            elif comparison == "leq":
                                mask &= ret_df[column_name] <= value
                            elif comparison == "eq":
                                mask &= ret_df[column_name] == value
                            elif comparison == "gt":
                                mask &= ret_df[column_name] > value
                            elif comparison == "lt":
                                mask &= ret_df[column_name] < value

            ret_df = ret_df[mask]

        valid_windows = ret_df["window_idx"].unique()
        filtered_df = filtered_df[filtered_df["window_idx"].isin(valid_windows)]
        logger.info(
            f"Apply {datatype} filtering: {df.shape} -> {filtered_df.shape}, unique windows left: {len(valid_windows)}"
        )
        return filtered_df

    def apply_discrete_query(
        self,
        df: pd.DataFrame,
        query_discrete: Optional[
            Dict[str, Dict[str, Union[List[str], None]]]
        ],  # str: DiscreteColumnInfo
    ) -> pd.DataFrame:
        """
        Extract windows from the DataFrame that have a specific discrete value in the given column,
        while dropping rows from windows that do not contain the value in that column.

        Parameters:
            - query_discrete: query info dict for discrete conditions
            format: {col_name: {including: [str] | None, excluding: [str] | None }}

        Returns:
            - A DataFrame containing only the rows where the specified column has the desired value,
              and removing windows that do not contain the value.
        """

        if query_discrete is None:
            logger.warning("No query info for discrete")
            return df

        filtered_df = df  # .copy()
        logger.info("Started discrete query")
        valid_window_idxs = set(filtered_df["window_idx"].unique())

        for col_name, details in query_discrete.items():
            logger.info(f"Processing: {col_name}")
            include_values = (
                set([])
                if details.get("including") is None
                else set(details.get("including")) # type: ignore
            )
            exclude_values = (
                set([])
                if details.get("excluding") is None
                else set(details.get("excluding")) # type: ignore
            )
            # Eliminate windows that contain any of the excluding elements
            if exclude_values:
                window_idxs_to_exclude = set(
                    filtered_df[filtered_df[col_name].isin(exclude_values)][
                        "window_idx"
                    ].unique()
                )
                valid_window_idxs -= window_idxs_to_exclude

            # Include windows that contain any of the including elements
            if include_values:
                window_idxs_to_include = set(
                    filtered_df[filtered_df[col_name].isin(include_values)][
                        "window_idx"
                    ].unique()
                )
                if window_idxs_to_include:
                    logger.info(f"Include values: {include_values}")
                    valid_window_idxs &= window_idxs_to_include
                else:
                    logger.warning(
                        f"No valid windows to include - skipping column: {col_name} with values: {include_values}"
                    )

        filtered_df = filtered_df[filtered_df["window_idx"].isin(valid_window_idxs)]
        logger.info(f"Apply discrete filtering: {df.shape} -> {filtered_df.shape}")
        return filtered_df

    def npdata_to_df(self, is_unittest: bool = False) -> pd.DataFrame:
        # TODO add unit test on this func.
        # TODO - combinations of timestamps/timeseries/continuous/discrete conditions, grouping cols not present.
        """
        Converts the numpy data into a dataframe
        the index is "window_idx" and the columns are the timeseries, continuous, and discrete columns
        """
        if os.path.exists(self.view_df_save_path) and not is_unittest:
            logger.info(f"Loading from {self.view_df_save_path}")
            df = pd.read_parquet(self.view_df_save_path)
            return df
        timestamps_data_np = self.timestamps_data_np
        timeseries_data_np = self.timeseries_data_np
        continuous_data_np = self.continuous_data_np
        discrete_data_np = self.discrete_data_np

        window_idxs = np.repeat(
            np.arange(timeseries_data_np.shape[0]), self.window_size
        )

        if len(timestamps_data_np) > 0:
            timestamps_reshaped = timestamps_data_np.reshape(-1)
        else:
            timestamps_reshaped = np.arange(
                timeseries_data_np.shape[0] * self.window_size
            )

        timeseries_reshaped = timeseries_data_np.reshape(
            -1, timeseries_data_np.shape[2]
        )
        continuous_reshaped = continuous_data_np.reshape(
            -1, continuous_data_np.shape[2]
        )
        discrete_reshaped = discrete_data_np.reshape(-1, discrete_data_np.shape[2])
        data_concatenated = np.concatenate(
            [timeseries_reshaped, continuous_reshaped, discrete_reshaped], axis=1
        )

        df = pd.DataFrame(
            data_concatenated,
            columns=self.timeseries_cols
            + self.continuous_cols
            + self.original_discrete_cols,
        )
        df["window_idx"] = window_idxs
        if self.timestamps_col:
            df[self.timestamps_col[0]] = pd.to_datetime(timestamps_reshaped)

        if not is_unittest:
            df.to_parquet(self.view_df_save_path)
            logger.info(f"Saved to {self.view_df_save_path}")

        return df

    def extract_attributes_from_query_with_llm(self, query: str) -> TimeSeriesViewQuery:
        # TODO needs unit testing
        # TODO - just added priority on the ordering. Test it.
        """
        Extracts the attributes from the query using the LLM.
        """
        # TODO - add test cases for this. Note - timestamp order not working too well
        timeseries_columns = self.timeseries_cols
        continuous_columns = self.continuous_cols
        discrete_columns = self.original_discrete_cols
        timestamps_col = self.timestamps_col

        parser = PydanticOutputParser(pydantic_object=TimeSeriesViewQuery)
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant that understands the user's query for finding/searching some data.
                    The user may ask about specific values/statistics/ranges like less than (lt) or greater than (gt), or greater or equal to (geq) or less or equal to (leq).

                    The data includes timeseries, continuous, discrete data, and timestamps.
                    The possible values for timeseries columns are: {timeseries_columns}
                    The possible values for continuous columns are: {continuous_columns}
                    The possible values for discrete columns are: {discrete_columns}
                    The possible values for timestamps are: {timestamps_col}

                    <Important> The user must explicitly specify the columns (possible shorthand notation is allowed) they want to query - DO NOT make any assumptions of which columns they are referring to if you cannot easily find it in the query. </Important>

                    For timestamp queries, you can now handle relative time ranges. Use the 'relative_time' field in the TimestampInfo object to specify these ranges. For example:
                    - For a specific hour: {{'hour': 20}} (for 8pm)
                    - For a specific minute: {{'minute': 30}}
                    - For a specific day of the week: {{'day_of_week': 1}} (1 for Monday, 7 for Sunday)
                    - For a specific day of the month: {{'day_of_month': 15}}
                    - For a specific month: {{'month': 6}} (for June)

                    When using relative_time, you don't need to specify geq and leq.

                    For ordering priority, start counting from 0, and do it in the order the user's query specifies.

                    You will return a JSON object that matches the following Pydantic model:
                    <Format Instructions>
                    {format_instructions}
                    </Format Instructions>


                    """,
                ),
                (
                    "human",
                    """
                    <Examples>
                    input: find 10 windows where the max counter.rtt_mean is more than .9 and the mean counter.rtt_max is less than .9. Order by most recent time and the max counter.rtt_max
                    output:
                        {{
                            'timeseries_columns': {{
                                'counter.rtt_mean': {{
                                    'statistics': {{
                                        'mean': None,
                                        'min': None,
                                        'max': {{'gt': 0.9, 'lt': None, 'geq': None, 'leq': None, 'eq': None}},
                                        'sd': None,
                                        'median': None
                                    }},
                                    'order_by': None
                                }}
                            }},
                            'continuous_columns': {{
                                'counter.rtt_max': {{
                                    'statistics': {{
                                        'mean': {{'gt': None, 'lt': .9, 'geq': None, 'leq': None, 'eq': None}},
                                        'min': None,
                                        'max': None,
                                        'sd': None,
                                        'median': None
                                    }},
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 1
                                    }}
                                }}
                            }},
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 0
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 10
                        }}

                    input: find 10 windows where the max counter.rtt_mean > 0.9 and the mean counter.rtt_max < 0.9 along with the throughput in the bottom 10 percentile. Order by most recent time and the max counter.rtt_max
                    output:
                        {{
                            'timeseries_columns': {{
                                'counter.rtt_mean': {{
                                    'statistics': {{
                                        'mean': None,
                                        'min': None,
                                        'max': {{'gt': 0.9, 'lt': None, 'geq': None, 'leq': None, 'eq': None}},
                                        'sd': None,
                                        'median': None,
                                        'percentile': {{'gt': None, 'lt': 10, 'geq': None, 'leq': None, 'eq': None, 'ignore_zeros': false}}
                                    }},
                                    'order_by': None
                                }}
                            }},
                            'continuous_columns': {{
                                'counter.rtt_max': {{
                                    'statistics': {{
                                        'mean': {{'gt': None, 'lt': .9, 'geq': None, 'leq': None, 'eq': None}},
                                        'min': None,
                                        'max': None,
                                        'sd': None,
                                        'median': None
                                    }},
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 1
                                    }}
                                }}
                            }},
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 0
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 10
                        }}

                    input: find 10 windows where the max counter.rtt_mean is greater than or equal to .9 and the mean counter.rtt_max is less than or equal to .9. Order by most recent time and the max counter.rtt_max
                    output:
                        {{
                            'timeseries_columns': {{
                                'counter.rtt_mean': {{
                                    'statistics': {{
                                        'mean': None,
                                        'min': None,
                                        'max': {{'gt': None, 'lt': None, 'geq': 0.9, 'leq': None, 'eq': None}},
                                        'sd': None,
                                        'median': None
                                    }},
                                    'order_by': None
                                }}
                            }},
                            'continuous_columns': {{
                                'counter.rtt_max': {{
                                    'statistics': {{
                                        'mean': {{'gt': None, 'lt': None, 'geq': None, 'leq': .9, 'eq': None}},
                                        'min': None,
                                        'max': None,
                                        'sd': None,
                                        'median': None
                                    }},
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 1
                                    }}
                                }}
                            }},
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 0
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 10
                        }}

                    input: find 10 windows where the max counter.rtt_mean is >= 0.9 and the mean counter.rtt_max is <= .9. Order by most recent time and the max counter.rtt_max
                    output:
                        {{
                            'timeseries_columns': {{
                                'counter.rtt_mean': {{
                                    'statistics': {{
                                        'mean': None,
                                        'min': None,
                                        'max': {{'gt': None, 'lt': None, 'geq': 0.9, 'leq': None, 'eq': None}},
                                        'sd': None,
                                        'median': None
                                    }},
                                    'order_by': None
                                }}
                            }},
                            'continuous_columns': {{
                                'counter.rtt_max': {{
                                    'statistics': {{
                                        'mean': {{'gt': None, 'lt': None, 'geq': None, 'leq': .9, 'eq': None}},
                                        'min': None,
                                        'max': None,
                                        'sd': None,
                                        'median': None
                                    }},
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 1
                                    }}
                                }}
                            }},
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 0
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 10
                        }}

                    input: Find windows where the mean continuous_condition_1 is greater than 4 and less than or equal to50, and the STD of continuous_condition_2 = 10, then discrete conditions are W, X, not Y and Z. time range is the first week of august 2024.  order by the max value in the windows for continuous_condition_1 and the oldest time.
                    output:
                        {{
                            "continuous_columns": {{
                                "continuous_condition_1": {{
                                    "statistics": {{
                                        "mean": {{
                                            "gt": 4,
                                            "leq": 50
                                        }}
                                    }},
                                    "order_by": {{
                                        "direction": "desc",
                                        "priority": 0
                                    }}
                                }},
                                "continuous_condition_2": {{
                                    "statistics": {{
                                        "sd": {{
                                            "eq": 10
                                        }}
                                    }}
                                }}
                            }}  ,
                            "discrete_columns": {{
                                "some_discrete_column": {{
                                    "including": ["W", "X"],
                                    "excluding": ["Y", "Z"]
                                }}
                            }},
                            "timestamps": {{
                                '{timestamps_col}': {{
                                    "geq": "2024-08-01T00:00:00Z",
                                    "leq": "2024-08-07T23:59:59Z"
                                }},
                                "order_by": {{
                                    "direction": "asc",
                                    "priority": 0
                                }}
                            }}
                        }}

                    input: Find windows where the mean continuous_condition_1 is greater than or equal to 11 and less than 50, and the STD of continuous_condition_2 = 10, then discrete conditions are W, X, not Y and Z. time range is the first week of august 2024.  order by the max value in the windows for continuous_condition_1 and the oldest time.
                    output:
                        {{
                            "continuous_columns": {{
                                "continuous_condition_1": {{
                                    "statistics": {{
                                        "mean": {{
                                            "geq": 11,
                                            "lt": 50
                                        }}
                                    }},
                                    "order_by": {{
                                        "direction": "desc",
                                        "priority": 0
                                    }}
                                }},
                                "continuous_condition_2": {{
                                    "statistics": {{
                                        "sd": {{
                                            "eq": 10
                                        }}
                                    }}
                                }}
                            }}  ,
                            "discrete_columns": {{
                                "some_discrete_column": {{
                                    "including": ["W", "X"],
                                    "excluding": ["Y", "Z"]
                                }}
                            }},
                            "timestamps": {{
                                '{timestamps_col}': {{
                                    "geq": "2024-08-01T00:00:00Z",
                                    "leq": "2024-08-07T23:59:59Z"
                                }},
                                "order_by": {{
                                    "direction": "asc",
                                    "priority": 0
                                }}
                            }}
                        }}

                    input: give me latest 5 windows of data from 8pm
                    output:
                        {{
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'relative_time': {{
                                        'hour': 20
                                    }},
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 0
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 5
                        }}

                    input: show me 10 windows from Mondays where the mean counter.rtt_mean is less than 0.5
                    output:
                        {{
                            'timeseries_columns': {{
                                'counter.rtt_mean': {{
                                    'statistics': {{
                                        'mean': {{'gt': None, 'lt': 0.5, 'geq': None, 'leq': None, 'eq': None}},
                                        'min': None,
                                        'max': None,
                                        'sd': None,
                                        'median': None
                                    }}
                                }}
                            }},
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'relative_time': {{
                                        'day_of_week': 1
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 10
                        }}

                    input: find 8 windows from the 15th of each month where the max counter.jitter_max_rec is greater than 0.1
                    output:
                        {{
                            'timeseries_columns': {{
                                'counter.jitter_max_rec': {{
                                    'statistics': {{
                                        'mean': None,
                                        'min': None,
                                        'max': {{'gt': 0.1, 'lt': None, 'geq': None, 'leq': None, 'eq': None}},
                                        'sd': None,
                                        'median': None
                                    }}
                                }}
                            }},
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'relative_time': {{
                                        'day_of_month': 15
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 8
                        }}

                    input: show me the latest 3 windows from June where the median counter.rtt_mean is between 0.3 and 0.7
                    output:
                        {{
                            'timeseries_columns': {{
                                'counter.rtt_mean': {{
                                    'statistics': {{
                                        'mean': None,
                                        'min': None,
                                        'max': None,
                                        'sd': None,
                                        'median': {{'gt': 0.3, 'lt': 0.7, 'geq': None, 'leq': None, 'eq': None}}
                                    }}
                                }}
                            }},
                            'timestamps': {{
                                '{timestamps_col}': {{
                                    'relative_time': {{
                                        'month': 6
                                    }},
                                    'order_by': {{
                                        'direction': 'desc',
                                        'priority': 0
                                    }}
                                }}
                            }},
                            'n_windows_to_return': 3
                        }}

                    input: Find the class A plan users whose throughput for video is below the 10th percentile of all of other user video throughput, ignore the zeros
                    output:
                        {{
                            "timeseries_columns": {{
                                "Throughput (Kbps)": {{
                                    "statistics": {{
                                        "percentile": {{
                                            "lt": 10,
                                            "ignore_zeros": true
                                        }}
                                    }},
                                }}
                            }}
                        }}

                    </Examples>
                    input: {query}
                    output: """,
                ),
            ]
        )
        output_parser = JsonOutputParser()
        chain = template | llm | output_parser
        ret = TimeSeriesViewQuery()
        try:
            ret = chain.invoke(
                {
                    "query": query,
                    "timeseries_columns": timeseries_columns,
                    "continuous_columns": continuous_columns,
                    "discrete_columns": discrete_columns,
                    "timestamps_col": timestamps_col,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            ret = TimeSeriesViewQuery(**ret)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
        logger.info(f"in extract_attributes_from_query_with_llm: {ret = }")
        ret = self.clean_query(ret)
        return ret

    def view_ts(
        self,
        text_query: str,
        search_set: List[str] = ["train", "val", "test"],  # TODO phase 2 - use this.
        n_windows_to_return: int = 5,
    ) -> Dict[str, np.ndarray]:
        """
        input: text_query, search_set
        output: Dict with keys: [timeseries, continuous, discrete, timestamp], values: np.ndarray
        """

        # parse the query with the LLM
        query = self.extract_attributes_from_query_with_llm(query=text_query)
        n_windows_to_return = query.n_windows_to_return # type: ignore
        query_dict = query.model_dump()
        logger.info(f"query_dict (readable format): {json.dumps(query_dict, indent=4)}")

        logger.info(f"{query_dict=}")
        filtered_df = (
            self.df.copy()
            if not os.path.exists(self.view_df_save_path)
            else pd.read_parquet(self.view_df_save_path)
        )
        # filter the columns to only include those from the query
        columns_to_include = (
            set(query.timeseries_columns.keys() if query.timeseries_columns else [])
            | set(query.continuous_columns.keys() if query.continuous_columns else [])
            | set(query.discrete_columns.keys() if query.discrete_columns else [])
            | set([self.timestamps_col[0]] if self.timestamps_col else [])
            | set(["window_idx"])
        )
        dropped_columns = [
            col for col in filtered_df.columns if col not in columns_to_include
        ] + ["window_idx", self.timestamps_col[0]]
        df_dropped_columns = filtered_df[dropped_columns]
        df_dropped_columns.to_parquet(
            "/tmp/df_dropped_columns_for_view_ui_saved.parquet"
        )
        del df_dropped_columns
        filtered_df = filtered_df[
            [col for col in filtered_df.columns if col in columns_to_include]
        ]
        filtered_df = self.apply_discrete_query(
            filtered_df, query_dict["discrete_columns"]
        )
        filtered_df = self.apply_continuous_query(
            filtered_df, query_dict["continuous_columns"], datatype="continuous"
        )
        filtered_df = self.apply_continuous_query(
            filtered_df, query_dict["timeseries_columns"], datatype="timeseries"
        )
        if self.timestamps_col:
            filtered_df = self.apply_timestamp_query(filtered_df, query.timestamps)
        # TODO add filtering based on the timestamp!!

        # order the windows
        logger.info("done with filtering, now ordering")
        filtered_df = self.order_df(filtered_df, query)
        # top_n_windows
        top_n_windows = filtered_df.head(n_windows_to_return * self.window_size)[
            "window_idx"
        ].unique()
        filtered_df = filtered_df[filtered_df["window_idx"].isin(top_n_windows)]

        logger.info("done ordering")

        # load back the dropped columns
        if os.path.exists("/tmp/df_dropped_columns_for_view_ui_saved.parquet"):
            df_dropped_columns = pd.read_parquet(
                "/tmp/df_dropped_columns_for_view_ui_saved.parquet"
            )
            # join on the "window_idx"
            df_dropped_columns = df_dropped_columns[
                df_dropped_columns["window_idx"].isin(top_n_windows)
            ]
            if len(df_dropped_columns) != len(filtered_df):
                raise ValueError(
                    f"Lengths of df_dropped_columns and filtered_df do not match: {len(df_dropped_columns)=}, {len(filtered_df)=}"
                )

            # concat instead of merge
            filtered_df = pd.concat([filtered_df, df_dropped_columns], axis=1)
            # drop last 2 columns (window_idx and timestamps_col)
            filtered_df = filtered_df.iloc[:, :-2]
            del df_dropped_columns

        ret_windows = self.convert_df_to_np(
            filtered_df,
            n_windows_to_return=n_windows_to_return,
        )
        logger.info("Finished view query")
        # ret_windows = self.get_n_windows(
        #     filtered_data_dict,
        #     query_dict.get("n_windows_to_return", n_windows_to_return),
        # )
        if len(ret_windows["timeseries"]) < n_windows_to_return:
            logger.warning(
                f"Less than {n_windows_to_return} windows found, returning {len(ret_windows['timeseries'])} windows"
            )

        # Check if each window has sorted values by timestamps
        if self.timestamps_col:
            for window_idx, window_df in filtered_df.groupby("window_idx"):
                if not window_df[self.timestamps_col[0]].is_monotonic_increasing:
                    raise ValueError(f"Window {window_idx} is not sorted by timestamps")
            logger.info("All windows are sorted by timestamps")

        return ret_windows

    def get_n_windows(
        self,
        filtered_data_dict: Dict[str, np.ndarray],
        n_windows_to_return: Optional[int],
    ) -> Dict[str, np.ndarray]:
        """
        Update arrays in filtered_data_dict to have only top n_windows_to_return.
        Parameters:
            - filtered_data_dict: dict containing windows array for each data type
            - n_windows_to_return: top windows number to return
        Returns:
            - updated filtered_data_dict
        """
        if n_windows_to_return is not None:
            for key, val in filtered_data_dict.items():
                filtered_data_dict[key] = val[:n_windows_to_return]

        return filtered_data_dict

    def convert_df_to_np(
        self, df: pd.DataFrame, n_windows_to_return: int
    ) -> Dict[str, np.ndarray]:
        # TODO unit test needed
        """
        Converts the filtered dataframe to numpy arrays. returns dict with keys:
        return: Dict with keys: [timeseries, continuous, discrete, timestamp]
        """
        logger.info("start convert_df_to_np")
        if len(df) == 0:
            return {
                "timeseries": np.array([]),
                "continuous": np.array([]),
                "discrete": np.array([]),
                "timestamp": np.array([]),
            }
        # Group columns by data type
        timeseries_cols = self.timeseries_cols
        continuous_cols = self.continuous_cols
        discrete_cols = self.original_discrete_cols
        timestamp_col = self.timestamps_col[0] if self.timestamps_col else None

        # Initialize dictionaries to store numpy arrays
        result = {"timeseries": [], "continuous": [], "discrete": [], "timestamp": []}

        # Sort DataFrame by window_idx to ensure correct order
        # df = df.sort_values(by=["window_idx", self.timestamps_col[0]])
        # The window IDXs are sorted in the order of order_by in the query, so commented out the above line

        # Group by window_idx and convert each group to numpy array
        for idx in range(n_windows_to_return):
            idx_start = idx * self.window_size
            idx_end = (idx + 1) * self.window_size
            if idx_start >= len(df) or idx_end > len(df):
                break
            window_df = df.iloc[idx_start:idx_end]
            result["timeseries"].append(window_df[timeseries_cols].values)
            result["continuous"].append(window_df[continuous_cols].values)
            result["discrete"].append(window_df[discrete_cols].values)
            if timestamp_col:
                result["timestamp"].append(
                    window_df[timestamp_col].values.reshape(-1, 1) # type: ignore
                )

        final_result: Dict[str, np.ndarray] = {}
        # Convert lists of arrays to 3D numpy arrays
        for key in result:
            if result[key]:
                final_result[key] = np.array(result[key])
        logger.info("done convert_df_to_np")
        return final_result

    def get_ordering_info(
        self, query: TimeSeriesViewQuery
    ) -> List[Tuple[str, str, int]]:
        ordering_info = []
        # TODO - add support for discrete columns
        for column_type in ["timeseries_columns", "continuous_columns", "timestamps"]:
            columns = getattr(query, column_type, None)
            if columns:
                for col, info in columns.items():
                    if info and info.order_by:
                        ordering_info.append(
                            (
                                col,
                                info.order_by.direction,
                                info.order_by.priority
                                or 0,  # Default priority to 0 if not specified
                            )
                        )
        # Sort by priority (lower priority first)
        ordering_info.sort(key=lambda x: x[2])
        return ordering_info

    def order_df(
        self, filtered_df: pd.DataFrame, query: TimeSeriesViewQuery
    ) -> pd.DataFrame:
        # TODO test
        # TODO later support some ordreing by discrete --> like order by the max number of discrete condition X in window
        """
        Orders the windows based on the query.
        """
        ordering_info = self.get_ordering_info(query)
        logger.info(f"{ordering_info=}")
        if len(ordering_info) == 0:
            return filtered_df

        # order by the windows with the max timestamp
        cols_to_order_by = [col for col, _, _ in ordering_info]
        aggregated_stats_df = (
            filtered_df[["window_idx"] + cols_to_order_by]
            .groupby("window_idx")
            .agg(["min", "max"])
        )

        sort_columns = []
        for col, direction, _ in ordering_info:
            if direction == "asc":
                sort_columns.append((col, "min"))
            else:
                sort_columns.append((col, "max"))

        # Order the windows
        if sort_columns:
            sorted_stats_df = aggregated_stats_df.sort_values(
                by=sort_columns,
                ascending=[direction == "asc" for _, direction, _ in ordering_info],
            )
        sorted_window_indices = sorted_stats_df.index
        filtered_df = (
            filtered_df.set_index("window_idx").loc[sorted_window_indices].reset_index()
        )
        # Note - this permutes the window_idx, we SHOULD NOT SORT AFTER THIS.

        return filtered_df


if __name__ == "__main__":
    view = TimeSeriesView(
        config_path=os.path.join(
            str(os.getenv("SYNTHEFY_PACKAGE_BASE")),
            "examples/configs/preprocessing_configs/config_twamp_one_month_preprocessing.json",
        ),
        precompute_stats=False,
    )
    # text_query = "Find me about 16 windows where the mean counter.rtt_mean is greater than .2 and less than .6, counter.jitter_max_rec > .05, distance_to_twamp_reflector_km <= 10, order by the most recent date and the max counter.rtt_mean, and for connection use Glasfaser, but excluding `DO1707_Gelsenkirchen-Schalke-Arena-80_73553635` and including `SY2886_Niedereschach-G-Stro-M_73509564` in column label"
    # text_query = "Show me where the mean counter.rtt_mean less than 0.697"
    # text_query = "Show me 10 windows of data from april"
    # text_query = "show me 10 windows from april where the mean counter.rtt_mean is less than 1000"
    text_query = "Show me where the mean counter.rtt_mean < .697"
    # text_query = "show me 10 windows of top 10 percentile counter.rtt_mean"
    logger.info(f"{text_query=}")
    search_set = ["test"]
    result = view.view_ts(text_query=text_query)
