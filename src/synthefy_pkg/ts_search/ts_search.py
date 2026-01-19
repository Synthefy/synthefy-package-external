import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import cdist

from synthefy_pkg.data.synthefy_dataset import SynthefyDataset
from synthefy_pkg.utils.basic_utils import load_pickle_from_path
from synthefy_pkg.utils.scaling_utils import (
    SCALER_FILENAMES,
)

# from synthefy_pkg.ts_search.utils.ts_search_view_utils import extract_search_set

COMPILE = False

# TODO: add more encoder types
ENCODER_TYPES = {}
DIST_TYPES = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


class TimeSeriesSearch:
    dataset: SynthefyDataset

    def __init__(
        self,
        config_path: str,
    ) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

        self.config_path = config_path

        self.dataset = SynthefyDataset(config_source=config_path)
        self.dataset.load_windows(
            window_types=["timestamp", "timeseries", "continuous", "discrete"]
        )
        self.dataset.load_columns_and_timestamps()

        # Load encoder and encoded data from saved in preprocessing
        self.load_encoded_data()

        self.timeseries_scalers = load_pickle_from_path(
            os.path.join(
                self.dataset.output_path,
                SCALER_FILENAMES["timeseries"],
            ),
            raise_error_if_not_exists=False,
            default_return_value={},
        )

    def load_encoded_data(self) -> None:
        encoded_data_path = os.path.join(
            self.dataset.output_path, "ts_search_encoded_timeseries_data.pkl"
        )
        if os.path.exists(encoded_data_path):
            logger.info("Loading encoded data")
            self.encoder_dict = load_pickle_from_path(encoded_data_path)
            self.encoder = self.encoder_dict.pop("encoder")
            self.encoded_data = self.encoder_dict.pop("encoded_data")
            logger.info(f"Loaded encoded data shape: {self.encoded_data.shape}")
        else:
            logger.info("No encoded data found, encoding from scratch")
            encoder_type = self.dataset.config.get("search_encoder", None)
            if encoder_type is None or encoder_type not in ENCODER_TYPES:
                raise ValueError(f"No encoder type specified or encoder '{encoder_type}' not found. Available encoders: {list(ENCODER_TYPES.keys())}")
            self.encoder = ENCODER_TYPES[encoder_type](self.dataset.config)

            self.encoded_data = self.encoder.encode(
                self.dataset.windows_data_dict["timeseries"]["windows"],
                col_names=self.dataset.timeseries_cols,
            )

    def search(
        self,
        query_np: np.ndarray,
        n_closest: int = 5,
        search_set: List[str] = ["train", "test", "val"],
        dist_metric: str = "cosine",
    ) -> Tuple[
        List[int],
        List[float],
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Search for the most similar `n_closest` window(s) to the given query
        in the encoded dataset using Euclidean distance.

        Parameters:
            - query (Union[np.array, torch.Tensor, pd.DataFrame, Dict]): The query time series window.

        Returns:
            - List[int]: The index of the most similar window.
            - List[float]: The Euclidean distance to the most similar window.
            - np.ndarray: The similar window
            - np.ndarray: The similar window's continuous conditions
            - np.ndarray: The similar window's discrete conditions
        """
        assert dist_metric in DIST_TYPES, (
            f"Distance metric {dist_metric} is not supported, choose from {DIST_TYPES}"
        )
        # Scale timeseries query window

        # TODO: Extract search set
        # encoded_data_search_set = extract_search_set(
        #     self.encoded_data, self.dataset.datatype_inds_dict, search_set
        # )
        # logger.info(f"Search dataset shape: {encoded_data_search_set.shape}")
        # timestamps_data_np = extract_search_set(
        #     self.dataset.windows_data_dict["timestamp"]["windows"],
        #     self.dataset.datatype_inds_dict,
        #     search_set,
        # )
        # timeseries_data_np = extract_search_set(
        #     self.dataset.windows_data_dict["timeseries"]["windows"],
        #     self.dataset.datatype_inds_dict,
        #     search_set,
        # )
        # continuous_data_np = extract_search_set(
        #     self.dataset.windows_data_dict["continuous"]["windows"],
        #     self.dataset.datatype_inds_dict,
        #     search_set,
        # )
        # discrete_data_np = extract_search_set(
        #     self.dataset.windows_data_dict["discrete"]["windows"],
        #     self.dataset.datatype_inds_dict,
        #     search_set,
        # )
        encoded_data_search_set = self.encoded_data
        timestamps_data_np = self.dataset.windows_data_dict["timestamp"][
            "windows"
        ]
        timeseries_data_np = self.dataset.windows_data_dict["timeseries"][
            "windows"
        ]
        continuous_data_np = self.dataset.windows_data_dict["continuous"][
            "windows"
        ]
        discrete_data_np = self.dataset.windows_data_dict["discrete"]["windows"]

        logger.info(f"{query_np.shape=}")
        # may not have timestcontinuousdiscrete data
        data_lengths = [
            len(data)
            for data in [
                timestamps_data_np,
                timeseries_data_np,
                continuous_data_np,
                discrete_data_np,
            ]
            if len(data) > 0
        ]
        assert all(length == data_lengths[0] for length in data_lengths), (
            "All present data types must have the same length"
        )
        assert query_np.shape[0] == 1, (
            f"More than one windows in query: {query_np.shape[0]}"
        )
        assert query_np.shape[2] == timeseries_data_np.shape[2], (
            f"Query num features ({query_np.shape[2]}) doesn't match combined data's ({encoded_data_search_set.shape[timeseries_data_np.shape[2]]})"
        )
        assert query_np.shape[1] == timeseries_data_np.shape[1], (
            f"Query window size ({query_np.shape[1]}) doesn't match combined data's ({encoded_data_search_set.shape[timeseries_data_np.shape[1]]})"
        )

        # Encode the query using the same encoder
        logger.info(f"{query_np.shape=}")
        # query_encoded = self.encoder.encode(query_np)
        query_encoded = self.encoder.encode(
            query_np, col_names=self.dataset.timeseries_cols
        )
        logger.info(f"{query_encoded.shape=}")

        assert query_encoded.shape[1] == encoded_data_search_set.shape[1], (
            f"Encoded query num features ({query_encoded.shape[1]}) doesn't match encoded combined data's ({encoded_data_search_set.shape[1]})"
        )

        # Calculate distances between the query and each window in the dataset
        distances = cdist(  # type: ignore
            query_encoded.astype(np.float64),
            encoded_data_search_set.astype(np.float64),
            metric=dist_metric,  # type: ignore
        ).flatten()
        sorted_indices = np.argsort(distances)
        top_n_indices = [int(i) for i in sorted_indices[:n_closest]]
        top_n_distances = distances[top_n_indices]

        logger.info(
            f"Most similar window found at index {top_n_indices} with distance {top_n_distances}"
        )

        if self.dataset.timestamps_col:
            x_axis = timestamps_data_np[top_n_indices]
        else:
            x_axis = np.tile(
                np.expand_dims(
                    np.array(list(range(timeseries_data_np.shape[1]))), axis=0
                ),
                (n_closest, 1),
            )

        return (
            top_n_indices,
            top_n_distances,
            x_axis,
            timeseries_data_np[top_n_indices],
            (
                continuous_data_np[top_n_indices]
                if len(continuous_data_np) > 0 and continuous_data_np.size > 0
                else None
            ),
            (
                discrete_data_np[top_n_indices]
                if len(discrete_data_np) > 0 and discrete_data_np.size > 0
                else None
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="config file path", required=True
    )

    config_path = parser.parse_args().config
    obj = TimeSeriesSearch(config_path=config_path)

    # Test search
    query_window_ind = 10
    query = obj.dataset.windows_data_dict["timeseries"]["windows"][
        query_window_ind : (query_window_ind + 1), :, :
    ]
    _, _, x_axis, querried_window1, querried_window2, querried_window3 = (
        obj.search(query, n_closest=10, search_set=["train", "test"])
    )
    logger.info(f"x_axis shape: {x_axis.shape if x_axis is not None else None}")
    logger.info(
        f"querried_window1 shape: {querried_window1.shape if querried_window1 is not None else None}"
    )
    logger.info(
        f"querried_window2 shape: {querried_window2.shape if querried_window2 is not None else None}"
    )
    logger.info(
        f"querried_window3 shape: {querried_window3.shape if querried_window3 is not None else None}"
    )
