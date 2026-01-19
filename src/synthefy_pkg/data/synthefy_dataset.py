import json
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
from loguru import logger

from synthefy_pkg.preprocessing.base_config import BaseConfig

DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
SCALER_TYPES = ["standard"]

COMPILE = False


class SynthefyDataset(BaseConfig):
    datatype_inds_dict: Dict[str, Dict[str, int]]

    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        super().__init__(config_source=config_source)

    def load_labels_description_by_arg(self, key: str) -> Any:
        """
        Load the labels description from a pickle file and extract the value for the given key.

        Parameters:
        - key (str): The key to extract from the labels description dictionary.

        Returns:
        - The value associated with the given key in the labels description dictionary.
        """
        labels_description_path = os.path.join(
            self.output_path, "labels_description.pkl"
        )
        if not os.path.exists(labels_description_path):
            raise FileNotFoundError(
                f"Labels description file {labels_description_path} not found."
            )

        with open(labels_description_path, "rb") as file:
            labels_description = pickle.load(file)

        if key in labels_description:
            return labels_description[key]
        else:
            raise KeyError(f"Key '{key}' not found in labels description.")

    def load_columns_and_timestamps(self):
        self.timeseries_cols = json.load(
            open(os.path.join(self.output_path, "timeseries_windows_columns.json"))
        )
        self.continuous_cols = json.load(
            open(os.path.join(self.output_path, "continuous_windows_columns.json"))
        )
        self.discrete_cols = json.load(
            open(os.path.join(self.output_path, "discrete_windows_columns.json"))
        )

    def load_original_discrete_colnames(self) -> Dict:
        """
        Load the original discrete column names from the encoders_dict.pkl file
        if no file, no discrete conditions - return {}
        """

        if os.path.exists(os.path.join(self.output_path, "encoders_dict.pkl")):
            discrete_encoders = pickle.load(
                open(
                    os.path.join(
                        self.output_path,
                        "encoders_dict.pkl",
                    ),
                    "rb",
                )
            )
        else:
            discrete_encoders = {}
        self.original_discrete_cols = []
        for encoder in discrete_encoders.values():
            self.original_discrete_cols.extend(list(encoder.feature_names_in_))

    def load_windows(
        self,
        window_types: List[str] = [
            "timeseries",
            "continuous",
            "discrete",
            "timestamp",
            "timestamp_conditions",
            "original_discrete",
        ],
    ) -> None:
        """
        Load saved preprocessed windows into self.windows_data_dict:
            1. For a window type store all windows concatenated under `windows` key
            2. For each dataset type store data under `{dataset_type}_windows`
            3. Collect start and end indices of dataset_type windows in the combined windows array
        """
        self.datatype_inds_dict = defaultdict(dict)
        for window_type in window_types:
            start_ind = 0
            for dataset_type in self.dataset_types:
                windows_filename = os.path.join(
                    self.output_path,
                    f"{dataset_type}_{self.windows_filenames_dict[window_type]['window_filename']}.npy",
                )
                logger.info(f"Loading {windows_filename}")
                if window_type == "timestamp" or window_type == "original_discrete":
                    with open(windows_filename, "rb") as file:
                        windows_data = np.load(file, allow_pickle=True)
                else:
                    with open(windows_filename, "rb") as file:
                        windows_data = np.load(file)
                # Used in view/search only
                if (
                    window_type in ["discrete", "original_discrete"]
                    and len(windows_data.shape) == 2
                ):
                    windows_data = np.repeat(
                        windows_data[:, np.newaxis, :], self.window_size, axis=1
                    )
                if window_type == "timeseries":
                    windows_data = windows_data.transpose(0, 2, 1)
                self.windows_data_dict[window_type][
                    f"{dataset_type}_windows"
                ] = windows_data

                if dataset_type == self.dataset_types[0]:
                    self.windows_data_dict[window_type]["windows"] = windows_data
                else:
                    self.windows_data_dict[window_type]["windows"] = np.concatenate(
                        (self.windows_data_dict[window_type]["windows"], windows_data),
                        axis=0,
                    )

                if not all(
                    element in self.datatype_inds_dict for element in self.dataset_types
                ):
                    end_ind = start_ind + len(windows_data)
                    self.datatype_inds_dict[dataset_type] = {
                        "start": start_ind,
                        "end": end_ind,
                    }
                    start_ind = end_ind


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file path", required=True)

    config_path = parser.parse_args().config
    obj = SynthefyDataset(config_source=config_path)
