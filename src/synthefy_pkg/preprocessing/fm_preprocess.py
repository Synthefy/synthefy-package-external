import argparse
import os
import pickle
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler

from synthefy_pkg.preprocessing.fm_preprocess_window import (
    WindowPreprocessor,
)
from synthefy_pkg.preprocessing.fm_text_embedder import (
    FoundationModelTextEmbedder,
)
from synthefy_pkg.utils.fm_utils import (
    load_data_from_directory,
)

DEFAULT_WINDOW_SIZE = 256
DEFAULT_STRIDE = 1
TEMP_FIX = True


class FoundationModelPreprocessor:
    def __init__(
        self,
        window_size: int = 512,
        stride: int = 1,
        metadata_types_to_use: List[str] = [
            "timestamp",
            "dataset_description",
            "text_description",
            "continuous",
            "retrieved_timeseries",
            "time_varying_textual_metadata",
        ],
    ):
        self.window_size = window_size
        self.stride = stride
        self.window_preprocessor = (
            None  # Will be initialized when metadata is available
        )
        self.metadata_types_to_use = metadata_types_to_use

    def preprocess_foundation_model_data(self, data_dir: str) -> None:
        """
        Preprocesses the data in the given directory.
        Saves the preprocessed data to the given directory.

        Args:
            data_dir: The directory containing the data to preprocess.

        Returns:
            None

        """
        df, metadata = load_data_from_directory(data_dir)

        # Initialize the window preprocessor with metadata
        self._initialize_window_preprocessor(metadata)

        # Initialize text embedder and generate embeddings
        text_embedder = FoundationModelTextEmbedder(metadata, data_dir)
        embeddings_dict = text_embedder.read_or_embed_fm_text_data(data_dir)

        total_windows = (len(df) - self.window_size) // self.stride + 1

        total_windows = max(
            total_windows, 1
        )  # we support data of size < window_size

        logger.info(f"Total windows: {total_windows}")

        all_timeseries = []
        all_metadata = []
        all_scalers = []
        for window_idx in range(total_windows):
            if window_idx % 100 == 0:
                logger.info(
                    f"Processing window {window_idx} of {total_windows}"
                )
            start_idx = window_idx * self.stride
            end_idx = start_idx + self.window_size
            window = df.iloc[start_idx:end_idx]

            # TODO: when ready to use non uniform values for a window
            # text description, uncomment the following code
            # # Get the corresponding slices of text embeddings for this window
            # window_text_description = np.array([])
            # if len(text_description_embedding) > 0:
            #     window_text_description = text_description_embedding[
            #         start_idx:end_idx
            #     ]

            window_time_varying_text = np.array([])
            if len(embeddings_dict["time_varying_text_embedding"]) > 0:
                window_time_varying_text = embeddings_dict[
                    "time_varying_text_embedding"
                ][start_idx:end_idx]

            processed_window_dict = self._process_window(
                window,
                window_time_varying_text,
                embeddings_dict,
            )

            all_timeseries.append(processed_window_dict["timeseries"])
            all_metadata.append(processed_window_dict["metadata"])
            all_scalers.append(processed_window_dict["scalers"])

        all_timeseries = np.stack(all_timeseries, axis=0)
        all_metadata = np.stack(all_metadata, axis=0)

        logger.info(f"All timeseries shape: {all_timeseries.shape}")
        logger.info(f"All metadata shape: {all_metadata.shape}")

        # save the data
        np.save(os.path.join(data_dir, "timeseries.npy"), all_timeseries)
        np.save(os.path.join(data_dir, "metadata.npy"), all_metadata)
        with open(os.path.join(data_dir, "scalers.pkl"), "wb") as f:
            pickle.dump(all_scalers, f)

    def _initialize_window_preprocessor(self, metadata: Dict[str, Any]) -> None:
        """Initialize the window preprocessor with metadata.

        Args:
            metadata: Dictionary of metadata for this data source
        """
        self.window_preprocessor = WindowPreprocessor(
            window_size=self.window_size,
            metadata=metadata,
            metadata_types_to_use=self.metadata_types_to_use,
            verbose=False,
        )

    def _process_window(
        self,
        window: pd.DataFrame,
        window_time_varying_text: np.ndarray,
        embeddings_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Union[torch.Tensor, Dict[str, StandardScaler]]]:
        """Process a window of data and return a dictionary with the window, metadata, and scaler.

        Args:
            window: DataFrame containing the window data
            window_time_varying_text: Time varying text metadata embeddings for this window
            embeddings_dict: Dictionary containing embeddings for various text metadata

        Returns:
            dict: Dictionary containing:
                - 'timeseries': tensor of the scaled window data - shape (1, window_size) since only univariate data for now
                - 'metadata': tensor of metadata (empty if no metadata) - shape (1, metadata_vector_length)
                - 'scalers': dictionary containing the fitted StandardScaler for timeseries and metadata

        Raises:
            ValueError: If window_preprocessor is not initialized
        """
        if self.window_preprocessor is None:
            raise ValueError(
                "Window preprocessor is not initialized. Call _initialize_window_preprocessor first."
            )

        processed_window_dict = self.window_preprocessor.preprocess_window(
            window,
            continuous_embeddings=embeddings_dict[
                "continuous_descriptions_embeddings"
            ],
            retrieved_ts_embeddings=embeddings_dict[
                "retrieved_ts_descriptions_embeddings"
            ],
            dataset_description_embedding=embeddings_dict[
                "dataset_description_embedding"
            ],
            text_description_embedding=embeddings_dict[
                "text_description_embedding"
            ],
            time_varying_text_embedding=window_time_varying_text,
        )
        return processed_window_dict


def main():
    """
    Main function to run the preprocessing pipeline from command line.

    Usage:
        python -m synthefy_pkg.preprocessing.fm_preprocess /path/to/data_directory
    """
    parser = argparse.ArgumentParser(
        description="Preprocess data for foundation models"
    )
    parser.add_argument(
        "--data_dir", type=str, help="Directory containing data to preprocess"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Window size for preprocessing",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Stride for window sliding",
    )
    parser.add_argument(
        "--metadata_types_to_use",
        type=str,
        nargs="+",
        default=[
            "timestamp",
            "dataset_description",
            "text_description",
            "continuous",
            "retrieved_timeseries",
            "time_varying_textual_metadata",
        ],
        help="Types of metadata to use in preprocessing",
    )

    args = parser.parse_args()

    preprocessor = FoundationModelPreprocessor(
        window_size=args.window_size,
        stride=args.stride,
        metadata_types_to_use=args.metadata_types_to_use,
    )

    logger.info(f"Preprocessing data in {args.data_dir}")
    preprocessor.preprocess_foundation_model_data(args.data_dir)
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
