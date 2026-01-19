import argparse
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from synthefy_pkg.utils.fm_utils import (
    get_columns_info_dict_from_metadata_json,
    load_dataframe_from_directory,
    load_metadata_from_directory,
)

TEXT_EMBEDDING_DIM = 384


class BaseFoundationModelTextEmbedder(ABC):
    def __init__(
        self,
        metadata: Dict[str, Any],
        output_dir: str,
        text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        verbose: bool = False,
    ):
        """Initialize the base text embedder.

        Base class for text embedding generation that handles common functionality between versions.

        Args:
            metadata: Dictionary containing dataset metadata including column information,
                     dataset title, and text column specifications.
            output_dir: Directory where embeddings will be saved as .npy files.
            text_embedding_model_name: Name of the sentence transformer model to use.
                                     Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            verbose: Whether to print verbose output. Defaults to False.
        """
        self.metadata = metadata
        self.output_dir = output_dir
        self.text_embedding_model_name = text_embedding_model_name
        self.verbose = verbose

        # Initialize device but not the model yet
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

    def _initialize_sentence_transformer(self) -> SentenceTransformer:
        """Initialize the sentence transformer model."""
        model = SentenceTransformer(self.text_embedding_model_name)
        model = model.to(self.device)
        model = model.eval()
        return model

    def embed_text(self, text: Union[str, List[str], None]) -> np.ndarray:
        """Embed text using sentence transformer.

        Common text embedding logic used by both embedder versions. Handles single strings,
        lists of strings, and missing values.

        Args:
            text: Text to embed. Can be:
                - A single string
                - A list of strings (may contain None/NaN values)
                - None

        Returns:
            numpy.ndarray: Embeddings with shape:
                - (TEXT_EMBEDDING_DIM,) for single string
                - (len(text), TEXT_EMBEDDING_DIM) for list of strings
                - (0,) for None input
                Missing values in lists are replaced with zero embeddings.
        """
        if self.model is None:
            self.model = self._initialize_sentence_transformer()

        if text is None:
            return np.array([])

        if isinstance(text, str):
            with torch.no_grad():
                embedding = self.model.encode(
                    text, convert_to_tensor=True, show_progress_bar=False
                )
                return embedding.cpu().numpy()

        if isinstance(text, list):
            text_array = np.array(text, dtype=object)
            valid_mask = np.array(
                [isinstance(x, str) and not pd.isna(x) for x in text_array]
            )
            valid_indices = np.where(valid_mask)[0]

            if sum(valid_mask) == 1:
                valid_texts = [text_array[valid_mask][0]]
            else:
                valid_texts = text_array[valid_mask].tolist()

            if not valid_texts:
                return np.zeros((len(text), TEXT_EMBEDDING_DIM))

            with torch.no_grad():
                valid_embeddings = (
                    self.model.encode(
                        valid_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )
                    .cpu()
                    .numpy()
                )

            result = np.zeros((len(text), TEXT_EMBEDDING_DIM))
            result[valid_indices] = valid_embeddings
            return result

    def _validate_metadata_exists(
        self,
        metadata_type: str,
        columns: Optional[Union[str, List[str]]],
        df: pd.DataFrame,
    ) -> None:
        """Validate that specified metadata columns exist in the dataframe.

        Common validation logic used by both embedder versions.

        Args:
            metadata_type: Type of metadata being validated (for error messages).
            columns: Column name(s) to validate. Can be a single string or list of strings.
            df: DataFrame to check for column existence.

        Raises:
            ValueError: If columns are specified but not found in the dataframe,
                      or if metadata type is specified but no columns are provided.
        """
        if columns is None:
            return

        columns = [columns] if isinstance(columns, str) else columns

        if len(columns) == 0:
            raise ValueError(
                f"Metadata type '{metadata_type}' specified but no columns provided"
            )

        for col in columns:
            if col not in df.columns:
                raise ValueError(
                    f"Metadata type '{metadata_type}' specified but column '{col}' not found in dataframe"
                )

    @abstractmethod
    def read_or_embed_fm_text_data(
        self, data_dir: str
    ) -> Dict[str, np.ndarray]:
        """Read existing embeddings or generate new ones.

        Abstract method that must be implemented by concrete embedder classes.
        Should either load cached embeddings from files or generate new ones.

        Args:
            data_dir: Directory containing the data files and where embeddings will be saved.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping embedding types to their numpy arrays.
            The exact structure depends on the embedder version.
        """
        pass


class FoundationModelTextEmbedder(BaseFoundationModelTextEmbedder):
    def __init__(
        self,
        metadata: Dict[str, Any],
        output_dir: str,
        text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        verbose: bool = False,
    ):
        super().__init__(
            metadata=metadata,
            output_dir=output_dir,
            text_embedding_model_name=text_embedding_model_name,
            verbose=verbose,
        )
        # Initialize v1-specific paths and attributes
        self.continuous_embeddings_path = os.path.join(
            self.output_dir, "continuous_embeddings.npy"
        )
        self.retrieved_ts_embeddings_path = os.path.join(
            self.output_dir, "retrieved_ts_embeddings.npy"
        )
        self.dataset_description_embedding_path = os.path.join(
            self.output_dir, "dataset_description_embedding.npy"
        )
        self.text_description_embedding_path = os.path.join(
            self.output_dir, "text_description_embedding.npy"
        )
        self.time_varying_text_embedding_path = os.path.join(
            self.output_dir, "time_varying_text_embedding.npy"
        )

    def embed_exogenous_columns(
        self,
        exogenous_or_retrieved: Literal["exogenous", "retrieved timeseries"],
    ) -> np.ndarray:
        """Embed continuous column names.

        Extracts continuous column names from metadata and embeds them using the sentence transformer.
        Saves the resulting embeddings to the specified output directory.

        Args:
            df: DataFrame containing the data (used for validation)

        Returns:
            Numpy array with shape:
            - (len(continuous_columns_descriptions), TEXT_EMBEDDING_DIM) if continuous columns exist
            - Empty array with shape (0,) if no continuous columns exist
        """
        columns_info_dict = get_columns_info_dict_from_metadata_json(
            self.metadata, v1=True
        )
        continuous_columns_descriptions = [
            info["description"]
            for info in columns_info_dict.get("continuous", {}).values()
        ]
        if (
            continuous_columns_descriptions is None
            or len(continuous_columns_descriptions) == 0
        ):
            embeddings = np.array([])

        else:
            cols_to_embed = [
                f"This is a {exogenous_or_retrieved} column: {col}"
                for col in continuous_columns_descriptions
            ]
            embeddings = (
                self.embed_text(cols_to_embed)
                if cols_to_embed
                else np.array([])
            )

        np.save(self.continuous_embeddings_path, embeddings)
        if self.verbose:
            logger.info(
                f"Saved continuous column embeddings to {self.continuous_embeddings_path}"
            )
        return embeddings

    def embed_retrieved_ts_columns(self) -> np.ndarray:
        """Embed retrieved timeseries column names.

        Currently a placeholder for future functionality. In the current implementation,
        this always returns an empty array as retrieved timeseries are not yet supported.
        Saves the resulting (empty) embeddings to the specified output directory.

        Returns:
            Empty numpy array with shape (0,)
        """
        # TODO: For now, we don't have retrieved timeseries, so we'll return an empty array
        # This can be extended if retrieved timeseries are added in the future
        empty_array = np.array([])
        np.save(self.retrieved_ts_embeddings_path, empty_array)
        if self.verbose:
            logger.info(
                f"Saved retrieved timeseries column embeddings to {self.retrieved_ts_embeddings_path}"
            )
        return empty_array

    def embed_dataset_description(self) -> np.ndarray:
        """Embed dataset description.

        Embeds the dataset title from metadata using the sentence transformer.
        Saves the resulting embedding to the specified output directory.

        Returns:
            Numpy array with shape:
            - (TEXT_EMBEDDING_DIM,) if dataset title exists
            - Empty array with shape (0,) if no dataset title exists
        """
        dataset_description = self.metadata.get("title", None)
        if dataset_description is None:
            embedding = np.array([])
        else:
            embedding = self.embed_text(
                f"This is dataset description: {dataset_description}"
            )
        np.save(self.dataset_description_embedding_path, embedding)
        if self.verbose:
            logger.info(
                f"Saved dataset description embedding to {self.dataset_description_embedding_path}"
            )
        return embedding

    # TODO: - (len(df), TEXT_EMBEDDING_DIM) if text description column exists
    def embed_text_description(self, df: pd.DataFrame) -> np.ndarray:
        """Embed text description for each row in the dataframe.

        Embeds the text in the column specified by 'text_description_column' in metadata.
        Missing values are handled by the embed_text method, which replaces them with zero embeddings.
        Saves the resulting embeddings to the specified output directory.

        Args:
            df: DataFrame containing the text description column

        Returns:
            Numpy array with shape:
            - (TEXT_EMBEDDING_DIM,) if text description column exists
            - Empty array with shape (0,) if no text description column is specified
        """
        text_description_column = self.metadata.get(
            "text_description_column", None
        )

        # Validate that the column exists in the dataframe if specified
        self._validate_metadata_exists(
            "text_description", text_description_column, df
        )

        # If column name wasn't provided, return empty array
        if text_description_column is None:
            embedding = np.array([])
        else:
            # Embed all values in the text description column
            text_descriptions = df[text_description_column].tolist()[0]
            embedding = self.embed_text(
                f"This is a window text description: {text_descriptions}"
            )

        np.save(self.text_description_embedding_path, embedding)
        if self.verbose:
            logger.info(
                f"Saved text description embeddings to {self.text_description_embedding_path}"
            )
        return embedding

    def embed_time_varying_text_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """Embed time varying text metadata for each row in the dataframe.

        Embeds the text in the column specified by 'textual_metadata_column' in metadata.
        Missing values are handled by the embed_text method, which replaces them with zero embeddings.
        Saves the resulting embeddings to the specified output directory.

        Args:
            df: DataFrame containing the textual metadata column

        Returns:
            Numpy array with shape:
            - (len(df), TEXT_EMBEDDING_DIM) if textual metadata column exists
            - Empty array with shape (0,) if no textual metadata column is specified
        """
        textual_metadata_column = self.metadata.get(
            "textual_metadata_column", None
        )

        # Validate that the column exists in the dataframe if specified
        self._validate_metadata_exists(
            "time_varying_textual_metadata", textual_metadata_column, df
        )

        # If column name wasn't provided, return empty array
        if textual_metadata_column is None:
            embedding = np.array([])
        else:
            # Embed each row's textual metadata
            text_values = df[textual_metadata_column].tolist()
            text_values = [
                f"This is a time varying text metadata: {text}"
                for text in text_values
            ]
            embedding = self.embed_text(text_values)

        np.save(self.time_varying_text_embedding_path, embedding)
        if self.verbose:
            logger.info(
                f"Saved time varying text metadata embeddings to {self.time_varying_text_embedding_path}"
            )
        return embedding

    def read_or_embed_fm_text_data(
        self, data_dir: str
    ) -> Dict[str, np.ndarray]:
        """Embed all text data or load existing embeddings.

        This method either loads existing embeddings from files if they exist,
        or generates new embeddings for all text data types and saves them to files.
        This allows for caching of embeddings to avoid redundant computation.

        Args:
            df: DataFrame containing the data with text columns to embed

        Returns:
            Dict containing:
            - continuous_embeddings: Array of shape (n_continuous_cols, TEXT_EMBEDDING_DIM) or (0,)
            - retrieved_ts_embeddings: Array of shape (0,) (placeholder for future functionality)
            - dataset_description_embedding: Array of shape (TEXT_EMBEDDING_DIM,) or (0,)
            - text_description_embedding: Array of shape (len(df), TEXT_EMBEDDING_DIM) or (0,)
            - time_varying_text_embedding: Array of shape (len(df), TEXT_EMBEDDING_DIM) or (0,)
        """

        # Check if all embedding files exist and load them if so
        if all(
            os.path.exists(path)
            for path in [
                self.continuous_embeddings_path,
                self.retrieved_ts_embeddings_path,
                self.dataset_description_embedding_path,
                self.text_description_embedding_path,
                self.time_varying_text_embedding_path,
            ]
        ):
            if self.verbose:
                logger.info("Loading existing embeddings from files")
            continuous_embeddings = np.load(self.continuous_embeddings_path)
            retrieved_ts_embeddings = np.load(self.retrieved_ts_embeddings_path)
            dataset_description_embedding = np.load(
                self.dataset_description_embedding_path
            )
            text_description_embedding = np.load(
                self.text_description_embedding_path
            )
            time_varying_text_embedding = np.load(
                self.time_varying_text_embedding_path
            )
        else:
            df = load_dataframe_from_directory(data_dir)
            if self.verbose:
                logger.info("Generating new embeddings")
            # Only initialize the model when we need to generate new embeddings
            if self.model is None:
                self.model = self._initialize_sentence_transformer()
            continuous_embeddings = self.embed_exogenous_columns(
                exogenous_or_retrieved="exogenous"
            )
            retrieved_ts_embeddings = self.embed_retrieved_ts_columns()
            dataset_description_embedding = self.embed_dataset_description()
            text_description_embedding = self.embed_text_description(df)
            time_varying_text_embedding = self.embed_time_varying_text_metadata(
                df
            )
        return {
            "continuous_descriptions_embeddings": continuous_embeddings,
            "retrieved_ts_descriptions_embeddings": retrieved_ts_embeddings,
            "dataset_description_embedding": dataset_description_embedding,
            "text_description_embedding": text_description_embedding,
            "time_varying_text_embedding": time_varying_text_embedding,
        }


class FoundationModelTextEmbedderV2(BaseFoundationModelTextEmbedder):
    def __init__(
        self,
        metadata: Dict[str, Any],
        output_dir: str,
        text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        verbose: bool = False,
    ):
        super().__init__(
            metadata=metadata,
            output_dir=output_dir,
            text_embedding_model_name=text_embedding_model_name,
            verbose=verbose,
        )
        # Initialize v2-specific paths and attributes
        self.timeseries_descriptions_embeddings_path = os.path.join(
            self.output_dir, "timseries_embeddings.npy"
        )

        self.continuous_descriptions_embeddings_path = os.path.join(
            self.output_dir, "continuous_embeddings.npy"
        )
        self.retrieved_ts_descriptions_embeddings_path = os.path.join(
            self.output_dir, "retrieved_ts_embeddings.npy"
        )
        self.dataset_description_embedding_path = os.path.join(
            self.output_dir, "dataset_description_embedding.npy"
        )
        self.text_description_embedding_path = os.path.join(
            self.output_dir, "text_description_embedding.npy"
        )
        self.time_varying_text_descriptions_embedding_path = os.path.join(
            self.output_dir, "time_varying_text_descriptions_embedding.npy"
        )
        self.time_varying_text_embedding_path = os.path.join(
            self.output_dir, "time_varying_text_embedding.npy"
        )

    def embed_timeseries_columns_descriptions(
        self, columns_info_dict: Dict[str, Any]
    ) -> np.ndarray:
        """Embed timeseries columns descriptions.

        Extracts timeseries column descriptions from metadata and embeds them using the sentence transformer.
        Saves the resulting embeddings to the specified output directory.

        Returns:
            Numpy array with shape:
            - (len(timeseries_columns_descriptions), TEXT_EMBEDDING_DIM) if timeseries columns exist
            - Empty array with shape (0,) if no timeseries columns descriptions exist
        """
        timeseries_columns_descriptions = [
            info["description"]
            for info in columns_info_dict.get("timeseries", {}).values()
        ]

        embeddings = (
            self.embed_text(timeseries_columns_descriptions)
            if timeseries_columns_descriptions
            else np.array([])
        )
        np.save(self.timeseries_descriptions_embeddings_path, embeddings)
        logger.info(
            f"Saved timeseries column embeddings to {self.timeseries_descriptions_embeddings_path}"
        )
        return embeddings

    def embed_retrieved_ts_columns_descriptions(self) -> np.ndarray:
        """Embed retrieved timeseries column names.

        Currently a placeholder for future functionality. In the current implementation,
        this always returns an empty array as retrieved timeseries are not yet supported.
        Saves the resulting (empty) embeddings to the specified output directory.

        Returns:
            Empty numpy array with shape (0,)
        """
        # TODO: For now, we don't have retrieved timeseries, so we'll return an empty array
        # This can be extended if retrieved timeseries are added in the future
        empty_array = np.array([])
        np.save(self.retrieved_ts_descriptions_embeddings_path, empty_array)
        logger.info(
            f"Saved retrieved timeseries column embeddings to {self.retrieved_ts_descriptions_embeddings_path}"
        )
        return empty_array

    def embed_time_varying_text_columns_descriptions(
        self, columns_info_dict: Dict[str, Any]
    ) -> np.ndarray:
        """Embed time_varying_text columns descriptions.

        Extracts time_varying_text column descriptions from metadata and embeds them using the sentence transformer.
        Saves the resulting embeddings to the specified output directory.

        Returns:
            Numpy array with shape:
            - (len(time_varying_text_columns_descriptions), TEXT_EMBEDDING_DIM) if time_varying_text columns exist
            - Empty array with shape (0,) if no time_varying_text columns descriptions exist
        """
        time_varying_text_columns_descriptions = [
            info["description"]
            for info in columns_info_dict.get("text", {}).values()
        ]

        embeddings = (
            self.embed_text(time_varying_text_columns_descriptions)
            if time_varying_text_columns_descriptions
            else np.array([])
        )
        np.save(self.time_varying_text_descriptions_embedding_path, embeddings)
        logger.info(
            f"Saved time_varying_text column embeddings to {self.time_varying_text_descriptions_embedding_path}"
        )
        return embeddings

    def embed_time_varying_text_metadata(
        self, df: pd.DataFrame, columns_info_dict: Dict[str, Any]
    ) -> np.ndarray:
        """Embed time varying text metadata for each row in the dataframe.

        Embeds the text in columns specified by 'time_varying_text_columns' from metadata.
        Each column is embedded separately, resulting in multiple embeddings per row.
        Missing values are handled by the embed_text method, which replaces them with zero embeddings.
        Saves the resulting embeddings to the specified output directory.

        Args:
            df: DataFrame containing the time varying text columns

        Returns:
            Numpy array with shape:
            - (len(df), TEXT_EMBEDDING_DIM, num_time_varying_columns) if time varying text columns exist
            - Empty array with shape (0,) if no time varying text columns are specified
        """
        time_varying_text_columns = list(
            columns_info_dict.get("text", {}).keys()
        )
        # If no time varying text columns are provided, return empty array
        if not time_varying_text_columns:
            embedding = np.array([])
        else:
            all_column_embeddings = []

            # Process each time varying text column separately
            for col in time_varying_text_columns:
                if col in df.columns:
                    # Get all values for this column
                    text_values = df[col].tolist()
                    # Format and embed
                    formatted_texts = [
                        f"This is a time varying text metadata: {text}"
                        for text in text_values
                    ]
                    column_embedding = self.embed_text(formatted_texts)
                    all_column_embeddings.append(column_embedding)
                else:
                    # Column not found, create zeros
                    zero_embedding = np.zeros((len(df), TEXT_EMBEDDING_DIM))
                    all_column_embeddings.append(zero_embedding)

            # Stack the embeddings along a new axis to get shape (num_columns, num_rows, embedding_dim)
            stacked = np.stack(all_column_embeddings, axis=0)

            # Transpose to get shape (num_rows, embedding_dim, num_columns)
            embedding = np.transpose(stacked, (1, 2, 0))

        np.save(self.time_varying_text_embedding_path, embedding)
        logger.info(
            f"Saved time varying text metadata embeddings to {self.time_varying_text_embedding_path}"
        )
        return embedding

    def read_or_embed_fm_text_data(
        self, data_dir: str
    ) -> Dict[str, np.ndarray]:
        """Embed all text data or load existing embeddings.

        This method either loads existing embeddings from files if they exist,
        or generates new embeddings for all text data types and saves them to files.
        This allows for caching of embeddings to avoid redundant computation.

        Args:
            data_dir: Directory containing the data with text columns to embed

        Returns:
            Dict containing:
            - timeseries_descriptions_embeddings: Array of shape (n_timeseries_cols, TEXT_EMBEDDING_DIM) or (0,)
            - retrieved_ts_descriptions_embeddings: Array of shape (0,) (placeholder for future functionality)
            - time_varying_text_descriptions_embeddings: Array of shape (n_time_varying_cols, TEXT_EMBEDDING_DIM) or (0,)
            - time_varying_text_embeddings: Array of shape (len(df), TEXT_EMBEDDING_DIM, n_time_varying_cols) or (0,)
        """

        # Check if all embedding files exist and load them if so
        if all(
            os.path.exists(path)
            for path in [
                self.timeseries_descriptions_embeddings_path,
                self.retrieved_ts_descriptions_embeddings_path,
                self.time_varying_text_descriptions_embedding_path,
                self.time_varying_text_embedding_path,
            ]
        ):
            logger.info("Loading existing embeddings from files")
            timeseries_descriptions_embeddings = np.load(
                self.timeseries_descriptions_embeddings_path
            )
            retrieved_ts_descriptions_embeddings = np.load(
                self.retrieved_ts_descriptions_embeddings_path
            )
            time_varying_text_descriptions_embeddings = np.load(
                self.time_varying_text_descriptions_embedding_path
            )
            time_varying_text_embeddings = np.load(
                self.time_varying_text_embedding_path
            )
        else:
            columns_info_dict = get_columns_info_dict_from_metadata_json(
                self.metadata, v1=False
            )
            df = load_dataframe_from_directory(data_dir)
            self.model = self._initialize_sentence_transformer()
            logger.info("Generating new embeddings")
            timeseries_descriptions_embeddings = (
                self.embed_timeseries_columns_descriptions(columns_info_dict)
            )
            retrieved_ts_descriptions_embeddings = (
                self.embed_retrieved_ts_columns_descriptions()
            )
            time_varying_text_descriptions_embeddings = (
                self.embed_time_varying_text_columns_descriptions(
                    columns_info_dict
                )
            )
            time_varying_text_embeddings = (
                self.embed_time_varying_text_metadata(df, columns_info_dict)
            )

        return {
            "timeseries_descriptions_embeddings": timeseries_descriptions_embeddings,
            "retrieved_ts_descriptions_embeddings": retrieved_ts_descriptions_embeddings,
            "time_varying_text_descriptions_embeddings": time_varying_text_descriptions_embeddings,
            "time_varying_text_embeddings": time_varying_text_embeddings,
        }


def main():
    """
    Main function to run the text embedding pipeline from command line.

    This function parses command line arguments, loads data from specified directories,
    initializes the appropriate text embedder version (v1 or v2), and generates text embeddings
    for all data types. It can process multiple directories in sequence.

    Usage:
        python3 src/synthefy_pkg/preprocessing/fm_text_embedder.py --data_dir /path/to/data_directory  --version v1
        python3 src/synthefy_pkg/preprocessing/fm_text_embedder.py --data_dir /path/to/data_directory  --version v2
    """
    parser = argparse.ArgumentParser(
        description="Generate text embeddings for foundation models"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",  # Accept one or more arguments
        required=True,
        help="Directory or list of directories containing data to embed",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the sentence transformer model to use",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["v1", "v2"],
        help="Version of the text embedder to use: v1 (original) or v2",
    )

    args = parser.parse_args()
    # Process each directory
    for data_dir in args.data_dir:
        logger.info(f"Processing directory: {data_dir}")

        # Check if there are parquet files in the current directory
        parquet_files = [
            f for f in os.listdir(data_dir) if f.endswith(".parquet")
        ]

        if not parquet_files:
            logger.info(
                f"No parquet files found in {data_dir}, searching subdirectories..."
            )
            # Walk through subdirectories to find parquet files
            subdirs_with_parquet = []
            for root, _, files in os.walk(data_dir):
                parquet_files = [f for f in files if f.endswith(".parquet")]
                if parquet_files:
                    subdirs_with_parquet.append(root)

            # Sort subdirectories by the integer at the end of the directory name
            def get_dir_number(dir_path):
                try:
                    # Extract the last part of the path and split by underscore
                    dir_name = os.path.basename(dir_path)
                    # Take the last element after splitting by underscore and convert to int
                    return int(dir_name.split("_")[-1])
                except (ValueError, IndexError):
                    # If conversion fails, return a large number to sort it last
                    return float("inf")

            # Process subdirectories sorted by their numeric suffix
            for root in sorted(subdirs_with_parquet, key=get_dir_number):
                logger.info(f"Found parquet files in subdirectory: {root}")
                # Process this subdirectory
                metadata = load_metadata_from_directory(root)

                # Initialize text embedder based on version
                if args.version == "v1":
                    text_embedder = FoundationModelTextEmbedder(
                        metadata=metadata,
                        output_dir=root,
                        text_embedding_model_name=args.model_name,
                        verbose=True,
                    )
                    logger.info(
                        f"Generating v1 text embeddings for data in {root}"
                    )
                    text_embedder.read_or_embed_fm_text_data(root)
                else:  # version v2
                    text_embedder = FoundationModelTextEmbedderV2(
                        metadata=metadata,
                        output_dir=root,
                        text_embedding_model_name=args.model_name,
                    )
                    logger.info(
                        f"Generating v2 text embeddings for data in {root}"
                    )
                    text_embedder.read_or_embed_fm_text_data(root)

                logger.info(f"Text embedding generation complete for {root}!")
        else:
            # Process the current directory directly
            metadata = load_metadata_from_directory(data_dir)

            # Initialize text embedder based on version
            if args.version == "v1":
                text_embedder = FoundationModelTextEmbedder(
                    metadata=metadata,
                    output_dir=data_dir,
                    text_embedding_model_name=args.model_name,
                    verbose=True,
                )
                logger.info(
                    f"Generating v1 text embeddings for data in {data_dir}"
                )
                text_embedder.read_or_embed_fm_text_data(data_dir)
            else:  # version v2
                text_embedder = FoundationModelTextEmbedderV2(
                    metadata=metadata,
                    output_dir=data_dir,
                    text_embedding_model_name=args.model_name,
                )
                logger.info(
                    f"Generating v2 text embeddings for data in {data_dir}"
                )
                text_embedder.read_or_embed_fm_text_data(data_dir)

            logger.info(f"Text embedding generation complete for {data_dir}!")
    logger.info("All directories processed successfully!")


if __name__ == "__main__":
    main()
