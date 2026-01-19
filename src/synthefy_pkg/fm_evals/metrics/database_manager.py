"""
Database management for evaluation results.

This module provides functionality to save evaluation results to a CSV database
that can be used for tracking and comparing model performance across runs.
"""

import os
from typing import Optional

import pandas as pd

from synthefy_pkg.fm_evals.formats.dataset_result_format import (
    DatasetResultFormat,
)
from synthefy_pkg.fm_evals.formats.metrics import SUPPORTED_METRICS


class DatabaseManager:
    """
    Database management for evaluation results.

    This module provides functionality to save evaluation results to a CSV database
    that can be used for tracking and comparing model performance across runs.
    """

    def __init__(self, output_directory: str):
        """
        Initialize the database manager.

        Parameters
        ----------
        output_directory : str
            Directory where the database CSV file will be stored
        """
        self.output_directory = output_directory
        self.database_path = os.path.join(
            output_directory, "results_database.csv"
        )
        self._ensure_database_exists()

    def _get_database_schema(self) -> list:
        """
        Generate database schema from SUPPORTED_METRICS.
        """
        base_columns = [
            "Dataset",
            "Model",
            "run_id",
            "results_path",
            "git_hash",
        ]
        metric_columns = SUPPORTED_METRICS
        return base_columns + metric_columns

    def _ensure_database_exists(self):
        """
        Create the database file if it doesn't exist.
        """
        if not os.path.exists(self.database_path):
            # Create empty database with proper schema
            schema = self._get_database_schema()
            empty_df = pd.DataFrame(columns=schema)
            empty_df.to_csv(self.database_path, index=False)

    def save_results(
        self,
        dataset_name: str,
        model_name: str,
        dataset_result: DatasetResultFormat,
        run_id: str,
        results_path: str,
        git_hash: Optional[str] = None,
    ):
        """
        Save evaluation results to the database.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset used for evaluation
        model_name : str
            Name of the model that was evaluated
        dataset_result : DatasetResultFormat
            The evaluation results to save
        run_id : str
            Unique identifier for this evaluation run
        results_path : str
            Path to the detailed results on disk
        git_hash : Optional[str]
            Git commit hash for reproducibility (default: None)
        """
        # Convert DatasetResultFormat to database row
        row = self._convert_to_database_row(
            dataset_name=dataset_name,
            model_name=model_name,
            dataset_result=dataset_result,
            run_id=run_id,
            results_path=results_path,
            git_hash=git_hash,
        )

        # Load existing database
        if os.path.exists(self.database_path):
            df = pd.read_csv(self.database_path)
        else:
            # Create new database with proper schema
            df = pd.DataFrame(columns=self._get_database_schema())

        # Append new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Save updated database
        df.to_csv(self.database_path, index=False)

    def _convert_to_database_row(
        self,
        dataset_name: str,
        model_name: str,
        dataset_result: DatasetResultFormat,
        run_id: str,
        results_path: str,
        git_hash: Optional[str],
    ) -> dict:
        """
        Convert DatasetResultFormat to database row format.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        model_name : str
            Name of the model
        dataset_result : DatasetResultFormat
            The evaluation results
        run_id : str
            Unique run identifier
        results_path : str
            Path to results on disk
        git_hash : Optional[str]
            Git commit hash

        Returns
        -------
        dict
            Database row as dictionary
        """
        row = {
            "Dataset": dataset_name,
            "Model": model_name,
            "run_id": run_id,
            "results_path": results_path,
            "git_hash": git_hash,
        }

        # Add all metrics dynamically
        if dataset_result.metrics:
            for metric_name in SUPPORTED_METRICS:
                row[metric_name] = getattr(
                    dataset_result.metrics, metric_name, None
                )
        else:
            # If no metrics available, set all metric columns to None
            for metric_name in SUPPORTED_METRICS:
                row[metric_name] = None

        return row

    def get_database_path(self) -> str:
        """
        Get the path to the database file.
        """
        return self.database_path
