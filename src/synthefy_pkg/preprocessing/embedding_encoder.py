import numpy as np
import torch
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Union, List
import math

COMPILE = True


class EmbeddingEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=torch.float32, min_dim=5, max_dim=50):
        self.dtype = dtype
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.category_maps_ = {}
        self.embeddings_ = {}
        self.embedding_dims_ = {}

    @staticmethod
    def calculate_embedding_dim(
        num_unique_values: int, min_dim: int = 5, max_dim: int = 50
    ) -> int:
        """
        Calculate the embedding dimension for a column based on the number of unique values.
        Formula: min(max_dim, max(min_dim, sqrt(num_unique)))

        Args:
        - num_unique_values (int): Number of unique categories in the column.

        Returns:
        - int: Calculated embedding dimension.
        """
        return min(max_dim, max(min_dim, int(math.sqrt(num_unique_values))))

    def fit(self, X, y=None):
        """
        Fit the encoder to the data.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Input data to fit.
        - y (Ignored): Not used, included for compatibility.
        """
        if isinstance(X, pd.DataFrame):
            columns = X.columns
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            columns = range(X.shape[1])

        for col in columns:
            categories = (
                np.unique(X[col])
                if isinstance(X, pd.DataFrame)
                else np.unique(X[:, col])
            )
            num_unique_values = len(categories)
            embedding_dim = self.calculate_embedding_dim(
                num_unique_values, self.min_dim, self.max_dim
            )
            self.category_maps_[col] = {cat: idx for idx, cat in enumerate(categories)}
            self.embeddings_[col] = torch.nn.Embedding(
                num_unique_values, embedding_dim, dtype=self.dtype
            )
            self.embedding_dims_[col] = embedding_dim

        self.embedding_columns = [
            f"embed_{col}_{i}"
            for col in columns
            for i in range(self.embedding_dims_[col])
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, ["category_maps_", "embeddings_"])

        if isinstance(X, pd.DataFrame):
            columns = X.columns
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            columns = range(X.shape[1])

        self.feature_names_in_ = list(columns)
        all_embeddings = []
        for col in columns:
            indices = np.vectorize(self.category_maps_[col].get)(
                X[col] if isinstance(X, pd.DataFrame) else X[:, col]
            )
            embeddings = self.embeddings_[col](torch.tensor(indices, dtype=torch.long))
            all_embeddings.append(embeddings.detach().numpy())

        return np.hstack(all_embeddings)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def inverse_transform(self, X):
        check_is_fitted(self, ["category_maps_", "embeddings_"])

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        start = 0
        original_data = []
        for col in self.category_maps_:
            end = start + self.embedding_dims_[col]
            X_col = X[:, start:end]

            # Convert input to PyTorch tensor
            X_tensor = torch.tensor(X_col, dtype=self.dtype)

            # Get the embeddings for all categories
            all_embeddings = self.embeddings_[col].weight.detach()

            # Compute cosine similarity between input and all embeddings
            similarities = torch.nn.functional.cosine_similarity(
                X_tensor.unsqueeze(1), all_embeddings.unsqueeze(0), dim=2
            )

            # Get the indices of the most similar embeddings
            closest_indices = torch.argmax(similarities, dim=1).numpy()

            # Map indices back to original categories
            inverse_category_map = {
                idx: cat for cat, idx in self.category_maps_[col].items()
            }
            original_categories = np.vectorize(inverse_category_map.get)(
                closest_indices
            )

            original_data.append(original_categories.reshape(-1, 1))
            start = end

        return np.hstack(original_data)

    def get_feature_names_out(self) -> List[str]:
        """
        Get column names for the embedded.

        Returns:
            - List[str]: List of new column names for the embeddings.
        """
        return self.embedding_columns

    def __getstate__(self):
        state = self.__dict__.copy()
        # Store state_dict of each embedding layer
        state["embeddings_"] = {
            col: emb.state_dict() for col, emb in self.embeddings_.items()
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Load each embedding layer from its state_dict
        self.embeddings_ = {
            col: torch.nn.Embedding.from_pretrained(
                torch.tensor(state["embeddings_"][col]["weight"]).clone().detach()
            )
            for col in state["embeddings_"]
        }


if __name__ == "__main__":
    import pandas as pd

    np.random.seed(42)

    # Generate synthetic data with more diverse unique values per column
    num_rows = 1000

    data = pd.DataFrame(
        {
            "Animal": np.random.choice(
                [
                    "cat",
                    "dog",
                    "mouse",
                    "elephant",
                    "lion",
                    "tiger",
                    "bear",
                    "wolf",
                    "fox",
                    "rabbit",
                ],
                size=num_rows,
            ),
            "Color": np.random.choice(
                [
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "black",
                    "white",
                    "purple",
                    "orange",
                    "pink",
                ],
                size=num_rows,
            ),
            "Country": np.random.choice(
                [
                    "USA",
                    "Canada",
                    "Mexico",
                    "Brazil",
                    "UK",
                    "Germany",
                    "France",
                    "China",
                    "India",
                    "Australia",
                    "Japan",
                    "Russia",
                ],
                size=num_rows,
            ),
            "Product": np.random.choice(
                ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], size=num_rows
            ),
            "AgeGroup": np.random.choice(
                ["child", "teen", "adult", "senior", "middle-aged"], size=num_rows
            ),
            "Count": np.random.randint(
                1, 100, size=num_rows
            ),  # Discrete integer values
        }
    )

    print("Sample of the generated data:")
    print(data.head())
    encoder = EmbeddingEncoder()

    encoder.fit(data)

    # Show the adaptive embedding dimensions for each column
    print("\nAdaptive embedding dimensions for each column:")
    for col in data.columns:
        embedding_dim = encoder.embedding_dims_[col]
        print(f"Column '{col}': {embedding_dim}")

    # Transform the data into embeddings
    embeddings = encoder.transform(data)

    # Generate new column names for the embeddings
    embedding_columns = encoder.get_feature_names_out()

    # Create a DataFrame with embeddings and the new column names
    embedded_df = pd.DataFrame(embeddings, columns=embedding_columns)
    print("\nSample of the embedded data:")
    print(embedded_df.head())

    # Inverse transform to get the original categories back
    original_data = encoder.inverse_transform(embeddings)
    pd.DataFrame(original_data, columns=data.columns)
    print("\nOriginal Categories:")
    print(original_data[:5])
