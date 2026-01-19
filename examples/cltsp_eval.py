import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from synthefy_pkg.experiments.cltsp_experiment import CLTSPExperiment


def get_cosine_distance(
    timeseries_embeddings: np.ndarray, condition_embeddings: np.ndarray
) -> np.ndarray:
    """
    Shape is (n_samples, embedding_dim) for both timeseries_embeddings and condition_embeddings

    Returns a numpy array of shape (n_samples,)
    """

    # Calculate cosine similarity between corresponding pairs
    similarities = np.array(
        [
            cosine_similarity(ts.reshape(1, -1), cn.reshape(1, -1))[0][0]
            for ts, cn in zip(timeseries_embeddings, condition_embeddings)
        ]
    )

    return similarities


def main(config_filepath: str, model_checkpoint_path: str, h5_file_path: str):
    experiment = CLTSPExperiment(config_filepath)
    predictions = experiment.predict(
        model_checkpoint_path, h5_file_path, synthetic_or_original="original"
    )

    timeseries_embeddings = predictions["timeseries_embeddings"]
    condition_embeddings = predictions["condition_embeddings"]

    cosine_distances = get_cosine_distance(
        timeseries_embeddings, condition_embeddings
    )

    df = pd.DataFrame(
        {
            "cosine_distances": cosine_distances,
        }
    )

    df.to_csv(
        f"cltsp_eval_datetime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_ppg_hr_samsung_cltsp.yaml",
        required=True,
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--h5_file_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.config, args.model_checkpoint_path, args.h5_file_path)
