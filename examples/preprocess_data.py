import argparse

from synthefy_pkg.data.synthefy_dataset import SynthefyDataset
from synthefy_pkg.preprocessing.preprocess import DataPreprocessor


def main(config_path: str, skip_embedding: bool = False):
    """
    Preprocess data and optionally embed timeseries for search.

    Args:
        config_path: Path to configuration file
        skip_embedding: If True, skip the embedding step
    """
    # Initialize and run the preprocessor
    preprocessor = DataPreprocessor(config_source=config_path)
    preprocessor.process_data()

    # Run embedding unless explicitly skipped
    if not skip_embedding:
        dataset = SynthefyDataset(config_source=config_path)
        dataset.load_windows(window_types=["timeseries"])
        preprocessor.embed_timeseries_for_search(
            dataset.windows_data_dict["timeseries"]["windows"],
            preprocessor.encoder_type,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="config file path", required=True
    )
    parser.add_argument(
        "--skip_timeseries_embeddings",
        action="store_true",
        help="Skip feature extraction",
    )
    args = parser.parse_args()

    main(args.config, skip_embedding=args.skip_timeseries_embeddings)
