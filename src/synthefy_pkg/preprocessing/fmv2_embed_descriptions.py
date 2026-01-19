#!/usr/bin/env python
"""
Dataset Description Embedder

This script processes multiple datasets by:
1. Reading metadata from each dataset directory
2. Extracting column descriptions/titles
3. Embedding these descriptions using a sentence transformer
4. Saving the embeddings to each dataset directory

Usage:
    python dataset_description_embedder.py --data_dir /path/to/parent/directory
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import text embedding dimension from your existing module
from synthefy_pkg.preprocessing.fm_text_embedder import TEXT_EMBEDDING_DIM
from synthefy_pkg.utils.fm_utils import load_metadata_from_directory

CHUNK_SIZE = 1000  # Adjust based on your memory constraints


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Embed dataset descriptions and save to individual dataset folders"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Parent directory containing all dataset folders",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the sentence transformer model to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    return parser.parse_args()


def find_dataset_dirs(parent_dir: str) -> List[str]:
    """Find all dataset directories within the parent directory."""
    # Look for directories that match the pattern 'name_number'
    dataset_dirs = []
    for item in os.listdir(parent_dir):
        path = os.path.join(parent_dir, item)
        if os.path.isdir(path) and "_" in item:
            try:
                # Check if the part after the underscore is a number
                _ = int(item.split("_")[-1])
                dataset_dirs.append(path)
            except ValueError:
                continue

    return dataset_dirs


def embed_descriptions(
    descriptions: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str] = None,
) -> np.ndarray:
    """Embed a list of descriptions using a sentence transformer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set a smaller batch size for GPU to avoid memory issues
    if device == "cuda" and batch_size > 16:
        logger.info(
            f"Reducing batch size from {batch_size} to 16 for GPU processing"
        )
        batch_size = 16

    try:
        # Load the model
        model = SentenceTransformer(model_name)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # Encode all descriptions
            embeddings = model.encode(
                descriptions,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True,
            )

            # Convert to numpy array
            return embeddings.cpu().numpy()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("GPU out of memory. Falling back to CPU")
            return embed_descriptions(
                descriptions, model_name, batch_size, device="cpu"
            )
        else:
            raise


def main():
    """Main function to process datasets and save embeddings."""
    args = parse_args()

    # Find all dataset directories
    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.verbose:
        print(f"Found {len(dataset_dirs)} dataset directories")

    # Create dictionary mapping dataset names to directories
    dataset_dir_dict = {os.path.basename(d): d for d in dataset_dirs}

    # Extract descriptions for each dataset
    descriptions_dict = {}
    for dataset_name, dataset_dir in tqdm(
        dataset_dir_dict.items(), desc="Extracting descriptions"
    ):
        try:
            metadata = load_metadata_from_directory(dataset_dir)
            description = metadata["columns"][0].get("title", None)
            if description is None:
                logger.warning(
                    f"No description found for {dataset_name}; skppint"
                )
                continue

            descriptions_dict[dataset_name] = description
            if args.verbose:
                print(f"{dataset_name}: {description}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

    # Get list of descriptions in the same order as dataset_names
    dataset_names = list(descriptions_dict.keys())
    descriptions = [descriptions_dict[name] for name in dataset_names]

    for i in range(0, len(descriptions), CHUNK_SIZE):
        chunk_names = dataset_names[i : i + CHUNK_SIZE]
        chunk_descriptions = descriptions[i : i + CHUNK_SIZE]

        print(
            f"Embedding chunk {i // CHUNK_SIZE + 1}/{len(descriptions) // CHUNK_SIZE + 1}..."
        )
        chunk_embeddings = embed_descriptions(
            descriptions=chunk_descriptions,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )

        # Save embeddings for this chunk
        for name, embedding in zip(chunk_names, chunk_embeddings):
            dataset_dir = dataset_dir_dict[name]
            output_path = os.path.join(dataset_dir, "description_embedding.npy")
            np.save(output_path, embedding)
            if args.verbose:
                print(f"Saved embedding to {output_path}")

    print(f"Successfully processed {len(descriptions_dict)} datasets")


if __name__ == "__main__":
    main()
