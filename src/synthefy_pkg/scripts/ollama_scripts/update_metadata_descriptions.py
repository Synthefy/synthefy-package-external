import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger


def load_json_file(file_path: Path) -> Dict:
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(file_path: Path, data: Dict) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_ollama_responses(
    series_dir: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """Load extract and expand responses from Ollama response files."""
    extract_file = series_dir / f"{series_dir.name}_extract_response.json"
    expand_file = series_dir / f"{series_dir.name}_expand_response.json"

    extracted_info = None
    expanded_description = None

    if extract_file.exists():
        response_data = load_json_file(extract_file)
        extracted_info = response_data.get("response", "").strip()
    else:
        logger.warning(f"Extract response file not found: {extract_file}")

    if expand_file.exists():
        response_data = load_json_file(expand_file)
        expanded_description = response_data.get("response", "").strip()
    else:
        logger.warning(f"Expand response file not found: {expand_file}")

    return extracted_info, expanded_description


def process_extracted_info(extracted_info: str, metadata: Dict) -> str:
    """Process extracted info to ensure it includes time period, frequency and original description."""
    info_parts = [extracted_info]

    # Add time period if available
    start_date = metadata.get("start_date", "")
    end_date = metadata.get("end_date", "")
    if start_date and end_date:
        time_period = f"{start_date} to {end_date}"
        info_parts.append(f"time_period: {time_period}")

    # Add frequency if available
    frequency = metadata.get("frequency", "")
    if frequency:
        info_parts.append(f"frequency: {frequency}")

    # Get original description from the first column
    original_desc = metadata["columns"][0]["description"]
    info_parts.append(f"title: {original_desc}")

    # Combine all parts with commas
    processed_info = ", ".join(info_parts)

    return processed_info.strip()


def update_metadata_file(metadata_file: Path) -> None:
    """Update a single metadata file with Ollama-generated descriptions."""
    try:
        # Load metadata
        metadata = load_json_file(metadata_file)

        # Load Ollama responses
        extracted_info, expanded_description = load_ollama_responses(
            metadata_file.parent
        )

        if not extracted_info:
            logger.error(f"No extracted info found for {metadata_file}")
            return

        # Process extracted info
        processed_extracted_info = process_extracted_info(
            extracted_info, metadata
        )

        # Save original description and update with processed info
        metadata["columns"][0]["old_description"] = metadata["columns"][0][
            "description"
        ]
        metadata["columns"][0]["description"] = processed_extracted_info

        # Add expanded description if available
        if expanded_description:
            metadata["columns"][0]["generated_description"] = (
                expanded_description
            )

        # Also update the top-level description to match
        metadata["description"] = processed_extracted_info

        # Save updated metadata
        save_json_file(metadata_file, metadata)
        logger.info(f"Successfully updated metadata file: {metadata_file}")

    except Exception as e:
        logger.error(f"Error processing metadata file {metadata_file}: {e}")


def process_directory(base_dir: Path) -> None:
    """Process all metadata files in subdirectories."""
    try:
        # Find all subdirectories
        subdirs = [
            d
            for d in base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        for subdir in subdirs:
            # Look for metadata file in subdir
            metadata_file = subdir / f"{subdir.name}_metadata.json"

            if metadata_file.exists():
                update_metadata_file(metadata_file)
            else:
                logger.warning(f"No metadata file found in {subdir}")

    except Exception as e:
        logger.error(f"Error processing directory {base_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Update metadata files with Ollama-generated descriptions"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing series subdirectories",
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return

    # Configure logging
    log_file = base_dir / "metadata_update.log"
    logger.add(log_file, rotation="100 MB", level="INFO")

    logger.info(f"Starting metadata update process in {base_dir}")
    process_directory(base_dir)
    logger.info("Metadata update process completed")


if __name__ == "__main__":
    main()
