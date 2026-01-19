import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List

# Prompt templates
EXTRACT_PROMPT_TEMPLATE = """
You are a specialized data extraction assistant. Your task is to extract structured information from a dataset description.

## INSTRUCTIONS
1. CRITICAL FORMAT REQUIREMENT:
   - Return ONLY comma-separated key-value pairs
   - MUST use commas between each key-value pair
   - NEVER repeat information
   - NO spaces between key-value pairs, ONLY commas
   - NO explanatory text, NO notes, NO suggestions
   - NO "Here is", NO "Let me know", NO other text
   - ONLY the key-value pairs themselves

2. OMISSION RULES (CRITICAL):
   - If a field is not found in the description, OMIT THE ENTIRE key-value pair
   - Do NOT include empty values or placeholders
   - Do NOT include "unknown", "N/A", or similar values
   - Do NOT include any file size, format, or storage-related information
   - Do NOT include any technical metadata
   - Do NOT include any IDs or internal references
   - NEVER repeat a key-value pair

3. EXTRACTION RULES:
   - location: Geographic location or country covered
   - subject: What is being measured
   - source: Organization providing the data
   - frequency: Data collection frequency
   - units: Measurement units used

4. FORMAT EXAMPLE (DO NOT INCLUDE THIS EXAMPLE IN YOUR RESPONSE):
   EXAMPLE ONLY -> location: Japan,frequency: Monthly,source: Bank of Japan,units: %
   DO NOT COPY THIS EXAMPLE - GENERATE YOUR OWN KEY-VALUE PAIRS BASED ON THE DESCRIPTION

5. STRICT RULES:
   - NO quotes, braces, or JSON syntax
   - NO newlines or line breaks
   - NO explanatory text before or after key-value pairs
   - NO technical details about file size/format
   - NO placeholder values
   - NO partial or incomplete key-value pairs
   - NEVER repeat any information
   - ALWAYS use commas between pairs
   - NO extra spaces between pairs

## DESCRIPTION TO PROCESS:
{clean_info}
"""

EXPAND_PROMPT_TEMPLATE = """
You are an expert in financial and economic datasets. Based on the following dataset description, create a clear, comprehensive explanation that could be used to understand what this dataset contains. Focus on explaining what the data represents, its source, geographic coverage, time period, and any other relevant characteristics. Do NOT include information about file size, format (such as CSV or Parquet), or technical storage details. Do NOT include newlines in your response meaning \\n or \\r. Respond with ONLY the description text, without any introductions, explanations, or additional comments.

Dataset description:
{clean_info}
Description:
"""


def clean_metadata(metadata: Dict) -> Dict:
    """Clean metadata by combining relevant fields into a description dictionary.

    Args:
        metadata (Dict): Original metadata dictionary

    Returns:
        Dict: Cleaned metadata with combined description fields
    """
    cleaned = {}
    description_parts = []

    # Add title if available
    if "title" in metadata and metadata["title"]:
        description_parts.append(f"Title: {metadata['title']}")

    # Add text descriptor if available
    if "text_descriptor" in metadata and metadata["text_descriptor"]:
        description_parts.append(f"Description: {metadata['text_descriptor']}")

    # Add ID if available
    if "id" in metadata and metadata["id"]:
        description_parts.append(f"ID: {metadata['id']}")

    # Add column information if available
    if (
        "columns" in metadata
        and isinstance(metadata["columns"], list)
        and len(metadata["columns"]) > 0
        and isinstance(metadata["columns"][0], dict)
    ):
        col = metadata["columns"][0]
        if "description" in col and col["description"]:
            description_parts.append(
                f"Column Description: {col['description']}"
            )
        if "title" in col and col["title"]:
            description_parts.append(f"Column Title: {col['title']}")

    # Combine all parts with separator
    if description_parts:
        cleaned["description"] = " | ".join(description_parts)

    return cleaned


def get_subdirs(data_dir: Path) -> Iterator[Path]:
    """Get all subdirectories matching the pattern {data_dir}_{i}.

    Args:
        data_dir (Path): Base directory containing subdirectories

    Returns:
        Iterator[Path]: Iterator of matching subdirectories
    """
    base_name = data_dir.name
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith(f"{base_name}_"):
            yield subdir


def process_batch(
    subdirs: List[Path], output_file: Path, include_expand: bool = True
):
    """Process a batch of subdirectories and generate prompts.

    Args:
        subdirs (List[Path]): List of subdirectories to process
        output_file (Path): Path to save the prompts
        include_expand (bool, optional): Whether to include expand prompts. Defaults to True.
    """
    prompts = []

    for subdir in subdirs:
        try:
            # Extract series_idx from the subdirectory name
            series_idx = int(subdir.name.split("_")[-1])
        except (ValueError, IndexError):
            print(
                f"Warning: Could not extract series index from directory name: {subdir.name}"
            )
            continue

        # Construct metadata file path
        metadata_file = subdir / f"{subdir.name}_metadata.json"

        if not metadata_file.exists():
            print(f"Warning: Metadata file not found in {subdir}")
            continue

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                if not isinstance(metadata, dict):
                    print(
                        f"Warning: Unexpected metadata format in {metadata_file}, expected dictionary"
                    )
                    continue

                metadata["series_idx"] = series_idx

                # Clean the metadata
                clean_info = clean_metadata(metadata)
                if not clean_info.get("description"):
                    print(f"Warning: No description found in {metadata_file}")
                    continue

                # Create extract prompt
                extract_prompt = {
                    "series_idx": series_idx,
                    "prompt_type": "extract",
                    "content": EXTRACT_PROMPT_TEMPLATE.format(
                        clean_info=clean_info["description"]
                    ),
                }
                prompts.append(extract_prompt)

                # Create expand prompt if requested
                if include_expand:
                    expand_prompt = {
                        "series_idx": series_idx,
                        "prompt_type": "expand",
                        "content": EXPAND_PROMPT_TEMPLATE.format(
                            clean_info=clean_info["description"]
                        ),
                    }
                    prompts.append(expand_prompt)

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {metadata_file}")
        except Exception as e:
            print(f"Error processing {metadata_file}: {str(e)}")

    # Write prompts to JSONL file
    if prompts:
        with open(output_file, "w") as f:
            for prompt in prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        return len(prompts)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare combined prompts from multiple metadata files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory containing subdirectories with metadata",
    )
    parser.add_argument(
        "--include-expand",
        action="store_true",
        help="Include generation of expand prompts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of series to process in each batch (default: 1000)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return

    # Create output directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # Process subdirectories in batches
    batch = []
    batch_num = 1
    total_prompts = 0
    total_series = 0

    for subdir in get_subdirs(data_dir):
        batch.append(subdir)

        if len(batch) >= args.batch_size:
            # Process the current batch
            output_file = (
                data_dir
                / f"{data_dir.name}_combined_prompts_batch_{batch_num:04d}.jsonl"
            )
            prompts_count = process_batch(
                batch, output_file, args.include_expand
            )

            if prompts_count > 0:
                series_count = prompts_count // (
                    2 if args.include_expand else 1
                )
                total_prompts += prompts_count
                total_series += series_count
                print(
                    f"Batch {batch_num}: Processed {series_count} series ({prompts_count} prompts)"
                )
                print(f"Saved to: {output_file}")

            batch = []
            batch_num += 1

    # Process remaining subdirectories
    if batch:
        output_file = (
            data_dir
            / f"{data_dir.name}_combined_prompts_batch_{batch_num:04d}.jsonl"
        )
        prompts_count = process_batch(batch, output_file, args.include_expand)

        if prompts_count > 0:
            series_count = prompts_count // (2 if args.include_expand else 1)
            total_prompts += prompts_count
            total_series += series_count
            print(
                f"Final batch {batch_num}: Processed {series_count} series ({prompts_count} prompts)"
            )
            print(f"Saved to: {output_file}")

    # Print final summary
    if total_series > 0:
        print("\nProcessing complete:")
        print(f"Total series processed: {total_series}")
        print(f"Total prompts generated: {total_prompts}")
        print(f"Output files saved in: {data_dir}")
    else:
        print("No series were processed")


if __name__ == "__main__":
    main()
