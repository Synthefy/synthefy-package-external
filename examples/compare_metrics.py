import argparse

from synthefy_pkg.scripts.compare_metrics import main

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Compare metrics between base and new JSON files."
    )
    parser.add_argument(
        "base_file", type=str, help="Path to the base metrics JSON file"
    )
    parser.add_argument("new_file", type=str, help="Path to the new metrics JSON file")
    args = parser.parse_args()

    main(args)
