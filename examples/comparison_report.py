import argparse
import os

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.postprocessing.report_comparator import ReportComparator

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the postprocess HTML report and convert it to PDF."
    )
    parser.add_argument("--config", type=str, help="Path to config file", required=True)

    parser.add_argument(
        "--use_scaled_data",
        action="store_true",
        help="Whether to use scaled data for comparison",
        default=False,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        help="Number of top results to show in the report",
        default=1,
    )

    args = parser.parse_args()

    # Get the output path where the comparison data is stored
    configuration = Configuration(config_filepath=args.config)
    save_dir = configuration.get_save_dir(SYNTHEFY_DATASETS_BASE)
    if args.use_scaled_data:
        output_path = output_path = os.path.join(save_dir, "comparison_eval", "scaled")
        html_report_path = os.path.join(
            save_dir, "reports", "comparison_report_scaled.html"
        )
        pdf_report_path = os.path.join(
            save_dir, "reports", "comparison_report_scaled.pdf"
        )
    else:
        output_path = output_path = os.path.join(
            save_dir, "comparison_eval", "unscaled"
        )
        html_report_path = os.path.join(
            save_dir, "reports", "comparison_report_unscaled.html"
        )
        pdf_report_path = os.path.join(
            save_dir, "reports", "comparison_report_unscaled.pdf"
        )

    comparator = ReportComparator(output_path=output_path)

    comparator.generate_html_report(output_html=html_report_path, top_k=args.top_k)
    comparator.convert_html_to_pdf(
        html_file=html_report_path,
        output_pdf=pdf_report_path,
    )


if __name__ == "__main__":
    main()


# Sample command to run this script:
"""
python examples/comparison_report.py \
    --config <config_path> \
    --use_scaled_data \
    --top_k 10
"""
