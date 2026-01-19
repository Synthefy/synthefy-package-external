import argparse
import os

from synthefy_pkg.postprocessing.report_generator import ReportGenerator


def main():
    """Parse arguments and generate the postprocessing reports."""
    parser = argparse.ArgumentParser(
        description="Generate the postprocess HTML report and convert it to PDF."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file used to postprocess "
        "(e.g., examples/configs/forecast_configs/config_air_quality.yaml)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Override the run_name in the config file",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Use this name as the model name in the report if provided, "
        "and use the denoiser name from the config file otherwise",
        default=None,
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        help="List of split types to include in the report (e.g., train val test)",
        default=["test"],
    )
    parser.add_argument(
        "--exclude_appendix",
        action="store_true",
        help="Exclude appendix sections from the report",
    )

    args = parser.parse_args()

    include_appendix = not args.exclude_appendix

    # Initialize report generator
    report_generator = ReportGenerator(
        config_path=args.config,
        run_name=args.run_name,
        model_name=args.model_name,
        splits=args.splits,
    )

    save_dir = report_generator.postprocessor.saved_data_path
    reports_dir = os.path.join(save_dir, "reports")

    # Generate HTML report
    report_generator.generate_html_report(
        output_html=os.path.join(reports_dir, "postprocessing_report.html"),
        include_appendix=include_appendix,
    )

    # Convert to PDF
    report_generator.convert_html_to_pdf(
        html_file=os.path.join(reports_dir, "postprocessing_report.html"),
        output_pdf=os.path.join(reports_dir, "postprocessing_report.pdf"),
    )


if __name__ == "__main__":
    main()


# Example usage:
"""
# Basic usage with all splits and features:
python examples/postprocessing_report.py \
    --config examples/configs/forecast_configs/config_air_quality_forecasting.yaml \
    --splits train val test \
    --exclude_appendix
"""

# For Chronos baseline model:
"""
python examples/postprocessing_report.py \
    --config examples/configs/forecast_configs/config_air_quality_forecasting.yaml \
    --run_name synthefy_forecasting_model_v1_air_quality_chronos \
    --model_name Chronos \
    --splits train val test \
    --exclude_appendix
"""
