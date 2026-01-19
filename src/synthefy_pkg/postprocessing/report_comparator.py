import os

import pdfkit
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from synthefy_pkg.postprocessing.utils import check_plot_exists

COMPILE = False


class ReportComparator:
    def __init__(self, output_path: str):
        """
        Initializes the ReportComparator with output path where the comparison plots are saved.

        Args:
            output_path (str): Path to the directory where the comparison output files were saved.
        """

        self.output_path = os.path.expanduser(output_path)

    def generate_html_report(
        self,
        output_html: str = "comparison_report.html",
        splits: list = ["test"],
        top_k: int = 1,
    ):
        """
        Generate an HTML report with comparative plots using Jinja2.

        Args:
            output_html (str): Path to save the HTML report. Defaults to "comparison_report.html".
            splits (list): List of dataset splits to include in the report. Defaults to ["test"].
            top_k (int): Number of top samples to include in the report for each split. Defaults to 1.

        Returns:
            str: HTML content to make HTML report file.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_html))
        os.makedirs(output_dir, exist_ok=True)

        # Prepare the template environment
        template_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "templates"
        )
        env = Environment(loader=FileSystemLoader(searchpath=template_dir))
        template = env.get_template("comparison_report.html")

        # Split type mapping
        split_mapping = {
            "train": "Training",
            "val": "Validation",
            "test": "Testing",
        }

        # Prepare data for MSE plot
        mse_plot_path = os.path.join(self.output_path, "mse_box_plot.png")
        check_plot_exists(mse_plot_path, f"MSE plot not found: {mse_plot_path}")

        # Prepare data for histogram plot
        hist_plot_path = os.path.join(self.output_path, "all_metrics_histogram.png")
        check_plot_exists(hist_plot_path, f"Histogram plot not found: {hist_plot_path}")

        # Prepare data for other sections
        sections = []
        for split in splits:
            section = {
                "id": split,
                "title": f"Comparison: {split_mapping[split]} Data",
                "images": [],
            }
            for i in range(top_k):

                # Get the path to the plot
                plot_path = os.path.join(
                    self.output_path,
                    f"{split}_timeseries_comparison",
                    f"comparison_samples_{i}.png",
                )

                # Check if the plot exists
                check_plot_exists(plot_path, f"Plot not found: {plot_path}")

                # Add the plot to the section
                section["images"].append(
                    {
                        "path": f"file://{plot_path}",
                        "title": f"{split.capitalize()} Sample {i}",
                    }
                )
            sections.append(section)

        # Render the HTML
        html_content = template.render(
            mse_plot_path=f"file://{mse_plot_path}",
            hist_plot_path=f"file://{hist_plot_path}",
            sections=sections,
        )

        # Save the HTML content to file
        with open(output_html, "w", encoding="utf-8") as file:
            file.write(html_content)

        logger.info(f"HTML report saved to {output_html}")

        return html_content

    def convert_html_to_pdf(
        self,
        html_file: str = "postprocessing_report.html",
        output_pdf: str = "postprocessing_report.pdf",
        wkhtmltopdf_path: str = "/usr/bin/wkhtmltopdf",
    ):
        """
        Convert an HTML file to a PDF.

        :param html_file: Path to the input HTML file. Defaults to "postprocessing_report.html".
        :param output_pdf: Path to save the generated PDF file. Defaults to "postprocessing_report.pdf".
        :param wkhtmltopdf_path: Path to the wkhtmltopdf binary. Defaults to "/usr/bin/wkhtmltopdf".

        :return: None. The PDF is saved at the specified path.
        """
        options = {"enable-local-file-access": True, "enable-internal-links": True}
        try:
            # Configure pdfkit with the wkhtmltopdf binary path
            config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

            # Convert HTML to PDF
            pdfkit.from_file(
                html_file, output_pdf, configuration=config, options=options
            )

            logger.info(f"PDF generated successfully: {output_pdf}")
        except Exception as e:
            logger.error(
                f"Failed to convert HTML to PDF. HTML file: {html_file}, Output PDF: {output_pdf}, Error: {e}"
            )
            raise RuntimeError(f"Failed to convert HTML to PDF.") from e
