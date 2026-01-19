import os

import pdfkit
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from synthefy_pkg.postprocessing.postprocess import (
    SYNTHEFY_DATASETS_BASE,
    Postprocessor,
)
from synthefy_pkg.postprocessing.utils import (
    check_plot_exists,
    get_image_base64,
)
from synthefy_pkg.scripts.generate_synthetic_dataset_with_baseline import (
    load_config,
)

COMPILE = False


class ReportGenerator:
    def __init__(
        self,
        config_path: str,
        run_name: str | None = None,
        model_name: str | None = None,
        splits: list[str] = ["test"],
    ):
        """
        Initialize the ReportGenerator object.

        :param config_path: Path to the configuration file used for postprocessing.
        :param run_name: Optional; Name of the run to be used in postprocessing. Default is None.
        :param model_name: Optional; Name of the model to be used in postprocessing. Default is None.
        :param splits: List of splits to be used in postprocessing. Default is ["test"].
        """

        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        # Validate the splits parameter
        valid_splits = {"train", "val", "test"}
        invalid_splits = [
            split for split in splits if split not in valid_splits
        ]
        if invalid_splits:
            raise ValueError(
                f"Invalid split types: {invalid_splits}. Valid options are {valid_splits}"
            )

        self.splits = splits
        self.split_mapping = {
            "train": "Training",
            "val": "Validation",
            "test": "Testing",
        }
        self.postprocessor = Postprocessor(
            config_filepath=config_path, splits=self.splits
        )

        if run_name is not None:
            self.postprocessor.saved_data_path = os.path.join(
                str(SYNTHEFY_DATASETS_BASE),
                self.postprocessor.configuration.generation_save_path,
                self.postprocessor.configuration.dataset_config.dataset_name,
                self.postprocessor.configuration.experiment_name,
                run_name,
            )
            self.postprocessor.figsave_path = os.path.join(
                str(self.postprocessor.saved_data_path), "plots/"
            )

        self.model_name = model_name
        if not model_name:
            self.model_name = (
                self.postprocessor.configuration.denoiser_config.denoiser_name
            )

    def __get_learning_plots(self):
        plot_path = os.path.join(
            self.postprocessor.figsave_path,
            "learning_curve.png",
        )

        curve_section = None
        if not os.path.exists(plot_path):
            logger.warning(
                f"No learning curve plots found in {self.postprocessor.figsave_path}"
            )
            return None

        else:
            # Main: Learning Curve Section
            curve_section = {
                "id": "curve_section",
                "title": "Learning Curve",
                "image": {
                    "base64": get_image_base64(plot_path),
                    "title": "learning_curve.png",
                },
            }

        return curve_section

    def __get_violin_plots(self):
        # Main: Violin Plot Section
        violin_section = {
            "id": "violin_section",
            "title": f"Violin Plots: Ground Truth vs {self.postprocessor.label} for Different KPIs",
            "subsections": [],
        }
        for split in self.splits:
            plot_path = os.path.join(
                self.postprocessor.figsave_path,
                f"violin_dist_{split}.png",
            )
            check_plot_exists(plot_path, f"Violin plot not found: {plot_path}")
            subsection = {
                "id": f"violin_{split}",
                "title": f"{self.split_mapping[split]} Data",
                "image": {
                    "base64": get_image_base64(plot_path),
                    "title": f"violin_dist_{split}.png",
                },
            }
            violin_section["subsections"].append(subsection)
        return violin_section

    def __get_line_plots(self):
        # Main: Line Plot Section
        line_section = {
            "id": "line_section",
            "title": f"Line Plots: Ground vs {self.postprocessor.label} over Time",
            "subsections": [],
        }
        for split in self.splits:
            subsection = {
                "id": f"line_section-{split}",
                "title": f"{self.split_mapping[split]} Data",
                "images": [],
            }
            for i in range(
                self.postprocessor.configuration.dataset_config.num_channels
            ):
                title = (
                    f"{split}: `{self.postprocessor.timeseries_col_names[i]}`"
                )
                plot_path = os.path.join(
                    self.postprocessor.figsave_path,
                    self.postprocessor.timeseries_col_names[i],
                    f"comparison_line_{split}_{self.postprocessor.timeseries_col_names[i]}.png",
                )
                check_plot_exists(
                    plot_path, f"Line plot not found: {plot_path}"
                )
                subsection["images"].append(
                    {
                        "base64": get_image_base64(plot_path),
                        "title": title,
                    }
                )
            line_section["subsections"].append(subsection)
        return line_section

    def __get_covariance_matrix_section(self):
        covariance_matrix_section = {
            "id": "covariance_matrix_section",
            "title": "Covariance Matrix: Continous Columns vs Real and Synthetic Timeseries Columns",
            "subsections": [],
        }

        for split in self.splits:
            real_covariance_matrix_path = os.path.join(
                self.postprocessor.figsave_path,
                f"real_covariance_matrix_{split}.png",
            )

            if not os.path.exists(real_covariance_matrix_path):
                logger.warning(
                    f"No real covariance matrix plots found in {self.postprocessor.figsave_path}"
                )

            else:
                subsection = {
                    "id": f"covariance_matrix_{split}",
                    "title": f"{self.split_mapping[split]} Data: Real",
                    "image": {
                        "base64": get_image_base64(real_covariance_matrix_path),
                        "title": f"real_covariance_matrix_{split}.png",
                    },
                }
                covariance_matrix_section["subsections"].append(subsection)

            synthetic_covariance_matrix_path = os.path.join(
                self.postprocessor.figsave_path,
                f"synthetic_covariance_matrix_{split}.png",
            )
            if not os.path.exists(synthetic_covariance_matrix_path):
                logger.warning(
                    f"No synthetic covariance matrix plots found in {self.postprocessor.figsave_path}"
                )

            else:
                subsection = {
                    "id": f"covariance_matrix_{split}",
                    "title": f"{self.split_mapping[split]} Data: {self.postprocessor.label}",
                    "image": {
                        "base64": get_image_base64(
                            synthetic_covariance_matrix_path
                        ),
                        "title": f"synthetic_covariance_matrix_{split}.png",
                    },
                }
                covariance_matrix_section["subsections"].append(subsection)

            difference_covariance_matrix_path = os.path.join(
                self.postprocessor.figsave_path,
                f"difference_covariance_matrix_{split}.png",
            )
            if not os.path.exists(difference_covariance_matrix_path):
                logger.warning(
                    f"No difference covariance matrix plots found in {self.postprocessor.figsave_path}"
                )

            else:
                subsection = {
                    "id": f"covariance_matrix_{split}",
                    "title": f"{self.split_mapping[split]} Data: Difference",
                    "image": {
                        "base64": get_image_base64(
                            difference_covariance_matrix_path
                        ),
                        "title": f"difference_covariance_matrix_{split}.png",
                    },
                }
                covariance_matrix_section["subsections"].append(subsection)

        return covariance_matrix_section

    def __get_html_section(self, filename: str, title: str, id: str):
        html_section = {
            "id": id,
            "subsections": [],
        }
        for split in self.splits:
            html_path = os.path.join(
                self.postprocessor.figsave_path,
                f"{filename}_{split}.html",
            )
            check_plot_exists(
                html_path, f"{filename} file does not exist: {html_path}"
            )
            subsection = {
                "id": f"{id}_{split}",
                "title": f"{title}: {self.split_mapping[split]} Dataset",
                "table": {
                    "content": open(html_path, "r").read(),
                    "title": f"{filename}_{split}.html",
                },
            }
            html_section["subsections"].append(subsection)
        return html_section

    def __get_appendix_section(self):
        toc = [
            {
                "id": "section1",
                "title": "Histogram Density Distribution",
                "subsections": [
                    {
                        "id": f"section1-{dt}",
                        "title": f"{self.split_mapping[dt]} Data",
                    }
                    for dt in self.splits
                ],
            },
            {
                "id": "section2",
                "title": "Histogram Plots: All Metrics",
                "subsections": [
                    {
                        "id": f"section2-{dt}",
                        "title": f"{self.split_mapping[dt]} Data",
                    }
                    for dt in self.splits
                ],
            },
            {
                "id": "section3",
                "title": "Line Plots: Best and Worst Windows",
                "subsections": [
                    {
                        "id": f"section3-{dt}",
                        "title": f"{self.split_mapping[dt]} Data",
                    }
                    for dt in self.splits
                ],
            },
            {
                "id": "section4",
                "title": "Fourier Plots: All Windows",
                "subsections": [
                    {
                        "id": f"section4-{dt}",
                        "title": f"{self.split_mapping[dt]} Data",
                    }
                    for dt in self.splits
                ],
            },
        ]

        sections = []

        # Section 1: Timeseries Histograms
        timeseries_section = {
            "id": "section1",
            "title": f"Histogram Plots: Ground Truth vs {self.postprocessor.label} Density Distribution",
            "description": f"The closer the overlap between the {self.postprocessor.label} and Ground Truth density distributions, the better.",
            "subsections": [],
        }
        for split in self.splits:
            subsection = {
                "id": f"section1-{split}",
                "title": f"{self.split_mapping[split]} Data",
                "images": [],
            }
            for i in range(
                self.postprocessor.configuration.dataset_config.num_channels
            ):
                title = (
                    f"{split}: `{self.postprocessor.timeseries_col_names[i]}`"
                )
                filename_suffix = f"all_data_{self.postprocessor.timeseries_col_names[i]}_{split}"
                plot_path = os.path.join(
                    self.postprocessor.figsave_path,
                    self.postprocessor.timeseries_col_names[i],
                    f"hist_{filename_suffix}.png",
                )
                check_plot_exists(
                    plot_path, f"Histogram plot not found: {plot_path}"
                )
                subsection["images"].append(
                    {
                        "base64": get_image_base64(plot_path),
                        "title": title,
                    }
                )
            timeseries_section["subsections"].append(subsection)
        sections.append(timeseries_section)

        # Section 2: Histograms for All Windows
        histograms_section = {
            "id": "section2",
            "title": "Histogram Plots: All Metrics",
            "description": "The closer the error distribution is concentrated near zero, the better.",
            "subsections": [],
        }
        for split in self.splits:
            subsection = {
                "id": f"section2-{split}",
                "title": f"{self.split_mapping[split]} Data",
                "images": [],
            }
            for var_i in self.postprocessor.metrics_dict[split].keys():
                df = self.postprocessor.metrics_dict[split][str(var_i)]
                for metric_i in df:
                    title = f"{split} {metric_i}: `{str(var_i)}`"
                    filename_suffix = (
                        f"metric_plot_{str(var_i)}_{metric_i}_{split}"
                    )
                    plot_path = os.path.join(
                        self.postprocessor.figsave_path,
                        var_i,
                        f"hist_{filename_suffix}.png",
                    )
                    check_plot_exists(
                        plot_path, f"Histogram plot not found: {plot_path}"
                    )
                    subsection["images"].append(
                        {
                            "base64": get_image_base64(plot_path),
                            "title": title,
                        }
                    )
            histograms_section["subsections"].append(subsection)
        sections.append(histograms_section)

        # Section 3: Line Plots of Best and Worst Windows
        line_plots_section = {
            "id": "section3",
            "title": "Line Plots: Best and Worst Windows",
            "description": f"The closer the {self.postprocessor.label} values match with the Ground Truth, the better. Usually it is very hard to have exact match.",
            "subsections": [],
        }
        for split in self.splits:
            subsection = {
                "id": f"section3-{split}",
                "title": f"{self.split_mapping[split]} Data",
                "images": [],
            }
            for best_worst in ["Best", "Worst"]:
                for i in range(
                    self.postprocessor.configuration.dataset_config.num_channels
                ):
                    for (
                        metric_i
                    ) in self.postprocessor.config.best_worst_metrics:
                        window_ind = self.postprocessor.best_worst_indices[
                            split
                        ][self.postprocessor.timeseries_col_names[i]][metric_i][
                            best_worst
                        ][0]
                        title = f"{metric_i}: {best_worst} {split} sample number ({window_ind}) of var `{self.postprocessor.timeseries_col_names[i]}`"
                        filename_suffix = f"{self.postprocessor.timeseries_col_names[i]}_{best_worst}_{metric_i}_{split}"
                        plot_path = os.path.join(
                            self.postprocessor.figsave_path,
                            self.postprocessor.timeseries_col_names[i],
                            f"line_{filename_suffix}.png",
                        )
                        check_plot_exists(
                            plot_path, f"Line plot not found: {plot_path}"
                        )
                        subsection["images"].append(
                            {
                                "base64": get_image_base64(plot_path),
                                "title": title,
                            }
                        )
                        if self.postprocessor.configuration.denoiser_config.use_probabilistic_forecast:
                            probabilistic_title = f"{metric_i}: {best_worst} {split} sample number ({window_ind}) of var `{self.postprocessor.timeseries_col_names[i]}` (Probabilistic)"
                            plot_path = os.path.join(
                                self.postprocessor.figsave_path,
                                self.postprocessor.timeseries_col_names[i],
                                f"line_{filename_suffix}_probabilistic.png",
                            )
                            subsection["images"].append(
                                {
                                    "base64": get_image_base64(plot_path),
                                    "title": probabilistic_title,
                                }
                            )
            line_plots_section["subsections"].append(subsection)
        sections.append(line_plots_section)

        # Section 4: Fourier Plots for All Windows
        try:
            fourier_section = {
                "id": "section4",
                "title": "Fourier Plots: All Windows",
                "description": f"The closer the overlap between the {self.postprocessor.label} and Ground Truth density distributions, the better.",
                "subsections": [],
            }
            for split in self.splits:
                subsection = {
                    "id": f"section4-{split}",
                    "title": f"{self.split_mapping[split]} Data",
                    "images": [],
                }
                for i in range(
                    self.postprocessor.configuration.dataset_config.num_channels
                ):
                    title = f"{split}: `{self.postprocessor.timeseries_col_names[i]}`"
                    filename_suffix = f"fourier_all_data_{self.postprocessor.timeseries_col_names[i]}_{split}"
                    plot_path = os.path.join(
                        self.postprocessor.figsave_path,
                        self.postprocessor.timeseries_col_names[i],
                        f"hist_{filename_suffix}.png",
                    )
                    check_plot_exists(
                        plot_path, f"Fourier plot not found: {plot_path}"
                    )
                    subsection["images"].append(
                        {
                            "base64": get_image_base64(plot_path),
                            "title": title,
                        }
                    )
                fourier_section["subsections"].append(subsection)
            sections.append(fourier_section)
        except Exception as e:
            logger.error(f"Error generating Fourier plots: {e}")

        return toc, sections

    def generate_html_report(
        self,
        output_html: str = "postprocessing_report.html",
        include_appendix: bool = True,
    ):
        """
        Generate an HTML report with plots generated from the postprocessing step using Jinja2.

        :param output_html: Path to save the HTML report.
        :param include_appendix: Boolean flag to include appendix sections in the report. Default is True.

        :return: HTML content as a string, saved to the specified file path.
        """

        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_html))
        os.makedirs(output_dir, exist_ok=True)

        # Prepare the template environment
        template_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "templates"
        )
        env = Environment(loader=FileSystemLoader(searchpath=template_dir))
        template = env.get_template("postprocess_report.html")

        # Main sections
        violin_section = self.__get_violin_plots()
        line_section = self.__get_line_plots()
        learning_section = self.__get_learning_plots()
        metrics_table_section = self.__get_html_section(
            filename="metrics_table",
            title="Metrics Table",
            id="metrics_table_section",
        )
        covariance_matrix_section = self.__get_covariance_matrix_section()

        # Appendix sections
        toc, sections = [], []
        if include_appendix:
            toc, sections = self.__get_appendix_section()

        # Render the HTML
        html_content = template.render(
            violin_section=violin_section,
            line_section=line_section,
            learning_section=learning_section,
            metrics_table_section=metrics_table_section,
            covariance_matrix_section=covariance_matrix_section,
            model_name=self.model_name,
            dataset_name=self.postprocessor.configuration.dataset_name,
            toc=toc,
            sections=sections,
            include_appendix=include_appendix,
            label=self.postprocessor.label,
        )

        # Save the HTML content to file
        with open(output_html, "w") as file:
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
        options = {
            "enable-local-file-access": True,
            "enable-internal-links": True,
        }
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
            raise RuntimeError("Failed to convert HTML to PDF.") from e
