import os
from typing import Any, Dict, Optional

import boto3
from fastapi import HTTPException
from loguru import logger

from synthefy_pkg.app.config import SummarizerSettings
from synthefy_pkg.app.data_models import AWSInfo, SummarizeResponse
from synthefy_pkg.app.utils.s3_utils import (
    create_presigned_url,
    upload_file_to_s3,
)
from synthefy_pkg.preprocessing.data_summarizer import DataSummarizer

DATA_SUMMARY_REPORT_DIR = "reports"


class SummarizeService:
    def __init__(self, settings: SummarizerSettings):
        """Initialize SummarizeService with the given settings.

        Args:
            settings: Configuration settings for summarize service
        """
        logger.info("Initializing SummarizeService...")
        self.settings = settings

    # TODO: plot generation is failing now. Need to fix it.
    async def process_file(
        self,
        file_path: str,
        aws_upload_info: AWSInfo,
        tmp_dir: str,
        config_dict: Optional[Dict[str, Any]] = None,
        group_cols: Optional[str] = None,
        skip_plots: bool = False,
    ) -> SummarizeResponse:
        """Process a file and generate summary reports.

        Args:
            file_path: Path to the input file
            aws_upload_info: AWS upload information
            config_dict: Optional configuration dictionary
            group_cols: Optional group columns for summarization
            skip_plots: Whether to skip generating plots
            tmp_dir: Temporary directory for processing

        Returns:
            SummarizeResponse containing summary information and PDF URL
        """
        try:
            # Initialize and run summarizer
            summarizer = DataSummarizer(
                data_input=file_path,
                save_path=tmp_dir,
                config=config_dict,
                group_cols=group_cols.split(",") if group_cols else None,
                skip_plots=skip_plots,
            )

            # Generate report
            html_path = os.path.join(
                tmp_dir, f"{self.settings.dataset_name}.html"
            )

            summarizer.summarize_metadata()
            summarizer.summarize_time_series()
            summarizer.generate_html_report(html_path)

            # Upload PDF to S3 only if bucket_name is not "local"
            if self.settings.bucket_name != "local":
                s3_client = boto3.client("s3")
                s3_key = str(
                    os.path.join(
                        aws_upload_info.user_id,
                        DATA_SUMMARY_REPORT_DIR,
                        self.settings.dataset_name,
                        f"{self.settings.dataset_name}.html",
                    )
                )

                if not upload_file_to_s3(
                    s3_client, html_path, self.settings.bucket_name, s3_key
                ):
                    raise HTTPException(
                        status_code=500, detail="Failed to upload HTML to S3"
                    )

                # Generate presigned URL
                presigned_url = create_presigned_url(
                    s3_client, self.settings.bucket_name, s3_key
                )
            else:
                # For local testing, just use the local HTML path as the URL
                presigned_url = f"file://{html_path}"

            # Get summary data and add URL
            summary_dict = summarizer.get_summary_dict()

            # Create and return SummarizeResponse object
            return SummarizeResponse(
                time_series_summary=summary_dict.get("time_series_summary"),
                time_range=summary_dict.get("time_range"),
                metadata_summary=summary_dict.get("metadata_summary"),
                columns_by_type=summary_dict.get("columns_by_type"),
                sample_counts=summary_dict.get("sample_counts"),
                html_url=presigned_url,
            )

        except Exception as e:
            logger.error(f"Error in process_file: {str(e)}")
            logger.error("Error traceback:", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to process data: {str(e)}"
            )
