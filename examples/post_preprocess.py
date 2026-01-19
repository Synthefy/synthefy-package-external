"""
Script to analyze and compare the distributions of train/val/test arrays for:
  1) Time-series and continuous metadata columns:
     - Overlaid histograms (Bokeh)
     - Overlaid KDE plots (Bokeh)
     - JSD & EMD distances between Train vs. Val/Test (written to CSV)
  2) Original discrete columns (e.g., wd, station):
     - Overlaid bar plots (Bokeh)
     - JSD distance between Train vs. Val/Test (written to CSV)

Additionally:
  - Covariance matrix between all time-series columns and all continuous columns.
  - Cross-correlation (max value and lag) between all time-series columns and all continuous columns.

All plots are combined into a single interactive HTML file with section headers.

Usage:
  python examples/post_preprocess.py --dataset_dir /path/to/air_quality
                     [--jsd_threshold 0.1 --emd_threshold 0.1]
                     [--use_unscaled_data]
                     [--output_dir /path/to/output]
                     [--pairwise_corr_figures]
                     [--downsample_factor 10]
"""

import argparse

from synthefy_pkg.utils.post_preprocess_utils import post_preprocess_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare distributions across train/val/test splits."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the directory containing the train/val/test .npy files and JSON column files.",
    )
    parser.add_argument(
        "--jsd_threshold",
        type=float,
        default=0.3,
        help="Threshold above which a JSD value is considered problematic.",
    )
    parser.add_argument(
        "--emd_threshold",
        type=float,
        default=0.3,
        help="Threshold above which an EMD value is considered problematic.",
    )
    parser.add_argument(
        "--use_unscaled_data",
        action="store_true",
        help="If passed, we will unscale/inverse-transform the data before analysis.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the HTML and CSV outputs (defaults to dataset_dir).",
    )
    parser.add_argument(
        "--pairwise_corr_figures",
        action="store_true",
        help="If passed, we will save pairwise cross-correlation figures.",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=10,
        help="Factor by which to downsample the data for plotting efficiency.",
    )
    args = parser.parse_args()

    # If --use_unscaled_data is not set, we keep using scaled data
    use_scaled_data = not args.use_unscaled_data

    post_preprocess_analysis(
        dataset_dir=args.dataset_dir,
        jsd_threshold=args.jsd_threshold,
        emd_threshold=args.emd_threshold,
        use_scaled_data=use_scaled_data,
        output_dir=args.output_dir,
        pairwise_corr_figures=args.pairwise_corr_figures,
        downsample_factor=args.downsample_factor,
    )


if __name__ == "__main__":
    main()
