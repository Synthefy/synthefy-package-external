#!/bin/bash

# Script to sync all summary results to S3
# Usage: ./sync_summary_results_to_s3.sh [--dry-run]

set -e  # Exit on any error

# Parse command line arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be synced without actually syncing"
            echo "  --help, -h   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Sync all results to S3"
            echo "  $0 --dry-run         # Show what would be synced"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "SYNCING SUMMARY RESULTS TO S3"
echo "========================================"

# Define the base paths
WORKSPACE_BASE="/workspace/fm_eval/summary_results"
S3_BUCKET="s3://synthefy-fm-eval-datasets/analysis"

# List of all dataset directories to sync
DATASETS=(
    "traffic" "solar_alabama" "weather_mpi" "goodrx" "spain_energy" "gpt-synthetic"
    "beijing_embassy" "ercot_load" "open_aq" "beijing_aq" "cgm"
    "mn_interstate" "blow_molding" "tac" "gas_sensor" "tetuan_power"
    "paris_mobility" "aus_electricity" "cursor_tabs" "walmart_sales"
    "complex_seasonal_timeseries" "mta_ridership" "pasta_sales"
    "austin_water" "ny_electricity2025" "fl_electricity" "tn_electricity"
    "pa_electricity" "car_electricity" "cal_electricity" "tx_electricity"
    "se_electricity" "ne_electricity" "az_electricity" "id_electricity"
    "or_electricity" "central_electricity" "eastern_electricity"
    "western_electricity" "southern_electricity" "northern_electricity"
    "tx_daily" "ne_daily" "ny_daily" "az_daily" "cal_daily" "nm_daily"
    "pa_daily" "tn_daily" "co_daily" "car_daily" "al_daily"
    "rideshare_uber" "rideshare_lyft" "causal_rivers" "bitcoin_price"
    "oikolab_weather" "blue_bikes" "web_visitors" "fred_md1" "fred_md2"
    "fred_md3" "fred_md4" "fred_md5" "fred_md6" "fred_md7" "fred_md8"
    "ecl" "rice_prices" "gold_prices" "sleep_lab" "mds_microgrid"
    "voip" "ev_sensors" "mujoco_halfcheetah_v2" "mujoco_ant_v2"
    "mujoco_hopper_v2" "mujoco_walker2d_v2" "cifar100" "openwebtext"
    "spriteworld" "SCM_tiny_obslag_synin_ns" "SCM_tiny_convlag_synin_ns"
    "SCM_medium_obslag_synin_s" "SCM_medium_convlag_synin_s"
    "SCM_large_convlag_synin_s" "dynamic_data" "stock_nasdaqtrader" "kitti"
    "wikipedia"
)

# Counters for reporting
total_datasets=${#DATASETS[@]}
synced_count=0
skipped_count=0
error_count=0

echo "Found $total_datasets datasets to process"
echo "Workspace base: $WORKSPACE_BASE"
echo "S3 bucket: $S3_BUCKET"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE: No actual syncing will be performed"
fi
echo ""

# Sync each dataset
for dataset in "${DATASETS[@]}"; do
    local_path="${WORKSPACE_BASE}/${dataset}/"
    s3_path="${S3_BUCKET}/${dataset}/"
    
    echo "Processing ${dataset}..."
    
    if [ -d "$local_path" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] Would sync: $local_path -> $s3_path"
            synced_count=$((synced_count + 1))
        else
            if aws s3 sync "$local_path" "$s3_path"; then
                echo "  ✓ Successfully synced ${dataset}"
                synced_count=$((synced_count + 1))
            else
                echo "  ✗ Failed to sync ${dataset}"
                error_count=$((error_count + 1))
            fi
        fi
    else
        echo "  ⚠ Warning: Local directory ${local_path} does not exist, skipping ${dataset}"
        skipped_count=$((skipped_count + 1))
    fi
done

echo ""
echo "========================================"
echo "SYNC SUMMARY"
echo "========================================"
echo "Total datasets: $total_datasets"
echo "Successfully synced: $synced_count"
echo "Skipped (missing): $skipped_count"
echo "Errors: $error_count"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a dry run. Use without --dry-run to perform actual syncing."
elif [ "$error_count" -eq 0 ]; then
    echo ""
    echo "✓ All available datasets synced successfully!"
else
    echo ""
    echo "⚠ Some datasets failed to sync. Check the output above for details."
    exit 1
fi

echo "========================================"
echo "S3 SYNC COMPLETED"
echo "========================================"
