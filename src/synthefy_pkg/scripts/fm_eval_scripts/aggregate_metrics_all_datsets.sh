#!/bin/bash

# Script to run download_and_aggregate.py on each dataset separately
# This allows for better error handling and progress tracking
# 
# Behavior: First dataset overwrites output files, subsequent datasets append to them

set -e  # Exit on any error (remove this line if you want to continue on errors)

# Default values
OUTPUT_DIR="/workspace/fm_eval"
KEEP_FILES=""
CONTINUE_ON_ERROR=false
S3_BASE_PATH="s3://synthefy-fm-eval-datasets/updated_jsons/"

# unsupported datasets
# "fmv3" "gift" "synthetic_medium_lag"

# All supported datasets (ordered as in eval.py, excluding gpt-synthetic which is handled separately)
DATASETS=(
    "traffic" "solar_alabama" "weather_mpi" "goodrx" "spain_energy"
    "beijing_embassy" "ercot_load" "open_aq" "beijing_aq" "cgm"
    "mn_interstate" "blow_molding" "tac" "gas_sensor" "tetuan_power"
    "paris_mobility" "aus_electricity" "cursor_tabs" "walmart_sales"
    "mta_ridership" "pasta_sales"
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

# List of specific dataset names for gpt-synthetic
GPT_SYNTHETIC_DATASET_NAMES=(
    "energy"
    "manufacturing"
    "retail"
    "supply_chain"
    "traffic"
)

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [DATASETS...]"
    echo ""
    echo "OPTIONS:"
    echo "  --output-dir DIR     Output directory for results (default: fm_evals)"
    echo "  --keep-files         Keep downloaded files after processing"
    echo "  --continue-on-error  Continue processing other datasets if one fails"
    echo "  --help               Show this help message"
    echo ""
    echo "DATASETS:"
    echo "  If no datasets are specified, all datasets will be processed."
    echo "  Specify individual datasets as space-separated arguments."
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Process all datasets"
    echo "  $0 fmv3 traffic solar_alabama         # Process specific datasets"
    echo "  $0 --output-dir results --keep-files  # Custom options"
    echo "  $0 --continue-on-error                # Continue on errors"
}

# Function to process a single dataset (unified for regular and gpt-synthetic)
process_dataset() {
    local dataset="$1"
    local dataset_name="$2"  # Optional: for gpt-synthetic datasets
    local output_dir="$3"
    local keep_files="$4"
    local append_results="$5"  # Optional: whether to append results
    
    # Determine if this is a gpt-synthetic run
    local is_gpt_synthetic=false
    local display_name="$dataset"
    
    if [ "$dataset" = "gpt-synthetic" ] && [ -n "$dataset_name" ]; then
        is_gpt_synthetic=true
        display_name="gpt-synthetic-$dataset_name"
    fi
    
    echo ""
    echo "============================================================"
    echo "Processing dataset: $display_name"
    echo "============================================================"
    
    # Build command
    local cmd="uv run src/synthefy_pkg/fm_evals/download_and_aggregate.py --datasets $dataset --output-dir $output_dir --s3-base-path $S3_BASE_PATH"
    
    if [ "$is_gpt_synthetic" = true ]; then
        cmd="$cmd --dataset-name $dataset_name"
    fi
    
    if [ -n "$keep_files" ]; then
        cmd="$cmd --keep-files"
    fi
    
    if [ -n "$append_results" ]; then
        cmd="$cmd --append-results"
    fi
    
    echo "Command: $cmd"
    echo ""
    
    # Run the command
    if eval "$cmd"; then
        echo "✅ Successfully processed $display_name"
        return 0
    else
        echo "❌ Failed to process $display_name"
        echo "Failed command: $cmd"
        return 1
    fi
}

# Parse command line arguments
SELECTED_DATASETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --keep-files)
            KEEP_FILES="--keep-files"
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            set +e  # Don't exit on errors
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            SELECTED_DATASETS+=("$1")
            shift
            ;;
    esac
done

# Determine which datasets to process
if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    # Process all datasets (regular + gpt-synthetic)
    ALL_DATASETS=("${DATASETS[@]}")
    for dataset_name in "${GPT_SYNTHETIC_DATASET_NAMES[@]}"; do
        ALL_DATASETS+=("gpt-synthetic-$dataset_name")
    done
    DATASETS_TO_PROCESS=("${ALL_DATASETS[@]}")
else
    DATASETS_TO_PROCESS=("${SELECTED_DATASETS[@]}")
fi

# Show what we're about to process
echo "Processing ${#DATASETS_TO_PROCESS[@]} datasets:"
for i in "${!DATASETS_TO_PROCESS[@]}"; do
    printf "  %2d. %s\n" $((i+1)) "${DATASETS_TO_PROCESS[$i]}"
done

echo ""
echo "Output directory: $OUTPUT_DIR"
if [ -n "$KEEP_FILES" ]; then
    echo "Keep files: Yes"
else
    echo "Keep files: No"
fi
if [ "$CONTINUE_ON_ERROR" = true ]; then
    echo "Continue on error: Yes"
else
    echo "Continue on error: No"
fi

# Process each dataset
successful=()
failed=()

for i in "${!DATASETS_TO_PROCESS[@]}"; do
    dataset="${DATASETS_TO_PROCESS[$i]}"
    echo ""
    echo "[$((i+1))/${#DATASETS_TO_PROCESS[@]}] Processing $dataset..."
    
    # Determine append behavior: overwrite for first dataset, append for rest
    append_flag=""
    if [ $i -gt 0 ]; then
        append_flag="--append-results"
    else
        append_flag=""
    fi
    
    # Determine if this is a gpt-synthetic dataset
    if [[ "$dataset" == gpt-synthetic-* ]]; then
        dataset_name="${dataset#gpt-synthetic-}"
        if process_dataset "gpt-synthetic" "$dataset_name" "$OUTPUT_DIR" "$KEEP_FILES" "$append_flag"; then
            successful+=("$dataset")
        else
            failed+=("$dataset")
            if [ "$CONTINUE_ON_ERROR" = false ]; then
                echo ""
                echo "Stopping due to error processing $dataset"
                echo "Use --continue-on-error to process remaining datasets"
                break
            fi
        fi
    else
        if process_dataset "$dataset" "" "$OUTPUT_DIR" "$KEEP_FILES" "$append_flag"; then
            successful+=("$dataset")
        else
            failed+=("$dataset")
            if [ "$CONTINUE_ON_ERROR" = false ]; then
                echo ""
                echo "Stopping due to error processing $dataset"
                echo "Use --continue-on-error to process remaining datasets"
                break
            fi
        fi
    fi
done

# Print summary
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Total datasets: ${#DATASETS_TO_PROCESS[@]}"
echo "Successful: ${#successful[@]}"
echo "Failed: ${#failed[@]}"

if [ ${#successful[@]} -gt 0 ]; then
    echo ""
    echo "✅ Successful datasets:"
    for dataset in "${successful[@]}"; do
        echo "  - $dataset"
    done
fi

if [ ${#failed[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed datasets:"
    for dataset in "${failed[@]}"; do
        echo "  - $dataset"
    done
    
    if [ "$CONTINUE_ON_ERROR" = true ]; then
        echo ""
        echo "To retry failed datasets, run:"
        echo "$0 --datasets ${failed[*]}"
    fi
fi

# Exit with error code if any datasets failed
if [ ${#failed[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
