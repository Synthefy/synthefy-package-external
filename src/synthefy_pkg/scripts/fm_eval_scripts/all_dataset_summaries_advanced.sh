#!/bin/bash

# Advanced script to run summarize.py on all supported datasets
# Features: parallel execution, better error handling, progress tracking, resume capability

set -e  # Exit on any error

# Configuration
SCRIPT_PATH="src/synthefy_pkg/fm_evals/summarize.py"
OUTPUT_DIR="/workspace/fm_eval/summary_results/"
MAX_SAMPLES=40
LOG_FILE="dataset_summary_results.log"
TIMING_FILE="dataset_timing_results.csv"
PROGRESS_FILE="dataset_progress.txt"
MAX_PARALLEL_JOBS=32  # Adjust based on your system
TIMEOUT_MINUTES=12000

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Initialize files
echo "Starting dataset summary run at $(date)" > "$LOG_FILE"
echo "Dataset,Status,Duration_Seconds,Error_Message,Start_Time,End_Time" > "$TIMING_FILE"
echo "0" > "$PROGRESS_FILE"  # Initialize progress counter

# List of all supported datasets
DATASETS=(
      # "fmv3"
      # "gift"
      # "synthetic_medium_lag"
      # "co2_monitor"
      # "news_sentiment"
      # "taiwan_aq"
      # "complex_seasonal_timeseries"
      # "external"
      # "external"
      "traffic"
      "solar_alabama"
      "weather_mpi"
      "goodrx"
      "spain_energy"
      "gpt-synthetic"
      "beijing_embassy"
      "ercot_load"
      "open_aq"
      "beijing_aq"
      "cgm"
      "mn_interstate"
      "blow_molding"
      "tac"
      "gas_sensor"
      "tetuan_power"
      "paris_mobility"
      "aus_electricity"
      "cursor_tabs"
      "walmart_sales"
      "complex_seasonal_timeseries"
      "mta_ridership"
      "pasta_sales"
      "austin_water"
      "ny_electricity2025"
      "fl_electricity"
      "tn_electricity"
      "pa_electricity"
      "car_electricity"
      "cal_electricity"
      "tx_electricity"
      "se_electricity"
      "ne_electricity"
      "az_electricity"
      "id_electricity"
      "or_electricity"
      "central_electricity"
      "eastern_electricity"
      "western_electricity"
      "southern_electricity"
      "northern_electricity"
      "tx_daily"
      "ne_daily"
      "ny_daily"
      "az_daily"
      "cal_daily"
      "nm_daily"
      "pa_daily"
      "tn_daily"
      "co_daily"
      "car_daily"
      "al_daily"
      "rideshare_uber"
      "rideshare_lyft"
      "causal_rivers"
      "bitcoin_price"
      "oikolab_weather"
      "blue_bikes"
      "web_visitors"
      "fred_md1"
      "fred_md2"
      "fred_md3"
      "fred_md4"
      "fred_md5"
      "fred_md6"
      "fred_md7"
      "fred_md8"
      "ecl"
      "rice_prices"
      "gold_prices"
      "sleep_lab"
      "mds_microgrid"
      "voip"
      "ev_sensors"
      "mujoco_halfcheetah_v2"
      "mujoco_ant_v2"
      "mujoco_hopper_v2"
      "mujoco_walker2d_v2"
      "cifar100"
      "openwebtext"
      "spriteworld"
      "SCM_tiny_obslag_synin_ns"
      "SCM_tiny_convlag_synin_ns"
      "SCM_medium_obslag_synin_s"
      "SCM_medium_convlag_synin_s"
      "SCM_large_convlag_synin_s"
      "dynamic_data"
      "stock_nasdaqtrader"
      "kitti"
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

# Function to check if dataset is already completed
is_dataset_completed() {
    local dataset="$1"
    grep -q "^$dataset,SUCCESS," "$TIMING_FILE" 2>/dev/null
}

# Function to run summarize for a single dataset (unified for regular and gpt-synthetic)
run_dataset_summary() {
    local dataset="$1"
    local dataset_name="$2"  # Optional: for gpt-synthetic datasets
    local start_time=$(date +%s)
    local start_time_str=$(date)
    local status="SUCCESS"
    local error_msg=""
    
    # Determine if this is a gpt-synthetic run
    local is_gpt_synthetic=false
    local display_name="$dataset"
    local output_dir_name="$dataset"
    local log_prefix="dataset"
    
    if [ "$dataset" = "gpt-synthetic" ] && [ -n "$dataset_name" ]; then
        is_gpt_synthetic=true
        display_name="gpt-synthetic-$dataset_name"
        output_dir_name="gpt-synthetic-$dataset_name"
        log_prefix="gpt-synthetic dataset"
    fi
    
    echo "[$(date)] Starting summary for $log_prefix: $display_name" | tee -a "$LOG_FILE"
    
    # Create dataset-specific output directory
    local dataset_output_dir="$OUTPUT_DIR/$output_dir_name"
    mkdir -p "$dataset_output_dir"
    
    # Build command arguments
    local cmd_args="--dataset $dataset --output-directory $dataset_output_dir --max-samples $MAX_SAMPLES --random-ordering"
    if [ "$is_gpt_synthetic" = true ]; then
        cmd_args="$cmd_args --dataset-name $dataset_name"
    fi
    
    # Run the command with timeout
    if timeout $((TIMEOUT_MINUTES * 60)) uv run "$SCRIPT_PATH" $cmd_args \
        > "${dataset_output_dir}/summary.log" 2>&1; then
        status="SUCCESS"
        echo "[$(date)] Successfully completed summary for $log_prefix: $display_name" | tee -a "$LOG_FILE"
    else
        local exit_code=$?
        status="FAILED"
        if [ $exit_code -eq 124 ]; then
            error_msg="TIMEOUT (${TIMEOUT_MINUTES} minutes)"
        elif [ $exit_code -eq 130 ]; then
            error_msg="INTERRUPTED"
        else
            error_msg="EXIT_CODE_$exit_code"
        fi
        echo "[$(date)] Failed to complete summary for $log_prefix: $display_name (Exit code: $exit_code)" | tee -a "$LOG_FILE"
    fi
    
    local end_time=$(date +%s)
    local end_time_str=$(date)
    local duration=$((end_time - start_time))
    
    # Record results in CSV
    echo "$display_name,$status,$duration,$error_msg,$start_time_str,$end_time_str" >> "$TIMING_FILE"
    
    # Update progress
    local current_progress=$(cat "$PROGRESS_FILE")
    echo $((current_progress + 1)) > "$PROGRESS_FILE"
    
    echo "[$(date)] Dataset $display_name completed in ${duration}s with status: $status" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# Function to show progress
show_progress() {
    local completed=$(cat "$PROGRESS_FILE")
    local total=$((${#DATASETS[@]} + ${#GPT_SYNTHETIC_DATASET_NAMES[@]}))
    local percentage=$((completed * 100 / total))
    echo "Progress: $completed/$total ($percentage%)"
}

# Function to run datasets in parallel (unified for regular and gpt-synthetic)
run_parallel_summaries() {
    local dataset_type="$1"  # "regular" or "gpt-synthetic"
    local -a remaining_datasets=()
    local -a dataset_list=()
    local batch_prefix=""
    
    if [ "$dataset_type" = "regular" ]; then
        dataset_list=("${DATASETS[@]}")
        batch_prefix=""
        
        # Filter out already completed datasets
        for dataset in "${dataset_list[@]}"; do
            if ! is_dataset_completed "$dataset"; then
                remaining_datasets+=("$dataset")
            fi
        done
    elif [ "$dataset_type" = "gpt-synthetic" ]; then
        dataset_list=("${GPT_SYNTHETIC_DATASET_NAMES[@]}")
        batch_prefix="gpt-synthetic "
        
        # Filter out already completed gpt-synthetic datasets
        for dataset_name in "${dataset_list[@]}"; do
            if ! is_dataset_completed "gpt-synthetic-$dataset_name"; then
                remaining_datasets+=("$dataset_name")
            fi
        done
    else
        echo "Error: Invalid dataset_type '$dataset_type'. Must be 'regular' or 'gpt-synthetic'"
        return 1
    fi
    
    echo "Running ${batch_prefix}summaries for ${#remaining_datasets[@]} remaining datasets..."
    echo "Using up to $MAX_PARALLEL_JOBS parallel jobs"
    
    # Process datasets in parallel batches
    for ((i=0; i<${#remaining_datasets[@]}; i+=MAX_PARALLEL_JOBS)); do
        local -a batch=()
        
        # Create batch of jobs
        for ((j=i; j<i+MAX_PARALLEL_JOBS && j<${#remaining_datasets[@]}; j++)); do
            batch+=("${remaining_datasets[j]}")
        done
        
        echo "Processing ${batch_prefix}batch: ${batch[*]}"
        
        # Run batch in parallel
        for dataset_item in "${batch[@]}"; do
            if [ "$dataset_type" = "regular" ]; then
                run_dataset_summary "$dataset_item" &
            else
                run_dataset_summary "gpt-synthetic" "$dataset_item" &
            fi
        done
        
        # Wait for all jobs in this batch to complete
        wait
        
        show_progress
    done
}

# Function to generate detailed summary report
generate_summary_report() {
    echo "Generating detailed summary report..." | tee -a "$LOG_FILE"
    
    local total_datasets=$((${#DATASETS[@]} + ${#GPT_SYNTHETIC_DATASET_NAMES[@]}))
    local successful=$(grep -c "SUCCESS" "$TIMING_FILE" || echo "0")
    local failed=$(grep -c "FAILED" "$TIMING_FILE" || echo "0")
    local total_time=$(awk -F',' 'NR>1 {sum+=$3} END {print sum+0}' "$TIMING_FILE")
    
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "DETAILED SUMMARY REPORT" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Total datasets: $total_datasets (${#DATASETS[@]} regular + ${#GPT_SYNTHETIC_DATASET_NAMES[@]} gpt-synthetic variants)" | tee -a "$LOG_FILE"
    echo "Successful: $successful" | tee -a "$LOG_FILE"
    echo "Failed: $failed" | tee -a "$LOG_FILE"
    echo "Total execution time: ${total_time}s ($(($total_time / 60))m $(($total_time % 60))s)" | tee -a "$LOG_FILE"
    if [ "$total_datasets" -gt 0 ]; then
        echo "Average time per dataset: $((total_time / total_datasets))s" | tee -a "$LOG_FILE"
    fi
    echo "" | tee -a "$LOG_FILE"
    
    if [ "$failed" -gt 0 ]; then
        echo "FAILED DATASETS:" | tee -a "$LOG_FILE"
        grep "FAILED" "$TIMING_FILE" | while IFS=',' read -r dataset status duration error_msg start_time end_time; do
            echo "  - $dataset: $error_msg (${duration}s)" | tee -a "$LOG_FILE"
        done
        echo "" | tee -a "$LOG_FILE"
    fi
    
    echo "SUCCESSFUL DATASETS:" | tee -a "$LOG_FILE"
    grep "SUCCESS" "$TIMING_FILE" | while IFS=',' read -r dataset status duration error_msg start_time end_time; do
        echo "  - $dataset: ${duration}s" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
    
    echo "Files created:" | tee -a "$LOG_FILE"
    echo "  - Detailed results: $TIMING_FILE" | tee -a "$LOG_FILE"
    echo "  - Full log: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "  - Dataset outputs: $OUTPUT_DIR/*/" | tee -a "$LOG_FILE"
}

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    # Kill any remaining background jobs
    jobs -p | xargs -r kill
    echo "Cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
echo "Advanced batch summary run for ${#DATASETS[@]} datasets + ${#GPT_SYNTHETIC_DATASET_NAMES[@]} gpt-synthetic variants"
echo "Output directory: $OUTPUT_DIR"
echo "Max samples per dataset: $MAX_SAMPLES"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Timeout per dataset: ${TIMEOUT_MINUTES} minutes"
echo "Log file: $LOG_FILE"
echo "Timing file: $TIMING_FILE"
echo ""

# Check for resume capability
completed_count=0
for dataset in "${DATASETS[@]}"; do
    if is_dataset_completed "$dataset"; then
        ((completed_count++))
    fi
done

gpt_completed_count=0
for dataset_name in "${GPT_SYNTHETIC_DATASET_NAMES[@]}"; do
    if is_dataset_completed "gpt-synthetic-$dataset_name"; then
        ((gpt_completed_count++))
    fi
done

if [ "$completed_count" -gt 0 ] || [ "$gpt_completed_count" -gt 0 ]; then
    echo "Found $completed_count already completed datasets and $gpt_completed_count completed gpt-synthetic variants. Resuming..."
fi

# # Run gpt-synthetic dataset summaries
# echo "========================================"
# echo "RUNNING GPT-SYNTHETIC DATASET SUMMARIES"
# echo "========================================"
# run_parallel_summaries "gpt-synthetic"

# # Run regular dataset summaries
echo "========================================"
echo "RUNNING REGULAR DATASET SUMMARIES"
echo "========================================"
run_parallel_summaries "regular"

# Generate final summary report
generate_summary_report

echo "Advanced batch summary run completed at $(date)"


# aws s3 sync /workspace_1/fm_eval/summary_results/ny_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/ne_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/or_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/tx_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/dynamic_data/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/SCM_large_convlag_synin_s/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/se_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/az_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/pa_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/SCM_medium_obslag_synin_s/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/traffic/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/mta_ridership/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/SCM_medium_convlag_synin_s/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/tx_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/stock_nasdaqtrader/ s3://synthefy-fm-eval-datasets/analysis/

# aws s3 sync /workspace_1/fm_eval/summary_results/cal_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/nm_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/az_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/id_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/ne_daily/ s3://synthefy-fm-eval-datasets/analysis/

# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md6/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/id_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/or_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/sleep_lab/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md7/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md8/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/ne_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/voip/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/eastern_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/ev_sensors/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/southern_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/az_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/az_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/ny_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/mds_microgrid/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/central_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/rice_prices/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/cifar100/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/cal_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/nm_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/openwebtext/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/ne_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/western_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/gold_prices/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md4/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/pa_daily/ s3://synthefy-fm-eval-datasets/analysis/

# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md5/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/rideshare_lyft/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/tn_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/rideshare_uber/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/causal_rivers/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md3/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/web_visitors/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/car_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md2/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/blue_bikes/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/bitcoin_price/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/co_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/fred_md1/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/oikolab_weather/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace_1/fm_eval/summary_results/al_daily/ s3://synthefy-fm-eval-datasets/analysis/

# aws s3 sync /workspace/fm_eval/summary_results/car_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/gas_sensor/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/mujoco_ant_v2/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/mujoco_halfcheetah_v2/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/ny_electricity2025/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/northern_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/walmart_sales/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/aus_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/cal_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/paris_mobility/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/blow_molding/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/ercot_load/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/tac/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/gpt-synthetic-supply_chain/gpt-synthetic/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/gpt-synthetic-energy/gpt-synthetic/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/pasta_sales/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/mujoco_walker2d_v2/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/gpt-synthetic-manufacturing/gpt-synthetic/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/beijing_embassy/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/spain_energy/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/mujoco_hopper_v2/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/goodrx/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/weather_mpi/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/cgm/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/austin_water/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/tetuan_power/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/gpt-synthetic-traffic/gpt-synthetic/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/gpt-synthetic-retail/gpt-synthetic/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/mn_interstate/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/open_aq/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/pa_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/tn_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/cursor_tabs/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/beijing_aq/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/fl_electricity/ s3://synthefy-fm-eval-datasets/analysis/

# aws s3 sync /workspace/fm_eval/summary_results/al_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/ne_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/co_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/fred_md1/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/tn_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/az_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/cal_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/id_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/fred_md3/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/web_visitors/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/eastern_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/central_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/western_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/car_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/southern_electricity/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/nm_daily/ s3://synthefy-fm-eval-datasets/analysis/
# aws s3 sync /workspace/fm_eval/summary_results/pa_daily/ s3://synthefy-fm-eval-datasets/analysis/