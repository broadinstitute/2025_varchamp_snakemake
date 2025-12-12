#!/bin/bash

# Unified Snakemake Pipeline Runner
# Supports both parallel and sequential batch execution
# Usage:
#   ./run_snakemake_pipeline.sh --parallel 18 19      # Run batches in parallel
#   ./run_snakemake_pipeline.sh --sequential 15 16 11 # Run batches sequentially
#   ./run_snakemake_pipeline.sh 13 14                 # Default: sequential

# Activate conda environment
source ~/software/anaconda3/etc/profile.d/conda.sh
conda activate varchamp

# Memory monitoring function
monitor_memory() {
    local snake_pid=$1
    local batch_name=$2
    local log_file="outputs/snakemake_logs/memory_${batch_name}.log"

    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$log_file")"

    echo "Memory monitoring started for $batch_name at $(date)" > "$log_file"

    while kill -0 "$snake_pid" 2>/dev/null; do
        local timestamp=$(date '+%H:%M:%S')
        local mem_info=$(free -h | grep Mem | awk '{print $3"/"$2}')
        local swap_info=$(free -h | grep Swap | awk '{print $3"/"$2}')
        local swap_pct=$(free | grep Swap | awk '{if($2>0) print int($3/$2*100); else print 0}')

        local message="$timestamp: RAM: $mem_info SWAP: $swap_info"

        if [ "$swap_pct" -gt 20 ]; then
            message="$message ⚠️  SWAP: $swap_pct%"
            echo "$message" | tee -a "$log_file"
            sleep 180  # Monitor more frequently when swap is high
        else
            echo "$message" >> "$log_file"
            sleep 900  # Normal 15-minute interval
        fi
    done

    echo "Memory monitoring finished for $batch_name at $(date)" >> "$log_file"
}

# Function to run batch with memory monitoring
run_batch_with_monitoring() {
    mkdir -p "outputs/snakemake_logs/"

    local batch_num=$1
    local snakefile="inputs/snakemake_files/Snakefile_batch${batch_num}"
    local log_file="outputs/snakemake_logs/snakemake_batch${batch_num}.log"

    echo "Starting batch $batch_num at $(date)"

    # Run snakemake directly from inputs/snakemake_files (no copy needed)
    nohup snakemake \
        --snakefile "$snakefile" \
        --rerun-incomplete \
        --cores 256 > "$log_file" 2>&1 &

    local snake_pid=$!
    echo "Snakemake PID: $snake_pid for batch $batch_num"

    # Start memory monitoring in background
    monitor_memory "$snake_pid" "batch${batch_num}" &
    local monitor_pid=$!

    # Return both PIDs for caller to manage
    echo "$snake_pid:$monitor_pid"
}

# Function to wait for batch and report status
wait_for_batch() {
    local batch_num=$1
    local snake_pid=$2
    local monitor_pid=$3

    # Wait for snakemake to complete
    wait "$snake_pid"
    local snake_exit_code=$?

    # Stop memory monitoring
    kill "$monitor_pid" 2>/dev/null || true

    if [ $snake_exit_code -eq 0 ]; then
        echo "Batch $batch_num completed successfully at $(date)"
    else
        echo "Batch $batch_num failed with exit code $snake_exit_code at $(date)"
    fi

    return $snake_exit_code
}

# Parse command line arguments
MODE="sequential"  # Default mode
BATCHES=()

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--parallel|--sequential] batch1 [batch2 ...]"
    echo "Examples:"
    echo "  $0 --parallel 18 19       # Run batches 18 and 19 in parallel"
    echo "  $0 --sequential 15 16 11  # Run batches sequentially"
    echo "  $0 13 14                  # Run batches 13 and 14 sequentially (default)"
    exit 1
fi

# Parse mode flag if present
if [[ "$1" == "--parallel" ]]; then
    MODE="parallel"
    shift
elif [[ "$1" == "--sequential" ]]; then
    MODE="sequential"
    shift
fi

# Collect batch numbers
BATCHES=("$@")

# Validate batch numbers
for batch in "${BATCHES[@]}"; do
    if ! [[ "$batch" =~ ^[0-9]+$ ]]; then
        echo "Error: Invalid batch number '$batch'. Must be a number."
        exit 1
    fi

    snakefile="inputs/snakemake_files/Snakefile_batch${batch}"
    if [ ! -f "$snakefile" ]; then
        echo "Error: Snakefile not found: $snakefile"
        exit 1
    fi
done

echo "==========================================="
echo "VarChAMP Snakemake Pipeline Runner"
echo "==========================================="
echo "Mode: $MODE"
echo "Batches: ${BATCHES[@]}"
echo "Start time: $(date)"
echo ""

if [ "$MODE" == "parallel" ]; then
    # Parallel execution mode
    echo "Running batches in parallel..."
    echo ""

    declare -A BATCH_PIDS
    declare -A MONITOR_PIDS

    # Start all batches
    for batch in "${BATCHES[@]}"; do
        pids=$(run_batch_with_monitoring "$batch")
        snake_pid=$(echo "$pids" | cut -d: -f1)
        monitor_pid=$(echo "$pids" | cut -d: -f2)

        BATCH_PIDS[$batch]=$snake_pid
        MONITOR_PIDS[$batch]=$monitor_pid
        echo ""
    done

    echo "All batches started. Waiting for completion..."
    echo ""

    # Wait for all batches to complete
    FAILED_BATCHES=()
    for batch in "${BATCHES[@]}"; do
        snake_pid=${BATCH_PIDS[$batch]}
        monitor_pid=${MONITOR_PIDS[$batch]}

        if ! wait_for_batch "$batch" "$snake_pid" "$monitor_pid"; then
            FAILED_BATCHES+=($batch)
        fi
        echo ""
    done

    # Report final status
    echo "==========================================="
    if [ ${#FAILED_BATCHES[@]} -eq 0 ]; then
        echo "All batches completed successfully at $(date)"
        echo "==========================================="
        exit 0
    else
        echo "Some batches failed at $(date)"
        echo "Failed batches: ${FAILED_BATCHES[@]}"
        echo "==========================================="
        echo "Check logs in outputs/snakemake_logs/ for details"
        exit 1
    fi

else
    # Sequential execution mode
    echo "Running batches sequentially..."
    echo ""

    for batch in "${BATCHES[@]}"; do
        echo "==========================================="
        echo "Running Batch $batch"
        echo "==========================================="

        pids=$(run_batch_with_monitoring "$batch")
        snake_pid=$(echo "$pids" | cut -d: -f1)
        monitor_pid=$(echo "$pids" | cut -d: -f2)

        if ! wait_for_batch "$batch" "$snake_pid" "$monitor_pid"; then
            echo ""
            echo "ERROR: Batch $batch failed"
            echo "Stopping pipeline. Check log: outputs/snakemake_logs/snakemake_batch${batch}.log"
            exit 1
        fi

        echo ""
    done

    echo "==========================================="
    echo "All batches completed successfully at $(date)"
    echo "==========================================="
    echo "Completed batches: ${BATCHES[@]}"
fi