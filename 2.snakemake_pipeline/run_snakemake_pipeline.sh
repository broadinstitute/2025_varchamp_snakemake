#!/bin/bash

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
            sleep 600  # Normal 10-minute interval
        fi
    done
    
    echo "Memory monitoring finished for $batch_name at $(date)" >> "$log_file"
}

# Function to run batch with memory monitoring
run_batch_with_monitoring() {
    local batch_num=$1
    local snakefile="Snakefile_batch${batch_num}"
    local log_file="outputs/snakemake_logs/snakemake_batch${batch_num}_new.log"
    
    echo "Starting batch $batch_num at $(date)"
    
    # Copy snakefile
    cp "inputs/snakemake_files/$snakefile" .
    
    # Start snakemake in background
    nohup snakemake \
        --snakefile "$snakefile" \
        --rerun-incomplete \
        --cores 256 > "$log_file" 2>&1 &
    
    local snake_pid=$!
    echo "Snakemake PID: $snake_pid for batch $batch_num"
    
    # Start memory monitoring in background
    monitor_memory "$snake_pid" "batch${batch_num}" &
    local monitor_pid=$!
    
    # Wait for snakemake to complete
    wait "$snake_pid"
    local snake_exit_code=$?
    
    # Stop memory monitoring
    kill "$monitor_pid" 2>/dev/null || true
    
    # Cleanup
    rm "$snakefile"
    
    if [ $snake_exit_code -eq 0 ]; then
        echo "Batch $batch_num completed successfully at $(date)"
    else
        echo "Batch $batch_num failed with exit code $snake_exit_code at $(date)"
    fi
    
    return $snake_exit_code
}

# cp inputs/snakemake_files/Snakefile_batch17 .
# nohup snakemake \
#     --snakefile Snakefile_batch17 \
#     --cores all &> outputs/snakemake_logs/snakemake_batch17.log

# cp inputs/snakemake_files/Snakefile_batch17_mc .
# nohup snakemake \
#     --snakefile Snakefile_batch17_mc \
#     --cores all &> outputs/snakemake_logs/snakemake_batch17_mc.log

# # Run batch 11
# cp inputs/snakemake_files/Snakefile_batch11 .
# snakemake \
#     --snakefile Snakefile_batch11 \
#     --cores all &> outputs/snakemake_logs/snakemake_batch11_new.log
# rm Snakefile_batch11

# # Run batch 12
# cp inputs/snakemake_files/Snakefile_batch12 .
# snakemake \
#     --snakefile Snakefile_batch12 \
#     --cores all &> outputs/snakemake_logs/snakemake_batch12_new.log
# rm Snakefile_batch12


# Run batches with memory monitoring
echo "Starting pipeline with memory monitoring at $(date)"

# # Run batch 13
# run_batch_with_monitoring 13

# # Run batch 14  
# run_batch_with_monitoring 14

# # Run batch 15
# run_batch_with_monitoring 15

# # Run batch 16
# run_batch_with_monitoring 16

# # Run batch 18
# run_batch_with_monitoring 18

# # Run batch 19
# run_batch_with_monitoring 19

# # Run batch 7
# run_batch_with_monitoring 7

# Run batch 8
run_batch_with_monitoring 8

echo "All batches completed at $(date)"