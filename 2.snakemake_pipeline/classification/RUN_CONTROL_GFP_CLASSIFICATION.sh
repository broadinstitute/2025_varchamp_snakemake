#!/bin/bash
# Run GFP-Filtered Control Classification for All Batches
#
# This script runs GFP-filtered classification specifically on control alleles
# for all batches. Control alleles are used as negative controls to assess
# the specificity of variant classifications.
#
# Output Structure:
# - classification_results/<batch>/<pipeline>/ - predictions, feature importance, classifier info, log
# - classification_analyses/<batch>/<pipeline>/ - metrics only
#
# All outputs use *_control_gfp_adj.* naming convention to distinguish from
# regular variant classifications.

# Activate conda environment
source "$HOME/software/anaconda3/etc/profile.d/conda.sh"
conda activate varchamp

# Navigate to pipeline directory
cd "$(dirname "$0")/.."

# Define batches with single-rep layout
SINGLE_REP_BATCHES=(
    "2024_01_23_Batch_7"
    "2024_02_06_Batch_8"
    "2025_01_27_Batch_13"
    "2025_01_28_Batch_14"
    "2025_03_17_Batch_15"
    "2025_03_17_Batch_16"
)

# Define batches with multi-rep layout
MULTI_REP_BATCHES=(
    "2024_12_09_Batch_11"
    "2024_12_09_Batch_12"
)

# Run single-rep batches
for batch in "${SINGLE_REP_BATCHES[@]}"; do
    batch_num=$(echo "$batch" | grep -o "Batch_[0-9]*" | grep -o "[0-9]*")
    echo "Running Batch $batch_num (single_rep): $batch"
    python classification/classify_gfp_filtered_control_cmd.py \
        --input_path outputs/batch_profiles/$batch/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet \
        --input_path_orig outputs/batch_profiles/$batch/profiles.parquet \
        --output_base_dir outputs/classification_analyses/$batch \
        --pipeline_name profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells \
        --plate_layout single_rep \
        --cc_threshold 20
    echo ""
done

# Run multi-rep batches
for batch in "${MULTI_REP_BATCHES[@]}"; do
    batch_num=$(echo "$batch" | grep -o "Batch_[0-9]*" | grep -o "[0-9]*")
    echo "Running Batch $batch_num (multi_rep): $batch"
    python classification/classify_gfp_filtered_control_cmd.py \
        --input_path outputs/batch_profiles/$batch/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet \
        --input_path_orig outputs/batch_profiles/$batch/profiles.parquet \
        --output_base_dir outputs/classification_analyses/$batch \
        --pipeline_name profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells \
        --plate_layout multi_rep \
        --cc_threshold 20
    echo ""
done

echo "All control GFP-filtered classifications completed!"
