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

# Batch 11 - Multi-rep layout
echo "Running Batch 11 (multi_rep)..."
python classification/classify_gfp_filtered_control_cmd.py \
    --input_path outputs/batch_profiles/2024_12_09_Batch_11/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet \
    --input_path_orig outputs/batch_profiles/2024_12_09_Batch_11/profiles.parquet \
    --output_base_dir outputs/classification_analyses/2024_12_09_Batch_11 \
    --pipeline_name profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells \
    --plate_layout multi_rep \
    --cc_threshold 20

# Batch 12 - Multi-rep layout
echo "Running Batch 12 (multi_rep)..."
python classification/classify_gfp_filtered_control_cmd.py \
    --input_path outputs/batch_profiles/2024_12_09_Batch_12/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet \
    --input_path_orig outputs/batch_profiles/2024_12_09_Batch_12/profiles.parquet \
    --output_base_dir outputs/classification_analyses/2024_12_09_Batch_12 \
    --pipeline_name profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells \
    --plate_layout multi_rep \
    --cc_threshold 20

# Batch 18 - Single-rep layout
echo "Running Batch 18 (single_rep)..."
python classification/classify_gfp_filtered_control_cmd.py \
    --input_path outputs/batch_profiles/2025_06_10_Batch_18/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet \
    --input_path_orig outputs/batch_profiles/2025_06_10_Batch_18/profiles.parquet \
    --output_base_dir outputs/classification_analyses/2025_06_10_Batch_18 \
    --pipeline_name profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells \
    --plate_layout single_rep \
    --cc_threshold 20

# Batch 19 - Single-rep layout
echo "Running Batch 19 (single_rep)..."
python classification/classify_gfp_filtered_control_cmd.py \
    --input_path outputs/batch_profiles/2025_06_10_Batch_19/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet \
    --input_path_orig outputs/batch_profiles/2025_06_10_Batch_19/profiles.parquet \
    --output_base_dir outputs/classification_analyses/2025_06_10_Batch_19 \
    --pipeline_name profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells \
    --plate_layout single_rep \
    --cc_threshold 20

echo "All control GFP-filtered classifications completed!"
