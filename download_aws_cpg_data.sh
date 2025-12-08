#!/bin/bash

## Download the necessary data and files for imaging analyses pipeline from AWS Cell-Painting Gallery (CPG)

## aws cpg paths, publicly available
AWS_IMG_PATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/images"
AWS_WORKSPACE_PATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace"
AWS_ANALYSIS_PATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace/analysis"

## Download the aws cpg data to your local directory
## To be replaced by user's choices 
CPG_IMG_PATH="./1.image_preprocess_qc/inputs/cpg_imgs" ## symbolic link to /data/shenrunx/igvf/varchamp/2021_09_01_VarChAMP_imgs
SNAKEMAKE_INPUT_PATH="./2.snakemake_pipeline/inputs"
BATCHES="2024_01_23_Batch_7 2024_02_06_Batch_8 2024_12_09_Batch_11 2024_12_09_Batch_12 2025_01_27_Batch_13 2025_01_28_Batch_14 2025_03_17_Batch_15 2025_03_17_Batch_16"

## Private batches
# BATCHES="2025_06_10_Batch_18 2025_06_10_Batch_19"
# BATCHES="2025_05_23_Batch_17"

## create directories
# mkdir -p $CPG_IMG_PATH
# mkdir -p $SNAKEMAKE_INPUT_PATH/single_cell_profiles
# mkdir -p $SNAKEMAKE_INPUT_PATH/metadata/platemaps

for batch_id in $BATCHES;
do
    ## download the raw img data
    # aws s3 sync --no-sign-request "$AWS_IMG_PATH/$batch_id/images" $CPG_IMG_PATH/$batch_id/images
    # aws s3 sync --no-sign-request \
    #     "$AWS_ANALYSIS_PATH/$batch_id" \
    #     "$CPG_IMG_PATH/$batch_id/analysis" \
    #     --exclude "*" \
    #     --include "**/Cells.csv" \
    #     --include "**/Cytoplasm.csv" \
    #     --include "**/Nuclei.csv" \
    #     --include "**/Image.csv" \
        # --recursive \
        # --dry-run

    ## download the CellProfiler single-cell profiles and metadata for snakemake analysis pipeline
    # aws s3 sync --no-sign-request --exclude "*.csv" "$AWS_WORKSPACE_PATH/backend/$batch_id/2025_04_01_B17A9R1" $SNAKEMAKE_INPUT_PATH/single_cell_profiles/$batch_id/2025_04_01_B17A9R1
    # aws s3 sync --no-sign-request --exclude "*.csv" "$AWS_WORKSPACE_PATH/backend/$batch_id/2025_04_21_B17A9R2_media_change" $SNAKEMAKE_INPUT_PATH/single_cell_profiles/$batch_id/2025_04_21_B17A9R2_media_change
    aws s3 sync --no-sign-request \
        "$AWS_WORKSPACE_PATH/metadata/platemaps/$batch_id/platemap" \
        $SNAKEMAKE_INPUT_PATH/metadata/platemaps/$batch_id/platemap \
        --exclude "*" \
        --include "*.txt" \
        --exclude "*-checkpoint.txt" #\ --dryrun
done