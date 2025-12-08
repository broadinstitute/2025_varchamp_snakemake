#!/bin/bash

## Download the necessary data and files for imaging analyses pipeline from AWS Cell-Painting Gallery (CPG)
## aws cpg paths, publicly available
AWS_WORKSPACE_PATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace"

## Download the aws cpg data to your local directory
## To be replaced by user's choices 
PLATEMAP_PATH="../inputs/platemaps"
BATCHES="2024_01_23_Batch_7 2024_02_06_Batch_8 2024_12_09_Batch_11 2024_12_09_Batch_12 2025_01_27_Batch_13 2025_01_28_Batch_14 2025_03_17_Batch_15 2025_03_17_Batch_16"

## Private alleles excluded from one-percent analysis
## No allele sequence confirmation results for these batches
## No QC could be done for these batches
## BATCHES="2025_05_23_Batch_17 2025_06_10_Batch_18 2025_06_10_Batch_19"

## create directories
mkdir -p $PLATEMAP_PATH

for batch_id in $BATCHES;
do
    aws s3 sync --no-sign-request "$AWS_WORKSPACE_PATH/metadata/platemaps/$batch_id" $PLATEMAP_PATH/$batch_id
done