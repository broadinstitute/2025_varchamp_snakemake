#!/bin/bash

BASEPATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace"

#### Upload data to S3
### Done on Dec 7, 2025, after removing all the incorrect reference and variant alleles from the VarChAMP datasets
### Involving alleles from Batch 7,8,11,12,13,14,15,16
# BATCHES="2024_01_23_Batch_7 2024_02_06_Batch_8 2024_12_09_Batch_11 2024_12_09_Batch_12 2025_01_27_Batch_13 2025_01_28_Batch_14 2025_03_17_Batch_15 2025_03_17_Batch_16"
# BATCHES="2025_03_17_Batch_15 2025_03_17_Batch_16"
BATCHES="2025_06_10_Batch_18 2025_06_10_Batch_19"

for batch_id in $BATCHES;
do
    # UPLOADPATH="$BASEPATH/metadata/platemaps/$batch_id/platemap"
    # aws s3 cp \
    #     ../outputs/corrected_platemaps/$batch_id/platemap \
    #     "$UPLOADPATH" \
    #     --recursive \
    #     --exclude "*" \
    #     --include "*.txt" \
    #     --include "*.csv" \
    #     --profile jump-cp-role
    #     # --dryrun

    # UPLOADPATH_BARCODE_PM="$BASEPATH/metadata/platemaps/$batch_id/barcode_platemap.csv"
    # aws s3 cp \
    #     ../../2.snakemake_pipeline/inputs/metadata/platemaps/$batch_id/barcode_platemap.csv \
    #     "$UPLOADPATH_BARCODE_PM" \
    #     --exclude "*" \
    #     --include "*.txt" \
    #     --include "*.csv" \
    #     --profile jump-cp-role #\
        # --dryrun
done