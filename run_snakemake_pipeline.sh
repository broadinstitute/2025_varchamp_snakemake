#!/bin/bash

## Run image QC
# python 1.image_preprocess_qc/scripts/1_calc_plate_bg.py --batch_list "2024_01_23_Batch_7,2024_02_06_Batch_8,2024_12_09_Batch_11,2024_12_09_Batch_12,2025_03_17_Batch_15,2025_03_17_Batch_16" --input_dir "1.image_preprocess_qc/inputs/cpg_imgs" --output_dir "1.image_well_qc_results/outputs/plate_bg_summary" --workers 256

# cp snakemake_files/Snakefile_batch7 /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline
# cp snakemake_files/Snakefile_batch8 /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline
# cp snakemake_files/Snakefile_batch11 /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline
# cp snakemake_files/Snakefile_batch12 /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline
# cp snakemake_files/Snakefile_batch15 /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline
# cp snakemake_files/Snakefile_batch16 /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline

# cd /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline

# snakemake \
#     --snakefile Snakefile_batch15 \
#     --directory /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline \
#     --cores 256 &> /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/3_outputs/1_snakemake_pipeline/4.sm_pipeline_outputs_tmp/snakemake_logs/Snakefile_batch15.log

# snakemake \
#     --snakefile Snakefile_batch16 \
#     --directory /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline \
#     --cores 256 &> /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/3_outputs/1_snakemake_pipeline/4.sm_pipeline_outputs_tmp/snakemake_logs/Snakefile_batch16.log

# snakemake \
#     --snakefile Snakefile_batch11 \
#     --directory /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline \
#     --cores 256 &> /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/3_outputs/1_snakemake_pipeline/4.sm_pipeline_outputs_tmp/snakemake_logs/Snakefile_batch11.log

# snakemake \
#     --snakefile Snakefile_batch12 \
#     --directory /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline \
#     --cores 256 &> /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/3_outputs/1_snakemake_pipeline/4.sm_pipeline_outputs_tmp/snakemake_logs/Snakefile_batch12.log

# snakemake \
#     --snakefile Snakefile_batch7 \
#     --directory /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline \
#     --cores 256 &> /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/3_outputs/1_snakemake_pipeline/4.sm_pipeline_outputs_tmp/snakemake_logs/Snakefile_batch7.log

# snakemake \
#     --snakefile Snakefile_batch8 \
#     --directory /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/2_analysis/1_snakemake_pipeline/2025_varchamp_snakemake/2.snakemake_pipeline \
#     --cores 256 &> /home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/3_outputs/1_snakemake_pipeline/4.sm_pipeline_outputs_tmp/snakemake_logs/Snakefile_batch8.log