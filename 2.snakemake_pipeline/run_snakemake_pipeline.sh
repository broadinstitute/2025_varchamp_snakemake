#!/bin/bash

# cp inputs/snakemake_files/Snakefile_batch17 .
# nohup snakemake \
#     --snakefile Snakefile_batch17 \
#     --cores all &> outputs/snakemake_logs/snakemake_batch17.log

# cp inputs/snakemake_files/Snakefile_batch17_mc .
# nohup snakemake \
#     --snakefile Snakefile_batch17_mc \
#     --cores all &> outputs/snakemake_logs/snakemake_batch17_mc.log

# Run batch 11
# cp inputs/snakemake_files/Snakefile_batch11 .
# snakemake \
#     --snakefile Snakefile_batch11 \
#     --cores 128 &> outputs/snakemake_logs/snakemake_batch11.log

# # Run batch 11
# cp inputs/snakemake_files/Snakefile_batch11 .
# snakemake \
#     --snakefile Snakefile_batch11 \
#     --cores 128 &> outputs/snakemake_logs/snakemake_batch11_new.log

# ## Run batch 12
# cp inputs/snakemake_files/Snakefile_batch12 .
# snakemake \
#     --snakefile Snakefile_batch12 \
#     --cores 128 &> outputs/snakemake_logs/snakemake_batch12_new.log

cp inputs/snakemake_files/Snakefile_batch15 .
nohup snakemake \
    --snakefile Snakefile_batch15 \
    --cores all &> outputs/snakemake_logs/snakemake_batch15_new.log
rm Snakefile_batch15

cp inputs/snakemake_files/Snakefile_batch16 .
nohup snakemake \
    --snakefile Snakefile_batch16 \
    --cores all &> outputs/snakemake_logs/snakemake_batch16_new.log
rm Snakefile_batch16

# Run batch 13
cp inputs/snakemake_files/Snakefile_batch13 .
snakemake \
    --snakefile Snakefile_batch13 \
    --cores 256 &> outputs/snakemake_logs/snakemake_batch13_new.log
rm Snakefile_batch13

## Run batch 14
cp inputs/snakemake_files/Snakefile_batch14 .
snakemake \
    --snakefile Snakefile_batch14 \
    --cores 256 &> outputs/snakemake_logs/snakemake_batch14_new.log
rm Snakefile_batch14

# Run batch 8
cp inputs/snakemake_files/Snakefile_batch8 .
nohup snakemake \
    --snakefile Snakefile_batch8 \
    --cores 256 &> outputs/snakemake_logs/snakemake_batch8_new.log
rm Snakefile_batch8

# Run batch 7
cp inputs/snakemake_files/Snakefile_batch7 .
nohup snakemake \
    --snakefile Snakefile_batch7 \
    --cores 256 &> outputs/snakemake_logs/snakemake_batch7_new.log
rm Snakefile_batch7

# Run batch 18
cp inputs/snakemake_files/Snakefile_batch18 .
snakemake \
    --snakefile Snakefile_batch18 \
    --cores 256 &> outputs/snakemake_logs/snakemake_batch18_new.log

## Run batch 19
cp inputs/snakemake_files/Snakefile_batch19 .
snakemake \
    --snakefile Snakefile_batch19 \
    --cores 256 &> outputs/snakemake_logs/snakemake_batch19_new.log