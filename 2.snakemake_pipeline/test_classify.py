#!/usr/bin/env python
"""
Test script for verifying classification pipeline changes.
Runs classification on a given batch and writes outputs to a Batch_<N>_test directory.

Usage:
    python test_classify.py --batch 2026_01_05_Batch_20
    python test_classify.py --batch 2026_01_05_Batch_21 --cc-threshold 20 --plate-layout single_rep
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import classification

PIPELINE = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", required=True, help="Full batch ID, e.g. 2026_01_05_Batch_20")
    p.add_argument("--cc-threshold", type=int, default=20)
    p.add_argument("--plate-layout", default="single_rep", choices=["single_rep", "multi_rep"])
    p.add_argument("--protein-channel-name", default="Protein")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    batch_input = args.batch
    # "2026_01_05_Batch_20" -> "Batch_20_test"
    batch_short = batch_input.split("_", 3)[-1]
    batch_output = f"{batch_short}_test"

    input_path = f"{base_dir}/outputs/batch_profiles/{batch_input}/{PIPELINE}.parquet"
    input_path_orig = f"{base_dir}/outputs/batch_profiles/{batch_input}/profiles_tcdropped_filtered_var_mad_outlier.parquet"

    results_dir = f"{base_dir}/outputs/classification_results/{batch_output}/{PIPELINE}"
    analyses_dir = f"{base_dir}/outputs/classification_analyses/{batch_output}/{PIPELINE}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(analyses_dir, exist_ok=True)

    feat_output_path = f"{results_dir}/feat_importance.csv"
    info_output_path = f"{results_dir}/classifier_info.csv"
    preds_output_path = f"{results_dir}/predictions.parquet"
    feat_output_path_gfp = f"{results_dir}/feat_importance_gfp_adj.csv"
    info_output_path_gfp = f"{results_dir}/classifier_info_gfp_adj.csv"
    preds_output_path_gfp = f"{results_dir}/predictions_gfp_adj.parquet"
    filtered_cell_path = f"{results_dir}/gfp_adj_filtered_cells_profiles.parquet"

    print("=" * 70)
    print(f"Test Classification Run for {batch_input}")
    print("=" * 70)
    print(f"Input path:        {input_path}")
    print(f"Input path (orig): {input_path_orig}")
    print(f"Results dir:       {results_dir}")
    print(f"Plate layout:      {args.plate_layout}")
    print(f"CC threshold:      {args.cc_threshold}")
    print(f"Protein channel:   {args.protein_channel_name}")
    print("=" * 70)

    for path in (input_path, input_path_orig):
        if not os.path.exists(path):
            sys.exit(f"ERROR: Input file not found: {path}")

    print("\nStarting classification...")
    classification.run_classify_workflow(
        input_path=input_path,
        input_path_orig=input_path_orig,
        feat_output_path=feat_output_path,
        info_output_path=info_output_path,
        preds_output_path=preds_output_path,
        feat_output_path_gfp=feat_output_path_gfp,
        info_output_path_gfp=info_output_path_gfp,
        preds_output_path_gfp=preds_output_path_gfp,
        filtered_cell_path=filtered_cell_path,
        cc_threshold=args.cc_threshold,
        plate_layout=args.plate_layout,
        protein_channel_name=args.protein_channel_name,
    )

    print(f"\nClassification completed. Results at: {results_dir}")

    print("\n" + "=" * 70)
    print("Verification: Checking for control alleles in GFP-adjusted results")
    print("=" * 70)
    df_info = pd.read_csv(info_output_path_gfp)
    print(f"Total classifiers in classifier_info_gfp_adj.csv: {len(df_info)}")
    print("Metadata_Control distribution:")
    print(df_info["Metadata_Control"].value_counts())

    if df_info["Metadata_Control"].any():
        print("\nSUCCESS: Control alleles ARE present in GFP-adjusted results!")
    else:
        print("\nWARNING: No control alleles found in GFP-adjusted results.")


if __name__ == "__main__":
    main()
