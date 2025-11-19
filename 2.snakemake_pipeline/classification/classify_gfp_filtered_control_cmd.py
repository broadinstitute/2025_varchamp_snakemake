"""
GFP-Filtered Control Classification CLI Script

This script provides a standalone command-line interface for running GFP-filtered
classification on control alleles. It supports both single-rep and multi-rep plate layouts.

Outputs consistent filenames (*_control_gfp_adj.*) following the standard directory structure:
- classification_results/<batch>/<pipeline>/ - predictions, feature importance, classifier info, log
- classification_analyses/<batch>/<pipeline>/ - metrics only

Usage:
    python classify_gfp_filtered_control_cmd.py \\
        --input_path <path_to_input_parquet> \\
        --input_path_orig <path_to_original_parquet> \\
        --output_base_dir <base_output_directory> \\
        --pipeline_name <pipeline_subdirectory> \\
        --plate_layout <single_rep|multi_rep> \\
        --cc_threshold <minimum_cells_per_well>

Example:
    python classify_gfp_filtered_control_cmd.py \\
        --input_path outputs/batch_profiles/2025_03_17_Batch_15/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet \\
        --input_path_orig outputs/batch_profiles/2025_03_17_Batch_15/profiles.parquet \\
        --output_base_dir outputs/classification_analyses/2025_03_17_Batch_15 \\
        --pipeline_name profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells \\
        --plate_layout single_rep \\
        --cc_threshold 20
"""

import os
import sys
import argparse
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import find_feat_cols, remove_nan_infs_columns

# Import classification functions
# Use absolute imports since we're running as a script
import classification.classify_helper_func as helper
import classification.classify_gfp_filter_func as gfp_filter
from classification.classify_single_rep_per_plate import get_common_plates, stratify_by_plate
from classification.analysis import calculate_class_metrics

# Extract needed functions and constants
add_control_annot = helper.add_control_annot
drop_low_cc_wells = helper.drop_low_cc_wells
control_group_runner_gfp_filtered = gfp_filter.control_group_runner_gfp_filtered
control_group_runner_fewer_rep_gfp_filtered = gfp_filter.control_group_runner_fewer_rep_gfp_filtered
GFP_INTENSITY_COLUMN = gfp_filter.GFP_INTENSITY_COLUMN


def main():
    parser = argparse.ArgumentParser(
        description="Run GFP-filtered classification on control alleles"
    )

    # Input/output arguments
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to feature-selected profiles parquet file"
    )
    parser.add_argument(
        "--input_path_orig",
        type=str,
        required=True,
        help="Path to original profiles parquet file (for GFP intensity data)"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base output directory (e.g., outputs/2025_03_17_Batch_15)"
    )
    parser.add_argument(
        "--pipeline_name",
        type=str,
        required=True,
        help="Pipeline subdirectory name (e.g., profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells)"
    )

    # Experimental design arguments
    parser.add_argument(
        "--plate_layout",
        type=str,
        required=True,
        choices=["single_rep", "multi_rep"],
        help="Experimental plate layout: single_rep or multi_rep"
    )
    parser.add_argument(
        "--cc_threshold",
        type=int,
        default=20,
        help="Minimum cell count threshold per well (default: 20)"
    )

    args = parser.parse_args()

    # Create output directories following the standard structure:
    # classification_results/<batch>/<pipeline>/ - for predictions, feature importance, etc.
    # classification_analyses/<batch>/<pipeline>/ - for metrics only
    results_dir = os.path.join(
        args.output_base_dir.replace("classification_analyses", "classification_results"),
        args.pipeline_name
    )
    analyses_dir = os.path.join(
        args.output_base_dir.replace("classification_results", "classification_analyses"),
        args.pipeline_name
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(analyses_dir, exist_ok=True)

    # Define output file paths with consistent naming convention
    # Most outputs go to classification_results
    feat_output_path = os.path.join(results_dir, "feat_importance_control_gfp_adj.csv")
    info_output_path = os.path.join(results_dir, "classifier_info_control_gfp_adj.csv")
    preds_output_path = os.path.join(results_dir, "predictions_control_gfp_adj.parquet")
    filtered_cell_path = os.path.join(results_dir, "filtered_cells_control_gfp_adj.parquet")
    log_file_path = os.path.join(results_dir, "classify_control_gfp_adj.log")

    # Metrics go to classification_analyses
    metrics_output_path = os.path.join(analyses_dir, "metrics_control_gfp_adj.csv")

    # Open log file
    log_file = open(log_file_path, "w")
    log_file.write("=== GFP-Filtered Control Classification ===\n")
    log_file.write(f"Input path: {args.input_path}\n")
    log_file.write(f"Original path: {args.input_path_orig}\n")
    log_file.write(f"Results output directory: {results_dir}\n")
    log_file.write(f"Analyses output directory: {analyses_dir}\n")
    log_file.write(f"Plate layout: {args.plate_layout}\n")
    log_file.write(f"Cell count threshold: {args.cc_threshold}\n")
    log_file.write("=" * 50 + "\n\n")

    # Create ParquetWriter for predictions
    schema = pa.schema([
        ("Classifier_ID", pa.string()),
        ("CellID", pa.string()),
        ("Label", pa.int64()),
        ("Prediction", pa.float32()),
        ("Metadata_Feature_Type", pa.string()),
        ("Metadata_Control", pa.bool_()),
    ])
    pq_writer = pq.ParquetWriter(preds_output_path, schema, compression="gzip")

    # Load data
    log_file.write("Loading data...\n")
    dframe_featsel = (
        pl.scan_parquet(args.input_path)
        .with_columns(
            pl.concat_str(
                [
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber",
                ],
                separator="_",
            ).alias("Metadata_CellID")
        )
    )

    # Join with original data if GFP intensity column is missing
    if GFP_INTENSITY_COLUMN not in dframe_featsel.columns:
        log_file.write("Joining with original data for GFP intensity...\n")
        dframe_orig = (
            pl.scan_parquet(args.input_path_orig)
            .with_columns(
                pl.concat_str([
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber"
                ], separator="_").alias("Metadata_CellID")
            )
            .select(pl.col(["Metadata_CellID", GFP_INTENSITY_COLUMN]))
        )
        dframe = dframe_featsel.join(
            dframe_orig, on="Metadata_CellID", how="left"
        ).collect().to_pandas()
    else:
        dframe = dframe_featsel.collect().to_pandas()

    log_file.write(f"Loaded {dframe.shape[0]} cells\n")

    # Data validation
    feat_col = find_feat_cols(dframe)
    if (~dframe[feat_col].isna()).any().any() or (~dframe[feat_col].isfinite()).all().all():
        log_file.write("Cleaning NaN and infinite values...\n")
        dframe = remove_nan_infs_columns(dframe)

    # Filter rows and add annotations
    dframe = dframe[~dframe["Metadata_well_position"].isna()]
    dframe = add_control_annot(dframe)
    dframe = dframe[~dframe["Metadata_control"].isna()]

    # Get control alleles based on plate layout
    if args.plate_layout == "single_rep":
        df_control = dframe[dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
        # Remove TC from analysis
        df_control = df_control[df_control["Metadata_node_type"] != "TC"].reset_index(drop=True)
        log_file.write(f"Single-rep layout: {df_control.shape[0]} control cells\n")
    else:  # multi_rep
        df_control = dframe[dframe["Metadata_node_type"] != "TC"].reset_index(drop=True)
        log_file.write(f"Multi-rep layout: {df_control.shape[0]} control cells\n")

    # Drop low cell count wells
    log_file.write("Dropping low cell count wells...\n")
    df_control = drop_low_cc_wells(df_control, args.cc_threshold, log_file)
    log_file.write(f"After QC: {df_control.shape[0]} control cells\n\n")

    # Run GFP-filtered control classification
    log_file.write("Running GFP-filtered control classification...\n")

    if args.plate_layout == "single_rep":
        df_feat, df_result, df_filtered_cells = control_group_runner_gfp_filtered(
            df_control,
            pq_writer=pq_writer,
            log_file=log_file,
            feat_type="GFP",
            min_cells_per_well=args.cc_threshold
        )
    else:  # multi_rep
        df_feat, df_result, df_filtered_cells = control_group_runner_fewer_rep_gfp_filtered(
            df_control,
            pq_writer=pq_writer,
            err_logger=log_file,
            feat_type="GFP",
            min_cells_per_well=args.cc_threshold
        )

    # Close parquet writer
    pq_writer.close()

    # Write output files
    log_file.write("\nWriting output files...\n")

    if df_feat.shape[0] > 0:
        df_feat.to_csv(feat_output_path, index=False)
        log_file.write(f"Feature importance: {feat_output_path}\n")
    else:
        log_file.write("WARNING: No feature importance results generated\n")

    if df_result.shape[0] > 0:
        df_result = df_result.drop_duplicates()
        df_result.to_csv(info_output_path, index=False)
        log_file.write(f"Classifier info: {info_output_path}\n")
    else:
        log_file.write("WARNING: No classifier info results generated\n")

    if df_filtered_cells.shape[0] > 0:
        df_filtered_cells.to_parquet(filtered_cell_path, index=False)
        log_file.write(f"Filtered cells: {filtered_cell_path}\n")
    else:
        log_file.write("WARNING: No filtered cells generated\n")

    log_file.write(f"Predictions: {preds_output_path}\n")

    log_file.write("\nGFP-filtered control classification completed successfully!\n")
    log_file.close()

    print(f"\n=== GFP-Filtered Control Classification Complete ===")
    print(f"Classification results: {results_dir}")
    print(f"Analysis metrics: {analyses_dir}")
    print(f"Log file: {log_file_path}")

    # Calculate classification metrics
    if df_result.shape[0] > 0 and os.path.exists(preds_output_path):
        print(f"\nCalculating classification metrics...")
        try:
            calculate_class_metrics(
                classifier_info=info_output_path,
                predictions=preds_output_path,
                metrics_file=metrics_output_path
            )
            print(f"✓ Metrics written to: {metrics_output_path}")
        except Exception as e:
            print(f"WARNING: Failed to calculate metrics: {e}")
    else:
        print("\nSkipping metrics calculation (no results generated)")


if __name__ == "__main__":
    main()
