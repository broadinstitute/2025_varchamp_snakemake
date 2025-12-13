"""
VarChAMP Classification Pipeline - Main Workflow Module

This module orchestrates the complete classification workflow for the VarChAMP 
(Variant Classification via Cell Painting) pipeline. It integrates multiple 
specialized classification modules to provide comprehensive variant analysis 
with optional GFP-intensity correction.

## Main Functionality

- **Data Loading & Preprocessing**: Handles Parquet files, data validation, and cell filtering
- **Multi-Modal Classification**: Supports different plate layouts (single_rep vs multi_rep)
- **Feature Type Processing**: Processes GFP, DNA, AGP, Mito, and Morph features
- **GFP-Intensity Correction**: Optional correction for GFP confounding effects
- **Control & Experimental Analysis**: Null distributions and variant-reference comparisons
- **Output Generation**: Feature importance, classifier metrics, and cell-level predictions

## Architecture

The workflow is modularized across several specialized modules:
- `classify_helper_func`: Core ML classifier, feature utilities, and helper functions
- `classify_single_rep_per_plate`: Single replicate per plate experimental design
- `classify_multi_rep_per_plate`: Multiple replicates per plate experimental design  
- `classify_gfp_filter_func`: GFP-intensity correction functionality

## Usage

This module is typically called through the Snakemake pipeline with specified
input/output paths and configuration parameters.
"""

import os
import sys
import warnings
import gc
import psutil
from typing import Union
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, remove_nan_infs_columns

# Import all helper functions and constants from classify_helper_func
from .classify_helper_func import (
    FEAT_TYPE_SET,
    control_type_helper,
    add_control_annot,
    drop_low_cc_wells,
    get_classifier_features,
    classifier
)

# Import specialized classification modules
from .classify_single_rep_per_plate import (
    experimental_runner, 
    control_group_runner
)
from .classify_multi_rep_per_plate import (
    control_group_runner_fewer_rep,
    experimental_runner_plate_rep
)
from .classify_gfp_filter_func import (
    experimental_runner_filter_gfp,
    experimental_runner_plate_rep_gfp_filtered,
    control_group_runner_gfp_filtered,
    control_group_runner_fewer_rep_gfp_filtered,
    GFP_INTENSITY_COLUMN
)

#######################################
# MEMORY MANAGEMENT FUNCTIONS
#######################################

def log_memory_usage(log_file, stage=""):
    """Log current memory usage including swap"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        log_file.write(f"[{stage}] Memory Usage:\n")
        log_file.write(f"  RSS: {memory_info.rss / 1024**3:.2f} GB\n")
        log_file.write(f"  VMS: {memory_info.vms / 1024**3:.2f} GB\n")
        log_file.write(f"  System RAM: {system_memory.used / 1024**3:.2f} / {system_memory.total / 1024**3:.2f} GB ({system_memory.percent}%)\n")
        log_file.write(f"  Swap: {swap_memory.used / 1024**3:.2f} / {swap_memory.total / 1024**3:.2f} GB ({swap_memory.percent}%)\n")
        log_file.flush()
    except Exception as e:
        log_file.write(f"Failed to log memory usage: {e}\n")


def force_memory_cleanup(log_file):
    """Force garbage collection and attempt to free memory"""
    if log_file is not None:
        log_file.write("Forcing memory cleanup...\n")

    # Force garbage collection multiple times
    for i in range(3):
        collected = gc.collect()
        if log_file is not None:
            log_file.write(f"  GC round {i+1}: collected {collected} objects\n")

    # Try to release unused memory back to OS (Linux only)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        if log_file is not None:
            log_file.write("  malloc_trim() called\n")
    except Exception as e:
        if log_file is not None:
            log_file.write(f"  malloc_trim() failed: {e}\n")

    if log_file is not None:
        log_file.flush()


def check_memory_limits(log_file, max_memory_gb=32):
    """Check if memory usage is approaching limits"""
    try:
        system_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        total_used_gb = (system_memory.used + swap_memory.used) / 1024**3
        
        if total_used_gb > max_memory_gb:
            log_file.write(f"WARNING: Memory usage ({total_used_gb:.2f} GB) exceeds limit ({max_memory_gb} GB)\n")
            return False
        return True
    except Exception as e:
        log_file.write(f"Failed to check memory limits: {e}\n")
        return True
    

#######################################
# MAIN WORKFLOW FUNCTION
#######################################

def run_classify_workflow(
    input_path: str,
    input_path_orig: str,
    feat_output_path: str,
    info_output_path: str,
    preds_output_path: str,
    feat_output_path_gfp: str,
    info_output_path_gfp: str,
    preds_output_path_gfp: str,
    filtered_cell_path: str,
    cc_threshold: int,
    plate_layout: str,
    use_gpu: Union[str, None] = None, ## "0,1"
    feat_output_path_control_gfp: Union[str, None] = None,
    info_output_path_control_gfp: Union[str, None] = None,
    preds_output_path_control_gfp: Union[str, None] = None,
    filtered_cell_path_control: Union[str, None] = None
):
    """
    Run workflow for single-cell classification
    
    Parameters:
    -----------
    input_path : str
        Path to main processed profiles parquet file
    input_path_orig : str  
        Path to original profiles parquet file (for GFP intensity data)
    feat_output_path : str
        Output path for feature importance CSV
    info_output_path : str
        Output path for classifier info CSV
    preds_output_path : str
        Output path for predictions parquet
    feat_output_path_gfp : str
        Output path for GFP-adjusted feature importance CSV
    info_output_path_gfp : str
        Output path for GFP-adjusted classifier info CSV
    preds_output_path_gfp : str
        Output path for GFP-adjusted predictions parquet
    filtered_cell_path : str
        Output path for GFP-filtered cell profiles parquet
    cc_threshold : int
        Minimum cell count threshold per well
    plate_layout : str
        Experimental layout: "single_rep" or "multi_rep"
    use_gpu : str, optional
        CUDA device specification
    """
    assert plate_layout in ("single_rep", "multi_rep"), f"Incorrect plate_layout: {plate_layout}, only 'single_rep' and 'multi_rep' allowed."

    if use_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

    # Initialize parquet for cell-level predictions
    if os.path.exists(preds_output_path):
        os.remove(preds_output_path)
    
    # Create a log file
    logfile_path = os.path.join('/'.join(preds_output_path.split("/")[:-1]), "classify.log")

    # Output the prediction.parquet schema
    schema = pa.schema([
        ("Classifier_ID", pa.string()),
        ("CellID", pa.string()),
        ("Label", pa.int64()),
        ("Prediction", pa.float32()),
        ("Metadata_Feature_Type", pa.string()),
        ("Metadata_Control", pa.bool_()),
    ])
    writer = pq.ParquetWriter(preds_output_path, schema, compression="gzip")
    writer_gfp = pq.ParquetWriter(preds_output_path_gfp, schema, compression="gzip")

    # Create control GFP writer if output paths are provided
    writer_control_gfp = None
    if preds_output_path_control_gfp is not None:
        writer_control_gfp = pq.ParquetWriter(preds_output_path_control_gfp, schema, compression="gzip")

    # Add CellID column
    dframe_featsel = (
        pl.scan_parquet(input_path)
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
        dframe_orig = (
            pl.scan_parquet(input_path_orig)
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

    # Data validation and preprocessing
    feat_col = find_feat_cols(dframe)
    try:
        assert (
            ~np.isnan(dframe[feat_col]).any().any()
        ), "Dataframe contains no NaN features."
        assert (
            np.isfinite(dframe[feat_col]).all().all()
        ), "Dataframe contains finite feature values."
    except AssertionError:
        dframe = remove_nan_infs_columns(dframe)

    # Filter rows with NaN Metadata
    dframe = dframe[~dframe["Metadata_well_position"].isna()]
    dframe = add_control_annot(dframe)
    dframe = dframe[~dframe["Metadata_control"].isna()]

    # Store the classifier feat_importance and classification_res
    feat_import_dfs, class_res_dfs = [], []
    feat_import_gfp_adj_dfs, class_res_gfp_adj_dfs, filtered_cells_gfp_adj_dfs = [], [], []
    feat_import_control_gfp_dfs, class_res_control_gfp_dfs, filtered_cells_control_gfp_dfs = [], [], []
    
    # Split data into experimental df with var and ref alleles
    df_exp = dframe[~dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
    
    with open(logfile_path, "w") as log_file:
        log_file.write(f"===============================================================================================================================================================\n")
        log_file.write("Dropping low cell count wells in ref. vs variant alleles:\n")
        print("Dropping low cell count wells in ref. vs variant alleles:")
        df_exp = drop_low_cc_wells(df_exp, cc_threshold, log_file)
        log_file.write(f"===============================================================================================================================================================\n\n")
        
        # Check the plate_layout for the correct classification set-up
        if (plate_layout=="single_rep"):
            # If the plate_layout is single_rep, with only one well per allele on a single plate
            # we can only get the control_df with the control labels
            df_control = dframe[dframe["Metadata_control"].astype("bool")].reset_index(
                drop=True
            )
            # Remove any remaining TC from analysis
            df_control = df_control[df_control["Metadata_node_type"] != "TC"].reset_index(
                drop=True
            )
            log_file.write("Dropping low cell count wells in ONLY the control alleles on the same plate:\n")
            print("Dropping low cell count wells in ONLY the control alleles on the same plate:")
            # Filter out wells with fewer than the cell count threshold
            df_control = drop_low_cc_wells(df_control, cc_threshold, log_file)
            print("Check ctrl df:")
            print(df_control)

            for feat in FEAT_TYPE_SET:
                print(feat)
                # GFP corrected version
                if feat == "GFP":
                    # Run experimental GFP filtering
                    df_feat_pro_exp_gfp_adj, df_result_pro_exp_gfp_adj, df_filtered_cells_gfp_adj = experimental_runner_filter_gfp(
                        df_exp, pq_writer=writer_gfp,
                        log_file=log_file, min_cells_per_well=cc_threshold
                    )
                    # Store to another set of feat_df, res_df and filtered_cells parquet
                    if (df_feat_pro_exp_gfp_adj.shape[0] > 0):
                        feat_import_gfp_adj_dfs += [df_feat_pro_exp_gfp_adj]
                        class_res_gfp_adj_dfs += [df_result_pro_exp_gfp_adj]
                        filtered_cells_gfp_adj_dfs += [df_filtered_cells_gfp_adj]

                    # Run control GFP filtering if output paths are provided
                    if writer_control_gfp is not None:
                        df_feat_pro_con_gfp, df_result_pro_con_gfp, df_filtered_cells_con_gfp = control_group_runner_gfp_filtered(
                            df_control, pq_writer=writer_control_gfp,
                            log_file=log_file, min_cells_per_well=cc_threshold
                        )
                        if (df_feat_pro_con_gfp.shape[0] > 0):
                            feat_import_control_gfp_dfs += [df_feat_pro_con_gfp]
                            class_res_control_gfp_dfs += [df_result_pro_con_gfp]
                            filtered_cells_control_gfp_dfs += [df_filtered_cells_con_gfp]

                    ## drop the gfp column during the inference if it was not selected during feature selection
                    if GFP_INTENSITY_COLUMN not in dframe_featsel.columns:
                        df_exp = df_exp.drop(GFP_INTENSITY_COLUMN, axis=1)
                        df_control = df_control.drop(GFP_INTENSITY_COLUMN, axis=1)
                
                df_feat_pro_con, df_result_pro_con = control_group_runner(
                    df_control, 
                    pq_writer=writer, 
                    log_file=log_file, 
                    feat_type=feat
                )
                df_feat_pro_exp, df_result_pro_exp = experimental_runner(
                    df_exp, 
                    pq_writer=writer, 
                    log_file=log_file, 
                    feat_type=feat
                )
                if (df_feat_pro_con.shape[0] > 0):
                    feat_import_dfs += [df_feat_pro_con, df_feat_pro_exp]
                    class_res_dfs += [df_result_pro_con, df_result_pro_exp]
                
        else:
            # If the plate_layout is multi_rep, with multiple wells per allele on a single plate
            # we can get control_df with every possible allele on the same plate
            # As long as it is not a TC:
            df_control = dframe[dframe["Metadata_node_type"] != "TC"].reset_index(
                drop=True
            )
            log_file.write("Dropping low cell count wells in every possible allele that could be used as controls:\n")
            print("Dropping low cell count wells in every possible allele that could be used as controls:")
            # Filter out wells with fewer than the cell count threshold
            df_control = drop_low_cc_wells(df_control, cc_threshold, log_file)
            
            for feat in FEAT_TYPE_SET:
                print(feat)
                # GFP corrected version
                if feat == "GFP":
                    # Run experimental GFP filtering
                    df_feat_pro_exp_gfp_adj, df_result_pro_exp_gfp_adj, df_filtered_cells_gfp_adj = experimental_runner_plate_rep_gfp_filtered(
                        df_exp, pq_writer=writer_gfp,
                        err_logger=log_file, min_cells_per_well=cc_threshold
                    )
                    # Store to another set of feat_df, res_df and filtered_cells parquet
                    if (df_feat_pro_exp_gfp_adj.shape[0] > 0):
                        feat_import_gfp_adj_dfs += [df_feat_pro_exp_gfp_adj]
                        class_res_gfp_adj_dfs += [df_result_pro_exp_gfp_adj]
                        filtered_cells_gfp_adj_dfs += [df_filtered_cells_gfp_adj]

                    # Run control GFP filtering if output paths are provided
                    if writer_control_gfp is not None:
                        df_feat_pro_con_gfp, df_result_pro_con_gfp, df_filtered_cells_con_gfp = control_group_runner_fewer_rep_gfp_filtered(
                            df_control, pq_writer=writer_control_gfp,
                            err_logger=log_file, min_cells_per_well=cc_threshold
                        )
                        if (df_feat_pro_con_gfp.shape[0] > 0):
                            feat_import_control_gfp_dfs += [df_feat_pro_con_gfp]
                            class_res_control_gfp_dfs += [df_result_pro_con_gfp]
                            filtered_cells_control_gfp_dfs += [df_filtered_cells_con_gfp]

                    ## drop the gfp column during the inference
                    if GFP_INTENSITY_COLUMN not in dframe_featsel.columns:
                        df_exp = df_exp.drop(GFP_INTENSITY_COLUMN, axis=1)
                        df_control = df_control.drop(GFP_INTENSITY_COLUMN, axis=1)

                df_feat_pro_con, df_result_pro_con = control_group_runner_fewer_rep(
                    df_control, pq_writer=writer, err_logger=log_file, feat_type=feat
                )
                df_feat_pro_exp, df_result_pro_exp = experimental_runner_plate_rep(
                    df_exp, 
                    pq_writer=writer, 
                    err_logger=log_file, 
                    feat_type=feat
                )
                if (df_feat_pro_con.shape[0] > 0):
                    feat_import_dfs += [df_feat_pro_con, df_feat_pro_exp]
                    class_res_dfs += [df_result_pro_con, df_result_pro_exp]

        # Close the parquet writers
        writer.close()
        writer_gfp.close()
        if writer_control_gfp is not None:
            writer_control_gfp.close()

    # Handle GFP-adjusted results if available
    if feat_import_gfp_adj_dfs:
        df_feat_gfp_adj = pd.concat(feat_import_gfp_adj_dfs, ignore_index=True)
        df_result_gfp_adj = pd.concat(class_res_gfp_adj_dfs, ignore_index=True)
        df_filtered_cell = pd.concat(filtered_cells_gfp_adj_dfs, ignore_index=True)
        df_result_gfp_adj = df_result_gfp_adj.drop_duplicates()

        # Write out GFP-adjusted results
        df_feat_gfp_adj.to_csv(feat_output_path_gfp, index=False)
        df_result_gfp_adj.to_csv(info_output_path_gfp, index=False)
        df_filtered_cell.to_parquet(filtered_cell_path, index=False)

    # Handle control GFP-adjusted results if available
    if feat_import_control_gfp_dfs and feat_output_path_control_gfp is not None:
        df_feat_control_gfp = pd.concat(feat_import_control_gfp_dfs, ignore_index=True)
        df_result_control_gfp = pd.concat(class_res_control_gfp_dfs, ignore_index=True)
        df_filtered_cell_control = pd.concat(filtered_cells_control_gfp_dfs, ignore_index=True)
        df_result_control_gfp = df_result_control_gfp.drop_duplicates()

        # Write out control GFP-adjusted results
        df_feat_control_gfp.to_csv(feat_output_path_control_gfp, index=False)
        df_result_control_gfp.to_csv(info_output_path_control_gfp, index=False)
        df_filtered_cell_control.to_parquet(filtered_cell_path_control, index=False)
    # else:
    #     # Create empty files if no GFP-adjusted results
    #     pd.DataFrame().to_csv(feat_output_path_gfp, index=False)
    #     pd.DataFrame().to_csv(info_output_path_gfp, index=False)
    #     pd.DataFrame().to_parquet(filtered_cell_path, index=False)

    # Concatenate results for both standard and GFP-adjusted analyses
    df_feat = pd.concat(feat_import_dfs, ignore_index=True)
    df_result = pd.concat(class_res_dfs, ignore_index=True)
    df_result = df_result.drop_duplicates()

    # Write out standard feature importance and classifier info
    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(info_output_path, index=False)

    # CLEANUP POINT: Final cleanup before function exit
    try:
        # Clean up all remaining large dataframes
        del dframe, df_exp
        if 'df_control' in locals():
            del df_control
        if 'df_feat' in locals():
            del df_feat
        if 'df_result' in locals():
            del df_result
        if 'df_feat_gfp_adj' in locals():
            del df_feat_gfp_adj, df_result_gfp_adj, df_filtered_cell

        # Force final memory cleanup
        force_memory_cleanup(None)  # Pass None since log file is closed

        # Log final memory state to console
        swap_memory = psutil.swap_memory()
        print(f"Final swap usage: {swap_memory.used / 1024**3:.2f} GB ({swap_memory.percent}%)")

    except Exception as e:
        print(f"Error during final cleanup: {e}")