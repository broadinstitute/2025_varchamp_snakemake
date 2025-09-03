"""
    Optimized GFP-filtered Classification Pipeline
    Filters single cells by similar GFP intensity ranges for improved variant vs reference classification
"""

import os
import sys
import warnings
import argparse
from typing import List, Tuple, Optional
import time
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from classify import *
from analysis import *
from scipy.stats import ttest_rel, ttest_ind, shapiro, wilcoxon
sys.path.append("..")
# from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns
warnings.filterwarnings("ignore")

## constants
FEAT_TYPE_SET = set(["GFP"]) ## , "DNA", "AGP", "Mito", "TxControl", "Morph"
GFP_INTENSITY_COLUMN = "Cells_Intensity_IntegratedIntensity_GFP" ## Cells_Intensity_MeanIntensity_GFP is another option


## paired t-test to detect difference in cell count and gfp intensity
def paired_ttest(dat, reference: str, var: str, value: str, min_num_rep: int=3):
    # pivot to wide: one row per plate
    wide_gfp = dat.pivot(index="Metadata_Plate",
                        columns="Metadata_gene_allele",
                        values=value)
    # drop any plate that doesn’t have both measurements
    wide_gfp = wide_gfp.dropna(subset=[reference, var])
    if wide_gfp.shape[0] >= min_num_rep:
        # now run paired t-test
        t_stat, p_val = ttest_rel(wide_gfp[var].astype(float), wide_gfp[reference].astype(float))
    else:
        t_stat, p_val = None, None

    # Calculate Cohen's d
    mean_diff = np.mean(wide_gfp[var]) - np.mean(wide_gfp[reference])
    pooled_std = np.sqrt((np.std(wide_gfp[var], ddof=1) ** 2 + np.std(wide_gfp[reference], ddof=1) ** 2) / 2)
    cohen_d = mean_diff / pooled_std

    summary_df = pl.DataFrame(
        {
            "t_stat": t_stat,
            "p_val": p_val,
            "cohen_d": cohen_d
        }
    )
    summary_df = summary_df.with_columns(
        pl.lit(reference).alias("Gene"), pl.lit(var).alias("Variant")
    )
    return summary_df


# GFP range optimization function with expanded ranges and ratio constraint
def find_optimal_gfp_range_fast(ref_gfp: np.ndarray, var_gfp: np.ndarray, 
                                quantile_pair: tuple=(0.25, 0.75),
                                min_cells_per_well: int = 20):
    """Ultra-fast vectorized GFP range optimization with single quantile pair"""
    # Expanded quantile range testing: from 10%-90% down to 30%-70%
    # quantile_pairs = [
    #     (0.2, 0.8), (0.22, 0.78), (0.25, 0.75), (0.27, 0.73), (0.3, 0.7)
    # ] ## (0.1, 0.9), (0.12, 0.88), (0.15, 0.85), (0.17, 0.83), 
    
    best_range = None
    max_total_cells = 0
    best_quantile_info = ""
    results = []
    
    # Vectorized quantile calculation for all ranges
    # all_quantiles = [q for pair in quantile_pairs for q in pair]
    # ref_qs = np.quantile(ref_gfp, all_quantiles)
    # var_qs = np.quantile(var_gfp, all_quantiles)
    # Test each quantile pair
    # for i, (low_q, high_q) in enumerate(quantile_pairs):
    # ref_low = ref_qs[i*2]
    # ref_high = ref_qs[i*2 + 1] 
    # var_low = var_qs[i*2]
    # var_high = var_qs[i*2 + 1]

    low_q, high_q = quantile_pair
    # Calculate quantiles directly for the single pair
    ref_low = np.quantile(ref_gfp, low_q)
    ref_high = np.quantile(ref_gfp, high_q)
    var_low = np.quantile(var_gfp, low_q)
    var_high = np.quantile(var_gfp, high_q)
    
    # Find overlapping range
    range_min = max(ref_low, var_low)
    range_max = min(ref_high, var_high)
    
    # Skip if invalid range
    if range_min >= range_max:
        results.append((f"{int(low_q*100)}-{int(high_q*100)}%", 0, 0, 0, "Invalid range", "N/A"))
        return None, None, 0, 0, "NO_SUITABLE_RANGE"
    
    # Vectorized cell counting
    ref_mask = (ref_gfp >= range_min) & (ref_gfp <= range_max)
    var_mask = (var_gfp >= range_min) & (var_gfp <= range_max)
    ref_count = np.sum(ref_mask)
    var_count = np.sum(var_mask)
    
    # Calculate sample size ratio
    if ref_count == 0 or var_count == 0:
        ratio_status = "Zero samples"
    else:
        ratio_status = f"{max(ref_count, var_count) / min(ref_count, var_count)}:.2f"
    
    results.append((f"{int(low_q*100)}-{int(high_q*100)}%", ref_count, var_count, 
                    ref_count + var_count, f"GFP: {range_min:.1f}-{range_max:.1f}", ratio_status))
    
    # Check minimum requirements and ratio constraint
    if (ref_count >= min_cells_per_well and var_count >= min_cells_per_well and
        ref_count > 0 and var_count > 0):
        total_cells = ref_count + var_count
        if total_cells > max_total_cells:
            max_total_cells = total_cells
            best_range = (range_min, range_max, ref_count, var_count)
            best_quantile_info = f"{int(low_q*100)}%-{int(high_q*100)}%"
    
    # Show all results
    # print("GFP Range Optimization Results (with ratio constraint ≤3x):")
    # print("Quantile Range | Ref Cells | Var Cells | Total | GFP Range      | Sample Ratio")
    # print("-" * 90)
    # for quantile, ref_c, var_c, total, gfp_range, ratio_info in results:
    #     # Check if this meets all criteria
    #     meets_min = ref_c >= min_cells_per_well and var_c >= min_cells_per_well
    #     meets_ratio = "Ratio:" in ratio_info and not "(>3x)" in ratio_info if ref_c > 0 and var_c > 0 else False
    #     status = "✅" if (meets_min and meets_ratio) else "❌"
    #     print(f"{status} {quantile:>12} | {ref_c:>8} | {var_c:>8} | {total:>5} | {gfp_range:>14} | {ratio_info}")

    if best_range is not None:
        return best_range[0], best_range[1], best_range[2], best_range[3], best_quantile_info
    else:
        return None, None, 0, 0, "NO_SUITABLE_RANGE"


"""
    Set up classification workflow with plate_layout of single replicate (single_rep) per plate
"""
def experimental_runner_filter_gfp(
    exp_dframe: pd.DataFrame,
    pq_writer,
    log_file,
    feat_type,
    allele_list,
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type",
    min_cells_per_well=20
):
    """
    Run Reference v.s. Variant experiments
    """
    exp_dframe = get_classifier_features(exp_dframe, feat_type)
    feat_cols = find_feat_cols(exp_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]

    if len(feat_cols) == 0:
        return pd.DataFrame(), pd.DataFrame()

    group_list = []
    pair_list = []
    feat_list = []
    info_list = []
    filtered_cell_list = []

    allele_ref_gene = list(set([allele.split("_")[0] for allele in allele_list]))
    exp_dframe = exp_dframe[exp_dframe["Metadata_gene_allele"].isin(allele_list+allele_ref_gene)]

    log_file.write(f"Running XGBboost classifiers w/ {feat_type} features on target variants:\n")
    ## First group the df by reference genes
    groups = exp_dframe.groupby(group_key_one).groups
    for key in tqdm(groups.keys()):
        dframe_grouped = exp_dframe.loc[groups[key]].reset_index(drop=True)
        # Ensure this gene has both reference and variants
        if dframe_grouped[threshold_key].unique().size < 2:
            continue
        df_group_one = dframe_grouped[
            dframe_grouped[threshold_key] != "allele" #== "disease_wt", sometimes the reference is misannotated (e.g. KRAS)
        ].reset_index(drop=True)
        df_group_one["Label"] = 1

        ## Then group the gene-specific df by different variant alleles
        subgroups = (
            dframe_grouped[dframe_grouped[threshold_key] == "allele"]
            .groupby(group_key_two)
            .groups
        )
        for subkey in subgroups.keys():
            df_group_two = dframe_grouped.loc[subgroups[subkey]].reset_index(drop=True)
            df_group_two["Label"] = 0
            plate_list = get_common_plates(df_group_one, df_group_two)

            ## Get ALL the wells for reference gene and variant alleles and pair up all possible combinations
            ref_wells = df_group_one["Metadata_well_position"].unique()
            var_wells = list(df_group_two["Metadata_well_position"].unique())
            ref_var_pairs = [(ref_well, var_well) for ref_well in ref_wells for var_well in var_wells]
            df_sampled_ = pd.concat([df_group_one, df_group_two], ignore_index=True)
            ## Per each ref-var well pair on the SAME plate, train and test the classifier
            for ref_var in ref_var_pairs:
                df_sampled = df_sampled_[df_sampled_["Metadata_well_position"].isin(ref_var)]
                ## Filter cells by GFP intensity ranges
                log_file.write(f"{key}, {subkey}, {ref_var}, Orig GFP paired t-test:")
                df_sampled_well_agg = pl.DataFrame(
                    df_sampled
                ).group_by(
                    ["Metadata_Plate", "Metadata_Well", "Metadata_gene_allele"]
                ).agg(
                    pl.col(col).median().alias(col)
                    for col in df_sampled.columns
                    if not col.startswith("Metadata_")
                ).unique()
                log_file.write(str(paired_ttest(
                    df_sampled_well_agg.to_pandas(), 
                    key, subkey, GFP_INTENSITY_COLUMN
                )))
                
                ## get the optimal gfp range for paired ref-var wells on each plate
                df_sampled_filtered = pd.DataFrame()
                # gfp_filtered_results = []
                for plate in plate_list:
                    df_plate = df_sampled[df_sampled["Metadata_Plate"] == plate]
                    ref_gfp = df_plate[
                        (df_plate["Label"] == 1) & (df_plate["Metadata_gene_allele"] == key)
                    ][GFP_INTENSITY_COLUMN].to_numpy()
                    var_gfp = df_plate[
                        (df_plate["Label"] == 0) & (df_plate["Metadata_gene_allele"] == subkey)
                    ][GFP_INTENSITY_COLUMN].to_numpy()
                    
                    gfp_low, gfp_high, ref_count, var_count, quantile_info = find_optimal_gfp_range_fast(
                        ref_gfp, var_gfp, quantile_pair=(0.25, 0.75), min_cells_per_well=min_cells_per_well
                    )
                    if gfp_low is not None:
                        # Filter cells within the optimal GFP range
                        df_plate_filtered = df_plate[
                            (df_plate[GFP_INTENSITY_COLUMN] >= gfp_low) & 
                            (df_plate[GFP_INTENSITY_COLUMN] <= gfp_high)
                        ].reset_index(drop=True)
                        df_plate_filtered_ref = df_plate_filtered[df_plate_filtered["Label"] == 1]
                        df_plate_filtered_var = df_plate_filtered[df_plate_filtered["Label"] == 0]
                        ## subsample the larger group to maintain a ratio <= 3
                        if max(df_plate_filtered_var.shape[0], df_plate_filtered_ref.shape[0]) / min(df_plate_filtered_var.shape[0], df_plate_filtered_ref.shape[0]) > 3:
                            if df_plate_filtered_var.shape[0] > df_plate_filtered_ref.shape[0]:
                                df_plate_filtered_var = df_plate_filtered_var.sample(
                                    n = df_plate_filtered_ref.shape[0] * 3 - 1,
                                    random_state=42
                                )
                            else:
                                df_plate_filtered_ref = df_plate_filtered_ref.sample(
                                    n = df_plate_filtered_var.shape[0] * 3 - 1,
                                    random_state=42
                                )
                        ## merge back the filtered ref and var dataframes
                        df_plate_filtered = pd.concat([
                            df_plate_filtered_ref, df_plate_filtered_var
                        ], ignore_index=True)
                        # fig, axes = plt.subplots(1,2,figsize=(10,4))
                        # sns.boxenplot(
                        #     x="Metadata_gene_allele",
                        #     y=GFP_INTENSITY_COLUMN,
                        #     data=df_plate,
                        #     ax=axes[0]
                        # )
                        # axes[0].set_title(f"{plate} Original GFP Distribution\n{ref_var[0]} vs {ref_var[1]}\nN={df_plate.shape[0]}")
                        # sns.boxenplot(
                        #     x="Metadata_gene_allele",
                        #     y=GFP_INTENSITY_COLUMN,
                        #     data=df_plate_filtered,
                        #     ax=axes[1]
                        # )
                        # axes[1].set_title(f"{plate} Filtered GFP Distribution\n{ref_var[0]} vs {ref_var[1]}\nN={df_plate_filtered.shape[0]}\nGFP: {gfp_low:.1f}-{gfp_high:.1f} ({quantile_info})")
                        # plt.savefig(f"{subkey}_{plate}_{ref_var[0]}_vs_{ref_var[1]}_gfp_filtered.png", dpi=150)

                        # Update df_sampled with filtered plate data
                        df_sampled_filtered = pd.concat([
                            df_sampled_filtered,
                            df_plate_filtered
                        ], ignore_index=True)
                        log_file.write(f"{key}, {subkey}, {ref_var}, {plate}, GFP range: {gfp_low:.2f}-{gfp_high:.2f}, Ref # cells: {ref_count}, Var # cells: {var_count}, Quantile: {quantile_info}, Status: SUCCESS\n")
                    else:
                        # gfp_filtered_results.append({
                        #     'pair_id': f"{plate}_{ref_var[0]}_vs_{ref_var[1]}",
                        #     'plate': plate,
                        #     'ref_well': ref_var[0],
                        #     'var_well': ref_var[1],
                        #     'gfp_min': None,
                        #     'gfp_max': None,
                        #     'quantile_range': 'FAILED',
                        #     'status': 'NO_SUITABLE_RANGE'
                        # })
                        log_file.write(f"{key}, {subkey}, {ref_var}, {plate}, GFP range: None-None, Ref # cells: None, Var # cells: None, Quantile: FAILED, Status: NO_SUITABLE_RANGE\n")

                if df_sampled_filtered.shape[0] == 0:
                    log_file.write(f"{key}, {subkey}, {ref_var}, Failed to correct for GFP on ANY PLATE and WELL pair. Skipping...\n")
                    continue

                log_file.write(f"{key}, {subkey}, {ref_var}, Corrected GFP paired t-test:")
                df_sampled_filterd_well_agg = pl.DataFrame(
                    df_sampled_filtered
                ).group_by(
                    ["Metadata_Plate", "Metadata_Well", "Metadata_gene_allele"]
                ).agg(
                    pl.col(col).median().alias(col)
                    for col in df_sampled.columns
                    if not col.startswith("Metadata_")
                ).unique()
                log_file.write(str(paired_ttest(
                    df_sampled_filterd_well_agg.to_pandas(), 
                    key, subkey, GFP_INTENSITY_COLUMN
                )))

                ## store the ref-vs-var tags for this filtered df
                df_sampled_filtered["Metadata_refvar_classify"] = f"{key}_{subkey}_{ref_var[0]}-{ref_var[1]}"
                ## stored the filtered cell list per this well pair across plates
                filtered_cell_list.append(df_sampled_filtered)

                # fig, axes = plt.subplots(1,2,figsize=(10,4))
                # sns.boxenplot(
                #     x="Metadata_gene_allele",
                #     y=GFP_INTENSITY_COLUMN,
                #     data=df_sampled_,
                #     ax=axes[0]
                # )
                # axes[0].set_title(f"{plate} Original GFP\n{ref_var[0]} vs {ref_var[1]}\nN={df_plate.shape[0]}")
                # sns.boxenplot(
                #     x="Metadata_gene_allele",
                #     y=GFP_INTENSITY_COLUMN,
                #     data=df_sampled_filtered,
                #     ax=axes[1]
                # )
                # axes[1].set_title(f"{plate} Filtered GFP\n{ref_var[0]} vs {ref_var[1]}\nN={df_plate_filtered.shape[0]}\nGFP: {gfp_low:.1f}-{gfp_high:.1f} ({quantile_info})")
                # plt.savefig(f"/home/shenrunx/igvf/varchamp/2025_Fang_CCM2_Imaging/data/interim/large_files/{subkey}_{ref_var[0]}_vs_{ref_var[1]}_gfp_filtered.png", dpi=150)

                ## during the inference, drop the gfp column
                df_sampled_filtered = df_sampled_filtered.drop(GFP_INTENSITY_COLUMN, axis=1)
                ## Define the func. for thread_map the plate on the same df_sampled
                def classify_by_plate_helper(plate):
                    df_train, df_test = stratify_by_plate(df_sampled_filtered, plate)
                    feat_importances, classifier_info, predictions = classifier(
                        df_train, df_test, log_file
                    )
                    return {plate: [feat_importances, classifier_info, predictions]}
                try:
                    result = thread_map(classify_by_plate_helper, plate_list)
                    pred_list = []
                    for res in result:
                        if list(res.values())[-1] is not None:
                            feat_list.append(list(res.values())[0][0])
                            group_list.append(key)
                            pair_list.append(f"{key}_{subkey}")
                            info_list.append(list(res.values())[0][1])
                            pred_list.append(list(res.values())[0][2])
                        else:
                            log_file.write(f"Skipped classification result for {key}_{subkey}\n")
                            print(f"Skipping classification result for {key}_{subkey}...")
                            feat_list.append([None] * len(feat_cols))
                            group_list.append(key)
                            pair_list.append(f"{key}_{subkey}")
                            info_list.append([None] * 10)

                    cell_preds = pd.concat(pred_list, axis=0)
                    cell_preds["Metadata_Feature_Type"] = feat_type
                    cell_preds["Metadata_Control"] = False
                    table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                    pq_writer.write_table(table)
                except Exception as e:
                    print(e)
                    log_file.write(f"{key}, {subkey} error: {e}\n")


    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Feature_Type"] = feat_type
    df_feat["Metadata_Control"] = False

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = False
    df_result["Metadata_Feature_Type"] = feat_type

    df_filtered_cells = pd.concat(filtered_cell_list, ignore_index=True)

    log_file.write(f"Finished running XGBboost classifiers w/ {feat_type} features on target variants.\n")
    log_file.write(f"===========================================================================\n\n")
    return df_feat, df_result, df_filtered_cells


def run_gfp_filtered_classification(
    allele_list: List[str],
    input_path: str,
    input_path_orig: str,
    feat_output_path: str,
    info_output_path: str,
    preds_output_path: str,
    filtered_cell_path: str,
    cc_threshold: int,
    use_gpu: Union[str, None] = "0,1",
):
    """
    Run workflow for single-cell classification
    """
    if use_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

    # Initialize parquet for cell-level predictions
    os.makedirs('/'.join(preds_output_path.split("/")[:-1]), exist_ok=True)

    # if os.path.exists(preds_output_path):
    #     os.remove(preds_output_path)
    
    ## create a log file
    logfile_path = os.path.join('/'.join(preds_output_path.split("/")[:-1]), "classify_filtered_gfp.log")

    ## output the prediction.parquet
    schema = pa.schema([
        ("Classifier_ID", pa.string()),
        ("CellID", pa.string()),
        ("Label", pa.int64()),
        ("Prediction", pa.float32()),
        ("Metadata_Feature_Type", pa.string()),
        ("Metadata_Control", pa.bool_()),
    ])
    writer = pq.ParquetWriter(preds_output_path, schema, compression="gzip")

    # Add CellID column
    dframe = (
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
        # .collect()
        # .to_pandas()
    )

    if GFP_INTENSITY_COLUMN not in dframe.columns:
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
            # .collect()
            # .to_pandas()
        )
        dframe = dframe.join(
            dframe_orig, on="Metadata_CellID", how="left"
        ).collect().to_pandas()
    else:
        dframe = dframe.collect().to_pandas()

    ## select the feature columns
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

    ## Filter rows with NaN Metadata
    dframe = dframe[~dframe["Metadata_well_position"].isna()]
    dframe = add_control_annot(dframe)
    dframe = dframe[~dframe["Metadata_control"].isna()]

    ## store the classifier feat_importance and classification_res
    feat_import_dfs, class_res_dfs, filtered_cells_dfs = [], [], []
    ## Split data into experimental df with var and ref alleles
    df_exp = dframe[~dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
    with open(logfile_path, "w") as log_file:
        log_file.write(f"===============================================================================================================================================================\n")
        log_file.write("Dropping low cell count wells in ref. vs variant alleles:\n")
        print("Dropping low cell count wells in ref. vs variant alleles:")
        df_exp = drop_low_cc_wells(df_exp, cc_threshold, log_file)
        log_file.write(f"===============================================================================================================================================================\n\n")
        # Check the plate_layout for the correct classification set-up

        for feat in FEAT_TYPE_SET:
            df_feat_pro_exp, df_result_pro_exp, df_filtered_cells = experimental_runner_filter_gfp(
                df_exp, pq_writer=writer, log_file=log_file, feat_type=feat, allele_list=allele_list,
                min_cells_per_well=cc_threshold
            )
            if (df_feat_pro_exp.shape[0] > 0):
                feat_import_dfs += [df_feat_pro_exp]
                class_res_dfs += [df_result_pro_exp]
                filtered_cells_dfs += [df_filtered_cells]

        ## Close the parquest writer
        writer.close()

    # Concatenate results for both protein and non-protein
    df_feat = pd.concat(
        feat_import_dfs, ignore_index=True
    )
    df_result = pd.concat(
        class_res_dfs, ignore_index=True
    )
    df_filtered_cell = pd.concat(
        filtered_cells_dfs, ignore_index=True
    )
    df_result = df_result.drop_duplicates()

    # Write out feature importance and classifier info
    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(info_output_path, index=False)
    df_filtered_cell.to_parquet(filtered_cell_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="GFP-filtered classification for variant analysis")
    parser.add_argument("--allele_file", required=True, help="Allele list file (one allele per line)")
    parser.add_argument("--batch", required=True, help="Comma-separated batch names (e.g., 2025_06_10_Batch_18,2025_06_10_Batch_19)")
    parser.add_argument("--input_dir", required=True, help="Directory containing batch profile parquet files")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--cc_threshold", type=int, default=20, help="Minimum cells per well after filtering")
    
    args = parser.parse_args()
    allele_list = pd.read_csv(args.allele_file, header=None)[0].tolist()

    #[allele.strip() for allele in args.alleles.split(",")]
    # Parse batch names and construct input paths
    batch = args.batch    
    input_path = os.path.join(args.input_dir, batch, "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells.parquet")
    input_path_orig = os.path.join(args.input_dir, batch, "profiles_tcdropped_filtered_var_mad_outlier.parquet")
    
    feat_output_path = os.path.join(args.output_dir, batch, "feat_importance.csv")
    preds_output_path = os.path.join(args.output_dir, batch, "predictions.parquet")
    info_output_path = os.path.join(args.output_dir, batch, "classifier_info.csv")
    filtered_cell_path = os.path.join(args.output_dir, batch, "filtered_gfp_cells.parquet")

    ## Run classification
    run_gfp_filtered_classification(
        allele_list=allele_list,
        input_path=input_path,
        input_path_orig=input_path_orig,
        feat_output_path=feat_output_path,
        info_output_path=info_output_path,
        preds_output_path=preds_output_path,
        filtered_cell_path=filtered_cell_path,
        cc_threshold=args.cc_threshold    
    )

    calculate_class_metrics(
        classifier_info=info_output_path,
        predictions=preds_output_path,
        metrics_file=os.path.join(args.output_dir, batch, "classification_metrics.csv"),
    )


if __name__ == "__main__":
    main()