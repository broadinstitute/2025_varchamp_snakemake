"""
GFP-Intensity Corrected Classification Module for VarChAMP Pipeline

This module provides advanced classification functionality that corrects for GFP intensity
differences between reference and variant alleles in cell painting experiments. The core
functionality addresses a critical confounding factor where genetic variants can affect
GFP expression levels, potentially biasing morphological classification results.

## Purpose and Motivation

In VarChAMP (Variant Classification via Cell Painting) experiments, genetic variants are
often introduced with GFP reporters. However, some variants can affect GFP expression
levels independently of their morphological effects. This creates a confounding variable
where classifiers might distinguish variants based on GFP intensity rather than true
morphological differences.

This module implements GFP-intensity correction by:
1. Analyzing GFP intensity distributions between reference and variant cells
2. Finding optimal overlapping GFP intensity ranges using quantile-based methods
3. Filtering cells to matched GFP intensity ranges before classification
4. Performing statistical validation with paired t-tests and effect size calculations

## Key Features

### Statistical Analysis
- Paired t-test comparisons between reference and variant GFP intensities
- Cohen's d effect size calculations for magnitude assessment  
- Before/after correction statistical validation

### Adaptive GFP Range Optimization
- Quantile-based range detection (25%-75% to 10%-90% fallback strategy)
- Vectorized optimization for computational efficiency
- Sample size balancing with 3:1 ratio constraints
- Minimum cell count requirements per well

### Classification Integration
- Seamless integration with existing XGBoost classification pipeline
- Parallel processing support for plate-wise analysis
- Comprehensive logging and error handling
- Output of filtered cell profiles for downstream analysis

## Workflow

1. **Pre-filtering Analysis**: Calculate paired t-tests on original GFP distributions
2. **Range Optimization**: Find optimal overlapping GFP intensity ranges per plate
3. **Cell Filtering**: Filter cells within matched GFP ranges
4. **Sample Balancing**: Subsample to maintain reasonable group size ratios
5. **Classification**: Run standard classification pipeline on filtered cells
6. **Post-filtering Validation**: Re-calculate statistics on corrected data
7. **Output Generation**: Export feature importance, metrics, and filtered cell profiles

## Constants and Configuration

- GFP_INTENSITY_COLUMN: Primary GFP intensity measurement column
- Quantile strategies: [(0.25, 0.75), (0.2, 0.8), (0.15, 0.85), (0.1, 0.9)]
- Default minimum cells per well: 20
- Maximum sample ratio constraint: 3:1

## Usage Context

This module is specifically designed for single-replicate-per-plate experimental layouts
where reference and variant alleles are tested in paired wells across multiple plates.
It requires GFP intensity measurements and is most effective when GFP intensity
differences are observed between reference and variant conditions.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from scipy.stats import ttest_rel, ttest_ind
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns
from .classify_helper_func import get_classifier_features, classifier
from .classify_single_rep_per_plate import get_common_plates, stratify_by_plate

# Constants
GFP_INTENSITY_COLUMN = "Cells_Intensity_IntegratedIntensity_GFP"  # Cells_Intensity_MeanIntensity_GFP is another option

#######################################
# GFP-INTENSITY CORRECTED CLASSIFICATION
# Statistical analysis and range optimization
#######################################
## paired t-test to detect difference in cell count and gfp intensity
def paired_ttest(dat, reference: str, var: str, value: str, min_num_rep: int=3):
    # pivot to wide: one row per plate
    wide_gfp = dat.pivot(index="Metadata_Plate",
                        columns="Metadata_gene_allele",
                        values=value)
    
    # Check if required columns exist
    if reference not in wide_gfp.columns or var not in wide_gfp.columns:
        t_stat, p_val = None, None
        mean_diff = np.nan
        pooled_std = np.nan
        cohen_d = np.nan
    else:
        # drop any plate that doesn't have both measurements
        wide_gfp = wide_gfp.dropna(subset=[reference, var])
        if wide_gfp.shape[0] >= min_num_rep:
            # now run paired t-test
            t_stat, p_val = ttest_rel(wide_gfp[var].astype(float), wide_gfp[reference].astype(float))
            # Calculate Cohen's d
            mean_diff = np.mean(wide_gfp[var]) - np.mean(wide_gfp[reference])
            pooled_std = np.sqrt((np.std(wide_gfp[var], ddof=1) ** 2 + np.std(wide_gfp[reference], ddof=1) ** 2) / 2)
            cohen_d = mean_diff / pooled_std
        else:
            t_stat, p_val = None, None
            mean_diff = np.nan
            pooled_std = np.nan
            cohen_d = np.nan

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


def ind_ttest(dat, reference: str, var: str, value: str, min_num_rep: int=3):
    ## Per each ref-var well pair on the SAME plate, train and test the classifier
    ## sort the wells to make sure they are from the same plate
    df_sampled = pd.DataFrame()
    for plate in dat["Metadata_Plate"].unique():
        dat = dat[dat["Metadata_Plate"]==plate].dropna().sort_values(["Metadata_gene_allele"])
        # count rows per group
        group_counts = dat.groupby("Metadata_gene_allele").size()
        min_count = group_counts.min()
        # print("Minimum rows in any group:", min_count)
        shuffled = dat.sample(frac=1, random_state=42).reset_index(drop=True)
        # Then take the first min_count rows per group
        sampled_df2 = (
            shuffled
            .groupby("Metadata_gene_allele", group_keys=False)
            .head(min_count)
        )
        df_sampled = pd.concat([df_sampled, sampled_df2])

    # m0 = smf.ols("Cell_count ~ Metadata_Well", data=df_sampled).fit()
    # df_sampled["resid"] = m0.resid
    # print(df_sampled)
    ## require at least two alleles per each VAR and WT group
    if df_sampled.shape[0] >= min_num_rep * 2:
        # now run paired t-test
        t_stat, p_val = ttest_ind(
            df_sampled.loc[df_sampled["Metadata_gene_allele"]==var, value].astype(int).values,
            df_sampled.loc[df_sampled["Metadata_gene_allele"]==reference, value].astype(int).values, 
            equal_var=False
        )
    else:
        t_stat, p_val = None, None
        
    # break
    summary_df = pl.DataFrame(
        {
            "t_stat": t_stat,
            "p_val": p_val,
            # "t_stat_res": res_t_stat,
            # "p_val_res": res_p_val
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
    # Check if arrays are empty
    if len(ref_gfp) == 0 or len(var_gfp) == 0:
        return None, None, 0, 0, "EMPTY_ARRAYS"
    
    # Expanded quantile range testing: from 10%-90% down to 30%-70%
    # quantile_pairs = [
    #     (0.2, 0.8), (0.22, 0.78), (0.25, 0.75), (0.27, 0.73), (0.3, 0.7)
    # ] ## (0.1, 0.9), (0.12, 0.88), (0.15, 0.85), (0.17, 0.83)
    
    best_range = None
    max_total_cells = 0
    best_quantile_info = ""
    results = []

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
    min_cells_per_well,
    feat_type="GFP",
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type"
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

    # allele_ref_gene = list(set([allele.split("_")[0] for allele in allele_list]))
    # exp_dframe = exp_dframe[exp_dframe["Metadata_gene_allele"].isin(allele_list+allele_ref_gene)]

    log_file.write(f"Running XGBboost classifiers w/ {feat_type} features on target variants corrected by GFP:\n")
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

                    ## by default: quantile_pair=(0.25, 0.75), overlapped interquartile range
                    quantile_pairs = [(0.25, 0.75), (0.2, 0.8), (0.15, 0.85), (0.1, 0.9)]
                    for quantile_pair in quantile_pairs:
                        # print(quantile_pair)
                        gfp_low, gfp_high, ref_count, var_count, quantile_info = find_optimal_gfp_range_fast(
                            ref_gfp, var_gfp, quantile_pair=quantile_pair, min_cells_per_well=min_cells_per_well
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
                            df_plate_filtered["Metadata_refvar_gfp_adj_classify"] = f"{key}_{subkey}_{ref_var[0]}-{ref_var[1]}_q{quantile_pair[0]}-{quantile_pair[1]}_plate"
                            # Update df_sampled with filtered plate data
                            df_sampled_filtered = pd.concat([
                                df_sampled_filtered,
                                df_plate_filtered
                            ], ignore_index=True)
                            log_file.write(f"{key}, {subkey}, {ref_var}, {plate}, GFP range: {gfp_low:.2f}-{gfp_high:.2f}, Ref # cells: {ref_count}, Var # cells: {var_count}, Quantile: {quantile_info}, Status: SUCCESS\n")
                            break

                        if gfp_low is None and quantile_pair==(0.1, 0.9):
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

    log_file.write(f"Finished running XGBboost classifiers w/ {feat_type} features on target variants corrected by GFP.\n")
    log_file.write(f"===========================================================================\n\n")
    return df_feat, df_result, df_filtered_cells


"""
    Set up classification workflow with plate_layout of multiple replicates (multi_rep) per plate
"""
def stratify_by_well_pair_exp_gfp_filtered(df_sampled: pd.DataFrame, well_pair_list: list, key: str, subkey: str, min_cells_per_well: int, log_file):
    """
        Stratify dframe by plate
        df_sampled: the data frame containing both ref. and var. alleles, each tested in 4 wells
        well_pair: a list of well pairs containing a ref. and a var. allele, with 1st pair for test and the rest pairs for training
    """
    ## Per each ref-var well pair on the SAME plate, train and test the classifier
    ## get the optimal gfp range for every paired ref-var wells on the plate
    df_sampled_well_agg = pl.DataFrame(
        df_sampled
    ).group_by(
        ["Metadata_Plate", "Metadata_Well", "Metadata_gene_allele"]
    ).agg(
        pl.col(GFP_INTENSITY_COLUMN).median().alias(GFP_INTENSITY_COLUMN)
    ).unique()
    log_file.write(str(ind_ttest(
        df_sampled_well_agg.to_pandas(), 
        key, subkey, GFP_INTENSITY_COLUMN
    )))
    
    df_sampled_filtered = pd.DataFrame()
    # print(well_pair_list)
    for ref_var in well_pair_list:
        print(ref_var)
        df_plate = df_sampled[df_sampled["Metadata_well_position"].isin(ref_var)]
        ## Filter cells by GFP intensity ranges
        log_file.write(f"{key}, {subkey}, {ref_var}, Orig GFP paired t-test:")
        
        ref_gfp = df_plate[
            (df_plate["Label"] == 1) & (df_plate["Metadata_gene_allele"] == key)
        ][GFP_INTENSITY_COLUMN].to_numpy()
        var_gfp = df_plate[
            (df_plate["Label"] == 0) & (df_plate["Metadata_gene_allele"] == subkey)
        ][GFP_INTENSITY_COLUMN].to_numpy()

        ## by default: quantile_pair=(0.25, 0.75), overlapped interquartile range
        quantile_pairs = [(0.25, 0.75), (0.2, 0.8), (0.15, 0.85), (0.1, 0.9)]
        for quantile_pair in quantile_pairs:
            log_file.write(f"q{quantile_pair[0]}-{quantile_pair[1]}")
            gfp_low, gfp_high, ref_count, var_count, quantile_info = find_optimal_gfp_range_fast(
                ref_gfp, var_gfp, quantile_pair=quantile_pair, min_cells_per_well=min_cells_per_well
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
                df_plate_filtered["Metadata_refvar_gfp_adj_classify"] = f"{key}_{subkey}_{ref_var[0]}-{ref_var[1]}_q{quantile_pair[0]}-{quantile_pair[1]}_plate"

                # Update df_sampled with filtered plate data
                df_sampled_filtered = pd.concat([
                    df_sampled_filtered,
                    df_plate_filtered
                ], ignore_index=True)
                log_file.write(f"{key}, {subkey}, {ref_var}, GFP range: {gfp_low:.2f}-{gfp_high:.2f}, Ref # cells: {ref_count}, Var # cells: {var_count}, Quantile: {quantile_info}, Status: SUCCESS\n")
                break

            if gfp_low is None and quantile_pair==(0.1, 0.9):
                log_file.write(f"{key}, {subkey}, {ref_var}, GFP range: None-None, Ref # cells: None, Var # cells: None, Quantile: FAILED, Status: NO_SUITABLE_RANGE\n")                        

    if df_sampled_filtered.shape[0] == 0:
        log_file.write(f"{key}, {subkey}, {ref_var}, Failed to correct for GFP on ANY PLATE and WELL pair. Skipping...\n")

    log_file.write(f"{key}, {subkey}, {ref_var}, Corrected GFP paired t-test:")
    df_sampled_filterd_well_agg = pl.DataFrame(
        df_sampled_filtered
    ).group_by(
        ["Metadata_Plate", "Metadata_Well", "Metadata_gene_allele"]
    ).agg(
        pl.col(GFP_INTENSITY_COLUMN).median().alias(GFP_INTENSITY_COLUMN)
    ).unique()
    log_file.write(str(ind_ttest(
        df_sampled_filterd_well_agg.to_pandas(), 
        key, subkey, GFP_INTENSITY_COLUMN
    )))

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
    # df_sampled_filtered = df_sampled_filtered.drop(GFP_INTENSITY_COLUMN, axis=1)
    
    ## Training on the rest three wells out of four
    df_train = df_sampled_filtered[
        (df_sampled_filtered["Metadata_well_position"].isin([well for pair in well_pair_list[1:] for well in pair]))
    ].reset_index(drop=True)
    ## Testing on the well_pair
    df_test = df_sampled_filtered[
        df_sampled_filtered["Metadata_well_position"].isin(well_pair_list[0])
    ].reset_index(drop=True)
    return df_train, df_test


def experimental_runner_plate_rep_gfp_filtered(
    exp_dframe: pd.DataFrame,
    pq_writer,
    err_logger,
    min_cells_per_well,
    feat_type="GFP",
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type",
):
    """
    Run Reference v.s. Variant experiments run on the same plate without tech. dups

    # df_feat_pro_exp, df_result_pro_exp = experimental_runner_plate_rep(df_exp, pq_writer=writer, err_logger=err_logger, feat_type=feat_type)
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

    err_logger.write(f"Logging errors when running real experiments w/ {feat_type} features:\n")
    groups = exp_dframe.groupby(group_key_one).groups
    for key in tqdm(groups.keys()):
        dframe_grouped = exp_dframe.loc[groups[key]].reset_index(drop=True)

        # Ensure this gene has both reference and variants
        if dframe_grouped[threshold_key].unique().size < 2:
            continue

        df_group_one = dframe_grouped[
            dframe_grouped["Metadata_node_type"] != "allele" #== "disease_wt", sometimes the reference is misannotated (e.g. KRAS)
        ].reset_index(drop=True)
        df_group_one["Label"] = 1
        ref_al_wells = df_group_one["Metadata_well_position"].unique()
        
        subgroups = (
            dframe_grouped[dframe_grouped["Metadata_node_type"] == "allele"]
            .groupby(group_key_two)
            .groups
        )

        for subkey in subgroups.keys():
            df_group_two = dframe_grouped.loc[subgroups[subkey]].reset_index(drop=True)
            df_group_two["Label"] = 0
            var_al_wells = df_group_two["Metadata_well_position"].unique()
            df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)

            if len(ref_al_wells) < 4:
                # ref_al_wells = np.random.choice(ref_al_wells, size=4)
                err_logger.write(f"{key}, {subkey} pair DOES NOT enough ref. alleles! Ref. allele wells in parquet: {ref_al_wells}\n")
                continue
            if len(var_al_wells) < 4:
                # var_al_wells = np.random.choice(var_al_wells, size=4)
                err_logger.write(f"{key}, {subkey} pair DOES NOT enough var. alleles! Var. allele wells in parquet: {var_al_wells}\n")
                continue
                
            well_pair_list = list(zip(ref_al_wells, var_al_wells))
            well_pair_nested_list = [[well_pair_list[i]] + well_pair_list[:i] + well_pair_list[i+1:] for i in range(len(well_pair_list))]
            ## try run classifier
            try:
                def classify_by_well_pair_exp_helper(df_sampled: pd.DataFrame, well_pair_list: list, log_file=err_logger):
                    """Helper func to run classifiers in parallel for var-ref alleles"""
                    df_train, df_test = stratify_by_well_pair_exp_gfp_filtered(df_sampled, well_pair_list, key, subkey, min_cells_per_well, log_file)
                    ## drop the gfp column during the inference
                    feat_importances, classifier_info, predictions = classifier(
                        df_train.drop(GFP_INTENSITY_COLUMN, axis=1), 
                        df_test.drop(GFP_INTENSITY_COLUMN, axis=1), 
                        log_file
                    )
                    well_pair = well_pair_list[0]
                    return {f"test_{well_pair[0]}_{well_pair[1]}": [df_train, df_test, feat_importances, classifier_info, predictions]}

                ## Bind df_sampled to the helper function
                classify_by_well_pair_bound = partial(classify_by_well_pair_exp_helper, df_sampled)
                result = thread_map(classify_by_well_pair_bound, well_pair_nested_list)
                
                pred_list = []
                for res in result:
                    if list(res.values())[-1] is not None:
                        filtered_cell_list.append(list(res.values())[0][0])
                        filtered_cell_list.append(list(res.values())[0][1])
                        feat_list.append(list(res.values())[0][2])
                        group_list.append(key)
                        pair_list.append(f"{key}_{subkey}")
                        info_list.append(list(res.values())[0][3])
                        pred_list.append(list(res.values())[0][4])
                    else:
                        err_logger.write(f"Skipped classification result for {key}_{subkey}\n")
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
                err_logger.write(f"{key}, {subkey} error: {e}")

    ### Store feature importance
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
    
    err_logger.write(f"Logging errors when running real experiments w/ {feat_type} features finished.\n")
    err_logger.write(f"===========================================================================\n\n")
    return df_feat, df_result, df_filtered_cells