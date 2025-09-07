"""Classification pipeline"""

import os
import sys
import warnings
from itertools import combinations
from typing import Union
import cupy as cp
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns

## constants
FEAT_TYPE_SET = ["GFP", "DNA", "AGP", "Mito", "Morph"] # , "TxControl" , 
GFP_INTENSITY_COLUMN = "Cells_Intensity_IntegratedIntensity_GFP" ## Cells_Intensity_MeanIntensity_GFP is another option


"""
    Util functions for annotations
"""
def control_type_helper(col_annot: str):
    """ helper func for annotating column "Metadata_control" """
    ## Only TC, NC, PC are used for constructing the null distribution because of multiple duplicates 
    if col_annot in ["TC", "NC", "PC"]:
        return True
    ## else labeled as not controls
    elif col_annot in ["disease_wt", "allele", "cPC", "cNC"]:
        return False
    else:
        return None


def add_control_annot(dframe):
    """annotating column "Metadata_control" """
    if "Metadata_control" not in dframe.columns:
        dframe["Metadata_control"] = dframe["Metadata_node_type"].apply(
            lambda x: control_type_helper(x)
        )
    return dframe


"""
    QC functions for filter out imaging wells with low cell counts
"""
def drop_low_cc_wells(dframe, cc_thresh, log_file):
    # Drop wells with cell counts lower than the threshold
    dframe["Metadata_Cell_ID"] = dframe.index
    cell_count = (
        dframe.groupby(["Metadata_Plate", "Metadata_Well"])["Metadata_Cell_ID"]
        .count()
        .reset_index(name="Metadata_Cell_Count")
    )
    ## get the cell counts per well per plate
    dframe = dframe.merge(
        cell_count,
        on=["Metadata_Plate", "Metadata_Well"],
    )
    dframe_dropped = (
        dframe[dframe["Metadata_Cell_Count"] < cc_thresh]
    )
    ## keep track of the alleles in a log file
    log_file.write(f"Number of wells dropped due to cell counts < {cc_thresh}: {len((dframe_dropped['Metadata_Plate']+dframe_dropped['Metadata_Well']+dframe_dropped['Metadata_gene_allele']).unique())}\n")
    dframe_dropped = dframe_dropped.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"])
    if (dframe_dropped.shape[0] > 0):
        for idx in dframe_dropped.index:
            log_file.write(f"{dframe_dropped.loc[idx, 'Metadata_Plate']}, {dframe_dropped.loc[idx, 'Metadata_Well']}:{dframe_dropped.loc[idx, 'Metadata_gene_allele']}\n")
            # print(f"{dframe_dropped.loc[idx, 'Metadata_Plate']}, {dframe_dropped.loc[idx, 'Metadata_Well']}:{dframe_dropped.loc[idx, 'Metadata_gene_allele']}\n")
    ## keep only the wells with cc >= cc_thresh
    dframe = (
        dframe[dframe["Metadata_Cell_Count"] >= cc_thresh]
        .drop(columns=["Metadata_Cell_Count"])
        .reset_index(drop=True)
    )
    return dframe


"""
    Utils functions for setting up trn/test sets for classifier training and testing
"""
def get_common_plates(dframe1, dframe2):
    """Helper func: get common plates in two dataframes"""
    plate_list = list(
        set(list(dframe1["Metadata_Plate"].unique()))
        & set(list(dframe2["Metadata_Plate"].unique()))
    )
    return plate_list


def stratify_by_plate(df_sampled: pd.DataFrame, plate: str):
    """Stratify dframe by plate"""
    # print(df_sampled.head())
    
    # OLD CODE - commented out due to assertion error when multiple platemaps match pattern
    # df_sampled_platemap = plate.split("_T")[0]
    # platemaps = df_sampled[df_sampled["Metadata_Plate"].str.contains(df_sampled_platemap)]["Metadata_plate_map_name"].to_list()
    # assert(len(set(platemaps))==1), f"Only one platemap should be associated with plate: {plate}."
    # platemap = platemaps[0]
    
    # NEW CODE - Get platemap for specific plate instead of pattern matching
    # This avoids assertion errors when multiple plates match the pattern
    plate_data = df_sampled[df_sampled["Metadata_Plate"] == plate]
    if plate_data.empty:
        # If exact plate match doesn't work, fall back to the old method but handle multiple platemaps
        df_sampled_platemap = plate.split("_T")[0]
        platemaps = df_sampled[df_sampled["Metadata_Plate"].str.contains(df_sampled_platemap)]["Metadata_plate_map_name"].to_list()
        if len(set(platemaps)) > 1:
            print(f"Warning: Multiple platemaps found for pattern {df_sampled_platemap}: {set(platemaps)}")
            print(f"Using first platemap: {platemaps[0]}")
        platemap = platemaps[0] if platemaps else None
    else:
        # Get the unique platemap(s) for this specific plate
        platemaps = plate_data["Metadata_plate_map_name"].unique().tolist()
        if len(platemaps) > 1:
            print(f"Warning: Multiple platemaps found for plate {plate}: {platemaps}")
            print(f"Using first platemap: {platemaps[0]}")
        platemap = platemaps[0] if platemaps else None
    
    if platemap is None:
        raise ValueError(f"No platemap found for plate: {plate}")

    # Train on data from same platemap but other plates
    df_train = df_sampled[
        (df_sampled["Metadata_plate_map_name"] == platemap)
        & (df_sampled["Metadata_Plate"] != plate)
    ].reset_index(drop=True)

    df_test = df_sampled[df_sampled["Metadata_Plate"] == plate].reset_index(drop=True)
    return df_train, df_test


def get_classifier_features(dframe: pd.DataFrame, feat_type: str):
    """Helper function to get dframe containing protein or non-protein features"""
    assert feat_type in FEAT_TYPE_SET, f"ONLY features in {FEAT_TYPE_SET} are allowed"
    feat_col = find_feat_cols(dframe)
    meta_col = find_meta_cols(dframe)

    if feat_type == "GFP":
        feat_col = [
            i
            for i in feat_col
            if (feat_type.lower() in i.lower())
            and ("Brightfield" not in i) ## excluding Brightfield features
        ]
    elif feat_type == "Morph":
        feat_col = [
            i
            for i in feat_col
            if ("GFP" not in i) and ("Brightfield" not in i) and ("TxControl" not in i)
        ]
    else:
        feat_col = [
            i
            for i in feat_col
            if (feat_type.lower() in i.lower())
            and ("Brightfield" not in i) and ("GFP" not in i) ## excluding Brightfield features and GFP features for other channel
        ]

    dframe = pd.concat([dframe[meta_col], dframe[feat_col]], axis=1)
    return dframe


"""
    Implementation of XGBoost Classifier
"""
def classifier(df_train, df_test, log_file, target="Label", shuffle=False):
    """
    This function train and test a classifier on the single-cell profiles from ref. and var. alleles.
    """

    feat_col = find_feat_cols(df_train)
    feat_col.remove(target)

    x_train, y_train = cp.array(df_train[feat_col].to_numpy()), df_train[[target]]
    x_test, y_test = cp.array(df_test[feat_col].to_numpy()), df_test[[target]]

    num_pos = df_train[df_train[target] == 1].shape[0]
    num_neg = df_train[df_train[target] == 0].shape[0]

    unique_plates = ",".join(sorted(df_train['Metadata_Plate'].unique()))
    gene_symbols = ",".join(sorted(df_train['Metadata_symbol'].unique()))
    wells = ",".join(sorted(df_train['Metadata_well_position'].unique()))

    if (num_pos == 0) or (num_neg == 0):
        log_file.write(f"Missing positive/negative labels for {gene_symbols} in wells: {wells} from plates {unique_plates}\n")
        log_file.write(f"Size of pos: {num_pos}, Size of neg: {num_neg}\n")
        print(f"Size of pos: {num_pos}, Size of neg: {num_neg}")
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, None, None

    scale_pos_weight = num_neg / num_pos

    if (scale_pos_weight > 100) or (scale_pos_weight < 0.01):
        log_file.write(f"Extreme class imbalance for {gene_symbols} in wells: {wells} from plates {unique_plates}\n")
        log_file.write(f"Scale_pos_weight: {scale_pos_weight}, Size of pos: {num_pos}, Size of neg: {num_neg}\n")
        print(
            f"Scale_pos_weight: {scale_pos_weight}, Size of pos: {num_pos}, Size of neg: {num_neg}"
        )
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, None, None

    le = LabelEncoder()
    y_train = cp.array(le.fit_transform(y_train))
    y_test = cp.array(le.fit_transform(y_test))

    if shuffle:
        # Create shuffled train labels
        y_train_shuff = y_train.copy()
        y_train_shuff["Label"] = np.random.permutation(y_train.values)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        tree_method="hist",
        device="cuda",
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
    ).fit(x_train, y_train, verbose=False)

    # get predictions and scores
    pred_score = model.predict_proba(x_test)[:, 1]

    # Return classifier info
    info_0 = df_test[df_test["Label"] == 0].iloc[0]
    info_1 = df_test[df_test["Label"] == 1].iloc[0]
    class_ID = (
        info_0["Metadata_Plate"]
        + "_"
        + info_0["Metadata_well_position"]
        + "_"
        + info_1["Metadata_well_position"]
    )
    classifier_df = pd.DataFrame({
        "Classifier_ID": [class_ID],
        "Plate": [info_0["Metadata_Plate"]],
        "trainsize_0": [sum(y_train.get() == 0)],
        "testsize_0": [sum(y_test.get() == 0)],
        "well_0": [info_0["Metadata_well_position"]],
        "allele_0": [info_0["Metadata_gene_allele"]],
        "trainsize_1": [sum(y_train.get() == 1)],
        "testsize_1": [sum(y_test.get() == 1)],
        "well_1": [info_1["Metadata_well_position"]],
        "allele_1": [info_1["Metadata_gene_allele"]],
    })

    # Store feature importance
    feat_importances = pd.Series(
        model.feature_importances_, index=df_train[feat_col].columns
    )

    # Return cell-level predictions
    cellID = df_test.apply(
        lambda row: f"{row['Metadata_Plate']}_{row['Metadata_well_position']}_{row['Metadata_ImageNumber']}_{row['Metadata_ObjectNumber']}",
        axis=1,
    ).to_list()

    pred_df = pd.DataFrame({
        "Classifier_ID": class_ID,
        "CellID": cellID,
        "Label": y_test.get(),
        "Prediction": pred_score,
    })

    return feat_importances, classifier_df, pred_df


"""
    Set up classification workflow with plate_layout of single replicate (single_rep) per plate
"""
def experimental_runner(
    exp_dframe: pd.DataFrame,
    pq_writer,
    log_file,
    feat_type,
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type",
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
                ## Define the func. for thread_map the plate on the same df_sampled
                def classify_by_plate_helper(plate):
                    df_train, df_test = stratify_by_plate(df_sampled, plate)
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
        #     break
        # break

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

    log_file.write(f"Finished running XGBboost classifiers w/ {feat_type} features on target variants.\n")
    log_file.write(f"===========================================================================\n\n")
    return df_feat, df_result


def control_group_runner(
    ctrl_dframe: pd.DataFrame,
    pq_writer,
    log_file,
    feat_type,
    group_key_one="Metadata_gene_allele",
    group_key_two="Metadata_plate_map_name",
    group_key_three="Metadata_well_position",
    threshold_key="Metadata_well_position"
):
    """
    Run null control experiments.
    """
    ctrl_dframe = get_classifier_features(ctrl_dframe, feat_type)
    feat_cols = find_feat_cols(ctrl_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]

    if len(feat_cols) == 0:
        return pd.DataFrame(), pd.DataFrame()

    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

    log_file.write(f"Running XGBboost classifiers w/ {feat_type} features on control alleles:\n")
    ## First group the df by reference genes
    groups = ctrl_dframe.groupby(group_key_one).groups
    for key in tqdm(groups.keys()):
        # groupby alleles
        dframe_grouped = ctrl_dframe.loc[groups[key]].reset_index(drop=True)
        # Skip controls with no replicates
        if dframe_grouped[threshold_key].unique().size < 2:
            continue

        # group by platemap
        subgroups = dframe_grouped.groupby(group_key_two).groups
        for key_two in subgroups.keys():
            dframe_grouped_two = dframe_grouped.loc[subgroups[key_two]].reset_index(
                drop=True
            )
            # If a well is not present on all four plates, drop well
            well_count = dframe_grouped_two.groupby(["Metadata_Well"])[
                "Metadata_Plate"
            ].nunique()
            well_to_drop = well_count[well_count < 4].index
            dframe_grouped_two = dframe_grouped_two[
                ~dframe_grouped_two["Metadata_Well"].isin(well_to_drop)
            ].reset_index(drop=True)

            # group by well
            sub_sub_groups = dframe_grouped_two.groupby(group_key_three).groups
            sampled_pairs = list(combinations(list(sub_sub_groups.keys()), r=2))

            for idx1, idx2 in sampled_pairs:
                df_group_one = dframe_grouped_two.loc[sub_sub_groups[idx1]].reset_index(
                    drop=True
                )
                df_group_one["Label"] = 1
                df_group_two = dframe_grouped_two.loc[sub_sub_groups[idx2]].reset_index(
                    drop=True
                )
                df_group_two["Label"] = 0
                df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)

                try:
                    plate_list = get_common_plates(df_group_one, df_group_two)

                    def classify_by_plate_helper(plate):
                        df_train, df_test = stratify_by_plate(df_sampled, plate)
                        feat_importances, classifier_info, predictions = classifier(
                            df_train, df_test, log_file
                        )
                        return {plate: [feat_importances, classifier_info, predictions]}

                    result = thread_map(classify_by_plate_helper, plate_list)

                    pred_list = []
                    for res in result:
                        if list(res.values())[-1] is not None:
                            feat_list.append(list(res.values())[0][0])
                            group_list.append(key)
                            pair_list.append(f"{idx1}_{idx2}")
                            info_list.append(list(res.values())[0][1])
                            pred_list.append(list(res.values())[0][2])
                        else:
                            log_file.write(f"Skipped classification result for {key}_{key_two}\n")
                            print(f"Skipping classification result for {key}_{key_two}...")
                            feat_list.append([None] * len(feat_cols))
                            group_list.append(key)
                            pair_list.append(f"{idx1}_{idx2}")
                            info_list.append([None] * 10)

                    cell_preds = pd.concat(pred_list, axis=0)
                    cell_preds["Metadata_Feature_Type"] = feat_type
                    cell_preds["Metadata_Control"] = True
                    table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                    pq_writer.write_table(table)
                except Exception as e:
                    print(e)
                    log_file.write(f"{key}, {key_two} error: {e}, wells per ctrl: {sub_sub_groups}\n")
            #     break
            # break

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Feature_Type"] = feat_type
    df_feat["Metadata_Control"] = True

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = True
    df_result["Metadata_Feature_Type"] = feat_type

    log_file.write(f"Finished running XGBboost classifiers w/ {feat_type} features on control alleles.\n")
    log_file.write(f"===========================================================================\n\n")
    return df_feat, df_result


"""
    Set up classification workflow with plate_layout of multiple replicates (multi_rep) per plate
"""
#######################################
# 2. RUN CLASSIFIERS ON CTRL ALLELES
# Resampling ctrl wells and run the classifiers on them for the null dist.
#######################################
def stratify_by_well_pair_ctrl(dframe_grouped_two: pd.DataFrame, well_pair_trn: tuple):
    """Stratify dframe by ctrl well pairs: one pair for training and one pair for testing"""
    sub_sub_groups = dframe_grouped_two.groupby("Metadata_well_position").groups
    assert len(sub_sub_groups.keys()) == 4, f"Number of wells per plate is not 4: {sub_sub_groups.keys()}"

    ## Train on data from well_pair_trn
    df_group_one = dframe_grouped_two.loc[sub_sub_groups[well_pair_trn[0]]].reset_index(
        drop=True
    )
    df_group_one["Label"] = 1
    df_group_two = dframe_grouped_two.loc[sub_sub_groups[well_pair_trn[1]]].reset_index(
        drop=True
    )
    df_group_two["Label"] = 0
    df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)
    df_train = df_sampled.reset_index(drop=True)

    ## Test on data from well_pair_test
    well_pair_test = tuple(key for key in sub_sub_groups.keys() if key not in well_pair_trn)
    df_group_3 = dframe_grouped_two.loc[sub_sub_groups[well_pair_test[0]]].reset_index(
        drop=True
    )
    df_group_3["Label"] = 1
    df_group_4 = dframe_grouped_two.loc[sub_sub_groups[well_pair_test[1]]].reset_index(
        drop=True
    )
    df_group_4["Label"] = 0
    df_sampled_test = pd.concat([df_group_3, df_group_4], ignore_index=True)
    df_test = df_sampled_test.reset_index(drop=True)
    return df_train, df_test


def control_group_runner_fewer_rep(
    ctrl_dframe: pd.DataFrame,
    pq_writer,
    err_logger,
    feat_type,
    group_key_one="Metadata_gene_allele",
    group_key_two="Metadata_plate_map_name",
    group_key_three="Metadata_well_position",
    threshold_key="Metadata_well_position",
    well_count_min=None
):
    """
    Run classifiers on control alleles.

    # df_feat_pro_con, df_result_pro_con = control_group_runner_fewer_rep(df_control, pq_writer=writer, err_logger=err_logger, feat_type=feat_type)
    """
    ctrl_dframe = get_classifier_features(ctrl_dframe, feat_type)
    feat_cols = find_feat_cols(ctrl_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]

    if len(feat_cols) == 0:
        return pd.DataFrame(), pd.DataFrame()

    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

    err_logger.write(f"Logging errors when running control experiments w/ {feat_type} features:\n")
    ## first we group the cells from the same Metadata_gene_allele
    groups = ctrl_dframe.groupby(group_key_one).groups
    for key in tqdm(groups.keys()):
        ## groupby alleles
        dframe_grouped = ctrl_dframe.loc[groups[key]].reset_index(drop=True)
        # Skip controls with no replicates
        if dframe_grouped[threshold_key].unique().size < 2:
            continue
        ## group by platemap
        subgroups = dframe_grouped.groupby(group_key_two).groups
        for key_two in subgroups.keys():
            ## for each platemap
            dframe_grouped_two = dframe_grouped.loc[subgroups[key_two]].reset_index(
                drop=True
            )
            ## If a well is not present on all four plates, drop well
            ## ONLY used when we have enough TECHNICAL-REPLICATE plates!!!
            if well_count_min is not None:
                well_count = dframe_grouped_two.groupby(["Metadata_Well"])[
                    "Metadata_Plate"
                ].nunique()
                well_to_drop = well_count[well_count < well_count_min].index
                dframe_grouped_two = dframe_grouped_two[
                    ~dframe_grouped_two["Metadata_Well"].isin(well_to_drop)
                ].reset_index(drop=True)

            ## group by well
            sub_sub_groups = dframe_grouped_two.groupby(group_key_three).groups
            sampled_pairs = list(combinations(list(sub_sub_groups.keys()), r=2))
            ## juxtapose each pair of wells against each other                
            try:
                def classify_by_well_pair_helper(df_sampled: pd.DataFrame, well_pair: tuple, log_file=err_logger):
                    """Helper func to run classifiers in parallel"""
                    df_train, df_test = stratify_by_well_pair_ctrl(df_sampled, well_pair)
                    feat_importances, classifier_info, predictions = classifier(
                        df_train, df_test, log_file
                    )
                    return {f"trn_{well_pair[0]}_{well_pair[1]}": [feat_importances, classifier_info, predictions]}
                
                ## Bind df_sampled to the helper function
                classify_by_well_pair_bound = partial(classify_by_well_pair_helper, dframe_grouped_two)
                result = thread_map(classify_by_well_pair_bound, sampled_pairs)
                pred_list = []
                for res in result:
                    if list(res.values())[-1] is not None:
                        feat_list.append(list(res.values())[0][0])
                        group_list.append(key)
                        pair_list.append(list(res.keys())[0])
                        info_list.append(list(res.values())[0][1])
                        pred_list.append(list(res.values())[0][2])
                    else:
                        err_logger.write(f"Skipped classification result for {key}_{key_two}\n")
                        print(f"Skipping classification result for {key}_{key_two}...")
                        feat_list.append([None] * len(feat_cols))
                        group_list.append(key)
                        pair_list.append(list(res.keys())[0])
                        info_list.append([None] * 10)

                cell_preds = pd.concat(pred_list, axis=0)
                cell_preds["Metadata_Feature_Type"] = feat_type
                cell_preds["Metadata_Control"] = True
                table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                pq_writer.write_table(table)
            except Exception as e:
                print(e)
                err_logger.write(f"{key}, {key_two} error: {e}, wells per ctrl: {sub_sub_groups}\n")

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Feature_Type"] = feat_type
    df_feat["Metadata_Control"] = True

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = True
    df_result["Metadata_Feature_Type"] = feat_type

    err_logger.write(f"Logging errors when running control experiments w/ {feat_type} features finished.\n")
    err_logger.write(f"==============================================================================\n\n")
    return df_feat, df_result
    

#######################################
# 3. RUN CLASSIFIERS ON VAR-REF ALLELES
# Construct 4-fold CV on var-vs-ref wells and run the classifiers on them.
#######################################
def stratify_by_well_pair_exp(df_sampled: pd.DataFrame, well_pair_list: list):
    """
        Stratify dframe by plate
        df_sampled: the data frame containing both ref. and var. alleles, each tested in 4 wells
        well_pair: a list of well pairs containing a ref. and a var. allele, with 1st pair for test and the rest pairs for training
    """
    ## Training on the rest three wells out of four
    df_train = df_sampled[
        (df_sampled["Metadata_well_position"].isin([well for pair in well_pair_list[1:] for well in pair]))
    ].reset_index(drop=True)
    ## Testing on the well_pair
    df_test = df_sampled[
        df_sampled["Metadata_well_position"].isin(well_pair_list[0])
    ].reset_index(drop=True)
    return df_train, df_test


def experimental_runner_plate_rep(
    exp_dframe: pd.DataFrame,
    pq_writer,
    err_logger,
    feat_type,
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type",
):
    """
    Run Reference v.s. Variant experiments run on the same plate without tech. dups

    # df_feat_pro_exp, df_result_pro_exp = experimental_runner_plate_rep(df_exp, pq_writer=writer, protein=True)
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
                    df_train, df_test = stratify_by_well_pair_exp(df_sampled, well_pair_list)
                    feat_importances, classifier_info, predictions = classifier(
                        df_train, df_test, log_file
                    )
                    well_pair = well_pair_list[0]
                    return {f"test_{well_pair[0]}_{well_pair[1]}": [feat_importances, classifier_info, predictions]}

                ## Bind df_sampled to the helper function
                classify_by_well_pair_bound = partial(classify_by_well_pair_exp_helper, df_sampled)
                result = thread_map(classify_by_well_pair_bound, well_pair_nested_list)
                
                pred_list = []
                for res in result:
                    if list(res.values())[-1] is not None:
                        feat_list.append(list(res.values())[0][0])
                        group_list.append(key)
                        pair_list.append(f"{key}_{subkey}")
                        info_list.append(list(res.values())[0][1])
                        pred_list.append(list(res.values())[0][2])
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

    err_logger.write(f"Logging errors when running real experiments w/ {feat_type} features finished.\n")
    err_logger.write(f"===========================================================================\n\n")
    return df_feat, df_result


#######################################
# 4. RUN CLASSIFIERS ON VAR-REF ALLELES
# Corrected by GFP intensity
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
    use_gpu: Union[str, None] = "0,1",
):
    """
    Run workflow for single-cell classification
    """
    assert plate_layout in ("single_rep", "multi_rep"), f"Incorrect plate_layout: {plate_layout}, only 'single_rep' and 'multi_rep' allowed."

    if use_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

    # Initialize parquet for cell-level predictions
    if os.path.exists(preds_output_path):
        os.remove(preds_output_path)
    
    ## create a log file
    logfile_path = os.path.join('/'.join(preds_output_path.split("/")[:-1]), "classify.log")

    ## Write out feature importance and classifier info
    # feat_output_path_gfp = feat_output_path.split(".")[0] + "_gfp_adj." + feat_output_path.split(".")[-1]
    # info_output_path_gfp = info_output_path.split(".")[0] + "_gfp_adj." + info_output_path.split(".")[-1]
    # preds_output_path_gfp = preds_output_path.split(".")[0] + "_gfp_adj." + preds_output_path.split(".")[-1]
    # filtered_cell_path = feat_output_path.split("/")[0] + "gfp_adj_filtered_cells_profiles.parquet"

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
    writer_gfp = pq.ParquetWriter(preds_output_path_gfp, schema, compression="gzip")

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
    feat_import_dfs, class_res_dfs = [], []
    feat_import_gfp_adj_dfs, class_res_gfp_adj_dfs, filtered_cells_gfp_adj_dfs = [], [], []
    ## Split data into experimental df with var and ref alleles
    df_exp = dframe[~dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
    writer = pq.ParquetWriter(preds_output_path, schema, compression="gzip")
    with open(logfile_path, "w") as log_file:
        log_file.write(f"===============================================================================================================================================================\n")
        log_file.write("Dropping low cell count wells in ref. vs variant alleles:\n")
        print("Dropping low cell count wells in ref. vs variant alleles:")
        df_exp = drop_low_cc_wells(df_exp, cc_threshold, log_file)
        log_file.write(f"===============================================================================================================================================================\n\n")
        # Check the plate_layout for the correct classification set-up
        if (plate_layout=="single_rep"):
            ## If the plate_layout is single_rep, with only one well per allele on a single plate
            ## we can only get the control_df with the control labels
            df_control = dframe[dframe["Metadata_control"].astype("bool")].reset_index(
                drop=True
            )
            # Remove any remaining TC from analysis
            df_control = df_control[df_control["Metadata_node_type"] != "TC"].reset_index(
                drop=True
            )
            log_file.write("Dropping low cell count wells in ONLY the control alleles on the same plate:\n")
            print("Dropping low cell count wells in ONLY the control alleles on the same plate:\n")
            # Filter out wells with fewer than the cell count threhsold
            df_control = drop_low_cc_wells(df_control, cc_threshold, log_file)
            print("Check ctrl df:")
            print(df_control)

            for feat in FEAT_TYPE_SET:
                print(feat)
                ## GFP corrected version
                if feat == "GFP":
                    df_feat_pro_exp_gfp_adj, df_result_pro_exp_gfp_adj, df_filtered_cells_gfp_adj = experimental_runner_filter_gfp(
                        df_exp, pq_writer=writer_gfp, 
                        log_file=log_file, min_cells_per_well=cc_threshold
                    )
                    ## store to another set of feat_df, res_df and filtered_cells parquet
                    if (df_feat_pro_exp_gfp_adj.shape[0] > 0):
                        feat_import_gfp_adj_dfs += [df_feat_pro_exp_gfp_adj]
                        class_res_gfp_adj_dfs += [df_result_pro_exp_gfp_adj]
                        filtered_cells_gfp_adj_dfs += [df_filtered_cells_gfp_adj]
                        
                df_feat_pro_con, df_result_pro_con = control_group_runner(
                    df_control, pq_writer=writer, log_file=log_file, feat_type=feat
                )
                df_feat_pro_exp, df_result_pro_exp = experimental_runner(
                    df_exp, pq_writer=writer, log_file=log_file, feat_type=feat
                )
                if (df_feat_pro_con.shape[0] > 0):
                    feat_import_dfs += [df_feat_pro_con, df_feat_pro_exp]
                    class_res_dfs += [df_result_pro_con, df_result_pro_exp]
                
        else:
            ## If the plate_layout is multi_rep, with multiple wells per allele on a single plate
            ## we can get control_df with every possible allele on the same plate
            ## As long as it is not a TC:
            df_control = dframe[dframe["Metadata_node_type"] != "TC"].reset_index(
                drop=True
            )
            log_file.write("Dropping low cell count wells in every possible allele that could be used as controls:\n")
            print("Dropping low cell count wells in every possible allele that could be used as controls:\n")
            # Filter out wells with fewer than the cell count threhsold
            df_control = drop_low_cc_wells(df_control, cc_threshold, log_file)
            
            for feat in FEAT_TYPE_SET:
                df_feat_pro_con, df_result_pro_con = control_group_runner_fewer_rep(
                    df_control, pq_writer=writer, err_logger=log_file, feat_type=feat
                )
                df_feat_pro_exp, df_result_pro_exp = experimental_runner_plate_rep(
                    df_exp, pq_writer=writer, err_logger=log_file, feat_type=feat
                )

                if (df_feat_pro_con.shape[0] > 0):
                    feat_import_dfs += [df_feat_pro_con, df_feat_pro_exp]
                    class_res_dfs += [df_result_pro_con, df_result_pro_exp]

        ## Close the parquest writer
        writer.close()

    # Concatenate results for both protein and non-protein
    df_feat = pd.concat(
        feat_import_dfs, ignore_index=True
    )
    
    df_result = pd.concat(
        class_res_dfs, ignore_index=True
    )
    df_result = df_result.drop_duplicates()

    df_feat_gfp_adj = pd.concat(
        feat_import_gfp_adj_dfs, ignore_index=True
    )
    df_result_gfp_adj = pd.concat(
        class_res_gfp_adj_dfs, ignore_index=True
    )
    df_filtered_cell = pd.concat(
        filtered_cells_gfp_adj_dfs, ignore_index=True
    )
    df_result_gfp_adj = df_result_gfp_adj.drop_duplicates()

    # Write out feature importance and classifier info
    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(info_output_path, index=False)
    
    df_feat_gfp_adj.to_csv(feat_output_path_gfp, index=False)
    df_result_gfp_adj.to_csv(info_output_path_gfp, index=False)
    df_filtered_cell.to_parquet(filtered_cell_path, index=False)