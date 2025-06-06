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
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns

## constants
FEAT_TYPE_SET = set(["GFP", "DNA", "AGP", "Mito", "Morph"])

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
    df_sampled_platemap = plate.split("_T")[0]
    platemaps = df_sampled[df_sampled["Metadata_Plate"].str.contains(df_sampled_platemap)]["Metadata_plate_map_name"].to_list()
    assert(len(set(platemaps))==1), "Only one platemap should be associated with plate: {plate}."
    platemap = platemaps[0]

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

    if feat_type != "Morph":
        feat_col = [
            i
            for i in feat_col
            if (feat_type.lower() in i.lower())
            and ("Brightfield" not in i) ## excluding Brightfield features
        ]
    else:
        feat_col = [
            i
            for i in feat_col
            if ("GFP" not in i) and ("Brightfield" not in i)
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
            dframe_grouped[threshold_key] == "disease_wt"
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
            dframe_grouped["Metadata_node_type"] == "disease_wt"
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


def run_classify_workflow(
    input_path: str,
    feat_output_path: str,
    info_output_path: str,
    preds_output_path: str,
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
        .collect()
        .to_pandas()
    )
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
    ## Split data into experimental df with var and ref alleles
    df_exp = dframe[~dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
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
            
            for feat in FEAT_TYPE_SET:
                df_feat_pro_con, df_result_pro_con = control_group_runner(
                    df_control, pq_writer=writer, log_file=log_file, feat_type=feat
                )
                df_feat_pro_exp, df_result_pro_exp = experimental_runner(
                    df_exp, pq_writer=writer, log_file=log_file, feat_type=feat
                )
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

    # Write out feature importance and classifier info
    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(info_output_path, index=False)