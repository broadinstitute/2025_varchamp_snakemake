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

warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns


def classifier(df_train, df_test, target="Label", shuffle=False):
    """
    This function runs classification.
    """

    feat_col = find_feat_cols(df_train)
    feat_col.remove(target)

    x_train, y_train = cp.array(df_train[feat_col].to_numpy()), df_train[[target]]
    x_test, y_test = cp.array(df_test[feat_col].to_numpy()), df_test[[target]]

    num_pos = df_train[df_train[target] == 1].shape[0]
    num_neg = df_train[df_train[target] == 0].shape[0]

    if (num_pos == 0) or (num_neg == 0):
        print(f"size of pos: {num_pos}, size of neg: {num_neg}")
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, np.nan

    scale_pos_weight = num_neg / num_pos

    if (scale_pos_weight > 100) or (scale_pos_weight < 0.01):
        print(
            f"scale_pos_weight: {scale_pos_weight}, size of pos: {num_pos}, size of neg: {num_neg}"
        )
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, np.nan

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


def get_classifier_features(dframe: pd.DataFrame, protein_feat: bool):
    """Helper function to get dframe containing protein or non-protein features"""
    feat_col = find_feat_cols(dframe)
    meta_col = find_meta_cols(dframe)

    if protein_feat:
        feat_col = [
            i
            for i in feat_col
            if ("GFP" in i)
            and ("DNA" not in i)
            and ("AGP" not in i)
            and ("Mito" not in i)
            and ("Brightfield" not in i)
        ]
    else:
        feat_col = [
            i
            for i in feat_col
            if ("GFP" not in i) and ("Brightfield" not in i) ## and ("AGP" not in i), test without AGP features
        ]

    dframe = pd.concat([dframe[meta_col], dframe[feat_col]], axis=1)
    return dframe


def stratify_by_plate(df_sampled: pd.DataFrame, plate: str):
    """Stratify dframe by plate"""
    platemap = "_".join(plate.split("T")[0].split("_")[-2:])

    # Train on data from same platemap but other plates
    df_train = df_sampled[
        (df_sampled["Metadata_plate_map_name"] == platemap)
        & (df_sampled["Metadata_Plate"] != plate)
    ].reset_index(drop=True)

    df_test = df_sampled[df_sampled["Metadata_Plate"] == plate].reset_index(drop=True)

    return df_train, df_test


def experimental_runner(
    exp_dframe: pd.DataFrame,
    pq_writer,
    protein=True,
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type",
):
    """
    Run Reference v.s. Variant experiments
    """
    exp_dframe = get_classifier_features(exp_dframe, protein)
    feat_cols = find_feat_cols(exp_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]

    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

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

        subgroups = (
            dframe_grouped[dframe_grouped["Metadata_node_type"] == "allele"]
            .groupby(group_key_two)
            .groups
        )

        for subkey in subgroups.keys():
            df_group_two = dframe_grouped.loc[subgroups[subkey]].reset_index(drop=True)
            df_group_two["Label"] = 0

            df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)

            plate_list = get_common_plates(df_group_one, df_group_two)

            def classify_by_plate_helper(plate):
                df_train, df_test = stratify_by_plate(df_sampled, plate)
                feat_importances, classifier_info, predictions = classifier(
                    df_train, df_test
                )
                return {plate: [feat_importances, classifier_info, predictions]}

            # try run classifier
            try:
                result = thread_map(classify_by_plate_helper, plate_list)

                pred_list = []
                for res in result:
                    if len(list(res.values())[0]) == 3:
                        feat_list.append(list(res.values())[0][0])
                        group_list.append(key)
                        pair_list.append(f"{key}_{subkey}")
                        info_list.append(list(res.values())[0][1])
                        pred_list.append(list(res.values())[0][2])
                    else:
                        print("res length not 3!")
                        feat_list.append([None] * len(feat_cols))
                        group_list.append(key)
                        pair_list.append(f"{key}_{subkey}")
                        info_list.append([None] * 10)

                cell_preds = pd.concat(pred_list, axis=0)
                cell_preds["Metadata_Protein"] = protein
                cell_preds["Metadata_Control"] = False
                table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                pq_writer.write_table(table)
            except Exception as e:
                print(e)

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = False

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = False
    return df_feat, df_result


def get_common_plates(dframe1, dframe2):
    """Helper func: get common plates in two dataframes"""
    plate_list = list(
        set(list(dframe1["Metadata_Plate"].unique()))
        & set(list(dframe2["Metadata_Plate"].unique()))
    )
    return plate_list


def control_group_runner(
    ctrl_dframe: pd.DataFrame,
    pq_writer,
    group_key_one="Metadata_gene_allele",
    group_key_two="Metadata_plate_map_name",
    group_key_three="Metadata_well_position",
    threshold_key="Metadata_well_position",
    protein=True,
):
    """
    Run null control experiments.
    """
    ctrl_dframe = get_classifier_features(ctrl_dframe, protein)
    feat_cols = find_feat_cols(ctrl_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]

    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

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
                            df_train, df_test
                        )
                        return {plate: [feat_importances, classifier_info, predictions]}

                    result = thread_map(classify_by_plate_helper, plate_list)

                    pred_list = []
                    for res in result:
                        if len(list(res.values())[0]) == 3:
                            feat_list.append(list(res.values())[0][0])
                            group_list.append(key)
                            pair_list.append(f"{idx1}_{idx2}")
                            info_list.append(list(res.values())[0][1])
                            pred_list.append(list(res.values())[0][2])
                        else:
                            print("res length does not equal three!")
                            feat_list.append([None] * len(feat_cols))
                            group_list.append(key)
                            pair_list.append(f"{idx1}_{idx2}")
                            info_list.append([None] * 10)

                    cell_preds = pd.concat(pred_list, axis=0)
                    cell_preds["Metadata_Protein"] = protein
                    cell_preds["Metadata_Control"] = True
                    table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                    pq_writer.write_table(table)
                except Exception as e:
                    print(e)

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = True

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = True
    return df_feat, df_result


def control_type_helper(col_annot: str):
    """helper func for annotating column "Metadata_control" """
    if col_annot in ["TC", "NC", "PC", "cPC", "cNC"]:
        return True
    elif col_annot in ["disease_wt", "allele"]:
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


def drop_low_cc_wells(dframe, cc_thresh):
    # Drop wells with cell counts lower than the threshold

    dframe["Metadata_Cell_ID"] = dframe.index
    cell_count = (
        dframe.groupby(["Metadata_Plate", "Metadata_Well"])["Metadata_Cell_ID"]
        .count()
        .reset_index(name="Metadata_Cell_Count")
    )

    dframe = dframe.merge(
        cell_count,
        on=["Metadata_Plate", "Metadata_Well"],
    )
    dframe = (
        dframe[dframe["Metadata_Cell_Count"] > cc_thresh]
        .drop(columns=["Metadata_Cell_Count"])
        .reset_index(drop=True)
    )
    return dframe


def run_classify_workflow(
    input_path: str,
    feat_output_path: str,
    info_output_path: str,
    preds_output_path: str,
    cc_threshold: int,
    use_gpu: Union[str, None] = "0,1",
):
    """
    Run workflow for single-cell classification
    """
    if use_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

    # Initialize parquet for cell-level predictions
    if os.path.exists(preds_output_path):
        os.remove(preds_output_path)
    schema = pa.schema([
        ("Classifier_ID", pa.string()),
        ("CellID", pa.string()),
        ("Label", pa.int64()),
        ("Prediction", pa.float32()),
        ("Metadata_Protein", pa.bool_()),
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

    # Split data into controls and alleles
    df_exp = dframe[~dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
    df_control = dframe[dframe["Metadata_control"].astype("bool")].reset_index(
        drop=True
    )

    # Remove any remaining TC from analysis
    df_control = df_control[df_control["Metadata_node_type"] != "TC"].reset_index(
        drop=True
    )

    # Filter out wells with fewer than the cell count threhsold
    df_control = drop_low_cc_wells(df_control, cc_threshold)
    df_exp = drop_low_cc_wells(df_exp, cc_threshold)

    # Protein feature analysis
    df_feat_pro_con, df_result_pro_con = control_group_runner(
        df_control, pq_writer=writer, protein=True
    )
    df_feat_pro_exp, df_result_pro_exp = experimental_runner(
        df_exp, pq_writer=writer, protein=True
    )

    # Non-protein feature analysis
    df_feat_no_pro_con, df_result_no_pro_con = control_group_runner(
        df_control, pq_writer=writer, protein=False
    )
    df_feat_no_pro_exp, df_result_no_pro_exp = experimental_runner(
        df_exp, pq_writer=writer, protein=False
    )
    writer.close()

    # Concatenate results for both protein and non-protein
    df_feat = pd.concat(
        [df_feat_pro_con, df_feat_no_pro_con, df_feat_pro_exp, df_feat_no_pro_exp],
        ignore_index=True,
    )
    df_result = pd.concat(
        [
            df_result_pro_con,
            df_result_no_pro_con,
            df_result_pro_exp,
            df_result_no_pro_exp,
        ],
        ignore_index=True,
    )
    df_result = df_result.drop_duplicates()

    # Write out feature importance and classifier info
    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(info_output_path, index=False)
