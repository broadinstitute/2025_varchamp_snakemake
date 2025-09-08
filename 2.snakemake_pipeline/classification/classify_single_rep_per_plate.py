import sys
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from itertools import combinations
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns
from .classify_helper_func import get_classifier_features, classifier

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