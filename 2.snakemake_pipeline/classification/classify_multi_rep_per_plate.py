import sys
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from itertools import combinations
from functools import partial
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns
from .classify_helper_func import get_classifier_features, classifier

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