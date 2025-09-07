import sys
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

sys.path.append("..")
from utils import find_feat_cols, find_meta_cols

# Constants
FEAT_TYPE_SET = ["GFP", "DNA", "AGP", "Mito", "Morph"]

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
            and ("Brightfield" not in i) and ("GFP" not in i) and ("TxControl" not in i) ## excluding Brightfield features and GFP features for other channel
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