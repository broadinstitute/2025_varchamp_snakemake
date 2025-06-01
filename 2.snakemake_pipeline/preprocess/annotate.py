"""Annotate dataframe with platemap"""
import pandas as pd
import numpy as np
from utils import remove_nan_infs_columns
from typing import List
import sys
sys.path.append("..")


def get_platemap(csv_path: str, plate: str) -> str:
    """Get the platemap .txt file"""
    barcode = pd.read_csv(csv_path)
    barcode_dict = dict(zip(barcode["Assay_Plate_Barcode"], barcode["Plate_Map_Name"]))
    return barcode_dict[plate]


def annotate_with_platemap(
    profile_path: str, platemap_path: str, output_file_path: str
):
    """Annotate dataframe using platemap"""

    profile = pd.read_parquet(profile_path, engine="pyarrow")
    platemap = pd.read_csv(platemap_path, sep="\t").copy()

    if "Metadata_Well" not in platemap.columns:
        platemap["Metadata_Well"] = platemap["well_position"]

    # Append 'Metadata_' to platemap column names
    platemap.columns = [
        f"Metadata_{x}" if not x.startswith("Metadata_") else x
        for x in platemap.columns
    ]

    # Annotate with platemap
    aligned_df = platemap.merge(profile, on=["Metadata_Well"], how="inner")

    aligned_df.to_parquet(path=output_file_path, compression="gzip", index=False)


def aggregate(plate_list: List[str], output_path: str):
    """Aggregate profiles and save"""
    df_plate_list = []
    for file in plate_list:
        if not file.endswith(".parquet"):
            continue
        df = pd.read_parquet(file)
        df_plate_list.append(df)
    df_agg = pd.concat(df_plate_list, ignore_index=True)
    df_agg.to_parquet(path=output_path, compression="gzip", index=False)


def filter_nan(input_path: str, output_path: str):
    """Filter nan and inf columns"""
    dframe = pd.read_parquet(input_path)
    dframe = remove_nan_infs_columns(dframe)
    dframe.to_parquet(output_path, compression="gzip", index=False)


def drop_nan_features(input_path: str, output_path: str, cell_threshold=100):
    """
    This function first drops the features that contain more than 100 rows of null cells,
    then drop the rest of null rows.
    """
    sc_profiles = pd.read_parquet(input_path)
    # Drop features with nan values
    sc_profiles.replace([np.inf, -np.inf], np.nan, inplace=True)
    _, c = np.where(sc_profiles.isna())
    feat, count = np.unique(c, return_counts=True)

    feat_to_remove = []
    row_to_remove = []

    for i, feat_idx in enumerate(feat):
        feat_name = sc_profiles.columns[feat_idx]
        if feat_name.startswith("Metadata_"):
            continue

        # If more than @cell_threshold rows have nanfeatures,
        # drop feature, otherwise drop rows
        if count[i] > cell_threshold:
            feat_to_remove.append(feat_name)
        else:
            row_to_remove = (
                row_to_remove + np.where(sc_profiles[feat_name].isna())[0].tolist()
            )
            row_to_remove = list(set(row_to_remove))

    sc_profiles.drop(feat_to_remove, axis=1, inplace=True)
    sc_profiles.drop(row_to_remove, axis=0, inplace=True)
    sc_profiles.reset_index(drop=True, inplace=True)
    print(
        f"Removed {len(feat_to_remove)} NaN features and {len(row_to_remove)} NaN rows."
    )
    feat_col = [i for i in sc_profiles.columns if "Metadata_" not in i]
    # Ensure no null rows or columns
    assert (
        ~np.isnan(sc_profiles[feat_col]).any().any()
    ), "Dataframe contain NaN features."
    assert (
        np.isfinite(sc_profiles[feat_col]).all().all()
    ), "Dataframe contain infinite feature values."

    sc_profiles.to_parquet(output_path, compression='gzip', index=False)


def drop_empty_wells(
    input_path: str,
    output_path: str,
    pert_col: str = "Metadata_control_type",
    pert_name: list = ["TC"],
):
    """Drop rows with no metadata and transfection controls"""
    dframe = pd.read_parquet(input_path)
    # drop rows with no well position metadata
    dframe = dframe[~dframe["Metadata_well_position"].isna()].reset_index(drop=True)
    # drop transfection control wells
    dframe = dframe[~dframe[pert_col].isin(pert_name)].reset_index(drop=True)
    dframe.to_parquet(output_path, compression="gzip", index=False)
