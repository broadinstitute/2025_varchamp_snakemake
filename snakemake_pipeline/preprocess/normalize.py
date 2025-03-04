"""RobustMAD Normalization"""
from functools import partial
import logging

from scipy.stats import median_abs_deviation

import numpy as np
import pandas as pd
import polars as pl
from typing import List, Tuple

import sys

sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns

logger = logging.getLogger(__name__)


def split_parquet(
    dframe_path, features=None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Split dframe into metadata, feature values, feature columns"""
    dframe = pd.read_parquet(dframe_path)
    if features is None:
        features = find_feat_cols(dframe)
    vals = np.empty((len(dframe), len(features)), dtype=np.float32)
    for i, c in enumerate(features):
        vals[:, i] = dframe[c]
    meta = dframe[find_meta_cols(dframe)].copy()
    return meta, vals, features


def merge_parquet(meta, vals, features, output_path) -> None:
    """Save the data in a parquet file resetting the index"""
    dframe = pd.DataFrame(vals, columns=features)
    for c in meta:
        dframe[c] = meta[c].reset_index(drop=True)
    dframe.to_parquet(output_path)


def add_metadata(stats: pd.DataFrame):
    """Add metadata to plate statistics"""
    parts = stats["feature"].str.split("_", expand=True)
    stats["compartment"] = parts[0].astype("category")
    stats["family"] = parts[range(3)].apply("_".join, axis=1).astype("category")
    return stats


def get_plate_stats_polar(lf: pl.LazyFrame) -> pl.DataFrame:
    """Compute plate level statistics with polars"""
    feat_col = [col for col in lf.columns if not col.startswith("Metadata_")]
    feat_col.append("Metadata_Plate")
    lframe = lf.select(pl.col(feat_col))

    groupby_plate = lframe.group_by("Metadata_Plate")
    median = groupby_plate.agg(pl.col(feat_col[:-1]).median().cast(pl.Float32))
    max_ = groupby_plate.agg(pl.col(feat_col[:-1]).max().cast(pl.Float32))
    min_ = groupby_plate.agg(pl.col(feat_col[:-1]).min().cast(pl.Float32))
    count = groupby_plate.agg(pl.col(feat_col[:-1]).count().cast(pl.Float32))
    mad_ = groupby_plate.agg(
        pl.col(feat_col[:-1])
        .map_elements(lambda col: (col - col.median()).abs().median())
        .cast(pl.Float32)
    )

    median = median.with_columns(stat=pl.lit("median"))
    mad_ = mad_.with_columns(stat=pl.lit("mad"))
    max_ = max_.with_columns(stat=pl.lit("max"))
    min_ = min_.with_columns(stat=pl.lit("min"))
    count = count.with_columns(stat=pl.lit("count"))

    stats = pl.concat([median, mad_, min_, max_, count])
    stats = stats.melt(id_vars=["Metadata_Plate", "stat"], variable_name="feature", value_name="value")
    stats = stats.collect()
    stats = stats.pivot(
        index=["Metadata_Plate", "feature"], columns="stat", values="value"
    )
    stats = stats.with_columns(
        abs_coef_var=pl.col("mad")
        / pl.col("median").fill_nan(0).abs().replace(np.inf, 0)
    )
    stats = stats.cast(
        {
            "min": pl.Float32,
            "max": pl.Float32,
            "count": pl.Float32,
            "median": pl.Float32,
            "mad": pl.Float32,
            "abs_coef_var": pl.Float32,
            "feature": pl.Categorical,
        }
    )

    return stats


def compute_norm_stats_polar(parquet_path: str, df_stats_path: str):
    """create platewise statistics for columns without nan/inf values only"""
    lf = pl.scan_parquet(parquet_path)
    lf_stats = get_plate_stats_polar(lf)
    logger.info("stats done.")
    dframe_stats = lf_stats.to_pandas()
    dframe_stats = add_metadata(dframe_stats)
    dframe_stats.to_parquet(df_stats_path, compression="gzip", index=False)


def get_plate_stats(dframe: pd.DataFrame):
    """Compute plate level statistics"""
    mad_fn = partial(median_abs_deviation, nan_policy="omit", axis=0)

    feat_cols = find_feat_cols(dframe)
    dframe = dframe[feat_cols + ["Metadata_Plate"]]
    median = dframe.groupby("Metadata_Plate", observed=True).median()
    max_ = dframe.groupby("Metadata_Plate", observed=True).max()
    min_ = dframe.groupby("Metadata_Plate", observed=True).min()
    count = dframe.groupby("Metadata_Plate", observed=True).count()
    mad = dframe.groupby("Metadata_Plate", observed=True).apply(mad_fn)
    mad = pd.DataFrame(index=mad.index, data=np.stack(mad.values), columns=feat_cols)

    median["stat"] = "median"
    mad["stat"] = "mad"
    min_["stat"] = "min"
    max_["stat"] = "max"
    count["stat"] = "count"

    stats = pd.concat([median, mad, min_, max_, count])
    stats.reset_index(inplace=True)
    stats = stats.melt(id_vars=["Metadata_Plate", "stat"], var_name="feature")
    stats = stats.pivot(
        index=["Metadata_Plate", "feature"], columns="stat", values="value"
    )
    stats.reset_index(inplace=True)
    stats["abs_coef_var"] = (
        (stats["mad"] / stats["median"]).fillna(0).abs().replace(np.inf, 0)
    )
    stats = stats.astype(
        {
            "min": np.float32,
            "max": np.float32,
            "count": np.int32,
            "median": np.float32,
            "mad": np.float32,
            "abs_coef_var": np.float32,
            "feature": "category",
        }
    )
    return stats


def compute_norm_stats(parquet_path: str, df_stats_path: str):
    """create platewise statistics for columns without nan/inf values only"""
    dframe = pd.read_parquet(parquet_path)
    logger.info("Removing nan and inf columns")
    dframe = remove_nan_infs_columns(dframe)
    dframe_stats = get_plate_stats(dframe)
    logger.info("stats done.")
    dframe_stats = add_metadata(dframe_stats)
    dframe_stats.to_parquet(df_stats_path, commpression="gzip", index=False)


def select_variant_features_polars(parquet_path, norm_stats_path, variant_feats_path):
    """
    Filtered out features that have mad == 0 or abs_coef_var>1e-3 in any plate.
    Using polar instead of pandas.
    """
    lframe = pl.scan_parquet(parquet_path)
    norm_stats = pl.scan_parquet(norm_stats_path)

    meta_col = [col for col in lframe.columns if col.startswith("Metadata_")]

    # Select variant_features
    norm_stats = norm_stats.filter(
        pl.col("mad") != 0, pl.col("abs_coef_var") > 1e-3
    ).collect()

    groups = norm_stats.group_by("Metadata_Plate").agg(pl.col("feature"))
    variant_features = set.intersection(set(*list(groups.select("feature").rows()[0])))
    # Select plates with variant features
    norm_stats = norm_stats.filter(pl.col("feature").is_in(variant_features))
    plate_list = norm_stats.select(pl.col("Metadata_Plate")).rows()
    plate_list = [x[0] for x in plate_list]
    lframe = lframe.filter(pl.col("Metadata_Plate").is_in(plate_list))
    lframe = lframe.select(pl.col(meta_col + list(variant_features)))
    lframe = lframe.collect().to_pandas()
    lframe.to_parquet(variant_feats_path, compression="gzip", index=False)


def select_variant_features(parquet_path, norm_stats_path, variant_feats_path):
    """
    Filtered out features that have mad == 0 or abs_coef_var>1e-3 in any plate.
    stats are computed using negative controls only
    """
    dframe = pd.read_parquet(parquet_path)
    norm_stats = pd.read_parquet(norm_stats_path)

    # Remove NaN and Inf
    dframe = remove_nan_infs_columns(dframe)

    # Select variant_features
    norm_stats = norm_stats.query("mad!=0 and abs_coef_var>1e-3")
    groups = norm_stats.groupby("Metadata_Plate", observed=True)["feature"]
    variant_features = set.intersection(*groups.agg(set).tolist())

    # Select plates with variant features
    norm_stats = norm_stats.query("feature in @variant_features")
    dframe = dframe.query("Metadata_Plate in @norm_stats.Metadata_Plate")

    # Filter features
    variant_features = sorted(variant_features)
    meta = dframe[find_meta_cols(dframe)]
    vals = dframe[variant_features].values
    merge_parquet(meta, vals, variant_features, variant_feats_path)


def robustmad(variant_feats_path, plate_stats_path, normalized_path):
    """Perform plate level RobustMAD (value - median)/mad"""
    meta, vals, features = split_parquet(variant_feats_path)
    plate_stats = pd.read_parquet(plate_stats_path)
    plate_stats = plate_stats.query("feature in @features")

    # get counts and sort by plate
    plates, counts = np.unique(meta["Metadata_Plate"], return_counts=True)
    ix = np.argsort(meta["Metadata_Plate"])
    meta = meta.iloc[ix]
    vals = vals[ix]

    # get mad and median matrices for MAD normalization
    mads = plate_stats.pivot(index="Metadata_Plate", columns="feature", values="mad")
    mads = mads.loc[plates, features].values
    medians = plate_stats.pivot(
        index="Metadata_Plate", columns="feature", values="median"
    )
    medians = medians.loc[plates, features].values

    # Get normalized features (epsilon = 0) for all plates that have MAD stats
    # -= and /= are inplace operations. i.e save memory
    vals -= np.repeat(medians, counts, axis=0)
    vals /= np.repeat(mads, counts, axis=0)

    merge_parquet(meta, vals, features, normalized_path)