"""Perform correction before feature selection"""
from tqdm.contrib.concurrent import thread_map

import pandas as pd
import polars as pl

def subtract_well_mean_polar(input_path: str, output_path: str):
    """Subtract the mean of each feature per well position using polar."""
    lf = pl.scan_parquet(input_path)
    feature_cols = [i for i in lf.columns if "Metadata_" not in i]
    lf = lf.with_columns(pl.col(feature_cols) - pl.mean(feature_cols).over("Metadata_Well"))
    df_well_corrected = lf.collect().to_pandas()
    df_well_corrected.to_parquet(output_path, compression="gzip")

def subtract_well_mean(input_path: str, output_path: str, parallel: bool = True):
    """Subtract the mean of each feature per well position."""
    ann_df = pd.read_parquet(input_path)
    feature_cols = ann_df.filter(regex="^(?!Metadata_)").columns.to_list()

    if parallel:
        def subtract_well_mean_parallel_helper(feature):
            return {
                feature: ann_df[feature] - ann_df.groupby("Metadata_Well")[feature].mean()
            }

        results = thread_map(subtract_well_mean_parallel_helper, feature_cols)

        for res in results:
            ann_df.update(pd.DataFrame(res))
    else:
        ann_df[feature_cols] = ann_df.groupby("Metadata_Well")[feature_cols].transform(
            lambda x: x - x.mean()
        )

    ann_df.to_parquet(output_path, index=False)
