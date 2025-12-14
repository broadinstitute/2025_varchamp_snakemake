"""
Cell Profiles Saver - Filter and save batch profiles with metadata

Usage:
    # Navigate to the script directory
    cd 3.downstream_analyses/scripts/cell_img_visualization/

    # Activate the varchamp conda environment and run
    source "$HOME/software/anaconda3/etc/profile.d/conda.sh" && \
    conda activate varchamp && \
    python cell_profiles_saver.py
"""

import os
import operator
import subprocess
import pickle
import glob
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from functools import reduce
from tqdm import tqdm

import sys
sys.path.append("../../..")
from img_utils import *

BATCH_PROFILES = "../../../2.snakemake_pipeline/outputs/batch_profiles/{}/profiles.parquet"
BATCH_NORM_PROFILES = "../../../2.snakemake_pipeline/outputs/batch_profiles/{}/profiles_tcdropped_filtered_var_mad.parquet"
BATCH_FILTERED_PROFILES = "../../../2.snakemake_pipeline/outputs/batch_profiles/{}/profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells.parquet"
IMG_ANALYSIS_DIR = "../../../1.image_preprocess_qc/inputs/cpg_imgs/{}/analysis"

def batch_filter_cells():
    """Main execution function for processing and saving batch profiles."""
    # Filter thresholds
    min_area_ratio = 0.15
    max_area_ratio = 0.3
    min_center = 50
    max_center = 1030
    # num_mad = 5
    # min_cells = 250

    batch_profiles = pl.DataFrame()
    for bio_rep, bio_rep_batches in BIO_REP_BATCHES_DICT.items():
        for batch_id in BIO_REP_BATCHES_DICT[bio_rep]:
            imagecsv_dir = IMG_ANALYSIS_DIR.format(batch_id)
            prof_path = BATCH_PROFILES.format(batch_id)
            # Get metadata
            profiles = pl.scan_parquet(prof_path).select(
                ["Metadata_well_position", "Metadata_plate_map_name", "Metadata_ImageNumber", "Metadata_ObjectNumber",
                "Metadata_symbol", "Metadata_gene_allele", "Metadata_node_type", "Metadata_Plate",
                "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
                "Nuclei_AreaShape_BoundingBoxMaximum_X", "Nuclei_AreaShape_BoundingBoxMaximum_Y",
                "Nuclei_AreaShape_BoundingBoxMinimum_X", "Nuclei_AreaShape_BoundingBoxMinimum_Y",
                "Cells_AreaShape_BoundingBoxMaximum_X", "Cells_AreaShape_BoundingBoxMaximum_Y", "Cells_AreaShape_BoundingBoxMinimum_X",
                "Cells_AreaShape_BoundingBoxMinimum_Y",	"Cells_AreaShape_Center_X",	"Cells_AreaShape_Center_Y",
                "Cells_Intensity_MeanIntensity_GFP", "Cells_Intensity_MedianIntensity_GFP", "Cells_Intensity_IntegratedIntensity_GFP"],
            ).collect()
            # print(profiles["Metadata_Plate"])

            # Filter based on cell to nucleus area
            profiles = profiles.with_columns(
                            (pl.col("Nuclei_AreaShape_Area")/pl.col("Cells_AreaShape_Area")).alias("Nucleus_Cell_Area"),
                            pl.concat_str([
                                "Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber",
                            ], separator="_").alias("Metadata_CellID"),
                    ).filter((pl.col("Nucleus_Cell_Area") > min_area_ratio) & (pl.col("Nucleus_Cell_Area") < max_area_ratio))

            # Filter cells too close to image edge
            profiles = profiles.filter(
                ((pl.col("Nuclei_AreaShape_Center_X") > min_center) & (pl.col("Nuclei_AreaShape_Center_X") < max_center) &
                (pl.col("Nuclei_AreaShape_Center_Y") > min_center) & (pl.col("Nuclei_AreaShape_Center_Y") < max_center)),
            )

            # Calculate mean, median and mad of gfp intensity for each allele
            ## mean
            means = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
                pl.col("Cells_Intensity_MeanIntensity_GFP").mean().alias("WellIntensityMean"),
            )
            profiles = profiles.join(means, on=["Metadata_Plate", "Metadata_well_position"])
            ## median
            medians = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
                pl.col("Cells_Intensity_MedianIntensity_GFP").median().alias("WellIntensityMedian"),
            )
            profiles = profiles.join(medians, on=["Metadata_Plate", "Metadata_well_position"])
            ## mad
            profiles = profiles.with_columns(
                (pl.col("Cells_Intensity_MedianIntensity_GFP") - pl.col("WellIntensityMedian")).abs().alias("Abs_dev"),
            )
            mad = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
                pl.col("Abs_dev").median().alias("Intensity_MAD"),
            )
            profiles = profiles.join(mad, on=["Metadata_Plate", "Metadata_well_position"])

            # ## Threshold is 5X
            # ## Used to be median well intensity + 5*mad implemented by Jess
            # ## Switching to mean well intensity + 5*mad implemented by Runxi
            # profiles = profiles.with_columns(
            #     (pl.col("WellIntensityMedian") + num_mad*pl.col("Intensity_MAD")).alias("Intensity_upper_threshold"), ## pl.col("WellIntensityMedian")
            #     (pl.col("WellIntensityMedian") - num_mad*pl.col("Intensity_MAD")).alias("Intensity_lower_threshold"), ## pl.col("WellIntensityMedian")
            # )
            # ## Filter by intensity MAD
            # profiles = profiles.filter(
            #     pl.col("Cells_Intensity_MeanIntensity_GFP") <= pl.col("Intensity_upper_threshold"),
            # ).filter(
            #     pl.col("Cells_Intensity_MeanIntensity_GFP") >= pl.col("Intensity_lower_threshold"),
            # )
            # Filter out alleles with fewer than 250 cells
            # keep_alleles = profiles.group_by("Metadata_gene_allele").count().filter(
            #     pl.col("count") >= min_cells,
            #     ).select("Metadata_gene_allele").to_series().to_list()
            # profiles = profiles.filter(pl.col("Metadata_gene_allele").is_in(keep_alleles))

            # add full crop coordinates
            profiles = profiles.with_columns(
                (pl.col("Nuclei_AreaShape_Center_X") - 50).alias("x_low").round().cast(pl.Int16),
                (pl.col("Nuclei_AreaShape_Center_X") + 50).alias("x_high").round().cast(pl.Int16),
                (pl.col("Nuclei_AreaShape_Center_Y") - 50).alias("y_low").round().cast(pl.Int16),
                (pl.col("Nuclei_AreaShape_Center_Y") + 50).alias("y_high").round().cast(pl.Int16),
            )

            # Read in all Image.csv to get ImageNumber:SiteNumber mapping and paths
            image_dat = []
            icfs = glob.glob(os.path.join(imagecsv_dir, "**/*Image.csv"), recursive=True)
            for icf in tqdm(icfs):
                fp = icf.split('/')[-2]
                # print(fp)
                plate, well = "-".join(fp.split("-")[:-2]), fp.split("-")[-2]
                # print(plate, well)
                image_dat.append(pl.read_csv(icf).select(
                    [
                        "ImageNumber",
                        "Metadata_Site",
                        "PathName_OrigDNA",
                        "FileName_OrigDNA",
                        "FileName_OrigGFP",
                        ],
                    ).with_columns(
                    pl.lit(plate).alias("Metadata_Plate"),
                    pl.lit(well).alias("Metadata_well_position"),
                    ))
            image_dat = pl.concat(image_dat).rename({"ImageNumber": "Metadata_ImageNumber"})

            # Create useful filepaths
            image_dat = image_dat.with_columns(
                pl.col("PathName_OrigDNA").str.replace(".*cpg0020-varchamp/", "").alias("Path_root"),
            )

            image_dat = image_dat.drop([
                "PathName_OrigDNA",
                "FileName_OrigDNA",
                "FileName_OrigGFP",
                "Path_root",
            ])
            # print(image_dat)

            # Append to profiles
            profiles = profiles.join(image_dat, on = ["Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber"])

            # Sort by allele, then image number
            profiles = profiles.with_columns(
                pl.concat_str(["Metadata_Plate", "Metadata_well_position", "Metadata_Site"], separator="_").alias("Metadata_SiteID"),
                pl.col("Metadata_gene_allele").str.replace("_", "-").alias("Protein_label"),
            )
            profiles = profiles.sort(["Protein_label", "Metadata_SiteID"])
            alleles = profiles.select("Protein_label").to_series().unique().to_list()
            batch_profiles = pl.concat([batch_profiles, profiles])

    batch_profiles.write_parquet("../../../2.snakemake_pipeline/outputs/batch_prof_filtered_metadata.parquet", compression="zstd")


def cast_numerics_to_f64(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col(c).cast(pl.Float64) 
        for c, dtype in df.schema.items() 
        if dtype in (pl.Int64, pl.Int32, pl.Float32)
    ])


def batch_mad_feat_filtered_cells():
    """Load the filtered batch profiles with metadata."""
    img_well_qc_sum_df = pl.read_parquet(IMG_QC_SUM_PARQUET_FILE)

    all_feat_profiles, filtered_feat_profiles = pl.DataFrame(), pl.DataFrame()
    for bio_rep, bio_rep_batches in BIO_REP_BATCHES_DICT.items():
        for batch_id in bio_rep_batches:
            print(f"Processing batch {batch_id}...")
            batch_alleles_filtered = (
                pl.scan_parquet(
                    BATCH_FILTERED_PROFILES.format(batch_id)
                ).with_columns(
                    pl.concat_str([
                        "Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber",
                    ], separator="_").alias("Metadata_CellID")
                )
                .select([
                    "Metadata_CellID"
                ])
            )
            batch_alleles = (
                pl.scan_parquet(
                    BATCH_NORM_PROFILES.format(batch_id)
                ).with_columns(
                    pl.concat_str([
                        "Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber",
                    ], separator="_").alias("Metadata_CellID")
                )
            )
            batch_alleles_filtered = batch_alleles_filtered.join(
                batch_alleles,
                on="Metadata_CellID",
                how="inner",
            )
            ## DNA valid well QC
            img_well_qc_sum = img_well_qc_sum_df.filter(
                pl.col("Metadata_Bio_Batch")==bio_rep
            )
            img_well_qc_sum_dapi = img_well_qc_sum.filter(
                pl.col("channel")=="DNA"
            ).with_columns(
                pl.col("plate").alias("Metadata_Plate"),
                pl.col("well").alias("Metadata_Well"),
            )

            batch_alleles_filtered = batch_alleles_filtered.join(
                img_well_qc_sum_dapi.select(
                    pl.col(["Metadata_Plate","Metadata_Well","is_bg"])
                ).lazy(),
                on=["Metadata_Plate","Metadata_Well"],
                how="left"
            ).filter(
                ~pl.col("is_bg")
            ).drop("is_bg")

            ## Non-filtered cells
            all_feat_profiles = pl.concat(
                [cast_numerics_to_f64(all_feat_profiles), cast_numerics_to_f64(batch_alleles.collect())],
                how="diagonal_relaxed"
            )
            ## Filtered cells
            filtered_feat_profiles = pl.concat(
                [cast_numerics_to_f64(filtered_feat_profiles), cast_numerics_to_f64(batch_alleles_filtered.collect())],
                how="diagonal_relaxed"
            )

    all_feat_profiles.write_parquet("../../../2.snakemake_pipeline/outputs/ref_var_var_mad_all_feats_all_cells.parquet", compression="zstd")
    filtered_feat_profiles.write_parquet("../../../2.snakemake_pipeline/outputs/ref_var_var_mad_all_feats_filtered_cells.parquet", compression="zstd")
        

if __name__ == "__main__":
    # batch_filter_cells()
    batch_mad_feat_filtered_cells()