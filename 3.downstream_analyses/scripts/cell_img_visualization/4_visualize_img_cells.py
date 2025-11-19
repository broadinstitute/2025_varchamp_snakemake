import os
import glob
import polars as pl
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread
from tqdm import tqdm
import re
import sys
import subprocess
import pickle


sys.path.append("../..")
from img_utils import *
from display_img import *

OUT_IMG_DIR = f"../../2.snakemake_pipeline/outputs/visualize_imgs"
OUT_CELL_DIR = f"../../2.snakemake_pipeline/outputs/visualize_cells"
BATCH_PROFILES = "../../2.snakemake_pipeline/outputs/batch_profiles/{}/profiles.parquet" 
IMG_ANALYSIS_DIR = "../../1.image_preprocess_qc/inputs/cpg_imgs/{}/analysis"


def get_allele_batch(allele, score_df):
    return score_df.filter(pl.col("gene_allele")==allele)["Metadata_Bio_Batch"].to_list()[0]


def save_allele_imgs(variant, feat, auroc_df, allele_meta_df_dict, img_well_qc_sum_dict, display=False, save_img=False):
    bio_rep = get_allele_batch(variant, auroc_df)
    auroc_df_batch = auroc_df.with_columns(
        pl.col(f"AUROC_Mean_{feat}").alias("AUROC_Mean"),
        pl.col(f"gene_allele").alias("allele_0")
    )
    ref_allele = variant.split("_")[0]
    ref_wells = allele_meta_df_dict[bio_rep].filter(pl.col("gene_allele")==ref_allele)["imaging_well"].to_list()
    var_wells = allele_meta_df_dict[bio_rep].filter(pl.col("gene_allele")==variant)["imaging_well"].to_list()
    target_file = [] #[f for f in os.listdir(f"{OUT_IMG_DIR}/{bio_rep}") if f.startswith(f"{variant}_{feat}")]
    if target_file:
        print(target_file, "exists.")
        output_dir = ""
        if not display:
            return None

    if save_img:
        output_dir = f"{OUT_IMG_DIR}/{bio_rep}"
        print(f"Img output at {output_dir}")
    else:
        output_dir = ""

    if bio_rep != "2024_12_Batch_11-12":
        if len(ref_wells)==1 and len(var_wells)==1:
            plot_allele(allele_meta_df_dict[bio_rep],
                            variant=variant, sel_channel=feat, 
                            auroc_df=auroc_df_batch, 
                            plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                            site="05", max_intensity=0.99, 
                            display=display,
                            imgs_dir=TIFF_IMGS_DIR, 
                            output_dir=output_dir)
        else:
            for ref_well in ref_wells:
                for var_well in var_wells:
                    plot_allele(allele_meta_df_dict[bio_rep],
                                variant=variant, sel_channel=feat, 
                                auroc_df=auroc_df_batch, 
                                plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                                site="05", max_intensity=0.99, 
                                display=display,
                                ref_well=[ref_well], 
                                var_well=[var_well],
                                imgs_dir=TIFF_IMGS_DIR, 
                                output_dir=output_dir)
    else:
        if len(ref_wells)==4 and len(var_wells)==4:
            plot_allele_single_plate(allele_meta_df_dict[bio_rep], ##.filter(pl.col("plate_map_name").str.contains("B13")
                                     variant=variant, sel_channel=feat, 
                                     auroc_df=auroc_df_batch, 
                                     plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                                     site="05", max_intensity=0.99, 
                                     display=display,
                                     imgs_dir=TIFF_IMGS_DIR, 
                                     output_dir=output_dir)
        else:
            ref_wells_idx = len(ref_wells) // 4
            var_wells_idx = len(var_wells) // 4
            for rw_idx in range(ref_wells_idx):
                for vw_idx in range(var_wells_idx):
                    plot_allele_single_plate(allele_meta_df_dict[bio_rep], ##.filter(pl.col("plate_map_name").str.contains("B13")
                                     variant=variant, sel_channel=feat, 
                                     auroc_df=auroc_df_batch, 
                                     plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                                     site="05", max_intensity=0.99,
                                     ref_well=ref_wells[rw_idx*4:rw_idx*4+4],
                                     var_well=var_wells[vw_idx*4:vw_idx*4+4],
                                     display=display,
                                     imgs_dir=TIFF_IMGS_DIR, 
                                     output_dir=output_dir)


def save_allele_cell_imgs(variant, feat, batch_profile_dict, auroc_df, allele_meta_df_dict, img_well_qc_sum_dict, display=False, save_img=False):
    bio_rep = get_allele_batch(variant, auroc_df)
    auroc_df_batch = auroc_df.with_columns(
        pl.col(f"AUROC_Mean_{feat}").alias("AUROC_Mean"),
        pl.col(f"gene_allele").alias("allele_0")
    )
    ref_allele = variant.split("_")[0]
    ref_wells = allele_meta_df_dict[bio_rep].filter(pl.col("gene_allele")==ref_allele)["imaging_well"].to_list()
    var_wells = allele_meta_df_dict[bio_rep].filter(pl.col("gene_allele")==variant)["imaging_well"].to_list()
    target_file = [f for f in os.listdir(f"{OUT_CELL_DIR}/{bio_rep}") if f.startswith(f"{variant}_{feat}")]
    if target_file:
        print(target_file, "exists.")
        output_dir = ""
        if not display:
            return None

    if save_img:
        output_dir = f"{OUT_CELL_DIR}/{bio_rep}"
        print(f"Img output at {output_dir}")
    else:
        output_dir = ""

    if bio_rep != "2024_12_Batch_11-12":
        if len(ref_wells)==1 and len(var_wells)==1:
            plot_allele_cell(allele_meta_df_dict[bio_rep],
                             variant=variant, sel_channel=feat,
                             batch_profile_dict=batch_profile_dict,
                             auroc_df=auroc_df_batch, 
                             plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                             site="05", max_intensity=0.99, 
                             display=display,
                            imgs_dir=TIFF_IMGS_DIR, 
                            output_dir=output_dir)
        else:
            for ref_well in ref_wells:
                for var_well in var_wells:
                    plot_allele_cell(allele_meta_df_dict[bio_rep],
                                     variant=variant, 
                                     sel_channel=feat,
                                     batch_profile_dict=batch_profile_dict,
                                     auroc_df=auroc_df_batch, 
                                     plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                                     site="05", max_intensity=0.99, 
                                     display=display,
                                     ref_well=[ref_well], 
                                     var_well=[var_well],
                                     imgs_dir=TIFF_IMGS_DIR, 
                                     output_dir=output_dir)
    else:
        if len(ref_wells)==4 and len(var_wells)==4:
            plot_allele_cell_single_plate(allele_meta_df_dict[bio_rep], ##.filter(pl.col("plate_map_name").str.contains("B13")
                                     variant=variant, sel_channel=feat,
                                     batch_profile_dict=batch_profile_dict,
                                     auroc_df=auroc_df_batch, 
                                     plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                                     site="05", max_intensity=0.99, 
                                     display=display,
                                     imgs_dir=TIFF_IMGS_DIR, 
                                     output_dir=output_dir)
        else:
            ref_wells_idx = len(ref_wells) // 4
            var_wells_idx = len(var_wells) // 4
            for rw_idx in range(ref_wells_idx):
                for vw_idx in range(var_wells_idx):
                    plot_allele_cell_single_plate(allele_meta_df_dict[bio_rep], ##.filter(pl.col("plate_map_name").str.contains("B13")
                                                  variant=variant, sel_channel=feat,
                                                  batch_profile_dict=batch_profile_dict,
                                                  auroc_df=auroc_df_batch, 
                                                  plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                                                  site="05", max_intensity=0.99,
                                                  ref_well=ref_wells[rw_idx*4:rw_idx*4+4],
                                                  var_well=var_wells[vw_idx*4:vw_idx*4+4],
                                                  display=display,
                                                  imgs_dir=TIFF_IMGS_DIR, 
                                                  output_dir=output_dir)


def main():
    allele_meta_df, img_well_qc_sum_df = pl.DataFrame(), pl.DataFrame()
    allele_meta_df_dict, img_well_qc_sum_dict = {}, {}

    for bio_rep, bio_rep_batches in BIO_REP_BATCHES_DICT.items():
        for batch_id in bio_rep_batches:
            allele_meta_df_batch = pl.DataFrame()
            platemaps = [file for file in os.listdir(PLATEMAP_DIR.format(batch_id=batch_id)) if file.endswith(".txt")]
            for platemap in platemaps:
                platemap_df = pl.read_csv(os.path.join(PLATEMAP_DIR.format(batch_id=batch_id), platemap), separator="\t", infer_schema_length=100000)
                allele_meta_df_batch = pl.concat([allele_meta_df_batch, 
                                            platemap_df.filter((pl.col("node_type").is_not_null()))], # (~pl.col("node_type").is_in(["TC","NC","PC"]))&
                                            how="diagonal_relaxed").sort("plate_map_name")
                allele_meta_df_batch = allele_meta_df_batch.with_columns(pl.col("plate_map_name").alias("plate_map")) ## str.split('_').list.get(0).
                # display(allele_meta_df.head())
            allele_meta_df = pl.concat([
                allele_meta_df,
                allele_meta_df_batch
            ], how="diagonal_relaxed")#.sort("plate_map_name") ## (~pl.col("node_type").is_in(["TC","NC","PC"]))&
        allele_meta_df_dict[bio_rep] = allele_meta_df_batch

        img_well_qc_sum = pl.read_csv(f"{IMGS_QC_METRICS_DIR}/{bio_rep}/plate-well-level_img_qc_sum.csv")
        img_well_qc_sum = img_well_qc_sum.with_columns(
            pl.col("channel").replace("DAPI", "DNA").alias("channel")
        )
        img_well_qc_sum_morph = img_well_qc_sum.filter(pl.col("channel")!="GFP")
        img_well_qc_sum_morph = img_well_qc_sum_morph.group_by(["plate","well"]).agg(
            pl.col("is_bg").max().alias("is_bg"),
            pl.col("s2n_ratio").mean().alias("s2n_ratio")
        ).with_columns(pl.lit("Morph").alias("channel"))
        img_well_qc_sum = pl.concat([
            img_well_qc_sum.select(pl.col(["plate","well","channel","is_bg","s2n_ratio"])),
            img_well_qc_sum_morph.select(pl.col(["plate","well","channel","is_bg","s2n_ratio"])),
        ], how="vertical_relaxed")
        img_well_qc_sum_dict[bio_rep] = img_well_qc_sum

    auroc_df = pl.read_csv(f"/home/shenrunx/igvf/varchamp/2025_laval_submitted/2_individual_assay_results/imaging/3_outputs/imaging_analyses_mislocalization_summary_clinvar.csv", 
                           infer_schema_length=100000)
    
    auroc_df_benign_gene = auroc_df.filter(
        (pl.col("Altered_95th_perc_both_batches_GFP")) & (pl.col("clinvar_clnsig_clean")=="2_Benign")
    )["Gene"].unique()
    
    allele_list = auroc_df.filter(
        pl.col("Gene").is_in(auroc_df_benign_gene)
    )["gene_allele"].unique()
    
    # for variant in tqdm(allele_list):
    #     for feat in ["GFP"]:
    #         save_allele_imgs(variant, feat, auroc_df, allele_meta_df_dict, img_well_qc_sum_dict, display=False, save_img=True)

    # To load the dictionary and DataFrames later
    with open("../../2.snakemake_pipeline/outputs/visualize_cells/batch_prof_dict.pkl", "rb") as f:
        batch_profiles = pickle.load(f)

    feat = "GFP"
    for variant in tqdm(allele_list):
        save_allele_cell_imgs(variant, "GFP", batch_profiles, auroc_df, allele_meta_df_dict, img_well_qc_sum_dict, save_img=True)


if __name__=="__main__":
    main()