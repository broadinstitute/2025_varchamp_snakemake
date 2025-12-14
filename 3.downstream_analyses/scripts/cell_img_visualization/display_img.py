import os
import operator
import subprocess
import pickle
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from functools import reduce
import sys

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# _PROJECT_ROOT points to 3.downstream_analyses/
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
# _REPO_ROOT points to 2025_varchamp_snakemake/ (where img_utils.py lives)
_REPO_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, ".."))

# Add the repo root to path for img_utils import
sys.path.insert(0, _REPO_ROOT)
from img_utils import *


BATCH_PROFILES = os.path.join(_REPO_ROOT, "2.snakemake_pipeline/outputs/batch_profiles/{}/profiles.parquet")
IMG_ANALYSIS_DIR = os.path.join(_REPO_ROOT, "1.image_preprocess_qc/inputs/cpg_imgs/{}/analysis")
BATCH_PROFILES_GFP_FILTERED = os.path.join(_REPO_ROOT, "2.snakemake_pipeline/outputs/classification_results/{}/profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells/gfp_adj_filtered_cells_profiles.parquet") 
GFP_INTENSITY_COLUMN = "Cells_Intensity_IntegratedIntensity_GFP" ## Cells_Intensity_MeanIntensity_GFP is another option


### STORE / READ IN THE BATCH PROFILES ###
### STORING THE PROFILES ###
# # Filter thresholds
# min_area_ratio = 0.15
# max_area_ratio = 0.3
# min_center = 50
# max_center = 1030
# # num_mad = 5
# # min_cells = 250

# batch_profiles = {}
# for bio_rep, bio_rep_batches in BIO_REP_BATCHES_DICT.items():
#     for batch_id in BIO_REP_BATCHES_DICT[bio_rep]:
#         imagecsv_dir = IMG_ANALYSIS_DIR.format(batch_id) #f"../../../8.1_upstream_analysis_runxi/2.raw_img_qc/inputs/images/{batch_id}/analysis"
#         prof_path = BATCH_PROFILES.format(batch_id)
#         # Get metadata
#         profiles = pl.scan_parquet(prof_path).select(
#             ["Metadata_well_position", "Metadata_plate_map_name", "Metadata_ImageNumber", "Metadata_ObjectNumber",
#             "Metadata_symbol", "Metadata_gene_allele", "Metadata_node_type", "Metadata_Plate",
#             "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
#             "Nuclei_AreaShape_BoundingBoxMaximum_X", "Nuclei_AreaShape_BoundingBoxMaximum_Y", 
#             "Nuclei_AreaShape_BoundingBoxMinimum_X", "Nuclei_AreaShape_BoundingBoxMinimum_Y", 
#             "Cells_AreaShape_BoundingBoxMaximum_X", "Cells_AreaShape_BoundingBoxMaximum_Y", "Cells_AreaShape_BoundingBoxMinimum_X",
#             "Cells_AreaShape_BoundingBoxMinimum_Y",	"Cells_AreaShape_Center_X",	"Cells_AreaShape_Center_Y",
#             "Cells_Intensity_MeanIntensity_GFP", "Cells_Intensity_MedianIntensity_GFP", "Cells_Intensity_IntegratedIntensity_GFP"],
#         ).collect()
#         # print(profiles["Metadata_Plate"])
    
#         # Filter based on cell to nucleus area
#         profiles = profiles.with_columns(
#                         (pl.col("Nuclei_AreaShape_Area")/pl.col("Cells_AreaShape_Area")).alias("Nucleus_Cell_Area"),
#                         pl.concat_str([
#                             "Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber",
#                             ], separator="_").alias("Metadata_CellID"),
#                 ).filter((pl.col("Nucleus_Cell_Area") > min_area_ratio) & (pl.col("Nucleus_Cell_Area") < max_area_ratio))
    
#         # Filter cells too close to image edge
#         profiles = profiles.filter(
#             ((pl.col("Nuclei_AreaShape_Center_X") > min_center) & (pl.col("Nuclei_AreaShape_Center_X") < max_center) &
#             (pl.col("Nuclei_AreaShape_Center_Y") > min_center) & (pl.col("Nuclei_AreaShape_Center_Y") < max_center)),
#         )
    
#         # Calculate mean, median and mad of gfp intensity for each allele
#         ## mean
#         means = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
#             pl.col("Cells_Intensity_MeanIntensity_GFP").mean().alias("WellIntensityMean"),
#         )
#         profiles = profiles.join(means, on=["Metadata_Plate", "Metadata_well_position"])
#         ## median
#         medians = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
#             pl.col("Cells_Intensity_MedianIntensity_GFP").median().alias("WellIntensityMedian"),
#         )
#         profiles = profiles.join(medians, on=["Metadata_Plate", "Metadata_well_position"])
#         ## mad
#         profiles = profiles.with_columns(
#             (pl.col("Cells_Intensity_MedianIntensity_GFP") - pl.col("WellIntensityMedian")).abs().alias("Abs_dev"),
#         )
#         mad = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
#             pl.col("Abs_dev").median().alias("Intensity_MAD"),
#         )
#         profiles = profiles.join(mad, on=["Metadata_Plate", "Metadata_well_position"])
    
#         # ## Threshold is 5X
#         # ## Used to be median well intensity + 5*mad implemented by Jess
#         # ## Switching to mean well intensity + 5*mad implemented by Runxi
#         # profiles = profiles.with_columns(
#         #     (pl.col("WellIntensityMedian") + num_mad*pl.col("Intensity_MAD")).alias("Intensity_upper_threshold"), ## pl.col("WellIntensityMedian")
#         #     (pl.col("WellIntensityMedian") - num_mad*pl.col("Intensity_MAD")).alias("Intensity_lower_threshold"), ## pl.col("WellIntensityMedian")
#         # )
#         # ## Filter by intensity MAD
#         # profiles = profiles.filter(
#         #     pl.col("Cells_Intensity_MeanIntensity_GFP") <= pl.col("Intensity_upper_threshold"),
#         # ).filter(
#         #     pl.col("Cells_Intensity_MeanIntensity_GFP") >= pl.col("Intensity_lower_threshold"),
#         # )
    
#         # Filter out alleles with fewer than 250 cells
#         # keep_alleles = profiles.group_by("Metadata_gene_allele").count().filter(
#         #     pl.col("count") >= min_cells,
#         #     ).select("Metadata_gene_allele").to_series().to_list()
#         # profiles = profiles.filter(pl.col("Metadata_gene_allele").is_in(keep_alleles))
    
#         # add full crop coordinates
#         profiles = profiles.with_columns(
#             (pl.col("Nuclei_AreaShape_Center_X") - 50).alias("x_low").round().cast(pl.Int16),
#             (pl.col("Nuclei_AreaShape_Center_X") + 50).alias("x_high").round().cast(pl.Int16),
#             (pl.col("Nuclei_AreaShape_Center_Y") - 50).alias("y_low").round().cast(pl.Int16),
#             (pl.col("Nuclei_AreaShape_Center_Y") + 50).alias("y_high").round().cast(pl.Int16),
#         )
    
#         # Read in all Image.csv to get ImageNumber:SiteNumber mapping and paths
#         image_dat = []
#         icfs = glob.glob(os.path.join(imagecsv_dir, "**/*Image.csv"), recursive=True)
#         for icf in tqdm(icfs):
#             fp = icf.split('/')[-2]
#             # print(fp)
#             plate, well = "-".join(fp.split("-")[:-2]), fp.split("-")[-2]
#             # print(plate, well)
#             image_dat.append(pl.read_csv(icf).select(
#                 [
#                     "ImageNumber",
#                     "Metadata_Site",
#                     "PathName_OrigDNA",
#                     "FileName_OrigDNA",
#                     "FileName_OrigGFP",
#                     ],
#                 ).with_columns(
#                 pl.lit(plate).alias("Metadata_Plate"),
#                 pl.lit(well).alias("Metadata_well_position"),
#                 ))
#         image_dat = pl.concat(image_dat).rename({"ImageNumber": "Metadata_ImageNumber"})
    
#         # Create useful filepaths
#         image_dat = image_dat.with_columns(
#             pl.col("PathName_OrigDNA").str.replace(".*cpg0020-varchamp/", "").alias("Path_root"),
#         )
    
#         image_dat = image_dat.drop([
#             "PathName_OrigDNA",
#             "FileName_OrigDNA",
#             "FileName_OrigGFP",
#             "Path_root",
#         ])
#         # print(image_dat)
    
#         # Append to profiles
#         profiles = profiles.join(image_dat, on = ["Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber"])
    
#         # Sort by allele, then image number
#         profiles = profiles.with_columns(
#             pl.concat_str(["Metadata_Plate", "Metadata_well_position", "Metadata_Site"], separator="_").alias("Metadata_SiteID"),
#             pl.col("Metadata_gene_allele").str.replace("_", "-").alias("Protein_label"),
#         )
#         profiles = profiles.sort(["Protein_label", "Metadata_SiteID"])
#         alleles = profiles.select("Protein_label").to_series().unique().to_list()
#         batch_profiles[batch_id] = profiles

# # Pickle the metadata dictionary
# _BATCH_PROF_DICT_PATH = os.path.join(_REPO_ROOT, "2.snakemake_pipeline/outputs/visualize_cells/batch_prof_dict.pkl")
# # with open(_BATCH_PROF_DICT_PATH, "wb") as f:
# #     pickle.dump(batch_profiles, f, pickle.HIGHEST_PROTOCOL)
# # To load the dictionary and DataFrames later
# with open(_BATCH_PROF_DICT_PATH, "rb") as f:
#     batch_profiles = pickle.load(f)


def plot_allele(pm, ref, var, sel_channel, plate_img_qc, auroc_df=None, site="05", ref_well=[], var_well=[], vmin=1., vmax=99., show_plot=False, imgs_dir="", output_dir=""):
    assert imgs_dir != "", "Image directory has to be input!"
    plt.clf()
    cmap = channel_to_cmap(sel_channel)
    channel = channel_dict[sel_channel]
    if auroc_df is not None:
        auroc = auroc_df.filter(pl.col("allele_0")==var)["AUROC_Mean"].mean()
    else:
        auroc = ""
    
    ## get the number of wells/images per allele
    plate_map = pm.filter(pl.col("gene_allele") == var).select("plate_map_name").to_pandas().values.flatten()
    # print(plate_map)

    wt_wells = pm.filter(pl.col("gene_allele") == ref).select("imaging_well").to_pandas().values.flatten()
    var_wells = pm.filter(pl.col("gene_allele") == var).select("imaging_well").to_pandas().values.flatten()
    if ref_well:
        wt_wells = [well for well in wt_wells if well in ref_well]
    if var_well:
        var_wells = [well for well in var_wells if well in var_well]

    pm_var = pm.filter(
        (pl.col("imaging_well").is_in(np.concatenate([wt_wells, var_wells]))),
        (pl.col("plate_map_name").is_in(plate_map))
    ).sort("node_type")

    fig, axes = plt.subplots((len(wt_wells)+len(var_wells))*2, 4, figsize=(15, (len(wt_wells)+len(var_wells))*8), sharex=True, sharey=True)
    for wt_var, pm_row in enumerate(pm_var.iter_rows(named=True)):
        well = pm_row["imaging_well"]  # Use the actual well from the current row
        if well in wt_wells:
            allele = ref
        else:
            allele = var

        for i in range(8):
            if i < 4:
                sel_plate = pm_row["imaging_plate_R1"]
            else:
                sel_plate = pm_row["imaging_plate_R2"]
                
            if "_" in sel_plate:
                batch_plate_map = sel_plate.split("_")[0]
            else:
                batch_plate_map = sel_plate
            
            batch = batch_dict[batch_plate_map]
            batch_img_dir = f'{imgs_dir}/{batch}/images'
            
            letter = well[0]
            row = letter_dict[letter]
            col = well[1:3]
            
            plate_img_dir = plate_dict[sel_plate][f"T{i%4+1}"]
            img_file = f"r{row}c{col}f{site}p01-ch{channel}sk1fk1fl1.tiff"
            # print(batch, well, plate_img_dir, img_file)

            if plate_img_qc is not None:
                img_qc_df = plate_img_qc.filter(
                    (pl.col("plate") == plate_img_dir.split("__")[0])
                    & (pl.col("well") == well)
                    & (pl.col("channel") == sel_channel)
                )
                # print(img_qc_df)
                is_bg_array = img_qc_df["is_bg"].to_numpy()
                if is_bg_array.size > 0:
                    is_bg = is_bg_array[0]
                else:
                    is_bg = True
                    
            if (os.path.exists(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}")):
                img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
            else:
                # Define your S3 path and local destination
                s3_path = f's3://cellpainting-gallery/cpg0020-varchamp/broad/images/{batch}/images/{plate_img_dir}/Images/{img_file}'
                local_path = f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}"
                # Build the aws cli command
                cmd = ['aws', 's3', 'cp', '--no-sign-request', s3_path, local_path]
                # Execute the command using subprocess
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Successfully downloaded from {s3_path} to {local_path}")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred: {e}")
                img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
            
            plot_idx = i+wt_var*4*2
            # Calculate display bounds from raw data (no normalization - display only)
            display_vmin = np.percentile(img, vmin)
            display_vmax = np.percentile(img, vmax)
            
            axes.flatten()[plot_idx].imshow(img, vmin=display_vmin, vmax=display_vmax, cmap=cmap)
            plot_label = f"{sel_channel}:{sel_plate},T{i%4+1}\nWell:{well},Site:{site}\n{allele}"
            axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
                    verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                    bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
            if is_bg:
                axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                    bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
            int_95 = str(int(round(np.percentile(img, 95))))
            axes.flatten()[plot_idx].text(0.97, 0.03, f"95th Intensity:{int_95}\nvmin:{vmin:.1f}%\nvmax:{vmax:.1f}%", color='white', fontsize=10,
                           verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                           bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
            axes.flatten()[plot_idx].axis("off")
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
    if show_plot:
        plt.show()
    
    if output_dir:
        file_name = f"{var}_{plate_map[0]}_{sel_channel}"
        if auroc:
            file_name = f"{file_name}_{auroc:.3f}"
        if ref_well:
            file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
        if var_well:
            file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
        fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
        
    plt.close(fig)


def plot_allele_single_plate(pm, variant, sel_channel, plate_img_qc, auroc_df=None, site="05", ref_well=[], var_well=[], vmin=1., vmax=99., show_plot=False, imgs_dir="", output_dir=""):
    assert imgs_dir != "", "Image directory has to be input!"
    plt.clf()
    cmap = channel_to_cmap(sel_channel)
    channel = channel_dict[sel_channel]
    if auroc_df is not None:
        auroc = auroc_df.filter(pl.col("allele_0")==variant)["AUROC_Mean"].mean()
    else:
        auroc = ""
    
    ## get the number of wells/images per allele
    plate_map = pm.filter(pl.col("gene_allele") == variant).select("plate_map_name").to_pandas().values.flatten()
    wt = variant.split("_")[0]
    wt_wells = pm.filter(pl.col("gene_allele") == wt).select("imaging_well").to_pandas().values.flatten()
    var_wells = pm.filter(pl.col("gene_allele") == variant).select("imaging_well").to_pandas().values.flatten()
    
    if ref_well:
        wt_wells = [well for well in wt_wells if well in ref_well]
    if var_well:
        var_wells = [well for well in var_wells if well in var_well]
    
    pm_var = pm.filter(
        (pl.col("imaging_well").is_in(np.concatenate([wt_wells, var_wells])))
        &(pl.col("plate_map_name").is_in(plate_map))
    ).sort(by=["gene_allele", "plate_map_name", "imaging_well"],
           descending=[True, False, False])
    # print(pm_var)
    # return None
    
    fig, axes = plt.subplots(4, 4, figsize=(15, 2*8), sharex=True, sharey=True)
    for plot_idx, pm_row in enumerate(pm_var.iter_rows(named=True)):
        well = pm_row["imaging_well"]
        allele = pm_row["gene_allele"]

        if plot_idx < 4 or (plot_idx >= 8 and plot_idx < 12):
            sel_plate = pm_row["imaging_plate_R1"]
        else:
            sel_plate = pm_row["imaging_plate_R2"]
            
        if "_" in sel_plate:
            batch_plate_map = sel_plate.split("_")[0]
        else:
            batch_plate_map = sel_plate
            
        batch = batch_dict[batch_plate_map]
        batch_img_dir = f'{imgs_dir}/{batch}/images'
        
        letter = well[0]
        row = letter_dict[letter]
        col = well[1:3]
        
        plate_img_dir = plate_dict[sel_plate]
        img_file = f"r{row}c{col}f{site}p01-ch{channel}sk1fk1fl1.tiff"

        # print(batch, well, plate_img_dir, img_file)
        # break

        if plate_img_qc is not None:
            is_bg_array = plate_img_qc.filter(
                (pl.col("plate") == plate_img_dir.split("__")[0])
                & (pl.col("well") == well)
                & (pl.col("channel") == sel_channel)
            )["is_bg"].to_numpy()
            if is_bg_array.size > 0:
                is_bg = is_bg_array[0]
            else:
                is_bg = True
                
        if (os.path.exists(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}")):
            img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
        else:
            # Define your S3 path and local destination
            s3_path = f's3://cellpainting-gallery/cpg0020-varchamp/broad/images/{batch}/images/{plate_img_dir}/Images/{img_file}'
            local_path = f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}"
            # Build the aws cli command
            cmd = ['aws', 's3', 'cp', '--no-sign-request', s3_path, local_path]
            # Execute the command using subprocess
            try:
                subprocess.run(cmd, check=True)
                print(f"Successfully downloaded from {s3_path} to {local_path}")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")
            img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
        
        # print(i, wt_var, plot_idx)
        # Calculate display bounds from raw data (no normalization - display only)
        display_vmin = np.percentile(img, vmin)
        display_vmax = np.percentile(img, vmax)
        
        axes.flatten()[plot_idx].imshow(img, vmin=display_vmin, vmax=display_vmax, cmap=cmap)
        plot_label = f"{sel_channel}:{sel_plate}\nWell:{well},Site:{site}\n{allele}"
        axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
                verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
        if is_bg:
            axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
        int_95 = str(int(round(np.percentile(img, 95))))
        axes.flatten()[plot_idx].text(0.97, 0.03, f"95th Intensity:{int_95}\nvmin:{vmin:.1f}%\nvmax:{vmax:.1f}%", color='white', fontsize=10,
                       verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                       bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
        axes.flatten()[plot_idx].axis("off")
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
    if show_plot:
        plt.show()
        
    file_name = f"{variant}_{sel_channel}"
    if auroc:
        file_name = f"{file_name}_{auroc:.3f}"
    if ref_well:
        file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
    if var_well:
        file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
        
    if output_dir:
        fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
    plt.close(fig)