import os
import operator
import subprocess
import pickle
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from functools import reduce
import sys
sys.path.append("../..")
from img_utils import *


BATCH_PROFILES = "../../2.snakemake_pipeline/outputs/batch_profiles/{}/profiles.parquet" 
IMG_ANALYSIS_DIR = "../../1.image_preprocess_qc/inputs/cpg_imgs/{}/analysis"
BATCH_PROFILES_GFP_FILTERED = "../../2.snakemake_pipeline/outputs/classification_results/{}/profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells/gfp_adj_filtered_cells_profiles.parquet" 
GFP_INTENSITY_COLUMN = "Cells_Intensity_IntegratedIntensity_GFP" ## Cells_Intensity_MeanIntensity_GFP is another option


# To load the dictionary and DataFrames later
with open("../../2.snakemake_pipeline/outputs/visualize_cells/batch_prof_dict.pkl", "rb") as f:
    batch_profiles = pickle.load(f)

allele_meta_df = pl.read_csv(IMG_METADATA_FILE, infer_schema_length=10000)
img_well_qc_sum_df = pl.read_csv(IMG_QC_SUM_DF_FILE, infer_schema_length=10000)

with open(IMG_METADATA_DICT_FILE, "rb") as f:
    allele_meta_df_dict = pickle.load(f)

with open(IMG_QC_SUM_DICT_FILE, "rb") as f:
    img_well_qc_sum_dict = pickle.load(f)


def get_allele_batch(allele, score_df):
    return score_df.filter(pl.col("gene_allele")==allele)["Metadata_Bio_Batch"].to_list()[0]
    

def crop_allele(allele: str, profile_df: pl.DataFrame, meta_plate: str, rep: str="", well: str="", site: str="") -> None:
    """Crop images and save metadata as numpy arrays for one allele.

    Parameters
    ----------
    allele : String
        Name of allele to process
    profile_df : String
        Dataframe with pathname and cell coordinates
    img_dir : String
        Directory where all images are stored
    out_dir : String
        Directory where numpy arrays should be saved
    """
    allele_df = profile_df.filter(
        (pl.col("Metadata_gene_allele")==allele) &
        (pl.col("Metadata_plate_map_name").str.contains(meta_plate))
    )
    
    if site:
        allele_df = profile_df.filter(
                (pl.col("Metadata_Site")==int(site))
        )

    if rep:
        allele_df = allele_df.filter(
            (pl.col("Metadata_Plate").str.contains(rep))
        )
        
    if well:
        allele_df = allele_df.filter(
            (pl.col("Metadata_well_position").str.contains(well))
        )

    return allele_df


# Compute distances from edges and find the most centered well
def compute_distance_cell(row, col, edge=1080):
    return min(row - 1, edge - row, col - 1, edge - col)  # Distance from nearest edge
    

def plot_allele_cell(pm, variant, sel_channel, batch_profile_dict, auroc_df, plate_img_qc, site="05", ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir=TIFF_IMGS_DIR, output_dir=""):
    cmap = channel_to_cmap(sel_channel)
    channel = channel_dict[sel_channel]
    auroc = auroc_df.filter(pl.col("allele_0")==variant)["AUROC_Mean"].mean()

    # if os.path.exists(os.path.join(output_dir, f"{variant}_{sel_channel}_{auroc:.3f}_cells.png")):
    #     print(f"Image for {variant} already exists.")
        # return None

    ## get the number of wells/images per allele
    wt = variant.split("_")[0]
    wt_wells = pm.filter(pl.col("gene_allele") == wt).select("imaging_well").to_pandas().values.flatten()
    var_wells = pm.filter(pl.col("gene_allele") == variant).select("imaging_well").to_pandas().values.flatten()
    plate_map = pm.filter(pl.col("gene_allele") == variant).select("plate_map_name").to_pandas().values.flatten()

    if ref_well:
        wt_wells = [well for well in wt_wells if well in ref_well]
    if var_well:
        var_wells = [well for well in var_wells if well in var_well]
    pm_var = pm.filter((pl.col("imaging_well").is_in(np.concatenate([wt_wells, var_wells])))&(pl.col("plate_map_name").is_in(plate_map))).sort("node_type")
    
    plt.clf()
    fig, axes = plt.subplots((len(wt_wells)+len(var_wells))*2, 4, figsize=(15, 16), sharex=True, sharey=True)
    for wt_var, pm_row in enumerate(pm_var.iter_rows(named=True)):
        # print(pm_row)
        if pm_row["node_type"] == "allele":
            well = var_wells[0]
            allele = variant
        else:
            well = wt_wells[0]
            allele = wt
        for i in range(8):
            plot_idx = i+wt_var*4*2
            if i < 4:
                sel_plate = pm_row["imaging_plate_R1"]
            else:
                sel_plate = pm_row["imaging_plate_R2"]

            batch = batch_dict[sel_plate.split("_")[0]]
            batch_img_dir = f'{imgs_dir}/{batch}/images'
            letter =  well[0]
            row, col = letter_dict[letter], well[1:3]
            # print(i, allele, well)
            plate_img_dir = plate_dict[sel_plate][f"T{i%4+1}"]
            img_file = f"r{row}c{col}f{site}p01-ch{channel}sk1fk1fl1.tiff"
            img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
            # print(np.percentile(img, 99) / np.median(img), np.percentile(img, 99) / np.percentile(img, 25))
            
            if plate_img_qc is not None:
                is_bg = plate_img_qc.filter((pl.col("plate") == plate_img_dir.split("__")[0]) & (pl.col("well") == well) & (pl.col("channel") == sel_channel))["is_bg"].to_numpy()[0]

            ## Draw cells
            cell_allele_coord_df = crop_allele(allele, batch_profile_dict[batch], sel_plate.split("P")[0], rep=f"T{i%4+1}", site=site[-1])      
            cell_allele_coord_df = cell_allele_coord_df.with_columns(
                pl.struct("Cells_AreaShape_Center_X", "Cells_AreaShape_Center_Y") # 'Nuclei_AreaShape_Center_X', 'Nuclei_AreaShape_Center_Y'
                .map_elements(lambda x: compute_distance_cell(x['Cells_AreaShape_Center_X'], x['Cells_AreaShape_Center_Y']), return_dtype=pl.Float32).cast(pl.Int16)
                .alias('dist2edge')
            ).sort(by=["dist2edge","Cells_AreaShape_Area"], descending=[True,True]).filter(pl.col("Cells_AreaShape_Area")>5000)

            if cell_allele_coord_df.is_empty():
                axes.flatten()[plot_idx].text(0.97, 0.97, "No high-quality\ncell available", color='black', fontsize=12,
                    verticalalignment='top', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                    bbox=dict(facecolor='white', alpha=0.3, linewidth=2)
                )
                # axes.flatten()[plot_idx].set_visible(False)
                x, y = img.shape[0] // 2, img.shape[1] // 2
                img_sub = img[
                    y-64:y+64, x-64:x+64
                ]
                axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)  ## np.percentile(img_sub, max_intensity*100)
            else:
                plot_yet = 0
                flag = False
                for cell_idx, cell_row in enumerate(cell_allele_coord_df.iter_rows(named=True)):
                    if plot_yet:
                        break
                    x, y = int(cell_row["Nuclei_AreaShape_Center_X"]), int(cell_row["Nuclei_AreaShape_Center_Y"])
                    # x, y = int(cell_allele_coord_df["Cells_AreaShape_Center_X"].to_numpy()[0]), int(cell_allele_coord_df["Cells_AreaShape_Center_Y"].to_numpy()[0])
                    ## flip the x and y for visualization
                    img_sub = img[
                        y-64:y+64, x-64:x+64
                    ]
                    ## skip the subimage due to poor cell quality
                    if img_sub.shape[0] == 0 or img_sub.shape[1] == 0 or np.percentile(img_sub, 90) <= np.median(img) or np.var(img_sub) < 1e4 or np.percentile(img_sub, 99) / np.percentile(img_sub, 25) < 2:
                        continue
                        
                    axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)  ## np.percentile(img_sub, max_intensity*100)
                    plot_label = f"{sel_channel}:{sel_plate},T{i%4+1}\nWell:{well},Site:{site}\n{allele}"
                    axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
                            verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                            bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
                    if is_bg:
                        axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
                            verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                            bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
                    # if flag:
                    #     axes.flatten()[plot_idx].text(0.97, 0.97, "Poor Quality FLAG", color='red', fontsize=10,
                    #         verticalalignment='top', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                    #         bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
                    int_95 = str(int(round(np.percentile(img_sub, 95))))
                    axes.flatten()[plot_idx].text(0.95, 0.05, f"95th Intensity:{int_95}", color='white', fontsize=10,
                                verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                                bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
                    plot_yet = 1
                    
            axes.flatten()[plot_idx].axis("off")
            
    fig.tight_layout()
    fig.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
    if display:
        plt.show()

    if output_dir:
        file_name = f"{variant}_{sel_channel}_cells"
        if auroc:
            file_name = f"{file_name}_{auroc:.3f}"
        if ref_well:
            file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
        if var_well:
            file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
        fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
        
    plt.close(fig)


def plot_allele_cell_by_id(pm, variant, sel_channel, batch_profile_dict, auroc_df, plate_img_qc, compartment="Cells", dim=64, cell_ids=[], ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir=TIFF_IMGS_DIR, output_dir=""):
    print("cell ids filtered")
    cmap = channel_to_cmap(sel_channel)
    channel = channel_dict[sel_channel]
    auroc = auroc_df.filter(pl.col("allele_0")==variant)["AUROC_Mean"].mean()

    # if os.path.exists(os.path.join(output_dir, f"{variant}_{sel_channel}_{auroc:.3f}_cells.png")):
    #     print(f"Image for {variant} already exists.")
        # return None

    ## get the number of wells/images per allele
    wt = variant.split("_")[0]
    wt_wells = pm.filter(pl.col("gene_allele") == wt).select("imaging_well").to_pandas().values.flatten()
    var_wells = pm.filter(pl.col("gene_allele") == variant).select("imaging_well").to_pandas().values.flatten()
    plate_map = pm.filter(pl.col("gene_allele") == variant).select("plate_map_name").to_pandas().values.flatten()

    if ref_well:
        wt_wells = [well for well in wt_wells if well in ref_well]
    if var_well:
        var_wells = [well for well in var_wells if well in var_well]
    pm_var = pm.filter((pl.col("imaging_well").is_in(np.concatenate([wt_wells, var_wells])))&(pl.col("plate_map_name").is_in(plate_map))).sort("node_type")
    
    plt.clf()
    fig, axes = plt.subplots((len(wt_wells)+len(var_wells))*2, 4, figsize=(15, 16), sharex=True, sharey=True)
    for wt_var, pm_row in enumerate(pm_var.iter_rows(named=True)):
        # print(pm_row)
        if pm_row["node_type"] == "allele":
            well = var_wells[0]
            allele = variant
        else:
            well = wt_wells[0]
            allele = wt
        for i in range(8):
            plot_idx = i+wt_var*4*2
            if i < 4:
                sel_plate = pm_row["imaging_plate_R1"]
            else:
                sel_plate = pm_row["imaging_plate_R2"]

            batch = batch_dict[sel_plate.split("_")[0]]
            batch_img_dir = f'{imgs_dir}/{batch}/images'
            letter =  well[0]
            row, col = letter_dict[letter], well[1:3]
            # print(i, allele, well)
            plate_img_dir = plate_dict[sel_plate][f"T{i%4+1}"]
            
            # print(np.percentile(img, 99) / np.median(img), np.percentile(img, 99) / np.percentile(img, 25))
            
            if plate_img_qc is not None:
                is_bg = plate_img_qc.filter((pl.col("plate") == plate_img_dir.split("__")[0]) & (pl.col("well") == well) & (pl.col("channel") == sel_channel))["is_bg"].to_numpy()[0]

            ## Draw cells
            
            cell_allele_coord_df = crop_allele(allele, batch_profile_dict[batch], sel_plate.split("P")[0], rep=f"T{i%4+1}")
            # print("No filtered:", allele, sel_plate, f"T{i%4+1}", cell_allele_coord_df.shape)
            cell_allele_coord_df = cell_allele_coord_df.with_columns(
                pl.struct("Cells_AreaShape_Center_X", "Cells_AreaShape_Center_Y") # 'Nuclei_AreaShape_Center_X', 'Nuclei_AreaShape_Center_Y'
                .map_elements(lambda x: compute_distance_cell(x['Cells_AreaShape_Center_X'], x['Cells_AreaShape_Center_Y']), return_dtype=pl.Float32).cast(pl.Int16)
                .alias('dist2edge'),
                pl.concat_str(
                    [
                        "Metadata_Plate",
                        "Metadata_well_position",
                        "Metadata_ImageNumber",
                        "Metadata_ObjectNumber",
                    ],
                    separator="_",
                ).alias("Metadata_CellID")
            )#.sort(by=["dist2edge","Cells_AreaShape_Area"], descending=[True,True]).filter(pl.col("Cells_AreaShape_Area")>5000)
            # print("No filtered:", cell_allele_coord_df.shape)
            if cell_ids:
                cell_allele_coord_df = cell_allele_coord_df.filter(pl.col("Metadata_CellID").is_in(cell_ids))
                # print(sel_plate, cell_allele_coord_df.shape)

            if cell_allele_coord_df.is_empty():
                axes.flatten()[plot_idx].text(0.97, 0.97, "No high-quality\ncell available", color='black', fontsize=12,
                    verticalalignment='top', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                    bbox=dict(facecolor='white', alpha=0.3, linewidth=2)
                )
                # axes.flatten()[plot_idx].set_visible(False)
                x, y = img.shape[0] // 2, img.shape[1] // 2
                img_sub = img[
                    y-dim//2:y+dim//2, x-dim//2:x+dim//2
                ]
                axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)  ## np.percentile(img_sub, max_intensity*100)
            else:
                plot_yet = 0
                flag = False
                for cell_idx, cell_row in enumerate(cell_allele_coord_df.iter_rows(named=True)):
                    if plot_yet:
                        break
                    
                    site = f"0{cell_row['Metadata_Site']}"
                    img_file = f"r{row}c{col}f{site}p01-ch{channel}sk1fk1fl1.tiff"
                    img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)

                    x, y = int(cell_row["Nuclei_AreaShape_Center_X"]), int(cell_row["Nuclei_AreaShape_Center_Y"])
                    if compartment == "Cells":
                        x_min, x_max = int(cell_row["Cells_AreaShape_BoundingBoxMinimum_X"]), int(cell_row["Cells_AreaShape_BoundingBoxMaximum_X"])
                        y_min, y_max = int(cell_row["Cells_AreaShape_BoundingBoxMinimum_Y"]), int(cell_row["Cells_AreaShape_BoundingBoxMaximum_Y"])
                        # x, y = int(cell_allele_coord_df["Cells_AreaShape_Center_X"].to_numpy()[0]), int(cell_allele_coord_df["Cells_AreaShape_Center_Y"].to_numpy()[0])
                        ## flip the x and y for visualization
                        img_sub = img[
                            y_min:y_max, x_min:x_max
                        ]
                    elif compartment == "Nuclei":
                        x_min, x_max = int(cell_row["Nuclei_AreaShape_BoundingBoxMinimum_X"]), int(cell_row["Nuclei_AreaShape_BoundingBoxMaximum_X"])
                        y_min, y_max = int(cell_row["Nuclei_AreaShape_BoundingBoxMinimum_Y"]), int(cell_row["Nuclei_AreaShape_BoundingBoxMaximum_Y"])
                        # x, y = int(cell_allele_coord_df["Cells_AreaShape_Center_X"].to_numpy()[0]), int(cell_allele_coord_df["Cells_AreaShape_Center_Y"].to_numpy()[0])
                        ## flip the x and y for visualization
                        img_sub = img[
                            y_min:y_max, x_min:x_max
                        ]
                    else:
                        img_sub = img[
                             y-dim//2:y+dim//2, x-dim//2:x+dim//2
                        ]
                    ## skip the subimage due to poor cell quality
                    if img_sub.shape[0] == 0 or img_sub.shape[1] == 0 or np.percentile(img_sub, 90) <= np.median(img) or np.var(img_sub) < 1e4 or np.percentile(img_sub, 99) / np.percentile(img_sub, 25) < 2:
                        continue
                        
                    img_sub = resize(img_sub, (dim, dim), preserve_range=True, anti_aliasing=True)
                    axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)  ## np.percentile(img_sub, max_intensity*100)
                    plot_label = f"{sel_channel}:{sel_plate},T{i%4+1}\nWell:{well},Site:{site}\n{allele}"
                    axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
                            verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                            bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
                    if is_bg:
                        axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
                            verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                            bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
                    # if flag:
                    #     axes.flatten()[plot_idx].text(0.97, 0.97, "Poor Quality FLAG", color='red', fontsize=10,
                    #         verticalalignment='top', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                    #         bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
                    int_95 = str(int(round(np.percentile(img_sub, 95))))
                    axes.flatten()[plot_idx].text(0.95, 0.05, f"95th Intensity:{int_95}", color='white', fontsize=10,
                                verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                                bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
                    plot_yet = 1
                    
            axes.flatten()[plot_idx].axis("off")
            
    fig.tight_layout()
    fig.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
    if display:
        plt.show()

    if output_dir:
        file_name = f"{variant}_{sel_channel}_cells"
        if auroc:
            file_name = f"{file_name}_{auroc:.3f}"
        if ref_well:
            file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
        if var_well:
            file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
        fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
        
    plt.close(fig)


def plot_allele_cell_single_plate(pm, variant, sel_channel, batch_profile_dict, plate_img_qc, auroc_df=None, site="05", ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir=TIFF_IMGS_DIR, output_dir=""):
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
        img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
        # print(np.percentile(img, 99) / np.median(img), np.percentile(img, 99) / np.percentile(img, 25))
        if plate_img_qc is not None:
            is_bg = plate_img_qc.filter(
                (pl.col("plate") == plate_img_dir.split("__")[0])
                & (pl.col("well") == well)
                & (pl.col("channel") == sel_channel))["is_bg"].to_numpy()[0]
            
        ## Full images
        # axes2.flatten()[plot_idx].imshow(img, vmin=0, vmax=np.percentile(img, max_intensity*100), cmap=cmap)
        # plot_label = f"{sel_channel}:{sel_plate},T{i%4+1}\nWell:{well},Site:{site}\n{allele}"
        # axes2.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
        #         verticalalignment='top', horizontalalignment='left', transform=axes2.flatten()[plot_idx].transAxes,
        #         bbox=dict(facecolor='black', alpha=0.3, linewidth=2))

        ## Draw cells
        cell_allele_coord_df = crop_allele(allele, batch_profile_dict[batch], sel_plate, well=well, site=site[-1])          
        cell_allele_coord_df = cell_allele_coord_df.with_columns(
            pl.struct("Cells_AreaShape_Center_X", "Cells_AreaShape_Center_Y") # 'Nuclei_AreaShape_Center_X', 'Nuclei_AreaShape_Center_Y'
            .map_elements(lambda x: compute_distance_cell(x['Cells_AreaShape_Center_X'], x['Cells_AreaShape_Center_Y']), return_dtype=pl.Float32).cast(pl.Int16)
            .alias('dist2edge')
        ).sort(by=["dist2edge","Cells_AreaShape_Area"], descending=[True,True]).filter(pl.col("Cells_AreaShape_Area")>5000)

        if cell_allele_coord_df.is_empty():
            axes.flatten()[plot_idx].text(0.97, 0.97, "No high-quality\ncell available", color='black', fontsize=12,
                verticalalignment='top', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                bbox=dict(facecolor='white', alpha=0.3, linewidth=2)
            )
            # axes.flatten()[plot_idx].set_visible(False)
            x, y = img.shape[0] // 2, img.shape[1] // 2
            img_sub = img[
                y-64:y+64, x-64:x+64
            ]
            axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)  ## np.percentile(img_sub, max_intensity*100)
        else:
            plot_yet = 0
            flag = False
            top_20_cell_allele_coord_df = cell_allele_coord_df #.head(20)
            img_subs = []
            ## go through top 20 if exists
            for cell_idx, cell_row in enumerate(top_20_cell_allele_coord_df.iter_rows(named=True)):
                if plot_yet:
                    break
                x, y = int(cell_row["Nuclei_AreaShape_Center_X"]), int(cell_row["Nuclei_AreaShape_Center_Y"])
                # x, y = int(cell_allele_coord_df["Cells_AreaShape_Center_X"].to_numpy()[0]), int(cell_allele_coord_df["Cells_AreaShape_Center_Y"].to_numpy()[0])
                ## flip the x and y for visualization
                img_sub = img[y-64:y+64, x-64:x+64]
                img_subs.append(img_sub)
                # Check if this is a good quality cell
                is_good_quality = not (img_sub.shape[0] == 0 or img_sub.shape[1] == 0 or 
                                      np.percentile(img_sub, 90) <= np.median(img) or 
                                      np.var(img_sub) < 1e4)
                # If it's good quality, plot it and break
                if is_good_quality:
                    axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)
                    plot_yet = 1
                    flag = False
                # If it's the last cell and we haven't plotted anything yet, plot this poor quality one
                elif cell_idx == len(top_20_cell_allele_coord_df) - 1:
                    if (img_sub.shape[0] == 128 and img_sub.shape[1] == 128):
                        axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)
                    else:
                        img_idx = len(img_subs) - 1
                        while (img_sub.shape[0] != 128 or img_sub.shape[1] != 128) and (img_idx >= 0):
                            img_sub = img_subs[img_idx]
                            img_idx -= 1
                        try:
                            axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)
                        except:
                            # axes.flatten()[plot_idx].set_visible(False)
                            x, y = img.shape[0] // 2, img.shape[1] // 2
                            img_sub = img[
                                y-64:y+64, x-64:x+64
                            ]
                            axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)
                    plot_yet = 1
                    flag = True
                else:
                    # Continue to next cell
                    continue
                axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)    
                plot_label = f"{sel_channel}:{sel_plate},\nWell:{well},Site:{site}\n{allele}"
                axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
                        verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                        bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
                if is_bg:
                    axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
                        verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                        bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
                if flag:
                    axes.flatten()[plot_idx].text(0.97, 0.97, "Poor Quality FLAG", color='red', fontsize=10,
                        verticalalignment='top', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                        bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
                int_95 = str(int(round(np.percentile(img_sub, 95))))
                axes.flatten()[plot_idx].text(0.95, 0.05, f"95th Intensity:{int_95}", color='white', fontsize=10,
                            verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                            bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
                plot_yet = 1
                
        axes.flatten()[plot_idx].axis("off")
            
    fig.tight_layout()
    fig.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
    if display:
        plt.show()

    if output_dir:
        file_name = f"{variant}_{sel_channel}_cells"
        if auroc:
            file_name = f"{file_name}_{auroc:.3f}"
        if ref_well:
            file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
        if var_well:
            file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
        fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
        
    plt.close(fig)


def save_allele_cell_imgs(variant, feat, batch_profile_dict, cell_ids=[], auroc_df=None, display=False, save_img=False):
    bio_rep = get_allele_batch(variant, auroc_df)

    if auroc_df is not None:
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
        # if len(ref_wells)==1 and len(var_wells)==1:
        #     if cell_ids:
        #         plot_allele_cell_by_id(pm, variant, sel_channel, batch_profile_dict, auroc_df, plate_img_qc, cell_ids=[], site="05", ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir=TIFF_IMGS_DIR, output_dir="")
        #     else:
        #         plot_allele_cell(allele_meta_df_dict[bio_rep],
        #                         variant=variant, sel_channel=feat,
        #                         batch_profile_dict=batch_profile_dict,
        #                         auroc_df=auroc_df_batch, 
        #                         plate_img_qc=img_well_qc_sum_dict[bio_rep], 
        #                         site="05", max_intensity=0.99, 
        #                         display=display,
        #                         imgs_dir=TIFF_IMGS_DIR, 
        #                         output_dir=output_dir)
        # else:
        for ref_well in ref_wells:
            for var_well in var_wells:
                if cell_ids:
                    plot_allele_cell_by_id(allele_meta_df_dict[bio_rep], 
                                           variant=variant, 
                                           sel_channel=feat, 
                                           batch_profile_dict=batch_profile_dict, 
                                           auroc_df=auroc_df_batch, 
                                           plate_img_qc=img_well_qc_sum_dict[bio_rep], 
                                           cell_ids=cell_ids, 
                                           ref_well=[ref_well],
                                           var_well=[var_well], max_intensity=0.99, 
                                           display=display, 
                                           imgs_dir=TIFF_IMGS_DIR, 
                                           output_dir=output_dir)
                else:
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


def plot_allele_separate_plot(pm, variant, sel_channel, plate_img_qc, auroc_df=None, site="05", ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir="", output_dir=""):
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
    
    # if len(wt_wells) > 1:
    #     # Get coordinates of wells
    #     well_coords = [well_to_coordinates(w) for w in set([ref_well_pl for ref_well_pl in wt_wells])]
    #     # Sort wells by max distance from edges (descending)
    #     wt_wells = [max(well_coords, key=lambda x: compute_distance(x[1], x[2]))[0]]
    pm_var = pm.filter((pl.col("imaging_well").is_in(np.concatenate([wt_wells, var_wells])))&(pl.col("plate_map_name").is_in(plate_map))).sort("node_type")

    # fig, axes = plt.subplots((len(wt_wells)+len(var_wells))*2, 4, figsize=(15, (len(wt_wells)+len(var_wells))*8), sharex=True, sharey=True)
    for wt_var, pm_row in enumerate(pm_var.iter_rows(named=True)):
        if "allele" in pm_row["node_type"]:
            if pm_row["node_type"] == "allele":
                well = var_wells[0]
                allele = variant
            else:
                well = wt_wells[0]
                allele = wt
        else:
            if pm_row["imaging_well"] in wt_wells:
                well = wt_wells[0]
                allele = wt
            else:
                well = var_wells[0]
                allele = variant

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

            fig, ax = plt.subplots()
            plate_img_dir = plate_dict[sel_plate][f"T{i%4+1}"]
            img_file = f"r{row}c{col}f{site}p01-ch{channel}sk1fk1fl1.tiff"
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

            img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
            plot_idx = i+wt_var*4*2
            # print(i, wt_var, plot_idx)
            ax.imshow(img, vmin=0, vmax=np.percentile(img, max_intensity*100), cmap=cmap)
            plot_label = f"{sel_channel}-{sel_plate}_T{i%4+1}_Well{well}_Site{site}_{allele}"
            # axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
            #         verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
            #         bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
            # if is_bg:
            #     axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
            #         verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
            #         bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
            # int_95 = str(int(round(np.percentile(img, 95))))
            # axes.flatten()[plot_idx].text(0.97, 0.03, f"95th Intensity:{int_95}\nSet vmax:{max_intensity*100:.0f}th perc.", color='white', fontsize=10,
            #                verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
            #                bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
            ax.axis("off")
        
            plt.tight_layout()
            # plt.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
            fig.savefig(os.path.join(output_dir, f"{plot_label}.png"), dpi=400, bbox_inches='tight')
    
    # if display:
    #     plt.show()
    
    # if output_dir:
    #     file_name = f"{variant}_{sel_channel}"
    #     if auroc:
    #         file_name = f"{file_name}_{auroc:.3f}"
    #     if ref_well:
    #         file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
    #     if var_well:
    #         file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
    #     fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
        
            plt.close(fig)


def plot_allele_cells_separate_figures(
    pm, 
    variant, 
    sel_channels, 
    batch_profile_dict, 
    plate_img_qc, 
    compartment="Cells", 
    dim=64,
    resize_fig=False,
    cell_ids=[], 
    ref_well=[], 
    var_well=[],
    target_feat="",
    imgs_dir=TIFF_IMGS_DIR, 
    n_cells=50
):
    """
    Generate two separate figures with cell crops - one for variant and one for reference.
    Each figure contains up to n_cells (default 50) cell crops with site, well, and info labels.
    
    Parameters:
    -----------
    pm : polars.DataFrame
        Plate mapping dataframe
    variant : str
        Variant allele name (e.g., "F9_Cys28Arg")
    sel_channels : list
        Channels to visualize (e.g., "GFP", "Mito", "AGP")
    batch_profile_dict : dict
        Dictionary containing batch profile data
    plate_img_qc : polars.DataFrame
        Image quality control dataframe
    cell_ids : list, optional
        Specific cell IDs to plot (if provided, filters to these cells)
    ref_well : list, optional
        Reference wells to include
    var_well : list, optional
        Variant wells to include
    n_cells : int, default 50
        Number of cells per figure
    """
    
    print(f"Generating separate figures for {variant} with {n_cells} cells each")
        
    # Get allele information
    wt = variant.split("_")[0]
    wt_wells = pm.filter(pl.col("gene_allele") == wt).select("imaging_well").to_pandas().values.flatten()
    var_wells = pm.filter(pl.col("gene_allele") == variant).select("imaging_well").to_pandas().values.flatten()
    plate_map = pm.filter(pl.col("gene_allele") == variant).select("plate_map_name").to_pandas().values.flatten()

    print(plate_map)
    # Filter wells if specified
    if ref_well:
        wt_wells = [well for well in wt_wells if well in ref_well]
    if var_well:
        var_wells = [well for well in var_wells if well in var_well]
    
    # Get plate mapping for both alleles
    pm_var = pm.filter(
        (pl.col("imaging_well").is_in(np.concatenate([wt_wells, var_wells]))) &
        (pl.col("plate_map_name").is_in(plate_map))
    ).sort("node_type")    
    # Collect all cell data for both alleles
    # ref_cells_data = []
    # var_cells_data = []
    target_cell_dict = {}
    for pm_row in pm_var.iter_rows(named=True):        
        # Determine allele type and wells
        if pm_row["node_type"] == "allele":
            current_wells = var_wells
            allele = variant
        else:
            current_wells = wt_wells  
            allele = wt
            
        well = current_wells[0] if len(current_wells) else None
        if not well:
            continue
            
        # Process all timepoints (8 total: 4 R1 + 4 R2)
        for i in range(8):
            if i < 4:
                sel_plate = pm_row["imaging_plate_R1"]
                timepoint = f"T{i%4+1}"
            else:
                sel_plate = pm_row["imaging_plate_R2"] 
                timepoint = f"T{i%4+1}"
                
            # Get batch and image info
            batch = batch_dict[sel_plate.split("_")[0]]
            batch_img_dir = f'{imgs_dir}/{batch}/images'
            letter = well[0]
            row, col = letter_dict[letter], well[1:3]
            plate_img_dir = plate_dict[sel_plate][timepoint]
                
            for sel_channel in sel_channels:
                channel = channel_dict[sel_channel]
                # Check background quality
                is_bg = False
                if plate_img_qc is not None:
                    try:
                        is_bg_array = plate_img_qc.filter(
                            (pl.col("plate") == plate_img_dir.split("__")[0]) &
                            (pl.col("well") == well) &
                            (pl.col("channel") == sel_channel)
                        )["is_bg"].to_numpy()
                        if is_bg_array.size > 0:
                            is_bg = is_bg_array[0]
                    except:
                        pass
                
                # Get cell coordinates for this allele/plate/timepoint
                try:
                    cell_coords_df = crop_allele(
                        allele, batch_profile_dict[batch], 
                        sel_plate.split("P")[0], rep=timepoint
                    )
                    
                    if cell_coords_df.is_empty():
                        continue
                        
                    # Add distance and cell ID columns
                    cell_coords_df = cell_coords_df.with_columns(
                        pl.struct("Cells_AreaShape_Center_X", "Cells_AreaShape_Center_Y")
                        .map_elements(
                            lambda x: compute_distance_cell(x['Cells_AreaShape_Center_X'], x['Cells_AreaShape_Center_Y']), 
                            return_dtype=pl.Float32
                        ).cast(pl.Int16).alias('dist2edge'),
                        pl.concat_str([
                            "Metadata_Plate", "Metadata_well_position", 
                            "Metadata_ImageNumber", "Metadata_ObjectNumber"
                        ], separator="_").alias("Metadata_CellID")
                    ).filter(
                        pl.col("Metadata_refvar_gfp_adj_classify").str.contains(variant)
                    ).unique(subset="Metadata_CellID")

                    # Filter by cell IDs if provided
                    if cell_ids:
                        cell_coords_df = cell_coords_df.filter(
                            pl.col("Metadata_CellID").is_in(cell_ids),
                        )

                    # Sort by quality metrics (distance from edge, cell area)
                    if target_feat:
                        cell_coords_df = cell_coords_df.sort(
                            by=[target_feat, "dist2edge", "Cells_AreaShape_Area"], 
                            descending=[True, True, True]
                        )#.filter(pl.col("Cells_AreaShape_Area") > 5000)
                    
                    # print(cell_coords_df.sort("Metadata_CellID"))
                    
                    cc = 0
                    # Process each cell in this image
                    for cell_row in cell_coords_df.iter_rows(named=True):
                        if cc == 50:
                            break
                        x, y = int(cell_row["Nuclei_AreaShape_Center_X"]), int(cell_row["Nuclei_AreaShape_Center_Y"])
                        site = f"0{cell_row['Metadata_Site']}"
                        img_file = f"r{row}c{col}f{site}p01-ch{channel}sk1fk1fl1.tiff"
                        try:
                            img = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
                        except:
                            print(f"Could not load image: {img_file}")
                            continue

                        if compartment == "Cells":
                            x_min, x_max = int(cell_row["Cells_AreaShape_BoundingBoxMinimum_X"]), int(cell_row["Cells_AreaShape_BoundingBoxMaximum_X"])
                            y_min, y_max = int(cell_row["Cells_AreaShape_BoundingBoxMinimum_Y"]), int(cell_row["Cells_AreaShape_BoundingBoxMaximum_Y"])
                            # x, y = int(cell_allele_coord_df["Cells_AreaShape_Center_X"].to_numpy()[0]), int(cell_allele_coord_df["Cells_AreaShape_Center_Y"].to_numpy()[0])
                            ## flip the x and y for visualization
                            img_sub = img[
                                y_min:y_max, x_min:x_max
                            ]
                        elif compartment == "Nuclei":
                            # x_min, x_max = int(cell_row["Nuclei_AreaShape_BoundingBoxMinimum_X"]), int(cell_row["Nuclei_AreaShape_BoundingBoxMaximum_X"])
                            # y_min, y_max = int(cell_row["Nuclei_AreaShape_BoundingBoxMinimum_Y"]), int(cell_row["Nuclei_AreaShape_BoundingBoxMaximum_Y"]) 
                            x, y = int(cell_row["Cells_AreaShape_Center_X"]), int(cell_row["Cells_AreaShape_Center_Y"])
                            ## flip the x and y for visualization
                            # img_sub = img[
                            #     y_min:y_max, x_min:x_max
                            # ]
                            img_sub = img[
                                y-dim//2:y+dim//2, x-dim//2:x+dim//2
                            ]
                        else:
                            img_sub = img[
                                y-dim//2:y+dim//2, x-dim//2:x+dim//2
                            ]
                            
                        # Quality filters
                        if ((img_sub.shape[0] == 0 or img_sub.shape[1] == 0 or 
                            np.percentile(img_sub, 90) <= np.median(img) or 
                            np.var(img_sub) < 1e4 or 
                            np.percentile(img_sub, 99) / np.percentile(img_sub, 25) < 2) and 
                            len(sel_channels) == 1):
                            continue
                        
                        if resize_fig:
                            img_sub = resize(img_sub, (dim, dim), preserve_range=True, anti_aliasing=True)
                        
                        # Store cell data if not exists
                        cell_id = cell_row["Metadata_CellID"]
                        if cell_id not in target_cell_dict:
                            cell_data = {
                                'allele': allele,
                                'well': well,
                                'site': site,
                                'plate': sel_plate,
                                'timepoint': timepoint,
                                f'img_crop_{sel_channel}': img_sub,
                                f'is_bg_{sel_channel}': is_bg,
                                f'intensity_95_{sel_channel}': np.percentile(img_sub, 95),
                            }
                            if target_feat:
                                cell_data[target_feat] = cell_row[target_feat]
                            target_cell_dict[cell_id] = cell_data
                        else:
                            target_cell_dict[cell_id].update(
                                {
                                    f'img_crop_{sel_channel}': img_sub,
                                    f'is_bg_{sel_channel}': is_bg,
                                    f'intensity_95_{sel_channel}': np.percentile(img_sub, 95)
                                }
                            )
                        cc += 1
                except Exception as e:
                    print(f"Error processing {allele} {timepoint}: {e}")
                    continue
    
    # print(f"Collected {len(ref_cells_data)} reference cells, {len(var_cells_data)} variant cells")
    return target_cell_dict


def viz_cell_crop(cell, cell_comp, max_intensity=.99, ax=None, axis_off=True):
    img_arr = cell[f"img_crop_{cell_comp}"]
    cmap = channel_to_cmap(cell_comp)
    key_feat_sel = list(cell.keys())[-1]
    if ax is None:
        plt.imshow(img_arr, vmin=0, vmax=np.percentile(img_arr, max_intensity*100), cmap=cmap)
        if axis_off:
            plt.axis('off')  # Turn off axis labels
        plt.show()
    else:
        ax.imshow(img_arr, vmin=0, vmax=np.percentile(img_arr, max_intensity*100), cmap=cmap)
        if axis_off:
            ax.axis('off')
        # plot_label = f"s2n ratio:\n{np.percentile(img_arr, 99) / np.percentile(img_arr, 25):.3f}"
        plot_label = f"{cell['allele']}\n95th intensity:{cell[f'intensity_95_{cell_comp}']:.1f}\n{cell[key_feat_sel]:.2f}"#cell["cell_id"]
        ax.text(0.05, 0.95, plot_label, color='white', fontsize=9,
                verticalalignment='top', horizontalalignment='left', 
                transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.2, linewidth=1))


def viz_cell_crop_multi(cell, max_intensity=0.99, ax=None, axis_off=True):
    """
    Visualize all available channels of a single cell with RGB color blending
    
    Parameters:
    -----------
    cell : dict
        Cell dictionary containing separate image crops for each channel
        Expected keys: f"img_crop_{channel}" for each channel
    max_intensity : float
        Percentile for intensity scaling (default 0.99)
    ax : matplotlib axis
        Axis to plot on (creates new if None)
    axis_off : bool
        Whether to turn off axis labels
    """
    # Find all available channels in the cell dict
    channels = []
    for key in cell.keys():
        if key.startswith("img_crop_"):
            channel = key.replace("img_crop_", "")
            channels.append(channel)
    
    if not channels:
        raise ValueError("No img_crop_{channel} keys found in cell dict list")
    
    # Get dimensions from first available channel
    first_channel = channels[0]
    first_img = cell[f"img_crop_{first_channel}"]
    height, width = first_img.shape
    
    # Initialize RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Process each channel
    for channel in channels:
        # Get image array for this channel
        channel_key = f"img_crop_{channel}"
        img_arr = cell[channel_key]
        
        # Calculate intensity scaling for this specific channel
        channel_vmax = np.percentile(img_arr, max_intensity * 100)
        
        # Normalize image to [0, 1]
        normalized_img = np.clip(img_arr / channel_vmax, 0, 1)
        
        # Get RGB color for this channel
        rgb_color = channel_to_rgb(channel)
        
        # Add this channel's contribution to RGB image
        for i in range(3):  # R, G, B
            rgb_image[:, :, i] += normalized_img * rgb_color[i]
    
    # Clip final RGB values to [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Plot the combined RGB image
    if ax is None:
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb_image)
        if axis_off:
            plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb_image)
        if axis_off:
            ax.axis('off')
        
        # Add label with cell information
        key_feat_sel = list(cell.keys())[-1]
        plot_label = f"{cell['allele']}\nChannels: {', '.join(sorted(channels))}\n{cell[key_feat_sel]:.2f}" ##95th intensity: {cell.get('intensity_95', 'N/A'):.1f}
        ax.text(0.05, 0.95, plot_label, color='white', fontsize=9,
                verticalalignment='top', horizontalalignment='left', 
                transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.2, linewidth=1))
    

def plot_var_key_feat_cell_crops(variant, key_feat_sel, auroc_df, sel_channel=["GFP"], compartment="Cells", resize_fig=False, top_num=200):
    batch_profiles_filtered = {}
    bio_rep = get_allele_batch(variant, auroc_df)
    # for bio_rep, bio_rep_batches in BIO_REP_BATCHES_DICT.items():
    for batch_id in BIO_REP_BATCHES_DICT[bio_rep]:
        # imagecsv_dir = IMG_ANALYSIS_DIR.format(batch_id) #f"../../../8.1_upstream_analysis_runxi/2.raw_img_qc/inputs/images/{batch_id}/analysis"
        prof_path = BATCH_PROFILES_GFP_FILTERED.format(batch_id)
        # print(prof_path)
        # Get metadata
        profiles_filt = pl.scan_parquet(prof_path).select(
            ["Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber", "Metadata_Plate", 
             "Metadata_refvar_gfp_adj_classify", GFP_INTENSITY_COLUMN, key_feat_sel
            #  "Nuclei_AreaShape_BoundingBoxMaximum_X", "Nuclei_AreaShape_BoundingBoxMaximum_Y", 
            #  "Nuclei_AreaShape_BoundingBoxMinimum_X", "Nuclei_AreaShape_BoundingBoxMinimum_Y", 
            # "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
            # "Cells_AreaShape_BoundingBoxMaximum_X", "Cells_AreaShape_BoundingBoxMaximum_Y", "Cells_AreaShape_BoundingBoxMinimum_X",
            # "Cells_AreaShape_BoundingBoxMinimum_Y",	"Cells_AreaShape_Center_X",	"Cells_AreaShape_Center_Y",
            # "Cells_Intensity_MeanIntensity_GFP", "Cells_Intensity_MedianIntensity_GFP", "Cells_Intensity_IntegratedIntensity_GFP"
            ],
        ).collect()
        # display(profiles_filt)
        # print(prof_path)
        # Sort by allele, then image number
        profiles_filt = profiles_filt.with_columns(
            # pl.concat_str(["Metadata_Plate", "Metadata_well_position", "Metadata_Site"], separator="_").alias("Metadata_SiteID"),
            pl.concat_str(
                [
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber",
                ],
                separator="_",
            ).alias("Metadata_CellID"),
        ).select(["Metadata_CellID", "Metadata_refvar_gfp_adj_classify", GFP_INTENSITY_COLUMN, key_feat_sel]).join(
            batch_profiles[batch_id], on="Metadata_CellID", how="inner"
        )
        # display(profiles_filt)
        # print(batch_profiles[batch_id].select(["Nuclei_AreaShape_BoundingBoxMaximum_X", "Nuclei_AreaShape_BoundingBoxMaximum_Y"]).head())
        # profiles = profiles.sort(["Protein_label", "Metadata_SiteID"])
        # alleles = profiles.select("Protein_label").to_series().unique().to_list()
        batch_profiles_filtered[batch_id] = profiles_filt

    select_cell_ids_df = pl.DataFrame()
    # for bio_rep, bio_rep_batches in BIO_REP_BATCHES_DICT.items():
    for batch_id in BIO_REP_BATCHES_DICT[bio_rep]:
        select_cell_ids_df = pl.concat([
            select_cell_ids_df,
            batch_profiles_filtered[batch_id].filter(
                (pl.col("Metadata_refvar_gfp_adj_classify").str.contains(variant)),
                # (pl.col("Cells_AreaShape_Area") < 7000) & (pl.col("Cells_AreaShape_Area") > 5000),
            ).sort("Metadata_gene_allele")
        ])
    # print(select_cell_ids_df[GFP_INTENSITY_COLUMN].max(), select_cell_ids_df[GFP_INTENSITY_COLUMN].min())
    
    var_gfp = select_cell_ids_df.filter(
          pl.col("Metadata_gene_allele")==variant
    )[GFP_INTENSITY_COLUMN].to_numpy()
    ref_gfp = select_cell_ids_df.filter(
          pl.col("Metadata_gene_allele")!=variant
    )[GFP_INTENSITY_COLUMN].to_numpy()
    
    gfp_results = find_optimal_gfp_range_fast(
        ref_gfp, var_gfp, quantile_pair=(.2, .8), min_cells_per_well=20
    )
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxenplot(
        data=select_cell_ids_df.to_pandas(), y=key_feat_sel, x="Metadata_gene_allele", ax=axes[0]
    )
    sns.boxenplot(
        data=select_cell_ids_df.to_pandas(), y=GFP_INTENSITY_COLUMN, x="Metadata_gene_allele", ax=axes[1]
    )
    plt.show()

    cell_ids = []
    cell_ids_df = select_cell_ids_df.filter(
        (pl.col(GFP_INTENSITY_COLUMN) > gfp_results[0]) & (pl.col(GFP_INTENSITY_COLUMN) < gfp_results[1])
    )
    # Calculate means for each group
    variant_mean = select_cell_ids_df.filter(pl.col("Metadata_gene_allele") == variant)[key_feat_sel].mean()
    non_variant_mean = select_cell_ids_df.filter(pl.col("Metadata_gene_allele") != variant)[key_feat_sel].mean()

    if variant_mean > non_variant_mean:
        # Variant higher: top 50 variant, bottom 50 non-variant
        variant_cells = select_cell_ids_df.filter(
            pl.col("Metadata_gene_allele") == variant
        ).sort(key_feat_sel, descending=True).head(top_num)["Metadata_CellID"].to_list()
        
        non_variant_cells = select_cell_ids_df.filter(
            pl.col("Metadata_gene_allele") != variant
        ).sort(key_feat_sel, descending=False).head(top_num)["Metadata_CellID"].to_list()
    else:
        # Non-variant higher: bottom 50 variant, top 50 non-variant
        variant_cells = select_cell_ids_df.filter(
            pl.col("Metadata_gene_allele") == variant
        ).sort(key_feat_sel, descending=False).head(top_num)["Metadata_CellID"].to_list()
        
        non_variant_cells = select_cell_ids_df.filter(
            pl.col("Metadata_gene_allele") != variant
        ).sort(key_feat_sel, descending=True).head(top_num)["Metadata_CellID"].to_list()

    cell_ids += (variant_cells + non_variant_cells)

    cell_img_dict = plot_allele_cells_separate_figures(
        pm=allele_meta_df_dict[bio_rep],
        variant=variant,
        sel_channels=sel_channel, 
        batch_profile_dict=batch_profiles_filtered,
        plate_img_qc=img_well_qc_sum_dict[bio_rep],
        compartment=compartment,
        target_feat=key_feat_sel,
        resize_fig=resize_fig,
        cell_ids=cell_ids,  # Uncomment to use specific cells
        imgs_dir=TIFF_IMGS_DIR,
        n_cells=50
    )

    var_data = {k: v for k, v in cell_img_dict.items() if v['allele']==variant}
    ref_data = {k: v for k, v in cell_img_dict.items() if v['allele']!=variant}

    var_data = sorted(var_data.items(), key=lambda x : x[1][key_feat_sel], reverse=True)
    ref_data = sorted(ref_data.items(), key=lambda x : x[1][key_feat_sel], reverse=False)
    print("Var len:", len(var_data), "Ref len:", len(ref_data))

    plt.clf()
    fig, axes = plt.subplots(3, 10, figsize=(25, 30))
    for idx, (cell_id, cell) in enumerate(ref_data[:30]):
        ## plot nuc
        if len(sel_channel) == 1:
            viz_cell_crop(cell, sel_channel[0], ax=axes[idx // 10, idx % 10])
        else:
            viz_cell_crop_multi(cell, ax=axes[idx // 10, idx % 10])
        # axes[idx // 10, idx % 10].set_title("\n".join([ex_meta[cell,1].split("_T")[0], "T"+ex_meta[cell,1].split("_T")[1]+"|nuc"]), fontsize=7)
        # axes[idx // 10 + 1, idx % 10].set_title("\n".join([ex_meta[cell,1].split("_T")[0], "T"+ex_meta[cell,1].split("_T")[1]+"|pro"]), fontsize=7)
    plt.subplots_adjust(wspace=0.05, hspace=-.9)
    plt.suptitle(f"Differential Feature: {key_feat_sel}", fontsize=12, y=.6)
    plt.show()

    plt.clf()
    fig, axes = plt.subplots(3, 10, figsize=(25, 30))
    for idx, (cell_id, cell) in enumerate(var_data[:30]):
        # print(idx)
        ## plot nuc
        if len(sel_channel) == 1:
            viz_cell_crop(cell, sel_channel[0], ax=axes[idx // 10, idx % 10])
        else:
            viz_cell_crop_multi(cell, ax=axes[idx // 10, idx % 10])
        # axes[idx // 10, idx % 10].set_title("\n".join([ex_meta[cell,1].split("_T")[0], "T"+ex_meta[cell,1].split("_T")[1]+"|nuc"]), fontsize=7)
        # axes[idx // 10 + 1, idx % 10].set_title("\n".join([ex_meta[cell,1].split("_T")[0], "T"+ex_meta[cell,1].split("_T")[1]+"|pro"]), fontsize=7)
    plt.subplots_adjust(wspace=0.05, hspace=-.9)
    plt.suptitle(f"Differential Feature: {key_feat_sel}", fontsize=12, y=.6)
    plt.show()
        

# GFP range optimization function with expanded ranges and ratio constraint
def find_optimal_gfp_range_fast(ref_gfp: np.ndarray, var_gfp: np.ndarray, 
                                quantile_pair: tuple=(0.25, 0.75),
                                min_cells_per_well: int = 20):
    """Ultra-fast vectorized GFP range optimization with single quantile pair"""
    # Check if arrays are empty
    if len(ref_gfp) == 0 or len(var_gfp) == 0:
        return None, None, 0, 0, "EMPTY_ARRAYS"
    
    # Expanded quantile range testing: from 10%-90% down to 30%-70%
    # quantile_pairs = [
    #     (0.2, 0.8), (0.22, 0.78), (0.25, 0.75), (0.27, 0.73), (0.3, 0.7)
    # ] ## (0.1, 0.9), (0.12, 0.88), (0.15, 0.85), (0.17, 0.83)
    
    best_range = None
    max_total_cells = 0
    best_quantile_info = ""
    results = []

    low_q, high_q = quantile_pair
    # Calculate quantiles directly for the single pair
    ref_low = np.quantile(ref_gfp, low_q)
    ref_high = np.quantile(ref_gfp, high_q)
    var_low = np.quantile(var_gfp, low_q)
    var_high = np.quantile(var_gfp, high_q)
    
    # Find overlapping range
    range_min = max(ref_low, var_low)
    range_max = min(ref_high, var_high)
    
    # Skip if invalid range
    if range_min >= range_max:
        results.append((f"{int(low_q*100)}-{int(high_q*100)}%", 0, 0, 0, "Invalid range", "N/A"))
        return None, None, 0, 0, "NO_SUITABLE_RANGE"
            
    # Vectorized cell counting
    ref_mask = (ref_gfp >= range_min) & (ref_gfp <= range_max)
    var_mask = (var_gfp >= range_min) & (var_gfp <= range_max)
    ref_count = np.sum(ref_mask)
    var_count = np.sum(var_mask)
    
    # Calculate sample size ratio
    if ref_count == 0 or var_count == 0:
        ratio_status = "Zero samples"
    else:
        ratio_status = f"{max(ref_count, var_count) / min(ref_count, var_count)}:.2f"
    
    results.append((f"{int(low_q*100)}-{int(high_q*100)}%", ref_count, var_count, 
                    ref_count + var_count, f"GFP: {range_min:.1f}-{range_max:.1f}", ratio_status))
    
    # Check minimum requirements and ratio constraint
    if (ref_count >= min_cells_per_well and var_count >= min_cells_per_well and
        ref_count > 0 and var_count > 0):
        total_cells = ref_count + var_count
        if total_cells > max_total_cells:
            max_total_cells = total_cells
            best_range = (range_min, range_max, ref_count, var_count)
            best_quantile_info = f"{int(low_q*100)}%-{int(high_q*100)}%"

    if best_range is not None:
        return best_range[0], best_range[1], best_range[2], best_range[3], best_quantile_info
    else:
        return None, None, 0, 0, "NO_SUITABLE_RANGE"
    

# def plot_allele_cell_multi(pm, variant, sel_channel, batch_profile_dict, auroc_df, plate_img_qc, site="05", ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir=TIFF_IMGS_DIR, output_dir=""):
#     # Detect input type
#     is_multichannel = isinstance(sel_channel, (list, tuple))
#     channels = sel_channel if is_multichannel else [sel_channel]
    
#     # For single channel, maintain original behavior
#     if not is_multichannel:
#         cmap = channel_to_cmap(sel_channel)
#         channel = channel_dict[sel_channel]

#     ## get the number of wells/images per allele
#     wt = variant.split("_")[0]
#     wt_wells = pm.filter(pl.col("gene_allele") == wt).select("imaging_well").to_pandas().values.flatten()
#     var_wells = pm.filter(pl.col("gene_allele") == variant).select("imaging_well").to_pandas().values.flatten()
#     plate_map = pm.filter(pl.col("gene_allele") == variant).select("plate_map_name").to_pandas().values.flatten()

#     if ref_well:
#         wt_wells = [well for well in wt_wells if well in ref_well]
#     if var_well:
#         var_wells = [well for well in var_wells if well in var_well]
#     pm_var = pm.filter((pl.col("imaging_well").is_in(np.concatenate([wt_wells, var_wells])))&(pl.col("plate_map_name").is_in(plate_map))).sort("node_type")
    
#     plt.clf()
#     fig, axes = plt.subplots((len(wt_wells)+len(var_wells))*2, 4, figsize=(15, 16), sharex=True, sharey=True)
    
#     for wt_var, pm_row in enumerate(pm_var.iter_rows(named=True)):
#         if pm_row["node_type"] == "allele":
#             well = var_wells[0]
#             allele = variant
#         else:
#             well = wt_wells[0]
#             allele = wt
            
#         for i in range(8):
#             plot_idx = i+wt_var*4*2
#             if i < 4:
#                 sel_plate = pm_row["imaging_plate_R1"]
#             else:
#                 sel_plate = pm_row["imaging_plate_R2"]

#             batch = batch_dict[sel_plate.split("_")[0]]
#             batch_img_dir = f'{imgs_dir}/{batch}/images'
#             letter = well[0]
#             row, col = letter_dict[letter], well[1:3]
#             plate_img_dir = plate_dict[sel_plate][f"T{i%4+1}"]
            
#             # Load images for all channels
#             channel_imgs = {}
#             for ch in channels:
#                 channel_num = channel_dict[ch]
#                 img_file = f"r{row}c{col}f{site}p01-ch{channel_num}sk1fk1fl1.tiff"
#                 channel_imgs[ch] = imread(f"{batch_img_dir}/{plate_img_dir}/Images/{img_file}", as_gray=True)
            
#             # For QC check, use first channel if multichannel
#             main_img = channel_imgs[channels[0]]
            
#             if plate_img_qc is not None:
#                 # Use first channel for QC check
#                 qc_channel = channels[0] if is_multichannel else sel_channel
#                 is_bg = plate_img_qc.filter((pl.col("plate") == plate_img_dir.split("__")[0]) & (pl.col("well") == well) & (pl.col("channel") == qc_channel))["is_bg"].to_numpy()[0]

#             ## Draw cells
#             cell_allele_coord_df = crop_allele(allele, batch_profile_dict[batch], sel_plate.split("P")[0], rep=f"T{i%4+1}", site=site[-1])      
#             cell_allele_coord_df = cell_allele_coord_df.with_columns(
#                 pl.struct("Cells_AreaShape_Center_X", "Cells_AreaShape_Center_Y")
#                 .map_elements(lambda x: compute_distance_cell(x['Cells_AreaShape_Center_X'], x['Cells_AreaShape_Center_Y']), return_dtype=pl.Float32).cast(pl.Int16)
#                 .alias('dist2edge')
#             ).sort(by=["dist2edge","Cells_AreaShape_Area"], descending=[True,True]).filter(pl.col("Cells_AreaShape_Area")>5000)

#             if cell_allele_coord_df.is_empty():
#                 axes.flatten()[plot_idx].text(0.97, 0.97, "No high-quality\ncell available", color='black', fontsize=12,
#                     verticalalignment='top', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
#                     bbox=dict(facecolor='white', alpha=0.3, linewidth=2)
#                 )
                
#                 x, y = main_img.shape[0] // 2, main_img.shape[1] // 2
                
#                 if is_multichannel:
#                     # Create multichannel composite
#                     channel_subs = {}
#                     for ch in channels:
#                         img_sub = channel_imgs[ch][y-64:y+64, x-64:x+64]
#                         channel_subs[ch] = img_sub
#                     composite = create_multichannel_composite(channel_subs, channels, max_intensity)
#                     axes.flatten()[plot_idx].imshow(composite)
#                 else:
#                     img_sub = main_img[y-64:y+64, x-64:x+64]
#                     axes.flatten()[plot_idx].imshow(img_sub, vmin=0, vmax=np.percentile(img_sub, max_intensity*100), cmap=cmap)
                    
#             else:
#                 plot_yet = 0
#                 for cell_idx, cell_row in enumerate(cell_allele_coord_df.iter_rows(named=True)):
#                     if plot_yet:
#                         break
#                     x, y = int(cell_row["Nuclei_AreaShape_Center_X"]), int(cell_row["Nuclei_AreaShape_Center_Y"])
                    
#                     # Get subimages for all channels
#                     channel_subs = {}
#                     main_img_sub = None
#                     for ch in channels:
#                         img_sub = channel_imgs[ch][y-64:y+64, x-64:x+64]
#                         channel_subs[ch] = img_sub
#                         if ch == channels[0]:  # Use first channel for quality checks
#                             main_img_sub = img_sub
                    
#                     ## skip the subimage due to poor cell quality (using main channel)
#                     if (main_img_sub.shape[0] == 0 or main_img_sub.shape[1] == 0 or 
#                         np.percentile(main_img_sub, 90) <= np.median(main_img) or 
#                         np.var(main_img_sub) < 1e4 or 
#                         np.percentile(main_img_sub, 99) / np.percentile(main_img_sub, 25) < 2):
#                         continue
                    
#                     # Display image
#                     if is_multichannel:
#                         composite = create_multichannel_composite(channel_subs, channels, max_intensity)
#                         axes.flatten()[plot_idx].imshow(composite)
#                         channel_label = "+".join(channels)
#                     else:
#                         axes.flatten()[plot_idx].imshow(main_img_sub, vmin=0, vmax=np.percentile(main_img_sub, max_intensity*100), cmap=cmap)
#                         channel_label = sel_channel
                    
#                     plot_label = f"{channel_label}:{sel_plate},T{i%4+1}\nWell:{well},Site:{site}\n{allele}"
#                     axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
#                             verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
#                             bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
                    
#                     if is_bg:
#                         axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
#                             verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
#                             bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
                    
#                     int_95 = str(int(round(np.percentile(main_img_sub, 95))))
#                     axes.flatten()[plot_idx].text(0.95, 0.05, f"95th Intensity:{int_95}", color='white', fontsize=10,
#                                 verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
#                                 bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
#                     plot_yet = 1
                    
#             axes.flatten()[plot_idx].axis("off")
            
#     fig.tight_layout()
#     fig.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
#     if display:
#         plt.show()

#     if output_dir:
#         if is_multichannel:
#             channel_str = "+".join(channels)
#         else:
#             channel_str = sel_channel
            
#         file_name = f"{variant}_{channel_str}_cells"
#         if auroc_df:
#             auroc = auroc_df.filter(pl.col("allele_0")==variant)["AUROC_Mean"].mean()
#             file_name = f"{file_name}_{auroc:.3f}"
#         if ref_well:
#             file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
#         if var_well:
#             file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
#         fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
        
#     plt.close(fig)


# def create_multichannel_composite(channel_subs, channels, max_intensity):
#     """Create RGB composite from multiple channel subimages"""
#     shape = next(iter(channel_subs.values())).shape
#     composite = np.zeros((*shape, 3))
    
#     # Channel-specific colors matching your cmap definitions
#     color_map = {
#         'DAPI': [0, 0, 1],      # Blue (#0000FF)
#         'GFP': [0.396, 0.996, 0.031],  # Green (#65fe08) 
#         'AGP': [1, 1, 0],       # Yellow (#FFFF00)
#         'Mito': [1, 0, 0],      # Red (#FF0000)
#         'Brightfield1': [1, 1, 1],     # White/Gray
#         'Brightfield2': [1, 1, 1],     # White/Gray
#         'Brightfield': [1, 1, 1]       # White/Gray
#     }
    
#     # Define layer order (first = bottom, last = top)
#     layer_order = ['DAPI', 'AGP', 'Mito', 'GFP', 'Brightfield1', 'Brightfield2', 'Brightfield']
    
#     # Sort channels by layer order
#     ordered_channels = []
#     for layer in layer_order:
#         if layer in channels:
#             ordered_channels.append(layer)
#     for ch in channels:
#         if ch not in ordered_channels:
#             ordered_channels.append(ch)
    
#     # Find global percentile across all channels to prevent saturation
#     all_values = []
#     for ch in ordered_channels:
#         all_values.extend(channel_subs[ch].flatten())
#     global_max = np.percentile(all_values, max_intensity*100)
    
#     for ch in ordered_channels:
#         img = channel_subs[ch]
#         # Normalize using global max to maintain relative intensities
#         img_norm = np.clip(img / global_max, 0, 1)
        
#         color = color_map.get(ch, [1, 1, 1])
        
#         # Direct addition without scaling
#         for i in range(3):
#             composite[:,:,i] += img_norm * color[i]
    
#     return np.clip(composite, 0, 1)