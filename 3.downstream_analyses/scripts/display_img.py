import os
from functools import reduce
import operator
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
import sys
sys.path.append("../..")
from img_utils import *


def plot_allele(pm, variant, sel_channel, plate_img_qc, auroc_df=None, site="05", ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir="", output_dir=""):
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

    fig, axes = plt.subplots((len(wt_wells)+len(var_wells))*2, 4, figsize=(15, (len(wt_wells)+len(var_wells))*8), sharex=True, sharey=True)
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
            
            plate_img_dir = plate_dict[sel_plate][f"T{i%4+1}"]
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
            
            plot_idx = i+wt_var*4*2
            # print(i, wt_var, plot_idx)
            axes.flatten()[plot_idx].imshow(img, vmin=0, vmax=np.percentile(img, max_intensity*100), cmap=cmap)
            plot_label = f"{sel_channel}:{sel_plate},T{i%4+1}\nWell:{well},Site:{site}\n{allele}"
            axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
                    verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                    bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
            if is_bg:
                axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                    bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
            int_95 = str(int(round(np.percentile(img, 95))))
            axes.flatten()[plot_idx].text(0.97, 0.03, f"95th Intensity:{int_95}\nSet vmax:{max_intensity*100:.0f}th perc.", color='white', fontsize=10,
                           verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                           bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
            axes.flatten()[plot_idx].axis("off")
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
    if display:
        plt.show()
    
    if output_dir:
        file_name = f"{variant}_{sel_channel}"
        if auroc:
            file_name = f"{file_name}_{auroc:.3f}"
        if ref_well:
            file_name = f"{file_name}_REF-{'_'.join(ref_well)}"
        if var_well:
            file_name = f"{file_name}_VAR-{'_'.join(var_well)}"
        fig.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=400, bbox_inches='tight')
        
    plt.close(fig)


def plot_allele_single_plate(pm, variant, sel_channel, plate_img_qc, auroc_df=None, site="05", ref_well=[], var_well=[], max_intensity=0.99, display=False, imgs_dir="", output_dir=""):
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
        axes.flatten()[plot_idx].imshow(img, vmin=0, vmax=np.percentile(img, max_intensity*100), cmap=cmap)
        plot_label = f"{sel_channel}:{sel_plate}\nWell:{well},Site:{site}\n{allele}"
        axes.flatten()[plot_idx].text(0.03, 0.97, plot_label, color='white', fontsize=10,
                verticalalignment='top', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
        if is_bg:
            axes.flatten()[plot_idx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected", color='red', fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', transform=axes.flatten()[plot_idx].transAxes,
                bbox=dict(facecolor='white', alpha=0.3, linewidth=2))
        int_95 = str(int(round(np.percentile(img, 95))))
        axes.flatten()[plot_idx].text(0.97, 0.03, f"95th Intensity:{int_95}\nSet vmax:{max_intensity*100:.0f}th perc.", color='white', fontsize=10,
                       verticalalignment='bottom', horizontalalignment='right', transform=axes.flatten()[plot_idx].transAxes,
                       bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
        axes.flatten()[plot_idx].axis("off")
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=-0.2, top=.99)
    
    if display:
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


def crop_allele(allele: str, profile_df: pl.DataFrame, meta_plate: str, rep: str="", well: str="", site: str="5") -> None:
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
    if rep:
        allele_df = profile_df.filter(
            (pl.col("Metadata_gene_allele")==allele) &
            (pl.col("Metadata_Site")==int(site)) &
            (pl.col("Metadata_plate_map_name").str.contains(meta_plate)) &
            (pl.col("Metadata_Plate").str.contains(rep))
        )
        
    if well:
        allele_df = profile_df.filter(
            (pl.col("Metadata_gene_allele")==allele) &
            (pl.col("Metadata_Site")==int(site)) &
            (pl.col("Metadata_plate_map_name").str.contains(meta_plate)) &
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

