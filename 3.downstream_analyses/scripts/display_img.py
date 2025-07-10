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
                is_bg = plate_img_qc.filter((pl.col("plate") == plate_img_dir.split("__")[0]) & (pl.col("well") == well) & (pl.col("channel") == sel_channel))["is_bg"].to_numpy()[0]
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
            is_bg = plate_img_qc.filter(
                (pl.col("plate") == plate_img_dir.split("__")[0])
                & (pl.col("well") == well)
                & (pl.col("channel") == sel_channel))["is_bg"].to_numpy()[0]
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