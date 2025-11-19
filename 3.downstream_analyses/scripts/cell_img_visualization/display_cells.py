"""
DEPRECATION NOTICE:
===================
This module contains legacy cell visualization code. For new projects, please use
the refactored modular system:

- cell_selector.py:    Cell selection and filtering functions
- cell_cropper.py:     Image cropping with multiple strategies
- cell_visualizer.py:  Visualization with configurable contrast control

See example_cell_visualization.py for usage examples.

This legacy file is kept for backward compatibility with existing workflows.
===================
"""

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


def viz_cell_crop(cell, cell_comp, max_intensity=.97, ax=None, axis_off=True):
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


def viz_cell_crop_multi(cell, max_intensity=0.97, ax=None, axis_off=True):
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
        # key_feat_sel = list(cell.keys())[-1]
        plot_label = f"{cell['allele']}\nChannels: {', '.join(sorted(channels))}" ## \n{cell[key_feat_sel]:.2f} | 95th intensity: {cell.get('intensity_95', 'N/A'):.1f}
        ax.text(0.05, 0.95, plot_label, color='white', fontsize=9,
                verticalalignment='top', horizontalalignment='left', 
                transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.2, linewidth=1))