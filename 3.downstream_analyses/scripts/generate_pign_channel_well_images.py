#!/usr/bin/env python
"""
Generate well-level images for PIGN variants per channel (DNA, AGP, GFP).

For each imaging channel (DNA, AGP, GFP/Protein), generates well images
for variants with AUROC > 0.9 in that channel. Skips existing images.

Usage:
    python generate_pign_channel_well_images.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from img_utils import letter_dict, channel_dict, plate_dict, channel_to_cmap
from downstream_utils import load_platemap, load_well_qc, get_well_qc_flag

# Configuration
AUROC_THRESHOLD = 0.9
GENE = "PIGN"

# Batch info
BATCH_20 = "2026_01_05_Batch_20"
BATCH_21 = "2026_01_05_Batch_21"

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
IMGS_DIR = PROJECT_ROOT / "1.image_preprocess_qc" / "inputs" / "cpg_imgs"
OUTPUT_DIR = PROJECT_ROOT / "2.snakemake_pipeline" / "outputs" / "gene_variant_well_images"
RESULTS_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"

# Channel configurations
# Note: "DNA" in AUROC files corresponds to "DAPI" imaging channel
CHANNELS = {
    "DNA": {
        "auroc_file": "altered_DNA_summary_auroc.csv",
        "auroc_col": "AUROC_Mean",
        "img_channel": "DAPI",  # DNA staining = DAPI channel
        "display_name": "DNA"
    },
    "AGP": {
        "auroc_file": "altered_AGP_summary_auroc.csv",
        "auroc_col": "AUROC_Mean",
        "img_channel": "AGP",
        "display_name": "AGP"
    },
    "GFP": {
        "auroc_file": "altered_GFP_summary_auroc.csv",
        "auroc_col": "AUROC_Mean",
        "img_channel": "GFP",
        "display_name": "GFP"
    }
}


def check_image_exists(variant: str, display_name: str, auroc: float) -> bool:
    """Check if image already exists."""
    filename = f"{variant}_{display_name}_{auroc:.3f}.png"
    return (OUTPUT_DIR / filename).exists()


def plot_variant_combined(variant: str, ref: str,
                          platemap_b20: pl.DataFrame, platemap_b21: pl.DataFrame,
                          well_qc_b20: pl.DataFrame, well_qc_b21: pl.DataFrame,
                          auroc: float, channel: str = "GFP",
                          display_name: str = None,
                          vmin_perc: float = 1., vmax_perc: float = 99.):
    """Generate combined well-level image for a variant vs reference across both batches."""
    if display_name is None:
        display_name = channel

    cmap = channel_to_cmap(channel)
    ch_num = channel_dict[channel]
    site = "05"
    timepoints = ["T1", "T2", "T3", "T4"]

    # Get well info for both batches
    var_well_b20 = None
    ref_well_b20 = None
    plate_map_b20 = None
    var_well_b21 = None
    ref_well_b21 = None
    plate_map_b21 = None

    # Batch 20
    var_rows_b20 = platemap_b20.filter(pl.col("gene_allele") == variant)
    ref_rows_b20 = platemap_b20.filter(pl.col("gene_allele") == ref)
    if len(var_rows_b20) > 0 and len(ref_rows_b20) > 0:
        var_well_b20 = var_rows_b20["well_position"][0]
        ref_well_b20 = ref_rows_b20["well_position"][0]
        plate_map_b20 = var_rows_b20["plate_map_name"][0]

    # Batch 21
    var_rows_b21 = platemap_b21.filter(pl.col("gene_allele") == variant)
    ref_rows_b21 = platemap_b21.filter(pl.col("gene_allele") == ref)
    if len(var_rows_b21) > 0 and len(ref_rows_b21) > 0:
        var_well_b21 = var_rows_b21["well_position"][0]
        ref_well_b21 = ref_rows_b21["well_position"][0]
        plate_map_b21 = var_rows_b21["plate_map_name"][0]

    if var_well_b20 is None and var_well_b21 is None:
        return None

    # Create figure: 4 rows x 4 columns
    fig, axes = plt.subplots(4, 4, figsize=(15, 16))

    def plot_well_row(row_idx, well, allele, batch, plate_map_name, well_qc):
        """Plot a single row of well images across timepoints."""
        if well is None or plate_map_name is None:
            for tidx in range(4):
                axes[row_idx, tidx].text(0.5, 0.5, "N/A", ha='center', va='center',
                                         transform=axes[row_idx, tidx].transAxes, fontsize=14)
                axes[row_idx, tidx].axis("off")
            return

        if plate_map_name not in plate_dict:
            for tidx in range(4):
                axes[row_idx, tidx].text(0.5, 0.5, "Plate not found", ha='center', va='center',
                                         transform=axes[row_idx, tidx].transAxes, fontsize=10)
                axes[row_idx, tidx].axis("off")
            return

        plate_info = plate_dict[plate_map_name]
        batch_img_dir = IMGS_DIR / batch / "images"
        letter = well[0]
        row_num = letter_dict[letter]
        col = well[1:3]

        for tidx, tp in enumerate(timepoints):
            plate_img_dir = plate_info[tp]
            img_file = f"r{row_num}c{col}f{site}p01-ch{ch_num}sk1fk1fl1.tiff"
            img_path = batch_img_dir / plate_img_dir / "Images" / img_file

            if img_path.exists():
                img = imread(str(img_path), as_gray=True)
                display_vmin = np.percentile(img, vmin_perc)
                display_vmax = np.percentile(img, vmax_perc)
                axes[row_idx, tidx].imshow(img, vmin=display_vmin, vmax=display_vmax, cmap=cmap)

                # Check QC flag
                is_bg = get_well_qc_flag(well_qc, plate_img_dir, well, channel)

                # Add labels
                int_95 = int(round(np.percentile(img, 95)))
                label = f"{channel}:{plate_map_name},{tp}\nWell:{well},Site:{site}\n{allele}"
                axes[row_idx, tidx].text(0.03, 0.97, label, color='white', fontsize=10,
                                         verticalalignment='top', horizontalalignment='left',
                                         transform=axes[row_idx, tidx].transAxes,
                                         bbox=dict(facecolor='black', alpha=0.3, linewidth=2))

                if is_bg:
                    axes[row_idx, tidx].text(0.03, 0.03, "FLAG:\nOnly Background\nNoise is Detected",
                                             color='red', fontsize=10, verticalalignment='bottom',
                                             horizontalalignment='left', transform=axes[row_idx, tidx].transAxes,
                                             bbox=dict(facecolor='white', alpha=0.3, linewidth=2))

                axes[row_idx, tidx].text(0.97, 0.03, f"95th Intensity:{int_95}\nvmin:{vmin_perc:.1f}%\nvmax:{vmax_perc:.1f}%",
                                         color='white', fontsize=10, verticalalignment='bottom',
                                         horizontalalignment='right', transform=axes[row_idx, tidx].transAxes,
                                         bbox=dict(facecolor='black', alpha=0.3, linewidth=2))
            else:
                axes[row_idx, tidx].text(0.5, 0.5, "Image not found", ha='center', va='center',
                                         transform=axes[row_idx, tidx].transAxes, fontsize=10)
            axes[row_idx, tidx].axis("off")

    # Plot all 4 rows
    plot_well_row(0, var_well_b20, variant, BATCH_20, plate_map_b20, well_qc_b20)
    plot_well_row(1, ref_well_b20, ref, BATCH_20, plate_map_b20, well_qc_b20)
    plot_well_row(2, var_well_b21, variant, BATCH_21, plate_map_b21, well_qc_b21)
    plot_well_row(3, ref_well_b21, ref, BATCH_21, plate_map_b21, well_qc_b21)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=-0.2, top=0.99)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}_{display_name}_{auroc:.3f}.png"
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return output_path


def main():
    print("=" * 80)
    print("Generating Per-Channel Well Images for PIGN Variants (AUROC > 0.9)")
    print("=" * 80)

    # Load platemaps
    print("\nLoading platemaps...")
    platemap_b20 = load_platemap(BATCH_20)
    platemap_b21 = load_platemap(BATCH_21)
    print(f"  Batch 20: {len(platemap_b20)} entries")
    print(f"  Batch 21: {len(platemap_b21)} entries")

    # Load QC data
    well_qc_b20 = load_well_qc(BATCH_20)
    well_qc_b21 = load_well_qc(BATCH_21)

    # Process each channel
    for channel_name, channel_config in CHANNELS.items():
        print(f"\n{'='*60}")
        print(f"Processing {channel_name} channel")
        print("=" * 60)

        # Load channel AUROC results
        auroc_file = RESULTS_DIR / channel_config["auroc_file"]
        if not auroc_file.exists():
            print(f"  Warning: {auroc_file} not found, skipping")
            continue

        results = pl.read_csv(auroc_file)

        # Filter for PIGN variants with AUROC > threshold
        pign_results = results.filter(
            (pl.col("allele_0").str.starts_with(f"{GENE}_")) &
            (pl.col(channel_config["auroc_col"]) > AUROC_THRESHOLD)
        ).sort(channel_config["auroc_col"], descending=True, nulls_last=True)

        print(f"  Found {len(pign_results)} PIGN variants with {channel_name} AUROC > {AUROC_THRESHOLD}")

        if len(pign_results) == 0:
            continue

        # Generate images
        generated = 0
        skipped = 0

        for row in pign_results.iter_rows(named=True):
            variant = row["allele_0"]
            auroc = row[channel_config["auroc_col"]]
            img_channel = channel_config["img_channel"]
            display_name = channel_config["display_name"]

            # Check if image exists
            if check_image_exists(variant, display_name, auroc):
                skipped += 1
                continue

            print(f"  Generating: {variant} ({display_name} AUROC: {auroc:.3f})")

            output_path = plot_variant_combined(
                variant, GENE,
                platemap_b20, platemap_b21,
                well_qc_b20, well_qc_b21,
                auroc, channel=img_channel,
                display_name=display_name
            )

            if output_path:
                generated += 1
            else:
                print(f"    Warning: Could not generate image for {variant}")

        print(f"\n  {channel_name} Summary:")
        print(f"    Generated: {generated}")
        print(f"    Skipped (existing): {skipped}")
        print(f"    Total: {generated + skipped}")

    print("\n" + "=" * 80)
    print(f"All images saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
