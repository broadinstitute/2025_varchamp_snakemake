#!/usr/bin/env python
"""
Generate well-level GFP images for top PIGN variants in Batch 20/21.

This script generates well images combining both biological replicates (Batch 20 and 21)
in a single image with 4 rows x 4 columns layout:
- Row 1: Variant, Batch 20, T1-T4
- Row 2: Reference, Batch 20, T1-T4
- Row 3: Variant, Batch 21, T1-T4
- Row 4: Reference, Batch 21, T1-T4

Usage:
    python generate_pign_well_images.py --top_n 10
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.io import imread

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from img_utils import letter_dict, channel_dict, plate_dict, channel_to_cmap
from downstream_utils import load_platemap, load_well_qc, get_well_qc_flag

# Batch info
BATCH_20 = "2026_01_05_Batch_20"
BATCH_21 = "2026_01_05_Batch_21"

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
IMGS_DIR = PROJECT_ROOT / "1.image_preprocess_qc" / "inputs" / "cpg_imgs"
OUTPUT_DIR = PROJECT_ROOT / "2.snakemake_pipeline" / "outputs" / "gene_variant_well_images"
RESULTS_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"


def plot_pign_variant_combined(variant: str, ref: str,
                                platemap_b20: pl.DataFrame, platemap_b21: pl.DataFrame,
                                well_qc_b20: pl.DataFrame, well_qc_b21: pl.DataFrame,
                                auroc: float, channel: str = "GFP",
                                vmin_perc: float = 1., vmax_perc: float = 99.):
    """Generate combined well-level image for a PIGN variant vs reference across both batches.

    Layout: 4 rows x 4 columns
    - Rows 1-2: Batch 20 (variant, reference)
    - Rows 3-4: Batch 21 (variant, reference)
    - Columns: T1, T2, T3, T4
    """

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

    if (var_well_b20 is None and var_well_b21 is None):
        print(f"    Warning: No wells found for {variant}")
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
    # Row 0: Variant, Batch 20
    plot_well_row(0, var_well_b20, variant, BATCH_20, plate_map_b20, well_qc_b20)
    # Row 1: Reference, Batch 20
    plot_well_row(1, ref_well_b20, ref, BATCH_20, plate_map_b20, well_qc_b20)
    # Row 2: Variant, Batch 21
    plot_well_row(2, var_well_b21, variant, BATCH_21, plate_map_b21, well_qc_b21)
    # Row 3: Reference, Batch 21
    plot_well_row(3, ref_well_b21, ref, BATCH_21, plate_map_b21, well_qc_b21)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=-0.2, top=0.99)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}_{channel}_{auroc:.3f}.png"
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate PIGN well images")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top variants")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Generating Well Images for Top {args.top_n} PIGN Variants")
    print("=" * 60)

    # Load classification results
    gfp_results = pl.read_csv(RESULTS_DIR / "altered_GFP_summary_auroc.csv")
    pign_results = gfp_results.filter(pl.col("allele_0").str.starts_with("PIGN_"))
    pign_results = pign_results.sort("AUROC_Mean", descending=True, nulls_last=True)
    top_pign = pign_results.head(args.top_n)

    print(f"\nTop {args.top_n} PIGN variants by GFP AUROC:")
    print(top_pign.select(["allele_0", "AUROC_Mean"]))

    # Load platemaps
    platemap_b20 = load_platemap(BATCH_20)
    platemap_b21 = load_platemap(BATCH_21)

    # Load QC data
    well_qc_b20 = load_well_qc(BATCH_20)
    well_qc_b21 = load_well_qc(BATCH_21)

    print(f"\nGenerating combined well images (both batches in one image)...")

    for row in top_pign.iter_rows(named=True):
        variant = row["allele_0"]
        auroc = row["AUROC_Mean"]
        ref = "PIGN"

        print(f"\n  {variant} (AUROC: {auroc:.3f})")

        output_path = plot_pign_variant_combined(
            variant, ref,
            platemap_b20, platemap_b21,
            well_qc_b20, well_qc_b21,
            auroc, channel="GFP"
        )
        if output_path:
            print(f"    Saved: {output_path.name}")

    print("\n" + "=" * 60)
    print(f"Well images saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
