#!/usr/bin/env python
"""
Generate well-level and cell crop images for PIGN variants.

This script generates:
1. Well-level images showing variant vs reference across timepoints
2. Cell crop images for top PIGN variants

Usage:
    python generate_pign_images.py [--top_n 20] [--channels GFP,AGP]
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
sys.path.insert(0, str(SCRIPT_DIR / "cell_img_visualization"))
from img_utils import *
from downstream_utils import load_platemap

# Output directories
WELL_IMG_DIR = PROJECT_ROOT / "2.snakemake_pipeline" / "outputs" / "gene_variant_well_images" / "pign"
CELL_CROP_DIR = PROJECT_ROOT / "2.snakemake_pipeline" / "outputs" / "gene_variant_cell_crop_images" / "pign"

# Batch info
BATCH_20 = "2026_01_05_Batch_20"
BATCH_21 = "2026_01_05_Batch_21"


def load_classification_results() -> pl.DataFrame:
    """Load classification results for PIGN variants."""
    results_dir = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"
    gfp_path = results_dir / "altered_GFP_summary_auroc.csv"

    if not gfp_path.exists():
        print("Error: Classification results not found. Run generate_batch20_21_results.py first.")
        return pl.DataFrame()

    df = pl.read_csv(gfp_path)
    # Filter for PIGN variants
    pign_df = df.filter(pl.col("allele_0").str.starts_with("PIGN"))
    return pign_df.sort("AUROC_Mean", descending=True, nulls_last=True)


def generate_well_image(variant: str, ref: str, channel: str, batch: str, platemap: pl.DataFrame,
                        output_dir: Path, auroc: float = None):
    """Generate a well-level image for a variant vs reference."""
    cmap = channel_to_cmap(channel)
    ch_num = channel_dict[channel]

    # Get wells for variant and reference
    var_rows = platemap.filter(pl.col("gene_allele") == variant)
    ref_rows = platemap.filter(pl.col("gene_allele") == ref)

    if len(var_rows) == 0 or len(ref_rows) == 0:
        print(f"  Warning: Wells not found for {variant} or {ref}")
        return

    var_well = var_rows["well_position"][0]
    ref_well = ref_rows["well_position"][0]

    # Get plate info
    plate_map_name = var_rows["plate_map_name"][0]

    # Determine batch image directory
    imgs_dir = PROJECT_ROOT / "1.image_preprocess_qc" / "inputs" / "cpg_imgs" / batch / "images"

    if plate_map_name not in plate_dict:
        print(f"  Warning: Plate map {plate_map_name} not found in plate_dict")
        return

    plate_info = plate_dict[plate_map_name]

    # Create figure with 2 rows (ref, var) x 4 columns (T1-T4)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    site = "05"  # Center site

    for tidx, timepoint in enumerate(["T1", "T2", "T3", "T4"]):
        plate_img_dir = plate_info[timepoint]

        # Plot reference
        ref_row = letter_dict[ref_well[0]]
        ref_col = ref_well[1:3]
        ref_img_file = f"r{ref_row}c{ref_col}f{site}p01-ch{ch_num}sk1fk1fl1.tiff"
        ref_img_path = imgs_dir / plate_img_dir / "Images" / ref_img_file

        if ref_img_path.exists():
            ref_img = imread(str(ref_img_path), as_gray=True)
            vmin, vmax = np.percentile(ref_img, [1, 99])
            axes[0, tidx].imshow(ref_img, vmin=vmin, vmax=vmax, cmap=cmap)
            axes[0, tidx].set_title(f"{ref}\n{timepoint}, Well:{ref_well}")
        else:
            axes[0, tidx].text(0.5, 0.5, "Image not found", ha='center', va='center', transform=axes[0, tidx].transAxes)
            axes[0, tidx].set_title(f"{ref}\n{timepoint}, Well:{ref_well}")
        axes[0, tidx].axis("off")

        # Plot variant
        var_row = letter_dict[var_well[0]]
        var_col = var_well[1:3]
        var_img_file = f"r{var_row}c{var_col}f{site}p01-ch{ch_num}sk1fk1fl1.tiff"
        var_img_path = imgs_dir / plate_img_dir / "Images" / var_img_file

        if var_img_path.exists():
            var_img = imread(str(var_img_path), as_gray=True)
            vmin, vmax = np.percentile(var_img, [1, 99])
            axes[1, tidx].imshow(var_img, vmin=vmin, vmax=vmax, cmap=cmap)
            axes[1, tidx].set_title(f"{variant}\n{timepoint}, Well:{var_well}")
        else:
            axes[1, tidx].text(0.5, 0.5, "Image not found", ha='center', va='center', transform=axes[1, tidx].transAxes)
            axes[1, tidx].set_title(f"{variant}\n{timepoint}, Well:{var_well}")
        axes[1, tidx].axis("off")

    # Add overall title
    title = f"{variant} vs {ref} - {channel}"
    if auroc is not None:
        title += f" (AUROC: {auroc:.3f})"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}_{batch.split('_')[-1]}_{channel}"
    if auroc is not None:
        filename += f"_{auroc:.3f}"
    filename += ".png"
    fig.savefig(output_dir / filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return output_dir / filename


def main():
    parser = argparse.ArgumentParser(description="Generate PIGN variant images")
    parser.add_argument("--top_n", type=int, default=30, help="Number of top variants to visualize")
    parser.add_argument("--channels", type=str, default="GFP,AGP", help="Comma-separated channels to visualize")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating PIGN Variant Images")
    print("=" * 60)

    # Load classification results
    pign_results = load_classification_results()
    if len(pign_results) == 0:
        print("No PIGN results found.")
        return

    print(f"\nFound {len(pign_results)} PIGN variants")
    print(f"Top variants by AUROC:")
    print(pign_results.select(["allele_0", "AUROC_Mean"]).head(10))

    # Load platemaps
    platemap_b20 = load_platemap(BATCH_20)
    platemap_b21 = load_platemap(BATCH_21)

    print(f"\nLoaded platemaps: Batch 20 ({len(platemap_b20)} wells), Batch 21 ({len(platemap_b21)} wells)")

    channels = args.channels.split(",")

    # Get top N variants
    top_variants = pign_results.head(args.top_n)

    print(f"\nGenerating well images for top {len(top_variants)} PIGN variants...")

    for row in top_variants.iter_rows(named=True):
        variant = row["allele_0"]
        auroc = row["AUROC_Mean"]
        ref = "PIGN"  # Reference is always PIGN

        print(f"\n  Processing {variant} (AUROC: {auroc:.3f})...")

        for channel in channels:
            # Generate for Batch 20
            if len(platemap_b20.filter(pl.col("gene_allele") == variant)) > 0:
                output_path = generate_well_image(
                    variant, ref, channel, BATCH_20, platemap_b20,
                    WELL_IMG_DIR, auroc
                )
                if output_path:
                    print(f"    Saved: {output_path.name}")

            # Generate for Batch 21
            if len(platemap_b21.filter(pl.col("gene_allele") == variant)) > 0:
                output_path = generate_well_image(
                    variant, ref, channel, BATCH_21, platemap_b21,
                    WELL_IMG_DIR, auroc
                )
                if output_path:
                    print(f"    Saved: {output_path.name}")

    print("\n" + "=" * 60)
    print("Image Generation Complete!")
    print(f"Well images saved to: {WELL_IMG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
