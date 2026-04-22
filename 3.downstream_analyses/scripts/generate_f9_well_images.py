#!/usr/bin/env python
"""
Generate AGP channel well-level images for randomly sampled F9 variants from Batch 13/14.

Layout: 4 rows x 4 columns per image
- Row 1: Variant, Batch 13, T1-T4
- Row 2: Reference (F9 wildtype), Batch 13, T1-T4
- Row 3: Variant, Batch 14, T1-T4
- Row 4: Reference (F9 wildtype), Batch 14, T1-T4

Usage:
    python generate_f9_well_images.py --n_samples 10 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from skimage.io import imread

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from img_utils import letter_dict, channel_dict, plate_dict, channel_to_cmap
from downstream_utils import load_platemap, load_well_qc, get_well_qc_flag

# Batch info
BATCH_13 = "2025_01_27_Batch_13"
BATCH_14 = "2025_01_28_Batch_14"

# Gene of interest
GENE = "F9"

# Channel
CHANNEL = "AGP"

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
IMGS_DIR = PROJECT_ROOT / "1.image_preprocess_qc" / "inputs" / "cpg_imgs"
OUTPUT_DIR = PROJECT_ROOT / "2.snakemake_pipeline" / "outputs" / "gene_variant_well_images" / GENE


def plot_f9_variant_combined(variant: str, ref: str,
                              platemap_b13: pl.DataFrame, platemap_b14: pl.DataFrame,
                              well_qc_b13: pl.DataFrame, well_qc_b14: pl.DataFrame,
                              channel: str = "AGP",
                              vmin_perc: float = 1., vmax_perc: float = 99.):
    """Generate combined well-level image for an F9 variant vs reference across both batches."""

    cmap = channel_to_cmap(channel)
    ch_num = channel_dict[channel]
    site = "05"
    timepoints = ["T1", "T2", "T3", "T4"]

    # Get well info for both batches
    var_well_b13 = None
    ref_well_b13 = None
    plate_map_b13 = None
    var_well_b14 = None
    ref_well_b14 = None
    plate_map_b14 = None

    # Batch 13
    var_rows_b13 = platemap_b13.filter(pl.col("gene_allele") == variant)
    ref_rows_b13 = platemap_b13.filter(pl.col("gene_allele") == ref)
    if len(var_rows_b13) > 0 and len(ref_rows_b13) > 0:
        var_well_b13 = var_rows_b13["well_position"][0]
        ref_well_b13 = ref_rows_b13["well_position"][0]
        plate_map_b13 = var_rows_b13["plate_map_name"][0]

    # Batch 14
    var_rows_b14 = platemap_b14.filter(pl.col("gene_allele") == variant)
    ref_rows_b14 = platemap_b14.filter(pl.col("gene_allele") == ref)
    if len(var_rows_b14) > 0 and len(ref_rows_b14) > 0:
        var_well_b14 = var_rows_b14["well_position"][0]
        ref_well_b14 = ref_rows_b14["well_position"][0]
        plate_map_b14 = var_rows_b14["plate_map_name"][0]

    if var_well_b13 is None and var_well_b14 is None:
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
                axes[row_idx, tidx].text(0.5, 0.5, f"Plate not found:\n{plate_map_name}", ha='center', va='center',
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
    plot_well_row(0, var_well_b13, variant, BATCH_13, plate_map_b13, well_qc_b13)
    plot_well_row(1, ref_well_b13, ref, BATCH_13, plate_map_b13, well_qc_b13)
    plot_well_row(2, var_well_b14, variant, BATCH_14, plate_map_b14, well_qc_b14)
    plot_well_row(3, ref_well_b14, ref, BATCH_14, plate_map_b14, well_qc_b14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=-0.2, top=0.99)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}_{channel}.png"
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate F9 well images (AGP channel)")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of variants to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Generating AGP Well Images for {args.n_samples} Random F9 Variants")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Set random seed
    np.random.seed(args.seed)

    # Load platemaps
    print("\nLoading platemaps...")
    platemap_b13 = load_platemap(BATCH_13)
    platemap_b14 = load_platemap(BATCH_14)
    print(f"  Batch 13: {len(platemap_b13)} entries")
    print(f"  Batch 14: {len(platemap_b14)} entries")

    # Load QC data (if available)
    well_qc_b13 = load_well_qc(BATCH_13)
    well_qc_b14 = load_well_qc(BATCH_14)

    # Find F9 variants (excluding the wildtype reference)
    f9_variants_b13 = platemap_b13.filter(
        (pl.col("gene_allele").str.starts_with(f"{GENE}_")) &
        (pl.col("node_type") == "allele")
    )["gene_allele"].unique().to_list()

    print(f"\nFound {len(f9_variants_b13)} F9 variants in Batch 13")

    # Randomly sample variants
    n_to_sample = min(args.n_samples, len(f9_variants_b13))
    sampled_variants = np.random.choice(f9_variants_b13, size=n_to_sample, replace=False)

    print(f"\nRandomly sampled {n_to_sample} variants:")
    for v in sampled_variants:
        print(f"  - {v}")

    # Generate images
    print(f"\nGenerating {CHANNEL} well images...")
    generated = 0

    for variant in sampled_variants:
        print(f"\n  Processing: {variant}")

        output_path = plot_f9_variant_combined(
            variant, GENE,
            platemap_b13, platemap_b14,
            well_qc_b13, well_qc_b14,
            channel=CHANNEL
        )

        if output_path:
            print(f"    Saved: {output_path.name}")
            generated += 1
        else:
            print(f"    Warning: Could not generate image for {variant}")

    print("\n" + "=" * 60)
    print(f"Generated {generated} images")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
