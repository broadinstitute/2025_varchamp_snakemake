#!/usr/bin/env python
"""
Generate well images and cell crops for ALL non-PIGN alleles in Batch 20/21.

This script generates:
1. Well-level GFP images for each variant vs wildtype
2. Cell crop images showing representative cells

Only GFP channel is generated.

Usage:
    python generate_gregor_allele_images.py [--cells_per_allele 8]
"""

import argparse
import os
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
sys.path.insert(0, str(SCRIPT_DIR / "cell_img_visualization"))

from img_utils import letter_dict, channel_dict, plate_dict, channel_to_cmap, TIFF_IMGS_DIR
from downstream_utils import (
    load_platemap,
    load_well_qc,
    get_well_qc_flag,
    load_batch_profiles_with_bbox,
)

# Import cell visualization utilities
try:
    from cell_selector import filter_cells_by_metadata, filter_by_quality_metrics, select_cells_percentile_random
    from cell_cropper import load_multichannel_cell_crops
    from cell_visualizer import viz_cell_single_channel
    HAS_CELL_VIZ = True
except ImportError:
    HAS_CELL_VIZ = False
    print("Warning: Cell visualization modules not found. Skipping cell crops.")

# Batch info
BATCH_20 = "2026_01_05_Batch_20"
BATCH_21 = "2026_01_05_Batch_21"
BATCHES = [BATCH_20, BATCH_21]

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
IMGS_DIR = PROJECT_ROOT / "1.image_preprocess_qc" / "inputs" / "cpg_imgs"
OUTPUT_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "gregor_alleles" / "images"
RESULTS_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"

# Genes to exclude
EXCLUDE_GENES = ["PIGN"]
TC = ["EGFP"]
NC = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
PC = ["ALK", "ALK_Arg1275Gln", "PTK2B"]
ALL_CONTROLS = TC + NC + PC


def get_all_non_pign_alleles() -> pl.DataFrame:
    """Get all non-PIGN alleles from classification summary."""
    summary_path = RESULTS_DIR / "imaging_analyses_classification_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Classification summary not found: {summary_path}")

    summary = pl.read_csv(summary_path)

    # Filter out PIGN and controls
    non_pign = summary.filter(
        ~pl.col("allele_0").str.starts_with("PIGN_") &
        ~pl.col("allele_0").is_in(ALL_CONTROLS)
    )

    # Sort by GFP AUROC
    non_pign = non_pign.sort("AUROC_GFP_Mean", descending=True)

    return non_pign


def plot_variant_well_images(variant: str, ref: str,
                              platemap_b20: pl.DataFrame, platemap_b21: pl.DataFrame,
                              well_qc_b20: pl.DataFrame, well_qc_b21: pl.DataFrame,
                              auroc: float, output_dir: Path,
                              vmin_perc: float = 1., vmax_perc: float = 99.):
    """Generate combined well-level GFP image for a variant vs reference across both batches."""

    channel = "GFP"
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
    fig, axes = plt.subplots(4, 4, figsize=(14, 15))

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
                batch_short = "B20" if "Batch_20" in batch else "B21"
                label = f"{batch_short},{tp}\nWell:{well}\n{allele}"
                axes[row_idx, tidx].text(0.03, 0.97, label, color='white', fontsize=9,
                                         verticalalignment='top', horizontalalignment='left',
                                         transform=axes[row_idx, tidx].transAxes,
                                         bbox=dict(facecolor='black', alpha=0.4, linewidth=1))

                if is_bg:
                    axes[row_idx, tidx].text(0.03, 0.03, "BG FLAG",
                                             color='red', fontsize=10, fontweight='bold',
                                             verticalalignment='bottom', horizontalalignment='left',
                                             transform=axes[row_idx, tidx].transAxes,
                                             bbox=dict(facecolor='white', alpha=0.5))

                axes[row_idx, tidx].text(0.97, 0.03, f"I95:{int_95}",
                                         color='white', fontsize=9, verticalalignment='bottom',
                                         horizontalalignment='right', transform=axes[row_idx, tidx].transAxes,
                                         bbox=dict(facecolor='black', alpha=0.4))
            else:
                axes[row_idx, tidx].text(0.5, 0.5, "Image N/A", ha='center', va='center',
                                         transform=axes[row_idx, tidx].transAxes, fontsize=10)
            axes[row_idx, tidx].axis("off")

    # Plot all 4 rows
    plot_well_row(0, var_well_b20, variant, BATCH_20, plate_map_b20, well_qc_b20)
    plot_well_row(1, ref_well_b20, ref, BATCH_20, plate_map_b20, well_qc_b20)
    plot_well_row(2, var_well_b21, variant, BATCH_21, plate_map_b21, well_qc_b21)
    plot_well_row(3, ref_well_b21, ref, BATCH_21, plate_map_b21, well_qc_b21)

    # Add title
    fig.suptitle(f"{variant} vs {ref} - GFP (AUROC: {auroc:.3f})", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.05, top=0.95)

    # Save
    well_dir = output_dir / "well_images"
    well_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}_GFP_{auroc:.3f}.png"
    output_path = well_dir / filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return output_path


def load_batch_profiles(batch: str) -> pl.DataFrame:
    return load_batch_profiles_with_bbox(batch)


def generate_cell_crop_figure(variant: str, ref: str, profiles: pl.DataFrame,
                              batch: str, auroc: float, output_dir: Path,
                              n_cells: int = 8):
    """Generate cell crop figure for a variant vs reference."""

    if not HAS_CELL_VIZ:
        return None

    # Filter cells by allele
    var_profiles = filter_cells_by_metadata(profiles, allele=variant)
    ref_profiles = filter_cells_by_metadata(profiles, allele=ref)

    if len(var_profiles) == 0 or len(ref_profiles) == 0:
        return None

    # Apply quality filters
    var_profiles = filter_by_quality_metrics(var_profiles, min_edge_dist=50)
    ref_profiles = filter_by_quality_metrics(ref_profiles, min_edge_dist=50)

    actual_n_cells = min(n_cells, len(var_profiles), len(ref_profiles))
    if actual_n_cells < 3:
        return None

    # Select cells from middle percentile range
    try:
        var_cells = select_cells_percentile_random(
            var_profiles, feature="Cells_AreaShape_Area",
            percentile_bins=[(30, 70)], n_per_bin=actual_n_cells * 3, seed=42
        )
        ref_cells = select_cells_percentile_random(
            ref_profiles, feature="Cells_AreaShape_Area",
            percentile_bins=[(30, 70)], n_per_bin=actual_n_cells * 3, seed=42
        )
    except Exception:
        return None

    if len(var_cells) == 0 or len(ref_cells) == 0:
        return None

    # Create figure: 2 rows x n_cells columns
    fig, axes = plt.subplots(2, actual_n_cells, figsize=(actual_n_cells * 2.5, 5))
    if actual_n_cells == 1:
        axes = axes.reshape(2, 1)

    channels = ["GFP"]

    # Plot variant cells (top row)
    cell_idx = 0
    plotted = 0
    while plotted < actual_n_cells and cell_idx < len(var_cells):
        try:
            cell_dict = var_cells.row(cell_idx, named=True)
            var_crops = load_multichannel_cell_crops(
                cell_dict, channels, imgs_dir=str(TIFF_IMGS_DIR),
                method='fixed', crop_size=128, site_col="Metadata_Site"
            )

            if np.std(var_crops["GFP"]) < 30:
                cell_idx += 1
                continue

            viz_cell_single_channel(
                var_crops['GFP'], channel='GFP', ax=axes[0, plotted],
                axis_off=True, percentile_low=1.0, percentile_high=99.0
            )
            plotted += 1
        except Exception:
            pass
        cell_idx += 1

    for i in range(plotted, actual_n_cells):
        axes[0, i].axis('off')

    # Plot reference cells (bottom row)
    cell_idx = 0
    plotted = 0
    while plotted < actual_n_cells and cell_idx < len(ref_cells):
        try:
            cell_dict = ref_cells.row(cell_idx, named=True)
            ref_crops = load_multichannel_cell_crops(
                cell_dict, channels, imgs_dir=str(TIFF_IMGS_DIR),
                method='fixed', crop_size=128, site_col="Metadata_Site"
            )

            if np.std(ref_crops["GFP"]) < 30:
                cell_idx += 1
                continue

            viz_cell_single_channel(
                ref_crops['GFP'], channel='GFP', ax=axes[1, plotted],
                axis_off=True, percentile_low=1.0, percentile_high=99.0
            )
            plotted += 1
        except Exception:
            pass
        cell_idx += 1

    for i in range(plotted, actual_n_cells):
        axes[1, i].axis('off')

    # Add row labels
    gene = variant.split("_")[0]
    short_var = variant.replace(f"{gene}_", "")
    axes[0, 0].text(-0.1, 0.5, short_var[:15], transform=axes[0, 0].transAxes,
                    fontsize=10, fontweight='bold', rotation=90, va='center', ha='right')
    axes[1, 0].text(-0.1, 0.5, f'{gene} (Ref)', transform=axes[1, 0].transAxes,
                    fontsize=10, fontweight='bold', rotation=90, va='center', ha='right')

    # Add title
    batch_short = "B20" if "Batch_20" in batch else "B21"
    fig.suptitle(f"{variant} vs {ref} - {batch_short} GFP (AUROC: {auroc:.3f})",
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.08, top=0.90)

    # Save
    cell_dir = output_dir / "cell_crops"
    cell_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}_{batch_short}_GFP_cells_{auroc:.3f}.png"
    output_path = cell_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate images for all non-PIGN alleles")
    parser.add_argument("--cells_per_allele", type=int, default=8, help="Number of cells per allele")
    parser.add_argument("--skip_wells", action="store_true", help="Skip well images")
    parser.add_argument("--skip_cells", action="store_true", help="Skip cell crops")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating GFP Images for ALL Non-PIGN Alleles (Batch 20 & 21)")
    print("=" * 70)

    # Get all non-PIGN alleles
    alleles_df = get_all_non_pign_alleles()
    print(f"\nTotal non-PIGN alleles: {len(alleles_df)}")

    # Load platemaps
    print("\nLoading platemaps...")
    platemap_b20 = load_platemap(BATCH_20)
    platemap_b21 = load_platemap(BATCH_21)
    print(f"  Batch 20 platemap entries: {len(platemap_b20)}")
    print(f"  Batch 21 platemap entries: {len(platemap_b21)}")

    # Load QC data
    well_qc_b20 = load_well_qc(BATCH_20)
    well_qc_b21 = load_well_qc(BATCH_21)

    # Load profiles for cell crops
    if not args.skip_cells and HAS_CELL_VIZ:
        print("\nLoading cell profiles...")
        profiles_b20 = load_batch_profiles(BATCH_20)
        profiles_b21 = load_batch_profiles(BATCH_21)
        print(f"  Batch 20: {len(profiles_b20)} cells")
        print(f"  Batch 21: {len(profiles_b21)} cells")
    else:
        profiles_b20 = pl.DataFrame()
        profiles_b21 = pl.DataFrame()

    # Process each allele
    print(f"\nProcessing {len(alleles_df)} alleles...")
    print("-" * 70)

    well_count = 0
    cell_count = 0

    for i, row in enumerate(alleles_df.iter_rows(named=True)):
        variant = row["allele_0"]
        auroc = row["AUROC_GFP_Mean"]
        gene = variant.split("_")[0] if "_" in variant else variant

        print(f"\n[{i+1}/{len(alleles_df)}] {variant} (GFP AUROC: {auroc:.3f})")

        # Generate well images
        if not args.skip_wells:
            well_path = plot_variant_well_images(
                variant, gene, platemap_b20, platemap_b21,
                well_qc_b20, well_qc_b21, auroc, OUTPUT_DIR
            )
            if well_path:
                print(f"  Well image: {well_path.name}")
                well_count += 1

        # Generate cell crops
        if not args.skip_cells and HAS_CELL_VIZ:
            # Check which batch has this variant
            var_in_b20 = len(profiles_b20.filter(pl.col("Metadata_gene_allele") == variant)) > 0
            var_in_b21 = len(profiles_b21.filter(pl.col("Metadata_gene_allele") == variant)) > 0

            if var_in_b20:
                cell_path = generate_cell_crop_figure(
                    variant, gene, profiles_b20, BATCH_20, auroc, OUTPUT_DIR, args.cells_per_allele
                )
                if cell_path:
                    print(f"  Cell crops (B20): {cell_path.name}")
                    cell_count += 1

            if var_in_b21:
                cell_path = generate_cell_crop_figure(
                    variant, gene, profiles_b21, BATCH_21, auroc, OUTPUT_DIR, args.cells_per_allele
                )
                if cell_path:
                    print(f"  Cell crops (B21): {cell_path.name}")
                    cell_count += 1

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total alleles processed: {len(alleles_df)}")
    print(f"  Well images generated: {well_count}")
    print(f"  Cell crop images generated: {cell_count}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
