#!/usr/bin/env python
"""
Generate cell crop images for top PIGN variants in Batch 20/21.

This script generates cell crop images showing representative cells for
each variant and reference allele.

Usage:
    python generate_pign_cell_crops.py --top_n 10 --cells_per_allele 10
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "cell_img_visualization"))

from img_utils import TIFF_IMGS_DIR
from cell_selector import filter_cells_by_metadata, filter_by_quality_metrics, select_cells_percentile_random
from cell_cropper import load_multichannel_cell_crops
from cell_visualizer import viz_cell_single_channel
from downstream_utils import load_batch_profiles_with_bbox as load_batch_profiles

# Batch info
BATCH_20 = "2026_01_05_Batch_20"
BATCH_21 = "2026_01_05_Batch_21"

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
OUTPUT_DIR = PROJECT_ROOT / "2.snakemake_pipeline" / "outputs" / "gene_variant_cell_crop_images"
RESULTS_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"



def generate_cell_crop_figure(variant: str, ref: str, profiles: pl.DataFrame,
                              batch: str, auroc: float, n_cells: int = 10):
    """Generate cell crop figure for a variant vs reference."""

    # Filter cells by allele
    var_profiles = filter_cells_by_metadata(profiles, allele=variant)
    ref_profiles = filter_cells_by_metadata(profiles, allele=ref)

    if len(var_profiles) == 0:
        print(f"    No cells found for {variant}")
        return None
    if len(ref_profiles) == 0:
        print(f"    No cells found for {ref}")
        return None

    # Apply quality filters
    var_profiles = filter_by_quality_metrics(var_profiles, min_edge_dist=50)
    ref_profiles = filter_by_quality_metrics(ref_profiles, min_edge_dist=50)

    if len(var_profiles) < n_cells:
        print(f"    Warning: Only {len(var_profiles)} variant cells after QC filter")
        n_cells = min(n_cells, len(var_profiles))
    if len(ref_profiles) < n_cells:
        print(f"    Warning: Only {len(ref_profiles)} reference cells after QC filter")

    if n_cells == 0:
        return None

    # Select cells from middle percentile range (45-55th) for representative samples
    try:
        var_cells = select_cells_percentile_random(
            var_profiles,
            feature="Cells_AreaShape_Area",
            percentile_bins=[(30, 70)],
            n_per_bin=n_cells * 2,  # Get more than needed to filter out bad crops
            seed=42
        )
        ref_cells = select_cells_percentile_random(
            ref_profiles,
            feature="Cells_AreaShape_Area",
            percentile_bins=[(30, 70)],
            n_per_bin=n_cells * 2,
            seed=42
        )
    except Exception as e:
        print(f"    Error selecting cells: {e}")
        return None

    if len(var_cells) == 0 or len(ref_cells) == 0:
        return None

    # Create figure: 2 rows (VAR on top, REF on bottom) x n_cells columns
    channels = ["GFP"]
    fig, axes = plt.subplots(2, n_cells, figsize=(n_cells * 3, 6))

    # Plot variant cells (top row)
    cell_idx = 0
    plotted = 0
    while plotted < n_cells and cell_idx < len(var_cells):
        try:
            # Get single cell row as dict
            cell_dict = var_cells.row(cell_idx, named=True)
            var_crops = load_multichannel_cell_crops(
                cell_dict,
                channels,
                imgs_dir=str(TIFF_IMGS_DIR),
                method='fixed',
                crop_size=128,
                site_col="Metadata_Site",
            )

            # Skip cells with low std (likely background)
            if np.std(var_crops["GFP"]) < 30:
                cell_idx += 1
                continue

            viz_cell_single_channel(
                var_crops['GFP'],
                channel='GFP',
                ax=axes[0, plotted],
                axis_off=True,
                percentile_low=1.0,
                percentile_high=99.0,
                title=f'V{plotted + 1}' if plotted == 0 else None
            )
            plotted += 1
            cell_idx += 1
        except Exception as e:
            cell_idx += 1
            continue

    # Fill remaining slots with empty if needed
    for i in range(plotted, n_cells):
        axes[0, i].axis('off')

    # Plot reference cells (bottom row)
    cell_idx = 0
    plotted = 0
    while plotted < n_cells and cell_idx < len(ref_cells):
        try:
            # Get single cell row as dict
            cell_dict = ref_cells.row(cell_idx, named=True)
            ref_crops = load_multichannel_cell_crops(
                cell_dict,
                channels,
                imgs_dir=str(TIFF_IMGS_DIR),
                method='fixed',
                crop_size=128,
                site_col="Metadata_Site",
            )

            # Skip cells with low std (likely background)
            if np.std(ref_crops["GFP"]) < 30:
                cell_idx += 1
                continue

            viz_cell_single_channel(
                ref_crops['GFP'],
                channel='GFP',
                ax=axes[1, plotted],
                axis_off=True,
                percentile_low=1.0,
                percentile_high=99.0,
                title=f'R{plotted + 1}' if plotted == 0 else None
            )
            plotted += 1
            cell_idx += 1
        except Exception as e:
            cell_idx += 1
            continue

    # Fill remaining slots with empty if needed
    for i in range(plotted, n_cells):
        axes[1, i].axis('off')

    # Add row labels
    axes[0, 0].text(-0.15, 0.5, variant.replace("PIGN_", ""),
                    transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold',
                    rotation=90, va='center', ha='right')
    axes[1, 0].text(-0.15, 0.5, 'PIGN (Ref)',
                    transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold',
                    rotation=90, va='center', ha='right')

    # Add title with AUROC
    fig.suptitle(f"{variant} vs PIGN - GFP Channel (AUROC: {auroc:.3f})", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.1, top=0.92)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_short = batch.split("_")[-1]
    filename = f"{variant}_{batch_short}_GFP_cells_{auroc:.3f}.png"
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate PIGN cell crop images")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top variants")
    parser.add_argument("--cells_per_allele", type=int, default=10, help="Number of cells per allele")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Generating Cell Crop Images for Top {args.top_n} PIGN Variants")
    print("=" * 60)

    # Load classification results
    gfp_results = pl.read_csv(RESULTS_DIR / "altered_GFP_summary_auroc.csv")
    pign_results = gfp_results.filter(pl.col("allele_0").str.starts_with("PIGN_"))
    pign_results = pign_results.sort("AUROC_Mean", descending=True, nulls_last=True)
    top_pign = pign_results.head(args.top_n)

    print(f"\nTop {args.top_n} PIGN variants by GFP AUROC:")
    print(top_pign.select(["allele_0", "AUROC_Mean"]))

    # Load profiles for both batches
    print("\nLoading cell profiles...")
    profiles_b20 = load_batch_profiles(BATCH_20)
    profiles_b21 = load_batch_profiles(BATCH_21)

    print(f"  Batch 20: {len(profiles_b20)} cells")
    print(f"  Batch 21: {len(profiles_b21)} cells")

    # Combine profiles
    all_profiles = pl.concat([profiles_b20, profiles_b21], how="diagonal_relaxed")
    print(f"  Total: {len(all_profiles)} cells")

    # Check PIGN cells
    pign_cells = all_profiles.filter(pl.col("Metadata_gene_allele").str.starts_with("PIGN"))
    print(f"  PIGN cells: {len(pign_cells)}")

    print(f"\nGenerating cell crop images...")

    for row in top_pign.iter_rows(named=True):
        variant = row["allele_0"]
        auroc = row["AUROC_Mean"]
        ref = "PIGN"

        print(f"\n  {variant} (AUROC: {auroc:.3f})")

        # Try Batch 20 first
        var_in_b20 = len(profiles_b20.filter(pl.col("Metadata_gene_allele") == variant)) > 0
        var_in_b21 = len(profiles_b21.filter(pl.col("Metadata_gene_allele") == variant)) > 0

        if var_in_b20:
            output_path = generate_cell_crop_figure(
                variant, ref, profiles_b20, BATCH_20, auroc, n_cells=args.cells_per_allele
            )
            if output_path:
                print(f"    Saved: {output_path.name}")

        if var_in_b21:
            output_path = generate_cell_crop_figure(
                variant, ref, profiles_b21, BATCH_21, auroc, n_cells=args.cells_per_allele
            )
            if output_path:
                print(f"    Saved: {output_path.name}")

    print("\n" + "=" * 60)
    print(f"Cell crop images saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
