"""
Example Usage Script for New Cell Visualization Modules

This script demonstrates how to use the refactored cell selection,
cropping, and visualization modules.

Author: VarChAMP Pipeline
Date: 2025
"""

import os
import sys
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append("../..")
from img_utils import TIFF_IMGS_DIR, BATCH_PROFILES

# Import new modules
from cell_selector import (
    filter_cells_by_metadata,
    select_cells_top_n,
    select_cells_percentile_random,
    filter_by_quality_metrics,
    compute_distance_to_edge,
    find_optimal_intensity_range,
    select_phenotype_extreme_cells
)
from cell_cropper import (
    extract_cell_crop,
    load_multichannel_cell_crops,
    batch_extract_cell_crops
)
from cell_visualizer import (
    normalize_channel,
    viz_cell_single_channel,
    viz_cell_multi_channel,
    viz_cell_grid,
    plot_cell_comparison
)


def example_1_basic_workflow():
    """
    Example 1: Basic workflow - load profiles, select cells, extract crops, visualize
    """
    print("=" * 60)
    print("Example 1: Basic Cell Visualization Workflow")
    print("=" * 60)

    # Step 1: Load batch profiles
    batch_id = "2024_01_Batch_7-8"
    profiles_path = BATCH_PROFILES.format(batch_id)

    if not os.path.exists(profiles_path):
        print(f"Profiles not found at {profiles_path}")
        return

    print(f"Loading profiles from {profiles_path}...")
    profiles = pl.read_parquet(profiles_path)

    # Add edge distance column if not present
    if "dist2edge" not in profiles.columns:
        profiles = profiles.with_columns(
            compute_distance_to_edge().alias("dist2edge")
        )

    print(f"Loaded {len(profiles)} cell profiles")

    # Step 2: Filter to specific variant
    variant = "BRCA1_V1"  # Replace with actual variant in your data
    print(f"\nFiltering to variant: {variant}")

    variant_cells = filter_cells_by_metadata(profiles, allele=variant)
    print(f"Found {len(variant_cells)} variant cells")

    # Step 3: Apply quality filters
    print("\nApplying quality filters...")
    variant_cells_filtered = filter_by_quality_metrics(
        variant_cells,
        min_edge_dist=50,
        area_range=(500, 5000)
    )
    print(f"After filtering: {len(variant_cells_filtered)} cells")

    # Step 4: Select top cells by GFP intensity
    feature = "Cells_Intensity_IntegratedIntensity_GFP"
    n_cells = 5
    print(f"\nSelecting top {n_cells} cells by {feature}")

    selected_cells = select_cells_top_n(
        variant_cells_filtered,
        feature=feature,
        n=n_cells,
        direction="high"
    )

    # Step 5: Extract cell crops
    print("\nExtracting cell crops...")
    channels = ['DAPI', 'AGP', 'Mito', 'GFP']

    # Extract first cell as example
    if len(selected_cells) > 0:
        first_cell = selected_cells[0]
        cell_crops = load_multichannel_cell_crops(
            first_cell,
            channels=channels,
            imgs_dir=TIFF_IMGS_DIR,
            method='bbox',
            target_size=128,
            recenter=True
        )

        if cell_crops:
            print(f"Successfully extracted {len(cell_crops)} channels")

            # Step 6: Visualize
            print("\nCreating visualization...")
            fig = viz_cell_grid(
                cell_crops,
                channels=channels,
                channel_mapping='morphology',
                contrast_percentiles=99.5,
                cell_info={
                    'Variant': variant,
                    'GFP Intensity': f"{first_cell[feature]:.1f}"
                },
                save_path='example_cell_grid.png'
            )
            print("Saved visualization to example_cell_grid.png")
            plt.close(fig)
        else:
            print("Failed to extract crops")
    else:
        print("No cells selected")


def example_2_contrast_comparison():
    """
    Example 2: Compare different contrast settings for same cell
    """
    print("\n" + "=" * 60)
    print("Example 2: Testing Different Contrast Thresholds")
    print("=" * 60)

    # Load profiles and select a cell (abbreviated version)
    batch_id = "2024_01_Batch_7-8"
    profiles_path = BATCH_PROFILES.format(batch_id)

    if not os.path.exists(profiles_path):
        print(f"Profiles not found")
        return

    profiles = pl.read_parquet(profiles_path)

    # Select one cell
    cell = profiles[0]

    # Extract crops
    channels = ['DAPI', 'AGP', 'Mito', 'GFP']
    cell_crops = load_multichannel_cell_crops(
        cell, channels, TIFF_IMGS_DIR,
        method='bbox', target_size=128, recenter=True
    )

    if not cell_crops:
        print("Failed to extract crops")
        return

    # Test different percentiles
    percentiles = [95.0, 97.0, 99.0, 99.5, 99.9]

    fig, axes = plt.subplots(1, len(percentiles), figsize=(20, 4))

    for i, percentile in enumerate(percentiles):
        viz_cell_multi_channel(
            cell_crops,
            channels=channels,
            channel_mapping='morphology',
            contrast_percentiles=percentile,
            ax=axes[i],
            title=f"p={percentile}"
        )

    plt.tight_layout()
    plt.savefig('example_contrast_comparison.png', dpi=150)
    print("Saved contrast comparison to example_contrast_comparison.png")
    plt.close(fig)


def example_3_variant_vs_reference():
    """
    Example 3: Compare variant vs reference cells side-by-side
    """
    print("\n" + "=" * 60)
    print("Example 3: Variant vs Reference Comparison")
    print("=" * 60)

    batch_id = "2024_01_Batch_7-8"
    profiles_path = BATCH_PROFILES.format(batch_id)

    if not os.path.exists(profiles_path):
        print(f"Profiles not found")
        return

    profiles = pl.read_parquet(profiles_path)

    # Add edge distance
    if "dist2edge" not in profiles.columns:
        profiles = profiles.with_columns(
            compute_distance_to_edge().alias("dist2edge")
        )

    # Select variant and reference
    variant = "BRCA1_V1"  # Replace with actual variant
    reference = "BRCA1_WT"  # Replace with actual reference

    variant_profiles = filter_cells_by_metadata(profiles, allele=variant)
    ref_profiles = filter_cells_by_metadata(profiles, allele=reference)

    if len(variant_profiles) == 0 or len(ref_profiles) == 0:
        print(f"Insufficient cells for comparison")
        return

    # Apply quality filters
    variant_profiles = filter_by_quality_metrics(variant_profiles, min_edge_dist=50)
    ref_profiles = filter_by_quality_metrics(ref_profiles, min_edge_dist=50)

    # Select phenotype-extreme cells
    feature = "Cells_Intensity_IntegratedIntensity_GFP"
    var_cells, ref_cells = select_phenotype_extreme_cells(
        variant_profiles,
        ref_profiles,
        feature=feature,
        n=1,  # Just get one cell each
        adaptive=True
    )

    # Extract crops
    channels = ['DAPI', 'AGP', 'Mito', 'GFP']

    var_crops = load_multichannel_cell_crops(
        var_cells[0], channels, TIFF_IMGS_DIR,
        method='bbox', target_size=128, recenter=True
    )

    ref_crops = load_multichannel_cell_crops(
        ref_cells[0], channels, TIFF_IMGS_DIR,
        method='bbox', target_size=128, recenter=True
    )

    if var_crops and ref_crops:
        # Create comparison plot
        fig = plot_cell_comparison(
            var_crops, ref_crops,
            channels=channels,
            channel_mapping='morphology',
            contrast_percentiles=99.5,
            variant_label=variant,
            ref_label=reference,
            save_path='example_comparison.png'
        )
        print("Saved comparison to example_comparison.png")
        plt.close(fig)
    else:
        print("Failed to extract crops")


def example_4_different_cropping_methods():
    """
    Example 4: Compare bbox vs fixed-size cropping
    """
    print("\n" + "=" * 60)
    print("Example 4: Comparing Cropping Methods")
    print("=" * 60)

    batch_id = "2024_01_Batch_7-8"
    profiles_path = BATCH_PROFILES.format(batch_id)

    if not os.path.exists(profiles_path):
        print(f"Profiles not found")
        return

    profiles = pl.read_parquet(profiles_path)
    cell = profiles[0]

    channels = ['DAPI', 'GFP']

    # Method 1: Bounding box with recentering
    print("\nExtracting with bbox method...")
    bbox_crops = load_multichannel_cell_crops(
        cell, channels, TIFF_IMGS_DIR,
        method='bbox', target_size=128, recenter=True
    )

    # Method 2: Fixed-size
    print("Extracting with fixed-size method...")
    fixed_crops = load_multichannel_cell_crops(
        cell, channels, TIFF_IMGS_DIR,
        method='fixed', crop_size=128
    )

    if bbox_crops and fixed_crops:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Bbox method
        viz_cell_single_channel(bbox_crops['DAPI'], 'DAPI', ax=axes[0, 0],
                               title='BBox - DAPI', contrast_percentile=99)
        viz_cell_single_channel(bbox_crops['GFP'], 'GFP', ax=axes[0, 1],
                               title='BBox - GFP', contrast_percentile=99)

        # Fixed-size method
        viz_cell_single_channel(fixed_crops['DAPI'], 'DAPI', ax=axes[1, 0],
                               title='Fixed - DAPI', contrast_percentile=99)
        viz_cell_single_channel(fixed_crops['GFP'], 'GFP', ax=axes[1, 1],
                               title='Fixed - GFP', contrast_percentile=99)

        plt.tight_layout()
        plt.savefig('example_cropping_comparison.png', dpi=150)
        print("Saved cropping comparison to example_cropping_comparison.png")
        plt.close(fig)
    else:
        print("Failed to extract crops")


def example_5_per_channel_contrast():
    """
    Example 5: Per-channel contrast optimization
    """
    print("\n" + "=" * 60)
    print("Example 5: Per-Channel Contrast Control")
    print("=" * 60)

    batch_id = "2024_01_Batch_7-8"
    profiles_path = BATCH_PROFILES.format(batch_id)

    if not os.path.exists(profiles_path):
        print(f"Profiles not found")
        return

    profiles = pl.read_parquet(profiles_path)
    cell = profiles[0]

    channels = ['DAPI', 'AGP', 'Mito', 'GFP']

    cell_crops = load_multichannel_cell_crops(
        cell, channels, TIFF_IMGS_DIR,
        method='bbox', target_size=128, recenter=True
    )

    if not cell_crops:
        print("Failed to extract crops")
        return

    # Different contrast strategies
    strategies = [
        ("Global 99.5", 99.5),
        ("Per-channel optimized", {
            'DAPI': 99.9,
            'AGP': 98.0,
            'Mito': 99.5,
            'GFP': 95.0
        })
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i, (label, percentiles) in enumerate(strategies):
        viz_cell_multi_channel(
            cell_crops,
            channels=channels,
            channel_mapping='gfp_inclusive',
            contrast_percentiles=percentiles,
            ax=axes[i],
            title=label
        )

    plt.tight_layout()
    plt.savefig('example_per_channel_contrast.png', dpi=150)
    print("Saved per-channel contrast example to example_per_channel_contrast.png")
    plt.close(fig)


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)

    print("\n" + "=" * 60)
    print("VarChAMP Cell Visualization Examples")
    print("=" * 60)
    print("\nThis script demonstrates the new modular cell visualization system.")
    print("Running examples...\n")

    try:
        # Run examples
        example_1_basic_workflow()
        example_2_contrast_comparison()
        example_3_variant_vs_reference()
        example_4_different_cropping_methods()
        example_5_per_channel_contrast()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the 'example_outputs' directory for results.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
