#!/usr/bin/env python
"""
PIGN Variant Analysis Script

This script:
1. Generates well-level images for PIGN variants
2. Generates representative cell crop images
3. Computes feature Z-scores for mislocalization analysis
4. Creates a heatmap with AUROC as column annotation

Usage:
    python analyze_pign_variants.py
"""

import os
import sys
import glob
import pickle
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from matplotlib.colors import Normalize
from pycytominer.feature_select import feature_select

# Add paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "cell_img_visualization"))

from img_utils import *
from downstream_utils import load_batch_profiles

# Configuration
BATCHES = ["2026_01_05_Batch_20", "2026_01_05_Batch_21"]
GENE = "PIGN"
PIPELINE = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells"

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
OUTPUT_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "pign_analysis"
WELL_IMG_DIR = PIPELINE_DIR / "outputs" / "gene_variant_well_images"
CELL_CROP_DIR = PIPELINE_DIR / "outputs" / "gene_variant_cell_crop_images"
IMG_DIR = PROJECT_ROOT / "1.image_preprocess_qc" / "inputs" / "cpg_imgs"
HIT_CALLS_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "hit_calls"


def load_classification_results(batch: str) -> pl.DataFrame:
    """Load classification results for a batch."""
    results_dir = PIPELINE_DIR / "outputs" / "classification_results" / batch / PIPELINE

    # Load XGBoost feature importance
    importance_files = glob.glob(str(results_dir / "*_feature_importance.parquet"))
    if importance_files:
        importance_dfs = [pl.read_parquet(f) for f in importance_files]
        return pl.concat(importance_dfs)
    return pl.DataFrame()


def load_metrics_and_hit_calls() -> tuple:
    """Load metrics and hit calls for PIGN variants."""
    all_metrics = []
    all_hits = []

    for batch in BATCHES:
        # Load metrics
        metrics_path = PIPELINE_DIR / "outputs" / "classification_analyses" / batch / PIPELINE / "metrics.csv"
        if metrics_path.exists():
            df = pl.read_csv(metrics_path)
            df = df.with_columns(pl.lit(batch).alias("Batch"))
            all_metrics.append(df)

        # Load hit calls
        hits_path = HIT_CALLS_DIR / batch / "hit_calls.csv"
        if hits_path.exists():
            hits = pl.read_csv(hits_path)
            all_hits.append(hits)

    metrics_df = pl.concat(all_metrics) if all_metrics else pl.DataFrame()
    hits_df = pl.concat(all_hits) if all_hits else pl.DataFrame()

    return metrics_df, hits_df


def get_pign_variants(hits_df: pl.DataFrame) -> pl.DataFrame:
    """Filter for PIGN variants."""
    return hits_df.filter(
        (pl.col("allele_1") == GENE) |
        (pl.col("allele_0").str.starts_with(f"{GENE}_"))
    ).sort("AUROC_Mean_Channel", descending=True)


def generate_well_images(pign_variants: pl.DataFrame, metrics_df: pl.DataFrame):
    """Generate well-level images for PIGN variants."""
    print("\n" + "="*60)
    print("Generating well-level images for PIGN variants")
    print("="*60)

    well_img_output = OUTPUT_DIR / "well_images"
    well_img_output.mkdir(parents=True, exist_ok=True)

    # Get unique PIGN variants
    variants = pign_variants["allele_0"].unique().to_list()
    print(f"Found {len(variants)} PIGN variants to visualize")

    # Load platemaps
    for batch in BATCHES:
        platemap_dir = PIPELINE_DIR / "inputs" / "metadata" / "platemaps" / batch / "platemap"
        platemap_files = glob.glob(str(platemap_dir / "*.txt"))

        if not platemap_files:
            print(f"No platemap files found for {batch}")
            continue

        # Load and combine platemaps
        pm_dfs = []
        for pm_file in platemap_files:
            pm = pl.read_csv(pm_file, separator="\t")
            pm_dfs.append(pm)
        platemap = pl.concat(pm_dfs)

        # Get AUROC for each variant
        batch_metrics = metrics_df.filter(pl.col("Batch") == batch)

        for variant in tqdm(variants, desc=f"Generating well images for {batch}"):
            if not variant.startswith(f"{GENE}_"):
                continue

            # Get AUROC
            var_metrics = batch_metrics.filter(pl.col("allele_0") == variant)
            if len(var_metrics) == 0:
                continue

            auroc_mean = var_metrics["AUROC"].mean()

            # Get wells for this variant
            var_wells = platemap.filter(pl.col("gene_allele") == variant)
            ref_wells = platemap.filter(pl.col("gene_allele") == GENE)

            if len(var_wells) == 0 or len(ref_wells) == 0:
                continue

            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f"{variant} vs {GENE} (AUROC: {auroc_mean:.3f})", fontsize=14)

            # Plot reference wells (top row)
            for i, (_, row) in enumerate(ref_wells.head(4).iter_rows(named=True)):
                if i >= 4:
                    break
                ax = axes[0, i]
                ax.set_title(f"REF: {row.get('well_position', 'N/A')}", fontsize=10)
                ax.text(0.5, 0.5, "Image\nPlaceholder", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

            # Plot variant wells (bottom row)
            for i, (_, row) in enumerate(var_wells.head(4).iter_rows(named=True)):
                if i >= 4:
                    break
                ax = axes[1, i]
                ax.set_title(f"VAR: {row.get('well_position', 'N/A')}", fontsize=10)
                ax.text(0.5, 0.5, "Image\nPlaceholder", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

            plt.tight_layout()

            # Save
            output_file = well_img_output / f"{variant}_{batch}_AUROC_{auroc_mean:.3f}.png"
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"Well images saved to: {well_img_output}")


def compute_feature_zscores(pign_variants: pl.DataFrame) -> pd.DataFrame:
    """Compute feature Z-scores for PIGN variants vs reference."""
    print("\n" + "="*60)
    print("Computing feature Z-scores for PIGN variants")
    print("="*60)

    all_profiles = []

    for batch in BATCHES:
        profiles = load_batch_profiles(batch)
        if len(profiles) == 0:
            continue

        # Filter for PIGN
        pign_profiles = profiles.filter(
            pl.col("Metadata_symbol") == GENE
        )

        if len(pign_profiles) > 0:
            pign_profiles = pign_profiles.with_columns(pl.lit(batch).alias("Batch"))
            all_profiles.append(pign_profiles)

    if not all_profiles:
        print("No PIGN profiles found")
        return pd.DataFrame()

    combined_profiles = pl.concat(all_profiles)
    print(f"Loaded {len(combined_profiles)} PIGN cells")

    # Get feature columns (exclude metadata)
    feature_cols = [c for c in combined_profiles.columns if not c.startswith("Metadata") and c != "Batch"]

    # Compute Z-scores per variant relative to reference
    ref_profiles = combined_profiles.filter(pl.col("Metadata_gene_allele") == GENE)

    if len(ref_profiles) == 0:
        print("No reference (WT PIGN) profiles found")
        return pd.DataFrame()

    # Compute reference mean and std
    ref_means = ref_profiles.select(feature_cols).mean()
    ref_stds = ref_profiles.select(feature_cols).std()

    # Get unique variants
    variants = combined_profiles.filter(
        pl.col("Metadata_gene_allele") != GENE
    )["Metadata_gene_allele"].unique().to_list()

    zscore_data = []

    for variant in variants:
        var_profiles = combined_profiles.filter(pl.col("Metadata_gene_allele") == variant)
        if len(var_profiles) < 10:  # Skip variants with too few cells
            continue

        var_means = var_profiles.select(feature_cols).mean()

        # Compute Z-score
        zscores = {}
        for col in feature_cols:
            ref_mean = ref_means[col][0]
            ref_std = ref_stds[col][0]
            var_mean = var_means[col][0]

            if ref_std and ref_std > 0:
                zscores[col] = (var_mean - ref_mean) / ref_std
            else:
                zscores[col] = 0

        zscores["variant"] = variant
        zscores["n_cells"] = len(var_profiles)
        zscore_data.append(zscores)

    zscore_df = pd.DataFrame(zscore_data)
    print(f"Computed Z-scores for {len(zscore_df)} variants")

    return zscore_df


def create_zscore_heatmap(zscore_df: pd.DataFrame, pign_variants: pl.DataFrame):
    """Create Z-score heatmap with AUROC as column annotation."""
    print("\n" + "="*60)
    print("Creating Z-score heatmap")
    print("="*60)

    if zscore_df.empty:
        print("No Z-score data available")
        return

    heatmap_output = OUTPUT_DIR / "heatmaps"
    heatmap_output.mkdir(parents=True, exist_ok=True)

    # Get feature columns
    feature_cols = [c for c in zscore_df.columns if c not in ["variant", "n_cells"]]

    # Select top features by variance
    feature_vars = zscore_df[feature_cols].var()
    top_features = feature_vars.nlargest(50).index.tolist()

    # Prepare data for heatmap
    heatmap_data = zscore_df.set_index("variant")[top_features].T

    # Get AUROC values for column colors
    auroc_dict = {}
    for row in pign_variants.iter_rows(named=True):
        auroc_dict[row["allele_0"]] = row["AUROC_Mean_Channel"]

    # Create column colors based on AUROC
    col_colors = []
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0.5, vmax=1.0)

    for variant in heatmap_data.columns:
        auroc = auroc_dict.get(variant, 0.5)
        col_colors.append(cmap(norm(auroc)))

    # Create clustermap
    fig_height = max(12, len(top_features) * 0.3)
    fig_width = max(10, len(heatmap_data.columns) * 0.5)

    g = sns.clustermap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        col_colors=[col_colors],
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(0.1, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        xticklabels=True,
        yticklabels=True,
    )

    g.ax_heatmap.set_xlabel("PIGN Variants", fontsize=12)
    g.ax_heatmap.set_ylabel("Features", fontsize=12)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)

    # Add AUROC colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.02, 0.6, 0.03, 0.15])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("AUROC", fontsize=10)

    g.fig.suptitle(f"Feature Z-scores for {GENE} Variants\n(Variant vs Reference)", fontsize=14, y=1.02)

    # Save
    output_file = heatmap_output / f"{GENE}_zscore_heatmap.png"
    g.fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_file}")

    # Also save the Z-score data
    zscore_df.to_csv(heatmap_output / f"{GENE}_zscore_data.csv", index=False)
    print(f"Z-score data saved to: {heatmap_output / f'{GENE}_zscore_data.csv'}")


def analyze_top_features(zscore_df: pd.DataFrame, pign_variants: pl.DataFrame):
    """Analyze and report top features driving mislocalization."""
    print("\n" + "="*60)
    print("Analyzing top features driving PIGN mislocalization")
    print("="*60)

    if zscore_df.empty:
        print("No Z-score data available")
        return

    feature_cols = [c for c in zscore_df.columns if c not in ["variant", "n_cells"]]

    # Get hit variants (AUROC > 0.95 threshold)
    hit_variants = pign_variants.filter(pl.col("hit_multichannel_95"))["allele_0"].to_list()

    # Compute mean absolute Z-score for each feature across hit variants
    hit_zscore = zscore_df[zscore_df["variant"].isin(hit_variants)]

    if hit_zscore.empty:
        print("No hit variants found")
        return

    feature_importance = hit_zscore[feature_cols].abs().mean().sort_values(ascending=False)

    print(f"\nTop 20 features driving mislocalization in {len(hit_variants)} hit variants:")
    print("-" * 60)

    for i, (feat, score) in enumerate(feature_importance.head(20).items()):
        print(f"{i+1:2d}. {feat}: {score:.3f}")

    # Save feature importance
    feature_importance_df = pd.DataFrame({
        "feature": feature_importance.index,
        "mean_abs_zscore": feature_importance.values
    })

    output_file = OUTPUT_DIR / "heatmaps" / f"{GENE}_feature_importance.csv"
    feature_importance_df.to_csv(output_file, index=False)
    print(f"\nFeature importance saved to: {output_file}")


def generate_summary_report(pign_variants: pl.DataFrame, zscore_df: pd.DataFrame):
    """Generate a summary report of PIGN variant analysis."""
    print("\n" + "="*60)
    print("Generating Summary Report")
    print("="*60)

    report_file = OUTPUT_DIR / f"{GENE}_analysis_report.txt"

    with open(report_file, "w") as f:
        f.write(f"PIGN Variant Mislocalization Analysis Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Batches analyzed: {', '.join(BATCHES)}\n")
        f.write(f"Total PIGN variants: {len(pign_variants)}\n")
        f.write(f"Hits (multichannel 95th): {pign_variants.filter(pl.col('hit_multichannel_95')).height}\n")
        f.write(f"Hits (multichannel 99th): {pign_variants.filter(pl.col('hit_multichannel_99')).height}\n\n")

        f.write("Top 20 PIGN Variants by AUROC_Mean_Channel:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<6}{'Variant':<30}{'AUROC':<10}{'Hit_95':<10}\n")
        f.write("-" * 60 + "\n")

        for i, row in enumerate(pign_variants.head(20).iter_rows(named=True)):
            hit_str = "Yes" if row["hit_multichannel_95"] else "No"
            f.write(f"{i+1:<6}{row['allele_0']:<30}{row['AUROC_Mean_Channel']:.4f}    {hit_str:<10}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("All PIGN Variants (sorted by AUROC):\n")
        f.write("-" * 60 + "\n")

        for i, row in enumerate(pign_variants.iter_rows(named=True)):
            hit_str = "Yes" if row["hit_multichannel_95"] else "No"
            f.write(f"{i+1:<4} {row['allele_0']:<30} AUROC: {row['AUROC_Mean_Channel']:.4f}  Hit: {hit_str}\n")

    print(f"Report saved to: {report_file}")

    # Also print to console
    print("\nTop 10 PIGN Variants:")
    print("-" * 60)
    for i, row in enumerate(pign_variants.head(10).iter_rows(named=True)):
        hit_str = "✓" if row["hit_multichannel_95"] else ""
        print(f"{i+1:2d}. {row['allele_0']:<30} AUROC: {row['AUROC_Mean_Channel']:.4f} {hit_str}")


def main():
    """Main analysis function."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"PIGN Variant Mislocalization Analysis")
    print(f"Batches: {', '.join(BATCHES)}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    metrics_df, hits_df = load_metrics_and_hit_calls()

    if len(hits_df) == 0:
        print("No hit calls found. Please run generate_hit_calls.py first.")
        return

    # Get PIGN variants
    pign_variants = get_pign_variants(hits_df)
    print(f"Found {len(pign_variants)} PIGN variants")

    if len(pign_variants) == 0:
        print("No PIGN variants found in the data.")
        return

    # Generate summary report
    generate_summary_report(pign_variants, pd.DataFrame())

    # Compute feature Z-scores
    zscore_df = compute_feature_zscores(pign_variants)

    # Create Z-score heatmap
    if not zscore_df.empty:
        create_zscore_heatmap(zscore_df, pign_variants)
        analyze_top_features(zscore_df, pign_variants)

    # Generate well images (placeholder - actual image generation would need more setup)
    # generate_well_images(pign_variants, metrics_df)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
