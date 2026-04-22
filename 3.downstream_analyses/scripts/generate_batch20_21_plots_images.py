#!/usr/bin/env python
"""
Generate classification plots and PIGN images for Batch 20 and 21.

This script generates:
1. AUROC distribution plots (B20_ctrl_var-wt_dist.png, B21_ctrl_var-wt_dist.png)
2. AUROC rank correlation plot (AUROC_rank_correlation.png)
3. Well-level images for top PIGN variants using display_img.py format
4. Cell crop images for top PIGN variants

Usage:
    python generate_batch20_21_plots_images.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "cell_img_visualization"))
from img_utils import *
from display_img import plot_allele
from downstream_utils import load_platemap, load_well_qc

# Constants
TC = ["EGFP"]
NC = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
PC = ["ALK", "ALK_Arg1275Gln", "PTK2B"]
TRN_IMBAL_THRES = 3
MIN_CLASS_NUM = 2
CHANNELS = ["GFP", "DNA", "AGP", "Morph", "GFP_ADJ"]

# Batch info
BIO_REP_NAME = "2026_01_Batch_20-21"
BATCH_20 = "2026_01_05_Batch_20"
BATCH_21 = "2026_01_05_Batch_21"

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
OUTPUT_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / BIO_REP_NAME
WELL_IMG_DIR = PROJECT_ROOT / "2.snakemake_pipeline" / "outputs" / "gene_variant_well_images"


def load_batch_metrics(batch: str, pipeline: str = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells") -> pl.DataFrame:
    """Load metrics for a batch."""
    metrics_dir = PIPELINE_DIR / "outputs" / "classification_analyses" / batch / pipeline
    metrics_path = metrics_dir / "metrics.csv"
    gfp_adj_path = metrics_dir / "metrics_gfp_adj.csv"

    if not metrics_path.exists():
        return pl.DataFrame()

    df = pl.read_csv(metrics_path)
    df = df.with_columns(pl.col("Metadata_Feature_Type").alias("Classifier_type"))

    if gfp_adj_path.exists():
        df_gfp_adj = pl.read_csv(gfp_adj_path)
        df_gfp_adj = df_gfp_adj.with_columns(pl.lit("GFP_ADJ").alias("Classifier_type"))
        df = pl.concat([df, df_gfp_adj], how="diagonal_relaxed")

    df = df.with_columns(pl.lit(batch).alias("Batch"))

    # Determine node type
    df = df.with_columns(
        pl.when(pl.col("allele_0").is_in(TC) | pl.col("allele_1").is_in(TC))
        .then(pl.lit("TC"))
        .when(pl.col("allele_0").is_in(NC) | pl.col("allele_1").is_in(NC))
        .then(pl.lit("NC"))
        .when(pl.col("allele_0").is_in(PC) | pl.col("allele_1").is_in(PC))
        .then(pl.lit("PC"))
        .otherwise(pl.lit("variant"))
        .alias("Node_Type")
    )

    return df


def compute_control_thresholds(metrics_df: pl.DataFrame, channel: str) -> dict:
    """Compute AUROC thresholds from control classifiers."""
    ctrl_df = metrics_df.filter(
        (pl.col("Metadata_Control") == True) &
        (pl.col("Classifier_type") == channel) &
        (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
    )

    if len(ctrl_df) == 0:
        return {"mean": 0.5, "95th": 0.5, "99th": 0.5}

    auroc_values = ctrl_df["AUROC"].to_numpy()
    return {
        "mean": float(np.mean(auroc_values)),
        "95th": float(np.percentile(auroc_values, 95)),
        "99th": float(np.percentile(auroc_values, 99))
    }


def generate_auroc_distribution_plots(batch20_df: pl.DataFrame, batch21_df: pl.DataFrame):
    """Generate AUROC distribution plots for each batch."""
    print("\nGenerating AUROC distribution plots...")

    for batch_num, batch_df in [("20", batch20_df), ("21", batch21_df)]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))

        node_types = ["TC", "NC", "PC", "variant"]
        channels_to_plot = ["GFP", "GFP_ADJ"]

        for change_idx, channel in enumerate(channels_to_plot):
            # Get control thresholds
            thresholds = compute_control_thresholds(batch_df, channel)

            for node_idx, node_type in enumerate(node_types):
                ax = axes[change_idx, node_idx]

                # Filter data
                plot_data = batch_df.filter(
                    (pl.col("Classifier_type") == channel) &
                    (pl.col("Node_Type") == node_type) &
                    (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
                )

                if len(plot_data) > 0:
                    sns.histplot(
                        data=plot_data.to_pandas(),
                        x="AUROC",
                        kde=False,
                        ax=ax,
                        color=sns.color_palette("Set2")[node_idx % len(sns.color_palette("Set2"))]
                    )

                    # Add threshold lines
                    ax.axvline(thresholds["95th"], color='red', linestyle='--', alpha=0.8, label=f"Ctrl_95th:{thresholds['95th']:.3f}")
                    ax.axvline(thresholds["99th"], color='purple', linestyle='--', alpha=0.8, label=f"Ctrl_99th:{thresholds['99th']:.3f}")
                    ax.legend(loc="upper left", fontsize=8)

                ax.set_title(f"{channel} - {node_type}", fontsize=11)
                ax.set_xlabel("AUROC")
                ax.grid(alpha=0.2)

        fig.suptitle(f"Batch {batch_num} - AUROC Distribution by Node Type", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.925)

        output_path = OUTPUT_DIR / f"B{batch_num}_ctrl_var-wt_dist.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")


def generate_rank_correlation_plot(batch20_df: pl.DataFrame, batch21_df: pl.DataFrame):
    """Generate AUROC rank correlation plot between bio replicates."""
    print("\nGenerating AUROC rank correlation plot...")

    plot_feats = ["GFP_ADJ", "GFP", "DNA", "AGP", "Morph"]
    fig, axes = plt.subplots(1, len(plot_feats), figsize=(20, 4))

    for classifier_type in plot_feats:
        # Get allele summaries for each batch
        set1_df = batch20_df.filter(
            (pl.col("Classifier_type") == classifier_type) &
            (pl.col("Metadata_Control") == False) &
            (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
        ).group_by("allele_0").agg(pl.col("AUROC").mean())

        set2_df = batch21_df.filter(
            (pl.col("Classifier_type") == classifier_type) &
            (pl.col("Metadata_Control") == False) &
            (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
        ).group_by("allele_0").agg(pl.col("AUROC").mean())

        if len(set1_df) == 0 or len(set2_df) == 0:
            continue

        # Calculate ranks
        set1_ranks = set1_df.with_columns(
            pl.col("AUROC").rank(method="average", descending=True).alias("AUROC_rank")
        )
        set2_ranks = set2_df.with_columns(
            pl.col("AUROC").rank(method="average", descending=True).alias("AUROC_rank")
        )

        # Join and calculate correlation
        joined_ranks = set1_ranks.join(set2_ranks, on="allele_0", how="inner", suffix="_rep2")

        if len(joined_ranks) > 2:
            spearman_corr, _ = spearmanr(joined_ranks["AUROC_rank"], joined_ranks["AUROC_rank_rep2"])

            ax_idx = plot_feats.index(classifier_type)
            sns.regplot(x="AUROC_rank", y="AUROC_rank_rep2", data=joined_ranks.to_pandas(), ax=axes[ax_idx])
            axes[ax_idx].set_title(f"Spearman's Correlation: {spearman_corr:.2f}\n{classifier_type}")
            axes[ax_idx].set_xlabel("AUROC Rank Batch 20")
            axes[ax_idx].set_ylabel("AUROC Rank Batch 21")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)

    output_path = OUTPUT_DIR / "AUROC_rank_correlation.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_pign_well_images(top_n: int = 10):
    """Generate well-level images for top PIGN variants using display_img.py."""
    print(f"\nGenerating well images for top {top_n} PIGN variants...")

    # Load classification results
    gfp_results = pl.read_csv(OUTPUT_DIR / "altered_GFP_summary_auroc.csv")
    pign_results = gfp_results.filter(pl.col("allele_0").str.starts_with("PIGN_"))
    pign_results = pign_results.sort("AUROC_Mean", descending=True, nulls_last=True)
    top_pign = pign_results.head(top_n)

    print(f"  Top {top_n} PIGN variants:")
    print(top_pign.select(["allele_0", "AUROC_Mean"]))

    # Load platemaps
    platemap_b20 = load_platemap(BATCH_20)
    platemap_b21 = load_platemap(BATCH_21)

    # Create combined platemap with imaging info
    # Add imaging plate info based on barcode_platemap
    def add_imaging_info(pm: pl.DataFrame, batch: str) -> pl.DataFrame:
        barcode_path = PIPELINE_DIR / "inputs" / "metadata" / "platemaps" / batch / "barcode_platemap.csv"
        barcode_df = pl.read_csv(barcode_path)

        # Get plate info from barcode
        plate_map_name = pm["plate_map_name"].unique()[0] if len(pm) > 0 else ""

        # Add imaging plate columns (R1 = first two timepoints, R2 = last two timepoints)
        barcodes = barcode_df["Assay_Plate_Barcode"].to_list()

        # For Batch 20/21, the layout is single plate with 4 timepoints
        if len(barcodes) >= 4:
            pm = pm.with_columns(
                pl.lit(barcodes[0]).alias("imaging_plate_R1"),  # T1
                pl.lit(barcodes[3]).alias("imaging_plate_R2"),  # T4
                pl.col("well_position").alias("imaging_well")
            )

        return pm

    platemap_b20 = add_imaging_info(platemap_b20, BATCH_20)
    platemap_b21 = add_imaging_info(platemap_b21, BATCH_21)

    # Load QC data
    well_qc_b20 = load_well_qc(BATCH_20)
    well_qc_b21 = load_well_qc(BATCH_21)

    # Create AUROC dataframe for display_img
    auroc_df = top_pign.with_columns(
        pl.col("AUROC_Mean").alias("AUROC_Mean")
    )

    # Generate images for each top PIGN variant
    for row in top_pign.iter_rows(named=True):
        variant = row["allele_0"]
        auroc = row["AUROC_Mean"]
        ref = "PIGN"

        print(f"\n  Processing {variant} (AUROC: {auroc:.3f})...")

        # Check which batch has this variant
        for batch, pm, well_qc in [(BATCH_20, platemap_b20, well_qc_b20), (BATCH_21, platemap_b21, well_qc_b21)]:
            if len(pm.filter(pl.col("gene_allele") == variant)) == 0:
                continue

            batch_num = batch.split("_")[-1]
            plate_map_name = pm.filter(pl.col("gene_allele") == variant)["plate_map_name"].unique()[0]

            try:
                plot_allele(
                    pm=pm,
                    ref=ref,
                    var=variant,
                    sel_channel="GFP",
                    plate_img_qc=well_qc,
                    auroc_df=auroc_df,
                    site="05",
                    vmin=1.,
                    vmax=99.,
                    show_plot=False,
                    imgs_dir=str(TIFF_IMGS_DIR),
                    output_dir=str(WELL_IMG_DIR)
                )
                print(f"    Saved well image for {variant} ({batch_num})")
            except Exception as e:
                print(f"    Error generating image for {variant} ({batch_num}): {e}")


def main():
    print("=" * 60)
    print("Generating Batch 20-21 Plots and PIGN Images")
    print("=" * 60)

    # Load metrics
    print("\nLoading metrics...")
    batch20_df = load_batch_metrics(BATCH_20)
    batch21_df = load_batch_metrics(BATCH_21)
    print(f"  Batch 20: {len(batch20_df)} classifiers")
    print(f"  Batch 21: {len(batch21_df)} classifiers")

    # Generate plots
    generate_auroc_distribution_plots(batch20_df, batch21_df)
    generate_rank_correlation_plot(batch20_df, batch21_df)

    # Generate PIGN well images
    generate_pign_well_images(top_n=10)

    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Well images saved to: {WELL_IMG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
