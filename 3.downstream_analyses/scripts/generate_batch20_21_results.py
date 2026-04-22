#!/usr/bin/env python
"""
Generate classification results for Batch 20 and 21.

This script generates:
1. Per-channel AUROC summary files matching existing format (altered_{channel}_summary_auroc.csv)
2. Combined classification summary

Output format matches existing batches in 3.downstream_analyses/outputs/2.classification_results/

Usage:
    python generate_batch20_21_results.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from img_utils import *

# Constants
TC = ["EGFP"]
NC = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
PC = ["ALK", "ALK_Arg1275Gln", "PTK2B"]

TRN_IMBAL_THRES = 3
MIN_CLASS_NUM = 2

# Batch 20 and 21 are biological replicates
BIO_REP_NAME = "2026_01_Batch_20-21"
BATCH_20 = "2026_01_05_Batch_20"
BATCH_21 = "2026_01_05_Batch_21"

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
OUTPUT_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / BIO_REP_NAME

# Channels to process - focusing on protein (GFP) first, then DNA, AGP, Morph
CHANNELS = ["GFP", "DNA", "AGP", "Morph"]


def load_batch_metrics(batch: str, pipeline: str = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells") -> pl.DataFrame:
    """Load metrics.csv and metrics_gfp_adj.csv for a batch."""
    metrics_dir = PIPELINE_DIR / "outputs" / "classification_analyses" / batch / pipeline

    metrics_path = metrics_dir / "metrics.csv"
    gfp_adj_path = metrics_dir / "metrics_gfp_adj.csv"

    if not metrics_path.exists():
        print(f"Warning: metrics.csv not found for {batch}")
        return pl.DataFrame()

    # Load main metrics
    df = pl.read_csv(metrics_path)

    # Add classifier type based on Metadata_Feature_Type
    df = df.with_columns(
        pl.col("Metadata_Feature_Type").alias("Classifier_type")
    )

    # Load GFP-adjusted metrics if available
    if gfp_adj_path.exists():
        df_gfp_adj = pl.read_csv(gfp_adj_path)
        df_gfp_adj = df_gfp_adj.with_columns(
            pl.lit("GFP_ADJ").alias("Classifier_type")
        )
        df = pl.concat([df, df_gfp_adj], how="diagonal_relaxed")

    # Add batch info
    df = df.with_columns(
        pl.lit(batch).alias("Batch")
    )

    # Determine node type (control vs variant)
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
    """Compute AUROC thresholds from control classifiers for a specific channel."""
    # Filter for controls that pass QC
    ctrl_df = metrics_df.filter(
        (pl.col("Metadata_Control") == True) &
        (pl.col("Classifier_type") == channel) &
        (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
    )

    if len(ctrl_df) == 0:
        return {"AUROC_mean": 0.5, "AUROC_std": 0.1, "AUROC_95": 0.5, "AUROC_99": 0.5}

    auroc_values = ctrl_df["AUROC"].to_numpy()

    return {
        "AUROC_mean": float(np.mean(auroc_values)),
        "AUROC_std": float(np.std(auroc_values)),
        "AUROC_95": float(np.percentile(auroc_values, 95)),
        "AUROC_99": float(np.percentile(auroc_values, 99))
    }


def generate_channel_summary(batch20_df: pl.DataFrame, batch21_df: pl.DataFrame, channel: str, ctrl_thresholds_b20: dict, ctrl_thresholds_b21: dict) -> pl.DataFrame:
    """
    Generate per-channel summary with BioRep columns matching existing format.

    Output columns:
    - allele_0
    - Altered_{channel}_95_BioRep1
    - Altered_{channel}_95_BioRep2
    - Altered_{channel}_95_both_batches
    - Altered_{channel}_99_BioRep1
    - Altered_{channel}_99_BioRep2
    - Altered_{channel}_99_both_batches
    - AUROC_BioRep1
    - AUROC_BioRep2
    - AUROC_Mean
    """
    results = []

    # Get all unique variants (allele_0) from both batches
    all_alleles_0 = set()

    for df in [batch20_df, batch21_df]:
        if len(df) > 0:
            channel_df = df.filter(
                (pl.col("Classifier_type") == channel) &
                (pl.col("Metadata_Control") == False) &
                (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
            )
            if len(channel_df) > 0:
                all_alleles_0.update(channel_df["allele_0"].unique().to_list())

    for allele in sorted(all_alleles_0):
        row = {"allele_0": allele}

        # Process BioRep1 (Batch 20)
        b20_channel = batch20_df.filter(
            (pl.col("Classifier_type") == channel) &
            (pl.col("allele_0") == allele) &
            (pl.col("Metadata_Control") == False) &
            (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
        )

        if len(b20_channel) >= MIN_CLASS_NUM:
            auroc_b20 = b20_channel["AUROC"].mean()
            hit_95_b20 = 1 if auroc_b20 > ctrl_thresholds_b20["AUROC_95"] else 0
            hit_99_b20 = 1 if auroc_b20 > ctrl_thresholds_b20["AUROC_99"] else 0
        else:
            auroc_b20 = None
            hit_95_b20 = None
            hit_99_b20 = None

        # Process BioRep2 (Batch 21)
        b21_channel = batch21_df.filter(
            (pl.col("Classifier_type") == channel) &
            (pl.col("allele_0") == allele) &
            (pl.col("Metadata_Control") == False) &
            (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
        )

        if len(b21_channel) >= MIN_CLASS_NUM:
            auroc_b21 = b21_channel["AUROC"].mean()
            hit_95_b21 = 1 if auroc_b21 > ctrl_thresholds_b21["AUROC_95"] else 0
            hit_99_b21 = 1 if auroc_b21 > ctrl_thresholds_b21["AUROC_99"] else 0
        else:
            auroc_b21 = None
            hit_95_b21 = None
            hit_99_b21 = None

        # Compute mean AUROC
        if auroc_b20 is not None and auroc_b21 is not None:
            auroc_mean = (auroc_b20 + auroc_b21) / 2
        elif auroc_b20 is not None:
            auroc_mean = auroc_b20
        elif auroc_b21 is not None:
            auroc_mean = auroc_b21
        else:
            auroc_mean = None

        # Determine if hit in both batches
        if hit_95_b20 is not None and hit_95_b21 is not None:
            hit_95_both = "true" if (hit_95_b20 == 1 and hit_95_b21 == 1) else "false"
        else:
            hit_95_both = "true" if (hit_95_b20 == 1 or hit_95_b21 == 1) else "false"

        if hit_99_b20 is not None and hit_99_b21 is not None:
            hit_99_both = "true" if (hit_99_b20 == 1 and hit_99_b21 == 1) else "false"
        else:
            hit_99_both = "true" if (hit_99_b20 == 1 or hit_99_b21 == 1) else "false"

        row[f"Altered_{channel}_95_BioRep1"] = hit_95_b20
        row[f"Altered_{channel}_95_BioRep2"] = hit_95_b21
        row[f"Altered_{channel}_95_both_batches"] = hit_95_both
        row[f"Altered_{channel}_99_BioRep1"] = hit_99_b20
        row[f"Altered_{channel}_99_BioRep2"] = hit_99_b21
        row[f"Altered_{channel}_99_both_batches"] = hit_99_both
        row["AUROC_BioRep1"] = auroc_b20
        row["AUROC_BioRep2"] = auroc_b21
        row["AUROC_Mean"] = auroc_mean

        results.append(row)

    return pl.DataFrame(results)


def main():
    print("=" * 60)
    print("Generating Classification Results for Batch 20-21")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load metrics from both batches
    print(f"\nLoading Batch 20 metrics...")
    batch20_df = load_batch_metrics(BATCH_20)
    print(f"  Loaded {len(batch20_df)} classifier results")

    print(f"\nLoading Batch 21 metrics...")
    batch21_df = load_batch_metrics(BATCH_21)
    print(f"  Loaded {len(batch21_df)} classifier results")

    # Process each channel
    all_channel_summaries = {}

    for channel in CHANNELS + ["GFP_ADJ"]:
        print(f"\n{'='*40}")
        print(f"Processing channel: {channel}")
        print(f"{'='*40}")

        # Compute control thresholds per batch per channel
        ctrl_thresholds_b20 = compute_control_thresholds(batch20_df, channel)
        ctrl_thresholds_b21 = compute_control_thresholds(batch21_df, channel)

        print(f"  Batch 20 control thresholds: 95th={ctrl_thresholds_b20['AUROC_95']:.3f}, 99th={ctrl_thresholds_b20['AUROC_99']:.3f}")
        print(f"  Batch 21 control thresholds: 95th={ctrl_thresholds_b21['AUROC_95']:.3f}, 99th={ctrl_thresholds_b21['AUROC_99']:.3f}")

        # Generate channel summary
        channel_summary = generate_channel_summary(batch20_df, batch21_df, channel, ctrl_thresholds_b20, ctrl_thresholds_b21)

        if len(channel_summary) > 0:
            # Save to CSV
            output_path = OUTPUT_DIR / f"altered_{channel}_summary_auroc.csv"
            channel_summary.write_csv(output_path)
            print(f"  Saved {len(channel_summary)} variants to {output_path}")

            # Count hits
            n_hits_95 = channel_summary.filter(pl.col(f"Altered_{channel}_95_both_batches") == "true").height
            n_hits_99 = channel_summary.filter(pl.col(f"Altered_{channel}_99_both_batches") == "true").height
            print(f"  Hits (95th both batches): {n_hits_95}")
            print(f"  Hits (99th both batches): {n_hits_99}")

            all_channel_summaries[channel] = channel_summary

    # Generate combined summary (focusing on GFP/Protein channel)
    print("\n" + "=" * 60)
    print("Generating Combined Summary")
    print("=" * 60)

    # Combine all variant data
    combined_variants = []

    if "GFP" in all_channel_summaries:
        gfp_summary = all_channel_summaries["GFP"]

        for row in gfp_summary.iter_rows(named=True):
            variant_data = {
                "allele_0": row["allele_0"],
                "AUROC_GFP_Mean": row["AUROC_Mean"],
                "Altered_95th_perc_both_batches_GFP": row["Altered_GFP_95_both_batches"] == "true",
                "Altered_99th_perc_both_batches_GFP": row["Altered_GFP_99_both_batches"] == "true",
            }

            # Add other channel data if available
            for channel in ["DNA", "AGP", "Morph", "GFP_ADJ"]:
                if channel in all_channel_summaries:
                    ch_df = all_channel_summaries[channel]
                    ch_row = ch_df.filter(pl.col("allele_0") == row["allele_0"])
                    if len(ch_row) > 0:
                        variant_data[f"AUROC_{channel}_Mean"] = ch_row["AUROC_Mean"][0]
                        variant_data[f"Altered_95th_perc_both_batches_{channel}"] = ch_row[f"Altered_{channel}_95_both_batches"][0] == "true"

            combined_variants.append(variant_data)

    if combined_variants:
        combined_df = pl.DataFrame(combined_variants)
        combined_df = combined_df.sort("AUROC_GFP_Mean", descending=True, nulls_last=True)

        combined_path = OUTPUT_DIR / "imaging_analyses_classification_summary.csv"
        combined_df.write_csv(combined_path)
        print(f"Saved combined summary to {combined_path}")

        # Print top variants
        print("\nTop 20 variants by GFP AUROC:")
        print(combined_df.select(["allele_0", "AUROC_GFP_Mean", "Altered_95th_perc_both_batches_GFP"]).head(20))

        # Print PIGN-specific results
        pign_variants = combined_df.filter(pl.col("allele_0").str.starts_with("PIGN_"))
        if len(pign_variants) > 0:
            print(f"\n{'='*60}")
            print(f"PIGN Variants Summary ({len(pign_variants)} total)")
            print(f"{'='*60}")
            pign_hits = pign_variants.filter(pl.col("Altered_95th_perc_both_batches_GFP") == True)
            print(f"PIGN hits at 95th percentile: {len(pign_hits)}")
            print("\nTop 20 PIGN variants by GFP AUROC:")
            print(pign_variants.select(["allele_0", "AUROC_GFP_Mean", "Altered_95th_perc_both_batches_GFP"]).head(20))

    print("\n" + "=" * 60)
    print("Classification Results Generation Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
