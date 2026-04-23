#!/usr/bin/env python
"""
Generate hit callings for VarChAMP classification results.

This script computes AUROC_Mean_Channel for each variant and calls hits
based on control thresholds. Also computes batch-normalized MislocScores
for cross-batch comparability.

MislocScore approaches implemented:
- ECDF (Percentile): Empirical CDF - best cross-batch consistency
- Effect CDF: Cohen's d via normal CDF - probability interpretation
- MinMax: Linear scaling between percentile anchors
- Sigmoid: Sigmoid-transformed z-score

Usage:
    python generate_hit_calls.py --batches "2026_01_05_Batch_20,2026_01_05_Batch_21"
"""

import argparse
import os
import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import norm, percentileofscore

# Constants
TC = ["EGFP"]
NC = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
PC = ["ALK", "ALK_Arg1275Gln", "PTK2B"]

TRN_IMBAL_THRES = 3
MIN_CLASS_NUM = 2
AUROC_THRESHOLDS = [0.95, 0.99]
FEAT_SETS = ["DAPI", "GFP", "AGP", "Mito"]

# Paths - use resolve() to get absolute path from __file__
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent.parent / "2.snakemake_pipeline"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "hit_calls"
QC_DIR = SCRIPT_DIR.parent.parent / "1.image_preprocess_qc" / "outputs" / "plate_bg_summary"


# =============================================================================
# MislocScore Functions (Batch-Normalized Mislocalization Scores)
# =============================================================================

def misloc_score_ecdf(auroc: float, ctrl_aurocs: np.ndarray) -> float:
    """
    Empirical CDF / Percentile rank.

    Returns the fraction of control AUROCs that this AUROC exceeds.
    Best cross-batch consistency (std of batch means ~0.0008).

    Interpretation: "This variant exceeds X% of controls"
    """
    if len(ctrl_aurocs) == 0:
        return 0.5
    return percentileofscore(ctrl_aurocs, auroc, kind='rank') / 100.0


def misloc_score_effect_cdf(auroc: float, ctrl_mean: float, ctrl_std: float) -> float:
    """
    Effect size CDF (Cohen's d via normal CDF).

    Computes Cohen's d and converts to probability using normal CDF.
    Good cross-batch consistency with probability interpretation.

    Interpretation: "P(variant exceeds random control) = score"
    """
    if ctrl_std == 0:
        return 0.5
    d = (auroc - ctrl_mean) / ctrl_std
    return norm.cdf(d)


def misloc_score_minmax(auroc: float, ctrl_aurocs: np.ndarray,
                        low_pct: float = 50, high_pct: float = 99) -> float:
    """
    Min-max scaling with batch anchors.

    Linearly scales AUROC between batch-specific percentile anchors.

    Interpretation: "0 = median control, 1 = exceeds 99% of controls"
    """
    if len(ctrl_aurocs) == 0:
        return 0.5
    low = np.percentile(ctrl_aurocs, low_pct)
    high = np.percentile(ctrl_aurocs, high_pct)
    if high == low:
        return 0.5
    return float(np.clip((auroc - low) / (high - low), 0, 1))


def misloc_score_sigmoid(auroc: float, ctrl_mean: float, ctrl_std: float,
                         k: float = 1.5, z0: float = 1.5) -> float:
    """
    Sigmoid-transformed z-score.

    Applies logistic function to z-score with tunable steepness (k) and inflection point (z0).

    Interpretation: Smoothly maps z-score to [0,1]
    """
    if ctrl_std == 0:
        return 0.5
    z = (auroc - ctrl_mean) / ctrl_std
    return 1.0 / (1.0 + np.exp(-k * (z - z0)))


def load_metrics(batch: str, pipeline: str = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells") -> pl.DataFrame:
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
        pl.lit(batch).alias("Batch"),
        pl.col("Full_Classifier_ID").str.split("_").list.last().alias("Batch_Short")
    )

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

    # Create allele set identifier
    df = df.with_columns(
        pl.concat_str([
            pl.col("allele_0").cast(pl.Utf8),
            pl.lit("_vs_"),
            pl.col("allele_1").cast(pl.Utf8)
        ]).alias("Allele_set")
    )

    return df


def load_well_qc(batch: str) -> pl.DataFrame:
    """Load well QC flags for a batch."""
    qc_path = QC_DIR / batch / "well_qc_flags.parquet"

    if not qc_path.exists():
        print(f"Warning: Well QC not found for {batch}, skipping QC filtering")
        return None

    return pl.read_parquet(qc_path)


def compute_control_thresholds(metrics_df: pl.DataFrame, feat_type: str) -> dict:
    """Compute AUROC thresholds and statistics from control classifiers."""
    # Filter for controls that pass QC
    ctrl_df = metrics_df.filter(
        (pl.col("Metadata_Control") == True) &
        (pl.col("Classifier_type") == feat_type) &
        (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
    )

    if len(ctrl_df) == 0:
        return {
            "AUROC_mean": 0.5,
            "AUROC_std": 0.1,
            "AUROC_95": 0.5,
            "AUROC_99": 0.5,
            "ctrl_aurocs": np.array([0.5])  # Fallback array for ECDF
        }

    auroc_values = ctrl_df["AUROC"].to_numpy()

    return {
        "AUROC_mean": float(np.mean(auroc_values)),
        "AUROC_std": float(np.std(auroc_values)),
        "AUROC_95": float(np.percentile(auroc_values, 95)),
        "AUROC_99": float(np.percentile(auroc_values, 99)),
        "ctrl_aurocs": auroc_values  # Store for ECDF computation
    }


def compute_hit_calls(batch: str) -> pl.DataFrame:
    """Compute hit calls for a single batch."""
    print(f"\n{'='*60}")
    print(f"Processing batch: {batch}")
    print(f"{'='*60}")

    # Load metrics
    metrics_df = load_metrics(batch)
    if len(metrics_df) == 0:
        return pl.DataFrame()

    print(f"Loaded {len(metrics_df)} classifier results")

    # Load well QC if available
    well_qc = load_well_qc(batch)

    if well_qc is not None:
        # Join QC flags for well_0
        metrics_df = metrics_df.join(
            well_qc.select(["plate", "well", "channel", "is_bg"]),
            left_on=["Plate", "well_0", "Metadata_Feature_Type"],
            right_on=["plate", "well", "channel"],
            how="left"
        ).rename({"is_bg": "well_0_is_bg"})

        # Fill missing QC flags with False
        metrics_df = metrics_df.with_columns(
            pl.col("well_0_is_bg").fill_null(False)
        )

        # Join QC flags for well_1
        metrics_df = metrics_df.join(
            well_qc.select(["plate", "well", "channel", "is_bg"]),
            left_on=["Plate", "well_1", "Metadata_Feature_Type"],
            right_on=["plate", "well", "channel"],
            how="left"
        ).rename({"is_bg": "well_1_is_bg"})

        metrics_df = metrics_df.with_columns(
            pl.col("well_1_is_bg").fill_null(False)
        )
    else:
        # No QC available, assume all wells pass
        metrics_df = metrics_df.with_columns(
            pl.lit(False).alias("well_0_is_bg"),
            pl.lit(False).alias("well_1_is_bg")
        )

    # Compute control thresholds for each feature type
    ctrl_thresholds = {}
    for feat in FEAT_SETS + ["GFP_ADJ"]:
        ctrl_thresholds[feat] = compute_control_thresholds(metrics_df, feat)
        print(f"  {feat} control AUROC: mean={ctrl_thresholds[feat]['AUROC_mean']:.3f}, "
              f"95th={ctrl_thresholds[feat]['AUROC_95']:.3f}, 99th={ctrl_thresholds[feat]['AUROC_99']:.3f}")

    # Filter for valid variant classifiers
    variant_df = metrics_df.filter(
        (pl.col("Metadata_Control") == False) &
        (~pl.col("well_0_is_bg")) &
        (~pl.col("well_1_is_bg")) &
        (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
    )

    print(f"\nValid variant classifiers: {len(variant_df)}")

    # Count effective classifiers per allele/feature
    effective_counts = (
        variant_df
        .group_by(["allele_0", "allele_1", "Allele_set", "Classifier_type"])
        .agg(pl.len().alias("Effective_Num_Classifier"))
    )

    # Join with variant data
    variant_df = variant_df.join(
        effective_counts,
        on=["allele_0", "allele_1", "Allele_set", "Classifier_type"],
        how="left"
    )

    # Filter for alleles with enough classifiers
    variant_df = variant_df.filter(
        pl.col("Effective_Num_Classifier") >= MIN_CLASS_NUM
    )

    print(f"Variants with >= {MIN_CLASS_NUM} classifiers: {len(variant_df)}")

    # Aggregate AUROC per allele per feature type
    allele_summary_list = []

    for feat in FEAT_SETS + ["GFP_ADJ"]:
        feat_df = variant_df.filter(pl.col("Classifier_type") == feat)

        if len(feat_df) == 0:
            continue

        # Aggregate per allele
        agg_df = (
            feat_df
            .group_by(["allele_0", "allele_1", "Allele_set"])
            .agg([
                pl.col("AUROC").mean().alias("AUROC_mean"),
                pl.col("AUROC").std().alias("AUROC_std"),
                pl.col("AUROC").max().alias("AUROC_max"),
                pl.col("AUROC").min().alias("AUROC_min"),
                pl.len().alias("n_classifiers"),
                pl.col("trainsize_0").mean().alias("avg_trainsize_0"),
                pl.col("trainsize_1").mean().alias("avg_trainsize_1"),
            ])
        )

        # Add feature type and control thresholds
        thres = ctrl_thresholds[feat]
        agg_df = agg_df.with_columns(
            pl.lit(feat).alias("Classifier_type"),
            pl.lit(thres["AUROC_95"]).alias("ctrl_AUROC_95"),
            pl.lit(thres["AUROC_99"]).alias("ctrl_AUROC_99"),
            pl.lit(thres["AUROC_mean"]).alias("ctrl_AUROC_mean"),
            pl.lit(thres["AUROC_std"]).alias("ctrl_AUROC_std"),
        )

        # Calculate z-score
        agg_df = agg_df.with_columns(
            ((pl.col("AUROC_mean") - pl.col("ctrl_AUROC_mean")) / pl.col("ctrl_AUROC_std")).alias("AUROC_zscore")
        )

        # Compute MislocScores for each allele
        auroc_means = agg_df["AUROC_mean"].to_numpy()
        ctrl_aurocs = thres["ctrl_aurocs"]
        ctrl_mean = thres["AUROC_mean"]
        ctrl_std = thres["AUROC_std"]

        # Compute all 4 MislocScores
        misloc_ecdf = np.array([misloc_score_ecdf(a, ctrl_aurocs) for a in auroc_means])
        misloc_cdf = np.array([misloc_score_effect_cdf(a, ctrl_mean, ctrl_std) for a in auroc_means])
        misloc_minmax = np.array([misloc_score_minmax(a, ctrl_aurocs) for a in auroc_means])
        misloc_sigmoid = np.array([misloc_score_sigmoid(a, ctrl_mean, ctrl_std) for a in auroc_means])

        agg_df = agg_df.with_columns([
            pl.Series("MislocScore_ecdf", misloc_ecdf),
            pl.Series("MislocScore_cdf", misloc_cdf),
            pl.Series("MislocScore_minmax", misloc_minmax),
            pl.Series("MislocScore_sigmoid", misloc_sigmoid),
        ])

        # Call hits based on raw AUROC thresholds (original method)
        agg_df = agg_df.with_columns(
            (pl.col("AUROC_mean") > pl.col("ctrl_AUROC_95")).alias("hit_95"),
            (pl.col("AUROC_mean") > pl.col("ctrl_AUROC_99")).alias("hit_99"),
        )

        # Call hits based on universal MislocScore thresholds (new method)
        # Using ECDF as primary (best cross-batch consistency)
        agg_df = agg_df.with_columns(
            (pl.col("MislocScore_ecdf") > 0.95).alias("hit_misloc_95"),
            (pl.col("MislocScore_ecdf") > 0.99).alias("hit_misloc_99"),
        )

        allele_summary_list.append(agg_df)

    if not allele_summary_list:
        return pl.DataFrame()

    # Combine all feature types
    allele_summary = pl.concat(allele_summary_list)

    # Compute mean AUROC and MislocScores across channels (excluding GFP_ADJ)
    channel_auroc = (
        allele_summary
        .filter(pl.col("Classifier_type").is_in(FEAT_SETS))
        .group_by(["allele_0", "allele_1", "Allele_set"])
        .agg([
            pl.col("AUROC_mean").mean().alias("AUROC_Mean_Channel"),
            pl.col("hit_95").sum().alias("n_channels_hit_95"),
            pl.col("hit_99").sum().alias("n_channels_hit_99"),
            pl.col("n_classifiers").sum().alias("total_classifiers"),
            # Mean MislocScores across channels
            pl.col("MislocScore_ecdf").mean().alias("MislocScore_ecdf_mean"),
            pl.col("MislocScore_cdf").mean().alias("MislocScore_cdf_mean"),
            pl.col("MislocScore_minmax").mean().alias("MislocScore_minmax_mean"),
            pl.col("MislocScore_sigmoid").mean().alias("MislocScore_sigmoid_mean"),
            # Count channels passing MislocScore thresholds
            pl.col("hit_misloc_95").sum().alias("n_channels_misloc_95"),
            pl.col("hit_misloc_99").sum().alias("n_channels_misloc_99"),
        ])
    )

    # Add batch info
    channel_auroc = channel_auroc.with_columns(
        pl.lit(batch).alias("Batch")
    )

    # Call overall hits (hit in at least 2 channels) - original method
    channel_auroc = channel_auroc.with_columns(
        (pl.col("n_channels_hit_95") >= 2).alias("hit_multichannel_95"),
        (pl.col("n_channels_hit_99") >= 2).alias("hit_multichannel_99"),
    )

    # Call overall hits using MislocScore (universal thresholds)
    # Method 1: Mean MislocScore across channels > threshold
    channel_auroc = channel_auroc.with_columns(
        (pl.col("MislocScore_ecdf_mean") > 0.95).alias("hit_misloc_mean_95"),
        (pl.col("MislocScore_ecdf_mean") > 0.99).alias("hit_misloc_mean_99"),
    )
    # Method 2: Hit in at least 2 channels using MislocScore
    channel_auroc = channel_auroc.with_columns(
        (pl.col("n_channels_misloc_95") >= 2).alias("hit_misloc_multichannel_95"),
        (pl.col("n_channels_misloc_99") >= 2).alias("hit_misloc_multichannel_99"),
    )

    # Sort by MislocScore_ecdf_mean descending (primary metric for ranking)
    channel_auroc = channel_auroc.sort("MislocScore_ecdf_mean", descending=True)

    print(f"\nUnique variants analyzed: {len(channel_auroc)}")
    print(f"Hits (AUROC multichannel 95th): {channel_auroc.filter(pl.col('hit_multichannel_95')).height}")
    print(f"Hits (AUROC multichannel 99th): {channel_auroc.filter(pl.col('hit_multichannel_99')).height}")
    print(f"Hits (MislocScore mean 95th): {channel_auroc.filter(pl.col('hit_misloc_mean_95')).height}")
    print(f"Hits (MislocScore mean 99th): {channel_auroc.filter(pl.col('hit_misloc_mean_99')).height}")

    return channel_auroc, allele_summary


def main():
    parser = argparse.ArgumentParser(description="Generate hit callings for VarChAMP batches")
    parser.add_argument("--batches", required=True, help="Comma-separated batch IDs")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Output directory")

    args = parser.parse_args()

    batches = [b.strip() for b in args.batches.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_hits = []
    all_summaries = []

    for batch in batches:
        result = compute_hit_calls(batch)
        if isinstance(result, tuple) and len(result) == 2:
            hits_df, summary_df = result
            if len(hits_df) > 0:
                all_hits.append(hits_df)
                all_summaries.append(summary_df.with_columns(pl.lit(batch).alias("Batch")))

                # Save per-batch results
                batch_dir = output_dir / batch
                batch_dir.mkdir(parents=True, exist_ok=True)

                hits_df.write_csv(batch_dir / "hit_calls.csv")
                summary_df.write_csv(batch_dir / "allele_summary_by_channel.csv")
                print(f"\nSaved results to {batch_dir}")

    # Combine all batches
    if all_hits:
        combined_hits = pl.concat(all_hits)
        combined_summary = pl.concat(all_summaries)

        combined_hits.write_csv(output_dir / "combined_hit_calls.csv")
        combined_summary.write_csv(output_dir / "combined_allele_summary.csv")

        print(f"\n{'='*60}")
        print("Combined Results")
        print(f"{'='*60}")
        print(f"Total variants: {len(combined_hits)}")
        print(f"\nOriginal AUROC-based hits:")
        print(f"  Multichannel 95th: {combined_hits.filter(pl.col('hit_multichannel_95')).height}")
        print(f"  Multichannel 99th: {combined_hits.filter(pl.col('hit_multichannel_99')).height}")
        print(f"\nMislocScore-based hits (universal thresholds):")
        print(f"  Mean score > 0.95: {combined_hits.filter(pl.col('hit_misloc_mean_95')).height}")
        print(f"  Mean score > 0.99: {combined_hits.filter(pl.col('hit_misloc_mean_99')).height}")
        print(f"  Multichannel 95th: {combined_hits.filter(pl.col('hit_misloc_multichannel_95')).height}")
        print(f"  Multichannel 99th: {combined_hits.filter(pl.col('hit_misloc_multichannel_99')).height}")

        # Print top hits by MislocScore
        print(f"\nTop 20 variants by MislocScore_ecdf_mean:")
        print(combined_hits.select([
            "Batch", "allele_0", "allele_1", "MislocScore_ecdf_mean",
            "AUROC_Mean_Channel", "hit_misloc_mean_95"
        ]).head(20))

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
