#!/usr/bin/env python
"""
Test and compare different mislocalization score normalization approaches.

This script implements and compares four approaches for converting AUROC values
to batch-normalized mislocalization scores in [0, 1]:

1. ECDF (Empirical CDF / Percentile Rank)
2. Sigmoid-transformed Z-score
3. Min-Max scaling with batch anchors
4. Effect Size CDF (Cohen's d via normal CDF)

The goal is to find a normalization that:
- Produces consistent control distributions across batches
- Handles the narrower null distribution in Batch 11-12 vs other batches
- Gives interpretable scores where 0 = normal, 1 = mislocalized

Usage:
    python test_misloc_scores.py

Output:
    Comparison plots in 3.downstream_analyses/outputs/misloc_score_comparison/
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm, percentileofscore
from typing import Tuple, Dict, List

# Constants
TC = ["EGFP"]
NC = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
PC = ["ALK", "ALK_Arg1275Gln", "PTK2B"]
TRN_IMBAL_THRES = 3
FEAT_SETS = ["DAPI", "GFP", "AGP", "Mito"]

# Batches to analyze (excluding newer batches for initial testing)
TEST_BATCHES = [
    "2024_01_23_Batch_7",
    "2024_02_06_Batch_8",
    "2024_12_09_Batch_11",
    "2024_12_09_Batch_12",
    "2025_01_27_Batch_13",
    "2025_01_28_Batch_14",
    "2025_03_17_Batch_15",
    "2025_03_17_Batch_16",
]

# Paths - use resolve() to get absolute path from __file__
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent.parent / "2.snakemake_pipeline"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "misloc_score_comparison"


# =============================================================================
# Scoring Functions
# =============================================================================

def misloc_score_ecdf(auroc: float, ctrl_aurocs: np.ndarray) -> float:
    """
    Approach 1: Empirical CDF / Percentile rank.

    Returns the fraction of control AUROCs that this AUROC exceeds.

    Pros: Non-parametric, naturally bounded [0,1], intuitive percentile interpretation
    Cons: Discrete jumps with small samples, requires storing control values

    Interpretation: "This variant exceeds X% of controls"
    """
    if len(ctrl_aurocs) == 0:
        return 0.5
    return percentileofscore(ctrl_aurocs, auroc, kind='rank') / 100.0


def misloc_score_sigmoid(auroc: float, ctrl_mean: float, ctrl_std: float,
                         k: float = 1.5, z0: float = 1.5) -> float:
    """
    Approach 2: Sigmoid-transformed z-score.

    Applies logistic function to z-score with tunable steepness (k) and inflection point (z0).

    Pros: Smooth, tunable steepness, can extrapolate
    Cons: Requires parameter tuning, assumes ~normal distribution

    Interpretation: Smoothly maps z-score to [0,1]
    """
    if ctrl_std == 0:
        return 0.5
    z = (auroc - ctrl_mean) / ctrl_std
    return 1.0 / (1.0 + np.exp(-k * (z - z0)))


def misloc_score_minmax(auroc: float, ctrl_aurocs: np.ndarray,
                        low_pct: float = 50, high_pct: float = 99) -> float:
    """
    Approach 3: Min-max scaling with batch anchors.

    Linearly scales AUROC between batch-specific percentile anchors.

    Pros: Simple, easy to explain
    Cons: Linear scaling, extreme values clip to 1.0

    Interpretation: "0 = median control, 1 = exceeds 99% of controls"
    """
    if len(ctrl_aurocs) == 0:
        return 0.5
    low = np.percentile(ctrl_aurocs, low_pct)
    high = np.percentile(ctrl_aurocs, high_pct)
    if high == low:
        return 0.5
    return np.clip((auroc - low) / (high - low), 0, 1)


def misloc_score_effect_cdf(auroc: float, ctrl_mean: float, ctrl_std: float) -> float:
    """
    Approach 4: Effect size CDF (Cohen's d via normal CDF).

    Computes Cohen's d and converts to probability using normal CDF.

    Pros: Statistically principled, probability interpretation, smooth, batch-normalized
    Cons: Assumes normality

    Interpretation: "P(variant exceeds random control) = score"
    """
    if ctrl_std == 0:
        return 0.5
    d = (auroc - ctrl_mean) / ctrl_std
    return norm.cdf(d)


# Vectorized versions for efficiency
def misloc_scores_ecdf_vec(aurocs: np.ndarray, ctrl_aurocs: np.ndarray) -> np.ndarray:
    """Vectorized ECDF scoring."""
    return np.array([misloc_score_ecdf(a, ctrl_aurocs) for a in aurocs])


def misloc_scores_sigmoid_vec(aurocs: np.ndarray, ctrl_mean: float, ctrl_std: float,
                               k: float = 1.5, z0: float = 1.5) -> np.ndarray:
    """Vectorized sigmoid scoring."""
    if ctrl_std == 0:
        return np.full_like(aurocs, 0.5)
    z = (aurocs - ctrl_mean) / ctrl_std
    return 1.0 / (1.0 + np.exp(-k * (z - z0)))


def misloc_scores_minmax_vec(aurocs: np.ndarray, ctrl_aurocs: np.ndarray,
                              low_pct: float = 50, high_pct: float = 99) -> np.ndarray:
    """Vectorized min-max scoring."""
    if len(ctrl_aurocs) == 0:
        return np.full_like(aurocs, 0.5)
    low = np.percentile(ctrl_aurocs, low_pct)
    high = np.percentile(ctrl_aurocs, high_pct)
    if high == low:
        return np.full_like(aurocs, 0.5)
    return np.clip((aurocs - low) / (high - low), 0, 1)


def misloc_scores_effect_cdf_vec(aurocs: np.ndarray, ctrl_mean: float, ctrl_std: float) -> np.ndarray:
    """Vectorized effect CDF scoring."""
    if ctrl_std == 0:
        return np.full_like(aurocs, 0.5)
    d = (aurocs - ctrl_mean) / ctrl_std
    return norm.cdf(d)


# =============================================================================
# Data Loading
# =============================================================================

def load_metrics(batch: str, pipeline: str = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells") -> pl.DataFrame:
    """Load metrics.csv for a batch."""
    metrics_dir = PIPELINE_DIR / "outputs" / "classification_analyses" / batch / pipeline
    metrics_path = metrics_dir / "metrics.csv"

    if not metrics_path.exists():
        print(f"Warning: metrics.csv not found for {batch}")
        return pl.DataFrame()

    df = pl.read_csv(metrics_path)

    # Add classifier type based on Metadata_Feature_Type
    df = df.with_columns(
        pl.col("Metadata_Feature_Type").alias("Classifier_type")
    )

    # Add batch info
    df = df.with_columns(
        pl.lit(batch).alias("Batch"),
        pl.lit(batch.split("_")[-1]).alias("Batch_Short")
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

    return df


def load_all_batches() -> pl.DataFrame:
    """Load metrics from all test batches."""
    dfs = []
    for batch in TEST_BATCHES:
        df = load_metrics(batch)
        if len(df) > 0:
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {batch}")

    if not dfs:
        raise ValueError("No data loaded from any batch")

    return pl.concat(dfs, how="diagonal_relaxed")


def get_control_stats(df: pl.DataFrame, batch: str, feat_type: str) -> Tuple[np.ndarray, float, float]:
    """Get control AUROC statistics for a batch and feature type."""
    ctrl_df = df.filter(
        (pl.col("Batch") == batch) &
        (pl.col("Metadata_Control") == True) &
        (pl.col("Classifier_type") == feat_type) &
        (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
    )

    if len(ctrl_df) == 0:
        return np.array([0.5]), 0.5, 0.1

    aurocs = ctrl_df["AUROC"].to_numpy()
    return aurocs, float(np.mean(aurocs)), float(np.std(aurocs))


# =============================================================================
# Score Computation
# =============================================================================

def compute_all_misloc_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all four mislocalization scores for each classifier."""
    results = []

    for batch in df["Batch"].unique().to_list():
        for feat_type in FEAT_SETS:
            # Get control statistics for this batch/feature
            ctrl_aurocs, ctrl_mean, ctrl_std = get_control_stats(df, batch, feat_type)

            # Get all classifiers for this batch/feature
            batch_feat_df = df.filter(
                (pl.col("Batch") == batch) &
                (pl.col("Classifier_type") == feat_type) &
                (pl.col("Training_imbalance") < TRN_IMBAL_THRES)
            )

            if len(batch_feat_df) == 0:
                continue

            aurocs = batch_feat_df["AUROC"].to_numpy()

            # Compute all scores
            scores_ecdf = misloc_scores_ecdf_vec(aurocs, ctrl_aurocs)
            scores_sigmoid = misloc_scores_sigmoid_vec(aurocs, ctrl_mean, ctrl_std)
            scores_minmax = misloc_scores_minmax_vec(aurocs, ctrl_aurocs)
            scores_cdf = misloc_scores_effect_cdf_vec(aurocs, ctrl_mean, ctrl_std)

            # Add scores to dataframe
            result_df = batch_feat_df.with_columns([
                pl.Series("MislocScore_ecdf", scores_ecdf),
                pl.Series("MislocScore_sigmoid", scores_sigmoid),
                pl.Series("MislocScore_minmax", scores_minmax),
                pl.Series("MislocScore_cdf", scores_cdf),
                pl.lit(ctrl_mean).alias("ctrl_mean"),
                pl.lit(ctrl_std).alias("ctrl_std"),
                pl.lit(len(ctrl_aurocs)).alias("n_controls"),
            ])

            results.append(result_df)

    return pl.concat(results, how="diagonal_relaxed")


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_control_distributions(df: pl.DataFrame, output_dir: Path):
    """Plot control score distributions across batches for all approaches."""
    ctrl_df = df.filter(pl.col("Metadata_Control") == True)

    score_cols = ["AUROC", "MislocScore_ecdf", "MislocScore_sigmoid",
                  "MislocScore_minmax", "MislocScore_cdf"]
    score_labels = ["Raw AUROC", "ECDF (Percentile)", "Sigmoid Z-score",
                    "Min-Max Scaling", "Effect Size CDF"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Group batches: 11-12 vs others
    ctrl_df = ctrl_df.with_columns(
        pl.when(pl.col("Batch_Short").is_in(["11", "12"]))
        .then(pl.lit("Batch 11-12"))
        .otherwise(pl.lit("Other Batches"))
        .alias("Batch_Group")
    )

    for idx, (col, label) in enumerate(zip(score_cols, score_labels)):
        ax = axes[idx]

        for batch_group in ["Batch 11-12", "Other Batches"]:
            group_data = ctrl_df.filter(pl.col("Batch_Group") == batch_group)[col].to_numpy()
            if len(group_data) > 0:
                ax.hist(group_data, bins=30, alpha=0.5, label=batch_group, density=True)

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"Control Distribution: {label}")
        ax.legend()

    # Remove empty subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "control_distributions_batch_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: control_distributions_batch_comparison.png")


def plot_control_distributions_by_batch(df: pl.DataFrame, output_dir: Path):
    """Plot control score distributions for each batch separately."""
    ctrl_df = df.filter(pl.col("Metadata_Control") == True)

    score_cols = ["AUROC", "MislocScore_ecdf", "MislocScore_sigmoid",
                  "MislocScore_minmax", "MislocScore_cdf"]
    score_labels = ["Raw AUROC", "ECDF", "Sigmoid", "MinMax", "Effect CDF"]

    batches = sorted(ctrl_df["Batch_Short"].unique().to_list())
    n_batches = len(batches)
    n_scores = len(score_cols)

    fig, axes = plt.subplots(n_scores, n_batches, figsize=(3*n_batches, 3*n_scores))

    for i, (col, label) in enumerate(zip(score_cols, score_labels)):
        for j, batch in enumerate(batches):
            ax = axes[i, j] if n_scores > 1 else axes[j]
            batch_data = ctrl_df.filter(pl.col("Batch_Short") == batch)[col].to_numpy()

            if len(batch_data) > 0:
                ax.hist(batch_data, bins=20, alpha=0.7, color='steelblue', density=True)
                ax.axvline(np.mean(batch_data), color='red', linestyle='--',
                          label=f'mean={np.mean(batch_data):.3f}')
                ax.axvline(np.median(batch_data), color='green', linestyle=':',
                          label=f'med={np.median(batch_data):.3f}')

            if i == 0:
                ax.set_title(f"Batch {batch}")
            if j == 0:
                ax.set_ylabel(label)
            if i == n_scores - 1:
                ax.set_xlabel("Score")

    plt.tight_layout()
    plt.savefig(output_dir / "control_distributions_by_batch.png", dpi=150)
    plt.close()
    print(f"Saved: control_distributions_by_batch.png")


def plot_score_correlations(df: pl.DataFrame, output_dir: Path):
    """Plot correlation matrix between scoring approaches."""
    score_cols = ["AUROC", "MislocScore_ecdf", "MislocScore_sigmoid",
                  "MislocScore_minmax", "MislocScore_cdf"]

    # Get all scores
    scores_df = df.select(score_cols).to_pandas()

    # Compute correlation matrix
    corr_matrix = scores_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, ax=ax, vmin=-1, vmax=1,
                xticklabels=["AUROC", "ECDF", "Sigmoid", "MinMax", "Effect CDF"],
                yticklabels=["AUROC", "ECDF", "Sigmoid", "MinMax", "Effect CDF"])
    ax.set_title("Correlation Between Scoring Approaches")

    plt.tight_layout()
    plt.savefig(output_dir / "score_correlations.png", dpi=150)
    plt.close()
    print(f"Saved: score_correlations.png")


def plot_variant_vs_control(df: pl.DataFrame, output_dir: Path):
    """Plot variant vs control score distributions for each approach."""
    score_cols = ["MislocScore_ecdf", "MislocScore_sigmoid",
                  "MislocScore_minmax", "MislocScore_cdf"]
    score_labels = ["ECDF (Percentile)", "Sigmoid Z-score",
                    "Min-Max Scaling", "Effect Size CDF"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(zip(score_cols, score_labels)):
        ax = axes[idx]

        # Controls
        ctrl_data = df.filter(pl.col("Metadata_Control") == True)[col].to_numpy()
        # Variants
        var_data = df.filter(pl.col("Node_Type") == "variant")[col].to_numpy()
        # Positive controls
        pc_data = df.filter(pl.col("Node_Type") == "PC")[col].to_numpy()

        if len(ctrl_data) > 0:
            ax.hist(ctrl_data, bins=30, alpha=0.5, label=f"Controls (n={len(ctrl_data)})",
                   density=True, color='blue')
        if len(var_data) > 0:
            ax.hist(var_data, bins=30, alpha=0.5, label=f"Variants (n={len(var_data)})",
                   density=True, color='orange')
        if len(pc_data) > 0:
            ax.hist(pc_data, bins=15, alpha=0.7, label=f"Pos Controls (n={len(pc_data)})",
                   density=True, color='red')

        # Add threshold lines
        ax.axvline(0.95, color='green', linestyle='--', alpha=0.7, label='0.95 threshold')
        ax.axvline(0.99, color='red', linestyle='--', alpha=0.7, label='0.99 threshold')

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution: {label}")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "variant_vs_control_distributions.png", dpi=150)
    plt.close()
    print(f"Saved: variant_vs_control_distributions.png")


def plot_batch_effect_assessment(df: pl.DataFrame, output_dir: Path):
    """Assess batch effect: do controls from different batches have similar score distributions?"""
    ctrl_df = df.filter(pl.col("Metadata_Control") == True)

    score_cols = ["MislocScore_ecdf", "MislocScore_sigmoid",
                  "MislocScore_minmax", "MislocScore_cdf"]
    score_labels = ["ECDF", "Sigmoid", "MinMax", "Effect CDF"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    batches = sorted(ctrl_df["Batch_Short"].unique().to_list())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batches)))

    for idx, (col, label) in enumerate(zip(score_cols, score_labels)):
        ax = axes[idx]

        # Box plot per batch
        data_by_batch = []
        for batch in batches:
            batch_data = ctrl_df.filter(pl.col("Batch_Short") == batch)[col].to_numpy()
            data_by_batch.append(batch_data)

        bp = ax.boxplot(data_by_batch, labels=batches, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_xlabel("Batch")
        ax.set_ylabel(label)
        ax.set_title(f"Control Scores by Batch: {label}")
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        # Add mean line
        means = [np.mean(d) for d in data_by_batch]
        ax.plot(range(1, len(batches)+1), means, 'r.-', label='Mean', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "batch_effect_boxplots.png", dpi=150)
    plt.close()
    print(f"Saved: batch_effect_boxplots.png")


def compute_summary_statistics(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """Compute summary statistics for each scoring approach."""
    ctrl_df = df.filter(pl.col("Metadata_Control") == True)
    var_df = df.filter(pl.col("Node_Type") == "variant")

    score_cols = ["AUROC", "MislocScore_ecdf", "MislocScore_sigmoid",
                  "MislocScore_minmax", "MislocScore_cdf"]

    stats_rows = []

    for col in score_cols:
        # Control statistics per batch
        ctrl_by_batch = []
        for batch in ctrl_df["Batch_Short"].unique().to_list():
            batch_data = ctrl_df.filter(pl.col("Batch_Short") == batch)[col].to_numpy()
            ctrl_by_batch.append({
                "batch": batch,
                "mean": np.mean(batch_data),
                "std": np.std(batch_data),
                "n": len(batch_data)
            })

        # Compute cross-batch consistency (std of batch means)
        batch_means = [b["mean"] for b in ctrl_by_batch]
        cross_batch_std = np.std(batch_means)

        # Overall control stats
        ctrl_all = ctrl_df[col].to_numpy()
        var_all = var_df[col].to_numpy()

        # Hits at different thresholds
        n_hits_95 = np.sum(var_all > 0.95) if col != "AUROC" else np.sum(var_all > np.percentile(ctrl_all, 95))
        n_hits_99 = np.sum(var_all > 0.99) if col != "AUROC" else np.sum(var_all > np.percentile(ctrl_all, 99))

        stats_rows.append({
            "Score": col,
            "ctrl_mean": np.mean(ctrl_all),
            "ctrl_std": np.std(ctrl_all),
            "ctrl_min": np.min(ctrl_all),
            "ctrl_max": np.max(ctrl_all),
            "cross_batch_std_of_means": cross_batch_std,
            "var_mean": np.mean(var_all),
            "var_std": np.std(var_all),
            "n_hits_95": n_hits_95,
            "n_hits_99": n_hits_99,
            "pct_hits_95": 100 * n_hits_95 / len(var_all),
            "pct_hits_99": 100 * n_hits_99 / len(var_all),
        })

    stats_df = pl.DataFrame(stats_rows)
    stats_df.write_csv(output_dir / "summary_statistics.csv")
    print(f"Saved: summary_statistics.csv")

    return stats_df


def print_batch_control_stats(df: pl.DataFrame):
    """Print control statistics by batch for comparison."""
    ctrl_df = df.filter(pl.col("Metadata_Control") == True)

    print("\n" + "="*80)
    print("CONTROL AUROC STATISTICS BY BATCH")
    print("="*80)

    for feat_type in FEAT_SETS:
        print(f"\n--- {feat_type} ---")
        print(f"{'Batch':<15} {'N':<6} {'Mean':<8} {'Std':<8} {'95th':<8} {'99th':<8}")
        print("-" * 55)

        for batch in sorted(ctrl_df["Batch_Short"].unique().to_list()):
            batch_data = ctrl_df.filter(
                (pl.col("Batch_Short") == batch) &
                (pl.col("Classifier_type") == feat_type)
            )["AUROC"].to_numpy()

            if len(batch_data) > 0:
                print(f"{batch:<15} {len(batch_data):<6} {np.mean(batch_data):<8.4f} "
                      f"{np.std(batch_data):<8.4f} {np.percentile(batch_data, 95):<8.4f} "
                      f"{np.percentile(batch_data, 99):<8.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MISLOCALIZATION SCORE COMPARISON TEST")
    print("="*80)

    # Load data
    print("\nLoading metrics from test batches...")
    df = load_all_batches()
    print(f"\nTotal classifiers loaded: {len(df)}")

    # Print raw control statistics
    print_batch_control_stats(df)

    # Compute all mislocalization scores
    print("\nComputing mislocalization scores...")
    scored_df = compute_all_misloc_scores(df)
    print(f"Scored {len(scored_df)} classifiers")

    # Save scored data
    scored_df.write_csv(OUTPUT_DIR / "all_classifiers_scored.csv")
    print(f"\nSaved: all_classifiers_scored.csv")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_control_distributions(scored_df, OUTPUT_DIR)
    plot_control_distributions_by_batch(scored_df, OUTPUT_DIR)
    plot_score_correlations(scored_df, OUTPUT_DIR)
    plot_variant_vs_control(scored_df, OUTPUT_DIR)
    plot_batch_effect_assessment(scored_df, OUTPUT_DIR)

    # Compute summary statistics
    print("\nComputing summary statistics...")
    stats_df = compute_summary_statistics(scored_df, OUTPUT_DIR)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(stats_df.to_pandas().to_string(index=False))

    print("\n" + "="*80)
    print("KEY METRIC: Cross-Batch Consistency (lower = better)")
    print("="*80)
    print("This measures the standard deviation of control means across batches.")
    print("Lower values indicate more consistent normalization across batches.\n")

    for row in stats_df.iter_rows(named=True):
        print(f"  {row['Score']:<25}: {row['cross_batch_std_of_means']:.4f}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nReview the generated plots to select the best approach.")


if __name__ == "__main__":
    main()
