#!/usr/bin/env python
"""
Gregor Alleles Analysis - Non-PIGN Mislocalization Hits from Batch 20 & 21

This script analyzes non-PIGN alleles with strong mislocalization phenotypes:
1. Identifies hits (passing control thresholds OR AUROC > 0.9)
2. Extracts feature importance (> 0.01 threshold)
3. Cleans features via correlation/variance filtering
4. Performs hierarchical clustering
5. Generates violin plots for top features per variant

Usage:
    python analyze_gregor_alleles.py
"""

import os
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from downstream_utils import load_batch_profiles
OUTPUT_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "gregor_alleles"
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
CLASSIFICATION_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"
HIT_CALLS_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "hit_calls"

# Analysis parameters
BATCHES = ["2026_01_05_Batch_20", "2026_01_05_Batch_21"]
PIPELINE = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells"
AUROC_THRESHOLD = 0.9  # For identifying strong phenotypes
FEATURE_IMPORTANCE_THRESHOLD = 0.01  # Min importance to include feature
CORRELATION_THRESHOLD = 0.85  # For decorrelation
VARIANCE_THRESHOLD = 0.01  # Min variance to keep feature

# Genes to exclude (PIGN variants already analyzed separately)
EXCLUDE_GENES = ["PIGN"]

# Control alleles
TC = ["EGFP"]
NC = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
PC = ["ALK", "ALK_Arg1275Gln", "PTK2B"]
ALL_CONTROLS = TC + NC + PC


def load_classification_summary():
    """Load the classification summary with per-channel AUROC values."""
    summary_path = CLASSIFICATION_DIR / "imaging_analyses_classification_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    raise FileNotFoundError(f"Classification summary not found: {summary_path}")


def load_hit_calls():
    """Load hit calls for both batches."""
    all_hits = []
    for batch in BATCHES:
        hit_path = HIT_CALLS_DIR / batch / "hit_calls.csv"
        if hit_path.exists():
            df = pd.read_csv(hit_path)
            all_hits.append(df)

    if all_hits:
        return pd.concat(all_hits, ignore_index=True)
    raise FileNotFoundError("No hit calls found")


def load_feature_importance(batch: str, gfp_adj: bool = False) -> pd.DataFrame:
    """Load feature importance for a batch."""
    results_dir = PIPELINE_DIR / "outputs" / "classification_results" / batch / PIPELINE

    if gfp_adj:
        fi_path = results_dir / "feat_importance_gfp_adj.csv"
    else:
        fi_path = results_dir / "feat_importance.csv"

    if fi_path.exists():
        return pd.read_csv(fi_path)
    return pd.DataFrame()


def identify_non_pign_hits(summary_df: pd.DataFrame, hit_calls_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify non-PIGN alleles with strong mislocalization phenotypes.

    Criteria:
    1. Hit called in at least 2 channels (95th percentile) OR
    2. Mean GFP AUROC > 0.9
    """
    # Filter out PIGN and control alleles
    non_pign = summary_df[
        ~summary_df["allele_0"].str.startswith(tuple(EXCLUDE_GENES)) &
        ~summary_df["allele_0"].isin(ALL_CONTROLS)
    ].copy()

    # Get hit status from hit_calls (average across batches if multiple)
    hits_summary = hit_calls_df.groupby("allele_0").agg({
        "hit_multichannel_95": "any",
        "hit_multichannel_99": "any",
        "n_channels_hit_95": "max",
        "n_channels_hit_99": "max",
        "AUROC_Mean_Channel": "mean"
    }).reset_index()

    # Merge with summary
    result = non_pign.merge(
        hits_summary[["allele_0", "hit_multichannel_95", "hit_multichannel_99", "n_channels_hit_95"]],
        on="allele_0",
        how="left"
    )

    # Apply criteria: hits OR high AUROC
    result["is_hit"] = (
        (result["hit_multichannel_95"] == True) |
        (result["AUROC_GFP_Mean"] > AUROC_THRESHOLD)
    )

    hits = result[result["is_hit"]].copy()
    hits = hits.sort_values("AUROC_GFP_Mean", ascending=False)

    return hits


def get_important_features(feat_imp_df: pd.DataFrame, variants: list,
                           threshold: float = FEATURE_IMPORTANCE_THRESHOLD) -> set:
    """
    Get features with importance > threshold for any of the specified variants.
    """
    # Identify metadata columns
    meta_cols = ["Group1", "Group2", "Metadata_Feature_Type", "Metadata_Control"]
    feature_cols = [c for c in feat_imp_df.columns if c not in meta_cols]

    important_features = set()

    for variant in variants:
        # Find rows for this variant (variant vs wildtype)
        gene = variant.split("_")[0] if "_" in variant else variant

        variant_rows = feat_imp_df[
            (feat_imp_df["Group2"].str.contains(variant, na=False)) |
            (feat_imp_df["Group1"].str.contains(variant, na=False))
        ]

        if len(variant_rows) == 0:
            continue

        # Get features above threshold
        for _, row in variant_rows.iterrows():
            for feat in feature_cols:
                try:
                    if float(row[feat]) > threshold:
                        important_features.add(feat)
                except (ValueError, TypeError):
                    continue

    return important_features


def filter_features_by_correlation_variance(
    data_df: pd.DataFrame,
    features: list,
    corr_threshold: float = CORRELATION_THRESHOLD,
    var_threshold: float = VARIANCE_THRESHOLD
) -> list:
    """
    Filter features by variance threshold and correlation.
    Keeps the feature with highest variance when correlation > threshold.
    """
    # Filter by variance first
    variances = data_df[features].var()
    var_features = variances[variances > var_threshold].index.tolist()

    if len(var_features) < 3:
        return var_features

    # Decorrelate
    corr_matrix = data_df[var_features].corr().abs()

    to_drop = set()
    sorted_by_var = variances[var_features].sort_values(ascending=False).index.tolist()

    for i, feat_i in enumerate(sorted_by_var):
        if feat_i in to_drop:
            continue
        for feat_j in sorted_by_var[i+1:]:
            if feat_j in to_drop:
                continue
            if corr_matrix.loc[feat_i, feat_j] > corr_threshold:
                to_drop.add(feat_j)  # Drop lower variance feature

    return [f for f in var_features if f not in to_drop]


def compute_variant_zscore_matrix(
    variants: list,
    features: list,
    batches: list = BATCHES
) -> pd.DataFrame:
    """
    Compute Z-score matrix for variants relative to their wildtype.
    """
    all_profiles = []

    for batch in batches:
        profiles = load_batch_profiles(batch)
        if len(profiles) == 0:
            continue
        profiles = profiles.with_columns(pl.lit(batch).alias("Batch"))
        all_profiles.append(profiles)

    if not all_profiles:
        return pd.DataFrame()

    combined = pl.concat(all_profiles)

    # Get available features
    available_features = [f for f in features if f in combined.columns]
    if not available_features:
        return pd.DataFrame()

    zscore_data = []

    for variant in variants:
        # Parse gene from variant name
        gene = variant.split("_")[0] if "_" in variant else None
        if gene is None:
            continue

        # Get variant cells
        var_profiles = combined.filter(pl.col("Metadata_gene_allele") == variant)
        if len(var_profiles) < 10:
            continue

        # Get reference (wildtype) cells
        ref_profiles = combined.filter(pl.col("Metadata_gene_allele") == gene)
        if len(ref_profiles) < 10:
            continue

        # Compute statistics
        ref_means = ref_profiles.select(available_features).mean().to_pandas().iloc[0]
        ref_stds = ref_profiles.select(available_features).std().to_pandas().iloc[0]
        var_means = var_profiles.select(available_features).mean().to_pandas().iloc[0]

        zscores = {"variant": variant, "gene": gene, "n_cells": len(var_profiles)}

        for feat in available_features:
            if ref_stds[feat] > 0:
                zscores[feat] = (var_means[feat] - ref_means[feat]) / ref_stds[feat]
            else:
                zscores[feat] = 0

        zscore_data.append(zscores)

    return pd.DataFrame(zscore_data)


def perform_clustering(zscore_df: pd.DataFrame, features: list) -> tuple:
    """
    Perform hierarchical clustering on variants based on feature Z-scores.
    Returns (cluster_labels, linkage_matrix, ordered_variants).
    """
    data_matrix = zscore_df.set_index("variant")[features].values
    data_matrix = np.nan_to_num(data_matrix, nan=0.0)

    if len(data_matrix) < 3:
        return np.ones(len(data_matrix), dtype=int), None, zscore_df["variant"].tolist()

    # Compute linkage
    Z = linkage(data_matrix, method='ward', metric='euclidean')

    # Auto-detect clusters (use moderate threshold)
    max_d = 0.7 * max(Z[:, 2]) if len(Z) > 0 else 1
    cluster_labels = fcluster(Z, max_d, criterion='distance')

    # Get dendrogram order
    dendro = dendrogram(Z, no_plot=True)
    ordered_idx = dendro['leaves']
    ordered_variants = zscore_df.iloc[ordered_idx]["variant"].tolist()

    return cluster_labels, Z, ordered_variants


def create_clustering_heatmap(
    zscore_df: pd.DataFrame,
    features: list,
    summary_df: pd.DataFrame,
    output_path: Path
):
    """Create clustered heatmap of variants vs features."""
    cluster_labels, Z, ordered_variants = perform_clustering(zscore_df, features)

    n_clusters = len(set(cluster_labels))
    print(f"  Found {n_clusters} clusters")

    # Prepare heatmap data
    heatmap_data = zscore_df.set_index("variant")[features].T

    # Clip extreme values
    heatmap_data = heatmap_data.clip(-5, 5)

    # Color mapping
    colors = sns.color_palette("husl", n_clusters)
    color_map = {i+1: colors[i] for i in range(n_clusters)}

    variant_to_cluster = dict(zip(zscore_df["variant"], cluster_labels))
    col_colors = [color_map[variant_to_cluster[v]] for v in heatmap_data.columns]

    # Get AUROC for annotation
    auroc_dict = dict(zip(summary_df["allele_0"], summary_df["AUROC_GFP_Mean"]))

    # Feature correlation linkage
    if heatmap_data.shape[0] > 1:
        try:
            corr_dist = 1 - heatmap_data.T.corr().values
            np.fill_diagonal(corr_dist, 0)
            corr_dist = np.clip(corr_dist, 0, 2)
            row_linkage = linkage(squareform(corr_dist), method='average')
        except:
            row_linkage = None
    else:
        row_linkage = None

    # Create compact figure with readable fonts
    # Smaller figure for better readability
    fig_width = 14
    fig_height = 18

    g = sns.clustermap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        vmin=-3, vmax=3,
        col_colors=[col_colors],
        row_linkage=row_linkage,
        col_linkage=Z,
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(0.08, 0.12),
        cbar_pos=None,  # Disable default colorbar, add manually
        xticklabels=True, yticklabels=True,
        linewidths=0.3,
    )

    # Larger fonts for axis labels
    g.ax_heatmap.set_xlabel("Variants (non-PIGN mislocalization hits)", fontsize=14, fontweight='bold')
    g.ax_heatmap.set_ylabel("Features (importance > 0.01, decorrelated)", fontsize=14, fontweight='bold')

    # Larger fonts for tick labels
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=60, ha="right", fontsize=11)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=9)

    # Add colorbar on the right side (center-right position)
    cbar_ax = g.fig.add_axes([0.92, 0.35, 0.02, 0.3])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-3, vmax=3))
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Z-score', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Legend with larger font
    legend_handles = [Patch(facecolor=color_map[c], label=f'Cluster {c} (n={sum(cluster_labels==c)})')
                      for c in sorted(color_map.keys())]
    g.ax_heatmap.legend(handles=legend_handles, title='Cluster',
                        bbox_to_anchor=(1.15, 1.0), loc='upper left', frameon=True,
                        fontsize=11, title_fontsize=12)

    # Title with larger font
    g.fig.suptitle(
        f"Non-PIGN Mislocalization Hits: Feature-based Clustering\n"
        f"({len(zscore_df)} variants, {n_clusters} clusters, {len(features)} features)",
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    g.fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

    return pd.DataFrame({
        'variant': zscore_df["variant"].values,
        'cluster': cluster_labels,
        'AUROC_GFP': [auroc_dict.get(v, np.nan) for v in zscore_df["variant"]]
    })


def get_top_features_per_variant(
    feat_imp_df: pd.DataFrame,
    variants: list,
    top_n: int = 5
) -> dict:
    """Get top N features for each variant based on feature importance."""
    meta_cols = ["Group1", "Group2", "Metadata_Feature_Type", "Metadata_Control"]
    feature_cols = [c for c in feat_imp_df.columns if c not in meta_cols]

    variant_features = {}

    for variant in variants:
        variant_rows = feat_imp_df[
            (feat_imp_df["Group2"].str.contains(variant, na=False)) |
            (feat_imp_df["Group1"].str.contains(variant, na=False))
        ]

        if len(variant_rows) == 0:
            continue

        # Aggregate importances across rows
        importances = variant_rows[feature_cols].mean()
        top_features = importances.nlargest(top_n).index.tolist()
        variant_features[variant] = top_features

    return variant_features


def create_violin_plots(
    variants: list,
    features: list,
    output_dir: Path,
    max_features: int = 10
):
    """Create violin plots for top features per variant."""
    # Load profiles
    all_profiles = []
    for batch in BATCHES:
        profiles = load_batch_profiles(batch)
        if len(profiles) > 0:
            profiles = profiles.with_columns(pl.lit(batch).alias("Batch"))
            all_profiles.append(profiles)

    if not all_profiles:
        print("  No profiles found for violin plots")
        return

    combined = pl.concat(all_profiles)

    violin_dir = output_dir / "violin_plots"
    violin_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants[:15]:  # Limit to top 15 variants
        gene = variant.split("_")[0] if "_" in variant else None
        if gene is None:
            continue

        # Get cells for variant and wildtype
        var_cells = combined.filter(pl.col("Metadata_gene_allele") == variant)
        wt_cells = combined.filter(pl.col("Metadata_gene_allele") == gene)

        if len(var_cells) < 10 or len(wt_cells) < 10:
            continue

        # Get available features
        plot_features = [f for f in features[:max_features] if f in combined.columns]

        if len(plot_features) == 0:
            continue

        # Prepare data for plotting
        var_data = var_cells.select(["Batch"] + plot_features).to_pandas()
        var_data["Type"] = "Variant"
        var_data["Allele"] = variant

        wt_data = wt_cells.select(["Batch"] + plot_features).to_pandas()
        wt_data["Type"] = "Wildtype"
        wt_data["Allele"] = gene

        plot_data = pd.concat([var_data, wt_data], ignore_index=True)

        # Create multi-panel figure
        n_features = len(plot_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, feat in enumerate(plot_features):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]

            # Truncate feature name for display
            short_feat = feat[:40] + "..." if len(feat) > 40 else feat

            sns.violinplot(
                data=plot_data,
                x="Batch",
                y=feat,
                hue="Type",
                split=True,
                ax=ax,
                palette={"Variant": "#e74c3c", "Wildtype": "#3498db"}
            )
            ax.set_title(short_feat, fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45)

            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            else:
                ax.get_legend().remove()

        # Hide empty subplots
        for idx in range(n_features, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle(f"{variant} vs {gene}\n(n={len(var_cells)} variant, n={len(wt_cells)} WT cells)",
                     fontsize=12, y=1.02)
        plt.tight_layout()

        output_path = violin_dir / f"{variant}_violins.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {variant}_violins.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Non-PIGN Mislocalization Alleles Analysis (Batch 20 & 21)")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1] Loading classification data...")
    summary_df = load_classification_summary()
    hit_calls_df = load_hit_calls()

    print(f"  Total alleles in summary: {len(summary_df)}")
    print(f"  Total hit call records: {len(hit_calls_df)}")

    # Step 2: Identify non-PIGN hits
    print("\n[2] Identifying non-PIGN mislocalization hits...")
    hits_df = identify_non_pign_hits(summary_df, hit_calls_df)

    print(f"  Non-PIGN hits found: {len(hits_df)}")

    # Save hit list
    hits_df.to_csv(OUTPUT_DIR / "non_pign_hits.csv", index=False)
    print(f"  Saved hit list to: non_pign_hits.csv")

    # Display top hits
    print("\n  Top 20 non-PIGN hits by GFP AUROC:")
    display_cols = ["allele_0", "AUROC_GFP_Mean", "AUROC_GFP_ADJ_Mean",
                    "hit_multichannel_95", "n_channels_hit_95"]
    display_cols = [c for c in display_cols if c in hits_df.columns]
    print(hits_df[display_cols].head(20).to_string())

    # Step 3: Extract important features
    print("\n[3] Extracting features with importance > 0.01...")
    hit_variants = hits_df["allele_0"].tolist()

    all_important_features = set()
    for batch in BATCHES:
        # Regular feature importance
        fi_df = load_feature_importance(batch, gfp_adj=False)
        if not fi_df.empty:
            features = get_important_features(fi_df, hit_variants, FEATURE_IMPORTANCE_THRESHOLD)
            all_important_features.update(features)
            print(f"  {batch}: Found {len(features)} important features")

        # GFP-adjusted
        fi_adj_df = load_feature_importance(batch, gfp_adj=True)
        if not fi_adj_df.empty:
            features = get_important_features(fi_adj_df, hit_variants, FEATURE_IMPORTANCE_THRESHOLD)
            all_important_features.update(features)
            print(f"  {batch} (GFP_ADJ): Found {len(features)} important features")

    important_features = list(all_important_features)
    print(f"  Total unique important features: {len(important_features)}")

    # Step 4: Compute Z-scores
    print("\n[4] Computing Z-scores for variants...")
    zscore_df = compute_variant_zscore_matrix(hit_variants, important_features)

    if len(zscore_df) == 0:
        print("  ERROR: No Z-score data computed!")
        return

    print(f"  Computed Z-scores for {len(zscore_df)} variants")

    # Step 5: Filter features
    print("\n[5] Filtering features by correlation and variance...")
    available_features = [f for f in important_features if f in zscore_df.columns]
    filtered_features = filter_features_by_correlation_variance(
        zscore_df, available_features, CORRELATION_THRESHOLD, VARIANCE_THRESHOLD
    )
    print(f"  Features after filtering: {len(filtered_features)}")

    # Save feature list
    pd.DataFrame({"feature": filtered_features}).to_csv(
        OUTPUT_DIR / "filtered_features.csv", index=False
    )

    # Step 6: Clustering
    print("\n[6] Performing hierarchical clustering...")
    cluster_df = create_clustering_heatmap(
        zscore_df, filtered_features, summary_df,
        OUTPUT_DIR / "clustering_heatmap.png"
    )

    # Save cluster assignments
    cluster_df.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
    print(f"  Saved cluster assignments")

    # Print cluster summary
    print("\n  Cluster summary:")
    for cluster_id in sorted(cluster_df["cluster"].unique()):
        cluster_vars = cluster_df[cluster_df["cluster"] == cluster_id]["variant"].tolist()
        mean_auroc = cluster_df[cluster_df["cluster"] == cluster_id]["AUROC_GFP"].mean()
        print(f"    Cluster {cluster_id}: {len(cluster_vars)} variants, mean AUROC = {mean_auroc:.3f}")
        for v in cluster_vars[:5]:
            print(f"      - {v}")
        if len(cluster_vars) > 5:
            print(f"      ... and {len(cluster_vars)-5} more")

    # Step 7: Get top features per variant for violin plots
    print("\n[7] Extracting top features per variant...")
    all_fi_dfs = []
    for batch in BATCHES:
        fi_df = load_feature_importance(batch, gfp_adj=False)
        if not fi_df.empty:
            all_fi_dfs.append(fi_df)
        fi_adj_df = load_feature_importance(batch, gfp_adj=True)
        if not fi_adj_df.empty:
            all_fi_dfs.append(fi_adj_df)

    if all_fi_dfs:
        combined_fi = pd.concat(all_fi_dfs, ignore_index=True)
        variant_top_features = get_top_features_per_variant(
            combined_fi, hit_variants, top_n=5
        )

        # Get union of top features
        top_feature_union = set()
        for feats in variant_top_features.values():
            top_feature_union.update(feats)

        # Save top features per variant
        top_feat_records = []
        for var, feats in variant_top_features.items():
            for f in feats:
                top_feat_records.append({"variant": var, "feature": f})
        pd.DataFrame(top_feat_records).to_csv(
            OUTPUT_DIR / "top_features_per_variant.csv", index=False
        )
        print(f"  Found {len(top_feature_union)} unique top features across variants")

    # Step 8: Create violin plots
    print("\n[8] Creating violin plots for top variants...")
    create_violin_plots(hit_variants, filtered_features, OUTPUT_DIR)

    # Step 9: Save Z-score matrix
    zscore_df.to_csv(OUTPUT_DIR / "zscore_matrix.csv", index=False)
    print(f"\n  Saved Z-score matrix")

    # Summary report
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey findings:")
    print(f"  - Non-PIGN hits identified: {len(hits_df)}")
    print(f"  - Features with importance > {FEATURE_IMPORTANCE_THRESHOLD}: {len(important_features)}")
    print(f"  - Features after correlation/variance filtering: {len(filtered_features)}")
    print(f"  - Clusters identified: {len(cluster_df['cluster'].unique())}")

    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - non_pign_hits.csv: List of all non-PIGN mislocalization hits")
    print("  - clustering_heatmap.png: Hierarchical clustering visualization")
    print("  - cluster_assignments.csv: Variant-to-cluster mapping")
    print("  - filtered_features.csv: Features used for analysis")
    print("  - top_features_per_variant.csv: Top 5 features per variant")
    print("  - zscore_matrix.csv: Full Z-score matrix")
    print("  - violin_plots/: Per-variant violin plots")


if __name__ == "__main__":
    main()
