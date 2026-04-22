#!/usr/bin/env python
"""
PIGN Variant Per-Channel Analysis using Feature Importance

Properly filters features based on classifier feature importance:
1. Loads feature importance from classification results
2. For each variant, selects top N features per channel
3. De-correlates features to remove redundancy
4. Creates clustered heatmaps per channel

Channels: DNA, AGP, GFP (Protein), GFP_ADJ, Morph
"""

import os
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from pycytominer.feature_select import feature_select

try:
    from dynamicTreeCut import cutreeHybrid
    HAS_DYNAMIC_TREECUT = True
except ImportError:
    HAS_DYNAMIC_TREECUT = False
    from scipy.cluster.hierarchy import fcluster

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from downstream_utils import load_batch_profiles
GENE = "PIGN"
AUROC_THRESHOLD = 0.9
TOP_N_FEATURES_PER_VARIANT = 5
CORRELATION_THRESHOLD = 0.9

# Paths
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
OUTPUT_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "pign_analysis"
CLASSIFICATION_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"

# Batch info
BATCHES = ["2026_01_05_Batch_20", "2026_01_05_Batch_21"]
PIPELINE = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells"

# Channel definitions - map analysis channel to feature pattern
CHANNELS = {
    "DNA": {
        "auroc_col": "AUROC_DNA_Mean",
        "feature_pattern": "_DNA_",
        "color": "#1f77b4"
    },
    "AGP": {
        "auroc_col": "AUROC_AGP_Mean",
        "feature_pattern": "_AGP_",
        "color": "#ff7f0e"
    },
    "GFP": {
        "auroc_col": "AUROC_GFP_Mean",
        "feature_pattern": "_Protein_",  # Protein channel in features
        "color": "#2ca02c"
    },
    "GFP_ADJ": {
        "auroc_col": "AUROC_GFP_ADJ_Mean",
        "feature_pattern": "_Protein_",  # Same features, different classifier
        "color": "#9467bd"
    },
    "Morph": {
        "auroc_col": "AUROC_Morph_Mean",
        "feature_pattern": "AreaShape_",  # Morphology features
        "color": "#d62728"
    }
}


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


def load_classification_summary():
    """Load the classification summary with per-channel AUROC values."""
    summary_path = CLASSIFICATION_DIR / "imaging_analyses_classification_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def get_channel_features(all_features: list, channel_name: str) -> list:
    """Get features belonging to a specific channel."""
    pattern = CHANNELS[channel_name]["feature_pattern"]
    return [f for f in all_features if pattern in f]


def get_top_features_per_variant(feat_imp_df: pd.DataFrame, channel_name: str,
                                  top_n: int = TOP_N_FEATURES_PER_VARIANT) -> dict:
    """
    Get top N important features for each variant in a specific channel.

    Returns dict: {variant: [list of top features]}
    """
    # Get feature columns (exclude metadata)
    meta_cols = ["Group1", "Group2", "Metadata_Feature_Type", "Metadata_Control"]
    feature_cols = [c for c in feat_imp_df.columns if c not in meta_cols]

    # Filter to channel-specific features
    pattern = CHANNELS[channel_name]["feature_pattern"]
    channel_features = [f for f in feature_cols if pattern in f]

    if not channel_features:
        print(f"  Warning: No features found for pattern '{pattern}'")
        return {}

    # Filter to PIGN variants
    pign_rows = feat_imp_df[
        (feat_imp_df["Group1"] == GENE) &
        (feat_imp_df["Group2"].str.contains(f"{GENE}_"))
    ].copy()

    if pign_rows.empty:
        return {}

    # For each variant, get top features by importance
    variant_top_features = {}

    for _, row in pign_rows.iterrows():
        variant = row["Group2"]
        # Clean variant name (remove PIGN_PIGN_ prefix if exists)
        if variant.startswith(f"{GENE}_{GENE}_"):
            variant = variant.replace(f"{GENE}_{GENE}_", f"{GENE}_")

        # Get feature importances for this row
        importances = row[channel_features].astype(float)

        # Get top N features
        top_features = importances.nlargest(top_n).index.tolist()

        if variant not in variant_top_features:
            variant_top_features[variant] = set()
        variant_top_features[variant].update(top_features)

    return variant_top_features


def aggregate_top_features(batches_top_features: list) -> dict:
    """Aggregate top features across batches."""
    aggregated = {}

    for batch_features in batches_top_features:
        for variant, features in batch_features.items():
            if variant not in aggregated:
                aggregated[variant] = set()
            aggregated[variant].update(features)

    return aggregated


def get_union_features(variant_top_features: dict) -> list:
    """Get union of all top features across variants."""
    all_features = set()
    for features in variant_top_features.values():
        all_features.update(features)
    return sorted(list(all_features))


def compute_zscore_matrix(variants: list, features: list) -> pd.DataFrame:
    """Compute Z-score matrix for variants vs reference."""
    all_profiles = []

    for batch in BATCHES:
        profiles = load_batch_profiles(batch)
        if len(profiles) == 0:
            continue

        # Filter for PIGN
        pign_profiles = profiles.filter(pl.col("Metadata_symbol") == GENE)
        if len(pign_profiles) > 0:
            pign_profiles = pign_profiles.with_columns(pl.lit(batch).alias("Batch"))
            all_profiles.append(pign_profiles)

    if not all_profiles:
        return pd.DataFrame()

    combined = pl.concat(all_profiles)

    # Get available features
    available_features = [f for f in features if f in combined.columns]
    if not available_features:
        return pd.DataFrame()

    # Compute reference statistics
    ref_profiles = combined.filter(pl.col("Metadata_gene_allele") == GENE)
    if len(ref_profiles) == 0:
        return pd.DataFrame()

    ref_means = ref_profiles.select(available_features).mean().to_pandas().iloc[0]
    ref_stds = ref_profiles.select(available_features).std().to_pandas().iloc[0]

    # Compute Z-scores for each variant
    zscore_data = []

    for variant in variants:
        var_profiles = combined.filter(pl.col("Metadata_gene_allele") == variant)
        if len(var_profiles) < 10:
            continue

        var_means = var_profiles.select(available_features).mean().to_pandas().iloc[0]

        zscores = {}
        for feat in available_features:
            if ref_stds[feat] > 0:
                zscores[feat] = (var_means[feat] - ref_means[feat]) / ref_stds[feat]
            else:
                zscores[feat] = 0

        zscores["variant"] = variant
        zscores["n_cells"] = len(var_profiles)
        zscore_data.append(zscores)

    return pd.DataFrame(zscore_data)


def decorrelate_features(zscore_df: pd.DataFrame, features: list,
                         threshold: float = CORRELATION_THRESHOLD) -> list:
    """Remove highly correlated features, keeping the one with highest variance."""
    if len(features) < 2:
        return features

    available = [f for f in features if f in zscore_df.columns]
    if len(available) < 2:
        return available

    # Compute correlation matrix
    corr_matrix = zscore_df[available].corr().abs()

    # Find features to drop
    to_drop = set()
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the one with lower variance
                var_i = zscore_df[available[i]].var()
                var_j = zscore_df[available[j]].var()
                if var_i < var_j:
                    to_drop.add(available[i])
                else:
                    to_drop.add(available[j])

    return [f for f in available if f not in to_drop]


def perform_clustering(data_matrix, min_cluster_size=5):
    """Perform hierarchical clustering with automatic cluster detection."""
    if len(data_matrix) < min_cluster_size:
        return np.ones(len(data_matrix), dtype=int), None, None, None

    # PCA
    n_components = min(len(data_matrix) - 1, data_matrix.shape[1] - 1, 20)
    if n_components < 2:
        return np.ones(len(data_matrix), dtype=int), None, None, None

    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data_matrix)

    # Clustering
    Z = linkage(data_pca, method='ward', metric='euclidean')
    dist_matrix = pdist(data_pca, metric='euclidean')

    if HAS_DYNAMIC_TREECUT and len(data_matrix) >= min_cluster_size * 2:
        result = cutreeHybrid(Z, dist_matrix, minClusterSize=min_cluster_size, deepSplit=2)
        cluster_labels = result["labels"]
    else:
        max_d = 0.7 * max(Z[:, 2]) if len(Z) > 0 else 1
        cluster_labels = fcluster(Z, max_d, criterion='distance')

    return cluster_labels, Z, data_pca, pca


def create_channel_heatmap(zscore_df: pd.DataFrame, features: list,
                           summary_df: pd.DataFrame, channel_name: str,
                           output_dir: Path, gfp_adj: bool = False):
    """Create clustered heatmap for a channel using feature importance filtered features."""

    channel_key = "GFP_ADJ" if gfp_adj else channel_name
    channel_info = CHANNELS[channel_key]
    auroc_col = channel_info["auroc_col"]

    # Filter for significant PIGN variants
    pign_summary = summary_df[summary_df["allele_0"].str.startswith(f"{GENE}_")].copy()

    if auroc_col not in pign_summary.columns:
        print(f"  Warning: {auroc_col} not found in summary")
        return None

    sig_variants = pign_summary[pign_summary[auroc_col] > AUROC_THRESHOLD]["allele_0"].tolist()

    if len(sig_variants) < 3:
        print(f"  Not enough significant variants for {channel_key}")
        return None

    # Filter zscore data
    zscore_filtered = zscore_df[zscore_df["variant"].isin(sig_variants)].copy()

    if len(zscore_filtered) < 3:
        print(f"  Not enough Z-score data for {channel_key}")
        return None

    # Get available features
    available_features = [f for f in features if f in zscore_filtered.columns]
    if len(available_features) < 3:
        print(f"  Not enough features for {channel_key}")
        return None

    # Decorrelate features
    decorr_features = decorrelate_features(zscore_filtered, available_features)
    print(f"  Features after decorrelation: {len(decorr_features)}")

    if len(decorr_features) < 3:
        decorr_features = available_features[:min(30, len(available_features))]

    # Prepare data
    heatmap_data = zscore_filtered.set_index("variant")[decorr_features].T
    data_matrix = zscore_filtered.set_index("variant")[decorr_features].values
    data_matrix = np.nan_to_num(data_matrix, nan=0.0)

    # Clustering
    min_cluster_size = max(3, len(sig_variants) // 10)
    cluster_labels, Z, data_pca, pca = perform_clustering(data_matrix, min_cluster_size)

    n_clusters = len(set(cluster_labels))
    print(f"  Found {n_clusters} clusters")

    # Colors
    colors = sns.color_palette("husl", n_clusters)
    unique_labels = sorted(np.unique(cluster_labels))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    auroc_dict = dict(zip(pign_summary["allele_0"], pign_summary[auroc_col]))
    row_colors_list = [color_map[c] for c in cluster_labels]

    cmap_auroc = plt.cm.RdYlGn
    norm_auroc = Normalize(vmin=AUROC_THRESHOLD, vmax=1.0)

    # Feature correlation linkage
    if heatmap_data.shape[0] > 1:
        try:
            corr_df = heatmap_data.T.corr()
            corr_dist = 1 - corr_df.values
            np.fill_diagonal(corr_dist, 0)
            corr_dist = np.clip(corr_dist, 0, 2)
            corr_dist_condensed = squareform(corr_dist)
            col_linkage = linkage(corr_dist_condensed, method='average', optimal_ordering=True)
        except:
            col_linkage = None
    else:
        col_linkage = None

    # Create heatmap
    fig_height = max(10, len(decorr_features) * 0.4)
    fig_width = max(10, len(heatmap_data.columns) * 0.5)

    g = sns.clustermap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        vmin=-3, vmax=3,
        col_colors=[row_colors_list],
        row_linkage=col_linkage,
        col_linkage=Z,
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(0.1, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.12),
        xticklabels=True, yticklabels=True,
        linewidths=0.05,
    )

    suffix = "_ADJ" if gfp_adj else ""
    g.ax_heatmap.set_xlabel(f"PIGN Variants (AUROC > {AUROC_THRESHOLD})", fontsize=11)
    g.ax_heatmap.set_ylabel(f"{channel_name}{suffix} Features (Top {TOP_N_FEATURES_PER_VARIANT}/variant, decorrelated)", fontsize=10)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)

    # Legend
    legend_handles = [Patch(facecolor=color_map[c], label=f'C{c} (n={sum(cluster_labels==c)})')
                      for c in sorted(color_map.keys())]
    g.ax_heatmap.legend(handles=legend_handles, title='Cluster',
                        bbox_to_anchor=(-0.15, 1.0), loc='upper left', frameon=True, fontsize=8)

    # AUROC colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_auroc, norm=norm_auroc)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.02, 0.6, 0.03, 0.12])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{channel_key} AUROC", fontsize=9)

    g.fig.suptitle(
        f"{channel_name}{suffix}: Feature Importance-based Clustering\n"
        f"({len(sig_variants)} variants, {n_clusters} clusters, {len(decorr_features)} features)",
        fontsize=12, y=1.02
    )

    # Save
    output_file = output_dir / f"PIGN_{channel_name}{suffix}_featimportance_heatmap.png"
    g.fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")

    # Return cluster assignments
    return pd.DataFrame({
        'variant': zscore_filtered["variant"].values,
        f'{channel_key}_cluster': cluster_labels,
        f'{channel_key}_AUROC': [auroc_dict.get(v, np.nan) for v in zscore_filtered["variant"]]
    })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dir = OUTPUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PIGN Per-Channel Analysis using Feature Importance")
    print(f"Top {TOP_N_FEATURES_PER_VARIANT} features per variant, decorrelated at r>{CORRELATION_THRESHOLD}")
    print("=" * 80)

    # Load classification summary
    summary_df = load_classification_summary()
    if summary_df.empty:
        print("Error: No classification summary found")
        return

    # Process each channel
    all_cluster_dfs = []

    for channel_name in ["DNA", "AGP", "GFP", "Morph"]:
        print(f"\n{'='*60}")
        print(f"Processing {channel_name} channel")
        print("=" * 60)

        # Load feature importance from both batches
        batches_top_features = []
        for batch in BATCHES:
            fi_df = load_feature_importance(batch, gfp_adj=False)
            if not fi_df.empty:
                top_feats = get_top_features_per_variant(fi_df, channel_name)
                if top_feats:
                    batches_top_features.append(top_feats)
                    print(f"  {batch}: Found top features for {len(top_feats)} variants")

        if not batches_top_features:
            print(f"  No feature importance data for {channel_name}")
            continue

        # Aggregate and get union of features
        aggregated = aggregate_top_features(batches_top_features)
        union_features = get_union_features(aggregated)
        print(f"  Union of top features: {len(union_features)}")

        # Compute Z-scores
        sig_variants = summary_df[
            (summary_df["allele_0"].str.startswith(f"{GENE}_")) &
            (summary_df[CHANNELS[channel_name]["auroc_col"]] > AUROC_THRESHOLD)
        ]["allele_0"].tolist()

        print(f"  Significant variants (AUROC > {AUROC_THRESHOLD}): {len(sig_variants)}")

        zscore_df = compute_zscore_matrix(sig_variants, union_features)

        if zscore_df.empty:
            print(f"  No Z-score data for {channel_name}")
            continue

        # Create heatmap
        cluster_df = create_channel_heatmap(
            zscore_df, union_features, summary_df, channel_name, heatmap_dir
        )

        if cluster_df is not None:
            all_cluster_dfs.append(cluster_df)

    # Process GFP_ADJ separately
    print(f"\n{'='*60}")
    print("Processing GFP_ADJ channel")
    print("=" * 60)

    batches_top_features = []
    for batch in BATCHES:
        fi_df = load_feature_importance(batch, gfp_adj=True)
        if not fi_df.empty:
            top_feats = get_top_features_per_variant(fi_df, "GFP")  # Same pattern as GFP
            if top_feats:
                batches_top_features.append(top_feats)
                print(f"  {batch}: Found top features for {len(top_feats)} variants")

    if batches_top_features:
        aggregated = aggregate_top_features(batches_top_features)
        union_features = get_union_features(aggregated)
        print(f"  Union of top features: {len(union_features)}")

        sig_variants = summary_df[
            (summary_df["allele_0"].str.startswith(f"{GENE}_")) &
            (summary_df["AUROC_GFP_ADJ_Mean"] > AUROC_THRESHOLD)
        ]["allele_0"].tolist()

        print(f"  Significant variants (AUROC > {AUROC_THRESHOLD}): {len(sig_variants)}")

        zscore_df = compute_zscore_matrix(sig_variants, union_features)

        if not zscore_df.empty:
            cluster_df = create_channel_heatmap(
                zscore_df, union_features, summary_df, "GFP", heatmap_dir, gfp_adj=True
            )
            if cluster_df is not None:
                all_cluster_dfs.append(cluster_df)

    # Save cluster labels
    if all_cluster_dfs:
        merged = all_cluster_dfs[0]
        for df in all_cluster_dfs[1:]:
            merged = merged.merge(df, on='variant', how='outer')

        cluster_file = OUTPUT_DIR / "PIGN_featimportance_cluster_labels.csv"
        merged.to_csv(cluster_file, index=False)
        print(f"\nCluster labels saved to: {cluster_file}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
