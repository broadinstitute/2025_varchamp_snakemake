#!/usr/bin/env python
"""
PIGN Variant Per-Channel Analysis Script

Generates:
1. Comprehensive per-channel report for all PIGN variants
2. Channel-specific heatmaps for variants with AUROC > 0.9 per channel (DNA, AGP, Protein, Morph)
3. Hierarchical clustering with automated optimal cluster detection per channel
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

# Try to import dynamicTreeCut
try:
    from dynamicTreeCut import cutreeHybrid
    HAS_DYNAMIC_TREECUT = True
except ImportError:
    HAS_DYNAMIC_TREECUT = False
    print("Warning: dynamicTreeCut not installed. Using scipy dendrogram cutting instead.")
    from scipy.cluster.hierarchy import fcluster

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GENE = "PIGN"
AUROC_THRESHOLD = 0.9

# Paths
OUTPUT_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "pign_analysis"
CLASSIFICATION_DIR = PROJECT_ROOT / "3.downstream_analyses" / "outputs" / "2.classification_results" / "2026_01_Batch_20-21"
ZSCORE_PATH = OUTPUT_DIR / "heatmaps" / "PIGN_zscore_data.csv"

# Channel definitions
CHANNELS = {
    "DNA": {
        "auroc_col": "AUROC_DNA_Mean",
        "altered_col": "Altered_95th_perc_both_batches_DNA",
        "feature_patterns": ["_DNA_"],
        "color": "#1f77b4"  # Blue
    },
    "AGP": {
        "auroc_col": "AUROC_AGP_Mean",
        "altered_col": "Altered_95th_perc_both_batches_AGP",
        "feature_patterns": ["_AGP_"],
        "color": "#ff7f0e"  # Orange
    },
    "Protein": {
        "auroc_col": "AUROC_GFP_Mean",
        "altered_col": "Altered_95th_perc_both_batches_GFP",
        "feature_patterns": ["_Protein_", "_GFP_"],
        "color": "#2ca02c"  # Green
    },
    "Morph": {
        "auroc_col": "AUROC_Morph_Mean",
        "altered_col": "Altered_95th_perc_both_batches_Morph",
        "feature_patterns": ["AreaShape_", "Granularity_", "RadialDistribution_"],
        "color": "#d62728"  # Red
    }
}


def load_classification_summary():
    """Load the classification summary with per-channel AUROC values."""
    summary_path = CLASSIFICATION_DIR / "imaging_analyses_classification_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def load_zscore_data():
    """Load the Z-score data."""
    if ZSCORE_PATH.exists():
        return pd.read_csv(ZSCORE_PATH)
    return pd.DataFrame()


def get_channel_features(zscore_df, channel_name):
    """Get features belonging to a specific channel."""
    channel_info = CHANNELS[channel_name]
    patterns = channel_info["feature_patterns"]

    feature_cols = [c for c in zscore_df.columns if c not in ["variant", "n_cells"]]
    channel_features = []

    for col in feature_cols:
        for pattern in patterns:
            if pattern in col:
                channel_features.append(col)
                break

    return channel_features


def filter_pign_variants(df):
    """Filter for PIGN variants only."""
    return df[df["allele_0"].str.startswith(f"{GENE}_")].copy()


def generate_channel_report(summary_df, output_path):
    """Generate comprehensive per-channel report for PIGN variants."""
    pign_df = filter_pign_variants(summary_df)

    if pign_df.empty:
        print("No PIGN variants found")
        return

    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PIGN VARIANT PER-CHANNEL MISLOCALIZATION ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total PIGN variants analyzed: {len(pign_df)}\n\n")

        # Summary per channel
        f.write("SUMMARY OF SIGNIFICANT VARIANTS PER CHANNEL (AUROC > 0.9):\n")
        f.write("-" * 100 + "\n")

        channel_hits = {}
        for channel_name, channel_info in CHANNELS.items():
            auroc_col = channel_info["auroc_col"]
            if auroc_col in pign_df.columns:
                hits = pign_df[pign_df[auroc_col] > AUROC_THRESHOLD]
                channel_hits[channel_name] = hits
                f.write(f"{channel_name:12s}: {len(hits):3d} variants with AUROC > {AUROC_THRESHOLD}\n")

        f.write("\n")

        # Detailed results per channel
        for channel_name, channel_info in CHANNELS.items():
            auroc_col = channel_info["auroc_col"]
            altered_col = channel_info["altered_col"]

            if auroc_col not in pign_df.columns:
                continue

            f.write("\n" + "=" * 100 + "\n")
            f.write(f"{channel_name.upper()} CHANNEL RESULTS\n")
            f.write("=" * 100 + "\n\n")

            # Sort by AUROC for this channel
            channel_df = pign_df.sort_values(auroc_col, ascending=False).copy()

            # Significant variants (AUROC > 0.9)
            sig_df = channel_df[channel_df[auroc_col] > AUROC_THRESHOLD]

            f.write(f"Significant Variants ({channel_name} AUROC > {AUROC_THRESHOLD}): {len(sig_df)}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Rank':<6}{'Variant':<35}{'AUROC':<12}{'Altered_95th':<15}\n")
            f.write("-" * 100 + "\n")

            for i, (_, row) in enumerate(sig_df.iterrows()):
                altered_str = "Yes" if row.get(altered_col, False) else "No"
                f.write(f"{i+1:<6}{row['allele_0']:<35}{row[auroc_col]:.6f}    {altered_str:<15}\n")

            # All variants sorted by this channel
            f.write(f"\nAll PIGN Variants (sorted by {channel_name} AUROC):\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Rank':<6}{'Variant':<35}{'AUROC':<12}{'Significant':<15}\n")
            f.write("-" * 100 + "\n")

            for i, (_, row) in enumerate(channel_df.iterrows()):
                sig_str = "***" if row[auroc_col] > AUROC_THRESHOLD else ""
                f.write(f"{i+1:<6}{row['allele_0']:<35}{row[auroc_col]:.6f}    {sig_str:<15}\n")

        # Cross-channel comparison
        f.write("\n" + "=" * 100 + "\n")
        f.write("CROSS-CHANNEL COMPARISON (Top 30 Variants)\n")
        f.write("=" * 100 + "\n\n")

        # Calculate mean AUROC across channels
        auroc_cols = [ch["auroc_col"] for ch in CHANNELS.values() if ch["auroc_col"] in pign_df.columns]
        pign_df["AUROC_Mean_All"] = pign_df[auroc_cols].mean(axis=1)

        top_df = pign_df.nlargest(30, "AUROC_Mean_All")

        f.write(f"{'Variant':<30}{'DNA':>10}{'AGP':>10}{'Protein':>10}{'Morph':>10}{'Mean':>10}{'Sig_Channels':<15}\n")
        f.write("-" * 100 + "\n")

        for _, row in top_df.iterrows():
            sig_channels = []
            for ch_name, ch_info in CHANNELS.items():
                if ch_info["auroc_col"] in pign_df.columns:
                    if row[ch_info["auroc_col"]] > AUROC_THRESHOLD:
                        sig_channels.append(ch_name[0])  # First letter

            sig_str = "".join(sig_channels) if sig_channels else "-"

            dna_val = row.get("AUROC_DNA_Mean", np.nan)
            agp_val = row.get("AUROC_AGP_Mean", np.nan)
            gfp_val = row.get("AUROC_GFP_Mean", np.nan)
            morph_val = row.get("AUROC_Morph_Mean", np.nan)
            mean_val = row["AUROC_Mean_All"]

            f.write(f"{row['allele_0']:<30}{dna_val:>10.4f}{agp_val:>10.4f}{gfp_val:>10.4f}{morph_val:>10.4f}{mean_val:>10.4f}  {sig_str:<15}\n")

        f.write("\nLegend: D=DNA, A=AGP, P=Protein, M=Morph (channels with AUROC > 0.9)\n")

    print(f"Per-channel report saved to: {output_path}")
    return pign_df


def create_channel_heatmap(zscore_df, summary_df, channel_name, output_dir):
    """Create heatmap for a specific channel with variants having AUROC > 0.9."""
    channel_info = CHANNELS[channel_name]
    auroc_col = channel_info["auroc_col"]

    # Filter PIGN variants
    pign_summary = filter_pign_variants(summary_df)

    if auroc_col not in pign_summary.columns:
        print(f"AUROC column {auroc_col} not found")
        return

    # Get variants with AUROC > 0.9 for this channel
    sig_variants = pign_summary[pign_summary[auroc_col] > AUROC_THRESHOLD]["allele_0"].tolist()

    if not sig_variants:
        print(f"No significant variants found for {channel_name}")
        return

    print(f"{channel_name}: {len(sig_variants)} variants with AUROC > {AUROC_THRESHOLD}")

    # Get channel-specific features
    channel_features = get_channel_features(zscore_df, channel_name)

    if not channel_features:
        print(f"No features found for {channel_name}")
        return

    print(f"  Features: {len(channel_features)}")

    # Filter zscore data for significant variants
    zscore_filtered = zscore_df[zscore_df["variant"].isin(sig_variants)].copy()

    if zscore_filtered.empty:
        print(f"No Z-score data for significant {channel_name} variants")
        return

    # Select top features by variance
    available_features = [f for f in channel_features if f in zscore_filtered.columns]
    if len(available_features) == 0:
        print(f"No overlapping features found for {channel_name}")
        return

    feature_vars = zscore_filtered[available_features].var()
    n_features = min(30, len(available_features))
    top_features = feature_vars.nlargest(n_features).index.tolist()

    # Prepare heatmap data
    heatmap_data = zscore_filtered.set_index("variant")[top_features].T

    # Get AUROC values for column colors
    auroc_dict = dict(zip(pign_summary["allele_0"], pign_summary[auroc_col]))

    col_colors = []
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=AUROC_THRESHOLD, vmax=1.0)

    for variant in heatmap_data.columns:
        auroc = auroc_dict.get(variant, AUROC_THRESHOLD)
        col_colors.append(cmap(norm(auroc)))

    # Create clustermap
    fig_height = max(8, len(top_features) * 0.4)
    fig_width = max(8, len(heatmap_data.columns) * 0.6)

    plt.figure(figsize=(fig_width + 2, fig_height + 2))

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

    # Improve labels
    g.ax_heatmap.set_xlabel(f"PIGN Variants (AUROC > {AUROC_THRESHOLD})", fontsize=11)
    g.ax_heatmap.set_ylabel(f"{channel_name} Features", fontsize=11)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)

    # Add AUROC colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.02, 0.6, 0.03, 0.15])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{channel_name} AUROC", fontsize=9)

    g.fig.suptitle(
        f"{channel_name} Channel: Feature Z-scores for Significant PIGN Variants\n"
        f"({len(sig_variants)} variants with {channel_name} AUROC > {AUROC_THRESHOLD})",
        fontsize=12, y=1.02
    )

    # Save
    output_file = output_dir / f"PIGN_{channel_name}_zscore_heatmap.png"
    g.fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Heatmap saved to: {output_file}")

    return sig_variants


def perform_clustering(data_matrix, min_cluster_size=5, deep_split=2):
    """
    Perform hierarchical clustering with automatic cluster detection.

    Uses dynamicTreeCut if available, otherwise falls back to scipy.
    """
    if len(data_matrix) < min_cluster_size:
        return np.ones(len(data_matrix), dtype=int)

    # PCA for dimensionality reduction (keep 90% variance)
    n_components = min(len(data_matrix) - 1, data_matrix.shape[1] - 1, 50)
    if n_components < 2:
        n_components = min(len(data_matrix) - 1, data_matrix.shape[1] - 1)

    pca = PCA(n_components=min(n_components, 0.9) if n_components > 2 else n_components, random_state=42)
    data_pca = pca.fit_transform(data_matrix)

    # Hierarchical clustering
    Z = linkage(data_pca, method='ward', metric='euclidean')
    dist_matrix = pdist(data_pca, metric='euclidean')

    if HAS_DYNAMIC_TREECUT:
        # Use dynamic tree cut for automatic cluster detection
        result = cutreeHybrid(
            Z,
            dist_matrix,
            minClusterSize=min_cluster_size,
            deepSplit=deep_split,
        )
        cluster_labels = result["labels"]
    else:
        # Fallback: use scipy fcluster with automatic threshold
        from scipy.cluster.hierarchy import inconsistent
        # Use inconsistency coefficient to determine cut
        max_d = 0.7 * max(Z[:, 2])  # Cut at 70% of max distance
        cluster_labels = fcluster(Z, max_d, criterion='distance')

    return cluster_labels, Z, data_pca, pca


def create_channel_heatmap_with_clustering(zscore_df, summary_df, channel_name, output_dir):
    """Create heatmap with hierarchical clustering for a specific channel."""
    channel_info = CHANNELS[channel_name]
    auroc_col = channel_info["auroc_col"]
    channel_color = channel_info["color"]

    # Filter PIGN variants
    pign_summary = filter_pign_variants(summary_df)

    if auroc_col not in pign_summary.columns:
        print(f"AUROC column {auroc_col} not found")
        return None

    # Get variants with AUROC > 0.9 for this channel
    sig_variants = pign_summary[pign_summary[auroc_col] > AUROC_THRESHOLD]["allele_0"].tolist()

    if len(sig_variants) < 3:
        print(f"Not enough significant variants for {channel_name} clustering (need >= 3)")
        return None

    print(f"{channel_name}: {len(sig_variants)} variants with AUROC > {AUROC_THRESHOLD}")

    # Get channel-specific features
    channel_features = get_channel_features(zscore_df, channel_name)

    if not channel_features:
        print(f"No features found for {channel_name}")
        return None

    # Filter zscore data for significant variants
    zscore_filtered = zscore_df[zscore_df["variant"].isin(sig_variants)].copy()

    if zscore_filtered.empty or len(zscore_filtered) < 3:
        print(f"Not enough Z-score data for {channel_name} clustering")
        return None

    # Select available features
    available_features = [f for f in channel_features if f in zscore_filtered.columns]
    if len(available_features) < 5:
        print(f"Not enough overlapping features for {channel_name}")
        return None

    # Feature selection - remove low variance and highly correlated features
    feature_data = zscore_filtered.set_index("variant")[available_features].copy()

    # Remove features with zero variance
    feature_vars = feature_data.var()
    feature_data = feature_data.loc[:, feature_vars > 0.01]

    if feature_data.shape[1] < 5:
        print(f"Not enough variable features for {channel_name}")
        return None

    # Select top features by variance for visualization
    n_features = min(40, feature_data.shape[1])
    top_features = feature_vars[feature_data.columns].nlargest(n_features).index.tolist()

    # Prepare data matrix for clustering (use all available features)
    data_matrix = feature_data.values

    # Handle NaN values
    data_matrix = np.nan_to_num(data_matrix, nan=0.0)

    # Perform clustering
    min_cluster_size = max(3, len(sig_variants) // 10)
    print(f"  Clustering with min_cluster_size={min_cluster_size}")

    try:
        cluster_labels, Z, data_pca, pca = perform_clustering(
            data_matrix,
            min_cluster_size=min_cluster_size,
            deep_split=2
        )
    except Exception as e:
        print(f"  Clustering failed: {e}")
        return None

    n_clusters = len(set(cluster_labels))
    print(f"  Found {n_clusters} clusters")

    # Create cluster color mapping
    colors = sns.color_palette("husl", n_clusters)
    unique_labels = sorted(np.unique(cluster_labels))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Get AUROC values for column annotation
    auroc_dict = dict(zip(pign_summary["allele_0"], pign_summary[auroc_col]))

    # Prepare heatmap data with top features
    heatmap_data = feature_data[top_features].T

    # Row colors for clusters
    row_colors_list = [color_map[c] for c in cluster_labels]

    # Column colors for AUROC
    cmap_auroc = plt.cm.RdYlGn
    norm_auroc = Normalize(vmin=AUROC_THRESHOLD, vmax=1.0)
    col_colors = [cmap_auroc(norm_auroc(auroc_dict.get(v, AUROC_THRESHOLD))) for v in heatmap_data.columns]

    # Feature correlation linkage for column ordering
    if heatmap_data.shape[0] > 1:
        corr_df = heatmap_data.T.corr()
        corr_dist = 1 - corr_df.values
        np.fill_diagonal(corr_dist, 0)
        corr_dist = np.clip(corr_dist, 0, 2)  # Ensure valid distances
        corr_dist_condensed = squareform(corr_dist)
        try:
            col_linkage = linkage(corr_dist_condensed, method='average', optimal_ordering=True)
        except:
            col_linkage = None
    else:
        col_linkage = None

    # Create clustermap
    fig_height = max(10, len(top_features) * 0.4)
    fig_width = max(10, len(heatmap_data.columns) * 0.5)

    g = sns.clustermap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        col_colors=[row_colors_list],  # Variant cluster colors
        row_colors=None,
        row_linkage=col_linkage,
        col_linkage=Z,  # Use variant clustering linkage
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(0.1, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.12),
        xticklabels=True,
        yticklabels=True,
        linewidths=0.05,
    )

    g.ax_heatmap.set_xlabel(f"PIGN Variants (clustered, AUROC > {AUROC_THRESHOLD})", fontsize=11)
    g.ax_heatmap.set_ylabel(f"{channel_name} Features", fontsize=11)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)

    # Add cluster legend
    legend_handles = [Patch(facecolor=color_map[c], label=f'C{c} (n={sum(cluster_labels==c)})')
                      for c in sorted(color_map.keys())]
    g.ax_heatmap.legend(
        handles=legend_handles,
        title='Cluster',
        bbox_to_anchor=(-0.15, 1.0),
        loc='upper left',
        frameon=True,
        fontsize=8
    )

    # Add AUROC colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_auroc, norm=norm_auroc)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.02, 0.6, 0.03, 0.12])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{channel_name} AUROC", fontsize=9)

    g.fig.suptitle(
        f"{channel_name} Channel: Hierarchical Clustering of PIGN Variants\n"
        f"({len(sig_variants)} variants, {n_clusters} clusters, AUROC > {AUROC_THRESHOLD})",
        fontsize=12, y=1.02
    )

    # Save heatmap
    output_file = output_dir / f"PIGN_{channel_name}_clustered_heatmap.png"
    g.fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Clustered heatmap saved to: {output_file}")

    # Create PCA scatter plot
    if data_pca.shape[1] >= 2:
        fig_pca, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PC1 vs PC2
        for c in unique_labels:
            mask = cluster_labels == c
            axes[0].scatter(
                data_pca[mask, 0],
                data_pca[mask, 1],
                c=[color_map[c]],
                label=f'C{c} (n={mask.sum()})',
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidth=0.5
            )
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0].set_title(f'{channel_name}: PC1 vs PC2')
        axes[0].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=9, title="Cluster")
        axes[0].grid(alpha=0.2)

        # PC3 vs PC4 if available
        if data_pca.shape[1] >= 4:
            for c in unique_labels:
                mask = cluster_labels == c
                axes[1].scatter(
                    data_pca[mask, 2],
                    data_pca[mask, 3],
                    c=[color_map[c]],
                    label=f'C{c}',
                    alpha=0.7,
                    s=60,
                    edgecolors='white',
                    linewidth=0.5
                )
            axes[1].set_xlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            axes[1].set_ylabel(f'PC4 ({pca.explained_variance_ratio_[3]:.1%})')
            axes[1].set_title(f'{channel_name}: PC3 vs PC4')
            axes[1].grid(alpha=0.2)
        else:
            axes[1].text(0.5, 0.5, f'Only {data_pca.shape[1]} PCs available',
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('PC3 vs PC4 (unavailable)')

        plt.tight_layout()
        pca_file = output_dir / f"PIGN_{channel_name}_cluster_PCA.png"
        fig_pca.savefig(pca_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  PCA plot saved to: {pca_file}")

    # Return cluster assignments
    cluster_df = pd.DataFrame({
        'variant': feature_data.index,
        f'{channel_name}_cluster': cluster_labels,
        f'{channel_name}_AUROC': [auroc_dict.get(v, np.nan) for v in feature_data.index]
    })

    return cluster_df


def create_summary_heatmap(zscore_df, summary_df, output_dir):
    """Create a summary heatmap showing top features across all significant variants."""
    # Get variants significant in at least one channel
    pign_summary = filter_pign_variants(summary_df)

    all_sig_variants = set()
    for channel_name, channel_info in CHANNELS.items():
        auroc_col = channel_info["auroc_col"]
        if auroc_col in pign_summary.columns:
            sig = pign_summary[pign_summary[auroc_col] > AUROC_THRESHOLD]["allele_0"].tolist()
            all_sig_variants.update(sig)

    if not all_sig_variants:
        print("No significant variants found across any channel")
        return

    print(f"\nCreating summary heatmap with {len(all_sig_variants)} variants significant in any channel")

    # Filter zscore data
    zscore_filtered = zscore_df[zscore_df["variant"].isin(all_sig_variants)].copy()

    if zscore_filtered.empty:
        print("No Z-score data available")
        return

    # Get feature columns
    feature_cols = [c for c in zscore_filtered.columns if c not in ["variant", "n_cells"]]

    # Select top features by absolute mean Z-score
    feature_importance = zscore_filtered[feature_cols].abs().mean()
    top_features = feature_importance.nlargest(50).index.tolist()

    # Create annotated feature labels showing channel
    def get_feature_channel(feat):
        for ch_name, ch_info in CHANNELS.items():
            for pattern in ch_info["feature_patterns"]:
                if pattern in feat:
                    return ch_name[0]  # First letter
        return "O"  # Other

    feature_labels = [f"{get_feature_channel(f)}:{f}" for f in top_features]

    # Prepare heatmap data
    heatmap_data = zscore_filtered.set_index("variant")[top_features].T
    heatmap_data.index = feature_labels

    # Calculate mean AUROC for sorting
    auroc_cols = [ch["auroc_col"] for ch in CHANNELS.values() if ch["auroc_col"] in pign_summary.columns]
    pign_summary["AUROC_Mean_All"] = pign_summary[auroc_cols].mean(axis=1)
    auroc_dict = dict(zip(pign_summary["allele_0"], pign_summary["AUROC_Mean_All"]))

    # Sort columns by mean AUROC
    sorted_cols = sorted(heatmap_data.columns, key=lambda x: auroc_dict.get(x, 0), reverse=True)
    heatmap_data = heatmap_data[sorted_cols]

    # Column colors by mean AUROC
    col_colors = []
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=AUROC_THRESHOLD, vmax=1.0)

    for variant in heatmap_data.columns:
        auroc = auroc_dict.get(variant, AUROC_THRESHOLD)
        col_colors.append(cmap(norm(auroc)))

    # Row colors by channel
    channel_colors = {"D": "#1f77b4", "A": "#ff7f0e", "P": "#2ca02c", "M": "#d62728", "O": "#7f7f7f"}
    row_colors = [channel_colors.get(label[0], "#7f7f7f") for label in feature_labels]

    # Create clustermap
    fig_height = max(12, len(top_features) * 0.35)
    fig_width = max(10, len(heatmap_data.columns) * 0.5)

    g = sns.clustermap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        col_colors=[col_colors],
        row_colors=[row_colors],
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(0.1, 0.12),
        cbar_pos=(0.02, 0.8, 0.03, 0.12),
        xticklabels=True,
        yticklabels=True,
        col_cluster=True,
        row_cluster=True,
    )

    g.ax_heatmap.set_xlabel("PIGN Variants (significant in any channel)", fontsize=11)
    g.ax_heatmap.set_ylabel("Top Features (prefix: D=DNA, A=AGP, P=Protein, M=Morph)", fontsize=10)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)

    # Add AUROC colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.02, 0.6, 0.03, 0.12])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Mean AUROC", fontsize=9)

    g.fig.suptitle(
        f"PIGN Variants: Top 50 Features Z-scores\n"
        f"({len(all_sig_variants)} variants with AUROC > {AUROC_THRESHOLD} in at least one channel)",
        fontsize=12, y=1.02
    )

    # Save
    output_file = output_dir / "PIGN_all_channels_zscore_heatmap.png"
    g.fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary heatmap saved to: {output_file}")


def main():
    """Main analysis function."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dir = OUTPUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PIGN Variant Per-Channel Analysis")
    print(f"AUROC threshold: {AUROC_THRESHOLD}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    summary_df = load_classification_summary()
    zscore_df = load_zscore_data()

    if summary_df.empty:
        print("No classification summary found")
        return

    if zscore_df.empty:
        print("No Z-score data found - heatmaps will not be generated")

    # Generate per-channel report
    print("\nGenerating per-channel report...")
    report_path = OUTPUT_DIR / f"{GENE}_per_channel_report.txt"
    pign_df = generate_channel_report(summary_df, report_path)

    # Generate channel-specific heatmaps with clustering
    if not zscore_df.empty:
        print("\nGenerating channel-specific heatmaps with hierarchical clustering...")

        all_cluster_dfs = []

        for channel_name in CHANNELS.keys():
            print(f"\nProcessing {channel_name}...")

            # Create heatmap with clustering
            cluster_df = create_channel_heatmap_with_clustering(
                zscore_df, summary_df, channel_name, heatmap_dir
            )

            if cluster_df is not None:
                all_cluster_dfs.append(cluster_df)

            # Also create the original simple heatmap
            create_channel_heatmap(zscore_df, summary_df, channel_name, heatmap_dir)

        # Merge all cluster assignments
        if all_cluster_dfs:
            print("\nMerging cluster assignments across channels...")
            merged_clusters = all_cluster_dfs[0]
            for df in all_cluster_dfs[1:]:
                merged_clusters = merged_clusters.merge(df, on='variant', how='outer')

            # Save cluster assignments
            cluster_output = OUTPUT_DIR / "PIGN_channel_cluster_labels.csv"
            merged_clusters.to_csv(cluster_output, index=False)
            print(f"Cluster labels saved to: {cluster_output}")

            # Generate cluster summary report
            generate_cluster_summary(merged_clusters, OUTPUT_DIR)

        # Generate summary heatmap
        print("\nGenerating summary heatmap...")
        create_summary_heatmap(zscore_df, summary_df, heatmap_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


def generate_cluster_summary(cluster_df, output_dir):
    """Generate a summary report of cluster assignments per channel."""
    report_path = output_dir / "PIGN_cluster_summary.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PIGN VARIANT CLUSTERING SUMMARY PER CHANNEL\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total variants with cluster assignments: {len(cluster_df)}\n\n")

        for channel in CHANNELS.keys():
            cluster_col = f"{channel}_cluster"
            auroc_col = f"{channel}_AUROC"

            if cluster_col not in cluster_df.columns:
                continue

            channel_data = cluster_df.dropna(subset=[cluster_col])

            f.write("-" * 80 + "\n")
            f.write(f"{channel.upper()} CHANNEL CLUSTERS\n")
            f.write("-" * 80 + "\n")

            n_clusters = int(channel_data[cluster_col].max())
            f.write(f"Number of clusters: {n_clusters}\n\n")

            for cluster_id in sorted(channel_data[cluster_col].unique()):
                cluster_variants = channel_data[channel_data[cluster_col] == cluster_id]
                f.write(f"Cluster {int(cluster_id)} ({len(cluster_variants)} variants):\n")

                # Sort by AUROC within cluster
                if auroc_col in cluster_variants.columns:
                    cluster_variants = cluster_variants.sort_values(auroc_col, ascending=False)

                for _, row in cluster_variants.iterrows():
                    auroc_val = row.get(auroc_col, np.nan)
                    auroc_str = f"{auroc_val:.4f}" if not np.isnan(auroc_val) else "N/A"
                    f.write(f"  - {row['variant']:<35} AUROC: {auroc_str}\n")
                f.write("\n")

            f.write("\n")

    print(f"Cluster summary saved to: {report_path}")


if __name__ == "__main__":
    main()
