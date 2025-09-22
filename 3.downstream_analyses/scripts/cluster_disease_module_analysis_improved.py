#!/usr/bin/env python3
"""
Improved Cluster and Disease Module Analysis

This script analyzes mislocalized variant clusters with enhanced visualizations:
1. Full CellProfiler feature names in heatmaps
2. Better color mapping and statistical annotations
3. Top 5 disease modules per cluster analysis
4. Clean, comprehensive output

Author: Claude Code
Date: 2025-09-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedClusterDiseaseModuleAnalyzer:
    """
    Enhanced analyzer with improved visualizations and comprehensive disease module analysis
    """

    def __init__(self):
        self.cluster_df = None
        self.disease_modules_df = None
        self.feature_columns = []
        self.metadata_columns = []
        self.cluster_stats = {}
        self.disease_module_mapping = {}

    def load_data(self):
        """Load cluster data and disease modules"""
        logger.info("Loading cluster data...")
        self.cluster_df = pd.read_csv('../outputs/mislocalized_variant_cluster.csv')

        logger.info("Loading disease modules...")
        # Only use the original disease_modules.csv file as requested
        self.disease_modules_df = pd.read_csv('../inputs/disease_modules.csv')

        # Identify feature and metadata columns
        self.metadata_columns = [col for col in self.cluster_df.columns if 'Metadata' in col]
        self.feature_columns = [col for col in self.cluster_df.columns
                               if 'Metadata' not in col and col != 'Unnamed: 0']

        logger.info(f"Loaded {len(self.cluster_df)} variants across {len(self.cluster_df['Metadata_cluster'].unique())} clusters")
        logger.info(f"Found {len(self.feature_columns)} feature columns")

    def extract_gene_from_variant(self, variant_name):
        """Extract gene symbol from variant name"""
        if pd.isna(variant_name):
            return None
        return str(variant_name).split('_')[0]

    def perform_cluster_feature_analysis(self):
        """
        Perform Mann-Whitney U tests for each cluster vs rest
        """
        logger.info("Performing cluster feature analysis...")

        clusters = sorted(self.cluster_df['Metadata_cluster'].unique())
        self.cluster_stats = {}

        for cluster_id in clusters:
            logger.info(f"Analyzing cluster {cluster_id}")

            # Split data: cluster vs rest
            cluster_mask = self.cluster_df['Metadata_cluster'] == cluster_id
            cluster_data = self.cluster_df[cluster_mask][self.feature_columns]
            rest_data = self.cluster_df[~cluster_mask][self.feature_columns]

            # Storage for results
            feature_stats = []

            for feature in self.feature_columns:
                try:
                    # Get feature values for cluster and rest
                    cluster_values = cluster_data[feature].dropna()
                    rest_values = rest_data[feature].dropna()

                    if len(cluster_values) > 0 and len(rest_values) > 0:
                        # Mann-Whitney U test
                        statistic, p_value = stats.mannwhitneyu(
                            cluster_values, rest_values,
                            alternative='two-sided'
                        )

                        # Effect size (Cohen's d approximation)
                        mean_cluster = cluster_values.mean()
                        mean_rest = rest_values.mean()
                        pooled_std = np.sqrt(((cluster_values.std()**2 * (len(cluster_values)-1)) +
                                            (rest_values.std()**2 * (len(rest_values)-1))) /
                                           (len(cluster_values) + len(rest_values) - 2))

                        cohens_d = (mean_cluster - mean_rest) / pooled_std if pooled_std > 0 else 0

                        feature_stats.append({
                            'feature': feature,
                            'cluster_mean': mean_cluster,
                            'rest_mean': mean_rest,
                            'fold_change': mean_cluster - mean_rest,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'cluster_size': len(cluster_values),
                            'rest_size': len(rest_values)
                        })

                except Exception as e:
                    logger.warning(f"Failed to analyze feature {feature} for cluster {cluster_id}: {e}")

            # Convert to DataFrame and apply multiple testing correction
            if feature_stats:
                stats_df = pd.DataFrame(feature_stats)

                # Multiple testing correction
                rejected, pvals_corrected, _, _ = multipletests(
                    stats_df['p_value'],
                    alpha=0.05,
                    method='fdr_bh'
                )

                stats_df['p_adj'] = pvals_corrected
                stats_df['significant'] = rejected
                stats_df['neg_log10_padj'] = -np.log10(stats_df['p_adj'] + 1e-100)

                # Sort by adjusted p-value
                stats_df = stats_df.sort_values('p_adj')

                self.cluster_stats[cluster_id] = stats_df

                logger.info(f"Cluster {cluster_id}: {rejected.sum()} significant features out of {len(feature_stats)}")

    def classify_disease_by_keywords(self, diseases_text):
        """Classify diseases based on keywords"""
        if pd.isna(diseases_text):
            return []

        disease_text = str(diseases_text).lower()

        # Enhanced disease classification
        disease_classifications = []

        if any(word in disease_text for word in ['gaucher', 'lysosomal', 'storage']):
            disease_classifications.append(('Lysosomal_Storage_Disorders', 0.9))
        if any(word in disease_text for word in ['cardiomyopathy', 'cardiac', 'heart']):
            disease_classifications.append(('Cardiomyopathy', 0.9))
        if any(word in disease_text for word in ['cancer', 'tumor', 'carcinoma', 'breast', 'ovarian']):
            disease_classifications.append(('Cancer_Syndromes', 0.8))
        if any(word in disease_text for word in ['neuropathy', 'neural', 'nerve']):
            disease_classifications.append(('Neuropathy', 0.8))
        if any(word in disease_text for word in ['metabolic', 'enzyme', 'deficiency']):
            disease_classifications.append(('Metabolic_Disorders', 0.7))
        if any(word in disease_text for word in ['coagulation', 'bleeding', 'hemophilia']):
            disease_classifications.append(('Coagulation_Disorders', 0.9))
        if any(word in disease_text for word in ['immunodeficiency', 'immune']):
            disease_classifications.append(('Immunodeficiency', 0.8))

        return disease_classifications if disease_classifications else [('Other_Genetic_Disorders', 0.5)]

    def map_disease_modules_to_clusters(self):
        """
        Enhanced disease module mapping with top 5 modules per cluster
        """
        logger.info("Mapping disease modules to clusters...")

        self.disease_module_mapping = {}

        for cluster_id in sorted(self.cluster_df['Metadata_cluster'].unique()):
            cluster_variants = self.cluster_df[self.cluster_df['Metadata_cluster'] == cluster_id]

            # Extract gene-allele combinations from this cluster
            cluster_gene_alleles = []
            cluster_diseases = []

            for idx, row in cluster_variants.iterrows():
                if 'Unnamed: 0' in row and pd.notna(row['Unnamed: 0']):
                    variant_name = row['Unnamed: 0']
                    cluster_gene_alleles.append(variant_name)

                # Also get disease names from the cluster data
                if 'Metadata_OMIM_disease_names' in row and pd.notna(row['Metadata_OMIM_disease_names']):
                    cluster_diseases.append(row['Metadata_OMIM_disease_names'])

            # Look up disease modules from original disease_modules.csv
            all_disease_classifications = []

            for gene_allele in cluster_gene_alleles:
                # Find matches in disease_modules.csv
                matches = self.disease_modules_df[self.disease_modules_df['gene_allele'] == gene_allele]

                if not matches.empty:
                    for _, match in matches.iterrows():
                        if pd.notna(match.get('OMIM_disease_names')):
                            disease_classifications = self.classify_disease_by_keywords(match['OMIM_disease_names'])
                            all_disease_classifications.extend(disease_classifications)

            # Also classify diseases from cluster metadata
            for disease_text in cluster_diseases:
                disease_classifications = self.classify_disease_by_keywords(disease_text)
                all_disease_classifications.extend(disease_classifications)

            # Aggregate disease module scores
            module_scores = defaultdict(list)
            for module, score in all_disease_classifications:
                module_scores[module].append(score)

            # Calculate average scores and get top 5
            module_avg_scores = {module: np.mean(scores) for module, scores in module_scores.items()}

            # Sort by score and get top 5
            top_5_modules = sorted(module_avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            self.disease_module_mapping[cluster_id] = {
                'cluster_size': len(cluster_variants),
                'gene_alleles': cluster_gene_alleles,
                'top_5_modules': top_5_modules,
                'primary_module': top_5_modules[0][0] if top_5_modules else 'Unknown',
                'primary_confidence': top_5_modules[0][1] if top_5_modules else 0.0,
                'module_diversity': len(module_scores)
            }

            logger.info(f"Cluster {cluster_id}: {len(cluster_gene_alleles)} variants, "
                       f"primary module: {self.disease_module_mapping[cluster_id]['primary_module']} "
                       f"(conf: {self.disease_module_mapping[cluster_id]['primary_confidence']:.3f})")

    def create_enhanced_heatmaps(self):
        """
        Create enhanced heatmaps with full feature names and better visualizations
        """
        logger.info("Creating enhanced heatmaps...")

        clusters = sorted(self.cluster_stats.keys())

        # Get top 5 features per cluster with full names
        all_top_features = set()
        cluster_feature_data = {}

        for cluster_id in clusters:
            top_features = self.cluster_stats[cluster_id].head(5)
            features = top_features['feature'].tolist()
            p_adj_values = top_features['p_adj'].tolist()
            neg_log10_values = top_features['neg_log10_padj'].tolist()

            all_top_features.update(features)
            cluster_feature_data[cluster_id] = {
                'features': features,
                'p_adj': p_adj_values,
                'neg_log10': neg_log10_values
            }

        # Create feature heatmap matrix
        feature_list = sorted(list(all_top_features))
        n_features = len(feature_list)
        n_clusters = len(clusters)

        # Matrix for -log10(p_adj) values
        heatmap_data = np.zeros((n_features, n_clusters))

        for i, feature in enumerate(feature_list):
            for j, cluster_id in enumerate(clusters):
                if feature in cluster_feature_data[cluster_id]['features']:
                    idx = cluster_feature_data[cluster_id]['features'].index(feature)
                    heatmap_data[i, j] = cluster_feature_data[cluster_id]['neg_log10'][idx]

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 24))
        gs = fig.add_gridspec(3, 1, height_ratios=[5, 2, 1], hspace=0.3)

        # 1. Feature significance heatmap
        ax1 = fig.add_subplot(gs[0])

        # Use a better colormap where darker = more significant
        im1 = ax1.imshow(heatmap_data, cmap='Blues', aspect='auto', vmin=0)

        # Set labels
        ax1.set_xticks(range(n_clusters))
        ax1.set_xticklabels([f'Cluster {c}' for c in clusters], fontsize=12)
        ax1.set_yticks(range(n_features))
        ax1.set_yticklabels(feature_list, fontsize=8)  # Full feature names
        ax1.set_title('Top Significantly Different Features per Cluster\n(-log₁₀(adjusted p-value))',
                      fontsize=16, fontweight='bold', pad=20)

        # Add significance annotations
        for i in range(n_features):
            for j in range(n_clusters):
                if heatmap_data[i, j] > 0:
                    # Get actual p_adj value
                    feature = feature_list[i]
                    cluster_id = clusters[j]
                    if feature in cluster_feature_data[cluster_id]['features']:
                        idx = cluster_feature_data[cluster_id]['features'].index(feature)
                        p_adj = cluster_feature_data[cluster_id]['p_adj'][idx]

                        # Add significance stars
                        stars = ''
                        if p_adj < 0.0001:
                            stars = '***'
                        elif p_adj < 0.001:
                            stars = '**'
                        elif p_adj < 0.01:
                            stars = '*'

                        if stars:
                            ax1.text(j, i, stars, ha='center', va='center',
                                   fontsize=10, fontweight='bold', color='red')

        # Add colorbar with scientific notation
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
        cbar1.set_label('-log₁₀(adjusted p-value)', fontsize=12)

        # 2. Top 5 Disease Modules Heatmap
        ax2 = fig.add_subplot(gs[1])

        # Prepare disease module data (top 5 per cluster)
        max_modules = 5
        module_matrix = np.zeros((max_modules, n_clusters))
        module_labels = [[''] * n_clusters for _ in range(max_modules)]

        for j, cluster_id in enumerate(clusters):
            cluster_info = self.disease_module_mapping.get(cluster_id, {})
            top_modules = cluster_info.get('top_5_modules', [])

            for i, (module, confidence) in enumerate(top_modules[:max_modules]):
                module_matrix[i, j] = confidence
                # Shorten module names for display
                short_module = module.replace('_', ' ').replace('Disorders', 'Dis.')
                if len(short_module) > 15:
                    short_module = short_module[:12] + '...'
                module_labels[i][j] = f"{short_module}\n{confidence:.2f}"

        # Create heatmap
        im2 = ax2.imshow(module_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)

        ax2.set_xticks(range(n_clusters))
        ax2.set_xticklabels([f'Cluster {c}' for c in clusters], fontsize=12)
        ax2.set_yticks(range(max_modules))
        ax2.set_yticklabels([f'Module {i+1}' for i in range(max_modules)], fontsize=10)
        ax2.set_title('Top 5 Disease Modules per Cluster (with confidence scores)',
                      fontsize=14, fontweight='bold')

        # Add module labels with confidence scores
        for i in range(max_modules):
            for j in range(n_clusters):
                if module_matrix[i, j] > 0:
                    ax2.text(j, i, module_labels[i][j], ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color='white' if module_matrix[i, j] > 0.5 else 'black')

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cbar2.set_label('Module Confidence Score', fontsize=12)

        # 3. Cluster summary stats
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')

        # Summary text
        summary_text = "Cluster Summary:\n"
        for cluster_id in clusters:
            cluster_info = self.disease_module_mapping.get(cluster_id, {})
            n_vars = cluster_info.get('cluster_size', 0)
            primary = cluster_info.get('primary_module', 'Unknown')
            conf = cluster_info.get('primary_confidence', 0)
            summary_text += f"C{cluster_id}: {n_vars} variants, {primary} ({conf:.2f})  "
            if (cluster_id + 1) % 3 == 0:  # New line every 3 clusters
                summary_text += "\n"

        ax3.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10,
                transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        plt.savefig('../outputs/enhanced_cluster_disease_module_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig('../outputs/enhanced_cluster_disease_module_heatmap.pdf', bbox_inches='tight')
        logger.info("Enhanced heatmap saved")

    def save_clean_results(self):
        """
        Save clean, consolidated results to a single comprehensive file
        """
        logger.info("Saving clean, consolidated results...")

        # Consolidate all results into a single comprehensive dataframe
        consolidated_results = []

        for cluster_id in sorted(self.cluster_stats.keys()):
            cluster_info = self.disease_module_mapping.get(cluster_id, {})
            stats_df = self.cluster_stats[cluster_id]

            # Get top 5 features for this cluster
            top_features = stats_df.head(5)

            for idx, (_, feature_row) in enumerate(top_features.iterrows()):
                # Get top 5 disease modules
                top_modules = cluster_info.get('top_5_modules', [])

                for module_rank, (module, module_conf) in enumerate(top_modules[:5]):
                    consolidated_results.append({
                        'cluster_id': cluster_id,
                        'cluster_size': cluster_info.get('cluster_size', 0),
                        'feature_rank': idx + 1,
                        'feature_name': feature_row['feature'],
                        'feature_p_adj': feature_row['p_adj'],
                        'feature_neg_log10_padj': feature_row['neg_log10_padj'],
                        'feature_fold_change': feature_row['fold_change'],
                        'feature_cohens_d': feature_row['cohens_d'],
                        'feature_significant': feature_row['significant'],
                        'module_rank': module_rank + 1,
                        'disease_module': module,
                        'module_confidence': module_conf,
                        'is_primary_module': (module_rank == 0)
                    })

        # Create comprehensive results dataframe
        results_df = pd.DataFrame(consolidated_results)

        # Save to single clean file
        output_file = '../outputs/cluster_disease_module_analysis_comprehensive.csv'
        results_df.to_csv(output_file, index=False)

        logger.info(f"Comprehensive results saved to {output_file}")

        # Generate summary statistics
        summary_stats = {
            'total_variants': len(self.cluster_df),
            'total_clusters': len(self.cluster_stats),
            'total_features_tested': len(self.feature_columns),
            'clusters_with_significant_features': sum(1 for stats in self.cluster_stats.values() if stats['significant'].any()),
            'total_significant_features': sum(stats['significant'].sum() for stats in self.cluster_stats.values())
        }

        # Save summary
        with open('../outputs/analysis_summary_final.txt', 'w') as f:
            f.write("VarChAMP Cluster-Disease Module Analysis - Final Summary\n")
            f.write("=" * 60 + "\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        return results_df

def main():
    """
    Main execution function
    """
    analyzer = ImprovedClusterDiseaseModuleAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Perform analyses
        analyzer.perform_cluster_feature_analysis()
        analyzer.map_disease_modules_to_clusters()

        # Create enhanced visualizations
        analyzer.create_enhanced_heatmaps()

        # Save clean results
        results_df = analyzer.save_clean_results()

        logger.info("Enhanced analysis completed successfully!")

        # Print summary
        print(f"\nEnhanced Analysis Results:")
        print(f"- {len(analyzer.cluster_df)} variants across {len(analyzer.cluster_stats)} clusters")
        print(f"- {len(analyzer.feature_columns)} morphological features tested")
        print(f"- Enhanced heatmaps with full feature names and significance annotations")
        print(f"- Top 5 disease modules per cluster with confidence scores")
        print(f"- Results consolidated into comprehensive output file")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()