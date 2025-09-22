#!/usr/bin/env python3
"""
Final Disease Module Analysis Script

This script provides a clean, comprehensive analysis of disease modules
using ONLY the original disease_modules.csv file.

Features:
- Clean disease name standardization
- Keyword-based disease classification
- Single comprehensive output file
- Clear, reproducible methodology

Author: Claude Code
Date: 2025-09-21
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalDiseaseModuleAnalyzer:
    """
    Clean, final disease module analyzer using only the original disease_modules.csv
    """

    def __init__(self):
        self.disease_modules_df = None

    def load_data(self):
        """Load the original disease modules file"""
        logger.info("Loading disease modules data...")
        self.disease_modules_df = pd.read_csv('../inputs/disease_modules.csv')
        logger.info(f"Loaded {len(self.disease_modules_df)} variants")

    def clean_disease_names(self, disease_text):
        """Clean and standardize disease names"""
        if pd.isna(disease_text) or disease_text == '':
            return []

        # Clean formatting
        cleaned = re.sub(r'["\'\[\]]', '', str(disease_text))
        diseases = re.split(r'[;|]\s*', cleaned)

        # Clean individual names
        cleaned_diseases = []
        for disease in diseases:
            disease = disease.strip()
            if disease and disease.lower() not in ['not provided', 'not specified', '']:
                disease = ' '.join(disease.split())
                disease = re.sub(r'[,\.]+$', '', disease)
                cleaned_diseases.append(disease)

        return list(set(cleaned_diseases))

    def classify_disease_modules(self, diseases):
        """Classify diseases into biological modules"""
        if not diseases:
            return [('Unclassified', 0.0)]

        disease_text = ' '.join(diseases).lower()

        # Disease classification rules with confidence scores
        classifications = []

        # High confidence classifications (0.9)
        if any(word in disease_text for word in ['gaucher', 'pompe', 'fabry', 'niemann', 'lysosomal']):
            classifications.append(('Lysosomal_Storage_Disorders', 0.9))

        if any(word in disease_text for word in ['hemophilia', 'bleeding', 'coagulation', 'factor ix', 'factor viii']):
            classifications.append(('Coagulation_Disorders', 0.9))

        if any(word in disease_text for word in ['arrhythmogenic', 'cardiomyopathy', 'dilated cardiomyopathy']):
            classifications.append(('Cardiomyopathy', 0.9))

        # Medium-high confidence (0.8)
        if any(word in disease_text for word in ['breast cancer', 'ovarian cancer', 'lynch syndrome', 'hereditary']):
            classifications.append(('Cancer_Syndromes', 0.8))

        if any(word in disease_text for word in ['neuropathy', 'charcot-marie', 'peripheral', 'neural']):
            classifications.append(('Neuropathy', 0.8))

        if any(word in disease_text for word in ['immunodeficiency', 'scid', 'immune deficiency']):
            classifications.append(('Immunodeficiency', 0.8))

        # Medium confidence (0.7)
        if any(word in disease_text for word in ['galactosemia', 'phenylketonuria', 'tyrosinemia', 'metabolic']):
            classifications.append(('Metabolic_Disorders', 0.7))

        if any(word in disease_text for word in ['muscular dystrophy', 'myopathy', 'muscle']):
            classifications.append(('Muscle_Disorders', 0.7))

        if any(word in disease_text for word in ['retinitis pigmentosa', 'leber', 'eye', 'retinal']):
            classifications.append(('Retinal_Disorders', 0.7))

        # Lower confidence (0.6)
        if any(word in disease_text for word in ['epilepsy', 'seizure', 'neurological']):
            classifications.append(('Neurological_Disorders', 0.6))

        if any(word in disease_text for word in ['connective tissue', 'collagen', 'marfan']):
            classifications.append(('Connective_Tissue_Disorders', 0.6))

        # Default classification
        if not classifications:
            classifications.append(('Other_Genetic_Disorders', 0.5))

        # Return highest confidence classification
        return sorted(classifications, key=lambda x: x[1], reverse=True)

    def process_disease_modules(self):
        """Main processing function"""
        logger.info("Processing disease modules...")

        results = []

        for idx, row in self.disease_modules_df.iterrows():
            gene_allele = row['gene_allele']
            gene_symbol = gene_allele.split('_')[0] if pd.notna(gene_allele) else 'Unknown'

            # Clean disease names
            diseases = self.clean_disease_names(row.get('OMIM_disease_names', ''))

            # Classify into disease modules
            classifications = self.classify_disease_modules(diseases)

            # Get primary and secondary classifications
            primary_module = classifications[0][0] if classifications else 'Unclassified'
            primary_confidence = classifications[0][1] if classifications else 0.0

            secondary_module = classifications[1][0] if len(classifications) > 1 else None
            secondary_confidence = classifications[1][1] if len(classifications) > 1 else None

            results.append({
                'gene_allele': gene_allele,
                'gene_symbol': gene_symbol,
                'cleaned_diseases': '; '.join(diseases),
                'disease_count': len(diseases),
                'primary_disease_module': primary_module,
                'primary_module_confidence': primary_confidence,
                'secondary_disease_module': secondary_module,
                'secondary_module_confidence': secondary_confidence,
                'total_classifications': len(classifications),
                'has_disease_annotation': len(diseases) > 0
            })

        return pd.DataFrame(results)

    def generate_summary_statistics(self, results_df):
        """Generate summary statistics"""
        logger.info("Generating summary statistics...")

        stats = {
            'total_variants': len(results_df),
            'variants_with_diseases': results_df['has_disease_annotation'].sum(),
            'unique_genes': results_df['gene_symbol'].nunique(),
            'disease_modules_identified': results_df['primary_disease_module'].nunique()
        }

        # Module distribution
        module_dist = results_df['primary_disease_module'].value_counts()

        # High confidence assignments
        high_conf = results_df[results_df['primary_module_confidence'] >= 0.8]

        summary_text = []
        summary_text.append("Disease Module Analysis - Final Summary")
        summary_text.append("=" * 50)
        summary_text.append("")
        summary_text.append(f"Total variants: {stats['total_variants']}")
        summary_text.append(f"Variants with disease annotations: {stats['variants_with_diseases']}")
        summary_text.append(f"Unique genes: {stats['unique_genes']}")
        summary_text.append(f"Disease modules identified: {stats['disease_modules_identified']}")
        summary_text.append(f"High confidence assignments (≥0.8): {len(high_conf)}")
        summary_text.append("")
        summary_text.append("Disease Module Distribution:")
        for module, count in module_dist.head(10).items():
            percentage = (count / len(results_df)) * 100
            summary_text.append(f"  {module}: {count} variants ({percentage:.1f}%)")
        summary_text.append("")
        summary_text.append("Top Genes by Disease Module:")
        for module in module_dist.head(5).index:
            if module != 'Unclassified':
                module_df = results_df[results_df['primary_disease_module'] == module]
                top_genes = module_df['gene_symbol'].value_counts().head(3)
                summary_text.append(f"  {module}:")
                for gene, count in top_genes.items():
                    avg_conf = module_df[module_df['gene_symbol'] == gene]['primary_module_confidence'].mean()
                    summary_text.append(f"    {gene}: {count} variants (avg conf: {avg_conf:.2f})")

        return '\n'.join(summary_text)

    def save_results(self, results_df):
        """Save final results"""
        logger.info("Saving final results...")

        # Save main results
        output_file = '../outputs/disease_modules_final_analysis.csv'
        results_df.to_csv(output_file, index=False)

        # Save summary
        summary = self.generate_summary_statistics(results_df)
        summary_file = '../outputs/disease_modules_final_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary)

        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary saved to {summary_file}")

        return output_file, summary_file

def main():
    """Main execution function"""
    analyzer = FinalDiseaseModuleAnalyzer()

    try:
        # Load and process data
        analyzer.load_data()
        results_df = analyzer.process_disease_modules()

        # Save results
        output_file, summary_file = analyzer.save_results(results_df)

        # Print summary statistics
        print(f"\nFinal Disease Module Analysis Complete:")
        print(f"- Processed {len(results_df)} variants")
        print(f"- {results_df['has_disease_annotation'].sum()} variants with disease annotations")
        print(f"- {results_df['primary_disease_module'].nunique()} disease modules identified")
        print(f"- Results saved to: {output_file}")
        print(f"- Summary saved to: {summary_file}")

        # Show top disease modules
        print(f"\nTop Disease Modules:")
        module_counts = results_df['primary_disease_module'].value_counts().head()
        for module, count in module_counts.items():
            print(f"  {module}: {count} variants")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()