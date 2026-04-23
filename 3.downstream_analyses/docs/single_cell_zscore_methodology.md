# Single-Cell Z-Score Analysis for Protein Abundance Change

## Overview

This document describes the single-cell Z-score methodology for detecting protein abundance changes in imaging data, adapted from the DUAL-IPA FACS-based approach.

## Motivation

The previous well-aggregated paired t-test approach:
- Aggregates single-cell GFP intensity to median per well
- Loses cellular heterogeneity information
- Requires paired ref-var wells on same plate

The single-cell Z-score approach:
- Preserves cell-level resolution
- Normalizes by reference population variability
- More directly comparable to DUAL-IPA methodology

## Statistical Framework

### Step 1: Background Handling

CellProfiler GFP intensities (`Cells_Intensity_MeanIntensity_GFP`) are already normalized to 0-1 scale. We do NOT subtract raw image background. Instead:
- Filter out wells flagged as `is_bg=True` (background-only wells from image QC)
- Z-score methodology inherently handles background by comparing variant to reference cells on the same plate

### Step 2: Reference Distribution (Per Gene, Per Plate)

For each reference allele on plate P:
```
1. Pool all single cells from reference wells on plate P
2. Compute mean reference intensity: μ_ref = mean(GFP_intensity)
3. Compute log2FC for each ref cell: log2FC = log2(cell_intensity / μ_ref)
4. Calculate distribution parameters:
   - μ_WT = mean(log2FC)  # should be ~0
   - σ_WT = std(log2FC)
```

### Step 3: Variant Z-Scores

For each variant cell on plate P:
```
Z_cell = (log2(variant_cell_intensity / μ_ref) - μ_WT) / σ_WT
```

### Step 4: Hierarchical Aggregation

```
1. Per-plate: mean_zscore = mean(Z_cell) for all variant cells on plate
2. Per-bio-batch: final_zscore = median(mean_zscore) across plates
3. Per-variant: sc_zscore = median(final_zscore) across bio batches
```

## QC Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| MIN_REF_CELLS | 20 | Minimum reference cells per gene-plate for stats |
| MIN_REF_WELLS_PER_PLATE | 1 | Minimum reference wells per gene-plate |
| Min GFP threshold | 1e-6 | Filter segmentation artifacts |
| Background wells | Exclude `is_bg=True` | Signal quality |

**Note:** Stricter thresholds (200 cells, 2 wells) dramatically reduce coverage. Relaxed thresholds maintain statistical validity while maximizing variant coverage.

## Data Sources

| Data | Source | Key Columns |
|------|--------|-------------|
| Single-cell profiles | `2.snakemake_pipeline/outputs/batch_profiles/{batch}/profiles.parquet` | `Cells_Intensity_MeanIntensity_GFP`, `Metadata_gene_allele` |
| Well QC flags | `3.downstream_analyses/outputs/0.img_metadata_qc/img_well_qc_sum_df.parquet` | `is_bg`, `channel=="GFP"` |

## Outputs

| File | Description |
|------|-------------|
| `prot_abundance_singlecell_zscore.csv` | Per-variant, per-bio-batch Z-scores |

Output schema:
- `gene_variant`: Variant identifier (e.g., "AGXT_Arg36Cys")
- `Gene`: Gene symbol
- `bio_batch`: Biological batch (e.g., "B_13-14")
- `final_zscore`: Median of plate-mean Z-scores
- `zscore_plate_variability`: Std across plates
- `median_log2FC`: Median log2 fold-change
- `n_plates`: Number of plates analyzed
- `total_cells`: Total variant cells analyzed

## Validation Results

### Correlation with DUAL-IPA

| Metric | Value |
|--------|-------|
| Overlapping variants | 314 |
| Pearson r | 0.347 (p = 2.6e-10) |
| Spearman ρ | 0.339 (p = 7.4e-10) |
| Direction concordance | 57.6% |

### Comparison with Paired T-test Method

| Metric | Value |
|--------|-------|
| Overlapping variants | 1,296 |
| Pearson r | 0.835 (p < 1e-300) |

The high correlation with the existing T-test method confirms both capture similar biological signal. The moderate correlation with DUAL-IPA (~0.35) reflects expected differences between imaging and FACS assays.

## Implementation

Notebook: `3.downstream_analyses/scripts/2b_protein_abundance_singlecell_zscore.ipynb`

Key functions:
- `compute_ref_stats_for_gene_plate()`: Calculate reference distribution parameters
- `compute_variant_zscores()`: Calculate cell-level Z-scores for variants

## Key Differences from DUAL-IPA

| Aspect | DUAL-IPA (FACS) | Imaging Adaptation |
|--------|-----------------|-------------------|
| Measurement | GFP:mCherry ratio | GFP MeanIntensity |
| Normalization | mCherry internal control | CellProfiler area normalization |
| Background | Empty well threshold | `is_bg` flag filtering |
| Cell count | ~800 cells/well threshold | Relaxed to 20 cells min |

## References

- DUAL-IPA methodology: Bhattacharyya et al. (FACS-based variant effect mapping)
- CellProfiler intensity features: `MeanIntensity` = sum(intensity) / cell_area
