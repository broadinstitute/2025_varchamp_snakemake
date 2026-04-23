# MislocScore Batch Normalization Analysis

**Date**: 2026-02-20
**Branch**: `feature/misloc-score-normalization`
**Status**: Investigation revealed fundamental issues - approach needs revision

## Problem Statement

The raw AUROC metric from VarChAMP classifiers has two issues:
1. **Counter-intuitive direction**: AUROC ~0.5 = normal, ~1 = mislocalized
2. **Batch variability**: Different batches have different null (control) distribution widths

### The Batch Effect Problem

| Batch | Mean ctrl_95 | Mean ctrl_std | Max ctrl_95 | OLD hits | NEW hits | Change |
|-------|-------------|---------------|-------------|----------|----------|--------|
| 7     | 0.8604      | 0.2391        | 0.9933      | 4        | 0        | -4     |
| 8     | 0.8759      | 0.2295        | 0.9843      | 0        | 1        | +1     |
| 11    | 0.7097      | 0.1307        | 0.8289      | 20       | 20       | 0      |
| 12    | 0.7494      | 0.1290        | 0.8920      | 13       | 15       | +2     |
| 13    | 0.8159      | 0.1499        | 0.9589      | 42       | 26       | -16    |
| 14    | 0.8071      | 0.1389        | 0.9415      | 48       | 41       | -7     |
| 15    | 0.8150      | 0.1600        | 0.9419      | 7        | 1        | -6     |
| 16    | 0.7557      | 0.1247        | 0.8986      | 40       | 33       | -7     |

**Total**: OLD = 174 hits → NEW = 137 hits (lost 37 hits, 21% reduction)

**Key observations**:
- Batches with **low ctrl_AUROC_95** (11, 12, 16) are stable or gain hits
- Batches with **high ctrl_AUROC_95** (7, 8, 13, 15) lose the most hits
- The mean ctrl_std is highest in Batch 7&8 (~0.24) and lowest in Batch 11&12 (~0.13)

## Approaches Tested

### 1. ECDF (Empirical CDF / Percentile Rank)
```python
def misloc_score_ecdf(auroc: float, ctrl_aurocs: np.ndarray) -> float:
    return percentileofscore(ctrl_aurocs, auroc, kind='rank') / 100.0
```
- **Interpretation**: "This variant exceeds X% of within-batch controls"
- **Cross-batch consistency**: Best (std of batch means = 0.0008)

### 2. Effect Size CDF
```python
def misloc_score_effect_cdf(auroc: float, ctrl_mean: float, ctrl_std: float) -> float:
    d = (auroc - ctrl_mean) / ctrl_std  # Cohen's d
    return norm.cdf(d)
```
- **Interpretation**: "P(variant exceeds random control) = score"
- **Cross-batch consistency**: Good (std = 0.0009)

### 3. Min-Max Scaling
```python
def misloc_score_minmax(auroc: float, ctrl_aurocs: np.ndarray, low_pct=50, high_pct=99):
    low = np.percentile(ctrl_aurocs, low_pct)
    high = np.percentile(ctrl_aurocs, high_pct)
    return np.clip((auroc - low) / (high - low), 0, 1)
```

### 4. Sigmoid Z-score
```python
def misloc_score_sigmoid(auroc: float, ctrl_mean: float, ctrl_std: float, k=1.5, z0=1.5):
    z = (auroc - ctrl_mean) / ctrl_std
    return 1.0 / (1.0 + np.exp(-k * (z - z0)))
```

## Methodology for 794-Variant Overlap Comparison

When comparing the old VarChAMP results (from `VarChAMP_data_supp_mat_PP.tsv`) with the new MislocScore approach, variants that appear in multiple batches (e.g., duplicate pairs 7&8, 11&12, 13&14, 15&16) needed aggregation. The current implementation uses:

- **MislocScore_ecdf_max**: MAX of MislocScore across batches
- **new_hit_any_batch**: TRUE if MislocScore > 0.95 in ANY batch

**Open question**: Should we use AVERAGE instead of MAX for cross-batch aggregation? The user raised this concern - using MAX may inflate scores for variants in batches with tight null distributions while masking issues in batches with wide distributions.

## Results Summary

### Cross-Batch Consistency (Control Score Distributions)
| Score Type | Std of Batch Means | Interpretation |
|------------|-------------------|----------------|
| Raw AUROC  | 0.035             | High batch variability |
| ECDF       | 0.0008            | Best consistency |
| Effect CDF | 0.0009            | Very good |
| Min-Max    | 0.0022            | Good |
| Sigmoid    | 0.0156            | Moderate |

**Conclusion**: ECDF provides the best cross-batch consistency for control distributions.

## Critical Issue Discovered

### The Problem with Universal Thresholds

When applying a universal threshold (MislocScore > 0.95) across batches:

**Batch 7 hit distribution (of 357 variants overlapping with 794-variant dataset):**
- Hits at MislocScore > 0.95: **0**
- Highest MislocScore in Batch 7: 0.945 (AUROC = 0.986)

**Batch 11 hit distribution (of 274 variants overlapping):**
- Hits at MislocScore > 0.95: **52**
- Variants easily reach MislocScore > 0.99

### Why This Happens

The ECDF approach normalizes scores *within* each batch. For Batch 7:
- Control AUROCs already span up to ~0.99
- A variant with AUROC = 0.986 only exceeds ~94.5% of controls
- Therefore MislocScore = 0.945 (just below 0.95 threshold)

For Batch 11:
- Control AUROCs max out around ~0.83
- A variant with AUROC = 0.90 exceeds ~99% of controls
- Therefore MislocScore = 0.99+ (easily above threshold)

### Impact on Hit Calling

Comparing 794 overlapping variants between old results and new MislocScore approach:

**Old approach (batch-specific AUROC thresholds):**
- Pathogenic hit rate: 15.7%
- Benign hit rate: 6.8%
- Odds ratio: 2.56

**New MislocScore approach (universal 0.95 threshold):**
- Pathogenic hit rate: 10.6%
- Benign hit rate: 14.4%
- Odds ratio: 0.71 (inverted!)

**Variants with changed hit status (135 total):**

| Change | Count | Pathogenic | Benign | VUS | Conflicting | Others |
|--------|-------|------------|--------|-----|-------------|--------|
| Lost hits (True→False) | 79 | 43 | 6 | 15 | 7 | 8 |
| Gained hits (False→True) | 56 | 24 | 15 | 2 | 10 | 5 |
| **Net change** | **-23** | **-19** | **+9** | **-13** | **+3** | **-3** |

**Critical observation**: The approach loses 43 pathogenic hits while gaining only 24, resulting in a net loss of 19 pathogenic hits. Meanwhile, it gains 15 benign hits while losing only 6, a net gain of 9 benign hits. This inversion explains the flipped odds ratio.

The loss of discriminatory power (OR from 2.56 to 0.71) indicates the approach is flawed.

## Root Cause Analysis

The batch normalization approach has a fundamental assumption: **the 95th percentile of controls should be a meaningful threshold across all batches**.

This assumption fails when:
1. Some batches have very wide null distributions (controls already achieving high AUROC)
2. The biological signal (variant mislocalization) may not exceed the technical noise in those batches
3. Using within-batch percentiles means batches with noisy controls will never call hits

## Potential Solutions to Explore

### Option 1: Use Pooled Controls
Instead of within-batch normalization, pool controls across all batches:
```python
all_ctrl_aurocs = np.concatenate([batch_ctrls for batch in batches])
misloc_score = percentileofscore(all_ctrl_aurocs, variant_auroc) / 100
```
- **Pro**: Variants from all batches compared to same reference
- **Con**: Ignores legitimate batch effects in imaging quality

### Option 2: Batch Quality Weighting
Weight hit calls by batch quality metrics:
```python
batch_quality = 1 / ctrl_auroc_95  # Lower control 95th = better batch
hit_confidence = misloc_score * batch_quality
```

### Option 3: Dual Threshold System
Use batch-specific thresholds but report universal scores:
- Keep original batch-specific hit calling for final calls
- Report MislocScore for cross-batch comparison only (not for thresholding)

### Option 4: Flag Low-Quality Batches
Identify batches where ctrl_AUROC_95 > 0.90 as "low sensitivity batches":
- Report hits from these batches separately
- Use lower MislocScore threshold (e.g., 0.90 instead of 0.95)

## Files Created/Modified

### New Files
- `3.downstream_analyses/scripts/test_misloc_scores.py` - Comparison test script
- `3.downstream_analyses/scripts/3-3_misloc_score_analysis.ipynb` - Analysis notebook
- `3.downstream_analyses/outputs/misloc_score_comparison/` - Output directory
  - `all_classifiers_scored.csv` - All classifiers with 4 MislocScore columns
  - `summary_statistics.csv` - Cross-batch consistency metrics
  - `hit_status_changes.csv` - Variants that changed hit status
  - `hit_status_changes_794_overlap.csv` - Same for 794 overlapping variants

### Modified Files
- `3.downstream_analyses/scripts/generate_hit_calls.py` - Added MislocScore computation

## Recommendations

1. **Do not use** the universal 0.95 MislocScore threshold for hit calling - it eliminates sensitivity for batches with wide null distributions

2. **Keep using** the original batch-specific AUROC threshold approach for hit calling

3. **Consider using** MislocScore for:
   - Cross-batch visualization and comparison
   - Ranking variants within ClinVar categories
   - Meta-analysis across studies

4. **Investigate** why Batch 7&8 have such wide null distributions:
   - Technical issues (imaging quality, staining variability)?
   - Different control alleles with more variability?
   - Plate layout effects?

## Code Reference

Key functions in `generate_hit_calls.py`:

```python
# Lines 15-29: MislocScore functions
def misloc_score_ecdf(auroc: float, ctrl_aurocs: np.ndarray) -> float:
    if len(ctrl_aurocs) == 0:
        return 0.5
    return percentileofscore(ctrl_aurocs, auroc, kind='rank') / 100.0

def misloc_score_effect_cdf(auroc: float, ctrl_mean: float, ctrl_std: float) -> float:
    if ctrl_std == 0:
        return 0.5
    d = (auroc - ctrl_mean) / ctrl_std
    return norm.cdf(d)
```

## Conclusion

The MislocScore batch normalization approach successfully achieves cross-batch consistency for control distributions (std reduced from 0.035 to 0.0008). However, this normalization eliminates the discriminatory power between pathogenic and benign variants because batches with wide null distributions (Batch 7&8) cannot achieve scores above the universal threshold.

**Key learning**: Batch normalization that equalizes control distributions is not the same as batch correction that preserves biological signal. The wide null distributions in some batches may reflect true technical limitations in detecting mislocalization, not just batch effects to be normalized away.

## Summary: Key Takeaways

1. **The ECDF approach works for its stated goal**: Cross-batch consistency of control scores improved dramatically (std: 0.035 → 0.0008)

2. **But the approach fails for hit calling**: Applying a universal threshold on normalized scores destroys discriminatory power because:
   - Batches with wide null distributions (7, 8, 13, 15) lose almost all hits
   - These batches happen to contain many pathogenic variants
   - Net effect: lose 19 pathogenic hits, gain 9 benign hits → OR inverts from 2.56 to 0.71

3. **Root cause**: Wide null distributions in certain batches mean that even strong biological signals (high AUROC) don't exceed the 95th percentile of controls in those batches

## Open Questions for User Decision

1. **Should we use AVERAGE instead of MAX** for aggregating MislocScores across duplicate batches?

2. **Should we keep the original batch-specific threshold approach** and abandon the universal MislocScore threshold idea?

3. **What threshold should we use** if we want to use MislocScore at all? (0.90? 0.85?)

4. **Should we investigate batch quality** to understand why some batches have such wide null distributions?

5. **Is the MislocScore still useful** for ranking/visualization even if not for thresholding?

## Session Archive

This document archives the work done on branch `feature/misloc-score-normalization`. The approach was tested but found to have fundamental issues with hit calling sensitivity. The original batch-specific threshold approach is still recommended for hit calling.
