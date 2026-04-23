# VarChAMP Classification Framework Documentation

## Overview

This document describes the VarChAMP (Variant Classification via Cell Painting) machine learning framework for classifying genetic variants based on single-cell morphological profiles. The framework is designed to:

1. Train XGBoost classifiers to distinguish variant alleles from reference (wildtype) alleles
2. Construct NULL distributions using control allele comparisons to establish significance thresholds
3. Aggregate classifier-level metrics to allele-level summaries for hit calling

This documentation is intended to enable re-implementation of the exact same ML workflow using different feature sets (e.g., deep learning features instead of CellProfiler features), while maintaining the same cell populations for training and testing.

---

## 1. Experimental Design and Plate Layouts

### 1.1 Two Plate Layout Types

The framework supports two experimental plate layouts, specified in the batch configuration JSON files (`inputs/configs/YYYY_MM_DD_Batch_X.json`):

#### **`single_rep` Layout** (Batches 7, 8, 13, 14, 15, 16)
- **Description**: Each allele is tested in **ONE well per plate**, with **4 technical replicate plates** using the identical platemap
- **Plate naming convention**: `YYYY_MM_DD_BX...P#T#` where:
  - `BX` = Batch number
  - `P#` = Platemap number (different allele layouts)
  - `T#` = **Technical replicate** (T1, T2, T3, T4 are 4 separate physical plates with the same platemap layout)
- **Key insight**: `_T1`, `_T2`, `_T3`, `_T4` suffixes indicate **technical replicates** (NOT timepoints!)
- **Example plates sharing the same platemap `B13A7A8P1_R1`**:
  ```
  2025_01_27_B13A7A8P1_T1
  2025_01_27_B13A7A8P1_T2
  2025_01_27_B13A7A8P1_T3
  2025_01_27_B13A7A8P1_T4
  ```

#### **`multi_rep` Layout** (Batches 11, 12)
- **Description**: Each allele is tested in **FOUR wells on a single plate** (4 within-plate replicates)
- **Plate naming convention**: `YYYY-MM-DD_BXA#R#` (no `_T#` suffix)
- **Example**: `2024-12-09_B11A1R1`
- **Key insight**: Within-plate well replicates replace technical replicate plates

### 1.2 Batch Pairing: Biological Replicates

Sequential batches with identical allele sets are **biological replicates**:

| Batch Pair | Layout | Notes |
|------------|--------|-------|
| **7 vs 8** | single_rep | Same alleles, independent experiments |
| **11 vs 12** | multi_rep | Same alleles, independent experiments |
| **13 vs 14** | single_rep | Same alleles, independent experiments |
| **15 vs 16** | single_rep | Same alleles, independent experiments |

These biological replicates should be analyzed independently first, then combined for meta-analysis to assess reproducibility.

---

## 2. Training/Testing Strategy

### 2.1 Core Principle: Leave-One-Out Cross-Validation

**Critical design goal**: Avoid overfitting by ensuring training and testing data come from **different spatial/experimental units**.

### 2.2 `single_rep` Layout: Plate-Level CV

**Implementation**: `classify_single_rep_per_plate.py`

For each variant-reference allele pair:

1. **Identify common plates**: Find plates containing both variant and reference wells
2. **For each ref-var well pair** across all plates:
   - **For each test plate** (leave-one-out):
     - **Training set**: All cells from the **other 3 technical replicate plates** (same platemap)
     - **Testing set**: All cells from the **held-out plate**
3. **Classifier ID format**: `{Plate}_{well_ref}_{well_var}`

```
Key function: stratify_by_plate(df_sampled, plate)
- Gets platemap from test plate
- Train on: (same platemap) AND (different plate)
- Test on: (specific plate)
```

**Result**: 4 classifiers per ref-var well pair (one per technical replicate plate). Multiple well pairs may exist if variant/reference are tested in multiple wells across different platemaps.

### 2.3 `multi_rep` Layout: Well-Pair CV

**Implementation**: `classify_multi_rep_per_plate.py`

For each variant-reference allele pair on the same plate:

1. **Pair wells**: Match ref wells to var wells (4 pairs typically)
2. **4-fold CV**: For each iteration:
   - **Training set**: 3 well-pairs (6 wells total)
   - **Testing set**: 1 well-pair (2 wells total)
3. **Classifier ID format**: `{Plate}_{well_ref}_{well_var}` (test pair)

```
Key function: stratify_by_well_pair_exp(df_sampled, well_pair_list)
- well_pair_list[0] = test pair
- well_pair_list[1:] = training pairs
```

**Result**: 4 classifiers per allele pair (one per well-pair fold).

---

## 3. Control Allele NULL Distribution

### 3.1 Purpose

Control alleles (same genotype, different wells) allow us to:
1. Estimate **well-position effects** in the feature space
2. Establish **significance thresholds** for calling true hits
3. Distinguish **biological signal** from **technical artifacts**

### 3.2 Control Types

Defined in batch config JSON:
- **TC (Transfection Control)**: `["EGFP"]` - Excluded from analysis
- **NC (Negative Control)**: `["RHEB", "MAPK9", "PRKACB", "SLIRP"]` - Expected AUROC ~0.5
- **PC (Positive Control)**: `["ALK", "ALK_Arg1275Gln", "PTK2B"]` - Expected AUROC ~0.5 (same allele)

### 3.3 Control Classification Workflow

**Single-rep layout** (`control_group_runner`):
1. Group cells by `Metadata_gene_allele` (same control allele)
2. Group by `Metadata_plate_map_name` (same platemap)
3. For each pair of wells containing the same control allele:
   - Run plate-level CV (same as experimental)
   - Label: well_1 = 1, well_2 = 0 (arbitrary assignment)

**Multi-rep layout** (`control_group_runner_fewer_rep`):
1. Group cells by allele, then by platemap
2. For each pair of 4 wells with same control allele:
   - Use 2 wells for training, 2 wells for testing
   - `stratify_by_well_pair_ctrl(df, well_pair_trn)` splits train/test

### 3.4 Interpreting Control AUROCs

| AUROC Range | Interpretation |
|-------------|----------------|
| ~0.5 | No distinguishing signal (expected for same-genotype comparisons) |
| 0.5-0.7 | Well-position effects or technical noise |
| >0.7-0.8 | Strong well-position effects (concerning if common) |

**Hit threshold**: 99th percentile of control AUROCs per feature type and batch.

---

## 4. XGBoost Classifier Details

### 4.1 Model Configuration

```python
xgb_params = {
    "objective": "binary:logistic",
    "n_estimators": 150,
    "tree_method": "hist",
    "learning_rate": 0.05,
    "scale_pos_weight": num_neg / num_pos,  # Class imbalance correction
    "n_jobs": 2,  # CPU parallelization
}
```

### 4.2 Feature Types

Classification is run separately for each feature channel:
- **GFP/Protein**: Protein localization features (configurable channel name)
- **DNA**: Nuclear features
- **AGP**: Actin/Golgi/Plasma membrane features
- **Mito**: Mitochondrial features
- **Morph**: All non-GFP morphological features

```python
FEAT_TYPE_SET = ["GFP", "DNA", "AGP", "Mito", "Morph"]
```

### 4.3 Data Quality Filters

1. **Cell count threshold**: Wells with < `cc_threshold` (default: 20) cells are dropped
2. **Training imbalance**: Classifiers with >100:1 or <1:100 class ratio are skipped
3. **Well presence**: Wells must be present on all 4 plates/replicates

### 4.4 Outputs Per Classifier

1. **Feature importance**: XGBoost feature importances (`df_feat`)
2. **Classifier info**: Train/test sizes, well identifiers (`classifier_df`)
3. **Cell-level predictions**: Probability scores for each cell (`pred_df`)

---

## 5. Metric Calculation and Aggregation

### 5.1 Per-Classifier Metrics

**Implementation**: `analysis.py::compute_metrics()`

Computed from cell-level predictions:
- **AUROC**: Area Under ROC Curve (`roc_auc_score`)
- **AUPRC**: Area Under Precision-Recall Curve
- **AUBPRC**: Background-corrected AUPRC
- **Macro F1**: Macro-averaged F1 score
- **Sensitivity/Specificity**: At threshold 0.5
- **Balanced Accuracy**: Average of sensitivity and specificity

### 5.2 Aggregation to Allele Level

**Implementation**: `analysis.py::compute_hits()`

1. **Filter by training imbalance**: Exclude classifiers with `Training_imbalance > trn_imbal_thres` (default: 3)
2. **Require minimum classifiers**: At least `min_num_classifier` (default: 2) per allele
3. **Compute mean AUROC**: Group by `(Classifier_type, allele_0, Allele_set, Batch)` and take mean

### 5.3 Hit Calling

```python
# Control threshold: 99th percentile of control AUROCs
AUROC_thresh = control_df.group_by(["Classifier_type", "Batch"]).quantile(0.99)

# Hit if mean AUROC > threshold
hit = (AUROC_mean > AUROC_thresh)
```

---

## 6. GFP-Intensity Correction (Optional)

### 6.1 Purpose

Genetic variants can affect GFP expression independently of morphology. This creates a confounding variable where classifiers might distinguish variants based on GFP intensity rather than true morphological differences.

### 6.2 Correction Workflow

**Implementation**: `classify_gfp_filter_func.py`

1. **Pre-correction t-test**: Paired t-test on GFP intensity between ref/var wells
2. **Range optimization**: Find overlapping GFP intensity quantile ranges
3. **Cell filtering**: Keep only cells within matched GFP ranges
4. **Ratio balancing**: Subsample to maintain ≤3:1 class ratio
5. **Post-correction t-test**: Validate GFP differences are reduced
6. **Classification**: Run standard pipeline on GFP-matched cells

### 6.3 Quantile Strategy

Try progressively wider quantile ranges until minimum cell count is met:
```python
quantile_pairs = [(0.25, 0.75), (0.2, 0.8), (0.15, 0.85), (0.1, 0.9)]
```

---

## 7. Adapting for New Feature Sets

### 7.1 Required Inputs

To use the same cells with different features (e.g., deep learning embeddings):

1. **Cell identifier**: `Metadata_CellID` constructed as:
   ```python
   f"{Metadata_Plate}_{Metadata_well_position}_{Metadata_ImageNumber}_{Metadata_ObjectNumber}"
   ```

2. **Metadata columns** (minimum required):
   - `Metadata_Plate`: Plate identifier
   - `Metadata_Well` / `Metadata_well_position`: Well position
   - `Metadata_ImageNumber`: Image field number
   - `Metadata_ObjectNumber`: Cell object number
   - `Metadata_plate_map_name`: Platemap identifier
   - `Metadata_gene_allele`: Allele identifier
   - `Metadata_symbol`: Gene symbol
   - `Metadata_node_type`: Category (`disease_wt`, `allele`, `TC`, `NC`, `PC`)

3. **Feature columns**: Any columns not starting with `Metadata_`

### 7.2 Parsing CellID

If your data only has `Metadata_CellID`, parse it:
```python
df[['Metadata_Plate', 'Metadata_well_position',
    'Metadata_ImageNumber', 'Metadata_ObjectNumber']] = \
    df['Metadata_CellID'].str.split('_', expand=True)
```

### 7.3 Implementation Steps

1. **Match cell populations**: Join your features with the original data on `Metadata_CellID`
2. **Verify metadata**: Ensure required metadata columns exist or can be derived
3. **Run classification**: Use the same `plate_layout` config as original batch
4. **Aggregate metrics**: Use the same aggregation pipeline

### 7.4 Code Modifications for New Features

```python
# In classify_helper_func.py, modify get_classifier_features()
def get_classifier_features(dframe: pd.DataFrame, feat_type: str):
    """
    For deep learning features:
    - feat_type could be "embedding_layer_X" or "all"
    - Filter feature columns by prefix/pattern matching
    """
    feat_col = [c for c in dframe.columns if not c.startswith("Metadata_")]
    # Add your feature filtering logic here
    return dframe
```

---

## 8. File Structure and Outputs

### 8.1 Input Files

```
inputs/
├── configs/
│   └── YYYY_MM_DD_Batch_X.json       # Batch configuration
├── metadata/platemaps/
│   └── YYYY_MM_DD_Batch_X/
│       ├── barcode_platemap.csv      # Plate -> Platemap mapping
│       └── platemap/                 # Well -> Allele mapping
└── single_cell_profiles/             # CellProfiler features (Parquet/SQLite)
```

### 8.2 Output Files

```
outputs/YYYY_MM_DD_Batch_X/
├── classifier_info.csv               # Classifier metadata
├── classifier_info_gfp_adj.csv       # GFP-adjusted classifier metadata
├── feat_importance.csv               # Feature importances
├── feat_importance_gfp_adj.csv       # GFP-adjusted feature importances
├── predictions.parquet               # Cell-level predictions
├── predictions_gfp_adj.parquet       # GFP-adjusted predictions
├── filtered_cells_gfp_adj.parquet    # GFP-matched cell profiles
├── metrics.csv                       # Per-classifier metrics
└── metrics_summary.csv               # Allele-level aggregated metrics
```

### 8.3 Classifier Info Schema

| Column | Description |
|--------|-------------|
| `Classifier_ID` | `{Plate}_{well_0}_{well_1}` |
| `Plate` | Test plate identifier |
| `well_0` | Reference well position |
| `allele_0` | Reference allele name |
| `trainsize_0` | Number of reference cells in training |
| `testsize_0` | Number of reference cells in testing |
| `well_1` | Variant well position |
| `allele_1` | Variant allele name |
| `trainsize_1` | Number of variant cells in training |
| `testsize_1` | Number of variant cells in testing |

### 8.4 Prediction Schema

| Column | Description |
|--------|-------------|
| `Classifier_ID` | Links to classifier info |
| `CellID` | `{Plate}_{well}_{ImageNumber}_{ObjectNumber}` |
| `Label` | Ground truth (0=variant, 1=reference) |
| `Prediction` | Predicted probability of class 1 |
| `Metadata_Feature_Type` | Feature channel used |
| `Metadata_Control` | Whether this is a control comparison |

---

## 9. Summary: Key Design Principles

1. **Never train and test on the same spatial unit**:
   - `single_rep`: Different plates (same platemap)
   - `multi_rep`: Different well-pairs (same plate)

2. **Use controls to establish null distribution**:
   - Same-allele, different-well comparisons capture technical noise
   - 99th percentile threshold avoids false positives from well effects

3. **Aggregate carefully**:
   - Filter by training class imbalance (<3:1)
   - Require minimum number of classifiers (≥2)
   - Report mean AUROC across valid classifiers

4. **Preserve cell identity for reproducibility**:
   - `Metadata_CellID` enables exact cell matching across feature sets
   - Same train/test splits ensure fair comparison

5. **Biological vs technical replicates**:
   - Technical: `_T#` suffix (same experiment, different imaging)
   - Biological: Batch pairs (independent experiments, same alleles)

---

## 10. Quick Reference: Batch Configurations

| Batch | Date | Layout | Biological Replicate |
|-------|------|--------|---------------------|
| 7 | 2024-01-23 | single_rep | with Batch 8 |
| 8 | 2024-02-06 | single_rep | with Batch 7 |
| 11 | 2024-12-09 | multi_rep | with Batch 12 |
| 12 | 2024-12-09 | multi_rep | with Batch 11 |
| 13 | 2025-01-27 | single_rep | with Batch 14 |
| 14 | 2025-01-28 | single_rep | with Batch 13 |
| 15 | 2025-03-17 | single_rep | with Batch 16 |
| 16 | 2025-03-17 | single_rep | with Batch 15 |

---

## Appendix A: Code Entry Points

```python
# Main workflow
from classification.classify import run_classify_workflow

run_classify_workflow(
    input_path="path/to/processed_profiles.parquet",
    input_path_orig="path/to/original_profiles.parquet",
    feat_output_path="outputs/feat_importance.csv",
    info_output_path="outputs/classifier_info.csv",
    preds_output_path="outputs/predictions.parquet",
    feat_output_path_gfp="outputs/feat_importance_gfp_adj.csv",
    info_output_path_gfp="outputs/classifier_info_gfp_adj.csv",
    preds_output_path_gfp="outputs/predictions_gfp_adj.parquet",
    filtered_cell_path="outputs/filtered_cells_gfp_adj.parquet",
    cc_threshold=20,
    plate_layout="single_rep",  # or "multi_rep"
    use_gpu=None,
    protein_channel_name="GFP"
)

# Metric calculation
from classification.analysis import calculate_class_metrics, compute_hits

metrics_df = calculate_class_metrics(
    classifier_info="outputs/classifier_info.csv",
    predictions="outputs/predictions.parquet",
    metrics_file="outputs/metrics.csv"
)

compute_hits(
    metrics_file="outputs/metrics.csv",
    metrics_summary_file="outputs/metrics_summary.csv",
    trn_imbal_thres=3,
    min_num_classifier=2
)
```

---

## Appendix B: Metadata Column Requirements

### Minimum Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Metadata_Plate` | str | Unique plate identifier | `2024_01_17_B7A1R1_P1T1` |
| `Metadata_Well` | str | Well position (A01-P24) | `B03` |
| `Metadata_well_position` | str | Same as Well or alternative format | `B03` |
| `Metadata_ImageNumber` | int | Field of view index | `1` |
| `Metadata_ObjectNumber` | int | Cell object index within image | `42` |
| `Metadata_plate_map_name` | str | Platemap identifier | `B7A1R1_P1` |
| `Metadata_gene_allele` | str | Full allele identifier | `GENE_Variant` |
| `Metadata_symbol` | str | Gene symbol | `GENE` |
| `Metadata_node_type` | str | Allele category | `disease_wt`, `allele`, `TC`, `NC`, `PC` |

### Derived Columns

| Column | Derivation |
|--------|------------|
| `Metadata_CellID` | `f"{Plate}_{well}_{ImageNumber}_{ObjectNumber}"` |
| `Metadata_control` | `True` if `node_type` in `["TC", "NC", "PC"]` |
| `Label` | Assigned during classification (0 or 1) |
