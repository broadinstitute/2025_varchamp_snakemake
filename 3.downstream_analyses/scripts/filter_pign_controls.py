#!/usr/bin/env python
"""
Filter VarChAMP parquet files to keep only PIGN gene alleles and control alleles.

This script filters batch profile parquet files and classification results
to extract only PIGN-related data and control alleles for the pign-cdg repository.

Usage:
    python filter_pign_controls.py

Author: VarChAMP Team
Date: 2026-02-22
"""

import polars as pl
from pathlib import Path
import shutil
from datetime import datetime

# Define paths
VARCHAMP_ROOT = Path("/data/users/shenrunx/igvf/varchamp/2025_varchamp_snakemake")
PIGN_CDG_ROOT = Path("/home/shenrunx/igvf/varchamp/pign-cdg/data/raw")

# Batch configurations
BATCHES = ["2026_01_05_Batch_20", "2026_01_05_Batch_21"]

# Control alleles from batch config
# TC (Transfection Control)
TC_ALLELES = ["EGFP"]
# NC (Negative Control)
NC_ALLELES = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
# PC (Positive Control)
PC_ALLELES = ["ALK", "ALK_Arg1275Gln", "PTK2B"]

# Combined control alleles
CONTROL_ALLELES = TC_ALLELES + NC_ALLELES + PC_ALLELES

# For classification files, we need to match allele names that include variants
# PIGN alleles follow the pattern: PIGN, PIGN_Xxx###Yyy
PIGN_ALLELE_PREFIX = "PIGN"


def is_pign_allele(allele: str) -> bool:
    """Check if an allele is PIGN or a PIGN variant."""
    if allele is None:
        return False
    return allele == "PIGN" or allele.startswith("PIGN_")


def get_pign_and_control_alleles(df: pl.DataFrame, allele_col: str) -> list:
    """Get list of all PIGN and control alleles from a dataframe."""
    unique_alleles = df[allele_col].unique().to_list()
    return [a for a in unique_alleles if a is not None and (is_pign_allele(a) or a in CONTROL_ALLELES)]


def filter_profile_parquet(input_path: Path, output_path: Path) -> dict:
    """
    Filter a profile parquet file to keep only PIGN and control alleles.

    Uses Metadata_symbol == 'PIGN' OR Metadata_gene_allele in CONTROL_ALLELES

    Returns stats about the filtering.
    """
    print(f"Processing: {input_path.name}")

    # Use lazy evaluation for memory efficiency
    df = pl.scan_parquet(input_path)

    # Apply filter
    filtered = df.filter(
        (pl.col("Metadata_symbol") == "PIGN") |
        (pl.col("Metadata_gene_allele").is_in(CONTROL_ALLELES))
    ).collect()

    # Get original count for stats
    original_count = pl.scan_parquet(input_path).select(pl.len()).collect().item()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(output_path)

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_rows": original_count,
        "filtered_rows": len(filtered),
        "reduction_pct": round((1 - len(filtered) / original_count) * 100, 2) if original_count > 0 else 0
    }

    print(f"  Original: {original_count:,} rows -> Filtered: {len(filtered):,} rows ({stats['reduction_pct']}% reduction)")

    return stats


def filter_classifier_csv(input_path: Path, output_path: Path) -> dict:
    """
    Filter classifier_info CSV files.

    Keep rows where allele_0 OR allele_1 is PIGN or a control allele.
    """
    print(f"Processing: {input_path.name}")

    df = pl.read_csv(input_path)
    original_count = len(df)

    # Get all PIGN alleles present in the data
    all_alleles = set(df["allele_0"].unique().to_list() + df["allele_1"].unique().to_list())
    pign_alleles = [a for a in all_alleles if a is not None and is_pign_allele(a)]
    target_alleles = pign_alleles + CONTROL_ALLELES

    # Filter
    filtered = df.filter(
        pl.col("allele_0").is_in(target_alleles) |
        pl.col("allele_1").is_in(target_alleles)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_csv(output_path)

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_rows": original_count,
        "filtered_rows": len(filtered),
        "pign_alleles_found": pign_alleles
    }

    print(f"  Original: {original_count:,} rows -> Filtered: {len(filtered):,} rows")
    print(f"  PIGN alleles found: {pign_alleles}")

    return stats


def filter_feat_importance_csv(input_path: Path, output_path: Path) -> dict:
    """
    Filter feature importance CSV files.

    Keep rows where Group1 (allele) is PIGN or a control allele.
    """
    print(f"Processing: {input_path.name}")

    df = pl.read_csv(input_path)
    original_count = len(df)

    # Get all PIGN alleles present
    all_alleles = df["Group1"].unique().to_list()
    pign_alleles = [a for a in all_alleles if a is not None and is_pign_allele(a)]
    target_alleles = pign_alleles + CONTROL_ALLELES

    # Filter
    filtered = df.filter(pl.col("Group1").is_in(target_alleles))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_csv(output_path)

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_rows": original_count,
        "filtered_rows": len(filtered),
        "pign_alleles_found": pign_alleles
    }

    print(f"  Original: {original_count:,} rows -> Filtered: {len(filtered):,} rows")

    return stats


def filter_predictions_parquet(input_path: Path, output_path: Path, classifier_ids: list) -> dict:
    """
    Filter predictions parquet files.

    Keep rows where Classifier_ID matches filtered classifier info.
    """
    print(f"Processing: {input_path.name}")

    df = pl.scan_parquet(input_path)
    original_count = pl.scan_parquet(input_path).select(pl.len()).collect().item()

    # Filter by classifier IDs
    filtered = df.filter(pl.col("Classifier_ID").is_in(classifier_ids)).collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(output_path)

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_rows": original_count,
        "filtered_rows": len(filtered)
    }

    print(f"  Original: {original_count:,} rows -> Filtered: {len(filtered):,} rows")

    return stats


def process_batch(batch: str) -> dict:
    """Process all files for a single batch."""
    print(f"\n{'='*60}")
    print(f"Processing batch: {batch}")
    print(f"{'='*60}")

    batch_stats = {"batch": batch, "files": []}

    # === Profile parquets ===
    print("\n--- Profile Parquets ---")

    profile_files = [
        ("profiles.parquet", "profiles_pign_controls.parquet"),
        ("profiles_tcdropped_filtered_var_mad_outlier.parquet", "profiles_normalized_pign_controls.parquet"),
        ("profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells.parquet", "profiles_featselect_pign_controls.parquet"),
    ]

    input_dir = VARCHAMP_ROOT / "2.snakemake_pipeline/outputs/batch_profiles" / batch
    output_dir = PIGN_CDG_ROOT / "aws_cpg" / batch

    for input_name, output_name in profile_files:
        input_path = input_dir / input_name
        output_path = output_dir / output_name

        if input_path.exists():
            stats = filter_profile_parquet(input_path, output_path)
            batch_stats["files"].append(stats)
        else:
            print(f"  SKIP: {input_name} not found")

    # === Classification Results ===
    print("\n--- Classification Results ---")

    pipeline = "profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells"
    class_input_dir = VARCHAMP_ROOT / f"2.snakemake_pipeline/outputs/classification_results/{batch}/{pipeline}"
    class_output_dir = PIGN_CDG_ROOT / "classification_results" / batch

    # First, filter classifier_info to get the list of relevant Classifier_IDs
    classifier_ids = []

    for csv_name in ["classifier_info.csv", "classifier_info_gfp_adj.csv"]:
        input_path = class_input_dir / csv_name
        output_name = csv_name.replace(".csv", "_pign_controls.csv")
        output_path = class_output_dir / output_name

        if input_path.exists():
            stats = filter_classifier_csv(input_path, output_path)
            batch_stats["files"].append(stats)

            # Collect classifier IDs for filtering predictions
            filtered_df = pl.read_csv(output_path)
            classifier_ids.extend(filtered_df["Classifier_ID"].to_list())

    classifier_ids = list(set(classifier_ids))
    print(f"  Collected {len(classifier_ids)} unique Classifier_IDs for filtering predictions")

    # Filter predictions parquets
    for parquet_name in ["predictions.parquet", "predictions_gfp_adj.parquet"]:
        input_path = class_input_dir / parquet_name
        output_name = parquet_name.replace(".parquet", "_pign_controls.parquet")
        output_path = class_output_dir / output_name

        if input_path.exists():
            stats = filter_predictions_parquet(input_path, output_path, classifier_ids)
            batch_stats["files"].append(stats)

    # Filter feature importance CSVs
    for csv_name in ["feat_importance.csv", "feat_importance_gfp_adj.csv"]:
        input_path = class_input_dir / csv_name
        output_name = csv_name.replace(".csv", "_pign_controls.csv")
        output_path = class_output_dir / output_name

        if input_path.exists():
            stats = filter_feat_importance_csv(input_path, output_path)
            batch_stats["files"].append(stats)

    # Filter gfp_adj_filtered_cells_profiles.parquet (single-cell profiles)
    gfp_profiles_path = class_input_dir / "gfp_adj_filtered_cells_profiles.parquet"
    if gfp_profiles_path.exists():
        output_path = class_output_dir / "gfp_adj_filtered_cells_profiles_pign_controls.parquet"
        stats = filter_profile_parquet(gfp_profiles_path, output_path)
        batch_stats["files"].append(stats)

    # Copy classify.log as-is
    log_path = class_input_dir / "classify.log"
    if log_path.exists():
        output_log = class_output_dir / "classify.log"
        output_log.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(log_path, output_log)
        print(f"  Copied: classify.log")

    # === Classification Analyses ===
    print("\n--- Classification Analyses ---")

    analysis_input_dir = VARCHAMP_ROOT / f"2.snakemake_pipeline/outputs/classification_analyses/{batch}/{pipeline}"

    for csv_name in ["metrics.csv", "metrics_gfp_adj.csv"]:
        input_path = analysis_input_dir / csv_name
        output_name = csv_name.replace(".csv", "_pign_controls.csv")
        output_path = class_output_dir / output_name

        if input_path.exists():
            stats = filter_classifier_csv(input_path, output_path)
            batch_stats["files"].append(stats)

    return batch_stats


def create_readme_files(all_stats: list):
    """Create README files documenting the filtered data."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # AWS CPG README
    aws_readme = f"""# VarChAMP PIGN + Controls Profile Data

## Source
Filtered from VarChAMP Batch 20 and 21 single-cell profile parquet files.

## Filtering Criteria
- **PIGN alleles**: All cells where `Metadata_symbol == 'PIGN'`
  - Includes disease_wt (PIGN) and all variants (PIGN_Xxx###Yyy)
- **Control alleles**: Cells where `Metadata_gene_allele` matches:
  - TC (Transfection Control): {TC_ALLELES}
  - NC (Negative Control): {NC_ALLELES}
  - PC (Positive Control): {PC_ALLELES}

## Files

### Per-batch directories

Each batch directory contains:

| File | Description | Pipeline Stage |
|------|-------------|----------------|
| `profiles_pign_controls.parquet` | Raw profiles from AWS S3 CPG | Annotated |
| `profiles_normalized_pign_controls.parquet` | Plate-normalized profiles | After MAD outlier removal |
| `profiles_featselect_pign_controls.parquet` | Feature-selected, quality-filtered cells | Final pipeline stage |

## Provenance

- **Source repository**: `/data/users/shenrunx/igvf/varchamp/2025_varchamp_snakemake`
- **Source directory**: `2.snakemake_pipeline/outputs/batch_profiles/`
- **Extraction date**: {timestamp}
- **Script**: `3.downstream_analyses/scripts/filter_pign_controls.py`

## Column Reference

Key metadata columns:
- `Metadata_symbol`: Gene symbol (e.g., 'PIGN', 'ALK')
- `Metadata_gene_allele`: Specific allele (e.g., 'PIGN', 'PIGN_Ala56Val', 'EGFP')
- `Metadata_plate_map_name`: Plate identifier
- `Metadata_well_position`: Well position (A01-P24)
- `Metadata_node_type`: Node type (disease_wt, allele, TC, NC, PC)
"""

    aws_readme_path = PIGN_CDG_ROOT / "aws_cpg" / "README.md"
    aws_readme_path.write_text(aws_readme)
    print(f"\nCreated: {aws_readme_path}")

    # Classification Results README
    class_readme = f"""# VarChAMP PIGN + Controls Classification Results

## Source
Filtered from VarChAMP Batch 20 and 21 classification results.

## Filtering Criteria

### For classifier/metrics CSVs:
- Keep rows where `allele_0` OR `allele_1` matches:
  - PIGN alleles: PIGN, PIGN_Xxx###Yyy pattern
  - Control alleles: {CONTROL_ALLELES}

### For predictions parquets:
- Keep rows with `Classifier_ID` matching filtered classifier info

### For feature importance CSVs:
- Keep rows where `Group1` (allele) matches PIGN or control alleles

### For single-cell profiles (gfp_adj_filtered_cells):
- Same criteria as profile parquets (`Metadata_symbol` or `Metadata_gene_allele`)

## Files

### Per-batch directories

| File | Description |
|------|-------------|
| `classifier_info_pign_controls.csv` | Classifier metadata (all feature types) |
| `classifier_info_gfp_adj_pign_controls.csv` | Classifier metadata (GFP-adjusted) |
| `predictions_pign_controls.parquet` | Cell-level predictions |
| `predictions_gfp_adj_pign_controls.parquet` | GFP-adjusted predictions |
| `gfp_adj_filtered_cells_profiles_pign_controls.parquet` | Single-cell profiles for GFP-adjusted analysis |
| `feat_importance_pign_controls.csv` | Feature importance scores |
| `feat_importance_gfp_adj_pign_controls.csv` | Feature importance (GFP-adjusted) |
| `metrics_pign_controls.csv` | Classification metrics (AUROC, etc.) |
| `metrics_gfp_adj_pign_controls.csv` | Metrics (GFP-adjusted) |
| `classify.log` | Original classification log (unfiltered, for reference) |

## Key Columns

### classifier_info / metrics
- `Classifier_ID`: Unique classifier identifier (Plate_well0_well1)
- `allele_0`, `allele_1`: The two alleles being compared
- `well_0`, `well_1`: Wells for each allele
- `AUROC`, `AUPRC`, etc.: Performance metrics (in metrics files)

### predictions
- `Classifier_ID`: Links to classifier_info
- `CellID`: Unique cell identifier
- `Label`: True class (0 or 1)
- `Prediction`: Model prediction probability

### feat_importance
- `Group1`: Allele name
- `Group2`: Well pair (well0_well1)
- Feature columns: Importance scores per morphological feature

## Provenance

- **Source repository**: `/data/users/shenrunx/igvf/varchamp/2025_varchamp_snakemake`
- **Source directories**:
  - `2.snakemake_pipeline/outputs/classification_results/`
  - `2.snakemake_pipeline/outputs/classification_analyses/`
- **Pipeline**: `profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells`
- **Extraction date**: {timestamp}
- **Script**: `3.downstream_analyses/scripts/filter_pign_controls.py`
"""

    class_readme_path = PIGN_CDG_ROOT / "classification_results" / "README.md"
    class_readme_path.write_text(class_readme)
    print(f"Created: {class_readme_path}")


def main():
    """Main entry point."""
    print("="*60)
    print("VarChAMP PIGN + Controls Data Extraction")
    print("="*60)
    print(f"\nSource: {VARCHAMP_ROOT}")
    print(f"Target: {PIGN_CDG_ROOT}")
    print(f"\nControl alleles: {CONTROL_ALLELES}")
    print(f"PIGN gene symbol filter: Metadata_symbol == 'PIGN'")

    all_stats = []

    for batch in BATCHES:
        batch_stats = process_batch(batch)
        all_stats.append(batch_stats)

    # Create README files
    create_readme_files(all_stats)

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)

    # Summary
    print("\nSummary:")
    for batch_stats in all_stats:
        print(f"\n{batch_stats['batch']}:")
        for file_stats in batch_stats["files"]:
            if "reduction_pct" in file_stats:
                print(f"  - {Path(file_stats['output_file']).name}: {file_stats['filtered_rows']:,} rows ({file_stats['reduction_pct']}% reduction)")
            else:
                print(f"  - {Path(file_stats['output_file']).name}: {file_stats['filtered_rows']:,} rows")


if __name__ == "__main__":
    main()
