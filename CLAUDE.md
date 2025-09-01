# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VarChAMP** (Variant Classification via Cell Painting) analysis pipeline for processing and analyzing imaging data from high-throughput cell painting experiments. The project identifies genetic variants that cause measurable phenotypic changes through machine learning classification of single-cell morphological profiles.

## Repository Structure

```
├── 1.image_preprocess_qc/          # Image quality control and preprocessing
│   ├── inputs/cpg_imgs/            # Raw Cell Painting Gallery images (symlink)
│   └── scripts/1_calc_plate_bg.py  # Calculate plate background metrics
├── 2.snakemake_pipeline/           # Core analysis pipeline using Snakemake
│   ├── inputs/
│   │   ├── configs/                # Batch configuration JSON files
│   │   ├── metadata/platemaps/     # Plate mapping files per batch
│   │   ├── single_cell_profiles/   # CellProfiler SQLite databases
│   │   └── snakemake_files/        # Individual Snakefile templates
│   ├── preprocess/                 # Data preprocessing modules
│   ├── classification/             # ML classification modules  
│   ├── profiling/                  # Cell profiling modules
│   └── rules/                      # Snakemake rule definitions
└── 3.downstream_analyses/          # Post-pipeline analysis notebooks
    ├── inputs/plate_well_qc_metrics/ # QC metrics by batch
    ├── outputs/                    # Analysis results and visualizations
    └── scripts/                    # Jupyter notebooks for analysis
```

## Environment Setup

The project uses conda/mamba for dependency management:

```bash
# Create and activate environment
conda env create -f env.yml
conda activate varchamp
```

Key dependencies include:
- **Snakemake 7.32.4** for workflow management
- **CUDA 12.6** for GPU acceleration  
- **Python 3.8** with scientific stack (pandas, numpy, polars, scikit-learn)
- **CellProfiler-related tools** (pycytominer, cytotable)

## Common Development Commands

### Data Download
Download Cell Painting Gallery data from AWS S3:
```bash
./download_aws_cpg_data.sh
```

### Image QC
Calculate plate background metrics:
```bash
python 1.image_preprocess_qc/scripts/1_calc_plate_bg.py \
    --batch_list "batch_names" \
    --input_dir "1.image_preprocess_qc/inputs/cpg_imgs" \
    --output_dir "1.image_preprocess_qc/outputs/plate_bg_summary" \
    --workers 64
```

### Snakemake Pipeline Execution
Navigate to pipeline directory and run:
```bash
cd 2.snakemake_pipeline/
./run_snakemake_pipeline.sh
```

Or run specific batch:
```bash
# Copy batch-specific Snakefile to working directory
cp inputs/snakemake_files/Snakefile_batch17 .

# Execute with full CPU utilization
snakemake \
    --snakefile Snakefile_batch17 \
    --cores all &> outputs/snakemake_logs/snakemake_batch17.log
```

### Analysis Notebooks
Run downstream analyses using Jupyter notebooks in `3.downstream_analyses/scripts/`:
- `0_allele_qc.ipynb` - Quality control for genetic alleles
- `1_flag_cell_count_hits.ipynb` - Identify cell count phenotypes
- `2_protein_abundance.ipynb` - Protein abundance analysis
- `3_classification_metrics.ipynb` - ML classification performance
- `4_visualize_imgs.ipynb` - Image visualization and inspection
- `5_aggregate_well_profiles.ipynb` - Well-level profile aggregation
- `6_feature_importance_analyses.ipynb` - Feature importance analysis
- `7_batch_effect_analyses.ipynb` - Batch effect assessment

## Architecture Overview

### Snakemake Workflow
The pipeline follows a standardized workflow defined in `rules/`:

1. **Data Conversion** (`preprocess.smk`): Convert SQLite to Parquet format
2. **Annotation**: Add plate mapping metadata to single-cell profiles  
3. **Aggregation**: Combine plates into batch-level profiles
4. **Filtering**: Remove poor quality cells and features
5. **Normalization**: Apply statistical normalization methods
6. **Feature Selection**: Select informative morphological features
7. **Classification**: Train ML models to distinguish genetic variants
8. **Analysis**: Generate performance metrics and visualizations

### Preprocessing Modules
Located in `2.snakemake_pipeline/preprocess/`:
- `to_parquet.py` - SQLite to Parquet conversion
- `annotate.py` - Metadata annotation and aggregation
- `clean.py` - Data quality filtering
- `filter.py` - Cell-level filtering
- `normalize.py` - Statistical normalization
- `feature_select.py` - Feature selection methods

### Classification System
Located in `2.snakemake_pipeline/classification/`:
- `classify.py` - ML model training and prediction
- `analysis.py` - Performance metric calculation

### Configuration System
Each batch has a JSON configuration file in `inputs/configs/` specifying:
- Input/output directories
- Batch metadata
- Pipeline parameters
- Plate mapping information

## Working with Batches

The pipeline processes experiments in "batches" (e.g., Batch_7, Batch_17). Each batch requires:

1. **Configuration file**: `inputs/configs/YYYY_MM_DD_Batch_X.json`
2. **Plate maps**: `inputs/metadata/platemaps/YYYY_MM_DD_Batch_X/`
3. **Single-cell profiles**: `inputs/single_cell_profiles/YYYY_MM_DD_Batch_X/`
4. **Snakefile template**: `inputs/snakemake_files/Snakefile_batch[X]`

## Image Visualization

The project includes sophisticated image visualization capabilities in `display_img.py`:
- `plot_allele()` - Display variant vs wildtype comparisons across timepoints
- `plot_allele_cell()` - Show individual cell crops with quality scoring  
- Quality control flagging for background-only images
- Support for multiple fluorescence channels (DNA, AGP, Mito, GFP, Morph)

## Tips for Development

- **GPU Usage**: The environment includes CUDA 12.6 for GPU-accelerated processing
- **Large Files**: Image data is stored as symlinks; actual files may be downloaded from S3 on-demand
- **Batch Processing**: Always work with complete batches rather than individual plates
- **Memory Management**: Use Polars for large dataset operations when possible
- **Parallel Processing**: Snakemake automatically parallelizes compatible steps