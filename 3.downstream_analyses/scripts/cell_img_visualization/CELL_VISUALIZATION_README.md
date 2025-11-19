# Cell Visualization Modules - User Guide

This guide describes the redesigned cell visualization system for VarChAMP, providing a clean, modular approach to selecting, cropping, and visualizing single cells from Cell Painting microscopy images.

## Overview

The system is organized into three focused modules:

1. **`cell_selector.py`** - Cell selection and filtering
2. **`cell_cropper.py`** - Image cropping with multiple strategies
3. **`cell_visualizer.py`** - Visualization with configurable contrast

## Quick Start

```python
import polars as pl
from cell_selector import select_cells_top_n, filter_by_quality_metrics
from cell_cropper import load_multichannel_cell_crops
from cell_visualizer import viz_cell_grid

# Load profiles
profiles = pl.read_parquet("path/to/profiles.parquet")

# Select top cells by feature
selected = select_cells_top_n(
    profiles,
    feature="Cells_Intensity_IntegratedIntensity_GFP",
    n=10,
    direction="high"
)

# Apply quality filters
filtered = filter_by_quality_metrics(
    selected,
    min_edge_dist=50,
    area_range=(500, 5000)
)

# Extract crops for first cell
cell_crops = load_multichannel_cell_crops(
    filtered[0],
    channels=['DAPI', 'AGP', 'Mito', 'GFP'],
    imgs_dir="/path/to/images",
    method='bbox',
    target_size=128,
    recenter=True
)

# Visualize with adjustable contrast
viz_cell_grid(
    cell_crops,
    contrast_percentiles=99.5,  # Try different values: 95, 97, 99, 99.5, 99.9
    channel_mapping='morphology',
    save_path='cell_viz.png'
)
```

## Module 1: cell_selector.py

Cell selection and filtering functions.

### Key Functions

#### `filter_cells_by_metadata()`
Filter cells by metadata fields (allele, plate, well, site).

```python
from cell_selector import filter_cells_by_metadata

# Filter to specific variant
variant_cells = filter_cells_by_metadata(
    profiles,
    allele='BRCA1_V1',
    plate='B7A1R1_P1',
    well='A01'
)
```

#### `select_cells_top_n()`
Select cells with extreme feature values.

```python
from cell_selector import select_cells_top_n

# Get 50 cells with highest GFP intensity
high_gfp = select_cells_top_n(
    profiles,
    feature='Cells_Intensity_IntegratedIntensity_GFP',
    n=50,
    direction='high'  # or 'low'
)
```

#### `select_cells_percentile_random()`
Random sampling from percentile bins.

```python
from cell_selector import select_cells_percentile_random

# Sample 20 cells each from bottom 10%, middle 40-60%, top 10%
selected = select_cells_percentile_random(
    profiles,
    feature='Cells_Intensity_IntegratedIntensity_GFP',
    percentile_bins=[(0, 10), (45, 55), (90, 100)],
    n_per_bin=20,
    seed=42
)
```

#### `select_cells_quality_weighted()`
Random sampling with quality-based probability weights.

```python
from cell_selector import select_cells_quality_weighted

# Sample cells weighted by edge distance and area
selected = select_cells_quality_weighted(
    profiles,
    n=100,
    edge_weight=0.4,
    area_weight=0.3,
    random_weight=0.3,
    seed=42
)
```

#### `filter_by_quality_metrics()`
Apply quality filters for edge distance, area, and intensity.

```python
from cell_selector import filter_by_quality_metrics

# Filter by multiple quality metrics
filtered = filter_by_quality_metrics(
    profiles,
    min_edge_dist=50,              # Minimum 50 pixels from edge
    area_range=(500, 5000),         # Cell area between 500-5000 pixels
    intensity_feature='Cells_Intensity_IntegratedIntensity_GFP',
    intensity_range=(1000, 10000)   # GFP intensity range
)
```

#### `find_optimal_intensity_range()`
Find overlapping intensity range between variant and reference.

```python
from cell_selector import find_optimal_intensity_range

# Find GFP intensity range with good overlap
intensity_range = find_optimal_intensity_range(
    variant_profiles,
    ref_profiles,
    intensity_feature='Cells_Intensity_IntegratedIntensity_GFP',
    quantile_range=(0.2, 0.8),  # Use 20-80th percentile
    min_cells_required=20
)

if intensity_range:
    # Filter both groups to this range
    var_filtered = filter_by_quality_metrics(
        variant_profiles,
        intensity_feature='Cells_Intensity_IntegratedIntensity_GFP',
        intensity_range=intensity_range
    )
```

#### `select_phenotype_extreme_cells()`
Select cells that maximize variant vs reference contrast.

```python
from cell_selector import select_phenotype_extreme_cells

# Adaptive selection to maximize phenotypic difference
var_cells, ref_cells = select_phenotype_extreme_cells(
    variant_profiles,
    ref_profiles,
    feature='Cells_Intensity_IntegratedIntensity_GFP',
    n=50,
    adaptive=True  # Automatically choose high/low based on group means
)
```

## Module 2: cell_cropper.py

Image cropping with multiple strategies.

### Key Functions

#### `extract_cell_crop()`
Extract a single channel crop using bbox or fixed-size method.

```python
from cell_cropper import extract_cell_crop

# Method 1: Bounding box with recentering and resize
crop_bbox = extract_cell_crop(
    cell_row,
    channel='GFP',
    imgs_dir='/path/to/images',
    method='bbox',
    recenter=True,    # Center nuclei at crop center
    target_size=128   # Resize to 128x128
)

# Method 2: Fixed-size centered extraction
crop_fixed = extract_cell_crop(
    cell_row,
    channel='DAPI',
    imgs_dir='/path/to/images',
    method='fixed',
    crop_size=64      # Extract 64x64 region around nuclei
)
```

#### `load_multichannel_cell_crops()`
Load crops for multiple channels for a single cell.

```python
from cell_cropper import load_multichannel_cell_crops

# Extract all channels
cell_crops = load_multichannel_cell_crops(
    cell_row,
    channels=['DAPI', 'AGP', 'Mito', 'GFP'],
    imgs_dir='/path/to/images',
    method='bbox',
    target_size=128,
    recenter=True
)

# Access individual channels
dapi_crop = cell_crops['DAPI']
gfp_crop = cell_crops['GFP']
```

#### `batch_extract_cell_crops()`
Extract crops for multiple cells.

```python
from cell_cropper import batch_extract_cell_crops

# Extract crops for 100 cells
all_crops = batch_extract_cell_crops(
    selected_cells,
    channels=['DAPI', 'GFP', 'Mito', 'AGP'],
    imgs_dir='/path/to/images',
    method='bbox',
    target_size=128
)

# Access specific cell and channel
cell_id = 12345
gfp_crop = all_crops[cell_id]['GFP']
```

### Cropping Methods

**Bounding Box Method (`method='bbox'`)**:
1. Extracts cell based on CellProfiler bounding box
2. Optionally recenters to put nuclei at image center
3. Resizes to target dimensions
4. Best for: Preserving full cell morphology

**Fixed-Size Method (`method='fixed'`)**:
1. Extracts fixed-size region centered on nuclei
2. No resizing needed
3. May clip large cells
4. Best for: Consistent dimensions, focusing on nuclei

## Module 3: cell_visualizer.py

Visualization with configurable contrast control.

### Key Functions

#### `normalize_channel()`
Flexible channel normalization.

```python
from cell_visualizer import normalize_channel

# Percentile-based (most common)
norm = normalize_channel(img, method='percentile', percentile=99.5)

# Min-max normalization
norm = normalize_channel(img, method='minmax')

# Manual bounds
norm = normalize_channel(img, method='manual', vmin=100, vmax=5000)
```

#### `viz_cell_single_channel()`
Display a single channel with adjustable contrast.

```python
from cell_visualizer import viz_cell_single_channel
import matplotlib.pyplot as plt

# Test different contrast levels
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, percentile in enumerate([95, 97, 99, 99.5, 99.9]):
    viz_cell_single_channel(
        crop, 'GFP',
        contrast_percentile=percentile,
        ax=axes[i],
        title=f'p={percentile}'
    )
```

#### `viz_cell_multi_channel()`
Display multi-channel RGB merged image.

```python
from cell_visualizer import viz_cell_multi_channel

# Morphology-focused (Mito=Red, AGP=Green, DAPI=Blue)
viz_cell_multi_channel(
    cell_crops,
    channel_mapping='morphology',
    contrast_percentiles=99.5
)

# GFP-inclusive RGB
viz_cell_multi_channel(
    cell_crops,
    channel_mapping='gfp_inclusive',
    contrast_percentiles={'DAPI': 99.9, 'GFP': 95.0, 'Mito': 99.5, 'AGP': 98.0}
)
```

#### `viz_cell_grid()`
Create grid showing individual channels and RGB merge.

```python
from cell_visualizer import viz_cell_grid

# Complete visualization with metadata
fig = viz_cell_grid(
    cell_crops,
    channels=['DAPI', 'AGP', 'Mito', 'GFP'],
    channel_mapping='morphology',
    contrast_percentiles=99.5,
    cell_info={
        'Variant': 'BRCA1_V1',
        'GFP Intensity': 8542.3,
        'AUROC': 0.87
    },
    save_path='cell_visualization.png',
    dpi=300
)
```

#### `plot_cell_comparison()`
Side-by-side comparison of variant and reference cells.

```python
from cell_visualizer import plot_cell_comparison

# Compare variant vs reference
fig = plot_cell_comparison(
    variant_crops,
    ref_crops,
    channels=['DAPI', 'AGP', 'Mito', 'GFP'],
    channel_mapping='morphology',
    contrast_percentiles=99.5,
    variant_label='BRCA1_V1',
    ref_label='BRCA1_WT',
    save_path='comparison.png'
)
```

### Channel Mapping Options

**`'morphology'`** (default):
- Red: Mitochondria
- Green: Golgi/ER (AGP)
- Blue: Nuclei (DAPI)
- *GFP not included*

**`'gfp_inclusive'`**:
- Red: (Mito + AGP) / 2
- Green: (GFP + AGP) / 2
- Blue: DAPI
- *Includes GFP signal*

**Custom mapping**:
```python
custom_mapping = {
    'DAPI': (0, 0, 1),      # Blue channel
    'GFP': (0, 1, 0),       # Green channel
    'Mito': (1, 0, 0),      # Red channel
    'AGP': (1, 1, 0)        # Yellow (red + green)
}

viz_cell_multi_channel(
    cell_crops,
    channel_mapping=custom_mapping,
    contrast_percentiles=99
)
```

## Contrast Control

All visualization functions support flexible contrast adjustment:

### Global Percentile
Apply same percentile to all channels:
```python
viz_cell_grid(cell_crops, contrast_percentiles=99.5)
```

### Per-Channel Percentiles
Optimize each channel independently:
```python
viz_cell_grid(
    cell_crops,
    contrast_percentiles={
        'DAPI': 99.9,   # High contrast for nuclei
        'AGP': 98.0,    # Lower for bright Golgi
        'Mito': 99.5,   # Medium-high for mitochondria
        'GFP': 95.0     # Lower for variable GFP
    }
)
```

### Testing Different Contrasts
```python
# Test range of percentiles
for p in [95, 97, 99, 99.5, 99.9]:
    viz_cell_grid(
        cell_crops,
        contrast_percentiles=p,
        save_path=f'cell_p{p}.png'
    )
```

## Complete Workflows

### Workflow 1: Visualize Top Cells by Feature

```python
import polars as pl
from cell_selector import select_cells_top_n, filter_by_quality_metrics, compute_distance_to_edge
from cell_cropper import load_multichannel_cell_crops
from cell_visualizer import viz_cell_grid

# Load and prepare profiles
profiles = pl.read_parquet("batch_profiles.parquet")
profiles = profiles.with_columns(compute_distance_to_edge().alias("dist2edge"))

# Select and filter cells
cells = select_cells_top_n(
    profiles,
    feature="Cells_Intensity_IntegratedIntensity_GFP",
    n=10,
    direction="high"
)
cells = filter_by_quality_metrics(cells, min_edge_dist=50)

# Visualize each cell
for i, cell_row in enumerate(cells.iter_rows(named=True)):
    crops = load_multichannel_cell_crops(
        cell_row,
        channels=['DAPI', 'AGP', 'Mito', 'GFP'],
        imgs_dir="/path/to/images",
        method='bbox',
        target_size=128
    )

    viz_cell_grid(
        crops,
        contrast_percentiles=99.5,
        save_path=f'cell_{i:03d}.png'
    )
```

### Workflow 2: Variant vs Reference Comparison

```python
from cell_selector import (
    filter_cells_by_metadata,
    filter_by_quality_metrics,
    select_phenotype_extreme_cells
)
from cell_cropper import load_multichannel_cell_crops
from cell_visualizer import plot_cell_comparison

# Load profiles
profiles = pl.read_parquet("batch_profiles.parquet")

# Filter to variant and reference
var_profiles = filter_cells_by_metadata(profiles, allele='BRCA1_V1')
ref_profiles = filter_cells_by_metadata(profiles, allele='BRCA1_WT')

# Apply quality filters
var_profiles = filter_by_quality_metrics(var_profiles, min_edge_dist=50)
ref_profiles = filter_by_quality_metrics(ref_profiles, min_edge_dist=50)

# Select phenotype-extreme cells
var_cells, ref_cells = select_phenotype_extreme_cells(
    var_profiles,
    ref_profiles,
    feature='Cells_Intensity_IntegratedIntensity_GFP',
    n=10,
    adaptive=True
)

# Compare first cells
var_crops = load_multichannel_cell_crops(
    var_cells[0], ['DAPI', 'AGP', 'Mito', 'GFP'],
    imgs_dir="/path/to/images", method='bbox', target_size=128
)

ref_crops = load_multichannel_cell_crops(
    ref_cells[0], ['DAPI', 'AGP', 'Mito', 'GFP'],
    imgs_dir="/path/to/images", method='bbox', target_size=128
)

plot_cell_comparison(
    var_crops, ref_crops,
    contrast_percentiles=99.5,
    variant_label='BRCA1_V1',
    ref_label='BRCA1_WT',
    save_path='variant_vs_ref.png'
)
```

### Workflow 3: Contrast Optimization

```python
from cell_cropper import load_multichannel_cell_crops
from cell_visualizer import viz_cell_grid

# Extract crops
cell_crops = load_multichannel_cell_crops(
    cell_row, ['DAPI', 'AGP', 'Mito', 'GFP'],
    imgs_dir="/path/to/images", method='bbox', target_size=128
)

# Test global percentiles
for p in [95, 97, 99, 99.5, 99.9]:
    viz_cell_grid(
        cell_crops,
        contrast_percentiles=p,
        save_path=f'contrast_global_p{p}.png'
    )

# Test per-channel optimization
strategies = {
    'conservative': {'DAPI': 99.9, 'AGP': 99.0, 'Mito': 99.5, 'GFP': 99.0},
    'aggressive': {'DAPI': 99.9, 'AGP': 97.0, 'Mito': 99.0, 'GFP': 95.0},
    'balanced': {'DAPI': 99.9, 'AGP': 98.0, 'Mito': 99.5, 'GFP': 97.0}
}

for name, percentiles in strategies.items():
    viz_cell_grid(
        cell_crops,
        contrast_percentiles=percentiles,
        save_path=f'contrast_strategy_{name}.png'
    )
```

## Tips and Best Practices

### Cell Selection
- **Always apply quality filters** (edge distance, area) before visualization
- **Use adaptive selection** for variant vs reference to maximize contrast
- **Find optimal intensity ranges** when comparing GFP between groups

### Cell Cropping
- **Bounding box method**: Best for preserving full cell morphology
- **Fixed-size method**: Best for consistent dimensions
- **Always use `recenter=True`** for cleaner visualizations
- **Target size 128x128**: Good balance of detail and file size

### Contrast Adjustment
- **Start with 99.5 percentile** - good default for most images
- **Use per-channel control** for images with varying brightness
- **AGP often needs lower percentiles** (97-98) due to bright puncta
- **GFP may need lower percentiles** (95-97) for high-expressing cells
- **Test multiple values** - visual inspection is key

### Visualization
- **Use `viz_cell_grid()`** for comprehensive single-cell view
- **Use `plot_cell_comparison()`** for variant vs reference
- **Save at 150-300 DPI** for publication-quality figures
- **Include cell metadata** in `cell_info` parameter

## Troubleshooting

**Q: Images not loading**
- Check that `imgs_dir` path is correct
- Verify plate_map_name is in plate_dict (img_utils.py)
- Ensure TIFF files exist at expected locations

**Q: Crops are all black or white**
- Try different contrast percentiles (95, 97, 99)
- Check that cells have valid intensity values
- Verify images loaded correctly (not corrupted)

**Q: Cells off-center**
- Use `recenter=True` in cropping functions
- Check that nuclei coordinates are valid
- Try fixed-size method instead of bounding box

**Q: RGB merge looks wrong**
- Verify all channels loaded successfully
- Try different channel_mapping options
- Adjust per-channel contrast percentiles

## Migration from Legacy Code

If you're using the old `display_cells.py` or `display_img.py`:

**Old code:**
```python
from display_cells import viz_cell_crop_multi
viz_cell_crop_multi(cell_dict, max_intensity=0.99)
```

**New code:**
```python
from cell_visualizer import viz_cell_multi_channel
viz_cell_multi_channel(cell_crops, contrast_percentiles=99.0)
```

Key changes:
- `max_intensity` → `contrast_percentiles`
- Better channel handling and RGB merging
- Configurable contrast per channel
- Cleaner function interfaces

## Examples

See `example_cell_visualization.py` for complete working examples of all functionality.

Run examples:
```bash
cd 3.downstream_analyses/scripts
python example_cell_visualization.py
```

## Support

For questions or issues, refer to:
- This README
- Function docstrings in each module
- Example script: `example_cell_visualization.py`
- Legacy backup: `display_cells_backup.py`
