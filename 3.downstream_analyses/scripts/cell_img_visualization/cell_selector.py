"""
Cell Selection Module for VarChAMP

This module provides functions for selecting and filtering cells from CellProfiler
profiles based on various criteria including feature values, quality metrics, and
intensity ranges.

Author: VarChAMP Pipeline
Date: 2025
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Union, Dict


def filter_cells_by_metadata(
    profiles: pl.DataFrame,
    allele: Optional[str] = None,
    plate: Optional[str] = None,
    well: Optional[Union[str, List[str]]] = None,
    site: Optional[str] = None,
) -> pl.DataFrame:
    """
    Filter cells by metadata fields (allele, plate, well, site).

    Parameters:
    -----------
    profiles : pl.DataFrame
        Cell profiles dataframe with metadata columns
    allele : str, optional
        Gene allele identifier (e.g., 'BRCA1_V1')
    plate : str, optional
        Plate identifier
    well : str or list, optional
        Well identifier(s) (e.g., 'A01' or ['A01', 'B02'])
    site : str, optional
        Site/FOV identifier (e.g., '05')

    Returns:
    --------
    pl.DataFrame : Filtered profiles
    """
    filtered = profiles

    if allele is not None:
        filtered = filtered.filter(pl.col("Metadata_gene_allele") == allele)

    if plate is not None:
        filtered = filtered.filter(pl.col("Metadata_Plate") == plate)

    if well is not None:
        if isinstance(well, list):
            filtered = filtered.filter(pl.col("Metadata_well_position").is_in(well))
        else:
            filtered = filtered.filter(pl.col("Metadata_well_position") == well)

    if site is not None:
        filtered = filtered.filter(pl.col("Metadata_Site") == site)

    return filtered


def select_cells_top_n(
    profiles: pl.DataFrame,
    feature: str,
    n: int = 50,
    direction: str = "high",
    group_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Select top N cells by feature value (highest or lowest).

    This is useful for selecting cells with extreme phenotypes.

    Parameters:
    -----------
    profiles : pl.DataFrame
        Cell profiles dataframe
    feature : str
        Feature column name to sort by
    n : int
        Number of cells to select
    direction : str
        'high' for highest values, 'low' for lowest values
    group_col : str, optional
        Column to group by before selection (e.g., 'Metadata_gene_allele')
        If provided, selects top N from each group

    Returns:
    --------
    pl.DataFrame : Selected cells
    """
    if group_col is not None:
        # Select top N from each group
        return (profiles
                .sort(feature, descending=(direction == "high"))
                .group_by(group_col)
                .head(n))
    else:
        # Select top N overall
        return (profiles
                .sort(feature, descending=(direction == "high"))
                .head(n))


def select_cells_percentile_random(
    profiles: pl.DataFrame,
    feature: str,
    percentile_bins: List[Tuple[float, float]],
    n_per_bin: int,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Randomly sample cells from specified percentile ranges.

    Useful for getting representative samples across feature value ranges.

    Parameters:
    -----------
    profiles : pl.DataFrame
        Cell profiles dataframe
    feature : str
        Feature column name to use for percentile calculation
    percentile_bins : list of tuples
        List of (lower, upper) percentile pairs, e.g., [(0, 10), (45, 55), (90, 100)]
    n_per_bin : int
        Number of cells to randomly sample from each bin
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pl.DataFrame : Selected cells from all bins

    Example:
    --------
    # Sample 20 cells each from bottom 10%, middle 40-60%, and top 10%
    selected = select_cells_percentile_random(
        profiles,
        feature="Cells_Intensity_IntegratedIntensity_GFP",
        percentile_bins=[(0, 10), (45, 55), (90, 100)],
        n_per_bin=20,
        seed=42
    )
    """
    selected_cells = []

    for lower_p, upper_p in percentile_bins:
        # Calculate percentile thresholds
        lower_val = profiles[feature].quantile(lower_p / 100.0)
        upper_val = profiles[feature].quantile(upper_p / 100.0)

        # Filter cells in this percentile range
        bin_cells = profiles.filter(
            (pl.col(feature) >= lower_val) & (pl.col(feature) <= upper_val)
        )

        # Randomly sample n_per_bin cells
        if len(bin_cells) > n_per_bin:
            sampled = bin_cells.sample(n=n_per_bin, seed=seed)
        else:
            sampled = bin_cells

        selected_cells.append(sampled)

    return pl.concat(selected_cells, how="vertical")


def select_cells_quality_weighted(
    profiles: pl.DataFrame,
    n: int,
    edge_dist_col: str = "dist2edge",
    area_col: str = "Cells_AreaShape_Area",
    edge_weight: float = 0.4,
    area_weight: float = 0.3,
    random_weight: float = 0.3,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Randomly sample cells with probability weighted by quality metrics.

    Quality score = (normalized_edge_dist * edge_weight +
                     normalized_area_score * area_weight +
                     random * random_weight)

    Parameters:
    -----------
    profiles : pl.DataFrame
        Cell profiles dataframe
    n : int
        Number of cells to select
    edge_dist_col : str
        Column name for edge distance
    area_col : str
        Column name for cell area
    edge_weight : float
        Weight for edge distance (0-1)
    area_weight : float
        Weight for area score (0-1)
    random_weight : float
        Weight for random component (0-1) - adds stochasticity
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pl.DataFrame : Selected cells
    """
    # Normalize edge distance to [0, 1]
    edge_min = profiles[edge_dist_col].min()
    edge_max = profiles[edge_dist_col].max()
    edge_norm = (profiles[edge_dist_col] - edge_min) / (edge_max - edge_min)

    # Normalize area - prefer mid-range cells (not too small, not too large)
    area_median = profiles[area_col].median()
    area_std = profiles[area_col].std()
    area_score = np.exp(-((profiles[area_col] - area_median) / area_std) ** 2)
    area_norm = (area_score - area_score.min()) / (area_score.max() - area_score.min())

    # Add random component
    np.random.seed(seed)
    random_score = np.random.random(len(profiles))

    # Calculate weighted quality score
    quality_score = (
        edge_norm * edge_weight +
        area_norm * area_weight +
        random_score * random_weight
    )

    # Add quality score to dataframe
    profiles_with_score = profiles.with_columns(
        pl.lit(quality_score).alias("_quality_score")
    )

    # Sample with probability proportional to quality score
    # Convert to numpy for weighted sampling
    weights = quality_score / quality_score.sum()
    indices = np.random.choice(len(profiles), size=n, replace=False, p=weights)

    selected = profiles_with_score[indices].drop("_quality_score")

    return selected


def filter_by_quality_metrics(
    profiles: pl.DataFrame,
    min_edge_dist: Optional[float] = None,
    area_range: Optional[Tuple[float, float]] = None,
    intensity_feature: Optional[str] = None,
    intensity_range: Optional[Tuple[float, float]] = None,
    edge_dist_col: str = "dist2edge",
    area_col: str = "Cells_AreaShape_Area",
) -> pl.DataFrame:
    """
    Filter cells by quality metrics (edge distance, area, intensity).

    Parameters:
    -----------
    profiles : pl.DataFrame
        Cell profiles dataframe
    min_edge_dist : float, optional
        Minimum distance from image edge (e.g., 50 pixels)
    area_range : tuple, optional
        (min_area, max_area) in pixels (e.g., (500, 5000))
    intensity_feature : str, optional
        Intensity feature column name for filtering
    intensity_range : tuple, optional
        (min_intensity, max_intensity) for the intensity feature
    edge_dist_col : str
        Column name for edge distance metric
    area_col : str
        Column name for cell area

    Returns:
    --------
    pl.DataFrame : Filtered profiles
    """
    filtered = profiles

    if min_edge_dist is not None:
        filtered = filtered.filter(pl.col(edge_dist_col) >= min_edge_dist)

    if area_range is not None:
        min_area, max_area = area_range
        filtered = filtered.filter(
            (pl.col(area_col) >= min_area) & (pl.col(area_col) <= max_area)
        )

    if intensity_feature is not None and intensity_range is not None:
        min_intensity, max_intensity = intensity_range
        filtered = filtered.filter(
            (pl.col(intensity_feature) >= min_intensity) &
            (pl.col(intensity_feature) <= max_intensity)
        )

    return filtered


def compute_distance_to_edge(
    x_col: str = "Nuclei_AreaShape_Center_X",
    y_col: str = "Nuclei_AreaShape_Center_Y",
    img_width: int = 2160,
    img_height: int = 2160,
) -> pl.Expr:
    """
    Create a Polars expression to compute distance to nearest image edge.

    Parameters:
    -----------
    x_col : str
        Column name for X coordinate
    y_col : str
        Column name for Y coordinate
    img_width : int
        Image width in pixels
    img_height : int
        Image height in pixels

    Returns:
    --------
    pl.Expr : Polars expression for distance calculation

    Example:
    --------
    profiles = profiles.with_columns(
        compute_distance_to_edge().alias("dist2edge")
    )
    """
    dist_left = pl.col(x_col)
    dist_right = img_width - pl.col(x_col)
    dist_top = pl.col(y_col)
    dist_bottom = img_height - pl.col(y_col)

    return pl.min_horizontal(dist_left, dist_right, dist_top, dist_bottom)


def find_optimal_intensity_range(
    variant_profiles: pl.DataFrame,
    ref_profiles: pl.DataFrame,
    intensity_feature: str,
    quantile_range: Tuple[float, float] = (0.2, 0.8),
    min_cells_required: int = 20,
) -> Optional[Tuple[float, float]]:
    """
    Find overlapping intensity range between variant and reference cells.

    This ensures that cells are compared within similar intensity ranges,
    which is important for GFP-based analyses where expression level matters.

    Parameters:
    -----------
    variant_profiles : pl.DataFrame
        Variant cell profiles
    ref_profiles : pl.DataFrame
        Reference cell profiles
    intensity_feature : str
        Intensity feature column name (e.g., 'Cells_Intensity_IntegratedIntensity_GFP')
    quantile_range : tuple
        (lower_quantile, upper_quantile) to use for range finding (default: 0.2 to 0.8)
    min_cells_required : int
        Minimum number of cells required in the overlapping range

    Returns:
    --------
    tuple or None : (min_intensity, max_intensity) or None if insufficient overlap

    Example:
    --------
    intensity_range = find_optimal_intensity_range(
        variant_profiles,
        ref_profiles,
        intensity_feature='Cells_Intensity_IntegratedIntensity_GFP',
        quantile_range=(0.2, 0.8)
    )
    if intensity_range:
        variant_filtered = filter_by_quality_metrics(
            variant_profiles,
            intensity_feature='Cells_Intensity_IntegratedIntensity_GFP',
            intensity_range=intensity_range
        )
    """
    # Calculate quantiles for both groups
    var_lower = variant_profiles[intensity_feature].quantile(quantile_range[0])
    var_upper = variant_profiles[intensity_feature].quantile(quantile_range[1])

    ref_lower = ref_profiles[intensity_feature].quantile(quantile_range[0])
    ref_upper = ref_profiles[intensity_feature].quantile(quantile_range[1])

    # Find overlapping range
    overlap_min = max(var_lower, ref_lower)
    overlap_max = min(var_upper, ref_upper)

    # Check if there's meaningful overlap
    if overlap_min >= overlap_max:
        return None

    # Check if enough cells fall in this range
    var_in_range = variant_profiles.filter(
        (pl.col(intensity_feature) >= overlap_min) &
        (pl.col(intensity_feature) <= overlap_max)
    )
    ref_in_range = ref_profiles.filter(
        (pl.col(intensity_feature) >= overlap_min) &
        (pl.col(intensity_feature) <= overlap_max)
    )

    if len(var_in_range) < min_cells_required or len(ref_in_range) < min_cells_required:
        return None

    return (overlap_min, overlap_max)


def select_phenotype_extreme_cells(
    variant_profiles: pl.DataFrame,
    ref_profiles: pl.DataFrame,
    feature: str,
    n: int = 50,
    adaptive: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Select cells with extreme phenotypes that maximize variant vs reference contrast.

    If variant mean > reference mean, select high-value variant cells and low-value ref cells.
    If variant mean < reference mean, select low-value variant cells and high-value ref cells.

    Parameters:
    -----------
    variant_profiles : pl.DataFrame
        Variant cell profiles
    ref_profiles : pl.DataFrame
        Reference cell profiles
    feature : str
        Feature to use for selection
    n : int
        Number of cells to select from each group
    adaptive : bool
        If True, adapt selection direction based on group means
        If False, always select high values for both

    Returns:
    --------
    tuple : (selected_variant_cells, selected_ref_cells)

    Example:
    --------
    var_cells, ref_cells = select_phenotype_extreme_cells(
        variant_profiles,
        ref_profiles,
        feature="Cells_Intensity_IntegratedIntensity_GFP",
        n=100
    )
    """
    if not adaptive:
        # Simple: select top N from both groups
        var_selected = select_cells_top_n(variant_profiles, feature, n, direction="high")
        ref_selected = select_cells_top_n(ref_profiles, feature, n, direction="high")
    else:
        # Adaptive: maximize phenotypic contrast
        var_mean = variant_profiles[feature].mean()
        ref_mean = ref_profiles[feature].mean()

        if var_mean > ref_mean:
            # Variant is higher - select high variant, low reference
            var_selected = select_cells_top_n(variant_profiles, feature, n, direction="high")
            ref_selected = select_cells_top_n(ref_profiles, feature, n, direction="low")
        else:
            # Variant is lower - select low variant, high reference
            var_selected = select_cells_top_n(variant_profiles, feature, n, direction="low")
            ref_selected = select_cells_top_n(ref_profiles, feature, n, direction="high")

    return var_selected, ref_selected
