"""
Shared helpers for downstream analysis scripts.

Factored out of the various generate_*.py / analyze_*.py scripts that all
need to load platemaps, well QC flags, and batch cell profiles from the
pipeline outputs. Every helper is keyed on a batch string such as
"2026_01_05_Batch_20" and resolves paths relative to PROJECT_ROOT.
"""

from pathlib import Path
from typing import Optional

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = PROJECT_ROOT / "2.snakemake_pipeline"
PLATEMAP_DIR = PIPELINE_DIR / "inputs" / "metadata" / "platemaps"
PLATE_BG_DIR = PROJECT_ROOT / "1.image_preprocess_qc" / "outputs" / "plate_bg_summary"
BATCH_PROFILES_DIR = PIPELINE_DIR / "outputs" / "batch_profiles"

BBOX_PROFILE_COLUMNS = [
    "Metadata_Plate",
    "Metadata_plate_map_name",
    "Metadata_well_position",
    "Metadata_ImageNumber",
    "Metadata_ObjectNumber",
    "Metadata_gene_allele",
    "Metadata_symbol",
    "Metadata_node_type",
    "Nuclei_AreaShape_Center_X",
    "Nuclei_AreaShape_Center_Y",
    "Nuclei_AreaShape_BoundingBoxMinimum_X",
    "Nuclei_AreaShape_BoundingBoxMinimum_Y",
    "Nuclei_AreaShape_BoundingBoxMaximum_X",
    "Nuclei_AreaShape_BoundingBoxMaximum_Y",
    "Cells_AreaShape_BoundingBoxMinimum_X",
    "Cells_AreaShape_BoundingBoxMinimum_Y",
    "Cells_AreaShape_BoundingBoxMaximum_X",
    "Cells_AreaShape_BoundingBoxMaximum_Y",
    "Cells_AreaShape_Area",
]


def load_platemap(batch: str) -> pl.DataFrame:
    """Concatenate all platemap .txt files for a batch. Empty DataFrame if none."""
    platemap_dir = PLATEMAP_DIR / batch / "platemap"
    platemap_files = list(platemap_dir.glob("*.txt"))
    if not platemap_files:
        return pl.DataFrame()
    dfs = [pl.read_csv(f, separator="\t") for f in platemap_files]
    return pl.concat(dfs, how="diagonal_relaxed")


def load_well_qc(batch: str) -> Optional[pl.DataFrame]:
    """Return per-well QC flags for a batch, or None if the parquet is missing."""
    qc_path = PLATE_BG_DIR / batch / "well_qc_flags.parquet"
    if qc_path.exists():
        return pl.read_parquet(qc_path)
    return None


def get_well_qc_flag(
    well_qc: Optional[pl.DataFrame], plate: str, well: str, channel: str
) -> bool:
    """Return True if a (plate, well, channel) is flagged as background-only.

    `plate` may be a full measurement name like "<plate>__<timestamp>"; only
    the portion before "__" is used to match the QC table's `plate` column.
    """
    if well_qc is None:
        return False
    plate_short = plate.split("__")[0]
    qc_row = well_qc.filter(
        (pl.col("plate") == plate_short)
        & (pl.col("well") == well)
        & (pl.col("channel") == channel)
    )
    if len(qc_row) > 0:
        return qc_row["is_bg"][0]
    return False


def load_batch_profiles(batch: str) -> pl.DataFrame:
    """Load profiles.parquet for a batch (all columns). Empty DataFrame if missing."""
    profiles_path = BATCH_PROFILES_DIR / batch / "profiles.parquet"
    if profiles_path.exists():
        return pl.read_parquet(profiles_path)
    return pl.DataFrame()


def load_batch_profiles_with_bbox(
    batch: str, sites_per_well: int = 9, img_size: int = 1080
) -> pl.DataFrame:
    """Load the metadata + bounding-box columns needed for single-cell cropping.

    Adds two derived columns:
      - Metadata_Site: derived from Metadata_ImageNumber mod `sites_per_well`
      - dist2edge: distance of the nucleus centroid to the nearest image edge,
        using a square `img_size x img_size` frame
    """
    profile_path = BATCH_PROFILES_DIR / batch / "profiles.parquet"
    if not profile_path.exists():
        return pl.DataFrame()

    profiles = pl.scan_parquet(profile_path).select(BBOX_PROFILE_COLUMNS).collect()
    return profiles.with_columns(
        (((pl.col("Metadata_ImageNumber") - 1) % sites_per_well) + 1).alias("Metadata_Site"),
        pl.min_horizontal(
            pl.col("Nuclei_AreaShape_Center_X"),
            pl.col("Nuclei_AreaShape_Center_Y"),
            pl.lit(img_size) - pl.col("Nuclei_AreaShape_Center_X"),
            pl.lit(img_size) - pl.col("Nuclei_AreaShape_Center_Y"),
        ).alias("dist2edge"),
    )
