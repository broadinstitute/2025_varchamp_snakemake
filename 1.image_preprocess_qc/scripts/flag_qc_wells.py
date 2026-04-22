#!/usr/bin/env python
"""Flag poor-quality imaging wells using signal-to-noise ratio and triangle thresholding.

For each batch:
1. Load well-level and plate-level intensity statistics
2. Compute signal-to-noise ratio: s2n_ratio = log10(perc_99 / plate_median_perc_50 + offset)
3. Use triangle thresholding to find optimal noise threshold per channel
4. Flag wells where s2n_ratio <= threshold as poor quality (is_bg=True)

Usage:
    python flag_qc_wells.py \
        --batch_ids "2026_01_05_Batch_20,2026_01_05_Batch_21" \
        --input_dir "../outputs/plate_bg_summary" \
        --output_dir "../outputs/plate_bg_summary"
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
from skimage.filters import threshold_triangle

CHANNELS_TO_PROCESS = ["DAPI", "GFP", "AGP", "Mito"]


def get_left_tail_threshold(data: np.ndarray) -> float:
    """Calculate noise threshold using Triangle method for left tail."""
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return 0.0

    max_val = np.max(data)
    inverted_data = max_val - data
    thresh_inverted = threshold_triangle(inverted_data)
    return max_val - thresh_inverted


def load_batch_stats(input_dir: Path, batch_id: str):
    """Load plate and well statistics for a batch."""
    plate_path = input_dir / batch_id / "plate_sum_stats.parquet"
    well_path = input_dir / batch_id / "plate_well_sum_stats.parquet"

    if not plate_path.exists() or not well_path.exists():
        print(f"Warning: Stats not found for batch {batch_id}")
        return None

    plate_stats = pl.read_parquet(plate_path)
    well_stats = pl.read_parquet(well_path)
    return plate_stats, well_stats


def compute_s2n_ratio(well_stats: pl.DataFrame, plate_stats: pl.DataFrame, batch_id: str) -> pl.DataFrame:
    """Compute signal-to-noise ratio for each well/channel."""
    well_stats = well_stats.with_columns(pl.lit(batch_id).alias("Metadata_Batch"))

    # Join with plate stats
    plate_median = plate_stats.select([
        "plate", "channel",
        pl.col("perc_50").alias("median_plate")
    ])

    well_stats = well_stats.join(plate_median, on=["plate", "channel"], how="left")

    result_dfs = []
    for plate in well_stats["plate"].unique().to_list():
        for channel in CHANNELS_TO_PROCESS:
            plate_channel_df = well_stats.filter(
                (pl.col("plate") == plate) & (pl.col("channel") == channel)
            )

            if len(plate_channel_df) == 0:
                continue

            plate_channel_df = plate_channel_df.with_columns(
                (pl.col("perc_99") / pl.col("median_plate")).alias("s2n_ratio_raw")
            )

            plate_channel_df = plate_channel_df.filter(
                (~pl.col("s2n_ratio_raw").is_infinite()) &
                (~pl.col("s2n_ratio_raw").is_nan())
            )

            if len(plate_channel_df) == 0:
                continue

            positive_values = plate_channel_df.filter(pl.col("s2n_ratio_raw") > 0)
            if len(positive_values) == 0:
                offset = 1e-6
            else:
                min_val = positive_values["s2n_ratio_raw"].min()
                offset = min_val / 2.0

            plate_channel_df = plate_channel_df.with_columns(
                np.log10(pl.col("s2n_ratio_raw") + offset).alias("s2n_ratio")
            )

            result_dfs.append(plate_channel_df)

    if not result_dfs:
        return pl.DataFrame()

    return pl.concat(result_dfs)


def flag_wells_with_thresholds(all_wells: pl.DataFrame, channel_thresholds: dict) -> pl.DataFrame:
    """Flag wells as poor quality based on channel thresholds."""
    result_dfs = []

    for channel, threshold in channel_thresholds.items():
        channel_df = all_wells.filter(pl.col("channel") == channel)
        if len(channel_df) == 0:
            continue

        channel_df = channel_df.with_columns(
            (pl.col("s2n_ratio") <= threshold).alias("is_bg")
        )
        result_dfs.append(channel_df)

    if not result_dfs:
        return pl.DataFrame()

    return pl.concat(result_dfs)


def process_batches(batch_ids: list, input_dir: Path, output_dir: Path):
    """Process multiple batches."""
    print(f"\n{'='*60}")
    print(f"Processing {len(batch_ids)} batches")
    print(f"{'='*60}")

    all_wells = pl.DataFrame()

    for batch_id in batch_ids:
        result = load_batch_stats(input_dir, batch_id)
        if result is None:
            continue

        plate_stats, well_stats = result
        print(f"  {batch_id}: {len(well_stats)} well/channel combinations")

        batch_wells = compute_s2n_ratio(well_stats, plate_stats, batch_id)
        if len(batch_wells) > 0:
            all_wells = pl.concat([all_wells, batch_wells], how="diagonal_relaxed")

    if len(all_wells) == 0:
        print("Error: No valid data found")
        return

    print(f"\nTotal: {len(all_wells)} well/channel combinations")

    # Compute thresholds
    print("\nComputing channel thresholds...")
    channel_thresholds = {}

    for channel in CHANNELS_TO_PROCESS:
        channel_data = all_wells.filter(pl.col("channel") == channel)
        if len(channel_data) == 0:
            continue

        s2n_values = channel_data["s2n_ratio"].to_numpy()
        threshold = get_left_tail_threshold(s2n_values)
        channel_thresholds[channel] = threshold

        n_below = np.sum(s2n_values <= threshold)
        pct_below = 100 * n_below / len(s2n_values)
        print(f"  {channel}: threshold={threshold:.4f}, flagged={n_below}/{len(s2n_values)} ({pct_below:.1f}%)")

    # Flag wells
    all_wells_flagged = flag_wells_with_thresholds(all_wells, channel_thresholds)

    # Save results
    for batch_id in batch_ids:
        batch_wells = all_wells_flagged.filter(pl.col("Metadata_Batch") == batch_id)
        if len(batch_wells) == 0:
            continue

        batch_output_dir = output_dir / batch_id
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = batch_output_dir / "well_qc_flags.parquet"
        batch_wells.write_parquet(output_path)

        n_flagged = batch_wells.filter(pl.col("is_bg")).height
        n_total = len(batch_wells)
        pct_flagged = 100 * n_flagged / n_total if n_total > 0 else 0

        print(f"  {batch_id}: {n_flagged}/{n_total} ({pct_flagged:.1f}%) flagged -> {output_path}")

    print("\nProcessing complete!")


def main():
    parser = argparse.ArgumentParser(description="Flag poor-quality imaging wells")
    parser.add_argument("--batch_ids", required=True, help="Comma-separated batch IDs")
    parser.add_argument("--input_dir", required=True, help="Input directory with plate stats")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    args = parser.parse_args()

    batch_ids = [b.strip() for b in args.batch_ids.split(",")]
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    process_batches(batch_ids, input_dir, output_dir)


if __name__ == "__main__":
    main()
