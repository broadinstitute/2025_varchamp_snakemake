"""
Cell Cropping Module for VarChAMP

This module provides functions for extracting cell image crops from multi-channel
microscopy images, with support for multiple cropping strategies including bounding
box-based and fixed-size extraction, with optional recentering on nuclei.

Author: VarChAMP Pipeline
Date: 2025
"""

import os
import warnings
import numpy as np
import polars as pl
from skimage.io import imread
from skimage.transform import resize
from typing import Dict, Tuple, Optional, List, Union
import sys

sys.path.append("../..")
from img_utils import letter_dict, channel_dict, plate_dict


def get_crop_coordinates_bbox(
    cell_row: Union[pl.DataFrame, Dict],
    recenter: bool = True,
    bbox_prefix: str = "Cells",
) -> Tuple[int, int, int, int]:
    """
    Calculate crop coordinates based on cell bounding box.

    Parameters:
    -----------
    cell_row : pl.DataFrame or dict
        Single cell row with bounding box columns:
        - {bbox_prefix}_AreaShape_BoundingBoxMinimum_X/Y
        - {bbox_prefix}_AreaShape_BoundingBoxMaximum_X/Y
        - Nuclei_AreaShape_Center_X/Y (if recenter=True)
    recenter : bool
        If True, shift bounding box to center nuclei at crop center
    bbox_prefix : str
        Prefix for bounding box columns ('Cells' or 'Nuclei')

    Returns:
    --------
    tuple : (x_min, y_min, x_max, y_max) crop coordinates
    """
    # Convert to dict if DataFrame row
    if isinstance(cell_row, pl.DataFrame):
        cell_dict = cell_row.to_dicts()[0]
    else:
        cell_dict = cell_row

    # Get bounding box coordinates
    x_min = int(cell_dict[f"{bbox_prefix}_AreaShape_BoundingBoxMinimum_X"])
    y_min = int(cell_dict[f"{bbox_prefix}_AreaShape_BoundingBoxMinimum_Y"])
    x_max = int(cell_dict[f"{bbox_prefix}_AreaShape_BoundingBoxMaximum_X"])
    y_max = int(cell_dict[f"{bbox_prefix}_AreaShape_BoundingBoxMaximum_Y"])

    if recenter:
        # Get nuclei center
        nuc_x = int(cell_dict["Nuclei_AreaShape_Center_X"])
        nuc_y = int(cell_dict["Nuclei_AreaShape_Center_Y"])

        # Calculate current bbox center
        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2

        # Calculate shift needed to center nuclei
        shift_x = nuc_x - bbox_center_x
        shift_y = nuc_y - bbox_center_y

        # Apply shift
        x_min += shift_x
        x_max += shift_x
        y_min += shift_y
        y_max += shift_y

    return x_min, y_min, x_max, y_max


def get_crop_coordinates_fixed(
    cell_row: Union[pl.DataFrame, Dict],
    crop_size: int = 128,
) -> Tuple[int, int, int, int]:
    """
    Calculate crop coordinates for fixed-size extraction centered on nuclei.

    Parameters:
    -----------
    cell_row : pl.DataFrame or dict
        Single cell row with nuclei center columns:
        - Nuclei_AreaShape_Center_X
        - Nuclei_AreaShape_Center_Y
    crop_size : int
        Size of square crop in pixels (e.g., 64, 128, 256)

    Returns:
    --------
    tuple : (x_min, y_min, x_max, y_max) crop coordinates
    """
    # Convert to dict if DataFrame row
    if isinstance(cell_row, pl.DataFrame):
        cell_dict = cell_row.to_dicts()[0]
    else:
        cell_dict = cell_row

    # Get nuclei center
    nuc_x = int(cell_dict["Nuclei_AreaShape_Center_X"])
    nuc_y = int(cell_dict["Nuclei_AreaShape_Center_Y"])

    # Calculate crop boundaries
    half_size = crop_size // 2
    x_min = nuc_x - half_size
    x_max = nuc_x + half_size
    y_min = nuc_y - half_size
    y_max = nuc_y + half_size

    return x_min, y_min, x_max, y_max


def validate_and_pad_crop(
    img: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    pad_value: float = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Validate crop coordinates and pad if they extend beyond image boundaries.

    Parameters:
    -----------
    img : np.ndarray
        Input image (2D)
    x_min, y_min, x_max, y_max : int
        Crop coordinates
    pad_value : float
        Value to use for padding (default: 0)

    Returns:
    --------
    tuple : (cropped_image, adjusted_coordinates)
        - cropped_image may be padded if original coords were out of bounds
        - adjusted_coordinates are the actual coordinates used (clipped to image)
    """
    img_height, img_width = img.shape

    # Calculate padding needed
    pad_left = max(0, -x_min)
    pad_right = max(0, x_max - img_width)
    pad_top = max(0, -y_min)
    pad_bottom = max(0, y_max - img_height)

    # Clip coordinates to image boundaries
    x_min_clipped = max(0, x_min)
    x_max_clipped = min(img_width, x_max)
    y_min_clipped = max(0, y_min)
    y_max_clipped = min(img_height, y_max)

    # Extract the valid region
    crop = img[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped]

    # Add padding if needed
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=pad_value
        )

    return crop, (x_min_clipped, y_min_clipped, x_max_clipped, y_max_clipped)


def get_image_path(
    batch_id: str,
    plate_map_name: str,
    well: str,
    site: str,
    channel: str,
    imgs_dir: str,
) -> str:
    """
    Construct image file path from metadata.

    Parameters:
    -----------
    batch_id : str
        Batch identifier from Metadata_Plate (e.g., '2025_01_27_B13A7A8P1_T1')
        Used to extract timepoint information
    plate_map_name : str
        Plate map name (e.g., 'B13A7A8P1_R1')
    well : str
        Well identifier (e.g., 'A01' or 'M18')
    site : str
        Site/FOV identifier (e.g., '05' or '1')
    channel : str
        Channel name (e.g., 'DAPI', 'GFP', 'Mito', 'AGP')
    imgs_dir : str
        Base directory for images

    Returns:
    --------
    str : Full path to image file
    """
    from img_utils import batch_dict

    # Convert well to row/col format
    row_letter = well[0]
    col_num = well[1:3]
    row_num = letter_dict[row_letter]

    # Get channel number
    channel_num = channel_dict[channel]

    # Extract plate map prefix (e.g., 'B13A7A8P1_R1' -> 'B13A7A8P1')
    if "_" in plate_map_name:
        plate_prefix = plate_map_name.split("_")[0]
    else:
        plate_prefix = plate_map_name

    # Get the actual batch directory name from batch_dict
    actual_batch_dir = batch_dict.get(plate_prefix)
    if actual_batch_dir is None:
        raise ValueError(f"Plate prefix '{plate_prefix}' from plate_map_name '{plate_map_name}' not found in batch_dict")

    # Get timepoint directory from plate_dict
    plate_info = plate_dict.get(plate_map_name)

    if plate_info is None:
        raise ValueError(f"Plate map name '{plate_map_name}' not found in plate_dict")

    # Handle both dict (multi-timepoint) and string (single timepoint) formats
    if isinstance(plate_info, dict):
        # Multi-timepoint batches - extract timepoint from batch_id (Metadata_Plate)
        # batch_id format: '2025_01_27_B13A7A8P1_T1'
        if '_T' in batch_id:
            timepoint = batch_id.split('_T')[-1]  # Extract '1', '2', '3', '4'
            timepoint_key = f"T{timepoint}"
            timepoint_dir = plate_info.get(timepoint_key)
            if timepoint_dir is None:
                # Fallback to first available timepoint
                timepoint_dir = list(plate_info.values())[0]
        else:
            # Use first available timepoint
            timepoint_dir = list(plate_info.values())[0]
    else:
        timepoint_dir = plate_info

    # Construct image filename
    img_filename = f"r{row_num}c{col_num}f{str(site).zfill(2)}p01-ch{channel_num}sk1fk1fl1.tiff"

    # Full path: imgs_dir/actual_batch_dir/images/timepoint_dir/Images/img_filename
    img_path = os.path.join(imgs_dir, actual_batch_dir, "images", timepoint_dir, "Images", img_filename)

    return img_path


def load_cell_image(
    cell_row: Union[pl.DataFrame, Dict],
    channel: str,
    imgs_dir: str,
    batch_col: str = "Metadata_Plate",
    plate_col: str = "Metadata_plate_map_name",
    well_col: str = "Metadata_well_position",
    site_col: str = "Metadata_Site",
) -> np.ndarray:
    """
    Load a single channel image for a cell based on its metadata.

    Parameters:
    -----------
    cell_row : pl.DataFrame or dict
        Single cell row with metadata columns
    channel : str
        Channel name to load
    imgs_dir : str
        Base directory for images
    batch_col, plate_col, well_col, site_col : str
        Column names for metadata

    Returns:
    --------
    np.ndarray : Loaded image (2D array)
    """
    # Convert to dict if DataFrame row
    if isinstance(cell_row, pl.DataFrame):
        cell_dict = cell_row.to_dicts()[0]
    else:
        cell_dict = cell_row

    batch_id = cell_dict[batch_col]
    plate_map_name = cell_dict[plate_col]
    well = cell_dict[well_col]
    site = str(cell_dict[site_col]).zfill(2)  # Ensure 2-digit site format

    img_path = get_image_path(batch_id, plate_map_name, well, site, channel, imgs_dir)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = imread(img_path)
    return img


def extract_cell_crop(
    cell_row: Union[pl.DataFrame, Dict],
    channel: str,
    imgs_dir: str,
    method: str = 'bbox',
    target_size: Optional[int] = None,
    canvas_size: Optional[int] = None,
    recenter: bool = True,
    crop_size: int = 128,
    pad_value: float = 0,
    batch_col: str = "Metadata_Plate",
    plate_col: str = "Metadata_plate_map_name",
    well_col: str = "Metadata_well_position",
    site_col: str = "Metadata_Site",
) -> np.ndarray:
    """
    Extract a cell crop from an image using specified method.

    Parameters:
    -----------
    cell_row : pl.DataFrame or dict
        Single cell row with metadata and morphology features
    channel : str
        Channel name to extract
    imgs_dir : str
        Base directory for images
    method : str
        'bbox' for bounding box extraction or 'fixed' for fixed-size extraction
    target_size : int, optional
        If provided, resize crop to this size (for bbox method).
        Note: Resizing distorts the image - prefer canvas_size instead.
    canvas_size : int, optional
        If provided, center the crop in a fixed-size canvas without resizing.
        This preserves original pixel values and is the preferred approach.
    recenter : bool
        For bbox method: recenter bbox on nuclei center
    crop_size : int
        For fixed method: size of square crop
    pad_value : float
        Value to use for padding when crop extends beyond image
    batch_col, plate_col, well_col, site_col : str
        Column names for metadata

    Returns:
    --------
    np.ndarray : Cropped (and optionally canvas-centered or resized) cell image

    Example:
    --------
    # Bounding box method with canvas centering (preferred - no distortion)
    crop = extract_cell_crop(
        cell_row, 'GFP', imgs_dir='/path/to/imgs',
        method='bbox', recenter=True, canvas_size=128
    )

    # Bounding box method with resize (distorts image)
    crop = extract_cell_crop(
        cell_row, 'GFP', imgs_dir='/path/to/imgs',
        method='bbox', recenter=True, target_size=128
    )

    # Fixed-size method
    crop = extract_cell_crop(
        cell_row, 'DAPI', imgs_dir='/path/to/imgs',
        method='fixed', crop_size=64
    )
    """
    # Load full image
    img = load_cell_image(cell_row, channel, imgs_dir,
                          batch_col, plate_col, well_col, site_col)

    # Get crop coordinates based on method
    if method == 'bbox':
        x_min, y_min, x_max, y_max = get_crop_coordinates_bbox(
            cell_row, recenter=recenter
        )
    elif method == 'fixed':
        x_min, y_min, x_max, y_max = get_crop_coordinates_fixed(
            cell_row, crop_size=crop_size
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bbox' or 'fixed'")

    # Extract crop with padding if needed
    crop, _ = validate_and_pad_crop(img, x_min, y_min, x_max, y_max, pad_value)

    # Option 1: Center in fixed canvas (preferred - no resize/distortion)
    if canvas_size is not None:
        h, w = crop.shape
        if h > canvas_size or w > canvas_size:
            # Warn and return raw crop if larger than canvas
            warnings.warn(
                f"Crop size ({h}x{w}) exceeds canvas_size ({canvas_size}). "
                "Returning raw crop without centering."
            )
        else:
            canvas = np.full((canvas_size, canvas_size), pad_value, dtype=crop.dtype)
            y_offset = (canvas_size - h) // 2
            x_offset = (canvas_size - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = crop
            crop = canvas

    # Option 2: Resize (existing behavior, bbox method only - distorts image)
    elif target_size is not None and method == 'bbox':
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crop = resize(crop, (target_size, target_size),
                         preserve_range=True, anti_aliasing=True)
            crop = crop.astype(img.dtype)

    return crop


def load_multichannel_cell_crops(
    cell_row: Union[pl.DataFrame, Dict],
    channels: List[str],
    imgs_dir: str,
    method: str = 'bbox',
    target_size: Optional[int] = None,
    canvas_size: Optional[int] = None,
    recenter: bool = True,
    crop_size: int = 128,
    batch_col: str = "Metadata_Plate",
    plate_col: str = "Metadata_plate_map_name",
    well_col: str = "Metadata_well_position",
    site_col: str = "Metadata_Site",
) -> Dict[str, np.ndarray]:
    """
    Load crops for multiple channels for a single cell.

    Parameters:
    -----------
    cell_row : pl.DataFrame or dict
        Single cell row with metadata
    channels : list of str
        Channel names to load (e.g., ['DAPI', 'GFP', 'Mito', 'AGP'])
    imgs_dir : str
        Base directory for images
    method : str
        'bbox' or 'fixed' extraction method
    target_size : int, optional
        Target size for resizing (bbox method). Distorts image - prefer canvas_size.
    canvas_size : int, optional
        Canvas size for centering (preferred - no distortion)
    recenter : bool
        Whether to recenter on nuclei (bbox method)
    crop_size : int
        Crop size for fixed method
    batch_col, plate_col, well_col, site_col : str
        Column names for metadata

    Returns:
    --------
    dict : Dictionary mapping channel names to crop arrays
        e.g., {'DAPI': array(...), 'GFP': array(...), ...}

    Example:
    --------
    # Preferred: use canvas_size (no distortion)
    cell_crops = load_multichannel_cell_crops(
        cell_row,
        channels=['DAPI', 'AGP', 'Mito', 'GFP'],
        imgs_dir='/path/to/imgs',
        method='bbox',
        canvas_size=128,
        recenter=True
    )
    """
    crops = {}

    for channel in channels:
        try:
            crop = extract_cell_crop(
                cell_row, channel, imgs_dir,
                method=method,
                target_size=target_size,
                canvas_size=canvas_size,
                recenter=recenter,
                crop_size=crop_size,
                batch_col=batch_col,
                plate_col=plate_col,
                well_col=well_col,
                site_col=site_col
            )
            crops[channel] = crop
        except FileNotFoundError as e:
            print(f"Warning: Could not load {channel} for cell: {e}")
            continue
        except Exception as e:
            print(f"Error loading {channel}: {e}")
            continue

    return crops


def batch_extract_cell_crops(
    cell_profiles: pl.DataFrame,
    channels: List[str],
    imgs_dir: str,
    method: str = 'bbox',
    target_size: Optional[int] = None,
    canvas_size: Optional[int] = None,
    recenter: bool = True,
    crop_size: int = 128,
    batch_col: str = "Metadata_Plate",
    plate_col: str = "Metadata_plate_map_name",
    well_col: str = "Metadata_well_position",
    site_col: str = "Metadata_Site",
    cell_id_col: str = "Metadata_ObjectNumber",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract crops for multiple cells and channels.

    Parameters:
    -----------
    cell_profiles : pl.DataFrame
        DataFrame with multiple cell rows
    channels : list of str
        Channels to extract
    imgs_dir : str
        Base directory for images
    method : str
        Extraction method ('bbox' or 'fixed')
    target_size : int, optional
        Target size for bbox method. Distorts image - prefer canvas_size.
    canvas_size : int, optional
        Canvas size for centering (preferred - no distortion)
    recenter : bool
        Whether to recenter on nuclei
    crop_size : int
        Crop size for fixed method
    batch_col, plate_col, well_col, site_col, cell_id_col : str
        Column names for metadata

    Returns:
    --------
    dict : Nested dictionary {cell_id: {channel: crop_array}}

    Example:
    --------
    # Preferred: use canvas_size (no distortion)
    all_crops = batch_extract_cell_crops(
        selected_cells,
        channels=['DAPI', 'GFP', 'Mito', 'AGP'],
        imgs_dir='/path/to/imgs',
        method='bbox',
        canvas_size=128
    )

    # Access a specific cell's GFP channel
    gfp_crop = all_crops[cell_id]['GFP']
    """
    all_crops = {}

    for row in cell_profiles.iter_rows(named=True):
        cell_id = row[cell_id_col]

        try:
            cell_crops = load_multichannel_cell_crops(
                row, channels, imgs_dir,
                method=method,
                target_size=target_size,
                canvas_size=canvas_size,
                recenter=recenter,
                crop_size=crop_size,
                batch_col=batch_col,
                plate_col=plate_col,
                well_col=well_col,
                site_col=site_col
            )

            if cell_crops:  # Only add if at least one channel was loaded
                all_crops[cell_id] = cell_crops

        except Exception as e:
            print(f"Warning: Could not extract crops for cell {cell_id}: {e}")
            continue

    return all_crops
