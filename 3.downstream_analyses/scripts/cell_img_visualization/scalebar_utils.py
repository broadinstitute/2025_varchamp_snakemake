"""
Scale Bar Utilities for Microscopy Images

This module provides functions to add publication-quality scale bars to matplotlib
figures containing microscopy images. Scale bars are added post-hoc as overlays,
without modifying the underlying image data or existing visualization functions.

Author: VarChAMP Analysis Pipeline
Date: 2025-12-15

Microscope Specifications (VarChAMP Dataset):
- Microscope: Perkin Elmer Phenix
- Objective: 20x magnification
- Binning: 2×2
- Pixel Size: 0.598 µm/pixel (verified from Index.idx.xml metadata)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Union, List, Tuple, Optional


def add_scale_bar(
    ax: plt.Axes,
    pixel_size_um: float,
    image_width_pixels: int,
    scale_bar_length_um: float = 10.0,
    location: str = 'lower right',
    color: str = 'white',
    fontsize: int = 10,
    bar_height_pixels: float = 2.0,
    padding_fraction: float = 0.05,
    label_offset_pixels: float = 3.0
) -> None:
    """
    Add a scale bar to a matplotlib axes object containing a microscopy image.

    This function adds a scale bar post-hoc to any matplotlib axes, making it
    compatible with existing visualization pipelines without requiring modifications
    to the underlying plotting functions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object containing the image to annotate
    pixel_size_um : float
        Size of one pixel in microns (µm/pixel). For VarChAMP images from
        Phenix microscope with 20x objective and 2×2 binning, use 0.598
    image_width_pixels : int
        Width of the image in pixels. If not known, can be extracted from
        the image array: ax.get_images()[0].get_array().shape[1]
    scale_bar_length_um : float, optional
        Desired length of the scale bar in microns. Default is 10 µm,
        appropriate for cell-level images (128×128 pixels)
    location : str, optional
        Position of the scale bar. Options: 'lower right', 'lower left',
        'upper right', 'upper left'. Default is 'lower right'
    color : str, optional
        Color of the scale bar and text. Default is 'white' for dark
        microscopy images. Use 'black' for bright-field images
    fontsize : int, optional
        Font size for the scale bar label. Default is 10
    bar_height_pixels : float, optional
        Height of the scale bar line in pixels. Default is 2.0 pixels
        for a thin, crisp line segment appearance
    padding_fraction : float, optional
        Padding from image edges as a fraction of image dimensions.
        Default is 0.05 (5% padding)
    label_offset_pixels : float, optional
        Vertical offset of label text from scale bar in pixels.
        Default is 3.0 pixels. Text appears above the scale bar line.

    Returns
    -------
    None
        The function modifies the axes in-place by adding patches and text

    Examples
    --------
    Add a 10 µm scale bar to a cell-level image (128×128 pixels):

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(cell_image)
    >>> add_scale_bar(ax, pixel_size_um=0.598, image_width_pixels=128,
    ...               scale_bar_length_um=10)
    >>> plt.show()

    Add a 100 µm scale bar to a well-level image (1080×1080 pixels):

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(well_image)
    >>> add_scale_bar(ax, pixel_size_um=0.598, image_width_pixels=1080,
    ...               scale_bar_length_um=100)
    >>> plt.show()

    Use with existing visualization functions:

    >>> from cell_visualizer import viz_cell_single_channel
    >>> fig, ax = plt.subplots()
    >>> viz_cell_single_channel(crop, channel='GFP', ax=ax)
    >>> add_scale_bar(ax, 0.598, 128, 10)
    >>> plt.show()

    Notes
    -----
    - The scale bar is added using data coordinates for pixel-perfect positioning
    - Works correctly even when axes are hidden (axis_off=True)
    - Does not modify the underlying image data
    - Can be called multiple times to add scale bars to multiple subplots

    Scale Bar Calculation:
        scale_bar_pixels = scale_bar_length_um / pixel_size_um
        Example: 10 µm / 0.598 µm/pixel = 16.72 pixels
    """
    # Get image dimensions from the axes
    # Try to get from imshow object first, fall back to parameter
    try:
        images = ax.get_images()
        if len(images) > 0:
            img_array = images[0].get_array()
            if len(img_array.shape) == 3:  # RGB image
                image_height_pixels, image_width_pixels_detected = img_array.shape[:2]
            else:  # Grayscale image
                image_height_pixels, image_width_pixels_detected = img_array.shape
            # Use detected width if available
            if image_width_pixels_detected > 0:
                image_width_pixels = image_width_pixels_detected
    except (IndexError, AttributeError):
        # If detection fails, assume square image
        image_height_pixels = image_width_pixels
    else:
        image_height_pixels = image_width_pixels

    # Calculate scale bar length in pixels
    scale_bar_pixels = scale_bar_length_um / pixel_size_um

    # Calculate padding in pixels
    padding_x = image_width_pixels * padding_fraction
    padding_y = image_height_pixels * padding_fraction

    # Calculate position based on location parameter
    # NOTE: In matplotlib imshow, (0,0) is at TOP-LEFT, Y increases DOWNWARD
    location = location.lower()

    if 'right' in location:
        # Right side: scale bar ends at padding distance from right edge
        bar_x_end = image_width_pixels - padding_x
        bar_x_start = bar_x_end - scale_bar_pixels
        text_x = (bar_x_start + bar_x_end) / 2  # Center of scale bar
        text_ha = 'center'
    else:  # 'left' in location
        # Left side: scale bar starts at padding distance from left edge
        bar_x_start = padding_x
        bar_x_end = bar_x_start + scale_bar_pixels
        text_x = (bar_x_start + bar_x_end) / 2
        text_ha = 'center'

    if 'lower' in location:
        # Lower (bottom): scale bar near bottom of image (high Y value)
        # Y increases downward, so bottom = image_height - padding
        bar_y = image_height_pixels - padding_y - bar_height_pixels
        # Text ABOVE bar means lower Y value (closer to 0)
        text_y = bar_y - label_offset_pixels
        text_va = 'bottom'  # Align text bottom edge to position above bar
    else:  # 'upper' in location
        # Upper (top): scale bar near top of image (low Y value)
        bar_y = padding_y
        # Text ABOVE bar at top means even lower Y value
        text_y = bar_y - label_offset_pixels
        text_va = 'bottom'  # Align text bottom edge to position

    # Create scale bar rectangle
    scale_bar_rect = Rectangle(
        (bar_x_start, bar_y),
        scale_bar_pixels,
        bar_height_pixels,
        facecolor=color,
        edgecolor='none',
        transform=ax.transData,
        zorder=1000  # Ensure scale bar appears on top
    )

    # Add scale bar to axes
    ax.add_patch(scale_bar_rect)

    # Format label text
    # Use appropriate precision based on scale bar length
    if scale_bar_length_um >= 10:
        label_text = f"{int(scale_bar_length_um)} µm"
    else:
        label_text = f"{scale_bar_length_um:.1f} µm"

    # Add text label
    ax.text(
        text_x,
        text_y,
        label_text,
        color=color,
        fontsize=fontsize,
        ha=text_ha,
        va=text_va,
        transform=ax.transData,
        zorder=1001,  # Text on top of scale bar
        weight='bold'  # Make text more visible
    )


def calculate_optimal_scale_bar_length(
    image_width_pixels: int,
    pixel_size_um: float,
    target_fraction: float = 0.2,
    round_to: Optional[List[int]] = None
) -> float:
    """
    Calculate an appropriate scale bar length for a given image size.

    This function suggests a scale bar length that is approximately a target
    fraction of the image width and rounds to standard values for clarity.

    Parameters
    ----------
    image_width_pixels : int
        Width of the image in pixels
    pixel_size_um : float
        Size of one pixel in microns (µm/pixel)
    target_fraction : float, optional
        Target scale bar length as a fraction of image width.
        Default is 0.2 (20% of image width)
    round_to : list of int, optional
        List of standard values to round to. The function will choose the
        closest value from this list. Default is [1, 2, 5, 10, 20, 50, 100, 200]

    Returns
    -------
    float
        Recommended scale bar length in microns

    Examples
    --------
    Calculate optimal scale bar for cell-level image (128×128 pixels):

    >>> calculate_optimal_scale_bar_length(128, 0.598)
    10

    Calculate optimal scale bar for well-level image (1080×1080 pixels):

    >>> calculate_optimal_scale_bar_length(1080, 0.598)
    100

    Use custom rounding values:

    >>> calculate_optimal_scale_bar_length(512, 0.598, round_to=[5, 25, 50, 100])
    50

    Notes
    -----
    The function aims for a scale bar that is visually prominent but not
    overwhelming, typically 15-25% of the image width.
    """
    if round_to is None:
        round_to = [1, 2, 5, 10, 20, 50, 100, 200]

    # Calculate image width in microns
    image_width_um = image_width_pixels * pixel_size_um

    # Calculate target scale bar length
    target_length_um = image_width_um * target_fraction

    # Find closest standard value
    round_to_array = np.array(round_to)
    closest_idx = np.argmin(np.abs(round_to_array - target_length_um))
    optimal_length = round_to_array[closest_idx]

    return float(optimal_length)


def add_scale_bar_to_axes_list(
    axes: Union[plt.Axes, List[plt.Axes], np.ndarray],
    pixel_size_um: float,
    image_width_pixels: Union[int, List[int]],
    scale_bar_length_um: Optional[Union[float, List[float]]] = None,
    **kwargs
) -> None:
    """
    Add scale bars to multiple axes objects.

    This is a convenience function for adding scale bars to multiple subplots
    in a single call. It handles flat lists, 2D arrays of axes, and supports
    different scale bar lengths for different axes.

    Parameters
    ----------
    axes : matplotlib.axes.Axes or list or numpy.ndarray
        Single axes object, list of axes, or 2D array of axes (from plt.subplots)
    pixel_size_um : float
        Size of one pixel in microns (µm/pixel)
    image_width_pixels : int or list of int
        Width of images in pixels. Can be a single value (applied to all axes)
        or a list matching the number of axes
    scale_bar_length_um : float or list of float, optional
        Scale bar length(s) in microns. If None, automatically calculated for
        each axes using calculate_optimal_scale_bar_length()
    **kwargs
        Additional keyword arguments passed to add_scale_bar()
        (location, color, fontsize, etc.)

    Returns
    -------
    None
        Modifies all axes in-place

    Examples
    --------
    Add scale bars to a grid of cell images:

    >>> fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    >>> # ... plot images on each axes ...
    >>> add_scale_bar_to_axes_list(axes, pixel_size_um=0.598,
    ...                            image_width_pixels=128,
    ...                            scale_bar_length_um=10)
    >>> plt.show()

    Add different scale bars to different image types:

    >>> fig, axes = plt.subplots(1, 2)
    >>> # axes[0] has cell image, axes[1] has well image
    >>> add_scale_bar_to_axes_list(
    ...     axes,
    ...     pixel_size_um=0.598,
    ...     image_width_pixels=[128, 1080],
    ...     scale_bar_length_um=[10, 100]
    ... )

    Automatically calculate scale bar lengths:

    >>> fig, axes = plt.subplots(2, 3)
    >>> # ... plot images ...
    >>> add_scale_bar_to_axes_list(axes, pixel_size_um=0.598,
    ...                            image_width_pixels=128)
    >>> # Automatically uses optimal scale bar length

    Notes
    -----
    - Skips axes that don't contain images (checked via ax.get_images())
    - Handles both 1D and 2D arrays of axes from plt.subplots()
    - Useful for batch processing multiple visualizations
    """
    # Convert single axes to list
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    elif isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = list(axes)

    # Convert scalar image_width_pixels to list
    if isinstance(image_width_pixels, int):
        image_widths = [image_width_pixels] * len(axes_list)
    else:
        image_widths = list(image_width_pixels)

    # Handle scale_bar_length_um
    if scale_bar_length_um is None:
        # Automatically calculate for each axes
        scale_bar_lengths = [
            calculate_optimal_scale_bar_length(width, pixel_size_um)
            for width in image_widths
        ]
    elif isinstance(scale_bar_length_um, (int, float)):
        scale_bar_lengths = [float(scale_bar_length_um)] * len(axes_list)
    else:
        scale_bar_lengths = list(scale_bar_length_um)

    # Add scale bars to each axes
    for ax, width, length in zip(axes_list, image_widths, scale_bar_lengths):
        # Skip axes without images
        if len(ax.get_images()) == 0:
            continue

        add_scale_bar(
            ax,
            pixel_size_um,
            width,
            scale_bar_length_um=length,
            **kwargs
        )


# Convenience constant for VarChAMP dataset
VARCHAMP_PIXEL_SIZE_UM = 0.598  # Phenix 20x, 2×2 binning


if __name__ == "__main__":
    # Demonstration code
    print("Scale Bar Utilities for Microscopy Images")
    print("=" * 60)
    print()
    print(f"VarChAMP Dataset Pixel Size: {VARCHAMP_PIXEL_SIZE_UM} µm/pixel")
    print()
    print("Example Calculations:")
    print("-" * 60)

    # Cell-level example
    cell_width = 128
    cell_scale_bar = 10
    cell_pixels = cell_scale_bar / VARCHAMP_PIXEL_SIZE_UM
    print(f"Cell-level image ({cell_width}×{cell_width} pixels):")
    print(f"  {cell_scale_bar} µm scale bar = {cell_pixels:.2f} pixels")
    print(f"  Optimal scale bar: {calculate_optimal_scale_bar_length(cell_width, VARCHAMP_PIXEL_SIZE_UM)} µm")
    print()

    # Well-level example
    well_width = 1080
    well_scale_bar = 100
    well_pixels = well_scale_bar / VARCHAMP_PIXEL_SIZE_UM
    print(f"Well-level image ({well_width}×{well_width} pixels):")
    print(f"  {well_scale_bar} µm scale bar = {well_pixels:.2f} pixels")
    print(f"  Optimal scale bar: {calculate_optimal_scale_bar_length(well_width, VARCHAMP_PIXEL_SIZE_UM)} µm")
