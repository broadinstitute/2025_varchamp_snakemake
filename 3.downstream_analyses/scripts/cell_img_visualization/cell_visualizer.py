"""
Cell Visualization Module for VarChAMP

This module provides functions for visualizing single-cell image crops with
configurable contrast control and multi-channel RGB merging optimized for
Cell Painting microscopy data.

Author: VarChAMP Pipeline
Date: 2025
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Union, Tuple
import sys

sys.path.append("../..")
from img_utils import channel_to_cmap, channel_to_rgb


def normalize_channel(
    channel: np.ndarray,
    method: str = 'percentile',
    percentile: float = 99.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Normalize channel intensity with configurable contrast control.

    Parameters:
    -----------
    channel : np.ndarray
        Input channel image
    method : str
        Normalization method:
        - 'percentile': normalize by percentile value (default)
        - 'minmax': normalize by actual min/max
        - 'manual': use provided vmin/vmax
    percentile : float
        Percentile for clipping when method='percentile' (e.g., 99.9, 99, 95)
    vmin : float, optional
        Manual minimum value (overrides method)
    vmax : float, optional
        Manual maximum value (overrides method)
    clip : bool
        Whether to clip values to [0, 1] (default: True)

    Returns:
    --------
    np.ndarray : Normalized channel with values in [0, 1]

    Example:
    --------
    # Using 99.9th percentile
    norm = normalize_channel(img, method='percentile', percentile=99.9)

    # Manual bounds
    norm = normalize_channel(img, method='manual', vmin=100, vmax=5000)

    # Full range normalization
    norm = normalize_channel(img, method='minmax')
    """
    # Determine vmax if not manually specified
    if vmax is None:
        if method == 'percentile':
            vmax = np.percentile(channel, percentile)
        elif method == 'minmax':
            vmax = channel.max()
        else:
            vmax = 1.0

    # Determine vmin if not manually specified
    if vmin is None:
        if method == 'minmax':
            vmin = channel.min()
        else:
            vmin = 0

    # Normalize
    if vmax > vmin:
        normalized = (channel - vmin) / (vmax - vmin)
    else:
        normalized = channel

    # Clip to valid range
    if clip:
        normalized = np.clip(normalized, 0, 1)

    return normalized


def create_rgb_merge(
    channel_crops: Dict[str, np.ndarray],
    channel_mapping: Union[str, Dict[str, Tuple[float, float, float]]] = 'morphology',
    contrast_percentiles: Optional[Union[float, Dict[str, float]]] = None,
    contrast_method: str = 'percentile',
) -> np.ndarray:
    """
    Create RGB merged image from multiple channels with configurable mapping.

    Parameters:
    -----------
    channel_crops : dict
        Dictionary mapping channel names to normalized image arrays
        e.g., {'DAPI': array(...), 'GFP': array(...), ...}
    channel_mapping : str or dict
        RGB mapping strategy:
        - 'morphology': Red=Mito, Green=AGP, Blue=DAPI (no GFP)
        - 'gfp_inclusive': Red=(Mito+AGP)/2, Green=(GFP+AGP)/2, Blue=DAPI
        - dict: Custom mapping {channel: (r_weight, g_weight, b_weight)}
    contrast_percentiles : float or dict, optional
        Percentile(s) for contrast adjustment
        If None, uses pre-normalized channels as-is
        If float: apply same percentile to all channels
        If dict: per-channel percentiles
    contrast_method : str
        Normalization method ('percentile', 'minmax', 'manual')

    Returns:
    --------
    np.ndarray : RGB image with shape (height, width, 3)

    Example:
    --------
    # Morphology-focused RGB
    rgb = create_rgb_merge(crops, channel_mapping='morphology')

    # Custom contrast per channel
    rgb = create_rgb_merge(
        crops,
        channel_mapping='gfp_inclusive',
        contrast_percentiles={'DAPI': 99.9, 'GFP': 95.0, 'Mito': 99.5, 'AGP': 98.0}
    )
    """
    # Get image dimensions from first available channel
    first_channel = list(channel_crops.values())[0]
    height, width = first_channel.shape

    # Initialize RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    # Normalize channels if percentiles provided
    if contrast_percentiles is not None:
        normalized_crops = {}
        for ch_name, ch_img in channel_crops.items():
            # Determine percentile for this channel
            if isinstance(contrast_percentiles, (int, float)):
                percentile = contrast_percentiles
            else:
                percentile = contrast_percentiles.get(ch_name, 99.0)

            normalized_crops[ch_name] = normalize_channel(
                ch_img, method=contrast_method, percentile=percentile
            )
    else:
        normalized_crops = channel_crops

    # Apply channel mapping
    if channel_mapping == 'morphology':
        # Red=Mito, Green=AGP, Blue=DAPI
        if 'Mito' in normalized_crops:
            rgb_image[:, :, 0] = normalized_crops['Mito']
        if 'AGP' in normalized_crops:
            rgb_image[:, :, 1] = normalized_crops['AGP']
        if 'DAPI' in normalized_crops:
            rgb_image[:, :, 2] = normalized_crops['DAPI']

    elif channel_mapping == 'gfp_inclusive':
        # Red=(Mito+AGP)/2, Green=(GFP+AGP)/2, Blue=DAPI
        red_channels = []
        if 'Mito' in normalized_crops:
            red_channels.append(normalized_crops['Mito'])
        if 'AGP' in normalized_crops:
            red_channels.append(normalized_crops['AGP'])
        if red_channels:
            rgb_image[:, :, 0] = np.mean(red_channels, axis=0)

        green_channels = []
        if 'GFP' in normalized_crops:
            green_channels.append(normalized_crops['GFP'])
        if 'AGP' in normalized_crops:
            green_channels.append(normalized_crops['AGP'])
        if green_channels:
            rgb_image[:, :, 1] = np.mean(green_channels, axis=0)

        if 'DAPI' in normalized_crops:
            rgb_image[:, :, 2] = normalized_crops['DAPI']

    elif isinstance(channel_mapping, dict):
        # Custom mapping: {channel: (r_weight, g_weight, b_weight)}
        for ch_name, rgb_weights in channel_mapping.items():
            if ch_name in normalized_crops:
                ch_img = normalized_crops[ch_name]
                rgb_image[:, :, 0] += ch_img * rgb_weights[0]
                rgb_image[:, :, 1] += ch_img * rgb_weights[1]
                rgb_image[:, :, 2] += ch_img * rgb_weights[2]

    else:
        raise ValueError(f"Unknown channel_mapping: {channel_mapping}")

    # Clip final RGB to [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)

    return rgb_image


def viz_cell_single_channel(
    crop: np.ndarray,
    channel: str,
    contrast_percentile: float = 99.0,
    contrast_method: str = 'percentile',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    axis_off: bool = True,
    title: Optional[str] = None,
    colorbar: bool = False,
) -> plt.Axes:
    """
    Visualize a single channel cell crop with adjustable contrast.

    Parameters:
    -----------
    crop : np.ndarray
        Cell crop image (2D array)
    channel : str
        Channel name (e.g., 'GFP', 'DAPI', 'Mito', 'AGP')
    contrast_percentile : float
        Percentile for contrast adjustment
    contrast_method : str
        Normalization method ('percentile', 'minmax', 'manual')
    vmin, vmax : float, optional
        Manual intensity bounds (overrides percentile)
    ax : matplotlib Axes, optional
        Axes to plot on (creates new if None)
    axis_off : bool
        Whether to turn off axis labels
    title : str, optional
        Title for the plot
    colorbar : bool
        Whether to add a colorbar

    Returns:
    --------
    plt.Axes : The axes object

    Example:
    --------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, channel in enumerate(['DAPI', 'AGP', 'Mito', 'GFP']):
        viz_cell_single_channel(
            crops[channel], channel,
            contrast_percentile=99.5,
            ax=axes[i]
        )
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize channel
    normalized = normalize_channel(
        crop,
        method=contrast_method,
        percentile=contrast_percentile,
        vmin=vmin,
        vmax=vmax
    )

    # Get colormap for this channel
    cmap = channel_to_cmap(channel)

    # Display
    im = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)

    if axis_off:
        ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10)

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def viz_cell_multi_channel(
    channel_crops: Dict[str, np.ndarray],
    channels: Optional[List[str]] = None,
    channel_mapping: str = 'morphology',
    contrast_percentiles: Optional[Union[float, Dict[str, float]]] = None,
    contrast_method: str = 'percentile',
    ax: Optional[plt.Axes] = None,
    axis_off: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Visualize multi-channel cell crop as RGB merged image.

    Parameters:
    -----------
    channel_crops : dict
        Dictionary mapping channel names to crop arrays
    channels : list of str, optional
        Specific channels to use (default: use all available)
    channel_mapping : str
        'morphology' or 'gfp_inclusive'
    contrast_percentiles : float or dict, optional
        Percentile(s) for contrast (default: None, use as-is)
    contrast_method : str
        Normalization method
    ax : matplotlib Axes, optional
        Axes to plot on
    axis_off : bool
        Whether to turn off axis labels
    title : str, optional
        Title for the plot

    Returns:
    --------
    plt.Axes : The axes object

    Example:
    --------
    viz_cell_multi_channel(
        cell_crops,
        channel_mapping='gfp_inclusive',
        contrast_percentiles={'DAPI': 99.9, 'GFP': 95.0, 'Mito': 99.5, 'AGP': 98.0}
    )
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Filter to specified channels if provided
    if channels is not None:
        filtered_crops = {k: v for k, v in channel_crops.items() if k in channels}
    else:
        filtered_crops = channel_crops

    # Create RGB merge
    rgb_image = create_rgb_merge(
        filtered_crops,
        channel_mapping=channel_mapping,
        contrast_percentiles=contrast_percentiles,
        contrast_method=contrast_method
    )

    # Display
    ax.imshow(rgb_image)

    if axis_off:
        ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10)

    return ax


def viz_cell_grid(
    channel_crops: Dict[str, np.ndarray],
    channels: Optional[List[str]] = None,
    channel_mapping: str = 'morphology',
    contrast_percentiles: Optional[Union[float, Dict[str, float]]] = None,
    contrast_method: str = 'percentile',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 150,
    cell_info: Optional[Dict] = None,
) -> plt.Figure:
    """
    Create a grid layout showing individual channels and RGB merged image.

    Layout:
    - Top row: Individual channel images (grayscale with channel-specific colormaps)
    - Bottom row: RGB merged image spanning full width

    Parameters:
    -----------
    channel_crops : dict
        Dictionary mapping channel names to crop arrays
    channels : list of str, optional
        Channels to display (default: all available, sorted)
    channel_mapping : str
        RGB mapping strategy for merged view
    contrast_percentiles : float or dict, optional
        Contrast percentiles (per-channel or global)
    contrast_method : str
        Normalization method
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
    cell_info : dict, optional
        Cell metadata to display (e.g., {'allele': 'BRCA1_V1', 'feature_value': 0.85})

    Returns:
    --------
    plt.Figure : The figure object

    Example:
    --------
    fig = viz_cell_grid(
        cell_crops,
        channels=['DAPI', 'AGP', 'Mito', 'GFP'],
        channel_mapping='gfp_inclusive',
        contrast_percentiles=99.5,
        save_path='cell_visualization.png'
    )
    """
    # Determine channels to display
    if channels is None:
        channels = sorted(channel_crops.keys())

    n_channels = len(channels)

    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, n_channels, height_ratios=[1, 4],
                          hspace=0.02, wspace=0.02)

    # Top row: Individual channels
    for i, channel in enumerate(channels):
        ax = fig.add_subplot(gs[0, i])
        viz_cell_single_channel(
            channel_crops[channel],
            channel=channel,
            contrast_percentile=contrast_percentiles if isinstance(contrast_percentiles, (int, float))
                               else contrast_percentiles.get(channel, 99.0) if contrast_percentiles else 99.0,
            contrast_method=contrast_method,
            ax=ax,
            axis_off=True,
            title=channel
        )

    # Bottom row: RGB merged image (spanning all columns)
    ax_combined = fig.add_subplot(gs[1, :])
    viz_cell_multi_channel(
        channel_crops,
        channels=channels,
        channel_mapping=channel_mapping,
        contrast_percentiles=contrast_percentiles,
        contrast_method=contrast_method,
        ax=ax_combined,
        axis_off=True
    )

    # Add cell info if provided
    if cell_info:
        info_text = '\n'.join([f"{k}: {v}" for k, v in cell_info.items()])
        ax_combined.text(0.02, 0.98, info_text,
                        transform=ax_combined.transAxes,
                        fontsize=9, color='white',
                        verticalalignment='top',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_cell_comparison(
    variant_crops: Dict[str, np.ndarray],
    ref_crops: Dict[str, np.ndarray],
    channels: Optional[List[str]] = None,
    channel_mapping: str = 'morphology',
    contrast_percentiles: Optional[Union[float, Dict[str, float]]] = None,
    contrast_method: str = 'percentile',
    variant_label: str = 'Variant',
    ref_label: str = 'Reference',
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create side-by-side comparison of variant and reference cells.

    Layout:
    - Left column: Variant cell (individual channels + RGB merge)
    - Right column: Reference cell (individual channels + RGB merge)

    Parameters:
    -----------
    variant_crops : dict
        Variant cell channel crops
    ref_crops : dict
        Reference cell channel crops
    channels : list of str, optional
        Channels to display
    channel_mapping : str
        RGB mapping strategy
    contrast_percentiles : float or dict, optional
        Contrast percentiles
    contrast_method : str
        Normalization method
    variant_label : str
        Label for variant cell
    ref_label : str
        Label for reference cell
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure

    Returns:
    --------
    plt.Figure : The figure object

    Example:
    --------
    fig = plot_cell_comparison(
        variant_crops, ref_crops,
        channels=['DAPI', 'AGP', 'Mito', 'GFP'],
        channel_mapping='morphology',
        contrast_percentiles=99.5,
        variant_label='BRCA1_V1',
        ref_label='BRCA1_WT'
    )
    """
    # Determine channels
    if channels is None:
        channels = sorted(set(variant_crops.keys()) & set(ref_crops.keys()))

    n_channels = len(channels)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_channels + 1, 2, hspace=0.15, wspace=0.05)

    # Individual channels (rows 0 to n_channels-1)
    for i, channel in enumerate(channels):
        # Variant channel
        ax_var = fig.add_subplot(gs[i, 0])
        viz_cell_single_channel(
            variant_crops[channel], channel,
            contrast_percentile=contrast_percentiles if isinstance(contrast_percentiles, (int, float))
                               else contrast_percentiles.get(channel, 99.0) if contrast_percentiles else 99.0,
            contrast_method=contrast_method,
            ax=ax_var,
            axis_off=True,
            title=f"{channel}" if i == 0 else None
        )
        if i == 0:
            ax_var.text(0.5, 1.15, variant_label, transform=ax_var.transAxes,
                       fontsize=12, fontweight='bold', ha='center')

        # Reference channel
        ax_ref = fig.add_subplot(gs[i, 1])
        viz_cell_single_channel(
            ref_crops[channel], channel,
            contrast_percentile=contrast_percentiles if isinstance(contrast_percentiles, (int, float))
                               else contrast_percentiles.get(channel, 99.0) if contrast_percentiles else 99.0,
            contrast_method=contrast_method,
            ax=ax_ref,
            axis_off=True,
            title=f"{channel}" if i == 0 else None
        )
        if i == 0:
            ax_ref.text(0.5, 1.15, ref_label, transform=ax_ref.transAxes,
                       fontsize=12, fontweight='bold', ha='center')

    # RGB merged images (bottom row)
    ax_var_rgb = fig.add_subplot(gs[n_channels, 0])
    viz_cell_multi_channel(
        variant_crops, channels,
        channel_mapping=channel_mapping,
        contrast_percentiles=contrast_percentiles,
        contrast_method=contrast_method,
        ax=ax_var_rgb,
        axis_off=True,
        title='RGB Merge'
    )

    ax_ref_rgb = fig.add_subplot(gs[n_channels, 1])
    viz_cell_multi_channel(
        ref_crops, channels,
        channel_mapping=channel_mapping,
        contrast_percentiles=contrast_percentiles,
        contrast_method=contrast_method,
        ax=ax_ref_rgb,
        axis_off=True,
        title='RGB Merge'
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
