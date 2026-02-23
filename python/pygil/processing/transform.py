"""
PyGIL Transform Module

This module provides image transformation operations,
replicating Boost.GIL's numeric extensions for image transformations.

Supports:
- Resize with interpolation (bilinear, nearest neighbor)
- Flip operations (vertical, horizontal)
- Rotation by arbitrary angles
"""

from typing import Tuple, Optional
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

# Import from core module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.image import Image
from core.view import ImageView
from core.pixel import PixelType


class Sampler:
    """
    Base class for image sampling methods.
    
    In GIL, samplers define how pixel values are interpolated
    when transforming images (e.g., during resize operations).
    
    GIL equivalent: boost::gil::sampler concept
    """
    pass


class BilinearSampler(Sampler):
    """
    Bilinear interpolation sampler.
    
    Interpolates pixel values using weighted average of 4 nearest neighbors.
    Provides smooth results for image scaling.
    
    GIL equivalent: boost::gil::bilinear_sampler
    """
    def __repr__(self):
        return "BilinearSampler()"


class NearestSampler(Sampler):
    """
    Nearest neighbor sampler.
    
    Uses the value of the closest pixel without interpolation.
    Faster but can produce blocky results.
    
    GIL equivalent: boost::gil::nearest_sampler (implied in GIL)
    """
    def __repr__(self):
        return "NearestSampler()"


# Sampler instances for convenience
def bilinear_sampler() -> BilinearSampler:
    """Get bilinear interpolation sampler."""
    return BilinearSampler()


def nearest_sampler() -> NearestSampler:
    """Get nearest neighbor sampler."""
    return NearestSampler()


def resize_view(src_view: ImageView, 
                width: int, 
                height: int,
                sampler: Optional[Sampler] = None) -> Image:
    """
    Resize an image view to new dimensions using interpolation.
    
    This replicates GIL's resize_view() from boost/gil/extension/numeric/resample.hpp.
    Creates a new image with the target dimensions and resamples the source view
    into it using the specified sampling method.
    
    Args:
        src_view: Source image view to resize
        width: Target width in pixels
        height: Target height in pixels
        sampler: Interpolation method (default: bilinear)
        
    Returns:
        New Image object with resized content
        
    GIL equivalent: boost::gil::resize_view(src, dst, sampler)
    
    Example:
        >>> img = Image(800, 600, RGB8)
        >>> v = view(img)
        >>> resized = resize_view(v, 400, 300)  # Resize to 400x300
        >>> resized.dimensions()
        (400, 300)
        
    Algorithm:
        - Calculate scale factors for width and height
        - Create coordinate grids for source and destination
        - For bilinear: Use scipy's interpolation for smooth results
        - For nearest: Use integer indexing for closest pixels
        - Process each channel independently
        - Clamp values to valid range for pixel type
    """
    if sampler is None:
        sampler = bilinear_sampler()
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Target dimensions must be positive: {width}x{height}")
    
    # Get source data
    src_data = src_view.to_array()
    src_height, src_width, channels = src_data.shape
    
    # Create output array
    dst_data = np.zeros((height, width, channels), dtype=src_data.dtype)
    
    if isinstance(sampler, NearestSampler):
        # Nearest neighbor interpolation (fast)
        # Calculate scale factors
        scale_y = src_height / height
        scale_x = src_width / width
        
        # For each destination pixel, find nearest source pixel
        for dst_y in range(height):
            for dst_x in range(width):
                # Map destination coordinates to source
                src_y = int(dst_y * scale_y)
                src_x = int(dst_x * scale_x)
                
                # Clamp to valid source coordinates
                src_y = min(src_y, src_height - 1)
                src_x = min(src_x, src_width - 1)
                
                # Copy pixel value
                dst_data[dst_y, dst_x] = src_data[src_y, src_x]
    
    else:  # BilinearSampler or default
        # Bilinear interpolation (smooth, higher quality)
        # Use scipy's zoom function which implements bilinear interpolation
        zoom_factors = (height / src_height, width / src_width, 1)
        
        # Process each channel independently
        for c in range(channels):
            dst_data[:, :, c] = ndimage.zoom(
                src_data[:, :, c],
                zoom_factors[:2],  # Only zoom spatial dimensions
                order=1,  # 1 = bilinear interpolation
                mode='nearest'  # Edge handling
            )
    
    # Create new image from resized data
    return Image.from_array(dst_data, src_view.pixel_type)


def flip_up_down(img_view: ImageView) -> Image:
    """
    Flip an image vertically (upside down).
    
    Creates a new image with rows reversed. The top becomes bottom and vice versa.
    
    Args:
        img_view: Source image view to flip
        
    Returns:
        New Image with vertically flipped content
        
    Note: For a zero-copy view-only flip, use flipped_up_down_view() instead.
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> flipped = flip_up_down(v)
        
    Algorithm:
        - Reverse the row order using NumPy slicing
        - data[::-1] reverses the first dimension (rows)
        - Create new Image from flipped data
    """
    flipped_data = img_view.to_array()[::-1, :, :]
    return Image.from_array(flipped_data, img_view.pixel_type)


def flip_left_right(img_view: ImageView) -> Image:
    """
    Flip an image horizontally (left-right mirror).
    
    Creates a new image with columns reversed. The left becomes right and vice versa.
    
    Args:
        img_view: Source image view to flip
        
    Returns:
        New Image with horizontally flipped content
        
    Note: For a zero-copy view-only flip, use flipped_left_right_view() instead.
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> flipped = flip_left_right(v)
        
    Algorithm:
        - Reverse the column order using NumPy slicing
        - data[:, ::-1] reverses the second dimension (columns)
        - Create new Image from flipped data
    """
    flipped_data = img_view.to_array()[:, ::-1, :]
    return Image.from_array(flipped_data, img_view.pixel_type)


def rotate(img_view: ImageView, 
           angle: float,
           resize_output: bool = False,
           fill_value: Optional[Tuple[int, ...]] = None) -> Image:
    """
    Rotate an image by an arbitrary angle.
    
    Rotates the image counter-clockwise by the specified angle in degrees.
    Uses bilinear interpolation for smooth results.
    
    Args:
        img_view: Source image view to rotate
        angle: Rotation angle in degrees (counter-clockwise, positive)
        resize_output: If True, expand output to fit entire rotated image
                      If False, keep same dimensions (may crop)
        fill_value: Pixel value for areas outside the original image
                   Default: black (zeros)
        
    Returns:
        New Image with rotated content
        
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> rotated = rotate(v, 45)  # Rotate 45 degrees
        >>> rotated_fitted = rotate(v, 45, resize_output=True)  # Expand to fit
        
    Algorithm:
        - Convert angle to radians
        - Create rotation matrix
        - If resize_output=True, calculate new dimensions to fit entire image
        - Apply affine transformation with bilinear interpolation
        - Use scipy.ndimage.rotate for efficient rotation
        - Process each channel independently
        - Fill empty areas with specified fill_value
    """
    src_data = img_view.to_array()
    height, width, channels = src_data.shape
    
    # Determine fill value (default to black)
    if fill_value is None:
        fill_value = tuple([0] * channels)
    elif isinstance(fill_value, int):
        fill_value = tuple([fill_value] * channels)
    
    # For each channel, apply rotation
    rotated_channels = []
    for c in range(channels):
        # Use scipy's rotate function with bilinear interpolation
        rotated_channel = ndimage.rotate(
            src_data[:, :, c],
            angle,  # Angle in degrees (counter-clockwise)
            reshape=resize_output,  # Whether to expand output
            order=1,  # 1 = bilinear interpolation
            mode='constant',  # How to handle borders
            cval=fill_value[c]  # Fill value for borders
        )
        rotated_channels.append(rotated_channel)
    
    # Stack channels back together
    rotated_data = np.stack(rotated_channels, axis=-1)
    
    # Ensure correct dtype
    rotated_data = rotated_data.astype(src_data.dtype)
    
    return Image.from_array(rotated_data, img_view.pixel_type)


def scale(img_view: ImageView, 
          scale_x: float, 
          scale_y: Optional[float] = None,
          sampler: Optional[Sampler] = None) -> Image:
    """
    Scale an image by specified factors.
    
    Convenience function for resizing by scale factors rather than absolute dimensions.
    
    Args:
        img_view: Source image view to scale
        scale_x: Horizontal scale factor (e.g., 2.0 = double width)
        scale_y: Vertical scale factor (default: same as scale_x for uniform scaling)
        sampler: Interpolation method (default: bilinear)
        
    Returns:
        New Image with scaled content
        
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> doubled = scale(v, 2.0)  # Double size (200x200)
        >>> wide = scale(v, 2.0, 1.0)  # Double width only (200x100)
        
    Algorithm:
        - If scale_y not provided, use scale_x for uniform scaling
        - Calculate new dimensions: new_width = width * scale_x
        - Call resize_view() with calculated dimensions
    """
    if scale_y is None:
        scale_y = scale_x
    
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"Scale factors must be positive: {scale_x}, {scale_y}")
    
    # Calculate new dimensions
    new_width = int(img_view.width * scale_x)
    new_height = int(img_view.height * scale_y)
    
    # Ensure at least 1 pixel
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    return resize_view(img_view, new_width, new_height, sampler)


def crop(img_view: ImageView, 
         x: int, 
         y: int, 
         width: int, 
         height: int) -> Image:
    """
    Crop a rectangular region from an image.
    
    Extracts a rectangular sub-region and returns it as a new image.
    
    Args:
        img_view: Source image view to crop from
        x: Starting x-coordinate (left edge)
        y: Starting y-coordinate (top edge)
        width: Width of crop region
        height: Height of crop region
        
    Returns:
        New Image containing the cropped region
        
    Raises:
        ValueError: If crop region extends beyond image boundaries
        
    Example:
        >>> img = Image(200, 200, RGB8)
        >>> v = view(img)
        >>> cropped = crop(v, 50, 50, 100, 100)  # 100x100 region at (50,50)
        
    Algorithm:
        - Validate crop region is within bounds
        - Extract sub-array using NumPy slicing
        - Create new Image from extracted data
    """
    # Validate boundaries
    if x < 0 or y < 0:
        raise ValueError(f"Crop coordinates must be non-negative: x={x}, y={y}")
    if x + width > img_view.width or y + height > img_view.height:
        raise ValueError(
            f"Crop region extends beyond image: "
            f"region=({x},{y},{width},{height}), "
            f"image=({img_view.width},{img_view.height})"
        )
    
    # Extract crop region
    cropped_data = img_view.to_array()[y:y+height, x:x+width, :]
    
    return Image.from_array(cropped_data, img_view.pixel_type)


def transpose(img_view: ImageView) -> Image:
    """
    Transpose an image (swap width and height).
    
    Reflects the image across its diagonal. Width becomes height and vice versa.
    
    Args:
        img_view: Source image view to transpose
        
    Returns:
        New Image with transposed dimensions
        
    Example:
        >>> img = Image(100, 200, RGB8)  # 100 wide, 200 tall
        >>> v = view(img)
        >>> transposed = transpose(v)
        >>> transposed.dimensions()
        (200, 100)  # 200 wide, 100 tall
        
    Algorithm:
        - Use NumPy's transpose to swap height and width dimensions
        - Keep channel dimension in place
        - Create new Image from transposed data
    """
    # Transpose spatial dimensions (swap axes 0 and 1, keep axis 2)
    transposed_data = np.transpose(img_view.to_array(), (1, 0, 2))
    
    return Image.from_array(transposed_data, img_view.pixel_type)


__all__ = [
    'Sampler',
    'BilinearSampler',
    'NearestSampler',
    'bilinear_sampler',
    'nearest_sampler',
    'resize_view',
    'flip_up_down',
    'flip_left_right',
    'rotate',
    'scale',
    'crop',
    'transpose',
]
