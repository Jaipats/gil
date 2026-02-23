"""
PyGIL Color Conversion Module

This module provides color space conversion utilities,
replicating Boost.GIL's color conversion from boost/gil/color_convert.hpp.

Supported conversions:
- RGB ↔ Grayscale
- RGB ↔ RGBA (with alpha channel handling)
- Grayscale ↔ RGBA
"""

from typing import Optional
import numpy as np
from .image import Image
from .pixel import PixelType, RGB8, RGBA8, GRAY8, RGB16, RGBA16, GRAY16


def color_convert(src_img: Image, target_type: PixelType, 
                  alpha_value: Optional[int] = None) -> Image:
    """
    Convert an image from one color space to another.
    
    This is the main color conversion function, analogous to GIL's
    color_convert() which handles conversions between different pixel types.
    
    Args:
        src_img: Source image to convert
        target_type: Target pixel type to convert to
        alpha_value: Alpha value to use when converting to RGBA (default: max value)
        
    Returns:
        New Image in the target color space
        
    Raises:
        ValueError: If the conversion is not supported
        
    GIL equivalent: boost::gil::color_convert(src, dst)
    
    Example:
        >>> rgb_img = Image(100, 100, RGB8)
        >>> gray_img = color_convert(rgb_img, GRAY8)
        >>> rgba_img = color_convert(rgb_img, RGBA8, alpha_value=255)
        
    Algorithm:
        - Detect source and target color spaces
        - Route to appropriate conversion function
        - Handle bit depth conversions if needed
    """
    src_type = src_img.pixel_type
    
    # Check if only bit depth differs (same color space)
    if (src_type.color_space == target_type.color_space and 
        src_type.bit_depth != target_type.bit_depth):
        return _convert_bit_depth(src_img, target_type)
    
    # Route to appropriate conversion function based on color spaces
    conversion_key = (src_type.color_space, target_type.color_space)
    
    if conversion_key == ('rgb', 'gray'):
        return rgb_to_gray(src_img, target_type)
    elif conversion_key == ('gray', 'rgb'):
        return gray_to_rgb(src_img, target_type)
    elif conversion_key == ('rgb', 'rgba'):
        return rgb_to_rgba(src_img, target_type, alpha_value)
    elif conversion_key == ('rgba', 'rgb'):
        return rgba_to_rgb(src_img, target_type)
    elif conversion_key == ('rgba', 'gray'):
        return rgba_to_gray(src_img, target_type)
    elif conversion_key == ('gray', 'rgba'):
        return gray_to_rgba(src_img, target_type, alpha_value)
    elif conversion_key[0] == conversion_key[1]:
        # Same color space, no conversion needed (already handled bit depth above)
        return src_img.copy()
    else:
        raise ValueError(
            f"Unsupported color conversion: {src_type.color_space} to {target_type.color_space}"
        )


def rgb_to_gray(src_img: Image, target_type: Optional[PixelType] = None) -> Image:
    """
    Convert RGB image to grayscale.
    
    Uses the standard luminance formula recommended by ITU-R BT.601:
    Gray = 0.299*R + 0.587*G + 0.114*B
    
    This formula weights green more heavily because human eyes are
    more sensitive to green wavelengths.
    
    Args:
        src_img: Source RGB image
        target_type: Target grayscale type (auto-detected if None)
        
    Returns:
        New grayscale Image
        
    GIL equivalent: color_converted_view<gray8_pixel_t>(rgb_view)
    
    Example:
        >>> rgb_img = Image(100, 100, RGB8)
        >>> gray_img = rgb_to_gray(rgb_img)
        
    Algorithm:
        - Extract R, G, B channels from source
        - Compute weighted sum: 0.299*R + 0.587*G + 0.114*B
        - Clamp result to valid range for target bit depth
        - Create new grayscale image with result
    """
    if src_img.pixel_type.color_space not in ('rgb', 'rgba'):
        raise ValueError(f"Source must be RGB or RGBA, got {src_img.pixel_type.color_space}")
    
    # Auto-detect target type if not specified
    if target_type is None:
        target_type = GRAY16 if src_img.pixel_type.bit_depth == 16 else GRAY8
    
    # ITU-R BT.601 luminance coefficients
    # These are the standard weights for converting RGB to grayscale
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    
    # Extract RGB channels (ignore alpha if present)
    rgb_data = src_img._data[:, :, :3].astype(np.float32)
    
    # Compute weighted sum along channel axis
    gray_data = np.dot(rgb_data, weights)
    
    # Clamp to valid range and convert to target dtype
    gray_data = np.clip(gray_data, 0, target_type.max_value)
    gray_data = gray_data.astype(target_type.dtype)
    
    # Reshape to include channel dimension
    gray_data = gray_data[:, :, np.newaxis]
    
    # Create new grayscale image
    return Image.from_array(gray_data, target_type)


def gray_to_rgb(src_img: Image, target_type: Optional[PixelType] = None) -> Image:
    """
    Convert grayscale image to RGB by replicating the gray value.
    
    Creates an RGB image where R = G = B = gray value.
    
    Args:
        src_img: Source grayscale image
        target_type: Target RGB type (auto-detected if None)
        
    Returns:
        New RGB Image
        
    GIL equivalent: color_converted_view<rgb8_pixel_t>(gray_view)
    
    Example:
        >>> gray_img = Image(100, 100, GRAY8)
        >>> rgb_img = gray_to_rgb(gray_img)
        
    Algorithm:
        - Extract grayscale channel
        - Replicate to create 3 identical channels (R=G=B)
        - Handle bit depth conversion if needed
    """
    if src_img.pixel_type.color_space != 'gray':
        raise ValueError(f"Source must be grayscale, got {src_img.pixel_type.color_space}")
    
    # Auto-detect target type if not specified
    if target_type is None:
        target_type = RGB16 if src_img.pixel_type.bit_depth == 16 else RGB8
    
    # Replicate grayscale channel to create RGB
    gray_channel = src_img._data[:, :, 0]
    rgb_data = np.stack([gray_channel, gray_channel, gray_channel], axis=-1)
    
    # Handle bit depth conversion if needed
    if src_img.pixel_type.bit_depth != target_type.bit_depth:
        if src_img.pixel_type.bit_depth == 8 and target_type.bit_depth == 16:
            rgb_data = (rgb_data.astype(np.float32) * 257).astype(target_type.dtype)
        elif src_img.pixel_type.bit_depth == 16 and target_type.bit_depth == 8:
            rgb_data = (rgb_data.astype(np.float32) / 257).astype(target_type.dtype)
    
    return Image.from_array(rgb_data, target_type)


def rgb_to_rgba(src_img: Image, target_type: Optional[PixelType] = None,
                alpha_value: Optional[int] = None) -> Image:
    """
    Convert RGB image to RGBA by adding an alpha channel.
    
    Args:
        src_img: Source RGB image
        target_type: Target RGBA type (auto-detected if None)
        alpha_value: Alpha value for all pixels (default: max value = opaque)
        
    Returns:
        New RGBA Image
        
    GIL equivalent: color_converted_view<rgba8_pixel_t>(rgb_view)
    
    Example:
        >>> rgb_img = Image(100, 100, RGB8)
        >>> rgba_img = rgb_to_rgba(rgb_img, alpha_value=255)  # Fully opaque
        
    Algorithm:
        - Copy RGB channels from source
        - Create alpha channel with specified value
        - Concatenate to form RGBA image
        - Handle bit depth conversion if needed
    """
    if src_img.pixel_type.color_space != 'rgb':
        raise ValueError(f"Source must be RGB, got {src_img.pixel_type.color_space}")
    
    # Auto-detect target type if not specified
    if target_type is None:
        target_type = RGBA16 if src_img.pixel_type.bit_depth == 16 else RGBA8
    
    # Default alpha to maximum value (fully opaque)
    if alpha_value is None:
        alpha_value = target_type.max_value
    
    # Create RGBA data by adding alpha channel
    height, width = src_img.height, src_img.width
    rgba_data = np.zeros((height, width, 4), dtype=target_type.dtype)
    
    # Copy RGB channels
    rgba_data[:, :, :3] = src_img._data
    
    # Set alpha channel
    rgba_data[:, :, 3] = alpha_value
    
    # Handle bit depth conversion if needed
    if src_img.pixel_type.bit_depth != target_type.bit_depth:
        if src_img.pixel_type.bit_depth == 8 and target_type.bit_depth == 16:
            rgba_data[:, :, :3] = (rgba_data[:, :, :3].astype(np.float32) * 257).astype(target_type.dtype)
        elif src_img.pixel_type.bit_depth == 16 and target_type.bit_depth == 8:
            rgba_data[:, :, :3] = (rgba_data[:, :, :3].astype(np.float32) / 257).astype(target_type.dtype)
    
    return Image.from_array(rgba_data, target_type)


def rgba_to_rgb(src_img: Image, target_type: Optional[PixelType] = None) -> Image:
    """
    Convert RGBA image to RGB by discarding the alpha channel.
    
    Simply removes the alpha channel, keeping only RGB.
    For alpha blending with a background, use a different function.
    
    Args:
        src_img: Source RGBA image
        target_type: Target RGB type (auto-detected if None)
        
    Returns:
        New RGB Image
        
    GIL equivalent: color_converted_view<rgb8_pixel_t>(rgba_view)
    
    Example:
        >>> rgba_img = Image(100, 100, RGBA8)
        >>> rgb_img = rgba_to_rgb(rgba_img)
        
    Algorithm:
        - Extract first 3 channels (RGB)
        - Discard channel 3 (alpha)
        - Handle bit depth conversion if needed
    """
    if src_img.pixel_type.color_space != 'rgba':
        raise ValueError(f"Source must be RGBA, got {src_img.pixel_type.color_space}")
    
    # Auto-detect target type if not specified
    if target_type is None:
        target_type = RGB16 if src_img.pixel_type.bit_depth == 16 else RGB8
    
    # Extract RGB channels (discard alpha)
    rgb_data = src_img._data[:, :, :3].copy()
    
    # Handle bit depth conversion if needed
    if src_img.pixel_type.bit_depth != target_type.bit_depth:
        if src_img.pixel_type.bit_depth == 8 and target_type.bit_depth == 16:
            rgb_data = (rgb_data.astype(np.float32) * 257).astype(target_type.dtype)
        elif src_img.pixel_type.bit_depth == 16 and target_type.bit_depth == 8:
            rgb_data = (rgb_data.astype(np.float32) / 257).astype(target_type.dtype)
    
    return Image.from_array(rgb_data, target_type)


def rgba_to_gray(src_img: Image, target_type: Optional[PixelType] = None) -> Image:
    """
    Convert RGBA image to grayscale (discards alpha).
    
    Converts RGB channels to grayscale using standard luminance formula,
    then discards the alpha channel.
    
    Args:
        src_img: Source RGBA image
        target_type: Target grayscale type (auto-detected if None)
        
    Returns:
        New grayscale Image
        
    Algorithm:
        - First convert RGBA to RGB (discard alpha)
        - Then convert RGB to grayscale
    """
    if src_img.pixel_type.color_space != 'rgba':
        raise ValueError(f"Source must be RGBA, got {src_img.pixel_type.color_space}")
    
    # Convert via RGB
    rgb_img = rgba_to_rgb(src_img)
    return rgb_to_gray(rgb_img, target_type)


def gray_to_rgba(src_img: Image, target_type: Optional[PixelType] = None,
                 alpha_value: Optional[int] = None) -> Image:
    """
    Convert grayscale image to RGBA.
    
    Replicates gray value to RGB channels and adds alpha channel.
    
    Args:
        src_img: Source grayscale image
        target_type: Target RGBA type (auto-detected if None)
        alpha_value: Alpha value for all pixels (default: max value = opaque)
        
    Returns:
        New RGBA Image
        
    Algorithm:
        - First convert grayscale to RGB
        - Then convert RGB to RGBA with alpha
    """
    if src_img.pixel_type.color_space != 'gray':
        raise ValueError(f"Source must be grayscale, got {src_img.pixel_type.color_space}")
    
    # Convert via RGB
    rgb_img = gray_to_rgb(src_img)
    return rgb_to_rgba(rgb_img, target_type, alpha_value)


def _convert_bit_depth(src_img: Image, target_type: PixelType) -> Image:
    """
    Convert image bit depth while keeping the same color space.
    
    Internal helper function for converting between 8-bit and 16-bit
    representations of the same color space.
    
    Args:
        src_img: Source image
        target_type: Target pixel type (same color space, different bit depth)
        
    Returns:
        New Image with converted bit depth
        
    Algorithm:
        - 8-bit to 16-bit: multiply by 257 (65535/255)
        - 16-bit to 8-bit: divide by 257
        - Preserve all channels (works for RGB, RGBA, Gray)
    """
    src_depth = src_img.pixel_type.bit_depth
    tgt_depth = target_type.bit_depth
    
    if src_depth == tgt_depth:
        return src_img.copy()
    
    data = src_img._data.astype(np.float32)
    
    if src_depth == 8 and tgt_depth == 16:
        # 8-bit to 16-bit: scale up
        converted_data = (data * 257).astype(target_type.dtype)
    elif src_depth == 16 and tgt_depth == 8:
        # 16-bit to 8-bit: scale down
        converted_data = (data / 257).astype(target_type.dtype)
    else:
        raise ValueError(f"Unsupported bit depth conversion: {src_depth} to {tgt_depth}")
    
    return Image.from_array(converted_data, target_type)


__all__ = [
    'color_convert',
    'rgb_to_gray',
    'gray_to_rgb',
    'rgb_to_rgba',
    'rgba_to_rgb',
    'rgba_to_gray',
    'gray_to_rgba',
]
