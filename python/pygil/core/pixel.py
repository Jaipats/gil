"""
PyGIL Pixel Types Module

This module defines pixel types and channel operations,
replicating Boost.GIL's pixel concepts from boost/gil/pixel.hpp.

In GIL, pixel types are templated classes like rgb8_pixel_t, rgba8_pixel_t.
Here we use Python enums and classes to represent these pixel formats.
"""

from enum import Enum
from typing import Tuple, Union
import numpy as np


class PixelType(Enum):
    """
    Enumeration of supported pixel formats.
    
    Mirrors GIL's pixel type system where each type specifies:
    - Color space (RGB, RGBA, Grayscale)
    - Bit depth per channel (8-bit, 16-bit)
    
    GIL equivalent: rgb8_pixel_t, rgba8_pixel_t, gray8_pixel_t, etc.
    """
    RGB8 = ('rgb', 8, 3)      # 8-bit RGB, 3 channels
    RGBA8 = ('rgba', 8, 4)    # 8-bit RGBA, 4 channels
    GRAY8 = ('gray', 8, 1)    # 8-bit grayscale, 1 channel
    RGB16 = ('rgb', 16, 3)    # 16-bit RGB, 3 channels
    RGBA16 = ('rgba', 16, 4)  # 16-bit RGBA, 4 channels
    GRAY16 = ('gray', 16, 1)  # 16-bit grayscale, 1 channel
    
    def __init__(self, color_space: str, bit_depth: int, num_channels: int):
        """
        Initialize pixel type with its properties.
        
        Args:
            color_space: Color space name ('rgb', 'rgba', 'gray')
            bit_depth: Bits per channel (8 or 16)
            num_channels: Number of channels (1, 3, or 4)
        """
        self.color_space = color_space
        self.bit_depth = bit_depth
        self.num_channels = num_channels
    
    @property
    def dtype(self) -> np.dtype:
        """
        Get the NumPy dtype for this pixel type.
        
        Returns:
            NumPy dtype (uint8 for 8-bit, uint16 for 16-bit)
            
        Algorithm:
            - Map bit depth to NumPy unsigned integer types
            - 8-bit -> np.uint8 (0-255 range)
            - 16-bit -> np.uint16 (0-65535 range)
        """
        if self.bit_depth == 8:
            return np.uint8
        elif self.bit_depth == 16:
            return np.uint16
        else:
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")
    
    @property
    def max_value(self) -> int:
        """
        Get the maximum value for a channel in this pixel type.
        
        Returns:
            Maximum channel value (255 for 8-bit, 65535 for 16-bit)
            
        This is used for normalization and value clamping operations.
        """
        return (1 << self.bit_depth) - 1  # 2^bit_depth - 1
    
    def __str__(self) -> str:
        """String representation of pixel type."""
        return f"{self.color_space}{self.bit_depth}"
    
    def __repr__(self) -> str:
        """Detailed representation of pixel type."""
        return f"PixelType.{self.name}({self.color_space}, {self.bit_depth}-bit, {self.num_channels} channels)"


# Convenience constants for commonly used pixel types
# These mirror GIL's typedefs: rgb8_pixel_t, rgba8_pixel_t, etc.
RGB8 = PixelType.RGB8
RGBA8 = PixelType.RGBA8
GRAY8 = PixelType.GRAY8
RGB16 = PixelType.RGB16
RGBA16 = PixelType.RGBA16
GRAY16 = PixelType.GRAY16


class ChannelAccessor:
    """
    Provides named access to pixel channels.
    
    Mirrors GIL's channel accessor concepts where you can access
    individual color channels by name (red, green, blue, alpha).
    
    GIL equivalent: get_color(pixel, red_t()), get_color(pixel, green_t())
    """
    
    @staticmethod
    def red(pixel: np.ndarray) -> Union[int, np.ndarray]:
        """
        Extract red channel from RGB/RGBA pixel(s).
        
        Args:
            pixel: Pixel array with shape (..., C) where C >= 3
            
        Returns:
            Red channel value(s)
            
        Algorithm:
            - Red is always channel 0 in RGB/RGBA format
        """
        if pixel.shape[-1] < 3:
            raise ValueError("Pixel must have at least 3 channels for red access")
        return pixel[..., 0]
    
    @staticmethod
    def green(pixel: np.ndarray) -> Union[int, np.ndarray]:
        """
        Extract green channel from RGB/RGBA pixel(s).
        
        Args:
            pixel: Pixel array with shape (..., C) where C >= 3
            
        Returns:
            Green channel value(s)
            
        Algorithm:
            - Green is always channel 1 in RGB/RGBA format
        """
        if pixel.shape[-1] < 3:
            raise ValueError("Pixel must have at least 3 channels for green access")
        return pixel[..., 1]
    
    @staticmethod
    def blue(pixel: np.ndarray) -> Union[int, np.ndarray]:
        """
        Extract blue channel from RGB/RGBA pixel(s).
        
        Args:
            pixel: Pixel array with shape (..., C) where C >= 3
            
        Returns:
            Blue channel value(s)
            
        Algorithm:
            - Blue is always channel 2 in RGB/RGBA format
        """
        if pixel.shape[-1] < 3:
            raise ValueError("Pixel must have at least 3 channels for blue access")
        return pixel[..., 2]
    
    @staticmethod
    def alpha(pixel: np.ndarray) -> Union[int, np.ndarray]:
        """
        Extract alpha channel from RGBA pixel(s).
        
        Args:
            pixel: Pixel array with shape (..., 4)
            
        Returns:
            Alpha channel value(s)
            
        Algorithm:
            - Alpha is always channel 3 in RGBA format
        """
        if pixel.shape[-1] != 4:
            raise ValueError("Pixel must have 4 channels for alpha access")
        return pixel[..., 3]
    
    @staticmethod
    def gray(pixel: np.ndarray) -> Union[int, np.ndarray]:
        """
        Extract grayscale value from grayscale pixel(s).
        
        Args:
            pixel: Pixel array with shape (..., 1)
            
        Returns:
            Grayscale value(s)
            
        Algorithm:
            - Grayscale is stored in channel 0
        """
        if pixel.shape[-1] != 1:
            raise ValueError("Pixel must have 1 channel for gray access")
        return pixel[..., 0]


def create_pixel(pixel_type: PixelType, values: Tuple[int, ...]) -> np.ndarray:
    """
    Create a pixel with specified type and channel values.
    
    This is a factory function for creating individual pixels,
    similar to GIL's pixel construction.
    
    Args:
        pixel_type: Type of pixel to create (RGB8, RGBA8, etc.)
        values: Tuple of channel values
        
    Returns:
        NumPy array representing the pixel with shape (num_channels,)
        
    Raises:
        ValueError: If number of values doesn't match pixel type channels
        
    Example:
        >>> pixel = create_pixel(RGB8, (255, 128, 0))  # Orange pixel
        >>> pixel.shape
        (3,)
        >>> pixel.dtype
        dtype('uint8')
        
    Algorithm:
        - Validate that values count matches pixel type channels
        - Create NumPy array with appropriate dtype
        - Clamp values to valid range for the bit depth
    """
    if len(values) != pixel_type.num_channels:
        raise ValueError(
            f"Expected {pixel_type.num_channels} values for {pixel_type}, "
            f"got {len(values)}"
        )
    
    # Create pixel array with appropriate dtype
    pixel = np.array(values, dtype=pixel_type.dtype)
    
    # Clamp values to valid range
    pixel = np.clip(pixel, 0, pixel_type.max_value)
    
    return pixel


def pixel_cast(pixel: np.ndarray, target_type: PixelType) -> np.ndarray:
    """
    Cast pixel from one type to another with proper scaling.
    
    Mirrors GIL's pixel type conversion which handles bit depth changes.
    For example, converting from 8-bit to 16-bit scales values appropriately.
    
    Args:
        pixel: Source pixel array
        target_type: Target pixel type
        
    Returns:
        Pixel array in target type
        
    Algorithm:
        - Detect source bit depth from dtype
        - If bit depths differ, scale values proportionally
        - 8-bit to 16-bit: multiply by 257 (65535/255)
        - 16-bit to 8-bit: divide by 257
        - Cast to target dtype
        
    Example:
        >>> pixel_8bit = np.array([255, 128, 0], dtype=np.uint8)
        >>> pixel_16bit = pixel_cast(pixel_8bit, RGB16)
        >>> pixel_16bit[0]  # 255 * 257 = 65535
        65535
    """
    # Determine source bit depth
    if pixel.dtype == np.uint8:
        src_bit_depth = 8
    elif pixel.dtype == np.uint16:
        src_bit_depth = 16
    else:
        raise ValueError(f"Unsupported source dtype: {pixel.dtype}")
    
    # If bit depths are the same, just cast
    if src_bit_depth == target_type.bit_depth:
        return pixel.astype(target_type.dtype)
    
    # Scale values when converting between bit depths
    if src_bit_depth == 8 and target_type.bit_depth == 16:
        # 8-bit to 16-bit: scale up by factor of 257
        scaled = pixel.astype(np.float32) * 257.0
        return scaled.astype(target_type.dtype)
    elif src_bit_depth == 16 and target_type.bit_depth == 8:
        # 16-bit to 8-bit: scale down by factor of 257
        scaled = pixel.astype(np.float32) / 257.0
        return scaled.astype(target_type.dtype)
    else:
        raise ValueError(f"Unsupported bit depth conversion: {src_bit_depth} to {target_type.bit_depth}")


__all__ = [
    'PixelType',
    'RGB8',
    'RGBA8',
    'GRAY8',
    'RGB16',
    'RGBA16',
    'GRAY16',
    'ChannelAccessor',
    'create_pixel',
    'pixel_cast',
]
