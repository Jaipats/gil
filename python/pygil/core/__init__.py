"""
PyGIL Core Module

This module contains the core image processing types and operations,
mirroring Boost.GIL's core functionality.

Components:
    - pixel: Pixel type definitions and channel operations
    - image: Image container class (data owner)
    - view: ImageView class (lightweight reference)
    - color_convert: Color space conversion utilities
"""

from .pixel import *
from .image import *
from .view import *
from .color_convert import *

__all__ = [
    # Pixel types
    'PixelType',
    'RGB8',
    'RGBA8',
    'GRAY8',
    'RGB16',
    'RGBA16',
    'GRAY16',
    
    # Image class
    'Image',
    
    # View functions
    'ImageView',
    'view',
    'const_view',
    'subimage_view',
    'flipped_up_down_view',
    'flipped_left_right_view',
    
    # Color conversion
    'color_convert',
    'rgb_to_gray',
    'gray_to_rgb',
    'rgb_to_rgba',
    'rgba_to_rgb',
]
