"""
PyGIL I/O Module

This module provides image input/output operations,
mirroring Boost.GIL's I/O extension functionality.

Components:
    - image_io: Read and write images in various formats
"""

from .image_io import *

__all__ = [
    'read_image',
    'write_image',
    'write_view',
    'png_tag',
    'jpeg_tag',
    'bmp_tag',
    'tiff_tag',
]
