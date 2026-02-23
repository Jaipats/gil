"""
PyGIL Processing Module

This module provides image processing operations,
mirroring Boost.GIL's image processing functionality.

Components:
    - transform: Image transformations (resize, flip, rotate)
    - convolve: Convolution operations with kernels
"""

from .transform import *
from .convolve import *

__all__ = [
    # Transform operations
    'resize_view',
    'flip_up_down',
    'flip_left_right',
    'rotate',
    'bilinear_sampler',
    'nearest_sampler',
    
    # Convolution operations
    'Kernel1D',
    'Kernel2D',
    'convolve_rows',
    'convolve_cols',
    'convolve_2d',
    'gaussian_kernel',
    'box_blur_kernel',
]
