"""
PyGIL: Python Generic Image Library

A Python library that replicates Boost.GIL's API and functionality using NumPy.
Provides efficient, generic image processing with a clean, type-safe interface.

This module exports the main PyGIL API, mirroring Boost.GIL's namespace structure.

Example usage:
    >>> import pygil as gil
    >>> 
    >>> # Read an image
    >>> img = gil.read_image("photo.jpg")
    >>> 
    >>> # Create a view
    >>> v = gil.view(img)
    >>> 
    >>> # Resize
    >>> resized = gil.resize_view(v, 400, 300)
    >>> 
    >>> # Convert to grayscale
    >>> gray = gil.color_convert(img, gil.GRAY8)
    >>> 
    >>> # Apply convolution
    >>> kernel = gil.gaussian_kernel(sigma=2.0)
    >>> blurred = gil.convolve_rows(gil.view(gray), kernel)
    >>> 
    >>> # Save result
    >>> gil.write_image(blurred, "output.jpg")

Module structure mirrors Boost.GIL:
    - Core types: Image, ImageView, PixelType
    - I/O: read_image, write_image, format tags
    - Processing: transformations, convolution
    - Color conversion: color_convert, rgb_to_gray, etc.
"""

__version__ = '0.1.0'
__author__ = 'PyGIL Contributors'

# Core module exports
from .core.pixel import (
    PixelType,
    RGB8, RGBA8, GRAY8,
    RGB16, RGBA16, GRAY16,
    ChannelAccessor,
    create_pixel,
    pixel_cast,
)

from .core.image import Image

from .core.view import (
    ImageView,
    view,
    const_view,
    subimage_view,
    flipped_up_down_view,
    flipped_left_right_view,
    rotated_view,
)

from .core.color_convert import (
    color_convert,
    rgb_to_gray,
    gray_to_rgb,
    rgb_to_rgba,
    rgba_to_rgb,
    rgba_to_gray,
    gray_to_rgba,
)

# I/O module exports
from .io.image_io import (
    read_image,
    write_image,
    write_view,
    read_and_convert_image,
    png_tag,
    jpeg_tag,
    bmp_tag,
    tiff_tag,
    ImageTag,
)

# Processing module exports - Transform
from .processing.transform import (
    resize_view,
    flip_up_down,
    flip_left_right,
    rotate,
    scale,
    crop,
    transpose,
    bilinear_sampler,
    nearest_sampler,
    BilinearSampler,
    NearestSampler,
)

# Processing module exports - Convolution
from .processing.convolve import (
    Kernel1D,
    Kernel2D,
    convolve_rows,
    convolve_cols,
    convolve_2d,
    gaussian_kernel,
    box_blur_kernel,
    sharpen_kernel,
    edge_detect_kernel,
)

# Define __all__ for clean namespace
__all__ = [
    # Version info
    '__version__',
    
    # Pixel types
    'PixelType',
    'RGB8', 'RGBA8', 'GRAY8',
    'RGB16', 'RGBA16', 'GRAY16',
    'ChannelAccessor',
    'create_pixel',
    'pixel_cast',
    
    # Image and View
    'Image',
    'ImageView',
    'view',
    'const_view',
    'subimage_view',
    'flipped_up_down_view',
    'flipped_left_right_view',
    'rotated_view',
    
    # Color conversion
    'color_convert',
    'rgb_to_gray',
    'gray_to_rgb',
    'rgb_to_rgba',
    'rgba_to_rgb',
    'rgba_to_gray',
    'gray_to_rgba',
    
    # I/O
    'read_image',
    'write_image',
    'write_view',
    'read_and_convert_image',
    'png_tag',
    'jpeg_tag',
    'bmp_tag',
    'tiff_tag',
    'ImageTag',
    
    # Transformations
    'resize_view',
    'flip_up_down',
    'flip_left_right',
    'rotate',
    'scale',
    'crop',
    'transpose',
    'bilinear_sampler',
    'nearest_sampler',
    'BilinearSampler',
    'NearestSampler',
    
    # Convolution
    'Kernel1D',
    'Kernel2D',
    'convolve_rows',
    'convolve_cols',
    'convolve_2d',
    'gaussian_kernel',
    'box_blur_kernel',
    'sharpen_kernel',
    'edge_detect_kernel',
]


def get_version() -> str:
    """
    Get the PyGIL version string.
    
    Returns:
        Version string in format 'major.minor.patch'
    """
    return __version__


def info():
    """
    Print PyGIL library information.
    
    Displays version, supported features, and basic usage information.
    """
    print(f"PyGIL - Python Generic Image Library v{__version__}")
    print("=" * 60)
    print("\nA Python library replicating Boost.GIL functionality")
    print("\nSupported features:")
    print("  - Image types: RGB8, RGBA8, GRAY8, RGB16, RGBA16, GRAY16")
    print("  - I/O formats: PNG, JPEG, BMP, TIFF")
    print("  - Transformations: resize, flip, rotate, crop")
    print("  - Convolution: 1D/2D kernels, Gaussian blur, edge detection")
    print("  - Color conversion: RGB ↔ Grayscale ↔ RGBA")
    print("\nQuick start:")
    print("  import pygil as gil")
    print("  img = gil.read_image('photo.jpg')")
    print("  gray = gil.color_convert(img, gil.GRAY8)")
    print("  gil.write_image(gray, 'output.jpg')")
    print("\nFor full documentation, see README.md")
