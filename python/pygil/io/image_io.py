"""
PyGIL Image I/O Module

This module provides image reading and writing functionality,
replicating Boost.GIL's I/O extension from boost/gil/extension/io/.

Supports common image formats: PNG, JPEG, BMP, TIFF
Uses Pillow (PIL) as the backend for actual file I/O operations.
"""

from typing import Optional, Union, BinaryIO
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

# Import from core module
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.image import Image
from core.pixel import PixelType, RGB8, RGBA8, GRAY8, RGB16, RGBA16, GRAY16


class ImageTag:
    """
    Base class for image format tags.
    
    In GIL, tags are used to specify image formats at compile time.
    We use similar tag objects in Python for API consistency.
    
    GIL equivalent: boost::gil::jpeg_tag, boost::gil::png_tag
    """
    def __init__(self, format_name: str, extensions: tuple):
        """
        Initialize an image format tag.
        
        Args:
            format_name: PIL/Pillow format name (e.g., 'PNG', 'JPEG')
            extensions: Tuple of file extensions for this format
        """
        self.format_name = format_name
        self.extensions = extensions
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.format_name})"


# Format tag instances
# These mirror GIL's tag system: jpeg_tag(), png_tag(), etc.

class PNGTag(ImageTag):
    """Tag for PNG format."""
    def __init__(self):
        super().__init__('PNG', ('.png', '.PNG'))

class JPEGTag(ImageTag):
    """Tag for JPEG format."""
    def __init__(self):
        super().__init__('JPEG', ('.jpg', '.jpeg', '.JPG', '.JPEG'))

class BMPTag(ImageTag):
    """Tag for BMP format."""
    def __init__(self):
        super().__init__('BMP', ('.bmp', '.BMP'))

class TIFFTag(ImageTag):
    """Tag for TIFF format."""
    def __init__(self):
        super().__init__('TIFF', ('.tif', '.tiff', '.TIF', '.TIFF'))


# Singleton instances for convenience (mirroring GIL's tag usage)
def png_tag() -> PNGTag:
    """Get PNG format tag. GIL equivalent: boost::gil::png_tag()"""
    return PNGTag()

def jpeg_tag() -> JPEGTag:
    """Get JPEG format tag. GIL equivalent: boost::gil::jpeg_tag()"""
    return JPEGTag()

def bmp_tag() -> BMPTag:
    """Get BMP format tag. GIL equivalent: boost::gil::bmp_tag()"""
    return BMPTag()

def tiff_tag() -> TIFFTag:
    """Get TIFF format tag. GIL equivalent: boost::gil::tiff_tag()"""
    return TIFFTag()


def _detect_format(path: Union[str, Path]) -> ImageTag:
    """
    Auto-detect image format from file extension.
    
    Args:
        path: File path
        
    Returns:
        Appropriate ImageTag for the format
        
    Raises:
        ValueError: If format cannot be detected
        
    Algorithm:
        - Extract file extension from path
        - Match against known format extensions
        - Return corresponding tag
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext in ('.png',):
        return png_tag()
    elif ext in ('.jpg', '.jpeg'):
        return jpeg_tag()
    elif ext in ('.bmp',):
        return bmp_tag()
    elif ext in ('.tif', '.tiff'):
        return tiff_tag()
    else:
        raise ValueError(f"Cannot detect format from extension: {ext}")


def read_image(path: Union[str, Path], 
               tag: Optional[ImageTag] = None) -> Image:
    """
    Read an image from a file.
    
    This is the main image loading function, mirroring GIL's read_image().
    Automatically detects the format if not specified via tag.
    
    Args:
        path: Path to the image file
        tag: Optional format tag (auto-detected if None)
        
    Returns:
        Image object containing the loaded pixel data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the format is unsupported
        
    GIL equivalent: boost::gil::read_image(filename, image, tag)
    
    Example:
        >>> # Auto-detect format
        >>> img = read_image("photo.jpg")
        >>> 
        >>> # Explicit format
        >>> img = read_image("photo.jpg", jpeg_tag())
        
    Algorithm:
        - Detect format from extension if tag not provided
        - Open file using PIL/Pillow
        - Convert PIL image to NumPy array
        - Detect pixel type from array properties
        - Create and return PyGIL Image object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Auto-detect format if not specified
    if tag is None:
        tag = _detect_format(path)
    
    # Open image with PIL
    pil_img = PILImage.open(path)
    
    # Convert PIL image to NumPy array
    # PIL images are in (height, width, channels) format by default
    img_array = np.array(pil_img)
    
    # Handle grayscale images (may be 2D)
    if img_array.ndim == 2:
        # Add channel dimension for grayscale
        img_array = img_array[:, :, np.newaxis]
    
    # Detect pixel type from array
    pixel_type = _detect_pixel_type_from_array(img_array)
    
    # Create PyGIL Image from array
    return Image.from_array(img_array, pixel_type)


def write_image(img: Image, path: Union[str, Path], 
                tag: Optional[ImageTag] = None,
                quality: int = 95) -> None:
    """
    Write an image to a file.
    
    Saves the image to disk in the specified format.
    Format is auto-detected from extension if tag not provided.
    
    Args:
        img: Image object to save
        path: Output file path
        tag: Optional format tag (auto-detected if None)
        quality: JPEG quality (0-100, only used for JPEG format)
        
    Raises:
        ValueError: If the format is unsupported
        
    GIL equivalent: boost::gil::write_view(filename, view, tag)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> write_image(img, "output.png")
        >>> write_image(img, "output.jpg", jpeg_tag(), quality=90)
        
    Algorithm:
        - Detect format from extension if tag not provided
        - Convert PyGIL Image to NumPy array
        - Create PIL Image from array
        - Save using PIL with format-specific options
    """
    path = Path(path)
    
    # Auto-detect format if not specified
    if tag is None:
        tag = _detect_format(path)
    
    # Get image data as NumPy array
    img_array = img._data
    
    # Handle different channel counts for PIL
    if img.num_channels == 1:
        # Grayscale: remove channel dimension for PIL
        pil_img = PILImage.fromarray(img_array[:, :, 0], mode='L')
    elif img.num_channels == 3:
        # RGB
        if img.dtype == np.uint16:
            # PIL doesn't support 16-bit RGB well, convert to 8-bit
            img_array = (img_array.astype(np.float32) / 257).astype(np.uint8)
        pil_img = PILImage.fromarray(img_array, mode='RGB')
    elif img.num_channels == 4:
        # RGBA
        if img.dtype == np.uint16:
            # Convert 16-bit to 8-bit
            img_array = (img_array.astype(np.float32) / 257).astype(np.uint8)
        pil_img = PILImage.fromarray(img_array, mode='RGBA')
    else:
        raise ValueError(f"Unsupported channel count: {img.num_channels}")
    
    # Save with format-specific options
    if isinstance(tag, JPEGTag):
        # JPEG specific: quality parameter
        pil_img.save(path, format=tag.format_name, quality=quality)
    elif isinstance(tag, PNGTag):
        # PNG specific: lossless compression
        pil_img.save(path, format=tag.format_name, optimize=True)
    else:
        # Generic save
        pil_img.save(path, format=tag.format_name)


def write_view(view, path: Union[str, Path], 
               tag: Optional[ImageTag] = None,
               quality: int = 95) -> None:
    """
    Write an image view to a file.
    
    Convenience function that works with ImageView objects.
    Internally converts the view to an image and saves it.
    
    Args:
        view: ImageView object to save
        path: Output file path
        tag: Optional format tag (auto-detected if None)
        quality: JPEG quality (0-100, only used for JPEG format)
        
    GIL equivalent: boost::gil::write_view(filename, view, tag)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> write_view(v, "output.png", png_tag())
        
    Algorithm:
        - Convert view to Image (creates copy)
        - Call write_image() with the image
    """
    # Convert view to image
    img = view.copy()
    
    # Write the image
    write_image(img, path, tag, quality)


def _detect_pixel_type_from_array(array: np.ndarray) -> PixelType:
    """
    Detect pixel type from NumPy array properties.
    
    Internal helper function to determine the appropriate PixelType
    based on array shape and dtype.
    
    Args:
        array: NumPy array with image data
        
    Returns:
        Detected PixelType
        
    Raises:
        ValueError: If array properties don't match any supported pixel type
        
    Algorithm:
        - Check number of channels (last dimension)
        - Check dtype for bit depth
        - Map to appropriate PixelType enum
    """
    if array.ndim != 3:
        raise ValueError(f"Array must have 3 dimensions, got {array.ndim}")
    
    channels = array.shape[2]
    dtype = array.dtype
    
    # Determine bit depth
    if dtype == np.uint8:
        bit_depth = 8
    elif dtype == np.uint16:
        bit_depth = 16
    else:
        # Unsupported dtype, default to 8-bit
        bit_depth = 8
    
    # Map channels and bit depth to pixel type
    if channels == 1:
        return GRAY16 if bit_depth == 16 else GRAY8
    elif channels == 3:
        return RGB16 if bit_depth == 16 else RGB8
    elif channels == 4:
        return RGBA16 if bit_depth == 16 else RGBA8
    else:
        raise ValueError(f"Unsupported channel count: {channels}")


def read_and_convert_image(path: Union[str, Path], 
                           target_type: PixelType,
                           tag: Optional[ImageTag] = None) -> Image:
    """
    Read an image and convert it to a specific pixel type.
    
    Convenience function that combines reading and color conversion.
    Useful when you need an image in a specific format.
    
    Args:
        path: Path to the image file
        target_type: Desired pixel type for the result
        tag: Optional format tag (auto-detected if None)
        
    Returns:
        Image in the specified pixel type
        
    Example:
        >>> # Read a color image and convert to grayscale
        >>> gray_img = read_and_convert_image("photo.jpg", GRAY8)
        >>> 
        >>> # Read any image and ensure it's RGBA
        >>> rgba_img = read_and_convert_image("photo.png", RGBA8)
        
    Algorithm:
        - Read image from file
        - Check if conversion is needed
        - Apply color_convert() if types differ
    """
    # Import here to avoid circular dependency
    from ..core.color_convert import color_convert
    
    # Read the image
    img = read_image(path, tag)
    
    # Convert if necessary
    if img.pixel_type != target_type:
        img = color_convert(img, target_type)
    
    return img


__all__ = [
    'ImageTag',
    'PNGTag',
    'JPEGTag',
    'BMPTag',
    'TIFFTag',
    'png_tag',
    'jpeg_tag',
    'bmp_tag',
    'tiff_tag',
    'read_image',
    'write_image',
    'write_view',
    'read_and_convert_image',
]
