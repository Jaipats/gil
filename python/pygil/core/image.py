"""
PyGIL Image Module

This module defines the Image class, the primary container for image data,
replicating Boost.GIL's image concepts from boost/gil/image.hpp.

In GIL, images own their pixel data and provide memory management.
Here we use NumPy arrays as the underlying data storage.
"""

from typing import Optional, Tuple, Union
import numpy as np
from .pixel import PixelType, RGB8, RGBA8, GRAY8


class Image:
    """
    Image container class that owns pixel data.
    
    Mirrors Boost.GIL's image<T> class which is the primary data owner.
    In GIL's design philosophy:
    - Image owns the data (memory allocation)
    - ImageView provides access to the data (no ownership)
    
    The data is stored as a NumPy array with shape (height, width, channels).
    This matches the row-major memory layout used in most image libraries.
    
    GIL equivalent: boost::gil::image<PixelType>
    
    Attributes:
        _data: NumPy array storing pixel data (shape: height x width x channels)
        _pixel_type: PixelType enum indicating the pixel format
    """
    
    def __init__(self, width: int, height: int, pixel_type: PixelType = RGB8):
        """
        Create a new image with specified dimensions and pixel type.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            pixel_type: Type of pixels (RGB8, RGBA8, GRAY8, etc.)
            
        Raises:
            ValueError: If width or height is non-positive
            
        Example:
            >>> img = Image(800, 600, RGB8)
            >>> img.width
            800
            >>> img.height
            600
            >>> img.num_channels
            3
            
        Algorithm:
            - Validate dimensions are positive
            - Create NumPy array with shape (height, width, channels)
            - Initialize with zeros (black image)
            - Store pixel type for later reference
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensions must be positive: width={width}, height={height}")
        
        self._pixel_type = pixel_type
        
        # Create NumPy array with shape (height, width, channels)
        # Initialize with zeros (black image)
        self._data = np.zeros(
            (height, width, pixel_type.num_channels),
            dtype=pixel_type.dtype
        )
    
    @classmethod
    def from_array(cls, array: np.ndarray, pixel_type: Optional[PixelType] = None) -> 'Image':
        """
        Create an Image from an existing NumPy array.
        
        This factory method wraps existing array data in an Image object,
        similar to GIL's interleaved_view() for existing memory.
        
        Args:
            array: NumPy array with shape (height, width, channels)
            pixel_type: Optional pixel type (auto-detected if not provided)
            
        Returns:
            New Image object wrapping the array data
            
        Raises:
            ValueError: If array shape or dtype is invalid
            
        Algorithm:
            - Validate array has 3 dimensions (height, width, channels)
            - Auto-detect pixel type from shape and dtype if not provided
            - Create Image object and replace its data with the array
            - Optionally copy data to ensure ownership
        """
        if array.ndim != 3:
            raise ValueError(f"Array must have 3 dimensions (H, W, C), got shape {array.shape}")
        
        height, width, channels = array.shape
        
        # Auto-detect pixel type if not provided
        if pixel_type is None:
            pixel_type = cls._detect_pixel_type(channels, array.dtype)
        
        # Create image with detected dimensions
        img = cls.__new__(cls)
        img._pixel_type = pixel_type
        
        # Copy the array to ensure this image owns its data
        img._data = array.astype(pixel_type.dtype, copy=True)
        
        return img
    
    @staticmethod
    def _detect_pixel_type(channels: int, dtype: np.dtype) -> PixelType:
        """
        Auto-detect pixel type from array properties.
        
        Args:
            channels: Number of color channels
            dtype: NumPy data type
            
        Returns:
            Detected PixelType
            
        Raises:
            ValueError: If channel count or dtype is unsupported
            
        Algorithm:
            - Determine bit depth from dtype (uint8 -> 8-bit, uint16 -> 16-bit)
            - Map channel count to color space:
              * 1 channel -> Grayscale
              * 3 channels -> RGB
              * 4 channels -> RGBA
        """
        # Determine bit depth
        if dtype == np.uint8:
            bit_depth = 8
        elif dtype == np.uint16:
            bit_depth = 16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Map channels to pixel type
        if channels == 1:
            return GRAY16 if bit_depth == 16 else GRAY8
        elif channels == 3:
            return RGB16 if bit_depth == 16 else RGB8
        elif channels == 4:
            return RGBA16 if bit_depth == 16 else RGBA8
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    
    def to_array(self) -> np.ndarray:
        """
        Get the underlying NumPy array.
        
        Returns a copy of the internal data to prevent external modifications.
        For zero-copy access, use ImageView instead.
        
        Returns:
            Copy of the pixel data as NumPy array
            
        GIL equivalent: Accessing pixels through image iterators
        """
        return self._data.copy()
    
    def copy(self) -> 'Image':
        """
        Create a deep copy of this image.
        
        Returns:
            New Image object with copied data
            
        Algorithm:
            - Create new Image with same dimensions and pixel type
            - Copy all pixel data to the new image
        """
        new_img = Image(self.width, self.height, self._pixel_type)
        new_img._data = self._data.copy()
        return new_img
    
    def dimensions(self) -> Tuple[int, int]:
        """
        Get image dimensions as (width, height) tuple.
        
        Returns:
            Tuple of (width, height)
            
        GIL equivalent: img.dimensions()
        """
        return (self.width, self.height)
    
    def fill(self, value: Union[int, Tuple[int, ...], np.ndarray]) -> None:
        """
        Fill entire image with a constant pixel value.
        
        Args:
            value: Pixel value to fill with (scalar or tuple of channel values)
            
        Example:
            >>> img = Image(100, 100, RGB8)
            >>> img.fill((255, 0, 0))  # Fill with red
            
        Algorithm:
            - Convert value to appropriate shape for broadcasting
            - Use NumPy's fill or broadcasting to set all pixels
        """
        if isinstance(value, (int, float)):
            # Single value for all channels
            self._data.fill(value)
        else:
            # Tuple or array of channel values
            value_array = np.array(value, dtype=self._pixel_type.dtype)
            if len(value_array) != self.num_channels:
                raise ValueError(
                    f"Value must have {self.num_channels} channels, "
                    f"got {len(value_array)}"
                )
            self._data[:] = value_array
    
    def recreate(self, width: int, height: int, pixel_type: Optional[PixelType] = None) -> None:
        """
        Recreate the image with new dimensions and/or pixel type.
        
        This destroys existing data and allocates new storage,
        similar to GIL's image.recreate() method.
        
        Args:
            width: New width in pixels
            height: New height in pixels
            pixel_type: New pixel type (keeps existing if None)
            
        Algorithm:
            - Update pixel type if provided
            - Deallocate old data (automatic with NumPy)
            - Allocate new array with new dimensions
        """
        if pixel_type is not None:
            self._pixel_type = pixel_type
        
        self._data = np.zeros(
            (height, width, self._pixel_type.num_channels),
            dtype=self._pixel_type.dtype
        )
    
    # Properties for convenient access
    
    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._data.shape[1]
    
    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._data.shape[0]
    
    @property
    def num_channels(self) -> int:
        """Number of color channels."""
        return self._data.shape[2]
    
    @property
    def pixel_type(self) -> PixelType:
        """Pixel type of this image."""
        return self._pixel_type
    
    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the pixel data."""
        return self._data.dtype
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Shape of image as (height, width, channels).
        
        Note: This matches NumPy convention but differs from dimensions()
        which returns (width, height) to match GIL convention.
        """
        return self._data.shape
    
    @property
    def size(self) -> int:
        """Total number of pixels (width * height)."""
        return self.width * self.height
    
    def __getitem__(self, key):
        """
        Get pixel or region using array indexing.
        
        Args:
            key: Index or slice for accessing pixels
            
        Returns:
            Pixel value(s) at the specified location
            
        Example:
            >>> img = Image(100, 100, RGB8)
            >>> pixel = img[50, 50]  # Get pixel at (row=50, col=50)
            >>> region = img[10:20, 30:40]  # Get rectangular region
        """
        return self._data[key]
    
    def __setitem__(self, key, value):
        """
        Set pixel or region using array indexing.
        
        Args:
            key: Index or slice for accessing pixels
            value: Pixel value(s) to set
            
        Example:
            >>> img = Image(100, 100, RGB8)
            >>> img[50, 50] = [255, 0, 0]  # Set pixel to red
        """
        self._data[key] = value
    
    def __repr__(self) -> str:
        """Detailed string representation of the image."""
        return (
            f"Image(width={self.width}, height={self.height}, "
            f"pixel_type={self._pixel_type}, dtype={self.dtype})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.width}x{self.height} {self._pixel_type} image"


__all__ = ['Image']
