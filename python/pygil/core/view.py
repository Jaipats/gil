"""
PyGIL Image View Module

This module defines ImageView and view factory functions,
replicating Boost.GIL's image view concepts from boost/gil/image_view.hpp.

In GIL's design philosophy:
- Views are lightweight, non-owning references to image data
- Views enable efficient operations without copying pixel data
- Multiple views can reference the same underlying image
"""

from typing import Optional, Tuple
import numpy as np
from .image import Image
from .pixel import PixelType


class ImageView:
    """
    Lightweight, non-owning reference to image data.
    
    Mirrors Boost.GIL's image_view<T> class which provides access to
    image pixels without owning the memory. This enables:
    - Zero-copy operations on image regions
    - Efficient chaining of transformations
    - Multiple views of the same image
    
    The view uses NumPy's array view mechanism, which means:
    - No data is copied when creating a view
    - Changes through the view affect the original image
    - Views become invalid if the source image is destroyed
    
    GIL equivalent: boost::gil::image_view<PixelType>
    
    Attributes:
        _data: NumPy array view (non-owning reference)
        _pixel_type: PixelType of the viewed data
        _const: Whether this is a const (read-only) view
    """
    
    def __init__(self, data: np.ndarray, pixel_type: PixelType, const: bool = False):
        """
        Create an image view from array data.
        
        Note: Typically you should use factory functions (view(), const_view())
        instead of calling this constructor directly.
        
        Args:
            data: NumPy array to view (shape: height x width x channels)
            pixel_type: Type of pixels in the data
            const: Whether this is a read-only view
            
        Algorithm:
            - Store reference to data (no copy)
            - Store pixel type and const flag
            - Mark array as read-only if const=True
        """
        self._data = data
        self._pixel_type = pixel_type
        self._const = const
        
        # Make array read-only if this is a const view
        if const:
            self._data.flags.writeable = False
    
    @property
    def width(self) -> int:
        """View width in pixels."""
        return self._data.shape[1]
    
    @property
    def height(self) -> int:
        """View height in pixels."""
        return self._data.shape[0]
    
    @property
    def num_channels(self) -> int:
        """Number of color channels."""
        return self._data.shape[2]
    
    @property
    def pixel_type(self) -> PixelType:
        """Pixel type of the viewed data."""
        return self._pixel_type
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape as (height, width, channels)."""
        return self._data.shape
    
    @property
    def is_const(self) -> bool:
        """Whether this is a read-only view."""
        return self._const
    
    def dimensions(self) -> Tuple[int, int]:
        """
        Get dimensions as (width, height) tuple.
        
        Returns:
            Tuple of (width, height)
            
        GIL equivalent: view.dimensions()
        """
        return (self.width, self.height)
    
    def to_array(self) -> np.ndarray:
        """
        Get the underlying array (returns view, not copy).
        
        Returns:
            NumPy array view of the pixel data
            
        Warning: This returns a view, not a copy. Modifications will
        affect the original image unless this is a const view.
        """
        return self._data
    
    def copy(self) -> Image:
        """
        Create a new Image by copying the viewed data.
        
        Returns:
            New Image object with copied pixel data
            
        Algorithm:
            - Create new Image with same dimensions and pixel type
            - Copy all viewed pixels to the new image
        """
        return Image.from_array(self._data.copy(), self._pixel_type)
    
    def __getitem__(self, key):
        """
        Get pixel or region using array indexing.
        
        Args:
            key: Index or slice for accessing pixels
            
        Returns:
            Pixel value(s) at the specified location
            
        Example:
            >>> view = view(img)
            >>> pixel = view[50, 50]  # Get pixel at (row=50, col=50)
        """
        return self._data[key]
    
    def __setitem__(self, key, value):
        """
        Set pixel or region using array indexing.
        
        Args:
            key: Index or slice for accessing pixels
            value: Pixel value(s) to set
            
        Raises:
            ValueError: If this is a const view
            
        Example:
            >>> v = view(img)
            >>> v[50, 50] = [255, 0, 0]  # Set pixel to red
        """
        if self._const:
            raise ValueError("Cannot modify const view")
        self._data[key] = value
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        const_str = "const " if self._const else ""
        return (
            f"ImageView({const_str}width={self.width}, height={self.height}, "
            f"pixel_type={self._pixel_type})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        const_str = "const " if self._const else ""
        return f"{const_str}{self.width}x{self.height} {self._pixel_type} view"


# View factory functions
# These mirror GIL's view creation functions

def view(img: Image) -> ImageView:
    """
    Create a mutable view of an image.
    
    Returns a lightweight reference to the image's pixel data that
    allows both reading and writing operations.
    
    Args:
        img: Source image to create view from
        
    Returns:
        Mutable ImageView of the image
        
    GIL equivalent: boost::gil::view(image)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> v[50, 50] = [255, 0, 0]  # Modifies original image
        
    Algorithm:
        - Create ImageView with reference to image's internal data
        - Allow write access (const=False)
    """
    return ImageView(img._data, img.pixel_type, const=False)


def const_view(img: Image) -> ImageView:
    """
    Create a read-only (const) view of an image.
    
    Returns a lightweight reference to the image's pixel data that
    allows reading but prevents modifications.
    
    Args:
        img: Source image to create view from
        
    Returns:
        Const (read-only) ImageView of the image
        
    GIL equivalent: boost::gil::const_view(image)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> cv = const_view(img)
        >>> pixel = cv[50, 50]  # OK: reading
        >>> cv[50, 50] = [255, 0, 0]  # Error: cannot modify const view
        
    Algorithm:
        - Create ImageView with reference to image's internal data
        - Mark as read-only (const=True)
        - Set NumPy array writeable flag to False
    """
    return ImageView(img._data, img.pixel_type, const=True)


def subimage_view(img_view: ImageView, x: int, y: int, width: int, height: int) -> ImageView:
    """
    Create a view of a rectangular sub-region of an image view.
    
    This creates a view of a rectangular portion of the source view,
    enabling operations on image regions without copying data.
    
    Args:
        img_view: Source view to extract sub-region from
        x: Starting x-coordinate (column) of the sub-region
        y: Starting y-coordinate (row) of the sub-region
        width: Width of the sub-region in pixels
        height: Height of the sub-region in pixels
        
    Returns:
        ImageView of the specified sub-region
        
    Raises:
        ValueError: If sub-region extends beyond view boundaries
        
    GIL equivalent: boost::gil::subimage_view(view, x, y, width, height)
    
    Example:
        >>> img = Image(200, 200, RGB8)
        >>> v = view(img)
        >>> sub_v = subimage_view(v, 50, 50, 100, 100)  # 100x100 region at (50,50)
        >>> sub_v.width
        100
        
    Algorithm:
        - Validate that sub-region is within view bounds
        - Create NumPy slice for the rectangular region
        - Return new ImageView referencing the sliced data
        - Preserve const-ness from source view
    """
    # Validate boundaries
    if x < 0 or y < 0:
        raise ValueError(f"Starting coordinates must be non-negative: x={x}, y={y}")
    if x + width > img_view.width or y + height > img_view.height:
        raise ValueError(
            f"Sub-region extends beyond view boundaries: "
            f"region=({x},{y},{width},{height}), "
            f"view=({img_view.width},{img_view.height})"
        )
    
    # Create slice of the data (no copy, just a view)
    # Note: NumPy uses row-major indexing: [rows, cols, channels]
    sub_data = img_view._data[y:y+height, x:x+width, :]
    
    return ImageView(sub_data, img_view.pixel_type, const=img_view.is_const)


def flipped_up_down_view(img_view: ImageView) -> ImageView:
    """
    Create a view with pixels flipped vertically (upside down).
    
    This creates a view where the image appears flipped vertically,
    without copying any data. The first row becomes the last row, etc.
    
    Args:
        img_view: Source view to flip
        
    Returns:
        ImageView with vertically flipped pixel access
        
    GIL equivalent: boost::gil::flipped_up_down_view(view)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> flipped = flipped_up_down_view(v)
        >>> # Reading from flipped[0, 0] reads from v[99, 0]
        
    Algorithm:
        - Use NumPy's slicing with negative step to reverse rows
        - data[::-1] reverses the first dimension (rows)
        - No data is copied, just view indexing is reversed
    """
    # Flip rows by reversing the first dimension
    flipped_data = img_view._data[::-1, :, :]
    
    return ImageView(flipped_data, img_view.pixel_type, const=img_view.is_const)


def flipped_left_right_view(img_view: ImageView) -> ImageView:
    """
    Create a view with pixels flipped horizontally (left-right mirror).
    
    This creates a view where the image appears flipped horizontally,
    without copying any data. The first column becomes the last column, etc.
    
    Args:
        img_view: Source view to flip
        
    Returns:
        ImageView with horizontally flipped pixel access
        
    GIL equivalent: boost::gil::flipped_left_right_view(view)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> flipped = flipped_left_right_view(v)
        >>> # Reading from flipped[0, 0] reads from v[0, 99]
        
    Algorithm:
        - Use NumPy's slicing with negative step to reverse columns
        - data[:, ::-1] reverses the second dimension (columns)
        - No data is copied, just view indexing is reversed
    """
    # Flip columns by reversing the second dimension
    flipped_data = img_view._data[:, ::-1, :]
    
    return ImageView(flipped_data, img_view.pixel_type, const=img_view.is_const)


def rotated_view(img_view: ImageView, num_quarters: int) -> ImageView:
    """
    Create a view rotated by 90-degree increments.
    
    This creates a view of the image rotated by multiples of 90 degrees,
    without copying data (uses NumPy's transpose and flip operations).
    
    Args:
        img_view: Source view to rotate
        num_quarters: Number of 90-degree rotations (1=90°, 2=180°, 3=270°)
        
    Returns:
        ImageView with rotated pixel access
        
    Note: For arbitrary angle rotations, use the rotate() function in
    the processing.transform module instead.
    
    Example:
        >>> img = Image(100, 200, RGB8)  # 100 wide, 200 tall
        >>> v = view(img)
        >>> rotated = rotated_view(v, 1)  # Rotate 90° clockwise
        >>> rotated.width  # Now 200 wide
        200
        >>> rotated.height  # Now 100 tall
        100
        
    Algorithm:
        - Normalize num_quarters to 0-3 range using modulo
        - 0: No rotation
        - 1 (90°): Transpose then flip horizontally
        - 2 (180°): Flip both vertically and horizontally
        - 3 (270°): Transpose then flip vertically
    """
    # Normalize to 0-3 range
    num_quarters = num_quarters % 4
    
    if num_quarters == 0:
        # No rotation
        return ImageView(img_view._data, img_view.pixel_type, const=img_view.is_const)
    elif num_quarters == 1:
        # 90° clockwise: transpose then flip horizontally
        rotated_data = np.transpose(img_view._data, (1, 0, 2))[:, ::-1, :]
    elif num_quarters == 2:
        # 180°: flip both dimensions
        rotated_data = img_view._data[::-1, ::-1, :]
    else:  # num_quarters == 3
        # 270° clockwise: transpose then flip vertically
        rotated_data = np.transpose(img_view._data, (1, 0, 2))[::-1, :, :]
    
    return ImageView(rotated_data, img_view.pixel_type, const=img_view.is_const)


__all__ = [
    'ImageView',
    'view',
    'const_view',
    'subimage_view',
    'flipped_up_down_view',
    'flipped_left_right_view',
    'rotated_view',
]
