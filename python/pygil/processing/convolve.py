"""
PyGIL Convolution Module

This module provides convolution operations with kernels,
replicating Boost.GIL's convolution from boost/gil/image_processing/convolve.hpp.

Supports:
- 1D and 2D kernels
- Row and column convolution
- Predefined kernels (Gaussian, box blur)
"""

from typing import Union, Optional
import numpy as np
from scipy import ndimage

# Import from core module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.image import Image
from core.view import ImageView


class Kernel1D:
    """
    One-dimensional convolution kernel.
    
    Represents a 1D kernel that can be applied to rows or columns of an image.
    Mirrors GIL's kernel_1d class from boost/gil/image_processing/kernel.hpp.
    
    GIL equivalent: boost::gil::kernel_1d<T>
    
    Attributes:
        values: NumPy array of kernel weights
        center: Index of the kernel center (anchor point)
        size: Number of elements in the kernel
    """
    
    def __init__(self, values: Union[list, np.ndarray], center: Optional[int] = None):
        """
        Create a 1D convolution kernel.
        
        Args:
            values: Kernel weights as list or array
            center: Index of center element (default: middle element)
            
        Raises:
            ValueError: If kernel is empty or center is out of bounds
            
        Example:
            >>> # Gaussian blur kernel
            >>> kernel = Kernel1D([0.25, 0.5, 0.25], center=1)
            >>> 
            >>> # Edge detection kernel
            >>> kernel = Kernel1D([-1, 0, 1], center=1)
            
        Algorithm:
            - Convert values to NumPy array
            - Validate that kernel is not empty
            - Set center to middle if not specified
            - Validate center is within bounds
        """
        self.values = np.array(values, dtype=np.float32)
        self.size = len(self.values)
        
        if self.size == 0:
            raise ValueError("Kernel cannot be empty")
        
        # Default center to middle element
        if center is None:
            self.center = self.size // 2
        else:
            self.center = center
        
        # Validate center
        if not (0 <= self.center < self.size):
            raise ValueError(f"Center {self.center} is out of bounds for kernel size {self.size}")
    
    @property
    def sum(self) -> float:
        """
        Sum of all kernel weights.
        
        Useful for normalization checks. For many kernels (e.g., blur),
        the sum should be 1.0 to preserve image brightness.
        """
        return np.sum(self.values)
    
    def normalize(self) -> 'Kernel1D':
        """
        Create a normalized copy of this kernel.
        
        Returns a new kernel with weights scaled so they sum to 1.0.
        This preserves image brightness after convolution.
        
        Returns:
            New normalized Kernel1D
            
        Algorithm:
            - Calculate sum of weights
            - Divide all weights by sum
            - Create new Kernel1D with normalized weights
        """
        kernel_sum = self.sum
        if kernel_sum == 0:
            raise ValueError("Cannot normalize kernel with sum = 0")
        
        normalized_values = self.values / kernel_sum
        return Kernel1D(normalized_values, self.center)
    
    def __repr__(self):
        return f"Kernel1D(size={self.size}, center={self.center}, sum={self.sum:.4f})"
    
    def __str__(self):
        return f"Kernel1D{self.values.tolist()}"


class Kernel2D:
    """
    Two-dimensional convolution kernel.
    
    Represents a 2D kernel for spatial filtering operations.
    Common for edge detection, sharpening, and other spatial filters.
    
    GIL equivalent: boost::gil::kernel_2d<T>
    
    Attributes:
        values: 2D NumPy array of kernel weights
        center_y: Row index of kernel center
        center_x: Column index of kernel center
    """
    
    def __init__(self, values: Union[list, np.ndarray], 
                 center_y: Optional[int] = None,
                 center_x: Optional[int] = None):
        """
        Create a 2D convolution kernel.
        
        Args:
            values: 2D array of kernel weights
            center_y: Row index of center (default: middle row)
            center_x: Column index of center (default: middle column)
            
        Raises:
            ValueError: If kernel dimensions are invalid or center is out of bounds
            
        Example:
            >>> # 3x3 edge detection kernel (Sobel)
            >>> sobel_x = Kernel2D([
            ...     [-1, 0, 1],
            ...     [-2, 0, 2],
            ...     [-1, 0, 1]
            ... ])
            
        Algorithm:
            - Convert values to 2D NumPy array
            - Validate dimensions
            - Set center to middle if not specified
            - Validate center is within bounds
        """
        self.values = np.array(values, dtype=np.float32)
        
        if self.values.ndim != 2:
            raise ValueError("Kernel must be 2-dimensional")
        
        self.height, self.width = self.values.shape
        
        if self.height == 0 or self.width == 0:
            raise ValueError("Kernel dimensions must be positive")
        
        # Default center to middle
        if center_y is None:
            self.center_y = self.height // 2
        else:
            self.center_y = center_y
        
        if center_x is None:
            self.center_x = self.width // 2
        else:
            self.center_x = center_x
        
        # Validate center
        if not (0 <= self.center_y < self.height):
            raise ValueError(f"Center Y {self.center_y} is out of bounds")
        if not (0 <= self.center_x < self.width):
            raise ValueError(f"Center X {self.center_x} is out of bounds")
    
    @property
    def sum(self) -> float:
        """Sum of all kernel weights."""
        return np.sum(self.values)
    
    def normalize(self) -> 'Kernel2D':
        """
        Create a normalized copy of this kernel.
        
        Returns a new kernel with weights scaled so they sum to 1.0.
        
        Returns:
            New normalized Kernel2D
        """
        kernel_sum = self.sum
        if kernel_sum == 0:
            raise ValueError("Cannot normalize kernel with sum = 0")
        
        normalized_values = self.values / kernel_sum
        return Kernel2D(normalized_values, self.center_y, self.center_x)
    
    def __repr__(self):
        return f"Kernel2D(shape=({self.height}, {self.width}), sum={self.sum:.4f})"


def convolve_rows(img_view: ImageView, kernel: Kernel1D) -> Image:
    """
    Apply 1D convolution to each row of the image.
    
    Convolves the kernel horizontally across each row, processing each
    color channel independently. This is efficient for separable filters.
    
    Args:
        img_view: Source image view
        kernel: 1D kernel to apply to rows
        
    Returns:
        New Image with row convolution applied
        
    GIL equivalent: boost::gil::convolve_rows(src, kernel, dst)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> kernel = gaussian_kernel(sigma=1.0, size=9)
        >>> blurred = convolve_rows(v, kernel)
        
    Algorithm:
        - Process each color channel independently
        - Apply 1D convolution along axis 1 (columns/width direction)
        - Use scipy.ndimage.convolve1d for efficient computation
        - Handle boundary conditions with 'reflect' mode
        - Clamp output to valid pixel range
    """
    src_data = img_view.to_array()
    height, width, channels = src_data.shape
    
    # Create output array
    dst_data = np.zeros_like(src_data, dtype=np.float32)
    
    # Process each channel independently
    for c in range(channels):
        # Apply 1D convolution along rows (axis 1)
        dst_data[:, :, c] = ndimage.convolve1d(
            src_data[:, :, c].astype(np.float32),
            kernel.values,
            axis=1,  # Convolve along columns (horizontal)
            mode='reflect'  # Boundary handling
        )
    
    # Clamp to valid range and convert to original dtype
    pixel_type = img_view.pixel_type
    dst_data = np.clip(dst_data, 0, pixel_type.max_value)
    dst_data = dst_data.astype(pixel_type.dtype)
    
    return Image.from_array(dst_data, pixel_type)


def convolve_cols(img_view: ImageView, kernel: Kernel1D) -> Image:
    """
    Apply 1D convolution to each column of the image.
    
    Convolves the kernel vertically down each column, processing each
    color channel independently. This is efficient for separable filters.
    
    Args:
        img_view: Source image view
        kernel: 1D kernel to apply to columns
        
    Returns:
        New Image with column convolution applied
        
    GIL equivalent: boost::gil::convolve_cols(src, kernel, dst)
    
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> kernel = gaussian_kernel(sigma=1.0, size=9)
        >>> blurred = convolve_cols(v, kernel)
        
    Algorithm:
        - Process each color channel independently
        - Apply 1D convolution along axis 0 (rows/height direction)
        - Use scipy.ndimage.convolve1d for efficient computation
        - Handle boundary conditions with 'reflect' mode
        - Clamp output to valid pixel range
    """
    src_data = img_view.to_array()
    height, width, channels = src_data.shape
    
    # Create output array
    dst_data = np.zeros_like(src_data, dtype=np.float32)
    
    # Process each channel independently
    for c in range(channels):
        # Apply 1D convolution along columns (axis 0)
        dst_data[:, :, c] = ndimage.convolve1d(
            src_data[:, :, c].astype(np.float32),
            kernel.values,
            axis=0,  # Convolve along rows (vertical)
            mode='reflect'  # Boundary handling
        )
    
    # Clamp to valid range and convert to original dtype
    pixel_type = img_view.pixel_type
    dst_data = np.clip(dst_data, 0, pixel_type.max_value)
    dst_data = dst_data.astype(pixel_type.dtype)
    
    return Image.from_array(dst_data, pixel_type)


def convolve_2d(img_view: ImageView, kernel: Kernel2D) -> Image:
    """
    Apply 2D convolution to the image.
    
    Applies a 2D spatial filter to the image. Processes each color
    channel independently.
    
    Args:
        img_view: Source image view
        kernel: 2D kernel to apply
        
    Returns:
        New Image with 2D convolution applied
        
    Example:
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> # Create edge detection kernel
        >>> sobel = Kernel2D([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        >>> edges = convolve_2d(v, sobel)
        
    Algorithm:
        - Process each color channel independently
        - Apply 2D convolution using scipy.ndimage.convolve
        - Handle boundary conditions with 'reflect' mode
        - Clamp output to valid pixel range
    """
    src_data = img_view.to_array()
    height, width, channels = src_data.shape
    
    # Create output array
    dst_data = np.zeros_like(src_data, dtype=np.float32)
    
    # Process each channel independently
    for c in range(channels):
        # Apply 2D convolution
        dst_data[:, :, c] = ndimage.convolve(
            src_data[:, :, c].astype(np.float32),
            kernel.values,
            mode='reflect'  # Boundary handling
        )
    
    # Clamp to valid range and convert to original dtype
    pixel_type = img_view.pixel_type
    dst_data = np.clip(dst_data, 0, pixel_type.max_value)
    dst_data = dst_data.astype(pixel_type.dtype)
    
    return Image.from_array(dst_data, pixel_type)


# Predefined kernel generators

def gaussian_kernel(sigma: float = 1.0, size: Optional[int] = None) -> Kernel1D:
    """
    Create a 1D Gaussian blur kernel.
    
    Generates a Gaussian kernel for smooth blurring. The Gaussian function
    provides the best balance between spatial and frequency localization.
    
    Args:
        sigma: Standard deviation of the Gaussian (controls blur amount)
        size: Kernel size (default: calculated from sigma)
        
    Returns:
        Normalized Kernel1D with Gaussian weights
        
    Example:
        >>> kernel = gaussian_kernel(sigma=1.0)
        >>> # Apply to image
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> blurred = convolve_rows(v, kernel)
        >>> blurred = convolve_cols(view(blurred), kernel)
        
    Algorithm:
        - If size not specified, calculate as 2 * ceil(3*sigma) + 1
          (covers 99.7% of Gaussian distribution)
        - Generate x coordinates centered at 0
        - Compute Gaussian: exp(-x^2 / (2*sigma^2))
        - Normalize so weights sum to 1.0
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    # Calculate size if not provided (cover 99.7% of distribution)
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    
    # Ensure size is odd for symmetric kernel
    if size % 2 == 0:
        size += 1
    
    # Generate kernel values
    center = size // 2
    x = np.arange(size) - center
    
    # Gaussian function: exp(-x^2 / (2*sigma^2))
    values = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize to sum to 1.0
    values = values / np.sum(values)
    
    return Kernel1D(values, center)


def box_blur_kernel(size: int) -> Kernel1D:
    """
    Create a box blur (averaging) kernel.
    
    Simple averaging kernel where all weights are equal.
    Faster than Gaussian but less smooth results.
    
    Args:
        size: Kernel size (should be odd for symmetry)
        
    Returns:
        Normalized Kernel1D with uniform weights
        
    Example:
        >>> kernel = box_blur_kernel(5)  # Average 5 pixels
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> blurred = convolve_rows(v, kernel)
        
    Algorithm:
        - Create array of ones with specified size
        - Normalize so weights sum to 1.0
        - Each weight = 1/size
    """
    if size <= 0:
        raise ValueError("Size must be positive")
    
    # Ensure size is odd
    if size % 2 == 0:
        size += 1
    
    # All weights equal
    values = np.ones(size) / size
    
    return Kernel1D(values, size // 2)


def sharpen_kernel() -> Kernel2D:
    """
    Create a sharpening kernel.
    
    Enhances edges and fine details in the image.
    
    Returns:
        Kernel2D for sharpening
        
    Example:
        >>> kernel = sharpen_kernel()
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> sharpened = convolve_2d(v, kernel)
        
    Algorithm:
        - Use standard sharpening matrix:
          [ 0, -1,  0]
          [-1,  5, -1]
          [ 0, -1,  0]
        - Center weight > sum of neighbors amplifies details
    """
    values = [
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ]
    return Kernel2D(values)


def edge_detect_kernel() -> Kernel2D:
    """
    Create a simple edge detection kernel.
    
    Detects edges in all directions.
    
    Returns:
        Kernel2D for edge detection
        
    Example:
        >>> kernel = edge_detect_kernel()
        >>> img = Image(100, 100, RGB8)
        >>> v = view(img)
        >>> edges = convolve_2d(v, kernel)
    """
    values = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]
    return Kernel2D(values)


__all__ = [
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
