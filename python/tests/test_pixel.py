"""
Unit tests for PyGIL pixel types and operations.

Tests the pixel.py module functionality including:
- Pixel type creation and properties
- Channel accessors
- Pixel casting between types
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygil.core.pixel import (
    PixelType, RGB8, RGBA8, GRAY8, RGB16,
    create_pixel, pixel_cast, ChannelAccessor
)


class TestPixelType(unittest.TestCase):
    """Test cases for PixelType enum and properties."""
    
    def test_pixel_type_properties(self):
        """Test that pixel types have correct properties."""
        # RGB8 should be 8-bit, 3 channels
        self.assertEqual(RGB8.bit_depth, 8)
        self.assertEqual(RGB8.num_channels, 3)
        self.assertEqual(RGB8.color_space, 'rgb')
        self.assertEqual(RGB8.max_value, 255)
        
        # RGBA8 should be 8-bit, 4 channels
        self.assertEqual(RGBA8.bit_depth, 8)
        self.assertEqual(RGBA8.num_channels, 4)
        self.assertEqual(RGBA8.color_space, 'rgba')
        
        # RGB16 should be 16-bit, 3 channels
        self.assertEqual(RGB16.bit_depth, 16)
        self.assertEqual(RGB16.max_value, 65535)
    
    def test_pixel_type_dtype(self):
        """Test that pixel types return correct NumPy dtypes."""
        self.assertEqual(RGB8.dtype, np.uint8)
        self.assertEqual(RGB16.dtype, np.uint16)


class TestCreatePixel(unittest.TestCase):
    """Test cases for pixel creation."""
    
    def test_create_rgb8_pixel(self):
        """Test creating an RGB8 pixel."""
        pixel = create_pixel(RGB8, (255, 128, 0))
        self.assertEqual(pixel.shape, (3,))
        self.assertEqual(pixel.dtype, np.uint8)
        np.testing.assert_array_equal(pixel, [255, 128, 0])
    
    def test_create_rgba8_pixel(self):
        """Test creating an RGBA8 pixel."""
        pixel = create_pixel(RGBA8, (255, 128, 0, 200))
        self.assertEqual(pixel.shape, (4,))
        np.testing.assert_array_equal(pixel, [255, 128, 0, 200])
    
    def test_create_gray8_pixel(self):
        """Test creating a grayscale pixel."""
        pixel = create_pixel(GRAY8, (128,))
        self.assertEqual(pixel.shape, (1,))
        self.assertEqual(pixel[0], 128)
    
    def test_create_pixel_wrong_channel_count(self):
        """Test that creating pixel with wrong channel count raises error."""
        with self.assertRaises(ValueError):
            create_pixel(RGB8, (255, 128))  # Only 2 values for 3 channels


class TestPixelCast(unittest.TestCase):
    """Test cases for pixel type casting."""
    
    def test_cast_8bit_to_16bit(self):
        """Test casting from 8-bit to 16-bit."""
        pixel_8 = np.array([255, 128, 0], dtype=np.uint8)
        pixel_16 = pixel_cast(pixel_8, RGB16)
        
        self.assertEqual(pixel_16.dtype, np.uint16)
        # 255 * 257 = 65535
        self.assertEqual(pixel_16[0], 65535)
        # 128 * 257 = 32896
        self.assertAlmostEqual(pixel_16[1], 32896, delta=1)
    
    def test_cast_16bit_to_8bit(self):
        """Test casting from 16-bit to 8-bit."""
        pixel_16 = np.array([65535, 32768, 0], dtype=np.uint16)
        pixel_8 = pixel_cast(pixel_16, RGB8)
        
        self.assertEqual(pixel_8.dtype, np.uint8)
        # 65535 / 257 ≈ 255
        self.assertAlmostEqual(pixel_8[0], 255, delta=1)
        # 32768 / 257 ≈ 127
        self.assertAlmostEqual(pixel_8[1], 127, delta=1)
    
    def test_cast_same_bit_depth(self):
        """Test casting within same bit depth just copies."""
        pixel = np.array([255, 128, 0], dtype=np.uint8)
        result = pixel_cast(pixel, RGB8)
        
        np.testing.assert_array_equal(result, pixel)


class TestChannelAccessor(unittest.TestCase):
    """Test cases for channel accessors."""
    
    def test_rgb_channel_access(self):
        """Test accessing individual RGB channels."""
        pixel = np.array([255, 128, 64], dtype=np.uint8)
        
        self.assertEqual(ChannelAccessor.red(pixel), 255)
        self.assertEqual(ChannelAccessor.green(pixel), 128)
        self.assertEqual(ChannelAccessor.blue(pixel), 64)
    
    def test_rgba_channel_access(self):
        """Test accessing RGBA channels including alpha."""
        pixel = np.array([255, 128, 64, 200], dtype=np.uint8)
        
        self.assertEqual(ChannelAccessor.red(pixel), 255)
        self.assertEqual(ChannelAccessor.alpha(pixel), 200)
    
    def test_gray_channel_access(self):
        """Test accessing grayscale channel."""
        pixel = np.array([128], dtype=np.uint8)
        
        self.assertEqual(ChannelAccessor.gray(pixel), 128)


if __name__ == '__main__':
    unittest.main()
