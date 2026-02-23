"""
Unit tests for PyGIL Image class.

Tests the image.py module functionality including:
- Image creation and initialization
- Image properties and dimensions
- Image copying and data access
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygil.core.image import Image
from pygil.core.pixel import RGB8, RGBA8, GRAY8, RGB16


class TestImageCreation(unittest.TestCase):
    """Test cases for Image creation."""
    
    def test_create_rgb8_image(self):
        """Test creating an RGB8 image."""
        img = Image(100, 50, RGB8)
        
        self.assertEqual(img.width, 100)
        self.assertEqual(img.height, 50)
        self.assertEqual(img.num_channels, 3)
        self.assertEqual(img.pixel_type, RGB8)
        self.assertEqual(img.dtype, np.uint8)
    
    def test_create_rgba8_image(self):
        """Test creating an RGBA8 image."""
        img = Image(200, 150, RGBA8)
        
        self.assertEqual(img.num_channels, 4)
        self.assertEqual(img.size, 200 * 150)
    
    def test_create_gray8_image(self):
        """Test creating a grayscale image."""
        img = Image(100, 100, GRAY8)
        
        self.assertEqual(img.num_channels, 1)
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            Image(0, 100, RGB8)
        
        with self.assertRaises(ValueError):
            Image(100, -50, RGB8)


class TestImageFromArray(unittest.TestCase):
    """Test cases for creating images from NumPy arrays."""
    
    def test_from_array_rgb8(self):
        """Test creating image from RGB8 array."""
        array = np.zeros((50, 100, 3), dtype=np.uint8)
        img = Image.from_array(array, RGB8)
        
        self.assertEqual(img.width, 100)
        self.assertEqual(img.height, 50)
        self.assertEqual(img.pixel_type, RGB8)
    
    def test_from_array_auto_detect(self):
        """Test auto-detection of pixel type from array."""
        # RGB8 array
        rgb_array = np.zeros((50, 100, 3), dtype=np.uint8)
        img = Image.from_array(rgb_array)
        self.assertEqual(img.pixel_type, RGB8)
        
        # RGBA8 array
        rgba_array = np.zeros((50, 100, 4), dtype=np.uint8)
        img = Image.from_array(rgba_array)
        self.assertEqual(img.pixel_type, RGBA8)
        
        # Gray8 array
        gray_array = np.zeros((50, 100, 1), dtype=np.uint8)
        img = Image.from_array(gray_array)
        self.assertEqual(img.pixel_type, GRAY8)


class TestImageOperations(unittest.TestCase):
    """Test cases for Image operations."""
    
    def test_image_copy(self):
        """Test copying an image."""
        img1 = Image(100, 50, RGB8)
        img1.fill((255, 0, 0))  # Fill with red
        
        img2 = img1.copy()
        
        # Should have same properties
        self.assertEqual(img1.width, img2.width)
        self.assertEqual(img1.height, img2.height)
        
        # Should be different objects
        self.assertIsNot(img1, img2)
        
        # Modifying one shouldn't affect the other
        img2.fill((0, 255, 0))  # Fill with green
        np.testing.assert_array_equal(img1[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(img2[0, 0], [0, 255, 0])
    
    def test_image_fill(self):
        """Test filling image with constant value."""
        img = Image(100, 50, RGB8)
        
        # Fill with red
        img.fill((255, 0, 0))
        
        # Check a few pixels
        np.testing.assert_array_equal(img[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(img[25, 50], [255, 0, 0])
    
    def test_image_dimensions(self):
        """Test dimensions() method."""
        img = Image(100, 50, RGB8)
        dims = img.dimensions()
        
        self.assertEqual(dims, (100, 50))  # (width, height)
    
    def test_image_shape(self):
        """Test shape property."""
        img = Image(100, 50, RGB8)
        shape = img.shape
        
        self.assertEqual(shape, (50, 100, 3))  # (height, width, channels)
    
    def test_image_indexing(self):
        """Test pixel access via indexing."""
        img = Image(100, 50, RGB8)
        
        # Set a pixel
        img[10, 20] = [255, 128, 64]
        
        # Read it back
        pixel = img[10, 20]
        np.testing.assert_array_equal(pixel, [255, 128, 64])
    
    def test_image_recreate(self):
        """Test recreating image with new dimensions."""
        img = Image(100, 50, RGB8)
        original_id = id(img._data)
        
        # Recreate with new dimensions
        img.recreate(200, 100)
        
        self.assertEqual(img.width, 200)
        self.assertEqual(img.height, 100)
        # Should have new data array
        self.assertNotEqual(id(img._data), original_id)


if __name__ == '__main__':
    unittest.main()
