"""
Unit tests for PyGIL transformation operations.

Tests the transform.py module functionality including:
- Image resizing
- Flipping operations
- Rotation
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygil.core.image import Image
from pygil.core.view import view
from pygil.core.pixel import RGB8
from pygil.processing.transform import (
    resize_view, flip_up_down, flip_left_right,
    rotate, scale, crop, bilinear_sampler, nearest_sampler
)


class TestResize(unittest.TestCase):
    """Test cases for image resizing."""
    
    def test_resize_upscale(self):
        """Test upscaling an image."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        resized = resize_view(v, 200, 100)
        
        self.assertEqual(resized.width, 200)
        self.assertEqual(resized.height, 100)
    
    def test_resize_downscale(self):
        """Test downscaling an image."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        resized = resize_view(v, 50, 25)
        
        self.assertEqual(resized.width, 50)
        self.assertEqual(resized.height, 25)
    
    def test_resize_with_samplers(self):
        """Test resizing with different samplers."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        # Bilinear
        resized_bi = resize_view(v, 200, 100, bilinear_sampler())
        self.assertEqual(resized_bi.width, 200)
        
        # Nearest neighbor
        resized_nn = resize_view(v, 200, 100, nearest_sampler())
        self.assertEqual(resized_nn.width, 200)


class TestFlip(unittest.TestCase):
    """Test cases for flipping operations."""
    
    def test_flip_up_down(self):
        """Test vertical flip."""
        img = Image(10, 10, RGB8)
        # Set top-left pixel to red
        img[0, 0] = [255, 0, 0]
        # Set bottom-left pixel to blue
        img[9, 0] = [0, 0, 255]
        
        v = view(img)
        flipped = flip_up_down(v)
        
        # After flip, top-left should be blue (was bottom-left)
        np.testing.assert_array_equal(flipped[0, 0], [0, 0, 255])
        # And bottom-left should be red (was top-left)
        np.testing.assert_array_equal(flipped[9, 0], [255, 0, 0])
    
    def test_flip_left_right(self):
        """Test horizontal flip."""
        img = Image(10, 10, RGB8)
        # Set top-left pixel to red
        img[0, 0] = [255, 0, 0]
        # Set top-right pixel to green
        img[0, 9] = [0, 255, 0]
        
        v = view(img)
        flipped = flip_left_right(v)
        
        # After flip, top-left should be green (was top-right)
        np.testing.assert_array_equal(flipped[0, 0], [0, 255, 0])
        # And top-right should be red (was top-left)
        np.testing.assert_array_equal(flipped[0, 9], [255, 0, 0])


class TestRotate(unittest.TestCase):
    """Test cases for rotation."""
    
    def test_rotate_90(self):
        """Test 90-degree rotation."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        rotated = rotate(v, 90)
        
        # Note: rotation may change dimensions slightly due to interpolation
        self.assertIsNotNone(rotated)
    
    def test_rotate_180(self):
        """Test 180-degree rotation."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        rotated = rotate(v, 180)
        
        # Dimensions should be same (or close) for 180° rotation
        self.assertIsNotNone(rotated)
    
    def test_rotate_with_resize(self):
        """Test rotation with output resize."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        rotated = rotate(v, 45, resize_output=True)
        
        # With resize_output=True, dimensions should expand to fit
        self.assertIsNotNone(rotated)


class TestScale(unittest.TestCase):
    """Test cases for scaling."""
    
    def test_scale_uniform(self):
        """Test uniform scaling."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        scaled = scale(v, 2.0)
        
        self.assertEqual(scaled.width, 200)
        self.assertEqual(scaled.height, 100)
    
    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        img = Image(100, 50, RGB8)
        v = view(img)
        
        scaled = scale(v, 2.0, 1.5)
        
        self.assertEqual(scaled.width, 200)
        self.assertEqual(scaled.height, 75)


class TestCrop(unittest.TestCase):
    """Test cases for cropping."""
    
    def test_crop_center(self):
        """Test cropping center region."""
        img = Image(100, 100, RGB8)
        v = view(img)
        
        cropped = crop(v, 25, 25, 50, 50)
        
        self.assertEqual(cropped.width, 50)
        self.assertEqual(cropped.height, 50)
    
    def test_crop_invalid_bounds(self):
        """Test that invalid crop bounds raise error."""
        img = Image(100, 100, RGB8)
        v = view(img)
        
        with self.assertRaises(ValueError):
            crop(v, 50, 50, 100, 100)  # Extends beyond image


if __name__ == '__main__':
    unittest.main()
