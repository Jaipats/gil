"""
Unit tests for PyGIL I/O operations.

Tests the image_io.py module functionality including:
- Image reading and writing
- Format detection
- Format tags
"""

import unittest
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygil.core.image import Image
from pygil.core.pixel import RGB8
from pygil.io.image_io import (
    png_tag, jpeg_tag, bmp_tag,
    write_image, read_image
)


class TestFormatTags(unittest.TestCase):
    """Test cases for format tags."""
    
    def test_png_tag(self):
        """Test PNG tag creation."""
        tag = png_tag()
        self.assertEqual(tag.format_name, 'PNG')
    
    def test_jpeg_tag(self):
        """Test JPEG tag creation."""
        tag = jpeg_tag()
        self.assertEqual(tag.format_name, 'JPEG')
    
    def test_bmp_tag(self):
        """Test BMP tag creation."""
        tag = bmp_tag()
        self.assertEqual(tag.format_name, 'BMP')


class TestImageIO(unittest.TestCase):
    """Test cases for image I/O operations."""
    
    def test_write_and_read_png(self):
        """Test writing and reading PNG images."""
        # Create a test image
        img = Image(50, 30, RGB8)
        img.fill((255, 128, 64))
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Write image
            write_image(img, temp_path, png_tag())
            
            # Read it back
            loaded_img = read_image(temp_path, png_tag())
            
            # Check properties
            self.assertEqual(loaded_img.width, img.width)
            self.assertEqual(loaded_img.height, img.height)
            self.assertEqual(loaded_img.num_channels, img.num_channels)
        
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_write_and_read_jpeg(self):
        """Test writing and reading JPEG images."""
        # Create a test image
        img = Image(50, 30, RGB8)
        img.fill((200, 100, 50))
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Write image
            write_image(img, temp_path, jpeg_tag(), quality=95)
            
            # Read it back
            loaded_img = read_image(temp_path, jpeg_tag())
            
            # Check properties (JPEG may have slight variations due to compression)
            self.assertEqual(loaded_img.width, img.width)
            self.assertEqual(loaded_img.height, img.height)
        
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_auto_format_detection(self):
        """Test automatic format detection from file extension."""
        img = Image(50, 30, RGB8)
        img.fill((150, 150, 150))
        
        # Write to temporary file (auto-detect format from extension)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Write without explicit tag
            write_image(img, temp_path)
            
            # Read without explicit tag
            loaded_img = read_image(temp_path)
            
            # Check it worked
            self.assertEqual(loaded_img.width, img.width)
            self.assertEqual(loaded_img.height, img.height)
        
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


if __name__ == '__main__':
    unittest.main()
