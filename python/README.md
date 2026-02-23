# PyGIL: Python Generic Image Library

A Python library that replicates [Boost.GIL](https://github.com/boostorg/gil)'s API and functionality using NumPy. PyGIL provides efficient, generic image processing with a clean, type-safe interface inspired by Boost.GIL's design philosophy.

## Features

- **Generic Image Types**: Support for RGB8, RGBA8, GRAY8, RGB16, RGBA16, GRAY16
- **View-Based Operations**: Efficient, zero-copy operations using image views
- **I/O Support**: Read and write PNG, JPEG, BMP, and TIFF formats
- **Transformations**: Resize, flip, rotate, crop with multiple interpolation methods
- **Convolution**: 1D and 2D kernels with predefined filters (Gaussian, edge detection, etc.)
- **Color Conversions**: RGB ↔ Grayscale ↔ RGBA with proper channel handling
- **NumPy Integration**: Fast array operations with familiar NumPy semantics
- **Comprehensive Documentation**: Detailed comments explaining algorithms and GIL equivalents

## Architecture

```
pygil/
├── core/           # Core image types and operations
│   ├── pixel.py           # Pixel types and channel operations
│   ├── image.py           # Image container (data owner)
│   ├── view.py            # ImageView (lightweight reference)
│   └── color_convert.py   # Color space conversions
├── io/             # Image I/O operations
│   └── image_io.py        # Read/write PNG, JPEG, BMP, TIFF
└── processing/     # Image processing operations
    ├── transform.py       # Resize, flip, rotate, crop
    └── convolve.py        # Convolution with kernels
```

## Installation

### From Source

```bash
cd python
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.21.0
- Pillow >= 9.0.0
- SciPy >= 1.7.0

## Quick Start

```python
import pygil as gil

# Read an image
img = gil.read_image("photo.jpg")
print(f"Image: {img.width}×{img.height}, {img.pixel_type}")

# Create a view (zero-copy reference)
v = gil.view(img)

# Resize with bilinear interpolation
resized = gil.resize_view(v, 400, 300)

# Convert to grayscale
gray = gil.color_convert(img, gil.GRAY8)

# Apply Gaussian blur
kernel = gil.gaussian_kernel(sigma=2.0)
blurred = gil.convolve_rows(gil.view(gray), kernel)
blurred = gil.convolve_cols(gil.view(blurred), kernel)

# Save result
gil.write_image(blurred, "output.jpg")
```

## Core Concepts

### Images and Views

Following Boost.GIL's design philosophy:

- **Image**: Owns pixel data (memory allocation)
- **ImageView**: Lightweight, non-owning reference to pixel data

```python
# Create an image (owns data)
img = gil.Image(800, 600, gil.RGB8)

# Create a mutable view
v = gil.view(img)

# Create a const (read-only) view
cv = gil.const_view(img)

# Create a sub-region view (no copy)
sub = gil.subimage_view(v, x=100, y=100, width=200, height=200)
```

### Pixel Types

```python
# 8-bit types (0-255 range)
gil.RGB8      # 8-bit RGB, 3 channels
gil.RGBA8     # 8-bit RGBA, 4 channels
gil.GRAY8     # 8-bit grayscale, 1 channel

# 16-bit types (0-65535 range)
gil.RGB16     # 16-bit RGB, 3 channels
gil.RGBA16    # 16-bit RGBA, 4 channels
gil.GRAY16    # 16-bit grayscale, 1 channel
```

### Image I/O

```python
# Read with auto-format detection
img = gil.read_image("photo.jpg")

# Read with explicit format
img = gil.read_image("photo.jpg", gil.jpeg_tag())

# Write with format tag
gil.write_image(img, "output.png", gil.png_tag())

# Write view directly
gil.write_view(gil.view(img), "output.png")
```

## Transformations

### Resize

```python
img = gil.read_image("photo.jpg")
v = gil.view(img)

# Resize with bilinear interpolation (smooth)
resized = gil.resize_view(v, 800, 600, gil.bilinear_sampler())

# Resize with nearest neighbor (fast)
resized = gil.resize_view(v, 800, 600, gil.nearest_sampler())

# Scale by factor
scaled = gil.scale(v, 2.0)  # Double size
scaled = gil.scale(v, 0.5)  # Half size
scaled = gil.scale(v, 2.0, 1.5)  # Non-uniform scaling
```

### Flip and Rotate

```python
# Flip operations (create new image)
flipped_v = gil.flip_up_down(v)
flipped_h = gil.flip_left_right(v)

# Rotate by arbitrary angle
rotated = gil.rotate(v, 45)  # 45 degrees counter-clockwise
rotated = gil.rotate(v, 45, resize_output=True)  # Expand to fit

# Zero-copy view flips (no data copy)
view_flipped = gil.flipped_up_down_view(v)
```

### Crop and Transpose

```python
# Crop rectangular region
cropped = gil.crop(v, x=50, y=50, width=200, height=200)

# Transpose (swap width and height)
transposed = gil.transpose(v)
```

## Color Conversion

```python
# RGB to Grayscale (ITU-R BT.601 formula)
gray = gil.rgb_to_gray(img)

# Grayscale to RGB
rgb = gil.gray_to_rgb(gray_img)

# RGB to RGBA (add alpha channel)
rgba = gil.rgb_to_rgba(img, alpha_value=255)  # Fully opaque

# RGBA to RGB (discard alpha)
rgb = gil.rgba_to_rgb(rgba_img)

# Generic conversion
gray = gil.color_convert(img, gil.GRAY8)
```

## Convolution

### 1D Convolution (Separable Filters)

```python
# Create Gaussian blur kernel
kernel = gil.gaussian_kernel(sigma=2.0, size=9)

# Apply to rows and columns separately (efficient for separable filters)
img = gil.read_image("photo.jpg")
v = gil.view(img)
blurred = gil.convolve_rows(v, kernel)
blurred = gil.convolve_cols(gil.view(blurred), kernel)
```

### 2D Convolution

```python
# Edge detection
edge_kernel = gil.edge_detect_kernel()
edges = gil.convolve_2d(v, edge_kernel)

# Sharpen
sharpen_kernel = gil.sharpen_kernel()
sharpened = gil.convolve_2d(v, sharpen_kernel)

# Custom 2D kernel
custom_kernel = gil.Kernel2D([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
result = gil.convolve_2d(v, custom_kernel)
```

### Predefined Kernels

```python
# Gaussian blur (smooth)
gaussian = gil.gaussian_kernel(sigma=1.5)

# Box blur (fast averaging)
box = gil.box_blur_kernel(size=5)

# Edge detection
edges = gil.edge_detect_kernel()

# Sharpening
sharpen = gil.sharpen_kernel()
```

## Demo Application

A Streamlit-based demo application is included to showcase PyGIL's capabilities:

```bash
cd demo
pip install -r requirements.txt
pip install -e ..  # Install pygil library
streamlit run app.py
```

The demo provides an interactive interface for:
- Image upload (PNG, JPEG, BMP)
- Real-time transformations (resize, flip, rotate)
- Color space conversions
- Side-by-side comparison
- Result download

## Running Tests

```bash
cd python
python -m pytest tests/
```

Or run individual test files:

```bash
python tests/test_pixel.py
python tests/test_image.py
python tests/test_transform.py
python tests/test_io.py
```

## API Design Philosophy

PyGIL closely mirrors Boost.GIL's design principles:

### 1. Separation of Image and View

```cpp
// GIL (C++)
gil::rgb8_image_t img(800, 600);
auto v = gil::view(img);
```

```python
# PyGIL (Python)
img = gil.Image(800, 600, gil.RGB8)
v = gil.view(img)
```

### 2. Generic Algorithms

```cpp
// GIL (C++)
gil::resize_view(src_view, dst_view, gil::bilinear_sampler());
```

```python
# PyGIL (Python)
resized = gil.resize_view(src_view, width, height, gil.bilinear_sampler())
```

### 3. Format Tags

```cpp
// GIL (C++)
gil::read_image("photo.jpg", img, gil::jpeg_tag());
```

```python
# PyGIL (Python)
img = gil.read_image("photo.jpg", gil.jpeg_tag())
```

## Performance Notes

- **Views are zero-copy**: Creating views doesn't copy pixel data
- **NumPy vectorization**: All operations leverage NumPy's optimized routines
- **SciPy acceleration**: Convolution and interpolation use SciPy's fast implementations
- **Memory efficiency**: Large images can be processed in-place using views

## Comparison with Boost.GIL

| Feature | Boost.GIL (C++) | PyGIL (Python) |
|---------|----------------|----------------|
| Generic types | Template-based | Enum + runtime dispatch |
| Memory management | Manual/RAII | NumPy arrays (automatic) |
| Views | Compile-time | Runtime with NumPy views |
| Type safety | Compile-time | Runtime with type hints |
| Performance | Near-optimal C++ | Vectorized NumPy (fast) |
| Ease of use | Learning curve | Python-friendly |

## Examples

### Complete Image Processing Pipeline

```python
import pygil as gil

# Read image
img = gil.read_image("input.jpg")
print(f"Loaded: {img.width}×{img.height}")

# Convert to grayscale
gray = gil.color_convert(img, gil.GRAY8)

# Resize to thumbnail
v = gil.view(gray)
thumbnail = gil.resize_view(v, 200, 150)

# Apply Gaussian blur
kernel = gil.gaussian_kernel(sigma=1.0)
blurred = gil.convolve_rows(gil.view(thumbnail), kernel)
blurred = gil.convolve_cols(gil.view(blurred), kernel)

# Save result
gil.write_image(blurred, "output.png")
```

### Working with Image Regions

```python
img = gil.read_image("photo.jpg")
v = gil.view(img)

# Extract face region (example coordinates)
face = gil.subimage_view(v, x=100, y=50, width=200, height=250)

# Process only the face region
face_img = face.copy()
processed = gil.color_convert(face_img, gil.GRAY8)

# Could composite back into original if needed
```

### Batch Processing

```python
from pathlib import Path

input_dir = Path("input_images")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    # Load
    img = gil.read_image(img_path)
    
    # Process
    v = gil.view(img)
    processed = gil.resize_view(v, 800, 600)
    
    # Save
    output_path = output_dir / f"{img_path.stem}_processed.jpg"
    gil.write_image(processed, output_path)
    print(f"Processed: {img_path.name}")
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional pixel formats (HSV, LAB, etc.)
- More image processing algorithms (morphology, filtering)
- Performance optimizations
- Additional format support
- Documentation improvements

## License

PyGIL is distributed under the Boost Software License, Version 1.0, matching the license of Boost.GIL.

## Acknowledgments

- **Boost.GIL**: The original Generic Image Library that inspired this project
- **NumPy**: For efficient array operations
- **Pillow**: For image I/O support
- **SciPy**: For image processing algorithms

## References

- [Boost.GIL Documentation](https://boostorg.github.io/gil/)
- [Boost.GIL GitHub](https://github.com/boostorg/gil)
- [GIL Design Guide](https://www.boost.org/doc/libs/release/libs/gil/doc/html/design_guide.html)

---

**Note**: This is a demonstration implementation. For production use, consider using established libraries like scikit-image, OpenCV, or Pillow directly. PyGIL is designed to showcase GIL's design concepts in Python.
