"""
PyGIL Streamlit Demo Application

This demo application showcases PyGIL's image processing capabilities
through an interactive web interface built with Streamlit.

Features:
- Image upload (PNG, JPEG, BMP)
- Real-time transformations (resize, flip, rotate)
- Color space conversions (RGB, Grayscale, RGBA)
- Side-by-side comparison
- Image download

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image as PILImage
import io
import sys
from pathlib import Path

# Add parent directory to path to import pygil
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pygil library
import pygil as gil


# Configure Streamlit page
st.set_page_config(
    page_title="PyGIL Demo",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def pil_to_gil_image(pil_img: PILImage.Image) -> gil.Image:
    """
    Convert a PIL Image to PyGIL Image.
    
    This helper function bridges between PIL (used by Streamlit's file uploader)
    and PyGIL's Image class.
    
    Args:
        pil_img: PIL Image object
        
    Returns:
        PyGIL Image object
        
    Algorithm:
        - Convert PIL image to NumPy array
        - Handle different channel configurations
        - Create PyGIL Image from the array
    """
    # Convert PIL image to NumPy array
    img_array = np.array(pil_img)
    
    # Handle grayscale images (add channel dimension if needed)
    if img_array.ndim == 2:
        img_array = img_array[:, :, np.newaxis]
    
    # Detect pixel type based on array properties
    channels = img_array.shape[2] if img_array.ndim == 3 else 1
    if channels == 1:
        pixel_type = gil.GRAY8
    elif channels == 3:
        pixel_type = gil.RGB8
    elif channels == 4:
        pixel_type = gil.RGBA8
    else:
        raise ValueError(f"Unsupported channel count: {channels}")
    
    # Create PyGIL Image
    return gil.Image.from_array(img_array, pixel_type)


def gil_to_pil_image(gil_img: gil.Image) -> PILImage.Image:
    """
    Convert a PyGIL Image to PIL Image.
    
    This helper function converts PyGIL images back to PIL format
    for display in Streamlit.
    
    Args:
        gil_img: PyGIL Image object
        
    Returns:
        PIL Image object
        
    Algorithm:
        - Extract NumPy array from PyGIL Image
        - Determine PIL mode based on channels
        - Create PIL Image from array
    """
    # Get image data as NumPy array
    img_array = gil_img.to_array()
    
    # Determine PIL mode based on channels
    if gil_img.num_channels == 1:
        # Grayscale: remove channel dimension for PIL
        return PILImage.fromarray(img_array[:, :, 0], mode='L')
    elif gil_img.num_channels == 3:
        # RGB
        return PILImage.fromarray(img_array, mode='RGB')
    elif gil_img.num_channels == 4:
        # RGBA
        return PILImage.fromarray(img_array, mode='RGBA')
    else:
        raise ValueError(f"Unsupported channel count: {gil_img.num_channels}")


def main():
    """
    Main Streamlit application entry point.
    
    Creates the UI layout and handles user interactions for:
    - Image upload
    - Transformation selection
    - Parameter adjustment
    - Result display and download
    """
    # Title and description
    st.title("🖼️ PyGIL Demo Application")
    st.markdown("""
    Interactive demo of **PyGIL** (Python Generic Image Library) - 
    a NumPy-based implementation of Boost.GIL concepts.
    
    Upload an image and apply various transformations in real-time!
    """)
    
    # Sidebar for controls
    st.sidebar.header("Image Upload")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a PNG, JPEG, or BMP image file"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        # The file uploader provides a file-like object that PIL can read
        pil_img = PILImage.open(uploaded_file)
        
        # Convert to PyGIL Image
        original_img = pil_to_gil_image(pil_img)
        
        # Store in session state for persistence
        st.session_state['original_img'] = original_img
        
        # Display image information
        st.sidebar.success("✅ Image loaded successfully!")
        st.sidebar.info(f"""
        **Image Properties:**
        - Dimensions: {original_img.width} × {original_img.height}
        - Channels: {original_img.num_channels}
        - Pixel Type: {original_img.pixel_type}
        - Data Type: {original_img.dtype}
        """)
        
        # Transformation controls
        st.sidebar.header("Transformations")
        
        # Select transformation type
        transform_type = st.sidebar.selectbox(
            "Select Transformation",
            ["None", "Resize", "Flip Vertical", "Flip Horizontal", 
             "Rotate", "Color Conversion"],
            help="Choose the type of transformation to apply"
        )
        
        # Initialize transformed image as original
        transformed_img = original_img
        
        # Apply selected transformation with parameters
        if transform_type == "Resize":
            st.sidebar.subheader("Resize Parameters")
            
            # Maintain aspect ratio option
            maintain_aspect = st.sidebar.checkbox(
                "Maintain Aspect Ratio", 
                value=True,
                help="Keep the original image proportions when resizing"
            )
            
            if maintain_aspect:
                # Single scale slider
                scale_factor = st.sidebar.slider(
                    "Scale Factor",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help="Scale factor for both dimensions (1.0 = original size)"
                )
                new_width = int(original_img.width * scale_factor)
                new_height = int(original_img.height * scale_factor)
            else:
                # Separate width and height sliders
                new_width = st.sidebar.slider(
                    "Width",
                    min_value=10,
                    max_value=original_img.width * 2,
                    value=original_img.width,
                    help="Target width in pixels"
                )
                new_height = st.sidebar.slider(
                    "Height",
                    min_value=10,
                    max_value=original_img.height * 2,
                    value=original_img.height,
                    help="Target height in pixels"
                )
            
            # Interpolation method
            interp_method = st.sidebar.radio(
                "Interpolation",
                ["Bilinear", "Nearest Neighbor"],
                help="Bilinear provides smoother results; Nearest is faster"
            )
            
            # Apply resize transformation
            # Create a view of the original image
            view = gil.view(original_img)
            
            # Select sampler based on user choice
            sampler = gil.bilinear_sampler() if interp_method == "Bilinear" else gil.nearest_sampler()
            
            # Perform resize using PyGIL's resize_view function
            transformed_img = gil.resize_view(view, new_width, new_height, sampler)
            
        elif transform_type == "Flip Vertical":
            # Apply vertical flip transformation
            # This flips the image upside down
            view = gil.view(original_img)
            transformed_img = gil.flip_up_down(view)
            
        elif transform_type == "Flip Horizontal":
            # Apply horizontal flip transformation
            # This creates a mirror image (left-right flip)
            view = gil.view(original_img)
            transformed_img = gil.flip_left_right(view)
            
        elif transform_type == "Rotate":
            st.sidebar.subheader("Rotation Parameters")
            
            # Rotation angle slider
            angle = st.sidebar.slider(
                "Rotation Angle (degrees)",
                min_value=0,
                max_value=360,
                value=0,
                step=1,
                help="Counter-clockwise rotation angle in degrees"
            )
            
            # Option to resize output to fit
            resize_output = st.sidebar.checkbox(
                "Expand to Fit",
                value=False,
                help="Expand output dimensions to fit entire rotated image"
            )
            
            # Apply rotation transformation
            view = gil.view(original_img)
            transformed_img = gil.rotate(view, angle, resize_output=resize_output)
            
        elif transform_type == "Color Conversion":
            st.sidebar.subheader("Color Conversion")
            
            # Get current color space
            current_space = original_img.pixel_type.color_space
            
            # Select target color space
            target_space = st.sidebar.selectbox(
                "Convert To",
                ["RGB", "Grayscale", "RGBA"],
                help="Target color space for conversion"
            )
            
            # Map selection to pixel type
            pixel_type_map = {
                "RGB": gil.RGB8,
                "Grayscale": gil.GRAY8,
                "RGBA": gil.RGBA8
            }
            target_type = pixel_type_map[target_space]
            
            # Apply color conversion if different from current
            if target_type != original_img.pixel_type:
                # Alpha value for RGBA conversion
                if target_space == "RGBA":
                    alpha_value = st.sidebar.slider(
                        "Alpha Value",
                        min_value=0,
                        max_value=255,
                        value=255,
                        help="Alpha channel value (255 = fully opaque)"
                    )
                    transformed_img = gil.color_convert(original_img, target_type, alpha_value)
                else:
                    transformed_img = gil.color_convert(original_img, target_type)
            else:
                st.sidebar.info(f"Image is already in {target_space} format")
        
        # Display original and transformed images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            # Convert PyGIL image to PIL for display
            original_pil = gil_to_pil_image(original_img)
            st.image(original_pil, use_container_width=True)
            st.caption(f"{original_img.width}×{original_img.height} | {original_img.pixel_type}")
        
        with col2:
            st.subheader("Transformed Image")
            # Convert transformed PyGIL image to PIL for display
            transformed_pil = gil_to_pil_image(transformed_img)
            st.image(transformed_pil, use_container_width=True)
            st.caption(f"{transformed_img.width}×{transformed_img.height} | {transformed_img.pixel_type}")
        
        # Download button for transformed image
        st.sidebar.header("Download Result")
        
        # Convert transformed image to bytes for download
        # Save to BytesIO buffer
        buffer = io.BytesIO()
        transformed_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Create download button
        st.sidebar.download_button(
            label="📥 Download Transformed Image",
            data=buffer,
            file_name="transformed_image.png",
            mime="image/png",
            help="Download the transformed image as PNG"
        )
        
    else:
        # No image uploaded yet
        st.info("👆 Please upload an image using the sidebar to get started!")
        
        # Show example usage
        st.markdown("""
        ### Features:
        
        1. **Resize** - Change image dimensions with bilinear or nearest-neighbor interpolation
        2. **Flip** - Flip vertically or horizontally
        3. **Rotate** - Rotate by any angle with optional expansion to fit
        4. **Color Conversion** - Convert between RGB, Grayscale, and RGBA
        
        ### PyGIL Library Overview:
        
        PyGIL is a Python implementation of Boost.GIL (Generic Image Library) concepts,
        providing:
        - Generic image types (RGB8, RGBA8, GRAY8, etc.)
        - Efficient NumPy-backed storage
        - View-based operations (zero-copy where possible)
        - Clean, type-safe API
        
        The transformations you see in this demo are powered by PyGIL's processing
        modules, which mirror Boost.GIL's functionality.
        """)


if __name__ == "__main__":
    main()
