"""
PyGIL Setup Script

Installation script for the PyGIL (Python Generic Image Library) package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    # Package metadata
    name="pygil",
    version="0.1.0",
    author="PyGIL Contributors",
    author_email="",
    description="Python Generic Image Library - A NumPy-based implementation of Boost.GIL concepts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boostorg/gil",
    
    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "demo", "demo.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discoverability
    keywords="image processing, computer vision, numpy, boost, gil, convolution, transformation",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/boostorg/gil/issues",
        "Source": "https://github.com/boostorg/gil",
        "Documentation": "https://github.com/boostorg/gil/tree/develop/python",
    },
    
    # Additional configuration
    include_package_data=True,
    zip_safe=False,
)
