"""
Setup script for VariantProject
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="variantproject",
    version="2.0.0",
    author="Hosuay",
    author_email="",
    description="AI-Assisted Molecular Exploration Tool for Drug Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hosuay/VariantProject",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "full": [
            "py3Dmol>=1.8.0",
            "selfies>=2.1.0",
            "tqdm>=4.62.0",
            "ipython>=7.30.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "variantproject=VariantProject_v2:main",
        ],
    },
)
