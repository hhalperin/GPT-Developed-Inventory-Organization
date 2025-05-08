"""Setup configuration for the inventory analysis package."""

from setuptools import setup, find_packages

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inventory_analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing and categorizing inventory items",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/inventory_analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "rapidfuzz>=2.0.0",
        "networkx>=2.6.0",
        "requests>=2.26.0",
        "joblib>=1.0.0",
        "openpyxl>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "inventory-analysis=scripts.run_inventory:main",
        ],
    },
) 