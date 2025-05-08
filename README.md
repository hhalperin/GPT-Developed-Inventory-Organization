# Integrating Generative AI in Supply Chains: Inventory Management Applications

A professional-grade Python package that leverages advanced AI technologies to transform inventory management through intelligent analysis, categorization, and optimization.

## Overview

This system combines the power of Large Language Models (LLMs) with traditional machine learning to provide comprehensive inventory analysis and management capabilities. It's designed to help supply chain professionals automate and enhance their inventory management processes through AI-driven insights.

## Key Features

### AI-Powered Analysis
- **GPT-based Description Enrichment**: Automatically enhances product descriptions using OpenAI's GPT models
- **Intelligent Categorization**: ML-based classification with GPT fallback for accurate product categorization
- **Smart Similarity Detection**: Advanced similarity scoring between inventory items
- **Automated Clustering**: Unsupervised learning to identify natural groupings in inventory

### Data Processing
- **Automated Data Loading**: Support for multiple input formats (Excel, CSV)
- **Intelligent Data Cleaning**: Automated data validation and cleaning
- **Smart Pipeline Management**: Resume capability from last successful step
- **Comprehensive Data Validation**: Built-in checks for data quality and completeness

### Visualization & Reporting
- **Interactive Visualizations**: Dynamic charts and graphs for inventory analysis
- **Customizable Reports**: Generate detailed reports in multiple formats
- **Real-time Statistics**: Track key metrics and performance indicators
- **Export Capabilities**: Multiple output formats for integration with other systems

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/inventory_system.git
cd inventory_system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### Command Line Interface

The package provides two main CLI tools:

1. **Standard Pipeline**:
```bash
python -m src.scripts.run_inventory input_file.xlsx --output-dir output/
```

2. **Resumable Pipeline** (recommended for large datasets):
```bash
python -m src.scripts.resume_pipeline input_file.xlsx --output-dir output/
```

#### Command Line Arguments

- **Data Processing**:
  - `--input-file`: Path to input data file (required)
  - `--output-dir`: Directory for output files (default: "output/")

- **AI Configuration**:
  - `--similarity-threshold`: Threshold for similarity scoring (default: 0.8)
  - `--dbscan-eps`: Epsilon parameter for DBSCAN clustering (default: 0.2)
  - `--min-samples`: Minimum samples for DBSCAN (default: 2)
  - `--kmeans-clusters`: Number of clusters for KMeans (default: 20)

- **Pipeline Control**:
  - `--no-enrich`: Skip description enrichment
  - `--no-cluster`: Skip clustering analysis
  - `--no-visualize`: Skip visualization generation
  - `--force`: Force rerun of all steps (resume_pipeline only)
  - `--log-level`: Set logging level (default: INFO)

### Python API

```python
from src.pipeline import InventoryPipeline
from pathlib import Path

# Initialize pipeline
pipeline = InventoryPipeline(
    similarity_threshold=0.8,
    dbscan_eps=0.2,
    min_samples=2,
    kmeans_clusters=20
)

# Run analysis
stats = pipeline.run_pipeline(
    input_file=Path("data.xlsx"),
    output_dir=Path("output"),
    enrich=True,
    cluster=True,
    visualize=True
)

# Access results
print(f"Processed {stats['total_items']} items")
print(f"Found {stats['unique_categories']} categories")
print(f"Generated {stats['clusters']} clusters")
```

## Project Structure

```
inventory_system/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Main pipeline orchestration
│   ├── data.py             # Data loading and preprocessing
│   ├── api.py              # OpenAI GPT integration
│   ├── classification.py   # ML-based categorization
│   ├── clustering.py       # Clustering algorithms
│   ├── similarity.py       # Similarity scoring
│   ├── visualize.py        # Visualization generation
│   ├── config.py           # Configuration settings
│   └── scripts/
│       ├── run_inventory.py    # Standard pipeline CLI
│       └── resume_pipeline.py  # Resumable pipeline CLI
├── tests/                  # Comprehensive test suite
├── data/                   # Data directory
│   ├── raw/               # Raw input data
│   ├── processed/         # Processed data
│   └── output/            # Generated outputs
├── requirements.txt       # Package dependencies
└── README.md             # Documentation
```

## Development

### Setting Up Development Environment

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_pipeline.py
```

### Code Quality

The project follows strict code quality standards:

- **Style**: PEP 8 compliance
- **Type Hints**: Full type annotation support
- **Documentation**: Google-style docstrings
- **Testing**: Comprehensive unit and integration tests

Check code quality:
```bash
# Style checking
flake8 src/ tests/

# Type checking
mypy src/

# Documentation
pdoc src/
```

## Performance Considerations

- **Large Datasets**: Use the resumable pipeline for datasets >10,000 items
- **API Usage**: Monitor OpenAI API usage through the built-in statistics
- **Memory Usage**: Configure batch sizes for large datasets
- **Parallel Processing**: Enable multiprocessing for faster processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License with additional academic use requirements. See the [LICENSE](LICENSE) file for details.

### Academic Use

If you use this software in academic research, you must:
1. Cite the associated research paper in any publications or presentations
2. Include the proper citation in your work (see LICENSE file for details)

### Citation

If you use this software in your research, please cite our paper:
**In second stage review at IJPR**

## Acknowledgments

- OpenAI for providing the GPT API
- Contributors and maintainers of the Python data science ecosystem
- The open-source community for their invaluable tools and libraries
