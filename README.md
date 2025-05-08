# Inventory Analysis System

A professional-grade Python package for analyzing and categorizing inventory data using machine learning and natural language processing.

## Features

- Data loading and cleaning
- GPT-based description enrichment
- Similarity scoring between items
- Machine learning-based categorization
- Clustering analysis
- Visualization generation
- Comprehensive test suite

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The package provides a command-line interface for running the complete analysis pipeline:

```bash
python -m src.scripts.run_inventory input_file.xlsx --output-dir output/
```

Optional arguments:
- `--similarity-threshold`: Threshold for similarity scoring (default: 0.8)
- `--dbscan-eps`: Epsilon parameter for DBSCAN (default: 0.2)
- `--min-samples`: Minimum samples for DBSCAN (default: 2)
- `--kmeans-clusters`: Number of clusters for KMeans (default: 20)
- `--no-enrich`: Skip description enrichment
- `--no-cluster`: Skip clustering analysis
- `--no-visualize`: Skip visualization generation
- `--log-level`: Set logging level (default: INFO)

### Python API

You can also use the package programmatically:

```python
from src.pipeline import InventoryPipeline
from pathlib import Path

# Create pipeline
pipeline = InventoryPipeline()

# Run analysis
stats = pipeline.run_pipeline(
    input_file=Path("data.xlsx"),
    output_dir=Path("output"),
    enrich=True,
    cluster=True,
    visualize=True
)

# Print statistics
print(f"Processed {stats['total_items']} items")
print(f"Found {stats['unique_categories']} categories")
```

## Project Structure

```
inventory_system/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Main pipeline class
│   ├── data.py             # Data loading and cleaning
│   ├── classification.py   # ML-based categorization
│   ├── clustering.py       # Clustering analysis
│   ├── similarity.py       # Similarity scoring
│   ├── visualize.py        # Visualization generation
│   ├── config.py           # Configuration settings
│   └── scripts/
│       └── run_inventory.py # CLI script
├── tests/                  # Test suite
├── requirements.txt        # Package dependencies
└── README.md              # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines. You can check your code with:

```bash
flake8 src/ tests/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 