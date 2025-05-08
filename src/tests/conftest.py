"""Test fixtures for inventory analysis."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from ..classification import MLCategorizer
from ..clustering import ClusterAnalyzer
from ..data import DataLoader
from ..similarity import SimilarityScorer
from ..visualize import Visualizer
from ..config import (
    CATALOG_NUMBER_COL,
    DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    CLUSTER_COL,
    MFR_CODE_COL,
    KMEANS_CLUSTERS
)

@pytest.fixture
def sample_data():
    """Create sample inventory data."""
    data = pd.DataFrame({
        CATALOG_NUMBER_COL: ["SKU1", "SKU2", "SKU3", "SKU4"],
        DESCRIPTION_COL: ["Blue Widget 10mm", "Red Widget 10mm", "Blue Widget 20mm", "Green Widget 15mm"],
        MAIN_CATEGORY_COL: ["Widgets", "Widgets", "Widgets", "Widgets"],
        SUB_CATEGORY_COL: ["Blue", "Red", "Blue", "Green"],
        MFR_CODE_COL: ["ABB", "SIEMENS", "ABB", "SCHNEIDER"]
    })
    return data

@pytest.fixture
def similarity_matrix():
    """Create sample similarity matrix."""
    return np.array([
        [1.0, 0.5, 0.8, 0.3],
        [0.5, 1.0, 0.4, 0.6],
        [0.8, 0.4, 1.0, 0.2],
        [0.3, 0.6, 0.2, 1.0]
    ])

@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()

@pytest.fixture
def ml_categorizer():
    """Create MLCategorizer instance."""
    return MLCategorizer()

@pytest.fixture
def cluster_analyzer():
    """Create ClusterAnalyzer instance."""
    return ClusterAnalyzer()

@pytest.fixture
def similarity_scorer():
    """Create SimilarityScorer instance."""
    return SimilarityScorer()

@pytest.fixture
def visualizer():
    """Create Visualizer instance."""
    return Visualizer()

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path

@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        "similarity_threshold": 0.7,
        "dbscan_eps": 0.5,
        "min_samples": 2,
        "kmeans_clusters": 3,
        "output_dir": str(Path("output")),
        "model_dir": str(Path("models"))
    }

@pytest.fixture
def mock_gpt_response():
    """Create a mock GPT API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Enhanced description"
                }
            }
        ],
        "usage": {
            "total_tokens": 100
        }
    } 