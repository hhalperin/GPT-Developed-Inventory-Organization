"""Tests for visualization module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.visualize import Visualizer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "CatalogNo": ["SKU1", "SKU2", "SKU3", "SKU4", "SKU5", "SKU6", "SKU7", "SKU8", "SKU9", "SKU10", "SKU11", "SKU12", "SKU13", "SKU14", "SKU15", "SKU16", "SKU17", "SKU18", "SKU19", "SKU20", "SKU21"],
        "Description": [f"Item {i}" for i in range(1, 22)],
        "Main Category": ["Cat1"]*10 + ["Cat2"]*11,
        "Sub-category": [f"Sub{i%7+1}" for i in range(21)],
        "Cluster": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    })

@pytest.fixture
def similarity_matrix():
    """Create a sample similarity matrix."""
    n = 21
    mat = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = mat[j, i] = 0.6 if abs(i-j) == 1 else 0.3
    return mat

@pytest.fixture
def visualizer():
    """Create a Visualizer instance for testing."""
    return Visualizer()

def test_init(visualizer):
    """Test Visualizer initialization."""
    assert isinstance(visualizer, object)

def test_plot_category_distribution(visualizer, sample_data, tmp_path):
    """Test category distribution plot."""
    output_file = tmp_path / "category_dist.png"
    visualizer.plot_category_distribution(sample_data, output_file)
    assert output_file.exists()

def test_plot_top_subcategory_distribution(visualizer, sample_data, tmp_path):
    """Test top subcategory distribution plot."""
    output_file = tmp_path / "top_subcategory_dist.png"
    visualizer.plot_top_subcategory_distribution(sample_data, output_file, top_n=5)
    assert output_file.exists()

def test_plot_cluster_distribution(visualizer, sample_data, tmp_path):
    """Test cluster distribution plot."""
    output_file = tmp_path / "cluster_dist.png"
    visualizer.plot_cluster_distribution(sample_data, output_file)
    assert output_file.exists()

def test_plot_category_cluster_heatmap(visualizer, sample_data, tmp_path):
    """Test category cluster heatmap plot."""
    output_file = tmp_path / "cat_cluster_heatmap.png"
    visualizer.plot_category_cluster_heatmap(sample_data, output_file)
    assert output_file.exists()

def test_plot_similarity_network(visualizer, sample_data, similarity_matrix, tmp_path):
    """Test similarity network plot."""
    output_file = tmp_path / "sim_network.png"
    visualizer.plot_similarity_network(sample_data, similarity_matrix, output_file)
    assert output_file.exists()

def test_plot_mds_visualization(visualizer, sample_data, similarity_matrix, tmp_path):
    """Test MDS visualization plot."""
    output_file = tmp_path / "mds.png"
    visualizer.plot_mds_visualization(sample_data, similarity_matrix, output_file)
    assert output_file.exists()

def test_save_visualization_results(visualizer, sample_data, similarity_matrix, tmp_path):
    """Test saving visualization results."""
    output_dir = tmp_path / "viz_out"
    visualizer.save_visualization_results(sample_data, similarity_matrix, output_dir)
    assert (output_dir / "category_distribution.png").exists()
    assert (output_dir / "top_subcategory_distribution.png").exists()
    assert (output_dir / "similarity_network.png").exists()
    assert (output_dir / "mds_visualization.png").exists()
    assert (output_dir / "cluster_distribution.png").exists()
    assert (output_dir / "category_cluster_heatmap.png").exists()
    assert (output_dir / "visualization_statistics.json").exists()

def test_plot_top_subcategory_distribution_error(visualizer, sample_data, tmp_path):
    """Test error handling for top subcategory distribution plot."""
    # Remove sub-category column to trigger error
    data = sample_data.drop(columns=["Sub-category"])
    output_file = tmp_path / "should_fail.png"
    # Should log error but not raise
    visualizer.plot_top_subcategory_distribution(data, output_file, top_n=5)
    # File should not exist
    assert not output_file.exists()

def test_get_visualization_statistics(visualizer, sample_data):
    """Test getting visualization statistics."""
    stats = visualizer.get_visualization_statistics(sample_data)
    assert isinstance(stats, dict)
    assert 'category_counts' in stats
    assert 'subcategory_distribution' in stats
    if 'Cluster' in sample_data.columns:
        assert 'cluster_distribution' in stats

def test_visualization_outputs_nonempty(visualizer, sample_data, similarity_matrix, tmp_path):
    """Test that all visualization output files are non-empty."""
    output_dir = tmp_path / "viz_out_nonempty"
    visualizer.save_visualization_results(sample_data, similarity_matrix, output_dir)
    files = [
        "category_distribution.png",
        "top_subcategory_distribution.png",
        "similarity_network.png",
        "mds_visualization.png",
        "cluster_distribution.png",
        "category_cluster_heatmap.png",
        "visualization_statistics.json"
    ]
    for fname in files:
        fpath = output_dir / fname
        assert fpath.exists(), f"{fname} does not exist"
        assert fpath.stat().st_size > 0, f"{fname} is empty" 