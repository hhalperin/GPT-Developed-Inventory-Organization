"""Tests for the pipeline module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline import InventoryPipeline

@pytest.fixture
def sample_data(tmp_path):
    df = pd.DataFrame({
        "CatalogNo": [f"SKU{i}" for i in range(1, 6)],
        "Description": [f"Item {i}" for i in range(1, 6)],
        "Main Category": ["Cat1", "Cat1", "Cat2", "Cat2", "Cat3"],
        "Sub-category": ["Sub1", "Sub2", "Sub3", "Sub1", "Sub2"]
    })
    file = tmp_path / "input.xlsx"
    df.to_excel(file, index=False)
    return file, df

@pytest.fixture
def pipeline(tmp_path):
    """Create an InventoryPipeline instance for testing."""
    return InventoryPipeline(
        input_file=tmp_path / "test_data.csv",
        output_dir=tmp_path / "output",
        api_key="test_key",
        similarity_threshold=0.8,
        dbscan_eps=0.5,
        min_samples=2,
        kmeans_clusters=3
    )

def test_init(pipeline, tmp_path):
    """Test pipeline initialization."""
    assert pipeline.input_file == tmp_path / "test_data.csv"
    assert pipeline.output_dir == tmp_path / "output"
    assert pipeline.api_key == "test_key"
    assert pipeline.similarity_threshold == 0.8
    assert pipeline.dbscan_eps == 0.5
    assert pipeline.min_samples == 2
    assert pipeline.kmeans_clusters == 3

@patch('src.pipeline.DataLoader')
@patch('src.pipeline.GPTClient')
@patch('src.pipeline.SimilarityScorer')
@patch('src.pipeline.MLCategorizer')
@patch('src.pipeline.ClusterAnalyzer')
@patch('src.pipeline.Visualizer')
def test_run_pipeline(mock_visualizer, mock_cluster, mock_categorizer,
                     mock_similarity, mock_gpt, mock_loader, pipeline, sample_data):
    """Test running the complete pipeline."""
    # Setup mocks
    mock_loader.return_value.data = sample_data
    mock_loader.return_value.cleaned_data = sample_data
    mock_gpt.return_value.enrich_descriptions.return_value = sample_data
    mock_categorizer.return_value.predict.return_value = ("Cat1", "Sub1", 0.9)
    mock_similarity.return_value.compute_similarity.return_value = np.array([[1.0, 0.5], [0.5, 1.0]])
    mock_cluster.return_value.get_clusters.return_value = [0, 1]
    
    # Run pipeline
    pipeline.run()
    
    # Verify all components were called
    mock_loader.return_value.load_data.assert_called_once()
    mock_loader.return_value.clean_data.assert_called_once()
    mock_gpt.return_value.enrich_descriptions.assert_called_once()
    mock_categorizer.return_value.train.assert_called_once()
    mock_similarity.return_value.compute_similarity.assert_called_once()
    mock_cluster.return_value.fit_dbscan.assert_called_once()
    mock_visualizer.return_value.create_visualizations.assert_called_once()

def test_save_results(pipeline, sample_data, tmp_path):
    """Test saving pipeline results."""
    pipeline.data = sample_data
    pipeline.cleaned_data = sample_data
    pipeline.enriched_data = sample_data
    pipeline.similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    pipeline.clusters = [0, 1]
    
    pipeline.save_results()
    
    # Check that output files exist
    assert (tmp_path / "output" / "cleaned_data.csv").exists()
    assert (tmp_path / "output" / "enriched_data.csv").exists()
    assert (tmp_path / "output" / "similarity_matrix.csv").exists()
    assert (tmp_path / "output" / "clusters.csv").exists()

def test_validate_input(pipeline, tmp_path):
    """Test input validation."""
    # Test valid input
    assert pipeline.validate_input()
    
    # Test invalid input file
    pipeline.input_file = tmp_path / "nonexistent.csv"
    assert not pipeline.validate_input()
    
    # Test invalid output directory
    pipeline.output_dir = tmp_path / "invalid" / "path"
    assert not pipeline.validate_input()

def test_pipeline_end_to_end(sample_data, tmp_path):
    input_file, df = sample_data
    output_dir = tmp_path / "output"
    pipeline = InventoryPipeline()
    summary = pipeline.run_pipeline(
        input_file=input_file,
        output_dir=output_dir,
        enrich=False,  # skip GPT for test speed
        cluster=True,
        visualize=True,
        thresholds=[0.5, 0.7]
    )
    # Check main output files
    files = [
        "processed_data.xlsx",
        "similarity_matrix.csv",
        "cluster_analysis.xlsx",
        "similarity_statistics.json",
        "distribution_statistics.json",
        "visualization_statistics.json",
        "category_change_statistics.json",
        "cluster_metrics.json",
        "category_distribution.png",
        "top_subcategory_distribution.png",
        "similarity_network.png",
        "mds_visualization.png",
        "cluster_distribution.png",
        "category_cluster_heatmap.png"
    ]
    for fname in files:
        fpath = output_dir / fname
        assert fpath.exists(), f"{fname} does not exist"
        assert fpath.stat().st_size > 0, f"{fname} is empty"
    # Check summary keys
    assert 'total_items' in summary
    assert 'unique_categories' in summary
    assert 'timing' in summary

def test_pipeline_skip_steps(sample_data, tmp_path):
    input_file, df = sample_data
    output_dir = tmp_path / "output_skip"
    pipeline = InventoryPipeline()
    summary = pipeline.run_pipeline(
        input_file=input_file,
        output_dir=output_dir,
        enrich=False,
        cluster=False,
        visualize=False
    )
    # Should still save processed data and similarity matrix
    assert (output_dir / "processed_data.xlsx").exists()
    assert (output_dir / "similarity_matrix.csv").exists()
    # Cluster/visualization files should not exist
    assert not (output_dir / "cluster_analysis.xlsx").exists()
    assert not (output_dir / "category_distribution.png").exists()

def test_pipeline_error_handling(tmp_path):
    pipeline = InventoryPipeline()
    # Nonexistent input file
    with pytest.raises(Exception):
        pipeline.run_pipeline(
            input_file=tmp_path / "nonexistent.xlsx",
            output_dir=tmp_path / "err_out"
        ) 