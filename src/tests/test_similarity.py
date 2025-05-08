"""Tests for similarity scoring module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.similarity import SimilarityScorer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "CatalogNo": ["SKU1", "SKU2", "SKU3"],
        "Description": [
            "Blue Widget 10mm",
            "Red Widget 10mm",
            "Blue Widget 20mm"
        ],
        "Main Category": ["Widgets", "Widgets", "Widgets"],
        "Sub-category": ["Blue", "Red", "Blue"]
    })

@pytest.fixture
def similarity_scorer():
    """Create a SimilarityScorer instance for testing."""
    return SimilarityScorer(threshold=0.5)

def test_init(similarity_scorer):
    """Test SimilarityScorer initialization."""
    assert similarity_scorer.threshold == 0.5
    assert similarity_scorer.sku_similarity_matrix is None
    assert similarity_scorer.desc_similarity_matrix is None

def test_compute_similarity(similarity_scorer, sample_data):
    """Test similarity computation."""
    similarity_matrix = similarity_scorer.compute_similarity(sample_data)
    
    assert isinstance(similarity_matrix, np.ndarray)
    assert similarity_matrix.shape == (len(sample_data), len(sample_data))
    assert np.all((similarity_matrix >= 0) & (similarity_matrix <= 1))
    assert np.all(np.diag(similarity_matrix) == 1.0)

def test_compute_similarity_matrix(similarity_scorer, sample_data):
    """Test computation of similarity matrix."""
    similarity_matrix = similarity_scorer.compute_similarity_matrix(sample_data)
    assert isinstance(similarity_matrix, np.ndarray)
    assert similarity_matrix.shape == (len(sample_data), len(sample_data))
    assert np.all(similarity_matrix >= 0) and np.all(similarity_matrix <= 1)
    assert np.allclose(similarity_matrix, similarity_matrix.T)  # Symmetric
    assert np.allclose(np.diag(similarity_matrix), 1)  # Self-similarity

def test_find_similar_items(similarity_scorer, sample_data):
    """Test finding similar items."""
    similar_pairs = similarity_scorer.find_similar_items(sample_data)
    assert isinstance(similar_pairs, list)
    for item1, item2, similarity in similar_pairs:
        assert isinstance(item1, str)
        assert isinstance(item2, str)
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        assert similarity >= similarity_scorer.threshold

def test_get_similarity_score(similarity_scorer, sample_data):
    """Test getting similarity score between two items."""
    similarity_scorer.compute_similarity(sample_data)
    score = similarity_scorer.get_similarity_score("SKU1", "SKU3")
    
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_save_similarity_matrix(similarity_scorer, sample_data, tmp_path):
    """Test saving similarity matrix."""
    similarity_scorer.compute_similarity(sample_data)
    output_file = tmp_path / "similarity_matrix.csv"
    similarity_scorer.save_similarity_matrix(output_file)
    sku_file = tmp_path / "similarity_matrix_sku.csv"
    desc_file = tmp_path / "similarity_matrix_desc.csv"
    assert sku_file.exists()
    assert desc_file.exists()
    saved_sku = pd.read_csv(sku_file)
    saved_desc = pd.read_csv(desc_file)
    assert saved_sku.shape == similarity_scorer.sku_similarity_matrix.shape
    assert saved_desc.shape == similarity_scorer.desc_similarity_matrix.shape
    assert np.allclose(saved_sku.values, similarity_scorer.sku_similarity_matrix)
    assert np.allclose(saved_desc.values, similarity_scorer.desc_similarity_matrix)

def test_get_similarity_statistics(similarity_scorer, sample_data):
    """Test getting similarity statistics."""
    stats = similarity_scorer.get_similarity_statistics(sample_data)
    assert isinstance(stats, dict)
    assert 'total_items' in stats
    assert 'similar_pairs' in stats
    assert 'average_sku_similarity' in stats
    assert 'average_desc_similarity' in stats
    assert 'max_sku_similarity' in stats
    assert 'max_desc_similarity' in stats
    assert 'similar_items_ratio' in stats
    assert 'connected_components' in stats
    assert 'largest_component_size' in stats
    assert 'per_item_stats' in stats

def test_build_similarity_graph(similarity_scorer, sample_data):
    """Test building similarity graph."""
    graph = similarity_scorer.build_similarity_graph(sample_data)
    assert len(graph.nodes) == len(sample_data)
    for node in graph.nodes:
        assert isinstance(node, str)
        assert 'description' in graph.nodes[node]
        assert 'category' in graph.nodes[node]
        assert 'subcategory' in graph.nodes[node]

def test_save_similarity_results(similarity_scorer, sample_data, tmp_path):
    """Test saving similarity results."""
    # Test CSV output
    csv_file = tmp_path / "similarity_results.csv"
    similarity_scorer.save_similarity_results(sample_data, csv_file)
    assert csv_file.exists()
    
    # Test Excel output
    xlsx_file = tmp_path / "similarity_results.xlsx"
    similarity_scorer.save_similarity_results(sample_data, xlsx_file)
    assert xlsx_file.exists()
    
    # Test invalid format
    with pytest.raises(ValueError):
        invalid_file = tmp_path / "results.txt"
        similarity_scorer.save_similarity_results(sample_data, invalid_file)

def test_compute_similarity_statistics(similarity_scorer, sample_data):
    """Test detailed similarity statistics for multiple thresholds."""
    thresholds = [0.3, 0.5, 0.7]
    stats = similarity_scorer.compute_similarity_statistics(sample_data, thresholds=thresholds)
    assert isinstance(stats, dict)
    for thresh in thresholds:
        assert thresh in stats
        s = stats[thresh]
        assert 'total_items' in s
        assert 'similar_pairs' in s
        assert 'average_sku_similarity' in s
        assert 'average_desc_similarity' in s
        assert 'max_sku_similarity' in s
        assert 'max_desc_similarity' in s
        assert 'similar_items_ratio' in s
        assert 'connected_components' in s
        assert 'largest_component_size' in s
        assert 'per_item_stats' in s
        # Per-item stats structure
        for sku, per_item in s['per_item_stats'].items():
            assert 'avg_sku_similarity' in per_item
            assert 'max_sku_similarity' in per_item
            assert 'avg_desc_similarity' in per_item
            assert 'max_desc_similarity' in per_item

def test_save_similarity_statistics(similarity_scorer, sample_data, tmp_path):
    """Test saving similarity statistics as JSON and CSV."""
    thresholds = [0.3, 0.5]
    stats = similarity_scorer.compute_similarity_statistics(sample_data, thresholds=thresholds)
    # JSON
    json_file = tmp_path / "similarity_stats.json"
    similarity_scorer.save_similarity_statistics(stats, json_file)
    assert json_file.exists()
    import json
    with open(json_file) as f:
        loaded = json.load(f)
    assert all(str(thresh) in loaded or float(thresh) in loaded for thresh in thresholds)
    # CSV
    csv_file = tmp_path / "similarity_stats.csv"
    similarity_scorer.save_similarity_statistics(stats, csv_file)
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert 'threshold' in df.columns
    assert set(df['threshold']) == set(thresholds)

def test_get_similarity_statistics(similarity_scorer, sample_data):
    """Test getting similarity statistics (legacy API)."""
    stats = similarity_scorer.get_similarity_statistics(sample_data)
    assert isinstance(stats, dict)
    assert 'total_items' in stats
    assert 'similar_pairs' in stats
    assert 'average_sku_similarity' in stats
    assert 'average_desc_similarity' in stats
    assert 'max_sku_similarity' in stats
    assert 'max_desc_similarity' in stats
    assert 'similar_items_ratio' in stats
    assert 'connected_components' in stats
    assert 'largest_component_size' in stats
    assert 'per_item_stats' in stats 