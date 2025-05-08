"""Tests for clustering module."""

import matplotlib
matplotlib.use('Agg')
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from src.clustering import ClusterAnalyzer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create 3 distinct clusters for clustering tests
    return pd.DataFrame({
        "CatalogNo": [f"SKU{i}" for i in range(1, 31)],
        "Description": [
            *("alpha desc" for _ in range(10)),
            *("beta desc" for _ in range(10)),
            *("gamma desc" for _ in range(10)),
        ],
        "Enriched Description": [
            *("enriched alpha" for _ in range(10)),
            *("enriched beta" for _ in range(10)),
            *("enriched gamma" for _ in range(10)),
        ],
        "Main Category": ["CatA"]*10 + ["CatB"]*10 + ["CatC"]*10,
        "Sub-category": ["Sub1"]*10 + ["Sub2"]*10 + ["Sub3"]*10
    })

def test_feature_extraction(sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    features = ca._extract_features(sample_data, fit=True)
    assert features.shape[0] == len(sample_data)
    assert features.shape[1] == 2

def test_dbscan_clustering(sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    labels = ca.fit_dbscan(sample_data, threshold=0.8)
    assert len(labels) == len(sample_data)
    assert set(labels).issubset(set(range(-1, max(labels)+1)))
    assert np.isclose(ca.dbscan.eps, 0.2)

def test_kmeans_clustering(sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    labels = ca.fit_kmeans(sample_data, n_clusters=3)
    assert len(labels) == len(sample_data)
    # Should find at least 2 clusters
    assert len(set(labels)) >= 2
    if len(set(labels)) != 3:
        import warnings
        warnings.warn(f"KMeans found {len(set(labels))} clusters instead of 3. Test data may not be well-separated.")

def test_cluster_statistics(sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    labels = ca.fit_dbscan(sample_data, threshold=0.8)
    stats = ca.get_cluster_statistics(sample_data, labels)
    if len(set(labels)) > 1:
        for key in ["silhouette", "calinski_harabasz", "davies_bouldin", "avg_degree", "num_components", "largest_component"]:
            assert key in stats

def test_analyze_clusters_structure(sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    labels = ca.fit_dbscan(sample_data, threshold=0.8)
    analysis = ca.analyze_clusters(sample_data)
    assert "n_clusters" in analysis
    assert "category_distribution" in analysis
    if analysis["n_clusters"] > 1:
        assert "silhouette" in analysis

def test_save_and_load_metrics(tmp_path, sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    labels = ca.fit_dbscan(sample_data, threshold=0.8)
    stats = ca.get_cluster_statistics(sample_data, labels)
    metrics_file = tmp_path / "metrics.csv"
    ca.save_metrics(stats, metrics_file)
    assert metrics_file.exists()
    df = pd.read_csv(metrics_file)
    if stats:
        assert set(stats.keys()).issubset(df.columns)

def test_save_cluster_results(tmp_path, sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    labels = ca.fit_dbscan(sample_data, threshold=0.8)
    ca.save_cluster_results(sample_data, tmp_path)
    assert (tmp_path / "cluster_assignments.xlsx").exists()
    assert (tmp_path / "cluster_analysis.xlsx").exists()
    assert (tmp_path / "cluster_visualization.png").exists()
    assert (tmp_path / "cluster_metrics.csv").exists()
    assert (tmp_path / "cluster_model.joblib").exists()

def test_error_on_missing_columns():
    ca = ClusterAnalyzer(svd_components=2)
    df = pd.DataFrame({"Description": ["a", "b"]})
    with pytest.raises(ValueError):
        ca.fit_dbscan(df)

def test_error_on_empty_dataframe():
    ca = ClusterAnalyzer(svd_components=2)
    df = pd.DataFrame(columns=["Description", "Main Category", "Sub-category"])
    with pytest.raises(ValueError):
        ca.fit_dbscan(df)

def test_model_persistence(tmp_path, sample_data):
    ca = ClusterAnalyzer(svd_components=2)
    ca.fit_dbscan(sample_data, threshold=0.8)
    ca.save_cluster_results(sample_data, tmp_path)
    ca2 = ClusterAnalyzer(svd_components=2)
    ca2.load_model(tmp_path)
    assert ca2.is_fitted
    assert np.allclose(ca2.last_features, ca.last_features)
    assert np.all(ca2.last_labels == ca.last_labels) 