"""Tests for the config module."""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from src.config import Config

@pytest.fixture
def config():
    """Create a Config instance for testing."""
    return Config()

def test_init(config):
    """Test Config initialization."""
    assert config.similarity_threshold == 0.8
    assert config.dbscan_eps == 0.5
    assert config.min_samples == 2
    assert config.kmeans_clusters == 3
    assert isinstance(config.output_dir, Path)
    assert isinstance(config.model_dir, Path)

def test_load_from_env(config, monkeypatch):
    """Test loading configuration from environment variables."""
    # Set environment variables
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.9")
    monkeypatch.setenv("DBSCAN_EPS", "0.6")
    monkeypatch.setenv("MIN_SAMPLES", "3")
    monkeypatch.setenv("KMEANS_CLUSTERS", "4")
    
    # Load configuration
    config.load_from_env()
    
    # Check values
    assert config.similarity_threshold == 0.9
    assert config.dbscan_eps == 0.6
    assert config.min_samples == 3
    assert config.kmeans_clusters == 4

def test_load_from_env_missing(config, monkeypatch):
    """Test loading configuration with missing environment variables."""
    # Clear environment variables
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.delenv("DBSCAN_EPS", raising=False)
    monkeypatch.delenv("MIN_SAMPLES", raising=False)
    monkeypatch.delenv("KMEANS_CLUSTERS", raising=False)
    
    # Load configuration
    config.load_from_env()
    
    # Check that default values are preserved
    assert config.similarity_threshold == 0.8
    assert config.dbscan_eps == 0.5
    assert config.min_samples == 2
    assert config.kmeans_clusters == 3

def test_load_from_file(config, tmp_path):
    """Test loading configuration from file."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    config_content = """
    similarity_threshold: 0.9
    dbscan_eps: 0.6
    min_samples: 3
    kmeans_clusters: 4
    """
    config_file.write_text(config_content)
    
    # Load configuration
    config.load_from_file(config_file)
    
    # Check values
    assert config.similarity_threshold == 0.9
    assert config.dbscan_eps == 0.6
    assert config.min_samples == 3
    assert config.kmeans_clusters == 4

def test_load_from_file_missing(config, tmp_path):
    """Test loading configuration from non-existent file."""
    # Try to load from non-existent file
    config_file = tmp_path / "nonexistent.yaml"
    config.load_from_file(config_file)
    
    # Check that default values are preserved
    assert config.similarity_threshold == 0.8
    assert config.dbscan_eps == 0.5
    assert config.min_samples == 2
    assert config.kmeans_clusters == 3

def test_save_to_file(config, tmp_path):
    """Test saving configuration to file."""
    # Set some values
    config.similarity_threshold = 0.9
    config.dbscan_eps = 0.6
    
    # Save configuration
    config_file = tmp_path / "config.yaml"
    config.save_to_file(config_file)
    
    # Check that file exists
    assert config_file.exists()
    
    # Load configuration back
    new_config = Config()
    new_config.load_from_file(config_file)
    
    # Check values
    assert new_config.similarity_threshold == 0.9
    assert new_config.dbscan_eps == 0.6

def test_validate_config(config):
    """Test configuration validation."""
    # Test valid configuration
    assert config.validate_config()
    
    # Test invalid similarity threshold
    config.similarity_threshold = 1.5
    assert not config.validate_config()
    
    # Test invalid DBSCAN parameters
    config.similarity_threshold = 0.8
    config.dbscan_eps = -0.1
    assert not config.validate_config()
    
    # Test invalid K-means clusters
    config.dbscan_eps = 0.5
    config.kmeans_clusters = 0
    assert not config.validate_config() 