"""Tests for classification module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.classification import MLCategorizer

def get_real_sample_data(n=10):
    df = pd.read_excel(Path('data/inventory_data.xlsx'))
    required = ['CatalogNo', 'Description', 'Main Category', 'Sub-category']
    # Drop rows with missing or empty required fields
    for col in required:
        df = df[df[col].notna() & (df[col].astype(str).str.strip() != '')]
    return df.head(n)

@pytest.fixture
def sample_data():
    return get_real_sample_data(10)

@pytest.fixture
def ml_categorizer():
    """Create MLCategorizer instance for testing."""
    return MLCategorizer()

def test_init(ml_categorizer):
    """Test MLCategorizer initialization."""
    assert ml_categorizer.tfidf is not None
    assert ml_categorizer.svd is not None
    assert ml_categorizer.scaler is not None
    assert ml_categorizer.kmeans is None
    assert not ml_categorizer.is_fitted

def test_validate_data(ml_categorizer):
    """Test data validation."""
    # Test valid data
    valid_data = pd.DataFrame({
        "CatalogNo": ["SKU1", "SKU2"],
        "Description": ["Desc 1", "Desc 2"],
        "Enriched Description": ["Item 1", "Item 2"],
        "Main Category": ["Cat1", "Cat2"],
        "Sub-category": ["Sub1", "Sub2"]
    })
    ml_categorizer._validate_data(valid_data)  # Should not raise
    
    # Test missing columns
    invalid_data = pd.DataFrame({
        "Enriched Description": ["Item 1", "Item 2"]
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        ml_categorizer._validate_data(invalid_data)
        
    # Test empty DataFrame
    empty_data = pd.DataFrame(columns=["Enriched Description", "Main Category", "Sub-category"])
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        ml_categorizer._validate_data(empty_data)
        
    # Test all missing descriptions
    missing_desc_data = pd.DataFrame({
        "Enriched Description": [None, None],
        "Main Category": ["Cat1", "Cat2"],
        "Sub-category": ["Sub1", "Sub2"]
    })
    with pytest.raises(ValueError, match="All descriptions are missing"):
        ml_categorizer._validate_data(missing_desc_data)

def test_prepare_features(ml_categorizer, sample_data):
    """Test feature preparation."""
    # Test fitting
    features = ml_categorizer.prepare_features(sample_data, fit=True)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 100  # SVD components
    
    # Test transform only
    features = ml_categorizer.prepare_features(sample_data, fit=False)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 100
    
    # Test invalid data
    invalid_data = pd.DataFrame({
        "CatalogNo": ["SKU1"],
        "Description": ["Desc 1"],
        "Enriched Description": ["Item 1"]
    })
    with pytest.raises(ValueError):
        ml_categorizer.prepare_features(invalid_data)

def test_train(ml_categorizer, sample_data):
    """Test model training."""
    ml_categorizer.train(sample_data)
    assert ml_categorizer.kmeans is not None
    assert hasattr(ml_categorizer.kmeans, 'labels_')
    assert ml_categorizer.is_fitted
    
    # Test training with single category
    single_cat_data = sample_data.copy()
    single_cat_data["Main Category"] = "Cat1"
    ml_categorizer.train(single_cat_data)  # Should not raise
    
    # Test training with single sub-category
    single_subcat_data = sample_data.copy()
    single_subcat_data["Sub-category"] = "Sub1"
    ml_categorizer.train(single_subcat_data)  # Should log warning but not raise

def test_predict(ml_categorizer, sample_data):
    """Test prediction."""
    # Test without training
    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        ml_categorizer.predict(sample_data)
    
    # Test with training
    ml_categorizer.train(sample_data)
    predictions = ml_categorizer.predict(sample_data)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(sample_data)
    assert "Predicted Main Category" in predictions.columns
    assert "Main Category Confidence" in predictions.columns
    assert "Predicted Sub-category" in predictions.columns
    assert "Sub-category Confidence" in predictions.columns
    assert all(0 <= conf <= 1 for conf in predictions["Main Category Confidence"])
    assert all(0 <= conf <= 1 for conf in predictions["Sub-category Confidence"])
    
    # Test invalid data
    invalid_data = pd.DataFrame({
        "Enriched Description": ["Item 1"]
    })
    with pytest.raises(ValueError):
        ml_categorizer.predict(invalid_data)

def test_fit_kmeans(ml_categorizer, sample_data):
    """Test KMeans fitting."""
    features = ml_categorizer.prepare_features(sample_data, fit=True)
    clusters = ml_categorizer.fit_kmeans(sample_data["Enriched Description"].tolist())
    assert isinstance(clusters, np.ndarray)
    assert len(clusters) == len(sample_data)
    
    # Test empty descriptions
    with pytest.raises(ValueError, match="Empty descriptions list provided"):
        ml_categorizer.fit_kmeans([])

def test_assign_clusters(ml_categorizer, sample_data):
    """Test cluster assignment."""
    # Test without training
    with pytest.raises(ValueError, match="Model must be fitted before cluster assignment"):
        ml_categorizer.assign_clusters(sample_data)
    
    # Test with training
    ml_categorizer.train(sample_data)
    result = ml_categorizer.assign_clusters(sample_data)
    assert isinstance(result, pd.DataFrame)
    assert CLUSTER_COL in result.columns
    assert "Predicted Main Category" in result.columns
    assert "Main Category Confidence" in result.columns
    assert "Predicted Sub-category" in result.columns
    assert "Sub-category Confidence" in result.columns

def test_save_models(ml_categorizer, sample_data, tmp_path):
    """Test model saving."""
    # Test without training
    with pytest.raises(ValueError, match="Model must be fitted before saving"):
        ml_categorizer.save_models(tmp_path)
    
    # Train models
    ml_categorizer.train(sample_data)
    
    # Save models
    models_dir = tmp_path / "models"
    ml_categorizer.save_models(models_dir)
    
    # Check that files exist
    assert (models_dir / "tfidf.joblib").exists()
    assert (models_dir / "svd.joblib").exists()
    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "kmeans.joblib").exists()
    assert (models_dir / "main_category_model.joblib").exists()
    assert (models_dir / "main_label_encoder.joblib").exists()

def test_load_models(ml_categorizer, sample_data, tmp_path):
    """Test model loading."""
    # Train and save models
    ml_categorizer.train(sample_data)
    models_dir = tmp_path / "models"
    ml_categorizer.save_models(models_dir)
    
    # Create new instance and load models
    new_categorizer = ml_categorizer.__class__()
    new_categorizer.load_models(models_dir)
    
    # Test prediction with loaded models
    predictions = new_categorizer.predict(sample_data)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(sample_data)
    assert "Predicted Main Category" in predictions.columns
    assert "Main Category Confidence" in predictions.columns
    assert "Predicted Sub-category" in predictions.columns
    assert "Sub-category Confidence" in predictions.columns
    
    # Test loading from non-existent directory
    with pytest.raises(ValueError):
        new_categorizer.load_models(tmp_path / "nonexistent") 