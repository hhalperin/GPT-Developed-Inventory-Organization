"""
Test suite for the category predictor module.
"""

import unittest
import pandas as pd
import numpy as np
from inventory_system.category_predictor import CategoryPredictor
from inventory_system.validation.data_validator import DataValidator

class TestCategoryPredictor(unittest.TestCase):
    """Test cases for the CategoryPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = CategoryPredictor()
        self.validator = DataValidator()
        
        # Create test data with explicit string types
        self.test_data = pd.DataFrame({
            "CatalogNo": pd.Series(["TEST001", "TEST002", "TEST003", "TEST004", "TEST005"], dtype="string"),
            "Description": pd.Series([
                "10A 120V Circuit Breaker",
                "20A 240V Circuit Breaker",
                "15A 120V GFCI Outlet",
                "30A 240V Circuit Breaker",
                "20A 120V GFCI Outlet"
            ], dtype="string"),
            "MfrCode": pd.Series(["MFR1", "MFR2", "MFR3", "MFR1", "MFR3"], dtype="string"),
            "Main Category": pd.Series(["Circuit Breakers", "Circuit Breakers", "Outlets", "Circuit Breakers", "Outlets"], dtype="string"),
            "Sub-category": pd.Series(["Standard", "Standard", "GFCI", "Standard", "GFCI"], dtype="string")
        })
        
        # Create test data for prediction
        self.prediction_data = pd.DataFrame({
            "CatalogNo": pd.Series(["TEST006", "TEST007"], dtype="string"),
            "Description": pd.Series([
                "30A 240V Circuit Breaker",
                "20A 120V GFCI Outlet"
            ], dtype="string"),
            "MfrCode": pd.Series(["MFR1", "MFR3"], dtype="string")
        })

    def test_initialization(self):
        """Test predictor initialization."""
        self.assertIsNotNone(self.predictor)
        self.assertFalse(self.predictor.is_trained)
        self.assertEqual(self.predictor.n_neighbors, 5)
        self.assertEqual(self.predictor.confidence_threshold, 0.7)

    def test_feature_extraction(self):
        """Test feature extraction methods."""
        # Test technical feature extraction
        features = self.predictor.extract_technical_features("10A 120V Circuit Breaker")
        self.assertEqual(features["current"], 10.0)
        self.assertEqual(features["voltage"], 120.0)
        
        # Test catalog feature extraction
        features = self.predictor.extract_catalog_features("TEST001")
        self.assertEqual(features["length"], 7)
        self.assertTrue(features["has_numbers"])
        self.assertTrue(features["has_letters"])

    def test_training(self):
        """Test model training."""
        # Validate test data
        is_valid, errors = self.validator.validate_dataframe(self.test_data)
        self.assertTrue(is_valid, f"Test data validation failed: {errors}")
        
        # Train model
        self.predictor.train(self.test_data)
        self.assertTrue(self.predictor.is_trained)
        
        # Check if models are trained
        self.assertIsNotNone(self.predictor.main_category_knn)
        self.assertIsNotNone(self.predictor.sub_category_knn)
        self.assertIsNotNone(self.predictor.main_category_rf)
        self.assertIsNotNone(self.predictor.sub_category_rf)

    def test_prediction(self):
        """Test category prediction."""
        # Train model first
        self.predictor.train(self.test_data)
        
        # Make predictions
        main_cats, sub_cats, main_conf, sub_conf = self.predictor.predict(self.prediction_data)
        
        # Check prediction results
        self.assertEqual(len(main_cats), len(self.prediction_data))
        self.assertEqual(len(sub_cats), len(self.prediction_data))
        self.assertEqual(len(main_conf), len(self.prediction_data))
        self.assertEqual(len(sub_conf), len(self.prediction_data))
        
        # Check confidence scores
        self.assertTrue(all(0 <= conf <= 1 for conf in main_conf))
        self.assertTrue(all(0 <= conf <= 1 for conf in sub_conf))

    def test_confidence_threshold(self):
        """Test confidence threshold functionality."""
        # Train model
        self.predictor.train(self.test_data)
        
        # Test with high confidence
        self.assertTrue(self.predictor.should_use_prediction(0.9, 0.9))
        
        # Test with low confidence
        self.assertFalse(self.predictor.should_use_prediction(0.5, 0.5))

    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train model
        self.predictor.train(self.test_data)
        
        # Save model
        self.predictor.save_model()
        
        # Create new predictor and load model
        new_predictor = CategoryPredictor()
        new_predictor.load_model(self.predictor.model_dir)
        
        # Check if loaded correctly
        self.assertTrue(new_predictor.is_trained)
        self.assertIsNotNone(new_predictor.main_category_knn)
        self.assertIsNotNone(new_predictor.sub_category_knn)

    def test_performance_evaluation(self):
        """Test model performance evaluation."""
        # Train model
        self.predictor.train(self.test_data)
        
        # Evaluate performance
        self.predictor._evaluate_performance(self.test_data)
        
        # Check performance metrics
        self.assertIn("knn", self.predictor.performance_metrics)
        self.assertIn("rf", self.predictor.performance_metrics)
        
        # Check accuracy values
        self.assertTrue(0 <= self.predictor.performance_metrics["knn"]["main_accuracy"] <= 1)
        self.assertTrue(0 <= self.predictor.performance_metrics["knn"]["sub_accuracy"] <= 1)

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        # Train model with optimization
        self.predictor.train(self.test_data)
        
        # Check if best parameters are stored
        self.assertIn("best_params", self.predictor.performance_metrics)
        self.assertIn("knn", self.predictor.performance_metrics["best_params"])
        self.assertIn("rf", self.predictor.performance_metrics["best_params"])

    def test_training_data_update(self):
        """Test training data update functionality."""
        # Train initial model
        self.predictor.train(self.test_data)
        
        # Create new data
        new_data = pd.DataFrame({
            "CatalogNo": ["TEST006"],
            "Description": ["40A 240V Circuit Breaker"],
            "MfrCode": ["MFR1"],
            "Main Category": ["Circuit Breakers"],
            "Sub-category": ["Standard"]
        })
        
        # Update training data
        self.predictor.update_training_data(new_data)
        
        # Check if model was retrained
        self.assertTrue(self.predictor.is_trained)

if __name__ == "__main__":
    unittest.main() 