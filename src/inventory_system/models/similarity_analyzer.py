"""
Similarity analysis and KNN model for category reuse prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import pickle
import os
from datetime import datetime
from .similarity_features import SimilarityFeaturePipeline, SimilarityConfig
from .ensemble_classifier import EnsembleClassifier, HierarchicalClassifier, ModelConfig
from .model_monitor import ModelMonitor

class SimilarityAnalyzer:
    """
    Handles similarity analysis and model training for category reuse prediction.
    """
    
    def __init__(self, n_neighbors: int = 5, min_samples_for_training: int = 10):
        """
        Initialize the similarity analyzer.
        
        Args:
            n_neighbors: Number of neighbors for KNN model
            min_samples_for_training: Minimum number of samples required for training
        """
        self.n_neighbors = n_neighbors
        self.min_samples_for_training = min_samples_for_training
        
        # Initialize components
        self.similarity_config = SimilarityConfig()
        self.model_config = ModelConfig(n_neighbors=n_neighbors)
        
        self.feature_pipeline = SimilarityFeaturePipeline(self.similarity_config)
        self.classifier = None
        self.hierarchical_classifier = None
        self.monitor = ModelMonitor()
        
        self.last_training_time = None
        self.training_history = []
        self.logger = logging.getLogger(__name__)
        
    def calculate_similarities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate similarity scores between items.
        
        Args:
            data: DataFrame with enriched descriptions and categories
            
        Returns:
            DataFrame with similarity features
        """
        return self.feature_pipeline.extract_features(data)
        
    def train_model(self, data: pd.DataFrame, similarity_features: pd.DataFrame, 
                   is_second_pass: bool = False) -> Dict[str, float]:
        """
        Train model using similarity scores and GPT-determined categories.
        
        Args:
            data: DataFrame with categories
            similarity_features: DataFrame with similarity scores
            is_second_pass: Whether this is the second pass for alignment
            
        Returns:
            Dictionary of performance metrics
        """
        if len(data) < self.min_samples_for_training:
            self.logger.warning(f"Not enough samples for training. Need {self.min_samples_for_training}, got {len(data)}")
            return None
            
        # Prepare features and target
        X = similarity_features.values
        y = data['Main Category'].values
        
        # Train appropriate model
        if is_second_pass and 'Sub Category' in data.columns:
            # Use hierarchical classifier for second pass
            self.hierarchical_classifier = HierarchicalClassifier(self.model_config)
            self.hierarchical_classifier.fit(X, y, data['Sub Category'].values)
            self.classifier = None
        else:
            # Use ensemble classifier for first pass
            self.classifier = EnsembleClassifier(self.model_config)
            self.classifier.fit(X, y)
            self.hierarchical_classifier = None
            
        # Evaluate performance
        y_pred = self.predict_categories(X)
        metrics = self.monitor.evaluate_performance(y, y_pred, len(data))
        
        self.last_training_time = datetime.now()
        self.training_history.append(metrics)
        
        return metrics._asdict()
        
    def predict_categories(self, X: np.ndarray) -> np.ndarray:
        """
        Predict categories using the appropriate model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted categories
        """
        if self.hierarchical_classifier is not None:
            return self.hierarchical_classifier.predict(X)
        elif self.classifier is not None:
            return self.classifier.predict(X)
        else:
            raise ValueError("No trained model available")
            
    def predict_category_reuse(self, new_item: pd.Series, existing_items: pd.DataFrame) -> Tuple[bool, float]:
        """
        Predict if a new item's category can be reused based on similarities.
        
        Args:
            new_item: Series with new item data
            existing_items: DataFrame with existing items
            
        Returns:
            Tuple of (can_reuse, confidence_score)
        """
        if self.classifier is None and self.hierarchical_classifier is None:
            raise ValueError("No trained model available")
            
        # Calculate similarities
        similarities = self.calculate_similarities(existing_items)
        
        # Get predictions and probabilities
        X = similarities.values
        if self.hierarchical_classifier is not None:
            proba = self.hierarchical_classifier.predict_proba(X)
        else:
            proba = self.classifier.predict_proba(X)
            
        # Calculate reuse confidence
        confidence = np.max(proba, axis=1)
        can_reuse = confidence > self.model_config.confidence_threshold
        
        return can_reuse, confidence
        
    def should_retrain(self, new_data_size: int, min_hours_since_last_train: float = 1.0) -> bool:
        """
        Determine if the model should be retrained based on new data and time since last training.
        
        Args:
            new_data_size: Number of new samples since last training
            min_hours_since_last_train: Minimum hours required since last training
            
        Returns:
            bool: Whether to retrain the model
        """
        if self.last_training_time is None:
            return True
            
        hours_since_last_train = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return (hours_since_last_train >= min_hours_since_last_train and 
                new_data_size >= self.min_samples_for_training)
        
    def align_categories(self, data: pd.DataFrame, similarity_features: pd.DataFrame) -> pd.DataFrame:
        """
        Perform second pass alignment of categories using the trained model.
        
        Args:
            data: DataFrame with initial categories
            similarity_features: DataFrame with similarity scores
            
        Returns:
            DataFrame with aligned categories
        """
        if self.classifier is None and self.hierarchical_classifier is None:
            raise ValueError("No trained model available")
            
        # Get predictions
        X = similarity_features.values
        aligned_categories = self.predict_categories(X)
        
        # Create new DataFrame with aligned categories
        aligned_data = data.copy()
        aligned_data['Aligned Category'] = aligned_categories
        
        # Calculate confidence scores
        if self.hierarchical_classifier is not None:
            proba = self.hierarchical_classifier.predict_proba(X)
        else:
            proba = self.classifier.predict_proba(X)
            
        aligned_data['Alignment Confidence'] = np.max(proba, axis=1)
        
        return aligned_data
        
    def save_model(self, path: str):
        """
        Save the trained model and related components.
        
        Args:
            path: Path to save the model
        """
        if self.classifier is None and self.hierarchical_classifier is None:
            raise ValueError("No model to save")
            
        model_data = {
            'classifier': self.classifier,
            'hierarchical_classifier': self.hierarchical_classifier,
            'feature_pipeline': self.feature_pipeline,
            'last_training_time': self.last_training_time,
            'training_history': self.training_history,
            'config': {
                'n_neighbors': self.n_neighbors,
                'min_samples_for_training': self.min_samples_for_training
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """
        Load a trained model and related components.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.classifier = model_data['classifier']
        self.hierarchical_classifier = model_data['hierarchical_classifier']
        self.feature_pipeline = model_data['feature_pipeline']
        self.last_training_time = model_data['last_training_time']
        self.training_history = model_data['training_history']
        
        config = model_data['config']
        self.n_neighbors = config['n_neighbors']
        self.min_samples_for_training = config['min_samples_for_training']
        
        self.logger.info(f"Model loaded from {path}")
        
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a performance report.
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            str: Report content
        """
        return self.monitor.generate_report(output_file) 