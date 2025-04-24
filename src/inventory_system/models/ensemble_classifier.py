"""
Ensemble classifier with hierarchical support for inventory categorization.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import joblib
import os

@dataclass
class ModelConfig:
    """Configuration for ensemble classifier."""
    n_neighbors: int = 5
    n_estimators: int = 100
    hidden_layer_sizes: Tuple[int, ...] = (100, 50)
    calibration_method: str = 'sigmoid'
    cv_folds: int = 5
    min_samples_leaf: int = 2
    confidence_threshold: float = 0.7

class BaseClassifier(ABC):
    """Base class for individual classifiers in the ensemble."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass

class KNNClassifier(BaseClassifier):
    """KNN classifier with calibration."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = KNeighborsClassifier(n_neighbors=config.n_neighbors)
        self.calibrated_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit and calibrate the classifier."""
        self.model.fit(X, y)
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=config.calibration_method,
            cv=config.cv_folds
        )
        self.calibrated_model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.calibrated_model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.calibrated_model.predict_proba(X)

class RandomForestClassifierWrapper(BaseClassifier):
    """Random Forest classifier with calibration."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            min_samples_leaf=config.min_samples_leaf
        )
        self.calibrated_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit and calibrate the classifier."""
        self.model.fit(X, y)
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=config.calibration_method,
            cv=config.cv_folds
        )
        self.calibrated_model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.calibrated_model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.calibrated_model.predict_proba(X)

class NeuralNetworkClassifier(BaseClassifier):
    """Neural Network classifier with calibration."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = MLPClassifier(
            hidden_layer_sizes=config.hidden_layer_sizes,
            max_iter=1000
        )
        self.calibrated_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit and calibrate the classifier."""
        self.model.fit(X, y)
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=self.config.calibration_method,
            cv=self.config.cv_folds
        )
        self.calibrated_model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.calibrated_model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.calibrated_model.predict_proba(X)

class HierarchicalClassifier:
    """Hierarchical classifier for coarse-to-fine categorization."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.coarse_classifier = None
        self.fine_classifiers = {}
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X: np.ndarray, y_coarse: np.ndarray, y_fine: np.ndarray) -> None:
        """Fit hierarchical classifier."""
        # Train coarse classifier
        self.coarse_classifier = EnsembleClassifier(self.config)
        self.coarse_classifier.fit(X, y_coarse)
        
        # Train fine classifiers for each coarse category
        for category in np.unique(y_coarse):
            mask = y_coarse == category
            if np.sum(mask) >= self.config.min_samples_leaf:
                fine_classifier = EnsembleClassifier(self.config)
                fine_classifier.fit(X[mask], y_fine[mask])
                self.fine_classifiers[category] = fine_classifier
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make hierarchical predictions."""
        # Predict coarse categories
        coarse_pred = self.coarse_classifier.predict(X)
        
        # Predict fine categories
        fine_pred = np.empty(len(X), dtype=object)
        for category, classifier in self.fine_classifiers.items():
            mask = coarse_pred == category
            if np.any(mask):
                fine_pred[mask] = classifier.predict(X[mask])
                
        return fine_pred
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get hierarchical prediction probabilities."""
        # Get coarse probabilities
        coarse_proba = self.coarse_classifier.predict_proba(X)
        
        # Get fine probabilities
        fine_proba = np.zeros((len(X), len(np.unique(y_fine))))
        for category, classifier in self.fine_classifiers.items():
            mask = coarse_pred == category
            if np.any(mask):
                fine_proba[mask] = classifier.predict_proba(X[mask])
                
        return fine_proba

class EnsembleClassifier:
    """Ensemble classifier combining multiple models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.classifiers = [
            KNNClassifier(config),
            RandomForestClassifierWrapper(config),
            NeuralNetworkClassifier(config)
        ]
        self.weights = None
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ensemble classifier."""
        # Train individual classifiers
        for classifier in self.classifiers:
            classifier.fit(X, y)
            
        # Calculate weights based on cross-validation performance
        self.weights = self._calculate_weights(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        # Get predictions from each classifier
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
        
        # Weighted voting
        weighted_predictions = np.zeros((len(X), len(np.unique(y))))
        for i, classifier in enumerate(self.classifiers):
            proba = classifier.predict_proba(X)
            weighted_predictions += self.weights[i] * proba
            
        return np.argmax(weighted_predictions, axis=1)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        # Get probabilities from each classifier
        probabilities = np.array([classifier.predict_proba(X) for classifier in self.classifiers])
        
        # Weighted average
        weighted_proba = np.zeros_like(probabilities[0])
        for i, proba in enumerate(probabilities):
            weighted_proba += self.weights[i] * proba
            
        return weighted_proba
        
    def _calculate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate classifier weights based on cross-validation performance."""
        weights = np.zeros(len(self.classifiers))
        
        for i, classifier in enumerate(self.classifiers):
            scores = cross_val_score(
                classifier.model,
                X,
                y,
                cv=self.config.cv_folds,
                scoring='accuracy'
            )
            weights[i] = np.mean(scores)
            
        # Normalize weights
        weights = weights / np.sum(weights)
        return weights
        
    def save(self, path: str) -> None:
        """Save the ensemble classifier."""
        model_data = {
            'classifiers': self.classifiers,
            'weights': self.weights,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")
        
    def load(self, path: str) -> None:
        """Load the ensemble classifier."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        model_data = joblib.load(path)
        self.classifiers = model_data['classifiers']
        self.weights = model_data['weights']
        self.config = model_data['config']
        
        self.logger.info(f"Model loaded from {path}") 