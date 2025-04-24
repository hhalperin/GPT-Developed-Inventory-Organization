"""
Enhanced category prediction module using ensemble methods and advanced NLP techniques.

This module implements a sophisticated category prediction system that combines:
- TF-IDF and count vectors for text feature extraction
- Technical feature extraction for specifications
- Ensemble learning with KNN and Random Forest
- Performance tracking and model persistence

The system is designed to predict product categories based on their descriptions,
with confidence scores to determine prediction reliability.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import joblib
import os
import json
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('openai').setLevel(logging.WARNING)

class CategoryPredictor:
    """
    Enhanced category prediction using ensemble methods and advanced NLP.
    
    This class implements a sophisticated category prediction system that combines
    multiple machine learning techniques to accurately predict product categories
    based on their descriptions. It features:
    
    - Text preprocessing with technical term handling
    - Multiple feature extraction methods (TF-IDF, count vectors)
    - Technical specification extraction
    - Ensemble learning with KNN and Random Forest
    - Performance tracking and model persistence
    
    The system is designed to be both accurate and efficient, with features like:
    - LRU caching for feature extraction
    - Parallel processing for large datasets
    - Comprehensive performance metrics
    """
    
    def __init__(self, n_neighbors: int = 3, confidence_threshold: float = 0.8):
        """
        Initialize the category predictor.
        
        Args:
            n_neighbors: Number of neighbors for KNN
            confidence_threshold: Minimum confidence score to use prediction
        """
        self.n_neighbors = n_neighbors
        self.confidence_threshold = confidence_threshold
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize feature extractors
        self._initialize_feature_extractors()
        
        # Initialize cache
        self.similarity_cache = {}
        
    def _initialize_models(self) -> None:
        """Initialize machine learning models."""
        try:
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1)
            self.rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            self.model = VotingClassifier(
                estimators=[('knn', self.knn), ('rf', self.rf)],
                voting='soft',
                n_jobs=-1
            )
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
            
    def _initialize_feature_extractors(self) -> None:
        """Initialize feature extraction components."""
        try:
            self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            self.label_encoder = LabelEncoder()
            self.logger.info("Feature extractors initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing feature extractors: {str(e)}")
            raise
            
    def train(self, data: pd.DataFrame, similarity_analyzer: Any = None) -> None:
        """
        Train the category predictor with enhanced validation.
        
        Args:
            data: DataFrame containing training data
            similarity_analyzer: Optional SimilarityAnalyzer instance
        """
        try:
            self.logger.info("Starting model training")
            
            # Validate input data
            if not self._validate_training_data(data):
                return
                
            # Prepare features and labels
            X, y_main, y_sub = self._prepare_training_data(data)
            
            # Split data for validation
            X_train, X_val, y_train_main, y_val_main, y_train_sub, y_val_sub = train_test_split(
                X, y_main, y_sub, test_size=0.2, random_state=42
            )
            
            # Train models
            self._train_models(X_train, y_train_main, y_train_sub)
            
            # Validate models
            self._validate_models(X_val, y_val_main, y_val_sub)
            
            # Save model
            self._save_model()
            
            self.is_trained = True
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.is_trained = False
            raise
            
    def _validate_training_data(self, data: pd.DataFrame) -> bool:
        """Validate training data."""
        try:
            # Check required columns
            required_columns = ['Main Category', 'Sub Category', 'Enriched Description']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Check for empty values
            if data[required_columns].isna().any().any():
                self.logger.error("Found empty values in required columns")
                return False
                
            # Check minimum samples
            if len(data) < self.n_neighbors + 1:
                self.logger.error(f"Insufficient training data (need at least {self.n_neighbors + 1} samples)")
                return False
                
            # Check category diversity
            if len(data['Main Category'].unique()) < 2 or len(data['Sub Category'].unique()) < 2:
                self.logger.error("Insufficient category diversity for training")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating training data: {str(e)}")
            return False
            
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        try:
            # Prepare text features
            X = data['Enriched Description'].fillna('').astype(str)
            X = self.tfidf.fit_transform(X).toarray()
            
            # Prepare labels
            y_main = self.label_encoder.fit_transform(data['Main Category'])
            y_sub = self.label_encoder.fit_transform(data['Sub Category'])
            
            return X, y_main, y_sub
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
            
    def _train_models(self, X: np.ndarray, y_main: np.ndarray, y_sub: np.ndarray) -> None:
        """Train the machine learning models."""
        try:
            # Train KNN
            self.knn.fit(X, y_main)
            
            # Train Random Forest
            self.rf.fit(X, y_main)
            
            # Train ensemble
            self.model.fit(X, y_main)
            
            self.logger.info("Models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise
            
    def _validate_models(self, X: np.ndarray, y_main: np.ndarray, y_sub: np.ndarray) -> None:
        """Validate model performance."""
        try:
            # Get predictions
            y_pred_main = self.model.predict(X)
            y_pred_sub = self.model.predict(X)
            
            # Calculate accuracy
            main_accuracy = accuracy_score(y_main, y_pred_main)
            sub_accuracy = accuracy_score(y_sub, y_pred_sub)
            
            # Log performance metrics
            self.logger.info(f"Main category accuracy: {main_accuracy:.2f}")
            self.logger.info(f"Sub category accuracy: {sub_accuracy:.2f}")
            
            # Log classification report
            self.logger.info("\nMain category classification report:")
            self.logger.info(classification_report(y_main, y_pred_main))
            
            self.logger.info("\nSub category classification report:")
            self.logger.info(classification_report(y_sub, y_pred_sub))
            
        except Exception as e:
            self.logger.error(f"Error validating models: {str(e)}")
            raise
            
    def _save_model(self) -> None:
        """Save the trained model and feature extractors."""
        try:
            # Create models directory if it doesn't exist
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = model_dir / f"model_{timestamp}.joblib"
            joblib.dump(self.model, model_path)
            
            # Save feature extractors
            feature_extractors = {
                'tfidf': self.tfidf,
                'label_encoder': self.label_encoder
            }
            feature_path = model_dir / f"feature_extractors_{timestamp}.joblib"
            joblib.dump(feature_extractors, feature_path)
            
            self.logger.info(f"Model and feature extractors saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def predict(self, data: pd.DataFrame, similarity_analyzer: Any = None, return_format: str = 'simple') -> Tuple:
        """
        Predict categories with enhanced error handling.
        
        Args:
            data: DataFrame containing items to predict
            similarity_analyzer: Optional SimilarityAnalyzer instance
            return_format: 'simple' or 'full' output format
            
        Returns:
            Tuple of predictions and confidence scores
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, attempting to load saved model")
            if not self._load_latest_model():
                self.logger.error("No trained model available")
                if return_format == 'simple':
                    return [], []
                return [], [], [], []
                
        try:
            # Prepare features
            X = data['Enriched Description'].fillna('').astype(str)
            X = self.tfidf.transform(X).toarray()
            
            # Get predictions
            main_pred = self.model.predict(X)
            sub_pred = self.model.predict(X)
            
            # Get confidence scores
            main_proba = self.model.predict_proba(X)
            sub_proba = self.model.predict_proba(X)
            
            # Calculate confidence scores
            main_conf = np.max(main_proba, axis=1)
            sub_conf = np.max(sub_proba, axis=1)
            
            # Convert to lists
            main_pred = list(main_pred)
            sub_pred = list(sub_pred)
            main_conf = list(main_conf)
            sub_conf = list(sub_conf)
            
            if return_format == 'simple':
                return main_pred, main_conf
            return main_pred, sub_pred, main_conf, sub_conf
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            if return_format == 'simple':
                return [], []
            return [], [], [], []
            
    def _load_latest_model(self) -> bool:
        """Load the latest saved model."""
        try:
            model_dir = Path("models")
            if not model_dir.exists():
                return False
                
            model_files = list(model_dir.glob("model_*.joblib"))
            if not model_files:
                return False
                
            latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
            self._load_model(str(latest_file))
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading latest model: {str(e)}")
            return False
            
    def _load_model(self, model_path: str) -> None:
        """Load model and feature extractors."""
        try:
            # Load model
            self.model = joblib.load(model_path)
            
            # Load feature extractors
            feature_path = model_path.replace("model_", "feature_extractors_")
            feature_extractors = joblib.load(feature_path)
            
            self.tfidf = feature_extractors['tfidf']
            self.label_encoder = feature_extractors['label_encoder']
            
            self.is_trained = True
            self.logger.info(f"Model and feature extractors loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Enhanced text preprocessing with technical term handling.
        
        This method performs comprehensive text preprocessing including:
        - Unit standardization (V, A, W, mm)
        - Technical abbreviation expansion
        - Special character handling
        - Case normalization
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            str: Preprocessed text with standardized units and expanded abbreviations
            
        Example:
            >>> preprocess_text("10V AC PCB SMD")
            "10 volts alternating current printed circuit board surface mount device"
        """
        # Convert to lowercase
        text = text.lower()
        
        # Standardize units
        unit_patterns = {
            r'(\d+)v\b': r'\1 volts',
            r'(\d+)a\b': r'\1 amps',
            r'(\d+)w\b': r'\1 watts',
            r'(\d+)mm\b': r'\1 millimeters'
        }
        for pattern, replacement in unit_patterns.items():
            text = re.sub(pattern, replacement, text)
            
        # Handle common abbreviations
        abbrev_dict = {
            'ac': 'alternating current',
            'dc': 'direct current',
            'pcb': 'printed circuit board',
            'smd': 'surface mount device'
        }
        words = text.split()
        words = [abbrev_dict.get(word, word) for word in words]
        text = ' '.join(words)
        
        # Remove special characters but preserve numbers and units
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        return text.strip()

    @lru_cache(maxsize=1000)
    def extract_technical_features(self, text: str) -> Dict[str, float]:
        """
        Extract technical specifications from text.
        
        This method identifies and extracts key technical specifications from
        product descriptions, including:
        - Voltage (V)
        - Current (A)
        - Power (W)
        - Dimensions (mm)
        - Weight (g)
        
        Args:
            text: Text to extract features from
            
        Returns:
            Dict[str, float]: Dictionary containing extracted technical features.
                            Missing features are set to 0.0.
                            
        Example:
            >>> extract_technical_features("12V 2A 24W 10mm 5g")
            {'voltage': 12.0, 'current': 2.0, 'power': 24.0,
             'dimension': 10.0, 'weight': 5.0}
        """
        features = {
            'voltage': 0.0,
            'current': 0.0,
            'power': 0.0,
            'dimension': 0.0,
            'weight': 0.0
        }
        
        # Extract voltage (V)
        voltage_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:v|volts?)', text.lower())
        if voltage_match:
            features['voltage'] = float(voltage_match.group(1))
            
        # Extract current (A)
        current_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:a|amps?)', text.lower())
        if current_match:
            features['current'] = float(current_match.group(1))
            
        # Extract power (W)
        power_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:w|watts?)', text.lower())
        if power_match:
            features['power'] = float(power_match.group(1))
            
        # Extract dimensions (mm)
        dim_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)', text.lower())
        if dim_match:
            features['dimension'] = float(dim_match.group(1))
            
        # Extract weight (g)
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)', text.lower())
        if weight_match:
            features['weight'] = float(weight_match.group(1))
            
        return features

    def prepare_features(self, descriptions: List[str]) -> np.ndarray:
        """
        Prepare features using multiple feature extraction methods.
        
        This method combines several feature extraction techniques:
        1. Text preprocessing
        2. TF-IDF vectorization
        3. Count vectorization
        4. Technical feature extraction
        
        The features are then:
        - Scaled using StandardScaler
        - Reduced using PCA to preserve 95% of variance
        
        Args:
            descriptions: List of item descriptions
            
        Returns:
            np.ndarray: Combined feature matrix with shape (n_samples, n_features)
        """
        # Preprocess descriptions
        preprocessed = [self.preprocess_text(desc) for desc in descriptions]
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf.transform(preprocessed).toarray()
        
        # Extract count features
        count_features = self.count_vec.transform(preprocessed).toarray()
        
        # Extract technical features
        with ThreadPoolExecutor() as executor:
            tech_features = list(executor.map(self.extract_technical_features, preprocessed))
        tech_matrix = np.array([[v for v in f.values()] for f in tech_features])
        
        # Combine all features
        combined_features = np.hstack([tfidf_features, count_features, tech_matrix])
        
        # Scale features
        combined_features = self.scaler.transform(combined_features)
        
        # Apply PCA
        combined_features = self.pca.transform(combined_features)
        
        return combined_features

    def _evaluate_performance(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Evaluate model performance with cross-validation.
        
        This method tracks:
        - Training accuracy
        - Cross-validation scores (if enough samples)
        - Feature importance (for Random Forest)
        
        Results are stored in self.performance_metrics for later analysis.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        # Calculate accuracy
        accuracy = self.model.score(X, y)
        self.performance_metrics['accuracy'].append(accuracy)
        
        # Determine number of samples per class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples_per_class = min(class_counts)
        
        # Only perform cross-validation if we have enough samples
        if min_samples_per_class >= 2:  # Need at least 2 samples per class for CV
            # Adjust number of folds based on minimum samples per class
            n_folds = min(5, min_samples_per_class)
            cv_scores = cross_val_score(self.model, X, y, cv=n_folds)
            self.performance_metrics['cross_val_scores'].append(cv_scores.mean())
            logging.info(f"Cross-validation scores (n_folds={n_folds}): {cv_scores.mean():.4f}")
        else:
            logging.warning(f"Skipping cross-validation: insufficient samples per class (min={min_samples_per_class})")
            self.performance_metrics['cross_val_scores'].append(None)
        
        # Calculate feature importance for RandomForest
        if self.use_ensemble:
            rf_model = self.model.named_estimators_['rf']
            feature_importance = dict(zip(
                range(X.shape[1]),
                rf_model.feature_importances_
            ))
            self.performance_metrics['feature_importance'] = feature_importance
            
        logging.info(f"Model performance: Accuracy={accuracy:.4f}")

    def should_use_prediction(self, confidence_score: float) -> bool:
        """
        Determine if prediction should be used based on confidence.
        
        Args:
            confidence_score: Confidence score to check
            
        Returns:
            bool: Whether to use the prediction
        """
        return float(confidence_score) >= self.confidence_threshold

    def update_training_data(self, new_descriptions: List[str], new_categories: List[str]) -> None:
        """
        Update model with new training data.
        
        This method implements incremental learning by:
        1. Retraining on new data
        2. Evaluating performance
        3. Saving updated model
        
        Args:
            new_descriptions: List of new item descriptions
            new_categories: List of new categories
            
        Raises:
            ValueError: If new_descriptions and new_categories have different lengths
        """
        # Prepare features for new data
        X_new = self.prepare_features(new_descriptions)
        y_new = self.label_encoder.transform(new_categories)
        
        # Update model with new data
        self.model.fit(X_new, y_new)
        
        # Evaluate performance on new data
        self._evaluate_performance(X_new, y_new)
        
        # Save updated model
        self._save_model() 