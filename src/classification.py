"""Category assignment and classification utilities for the inventory tool."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.cluster import KMeans
from pathlib import Path
import re

from .config import (
    SVD_COMPONENTS,
    TFIDF_MAX_FEATURES,
    RANDOM_FOREST_ESTIMATORS,
    ML_CONFIDENCE_THRESHOLD,
    KMEANS_CLUSTERS,
    DESCRIPTION_COL,
    ENRICHED_DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    CLUSTER_COL,
    CATALOG_NUMBER_COL,
    MFR_CODE_COL
)

logger = logging.getLogger(__name__)

class MLCategorizer:
    """Machine learning-based categorizer for inventory items."""
    
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES, n_components: int = SVD_COMPONENTS):
        """Initialize the categorizer.
        
        Args:
            max_features: Maximum number of features for TF-IDF.
            n_components: Number of components for SVD.
        """
        self.tfidf = TfidfVectorizer(max_features=max_features)
        self.svd = TruncatedSVD(n_components=n_components)
        self.scaler = StandardScaler()
        self.kmeans = None
        self.is_fitted = False
        self.main_category_model = RandomForestClassifier(n_estimators=RANDOM_FOREST_ESTIMATORS)
        self.sub_category_models = {}
        self.main_label_encoder = LabelEncoder()
        self.sub_label_encoders = {}
        self.confidence_threshold = ML_CONFIDENCE_THRESHOLD
        
    def _extract_sku_features(self, sku: str) -> np.ndarray:
        """Extract features from SKU.
        
        Args:
            sku: SKU string.
            
        Returns:
            Array of SKU features [length, digit_count, alpha_count].
        """
        if not isinstance(sku, str):
            return np.array([0, 0, 0])
        return np.array([
            len(sku),
            sum(c.isdigit() for c in sku),
            sum(c.isalpha() for c in sku)
        ])
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure.
        
        Args:
            data: DataFrame to validate.
            
        Raises:
            ValueError: If data is invalid.
        """
        required_cols = [DESCRIPTION_COL, MAIN_CATEGORY_COL, SUB_CATEGORY_COL, CATALOG_NUMBER_COL]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if len(data) == 0:
            raise ValueError("Empty DataFrame provided")
            
        if data[DESCRIPTION_COL].isna().all():
            raise ValueError("All descriptions are missing")
        
    def prepare_features(self, data: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Prepare features for clustering.
        
        Args:
            data: DataFrame containing inventory data.
            fit: Whether to fit the transformers.
            
        Returns:
            Feature matrix.
            
        Raises:
            ValueError: If data is invalid.
        """
        self._validate_data(data)
        
        # Extract text features
        desc_col = ENRICHED_DESCRIPTION_COL if ENRICHED_DESCRIPTION_COL in data.columns else DESCRIPTION_COL
        descriptions = data[desc_col].fillna('')
        
        # Transform text to TF-IDF features
        if fit:
            tfidf_features = self.tfidf.fit_transform(descriptions)
        else:
            tfidf_features = self.tfidf.transform(descriptions)
            
        # Apply SVD for dimensionality reduction
        if fit:
            svd_features = self.svd.fit_transform(tfidf_features)
        else:
            svd_features = self.svd.transform(tfidf_features)
            
        # Extract SKU features
        sku_features = np.array([self._extract_sku_features(sku) for sku in data[CATALOG_NUMBER_COL]])
        
        # Combine features
        features = np.hstack([svd_features, sku_features])
        
        return features
        
    def train(self, data: pd.DataFrame) -> None:
        """Train the categorization model.
        
        Args:
            data: DataFrame containing inventory data.
            
        Raises:
            ValueError: If data is invalid or training fails.
        """
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Initialize and fit KMeans
            self.kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42)
            self.kmeans.fit(scaled_features)
            
            # Train main category model
            main_categories = data[MAIN_CATEGORY_COL].fillna('Unknown')
            self.main_label_encoder.fit(main_categories)
            y_main = self.main_label_encoder.transform(main_categories)
            self.main_category_model.fit(features, y_main)
            
            # Train sub-category models for each main category
            for main_cat in self.main_label_encoder.classes_:
                mask = data[MAIN_CATEGORY_COL] == main_cat
                if not mask.any():
                    continue
                    
                sub_categories = data.loc[mask, SUB_CATEGORY_COL].fillna('Unknown')
                if len(sub_categories.unique()) < 2:
                    logger.warning(f"Skipping sub-category model for {main_cat}: insufficient unique categories")
                    continue
                    
                sub_encoder = LabelEncoder()
                sub_encoder.fit(sub_categories)
                y_sub = sub_encoder.transform(sub_categories)
                
                sub_model = RandomForestClassifier(n_estimators=RANDOM_FOREST_ESTIMATORS)
                sub_model.fit(features[mask], y_sub)
                
                self.sub_category_models[main_cat] = sub_model
                self.sub_label_encoders[main_cat] = sub_encoder
                
            self.is_fitted = True
            logger.info("Successfully trained categorization models")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ValueError(f"Training failed: {str(e)}")
        
    def predict(self, data: pd.DataFrame, gpt_client=None) -> pd.DataFrame:
        """Predict categories for new data.
        
        Args:
            data: DataFrame containing inventory data.
            gpt_client: Optional GPT client for fallback categorization.
            
        Returns:
            DataFrame with predicted categories and confidence scores.
            
        Raises:
            ValueError: If model is not fitted or data is invalid.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self._validate_data(data)
            
        # Prepare features
        features = self.prepare_features(data, fit=False)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict clusters
        clusters = self.kmeans.predict(scaled_features)
        
        # Predict main categories with confidence
        y_main_proba = self.main_category_model.predict_proba(features)
        y_main_pred = self.main_category_model.predict(features)
        main_categories = self.main_label_encoder.inverse_transform(y_main_pred)
        main_confidences = np.max(y_main_proba, axis=1)
        
        # Predict sub-categories with confidence
        sub_categories = []
        sub_confidences = []
        
        for i, (main_cat, main_conf) in enumerate(zip(main_categories, main_confidences)):
            if main_cat in self.sub_category_models and main_conf >= self.confidence_threshold:
                sub_model = self.sub_category_models[main_cat]
                sub_encoder = self.sub_label_encoders[main_cat]
                y_sub_proba = sub_model.predict_proba([features[i]])
                y_sub_pred = sub_model.predict([features[i]])
                sub_cat = sub_encoder.inverse_transform(y_sub_pred)[0]
                sub_conf = np.max(y_sub_proba)
            else:
                sub_cat = 'Unknown'
                sub_conf = 0.0
                
            # GPT fallback if confidence is low
            if (main_conf < self.confidence_threshold or sub_conf < self.confidence_threshold) and gpt_client:
                try:
                    catalog_no = data.iloc[i][CATALOG_NUMBER_COL]
                    mfr_code = data.iloc[i].get(MFR_CODE_COL, 'Unknown')
                    desc = data.iloc[i].get(ENRICHED_DESCRIPTION_COL, data.iloc[i][DESCRIPTION_COL])
                    
                    if main_conf < self.confidence_threshold:
                        main_prompt = (
                            f"Determine the main category for this electrical part. "
                            f"CatalogNo: {catalog_no}, Manufacturer: {mfr_code}, "
                            f"Description: {desc}"
                        )
                        main_cat = gpt_client.call_gpt_api(main_prompt)
                        main_conf = 0.0  # Indicate fallback
                        
                    if sub_conf < self.confidence_threshold:
                        sub_prompt = (
                            f"Determine the sub-category for this electrical part. "
                            f"CatalogNo: {catalog_no}, Manufacturer: {mfr_code}, "
                            f"Description: {desc}, Main Category: {main_cat}"
                        )
                        sub_cat = gpt_client.call_gpt_api(sub_prompt)
                        sub_conf = 0.0  # Indicate fallback
                        
                except Exception as e:
                    logger.error(f"GPT fallback failed for row {i}: {e}")
                    if main_conf < self.confidence_threshold:
                        main_cat = 'Unknown'
                    if sub_conf < self.confidence_threshold:
                        sub_cat = 'Unknown'
                        
            sub_categories.append(sub_cat)
            sub_confidences.append(sub_conf)
            
        result = data.copy()
        result[CLUSTER_COL] = clusters
        result['Predicted Main Category'] = main_categories
        result['Main Category Confidence'] = main_confidences
        result['Predicted Sub-category'] = sub_categories
        result['Sub-category Confidence'] = sub_confidences
        
        # Store original categories for change tracking
        result['Original Main Category'] = data[MAIN_CATEGORY_COL]
        result['Original Sub-category'] = data[SUB_CATEGORY_COL]
        
        # Compute category changes
        result['Main Category Changed'] = result['Predicted Main Category'] != result['Original Main Category']
        result['Sub-category Changed'] = result['Predicted Sub-category'] != result['Original Sub-category']
        
        # Log statistics
        main_changes = result['Main Category Changed'].sum()
        sub_changes = result['Sub-category Changed'].sum()
        logger.info(f"Category changes: {main_changes} main categories, {sub_changes} sub-categories")
        
        return result
        
    def reevaluate_categories(self, data: pd.DataFrame, gpt_client) -> pd.DataFrame:
        """Reevaluate categories using GPT for all items.
        
        Args:
            data: DataFrame containing inventory data.
            gpt_client: GPT client for categorization.
            
        Returns:
            DataFrame with reevaluated categories and change statistics.
        """
        logger.info("Starting category reevaluation...")
        
        result = data.copy()
        result['GPT Main Category'] = None
        result['GPT Sub-category'] = None
        
        for i, row in result.iterrows():
            try:
                catalog_no = row[CATALOG_NUMBER_COL]
                mfr_code = row.get(MFR_CODE_COL, 'Unknown')
                desc = row.get(ENRICHED_DESCRIPTION_COL, row[DESCRIPTION_COL])
                
                # Get main category
                main_prompt = (
                    f"Determine the main category for this electrical part. "
                    f"CatalogNo: {catalog_no}, Manufacturer: {mfr_code}, "
                    f"Description: {desc}"
                )
                main_cat = gpt_client.call_gpt_api(main_prompt)
                result.at[i, 'GPT Main Category'] = main_cat
                
                # Get sub-category
                sub_prompt = (
                    f"Determine the sub-category for this electrical part. "
                    f"CatalogNo: {catalog_no}, Manufacturer: {mfr_code}, "
                    f"Description: {desc}, Main Category: {main_cat}"
                )
                sub_cat = gpt_client.call_gpt_api(sub_prompt)
                result.at[i, 'GPT Sub-category'] = sub_cat
                
            except Exception as e:
                logger.error(f"GPT reevaluation failed for row {i}: {e}")
                result.at[i, 'GPT Main Category'] = 'Unknown'
                result.at[i, 'GPT Sub-category'] = 'Unknown'
                
        # Compute changes
        result['Main Category Changed'] = result['GPT Main Category'] != result[MAIN_CATEGORY_COL]
        result['Sub-category Changed'] = result['GPT Sub-category'] != result[SUB_CATEGORY_COL]
        
        # Log statistics
        main_changes = result['Main Category Changed'].sum()
        sub_changes = result['Sub-category Changed'].sum()
        logger.info(f"GPT reevaluation changes: {main_changes} main categories, {sub_changes} sub-categories")
        
        return result
        
    def save_models(self, output_dir: Path) -> None:
        """Save trained models to files.
        
        Args:
            output_dir: Directory to save models.
            
        Raises:
            ValueError: If model is not fitted or saving fails.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        try:
            output_dir.mkdir(exist_ok=True)
            
            # Save main category model and encoder
            joblib.dump(self.main_category_model, output_dir / 'main_category_model.joblib')
            joblib.dump(self.main_label_encoder, output_dir / 'main_label_encoder.joblib')
            
            # Save feature processing models
            joblib.dump(self.tfidf, output_dir / 'tfidf.joblib')
            joblib.dump(self.svd, output_dir / 'svd.joblib')
            joblib.dump(self.scaler, output_dir / 'scaler.joblib')
            
            # Save sub-category models
            for main_cat, model in self.sub_category_models.items():
                safe_name = "".join(c if c.isalnum() else "_" for c in main_cat)
                joblib.dump(model, output_dir / f'sub_model_{safe_name}.joblib')
                joblib.dump(self.sub_label_encoders[main_cat], output_dir / f'sub_encoder_{safe_name}.joblib')
            
            # Save KMeans model
            joblib.dump(self.kmeans, output_dir / 'kmeans.joblib')
            
            logger.info(f"Saved models to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
            raise ValueError(f"Failed to save models: {str(e)}")
        
    def load_models(self, input_dir: Path) -> None:
        """Load trained models from files.
        
        Args:
            input_dir: Directory containing model files.
            
        Raises:
            ValueError: If loading fails.
        """
        try:
            # Load main category model and encoder
            self.main_category_model = joblib.load(input_dir / 'main_category_model.joblib')
            self.main_label_encoder = joblib.load(input_dir / 'main_label_encoder.joblib')
            
            # Load feature processing models
            self.tfidf = joblib.load(input_dir / 'tfidf.joblib')
            self.svd = joblib.load(input_dir / 'svd.joblib')
            self.scaler = joblib.load(input_dir / 'scaler.joblib')
            
            # Load sub-category models
            self.sub_category_models = {}
            self.sub_label_encoders = {}
            for main_cat in self.main_label_encoder.classes_:
                safe_name = "".join(c if c.isalnum() else "_" for c in main_cat)
                try:
                    model_path = input_dir / f'sub_model_{safe_name}.joblib'
                    encoder_path = input_dir / f'sub_encoder_{safe_name}.joblib'
                    if model_path.exists() and encoder_path.exists():
                        self.sub_category_models[main_cat] = joblib.load(model_path)
                        self.sub_label_encoders[main_cat] = joblib.load(encoder_path)
                except Exception as e:
                    logger.warning(f"Could not load sub-category model for {main_cat}: {e}")
                    
            # Load KMeans model
            self.kmeans = joblib.load(input_dir / 'kmeans.joblib')
            
            self.is_fitted = True
            logger.info(f"Loaded models from {input_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise ValueError(f"Failed to load models: {str(e)}") 