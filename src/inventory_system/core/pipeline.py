"""
Main pipeline module for the inventory system.
Implements the core processing logic for inventory categorization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from Levenshtein import distance
from joblib import dump, load

from inventory_system.config import SystemConfig
from inventory_system.services.description_enricher import DescriptionEnricher
from inventory_system.utils.metrics import calculate_metrics
from inventory_system.utils.monitoring import log_metrics

logger = logging.getLogger(__name__)

class InventoryPipeline:
    """Implements the core inventory processing pipeline."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.description_enricher = DescriptionEnricher(config.gpt)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize feature extractors
        self.tfidf = TfidfVectorizer(
            max_features=config.model.max_features,
            ngram_range=config.model.ngram_range
        )
        self.sentence_transformer = SentenceTransformer(config.similarity.text_model_name)
        
        # Initialize caches
        self.similarity_cache = {}
        self.embeddings_cache = {}
        
    def _initialize_models(self) -> None:
        """Initialize the categorization and verification models."""
        try:
            # Initialize categorization model
            self.categorization_model = self.config.get_model("categorization")
            
            # Initialize verification model
            self.verification_model = self.config.get_model("verification")
            
            # Initialize feature extractors
            self.feature_extractors = {
                "text": self.config.get_feature_extractor("text"),
                "numeric": self.config.get_feature_extractor("numeric"),
                "categorical": self.config.get_feature_extractor("categorical")
            }
            
            logger.info("Models and feature extractors initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            bool: True if validation passes
        """
        required_columns = ['CatalogNo', 'Description']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False
            
        if data.empty:
            logger.error("Input data is empty")
            return False
            
        return True
        
    def enrich_descriptions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich item descriptions using GPT.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with enriched descriptions
        """
        logger.info("Starting description enrichment...")
        
        # Check for existing enriched descriptions
        checkpoint_path = Path(self.config.checkpoints_dir) / "enriched_descriptions.csv"
        if checkpoint_path.exists():
            logger.info("Loading existing enriched descriptions...")
            return pd.read_csv(checkpoint_path)
        
        # Process in batches
        batch_size = self.config.gpt.batch_size
        enriched_data = data.copy()
        processed_items = set()
        
        # Load progress if exists
        progress_path = Path(self.config.checkpoints_dir) / "enrichment_progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                processed_items = set(progress['processed_items'])
                logger.info(f"Resuming from {len(processed_items)} processed items")
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i+batch_size]
            
            for idx, item in batch.iterrows():
                try:
                    # Skip already processed items
                    if item['CatalogNo'] in processed_items:
                        continue
                    
                    # Enrich description
                    enriched_desc = self.description_enricher.enrich(item['Description'])
                    enriched_data.at[idx, 'Enriched Description'] = enriched_desc
                    
                    # Mark as processed
                    processed_items.add(item['CatalogNo'])
                    
                    # Save progress every 100 items
                    if len(processed_items) % 100 == 0:
                        with open(progress_path, 'w') as f:
                            json.dump({'processed_items': list(processed_items)}, f)
                        enriched_data.to_csv(checkpoint_path, index=False)
                        logger.info(f"Saved progress: {len(processed_items)} items processed")
                        
                except Exception as e:
                    logger.error(f"Error enriching item {item['CatalogNo']}: {str(e)}")
                    enriched_data.at[idx, 'Enrichment Error'] = str(e)
                    continue
            
            # Save batch progress
            enriched_data.to_csv(checkpoint_path, index=False)
            
        logger.info("Description enrichment completed")
        return enriched_data
        
    def initial_categorization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial categorization using GPT.
        
        Args:
            data: DataFrame with enriched descriptions
            
        Returns:
            pd.DataFrame: DataFrame with initial categories
        """
        logger.info("Starting initial categorization...")
        
        # Check for existing categorizations
        checkpoint_path = Path(self.config.checkpoints_dir) / "initial_categories.csv"
        if checkpoint_path.exists():
            logger.info("Loading existing initial categories...")
            return pd.read_csv(checkpoint_path)
        
        # Process in batches
        batch_size = self.config.gpt.batch_size
        categorized_data = data.copy()
        processed_items = set()
        
        # Load progress if exists
        progress_path = Path(self.config.checkpoints_dir) / "categorization_progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                processed_items = set(progress['processed_items'])
                logger.info(f"Resuming from {len(processed_items)} processed items")
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i+batch_size]
            
            for idx, item in batch.iterrows():
                try:
                    # Skip already processed items
                    if item['CatalogNo'] in processed_items:
                        continue
                    
                    # Skip items with enrichment errors
                    if 'Enrichment Error' in item and pd.notna(item['Enrichment Error']):
                        categorized_data.at[idx, 'Categorization Error'] = "Skipped due to enrichment error"
                        continue
                    
                    # Categorize item
                    categories = self.description_enricher.categorize(item['Enriched Description'])
                    categorized_data.at[idx, 'Main Category'] = categories['main']
                    categorized_data.at[idx, 'Sub Category'] = categories['sub']
                    
                    # Mark as processed
                    processed_items.add(item['CatalogNo'])
                    
                    # Save progress every 100 items
                    if len(processed_items) % 100 == 0:
                        with open(progress_path, 'w') as f:
                            json.dump({'processed_items': list(processed_items)}, f)
                        categorized_data.to_csv(checkpoint_path, index=False)
                        logger.info(f"Saved progress: {len(processed_items)} items categorized")
                        
                except Exception as e:
                    logger.error(f"Error categorizing item {item['CatalogNo']}: {str(e)}")
                    categorized_data.at[idx, 'Categorization Error'] = str(e)
                    continue
            
            # Save batch progress
            categorized_data.to_csv(checkpoint_path, index=False)
            
        logger.info("Initial categorization completed")
        return categorized_data
        
    def _calculate_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two items.
        
        Args:
            item1: First item
            item2: Second item
            
        Returns:
            float: Similarity score
        """
        # Check cache
        cache_key = f"{item1['CatalogNo']}_{item2['CatalogNo']}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Calculate CatalogNo similarity
        catalog_similarity = 1 - (distance(item1['CatalogNo'], item2['CatalogNo']) / 
                                max(len(item1['CatalogNo']), len(item2['CatalogNo'])))
        
        # Calculate description similarity
        if item1['CatalogNo'] not in self.embeddings_cache:
            self.embeddings_cache[item1['CatalogNo']] = self.sentence_transformer.encode(
                item1['Enriched Description']
            )
        if item2['CatalogNo'] not in self.embeddings_cache:
            self.embeddings_cache[item2['CatalogNo']] = self.sentence_transformer.encode(
                item2['Enriched Description']
            )
            
        desc_similarity = np.dot(
            self.embeddings_cache[item1['CatalogNo']],
            self.embeddings_cache[item2['CatalogNo']]
        )
        
        # Combine similarities with weights
        similarity = (
            self.config.similarity.catalog_no_weight * catalog_similarity +
            self.config.similarity.description_weight * desc_similarity
        )
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        return similarity
        
    def iterative_categorization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform iterative categorization using ML model.
        
        Args:
            data: DataFrame with initial categories
            
        Returns:
            pd.DataFrame: DataFrame with final categories
        """
        logger.info("Starting iterative categorization...")
        
        # Check for existing progress
        checkpoint_path = Path(self.config.checkpoints_dir) / "iterative_categories.csv"
        if checkpoint_path.exists():
            logger.info("Loading existing iterative categories...")
            return pd.read_csv(checkpoint_path)
        
        categorized_data = data.copy()
        processed_items = set()
        
        # Load progress if exists
        progress_path = Path(self.config.checkpoints_dir) / "iterative_progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                processed_items = set(progress['processed_items'])
                logger.info(f"Resuming from {len(processed_items)} processed items")
        
        # Initialize model performance tracking
        performance_history = []
        
        # Process items in batches
        batch_size = self.config.model.batch_size
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i+batch_size]
            
            for idx, item in batch.iterrows():
                try:
                    if item['CatalogNo'] in processed_items:
                        continue
                    
                    # Skip items with previous errors
                    if 'Categorization Error' in item and pd.notna(item['Categorization Error']):
                        categorized_data.at[idx, 'Iterative Error'] = "Skipped due to previous error"
                        continue
                    
                    # Find similar items
                    similar_items = []
                    for other_idx, other_item in categorized_data.iterrows():
                        if other_item['CatalogNo'] in processed_items:
                            similarity = self._calculate_similarity(item, other_item)
                            similar_items.append((similarity, other_item))
                    
                    if similar_items:
                        # Sort by similarity
                        similar_items.sort(reverse=True, key=lambda x: x[0])
                        
                        # Get features for ML prediction
                        features = []
                        for similarity, similar_item in similar_items[:self.config.model.n_neighbors]:
                            features.extend([
                                similarity,
                                self._calculate_similarity(item, similar_item)
                            ])
                        
                        # Pad features if needed
                        while len(features) < self.config.model.n_neighbors * 2:
                            features.append(0)
                        
                        # Predict using ensemble
                        predictions = []
                        confidences = []
                        for model in self.calibrated_models.values():
                            if hasattr(model, 'predict_proba'):
                                pred = model.predict([features])[0]
                                conf = model.predict_proba([features])[0].max()
                                predictions.append(pred)
                                confidences.append(conf)
                        
                        # Use majority vote with confidence
                        if confidences and max(confidences) > self.config.model.confidence_threshold:
                            main_category = max(set(predictions), key=predictions.count)
                            categorized_data.at[idx, 'Main Category'] = main_category
                            categorized_data.at[idx, 'Prediction Confidence'] = max(confidences)
                        else:
                            # Use GPT for categorization
                            result = self.description_enricher.categorize_item(item)
                            categorized_data.at[idx, 'Main Category'] = result['Main Category']
                            categorized_data.at[idx, 'Sub Category'] = result['Sub Category']
                            
                            # Update model with new data
                            if self.config.model.online_learning:
                                self._update_model(item, result)
                                
                                # Track model performance
                                performance = self._evaluate_model_performance()
                                performance_history.append(performance)
                                
                                # Save model if performance improves
                                if self._should_save_model(performance_history):
                                    self._save_model()
                                    logger.info("Saved improved model")
                    
                    # Mark as processed
                    processed_items.add(item['CatalogNo'])
                    
                    # Save progress every 100 items
                    if len(processed_items) % 100 == 0:
                        with open(progress_path, 'w') as f:
                            json.dump({
                                'processed_items': list(processed_items),
                                'performance_history': performance_history
                            }, f)
                        categorized_data.to_csv(checkpoint_path, index=False)
                        logger.info(f"Saved progress: {len(processed_items)} items processed")
                        
                except Exception as e:
                    logger.error(f"Error processing item {item['CatalogNo']}: {str(e)}")
                    categorized_data.at[idx, 'Iterative Error'] = str(e)
                    continue
            
            # Save batch progress
            categorized_data.to_csv(checkpoint_path, index=False)
            
        logger.info("Iterative categorization completed")
        return categorized_data
        
    def _evaluate_model_performance(self) -> Dict[str, float]:
        """Evaluate current model performance."""
        # Implement performance evaluation logic
        # This could include accuracy, precision, recall, etc.
        return {
            'accuracy': 0.0,  # Placeholder
            'precision': 0.0,  # Placeholder
            'recall': 0.0  # Placeholder
        }
        
    def _should_save_model(self, performance_history: List[Dict[str, float]], 
                         window_size: int = 5) -> bool:
        """Determine if model should be saved based on performance history."""
        if len(performance_history) < window_size:
            return False
            
        recent_performance = performance_history[-window_size:]
        avg_accuracy = sum(p['accuracy'] for p in recent_performance) / window_size
        
        # Save if recent performance is better than historical average
        historical_avg = sum(p['accuracy'] for p in performance_history[:-window_size]) / \
                        (len(performance_history) - window_size)
                        
        return avg_accuracy > historical_avg
        
    def _update_model(self, item: Dict[str, Any], result: Dict[str, Any]):
        """
        Update ML model with new data.
        
        Args:
            item: New item
            result: Categorization result
        """
        # Get features for the new item
        features = []
        for other_item in self.embeddings_cache:
            similarity = self._calculate_similarity(item, {'CatalogNo': other_item})
            features.append(similarity)
        
        # Pad features if needed
        while len(features) < self.config.model.n_neighbors:
            features.append(0)
        
        # Update each model
        for model in self.calibrated_models.values():
            if hasattr(model, 'partial_fit'):
                model.partial_fit(
                    [features],
                    [result['Main Category']],
                    classes=np.unique(list(self.embeddings_cache.keys()))
                )
                
    def _save_model(self):
        """Save the current state of the ML model."""
        model_path = Path(self.config.models_dir) / "ensemble_model.joblib"
        dump(self.calibrated_models, model_path)
        
    def align_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align categories using the trained ML model.
        
        Args:
            data: DataFrame with categories
            
        Returns:
            pd.DataFrame: DataFrame with aligned categories
        """
        logger.info("Starting category alignment...")
        
        # Check for existing progress
        checkpoint_path = Path(self.config.checkpoints_dir) / "aligned_categories.csv"
        if checkpoint_path.exists():
            logger.info("Loading existing aligned categories...")
            return pd.read_csv(checkpoint_path)
        
        aligned_data = data.copy()
        processed_items = set()
        
        # Load progress if exists
        progress_path = Path(self.config.checkpoints_dir) / "alignment_progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                processed_items = set(progress['processed_items'])
                logger.info(f"Resuming from {len(processed_items)} processed items")
        
        # First pass alignment
        for idx, item in tqdm(aligned_data.iterrows()):
            try:
                if item['CatalogNo'] in processed_items:
                    continue
                
                # Skip items with previous errors
                if 'Iterative Error' in item and pd.notna(item['Iterative Error']):
                    aligned_data.at[idx, 'Alignment Error'] = "Skipped due to previous error"
                    continue
                
                # Get features
                features = []
                for other_item in self.embeddings_cache:
                    similarity = self._calculate_similarity(item, {'CatalogNo': other_item})
                    features.append(similarity)
                
                # Pad features if needed
                while len(features) < self.config.model.n_neighbors:
                    features.append(0)
                
                # Predict using ensemble
                predictions = []
                confidences = []
                for model in self.calibrated_models.values():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict([features])[0]
                        conf = model.predict_proba([features])[0].max()
                        predictions.append(pred)
                        confidences.append(conf)
                
                # Apply alignment if confidence is high
                if confidences and max(confidences) > self.config.model.confidence_threshold:
                    main_category = max(set(predictions), key=predictions.count)
                    if main_category != item['Main Category']:
                        aligned_data.at[idx, 'Main Category'] = main_category
                        aligned_data.at[idx, 'Alignment Confidence'] = max(confidences)
                        aligned_data.at[idx, 'Original Category'] = item['Main Category']
                
                # Mark as processed
                processed_items.add(item['CatalogNo'])
                
                # Save progress every 100 items
                if len(processed_items) % 100 == 0:
                    with open(progress_path, 'w') as f:
                        json.dump({'processed_items': list(processed_items)}, f)
                    aligned_data.to_csv(checkpoint_path, index=False)
                    logger.info(f"Saved progress: {len(processed_items)} items aligned")
                    
            except Exception as e:
                logger.error(f"Error aligning item {item['CatalogNo']}: {str(e)}")
                aligned_data.at[idx, 'Alignment Error'] = str(e)
                continue
        
        # Save progress
        aligned_data.to_csv(checkpoint_path, index=False)
        
        # Optional second pass
        if self.config.model.online_learning:
            logger.info("Performing second pass alignment...")
            for idx, item in tqdm(aligned_data.iterrows()):
                try:
                    # Skip items with alignment errors
                    if 'Alignment Error' in item and pd.notna(item['Alignment Error']):
                        continue
                    
                    # Update model with aligned data
                    self._update_model(item, {
                        'Main Category': item['Main Category'],
                        'Sub Category': item['Sub Category']
                    })
                    
                except Exception as e:
                    logger.error(f"Error in second pass for item {item['CatalogNo']}: {str(e)}")
                    continue
        
        logger.info("Category alignment completed")
        return aligned_data
        
    def export_results(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Export final results.
        
        Args:
            data: Final DataFrame
            output_path: Path to save results
        """
        logger.info(f"Exporting results to {output_path}")
        
        # Calculate metrics
        metrics = calculate_metrics(data)
        log_metrics(metrics)
        
        # Save results
        data.to_excel(output_path, index=False)
        
        # Save metrics
        metrics_path = Path(output_path).parent / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
    def test_pipeline(self, data: pd.DataFrame, fraction: float = 0.1) -> Dict[str, Any]:
        """
        Test the pipeline with a sample of data.
        
        Args:
            data: Input DataFrame
            fraction: Fraction of data to use for testing
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Running test pipeline...")
        
        # Sample data
        test_data = data.sample(frac=fraction)
        
        # Run through pipeline
        enriched_data = self.enrich_descriptions(test_data)
        initial_categories = self.initial_categorization(enriched_data)
        final_categories = self.iterative_categorization(initial_categories)
        aligned_categories = self.align_categories(final_categories)
        
        # Calculate metrics
        metrics = calculate_metrics(aligned_categories)
        
        return {
            'metrics': metrics,
            'sample_size': len(test_data),
            'processing_time': datetime.now() - datetime.now()  # TODO: Implement timing
        }
        
    def verify_categorization(self, item: Dict[str, Any], category: str) -> Tuple[bool, float]:
        """
        Verify the categorization of an item using the verification model.
        
        Args:
            item: Item data
            category: Assigned category
            
        Returns:
            Tuple of (is_correct, confidence)
        """
        try:
            # Extract features for verification
            features = self._extract_features(item)
            
            # Get verification result
            is_correct, confidence = self.verification_model.verify(
                features=features,
                category=category
            )
            
            return is_correct, confidence
        except Exception as e:
            logger.error(f"Error in verification: {str(e)}")
            return False, 0.0

    def _categorize_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Categorize items using the ensemble model with verification.
        
        Args:
            items: List of items to categorize
            
        Returns:
            List of categorized items
        """
        categorized_items = []
        
        for item in items:
            try:
                # Extract features
                features = self._extract_features(item)
                
                # Get initial categorization
                category, confidence = self.categorization_model.predict(features)
                
                # Verify the categorization
                is_correct, verification_confidence = self.verify_categorization(item, category)
                
                # If verification fails or confidence is low, mark for review
                if not is_correct or verification_confidence < self.config.verification_threshold:
                    item['needs_review'] = True
                    item['verification_confidence'] = verification_confidence
                else:
                    item['needs_review'] = False
                    item['verification_confidence'] = verification_confidence
                
                # Add categorization results
                item['category'] = category
                item['confidence'] = confidence
                
                categorized_items.append(item)
                
            except Exception as e:
                logger.error(f"Error categorizing item: {str(e)}")
                item['error'] = str(e)
                categorized_items.append(item)
        
        return categorized_items

    def _update_models(self, feedback_data: List[Dict[str, Any]]) -> None:
        """
        Update models based on feedback data.
        
        Args:
            feedback_data: List of items with feedback
        """
        try:
            # Prepare training data
            X = []
            y = []
            verification_data = []
            
            for item in feedback_data:
                features = self._extract_features(item)
                X.append(features)
                y.append(item['correct_category'])
                
                # Prepare verification data
                verification_data.append({
                    'features': features,
                    'category': item['category'],
                    'is_correct': item['category'] == item['correct_category']
                })
            
            # Update categorization model
            self.categorization_model.update(X, y)
            
            # Update verification model
            self.verification_model.update(verification_data)
            
            logger.info("Models updated successfully with feedback data")
            
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            raise 