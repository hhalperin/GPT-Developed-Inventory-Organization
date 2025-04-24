"""
Category classification module for the inventory system.
Handles item categorization using GPT and similarity analysis.
"""

import os
import time
import logging
from openai import OpenAI
from typing import Dict, Any, Tuple, List
import pandas as pd
from tqdm.auto import tqdm
from .category_predictor import CategoryPredictor
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import numpy as np
import multiprocessing as mp
import gc

# Suppress OpenAI client messages
logging.getLogger("openai").setLevel(logging.WARNING)

class CategoryClassifier:
    """Handles item categorization using GPT and similarity analysis."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the category classifier.
        
        Args:
            api_key: OpenAI API key. If None, uses key from environment variable.
            model: GPT model to use. If None, uses model from environment variable.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-nano-2025-04-14')
        self.client = OpenAI(api_key=self.api_key)
        self.max_tokens = 35
        self.top_p = 0.05
        self.temperature = 0.1
        
        # Initialize category predictor
        self.category_predictor = CategoryPredictor()
        
        self.existing_categories: Dict[str, List[Dict[str, Any]]] = {}
        self.category_change_dict: Dict[str, Dict[str, Any]] = {}
        self.timing_info: Dict[str, float] = {}
        self.categorization_stats: Dict[str, float] = {}
        self.api_calls_saved = 0
        self.total_items = 0
        
        logging.info("Category classifier initialized")

    def call_gpt_api(self, prompt: str) -> str:
        """
        Call the GPT API with a given prompt.
        
        Args:
            prompt: The prompt to send to GPT
            
        Returns:
            str: GPT's response
        """
        time.sleep(0.01)  # Rate limiting
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional electrician and expert in electrical supply parts. Provide short, plain-English responses."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API call error: {e}")
            return ""

    def determine_category_gpt(self, catalog_no: str, mfr_code: str, enriched_desc: str, 
                            category_type: str, main_category: str = "") -> str:
        """
        Determine category using GPT.
        
        Args:
            catalog_no: Item's catalog number
            mfr_code: Manufacturer code
            enriched_desc: Enriched description
            category_type: 'main' or 'sub'
            main_category: Main category for sub-category determination
            
        Returns:
            str: Determined category
        """
        if category_type == "main":
            prompt = (
                f"Determine the main category of {catalog_no} using description: '{enriched_desc}' "
                f"and manufacturer: {mfr_code}. Format output as: 'Main Category: (main_category)'"
            )
        else:
            prompt = (
                f"Determine the sub-category of {catalog_no} more specific than {main_category} "
                f"using description: '{enriched_desc}' and manufacturer: {mfr_code}. "
                f"Format output as: 'Sub Category: (sub_category)'"
            )

        # Add delay between API calls
        time.sleep(2)  # Base delay of 2 seconds between calls
        
        max_retries = 3
        base_delay = 5  # Base delay for exponential backoff
        current_delay = base_delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional electrician and expert in electrical supply parts. Provide short, plain-English responses."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to get GPT category after {max_retries} attempts: {str(e)}")
                    return "Unknown"  # Return a default category on final failure
                else:
                    logging.warning(f"GPT API attempt {attempt + 1} failed, retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff

    def _process_batch(self, batch_items, training_data_copy, similarity_analyzer):
        """
        Process a batch of items to assign categories.
        
        Args:
            batch_items: DataFrame containing items to process
            training_data_copy: Copy of training data
            similarity_analyzer: SimilarityAnalyzer instance
            
        Returns:
            List of results for each item in the batch
        """
        try:
            # Disable logging in worker processes
            logging.getLogger().handlers = []
            logging.getLogger().propagate = False
            
            # Convert DataFrames to more efficient formats for processing
            batch_items = batch_items.copy()
            training_data_copy = training_data_copy.copy()
            
            batch_results = []
            for idx, row in batch_items.iterrows():
                try:
                    # Skip items with empty descriptions
                    if not row['Enriched Description'].strip():
                        continue
                    
                    # Get similarity scores with all training data
                    current_item = pd.DataFrame([row])
                    
                    # Compute similarity scores
                    similarity_scores = similarity_analyzer.compute_similarity_scores(training_data_copy)
                    
                    if not similarity_scores or 'description_similarity' not in similarity_scores:
                        raise ValueError("No similarity scores available")
                        
                    # Get description similarity scores for current item
                    current_sku = row['CatalogNo']
                    desc_similarity = similarity_scores['description_similarity'].get(current_sku, {})
                    
                    if not desc_similarity:
                        raise ValueError("No description similarity scores available")
                    
                    # Get most similar item and its category
                    valid_similarities = {
                        sku: score for sku, score in desc_similarity.items() 
                        if sku != current_sku and training_data_copy.loc[training_data_copy['CatalogNo'] == sku, 'Main Category'].notna().any()
                    }
                    
                    if not valid_similarities:
                        raise ValueError("No valid similar items found")
                    
                    most_similar_sku = max(valid_similarities.items(), key=lambda x: x[1])[0]
                    similarity_score = valid_similarities[most_similar_sku]
                    
                    # Find the most similar item in training data
                    similar_items = training_data_copy[training_data_copy['CatalogNo'] == most_similar_sku]
                    if similar_items.empty:
                        raise ValueError("Most similar item not found in training data")
                        
                    most_similar_item = similar_items.iloc[0]
                    
                    # Predict if this item should have the same category
                    should_use_same_category, confidence = self.category_predictor.predict(
                        current_item,
                        similarity_analyzer,
                        return_format='simple'
                    )
                    
                    # Handle empty predictions
                    if not confidence or not should_use_same_category:
                        raise ValueError("Empty predictions from model")
                        
                    # Convert confidence to float if it's a list
                    confidence_score = float(confidence[0]) if isinstance(confidence, list) else float(confidence)
                    
                    if confidence_score >= self.category_predictor.confidence_threshold:
                        result = {
                            'idx': idx,
                            'main_category': most_similar_item['Main Category'],
                            'sub_category': most_similar_item['Sub Category'],
                            'confidence': confidence_score,
                            'method': 'similarity'
                        }
                    else:
                        # Use GPT for categorization
                        main_cat = self.determine_category_gpt(
                            catalog_no=row['CatalogNo'],
                            mfr_code=row['MfrCode'],
                            enriched_desc=row['Enriched Description'],
                            category_type='main'
                        )
                        
                        sub_cat = self.determine_category_gpt(
                            catalog_no=row['CatalogNo'],
                            mfr_code=row['MfrCode'],
                            enriched_desc=row['Enriched Description'],
                            category_type='sub',
                            main_category=main_cat
                        )
                        
                        result = {
                            'idx': idx,
                            'main_category': main_cat,
                            'sub_category': sub_cat,
                            'confidence': 1.0,
                            'method': 'gpt'
                        }
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    # Fall back to GPT if any step fails
                    try:
                        main_cat = self.determine_category_gpt(
                            catalog_no=row['CatalogNo'],
                            mfr_code=row['MfrCode'],
                            enriched_desc=row['Enriched Description'],
                            category_type='main'
                        )
                        
                        sub_cat = self.determine_category_gpt(
                            catalog_no=row['CatalogNo'],
                            mfr_code=row['MfrCode'],
                            enriched_desc=row['Enriched Description'],
                            category_type='sub',
                            main_category=main_cat
                        )
                        
                        result = {
                            'idx': idx,
                            'main_category': main_cat,
                            'sub_category': sub_cat,
                            'confidence': 1.0,
                            'method': 'gpt_fallback'
                        }
                        batch_results.append(result)
                    except Exception as gpt_error:
                        continue
                        
            return batch_results
            
        except Exception as e:
            logging.error(f"Error in batch processing: {str(e)}")
            return []
        finally:
            # Clean up resources
            del batch_items
            del training_data_copy
            gc.collect()

    def assign_category(self, data: pd.DataFrame, similarity_analyzer: Any = None) -> pd.DataFrame:
        """
        Assign categories to items using incremental learning based on similarity.
        
        Args:
            data: DataFrame containing item descriptions
            similarity_analyzer: Optional SimilarityAnalyzer instance
            
        Returns:
            DataFrame with assigned categories
        """
        try:
            # Initialize necessary columns if they don't exist
            if 'Main Category' not in data.columns:
                data['Main Category'] = None
            if 'Sub Category' not in data.columns:
                data['Sub Category'] = None
            if 'Category Confidence' not in data.columns:
                data['Category Confidence'] = None
                
            total_items = len(data)
            logging.info(f"Starting categorization of {total_items} items")
            
            # Initialize training data with all items (uncategorized)
            training_data = data.copy()
            
            # First 4 items: Always use GPT to build initial training set
            initial_items = data.iloc[:4]
            for idx, row in initial_items.iterrows():
                try:
                    # Skip items with empty descriptions
                    if not row['Enriched Description'].strip():
                        logging.warning(f"Skipping item {row['CatalogNo']} due to empty description")
                        continue
                    
                    # Get main category
                    main_cat = self.determine_category_gpt(
                        catalog_no=row['CatalogNo'],
                        mfr_code=row['MfrCode'],
                        enriched_desc=row['Enriched Description'],
                        category_type='main'
                    )
                    
                    # Get sub category
                    sub_cat = self.determine_category_gpt(
                        catalog_no=row['CatalogNo'],
                        mfr_code=row['MfrCode'],
                        enriched_desc=row['Enriched Description'],
                        category_type='sub',
                        main_category=main_cat
                    )
                    
                    data.at[idx, 'Main Category'] = main_cat
                    data.at[idx, 'Sub Category'] = sub_cat
                    data.at[idx, 'Category Confidence'] = 1.0
                    
                    # Update training data
                    training_data.at[idx, 'Main Category'] = main_cat
                    training_data.at[idx, 'Sub Category'] = sub_cat
                    training_data.at[idx, 'Category Confidence'] = 1.0
                    
                except Exception as e:
                    logging.error(f"Error in GPT categorization for item {idx}: {str(e)}")
                    continue
            
            # Train initial model
            self.category_predictor.train(training_data, similarity_analyzer)
            
            # Process remaining items in smaller batches
            uncategorized_items = data[data['Main Category'].isna()]
            if not uncategorized_items.empty:
                logging.info(f"Processing {len(uncategorized_items)} remaining items")
                
                # Use smaller batch size and fewer workers
                batch_size = 25  # Reduced from 50
                max_workers = min(4, mp.cpu_count())  # Limit to 4 workers
                
                # Split items into batches
                batches = [uncategorized_items.iloc[i:i+batch_size] 
                          for i in range(0, len(uncategorized_items), batch_size)]
                
                # Process batches sequentially with limited parallelization
                for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
                    try:
                        # Process each item in the batch
                        for idx, row in batch.iterrows():
                            try:
                                # Skip items with empty descriptions
                                if not row['Enriched Description'].strip():
                                    continue
                                
                                # Get similarity scores
                                current_item = pd.DataFrame([row])
                                similarity_scores = similarity_analyzer.compute_similarity_scores(training_data)
                                
                                if not similarity_scores or 'description_similarity' not in similarity_scores:
                                    raise ValueError("No similarity scores available")
                                    
                                # Get description similarity scores
                                current_sku = row['CatalogNo']
                                desc_similarity = similarity_scores['description_similarity'].get(current_sku, {})
                                
                                if not desc_similarity:
                                    raise ValueError("No description similarity scores available")
                                
                                # Get most similar item
                                valid_similarities = {
                                    sku: score for sku, score in desc_similarity.items() 
                                    if sku != current_sku and training_data.loc[training_data['CatalogNo'] == sku, 'Main Category'].notna().any()
                                }
                                
                                if not valid_similarities:
                                    raise ValueError("No valid similar items found")
                                
                                most_similar_sku = max(valid_similarities.items(), key=lambda x: x[1])[0]
                                similarity_score = valid_similarities[most_similar_sku]
                                
                                # Find the most similar item in training data
                                similar_items = training_data[training_data['CatalogNo'] == most_similar_sku]
                                if similar_items.empty:
                                    raise ValueError("Most similar item not found in training data")
                                    
                                most_similar_item = similar_items.iloc[0]
                                
                                # Predict category
                                should_use_same_category, confidence = self.category_predictor.predict(
                                    current_item,
                                    similarity_analyzer,
                                    return_format='simple'
                                )
                                
                                if not confidence or not should_use_same_category:
                                    raise ValueError("Empty predictions from model")
                                    
                                confidence_score = float(confidence[0]) if isinstance(confidence, list) else float(confidence)
                                
                                if confidence_score >= self.category_predictor.confidence_threshold:
                                    data.at[idx, 'Main Category'] = most_similar_item['Main Category']
                                    data.at[idx, 'Sub Category'] = most_similar_item['Sub Category']
                                    data.at[idx, 'Category Confidence'] = confidence_score
                                    
                                    # Update training data
                                    training_data.at[idx, 'Main Category'] = most_similar_item['Main Category']
                                    training_data.at[idx, 'Sub Category'] = most_similar_item['Sub Category']
                                    training_data.at[idx, 'Category Confidence'] = confidence_score
                                else:
                                    # Use GPT for categorization
                                    main_cat = self.determine_category_gpt(
                                        catalog_no=row['CatalogNo'],
                                        mfr_code=row['MfrCode'],
                                        enriched_desc=row['Enriched Description'],
                                        category_type='main'
                                    )
                                    
                                    sub_cat = self.determine_category_gpt(
                                        catalog_no=row['CatalogNo'],
                                        mfr_code=row['MfrCode'],
                                        enriched_desc=row['Enriched Description'],
                                        category_type='sub',
                                        main_category=main_cat
                                    )
                                    
                                    data.at[idx, 'Main Category'] = main_cat
                                    data.at[idx, 'Sub Category'] = sub_cat
                                    data.at[idx, 'Category Confidence'] = 1.0
                                    
                                    # Update training data
                                    training_data.at[idx, 'Main Category'] = main_cat
                                    training_data.at[idx, 'Sub Category'] = sub_cat
                                    training_data.at[idx, 'Category Confidence'] = 1.0
                                    
                            except Exception as e:
                                logging.error(f"Error processing item {idx}: {str(e)}")
                                continue
                                
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                        continue
                    
            logging.info("Categorization completed")
            return data
            
        except Exception as e:
            logging.error(f"Error in assign_category: {str(e)}")
            raise
        finally:
            # Clean up resources
            gc.collect()

    def reevaluate_categories(self, data: pd.DataFrame, confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Reevaluate categories based on confidence filtering.
        
        Args:
            data: DataFrame containing items to reevaluate
            confidence_threshold: Minimum confidence score to keep prediction
            
        Returns:
            DataFrame with updated categories
        """
        try:
            # Initialize predictor if not already done
            if not hasattr(self, 'predictor'):
                self.predictor = CategoryPredictor()
            
            # Get predictions and confidence scores
            main_pred, sub_pred, main_conf, sub_conf = self.predictor.predict(
                data, 
                return_format='full'
            )
            
            # Track changes
            changes = 0
            
            # Update categories based on confidence
            for idx in range(len(data)):
                if main_conf[idx] >= confidence_threshold:
                    # Update main category if confidence is high enough
                    if data.iloc[idx]['Main Category'] != main_pred[idx]:
                        data.at[idx, 'Main Category'] = main_pred[idx]
                        changes += 1
                        
                if sub_conf[idx] >= confidence_threshold:
                    # Update sub category if confidence is high enough
                    if data.iloc[idx]['Sub Category'] != sub_pred[idx]:
                        data.at[idx, 'Sub Category'] = sub_pred[idx]
                        changes += 1
                        
                # Update confidence scores
                data.at[idx, 'Category Confidence'] = max(main_conf[idx], sub_conf[idx])
                
            logging.info(f"Reevaluated {len(data)} items, made {changes} changes")
            return data
            
        except Exception as e:
            logging.error(f"Error during category reevaluation: {str(e)}")
            return data

    def _update_category_tracking(self, catalog_no: str, main_category: str, sub_category: str) -> None:
        """Update internal category tracking."""
        new_row = {
            "CatalogNo": catalog_no,
            "Main Category": main_category,
            "Sub-category": sub_category
        }
        
        if main_category not in self.existing_categories:
            self.existing_categories[main_category] = []
        self.existing_categories[main_category].append(new_row) 