"""
Module for GPT-based description enrichment and categorization.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inventory_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPTProcessor:
    """Class for processing item descriptions using GPT."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize the GPT processor.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
            model_name: GPT model name. If None, will try to load from environment.
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
            
        self.model_name = model_name or os.getenv('OPENAI_MODEL')
        if not self.model_name:
            raise ValueError("OpenAI model name not found. Please set OPENAI_MODEL in .env file.")
            
        self.client = OpenAI(api_key=self.api_key)
        self.max_tokens = 35
        self.top_p = 0.05
        self.temperature = 0.1
        
        # Initialize clustering attributes
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.kmeans = KMeans(n_clusters=50, random_state=42)  # Adjust n_clusters based on data size
        self.cluster_centers = None
        self.item_clusters = {}
        self.cluster_categories = defaultdict(lambda: {'main': None, 'sub': None})
        
        logger.info(f"Initialized GPTProcessor with model: {self.model_name}")

    def compute_clusters(self, data: pd.DataFrame) -> None:
        """Compute clusters for items using K-means."""
        try:
            # Prepare text data - use original descriptions for initial clustering
            descriptions = data['Description'].fillna("")
            catalog_nos = data['CatalogNo'].fillna("")
            
            # Combine descriptions and catalog numbers for better clustering
            combined_text = descriptions + " " + catalog_nos
            
            # Vectorize text
            X = self.vectorizer.fit_transform(combined_text)
            
            # Fit K-means
            self.kmeans.fit(X)
            self.cluster_centers = self.kmeans.cluster_centers_
            
            # Store cluster assignments
            for idx, (catalog_no, cluster_id) in enumerate(zip(catalog_nos, self.kmeans.labels_)):
                self.item_clusters[catalog_no] = cluster_id
                
                # Only store categories if they exist and are not empty
                if 'Main Category' in data.columns and 'Sub Category' in data.columns:
                    row = data[data['CatalogNo'] == catalog_no].iloc[0]
                    main_category = row['Main Category']
                    sub_category = row['Sub Category']
                    if pd.notna(main_category) and pd.notna(sub_category) and main_category and sub_category:
                        self.cluster_categories[cluster_id]['main'] = main_category
                        self.cluster_categories[cluster_id]['sub'] = sub_category
            
            logger.info(f"Computed {self.kmeans.n_clusters} clusters for {len(data)} items")
            
        except Exception as e:
            logger.error(f"Error computing clusters: {e}")
            raise

    def find_similar_items(self, catalog_no: str, data: pd.DataFrame, n_similar: int = 5) -> List[str]:
        """Find similar items using cluster information and cosine similarity."""
        try:
            if catalog_no not in self.item_clusters:
                return []
                
            cluster_id = self.item_clusters[catalog_no]
            cluster_items = [item for item, c_id in self.item_clusters.items() 
                           if c_id == cluster_id and item != catalog_no]
            
            if not cluster_items:
                return []
            
            # Get vector for current item
            item_idx = data[data['CatalogNo'] == catalog_no].index[0]
            item_desc = data.at[item_idx, 'Description']  # Use original description
            
            # Skip if description is NaN or empty
            if pd.isna(item_desc) or not item_desc.strip():
                return []
                
            item_vector = self.vectorizer.transform([item_desc])
            
            # Get vectors for cluster items
            cluster_data = data[data['CatalogNo'].isin(cluster_items)]
            valid_descs = cluster_data['Description'].dropna()  # Use original descriptions
            
            if valid_descs.empty:
                return []
                
            cluster_vectors = self.vectorizer.transform(valid_descs)
            
            # Compute similarities
            similarities = cosine_similarity(item_vector, cluster_vectors)[0]
            
            # Get top n similar items
            similar_indices = np.argsort(similarities)[-n_similar:]
            similar_items = [cluster_items[i] for i in similar_indices]
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []

    def enrich_description(self, description: str, catalog_no: str, mfr_code: str) -> str:
        """Enrich an item description using GPT with exponential backoff."""
        max_retries = 5
        base_delay = 2
        max_delay = 60
        
        for attempt in range(max_retries):
            try:
                prompt = f"Create a enriched part description in plain english (10 words or less). CatalogNo: {catalog_no}, Manufacturer: {mfr_code}, Abbreviated Description: {description}."
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional electrician and expert in electrical supply parts. Your expertise includes knowledge of the parts use cases and functionality."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Error enriching description after {max_retries} attempts: {e}")
                    return description

    def determine_category(self, catalog_no: str, mfr_code: str, enriched_description: str, category_type: str, main_category: str = '') -> str:
        """Determine category using GPT with exponential backoff."""
        max_retries = 5  # Increased from 3 to 5
        base_delay = 2  # Base delay in seconds
        max_delay = 60  # Maximum delay in seconds
        
        for attempt in range(max_retries):
            try:
                if category_type == 'main':
                    prompt = f"Determine the main category of {catalog_no} using description: '{enriched_description}' and manufacturer: {mfr_code}. Format output as: 'Main Category: (main_category)'"
                else:
                    prompt = f"Determine the sub-category of {catalog_no} more specific than {main_category} using description: '{enriched_description}' and manufacturer: {mfr_code}. Format output as: 'Sub Category: (sub_category)'"
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional electrician and expert in electrical supply parts. Your expertise includes knowledge of the parts use cases and functionality."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    temperature=self.temperature
                )
                
                result = response.choices[0].message.content.strip()
                if category_type == 'main':
                    category = result.replace('Main Category:', '').strip()
                else:
                    category = result.replace('Sub Category:', '').strip()
                
                return category.capitalize()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Error determining category after {max_retries} attempts: {e}")
                    return "Unknown"

    def assign_category(self, data: pd.DataFrame, idx: int, row: pd.Series) -> Tuple[str, str]:
        """Assign categories using clustering or GPT."""
        catalog_no = row['CatalogNo']
        mfr_code = row['MfrCode']
        enriched_description = data.at[idx, 'Enriched Description']

        # Try to find similar items in the same cluster
        similar_items = self.find_similar_items(catalog_no, data)
        
        # Check if any similar items have categories
        for similar_catalog_no in similar_items:
            similar_idx = data[data['CatalogNo'] == similar_catalog_no].index[0]
            main_category = data.at[similar_idx, 'Main Category']
            sub_category = data.at[similar_idx, 'Sub Category']
            
            if main_category and sub_category:
                # Update cluster categories
                cluster_id = self.item_clusters[catalog_no]
                self.cluster_categories[cluster_id]['main'] = main_category
                self.cluster_categories[cluster_id]['sub'] = sub_category
                return main_category, sub_category

        # If no similar items with categories found, use GPT
        main_category = self.determine_category(catalog_no, mfr_code, enriched_description, 'main')
        sub_category = self.determine_category(catalog_no, mfr_code, enriched_description, 'sub', main_category)
        
        # Update cluster categories
        cluster_id = self.item_clusters[catalog_no]
        self.cluster_categories[cluster_id]['main'] = main_category
        self.cluster_categories[cluster_id]['sub'] = sub_category
        
        return main_category, sub_category

    def process_batch(self, data: pd.DataFrame, batch_size: int = 10) -> pd.DataFrame:
        """Process a batch of items with periodic saving."""
        try:
            # Add new columns if they don't exist
            if 'Enriched Description' not in data.columns:
                data['Enriched Description'] = ''
            if 'Main Category' not in data.columns:
                data['Main Category'] = ''
            if 'Sub Category' not in data.columns:
                data['Sub Category'] = ''
            if 'GPT Processed' not in data.columns:
                data['GPT Processed'] = False

            # Compute clusters for the entire dataset
            logger.info("Computing clusters")
            self.compute_clusters(data)

            # Calculate total items to process
            unprocessed_mask = ~data['GPT Processed']
            total_items = unprocessed_mask.sum()
            items_processed = 0
            save_interval = max(1, total_items // 10)  # Save every 10% of items
            
            logger.info(f"Starting to process {total_items} items in batches of {batch_size}")
            
            # Create progress bar for batches
            with tqdm(total=total_items, desc="Processing items", unit="item") as pbar:
                # Process in batches
                for i in range(0, len(data), batch_size):
                    batch = data.iloc[i:i+batch_size]
                    batch_unprocessed = batch[~batch['GPT Processed']]
                    
                    if len(batch_unprocessed) == 0:
                        continue
                        
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
                    
                    for idx, row in batch_unprocessed.iterrows():
                        try:
                            # Enrich description
                            logger.debug(f"Enriching description for item {idx}")
                            data.at[idx, 'Enriched Description'] = self.enrich_description(
                                row['Description'],
                                row['CatalogNo'],
                                row['MfrCode']
                            )
                            
                            # Assign categories using clustering or GPT
                            logger.debug(f"Assigning categories for item {idx}")
                            main_category, sub_category = self.assign_category(data, idx, row)
                            
                            data.at[idx, 'Main Category'] = main_category
                            data.at[idx, 'Sub Category'] = sub_category
                            data.at[idx, 'GPT Processed'] = True
                            
                            # Update progress
                            items_processed += 1
                            pbar.update(1)
                            pbar.set_postfix({
                                'Main Category': main_category,
                                'Sub Category': sub_category,
                                'Cluster': self.item_clusters.get(row['CatalogNo'], -1)
                            })
                            
                            # Save progress every 10% of items
                            if items_processed % save_interval == 0:
                                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                progress_file = f'data/output/progress_{timestamp}.xlsx'
                                data.to_excel(progress_file, index=False)
                                logger.info(f"Saved progress after processing {items_processed} items to {progress_file}")
                                
                        except Exception as e:
                            logger.error(f"Error processing item {idx}: {e}")
                            # Save progress even if an item fails
                            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                            progress_file = f'data/output/progress_{timestamp}.xlsx'
                            data.to_excel(progress_file, index=False)
                            logger.info(f"Saved progress after error to {progress_file}")
                            continue
                
                    # Add delay to avoid rate limits
                    time.sleep(1)
                
            logger.info("Processing completed successfully")
            return data
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Save progress even if the entire batch fails
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            progress_file = f'data/output/progress_{timestamp}.xlsx'
            data.to_excel(progress_file, index=False)
            logger.info(f"Saved progress after batch error to {progress_file}")
            raise 