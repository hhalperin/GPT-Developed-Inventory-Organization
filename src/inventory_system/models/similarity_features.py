"""
Advanced similarity feature extraction for inventory items.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
from sklearn.preprocessing import StandardScaler
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SimilarityConfig:
    """Configuration for similarity feature extraction."""
    text_model_name: str = "all-MiniLM-L6-v2"
    catalog_pattern: str = r"([A-Z]+)(\d+)([A-Z]*)"
    min_confidence: float = 0.7
    cache_size: int = 1000

class SimilarityFeatureExtractor(ABC):
    """Base class for similarity feature extractors."""
    
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract similarity features from data."""
        pass

class TextSimilarityExtractor(SimilarityFeatureExtractor):
    """Extracts text-based similarity features using sentence transformers."""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.model = SentenceTransformer(config.text_model_name)
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract text similarity features."""
        features = pd.DataFrame(index=data.index)
        
        # Get embeddings for descriptions
        descriptions = data['Enriched Description'].fillna('')
        embeddings = self._get_embeddings(descriptions)
        
        # Calculate pairwise similarities
        similarity_matrix = self._calculate_cosine_similarity(embeddings)
        
        # Extract features
        features['max_text_similarity'] = np.max(similarity_matrix, axis=1)
        features['mean_text_similarity'] = np.mean(similarity_matrix, axis=1)
        features['text_similarity_std'] = np.std(similarity_matrix, axis=1)
        
        return features
        
    def _get_embeddings(self, texts: pd.Series) -> np.ndarray:
        """Get embeddings for texts, using cache when possible."""
        embeddings = []
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                embedding = self.model.encode(text)
                self.cache[text] = embedding
                embeddings.append(embedding)
                
                # Maintain cache size
                if len(self.cache) > self.config.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                    
        return np.array(embeddings)
        
    def _calculate_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix."""
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norm
        return np.dot(normalized, normalized.T)

class CatalogSimilarityExtractor(SimilarityFeatureExtractor):
    """Extracts catalog number similarity features."""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.pattern = re.compile(config.catalog_pattern)
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract catalog number similarity features."""
        features = pd.DataFrame(index=data.index)
        
        # Parse catalog numbers
        catalog_numbers = data['CatalogNo'].astype(str)
        parsed_numbers = catalog_numbers.apply(self._parse_catalog_number)
        
        # Calculate similarity features
        features['prefix_similarity'] = self._calculate_prefix_similarity(parsed_numbers)
        features['number_similarity'] = self._calculate_number_similarity(parsed_numbers)
        features['suffix_similarity'] = self._calculate_suffix_similarity(parsed_numbers)
        
        return features
        
    def _parse_catalog_number(self, catalog_no: str) -> Tuple[str, int, str]:
        """Parse catalog number into prefix, number, and suffix."""
        match = self.pattern.match(catalog_no)
        if match:
            prefix, number, suffix = match.groups()
            return prefix, int(number), suffix
        return '', 0, ''
        
    def _calculate_prefix_similarity(self, parsed_numbers: pd.Series) -> pd.Series:
        """Calculate prefix similarity."""
        prefixes = parsed_numbers.apply(lambda x: x[0])
        return prefixes.apply(lambda x: sum(prefixes == x) / len(prefixes))
        
    def _calculate_number_similarity(self, parsed_numbers: pd.Series) -> pd.Series:
        """Calculate number similarity."""
        numbers = parsed_numbers.apply(lambda x: x[1])
        max_number = numbers.max()
        if max_number == 0:
            return pd.Series(0, index=numbers.index)
        # Calculate similarity for each number
        similarities = []
        for num in numbers:
            similarity = 1 - abs(numbers - num) / max_number
            similarities.append(similarity.mean())
        return pd.Series(similarities, index=numbers.index)
        
    def _calculate_suffix_similarity(self, parsed_numbers: pd.Series) -> pd.Series:
        """Calculate suffix similarity."""
        suffixes = parsed_numbers.apply(lambda x: x[2])
        return suffixes.apply(lambda x: sum(suffixes == x) / len(suffixes))

class DomainFeatureExtractor(SimilarityFeatureExtractor):
    """Extracts domain-specific similarity features."""
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract domain-specific similarity features."""
        features = pd.DataFrame(index=data.index)
        
        # Material similarity
        if 'Material' in data.columns:
            features['material_similarity'] = self._calculate_material_similarity(data['Material'])
            
        # Dimension similarity
        if all(col in data.columns for col in ['Length', 'Width', 'Height']):
            features['dimension_similarity'] = self._calculate_dimension_similarity(
                data[['Length', 'Width', 'Height']]
            )
            
        # Specification similarity
        if 'Specifications' in data.columns:
            features['spec_similarity'] = self._calculate_spec_similarity(data['Specifications'])
            
        return features
        
    def _calculate_material_similarity(self, materials: pd.Series) -> pd.Series:
        """Calculate material similarity."""
        return materials.apply(lambda x: sum(materials == x) / len(materials))
        
    def _calculate_dimension_similarity(self, dimensions: pd.DataFrame) -> pd.Series:
        """Calculate dimension similarity."""
        normalized = (dimensions - dimensions.mean()) / dimensions.std()
        return 1 - np.linalg.norm(normalized, axis=1) / np.max(np.linalg.norm(normalized, axis=1))
        
    def _calculate_spec_similarity(self, specs: pd.Series) -> pd.Series:
        """Calculate specification similarity."""
        return specs.apply(lambda x: sum(specs == x) / len(specs))

class SimilarityFeaturePipeline:
    """Pipeline for extracting all similarity features."""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.extractors = [
            TextSimilarityExtractor(config),
            CatalogSimilarityExtractor(config),
            DomainFeatureExtractor()
        ]
        self.scaler = StandardScaler()
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all similarity features."""
        features_list = []
        
        for extractor in self.extractors:
            try:
                features = extractor.extract_features(data)
                features_list.append(features)
            except Exception as e:
                logging.error(f"Error in {extractor.__class__.__name__}: {e}")
                
        # Combine all features
        combined_features = pd.concat(features_list, axis=1)
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(combined_features),
            index=combined_features.index,
            columns=combined_features.columns
        )
        
        return scaled_features 