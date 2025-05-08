"""Similarity scoring module for inventory analysis."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CATALOG_NUMBER_COL,
    DESCRIPTION_COL,
    ENRICHED_DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

class SimilarityScorer:
    """Computes similarity between inventory items."""
    
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        """Initialize similarity scorer.
        
        Args:
            threshold: Similarity threshold for considering items similar.
        """
        self.threshold = threshold
        self.sku_similarity_matrix = None
        self.desc_similarity_matrix = None
        self._sku_to_index = None
        self.sku_similarity_dicts = {}  # Per-SKU similarity dictionaries
        self.desc_similarity_dicts = {}  # Per-SKU description similarity dictionaries
        
    def compute_sku_similarity(self, data: pd.DataFrame) -> np.ndarray:
        """Compute SKU similarity matrix using RapidFuzz.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            SKU similarity matrix.
        """
        n_items = len(data)
        similarity_matrix = np.zeros((n_items, n_items))
        self._sku_to_index = {sku: i for i, sku in enumerate(data[CATALOG_NUMBER_COL])}
        
        for i in range(n_items):
            for j in range(i, n_items):
                # SKU similarity using fuzzy matching
                sku_sim = fuzz.ratio(
                    str(data.iloc[i][CATALOG_NUMBER_COL]),
                    str(data.iloc[j][CATALOG_NUMBER_COL])
                ) / 100.0
                
                # Make matrix symmetric
                similarity_matrix[i, j] = sku_sim
                similarity_matrix[j, i] = sku_sim
                
                # Store in per-SKU dictionaries
                sku1 = data.iloc[i][CATALOG_NUMBER_COL]
                sku2 = data.iloc[j][CATALOG_NUMBER_COL]
                if sku1 not in self.sku_similarity_dicts:
                    self.sku_similarity_dicts[sku1] = {}
                if sku2 not in self.sku_similarity_dicts:
                    self.sku_similarity_dicts[sku2] = {}
                self.sku_similarity_dicts[sku1][sku2] = sku_sim
                self.sku_similarity_dicts[sku2][sku1] = sku_sim
                
        self.sku_similarity_matrix = similarity_matrix
        return similarity_matrix
        
    def compute_description_similarity(self, data: pd.DataFrame) -> np.ndarray:
        """Compute description similarity matrix using TF-IDF + cosine similarity.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            Description similarity matrix.
        """
        # Use enriched descriptions if available, fall back to regular descriptions
        desc_col = ENRICHED_DESCRIPTION_COL if ENRICHED_DESCRIPTION_COL in data.columns else DESCRIPTION_COL
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data[desc_col].fillna(""))
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Store in per-SKU dictionaries
        for i, sku1 in enumerate(data[CATALOG_NUMBER_COL]):
            if sku1 not in self.desc_similarity_dicts:
                self.desc_similarity_dicts[sku1] = {}
            for j, sku2 in enumerate(data[CATALOG_NUMBER_COL]):
                self.desc_similarity_dicts[sku1][sku2] = float(similarity_matrix[i, j])
                
        self.desc_similarity_matrix = similarity_matrix
        return similarity_matrix
        
    def compute_similarity_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Compute combined similarity matrix.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            Combined similarity matrix.
        """
        # Compute SKU and description similarities
        sku_sim = self.compute_sku_similarity(data)
        desc_sim = self.compute_description_similarity(data)
        
        # Combine similarities (weighted average)
        combined_sim = 0.4 * sku_sim + 0.6 * desc_sim
        
        # Update DataFrame with similarity information
        self._update_dataframe_similarities(data)
        
        return combined_sim
        
    def _update_dataframe_similarities(self, data: pd.DataFrame) -> None:
        """Update DataFrame with similarity information."""
        # Initialize similarity columns if they don't exist
        for col in ['Highest SKU Similarity', 'Most Similar SKU', 
                   'Highest Description Similarity', 'Most Similar Description SKU',
                   'Average Similarity', 'High Similarity Average']:
            if col not in data.columns:
                data[col] = None
                
        # Update per-SKU statistics
        for idx, row in data.iterrows():
            sku = row[CATALOG_NUMBER_COL]
            
            # SKU similarities
            sku_sims = self.sku_similarity_dicts[sku]
            sku_sims_no_self = {k: v for k, v in sku_sims.items() if k != sku}
            if sku_sims_no_self:
                data.at[idx, 'Highest SKU Similarity'] = max(sku_sims_no_self.values())
                data.at[idx, 'Most Similar SKU'] = max(sku_sims_no_self.items(), key=lambda x: x[1])[0]
                
            # Description similarities
            desc_sims = self.desc_similarity_dicts[sku]
            desc_sims_no_self = {k: v for k, v in desc_sims.items() if k != sku}
            if desc_sims_no_self:
                data.at[idx, 'Highest Description Similarity'] = max(desc_sims_no_self.values())
                data.at[idx, 'Most Similar Description SKU'] = max(desc_sims_no_self.items(), key=lambda x: x[1])[0]
                
            # Average similarities
            all_sims = list(sku_sims_no_self.values())
            high_sims = [s for s in all_sims if s >= self.threshold]
            data.at[idx, 'Average Similarity'] = np.mean(all_sims) if all_sims else None
            data.at[idx, 'High Similarity Average'] = np.mean(high_sims) if high_sims else None
            
    def find_similar_items(self, data: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find pairs of similar items.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            List of tuples containing (item1, item2, similarity).
        """
        if self.sku_similarity_matrix is None or self.desc_similarity_matrix is None:
            self.compute_similarity_matrix(data)
            
        similar_pairs = []
        n_items = len(data)
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                sku_sim = self.sku_similarity_matrix[i, j]
                desc_sim = self.desc_similarity_matrix[i, j]
                combined_sim = 0.4 * sku_sim + 0.6 * desc_sim
                
                if combined_sim >= self.threshold:
                    similar_pairs.append((
                        data.iloc[i][CATALOG_NUMBER_COL],
                        data.iloc[j][CATALOG_NUMBER_COL],
                        float(combined_sim)
                    ))
                    
        return similar_pairs
        
    def build_similarity_graph(self, data: pd.DataFrame) -> nx.Graph:
        """Build graph of similar items.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            NetworkX graph.
        """
        if self.sku_similarity_matrix is None or self.desc_similarity_matrix is None:
            self.compute_similarity_matrix(data)
            
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, row in data.iterrows():
            G.add_node(
                row[CATALOG_NUMBER_COL],
                description=row[DESCRIPTION_COL],
                category=row[MAIN_CATEGORY_COL],
                subcategory=row[SUB_CATEGORY_COL]
            )
            
        # Add edges for similar items
        similar_pairs = self.find_similar_items(data)
        for item1, item2, similarity in similar_pairs:
            G.add_edge(item1, item2, weight=similarity)
            
        return G
        
    def get_similarity_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about similarities.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            Dictionary containing statistics.
        """
        if self.sku_similarity_matrix is None or self.desc_similarity_matrix is None:
            self.compute_similarity_matrix(data)
            
        similar_pairs = self.find_similar_items(data)
        
        # Build graph for connected component analysis
        G = self.build_similarity_graph(data)
        
        stats = {
            'total_items': len(data),
            'similar_pairs': len(similar_pairs),
            'average_sku_similarity': float(np.mean(self.sku_similarity_matrix)),
            'average_desc_similarity': float(np.mean(self.desc_similarity_matrix)),
            'max_sku_similarity': float(np.max(self.sku_similarity_matrix)),
            'max_desc_similarity': float(np.max(self.desc_similarity_matrix)),
            'similar_items_ratio': len(similar_pairs) / len(data),
            'connected_components': nx.number_connected_components(G),
            'largest_component_size': len(max(nx.connected_components(G), key=len)),
            'per_item_stats': {
                'avg_similarity_mean': float(data['Average Similarity'].mean()),
                'high_similarity_mean': float(data['High Similarity Average'].mean()),
                'items_with_high_similarity': int((data['High Similarity Average'] > 0).sum())
            }
        }
        
        logger.info(f"Similarity statistics: {stats}")
        return stats
        
    def save_similarity_results(self, data: pd.DataFrame, output_file: Path) -> None:
        """Save similarity results to file.
        
        Args:
            data: DataFrame containing inventory data.
            output_file: Path to save results.
        """
        # Compute similarity matrix if not already computed
        if self.sku_similarity_matrix is None or self.desc_similarity_matrix is None:
            self.compute_similarity_matrix(data)
            
        # Get similar pairs
        similar_pairs = self.find_similar_items(data)
        
        # Create results DataFrame
        results = pd.DataFrame(similar_pairs, columns=['Item1', 'Item2', 'Similarity'])
        
        # Add item details
        item_details = data[[CATALOG_NUMBER_COL, DESCRIPTION_COL, MAIN_CATEGORY_COL, SUB_CATEGORY_COL]]
        
        results = (
            results
            .merge(
                item_details,
                left_on='Item1',
                right_on=CATALOG_NUMBER_COL,
                suffixes=('', '_1')
            )
            .merge(
                item_details,
                left_on='Item2',
                right_on=CATALOG_NUMBER_COL,
                suffixes=('_1', '_2')
            )
        )
        
        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_type = output_file.suffix.lower()
        if file_type == '.csv':
            results.to_csv(output_file, index=False)
        elif file_type == '.xlsx':
            results.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    def save_similarity_matrix(self, output_file: Path) -> None:
        """Save the similarity matrices to files.
        
        Args:
            output_file: Base path for output files.
        """
        if self.sku_similarity_matrix is None or self.desc_similarity_matrix is None:
            raise ValueError("Similarity matrices not computed.")
            
        # Save SKU similarity matrix
        sku_file = output_file.parent / f"{output_file.stem}_sku{output_file.suffix}"
        pd.DataFrame(self.sku_similarity_matrix).to_csv(sku_file, index=False)
        
        # Save description similarity matrix
        desc_file = output_file.parent / f"{output_file.stem}_desc{output_file.suffix}"
        pd.DataFrame(self.desc_similarity_matrix).to_csv(desc_file, index=False)
        
        logger.info(f"Saved similarity matrices to {sku_file} and {desc_file}")

    def compute_similarity(self, data: pd.DataFrame) -> np.ndarray:
        """Alias for compute_similarity_matrix for compatibility with tests and requirements."""
        return self.compute_similarity_matrix(data)

    def get_similarity_score(self, sku1: str, sku2: str) -> float:
        """Get similarity score between two SKUs from the similarity matrix."""
        if self.sku_similarity_matrix is None or self.desc_similarity_matrix is None:
            raise ValueError("Similarity matrices not computed.")
        i = self._sku_to_index.get(sku1)
        j = self._sku_to_index.get(sku2)
        if i is None or j is None:
            raise KeyError(f"SKU not found: {sku1 if i is None else sku2}")
        return float(self.sku_similarity_matrix[i, j])

    def compute_similarity_statistics(self, data: pd.DataFrame, thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """Compute detailed similarity statistics for multiple thresholds and per-item stats."""
        if thresholds is None:
            thresholds = [self.threshold]
        stats = {}
        for thresh in thresholds:
            self.threshold = thresh
            self.compute_similarity_matrix(data)
            similar_pairs = self.find_similar_items(data)
            G = self.build_similarity_graph(data)
            per_item_stats = {
                sku: {
                    'avg_sku_similarity': float(np.mean([v for k, v in self.sku_similarity_dicts[sku].items() if k != sku])),
                    'max_sku_similarity': float(np.max([v for k, v in self.sku_similarity_dicts[sku].items() if k != sku])),
                    'avg_desc_similarity': float(np.mean([v for k, v in self.desc_similarity_dicts[sku].items() if k != sku])),
                    'max_desc_similarity': float(np.max([v for k, v in self.desc_similarity_dicts[sku].items() if k != sku])),
                }
                for sku in self.sku_similarity_dicts
            }
            stats[thresh] = {
                'total_items': len(data),
                'similar_pairs': len(similar_pairs),
                'average_sku_similarity': float(np.mean(self.sku_similarity_matrix)),
                'average_desc_similarity': float(np.mean(self.desc_similarity_matrix)),
                'max_sku_similarity': float(np.max(self.sku_similarity_matrix)),
                'max_desc_similarity': float(np.max(self.desc_similarity_matrix)),
                'similar_items_ratio': len(similar_pairs) / len(data),
                'connected_components': nx.number_connected_components(G),
                'largest_component_size': len(max(nx.connected_components(G), key=len)),
                'per_item_stats': per_item_stats
            }
            logger.info(f"Similarity statistics for threshold {thresh}: {stats[thresh]}")
        return stats

    def save_similarity_statistics(self, stats: Dict[str, Any], output_file: Path) -> None:
        """Save similarity statistics to a file (JSON or CSV)."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.suffix.lower() == '.json':
            import json
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
        elif output_file.suffix.lower() == '.csv':
            # Flatten for CSV
            rows = []
            for thresh, s in stats.items():
                row = {'threshold': thresh}
                row.update({k: v for k, v in s.items() if k != 'per_item_stats'})
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported file type: {output_file.suffix}")
        logger.info(f"Saved similarity statistics to {output_file}") 