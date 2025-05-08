"""Clustering module for inventory analysis."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine

from .config import (
    DEFAULT_DBSCAN_EPS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_KMEANS_CLUSTERS,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    CATALOG_NUMBER_COL,
    DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    CLUSTER_COL,
    KMEANS_CLUSTERS,
    SVD_COMPONENTS,
    TFIDF_MAX_FEATURES,
    ENRICHED_DESCRIPTION_COL,
    DEFAULT_SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """Clustering and cluster analysis for inventory items."""
    
    def __init__(self, tfidf_max_features: int = TFIDF_MAX_FEATURES, svd_components: int = SVD_COMPONENTS):
        self.tfidf = TfidfVectorizer(max_features=tfidf_max_features)
        self.svd = TruncatedSVD(n_components=svd_components)
        self.scaler = StandardScaler()
        self.kmeans = None
        self.dbscan = None
        self.is_fitted = False
        self.last_features = None
        self.last_labels = None
        self.last_method = None
        self.last_metrics = None
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure.
        
        Args:
            data: DataFrame to validate.
            
        Raises:
            ValueError: If data is invalid.
        """
        required_cols = [DESCRIPTION_COL, MAIN_CATEGORY_COL, SUB_CATEGORY_COL]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if len(data) == 0:
            raise ValueError("Empty DataFrame provided")
            
        if data[DESCRIPTION_COL].isna().all():
            raise ValueError("All descriptions are missing")
            
    def _extract_features(self, data: pd.DataFrame, fit: bool = True) -> np.ndarray:
        desc_col = ENRICHED_DESCRIPTION_COL if ENRICHED_DESCRIPTION_COL in data.columns else DESCRIPTION_COL
        descriptions = data[desc_col].fillna("")
        if fit:
            tfidf_features = self.tfidf.fit_transform(descriptions)
            svd_features = self.svd.fit_transform(tfidf_features)
        else:
            tfidf_features = self.tfidf.transform(descriptions)
            svd_features = self.svd.transform(tfidf_features)
        return svd_features
        
    def fit_dbscan(self, data: pd.DataFrame, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> np.ndarray:
        """Cluster using DBSCAN with cosine distance on TF-IDF+SVD features."""
        features = self._extract_features(data, fit=True)
        # Compute cosine distance matrix
        normed = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        cosine_dist = 1 - np.dot(normed, normed.T)
        # DBSCAN expects a condensed distance matrix, but we use precomputed
        eps = 1 - threshold
        dbscan = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
        labels = dbscan.fit_predict(cosine_dist)
        self.dbscan = dbscan
        self.is_fitted = True
        self.last_features = features
        self.last_labels = labels
        self.last_method = 'dbscan'
        logger.info(f"DBSCAN clustering complete: {len(set(labels)) - (1 if -1 in labels else 0)} clusters found.")
        return labels
        
    def fit_kmeans(self, data: pd.DataFrame, n_clusters: int = KMEANS_CLUSTERS) -> np.ndarray:
        features = self._extract_features(data, fit=True)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        self.kmeans = kmeans
        self.is_fitted = True
        self.last_features = features
        self.last_labels = labels
        self.last_method = 'kmeans'
        logger.info(f"KMeans clustering complete: {n_clusters} clusters assigned.")
        return labels
        
    def get_clusters(self, method: str = 'dbscan') -> np.ndarray:
        if not self.is_fitted or self.last_method != method:
            raise ValueError(f"No clustering results for method: {method}")
        return self.last_labels
        
    def analyze_clusters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clustering results.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            Dictionary containing analysis results.
            
        Raises:
            ValueError: If clusters not computed.
        """
        if not self.is_fitted:
            raise ValueError("Must compute clusters first")
            
        # Get features for metrics
        features = self.last_features
        
        # Basic statistics
        analysis = {
            'n_clusters': len(set(self.last_labels)) - (1 if -1 in self.last_labels else 0),
            'n_noise': list(self.last_labels).count(-1) if -1 in self.last_labels else 0,
            'cluster_sizes': pd.Series(self.last_labels).value_counts().to_dict()
        }
        
        # Category distribution
        analysis['category_distribution'] = {}
        for cluster in set(self.last_labels):
            if cluster == -1:
                continue
            mask = self.last_labels == cluster
            analysis['category_distribution'][cluster] = {
                'main_categories': data.loc[mask, MAIN_CATEGORY_COL].value_counts().to_dict(),
                'sub_categories': data.loc[mask, SUB_CATEGORY_COL].value_counts().to_dict()
            }
            
        # Clustering metrics
        metrics = self.get_cluster_statistics(data, self.last_labels)
        analysis.update(metrics)
        
        return analysis
        
    def get_cluster_statistics(self, data: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        features = self.last_features
        mask = labels != -1
        stats = {}
        if np.unique(labels[mask]).size < 2:
            logger.warning("Not enough clusters for metrics.")
            return stats
        try:
            stats['silhouette'] = silhouette_score(features[mask], labels[mask])
        except Exception as e:
            stats['silhouette'] = None
            logger.warning(f"Silhouette score failed: {e}")
        try:
            stats['calinski_harabasz'] = calinski_harabasz_score(features[mask], labels[mask])
        except Exception as e:
            stats['calinski_harabasz'] = None
            logger.warning(f"Calinski-Harabasz score failed: {e}")
        try:
            stats['davies_bouldin'] = davies_bouldin_score(features[mask], labels[mask])
        except Exception as e:
            stats['davies_bouldin'] = None
            logger.warning(f"Davies-Bouldin score failed: {e}")
        # NetworkX graph metrics
        try:
            G = nx.Graph()
            for i, idx1 in enumerate(np.where(mask)[0]):
                for j, idx2 in enumerate(np.where(mask)[0]):
                    if i < j and labels[idx1] == labels[idx2]:
                        G.add_edge(idx1, idx2)
            degrees = [d for n, d in G.degree()]
            stats['avg_degree'] = float(np.mean(degrees)) if degrees else 0.0
            components = list(nx.connected_components(G))
            stats['num_components'] = len(components)
            stats['largest_component'] = max((len(c) for c in components), default=0)
        except Exception as e:
            stats['avg_degree'] = None
            stats['num_components'] = None
            stats['largest_component'] = None
            logger.warning(f"NetworkX metrics failed: {e}")
        self.last_metrics = stats
        logger.info(f"Cluster statistics: {stats}")
        return stats
        
    def visualize_clusters(self, data: pd.DataFrame, output_file: Path) -> None:
        """Create and save cluster visualization.
        
        Args:
            data: DataFrame containing inventory data.
            output_file: Path to save visualization.
            
        Raises:
            ValueError: If clusters not computed.
        """
        if not self.is_fitted:
            raise ValueError("Must compute clusters first")
            
        # Get features for visualization
        features = self.last_features
        
        # Reduce dimensionality for visualization
        mds = MDS(n_components=2, random_state=42)
        coords = mds.fit_transform(features)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot points
        scatter = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=self.last_labels,
            cmap='viridis',
            alpha=0.6
        )
        
        # Add labels for some points
        for i, (x, y) in enumerate(coords):
            if i % 10 == 0:  # Label every 10th point
                plt.annotate(
                    data.iloc[i][CATALOG_NUMBER_COL],
                    (x, y),
                    fontsize=8
                )
                
        # Add legend
        plt.colorbar(scatter, label='Cluster')
        
        # Add title and labels
        plt.title('Cluster Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_cluster_results(self, data: pd.DataFrame, output_dir: Path) -> None:
        """Save clustering results.
        
        Args:
            data: DataFrame containing inventory data.
            output_dir: Directory to save results.
            
        Raises:
            ValueError: If clusters not computed.
        """
        if not self.is_fitted:
            raise ValueError("Must compute clusters first")
            
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cluster assignments
            results = data.copy()
            results[CLUSTER_COL] = self.last_labels
            results.to_excel(output_dir / 'cluster_assignments.xlsx', index=False)
            
            # Save cluster analysis
            analysis = self.analyze_clusters(data)
            analysis_df = pd.DataFrame.from_dict(analysis, orient='index')
            analysis_df.to_excel(output_dir / 'cluster_analysis.xlsx')
            
            # Create and save visualization
            self.visualize_clusters(data, output_dir / 'cluster_visualization.png')
            
            # Save metrics
            self.save_metrics(self.last_metrics, output_dir / 'cluster_metrics.csv')
            
            # Save model
            joblib.dump(self, output_dir / 'cluster_model.joblib')
            
        except Exception as e:
            logger.error(f"Failed to save cluster results: {str(e)}")
            raise ValueError(f"Failed to save cluster results: {str(e)}")
            
    def load_model(self, input_dir: Path) -> None:
        """Load trained model from file.
        
        Args:
            input_dir: Directory containing model file.
            
        Raises:
            ValueError: If loading fails.
        """
        try:
            model_path = input_dir / 'cluster_model.joblib'
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
                
            loaded_model = joblib.load(model_path)
            self.__dict__.update(loaded_model.__dict__)
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")

    def save_metrics(self, metrics: Dict[str, Any], output_file: Path) -> None:
        pd.DataFrame([metrics]).to_csv(output_file, index=False)
        logger.info(f"Cluster metrics saved to {output_file}") 