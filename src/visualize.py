"""Visualization module for inventory analysis."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import networkx as nx
import json

from .config import (
    CATALOG_NUMBER_COL,
    DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    CLUSTER_COL
)

logger = logging.getLogger(__name__)

class Visualizer:
    """Creates visualizations for inventory analysis."""
    
    def __init__(self):
        """Initialize visualizer."""
        pass
        
    def plot_category_distribution(self, data: pd.DataFrame, output_file: Path) -> None:
        """Plot category distribution.
        
        Args:
            data: DataFrame containing inventory data.
            output_file: Path to save plot.
        """
        # Get category counts
        category_counts = data[MAIN_CATEGORY_COL].value_counts()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        
    def plot_cluster_distribution(self, data: pd.DataFrame, output_file: Path) -> None:
        """Plot cluster distribution.
        
        Args:
            data: DataFrame containing inventory data.
            output_file: Path to save plot.
        """
        if CLUSTER_COL not in data.columns:
            raise ValueError("No cluster assignments found in data")
            
        # Get cluster counts
        cluster_counts = data[CLUSTER_COL].value_counts().sort_index()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
        plt.title('Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save plot
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        
    def plot_category_cluster_heatmap(self, data: pd.DataFrame, output_file: Path) -> None:
        """Plot heatmap of categories vs clusters.
        
        Args:
            data: DataFrame containing inventory data.
            output_file: Path to save plot.
        """
        if CLUSTER_COL not in data.columns:
            raise ValueError("No cluster assignments found in data")
            
        # Create cross-tabulation
        cross_tab = pd.crosstab(data[MAIN_CATEGORY_COL], data[CLUSTER_COL])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Category vs Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Category')
        plt.tight_layout()
        
        # Save plot
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        
    def plot_similarity_network(self, data: pd.DataFrame, similarity_matrix: np.ndarray, output_file: Path) -> None:
        """Plot network of similar items.
        
        Args:
            data: DataFrame containing inventory data.
            similarity_matrix: Pre-computed similarity matrix.
            output_file: Path to save plot.
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, row in data.iterrows():
            G.add_node(
                row[CATALOG_NUMBER_COL],
                category=row[MAIN_CATEGORY_COL],
                cluster=data[CLUSTER_COL].iloc[i] if CLUSTER_COL in data.columns else None
            )
            
        # Add edges for similar items
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if similarity_matrix[i, j] > 0.5:  # Only show strong similarities
                    G.add_edge(
                        data.iloc[i][CATALOG_NUMBER_COL],
                        data.iloc[j][CATALOG_NUMBER_COL],
                        weight=float(similarity_matrix[i, j])
                    )
                    
        # Create plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        if CLUSTER_COL in data.columns:
            node_colors = [G.nodes[n]['cluster'] for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='tab10')
        else:
            nx.draw_networkx_nodes(G, pos)
            
        # Draw edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title('Similarity Network')
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        
    def plot_mds_visualization(self, data: pd.DataFrame, similarity_matrix: np.ndarray, output_file: Path) -> None:
        """Plot MDS visualization of items.
        
        Args:
            data: DataFrame containing inventory data.
            similarity_matrix: Pre-computed similarity matrix.
            output_file: Path to save plot.
        """
        # Compute MDS embedding
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(1 - similarity_matrix)  # Convert similarity to distance
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot points
        if CLUSTER_COL in data.columns:
            scatter = plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=data[CLUSTER_COL],
                cmap='tab10'
            )
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1])
            
        # Add labels
        for i, row in data.iterrows():
            plt.annotate(
                row[CATALOG_NUMBER_COL],
                (embedding[i, 0], embedding[i, 1]),
                xytext=(5, 5),
                textcoords='offset points'
            )
            
        plt.title('MDS Visualization')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.tight_layout()
        
        # Save plot
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        
    def get_visualization_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics for visualization.
        
        Args:
            data: DataFrame containing inventory data.
            
        Returns:
            Dictionary containing statistics.
        """
        stats = {
            'total_items': len(data),
            'category_distribution': data[MAIN_CATEGORY_COL].value_counts().to_dict(),
            'subcategory_distribution': data[SUB_CATEGORY_COL].value_counts().to_dict(),
            'category_counts': {
                'main': len(data[MAIN_CATEGORY_COL].unique()),
                'sub': len(data[SUB_CATEGORY_COL].unique())
            }
        }
        
        if CLUSTER_COL in data.columns:
            stats.update({
                'cluster_distribution': data[CLUSTER_COL].value_counts().to_dict(),
                'clusters': len(data[CLUSTER_COL].unique())
            })
            
        return stats
        
    def plot_top_subcategory_distribution(self, data: pd.DataFrame, output_file: Path, top_n: int = 20) -> None:
        """Plot top-N sub-category distribution.
        
        Args:
            data: DataFrame containing inventory data.
            output_file: Path to save plot.
            top_n: Number of top sub-categories to plot.
        """
        try:
            subcat_counts = data[SUB_CATEGORY_COL].value_counts().head(top_n)
            plt.figure(figsize=(12, 6))
            sns.barplot(x=subcat_counts.index, y=subcat_counts.values)
            plt.title(f'Top {top_n} Sub-category Distribution')
            plt.xlabel('Sub-category')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot top-N sub-category distribution: {e}")

    def save_visualization_results(self, data: pd.DataFrame, similarity_matrix: np.ndarray, output_dir: Path) -> None:
        """Save all visualizations.
        
        Args:
            data: DataFrame containing inventory data.
            similarity_matrix: Pre-computed similarity matrix.
            output_dir: Directory to save visualizations.
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Generate visualizations
            self.plot_category_distribution(data, output_dir / 'category_distribution.png')
            self.plot_top_subcategory_distribution(data, output_dir / 'top_subcategory_distribution.png', top_n=20)
            self.plot_similarity_network(data, similarity_matrix, output_dir / 'similarity_network.png')
            self.plot_mds_visualization(data, similarity_matrix, output_dir / 'mds_visualization.png')
            if CLUSTER_COL in data.columns:
                self.plot_cluster_distribution(data, output_dir / 'cluster_distribution.png')
                self.plot_category_cluster_heatmap(data, output_dir / 'category_cluster_heatmap.png')
            # Save statistics
            stats = self.get_visualization_statistics(data)
            with open(output_dir / 'visualization_statistics.json', 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save visualization results: {e}") 