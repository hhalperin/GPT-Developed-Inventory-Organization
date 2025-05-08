"""Main pipeline for inventory analysis."""

import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE,
    OUTPUT_DIR,
    FINAL_CATEGORIES,
    CATALOG_NUMBER_COL,
    DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    ENRICHED_DESCRIPTION_COL,
    CLUSTER_COL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_DBSCAN_EPS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_KMEANS_CLUSTERS,
    MFR_CODE_COL
)
from .data import DataLoader
from .api import GPTClient
from .similarity import SimilarityScorer
from .classification import MLCategorizer
from .clustering import ClusterAnalyzer
from .visualize import Visualizer

logger = logging.getLogger(__name__)

class InventoryPipeline:
    """Main pipeline for inventory analysis."""
    
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        dbscan_eps: float = DEFAULT_DBSCAN_EPS,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        kmeans_clusters: int = DEFAULT_KMEANS_CLUSTERS,
        gpt_api_key: str = None
    ):
        """Initialize pipeline.
        
        Args:
            similarity_threshold: Threshold for similarity scoring.
            dbscan_eps: Epsilon parameter for DBSCAN (not passed to ClusterAnalyzer).
            min_samples: Minimum samples for DBSCAN (not passed to ClusterAnalyzer).
            kmeans_clusters: Number of clusters for KMeans.
            gpt_api_key: API key for GPT.
        """
        self.data_loader = DataLoader()
        self.ml_categorizer = MLCategorizer()
        self.cluster_analyzer = ClusterAnalyzer()
        self.similarity_scorer = SimilarityScorer(threshold=similarity_threshold)
        self.visualizer = Visualizer()
        self.gpt_client = GPTClient(api_key=gpt_api_key)
        
        self.data: Optional[pd.DataFrame] = None
        self.similarity_matrix = None
        self.clusters = None
        # Store config for reference
        self.similarity_threshold = similarity_threshold
        self.dbscan_eps = dbscan_eps
        self.min_samples = min_samples
        self.kmeans_clusters = kmeans_clusters
        self.gpt_api_key = gpt_api_key
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and clean data.
        
        Args:
            file_path: Path to data file.
            
        Returns:
            Loaded and cleaned DataFrame.
        """
        self.data = self.data_loader.load_data(file_path)
        self.data = self.data_loader.clean_data()
        return self.data
        
    def enrich_descriptions(self) -> None:
        """Enrich item descriptions using GPT."""
        if self.data is None:
            raise ValueError("No data loaded")
        # Only enrich if not already present or mostly missing
        if ENRICHED_DESCRIPTION_COL in self.data.columns and self.data[ENRICHED_DESCRIPTION_COL].notna().sum() > 0.8 * len(self.data):
            logger.info("Descriptions already enriched")
            return
        logger.info("Enriching descriptions using GPT...")
        try:
            enriched = []
            for idx, row in self.data.iterrows():
                catalog_no = row.get(CATALOG_NUMBER_COL, "")
                mfr_code = row.get(MFR_CODE_COL, "")
                description = row.get(DESCRIPTION_COL, "")
                enriched_desc = self.gpt_client.enrich_description(catalog_no, mfr_code, description)
                enriched.append(enriched_desc)
                if (idx+1) % 10 == 0 or (idx+1) == len(self.data):
                    logger.info(f"Enriched {idx+1}/{len(self.data)} descriptions...")
            self.data[ENRICHED_DESCRIPTION_COL] = enriched
            num_enriched = sum(x != "Unknown" for x in enriched)
            logger.info(f"Enrichment complete: {num_enriched}/{len(self.data)} items enriched successfully.")
        except Exception as e:
            logger.error(f"GPT enrichment failed: {e}")
            raise
        # Save enriched data
        self.data.to_excel(OUTPUT_DIR / 'enriched_data.xlsx', index=False)
        logger.info("Saved enriched data to enriched_data.xlsx")
        
    def compute_similarities(self) -> None:
        """Compute similarity scores between items."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        self.similarity_matrix = self.similarity_scorer.compute_similarity_matrix(self.data)
        
    def assign_categories(self) -> None:
        """Assign categories to items using ML and GPT fallback."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Train ML model if needed
        if not self.ml_categorizer.is_fitted:
            self.ml_categorizer.train(self.data)
            
        # Get ML predictions
        predictions = self.ml_categorizer.predict(self.data)
        self.data['Predicted Main Category'] = predictions['Predicted Main Category']
        self.data['Main Category Confidence'] = predictions['Main Category Confidence']
        self.data['Predicted Sub-category'] = predictions['Predicted Sub-category']
        self.data['Sub-category Confidence'] = predictions['Sub-category Confidence']
        
        # Fallback to GPT for low-confidence predictions
        threshold = getattr(self.ml_categorizer, 'confidence_threshold', 0.75)
        for idx, row in self.data.iterrows():
            if row['Main Category Confidence'] < threshold:
                desc = row.get(ENRICHED_DESCRIPTION_COL, row.get(DESCRIPTION_COL, ""))
                try:
                    main_prompt = f"Determine the main category for: {desc}"
                    main_cat = self.gpt_client.call_gpt_api(main_prompt)
                    sub_prompt = f"Determine the sub-category for: {desc} (main category: {main_cat})"
                    sub_cat = self.gpt_client.call_gpt_api(sub_prompt)
                    self.data.at[idx, 'Predicted Main Category'] = main_cat
                    self.data.at[idx, 'Predicted Sub-category'] = sub_cat
                    self.data.at[idx, 'Main Category Confidence'] = 0.0  # Indicate fallback
                    self.data.at[idx, 'Sub-category Confidence'] = 0.0
                except Exception as e:
                    logger.error(f"GPT fallback failed for row {idx}: {e}")
                    self.data.at[idx, 'Predicted Main Category'] = 'Unknown'
                    self.data.at[idx, 'Predicted Sub-category'] = 'Unknown'
                    
        # Save categorized data
        self.data['Main Category'] = self.data['Predicted Main Category']
        self.data['Sub-category'] = self.data['Predicted Sub-category']
        self.data.to_excel(OUTPUT_DIR / 'categorized_data.xlsx', index=False)
        logger.info("Saved categorized data to categorized_data.xlsx")
        
    def perform_clustering(self) -> None:
        """Perform clustering analysis."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Get cluster assignments
        self.clusters = self.cluster_analyzer.get_clusters(self.data)
        self.data[CLUSTER_COL] = self.clusters
        
    def generate_visualizations(self, output_dir: Path) -> None:
        """Generate visualizations.
        
        Args:
            output_dir: Directory to save visualizations.
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        if self.similarity_matrix is None:
            raise ValueError("Must compute similarities first")
            
        self.visualizer.save_visualization_results(
            self.data,
            self.similarity_matrix,
            output_dir
        )
        
    def save_results(self, output_dir: Path) -> None:
        """Save analysis results.
        
        Args:
            output_dir: Directory to save results.
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        self.data.to_excel(output_dir / 'processed_data.xlsx', index=False)
        
        # Save similarity matrix
        if self.similarity_matrix is not None:
            pd.DataFrame(self.similarity_matrix).to_csv(
                output_dir / 'similarity_matrix.csv',
                index=False
            )
            
        # Save cluster analysis
        if self.clusters is not None:
            cluster_analysis = self.cluster_analyzer.analyze_clusters(self.data)
            pd.DataFrame.from_dict(cluster_analysis).to_excel(
                output_dir / 'cluster_analysis.xlsx'
            )
            
    def save_statistics(self, output_dir: Path, thresholds: Optional[List[float]] = None) -> None:
        """Aggregate and save all statistics and evaluation outputs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Similarity statistics
        sim_stats = self.similarity_scorer.compute_similarity_statistics(self.data, thresholds=thresholds)
        self.similarity_scorer.save_similarity_statistics(sim_stats, output_dir / 'similarity_statistics.json')
        # Category change statistics (if reevaluation was performed)
        if 'Main Category Changed' in self.data.columns and 'Sub-category Changed' in self.data.columns:
            main_changed = int(self.data['Main Category Changed'].sum())
            sub_changed = int(self.data['Sub-category Changed'].sum())
            total = len(self.data)
            cat_change_stats = {
                'main_category_changed': main_changed,
                'sub_category_changed': sub_changed,
                'main_category_changed_pct': main_changed / total if total else 0,
                'sub_category_changed_pct': sub_changed / total if total else 0
            }
            import json
            with open(output_dir / 'category_change_statistics.json', 'w') as f:
                json.dump(cat_change_stats, f, indent=2)
            logger.info(f"Category change statistics: {cat_change_stats}")
        # Category & cluster distributions
        cat_dist = self.data[MAIN_CATEGORY_COL].value_counts().to_dict()
        subcat_dist = self.data[SUB_CATEGORY_COL].value_counts().to_dict()
        cluster_dist = self.data[CLUSTER_COL].value_counts().to_dict() if CLUSTER_COL in self.data.columns else {}
        dist_stats = {
            'category_distribution': cat_dist,
            'subcategory_distribution': subcat_dist,
            'cluster_distribution': cluster_dist
        }
        import json
        with open(output_dir / 'distribution_statistics.json', 'w') as f:
            json.dump(dist_stats, f, indent=2)
        logger.info(f"Distribution statistics: {dist_stats}")
        # Cluster metrics
        if self.clusters is not None:
            cluster_metrics = self.cluster_analyzer.get_cluster_statistics(self.data, self.clusters)
            import json
            with open(output_dir / 'cluster_metrics.json', 'w') as f:
                json.dump(cluster_metrics, f, indent=2)
            logger.info(f"Cluster metrics: {cluster_metrics}")

    def run_pipeline(
        self,
        input_file: Path,
        output_dir: Path,
        enrich: bool = True,
        cluster: bool = True,
        visualize: bool = True,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Run the complete analysis pipeline.
        
        Args:
            input_file: Path to input data file.
            output_dir: Directory to save results.
            enrich: Whether to enrich descriptions.
            cluster: Whether to perform clustering.
            visualize: Whether to generate visualizations.
            thresholds: List of similarity thresholds for statistics.
        
        Returns:
            Dictionary containing pipeline statistics.
        """
        import time
        t0 = time.time()
        # Load data
        self.load_data(input_file)
        t1 = time.time(); logger.info(f"Data loading: {t1-t0:.2f}s")
        # Enrich descriptions if requested
        if enrich:
            self.enrich_descriptions()
        t2 = time.time(); logger.info(f"Description enrichment: {t2-t1:.2f}s")
        # Compute similarities
        self.compute_similarities()
        t3 = time.time(); logger.info(f"Similarity computation: {t3-t2:.2f}s")
        # Assign categories
        self.assign_categories()
        t4 = time.time(); logger.info(f"Category assignment: {t4-t3:.2f}s")
        # Perform clustering if requested
        if cluster:
            self.perform_clustering()
        t5 = time.time(); logger.info(f"Clustering: {t5-t4:.2f}s")
        # Generate visualizations if requested
        if visualize:
            self.generate_visualizations(output_dir)
        t6 = time.time(); logger.info(f"Visualization: {t6-t5:.2f}s")
        # Save results
        self.save_results(output_dir)
        # Save statistics
        self.save_statistics(output_dir, thresholds=thresholds)
        t7 = time.time(); logger.info(f"Statistics & output: {t7-t6:.2f}s")
        # Return statistics summary
        summary = {
            'total_items': len(self.data),
            'unique_categories': len(self.data[MAIN_CATEGORY_COL].unique()),
            'unique_subcategories': len(self.data[SUB_CATEGORY_COL].unique()),
            'clusters': len(set(self.clusters)) if self.clusters is not None else None,
            'timing': {
                'data_loading': t1-t0,
                'enrichment': t2-t1,
                'similarity': t3-t2,
                'category_assignment': t4-t3,
                'clustering': t5-t4,
                'visualization': t6-t5,
                'statistics': t7-t6,
                'total': t7-t0
            }
        }
        logger.info(f"Pipeline summary: {summary}")
        return summary
            
def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Inventory Analysis Pipeline")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to input Excel file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for output files"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Threshold for similarity scoring"
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=DEFAULT_DBSCAN_EPS,
        help="Epsilon parameter for DBSCAN"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum samples parameter for DBSCAN"
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=DEFAULT_KMEANS_CLUSTERS,
        help="Number of clusters for KMeans"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key for GPT enrichment"
    )
    
    args = parser.parse_args()
    
    pipeline = InventoryPipeline(
        similarity_threshold=args.similarity_threshold,
        dbscan_eps=args.dbscan_eps,
        min_samples=args.min_samples,
        kmeans_clusters=args.kmeans_clusters,
        gpt_api_key=args.api_key
    )
    
    pipeline.run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir
    )
    
if __name__ == "__main__":
    main() 