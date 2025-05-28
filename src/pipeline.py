"""Main pipeline for inventory analysis."""

import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add tqdm import with fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

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
        
    def enrich_descriptions(self, output_dir: Optional[Path] = None) -> None:
        print('[DEBUG] Entering enrich_descriptions')
        if self.data is None:
            raise ValueError("No data loaded")
        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_path = output_dir / 'enriched_data.xlsx'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Resume logic
        if output_path.exists():
            logger.info(f"Loading enriched data from {output_path}")
            self.data = pd.read_excel(output_path)
            print('[DEBUG] Loaded enriched data from file, skipping enrichment')
            return
        # Only enrich if not already present or mostly missing
        if ENRICHED_DESCRIPTION_COL in self.data.columns and self.data[ENRICHED_DESCRIPTION_COL].notna().sum() > 0.8 * len(self.data):
            logger.info("Descriptions already enriched")
            print('[DEBUG] Descriptions already enriched, skipping')
            return
        logger.info("Enriching descriptions using GPT...")
        try:
            enriched = []
            for idx, row in tqdm(list(self.data.iterrows()), desc='Enriching descriptions', leave=True, disable=False):
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
            print(f'[DEBUG] Exception in enrich_descriptions: {e}')
            raise
        # Save enriched data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_excel(output_path, index=False)
        logger.info(f"Saved enriched data to {output_path}")
        print('[DEBUG] Exiting enrich_descriptions')
        
    def compute_similarities(self) -> None:
        print('[DEBUG] Entering compute_similarities')
        if self.data is None:
            raise ValueError("No data loaded")
        self.similarity_matrix = self.similarity_scorer.compute_similarity_matrix(self.data)
        print('[DEBUG] Exiting compute_similarities')
        
    def assign_categories(self, output_dir: Optional[Path] = None) -> None:
        print('[DEBUG] Entering assign_categories')
        if self.data is None:
            raise ValueError("No data loaded")
        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_path = output_dir / 'categorized_data.xlsx'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resume logic
        if output_path.exists():
            logger.info(f"Loading categorized data from {output_path}")
            self.data = pd.read_excel(output_path)
            print('[DEBUG] Loaded categorized data from file, skipping categorization')
            return

        # Initialize prediction columns
        self.data['Predicted Main Category'] = 'Unknown'
        self.data['Predicted Sub-category'] = 'Unknown'
        self.data['Main Category Confidence'] = 0.0
        self.data['Sub-category Confidence'] = 0.0
        self.data['Categorization Reason'] = ''

        # Get existing categorized data for training
        categorized_data = self.data[
            (self.data[MAIN_CATEGORY_COL].notna()) & 
            (self.data[MAIN_CATEGORY_COL] != 'Unknown') &
            (self.data[SUB_CATEGORY_COL].notna()) & 
            (self.data[SUB_CATEGORY_COL] != 'Unknown')
        ]

        if not categorized_data.empty:
            logger.info(f"Training ML model on {len(categorized_data)} pre-categorized items")
            self.ml_categorizer.train(categorized_data)
            
            # Get ML predictions for all items
            predictions = self.ml_categorizer.predict(self.data)
            self.data['Predicted Main Category'] = predictions['Predicted Main Category']
            self.data['Main Category Confidence'] = predictions['Main Category Confidence']
            self.data['Predicted Sub-category'] = predictions['Predicted Sub-category']
            self.data['Sub-category Confidence'] = predictions['Sub-category Confidence']
            self.data['Categorization Reason'] = 'ML prediction'

        # Set confidence threshold
        threshold = getattr(self.ml_categorizer, 'confidence_threshold', 0.75)
        
        # Track statistics
        ml_used = 0
        gpt_used = 0
        gpt_failed = 0

        # Process each item
        for idx, row in tqdm(list(self.data.iterrows()), desc='Assigning categories', leave=True, disable=False):
            use_gpt = False
            
            # Determine if we need GPT
            if not self.ml_categorizer.is_fitted:
                use_gpt = True
            elif (row['Main Category Confidence'] < threshold or 
                  row['Predicted Main Category'] == 'Unknown' or
                  row['Sub-category Confidence'] < threshold or 
                  row['Predicted Sub-category'] == 'Unknown'):
                use_gpt = True
            
            if use_gpt:
                gpt_used += 1
                desc = row.get(ENRICHED_DESCRIPTION_COL, row.get(DESCRIPTION_COL, ""))
                try:
                    main_prompt = f"Determine the main category for: {desc}"
                    main_cat = self.gpt_client.call_gpt_api(main_prompt)
                    sub_prompt = f"Determine the sub-category for: {desc} (main category: {main_cat})"
                    sub_cat = self.gpt_client.call_gpt_api(sub_prompt)
                    self.data.at[idx, 'Predicted Main Category'] = main_cat
                    self.data.at[idx, 'Predicted Sub-category'] = sub_cat
                    self.data.at[idx, 'Main Category Confidence'] = 0.0
                    self.data.at[idx, 'Sub-category Confidence'] = 0.0
                    self.data.at[idx, 'Categorization Reason'] = 'Used GPT (ML not trained or low confidence)'
                except Exception as e:
                    logger.error(f"GPT fallback failed for row {idx}: {e}")
                    self.data.at[idx, 'Predicted Main Category'] = 'Unknown'
                    self.data.at[idx, 'Predicted Sub-category'] = 'Unknown'
                    self.data.at[idx, 'Main Category Confidence'] = 0.0
                    self.data.at[idx, 'Sub-category Confidence'] = 0.0
                    self.data.at[idx, 'Categorization Reason'] = f'GPT failed: {e}'
                    gpt_failed += 1
            else:
                ml_used += 1

        # Log statistics
        total = len(self.data)
        logger.info(f"Categorization complete:")
        logger.info(f"- ML model used: {ml_used}/{total} ({ml_used/total*100:.1f}%)")
        logger.info(f"- GPT used: {gpt_used}/{total} ({gpt_used/total*100:.1f}%)")
        logger.info(f"- GPT failures: {gpt_failed}/{gpt_used} ({gpt_failed/gpt_used*100:.1f}% of GPT attempts)")

        # Save failures
        failures = self.data[self.data['Predicted Main Category'] == 'Unknown']
        if not failures.empty:
            failures.to_csv(output_dir / 'categorization_failures.csv', index=False)

        # Save final categories
        self.data['Main Category'] = self.data['Predicted Main Category']
        self.data['Sub-category'] = self.data['Predicted Sub-category']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_excel(output_path, index=False)
        logger.info(f"Saved categorized data to {output_path}")
        
        # Save ML models if trained
        if self.ml_categorizer.is_fitted:
            self.ml_categorizer.save_models(Path('data/models'))
            
        print('[DEBUG] Exiting assign_categories')
        
    def perform_clustering(self) -> None:
        print('[DEBUG] Entering perform_clustering')
        if self.data is None:
            raise ValueError("No data loaded")
        # Fit DBSCAN before getting clusters
        self.cluster_analyzer.fit_dbscan(self.data)
        self.clusters = self.cluster_analyzer.get_clusters()
        self.data[CLUSTER_COL] = self.clusters
        # Save clustering model/results
        self.cluster_analyzer.save_cluster_results(self.data, Path('data/models'))
        print('[DEBUG] Exiting perform_clustering')
        
    def generate_visualizations(self, output_dir: Path) -> None:
        print('[DEBUG] Entering generate_visualizations')
        if self.data is None:
            raise ValueError("No data loaded")
        if self.similarity_matrix is None:
            raise ValueError("Must compute similarities first")
        self.visualizer.save_visualization_results(
            self.data,
            self.similarity_matrix,
            output_dir
        )
        # Also save to data/visualizations/
        self.visualizer.save_visualization_results(
            self.data,
            self.similarity_matrix,
            Path('data/visualizations')
        )
        print('[DEBUG] Exiting generate_visualizations')
        
    def save_results(self, output_dir: Path) -> None:
        print('[DEBUG] Entering save_results')
        if self.data is None:
            raise ValueError("No data loaded")
        output_path = output_dir / 'processed_data.xlsx'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_excel(output_path, index=False)
        # Save similarity matrix
        if self.similarity_matrix is not None:
            sim_path = output_dir / 'similarity_matrix.csv'
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.similarity_matrix).to_csv(sim_path, index=False)
        # Save cluster analysis
        if self.clusters is not None:
            cluster_analysis = self.cluster_analyzer.analyze_clusters(self.data)
            ca_path = output_dir / 'cluster_analysis.xlsx'
            ca_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict(cluster_analysis).to_excel(ca_path)
        print('[DEBUG] Exiting save_results')
        
    def save_statistics(self, output_dir: Path, thresholds: Optional[List[float]] = None) -> None:
        print('[DEBUG] Entering save_statistics')
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
        print('[DEBUG] Exiting save_statistics')

    def run_pipeline(
        self,
        input_file: Path,
        output_dir: Path,
        enrich: bool = True,
        cluster: bool = True,
        visualize: bool = True,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        print('[DEBUG] Entering run_pipeline')
        import time
        t0 = time.time()
        try:
            # Load data
            print('[DEBUG] Loading data')
            self.load_data(input_file)
            t1 = time.time(); logger.info(f"Data loading: {t1-t0:.2f}s")
            # Enrich descriptions if requested
            if enrich:
                print('[DEBUG] Enriching descriptions')
                self.enrich_descriptions(output_dir)
            t2 = time.time(); logger.info(f"Description enrichment: {t2-t1:.2f}s")
            # Compute similarities
            print('[DEBUG] Computing similarities')
            self.compute_similarities()
            t3 = time.time(); logger.info(f"Similarity computation: {t3-t2:.2f}s")
            # Assign categories
            print('[DEBUG] Assigning categories')
            self.assign_categories(output_dir)
            t4 = time.time(); logger.info(f"Category assignment: {t4-t3:.2f}s")
            # Perform clustering if requested
            if cluster:
                print('[DEBUG] Performing clustering')
                self.perform_clustering()
            t5 = time.time(); logger.info(f"Clustering: {t5-t4:.2f}s")
            # Generate visualizations if requested
            if visualize:
                print('[DEBUG] Generating visualizations')
                self.generate_visualizations(output_dir)
            t6 = time.time(); logger.info(f"Visualization: {t6-t5:.2f}s")
            # Save results
            print('[DEBUG] Saving results')
            self.save_results(output_dir)
            # Save statistics
            print('[DEBUG] Saving statistics')
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
            print('[DEBUG] Exiting run_pipeline')
            return summary
        except Exception as e:
            print(f'[DEBUG] Exception in run_pipeline: {e}')
            raise
            
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