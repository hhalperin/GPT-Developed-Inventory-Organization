#!/usr/bin/env python3
"""Script to run the inventory analysis pipeline."""

import argparse
import logging
from pathlib import Path
import os
import pandas as pd

from src.pipeline import InventoryPipeline
from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_DBSCAN_EPS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_KMEANS_CLUSTERS
)

def main():
    print("[DEBUG] Starting main() in run_inventory.py")
    # --- Auto-create subset if needed ---
    subset_path = "data/inventory_data_subset.xlsx"
    full_path = "data/inventory_data.xlsx"
    if not os.path.exists(subset_path):
        if os.path.exists(full_path):
            print(f"Creating subset file: {subset_path} from {full_path} (first 1000 rows)")
            df = pd.read_excel(full_path)
            df.head(1000).to_excel(subset_path, index=False)
        else:
            print(f"ERROR: {full_path} does not exist. Please provide your inventory data.")
            return
    # --- End auto-create block ---
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run inventory analysis pipeline")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input data file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
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
        "--no-enrich",
        action="store_true",
        help="Skip description enrichment"
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Skip clustering analysis"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (optional, can also be set in .env file as OPENAI_API_KEY)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create pipeline
    print("[DEBUG] Creating InventoryPipeline")
    pipeline = InventoryPipeline(
        similarity_threshold=args.similarity_threshold,
        dbscan_eps=args.dbscan_eps,
        min_samples=args.min_samples,
        kmeans_clusters=args.kmeans_clusters,
        gpt_api_key=args.api_key
    )
    
    # Run pipeline
    print("[DEBUG] Running pipeline.run_pipeline()")
    print(f"[DEBUG] Input file: {args.input_file}")
    print(f"[DEBUG] Output dir: {args.output_dir}")
    print(f"[DEBUG] Enrich: {not args.no_enrich}, Cluster: {not args.no_cluster}, Visualize: {not args.no_visualize}")
    stats = pipeline.run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        enrich=not args.no_enrich,
        cluster=not args.no_cluster,
        visualize=not args.no_visualize
    )
    print("[DEBUG] Pipeline run complete.")
    
    # Print statistics
    print("\nPipeline Statistics:")
    print(f"Total items processed: {stats.get('total_items')}")
    print(f"Unique categories: {stats.get('unique_categories')}")
    print(f"Unique subcategories: {stats.get('unique_subcategories')}")
    if stats.get('clusters') is not None:
        print(f"Number of clusters: {stats['clusters']}")
    print("[DEBUG] End of main() in run_inventory.py")

if __name__ == "__main__":
    main() 