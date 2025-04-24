"""
Performance analysis and visualization module for the category prediction system.

This module provides tools for:
- Analyzing model performance metrics
- Generating visualizations
- Creating performance reports
- Tracking model improvements over time
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy import interp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerformanceAnalyzer:
    """
    Analyzes and visualizes performance metrics for the category prediction system.
    
    This class provides comprehensive analysis of model performance including:
    - Accuracy trends over time
    - Feature importance analysis
    - Confidence score distributions
    - Cross-validation results
    - Model comparison metrics
    - Category-specific metrics
    - Similarity analysis
    - Process metrics
    """
    
    def __init__(self, data: pd.DataFrame, model_dir: str = "models"):
        """
        Initialize the performance analyzer.
        
        Args:
            data: DataFrame containing inventory data
            model_dir: Directory containing model files
        """
        self.data = data
        self.model_dir = model_dir
        self.metrics_history = []
        self._load_metrics_history()
        
    def _load_metrics_history(self) -> None:
        """Load performance metrics from all saved models."""
        try:
            for filename in os.listdir(self.model_dir):
                if filename.endswith('_metadata.json'):
                    with open(os.path.join(self.model_dir, filename), 'r') as f:
                        metadata = json.load(f)
                        self.metrics_history.append({
                            'timestamp': metadata['timestamp'],
                            'metrics': metadata['performance_metrics'],
                            'params': metadata['model_params']
                        })
            self.metrics_history.sort(key=lambda x: x['timestamp'])
            logging.info(f"Loaded metrics from {len(self.metrics_history)} models")
        except Exception as e:
            logging.error(f"Error loading metrics history: {e}")
            
    def analyze_performance(self, output_dir: str = "output/analysis") -> Dict[str, Any]:
        """
        Run comprehensive performance analysis.
        
        Args:
            output_dir: Directory to save analysis results
            
        Returns:
            Dict containing all analysis results
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate all metrics
            metrics = {
                'basic_metrics': self._calculate_basic_metrics(),
                'category_metrics': self._calculate_category_metrics(),
                'similarity_metrics': self._calculate_similarity_metrics(),
                'process_metrics': self._calculate_process_metrics(),
                'model_metrics': self._calculate_model_metrics()
            }
            
            # Generate visualizations
            self.generate_visualizations(output_dir)
            
            # Save metrics to file
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
                
            # Log metrics
            self._log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in performance analysis: {e}")
            raise
            
    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        return {
            'total_items': len(self.data),
            'total_categories': self.data['Main Category'].nunique(),
            'total_subcategories': self.data['Sub Category'].nunique(),
            'category_distribution': self.data['Main Category'].value_counts().to_dict(),
            'subcategory_distribution': self.data['Sub Category'].value_counts().to_dict(),
            'processing_completion': self.data['GPT Processed'].mean()
        }
        
    def _calculate_category_metrics(self) -> Dict[str, Any]:
        """Calculate category-specific metrics."""
        metrics = {}
        for category in self.data['Main Category'].unique():
            category_data = self.data[self.data['Main Category'] == category]
            metrics[category] = {
                'count': len(category_data),
                'subcategories': category_data['Sub Category'].nunique(),
                'avg_confidence': category_data.get('Prediction Confidence', 0).mean(),
                'stability': self._calculate_category_stability(category)
            }
        return metrics
        
    def _calculate_similarity_metrics(self) -> Dict[str, Any]:
        """Calculate similarity-related metrics."""
        return {
            'similarity_distribution': self._analyze_similarity_distribution(),
            'similarity_correlation': self._analyze_similarity_correlation(),
            'threshold_impact': self._analyze_threshold_impact()
        }
        
    def _calculate_process_metrics(self) -> Dict[str, Any]:
        """Calculate process-related metrics."""
        return {
            'iteration_success': self._calculate_iteration_success(),
            'gpt_ml_ratio': self._calculate_gpt_ml_ratio(),
            'alignment_success': self._calculate_alignment_success(),
            'processing_times': self._analyze_processing_times()
        }
        
    def _calculate_model_metrics(self) -> Dict[str, Any]:
        """Calculate model-specific metrics."""
        return {
            'learning_curves': self._analyze_learning_curves(),
            'roc_curves': self._analyze_roc_curves(),
            'calibration': self._analyze_calibration(),
            'convergence': self._analyze_convergence()
        }
        
    def generate_visualizations(self, output_dir: str) -> None:
        """Generate all visualizations."""
        # Category visualizations
        self._plot_category_hierarchy(output_dir)
        self._plot_category_distribution(output_dir)
        self._plot_category_stability(output_dir)
        
        # Model performance visualizations
        self._plot_learning_curves(output_dir)
        self._plot_roc_curves(output_dir)
        self._plot_calibration(output_dir)
        self._plot_convergence(output_dir)
        
        # Similarity visualizations
        self._plot_similarity_matrix(output_dir)
        self._plot_similarity_correlation(output_dir)
        self._plot_threshold_impact(output_dir)
        
        # Process visualizations
        self._plot_iteration_success(output_dir)
        self._plot_gpt_ml_ratio(output_dir)
        self._plot_alignment_success(output_dir)
        self._plot_processing_times(output_dir)
        
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to console."""
        logging.info("\nPerformance Analysis Results:")
        logging.info("=" * 50)
        
        # Basic metrics
        logging.info("\nBasic Metrics:")
        logging.info(f"Total items: {metrics['basic_metrics']['total_items']}")
        logging.info(f"Total categories: {metrics['basic_metrics']['total_categories']}")
        logging.info(f"Total subcategories: {metrics['basic_metrics']['total_subcategories']}")
        logging.info(f"Processing completion: {metrics['basic_metrics']['processing_completion']:.2%}")
        
        # Category metrics
        logging.info("\nCategory Metrics:")
        for category, cat_metrics in metrics['category_metrics'].items():
            logging.info(f"\n{category}:")
            logging.info(f"  Count: {cat_metrics['count']}")
            logging.info(f"  Subcategories: {cat_metrics['subcategories']}")
            logging.info(f"  Average confidence: {cat_metrics['avg_confidence']:.2f}")
            logging.info(f"  Stability: {cat_metrics['stability']:.2f}")
            
        # Process metrics
        logging.info("\nProcess Metrics:")
        logging.info(f"Iteration success rate: {metrics['process_metrics']['iteration_success']:.2%}")
        logging.info(f"GPT vs ML ratio: {metrics['process_metrics']['gpt_ml_ratio']:.2f}")
        logging.info(f"Alignment success rate: {metrics['process_metrics']['alignment_success']:.2%}")
        
        # Model metrics
        logging.info("\nModel Metrics:")
        logging.info(f"Model convergence: {metrics['model_metrics']['convergence']:.2f}")
        
    # Additional helper methods for specific analyses
    def _calculate_category_stability(self, category: str) -> float:
        """Calculate category stability over iterations."""
        # Implementation details
        return 0.0  # Placeholder
        
    def _analyze_similarity_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of similarity scores."""
        # Implementation details
        return {}  # Placeholder
        
    def _analyze_similarity_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between different similarity metrics."""
        # Implementation details
        return {}  # Placeholder
        
    def _analyze_threshold_impact(self) -> Dict[str, Any]:
        """Analyze impact of similarity thresholds on performance."""
        # Implementation details
        return {}  # Placeholder
        
    def _calculate_iteration_success(self) -> float:
        """Calculate success rate of iterations."""
        # Implementation details
        return 0.0  # Placeholder
        
    def _calculate_gpt_ml_ratio(self) -> float:
        """Calculate ratio of GPT vs ML model usage."""
        # Implementation details
        return 0.0  # Placeholder
        
    def _calculate_alignment_success(self) -> float:
        """Calculate success rate of category alignments."""
        # Implementation details
        return 0.0  # Placeholder
        
    def _analyze_processing_times(self) -> Dict[str, Any]:
        """Analyze processing times."""
        # Implementation details
        return {}  # Placeholder
        
    def _analyze_learning_curves(self) -> Dict[str, Any]:
        """Analyze learning curves."""
        # Implementation details
        return {}  # Placeholder
        
    def _analyze_roc_curves(self) -> Dict[str, Any]:
        """Analyze ROC curves."""
        # Implementation details
        return {}  # Placeholder
        
    def _analyze_calibration(self) -> Dict[str, Any]:
        """Analyze model calibration."""
        # Implementation details
        return {}  # Placeholder
        
    def _analyze_convergence(self) -> float:
        """Analyze model convergence."""
        # Implementation details
        return 0.0  # Placeholder
        
    # Visualization methods
    def _plot_category_hierarchy(self, output_dir: str) -> None:
        """Plot category hierarchy."""
        # Implementation details
        pass
        
    def _plot_category_distribution(self, output_dir: str) -> None:
        """Plot category distribution."""
        # Implementation details
        pass
        
    def _plot_category_stability(self, output_dir: str) -> None:
        """Plot category stability."""
        # Implementation details
        pass
        
    def _plot_learning_curves(self, output_dir: str) -> None:
        """Plot learning curves."""
        # Implementation details
        pass
        
    def _plot_roc_curves(self, output_dir: str) -> None:
        """Plot ROC curves."""
        # Implementation details
        pass
        
    def _plot_calibration(self, output_dir: str) -> None:
        """Plot calibration curves."""
        # Implementation details
        pass
        
    def _plot_convergence(self, output_dir: str) -> None:
        """Plot convergence analysis."""
        # Implementation details
        pass
        
    def _plot_similarity_matrix(self, output_dir: str) -> None:
        """Plot similarity matrix."""
        # Implementation details
        pass
        
    def _plot_similarity_correlation(self, output_dir: str) -> None:
        """Plot similarity correlation."""
        # Implementation details
        pass
        
    def _plot_threshold_impact(self, output_dir: str) -> None:
        """Plot threshold impact analysis."""
        # Implementation details
        pass
        
    def _plot_iteration_success(self, output_dir: str) -> None:
        """Plot iteration success rates."""
        # Implementation details
        pass
        
    def _plot_gpt_ml_ratio(self, output_dir: str) -> None:
        """Plot GPT vs ML ratio."""
        # Implementation details
        pass
        
    def _plot_alignment_success(self, output_dir: str) -> None:
        """Plot alignment success rates."""
        # Implementation details
        pass
        
    def _plot_processing_times(self, output_dir: str) -> None:
        """Plot processing times."""
        # Implementation details
        pass 