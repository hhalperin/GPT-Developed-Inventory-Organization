"""
Model monitoring and evaluation system.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    timestamp: str
    n_samples: int
    confusion_matrix: np.ndarray
    class_report: Dict[str, Dict[str, float]]

class ModelMonitor:
    """Monitors model performance and detects drift."""
    
    def __init__(self, output_dir: str = "monitoring"):
        """
        Initialize the model monitor.
        
        Args:
            output_dir: Directory to store monitoring data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        
    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           n_samples: int) -> PerformanceMetrics:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            n_samples: Number of samples evaluated
            
        Returns:
            PerformanceMetrics object
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create metrics object
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            timestamp=datetime.now().isoformat(),
            n_samples=n_samples,
            confusion_matrix=cm,
            class_report=report
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
        
    def detect_drift(self, window_size: int = 10, threshold: float = 0.1) -> bool:
        """
        Detect performance drift.
        
        Args:
            window_size: Number of recent metrics to consider
            threshold: Performance degradation threshold
            
        Returns:
            bool: Whether drift was detected
        """
        if len(self.metrics_history) < window_size:
            return False
            
        # Get recent metrics
        recent_metrics = self.metrics_history[-window_size:]
        
        # Calculate average performance
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        avg_f1 = np.mean([m.f1 for m in recent_metrics])
        
        # Calculate baseline performance
        baseline_metrics = self.metrics_history[:-window_size]
        if baseline_metrics:
            baseline_accuracy = np.mean([m.accuracy for m in baseline_metrics])
            baseline_f1 = np.mean([m.f1 for m in baseline_metrics])
            
            # Check for significant degradation
            accuracy_drift = (baseline_accuracy - avg_accuracy) > threshold
            f1_drift = (baseline_f1 - avg_f1) > threshold
            
            if accuracy_drift or f1_drift:
                self.logger.warning(
                    f"Performance drift detected: "
                    f"Accuracy: {baseline_accuracy:.3f} -> {avg_accuracy:.3f}, "
                    f"F1: {baseline_f1:.3f} -> {avg_f1:.3f}"
                )
                return True
                
        return False
        
    def analyze_category_stability(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze category stability and consistency.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with stability analysis
        """
        # Get unique categories
        categories = np.unique(np.concatenate([y_true, y_pred]))
        
        # Calculate category-wise metrics
        stability_analysis = {}
        for category in categories:
            # Get category-specific metrics
            mask = y_true == category
            if np.any(mask):
                accuracy = accuracy_score(y_true[mask], y_pred[mask])
                precision = precision_score(y_true[mask], y_pred[mask], average='weighted')
                recall = recall_score(y_true[mask], y_pred[mask], average='weighted')
                f1 = f1_score(y_true[mask], y_pred[mask], average='weighted')
                
                stability_analysis[category] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'n_samples': np.sum(mask)
                }
                
        return stability_analysis
        
    def _save_metrics(self, metrics: PerformanceMetrics) -> None:
        """Save performance metrics to file."""
        # Convert metrics to dictionary
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1': metrics.f1,
            'timestamp': metrics.timestamp,
            'n_samples': metrics.n_samples,
            'confusion_matrix': metrics.confusion_matrix.tolist(),
            'class_report': metrics.class_report
        }
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        self.logger.info(f"Metrics saved to {metrics_file}")
        
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            str: Report content
        """
        if not self.metrics_history:
            return "No metrics available for report generation."
            
        # Get latest metrics
        latest = self.metrics_history[-1]
        
        # Generate report
        report = [
            "Model Performance Report",
            "======================",
            f"Timestamp: {latest.timestamp}",
            f"Number of samples: {latest.n_samples}",
            "",
            "Overall Metrics:",
            f"Accuracy: {latest.accuracy:.3f}",
            f"Precision: {latest.precision:.3f}",
            f"Recall: {latest.recall:.3f}",
            f"F1 Score: {latest.f1:.3f}",
            "",
            "Category-wise Performance:",
        ]
        
        # Add category-wise metrics
        for category, metrics in latest.class_report.items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                report.extend([
                    f"\nCategory: {category}",
                    f"Precision: {metrics['precision']:.3f}",
                    f"Recall: {metrics['recall']:.3f}",
                    f"F1 Score: {metrics['f1-score']:.3f}",
                    f"Support: {metrics['support']}"
                ])
                
        # Add drift analysis
        drift_detected = self.detect_drift()
        report.extend([
            "",
            "Drift Analysis:",
            f"Drift detected: {drift_detected}"
        ])
        
        report_content = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
                
        return report_content 