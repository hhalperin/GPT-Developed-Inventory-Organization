"""
Metrics monitoring module for the inventory system.
Tracks system performance and health metrics.
"""

import logging
import time
from typing import Dict, Any, List
from datetime import datetime
import json
import os
from ..config.settings import MODEL_DIR

class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics: Dict[str, Any] = {
            "api_calls": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "latency": []
            },
            "processing": {
                "items_processed": 0,
                "processing_time": 0,
                "batch_size": 0
            },
            "model": {
                "predictions": 0,
                "accuracy": 0,
                "confidence": []
            },
            "system": {
                "memory_usage": 0,
                "cpu_usage": 0,
                "disk_usage": 0
            }
        }
        self.start_time = time.time()
        self.metrics_file = os.path.join(MODEL_DIR, "metrics.json")

    def record_api_call(self, success: bool, latency: float) -> None:
        """
        Record an API call metric.
        
        Args:
            success: Whether the API call was successful
            latency: API call latency in seconds
        """
        self.metrics["api_calls"]["total"] += 1
        if success:
            self.metrics["api_calls"]["successful"] += 1
        else:
            self.metrics["api_calls"]["failed"] += 1
        self.metrics["api_calls"]["latency"].append(latency)

    def record_processing(self, items_processed: int, processing_time: float) -> None:
        """
        Record processing metrics.
        
        Args:
            items_processed: Number of items processed
            processing_time: Time taken for processing in seconds
        """
        self.metrics["processing"]["items_processed"] += items_processed
        self.metrics["processing"]["processing_time"] += processing_time
        self.metrics["processing"]["batch_size"] = items_processed

    def record_prediction(self, confidence: float, correct: bool) -> None:
        """
        Record model prediction metrics.
        
        Args:
            confidence: Prediction confidence score
            correct: Whether the prediction was correct
        """
        self.metrics["model"]["predictions"] += 1
        if correct:
            self.metrics["model"]["accuracy"] = (
                (self.metrics["model"]["accuracy"] * (self.metrics["model"]["predictions"] - 1) + 1) /
                self.metrics["model"]["predictions"]
            )
        else:
            self.metrics["model"]["accuracy"] = (
                self.metrics["model"]["accuracy"] * (self.metrics["model"]["predictions"] - 1) /
                self.metrics["model"]["predictions"]
            )
        self.metrics["model"]["confidence"].append(confidence)

    def record_system_metrics(self, memory_usage: float, cpu_usage: float, disk_usage: float) -> None:
        """
        Record system resource metrics.
        
        Args:
            memory_usage: Memory usage percentage
            cpu_usage: CPU usage percentage
            disk_usage: Disk usage percentage
        """
        self.metrics["system"]["memory_usage"] = memory_usage
        self.metrics["system"]["cpu_usage"] = cpu_usage
        self.metrics["system"]["disk_usage"] = disk_usage

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        # Calculate average latency
        if self.metrics["api_calls"]["latency"]:
            avg_latency = sum(self.metrics["api_calls"]["latency"]) / len(self.metrics["api_calls"]["latency"])
            self.metrics["api_calls"]["average_latency"] = avg_latency
        
        # Calculate items per second
        total_time = time.time() - self.start_time
        if total_time > 0:
            self.metrics["processing"]["items_per_second"] = (
                self.metrics["processing"]["items_processed"] / total_time
            )
        
        return self.metrics

    def save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.get_metrics()
            }
            
            # Create metrics directory if it doesn't exist
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            
            # Save metrics to file
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logging.info(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")

    def load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from file.
        
        Returns:
            Dictionary containing saved metrics
        """
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")
            return {}

    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        self.metrics = {
            "api_calls": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "latency": []
            },
            "processing": {
                "items_processed": 0,
                "processing_time": 0,
                "batch_size": 0
            },
            "model": {
                "predictions": 0,
                "accuracy": 0,
                "confidence": []
            },
            "system": {
                "memory_usage": 0,
                "cpu_usage": 0,
                "disk_usage": 0
            }
        }
        self.start_time = time.time()
        logging.info("Metrics reset") 