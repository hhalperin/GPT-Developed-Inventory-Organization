"""
Test suite for the metrics monitoring module.
"""

import unittest
import os
import json
import time
from datetime import datetime
from inventory_system.monitoring.metrics import MetricsCollector

class TestMetricsCollector(unittest.TestCase):
    """Test cases for the MetricsCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = MetricsCollector()
        self.test_metrics_file = os.path.join("tests", "test_metrics.json")
        
        # Create test directory if it doesn't exist
        os.makedirs(os.path.dirname(self.test_metrics_file), exist_ok=True)

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_metrics_file):
            os.remove(self.test_metrics_file)

    def test_initialization(self):
        """Test metrics collector initialization."""
        self.assertIsNotNone(self.metrics)
        self.assertEqual(self.metrics.metrics["api_calls"]["total"], 0)
        self.assertEqual(self.metrics.metrics["processing"]["items_processed"], 0)
        self.assertEqual(self.metrics.metrics["model"]["predictions"], 0)
        self.assertEqual(self.metrics.metrics["system"]["memory_usage"], 0)

    def test_record_api_call(self):
        """Test API call recording."""
        # Test successful API call
        self.metrics.record_api_call(True, 0.5)
        self.assertEqual(self.metrics.metrics["api_calls"]["total"], 1)
        self.assertEqual(self.metrics.metrics["api_calls"]["successful"], 1)
        self.assertEqual(self.metrics.metrics["api_calls"]["failed"], 0)
        self.assertEqual(len(self.metrics.metrics["api_calls"]["latency"]), 1)
        
        # Test failed API call
        self.metrics.record_api_call(False, 0.3)
        self.assertEqual(self.metrics.metrics["api_calls"]["total"], 2)
        self.assertEqual(self.metrics.metrics["api_calls"]["successful"], 1)
        self.assertEqual(self.metrics.metrics["api_calls"]["failed"], 1)
        self.assertEqual(len(self.metrics.metrics["api_calls"]["latency"]), 2)

    def test_record_processing(self):
        """Test processing metrics recording."""
        self.metrics.record_processing(100, 5.0)
        self.assertEqual(self.metrics.metrics["processing"]["items_processed"], 100)
        self.assertEqual(self.metrics.metrics["processing"]["processing_time"], 5.0)
        self.assertEqual(self.metrics.metrics["processing"]["batch_size"], 100)

    def test_record_prediction(self):
        """Test prediction metrics recording."""
        # Test correct prediction
        self.metrics.record_prediction(0.9, True)
        self.assertEqual(self.metrics.metrics["model"]["predictions"], 1)
        self.assertEqual(self.metrics.metrics["model"]["accuracy"], 1.0)
        self.assertEqual(len(self.metrics.metrics["model"]["confidence"]), 1)
        
        # Test incorrect prediction
        self.metrics.record_prediction(0.8, False)
        self.assertEqual(self.metrics.metrics["model"]["predictions"], 2)
        self.assertEqual(self.metrics.metrics["model"]["accuracy"], 0.5)
        self.assertEqual(len(self.metrics.metrics["model"]["confidence"]), 2)

    def test_record_system_metrics(self):
        """Test system metrics recording."""
        self.metrics.record_system_metrics(0.7, 0.5, 0.3)
        self.assertEqual(self.metrics.metrics["system"]["memory_usage"], 0.7)
        self.assertEqual(self.metrics.metrics["system"]["cpu_usage"], 0.5)
        self.assertEqual(self.metrics.metrics["system"]["disk_usage"], 0.3)

    def test_get_metrics(self):
        """Test metrics retrieval."""
        # Record some metrics
        self.metrics.record_api_call(True, 0.5)
        self.metrics.record_processing(100, 5.0)
        self.metrics.record_prediction(0.9, True)
        self.metrics.record_system_metrics(0.7, 0.5, 0.3)
        
        # Get metrics
        metrics = self.metrics.get_metrics()
        
        # Check metrics structure
        self.assertIn("api_calls", metrics)
        self.assertIn("processing", metrics)
        self.assertIn("model", metrics)
        self.assertIn("system", metrics)
        
        # Check calculated metrics
        self.assertIn("average_latency", metrics["api_calls"])
        self.assertIn("items_per_second", metrics["processing"])

    def test_save_metrics(self):
        """Test metrics saving."""
        # Record some metrics
        self.metrics.record_api_call(True, 0.5)
        self.metrics.record_processing(100, 5.0)
        
        # Save metrics
        self.metrics.metrics_file = self.test_metrics_file
        self.metrics.save_metrics()
        
        # Check if file exists and contains valid JSON
        self.assertTrue(os.path.exists(self.test_metrics_file))
        with open(self.test_metrics_file, 'r') as f:
            saved_metrics = json.load(f)
            self.assertIn("timestamp", saved_metrics)
            self.assertIn("metrics", saved_metrics)

    def test_load_metrics(self):
        """Test metrics loading."""
        # Create test metrics file
        test_metrics = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "api_calls": {
                    "total": 10,
                    "successful": 8,
                    "failed": 2,
                    "latency": [0.5, 0.6]
                }
            }
        }
        with open(self.test_metrics_file, 'w') as f:
            json.dump(test_metrics, f)
        
        # Load metrics
        self.metrics.metrics_file = self.test_metrics_file
        loaded_metrics = self.metrics.load_metrics()
        
        # Check loaded metrics
        self.assertEqual(loaded_metrics["metrics"]["api_calls"]["total"], 10)
        self.assertEqual(loaded_metrics["metrics"]["api_calls"]["successful"], 8)
        self.assertEqual(loaded_metrics["metrics"]["api_calls"]["failed"], 2)

    def test_reset_metrics(self):
        """Test metrics reset."""
        # Record some metrics
        self.metrics.record_api_call(True, 0.5)
        self.metrics.record_processing(100, 5.0)
        
        # Reset metrics
        self.metrics.reset_metrics()
        
        # Check if metrics are reset
        self.assertEqual(self.metrics.metrics["api_calls"]["total"], 0)
        self.assertEqual(self.metrics.metrics["processing"]["items_processed"], 0)
        self.assertEqual(self.metrics.metrics["model"]["predictions"], 0)
        self.assertEqual(self.metrics.metrics["system"]["memory_usage"], 0)

if __name__ == "__main__":
    unittest.main() 