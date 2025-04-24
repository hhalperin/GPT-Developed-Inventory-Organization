"""
Configuration settings for the inventory categorization system.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
from pathlib import Path

@dataclass
class GPTConfig:
    """Configuration for GPT processing."""
    model_name: str = os.getenv('OPENAI_MODEL', 'gpt-4-turbo')
    temperature: float = 0.7
    max_tokens: int = 150
    batch_size: int = 10
    delay_between_batches: float = 1.0
    description_prompt: str = "Enrich the following product description with additional details: {description}"
    category_prompt: str = "Categorize the following product into main and sub categories: {description}"

@dataclass
class SimilarityConfig:
    """Configuration for similarity analysis."""
    text_model_name: str = "all-MiniLM-L6-v2"
    n_neighbors: int = 5
    min_samples_for_training: int = 50
    retrain_interval_hours: int = 24
    min_samples_for_retraining: int = 10
    main_category_threshold: float = 0.8
    sub_category_threshold: float = 0.9
    catalog_no_weight: float = 0.3
    description_weight: float = 0.7

@dataclass
class ModelConfig:
    """Configuration for ensemble model."""
    n_neighbors: int = 5
    n_estimators: int = 100
    hidden_layer_sizes: Tuple[int, ...] = (100, 50)
    calibration_method: str = "isotonic"
    cv_folds: int = 5
    min_samples_leaf: int = 5
    confidence_threshold: float = 0.8
    online_learning: bool = True
    batch_size: int = 100

@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""
    metrics_window_size: int = 100
    drift_threshold: float = 0.1
    performance_report_interval: int = 24  # hours
    log_level: str = "INFO"
    checkpoint_interval: int = 100  # items

@dataclass
class SystemConfig:
    """Main system configuration."""
    gpt: GPTConfig = GPTConfig()
    similarity: SimilarityConfig = SimilarityConfig()
    model: ModelConfig = ModelConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Directory paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models"
    monitoring_dir: str = "monitoring"
    reports_dir: str = "reports"
    checkpoints_dir: str = "checkpoints"
    
    # File patterns
    input_file_pattern: str = "*.csv"
    output_file_suffix: str = "_processed"
    
    # Environment variables
    openai_api_key_env: str = "OPENAI_API_KEY"
    
    def __post_init__(self):
        """Create necessary directories and validate settings."""
        self._create_directories()
        self._validate_settings()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.monitoring_dir,
            self.reports_dir,
            self.checkpoints_dir
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_settings(self):
        """Validate configuration settings."""
        if self.gpt.temperature < 0 or self.gpt.temperature > 1:
            raise ValueError("GPT temperature must be between 0 and 1")
        
        if self.similarity.main_category_threshold < 0 or self.similarity.main_category_threshold > 1:
            raise ValueError("Main category threshold must be between 0 and 1")
        
        if self.similarity.sub_category_threshold < 0 or self.similarity.sub_category_threshold > 1:
            raise ValueError("Sub category threshold must be between 0 and 1")
        
        if self.model.confidence_threshold < 0 or self.model.confidence_threshold > 1:
            raise ValueError("Model confidence threshold must be between 0 and 1")

# Default configuration
default_config = SystemConfig() 