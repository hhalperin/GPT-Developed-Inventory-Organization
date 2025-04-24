"""
Inventory System Package

This package provides tools for inventory management, including:
- Category prediction
- Description enrichment
- Similarity analysis
- Performance tracking
"""

from .config.config import Config
from .core.workflow import Workflow
from .models.category_predictor import CategoryPredictor
from .analysis.performance_analyzer import PerformanceAnalyzer

__all__ = [
    'Config',
    'Workflow',
    'CategoryPredictor',
    'PerformanceAnalyzer'
]

__version__ = "0.1.0" 