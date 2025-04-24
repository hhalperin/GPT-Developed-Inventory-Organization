"""
Centralized logging configuration for the inventory system.
"""

import logging
import os
from pathlib import Path
import portalocker
import time

def setup_logging():
    """
    Set up logging configuration for the application.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "inventory_system.log"),
            logging.StreamHandler()
        ]
    )
    
    # Suppress OpenAI client messages
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    # Get logger for this module
    logger = logging.getLogger('inventory_system')
    
    return logger 