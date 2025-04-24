"""
Command-line interface for the inventory system.
"""

import click
import logging
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from inventory_system.core.pipeline import InventoryPipeline
from inventory_system.core.data_processor import DataProcessor
import json

# Load environment variables
load_dotenv()

# Get logger for this module
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Inventory System CLI"""
    pass

@cli.command()
@click.option('--data-path', required=True, help='Path to input data file')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed logging')
@click.option('--force', is_flag=True, help='Force reprocessing even if processed data exists')
def run(data_path, verbose, debug, force):
    """Run the inventory processing pipeline"""
    if verbose or debug:
        logger.setLevel(logging.DEBUG)
        # Enable debug logging for all modules
        logging.getLogger('inventory_system').setLevel(logging.DEBUG)
        logging.getLogger('openai').setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Get API key and model from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL', 'gpt-4-turbo')  # Default to gpt-4-turbo if not set
    
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Check if input file exists
        if not os.path.exists(data_path):
            logger.error(f"Input file not found: {data_path}")
            return
            
        # Read the Excel file
        logger.info(f"Reading data from {data_path}")
        try:
            data = pd.read_excel(data_path)
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            return
            
        # Check if data is empty
        if data.empty:
            logger.error("Input data is empty")
            return
            
        # Initialize pipeline
        pipeline = InventoryPipeline(api_key=api_key, model=model)
        
        if force:
            logger.info("Force flag set - will reprocess data even if processed data exists")
            
        # Set and validate data
        pipeline.set_data(data)
        
        # Process the data
        try:
            # Check for existing processed data
            if not force:
                existing_data, stage = pipeline._check_existing_data(data)
                if existing_data is not None:
                    logger.info(f"Found existing {stage} data")
                    if stage == "final":
                        logger.info("Data is already fully processed")
                        logger.info(f"Processed {len(existing_data)} items")
                        return
                    logger.info(f"Will resume processing from {stage} stage")
            
            # Process the data
            result = pipeline.process_data(data)
            logger.info("Pipeline completed successfully")
            logger.info(f"Processed {len(result)} items")
            
            # Show processing statistics
            if verbose:
                logger.info("\nProcessing Statistics:")
                logger.info(f"Total items: {len(result)}")
                if 'Main Category' in result.columns:
                    logger.info(f"Main categories: {result['Main Category'].nunique()}")
                if 'Sub Category' in result.columns:
                    logger.info(f"Sub categories: {result['Sub Category'].nunique()}")
                if 'Category Confidence' in result.columns:
                    logger.info(f"Average confidence: {result['Category Confidence'].mean():.2%}")
            
        except ValueError as e:
            logger.error(f"Data validation error: {str(e)}")
            return
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            if debug:
                logger.exception("Full traceback:")
            return
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if debug:
            logger.exception("Full traceback:")
        return

if __name__ == '__main__':
    cli() 