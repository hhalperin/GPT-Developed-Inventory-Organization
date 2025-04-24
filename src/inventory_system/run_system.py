"""
Main script to run the inventory categorization system.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import json

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from inventory_system.models.gpt_processor import GPTProcessor
from inventory_system.models.similarity_analyzer import SimilarityAnalyzer
from inventory_system.models.similarity_features import SimilarityConfig
from inventory_system.models.ensemble_classifier import ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inventory_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the system."""
    directories = [
        'data/input',
        'data/processed',
        'models',
        'monitoring',
        'reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_processed_data(input_file: str) -> bool:
    """Check if data has already been fully processed."""
    input_name = Path(input_file).stem
    processed_path = Path('data/processed') / f"{input_name}_processed.csv"
    
    # Check if we have a final processed file
    if processed_path.exists():
        logger.info(f"Found final processed file: {processed_path}")
        return True
        
    # Check for progress files
    output_dir = Path('data/output')
    progress_files = sorted(output_dir.glob('progress_*.xlsx'))
    
    if progress_files:
        latest_progress = progress_files[-1]
        progress_data = pd.read_excel(latest_progress)
        
        # Check if all items are processed
        if 'GPT Processed' in progress_data.columns:
            if progress_data['GPT Processed'].all():
                logger.info(f"Found complete progress file: {latest_progress}")
                return True
            else:
                logger.info(f"Found incomplete progress file: {latest_progress}")
                return False
                
    logger.info("No processed data found")
    return False

def load_data(input_file: str) -> pd.DataFrame:
    """Load and preprocess input data."""
    logger.info(f"Loading data from {input_file}")
    
    try:
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            data = pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file}")
            
        # Basic preprocessing
        data = data.fillna('')
        logger.info(f"Loaded {len(data)} rows from {input_file}")
        logger.info(f"Columns: {', '.join(data.columns)}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def process_data(data: pd.DataFrame, gpt_processor: GPTProcessor, analyzer: SimilarityAnalyzer) -> pd.DataFrame:
    """Process data using GPT and similarity analysis."""
    try:
        # First check for completed data in output directory
        output_dir = Path('data/output')
        completed_files = sorted(output_dir.glob('enriched_descriptions_*.xlsx'))
        
        if completed_files:
            latest_completed = completed_files[-1]
            logger.info(f"Found completed data file: {latest_completed}")
            completed_data = pd.read_excel(latest_completed)
            
            # Log the columns we found in the completed data
            logger.info(f"Columns in completed data: {', '.join(completed_data.columns)}")
            
            if 'CatalogNo' in completed_data.columns:
                # Create mappings for all processed data, checking if columns exist
                processed_map = {}
                
                # Define the columns we want to map and their possible alternative names
                column_mappings = {
                    'Enriched Description': ['Enriched Description', 'Description'],
                    'Main Category': ['Main Category', 'Category'],
                    'Sub Category': ['Sub Category', 'SubCategory'],
                    'GPT Processed': ['GPT Processed', 'Processed']
                }
                
                # Find the correct column names
                for target_col, possible_names in column_mappings.items():
                    for col_name in possible_names:
                        if col_name in completed_data.columns:
                            processed_map[target_col] = dict(zip(completed_data['CatalogNo'], completed_data[col_name]))
                            logger.info(f"Using column '{col_name}' for {target_col}")
                            break
                
                if not processed_map:
                    logger.warning("No matching columns found in completed data")
                    return data
                
                # Update current data with completed values
                for col, mapping in processed_map.items():
                    if col in data.columns:
                        data[col] = data['CatalogNo'].map(mapping).fillna(data[col])
                
                logger.info("Using completed data from output directory")
            
        # If no completed data found, check progress files
        elif progress_files := sorted(output_dir.glob('progress_*.xlsx')):
            latest_progress = progress_files[-1]
            logger.info(f"Found progress file: {latest_progress}")
            progress_data = pd.read_excel(latest_progress)
            
            # Log the columns we found in the progress data
            logger.info(f"Columns in progress data: {', '.join(progress_data.columns)}")
            
            if 'CatalogNo' in progress_data.columns:
                # Create mappings for processed data, checking if columns exist
                processed_map = {}
                
                # Define the columns we want to map and their possible alternative names
                column_mappings = {
                    'Enriched Description': ['Enriched Description', 'Description'],
                    'Main Category': ['Main Category', 'Category'],
                    'Sub Category': ['Sub Category', 'SubCategory'],
                    'GPT Processed': ['GPT Processed', 'Processed']
                }
                
                # Find the correct column names
                for target_col, possible_names in column_mappings.items():
                    for col_name in possible_names:
                        if col_name in progress_data.columns:
                            processed_map[target_col] = dict(zip(progress_data['CatalogNo'], progress_data[col_name]))
                            logger.info(f"Using column '{col_name}' for {target_col}")
                            break
                
                if not processed_map:
                    logger.warning("No matching columns found in progress data")
                    return data
                
                # Update current data with processed values
                for col, mapping in processed_map.items():
                    if col in data.columns:
                        data[col] = data['CatalogNo'].map(mapping).fillna(data[col])
                
                logger.info("Using progress data from output directory")
                
        else:
            # If no existing data found, process all items
            logger.info("No existing data found, processing all items")
            data = gpt_processor.process_batch(data)
        
        # Check if we need to categorize the items
        if 'Main Category' not in data.columns or 'Sub Category' not in data.columns:
            logger.info("Categories not found, running categorization")
            data = gpt_processor.process_batch(data)
        else:
            # Check if any items are missing categories
            missing_categories = data[data['Main Category'].isna() | data['Sub Category'].isna()]
            if not missing_categories.empty:
                logger.info(f"Found {len(missing_categories)} items missing categories, processing them")
                data = gpt_processor.process_batch(data)
            else:
                logger.info("All items already have categories, skipping categorization")
        
        # Run analysis and visualization steps
        logger.info("Starting similarity and category analysis")
        try:
            data = train_and_analyze(data, analyzer)
        except Exception as e:
            logger.error(f"Error during similarity analysis: {e}")
            # Try to continue with the data we have
            return data
        
        return data
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def save_processed_data(data: pd.DataFrame, output_file: str):
    """Save processed data to output file."""
    try:
        logger.info(f"Saving processed data to {output_file}")
        data.to_excel(output_file, index=False)
        
        # Log summary statistics
        total_items = len(data)
        processed_items = data['GPT Processed'].sum() if 'GPT Processed' in data.columns else 0
        
        # Only calculate category statistics if the columns exist
        if 'Main Category' in data.columns and 'Sub Category' in data.columns:
            unique_categories = data['Main Category'].nunique()
            logger.info(f"Summary Statistics:")
            logger.info(f"Total Items: {total_items}")
            logger.info(f"Processed Items: {processed_items}")
            logger.info(f"Unique Categories: {unique_categories}")
        else:
            logger.info(f"Summary Statistics:")
            logger.info(f"Total Items: {total_items}")
            logger.info(f"Processed Items: {processed_items}")
            logger.info("Categories not yet processed")
            
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

def train_and_analyze(data: pd.DataFrame, analyzer: SimilarityAnalyzer) -> pd.DataFrame:
    """Train models and analyze similarities."""
    logger.info("Starting similarity analysis and model training")
    
    # Calculate similarity features
    logger.info("Calculating similarity features")
    try:
        similarity_features = analyzer.calculate_similarities(data)
        
        # Save similarity features
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        similarity_features.to_csv(f'data/processed/similarity_features_{timestamp}.csv', index=False)
        logger.info(f"Saved similarity features to data/processed/similarity_features_{timestamp}.csv")
        
        # First pass training
        logger.info("First pass training")
        metrics = analyzer.train_model(data, similarity_features, is_second_pass=False)
        logger.info(f"First pass metrics: {metrics}")
        
        # Save first pass model and metrics
        analyzer.save_model(f'models/first_pass_model_{timestamp}.pkl')
        logger.info(f"Saved first pass model to models/first_pass_model_{timestamp}.pkl")
        
        # Save first pass metrics
        with open(f'models/first_pass_metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f)
        logger.info(f"Saved first pass metrics to models/first_pass_metrics_{timestamp}.json")
        
        # Second pass alignment
        logger.info("Second pass alignment")
        aligned_data = analyzer.align_categories(data, similarity_features)
        
        # Save aligned data
        aligned_data.to_csv(f'data/processed/aligned_data_{timestamp}.csv', index=False)
        logger.info(f"Saved aligned data to data/processed/aligned_data_{timestamp}.csv")
        
        # Save second pass model and metrics
        analyzer.save_model(f'models/second_pass_model_{timestamp}.pkl')
        logger.info(f"Saved second pass model to models/second_pass_model_{timestamp}.pkl")
        
        # Generate and save visualizations
        try:
            from inventory_system.analysis.performance_analyzer import PerformanceAnalyzer
            performance_analyzer = PerformanceAnalyzer(aligned_data)
            
            # Save accuracy trend plot
            performance_analyzer.plot_accuracy_trend(f'monitoring/accuracy_trend_{timestamp}.png')
            
            # Save feature importance plot
            performance_analyzer.plot_feature_importance(f'monitoring/feature_importance_{timestamp}.png')
            
            # Save confidence distribution plot
            performance_analyzer.plot_confidence_distribution(
                aligned_data['Alignment Confidence'].values,
                f'monitoring/confidence_distribution_{timestamp}.png'
            )
            
            # Save confusion matrix plot
            performance_analyzer.plot_confusion_matrix(
                aligned_data['Main Category'].values,
                aligned_data['Aligned Category'].values,
                f'monitoring/confusion_matrix_{timestamp}.png'
            )
            
            logger.info("Generated and saved performance visualizations")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        # Generate and save performance report
        report = analyzer.generate_performance_report(
            f'reports/performance_report_{timestamp}.txt'
        )
        logger.info(f"Performance report generated: {report}")
        
        return aligned_data
    except Exception as e:
        logger.error(f"Error during similarity analysis: {e}")
        # Return the original data if analysis fails
        return data

def main():
    """Main function to run the system."""
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Setup directories
        setup_directories()
        
        # Initialize components
        logger.info("Initializing system components")
        gpt_processor = GPTProcessor()
        analyzer = SimilarityAnalyzer()
        
        # Process input files
        input_dir = Path('data/input')
        input_file = input_dir / 'input.xlsx'
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return
            
        # Load data
        data = load_data(str(input_file))
        
        # Process with GPT and analyze
        processed_data = process_data(data, gpt_processor, analyzer)
        
        logger.info("System run completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
            
if __name__ == "__main__":
    main() 