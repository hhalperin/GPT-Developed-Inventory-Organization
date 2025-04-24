import pandas as pd
import numpy as np
import logging
from pathlib import Path

def analyze_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze the input data and generate basic statistics.
    
    Args:
        df: Input DataFrame to analyze
        output_dir: Directory to save analysis results
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Print basic information
        logging.info("\nDataset Info:")
        logging.info("-" * 50)
        logging.info(f"Number of rows: {len(df)}")
        logging.info(f"Number of columns: {len(df.columns)}")
        logging.info("\nColumns:")
        logging.info("-" * 50)
        for col in df.columns:
            logging.info(f"- {col}")
        
        # Save basic statistics to a text file
        with open(output_path / "data_analysis.txt", "w") as f:
            f.write("Dataset Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of rows: {len(df)}\n")
            f.write(f"Number of columns: {len(df.columns)}\n\n")
            f.write("Columns:\n")
            f.write("-" * 50 + "\n")
            for col in df.columns:
                f.write(f"- {col}\n")
            
            f.write("\nData Types and Missing Values:\n")
            f.write("-" * 50 + "\n")
            f.write(df.info().to_string())
        
        logging.info(f"Analysis results saved to {output_path / 'data_analysis.txt'}")
        
    except Exception as e:
        logging.error(f"Error in data analysis: {str(e)}")
        raise 