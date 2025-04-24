"""
Data processing module for the inventory system.
Handles data loading, cleaning, and validation.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from datetime import datetime

class DataProcessor:
    """Handles data loading, cleaning, and validation."""
    
    # Manufacturer code to full name mapping
    MFR_DICT: Dict[str, str] = {
        "DOT": "Dottie",
        "CH": "Eaton",
        "BLINE": "Cooper B-Line",
        "MIL": "Milbank",
        "LEV": "Leviton",
        "ITE": "Siemens",
        "GEIND": "General Electric Industrial",
        "UNIPA": "Union Pacific",
        "GARV": "Garvin Industries",
        "FIT": "American Fittings",
        "TAY": "TayMac",
        "ARL": "Arlington",
        "AMFI": "American Fittings",
        "BPT": "Bridgeport",
        "CCHO": "Eaton Course-Hinds",
        "HARGR": "Harger",
        "CARLN": "Carlon",
        "MULB": "Mulberry",
        "SOLAR": "Solarline",
        "ENERL": "Enerlites",
        "HUBWD": "Hubble Wiring Device",
        "DMC": "DMC Power",
        "INT": "Intermatic",
        "LUT": "Lutron",
        "LITTE": "Littelfuse",
        "GRNGA": "GreenGate",
        "WATT": "Wattstopper",
        "SENSO": "Sensor Switch",
        "CHE": "Eaton Crouse Hinds",
        "OZ": "OZ Gedney",
    }

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the data processor.
        
        Args:
            data: Optional DataFrame to process
        """
        self.basic_required_columns = [
            'CatalogNo', 'MfrCode', 'Description'
        ]
        self.full_required_columns = [
            'CatalogNo', 'MfrCode', 'Description', 'Enriched Description',
            'Main Category', 'Sub Category', 'Category Confidence'
        ]
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.processed_data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel file with enhanced validation.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            ValueError: If file is invalid or data is corrupted
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
                
            # Load data with error handling
            try:
                self.data = pd.read_excel(file_path)
            except Exception as e:
                raise ValueError(f"Error reading Excel file: {str(e)}")
                
            # Validate required columns
            missing_columns = [col for col in self.full_required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Initial data quality report
            self._log_data_quality(self.data, "Initial data quality")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data with enhanced error handling.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info("Starting data cleaning")
            cleaned_data = data.copy()
            
            # Track removed rows
            removed_rows = {
                'empty_values': 0,
                'invalid_description': 0,
                'duplicates': 0
            }
            
            # Remove rows with empty required values
            required_cols = ['CatalogNo', 'MfrCode', 'Description']
            mask = cleaned_data[required_cols].isna().any(axis=1)
            removed_rows['empty_values'] = mask.sum()
            cleaned_data = cleaned_data[~mask]
            
            # Validate description length
            mask = cleaned_data['Description'].str.len() < 3
            removed_rows['invalid_description'] = mask.sum()
            cleaned_data = cleaned_data[~mask]
            
            # Remove duplicates
            mask = cleaned_data.duplicated(subset=['CatalogNo'], keep='first')
            removed_rows['duplicates'] = mask.sum()
            cleaned_data = cleaned_data[~mask]
            
            # Log cleaning results
            self._log_cleaning_results(removed_rows, len(cleaned_data))
            
            # Final data quality report
            self._log_data_quality(cleaned_data, "Final data quality")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise
            
    def _log_data_quality(self, data: pd.DataFrame, stage: str) -> None:
        """Log data quality metrics."""
        self.logger.info(f"\n{stage} report:")
        self.logger.info(f"Total rows: {len(data)}")
        self.logger.info(f"Missing values per column:")
        for col in data.columns:
            missing = data[col].isna().sum()
            if missing > 0:
                self.logger.info(f"  {col}: {missing} ({missing/len(data)*100:.1f}%)")
                
    def _log_cleaning_results(self, removed_rows: Dict[str, int], final_count: int) -> None:
        """Log results of data cleaning."""
        self.logger.info("\nData cleaning results:")
        for reason, count in removed_rows.items():
            if count > 0:
                self.logger.info(f"Removed {count} rows due to {reason}")
        self.logger.info(f"Final row count: {final_count}")
        
    def validate_data(self, data: pd.DataFrame, check_full: bool = False) -> bool:
        """
        Validate the input data.
        
        Args:
            data: DataFrame to validate
            check_full: Whether to check for all required columns or just basic ones
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            self.logger.info("Starting data validation")
            
            # Make a copy of the data for validation
            validation_data = data.copy()
            
            # Clean data before validation
            validation_data = self._clean_validation_data(validation_data)
            
            # Check required columns
            required_columns = self.full_required_columns if check_full else self.basic_required_columns
            missing_columns = [col for col in required_columns if col not in validation_data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Check for empty values in required columns
            for col in self.basic_required_columns:
                empty_count = validation_data[col].isna().sum()
                if empty_count > 0:
                    self.logger.warning(f"Found {empty_count} empty values in column: {col}")
                    # Remove rows with empty values
                    validation_data = validation_data.dropna(subset=[col])
                    
            # Check description lengths
            if 'Description' in validation_data.columns:
                short_desc = validation_data[validation_data['Description'].str.len() < 3]
                if not short_desc.empty:
                    self.logger.warning(f"Found {len(short_desc)} descriptions shorter than 3 characters")
                    # Remove rows with short descriptions
                    validation_data = validation_data[validation_data['Description'].str.len() >= 3]
                    
            # Check for duplicates
            duplicates = validation_data[validation_data.duplicated(subset=['CatalogNo'], keep=False)]
            if not duplicates.empty:
                self.logger.warning(f"Found {len(duplicates)} duplicate catalog numbers")
                # Keep first occurrence of each catalog number
                validation_data = validation_data.drop_duplicates(subset=['CatalogNo'], keep='first')
                
            # Update the data if we removed any rows
            if len(validation_data) < len(data):
                self.logger.info(f"Removed {len(data) - len(validation_data)} invalid rows during validation")
                self.data = validation_data
                
            self.logger.info("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            return False
            
    def _clean_validation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data before validation.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Convert MfrCode to string and handle empty values
        if 'MfrCode' in cleaned_data.columns:
            cleaned_data['MfrCode'] = cleaned_data['MfrCode'].astype(str)
            cleaned_data['MfrCode'] = cleaned_data['MfrCode'].replace('nan', '')
            
        # Convert Description to string and handle empty values
        if 'Description' in cleaned_data.columns:
            cleaned_data['Description'] = cleaned_data['Description'].astype(str)
            cleaned_data['Description'] = cleaned_data['Description'].replace('nan', '')
            
        # Convert CatalogNo to string and handle empty values
        if 'CatalogNo' in cleaned_data.columns:
            cleaned_data['CatalogNo'] = cleaned_data['CatalogNo'].astype(str)
            cleaned_data['CatalogNo'] = cleaned_data['CatalogNo'].replace('nan', '')
            
        return cleaned_data

    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed data."""
        if self.processed_data is None:
            raise ValueError("Data has not been processed yet. Call clean_data() first.")
        return self.processed_data

    def to_dict(self) -> Dict[Any, Any]:
        """Convert processed data to dictionary format."""
        if self.processed_data is None:
            raise ValueError("Data has not been processed yet. Call clean_data() first.")
        return self.processed_data.to_dict(orient='index') 