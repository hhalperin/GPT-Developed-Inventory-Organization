"""Data loading and preprocessing module."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

from .config import (
    CATALOG_NUMBER_COL,
    DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    MFR_CODE_COL,
    MFR_DICT
)

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and preprocessing."""
    
    def __init__(self):
        """Initialize data loader."""
        self.data = None
        self.raw_data = None
        self.cleaned_data = None
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            file_path: Path to data file.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not supported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_type = file_path.suffix.lower()
        if file_type == '.csv':
            self.raw_data = pd.read_csv(file_path)
        elif file_type == '.xlsx':
            self.raw_data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        self.data = self.raw_data.copy()
        return self.data
        
    def clean_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data frame to match monolith logic."""
        df = df.copy()
        logger.info("Starting data cleaning...")
        # Remove rows with missing or EMPTY catalog numbers
        before = len(df)
        df = df[df[CATALOG_NUMBER_COL].notna()]
        df = df[df[CATALOG_NUMBER_COL].astype(str) != "EMPTY"]
        logger.info(f"Removed {before - len(df)} rows with missing or EMPTY CatalogNo.")
        # Ensure all required columns exist
        required_cols = [CATALOG_NUMBER_COL, DESCRIPTION_COL, MFR_CODE_COL, MAIN_CATEGORY_COL, SUB_CATEGORY_COL]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "Unknown" if col != CATALOG_NUMBER_COL else ""
                logger.info(f"Added missing column: {col}")
        # Enforce string type for key columns
        for col in required_cols:
            df[col] = df[col].astype(str)
        # Fill missing descriptions
        df[DESCRIPTION_COL] = df[DESCRIPTION_COL].replace({None: "Unknown", np.nan: "Unknown"}).fillna("Unknown")
        # Fill missing categories
        df[MAIN_CATEGORY_COL] = df[MAIN_CATEGORY_COL].replace({None: "Uncategorized", np.nan: "Uncategorized"}).fillna("Uncategorized")
        df[SUB_CATEGORY_COL] = df[SUB_CATEGORY_COL].replace({None: "Uncategorized", np.nan: "Uncategorized"}).fillna("Uncategorized")
        # Map MfrCode using MFR_DICT
        if MFR_CODE_COL in df.columns:
            df[MFR_CODE_COL] = df[MFR_CODE_COL].apply(lambda x: MFR_DICT.get(x, x) if pd.notna(x) else "Unknown")
            logger.info("Mapped MfrCode using MFR_DICT.")
        # Log statistics
        logger.info(f"Cleaned data: {len(df)} rows, {df[CATALOG_NUMBER_COL].nunique()} unique SKUs.")
        logger.info(f"Unique manufacturers: {df[MFR_CODE_COL].nunique()}")
        logger.info(f"Unique main categories: {df[MAIN_CATEGORY_COL].nunique()}")
        logger.info(f"Unique sub-categories: {df[SUB_CATEGORY_COL].nunique()}")
        logger.info(f"Missing descriptions: {df[DESCRIPTION_COL].isna().sum()}")
        logger.info(f"Missing main categories: {df[MAIN_CATEGORY_COL].isna().sum()}")
        logger.info(f"Missing sub-categories: {df[SUB_CATEGORY_COL].isna().sum()}")
        return df
        
    def clean_data(self) -> pd.DataFrame:
        """Clean loaded data.
        
        Returns:
            Cleaned DataFrame.
            
        Raises:
            ValueError: If no data loaded.
        """
        if self.raw_data is None:
            raise ValueError("No data loaded")
            
        self.cleaned_data = self.clean_data_frame(self.raw_data)
        return self.cleaned_data
        
    def get_manufacturer_code(self, catalog_number: str) -> str:
        """Extract manufacturer code from catalog number.
        
        Args:
            catalog_number: Catalog number to extract from.
            
        Returns:
            Extracted manufacturer code.
        """
        if not isinstance(catalog_number, str):
            return "Unknown"
            
        # Try to match against manufacturer prefixes
        for mfr, prefixes in MFR_DICT.items():
            for prefix in prefixes:
                if catalog_number.upper().startswith(prefix):
                    return mfr
                    
        return "Unknown"
        
    def get_bin_data(self) -> Dict[str, Any]:
        """Get data in dictionary format.
        
        Returns:
            Dictionary containing data.
            
        Raises:
            ValueError: If no cleaned data available.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available")
            
        return {
            'catalog_numbers': self.cleaned_data[CATALOG_NUMBER_COL].tolist(),
            'descriptions': self.cleaned_data[DESCRIPTION_COL].tolist(),
            'manufacturer_codes': self.cleaned_data[MFR_CODE_COL].tolist(),
            'main_categories': self.cleaned_data[MAIN_CATEGORY_COL].tolist(),
            'sub_categories': self.cleaned_data[SUB_CATEGORY_COL].tolist()
        }
        
    def save_data(self, output_file: Path) -> None:
        """Save cleaned data to file.
        
        Args:
            output_file: Path to save file.
            
        Raises:
            ValueError: If no cleaned data available.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available")
            
        # Create parent directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file type
        file_type = output_file.suffix.lower()
        if file_type == '.csv':
            self.cleaned_data.to_csv(output_file, index=False)
        elif file_type == '.xlsx':
            self.cleaned_data.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    def load_categorized_data(self, file_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """Load pre-categorized data.
        
        Args:
            file_path: Path to categorized data file.
            
        Returns:
            Loaded DataFrame or None if file not found.
        """
        if file_path is None:
            return None
            
        if not file_path.exists():
            return None
            
        file_type = file_path.suffix.lower()
        if file_type == '.csv':
            return pd.read_csv(file_path)
        elif file_type == '.xlsx':
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the data.
        
        Returns:
            Dictionary containing statistics.
            
        Raises:
            ValueError: If no cleaned data available.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available")
            
        # Add manufacturer codes if not present
        if MFR_CODE_COL not in self.cleaned_data.columns:
            self.cleaned_data[MFR_CODE_COL] = self.cleaned_data[CATALOG_NUMBER_COL].apply(self.get_manufacturer_code)
            
        return {
            'total_items': len(self.cleaned_data),
            'unique_manufacturers': len(self.cleaned_data[MFR_CODE_COL].unique()),
            'unique_categories': len(self.cleaned_data[MAIN_CATEGORY_COL].unique()),
            'unique_subcategories': len(self.cleaned_data[SUB_CATEGORY_COL].unique()),
            'missing_descriptions': int(self.cleaned_data[DESCRIPTION_COL].isna().sum()),
            'missing_categories': int(self.cleaned_data[MAIN_CATEGORY_COL].isna().sum()),
            'manufacturer_distribution': self.cleaned_data[MFR_CODE_COL].value_counts().to_dict(),
            'category_distribution': self.cleaned_data[MAIN_CATEGORY_COL].value_counts().to_dict(),
            'subcategory_distribution': self.cleaned_data[SUB_CATEGORY_COL].value_counts().to_dict()
        }
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data format.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        required_columns = [CATALOG_NUMBER_COL, DESCRIPTION_COL]
        return all(col in data.columns for col in required_columns)