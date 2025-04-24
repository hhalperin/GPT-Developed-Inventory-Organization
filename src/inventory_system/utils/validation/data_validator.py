"""
Data validation module for the inventory system.
Handles data quality checks and validation.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
from ..config.settings import (
    MIN_DESCRIPTION_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    REQUIRED_FIELDS
)

class DataValidator:
    """Handles data validation and quality checks."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_errors: List[Dict[str, Any]] = []
        self.validation_warnings: List[Dict[str, Any]] = []

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the input DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required fields
        self._check_required_fields(df)
        
        # Check data types
        self._check_data_types(df)
        
        # Check description lengths
        self._check_description_lengths(df)
        
        # Check for duplicates
        self._check_duplicates(df)
        
        # Check for missing values
        self._check_missing_values(df)
        
        # Log validation results
        self._log_validation_results()
        
        return len(self.validation_errors) == 0, self.validation_errors

    def _check_required_fields(self, df: pd.DataFrame) -> None:
        """Check if all required fields are present."""
        missing_fields = [field for field in REQUIRED_FIELDS if field not in df.columns]
        if missing_fields:
            self.validation_errors.append({
                "type": "missing_field",
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "fields": missing_fields
            })

    def _check_data_types(self, df: pd.DataFrame) -> None:
        """Check if data types are correct."""
        expected_types = {
            "CatalogNo": str,
            "Description": str,
            "MfrCode": str
        }
        
        for field, expected_type in expected_types.items():
            if field in df.columns and not df[field].dtype == expected_type:
                self.validation_errors.append({
                    "type": "invalid_type",
                    "message": f"Field {field} has incorrect type. Expected {expected_type}, got {df[field].dtype}",
                    "field": field,
                    "expected_type": expected_type,
                    "actual_type": df[field].dtype
                })

    def _check_description_lengths(self, df: pd.DataFrame) -> None:
        """Check if description lengths are within limits."""
        if "Description" in df.columns:
            too_short = df[df["Description"].str.len() < MIN_DESCRIPTION_LENGTH]
            too_long = df[df["Description"].str.len() > MAX_DESCRIPTION_LENGTH]
            
            if not too_short.empty:
                self.validation_errors.append({
                    "type": "description_too_short",
                    "message": f"Found {len(too_short)} descriptions shorter than {MIN_DESCRIPTION_LENGTH} characters",
                    "count": len(too_short),
                    "min_length": MIN_DESCRIPTION_LENGTH
                })
            
            if not too_long.empty:
                self.validation_warnings.append({
                    "type": "description_too_long",
                    "message": f"Found {len(too_long)} descriptions longer than {MAX_DESCRIPTION_LENGTH} characters",
                    "count": len(too_long),
                    "max_length": MAX_DESCRIPTION_LENGTH
                })

    def _check_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate catalog numbers."""
        duplicates = df[df.duplicated(subset=["CatalogNo"], keep=False)]
        if not duplicates.empty:
            self.validation_errors.append({
                "type": "duplicate_catalog",
                "message": f"Found {len(duplicates)} duplicate catalog numbers",
                "count": len(duplicates),
                "catalog_numbers": duplicates["CatalogNo"].tolist()
            })

    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values in required fields."""
        for field in REQUIRED_FIELDS:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    self.validation_errors.append({
                        "type": "missing_values",
                        "message": f"Field {field} has {missing} missing values",
                        "field": field,
                        "count": missing
                    })

    def _log_validation_results(self) -> None:
        """Log validation results."""
        if self.validation_errors:
            logging.error(f"Data validation failed with {len(self.validation_errors)} errors")
            for error in self.validation_errors:
                logging.error(f"Validation error: {error['message']}")
        
        if self.validation_warnings:
            logging.warning(f"Data validation produced {len(self.validation_warnings)} warnings")
            for warning in self.validation_warnings:
                logging.warning(f"Validation warning: {warning['message']}")
        
        if not self.validation_errors and not self.validation_warnings:
            logging.info("Data validation passed successfully") 