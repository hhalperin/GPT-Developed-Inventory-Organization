"""
Test suite for the data validator module.
"""

import unittest
import pandas as pd
import numpy as np
from inventory_system.validation.data_validator import DataValidator

class TestDataValidator(unittest.TestCase):
    """Test cases for the DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        
        # Create valid test data with explicit string types
        self.valid_data = pd.DataFrame({
            "CatalogNo": pd.Series(["TEST001", "TEST002", "TEST003"], dtype="string"),
            "Description": pd.Series([
                "10A 120V Circuit Breaker",
                "20A 240V Circuit Breaker",
                "15A 120V GFCI Outlet"
            ], dtype="string"),
            "MfrCode": pd.Series(["MFR1", "MFR2", "MFR3"], dtype="string"),
            "Main Category": pd.Series(["Circuit Breakers", "Circuit Breakers", "Outlets"], dtype="string"),
            "Sub-category": pd.Series(["Standard", "Standard", "GFCI"], dtype="string")
        })
        
        # Create invalid test data
        self.invalid_data = pd.DataFrame({
            "CatalogNo": pd.Series(["TEST001", None, "TEST003"], dtype="string"),
            "Description": pd.Series(["", "20A 240V Circuit Breaker", None], dtype="string"),
            "MfrCode": pd.Series(["MFR1", "MFR2", "MFR3"], dtype="string"),
            "Main Category": pd.Series(["Circuit Breakers", None, "Outlets"], dtype="string"),
            "Sub-category": pd.Series(["Standard", "Standard", None], dtype="string")
        })

    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator)
        self.assertEqual(len(self.validator.validation_errors), 0)
        self.assertEqual(len(self.validator.validation_warnings), 0)

    def test_validate_dataframe_valid(self):
        """Test validation of valid dataframe."""
        is_valid, errors = self.validator.validate_dataframe(self.valid_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(self.validator.validation_errors), 0)
        self.assertEqual(len(self.validator.validation_warnings), 0)

    def test_validate_dataframe_invalid(self):
        """Test validation of invalid dataframe."""
        is_valid, errors = self.validator.validate_dataframe(self.invalid_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertGreater(len(self.validator.validation_errors), 0)

    def test_check_required_fields(self):
        """Test required fields validation."""
        # Test with missing required field
        data = self.valid_data.drop(columns=["CatalogNo"])
        self.validator._check_required_fields(data)
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Test with all required fields
        self.validator.validation_errors = []
        self.validator._check_required_fields(self.valid_data)
        self.assertEqual(len(self.validator.validation_errors), 0)

    def test_check_data_types(self):
        """Test data type validation."""
        # Test with incorrect data types
        data = self.valid_data.copy()
        data["CatalogNo"] = data["CatalogNo"].astype(float)
        self.validator._check_data_types(data)
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Test with correct data types
        self.validator.validation_errors = []
        self.validator._check_data_types(self.valid_data)
        self.assertEqual(len(self.validator.validation_errors), 0)

    def test_check_description_lengths(self):
        """Test description length validation."""
        # Test with too short description
        data = self.valid_data.copy()
        data.loc[0, "Description"] = "A"
        self.validator._check_description_lengths(data)
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Test with too long description
        self.validator.validation_errors = []
        data.loc[0, "Description"] = "A" * 1001
        self.validator._check_description_lengths(data)
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Test with valid descriptions
        self.validator.validation_errors = []
        self.validator._check_description_lengths(self.valid_data)
        self.assertEqual(len(self.validator.validation_errors), 0)

    def test_check_duplicates(self):
        """Test duplicate validation."""
        # Test with duplicates
        data = self.valid_data.copy()
        data = pd.concat([data, data.iloc[0:1]], ignore_index=True)
        self.validator._check_duplicates(data)
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Test without duplicates
        self.validator.validation_errors = []
        self.validator._check_duplicates(self.valid_data)
        self.assertEqual(len(self.validator.validation_errors), 0)

    def test_check_missing_values(self):
        """Test missing values validation."""
        # Test with missing values
        self.validator._check_missing_values(self.invalid_data)
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Test without missing values
        self.validator.validation_errors = []
        self.validator._check_missing_values(self.valid_data)
        self.assertEqual(len(self.validator.validation_errors), 0)

    def test_log_validation_results(self):
        """Test validation results logging."""
        # Test with errors
        self.validator._check_missing_values(self.invalid_data)
        self.validator._log_validation_results()
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Test without errors
        self.validator.validation_errors = []
        self.validator._check_missing_values(self.valid_data)
        self.validator._log_validation_results()
        self.assertEqual(len(self.validator.validation_errors), 0)

if __name__ == "__main__":
    unittest.main() 