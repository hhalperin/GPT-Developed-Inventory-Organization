"""
Test suite for the main workflow module.
"""

import unittest
import os
from pathlib import Path
from dotenv import load_dotenv
from inventory_system.core.workflow import Workflow
from inventory_system.config.config import Config

class TestWorkflow(unittest.TestCase):
    """Test cases for the Workflow class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration
        self.config = Config()
        
        # Set up test directories
        self.test_data_dir = Path("tests/data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create test input file if it doesn't exist
        self.input_path = self.test_data_dir / "test_input.xlsx"
        if not self.input_path.exists():
            # Create a minimal test Excel file
            import pandas as pd
            test_data = pd.DataFrame({
                "CatalogNo": ["TEST001"],
                "Description": ["10A 120V Circuit Breaker"],
                "MfrCode": ["MFR1"]
            })
            test_data.to_excel(self.input_path, index=False)
    
    def test_workflow_execution(self):
        """Test the complete workflow execution."""
        workflow = Workflow(self.config)
        
        try:
            output_path = workflow.run(str(self.input_path))
            self.assertTrue(Path(output_path).exists())
            print(f"Workflow completed successfully. Results saved to: {output_path}")
        except Exception as e:
            self.fail(f"Error running workflow: {str(e)}")
    
    def test_pipeline_testing(self):
        """Test the pipeline testing functionality."""
        workflow = Workflow(self.config)
        
        try:
            results = workflow.test_pipeline(str(self.input_path), fraction=0.1)
            self.assertIsInstance(results, dict)
            print(f"Test pipeline completed successfully with results: {results}")
        except Exception as e:
            self.fail(f"Error in test pipeline: {str(e)}")

if __name__ == "__main__":
    unittest.main() 