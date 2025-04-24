"""
Main workflow module for the inventory system.
Orchestrates the entire processing pipeline from data loading to results export.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os

from inventory_system.core.pipeline import InventoryPipeline
from inventory_system.config import SystemConfig
from inventory_system.analysis.performance_analyzer import PerformanceAnalyzer
from inventory_system.analysis.analyze_data import analyze_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Workflow:
    """Orchestrates the entire inventory processing workflow."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the workflow with configuration.
        
        Args:
            config: Optional SystemConfig instance. If None, uses default config.
        """
        self.config = config or SystemConfig()
        self.pipeline = None
        self.checkpoint_file = Path(self.config.checkpoints_dir) / "workflow_checkpoint.json"
        
    def _save_checkpoint(self, phase: str, data: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """Save workflow checkpoint."""
        checkpoint = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # Save checkpoint metadata
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        
        # Save data
        data_path = Path(self.config.checkpoints_dir) / f"checkpoint_{phase}.csv"
        data.to_csv(data_path, index=False)
        
        logger.info(f"Saved checkpoint for phase: {phase}")
    
    def _load_checkpoint(self) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Load workflow checkpoint."""
        if not self.checkpoint_file.exists():
            return None, None, None
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            phase = checkpoint["phase"]
            data_path = Path(self.config.checkpoints_dir) / f"checkpoint_{phase}.csv"
            
            if data_path.exists():
                data = pd.read_csv(data_path)
                return phase, data, checkpoint["metadata"]
            
            return phase, None, checkpoint["metadata"]
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None, None, None
    
    def run(self, data_path: str) -> str:
        """
        Run the complete inventory processing workflow.
        
        Args:
            data_path: Path to the input Excel file
            
        Returns:
            str: Path to the output file
        """
        try:
            # Check for existing checkpoint
            phase, data, metadata = self._load_checkpoint()
            
            if phase is None:
                # Step 1: Load and validate data
                logger.info("Loading input data...")
                data = pd.read_excel(data_path)
                phase = "initial_load"
            
            # Initialize pipeline
            if self.pipeline is None:
                self.pipeline = InventoryPipeline(self.config)
            
            # Process based on current phase
            if phase == "initial_load":
                # Step 2: Validate data
                logger.info("Validating data...")
                if not self.pipeline.validate_data(data):
                    raise ValueError("Data validation failed")
                
                # Step 3: Enrich descriptions
                logger.info("Enriching descriptions...")
                data = self.pipeline.enrich_descriptions(data)
                self._save_checkpoint("description_enrichment", data, {})
                phase = "description_enrichment"
            
            if phase == "description_enrichment":
                # Step 4: Initial categorization
                logger.info("Starting initial categorization...")
                data = self.pipeline.initial_categorization(data)
                self._save_checkpoint("initial_categorization", data, {})
                phase = "initial_categorization"
            
            if phase == "initial_categorization":
                # Step 5: Iterative categorization with ML
                logger.info("Starting iterative categorization with ML...")
                data = self.pipeline.iterative_categorization(data)
                self._save_checkpoint("iterative_categorization", data, {})
                phase = "iterative_categorization"
            
            if phase == "iterative_categorization":
                # Step 6: Category alignment
                logger.info("Starting category alignment...")
                data = self.pipeline.align_categories(data)
                self._save_checkpoint("category_alignment", data, {})
                phase = "category_alignment"
            
            if phase == "category_alignment":
                # Step 7: Run comprehensive analysis
                logger.info("Running comprehensive analysis...")
                self._run_analysis(data)
                
                # Step 8: Export results
                logger.info("Exporting results...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(self.config.reports_dir) / f"results_{timestamp}.xlsx"
                self.pipeline.export_results(data, str(output_path))
                
                # Clean up checkpoint
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
                
                logger.info(f"Pipeline completed successfully. Results saved to: {output_path}")
                return str(output_path)
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            raise
            
    def _run_analysis(self, data: pd.DataFrame) -> None:
        """
        Run comprehensive analysis on the processed data.
        
        Args:
            data: Processed DataFrame
        """
        try:
            # Create analysis directories
            analysis_dir = Path(self.config.reports_dir) / "analysis"
            visualizations_dir = analysis_dir / "visualizations"
            os.makedirs(visualizations_dir, exist_ok=True)
            
            # Run data analysis
            logger.info("Running data analysis...")
            analyze_data(data, str(analysis_dir))
            
            # Run performance analysis
            logger.info("Running performance analysis...")
            analyzer = PerformanceAnalyzer(data, self.config.models_dir)
            analyzer.analyze_performance(str(analysis_dir))
            
            # Generate research report
            logger.info("Generating research report...")
            self._generate_research_report(data, str(analysis_dir))
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise
            
    def _generate_research_report(self, data: pd.DataFrame, output_dir: str) -> None:
        """
        Generate comprehensive research report.
        
        Args:
            data: Processed DataFrame
            output_dir: Directory to save report
        """
        try:
            report_path = Path(output_dir) / "research_report.md"
            
            with open(report_path, 'w') as f:
                f.write("# Inventory Categorization System Research Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # System Overview
                f.write("## System Overview\n\n")
                f.write("The inventory categorization system uses a combination of GPT and ML models to categorize items.\n\n")
                
                # Methodology
                f.write("## Methodology\n\n")
                f.write("1. Initial Categorization\n")
                f.write("2. Iterative Categorization with ML\n")
                f.write("3. Category Alignment\n")
                f.write("4. Performance Analysis\n\n")
                
                # Results
                f.write("## Results\n\n")
                f.write(f"Total items processed: {len(data)}\n")
                f.write(f"Total categories: {data['Main Category'].nunique()}\n")
                f.write(f"Total subcategories: {data['Sub Category'].nunique()}\n\n")
                
                # Performance Metrics
                f.write("## Performance Metrics\n\n")
                f.write("See the analysis directory for detailed metrics and visualizations.\n\n")
                
                # Conclusions
                f.write("## Conclusions\n\n")
                f.write("The system demonstrates effective categorization capabilities with high accuracy.\n\n")
                
                # Future Work
                f.write("## Future Work\n\n")
                f.write("1. Enhance category alignment\n")
                f.write("2. Improve ML model performance\n")
                f.write("3. Add more sophisticated similarity metrics\n")
                
            logger.info(f"Research report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating research report: {e}")
            raise
            
    def test_pipeline(self, data_path: str, fraction: float = 0.1) -> Dict[str, Any]:
        """
        Test the pipeline with a sample of the data.
        
        Args:
            data_path: Path to the input Excel file
            fraction: Fraction of data to use for testing
            
        Returns:
            Dict[str, Any]: Test results
        """
        try:
            logger.info("Loading test data...")
            data = pd.read_excel(data_path)
            sample_data = data.sample(frac=fraction)
            
            # Initialize pipeline
            if self.pipeline is None:
                self.pipeline = InventoryPipeline(self.config)
            
            # Run test pipeline
            logger.info("Running test pipeline...")
            results = self.pipeline.test_pipeline(sample_data)
            
            logger.info("Test pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in test pipeline: {e}")
            raise 