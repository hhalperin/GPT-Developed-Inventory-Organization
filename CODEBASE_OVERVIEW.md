# CODEBASE OVERVIEW

## /src - Main source code directory containing the inventory system implementation
- /inventory_system - Core inventory system package
  - config.py - Configuration management for the inventory system
  - run_system.py - Main entry point for running the inventory system
  - cli.py - Command-line interface implementation
  - __init__.py - Package initialization file
  - README.md - Package documentation

  - /models - Machine learning models and related components
    - ensemble_classifier.py - Implementation of ensemble classification model
    - similarity_features.py - Feature extraction for similarity analysis
    - gpt_processor.py - GPT-based text processing implementation
    - category_classifier.py - Category classification model implementation
    - similarity_analyzer.py - Similarity analysis functionality
    - model_monitor.py - Model monitoring and evaluation
    - category_predictor.py - Category prediction implementation

  - /utils - Utility functions and helpers
    - logging_config.py - Logging configuration setup
    - /monitoring - System monitoring utilities
    - /validation - Data validation utilities

  - /analysis - Data analysis and performance evaluation
    - performance_analyzer.py - Performance analysis implementation
    - analyze_data.py - Data analysis utilities

  - /services - Service layer implementations
    - description_enricher.py - Service for enriching product descriptions

  - /core - Core system components
    - pipeline.py - Main processing pipeline implementation
    - workflow.py - Workflow management implementation
    - data_processor.py - Data processing utilities

## /tests - Test suite for the inventory system
- test_workflow.py - Workflow component tests
- test_category_predictor.py - Category predictor tests
- test_data_validator.py - Data validation tests
- test_metrics.py - Metrics calculation tests

## /data - Data storage and processing directories
- /processed - Processed data files
- /output - System output files
- /input - Input data files

## /models - Trained model storage
- Contains trained model files, feature extractors, and metadata
- Files follow naming convention: {type}_{timestamp}.{extension}

## /logs - System logging directory
- inventory_system.log - Main system log
- Various component-specific log files

## Root Level Files
- requirements.txt - Python package dependencies
- README.md - Project documentation
- run_tests.ps1 - Test execution script
- debug.ps1 - Debugging utilities
- .gitignore - Git ignore rules 