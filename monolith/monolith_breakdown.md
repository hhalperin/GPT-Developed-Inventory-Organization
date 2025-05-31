# Monolith Breakdown and Migration Analysis

## Overview
The `monolith.py` file is a 1019-line single-file implementation of an inventory management and categorization system. This document breaks down the monolith into logical chunks to facilitate migration to an Object-Oriented Programming (OOP) structure.

## 1. **Configuration and Constants Section** (Lines 1-88)

### Purpose
- Sets up global configuration variables, API keys, and constants
- Defines manufacturer code mapping dictionary
- Configures logging and warnings

### Components
- **API Configuration**: `MODEL`, `MAX_TOKENS`, `TOP_P`, `TEMPERATURE`, `API_URL`
- **Manufacturer Mapping**: `MFR_DICT` (30+ manufacturer code mappings)
- **Logging Setup**: Basic logging configuration with INFO level
- **Imports**: All required libraries and dependencies

### Migration Notes
- This maps to `src/config.py` with some enhancements
- API key handling needs to be environment-variable based instead of Google Colab userdata
- Manufacturer dictionary is already migrated

## 2. **Data Utility Functions** (Lines 89-114)

### Purpose
- Standalone utility functions for data cleaning and processing

### Components
- `clean_data_frame(df)`: Removes empty rows and expands manufacturer codes
- `get_bin_data(df)`: Converts DataFrame to dictionary format

### Migration Notes
- These map to methods in `src/data.py` DataLoader class
- Logic is preserved but wrapped in class methods

## 3. **Core WarehouseInventory Class** (Lines 115-423)

### Purpose
- Main business logic class that handles inventory management, GPT enrichment, similarity scoring, and categorization

### Key Methods Breakdown

#### **Initialization** (Lines 122-140)
- Sets up data structure with required columns
- Initializes similarity dictionaries and tracking variables
- Creates ML model instance

#### **GPT API Integration** (Lines 141-201)
- `call_gpt_api(prompt)`: Static method for API calls
- `enrich_description()`: Single item description enrichment
- `enrich_all_descriptions()`: Batch enrichment with progress tracking

#### **Similarity Computing** (Lines 202-263)
- `compute_similarity_scores()`: Computes SKU and description similarities using fuzzy matching and TF-IDF
- `update_dataframe_with_similarity()`: Updates DataFrame with highest similarity scores and most similar items
- Complex nested dictionary management for similarity scores

#### **Category Assignment** (Lines 264-307)
- `assign_category()`: ML + GPT fallback categorization
- `determine_category_gpt()`: GPT-based category determination
- `update_existing_categories()`: Internal category tracking

#### **Advanced Analytics** (Lines 308-422)
- `reevaluate_categories()`: Re-categorizes all items and tracks changes
- `apply_threshold_filter()`: Filters similarities by threshold
- `compute_average_scores()`: Computes average similarity metrics
- `test_on_subset()`: Runs analysis on data subsets with multiple thresholds
- `show_dataframe()`: Data preview functionality

### Migration Notes
- This is the largest and most complex section
- Already partially migrated across multiple `src/` modules:
  - API calls → `src/api.py`
  - Similarity → `src/similarity.py`
  - Categorization → `src/classification.py`
- Some methods missing or incomplete in current migration

## 4. **MLCategorizer Class** (Lines 424-519)

### Purpose
- Machine learning model for automated categorization using RandomForest and clustering

### Key Methods
- `__init__()`: Initializes ML models and encoders
- `_extract_sku_features()`: Feature extraction from SKU strings
- `_compute_svd()`: Dimensionality reduction setup
- `prepare_features()`: TF-IDF + SVD feature preparation
- `train()`: Training main and sub-category models
- `predict()`: Prediction with confidence scores
- `fit_kmeans()` & `assign_clusters()`: Clustering functionality
- `save_models()`: Model persistence

### Migration Notes
- Already migrated to `src/classification.py` 
- Implementation appears complete and enhanced in src version

## 5. **Main Execution Pipeline** (Lines 520-583)

### Purpose
- Orchestrates the complete workflow from data loading to final output

### Workflow Steps
1. **Data Loading**: Load from categorized > enriched > raw data (resume logic)
2. **Initialization**: Create WarehouseInventory instance
3. **ML Training**: Train categorization models
4. **Category Assignment**: Run ML + GPT categorization if needed
5. **Similarity Analysis**: Compute similarity scores and relationships
6. **Category Reevaluation**: Re-evaluate categories with GPT
7. **Testing**: Run subset testing with multiple thresholds
8. **Output**: Display results

### Migration Notes
- This maps to `src/pipeline.py` InventoryPipeline class
- Resume logic (checking for existing files) is critical functionality
- Error handling and logging preserved

## 6. **DBSCAN Clustering Analysis** (Lines 584-745)

### Purpose
- Advanced clustering analysis using DBSCAN with threshold-based parameter sweeps

### Components
- **File Processing**: Loads multiple threshold result files
- **Feature Extraction**: TF-IDF vectorization across all data
- **DBSCAN Analysis**: Runs DBSCAN with various eps parameters
- **Metrics Computation**: Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- **Network Analysis**: NetworkX graph metrics (degree, components)
- **Visualization**: Multiple plots saved as PNG files

### Migration Notes
- This is advanced analytics functionality
- Partially migrated to `src/clustering.py` but missing network analysis
- Visualization components should be in `src/visualize.py`

## 7. **KMeans Clustering Analysis** (Lines 746-810)

### Purpose
- KMeans clustering with cluster count sweeps and metrics evaluation

### Components
- **Data Loading**: Loads enriched descriptions
- **Feature Preparation**: TF-IDF vectorization
- **Cluster Sweeps**: Tests K=5 to K=50 in steps of 5
- **Metrics Evaluation**: Same clustering metrics as DBSCAN
- **Results Output**: CSV files and visualizations

### Migration Notes
- Also part of clustering analysis
- Integrated into `src/clustering.py` ClusterAnalyzer class

## 8. **Final Clustering Implementation** (Lines 811-874)

### Purpose
- Production clustering implementation using optimal parameters

### Components
- **Data Loading**: Loads enriched data
- **TF-IDF Setup**: Feature extraction pipeline
- **Two-Level Clustering**: 
  - Main categories (threshold 0.80)
  - Sub-categories (threshold 0.90) within main clusters
- **Output Generation**: Final categorized Excel file

### Migration Notes
- This is the actual production clustering logic
- Should be integrated into clustering module
- Uses hardcoded thresholds that should be configurable

## 9. **Visualization and Analysis** (Lines 875-1019)

### Purpose
- Generate final reports, statistics, and visualizations

### Components
- **Distribution Analysis**: Category and sub-category counts
- **Visualization Generation**: Bar plots for distributions
- **Quality Metrics**: Silhouette scores for clustering evaluation
- **Report Output**: CSV files and PNG plots
- **Summary Statistics**: Console output of key metrics

### Migration Notes
- This functionality is partially in `src/visualize.py`
- Some analysis missing (silhouette scoring, detailed statistics)
- Duplicate code block (lines 955-1019 repeat lines 875-939)

## Critical Missing Functionality in Current Migration

### 1. **Resume Logic**
- File existence checking for incremental processing
- Loading from categorized_data.xlsx, enriched_data.xlsx, or raw data

### 2. **Threshold Testing**
- `test_on_subset()` method with multiple threshold analysis
- Results saving in separate directories

### 3. **Network Analysis**
- NetworkX graph construction and analysis
- Degree calculations, connected components

### 4. **Two-Level Clustering**
- DBSCAN main category clustering (threshold 0.80)
- DBSCAN sub-category clustering (threshold 0.90) within main clusters

### 5. **Advanced Visualization**
- Multiple clustering metric plots
- Network visualizations
- Comprehensive reporting

### 6. **Category Reevaluation**
- Complete GPT-based re-categorization with change tracking
- Statistics on category changes

## Recommended Migration Priority

1. **HIGH PRIORITY**: Resume logic and file handling
2. **HIGH PRIORITY**: Complete similarity scoring implementation
3. **MEDIUM PRIORITY**: Advanced clustering with network analysis
4. **MEDIUM PRIORITY**: Comprehensive visualization suite
5. **LOW PRIORITY**: Threshold testing and analysis
6. **LOW PRIORITY**: Category reevaluation functionality

## File Mapping Summary

| Monolith Section | Target Module | Status |
|------------------|---------------|---------|
| Configuration | `src/config.py` | ✅ Complete |
| Data Utils | `src/data.py` | ✅ Complete |
| GPT API | `src/api.py` | ✅ Complete |
| Similarity | `src/similarity.py` | ⚠️ Partial |
| ML Classification | `src/classification.py` | ✅ Complete |
| Clustering | `src/clustering.py` | ⚠️ Partial |
| Visualization | `src/visualize.py` | ⚠️ Partial |
| Pipeline | `src/pipeline.py` | ⚠️ Partial |

## Next Steps

1. **Audit each `src/` module** against monolith functionality
2. **Identify and implement missing methods** 
3. **Ensure data flow compatibility** between modules
4. **Validate that complex nested logic** (similarity scoring, category tracking) is preserved
5. **Test resume logic** and incremental processing
6. **Implement missing advanced analytics** features 