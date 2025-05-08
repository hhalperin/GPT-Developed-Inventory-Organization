"""Configuration settings for the inventory tool."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional

class Config:
    """Configuration class for the inventory tool."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        self.similarity_threshold = 0.8
        self.dbscan_eps = 0.5
        self.min_samples = 2
        self.kmeans_clusters = 3
        self.output_dir = Path("data/output")
        self.model_dir = Path("data/models")
        self.viz_dir = Path("data/visualizations")
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.viz_dir.mkdir(exist_ok=True, parents=True)
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        if os.getenv("SIMILARITY_THRESHOLD"):
            self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
        if os.getenv("DBSCAN_EPS"):
            self.dbscan_eps = float(os.getenv("DBSCAN_EPS"))
        if os.getenv("MIN_SAMPLES"):
            self.min_samples = int(os.getenv("MIN_SAMPLES"))
        if os.getenv("KMEANS_CLUSTERS"):
            self.kmeans_clusters = int(os.getenv("KMEANS_CLUSTERS"))
        if os.getenv("OUTPUT_DIR"):
            self.output_dir = Path(os.getenv("OUTPUT_DIR"))
        if os.getenv("MODEL_DIR"):
            self.model_dir = Path(os.getenv("MODEL_DIR"))
        if os.getenv("VIZ_DIR"):
            self.viz_dir = Path(os.getenv("VIZ_DIR"))
    
    def load_from_file(self, file_path: Path) -> None:
        """Load configuration from a YAML file."""
        if not file_path.exists():
            return
        
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        if config_data:
            self.similarity_threshold = config_data.get('similarity_threshold', self.similarity_threshold)
            self.dbscan_eps = config_data.get('dbscan_eps', self.dbscan_eps)
            self.min_samples = config_data.get('min_samples', self.min_samples)
            self.kmeans_clusters = config_data.get('kmeans_clusters', self.kmeans_clusters)
            if 'output_dir' in config_data:
                self.output_dir = Path(config_data['output_dir'])
            if 'model_dir' in config_data:
                self.model_dir = Path(config_data['model_dir'])
            if 'viz_dir' in config_data:
                self.viz_dir = Path(config_data['viz_dir'])
    
    def save_to_file(self, file_path: Path) -> None:
        """Save configuration to a YAML file."""
        config_data = {
            'similarity_threshold': self.similarity_threshold,
            'dbscan_eps': self.dbscan_eps,
            'min_samples': self.min_samples,
            'kmeans_clusters': self.kmeans_clusters,
            'output_dir': str(self.output_dir),
            'model_dir': str(self.model_dir),
            'viz_dir': str(self.viz_dir)
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def validate_config(self) -> bool:
        """Validate configuration values."""
        if not (0 <= self.similarity_threshold <= 1):
            return False
        if self.dbscan_eps <= 0:
            return False
        if self.min_samples < 1:
            return False
        if self.kmeans_clusters < 1:
            return False
        return True

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4.1-nano-2025-04-14"
MAX_TOKENS = 35
TOP_P = 0.05
TEMPERATURE = 0.1
API_URL = "https://api.openai.com/v1/chat/completions"

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"
VIZ_DIR = DATA_DIR / "visualizations"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# Input/Output files
INVENTORY_DATA = DATA_DIR / "inventory_data.xlsx"
ENRICHED_DATA = DATA_DIR / "enriched_data.xlsx"
CATEGORIZED_DATA = DATA_DIR / "categorized_data.xlsx"
FINAL_CATEGORIES = OUTPUT_DIR / "final_categories.xlsx"

# Clustering parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_DBSCAN_EPS = 0.2  # 1.0 - DEFAULT_SIMILARITY_THRESHOLD
DEFAULT_MIN_SAMPLES = 2
DEFAULT_KMEANS_CLUSTERS = 20

# ML parameters
SVD_COMPONENTS = 100
TFIDF_MAX_FEATURES = 1000
RANDOM_FOREST_ESTIMATORS = 100
ML_CONFIDENCE_THRESHOLD = 0.75

# Manufacturer mapping
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

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_FILE = OUTPUT_DIR / "inventory_tool.log"

# Machine learning parameters
SVD_COMPONENTS = 100
TFIDF_MAX_FEATURES = 1000
RANDOM_FOREST_ESTIMATORS = 100
ML_CONFIDENCE_THRESHOLD = 0.8

# Clustering parameters
KMEANS_CLUSTERS = 20
DBSCAN_EPS = 0.2
DBSCAN_MIN_SAMPLES = 2

# Category lists
FINAL_CATEGORIES = [
    'Electronics',
    'Mechanical',
    'Tools',
    'Office Supplies',
    'Safety Equipment',
    'Lab Equipment',
    'Consumables',
    'Other'
]

# Column names
CATALOG_NUMBER_COL = 'CatalogNo'
DESCRIPTION_COL = 'Description'
MAIN_CATEGORY_COL = 'Main Category'
SUB_CATEGORY_COL = 'Sub-category'
ENRICHED_DESCRIPTION_COL = 'Enriched Description'
CLUSTER_COL = 'Cluster'
MFR_CODE_COL = 'MfrCode'

# File paths
CATEGORIZED_DATA_FILE = 'categorized_data.xlsx'
SIMILARITY_MATRIX_FILE = 'similarity_matrix.npy'
CLUSTER_ASSIGNMENTS_FILE = 'cluster_assignments.xlsx'
SUMMARY_REPORT_FILE = 'summary_report.html'

# API configuration
API_TIMEOUT = 30
API_MAX_RETRIES = 3
API_BATCH_SIZE = 100

# Directory paths
OUTPUT_DIR = Path('../data/output')
MODEL_DIR = Path('../data/models')
VIZ_DIR = Path('../data/visualizations')

# Machine learning parameters
SVD_COMPONENTS = 100
TFIDF_MAX_FEATURES = 1000
RANDOM_FOREST_ESTIMATORS = 100
ML_CONFIDENCE_THRESHOLD = 0.8

# Clustering parameters
KMEANS_CLUSTERS = 20
DBSCAN_EPS = 0.2
DBSCAN_MIN_SAMPLES = 2

# Category lists
FINAL_CATEGORIES = [
    'Electronics',
    'Mechanical',
    'Tools',
    'Office Supplies',
    'Safety Equipment',
    'Lab Equipment',
    'Consumables',
    'Other'
]

# Column names
CATALOG_NUMBER_COL = 'CatalogNo'
DESCRIPTION_COL = 'Description'
MAIN_CATEGORY_COL = 'Main Category'
SUB_CATEGORY_COL = 'Sub-category'
ENRICHED_DESCRIPTION_COL = 'Enriched Description'
CLUSTER_COL = 'Cluster'
MFR_CODE_COL = 'MfrCode'

# API configuration
API_ENDPOINT = 'https://api.example.com/v1'
API_TIMEOUT = 30  # seconds
API_RETRIES = 3

# File paths
DATA_FILE = OUTPUT_DIR / 'inventory_data.csv'
MODEL_FILE = MODEL_DIR / 'inventory_model.pkl'
SIMILARITY_FILE = OUTPUT_DIR / 'similarity_matrix.csv'
CLUSTER_FILE = OUTPUT_DIR / 'cluster_assignments.csv'
REPORT_FILE = VIZ_DIR / 'inventory_report.html'

# Visualization settings
PLOT_WIDTH = 12
PLOT_HEIGHT = 8
PLOT_DPI = 300
PLOT_STYLE = 'seaborn'
COLOR_PALETTE = 'viridis'

# Error messages
ERR_FILE_NOT_FOUND = 'File not found: {}'
ERR_INVALID_FORMAT = 'Invalid file format: {}'
ERR_MISSING_COLUMNS = 'Missing required columns: {}'
ERR_EMPTY_DATA = 'No data available'
ERR_MODEL_NOT_TRAINED = 'Model not trained'
ERR_INVALID_CATEGORY = 'Invalid category: {}'
ERR_API_ERROR = 'API error: {}'

# Success messages
MSG_DATA_LOADED = 'Data loaded successfully: {} rows'
MSG_MODEL_TRAINED = 'Model trained successfully'
MSG_PREDICTIONS_MADE = 'Made predictions for {} items'
MSG_REPORT_GENERATED = 'Report generated: {}'

# Default values
DEFAULT_BATCH_SIZE = 1000
DEFAULT_NUM_THREADS = 4
DEFAULT_CACHE_SIZE = '1GB'
DEFAULT_TIMEOUT = 60  # seconds

# Feature engineering
TEXT_FEATURES = [
    DESCRIPTION_COL,
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL
]

NUMERIC_FEATURES = [
    'Price',
    'Quantity',
    'ReorderPoint'
]

CATEGORICAL_FEATURES = [
    MAIN_CATEGORY_COL,
    SUB_CATEGORY_COL,
    'Manufacturer',
    'Supplier'
]

# Model parameters
MODEL_PARAMS = {
    'n_estimators': RANDOM_FOREST_ESTIMATORS,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Validation settings
VALIDATION_SPLIT = 0.2
CROSS_VAL_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 10

# Performance metrics
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1'
]

# Cache settings
CACHE_ENABLED = True
CACHE_DIR = OUTPUT_DIR / 'cache'
CACHE_EXPIRY = 3600  # seconds

# Threading settings
MAX_THREADS = 8
THREAD_TIMEOUT = 300  # seconds

# Batch processing
BATCH_SIZE = 1000
MAX_BATCHES = 100

# Database settings
DB_HOST = 'localhost'
DB_PORT = 5432
DB_NAME = 'inventory'
DB_USER = 'admin'
DB_PASSWORD = None  # Set via environment variable

# API rate limiting
RATE_LIMIT = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Data columns
CATALOG_NUMBER_COL = 'CatalogNo'
DESCRIPTION_COL = 'Description'
MAIN_CATEGORY_COL = 'Main Category'
SUB_CATEGORY_COL = 'Sub-category'
MFR_CODE_COL = 'MfrCode'
CLUSTER_COL = 'Cluster'

# File paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
VISUALIZATIONS_DIR = 'visualizations'

# Model parameters
TFIDF_MAX_FEATURES = 1000
SVD_COMPONENTS = 100
KMEANS_CLUSTERS = 10
SIMILARITY_THRESHOLD = 0.5

# Manufacturer codes
MFR_DICT = {
    'ABB': ['ABB', '1SDA'],
    'SIEMENS': ['3VA', '3VL', '3RV'],
    'SCHNEIDER': ['LV4', 'NSX', 'CVS'],
    'EATON': ['NZMN', 'NZMP', 'PKZ'],
    'GE': ['FE', 'FD', 'FB'],
    'MITSUBISHI': ['NF', 'WS'],
    'CHINT': ['NM8', 'NM1'],
    'LEGRAND': ['DPX', 'DMX'],
    'HYUNDAI': ['HGM', 'HMD'],
    'LS': ['TS', 'TD']
} 