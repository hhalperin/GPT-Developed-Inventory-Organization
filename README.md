# Inventory System with AI-Powered Categorization

This project implements an intelligent inventory management system that uses AI to automatically categorize and enrich item descriptions. The system combines traditional inventory management with modern AI capabilities to provide accurate and detailed item categorization.

## Features

- **AI-Powered Categorization**: Uses GPT models to automatically categorize inventory items
- **Description Enrichment**: Automatically enhances item descriptions with additional details
- **Similarity-Based Matching**: Implements fuzzy matching to find similar items
- **Configurable System**: Easy to configure through a centralized configuration system
- **Batch Processing**: Efficiently processes large datasets in batches
- **Comprehensive Analysis**: Generates detailed performance metrics and visualizations
- **Research Report Generation**: Automatically creates research reports with system performance
- **Checkpoint System**: Saves progress and allows resuming from any stage
- **Monitoring**: Built-in monitoring and logging capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/inventory_system.git
cd inventory_system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Prepare your inventory data in Excel format
2. Configure the system settings in `src/inventory_system/config.py`
3. Run the main processing script:
```bash
python src/main.py
```

The system will:
- Process your inventory data
- Categorize items using AI
- Generate performance metrics
- Create visualizations
- Generate a research report

All results will be saved in the `reports/analysis` directory.

## Project Structure

```
inventory_system/
├── src/
│   ├── inventory_system/
│   │   ├── core/
│   │   │   ├── pipeline.py      # Main processing pipeline
│   │   │   └── workflow.py      # Workflow orchestration
│   │   ├── models/
│   │   │   ├── gpt_processor.py # GPT model integration
│   │   │   └── similarity_matcher.py
│   │   ├── analysis/
│   │   │   ├── performance_analyzer.py
│   │   │   └── analyze_data.py
│   │   ├── utils/
│   │   │   └── data_processor.py
│   │   ├── config.py
│   │   └── __init__.py
│   └── main.py
├── data/
│   ├── raw/          # Input data
│   └── processed/    # Processed data
├── reports/
│   ├── analysis/     # Analysis results
│   │   ├── metrics/  # Performance metrics
│   │   └── visualizations/  # Generated plots
│   └── checkpoints/  # Progress checkpoints
├── logs/             # System logs
├── tests/            # Unit tests
├── requirements.txt
└── README.md
```

## Configuration

The system can be configured through the `SystemConfig` class in `config.py`. Key configuration options include:

- GPT model settings
- Similarity matching thresholds
- Processing batch sizes
- File paths and patterns
- Environment variables

## Analysis and Reporting

The system generates comprehensive analysis including:

1. **Performance Metrics**
   - Basic metrics (accuracy, precision, recall)
   - Category-specific metrics
   - Similarity analysis
   - Process metrics
   - Model performance metrics

2. **Visualizations**
   - Category distribution plots
   - Learning curves
   - ROC curves
   - Processing time analysis
   - Similarity matrices

3. **Research Report**
   - System overview
   - Methodology
   - Results and statistics
   - Performance analysis
   - Conclusions and future work

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 