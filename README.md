# Advanced Pump Maintenance Predictor

A machine learning system for predicting maintenance needs of water injection pumps using advanced ML libraries and best practices.

## Features

- Comprehensive data preprocessing and feature engineering
- Multiple ML models (Random Forest and XGBoost)
- Model performance evaluation and comparison
- Feature importance analysis
- Visualization utilities
- Logging system
- Model persistence

## Project Structure

```
pump_predictor/
├── config.py           # Configuration settings
├── data/
│   └── preprocessing.py # Data preprocessing utilities
├── models/
│   ├── base_model.py    # Abstract base model class
│   ├── random_forest_model.py
│   └── xgboost_model.py
├── utils/
│   ├── logger.py        # Logging configuration
│   └── visualization.py # Visualization utilities
└── main.py             # Main execution module
```

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your pump data in `data/pump_data.csv`
2. Run the predictor:
```bash
python -m pump_predictor.main
```

## Data Format

The input CSV should contain the following columns:
- pressure
- temperature
- speed
- vibration
- oil_level
- runtime_hours
- needs_maintenance (target variable)

## Model Output

The system will:
1. Train both Random Forest and XGBoost models
2. Compare their performance
3. Save the best performing model
4. Generate feature importance visualizations
5. Log all activities and results