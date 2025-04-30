# Payment Default Prediction

This repository contains an end-to-end solution for predicting loan payment defaults based on historical payment data. It includes data ingestion, preprocessing, feature engineering, model training and validation, and a scoring function for new data.

## Project Structure
```
├── README.md                        # Project documentation (this file)
├── data/                            # Raw and processed datasets
│   ├── payment_default.csv         # Original default indicators (target variable)
│   ├── payment_history.csv         # Original payment history data
│   ├── defaults.pkl                # Serialized defaults DataFrame (optional caching)
│   └── history.pkl                 # Serialized history DataFrame (optional caching)
├── default_predictions.csv          # Sample output predictions
├── models/                          # Saved trained model artifacts (e.g., .pkl files)
├── solution_1/                      # First solution implementation (standalone)
│   ├── app.py                       # CLI app for end-to-end scoring
│   └── requirements.txt             # Dependencies for solution_1
├── src/                             # Modular pipeline components
│   ├── ingest_data.py               # CSV loading and normalization
│   ├── feature_engineering.py       # Imputation and aggregation functions
│   ├── train_model.py               # Model training, tuning, and evaluation
│   └── score_model.py               # Scoring function for new data
├── data_insights.ipynb              # Jupyter notebook for exploratory data analysis
├── simple.ipynb                     # Simplified end-to-end demonstration notebook
└── test.py                          # Quick local script for testing pipeline
```

## Requirements

- Python 3.8+  
- See `solution_1/requirements.txt` or the project-wide `requirements.txt`:
  ```
  numpy>=1.21.0
  pandas>=1.3.0
  scipy>=1.7.0
  scikit-learn>=1.0.0
  xgboost>=1.6.0
  tqdm>=4.60.0
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Azharudeen10/shopup_project.git
   cd shopup_project
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### CLI Application (`solution_1/app.py`)

```bash
python solution_1/app.py \
  --history data/payment_history.csv \
  --defaults data/payment_default.csv \
  --output default_predictions.csv
```

### Modular Scripts (`src/`)

1. **Ingestion**: Load and normalize data
   ```bash
   python -c "from src.ingest_data import load_data; load_data('data/payment_history.csv','data/payment_default.csv')"
   ```
2. **Feature Engineering**: Imputation & aggregation
3. **Training**: Hyperparameter tuning and model evaluation
4. **Scoring**: Generate predictions on new data

### Notebooks

- **`data_insights.ipynb`**: In-depth EDA with visualizations and statistical summaries.  
- **`simple.ipynb`**: Concise demonstration of the full pipeline from ingestion to scoring.

## Logging & Progress

- **Logging**: Uses Python's `logging` for informative, timestamped status updates.  
- **Progress Bars**: `tqdm` provides spinners during hyperparameter tuning.

## Future Enhancements

- **Model Deployment**: Wrap the pipeline into a web service (FastAPI/Streamlit).  
- **Advanced Feature Engineering**: Include outlier detection, time-based features, and ensemble methods.  
- **Monitoring**: Track prediction performance in production and implement model retraining triggers.

## License

Released under the MIT License. Contributions welcome!

