# Credit Default Prediction System

This project provides two solutions for predicting credit defaults based on payment history data.


 ## Project Structure
 ```
 ├── EDA.ipynb # Exploratory Data Analysis notebook
 ├── requirements.txt # Python dependencies
 ├── client_dashboard.py # Data visualization client wise
 ├── solution_1/ # Monolithic solution
 │ └── app.py # Streamlit application (all-in-one)
 └── solution_2/ # Modular solution
   ├── app.py # Streamlit application (orchestrator)
   ├── feature_engineering.py # Feature generation logic
   ├── ingest_data.py # Data loading and validation
   ├── score_model.py # Prediction scoring
   └── train_model.py # Model training and evaluation
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
 
## Requirements
 This part alone take some more amount of data and time to installation.
 - Python 3.8+  
   ```
   pandas>=1.5.0
   numpy>=1.23.0
   streamlit>=1.18.0
   scipy>=1.9.0
   scikit-learn>=1.1.0
   xgboost>=1.7.0
   matplotlib>=3.5.0
   seaborn>=0.12.0
   ```

  
 ## Solution 1: Monolithic Approach
 
 A single-file implementation that handles:
 - Data loading
 - Feature engineering
 - Model training
 - Prediction generation
 
 ### Features
 - Imputes missing payment status via nearest centroid
 - Calculates payment delay and paid ratio metrics
 - Aggregates per-client features
 - Trains multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
 - Provides performance metrics (Accuracy, AUC-ROC)
 - Generates downloadable predictions
 
 ### Usage
 ```bash
 streamlit run solution_1/app.py
 ```

## Solution 2: Modular Approach
 
 A component-based architecture that separates concerns into distinct modules:
 Components
 
 **Data Ingestion** (ingest_data.py):
 - Loads and validates input CSVs
 - Normalizes column names

 **Feature Engineering** (feature_engineering.py):
 - Calculates payment patterns
 - Generates aggregated client features
 - Handles missing data

 **Model Training** (train_model.py):
 - Standardizes features
 - Performs hyperparameter tuning
 - Evaluates multiple models
 - Selects best performer

 **Prediction Scoring** (score_model.py):
 - Generates probability scores
 - Makes final predictions
 - Formats output

 ### Usage
 ```bash
 streamlit run solution_2/app.py
 ```
 
 ### Notebooks
 - **`EDA.ipynb`**: In-depth EDA with visualizations and statistical summaries.
Kindly get into the Data Analysis to find the insights extracted from the data 

 ### Client dashboard
 - Execute the model once.
 - Enter a Client ID and click Show Data to see both the payment‐history plots and the default prediction.

### Usage
 ```bash
 streamlit run client_dashboard.py
 ```
 
 ## Future Enhancements
 - **Dynamic column handling**: Improve to handle the columns dynamically instead of selecting and fitting in input X.
 - **Realtime prediction**: Using apache kafka to make consume the data and get the realtime prdiction instead of batch.
 - **Unittesting**: Create a unittesting file to catch the development mistakes
 - **Processing**: Implementation of pyspark to process very huge data files.
 - **Front-end**: Can improve the front-end in a better way.
 - **Model Deployment**: Wrap the pipeline into a web service (FastAPI/Streamlit).  
 - **Advanced Feature Engineering**: Include outlier detection, time-based features, and ensemble methods.  
 - **Monitoring**: Track prediction performance in production and implement model retraining triggers.

 ## Developer note
 Happy coding!
