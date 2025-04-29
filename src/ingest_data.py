# ingest_data.py
import pandas as pd

def load_data(history_path: str, defaults_path: str):
    """
    Load raw CSVs, normalize columns, and return DataFrames.
    """
    history = pd.read_csv(history_path)
    defaults = pd.read_csv(defaults_path)

    # normalize column names
    history.columns = history.columns.str.strip().str.lower()
    defaults.columns = defaults.columns.str.strip().str.lower()

    return history, defaults
