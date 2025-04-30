import pandas as pd
import logging

def load_data(history_path: str, defaults_path: str):
    """Load and lowercase the two CSVs."""
    logging.info(f"Loading history from {history_path}")
    history = pd.read_csv(history_path)
    logging.info(f"Loading defaults from {defaults_path}")
    defaults = pd.read_csv(defaults_path)

    # normalize
    history.columns  = history.columns.str.strip().str.lower()
    defaults.columns = defaults.columns.str.strip().str.lower()
    return history, defaults
