# score_model.py
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def score_defaults(history_path, defaults_path, model_path, scaler_path, output_csv):
    # load histories and defaults
    history = pd.read_csv(history_path)
    defaults = pd.read_csv(defaults_path)
    model = pd.read_pickle(model_path)
    scaler= pd.read_pickle(scaler_path)

    # impute & aggregate (reuse functions from feature_engineering)
    from feature_engineering import impute_status_similarity, aggregate_history
    history = impute_status_similarity(history)
    features = aggregate_history(history, defaults)

    # merge and predict
    data = defaults.merge(features, on='client_id', how='left').fillna(0)
    X = data.drop(columns=['client_id','default'])

    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)[:,1]
    preds = (probs>=0.5).astype(int)

    out = pd.DataFrame({
        'client_id': data['client_id'],
        'probability_of_default': probs,
        'predicted_default': preds
    })
    out.to_csv(output_csv, index=False)
    print(f"Saved scoring results to {output_csv}")