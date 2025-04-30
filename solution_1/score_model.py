import pandas as pd
import logging

def score_defaults(history: pd.DataFrame, defaults: pd.DataFrame,
                   model, scaler, feature_df, target_col='default'):
    """
    history: raw payment history
    defaults: default CSV
    model: trained classifier
    scaler: fitted StandardScaler
    feature_df: DataFrame from make_features()
    """
    df = feature_df.merge(defaults[['client_id', target_col]], on='client_id', how='left').fillna(0)
    X = df.drop(columns=[target_col,'client_id'])
    X_s = scaler.transform(X)

    prob = model.predict_proba(X_s)[:,1]
    return pd.DataFrame({
        'client_id': df['client_id'],
        'probability_of_default': prob,
        'predicted_default': (prob>=0.5).astype(int)
    })
