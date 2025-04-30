import os
import sys
import logging

import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ----------------------------------------------------------------------------
# 0) Configure logging to print INFO+ to stdout
# ----------------------------------------------------------------------------
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
root.handlers.clear()
root.addHandler(handler)
logging.getLogger("streamlit").setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# 1) Core pipeline function with caching
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def full_pipeline(history_df: pd.DataFrame, defaults_df: pd.DataFrame):
    """Run the entire end-to-end pipeline and return (metrics, predictions_df)."""

    # Normalize column names
    history_df = history_df.copy()
    defaults_df = defaults_df.copy()
    history_df.columns = history_df.columns.str.strip().str.lower()
    defaults_df.columns = defaults_df.columns.str.strip().str.lower()
    logging.info("Data ingested")

    # Impute payment_status == -2 via nearest centroid
    valid = history_df[history_df['payment_status'] != -2]
    centroids = valid.groupby('payment_status')[['bill_amt', 'paid_amt']].mean()
    history_df['payment_status_imputed'] = history_df.apply(
        lambda row: row['payment_status'] if row['payment_status'] != -2 else int(
            cdist([[row['bill_amt'], row['paid_amt']]], centroids.values).argmin()
        ),
        axis=1
    )
    logging.info("Data imputed")

    # Compute delay & paid_ratio
    history_df['delay'] = history_df['payment_status_imputed'].apply(lambda x: 0 if x == -1 else x)
    history_df['paid_ratio'] = np.where(
        history_df['bill_amt'] > 0,
        history_df['paid_amt'] / history_df['bill_amt'],
        0
    )

    # Aggregate per-client features
    def compute_slope(grp):
        return np.polyfit(grp['month'], grp['delay'], 1)[0] if len(grp) > 1 else 0.0
    logging.info("Aggregating per-client features")
    grp = history_df.groupby('client_id')
    features = pd.DataFrame({
        'client_id':        grp.size().index,
        'count_on_time':    grp.apply(lambda g: (g['delay'] == 0).sum()),
        'count_delayed':    grp.apply(lambda g: (g['delay'] > 0).sum()),
        'max_delay':        grp['delay'].max(),
        'avg_delay':        grp['delay'].mean(),
        'std_delay':        grp['delay'].std().fillna(0),
        'delay_trend':      grp.apply(compute_slope),
        'avg_paid_ratio':   grp['paid_ratio'].mean(),
        'std_paid_ratio':   grp['paid_ratio'].std().fillna(0),
        'min_paid_ratio':   grp['paid_ratio'].min(),
        'max_paid_ratio':   grp['paid_ratio'].max(),
        'avg_bill_amt':     grp['bill_amt'].mean(),
        'std_bill_amt':     grp['bill_amt'].std().fillna(0)
    }).reset_index(drop=True)

    # Merge with defaults (client_id, default, credit_given)
    data = pd.merge(defaults_df, features, on='client_id', how='left').fillna(0)

    # Prepare X, y
    feature_cols = ['credit_given'] + [c for c in features.columns if c != 'client_id']
    X = data[feature_cols]
    y = data['default']
    logging.info("Prepared X and y")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    logging.info("Performed train/test split")

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # Hyperparameter tuning RandomForest
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid = GridSearchCV(
        rf,
        {'n_estimators': [100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]},
        cv=3, scoring='roc_auc', n_jobs=-1
    )
    grid.fit(X_train_s, y_train)
    best_rf = grid.best_estimator_
    logging.info(f"RF best params: {grid.best_params_}")

    # Evaluate models
    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ("DecisionTree", DecisionTreeClassifier(class_weight='balanced', random_state=42)),
        ("RandomForest", best_rf),
        ("XGBoost", XGBClassifier(eval_metric='auc', random_state=42))
    ]
    metrics = {}
    for name, mdl in models:
        if name != "RandomForest": mdl.fit(X_train_s, y_train)
        y_pred, y_prob = mdl.predict(X_test_s), mdl.predict_proba(X_test_s)[:, 1]
        metrics[name] = {'accuracy': accuracy_score(y_test, y_pred), 'auc': roc_auc_score(y_test, y_prob)}
        logging.info(f"{name} -> Acc: {metrics[name]['accuracy']:.4f}, AUC: {metrics[name]['auc']:.4f}")

    # Full-data predictions
    preds = data[['client_id']].copy()
    full_X = scaler.transform(X)
    preds['probability_of_default'] = best_rf.predict_proba(full_X)[:, 1]
    preds['predicted_default'] = (preds['probability_of_default'] >= 0.5).astype(int)

    # Sort by client_id
    preds = preds.sort_values('client_id').reset_index(drop=True)
    logging.info("Sorted predictions by client_id")

    return metrics, preds

# ----------------------------------------------------------------------------
# 2) Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Credit Default Scorer", layout="wide")
st.title("🎯 Credit Default Prediction")

st.markdown("Upload your **payment_history.csv** and **payment_default.csv** to begin:")
hist_file = st.file_uploader("Payment History CSV", type="csv")
def_file = st.file_uploader("Payment Default CSV", type="csv")

if hist_file and def_file:
    history_df, defaults_df = pd.read_csv(hist_file), pd.read_csv(def_file)

    if 'results' not in st.session_state:
        with st.spinner('Running pipeline...'):
            logging.info("Running pipeline")
            metrics, preds_df = full_pipeline(history_df, defaults_df)
        st.session_state.results = (metrics, preds_df)
    else:
        metrics, preds_df = st.session_state.results

    # Show metrics
    st.markdown("## Model Performance")
    cols = st.columns(len(metrics))
    for col, (name, m) in zip(cols, metrics.items()):
        col.metric(name, f"AUC {m['auc']:.3f}", f"Acc {m['accuracy']:.3f}")

    # Show sample
    st.markdown("## Sample Predictions")
    st.dataframe(preds_df.head(10))

    # Download button
    csv = preds_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="default_predictions.csv", mime="text/csv", key="download-csv")
    logging.info("Predictions CSV ready for download")
else:
    st.info("Awaiting both CSV uploads…")
