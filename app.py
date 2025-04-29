import numpy as np
import pandas as pd
import warnings
# Suppress deprecation warnings (pandas, etc.)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ----------------------------------------------------------------------------
# Configure logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def assign_nearest_status(row, centroids):
    """
    Impute payment_status == -2 by nearest centroid based on bill_amt and paid_amt.
    """
    if row['payment_status'] != -2:
        return row['payment_status']
    point = np.array([[row['bill_amt'], row['paid_amt']]])
    dists = cdist(point, centroids.values, metric='euclidean').ravel()
    return int(centroids.index[dists.argmin()])

def compute_slope(df):
    """Compute slope of delay over month for each client."""
    if len(df) > 1:
        return np.polyfit(df['month'], df['delay'], 1)[0]
    else:
        return 0.0

# ----------------------------------------------------------------------------
# 1) Load & normalize data
# ----------------------------------------------------------------------------
logging.info("Data ingestion started")
history  = pd.read_csv('data/payment_history.csv')
defaults = pd.read_csv('data/payment_default.csv')
logging.info("Data ingestion completed")

history.columns  = history.columns.str.strip().str.lower()
defaults.columns = defaults.columns.str.strip().str.lower()

# ----------------------------------------------------------------------------
# 2) Preprocess history & feature engineering
# ----------------------------------------------------------------------------
logging.info("Preprocessing history and feature engineering started")
# Impute payment_status == -2
valid = history[history['payment_status'] != -2]
centroids = valid.groupby('payment_status')[['bill_amt', 'paid_amt']].mean()
history['payment_status_imputed'] = history.apply(
    assign_nearest_status,
    axis=1,
    args=(centroids,)
)
# Compute delay & paid_ratio
history['delay'] = history['payment_status_imputed'].apply(lambda x: 0 if x == -1 else x)
history['paid_ratio'] = np.where(
    history['bill_amt'] > 0,
    history['paid_amt'] / history['bill_amt'],
    0
)
# Aggregate per-client
grp = history.groupby('client_id')
features = pd.DataFrame({
    'count_on_time':    grp.apply(lambda x: (x['delay'] == 0).sum()),
    'count_delayed':    grp.apply(lambda x: (x['delay'] > 0).sum()),
    'max_delay':        grp['delay'].max(),
    'avg_delay':        grp['delay'].mean(),
    'std_delay':        grp['delay'].std().fillna(0),
    'last_month_delay': history[history['month'] == history['month'].max()]
                          .set_index('client_id')['delay'],
    'delay_trend':      grp.apply(compute_slope),
    'avg_paid_ratio':   grp['paid_ratio'].mean(),
    'std_paid_ratio':   grp['paid_ratio'].std().fillna(0),
    'min_paid_ratio':   grp['paid_ratio'].min(),
    'max_paid_ratio':   grp['paid_ratio'].max(),
    'avg_bill_amt':     grp['bill_amt'].mean(),
    'std_bill_amt':     grp['bill_amt'].std().fillna(0)
})
features.reset_index(inplace=True)
logging.info("Feature engineering completed")

# ----------------------------------------------------------------------------
# 3) Merge features with defaults and prepare data
# ----------------------------------------------------------------------------
logging.info("Merging features and preparing dataset")
data = pd.merge(defaults, features, on='client_id', how='left').fillna(0)
target_col   = 'default'
feature_cols = ['credit_given'] + [c for c in features.columns if c != 'client_id']
X = data[feature_cols]
y = data[target_col]
logging.info("Data preparation completed")

# ----------------------------------------------------------------------------
# 4) Train/Test split and scaling
# ----------------------------------------------------------------------------
logging.info("Train/test split started")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
logging.info("Train/test split completed")

logging.info("Scaling numeric features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
logging.info("Scaling completed")

# ----------------------------------------------------------------------------
# 5) Hyperparameter tuning for Random Forest
# ----------------------------------------------------------------------------
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators':     [100, 200],
    'max_depth':        [None, 5, 10],
    'min_samples_split':[2, 5]
}
grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
logging.info("Random Forest hyperparameter tuning started")
# single-step progress indicator
for _ in tqdm(range(1), desc="Tuning RF", ascii=True):
    grid.fit(X_train_scaled, y_train)
best_rf = grid.best_estimator_
logging.info("RF tuning completed – best params: %s", grid.best_params_)

# ----------------------------------------------------------------------------
# 6) Evaluate multiple models on the test set
# ----------------------------------------------------------------------------
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(class_weight='balanced', random_state=42)),
    ("Random Forest (tuned)", best_rf),
    ("XGBoost", XGBClassifier(eval_metric='auc', random_state=42))  # Fix: add model name
]


logging.info("Model evaluation started")
for name, mdl in models:
    logging.info("Evaluating %s", name)
    if name != "Random Forest (tuned)":
        mdl.fit(X_train_scaled, y_train)
    y_pred  = mdl.predict(X_test_scaled)
    y_proba = mdl.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, digits=4)
    logging.info(
        "%s -> Accuracy: %.4f, ROC AUC: %.4f", name, acc, auc
    )
    logging.debug("Classification report for %s:\n%s", name, report)
logging.info("Model evaluation completed")

# ----------------------------------------------------------------------------
# 7) Feature importances (from best RF)
# ----------------------------------------------------------------------------
logging.info("Top 10 feature importances (Random Forest)")
importances = pd.Series(best_rf.feature_importances_, index=feature_cols)
top_imp = importances.sort_values(ascending=False).head(10)
for feat, val in top_imp.items():
    logging.info("%s: %.4f", feat, val)

# ----------------------------------------------------------------------------
# 8) Scoring function for new data & example run
# ----------------------------------------------------------------------------
def score_defaults(history_df, defaults_df, model, scaler, feature_columns, threshold=0.5):
    h = history_df.copy()
    valid = h[h['payment_status'] != -2]
    cents = valid.groupby('payment_status')[['bill_amt','paid_amt']].mean()
    h['payment_status_imputed'] = h.apply(
        lambda r: assign_nearest_status(r, cents), axis=1
    )
    h['delay'] = h['payment_status_imputed'].apply(lambda x: 0 if x == -1 else x)
    h['paid_ratio'] = np.where(
        h['bill_amt']>0, h['paid_amt']/h['bill_amt'], 0
    )
    grp2 = h.groupby('client_id')
    feats2 = pd.DataFrame({
        'count_on_time': grp2.apply(lambda x: (x['delay']==0).sum()),
        'count_delayed': grp2.apply(lambda x: (x['delay']>0).sum()),
        'max_delay':     grp2['delay'].max(),
        'avg_delay':     grp2['delay'].mean(),
        'std_delay':     grp2['delay'].std().fillna(0),
        'last_month_delay': h[h['month']==h['month'].max()]
                                .set_index('client_id')['delay'],
        'delay_trend':   grp2.apply(compute_slope),
        'avg_paid_ratio': grp2['paid_ratio'].mean(),
        'std_paid_ratio': grp2['paid_ratio'].std().fillna(0),
        'min_paid_ratio': grp2['paid_ratio'].min(),
        'max_paid_ratio': grp2['paid_ratio'].max(),
        'avg_bill_amt':   grp2['bill_amt'].mean(),
        'std_bill_amt':   grp2['bill_amt'].std().fillna(0)
    }).reset_index()
    feats2 = pd.merge(feats2, defaults_df[['client_id','credit_given']],
                      on='client_id', how='left').fillna(0)
    Xh = feats2[feature_columns]
    Xh_sc = scaler.transform(Xh)
    prob = model.predict_proba(Xh_sc)[:,1]
    pred = (prob>=threshold).astype(int)
    return pd.DataFrame({
        'client_id': feats2['client_id'],
        'probability_of_default': prob,
        'predicted_default': pred
    })

logging.info("Scoring new data")
results = score_defaults(history, defaults, best_rf, scaler, feature_cols)
logging.info("Sample scoring results:\n%s", results.head())
results.to_csv('default_predictions.csv', index=False)
logging.info("Saved predictions to 'default_predictions.csv'")
