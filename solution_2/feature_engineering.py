import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import logging

def assign_nearest_status(row, centroids):
    if row['payment_status'] != -2:
        return row['payment_status']
    point = np.array([[row['bill_amt'], row['paid_amt']]])
    dists = cdist(point, centroids.values, metric='euclidean').ravel()
    return int(centroids.index[dists.argmin()])

def compute_slope(df):
    if len(df) > 1:
        return np.polyfit(df['month'], df['payment_status_imputed'], 1)[0]
    else:
        return 0.0

def make_features(history: pd.DataFrame, defaults: pd.DataFrame):
    """Impute, compute delay/paid_ratio, aggregate per client + merge credit_given."""
    logging.info("Computing centroids for -2 imputation")
    valid = history[history['payment_status'] != -2]
    centroids = valid.groupby('payment_status')[['bill_amt','paid_amt']].mean()

    logging.info("Imputing missing statuses")
    history['payment_status_imputed'] = history.apply(
        assign_nearest_status, axis=1, args=(centroids, )
    )
    history['delay'] = history['payment_status_imputed'].apply(lambda x: 0 if x == -1 else x)
    history['paid_ratio'] = np.where(history['bill_amt']>0, history['paid_amt']/history['bill_amt'], 0)

    logging.info("Aggregating per-client features")
    grp = history.groupby('client_id')
    feats = pd.DataFrame({
        'count_on_time':    grp.apply(lambda g: (g['delay'] == 0).sum()),
        'count_delayed':    grp.apply(lambda g: (g['delay'] > 0).sum()),
        'avg_delay':        grp['delay'].mean(),
        'max_delay':        grp['delay'].max(),
        'delay_ratio':      grp.apply(lambda g: (g['delay'] > 0).mean()),
        'avg_paid_ratio':   grp['paid_ratio'].mean(),
        'std_paid_ratio':   grp['paid_ratio'].std().fillna(0),
        'delay_trend':      grp.apply(compute_slope),
    }).reset_index()  # now feats has `client_id`

    # merge in credit_given by client_id
    feats = feats.merge(
        defaults[['client_id','credit_given']],
        on='client_id', how='left'
    ).fillna(0)

    return feats
