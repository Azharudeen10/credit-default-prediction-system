# feature_engineering.py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def impute_status_similarity(history: pd.DataFrame) -> pd.DataFrame:
    valid = history[history['payment_status'] != -2]
    centroids = valid.groupby('payment_status')[['bill_amt','paid_amt']].mean()
    def assign(row):
        if row['payment_status'] != -2:
            return row['payment_status']
        p = np.array([[row['bill_amt'], row['paid_amt']]])
        d = cdist(p, centroids.values, metric='euclidean').ravel()
        return int(centroids.index[d.argmin()])
    history['payment_status_imputed'] = history.apply(assign, axis=1)
    return history


def aggregate_history(history: pd.DataFrame, defaults: pd.DataFrame) -> pd.DataFrame:
    # compute derived
    history['delay'] = history['payment_status_imputed'].apply(lambda x: 0 if x==-1 else x)
    history['paid_ratio'] = np.where(history['bill_amt']>0, history['paid_amt']/history['bill_amt'], 0)

    def slope(g): return np.polyfit(g.month, g.delay, 1)[0] if len(g)>1 else 0.0

    grp = history.groupby('client_id')
    feats = pd.DataFrame({
        'count_on_time': grp.apply(lambda g: (g.delay==0).sum()),
        'count_delayed': grp.apply(lambda g: (g.delay>0).sum()),
        'avg_delay':     grp.delay.mean(),
        'max_delay':     grp.delay.max(),
        'delay_trend':   grp.apply(slope),
        'avg_paid_ratio':grp.paid_ratio.mean(),
        'bill_sum':      grp.bill_amt.sum(),
        'paid_sum':      grp.paid_amt.sum(),
    }).reset_index()

    # merge credit_given
    feats = feats.merge(defaults[['client_id','credit_given']], on='client_id', how='left')
    return feats
