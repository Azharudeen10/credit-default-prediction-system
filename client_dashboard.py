import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import cdist
import os

# Paths (adjust as needed)
HISTORY_PATH     = 'data/payment_history.csv'
PREDICTIONS_PATH = 'data/default_predictions.csv'

def check_files():
    missing = []
    for p in (HISTORY_PATH, PREDICTIONS_PATH):
        if not os.path.exists(p):
            missing.append(p)
    return missing

@st.cache_data
def load_history(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # ————— Impute -2 via nearest‐centroid on (bill_amt, paid_amt) —————
    valid    = df[df['payment_status'] != -2]
    centroids = valid.groupby('payment_status')[['bill_amt','paid_amt']].mean()

    def assign_nearest_status(row):
        if row['payment_status'] != -2:
            return row['payment_status']
        point = np.array([[row['bill_amt'], row['paid_amt']]])
        dists = cdist(point, centroids.values, metric='euclidean').ravel()
        return int(centroids.index[dists.argmin()])

    df['payment_status_imputed'] = df.apply(assign_nearest_status, axis=1)
    return df

@st.cache_data
def load_predictions(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df

def plot_client_history(df: pd.DataFrame, client_id: int):
    client_df = df[df.client_id == client_id].sort_values("month")
    if client_df.empty:
        st.write(f"No history found for client {client_id}")
        return

    months = client_df['month'].astype(int).tolist()
    x = np.arange(len(months))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ─── Payment Status (Delay Code) ──────────────────────
    ax0 = axes[0]
    ax0.plot(months, client_df['payment_status_imputed'], marker='o', linestyle='-')
    ax0.set_title("Payment Status (Delay Code)")
    ax0.set_xlabel("Month")
    ax0.set_ylabel("Delay Code")
    ax0.set_xticks(months)
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ─── Bill vs Paid Amount ──────────────────────────────
    ax1 = axes[1]
    width = 0.35
    ax1.bar(x - width/2, client_df['bill_amt'], width, label='Bill Amt')
    ax1.bar(x + width/2, client_df['paid_amt'], width, label='Paid Amt')
    ax1.set_title("Bill vs Paid Amount")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Amount")
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.legend()

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Client Payment History & Default Prediction")

    # 1. Check for required files
    missing = check_files()
    if missing:
        st.error(
            "Required files not found:\n" +
            "\n".join(f"- {p}" for p in missing) +
            "\n\nPlease run the training/scoring pipeline first to generate them."
        )
        return

    # 2. Load data (with imputation baked in)
    history_df     = load_history(HISTORY_PATH)
    predictions_df = load_predictions(PREDICTIONS_PATH)

    # 3. User input
    client_id = st.number_input("Enter Client ID:", min_value=1, step=1, value=1)

    if st.button("Show Data"):
        st.subheader(f"Payment History for Client {client_id}")
        plot_client_history(history_df, client_id)

        st.subheader("Predicted Default Status")
        result = predictions_df[predictions_df.client_id == client_id]
        if result.empty:
            st.write("No prediction found for this client.")
        else:
            prob = result['probability_of_default'].iloc[0]
            pred = result['predicted_default'].iloc[0]
            st.markdown(f"- **Probability of Default:** {prob:.2f}")
            st.markdown(f"- **Predicted Default:** {'Yes' if pred == 1 else 'No'}")

if __name__ == "__main__":
    main()
