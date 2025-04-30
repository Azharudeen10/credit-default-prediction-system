# streamlit_app.py
import sys
import os
import logging
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# 0) Configure logging to print INFO+ to stdout in the terminal
# ----------------------------------------------------------------------------
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
root.handlers.clear()
root.addHandler(handler)

# Also show Streamlit internal logs if desired
logging.getLogger("streamlit").setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Imports for your pipeline
# ----------------------------------------------------------------------------
from ingest_data import load_data
from feature_engineering import make_features
from train_model import train_and_eval
from score_model import score_defaults

# ----------------------------------------------------------------------------
# Streamlit page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Credit Default Scorer", layout="wide")
st.title("🎯 Credit Default Prediction")

# ----------------------------------------------------------------------------
# 1) File upload
# ----------------------------------------------------------------------------
st.markdown("### Step 1: Upload your CSV files")
hist_file = st.file_uploader("Payment History CSV", type="csv", key="hist")
def_file  = st.file_uploader("Payment Default CSV", type="csv", key="def")

if hist_file and def_file:
    # 1a) Persist uploads
    os.makedirs("data", exist_ok=True)
    hist_path = "data/payment_history.csv"
    def_path  = "data/payment_default.csv"
    with open(hist_path, "wb") as f:
        f.write(hist_file.getbuffer())
    with open(def_path, "wb") as f:
        f.write(def_file.getbuffer())
    st.success("✅ Files saved to `data/`")
    logging.info("Files written to disk")

    # placeholders for UI progress
    status = st.empty()
    bar    = st.progress(0)
    step   = 0
    total  = 5

    # ----------------------------------------------------------------------------
    # 2) Load & feature-engineer (cached)
    # ----------------------------------------------------------------------------
    status.text("🔍 Step 2/5: Loading data & engineering features…")
    @st.cache_data(show_spinner=False)
    def cached_ingest_and_feature(h_path, d_path):
        logging.info("Loading CSVs")
        hist_df, defs_df = load_data(h_path, d_path)
        logging.info("Engineering features")
        feats_df = make_features(hist_df, defs_df)
        return hist_df, defs_df, feats_df

    history_df, defaults_df, feats_df = cached_ingest_and_feature(hist_path, def_path)
    step += 1; bar.progress(step/total)
    logging.info("Data loaded and features created")

    # ----------------------------------------------------------------------------
    # 3) Prepare training set
    # ----------------------------------------------------------------------------
    status.text("🧮 Step 3/5: Preparing training dataset…")
    train_df = feats_df.merge(
        defaults_df[["client_id","default"]],
        on="client_id", how="left"
    ).fillna(0)
    step += 1; bar.progress(step/total)
    logging.info("Training DataFrame prepared")

    # ----------------------------------------------------------------------------
    # 4) Train & evaluate models (cached)
    # ----------------------------------------------------------------------------
    status.text("⚙️ Step 4/5: Training and evaluating models…")
    @st.cache_resource(show_spinner=False)
    def cached_train(df):
        logging.info("Starting model training")
        return train_and_eval(df)

    results, best_model, scaler = cached_train(train_df)
    step += 1; bar.progress(step/total)
    logging.info("Model training and evaluation complete")

    # Display model performance
    status.text("🎯 Step 5/5: Displaying performance & scoring…")
    st.markdown("### Model Performance")
    cols = st.columns(len(results))
    for col, (name, metrics) in zip(cols, results.items()):
        col.metric(
            label=name,
            value=f"AUC {metrics['auc']:.3f}",
            delta=f"Acc {metrics['accuracy']:.3f}"
        )

    # ----------------------------------------------------------------------------
    # 5) Score full dataset
    # ----------------------------------------------------------------------------
    logging.info("Scoring all clients")
    preds_df = score_defaults(history_df, defaults_df, best_model, scaler, feats_df)
    out_path = "data/default_predictions.csv"
    preds_df.to_csv(out_path, index=False)
    bar.progress(1.0)
    st.balloons()
    logging.info(f"Predictions saved to {out_path}")

    # Show sample and download
    st.markdown("### Sample Predictions")
    st.dataframe(preds_df.head(10))

    st.markdown("### Download Full Predictions")
    st.download_button(
        label="📥 Download CSV",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name="default_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload both CSV files to get started.")
