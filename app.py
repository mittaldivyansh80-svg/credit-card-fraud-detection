import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "creditcard.csv")
MODEL_PATH = os.path.join(ROOT, "models", "fraud_model.joblib")
META_PATH = os.path.join(ROOT, "models", "metadata.joblib")
sys.path.insert(0, ROOT)

from src.train_model import train_model, FEATURES

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f5f7fa !important;
    color: #1f2937 !important;
}

/* Main content area */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1, h2, h3 {
    color: #172b4d !important;
    font-weight: 650 !important;
}

h1 {
    font-size: 1.9rem !important;
}

/* Paragraphs and general text */
p, li, label, .stMarkdown {
    color: #263445 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}

/* Input labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #263445 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Text and number input boxes */
[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    -webkit-text-fill-color: #1f2937 !important;
    border-color: #cbd5e1 !important;
    opacity: 1 !important;
}

/* Dropdowns */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
}

[data-baseweb="select"] span,
[data-baseweb="select"] input {
    color: #1f2937 !important;
    -webkit-text-fill-color: #1f2937 !important;
}

/* Number input buttons */
[data-testid="stNumberInput"] button {
    background-color: #f8fafc !important;
    color: #334155 !important;
    border-color: #cbd5e1 !important;
}

/* Form container */
div[data-testid="stForm"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 22px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 6px;
    font-weight: 600;
    min-height: 42px;
}

/* Primary action button */
.stButton > button[kind="primary"] {
    background-color: #245b78 !important;
    color: #ffffff !important;
    border: 1px solid #245b78 !important;
}

/* Result cards */
.result {
    padding: 18px 20px;
    border-radius: 8px;
    border: 1px solid;
    margin-top: 14px;
}

.safe {
    background-color: #edf7f0 !important;
    border-color: #b9dec5 !important;
    color: #205b37 !important;
}

.risk {
    background-color: #fff0f0 !important;
    border-color: #edc1c1 !important;
    color: #8c2929 !important;
}

/* Supporting text */
.muted,
.small-muted {
    color: #64748b !important;
    font-size: 14px !important;
}

/* Improve metric visibility */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #172b4d !important;
}

/* Separators */
hr {
    border-color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)


st.title("Credit Card Fraud Detection")
st.markdown('<p class="muted">Enter transaction details and get a prediction from the model trained on your uploaded dataset.</p>', unsafe_allow_html=True)

if not os.path.isfile(DATA_PATH):
    st.error("Dataset not found. Place your supplied CSV at data/creditcard.csv.")
    st.stop()

@st.cache_resource(show_spinner="Training the model on your dataset. This happens on first launch...")
def load_model(path, modified):
    return train_model(path, MODEL_PATH, META_PATH)

try:
    model, meta, df = load_model(DATA_PATH, os.path.getmtime(DATA_PATH))
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

k1, k2, k3 = st.columns(3)
k1.metric("Transactions in dataset", f"{meta['rows']:,}")
k2.metric("Legitimate records", f"{meta['legitimate_count']:,}")
k3.metric("Fraud-labelled records", f"{meta['fraud_count']:,}")

st.divider()
single, bulk = st.tabs(["Enter transaction", "Upload CSV"])

with single:
    st.subheader("Transaction details")
    st.caption("Enter the available transaction parameters. The model was trained using these same fields.")
    categories = meta["merchant_categories"]
    with st.form("transaction_form"):
        left, right = st.columns(2)
        with left:
            amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
            hour = st.number_input("Transaction hour (0–23)", min_value=0, max_value=23, value=12, step=1)
            merchant = st.selectbox("Merchant category", categories)
            foreign = st.selectbox("Foreign transaction", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with right:
            mismatch = st.selectbox("Location mismatch", [0, 1], format_func=lambda x: "Yes" if x else "No")
            trust = st.slider("Device trust score", 0, 100, 70)
            velocity = st.number_input("Transactions in last 24 hours", min_value=0, value=2, step=1)
            age = st.number_input("Cardholder age", min_value=18, max_value=120, value=35, step=1)
        submit = st.form_submit_button("Check transaction", type="primary", use_container_width=True)
    if submit:
        row = pd.DataFrame([{
            "amount": amount,
            "transaction_hour": hour,
            "merchant_category": merchant,
            "foreign_transaction": foreign,
            "location_mismatch": mismatch,
            "device_trust_score": trust,
            "velocity_last_24h": velocity,
            "cardholder_age": age,
        }], columns=FEATURES)
        score = float(model.predict_proba(row)[0, 1])
        if score >= 0.5:
            st.markdown(f'<div class="result risk"><h3>Flagged for review</h3><p>Estimated fraud score: <b>{score:.1%}</b></p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result safe"><h3>Not flagged by the model</h3><p>Estimated fraud score: <b>{score:.1%}</b></p></div>', unsafe_allow_html=True)
        st.caption("This is a model estimate, not proof of fraud. Review flagged transactions before taking action.")

with bulk:
    st.subheader("Check multiple transactions")
    st.write("Upload a CSV containing the same feature columns. The target column `is_fraud` is optional and is not used for prediction.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in incoming.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                X = incoming[FEATURES].copy()
                if X.isna().any().any():
                    st.error("Some required fields are blank. Fill them before uploading.")
                elif st.button("Generate results", type="primary"):
                    scores = model.predict_proba(X)[:, 1]
                    results = incoming.copy()
                    results["fraud_score"] = scores
                    results["prediction"] = np.where(scores >= 0.5, "Flagged for review", "Not flagged")
                    r1, r2 = st.columns(2)
                    r1.metric("Transactions checked", f"{len(results):,}")
                    r2.metric("Flagged for review", f"{int((scores >= 0.5).sum()):,}")
                    st.dataframe(results.head(500), use_container_width=True, hide_index=True)
                    st.download_button("Download results CSV", results.to_csv(index=False).encode("utf-8"), "fraud_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

st.divider()
st.caption("Academic project. A model score is an estimate and should not be treated as a definitive fraud determination.")
